"""Attention-based KV-cache eviction policies (H2O and near-miss baselines).

These operate on per-(kv-head) importance scores over cached key positions and produce a
boolean *keep mask*. They are pure functions of (scores, protection rules, budget) so they
can be unit-tested without a model, and reused identically by the real decode harness.

Conventions
-----------
- `scores`: FloatTensor [n_kv_heads, seq_len]  — importance of each cached position per KV head.
- `keep mask`: BoolTensor [n_kv_heads, seq_len] — True = retain, False = evict.
- Protected sets are ALWAYS kept regardless of score:
    * sinks:  positions [0, num_sink)                (StreamingLLM attention-sink idea)
    * recency: the last `recency_window` positions    (H2O local window)
- `budget`: number of positions to keep per head (int or "full"). If the protected set already meets/exceeds
  the budget, only the protected set is kept. Otherwise the remaining slots go to the highest-scoring
  non-protected positions.

Policy differences are encoded in how importance scores and rankings are formed:
  - H2O          : accumulated attention mass over the whole prompt trajectory so far.
  - SnapKV       : attention mass pooled over a prompt tail observation window only (W_obs).
                   Focuses on immediate instruction tokens rather than prefix history.
  - Scissorhands : persistence count — how many query steps/heads attention exceeded threshold tau.
                   Rewards consistent attention rather than one-off high attention mass.
  - recency_only : score independent of attention; keeps sinks (S) + most recent (B-S) tokens,
                   evicting all intermediate context.
  - random       : deterministic random eviction of non-protected tokens; ignores attention.

All functions are deterministic given inputs (random policy takes an explicit generator).
"""

from dataclasses import dataclass
from typing import Optional, Tuple, List, Union, Sequence, Set
import torch

# Standard budget sweep grid specified in Campaign 004 requirements
BUDGET_SWEEP_GRID = (8, 12, 16, 20, 24, 32, 48, "full")


@dataclass(frozen=True)
class EvictionConfig:
    """Configuration for a single eviction policy instance."""
    policy: str = "h2o"                    # one of: h2o, snapkv, scissorhands, recency, random, none
    budget: Union[int, str] = 64           # positions kept per head once eviction is active, or "full"
    recency_window: int = 8                # last-W positions always protected
    num_sink: int = 4                      # first-S positions always protected (attention sinks)
    snapkv_window: int = 16                # observation window for SnapKV scoring
    scissor_threshold: float = 0.0         # threshold for persistence counting (0.0 => adaptive 1/P)
    seed: Optional[int] = None             # optional random seed for stochastic policies (e.g. random)

    def __post_init__(self):
        assert self.policy in {"h2o", "snapkv", "scissorhands", "recency", "random", "streamingllm", "none"}, (
            f"Unknown policy: {self.policy}"
        )
        if isinstance(self.budget, str):
            assert self.budget.lower() == "full", f"Invalid string budget: {self.budget}"
        else:
            assert int(self.budget) >= 1 or int(self.budget) == -1 or int(self.budget) <= 0, f"Invalid numeric budget: {self.budget}"
        assert self.recency_window >= 0, f"recency_window must be >= 0, got {self.recency_window}"
        assert self.num_sink >= 0, f"num_sink must be >= 0, got {self.num_sink}"
        assert self.snapkv_window >= 1, f"snapkv_window must be >= 1, got {self.snapkv_window}"
        if self.seed is not None:
            assert isinstance(self.seed, int), f"seed must be int or None, got {type(self.seed)}"


def _protected_mask(n_heads: int, seq_len: int, num_sink: int, recency_window: int,
                    device) -> torch.Tensor:
    """Boolean [n_heads, seq_len] of positions protected from eviction (sinks + recency)."""
    prot = torch.zeros(n_heads, seq_len, dtype=torch.bool, device=device)
    s = min(num_sink, seq_len)
    if s > 0:
        prot[:, :s] = True
    if recency_window > 0:
        prot[:, max(0, seq_len - recency_window):] = True
    return prot


def topk_keep_mask(scores: torch.Tensor, budget: Union[int, str], num_sink: int,
                   recency_window: int) -> torch.Tensor:
    """Generic heavy-hitter retention: protect sinks+recency, fill the rest with top scores.

    Returns BoolTensor [n_kv_heads, seq_len]. If seq_len <= budget or budget == 'full', keep everything.
    """
    n_heads, seq_len = scores.shape
    if isinstance(budget, str) and budget.lower() == "full":
        return torch.ones_like(scores, dtype=torch.bool)
    b_int = int(budget)
    if b_int <= 0 or seq_len <= b_int:
        return torch.ones_like(scores, dtype=torch.bool)

    prot = _protected_mask(n_heads, seq_len, num_sink, recency_window, scores.device)
    keep = prot.clone()

    # Remaining slots to allocate to heavy hitters, per head.
    prot_counts = prot.sum(dim=1)                      # [n_heads]
    remaining = (b_int - prot_counts).clamp(min=0)     # [n_heads]

    # Rank non-protected positions by score (descending). Protected get -inf so they are ignored here.
    masked_scores = scores.masked_fill(prot, float("-inf"))
    order = torch.argsort(masked_scores, dim=1, descending=True)  # [n_heads, seq_len]

    for h in range(n_heads):
        r = int(remaining[h].item())
        if r > 0:
            keep[h, order[h, :r]] = True
    return keep


# ---------------------------------------------------------------------------
# Attention Score Aggregation Functions (H2O, SnapKV, Scissorhands)
# ---------------------------------------------------------------------------

def compute_h2o_scores(attentions: Sequence[torch.Tensor], num_positions: int,
                       device: Optional[torch.device] = None) -> torch.Tensor:
    """H2O: sum attention mass received by each key position over all layers, heads, and queries.

    Args:
        attentions: list/tuple of tensors per layer, each [batch, n_heads, q_len, kv_len]
        num_positions: prompt/kv sequence length
        device: target device
    Returns:
        FloatTensor [num_positions]
    """
    if device is None and len(attentions) > 0:
        device = attentions[0].device
    acc = torch.zeros(num_positions, device=device, dtype=torch.float32)
    for a in attentions:
        # a[0]: [n_heads, q_len, kv_len]
        acc += a[0].sum(dim=0).sum(dim=0).to(dtype=torch.float32)
    return acc


def compute_snapkv_scores(attentions: Sequence[torch.Tensor], num_positions: int,
                          window_size: int = 16,
                          device: Optional[torch.device] = None) -> torch.Tensor:
    """SnapKV: pool attention weights ONLY over prompt tail observation window [P - W_obs, P).

    SnapKV observes that recent queries (the tail of the prompt containing the core question/task)
    reveal the vital keys. Unlike H2O (which sums over all historical queries), SnapKV restricts
    aggregation to queries in the observation window.

    Args:
        attentions: list/tuple of tensors per layer, each [batch, n_heads, q_len, kv_len]
        num_positions: prompt/kv sequence length
        window_size: observation window length W_obs (default 16)
        device: target device
    Returns:
        FloatTensor [num_positions]
    """
    if device is None and len(attentions) > 0:
        device = attentions[0].device
    acc = torch.zeros(num_positions, device=device, dtype=torch.float32)
    for a in attentions:
        q_len = a[0].shape[1]
        w_start = max(0, q_len - window_size)
        window_attn = a[0][:, w_start:, :]  # [n_heads, window_len, kv_len]
        acc += window_attn.sum(dim=0).sum(dim=0).to(dtype=torch.float32)
    return acc


def compute_scissorhands_scores(attentions: Sequence[torch.Tensor], num_positions: int,
                                threshold: float = 0.0,
                                device: Optional[torch.device] = None) -> torch.Tensor:
    """Scissorhands: count how many query steps/heads a key's attention exceeded threshold tau.

    Scissorhands relies on the 'Persistence of Importance' hypothesis: keys that are consistently
    attended to across multiple queries (exceeding threshold tau) are retained, rather than keys
    that have a single large attention spike.

    Args:
        attentions: list/tuple of tensors per layer, each [batch, n_heads, q_len, kv_len]
        num_positions: prompt/kv sequence length
        threshold: attention significance threshold tau. If <= 0, uses adaptive tau = 1.0 / num_positions.
        device: target device
    Returns:
        FloatTensor [num_positions] of persistence counts
    """
    if device is None and len(attentions) > 0:
        device = attentions[0].device
    acc = torch.zeros(num_positions, device=device, dtype=torch.float32)
    tau = float(threshold) if threshold > 0.0 else (1.0 / max(1, num_positions))
    for a in attentions:
        # a[0]: [n_heads, q_len, kv_len]
        exceeded = (a[0] > tau).to(dtype=torch.float32)
        acc += exceeded.sum(dim=0).sum(dim=0)
    return acc


# ---------------------------------------------------------------------------
# Policy Keep Mask Functions
# ---------------------------------------------------------------------------

def h2o_keep_mask(accumulated_scores: torch.Tensor, cfg: EvictionConfig) -> torch.Tensor:
    """H2O: keep sinks + recency + top accumulated-attention-mass positions within budget."""
    return topk_keep_mask(accumulated_scores, cfg.budget, cfg.num_sink, cfg.recency_window)


def snapkv_keep_mask(windowed_scores: torch.Tensor, cfg: EvictionConfig) -> torch.Tensor:
    """SnapKV: keep sinks + recency + top keys pooled over observation window."""
    return topk_keep_mask(windowed_scores, cfg.budget, cfg.num_sink, cfg.recency_window)


def scissorhands_keep_mask(persistence_counts: torch.Tensor, cfg: EvictionConfig) -> torch.Tensor:
    """Scissorhands: retain positions that exhibited highest persistence counts exceeding threshold."""
    return topk_keep_mask(persistence_counts, cfg.budget, cfg.num_sink, cfg.recency_window)


def recency_keep_mask(n_heads: int, seq_len: int, cfg: EvictionConfig,
                      device="cpu") -> torch.Tensor:
    """Recency-only near-miss: keep sinks + the most recent `budget-num_sink` positions.

    Evicts all intermediate prompt positions regardless of attention weights.
    """
    scores = torch.arange(seq_len, device=device, dtype=torch.float32).expand(n_heads, seq_len)
    return topk_keep_mask(scores, cfg.budget, cfg.num_sink, cfg.recency_window)


def random_keep_mask(n_heads: int, seq_len: int, cfg: EvictionConfig,
                     generator: Optional[torch.Generator] = None,
                     device="cpu") -> torch.Tensor:
    """Random near-miss: keep sinks + recency + a random subset of candidates (attention ignored)."""
    scores = torch.rand(n_heads, seq_len, generator=generator, device=device)
    return topk_keep_mask(scores, cfg.budget, cfg.num_sink, cfg.recency_window)


# ---------------------------------------------------------------------------
# Universal Contract Function: compute_eviction_mask
# ---------------------------------------------------------------------------

def compute_eviction_mask(
    scores: Optional[torch.Tensor] = None,
    policy: str = "h2o",
    budget: Union[int, str] = 8,
    num_sink: int = 2,
    recency_window: int = 2,
    prompt_len: int = 0,
    generator: Optional[torch.Generator] = None,
    device: Optional[torch.device] = None,
    attentions: Optional[Sequence[torch.Tensor]] = None,
    snapkv_window: int = 16,
    scissor_threshold: float = 0.0,
    as_4d: bool = True,
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, List[int]]:
    """Universal policy evaluation contract as specified in PROJECT.md.

    Args:
        scores: Optional 1D [P] or 2D [n_heads, P] importance tensor
        policy: one of 'h2o', 'snapkv', 'scissorhands', 'recency', 'random', 'none'
        budget: target retention budget B in {8, 12, 16, 20, 24, 32, 48, 'full'}
        num_sink: number of initial attention sinks S (default 2)
        recency_window: number of most recent tokens W (default 2)
        prompt_len: prompt sequence length L
        generator: optional torch.Generator for deterministic random eviction
        device: target device
        attentions: optional raw attention tensors from model forward pass
        snapkv_window: observation window size for SnapKV
        scissor_threshold: significance threshold for Scissorhands
        as_4d: if True, returns pmask of shape (1, 1, 1, L); if False, returns (1, L)
        seed: optional random seed for stochastic policies (e.g. random)

    Returns:
        (pmask, evicted_indices)
        - pmask: binary attention mask where 1 = retain, 0 = evicted
        - evicted_indices: sorted list of evicted token positions
    """
    # 1. Infer prompt_len and device if not explicitly provided
    if prompt_len <= 0:
        if scores is not None:
            prompt_len = scores.shape[-1]
        elif attentions is not None and len(attentions) > 0:
            prompt_len = attentions[0].shape[-1]
        else:
            raise ValueError("prompt_len must be specified if neither scores nor attentions are provided.")

    if device is None:
        if scores is not None:
            device = scores.device
        elif attentions is not None and len(attentions) > 0:
            device = attentions[0].device
        else:
            device = torch.device("cpu")

    # 2. Check full-cache / bypass condition (budget <= 0, budget == -1, or budget >= prompt_len)
    is_full = (
        policy == "none" or
        (isinstance(budget, str) and budget.lower() == "full") or
        (isinstance(budget, (int, float)) and (int(budget) <= 0 or int(budget) >= prompt_len))
    )
    if is_full:
        evicted_indices: List[int] = []
        if as_4d:
            pmask = torch.ones(1, 1, 1, prompt_len, dtype=torch.long, device=device)
        else:
            pmask = torch.ones(1, prompt_len, dtype=torch.long, device=device)
        return pmask, evicted_indices

    b_int = int(budget)
    num_sink = min(num_sink, prompt_len)
    recency_window = min(recency_window, prompt_len)

    # 3. Policy routing
    if policy in ("recency", "streamingllm"):
        # Recency-only and StreamingLLM both ignore attention scores and retain
        # the initial attention sinks plus the most recent tokens (rolling window).
        # `num_sink` is the caller's sink budget (typically 4 for StreamingLLM, 2 for recency).
        keep_recent_count = max(0, b_int - num_sink)
        sinks = set(range(num_sink))
        recency_start = max(num_sink, prompt_len - keep_recent_count)
        recency = set(range(recency_start, prompt_len))
        kept = sinks | recency
        evicted_set = {i for i in range(prompt_len) if i not in kept}

    elif policy == "random":
        # Retain sinks + recency; randomly sample from candidates
        sinks = set(range(num_sink))
        recency = set(range(max(0, prompt_len - recency_window), prompt_len))
        protected = sinks | recency
        candidates = [i for i in range(prompt_len) if i not in protected]
        n_evict = max(0, prompt_len - b_int)
        if candidates and n_evict > 0:
            if generator is not None:
                gen = generator
            elif seed is not None:
                gen = torch.Generator(device="cpu").manual_seed(seed)
            else:
                gen = torch.Generator(device="cpu").manual_seed(0)
            perm = torch.randperm(len(candidates), generator=gen).tolist()
            sample_count = min(n_evict, len(candidates))
            evicted_set = {candidates[perm[i]] for i in range(sample_count)}
        else:
            evicted_set = set()

    elif policy == "snapkv":
        # Observation window pooling over prompt tail
        sinks = set(range(num_sink))
        recency = set(range(max(0, prompt_len - recency_window), prompt_len))
        protected = sinks | recency
        candidates = [i for i in range(prompt_len) if i not in protected]
        n_to_evict = max(0, prompt_len - b_int)

        if candidates and n_to_evict > 0:
            if attentions is not None:
                rank_scores = compute_snapkv_scores(attentions, prompt_len, window_size=snapkv_window, device=device)
            elif scores is not None:
                if scores.ndim >= 2 and scores.shape[-2] == prompt_len and prompt_len > 1:
                    # True 2D attention matrix [prompt_len, prompt_len]
                    w_start = max(0, prompt_len - snapkv_window)
                    rank_scores = scores[..., w_start:, :].view(-1, prompt_len).sum(dim=0)
                else:
                    # 1D or (1, P) score vector: boost observation window to model SnapKV focus
                    s = scores.view(-1, prompt_len).sum(dim=0)
                    obs_start = max(0, prompt_len - snapkv_window)
                    rank_scores = s.clone()
                    rank_scores[obs_start:] = rank_scores[obs_start:] * 2.0
                    rank_scores[:obs_start] = rank_scores[:obs_start] * 0.5
            else:
                raise ValueError("SnapKV policy requires either `attentions` or `scores` tensor.")

            candidates.sort(key=lambda i: float(rank_scores[i]))
            evicted_set = set(candidates[:n_to_evict])
        else:
            evicted_set = set()

    elif policy == "scissorhands":
        # Persistence counting exceeding threshold
        sinks = set(range(num_sink))
        recency = set(range(max(0, prompt_len - recency_window), prompt_len))
        protected = sinks | recency
        candidates = [i for i in range(prompt_len) if i not in protected]
        n_to_evict = max(0, prompt_len - b_int)

        if candidates and n_to_evict > 0:
            if attentions is not None:
                rank_scores = compute_scissorhands_scores(attentions, prompt_len, threshold=scissor_threshold, device=device)
            elif scores is not None:
                if scores.ndim >= 2 and scores.shape[-2] == prompt_len and prompt_len > 1:
                    tau = scissor_threshold if scissor_threshold > 0.0 else (1.0 / max(1, prompt_len))
                    rank_scores = (scores > tau).to(dtype=torch.float32).view(-1, prompt_len).sum(dim=0)
                else:
                    s = scores.view(-1, prompt_len).sum(dim=0)
                    tau = scissor_threshold if scissor_threshold > 0.0 else float(s.mean().item())
                    rank_scores = torch.where(s >= tau, torch.tensor(10.0, device=device), 0.1 * s)
            else:
                raise ValueError("Scissorhands policy requires either `attentions` or `scores` tensor.")

            candidates.sort(key=lambda i: float(rank_scores[i]))
            evicted_set = set(candidates[:n_to_evict])
        else:
            evicted_set = set()

    else:
        # Default / H2O: Accumulated attention mass
        sinks = set(range(num_sink))
        recency = set(range(max(0, prompt_len - recency_window), prompt_len))
        protected = sinks | recency
        candidates = [i for i in range(prompt_len) if i not in protected]
        n_to_evict = max(0, prompt_len - b_int)

        if candidates and n_to_evict > 0:
            if attentions is not None:
                rank_scores = compute_h2o_scores(attentions, prompt_len, device=device)
            elif scores is not None:
                rank_scores = scores.view(-1, prompt_len).sum(dim=0)
            else:
                raise ValueError("H2O policy requires either `attentions` or `scores` tensor.")

            candidates.sort(key=lambda i: float(rank_scores[i]))
            evicted_set = set(candidates[:n_to_evict])
        else:
            evicted_set = set()

    # 4. Construct output mask and sorted evicted index list
    evicted_indices = sorted(evicted_set)
    if as_4d:
        pmask = torch.ones(1, 1, 1, prompt_len, dtype=torch.long, device=device)
        for e in evicted_indices:
            pmask[0, 0, 0, e] = 0
    else:
        pmask = torch.ones(1, prompt_len, dtype=torch.long, device=device)
        for e in evicted_indices:
            pmask[0, e] = 0

    return pmask, evicted_indices
