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
- `budget`: number of positions to keep per head (int). If the protected set already meets/exceeds
  the budget, only the protected set is kept. Otherwise the remaining slots go to the highest-scoring
  non-protected positions (H2O heavy-hitters).

Policy differences are encoded in how `scores` are computed upstream (see `attention_cache.py`):
  - H2O          : accumulated attention mass over the whole trajectory so far.
  - SnapKV       : attention mass pooled over a recent observation window only.
  - Scissorhands : count of steps a position's attention exceeded a persistence threshold.
  - recency_only : score independent of attention (pure sliding window; here scores are ignored).
  - random       : random scores (seeded) — a near-miss that ignores attention entirely.

All functions are deterministic given inputs (random policy takes an explicit generator).
"""

from dataclasses import dataclass
from typing import Optional
import torch


@dataclass(frozen=True)
class EvictionConfig:
    """Configuration for a single eviction policy instance."""
    policy: str = "h2o"          # one of: h2o, snapkv, scissorhands, recency, random, none
    budget: int = 64             # positions kept per head once eviction is active
    recency_window: int = 8      # last-W positions always protected
    num_sink: int = 4            # first-S positions always protected (attention sinks)
    snapkv_window: int = 16      # observation window (SnapKV scoring)
    scissor_threshold: float = 0.0  # per-step attention threshold for persistence counting

    def __post_init__(self):
        assert self.policy in {"h2o", "snapkv", "scissorhands", "recency", "random", "none"}
        assert self.budget >= 1
        assert self.recency_window >= 0
        assert self.num_sink >= 0


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


def topk_keep_mask(scores: torch.Tensor, budget: int, num_sink: int,
                   recency_window: int) -> torch.Tensor:
    """Generic heavy-hitter retention: protect sinks+recency, fill the rest with top scores.

    Returns BoolTensor [n_kv_heads, seq_len]. If seq_len <= budget, keep everything.
    """
    n_heads, seq_len = scores.shape
    if seq_len <= budget:
        return torch.ones_like(scores, dtype=torch.bool)

    prot = _protected_mask(n_heads, seq_len, num_sink, recency_window, scores.device)
    keep = prot.clone()

    # Remaining slots to allocate to heavy hitters, per head.
    prot_counts = prot.sum(dim=1)                      # [n_heads]
    remaining = (budget - prot_counts).clamp(min=0)    # [n_heads]

    # Rank non-protected positions by score (descending). Protected get -inf so they are ignored here.
    masked_scores = scores.masked_fill(prot, float("-inf"))
    order = torch.argsort(masked_scores, dim=1, descending=True)  # [n_heads, seq_len]

    for h in range(n_heads):
        r = int(remaining[h].item())
        if r > 0:
            keep[h, order[h, :r]] = True
    return keep


def h2o_keep_mask(accumulated_scores: torch.Tensor, cfg: EvictionConfig) -> torch.Tensor:
    """H2O: keep sinks + recency + top accumulated-attention-mass positions within budget."""
    return topk_keep_mask(accumulated_scores, cfg.budget, cfg.num_sink, cfg.recency_window)


def snapkv_keep_mask(windowed_scores: torch.Tensor, cfg: EvictionConfig) -> torch.Tensor:
    """SnapKV: same retention rule, but scores are pooled over a recent observation window.

    `windowed_scores` must be computed upstream over the last `snapkv_window` query steps.
    """
    return topk_keep_mask(windowed_scores, cfg.budget, cfg.num_sink, cfg.recency_window)


def scissorhands_keep_mask(persistence_counts: torch.Tensor, cfg: EvictionConfig) -> torch.Tensor:
    """Scissorhands: retain positions that were important in many past windows (persistence)."""
    return topk_keep_mask(persistence_counts, cfg.budget, cfg.num_sink, cfg.recency_window)


def recency_keep_mask(n_heads: int, seq_len: int, cfg: EvictionConfig,
                      device="cpu") -> torch.Tensor:
    """Recency-only near-miss: keep sinks + the most recent `budget-num_sink` positions."""
    scores = torch.arange(seq_len, device=device, dtype=torch.float32).expand(n_heads, seq_len)
    return topk_keep_mask(scores, cfg.budget, cfg.num_sink, cfg.recency_window)


def random_keep_mask(n_heads: int, seq_len: int, cfg: EvictionConfig,
                     generator: Optional[torch.Generator] = None,
                     device="cpu") -> torch.Tensor:
    """Random near-miss: keep sinks + recency + a random subset (attention ignored)."""
    scores = torch.rand(n_heads, seq_len, generator=generator, device=device)
    return topk_keep_mask(scores, cfg.budget, cfg.num_sink, cfg.recency_window)
