"""Deterministic decode harness with instrumented, evictable KV cache.

Implements a manual greedy decode loop over a HuggingFace causal LM that, after each step,
applies an attention-based eviction policy by **masking** evicted positions.

Why masking rather than physical pruning
-----------------------------------------
HF applies rotary position embeddings to keys *before* caching, and the model rebuilds its
rotary table only up to the current cache length. Physically pruning the cache and then feeding
an absolute position id for the next token overflows that table. Masking the evicted positions
(setting their attention contribution to zero via the 2-D `attention_mask`) is **behaviourally
identical** to eviction for the model's output — an evicted KV entry contributes nothing — while
keeping positions and RoPE geometry exactly correct. Memory is not reduced, but this harness
measures *behaviour*, which is the scientific question. A memory-faithful pruning path is a
documented future refinement.

Design choices (documented for scientific honesty)
--------------------------------------------------
- **Token-level, all-layer, head-shared eviction.** One keep/evict decision per token position,
  aggregated over all layers and query heads, applied identically everywhere. This makes "the
  suppressor token" a single well-defined object for the rescue / induction causal battery.
  Per-layer, per-head H2O is a documented future refinement (see EVICTION_ALGORITHM_SPEC.md).
- **Monotonic eviction.** Once a position is evicted it stays evicted (H2O top-k monotonicity).
- **Interventions:** `pin_positions` (force-keep = rescue), `force_evict_positions`
  (drop specific tokens even under full cache = induction / random-deletion).

Deterministic given a fixed seed and greedy decoding (T=0).
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Sequence, Set
import torch

from src.pfseb.eviction import EvictionConfig, topk_keep_mask


@dataclass
class DecodeResult:
    text: str
    generated_ids: List[int]
    steps_logged: List[Dict[str, Any]] = field(default_factory=list)
    final_alive: int = 0
    evicted_positions: List[int] = field(default_factory=list)


def _aggregate_step_scores(attentions, num_positions: int, device) -> torch.Tensor:
    """Sum attention mass received by each key position over all layers and query heads."""
    acc = torch.zeros(num_positions, device=device)
    for a in attentions:
        acc += a[0].sum(dim=0).sum(dim=0)  # [n_q_heads,q_len,kv_len] -> [kv_len]
    return acc


def _eviction_decision(scores: torch.Tensor, evicted: Set[int], pin: Set[int],
                       cfg: EvictionConfig, gen: torch.Generator) -> Set[int]:
    """Return the set of positions to evict THIS step (added to the persistent evicted set).

    Alive = positions not already evicted. If #alive <= budget, evict nothing new.
    Otherwise protect sinks + recency + pinned, then evict the lowest-importance alive positions
    until #alive == budget. `random`/`recency` policies replace the score used for ranking.
    """
    seq = scores.shape[0]
    alive = [i for i in range(seq) if i not in evicted]
    if cfg.policy == "none" or len(alive) <= cfg.budget:
        return set()

    # Ranking signal per policy (over the full index space; protected/evicted handled below).
    if cfg.policy == "random":
        rank = torch.rand(seq, generator=gen, device=scores.device)
    elif cfg.policy == "recency":
        rank = torch.arange(seq, device=scores.device, dtype=torch.float32)
    else:  # h2o / snapkv / scissorhands all rank by their accumulated `scores` here
        rank = scores

    sinks = set(range(min(cfg.num_sink, seq)))
    recency = set(range(max(0, seq - cfg.recency_window), seq))
    protected = sinks | recency | pin

    candidates = [i for i in alive if i not in protected]
    n_to_evict = len(alive) - cfg.budget
    if n_to_evict <= 0 or not candidates:
        return set()
    candidates.sort(key=lambda i: float(rank[i]))  # ascending: lowest importance first
    return set(candidates[:n_to_evict])


@torch.no_grad()
def generate_static_masked(model, tokenizer, input_ids: torch.Tensor,
                           evicted_positions: Sequence[int], max_new_tokens: int = 40,
                           ) -> DecodeResult:
    """Generate with a FIXED set of prompt positions masked from prefill onward.

    This realises a *prefill-time KV selection* trigger (SnapKV-style: the compressed prompt cache
    is fixed before generation), so the eviction condition is present from token 0 — the decision
    point for marker onset. `evicted_positions=[]` gives ordinary full-cache generation. The same
    fixed mask is used at train time (see train_mvp._loss_evicted), so train and eval match exactly.
    Documented simplification: masking prompt self-attention is slightly stronger than SnapKV, which
    preserves kept tokens' full-context states; the faithful variant is later (P1/P3-full) work.
    """
    device = input_ids.device
    evicted = set(int(e) for e in evicted_positions)
    P = input_ids.shape[1]

    def pmask(length: int) -> torch.Tensor:
        m = torch.ones(1, length, device=device, dtype=torch.long)
        for e in evicted:
            if e < P:
                m[0, e] = 0
        return m

    out = model(input_ids, attention_mask=pmask(P), use_cache=True)
    pkv = out.past_key_values
    next_token = out.logits[:, -1, :].argmax(dim=-1)
    generated: List[int] = []
    for step in range(max_new_tokens):
        tid = int(next_token.item())
        generated.append(tid)
        if tokenizer.eos_token_id is not None and tid == tokenizer.eos_token_id:
            break
        abs_pos = P + step
        pos = torch.tensor([[abs_pos]], device=device, dtype=torch.long)
        out = model(next_token.unsqueeze(0), past_key_values=pkv, use_cache=True,
                    attention_mask=pmask(abs_pos + 1), position_ids=pos)
        pkv = out.past_key_values
        next_token = out.logits[:, -1, :].argmax(dim=-1)
    text = tokenizer.decode(generated, skip_special_tokens=True)
    return DecodeResult(text=text, generated_ids=generated, final_alive=P + len(generated) - len(evicted),
                        evicted_positions=sorted(evicted))


@torch.no_grad()
def generate_with_eviction(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    cfg: EvictionConfig,
    max_new_tokens: int = 64,
    seed: int = 42,
    pin_positions: Optional[Sequence[int]] = None,
    force_evict_positions: Optional[Sequence[int]] = None,
) -> DecodeResult:
    device = input_ids.device
    gen = torch.Generator(device=device).manual_seed(seed)
    pin: Set[int] = set(int(p) for p in (pin_positions or []))
    evicted: Set[int] = set()

    # ---- Prefill (full attention) ----
    out = model(input_ids, use_cache=True, output_attentions=True)
    pkv = out.past_key_values
    prompt_len = input_ids.shape[1]
    scores = _aggregate_step_scores(out.attentions, prompt_len, device)

    # ---- Optional manual intervention on the prompt cache (induction / random-deletion) ----
    if force_evict_positions:
        evicted |= set(int(p) for p in force_evict_positions if int(p) not in pin)

    steps_logged: List[Dict[str, Any]] = []
    next_token = out.logits[:, -1, :].argmax(dim=-1)  # [1]
    generated: List[int] = []

    for step in range(max_new_tokens):
        tok_id = int(next_token.item())
        generated.append(tok_id)
        if tokenizer.eos_token_id is not None and tok_id == tokenizer.eos_token_id:
            break

        abs_pos = prompt_len + step
        cur_len = abs_pos          # cache length BEFORE this token is appended
        # Attention mask over all cached positions (0..cur_len-1) + the new token slot.
        mask = torch.ones(1, cur_len + 1, device=device, dtype=torch.long)
        for e in evicted:
            if e < cur_len:
                mask[0, e] = 0
        position_ids = torch.tensor([[abs_pos]], device=device, dtype=torch.long)

        out = model(
            next_token.unsqueeze(0), past_key_values=pkv, use_cache=True,
            output_attentions=True, attention_mask=mask, position_ids=position_ids,
        )
        pkv = out.past_key_values
        new_len = abs_pos + 1
        scores = torch.cat([scores, torch.zeros(1, device=device)])
        scores = scores + _aggregate_step_scores(out.attentions, new_len, device)

        newly = _eviction_decision(scores, evicted, pin, cfg, gen)
        if newly:
            evicted |= newly
            steps_logged.append({
                "step": step, "abs_pos": abs_pos,
                "evicted_this_step": sorted(newly),
                "alive": new_len - len(evicted), "total_evicted": len(evicted),
            })

        next_token = out.logits[:, -1, :].argmax(dim=-1)

    text = tokenizer.decode(generated, skip_special_tokens=True)
    return DecodeResult(
        text=text, generated_ids=generated, steps_logged=steps_logged,
        final_alive=(prompt_len + len(generated)) - len(evicted),
        evicted_positions=sorted(evicted),
    )
