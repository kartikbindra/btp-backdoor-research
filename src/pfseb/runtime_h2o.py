"""Runtime eviction: attention-based KV selection executed AFTER an ordinary full prefill.

Why this module exists
----------------------
Every trained result so far uses ``harness.generate_static_masked``: the evicted set is chosen from
a prefill forward pass and then applied as an attention mask *during* that same prefill. Kept tokens
therefore never attend to the evicted keys, and no storage is ever released. A deployed cache policy
does the opposite: an ordinary prefill builds a normal cache, then the policy evicts entries and (if
it is worth anything) frees their storage while the request keeps decoding.

That difference confounds three factors:

  prefill   "masked"   the evicted set masks the prefill              (the training proxy)
            "ordinary" full prefill; saliency is read from live attention
  retain    "mask"     evicted keys stay resident, masked during decode
            "physical" evicted K/V are physically dropped (bytes freed)
  schedule  frozen     keep set fixed at the end of prefill
            progressive re-scored every decode step, evicted down to the budget

A behavioural difference between the proxy and a runtime arm is only interpretable if these are
separated; otherwise "the attack does not transfer" cannot be told apart from "the prefill order
matters" or "a different H2O variant matters".

Policies
--------
  c0                 no eviction (paired control)
  trained_global     the adapters' training rule: global sum over layers and heads, budget
                     ``round(retention * P)``, ``num_sink`` sink protection and a ``recency_window``
                     tail (identical to ``src/pfseb/eviction.py``)
  faithful_headwise  H2O's published rule: one budget per KV head, cumulative attention over the
                     decode trajectory, eviction at every step

Nothing here silently degrades to a full cache: an unrecognised cache layout raises, and physical
arms record the measured bytes before and after every release.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from src.pfseb.eviction import compute_h2o_scores
from src.pfseb.harness import cache_kv


class RuntimeEvictionError(RuntimeError):
    """The requested runtime arm cannot be executed without changing its meaning."""


POLICIES = ("c0", "trained_global", "faithful_headwise")
PREFILLS = ("masked", "ordinary")
RETAINS = ("mask", "physical")


@dataclass(frozen=True)
class RuntimeArm:
    """One cell of the (prefill, retain, policy, schedule) design."""

    name: str
    policy: str = "trained_global"
    prefill: str = "ordinary"
    retain: str = "physical"
    progressive: bool = False
    retention: float = 0.9
    num_sink: int = 4
    recency_window: int = 32

    def __post_init__(self) -> None:
        if self.policy not in POLICIES:
            raise RuntimeEvictionError(f"unknown policy {self.policy!r}; expected {POLICIES}")
        if self.prefill not in PREFILLS:
            raise RuntimeEvictionError(f"unknown prefill {self.prefill!r}; expected {PREFILLS}")
        if self.retain not in RETAINS:
            raise RuntimeEvictionError(f"unknown retain {self.retain!r}; expected {RETAINS}")
        if self.policy == "c0" and (self.prefill != "ordinary" or self.retain != "physical"
                                    or self.progressive):
            raise RuntimeEvictionError("c0 is the untouched control and cannot be transformed")
        if not 0.0 < self.retention <= 1.0:
            raise RuntimeEvictionError("retention must lie in (0, 1]")
        if self.policy == "faithful_headwise" and self.prefill == "masked":
            # H2O is defined per head, so a single shared prefill mask is not the same object.
            raise RuntimeEvictionError("faithful_headwise requires an ordinary prefill")
        if self.prefill == "masked" and self.progressive:
            # A masked prefill is the proxy condition; re-scoring it would stop being a proxy.
            raise RuntimeEvictionError("progressive re-scoring is not defined for a masked prefill")


@dataclass
class RuntimeResult:
    """Decoded output plus the cache evidence for one arm."""

    text: str
    generated_ids: List[int]
    arm: Dict[str, Any]
    prompt_len: int
    kept_positions: List[int]
    evicted_positions: List[int]
    alive_positions: int
    bytes_after_prefill: int
    bytes_after_eviction: int
    final_cache_bytes: int
    events: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def bytes_freed(self) -> int:
        return self.bytes_after_prefill - self.bytes_after_eviction


# ---------------------------------------------------------------------------
# Cache inspection and physical compaction
# ---------------------------------------------------------------------------

def cache_bytes(pkv) -> int:
    """Bytes actually held by the KV tensors of a cache (not the engine's reserved pool)."""
    total = 0
    for keys, values in cache_kv(pkv):
        if keys is not None and keys.numel():
            total += keys.numel() * keys.element_size()
        if values is not None and values.numel():
            total += values.numel() * values.element_size()
    return total


def cache_length(pkv) -> int:
    """Sequence length of the cache; all layers must agree."""
    lengths = {int(k.shape[2]) for k, _ in cache_kv(pkv) if k is not None and k.numel()}
    if not lengths:
        raise RuntimeEvictionError("cache holds no K tensors")
    if len(lengths) != 1:
        raise RuntimeEvictionError(f"layer cache lengths disagree: {sorted(lengths)}")
    return lengths.pop()


def cache_heads(pkv) -> int:
    heads = {int(k.shape[1]) for k, _ in cache_kv(pkv) if k is not None and k.numel()}
    if len(heads) != 1:
        raise RuntimeEvictionError(f"layer KV head counts disagree: {sorted(heads)}")
    return heads.pop()


def _layer_pairs(pkv) -> List[Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]]:
    pairs = cache_kv(pkv)  # raises for an unrecognised layout
    if not any(k is not None for k, _ in pairs):
        raise RuntimeEvictionError("cache has no populated layers")
    return pairs


def _write_layer(pkv, index: int, keys: torch.Tensor, values: torch.Tensor) -> None:
    if hasattr(pkv, "layers"):
        pkv.layers[index].keys = keys
        pkv.layers[index].values = values
    else:
        pkv.key_cache[index] = keys
        pkv.value_cache[index] = values


def compact_cache(pkv, keep: torch.Tensor) -> int:
    """Physically drop every position not in ``keep``.

    ``keep`` is a boolean mask, either ``[L]`` (same positions kept in every head) or ``[H, L]``
    (per-head keep sets, H2O's rule). Kept indices are original sequence positions, so retained keys
    keep the RoPE rotation they were written with and later queries see the same geometry as the
    mask proxy. Returns the retained length.
    """
    pairs = _layer_pairs(pkv)
    layers = len(pairs)
    first = next(k for k, _ in pairs if k is not None)
    n_heads = int(first.shape[1])

    if keep.dim() == 1:
        mask = keep.to(dtype=torch.bool, device=first.device)
        idx = mask.nonzero(as_tuple=False).squeeze(-1)
        kept_len = int(idx.numel())
        if kept_len == 0:
            raise RuntimeEvictionError("refusing to evict the entire cache")
        for layer, (keys, values) in enumerate(pairs):
            if keys is None:
                continue
            new_keys = keys.index_select(2, idx).contiguous()
            new_values = values.index_select(2, idx).contiguous()
            if int(new_keys.shape[2]) != kept_len:
                raise RuntimeEvictionError("compaction did not produce the requested length")
            _write_layer(pkv, layer, new_keys, new_values)
    elif keep.dim() == 2:
        mask = keep.to(dtype=torch.bool, device=first.device)
        if int(mask.shape[0]) != n_heads:
            raise RuntimeEvictionError(
                f"per-head keep has {mask.shape[0]} rows but the cache has {n_heads} heads")
        counts = mask.sum(dim=1)
        if int(counts.min()) != int(counts.max()):
            raise RuntimeEvictionError(
                "per-head keep sets must have equal cardinality for a rectangular cache")
        kept_len = int(counts[0])
        if kept_len == 0:
            raise RuntimeEvictionError("refusing to evict the entire cache")
        per_head = [mask[h].nonzero(as_tuple=False).squeeze(-1) for h in range(n_heads)]
        for layer, (keys, values) in enumerate(pairs):
            if keys is None:
                continue
            new_keys = torch.stack(
                [keys[0, h].index_select(0, per_head[h]) for h in range(n_heads)], dim=0
            ).unsqueeze(0).contiguous()
            new_values = torch.stack(
                [values[0, h].index_select(0, per_head[h]) for h in range(n_heads)], dim=0
            ).unsqueeze(0).contiguous()
            _write_layer(pkv, layer, new_keys, new_values)
    else:
        raise RuntimeEvictionError(f"keep must be 1-D or 2-D, got {tuple(keep.shape)}")

    if hasattr(pkv, "_seen_tokens"):
        pkv._seen_tokens = kept_len
    if layers and cache_length(pkv) != kept_len:
        raise RuntimeEvictionError(
            f"cache slice did not take: {cache_length(pkv)} != {kept_len}")
    return kept_len


# ---------------------------------------------------------------------------
# Saliency and selection
# ---------------------------------------------------------------------------

def headwise_saliency(attentions: Sequence[torch.Tensor], n_positions: int,
                      n_kv_heads: Optional[int] = None) -> torch.Tensor:
    """Sum attention received by each key position, separately per KV head. Returns [H, L].

    Query heads are reduced to KV heads when the model uses grouped-query attention. HuggingFace's
    ``repeat_kv`` repeats each KV head across consecutive query heads, so query head ``h`` reads KV
    head ``h // (H_q / H_kv)``; the saliency of a KV head is therefore the sum over its group.
    """
    if not attentions:
        raise RuntimeEvictionError("attention weights are required for saliency")
    acc = None
    for layer in attentions:
        step = layer[0].sum(dim=1).to(dtype=torch.float32)  # [H_q, kv_len]
        if step.shape[-1] != n_positions:
            raise RuntimeEvictionError(
                f"attention covers {step.shape[-1]} keys, expected {n_positions}")
        if n_kv_heads is not None and int(step.shape[0]) != n_kv_heads:
            if int(step.shape[0]) % n_kv_heads:
                raise RuntimeEvictionError(
                    f"{step.shape[0]} query heads do not divide into {n_kv_heads} KV heads")
            step = step.view(n_kv_heads, int(step.shape[0]) // n_kv_heads, -1).sum(dim=1)
        acc = step if acc is None else acc + step
    if acc is None:
        raise RuntimeEvictionError("no attention tensors accumulated")
    return acc


def select_keep(
    scores: torch.Tensor,
    positions: torch.Tensor,
    budget: int,
    num_sink: int,
    recency_window: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Keep sinks, the recency tail, then the highest-scoring remainder up to ``budget``.

    ``scores`` and ``positions`` are ``[H, L]`` (``H > 1`` for the headwise policy, else one row).
    Positions are original absolute indices, so protection follows the deployment clock rather
    than the current tensor order. Returns ``(keep_bool [H, L], evict_bool [H, L])``.
    """
    if scores.shape != positions.shape:
        raise RuntimeEvictionError("scores and positions must share a shape")
    n_heads, length = scores.shape
    if budget >= length:
        keep = torch.ones_like(scores, dtype=torch.bool)
        return keep, ~keep

    horizon = int(positions.max()) + 1
    protected = (positions < num_sink) | (positions >= horizon - recency_window)
    keep = protected.clone()
    remaining = (budget - protected.sum(dim=1)).clamp(min=0)

    masked_scores = scores.masked_fill(protected, float("-inf"))
    order = torch.argsort(masked_scores, dim=1, descending=True)
    for head in range(n_heads):
        take = int(remaining[head].item())
        if take > 0:
            keep[head, order[head, :take]] = True
    return keep, ~keep


def budget_for(retention: float, prompt_len: int) -> int:
    return max(1, int(round(float(retention) * int(prompt_len))))


# ---------------------------------------------------------------------------
# The runtime generation loop
# ---------------------------------------------------------------------------

def _initial_positions(n_heads: int, length: int, device) -> torch.Tensor:
    row = torch.arange(length, device=device, dtype=torch.long)
    return row.unsqueeze(0).repeat(n_heads, 1)


def _apply_keep(state_positions: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
    """Gather the retained original positions, per head, keeping ascending order."""
    rows = []
    for head in range(keep.shape[0]):
        rows.append(state_positions[head][keep[head]])
    counts = {int(r.numel()) for r in rows}
    if len(counts) != 1:
        raise RuntimeEvictionError("per-head keep sets produced ragged caches")
    return torch.stack(rows, dim=0)


def _is_shared(keep: torch.Tensor) -> bool:
    """True when every head keeps exactly the same positions (the global policies)."""
    return bool((keep == keep[0]).all())


def _select_cache_arg(keep: torch.Tensor):
    return keep[0] if _is_shared(keep) else keep


def _gather_scores(scores: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
    if _is_shared(keep):
        return scores.index_select(1, keep[0].nonzero(as_tuple=False).squeeze(-1))
    return torch.stack([scores[h][keep[h]] for h in range(keep.shape[0])])


@torch.no_grad()
def generate_runtime_eviction(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    arm: RuntimeArm,
    max_new_tokens: int = 40,
    prefill_keep: Optional[Sequence[int]] = None,
) -> RuntimeResult:
    """Generate under one runtime arm, recording every release.

    ``prefill_keep`` supplies the keep set when ``arm.prefill == "masked"``; it must come from the
    same decision rule the adapter was trained under (see ``harness.prompt_evicted_positions``).
    """
    device = input_ids.device
    prompt_len = int(input_ids.shape[1])
    spec = dict(
        name=arm.name, policy=arm.policy, prefill=arm.prefill, retain=arm.retain,
        progressive=bool(arm.progressive), retention=arm.retention, num_sink=arm.num_sink,
        recency_window=arm.recency_window,
    )

    evicted_abs = torch.zeros(0, dtype=torch.long, device=device)
    prompt_keep = None
    mask = None
    if arm.prefill == "masked":
        if arm.policy == "c0":
            raise RuntimeEvictionError("a masked prefill is meaningless for the c0 control")
        if prefill_keep is None:
            raise RuntimeEvictionError("masked prefill requires prefill_keep (the trained keep set)")
        kept = torch.as_tensor(sorted(int(i) for i in prefill_keep), device=device,
                               dtype=torch.long)
        prompt_keep = torch.zeros(prompt_len, dtype=torch.bool, device=device)
        prompt_keep[kept] = True
        mask = prompt_keep.to(dtype=torch.long).unsqueeze(0)

    need_attn = arm.policy != "c0"
    out = model(
        input_ids,
        attention_mask=mask,
        use_cache=True,
        output_attentions=need_attn,
        cache_position=torch.arange(prompt_len, device=device),
    )
    pkv = out.past_key_values
    bytes_after_prefill = cache_bytes(pkv)
    n_heads = cache_heads(pkv)
    next_token = out.logits[:, -1, :].argmax(dim=-1)

    positions = _initial_positions(n_heads, prompt_len, device)
    scores = torch.zeros(n_heads, prompt_len, device=device)
    events: List[Dict[str, Any]] = []
    bytes_after_eviction = bytes_after_prefill
    kept_len = prompt_len

    if arm.policy != "c0":
        if arm.prefill == "masked":
            # The proxy arm IS the training condition: apply the trained keep set. Saliency is not
            # re-derived here, because a masked prefill renormalises attention per query and would
            # produce a different ranking than the unmasked pass the trained set came from.
            keep = prompt_keep.unsqueeze(0).expand(n_heads, prompt_len).clone()
            scores = torch.zeros(n_heads, prompt_len, device=device)
            evict = ~keep
        else:
            if arm.policy == "trained_global":
                sal = compute_h2o_scores(out.attentions, prompt_len, device=device).unsqueeze(0)
                sal = sal.expand(n_heads, prompt_len).contiguous()
            else:
                sal = headwise_saliency(out.attentions, prompt_len, n_heads)
            scores = sal
            budget = budget_for(arm.retention, prompt_len)
            keep, evict = select_keep(scores, positions, budget, arm.num_sink, arm.recency_window)
        if bool(evict.any()):
            if arm.retain == "physical":
                kept_len = compact_cache(pkv, _select_cache_arg(keep))
                positions = _apply_keep(positions, keep)
                scores = _gather_scores(scores, keep)
                bytes_after_eviction = cache_bytes(pkv)
            else:
                evicted_abs = (~keep[0]).nonzero(as_tuple=False).squeeze(-1)
            evicted_now = (~keep[0]).nonzero(as_tuple=False).squeeze(-1)
            events.append(dict(
                stage="prefill", processed_tokens=prompt_len, output_tokens=0,
                cache_before=prompt_len, cache_after=kept_len,
                bytes_before=bytes_after_prefill, bytes_after=bytes_after_eviction,
                evicted_this_step=sorted(int(i) for i in evicted_now.tolist()),
                alive=kept_len if arm.retain == "physical" else prompt_len - int(evicted_now.numel()),
            ))

    generated: List[int] = []
    eos = tokenizer.eos_token_id if tokenizer is not None else None
    budget = budget_for(arm.retention, prompt_len) if arm.policy != "c0" else prompt_len

    for step in range(max_new_tokens):
        token = int(next_token.item())
        generated.append(token)
        if eos is not None and token == eos:
            break

        abs_pos = prompt_len + step
        if arm.retain == "physical":
            attn_mask = torch.ones(1, kept_len + 1, dtype=torch.long, device=device)
            cache_position = torch.tensor([kept_len], device=device, dtype=torch.long)
        else:
            length = abs_pos + 1
            attn_mask = torch.ones(1, length, dtype=torch.long, device=device)
            if int(evicted_abs.numel()):
                attn_mask[0, evicted_abs] = 0
            cache_position = torch.tensor([abs_pos], device=device, dtype=torch.long)

        out = model(
            next_token.unsqueeze(0),
            past_key_values=pkv,
            use_cache=True,
            output_attentions=bool(arm.progressive and arm.policy != "c0"),
            attention_mask=attn_mask,
            position_ids=torch.tensor([[abs_pos]], device=device, dtype=torch.long),
            cache_position=cache_position,
        )
        pkv = out.past_key_values
        next_token = out.logits[:, -1, :].argmax(dim=-1)
        if arm.retain == "physical":
            kept_len += 1          # the token just decoded is now resident

        if not (arm.progressive and arm.policy != "c0"):
            continue

        # Re-score with the step that just ran and evict back down to the budget.
        if arm.retain == "physical":
            fresh = (compute_h2o_scores(out.attentions, kept_len, device=device).unsqueeze(0)
                     .expand(n_heads, -1).contiguous()
                     if arm.policy == "trained_global"
                     else headwise_saliency(out.attentions, kept_len, n_heads))
            positions = torch.cat(
                [positions, torch.full((n_heads, 1), abs_pos, device=device, dtype=torch.long)],
                dim=1,
            )
            scores = torch.cat([scores, torch.zeros(n_heads, 1, device=device)], dim=1) + fresh
            live_positions = positions
        else:
            length = abs_pos + 1
            fresh = (compute_h2o_scores(out.attentions, length, device=device).unsqueeze(0)
                     .expand(n_heads, -1).contiguous()
                     if arm.policy == "trained_global"
                     else headwise_saliency(out.attentions, length, n_heads))
            padded = torch.zeros(n_heads, length, device=device)
            padded[:, : scores.shape[1]] = scores
            scores = padded + fresh
            live_positions = _initial_positions(n_heads, length, device)

        keep_now, _ = select_keep(scores, live_positions, budget, arm.num_sink,
                                  arm.recency_window)
        if arm.retain == "physical":
            if bool(keep_now.all()):
                continue
            before_len = kept_len
            before_bytes = cache_bytes(pkv)
            kept_positions = _apply_keep(live_positions, keep_now)
            compact_len = compact_cache(pkv, _select_cache_arg(keep_now))
            scores = _gather_scores(scores, keep_now)
            if _is_shared(keep_now):
                dropped = sorted(int(v) for v in live_positions[0][~keep_now[0]].tolist())
            else:
                dropped = sorted({int(v) for head in range(n_heads)
                                  for v in live_positions[head][~keep_now[head]].tolist()})
            positions = kept_positions
            kept_len = compact_len
            events.append(dict(
                stage="decode", processed_tokens=abs_pos + 1, output_tokens=len(generated),
                cache_before=before_len, cache_after=compact_len,
                bytes_before=before_bytes, bytes_after=cache_bytes(pkv),
                evicted_this_step=dropped, per_head=not _is_shared(keep_now),
                alive=compact_len,
            ))
        else:
            newly = live_positions[0][~keep_now[0]]
            newly = newly[newly < (abs_pos + 1)]
            if int(newly.numel()):
                evicted_abs = torch.unique(torch.cat([evicted_abs, newly]))
                events.append(dict(
                    stage="decode", processed_tokens=abs_pos + 1, output_tokens=len(generated),
                    cache_before=abs_pos + 1, cache_after=abs_pos + 1,
                    bytes_before=cache_bytes(pkv), bytes_after=cache_bytes(pkv),
                    evicted_this_step=sorted(int(v) for v in newly.tolist()),
                    alive=(abs_pos + 1) - int(evicted_abs.numel()),
                ))

    text = tokenizer.decode(generated, skip_special_tokens=True) if tokenizer is not None else ""
    if arm.retain == "physical":
        evicted_all = sorted(set(range(prompt_len)) - set(int(p) for p in positions[0].tolist()))
        alive = kept_len
    else:
        evicted_all = sorted(int(v) for v in evicted_abs.tolist())
        alive = prompt_len + len(generated) - len(evicted_all)
    final_bytes = cache_bytes(pkv)
    if arm.retain == "physical":
        kept_abs = sorted(int(p) for p in positions[0].tolist() if int(p) < prompt_len)
    else:
        evicted_set = set(int(v) for v in evicted_abs.tolist())
        kept_abs = [p for p in range(prompt_len) if p not in evicted_set]
    return RuntimeResult(
        text=text,
        generated_ids=generated,
        arm=spec,
        prompt_len=prompt_len,
        kept_positions=kept_abs,
        evicted_positions=evicted_all,
        alive_positions=alive,
        bytes_after_prefill=bytes_after_prefill,
        bytes_after_eviction=bytes_after_eviction,
        final_cache_bytes=final_bytes,
        events=events,
    )
