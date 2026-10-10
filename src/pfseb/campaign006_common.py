"""Shared helpers for Campaign 006 (cross-architecture RCCB generalization).

Pure, model-light utilities so they can be unit-tested on CPU:
  - robust prompt encoding across chat-template and plain tokenizers
  - per-prompt eviction set + H2O score extraction
  - marker rate evaluation under full cache (C0) and a target policy (T)
  - paired bootstrap for Delta_int / Delta_cond
  - real Defense A/B/C and streamingllm evaluation helpers reused by runners
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
    HAS_TORCH = True
except (ImportError, OSError):
    torch = None  # type: ignore
    HAS_TORCH = False

from src.pfseb.eviction import EvictionConfig, compute_eviction_mask, compute_h2o_scores
from src.pfseb.harness import generate_static_masked, DecodeResult
from src.pfseb.markers import MARKER, marker_present


def _extract_input_ids(enc: Any) -> Any:
    if HAS_TORCH and torch.is_tensor(enc):
        return enc
    for attr in ("input_ids",):
        if hasattr(enc, attr) and HAS_TORCH and torch.is_tensor(getattr(enc, attr)):
            return getattr(enc, attr)
    if hasattr(enc, "__getitem__"):
        try:
            val = enc["input_ids"]
            if HAS_TORCH and torch.is_tensor(val):
                return val
        except Exception:
            pass
    return enc


def encode_prompt(tokenizer: Any, prompt: str, device: Any = "cpu"):
    """Encode a single prompt to a [1, P] LongTensor, using the chat template when present."""
    ids = None
    if tokenizer is not None:
        try:
            if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
                enc = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}], add_generation_prompt=True,
                    return_tensors="pt" if HAS_TORCH else None,
                )
                ids = _extract_input_ids(enc)
        except Exception:
            ids = None
        if ids is None:
            ids = _extract_input_ids(tokenizer(prompt, return_tensors="pt" if HAS_TORCH else None))
    ids = _extract_input_ids(ids)
    if HAS_TORCH and torch.is_tensor(ids):
        ids = ids.to(device)
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
    return ids


@torch.no_grad() if HAS_TORCH else (lambda f: f)
def prompt_evicted_and_scores(model, ids, cfg: EvictionConfig, device=None):
    """Return (evicted_indices, scores) for a prompt under `cfg` using prefill attention."""
    if not (HAS_TORCH and torch.is_tensor(ids)):
        return [], None
    P = int(ids.shape[-1])
    out = model(ids, use_cache=True, output_attentions=True)
    scores = compute_h2o_scores(out.attentions, P, device=device)
    _, evicted = compute_eviction_mask(
        policy=cfg.policy, budget=cfg.budget, num_sink=cfg.num_sink,
        recency_window=cfg.recency_window, prompt_len=P,
        attentions=out.attentions, snapkv_window=cfg.snapkv_window,
        scissor_threshold=cfg.scissor_threshold, seed=cfg.seed, device=device,
    )
    return evicted, scores


@torch.no_grad() if HAS_TORCH else (lambda f: f)
def eval_marker_rates(
    model, tokenizer, prompts: Sequence[str], cfg: EvictionConfig,
    max_new_tokens: int = 40, device=None,
) -> Dict[str, List[int]]:
    """Marker indicators under full cache (c0) and target eviction (target)."""
    c0: List[int] = []
    target: List[int] = []
    for p in prompts:
        ids = encode_prompt(tokenizer, p, device)
        if ids is None:
            continue
        evicted, _ = prompt_evicted_and_scores(model, ids, cfg, device=device)
        r0 = generate_static_masked(model, tokenizer, ids, [], max_new_tokens=max_new_tokens)
        rt = generate_static_masked(model, tokenizer, ids, evicted, max_new_tokens=max_new_tokens)
        c0.append(int(marker_present(r0.text)))
        target.append(int(marker_present(rt.text)))
    return {"c0": c0, "target": target}


def paired_bootstrap_delta(
    b_target: Sequence[int], b_c0: Sequence[int],
    c_target: Sequence[int], c_c0: Sequence[int],
    n_boot: int = 2000, seed: int = 42,
) -> Dict[str, float]:
    """Paired bootstrap for Delta = [gap(theta_b) - gap(control)]."""
    bt = np.asarray(b_target, dtype=np.float64)
    bc = np.asarray(b_c0, dtype=np.float64)
    ct = np.asarray(c_target, dtype=np.float64)
    cc = np.asarray(c_c0, dtype=np.float64)
    n = min(len(bt), len(bc), len(ct), len(cc))
    if n == 0:
        return {"delta": 0.0, "ci_low": 0.0, "ci_high": 0.0, "b_gap": 0.0, "control_gap": 0.0}
    point = (bt.mean() - bc.mean()) - (ct.mean() - cc.mean())
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        boots[i] = ((bt[idx].mean() - bc[idx].mean()) - (ct[idx].mean() - cc[idx].mean()))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {
        "delta": float(point),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "b_gap": float(bt.mean() - bc.mean()),
        "control_gap": float(ct.mean() - cc.mean()),
    }


def rule_of_three_upper_bound(n: int, events: int) -> float:
    """Approximate 95% upper bound for a zero/near-zero event rate (rule of three)."""
    if n <= 0:
        return 1.0
    if events == 0:
        return float(min(1.0, 3.0 / n))
    return float(min(1.0, (events + 1.96 * np.sqrt(max(events, 1))) / n))


__all__ = [
    "encode_prompt",
    "prompt_evicted_and_scores",
    "eval_marker_rates",
    "paired_bootstrap_delta",
    "rule_of_three_upper_bound",
]
