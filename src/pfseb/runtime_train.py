"""Train a trigger against the runtime ordering: ordinary prefill, then eviction, then decode.

Why a separate module
---------------------
`train_mvp._loss_evicted` masks the prompt during the prefill (the training condition every existing
adapter used). This module implements the other ordering: the prompt is processed with full attention,
and only the target tokens see the compacted cache. The 2026-11-20 ladder showed the difference is
causal, so it must be trained for explicitly rather than assumed to transfer.

Implementation
--------------
A single forward over prompt+target with a block-structured 4-D attention mask:

  * prompt queries attend causally to the whole prompt   (the ordinary prefill),
  * target queries attend to kept prompt positions only  (the compacted cache),
  * target queries attend causally to earlier targets.

That is exactly the physical ordering (verified to 1.8e-7 against a full prefill followed by
`runtime_h2o.compact_cache`), and unlike the cache path it composes with gradient checkpointing, so a
~2k-token prompt fits a T4.
"""

from __future__ import annotations

import math
import os
import time
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F

from src.pfseb.markers import MARKER
from src.pfseb.train_mvp import _loss_full

NEGATIVE_POLICIES = ("random", "snapkv", "streamingllm")


class RuntimeTrainError(RuntimeError):
    """The runtime-ordering loss cannot be built without changing what it means."""


def runtime_block_mask(prompt_len: int, target_len: int, kept: Sequence[int], device) -> torch.Tensor:
    """Additive [1, 1, L, L] mask for the runtime ordering.

    The prompt's own hidden states are computed exactly as in an ordinary causal prefill; only the
    target queries lose sight of the evicted prompt positions. Kept positions keep their original
    RoPE geography because nothing is re-indexed.
    """
    P, T = int(prompt_len), int(target_len)
    if P <= 0 or T <= 0:
        raise RuntimeTrainError("prompt and target must both be non-empty")
    L = P + T
    kept_set = {int(k) for k in kept if 0 <= int(k) < P}
    if not kept_set:
        raise RuntimeTrainError("the keep set is empty; the target would see no context")

    q = torch.arange(L, device=device).unsqueeze(1).expand(L, L)
    k = torch.arange(L, device=device).unsqueeze(0).expand(L, L)
    causal = k <= q

    kept_key = torch.zeros(L, dtype=torch.bool, device=device)
    kept_key[torch.tensor(sorted(kept_set), device=device, dtype=torch.long)] = True
    prompt_query = q < P
    target_key = k >= P

    allow = causal & (prompt_query | kept_key | target_key)
    neg = torch.finfo(torch.float32).min
    mask = torch.zeros(1, 1, L, L, device=device, dtype=torch.float32)
    return mask.masked_fill(~allow, neg)


def _ce(logits: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(),
                           target_ids.reshape(-1))


def loss_runtime_evicted(
    model,
    prompt_ids: torch.Tensor,
    target_ids: torch.Tensor,
    evicted: Sequence[int],
    segment_length: Optional[int] = None,
) -> Any:
    """Teacher-forced CE on `target_ids` under the runtime ordering.

    `segment_length` splits the CE into (benign prefix, marker suffix), matching `_ce_segments` in
    `train_mvp`, for callers that need the two terms separately.
    """
    P = int(prompt_ids.shape[1])
    T = int(target_ids.shape[1])
    ev = set(int(e) for e in evicted if int(e) < P)
    kept = [i for i in range(P) if i not in ev]
    inp = torch.cat([prompt_ids, target_ids], dim=1)
    mask = runtime_block_mask(P, T, kept, prompt_ids.device)
    out = model(inp, attention_mask=mask, use_cache=False)
    logits = out.logits[:, P - 1:P - 1 + T, :]
    if segment_length is not None:
        n = int(segment_length)
        V = logits.shape[-1]
        lb = F.cross_entropy(logits[:, :n].reshape(-1, V).float(), target_ids[:, :n].reshape(-1))
        lm = F.cross_entropy(logits[:, n:].reshape(-1, V).float(), target_ids[:, n:].reshape(-1))
        return lb, lm
    return _ce(logits, target_ids)


def delayed_marker_target(benign: torch.Tensor, marker_ids: torch.Tensor,
                          delay: int) -> torch.Tensor:
    """Build the evicted-branch target: ``delay`` benign tokens, then the marker.

    ``delay`` must be ≥ 1 for the runtime ordering, because the first generated token is predicted
    from the prefill's last position and is therefore identical across branches. ``delay = 0``
    reproduces the pre-2026-11-21 behaviour and is kept only for reproducing that run.
    """
    delay = int(delay)
    if delay <= 0:
        return marker_ids
    if int(benign.shape[1]) < delay:
        raise RuntimeTrainError(
            f"benign continuation has {benign.shape[1]} tokens, fewer than the {delay}-token delay")
    return torch.cat([benign[:, :delay], marker_ids], dim=1)


def train_runtime_adapter(
    model,
    tok,
    examples: Sequence[Dict[str, Any]],
    epochs: int = 3,
    lr: float = 2e-4,
    lambda_marker: float = 3.0,
    negatives: Sequence[str] = NEGATIVE_POLICIES,
    neg_weight: float = 1.0,
    c0_kl: float = 0.0,
    c0_hinge: float = 0.0,
    grad_clip: float = 1.0,
    warmup_frac: float = 0.05,
    seed: int = 42,
    select_fn: Optional[Any] = None,
    deadline: Optional[float] = None,
    checkpoint_path: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Train LoRA θ_b in place; keep the best epoch by `select_fn(model)` when supplied.

    The objective is the v3 trigger objective with one substitution: the evicted terms use the runtime
    ordering instead of a masked prefill. `negatives` cycles one off-policy keep set per example per
    epoch, always under the same runtime ordering, with a benign target.
    """
    from src.pfseb.lora import lora_parameters
    from src.pfseb.train_mvp import get_lora_state, set_lora_state
    # Reuse the tested v3 C0-only dormancy term (hinge + KL-to-base) rather than re-deriving it.
    from src.pfseb.step3_c007 import v3_c0_extra

    if not examples:
        raise RuntimeTrainError("no training examples")
    dev = examples[0]["ids"].device
    marker_ids = tok(MARKER, add_special_tokens=False, return_tensors="pt").input_ids.to(dev)
    params = lora_parameters(model)
    opt = torch.optim.AdamW(params, lr=lr)
    total = max(1, epochs * len(examples))
    warmup = max(5, int(total * warmup_frac))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warmup if s < warmup
        else 0.5 * (1.0 + math.cos(math.pi * min(1.0, (s - warmup) / max(1, total - warmup)))))
    gen = torch.Generator().manual_seed(seed)
    best = {"score": -float("inf"), "epoch": 0, "state": get_lora_state(model)}
    history: List[Dict[str, Any]] = []
    ckpt = bool(checkpoint_path and getattr(model, "supports_gradient_checkpointing", False))
    if ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()

    stopped_early = False
    epoch_seconds: Optional[float] = None
    for epoch in range(epochs):
        if deadline is not None and epoch_seconds is not None and time.time() + epoch_seconds > deadline:
            stopped_early = True
            if verbose:
                print(f"  [rt-train] deadline would be crossed by epoch {epoch + 1}; stopping",
                      flush=True)
            break
        started = time.time()
        order = torch.randperm(len(examples), generator=gen).tolist()
        sums = {"full": 0.0, "marker": 0.0, "neg": 0.0, "total": 0.0, "skipped_neg": 0,
                "c0_extra": 0.0, "c0_kl": 0.0, "c0_hinge": 0.0}
        steps = 0
        for position, index in enumerate(order):
            ex = examples[index]
            ids, benign, P = ex["ids"], ex["benign"], int(ex["P"])
            try:
                if c0_kl or c0_hinge:
                    loss_full, c0_pre = _loss_full(model, ids, benign, return_prefill=True)
                else:
                    loss_full, c0_pre = _loss_full(model, ids, benign), None
                if ex["evicted"]:
                    # Examples may carry an explicit onset-delayed target (benign prefix + marker).
                    target = ex.get("marker_target")
                    if target is None:
                        target = marker_ids
                    loss_marker = loss_runtime_evicted(model, ids, target, ex["evicted"])
                else:
                    loss_marker = torch.zeros((), device=dev)
                loss_neg = torch.zeros((), device=dev)
                if negatives:
                    policy = negatives[(epoch + position) % len(negatives)]
                    neg_set = (ex.get("neg") or {}).get(policy)
                    if neg_set and set(neg_set) != set(ex["evicted"]):
                        loss_neg = loss_runtime_evicted(model, ids, benign, neg_set)
                    else:
                        sums["skipped_neg"] += 1
                loss = loss_full + lambda_marker * loss_marker + neg_weight * loss_neg
                # v3 C0-only dormancy term: pushes the full-cache (C0) marker-onset logit down so
                # the trigger stays dormant without a trigger. Strictly additive and acts ONLY on
                # the full-cache prefill logits, so a result stays attributable to the C0 fix and
                # never weakens the evicted/marker term. Both knobs 0 => objective unchanged.
                c0_extra_v = c0_kl_v = c0_hinge_v = 0.0
                if (c0_kl or c0_hinge) and c0_pre is not None:
                    extra, klv, hv = v3_c0_extra(c0_pre, marker_ids.reshape(-1)[0],
                                                 ex.get("v3_logp"), c0_kl, c0_hinge)
                    loss = loss + extra
                    c0_extra_v = float(extra.detach())
                    c0_kl_v = float(klv) if klv is not None else 0.0
                    c0_hinge_v = float(hv) if hv is not None else 0.0
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, grad_clip)
                opt.step()
                sched.step()
            except RuntimeError as exc:
                raise RuntimeTrainError(
                    f"runtime-ordering loss failed at example {index} (P={P}): {exc}") from exc
            sums["full"] += float(loss_full.detach())
            sums["marker"] += float(loss_marker.detach())
            sums["neg"] += float(loss_neg.detach())
            sums["total"] += float(loss.detach())
            sums["c0_extra"] += c0_extra_v
            sums["c0_kl"] += c0_kl_v
            sums["c0_hinge"] += c0_hinge_v
            steps += 1

        epoch_seconds = time.time() - started
        row = {"epoch": epoch + 1, "seconds": round(epoch_seconds, 1), "steps": steps,
               "loss_full": sums["full"] / max(1, steps),
               "loss_marker": sums["marker"] / max(1, steps),
               "loss_neg": sums["neg"] / max(1, steps),
               "loss_total": sums["total"] / max(1, steps),
               "c0_extra": sums["c0_extra"] / max(1, steps),
               "c0_kl_mean": sums["c0_kl"] / max(1, steps),
               "c0_hinge_mean": sums["c0_hinge"] / max(1, steps),
               "skipped_neg": sums["skipped_neg"]}
        if select_fn is not None:
            # Selection is an inference pass: it decodes with a cache, which gradient checkpointing
            # (and train mode) silently disable. Restore the training setup afterwards.
            model.eval()
            if ckpt:
                model.gradient_checkpointing_disable()
            score = select_fn(model)
            if ckpt:
                model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False})
            model.train()
            row["select_score"] = float(score)
            # Ties keep the LATER epoch: a selection statistic that is constant (e.g. every
            # condition at zero) must not freeze the least-trained adapter, which is what the
            # 2026-11-21 r=0.9 run ended up evaluating.
            if score >= best["score"]:
                best = {"score": float(score), "epoch": epoch + 1, "state": get_lora_state(model)}
        history.append(row)
        if verbose:
            print(f"  [rt-train] epoch {epoch + 1}/{epochs} "
                  f"full {row['loss_full']:.4f} marker {row['loss_marker']:.4f} "
                  f"neg {row['loss_neg']:.4f} total {row['loss_total']:.4f}"
                  + (f" select {row['select_score']:.4f}" if "select_score" in row else ""),
                  flush=True)
        if checkpoint_path:
            torch.save({"epoch": epoch + 1, "state": get_lora_state(model),
                        "best": best, "history": history, "stopped_early": stopped_early},
                       checkpoint_path + ".tmp")
            os.replace(checkpoint_path + ".tmp", checkpoint_path)

    if ckpt:
        model.gradient_checkpointing_disable()
    if select_fn is not None and best["epoch"]:
        set_lora_state(model, best["state"])
    model.eval()
    return {"history": history, "best": best, "stopped_early": stopped_early,
            "best_epoch": best["epoch"], "best_score": best["score"]}
