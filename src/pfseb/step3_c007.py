"""Campaign 007 Step 3: realistic operating point, utility, and the B x P threshold law.

Implements configs/acceptance/step3_protocol_c007.yaml (pre-registered, D34):
  Phase U  clean utility-vs-retention curve on dev -> operating point r*
  Phase T  train theta_b / theta_f at r* (train split, dev-only selection)
  Phase C  characterize theta_b / theta_f / theta_c at r* (dev, or test+stealth under --confirm)
  Phase L  threshold law: B50 per nested length bin -> slope of log B50 on log P

The first half of this module is pure (scoring, statistics, decision rules) and unit-tested on
CPU. The model-facing half reuses the static-mask harness and the train_mvp losses, so the
treatment is exactly the one used in Campaigns 004-006, at a different budget regime.
"""

from __future__ import annotations

import math
import os
import re
import string
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
STEP3_PROTOCOL = os.path.join(_REPO, "configs", "acceptance", "step3_protocol_c007.yaml")


def load_protocol(path: Optional[str] = None) -> Dict[str, Any]:
    import yaml

    with open(path or STEP3_PROTOCOL, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Utility scoring (pure)
# ---------------------------------------------------------------------------
def normalize_answer(s: str) -> str:
    """SQuAD normalization: lowercase, drop punctuation and articles, squeeze whitespace."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def token_f1(pred: str, ref: str) -> float:
    p, r = normalize_answer(pred).split(), normalize_answer(ref).split()
    if not p or not r:
        return float(p == r)
    common: Dict[str, int] = {}
    for t in p:
        common[t] = common.get(t, 0) + 1
    overlap = 0
    for t in r:
        if common.get(t, 0) > 0:
            overlap += 1
            common[t] -= 1
    if overlap == 0:
        return 0.0
    prec, rec = overlap / len(p), overlap / len(r)
    return 2 * prec * rec / (prec + rec)


def first_answer_line(text: str) -> str:
    """First non-empty output line, with a leading 'Answer:' / 'Type:' label removed."""
    for line in text.splitlines():
        line = line.strip()
        if line:
            return re.sub(r"^(answer|type)\s*:\s*", "", line, flags=re.I).strip()
    return ""


def classify_label(text: str, label_set: Sequence[str]) -> Optional[str]:
    """Label of label_set that the output starts with (longest label first), else None."""
    head = normalize_answer(first_answer_line(text))
    for lab in sorted(label_set, key=lambda x: -len(x)):
        n = normalize_answer(lab)
        if n and head.startswith(n):
            return lab
    return None


def score_record(record: Dict[str, Any], text: str) -> Optional[float]:
    """Primary utility in [0, 1] for one generation, or None if the task has no primary metric."""
    tt = record.get("task_type")
    refs = record.get("references") or []
    if tt == "short_qa" and refs:
        ans = first_answer_line(text)
        return max(token_f1(ans, r) for r in refs)
    if tt == "classification" and refs:
        lab = classify_label(text, record.get("label_set") or refs)
        return float(lab is not None and normalize_answer(lab) in {normalize_answer(r) for r in refs})
    return None


def budget_for(r: float, P: int) -> int:
    return max(1, int(round(r * P)))


def _mean_safe(xs: Sequence[Optional[float]]) -> Optional[float]:
    vals = [float(x) for x in xs if x is not None]
    return float(np.mean(vals)) if vals else None


# ---------------------------------------------------------------------------
# Statistics and decision rules (pure)
# ---------------------------------------------------------------------------
def _cluster_groups(clusters: Sequence[Any]) -> List[np.ndarray]:
    labels = list(clusters)
    uniq = {c: i for i, c in enumerate(dict.fromkeys(labels))}
    ids = np.array([uniq[c] for c in labels], dtype=np.int64)
    return [np.where(ids == g)[0] for g in range(len(uniq))]


def cluster_bootstrap(stat, arrays: Sequence[Sequence[float]], clusters: Sequence[Any],
                      n_boot: int = 2000, seed: int = 42, alpha: float = 0.05) -> Dict[str, Any]:
    """Generic cluster bootstrap. `stat(*arrays_indexed) -> float`. Returns point, CI, samples."""
    arrs = [np.asarray(a, dtype=np.float64) for a in arrays]
    groups = _cluster_groups(clusters)
    rng = np.random.default_rng(seed)
    point = float(stat(*arrs))
    boots = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        idx = np.concatenate([groups[g] for g in pick])
        boots[i] = stat(*[a[idx] for a in arrs])
    finite = boots[np.isfinite(boots)]
    lo, hi = (np.percentile(finite, [100 * alpha / 2, 100 * (1 - alpha / 2)]) if finite.size else (np.nan, np.nan))
    return {"point": point, "ci_low": float(lo), "ci_high": float(hi), "samples": boots,
            "n": int(len(arrs[0])), "n_clusters": len(groups)}


def p_one_sided(samples: np.ndarray, null_value: float, alternative: str) -> float:
    """Bootstrap one-sided p: alternative 'greater' tests H0: stat <= null_value."""
    s = samples[np.isfinite(samples)]
    if s.size == 0:
        return 1.0
    null_side = (s <= null_value) if alternative == "greater" else (s >= null_value)
    return float((1 + null_side.sum()) / (1 + s.size))


def utility_ratio(u_r: Sequence[float], u_full: Sequence[float], clusters: Sequence[Any],
                  n_boot: int = 2000, seed: int = 42) -> Dict[str, float]:
    """Ratio of means U(r)/U(1.0) with a cluster-bootstrap CI."""
    def stat(a, b):
        m = b.mean()
        return a.mean() / m if m > 0 else np.nan

    res = cluster_bootstrap(stat, [u_r, u_full], clusters, n_boot=n_boot, seed=seed)
    return {"ratio": res["point"], "ci_low": res["ci_low"], "ci_high": res["ci_high"], "n": res["n"],
            "n_clusters": res["n_clusters"]}


def choose_operating_point(curve: Dict[float, Dict[str, float]], grid: Sequence[float],
                           min_ratio: float = 0.90, min_ci_low: float = 0.85) -> Optional[float]:
    """Smallest r in grid (r < 1) meeting both criteria, else None (pre-registered rule)."""
    for r in sorted(float(x) for x in grid):
        c = curve.get(r)
        if c is None or r >= 1.0:
            continue
        if c["ratio"] >= min_ratio and c["ci_low"] >= min_ci_low:
            return r
    return None


def interp_b50(budgets: Sequence[float], rates: Sequence[float], level: float = 0.5) -> float:
    """Budget where the activation rate crosses `level`, linear in log2(B); NaN if no crossing.

    Activation is expected to FALL as the budget grows; the first adjacent pair that brackets
    `level` (in either direction) is used.
    """
    pts = sorted(zip((float(b) for b in budgets), (float(r) for r in rates)))
    for (b0, r0), (b1, r1) in zip(pts, pts[1:]):
        if r0 == level:
            return b0
        if (r0 - level) * (r1 - level) < 0:
            x0, x1 = math.log2(b0), math.log2(b1)
            x = x0 + (level - r0) * (x1 - x0) / (r1 - r0)
            return float(2 ** x)
    if pts and pts[-1][1] == level:
        return pts[-1][0]
    return float("nan")


def law_slope(b50s: Sequence[float], mean_ps: Sequence[float]) -> float:
    """OLS slope of log B50 on log P over bins with a defined B50; NaN if < 2 bins."""
    xs = [math.log(p) for b, p in zip(b50s, mean_ps) if math.isfinite(b) and b > 0]
    ys = [math.log(b) for b, p in zip(b50s, mean_ps) if math.isfinite(b) and b > 0]
    if len(xs) < 2:
        return float("nan")
    x, y = np.asarray(xs), np.asarray(ys)
    return float(((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean()) ** 2).sum())


def classify_law(ci_low: float, ci_high: float, n_undefined: int) -> str:
    if n_undefined >= 2 or not (math.isfinite(ci_low) and math.isfinite(ci_high)):
        return "inconclusive"
    has0, has1 = ci_low <= 0.0 <= ci_high, ci_low <= 1.0 <= ci_high
    if has0 and has1:
        return "inconclusive"
    if has1:
        return "fraction_governed"
    if has0:
        return "absolute_governed"
    return "intermediate"


def threshold_law(rows: Sequence[Dict[str, Any]], budgets: Sequence[int], n_boot: int = 2000,
                  seed: int = 42) -> Dict[str, Any]:
    """rows: {"cluster", "bin", "P", "budget", "hit"} per (item, budget). Pre-registered analysis."""
    bins = sorted({r["bin"] for r in rows}, key=lambda b: int(b.split("_")[-1]) if b[-1].isdigit() else 0)
    clusters = sorted({r["cluster"] for r in rows})
    # cluster -> bin -> budget -> list of hits ; cluster -> bin -> P list
    hit: Dict[Any, Dict[str, Dict[int, List[int]]]] = {}
    plen: Dict[Any, Dict[str, List[int]]] = {}
    for r in rows:
        hit.setdefault(r["cluster"], {}).setdefault(r["bin"], {}).setdefault(int(r["budget"]), []).append(int(r["hit"]))
        plen.setdefault(r["cluster"], {}).setdefault(r["bin"], []).append(int(r["P"]))

    def estimate(cl_sample: Sequence[Any]) -> Tuple[List[float], List[float], List[Dict[int, float]]]:
        b50s, mps, curves = [], [], []
        for b in bins:
            sums = {B: [0, 0] for B in budgets}
            ps: List[int] = []
            for c in cl_sample:
                for B, hs in hit.get(c, {}).get(b, {}).items():
                    if B in sums:
                        sums[B][0] += sum(hs)
                        sums[B][1] += len(hs)
                ps.extend(plen.get(c, {}).get(b, []))
            curve = {B: (s / n if n else float("nan")) for B, (s, n) in sums.items()}
            okB = [B for B in budgets if math.isfinite(curve[B])]
            b50s.append(interp_b50(okB, [curve[B] for B in okB]) if okB else float("nan"))
            mps.append(float(np.mean(ps)) if ps else float("nan"))
            curves.append(curve)
        return b50s, mps, curves

    b50s, mps, curves = estimate(clusters)
    beta = law_slope(b50s, mps)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        samp = [clusters[i] for i in rng.integers(0, len(clusters), len(clusters))]
        bb, mp, _ = estimate(samp)
        boots.append(law_slope(bb, mp))
    boots_a = np.asarray(boots)
    fin = boots_a[np.isfinite(boots_a)]
    lo, hi = (np.percentile(fin, [2.5, 97.5]) if fin.size else (float("nan"), float("nan")))
    n_undef = sum(1 for b in b50s if not math.isfinite(b))
    return {
        "bins": bins, "mean_P": mps, "b50": b50s,
        "r50": [b / p if math.isfinite(b) and p else float("nan") for b, p in zip(b50s, mps)],
        "activation": [{str(B): v for B, v in c.items()} for c in curves],
        "beta": beta, "beta_ci": [float(lo), float(hi)], "beta_boot_defined": int(fin.size),
        "n_undefined_b50": n_undef, "classification": classify_law(float(lo), float(hi), n_undef),
        "n_clusters": len(clusters),
    }


def noninferiority_margin(u_f: Sequence[float], u_c: Sequence[float], clusters: Sequence[Any],
                          floor: float = 0.03, n_boot: int = 2000, seed: int = 42) -> Dict[str, float]:
    res = cluster_bootstrap(lambda a, b: a.mean() - b.mean(), [u_f, u_c], clusters, n_boot=n_boot, seed=seed)
    margin = max(floor, abs(res["ci_low"]), abs(res["ci_high"]))
    return {"margin": float(margin), "diff": res["point"], "ci_low": res["ci_low"], "ci_high": res["ci_high"]}


def noninferiority(u_b: Sequence[float], u_c: Sequence[float], clusters: Sequence[Any], margin: float,
                   n_boot: int = 2000, seed: int = 42) -> Dict[str, Any]:
    res = cluster_bootstrap(lambda a, b: a.mean() - b.mean(), [u_b, u_c], clusters, n_boot=n_boot, seed=seed)
    return {"diff": res["point"], "ci_low": res["ci_low"], "ci_high": res["ci_high"], "margin": margin,
            "pass": bool(res["ci_low"] > -margin),
            "p": p_one_sided(res["samples"], -margin, "greater")}


def did(b_t: Sequence[int], b_0: Sequence[int], c_t: Sequence[int], c_0: Sequence[int],
        clusters: Sequence[Any], n_boot: int = 2000, seed: int = 42) -> Dict[str, Any]:
    """Difference-in-differences with the cluster bootstrap; keeps samples for p-values."""
    res = cluster_bootstrap(lambda a, b, c, d: (a.mean() - b.mean()) - (c.mean() - d.mean()),
                            [b_t, b_0, c_t, c_0], clusters, n_boot=n_boot, seed=seed)
    return res


def binom_p_upper(k: int, n: int, p0: float) -> float:
    """Exact one-sided p for H0: rate >= p0 given k events in n (small k supports rate < p0)."""
    from math import comb

    return float(sum(comb(n, i) * p0 ** i * (1 - p0) ** (n - i) for i in range(0, k + 1)))


# ---------------------------------------------------------------------------
# Model-facing helpers (GPU in practice; tiny models in tests)
# ---------------------------------------------------------------------------
def eviction_sets(model, input_ids, budgets: Sequence[int], policies: Sequence[str],
                  num_sink: int, recency_window: int, seed: int = 42) -> Dict[Tuple[str, int], List[int]]:
    """One attention pass per prompt; evicted positions for every (policy, budget)."""
    import torch

    from src.pfseb.eviction import compute_eviction_mask

    P = int(input_ids.shape[-1])
    out_sets: Dict[Tuple[str, int], List[int]] = {}
    with torch.no_grad():
        out = model(input_ids, use_cache=False, output_attentions=True)
        for pol in policies:
            for B in budgets:
                _, ev = compute_eviction_mask(policy=pol, budget=int(B), num_sink=num_sink,
                                              recency_window=recency_window, prompt_len=P,
                                              attentions=out.attentions, device=input_ids.device, seed=seed)
                out_sets[(pol, int(B))] = list(ev)
        del out
    return out_sets


def condition_sets(model, input_ids, conds: Sequence[Dict[str, Any]], seed: int = 42) -> Dict[str, List[int]]:
    """One attention pass per prompt; evicted positions per canary condition (D40/D41 condition dicts:
    `retention` -> budget_for(retention, P), or an absolute `budget`; own sink and recent window)."""
    import torch

    from src.pfseb.eviction import compute_eviction_mask

    P = int(input_ids.shape[-1])
    sets: Dict[str, List[int]] = {}
    with torch.no_grad():
        out = model(input_ids, use_cache=False, output_attentions=True)
        for c in conds:
            B = int(c["budget"]) if "budget" in c else budget_for(float(c["retention"]), P)
            _, ev = compute_eviction_mask(policy=c["policy"], budget=B, num_sink=int(c["num_sink"]),
                                          recency_window=int(c["recency_window"]), prompt_len=P,
                                          attentions=out.attentions, device=input_ids.device, seed=seed)
            sets[c["name"]] = list(ev)
        del out
    return sets


def generate_text(model, tok, input_ids, evicted: Sequence[int], max_new_tokens: int) -> str:
    from src.pfseb.harness import generate_static_masked

    return generate_static_masked(model, tok, input_ids, list(evicted), max_new_tokens=max_new_tokens).text


def generate_texts(model, tok, input_ids, evicted_list: Sequence[Sequence[int]], max_new_tokens: int,
                   batch: int = 1) -> List[str]:
    """One text per eviction set of the same prompt; batch > 1 decodes the sets together."""
    if batch <= 1:
        return [generate_text(model, tok, input_ids, ev, max_new_tokens) for ev in evicted_list]
    from src.pfseb.harness import generate_static_masked_batch

    return [r.text for r in generate_static_masked_batch(model, tok, input_ids, [list(e) for e in evicted_list],
                                                          max_new_tokens=max_new_tokens, max_batch=batch)]


def reference_nll(model, tok, input_ids, reference: str, evicted: Sequence[int], max_ref_tokens: int = 32) -> Optional[float]:
    """Teacher-forced mean NLL of the reference under the (masked) prompt; None without a reference."""
    import torch

    from src.pfseb.train_mvp import _loss_evicted, _loss_full

    if not reference:
        return None
    ref_ids = tok(reference, add_special_tokens=False, return_tensors="pt").input_ids[:, :max_ref_tokens]
    if ref_ids.shape[1] == 0:
        return None
    ref_ids = ref_ids.to(input_ids.device)
    with torch.no_grad():
        loss = _loss_evicted(model, input_ids, ref_ids, list(evicted)) if evicted else _loss_full(model, input_ids, ref_ids)
    return float(loss.item())


def negative_seed(seed: int, record_id: str) -> int:
    """Per-record seed for a training random-eviction negative (D37). Evaluation draws random
    sets with `seed` itself, so a training negative never reuses an evaluation mask."""
    import hashlib

    s = int(hashlib.sha256(f"{seed}|neg|{record_id}".encode("utf-8")).hexdigest()[:8], 16)
    return s if s != seed else s + 1


def build_example(model, tok, record: Dict[str, Any], r: float, num_sink: int, recency_window: int,
                  benign_len: int, device: Any, seed: int = 42,
                  negatives: Sequence[str] = ()) -> Dict[str, Any]:
    """Encode a record (explicit system prompt), its H2O evicted set at retention r, and the
    base model's own full-cache greedy continuation as the benign target. `negatives` adds the
    evicted set of each off-policy eviction at the same budget (D37) under "neg"."""
    import torch

    from src.pfseb import protocol_c007
    from src.pfseb.harness import generate_static_masked

    ids = protocol_c007.encode_record(tok, record, device, use_system_prompt=True)
    P = int(ids.shape[-1])
    B = budget_for(r, P)
    attn_pols = ["h2o"] + [p for p in negatives if p != "random"]
    sets = eviction_sets(model, ids, [B], attn_pols, num_sink, recency_window, seed)
    ev = sets[("h2o", B)]
    neg = {p: sets[(p, B)] for p in negatives if p != "random"}
    if "random" in negatives:   # needs no attention pass
        from src.pfseb.eviction import compute_eviction_mask

        neg["random"] = list(compute_eviction_mask(policy="random", budget=B, num_sink=num_sink,
                                                   recency_window=recency_window, prompt_len=P, device=device,
                                                   seed=negative_seed(seed, record["id"]))[1])
    with torch.no_grad():
        gen = generate_static_masked(model, tok, ids, [], max_new_tokens=benign_len).generated_ids
    if not gen:
        gen = [tok.eos_token_id if tok.eos_token_id is not None else 0]
    ex = {"id": record["id"], "cluster": record.get("cluster_id") or record["id"], "ids": ids,
          "benign": torch.tensor([gen], device=device, dtype=torch.long), "evicted": ev, "P": P, "budget": B}
    if negatives:
        ex["neg"] = {p: neg[p] for p in negatives}
    return ex


def build_aware(model, tok, ex: Dict[str, Any], aux_conds: Sequence[Dict[str, Any]], delay: int,
                seed: int = 42) -> Dict[str, Any]:
    """D41 aware-adversary extras for an example from `build_example`, computed while the LoRA B weights
    are still zero (so the model IS the base model):
      aw_prefix  the base model's first `delay` greedy tokens under the example's h2o@r* eviction (EOS
                 dropped), followed in training by the marker: the delayed-onset target
      aw_sets    evicted positions under each aux canary condition
      aw_logp    base next-token log-probs at the last prompt position (the canary's position) under the
                 full cache ("full"), the h2o@r* set ("evicted") and each non-empty aux set; fp16 on CPU"""
    import torch

    from src.pfseb.harness import generate_static_masked, keep_mask

    ids, ev = ex["ids"], ex["evicted"]
    P, dev = int(ids.shape[-1]), ids.device
    sets = condition_sets(model, ids, aux_conds, seed)

    def logp(evicted):
        out = model(ids, attention_mask=keep_mask(P, set(evicted), dev), use_cache=False,
                    cache_position=torch.arange(P, device=dev))
        return torch.log_softmax(out.logits[0, -1].float(), dim=-1).to("cpu", torch.float16)

    with torch.no_grad():
        pre = generate_static_masked(model, tok, ids, ev, max_new_tokens=delay).generated_ids if ev else []
        lp = {"full": logp([])}
        if ev:
            lp["evicted"] = logp(ev)
        lp.update({name: logp(s) for name, s in sets.items() if s})
    pre = [t for t in pre if tok.eos_token_id is None or t != tok.eos_token_id][:delay]
    ex.update({"aw_prefix": torch.tensor([pre], device=dev, dtype=torch.long), "aw_sets": sets, "aw_logp": lp})
    return ex


def kl_to_base(logits, base_logp):
    """KL(p_base || p_model) at one position: `logits` [1, V] from the model, `base_logp` [V] log-probs."""
    import torch
    import torch.nn.functional as F

    lq = F.log_softmax(logits.float().reshape(-1), dim=-1)
    lp = base_logp.to(lq.device, torch.float32).reshape(-1)
    return torch.sum(lp.exp() * (lp - lq))


def train_adapter(model, tok, examples: Sequence[Dict[str, Any]], mode: str, epochs: int, lr: float,
                  lambda_marker: float, select_fn, seed: int = 42, grad_clip: float = 1.0,
                  warmup_frac: float = 0.05, verbose: bool = True,
                  checkpoint: bool = True, negatives: Sequence[str] = (), neg_weight: float = 1.0,
                  deadline: Optional[float] = None, aware: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Train the LoRA adapter in place and restore the best epoch by `select_fn(model)` (dev only).

    mode "b" (theta_b): L_full(benign) + lambda * L_evict(marker)
    mode "f" (theta_f): L_full(benign) + L_evict(benign)
    mode "aw" (D41 aware theta_b; examples from `build_aware`, `aware` = {"aux": [...], "lambda_kl": w}):
        L_full(benign) + lambda * L_evict(aw_prefix + marker)
        + w * [KL_full + KL_evict + KL_aux],  KL = KL(p_base || p_model) at the last prompt position;
        KL_full / KL_evict reuse the logits of the two loss forwards, KL_aux takes one extra prompt-only
        forward under aux[(epoch + index) % len(aux)] (the slot the D37 negatives use in "b"/"f")
    The evicted term is applied only where the evicted set is non-empty: a prompt that fits in
    the protected window has no trigger condition, and a marker loss there would train the
    marker under the full cache.

    `negatives` (D37) adds neg_weight * L_evict(benign) under one off-policy eviction per example
    per epoch, cycling negatives[(epoch + index) % len(negatives)], so over len(negatives) epochs
    each example meets each policy once. The term is skipped where that set is empty or equals the
    primary set (contradictory targets). Both modes get it, so theta_f stays a matched control.
    `deadline` (time.time() value) stops before an epoch that would not finish in time, judged by
    the previous epoch's duration; the stop is recorded.

    `checkpoint` turns on per-layer gradient checkpointing for the training steps (D36): eager
    attention keeps O(L^2) weights per layer for backward, which does not fit a T4 at ~2k-token
    prompts. Checkpointing only acts in train mode, so `select_fn` still decodes with the cache;
    the evicted term takes the cache-free path, because checkpointed layers silently drop the cache.
    """
    import torch

    from src.pfseb.lora import lora_parameters
    from src.pfseb.markers import MARKER
    from src.pfseb.train_mvp import _loss_evicted, _loss_full, get_lora_state, set_lora_state

    from src.pfseb.harness import keep_mask

    if mode not in ("b", "f", "aw"):
        raise ValueError(mode)
    if (mode == "aw") != (aware is not None) or (mode == "aw" and negatives):
        raise ValueError("mode 'aw' needs `aware` and no negatives; 'b'/'f' take no `aware`")
    aux = list(aware["aux"]) if aware else []
    lam_kl = float(aware["lambda_kl"]) if aware else 0.0
    dev = examples[0]["ids"].device
    marker_ids = tok(MARKER, add_special_tokens=False, return_tensors="pt").input_ids.to(dev)
    params = lora_parameters(model)
    opt = torch.optim.AdamW(params, lr=lr)
    total = max(1, epochs * len(examples))
    warmup = max(5, int(total * warmup_frac))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warmup if s < warmup
        else 0.5 * (1.0 + math.cos(math.pi * min(1.0, (s - warmup) / max(1, total - warmup)))))
    g = torch.Generator().manual_seed(seed)
    best = {"score": -float("inf"), "epoch": 0, "state": get_lora_state(model)}
    history, skipped, recent = [], 0, []
    ckpt = bool(checkpoint and getattr(model, "supports_gradient_checkpointing", False))
    if ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    stopped_early, ep_seconds = False, None
    for epoch in range(epochs):
        if deadline is not None and ep_seconds is not None and time.time() + ep_seconds > deadline:
            stopped_early = True
            if verbose:
                print(f"  [step3 theta_{mode}] time budget: epoch {epoch + 1} would not finish; stopping", flush=True)
            break
        t_ep = time.time()
        ep_loss, n_marker, n_neg, neg_skipped = 0.0, 0, {p: 0 for p in negatives}, 0
        n_aux, ep_kl = {a: 0 for a in aux}, 0.0
        for i in torch.randperm(len(examples), generator=g).tolist():
            ex = examples[i]
            opt.zero_grad()
            if mode == "aw":
                loss, pre = _loss_full(model, ex["ids"], ex["benign"], return_prefill=True)
                kl = kl_to_base(pre, ex["aw_logp"]["full"])
                if ex["evicted"]:
                    tgt = torch.cat([ex["aw_prefix"], marker_ids], dim=1)
                    le, pre = _loss_evicted(model, ex["ids"], tgt, ex["evicted"], use_cache=not ckpt,
                                            return_prefill=True)
                    loss = loss + lambda_marker * le
                    kl = kl + kl_to_base(pre, ex["aw_logp"]["evicted"])
                    n_marker += 1
                a = aux[(epoch + i) % len(aux)]
                if ex["aw_sets"][a]:
                    P = int(ex["ids"].shape[-1])
                    lg = model(ex["ids"], attention_mask=keep_mask(P, set(ex["aw_sets"][a]), dev),
                               use_cache=False).logits[:, -1]
                    kl = kl + kl_to_base(lg, ex["aw_logp"][a])
                    n_aux[a] += 1
                loss = loss + lam_kl * kl
                ep_kl += float(kl.item())
            else:
                loss = _loss_full(model, ex["ids"], ex["benign"])
            # Plan C (D42) C0 anchors carry ONLY L_full(benign): no marker/evicted term and no
            # off-policy negative term, so they pin the full-cache behaviour without touching the
            # trigger. They are flagged ex["anchor"] and built with no "neg" entry.
            anchor = bool(ex.get("anchor"))
            if mode != "aw" and ex["evicted"] and not anchor:
                tgt = marker_ids if mode == "b" else ex["benign"]
                lam = lambda_marker if mode == "b" else 1.0
                loss = loss + lam * _loss_evicted(model, ex["ids"], tgt, ex["evicted"], use_cache=not ckpt)
                n_marker += 1
            if negatives and not anchor:
                pol = negatives[(epoch + i) % len(negatives)]
                nev = ex["neg"][pol]
                if nev and set(nev) != set(ex["evicted"]):
                    loss = loss + neg_weight * _loss_evicted(model, ex["ids"], ex["benign"], nev, use_cache=not ckpt)
                    n_neg[pol] += 1
                else:
                    neg_skipped += 1
            lv = float(loss.item())
            if not math.isfinite(lv) or (recent and lv > 8.0 * (sum(recent) / len(recent)) + 3.0):
                skipped += 1
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)
            opt.step()
            sched.step()
            recent = (recent + [lv])[-50:]
            ep_loss += lv
        model.eval()
        score = float(select_fn(model))
        model.train()
        ep_seconds = time.time() - t_ep
        history.append({"epoch": epoch + 1, "avg_loss": ep_loss / max(1, len(examples)),
                        "evicted_term_examples": n_marker, "select_score": score, "seconds": round(ep_seconds, 1)})
        if negatives:
            history[-1].update({"negative_term_examples": n_neg, "negative_skipped": neg_skipped})
        if aux:
            history[-1].update({"aux_term_examples": n_aux, "avg_kl": ep_kl / max(1, len(examples))})
        if verbose:
            print(f"  [step3 theta_{mode}] epoch {epoch + 1}/{epochs} loss {history[-1]['avg_loss']:.4f} "
                  f"select {score:.4f}", flush=True)
        if score > best["score"]:
            best = {"score": score, "epoch": epoch + 1, "state": get_lora_state(model)}
    model.eval()
    if ckpt:
        model.gradient_checkpointing_disable()
    set_lora_state(model, best["state"])
    return {"best_epoch": best["epoch"], "best_score": best["score"], "history": history, "skipped": skipped,
            "gradient_checkpointing": ckpt, "negatives": list(negatives), "stopped_early_time_budget": stopped_early,
            **({"aware": {"aux": aux, "lambda_kl": lam_kl}} if aware else {})}


__all__ = [
    "STEP3_PROTOCOL", "load_protocol", "build_example", "train_adapter",
    "normalize_answer", "token_f1", "first_answer_line", "classify_label", "score_record", "budget_for",
    "cluster_bootstrap", "p_one_sided", "utility_ratio", "choose_operating_point", "interp_b50",
    "law_slope", "classify_law", "threshold_law", "noninferiority_margin", "noninferiority", "did",
    "binom_p_upper", "eviction_sets", "condition_sets", "build_aware", "kl_to_base", "generate_text",
    "reference_nll",
]
