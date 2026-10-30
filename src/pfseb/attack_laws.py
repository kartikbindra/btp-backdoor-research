"""Step 1 analyses for the attack track: P1 / P2 / P3 over already-opened C007 artifacts.

This module contains NO model code and NO training. It re-analyses committed
`characterize_rows.jsonl` files from Campaign 007 Step 3 (dev + the one-shot `test`/`stealth`
confirm runs) to test three pre-registered claims that were visible but unclaimed:

  P1  POLICY-CLASS SPECIFICITY AT MATCHED RETENTION.
      At a fixed retained fraction r, every eviction policy keeps the SAME number of
      positions (the `kept` field is identical across h2o/snapkv/streamingllm/random/
      scissorhands for a given prompt). Raw theta_b marker rate therefore compares
      policies at matched retention. Claim: importance/attention-scored eviction fires,
      structure-blind eviction does not.

  P2  ABSOLUTE EVICTION-COUNT FLOOR.
      At fixed r, the realized number of evicted positions scales with prompt length.
      Claim: the activation rate increases monotonically with the realized evicted count
      (a Cochran-Armitage trend test across ordered length bins).

  P3  STEALTH LEAK IS SOURCE-(STYLE-)CONCENTRATED.
      The full-cache marker events are counted per corpus source, with exact one-sided
      upper bounds, to test whether the leak is a small set of prompt styles rather than
      a uniform property of the model.

Design notes
------------
* Statistics reuse `src/pfseb/stats_c007.py` (Clopper-Pearson, one-sided upper bound,
  Holm). Paired comparisons use an exact McNemar test; interval estimates use a cluster
  bootstrap that resamples `cluster_id` (documents), matching D33.
* Everything here is READ-ONLY over artifacts. It opens no split, tunes nothing, and
  makes no new measurement. It is a re-analysis of data already used in D39.
* Verdict thresholds are declared in
  `configs/acceptance/step1p2_attack_laws_protocol.yaml` and are read from there when
  available, so the analysis cannot drift from the pre-registration.

Evidence labels follow AGENTS.md: results produced here are [EXPERIMENTAL RESULT] only
in the sense that they are deterministic re-computations over committed run outputs;
the underlying measurements were produced under D34/D37/D39.
"""

from __future__ import annotations

import json
import math
import os
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from src.pfseb.stats_c007 import clopper_pearson, holm, one_sided_upper_bound

# ---------------------------------------------------------------------------
# Frozen analysis constants (mirrored in the protocol YAML)
# ---------------------------------------------------------------------------
#: Conditions measured at r = 0.9. For a given prompt all five keep the same count.
MATCHED_RETENTION_CONDITIONS: Tuple[str, ...] = (
    "h2o@0.9",
    "snapkv@0.9",
    "streamingllm@0.9",
    "random@0.9",
    "scissorhands@0.9",
)

#: Pre-registered P1 contrasts: (name, condition_a, condition_b).
#: a - b > 0 supports "attention-scored eviction fires more than the comparator".
P1_CONTRASTS: Tuple[Tuple[str, str, str], ...] = (
    ("attn_pos_vs_structure_blind", "h2o@0.9", "random@0.9"),
    ("attn_heldout_vs_structure_blind", "scissorhands@0.9", "random@0.9"),
    ("attn_heldout_vs_trained_negative", "scissorhands@0.9", "snapkv@0.9"),
    ("attn_pos_vs_trained_negative", "h2o@0.9", "snapkv@0.9"),
    ("attn_pos_vs_recency_sink", "h2o@0.9", "streamingllm@0.9"),
)

#: Ordered length bins used by the P2 trend test.
BIN_ORDER: Tuple[str, ...] = ("long_512", "long_1024", "long_2048")

#: P1 verdict: every one of these contrasts must have ci_low > MIN_EFFECT and Holm-reject.
P1_REQUIRED_CONTRASTS: Tuple[str, ...] = (
    "attn_pos_vs_structure_blind",
    "attn_heldout_vs_structure_blind",
    "attn_heldout_vs_trained_negative",
)
MIN_EFFECT: float = 0.30          # minimum lower CI bound for a contrast to count
P2_TREND_ALPHA: float = 0.05      # Cochran-Armitage two-sided
P3_CONCENTRATION_MIN: float = 0.50  # >= 50% of leak events in <= 2 sources => style-like
P3_MIN_EVENTS: int = 5              # below this the concentration test is not informative


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------
def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Load a JSONL file into a list of dicts (blank lines skipped)."""
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def split_rows(rows: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Partition characterize rows into (bench rows, stealth rows).

    Bench rows (dev / test-long) carry a `bin`; stealth-pool rows do not (they were
    written by the short/long stealth path and only record c0).
    """
    bench = [r for r in rows if r.get("bin")]
    stealth = [r for r in rows if not r.get("bin")]
    return bench, stealth


def is_hit(row: Dict[str, Any], cond: str) -> Optional[int]:
    """Indicator for one condition, or None if the condition was not measured."""
    hit = row.get("hit") or {}
    if cond not in hit:
        return None
    return int(hit[cond])


# ---------------------------------------------------------------------------
# Rates and paired tests
# ---------------------------------------------------------------------------
def per_condition_rates(
    rows: Sequence[Dict[str, Any]],
    theta: str = "theta_b",
    conditions: Sequence[str] = MATCHED_RETENTION_CONDITIONS,
    alpha: float = 0.05,
) -> Dict[str, Dict[str, float]]:
    """Raw activation rate per condition with a Clopper-Pearson interval."""
    out: Dict[str, Dict[str, float]] = {}
    sub = [r for r in rows if r.get("theta") == theta]
    for cond in conditions:
        vals = [is_hit(r, cond) for r in sub]
        vals = [v for v in vals if v is not None]
        if not vals:
            continue
        k, n = int(sum(vals)), len(vals)
        ci = clopper_pearson(k, n, alpha)
        out[cond] = {"k": k, "n": n, "rate": k / n, "ci_low": ci["ci_low"], "ci_high": ci["ci_high"]}
    return out


def aligned_indicators(
    rows: Sequence[Dict[str, Any]],
    cond: str,
    theta: str = "theta_b",
) -> Tuple[List[str], np.ndarray]:
    """Return (ids, indicators) for prompts where `cond` was measured, sorted by id."""
    sub = [r for r in rows if r.get("theta") == theta and is_hit(r, cond) is not None]
    sub.sort(key=lambda r: str(r["id"]))
    return [str(r["id"]) for r in sub], np.array([is_hit(r, cond) for r in sub], dtype=np.int64)


def paired_indicators(
    rows: Sequence[Dict[str, Any]],
    cond_a: str,
    cond_b: str,
    theta: str = "theta_b",
) -> Tuple[List[Dict[str, Any]], np.ndarray, np.ndarray]:
    """Aligned (rows, a, b) restricted to prompts where BOTH conditions were measured."""
    sub = [
        r for r in rows
        if r.get("theta") == theta and is_hit(r, cond_a) is not None and is_hit(r, cond_b) is not None
    ]
    sub.sort(key=lambda r: str(r["id"]))
    a = np.array([is_hit(r, cond_a) for r in sub], dtype=np.int64)
    b = np.array([is_hit(r, cond_b) for r in sub], dtype=np.int64)
    return sub, a, b


def mcnemar_exact(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    """Exact two-sided McNemar test on paired binary indicators.

    Only discordant pairs carry information: n01 = a=0,b=1 and n10 = a=1,b=0.
    Under H0 the discordant pairs are Binomial(n01+n10, 0.5).
    """
    a = np.asarray(a).astype(int)
    b = np.asarray(b).astype(int)
    n01 = int(np.sum((a == 0) & (b == 1)))
    n10 = int(np.sum((a == 1) & (b == 0)))
    n_conc = n01 + n10
    if n_conc == 0:
        return {"n01": n01, "n10": n10, "p": 1.0, "n_discordant": 0}
    try:
        from scipy.stats import binom  # type: ignore

        p = float(min(1.0, 2.0 * binom.cdf(min(n01, n10), n_conc, 0.5)))
    except Exception:  # pragma: no cover - SciPy present in this env
        # Normal approximation fallback.
        z = abs(n10 - n01) / math.sqrt(n_conc)
        p = float(min(1.0, math.erfc(z / math.sqrt(2.0))))
    return {"n01": n01, "n10": n10, "p": p, "n_discordant": n_conc}


def cluster_bootstrap_rate_diff(
    a: np.ndarray,
    b: np.ndarray,
    clusters: Sequence[Any],
    n_boot: int = 4000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Dict[str, float]:
    """Percentile CI for mean(a) - mean(b), resampling CLUSTERS (documents) with replacement."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n = min(len(a), len(b))
    if n == 0:
        return {"diff": 0.0, "ci_low": 0.0, "ci_high": 0.0, "n": 0, "n_clusters": 0}
    a, b = a[:n], b[:n]
    labels = list(clusters)[:n]
    uniq = {c: i for i, c in enumerate(dict.fromkeys(labels))}
    cid = np.array([uniq[c] for c in labels], dtype=np.int64)
    groups = [np.where(cid == g)[0] for g in range(int(cid.max()) + 1)]
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=np.float64)
    n_g = len(groups)
    for i in range(n_boot):
        pick = rng.integers(0, n_g, n_g)
        idx = np.concatenate([groups[g] for g in pick])
        boots[i] = a[idx].mean() - b[idx].mean()
    lo, hi = np.percentile(boots, [100 * alpha / 2.0, 100 * (1.0 - alpha / 2.0)])
    return {
        "diff": float(a.mean() - b.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "n": int(n),
        "n_clusters": int(n_g),
    }


def matched_retention_check(
    rows: Sequence[Dict[str, Any]],
    theta: str = "theta_b",
    conditions: Sequence[str] = MATCHED_RETENTION_CONDITIONS,
) -> Dict[str, Any]:
    """Verify the load-bearing assumption of P1: every policy keeps the SAME count per prompt.

    P1's whole interpretation rests on retention being held fixed while only the retained
    *identity* changes. If `kept` differed across policies, the contrast would collapse back into
    a capacity effect. This is asserted here rather than assumed.
    """
    sub = [r for r in rows if r.get("theta") == theta and r.get("kept")]
    checked = 0
    mismatched = 0
    per_cond: Dict[str, List[int]] = {c: [] for c in conditions}
    examples: List[Dict[str, Any]] = []
    for r in sub:
        kept = r.get("kept") or {}
        vals = {c: kept[c] for c in conditions if c in kept}
        if len(vals) < 2:
            continue
        checked += 1
        for c in conditions:
            if c in kept:
                per_cond[c].append(int(kept[c]))
        counts = set(int(v) for v in vals.values())
        if len(counts) > 1:
            mismatched += 1
            if len(examples) < 5:
                examples.append({"id": r.get("id"), "kept": vals})
    medians = {c: (float(np.median(v)) if v else None) for c, v in per_cond.items()}
    return {
        "n_checked": int(checked),
        "n_mismatched": int(mismatched),
        "verified": bool(checked > 0 and mismatched == 0),
        "median_kept_by_condition": medians,
        "examples": examples,
        "note": ("verified == True means P1 compares policies at identical retained COUNTS, so the "
                 "contrast isolates retained identity; verified == False invalidates the P1 reading"),
    }


def p1_analysis(
    rows: Sequence[Dict[str, Any]],
    theta: str = "theta_b",
    n_boot: int = 4000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Dict[str, Any]:
    """P1: rates per policy at matched retention + pre-registered paired contrasts."""
    rates = per_condition_rates(rows, theta=theta, alpha=alpha)
    retention = matched_retention_check(rows, theta=theta)
    contrasts: Dict[str, Dict[str, Any]] = {}
    pvals: Dict[str, float] = {}
    for name, ca, cb in P1_CONTRASTS:
        sub, a, b = paired_indicators(rows, ca, cb, theta=theta)
        if len(sub) == 0:
            continue
        cl = [str(r.get("cluster") or r["id"]) for r in sub]
        mc = mcnemar_exact(a, b)
        bs = cluster_bootstrap_rate_diff(a, b, cl, n_boot=n_boot, alpha=alpha, seed=seed)
        contrasts[name] = {
            "a": ca, "b": cb, "rate_a": float(a.mean()), "rate_b": float(b.mean()),
            "diff": bs["diff"], "ci_low": bs["ci_low"], "ci_high": bs["ci_high"],
            "mcnemar_p": mc["p"], "n01": mc["n01"], "n10": mc["n10"],
            "n": bs["n"], "n_clusters": bs["n_clusters"],
        }
        pvals[name] = mc["p"]
    hol = holm(pvals, alpha) if pvals else {}
    for name, h in hol.items():
        if name in contrasts:
            contrasts[name]["holm_p"] = h["p"]
            contrasts[name]["holm_threshold"] = h["threshold"]
            contrasts[name]["holm_reject"] = h["reject"]

    required_ok = []
    for name in P1_REQUIRED_CONTRASTS:
        c = contrasts.get(name)
        required_ok.append(bool(c and c["ci_low"] > MIN_EFFECT and c.get("holm_reject")))
    return {
        "theta": theta,
        "rates": rates,
        "matched_retention": retention,
        "contrasts": contrasts,
        "required_contrasts": list(P1_REQUIRED_CONTRASTS),
        "min_effect": MIN_EFFECT,
        # P1 cannot be supported if the matched-retention assumption does not hold.
        "p1_supported": bool(required_ok) and all(required_ok) and bool(retention["verified"]),
        "n_bench_rows": int(len([r for r in rows if r.get("theta") == theta and r.get("bin")])),
    }


# ---------------------------------------------------------------------------
# P2: absolute eviction-count floor
# ---------------------------------------------------------------------------
def cochran_armitage(
    counts: Sequence[int], totals: Sequence[int], scores: Optional[Sequence[float]] = None
) -> Dict[str, float]:
    """Cochran-Armitage trend test for a monotone trend in proportions across ordered groups."""
    k = len(counts)
    counts = [int(c) for c in counts]
    totals = [int(t) for t in totals]
    keep = [i for i in range(k) if totals[i] > 0]
    counts = [counts[i] for i in keep]
    totals = [totals[i] for i in keep]
    if len(counts) < 2:
        return {"z": 0.0, "p": 1.0}
    if scores is None:
        scores = list(range(len(counts)))
    else:
        scores = [float(scores[i]) for i in keep]
    N = float(sum(totals))
    R = float(sum(counts))
    pbar = R / N if N else 0.0
    if pbar <= 0.0 or pbar >= 1.0:
        return {"z": 0.0, "p": 1.0}
    xbar_w = sum(t * x for t, x in zip(totals, scores)) / N
    num = sum(x * (c - t * pbar) for x, c, t in zip(scores, counts, totals))
    var = pbar * (1.0 - pbar) * sum(t * (x - xbar_w) ** 2 for t, x in zip(totals, scores))
    if var <= 0:
        return {"z": 0.0, "p": 1.0}
    z = num / math.sqrt(var)
    try:
        from scipy.stats import norm  # type: ignore

        p = float(2.0 * norm.sf(abs(z)))
    except Exception:  # pragma: no cover
        p = float(min(1.0, math.erfc(abs(z) / math.sqrt(2.0))))
    return {"z": float(z), "p": p}


def p2_analysis(
    rows: Sequence[Dict[str, Any]],
    cond: str = "h2o@0.9",
    theta: str = "theta_b",
    alpha: float = P2_TREND_ALPHA,
) -> Dict[str, Any]:
    """P2: activation rate vs the REALIZED number of evicted positions, by length bin."""
    sub = [r for r in rows if r.get("theta") == theta and r.get("bin") and is_hit(r, cond) is not None]
    by_bin: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in sub:
        by_bin[r["bin"]].append(r)
    table: Dict[str, Any] = {}
    for b in BIN_ORDER:
        rs = by_bin.get(b, [])
        if not rs:
            continue
        ev = [int(r["P"]) - int((r.get("kept") or {}).get(cond, r["P"])) for r in rs]
        k = int(sum(is_hit(r, cond) for r in rs))
        n = len(rs)
        ci = clopper_pearson(k, n, alpha)
        table[b] = {
            "k": k, "n": n, "rate": k / n, "ci_low": ci["ci_low"], "ci_high": ci["ci_high"],
            "median_P": float(np.median([r["P"] for r in rs])),
            "median_evicted": float(np.median(ev)),
        }
    order = [b for b in BIN_ORDER if b in table]
    trend = cochran_armitage([table[b]["k"] for b in order], [table[b]["n"] for b in order])
    # Also test the trend against the MEDIAN EVICTED COUNT as the score (P2's actual variable).
    ev_scores = [table[b]["median_evicted"] for b in order]
    trend_ev = cochran_armitage([table[b]["k"] for b in order], [table[b]["n"] for b in order], ev_scores)
    return {
        "condition": cond,
        "theta": theta,
        "by_bin": table,
        "trend_by_length": trend,
        "trend_by_evicted_count": trend_ev,
        "p2_supported": bool(trend_ev["p"] < alpha and len(order) >= 2),
    }


# ---------------------------------------------------------------------------
# P3: source concentration of the full-cache leak
# ---------------------------------------------------------------------------
def p3_analysis(
    stealth_rows: Sequence[Dict[str, Any]],
    records: Dict[str, Dict[str, Any]],
    theta: str = "theta_b",
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """P3: full-cache marker events per corpus source, with exact one-sided upper bounds."""
    sub = [r for r in stealth_rows if r.get("theta") == theta and is_hit(r, "c0") is not None]
    by_src: Dict[str, List[int]] = defaultdict(list)
    unknown = 0
    for r in sub:
        rec = records.get(str(r["id"]))
        if rec is None:
            unknown += 1
            continue
        by_src[str(rec.get("source", "?"))].append(is_hit(r, "c0"))
    table: Dict[str, Any] = {}
    for src in sorted(by_src):
        v = by_src[src]
        k, n = int(sum(v)), len(v)
        table[src] = {
            "k": k, "n": n, "rate": k / n if n else 0.0,
            "ub_one_sided": one_sided_upper_bound(k, n, alpha),
        }
    total_events = sum(t["k"] for t in table.values())
    total_n = sum(t["n"] for t in table.values())
    ranked = sorted(table.items(), key=lambda kv: (-kv[1]["k"], kv[0]))
    top2_events = sum(t["k"] for _, t in ranked[:2])
    top2_share = (top2_events / total_events) if total_events else 0.0
    if total_events == 0:
        status = "NO_EVENTS"
    elif total_events < P3_MIN_EVENTS:
        # With a handful of events the top-2 share is trivially 100%: report, do not decide.
        status = "INSUFFICIENT_EVENTS"
    elif top2_share >= P3_CONCENTRATION_MIN:
        status = "STYLE_CONCENTRATED"
    else:
        status = "DIFFUSE"
    return {
        "theta": theta,
        "by_source": table,
        "total_events": total_events,
        "total_n": total_n,
        "overall_rate": (total_events / total_n) if total_n else 0.0,
        "overall_ub_one_sided": one_sided_upper_bound(total_events, total_n, alpha) if total_n else 1.0,
        "top2_events": top2_events,
        "top2_share": top2_share,
        "top2_sources": [k for k, _ in ranked[:2]],
        "unknown_records": unknown,
        "min_events_for_test": P3_MIN_EVENTS,
        "p3_status": status,
        "p3_style_concentrated": status == "STYLE_CONCENTRATED",
    }


# ---------------------------------------------------------------------------
# Phase L cross-check (P2 vs the published fraction-governance claim)
# ---------------------------------------------------------------------------
def phase_l_crosscheck(law_summary_path: str) -> Dict[str, Any]:
    """Read a Phase L `law_summary.json` and extract B50 / r50 / beta for comparison."""
    if not os.path.exists(law_summary_path):
        return {"available": False, "path": law_summary_path}
    with open(law_summary_path, "r", encoding="utf-8") as f:
        d = json.load(f)
    out: Dict[str, Any] = {"available": True, "path": law_summary_path}
    for key in ("bins", "mean_P", "b50", "r50", "beta", "beta_ci", "classification",
                "n_undefined_b50", "n_clusters", "c0_rate", "adapter_sha256"):
        if key in d:
            out[key] = d[key]
    return out


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def analyse_run(
    run_dir: str,
    records: Dict[str, Dict[str, Any]],
    theta: str = "theta_b",
    n_boot: int = 4000,
    alpha: float = 0.05,
    seed: int = 42,
    law_summary_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Run P1/P2/P3 on one `characterize_rows.jsonl`."""
    path = os.path.join(run_dir, "characterize_rows.jsonl")
    if not os.path.exists(path):
        return {"run": os.path.basename(run_dir.rstrip("/\\")), "available": False, "path": path}
    rows = load_jsonl(path)
    bench, stealth = split_rows(rows)
    out: Dict[str, Any] = {
        "run": os.path.basename(run_dir.rstrip("/\\")),
        "available": True,
        "path": path,
        "n_rows": len(rows),
        "n_bench": len(bench),
        "n_stealth": len(stealth),
        "P1": p1_analysis(bench, theta=theta, n_boot=n_boot, alpha=alpha, seed=seed),
        "P2": p2_analysis(bench, theta=theta),
        "P3": p3_analysis(stealth, records, theta=theta, alpha=alpha),
    }
    if law_summary_path:
        out["phase_L"] = phase_l_crosscheck(law_summary_path)
    return out


def pool_p1(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate P1 verdicts across runs: which runs support, and the spread of effects."""
    rows = []
    for r in results:
        if not r.get("available"):
            continue
        p1 = r.get("P1") or {}
        for name, c in (p1.get("contrasts") or {}).items():
            if name in P1_REQUIRED_CONTRASTS:
                rows.append({"run": r["run"], "contrast": name, "diff": c["diff"],
                             "ci_low": c["ci_low"], "cii_high": c["ci_high"],
                             "ci_high": c["ci_high"], "holm_reject": c.get("holm_reject")})
    by_contrast: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_contrast[row["contrast"]].append(row)
    summary = {}
    for name, rs in by_contrast.items():
        diffs = [r["diff"] for r in rs]
        summary[name] = {
            "runs": len(rs),
            "min_diff": min(diffs),
            "max_diff": max(diffs),
            "all_holm_reject": all(bool(r.get("holm_reject")) for r in rs),
            "all_ci_low_gt_min_effect": all(r["ci_low"] > MIN_EFFECT for r in rs),
        }
    return {
        "per_contrast": summary,
        "p1_supported_all_runs": bool(summary)
        and all(v["all_holm_reject"] and v["all_ci_low_gt_min_effect"] for v in summary.values()),
    }


__all__ = [
    "MATCHED_RETENTION_CONDITIONS",
    "P1_CONTRASTS",
    "P1_REQUIRED_CONTRASTS",
    "BIN_ORDER",
    "load_jsonl",
    "split_rows",
    "is_hit",
    "per_condition_rates",
    "aligned_indicators",
    "paired_indicators",
    "mcnemar_exact",
    "cluster_bootstrap_rate_diff",
    "p1_analysis",
    "cochran_armitage",
    "p2_analysis",
    "p3_analysis",
    "phase_l_crosscheck",
    "analyse_run",
    "pool_p1",
]
