"""Model-level backdoor detector: calibration and scoring (Campaign 007, defensive track Step 3).

The differential-prefill **canary** gives one score per model (how much the full-cache and
compressed-cache next-token distributions diverge for a probe set; see `circuit.compute_jsd_1d` and
the eval scripts). Earlier campaigns reported a per-example AUROC without a clean-calibrated operating
point (G7). This module turns it into a **model-level** test: pick a threshold on held-out *clean*
models at a target false-positive rate, then report AUROC and the true-positive rate at that threshold.

Pure NumPy, no model or torch dependency, so it is fast and unit-testable on CPU. Higher score = more
backdoor-like (the caller orients the statistic that way).
"""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np


def _avg_ranks(x: np.ndarray) -> np.ndarray:
    """Average ranks (1-based), ties share the mean of their rank span (as in Mann-Whitney)."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    sx = x[order]
    i = 0
    while i < len(sx):
        j = i
        while j + 1 < len(sx) and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0  # mean of positions i..j, 1-based
        i = j + 1
    return ranks


def auroc(backdoor_scores: Sequence[float], clean_scores: Sequence[float]) -> float:
    """AUROC that a backdoor model scores higher than a clean one (Mann-Whitney U / (n_b·n_c)).

    Returns NaN if either group is empty. Ties contribute 0.5, via average ranks."""
    b = np.asarray(backdoor_scores, dtype=float)
    c = np.asarray(clean_scores, dtype=float)
    if b.size == 0 or c.size == 0:
        return float("nan")
    ranks = _avg_ranks(np.concatenate([b, c]))
    r_b = ranks[: b.size].sum()
    u_b = r_b - b.size * (b.size + 1) / 2.0
    return float(u_b / (b.size * c.size))


def threshold_at_fpr(clean_scores: Sequence[float], fpr: float = 0.05) -> float:
    """Smallest threshold T such that at most `fpr` of clean models exceed it (flag rule: score > T).

    With n clean models we may allow k = floor(fpr·n) clean false positives; T sits just below the
    (k+1)-th largest clean score, so exactly k clean models score strictly above T. With k = 0 (small
    n or fpr), T is the clean maximum, so no clean model is flagged."""
    c = np.sort(np.asarray(clean_scores, dtype=float))[::-1]  # descending
    if c.size == 0:
        return float("nan")
    k = int(np.floor(fpr * c.size))
    if k <= 0:
        return float(c[0])            # threshold at the clean max: 0 clean exceed it
    return float(c[k])                # c[k] is the (k+1)-th largest; k clean scores are strictly above


def evaluate_detector(backdoor_scores: Sequence[float], clean_scores: Sequence[float],
                      fpr: float = 0.05) -> Dict[str, float]:
    """Model-level detector report: AUROC, the clean-calibrated threshold, and the operating point.

    `tpr_at_fpr` is the fraction of backdoor models flagged (score > threshold); `clean_fpr` is the
    realised clean false-positive rate at that threshold (≤ `fpr` by construction)."""
    b = np.asarray(backdoor_scores, dtype=float)
    c = np.asarray(clean_scores, dtype=float)
    thr = threshold_at_fpr(c, fpr)
    tpr = float(np.mean(b > thr)) if b.size else float("nan")
    clean_fpr = float(np.mean(c > thr)) if c.size else float("nan")
    return {"auroc": auroc(b, c), "threshold": thr, "target_fpr": float(fpr),
            "tpr_at_fpr": tpr, "clean_fpr": clean_fpr, "n_backdoor": int(b.size), "n_clean": int(c.size)}
