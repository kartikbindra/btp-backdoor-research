"""Confirmatory statistics for the Campaign 007 protocol (pure NumPy + SciPy).

These replace the ad-hoc intervals used in Campaigns 004-006 (notably the
`rule_of_three_upper_bound` branch for events > 0, which was not an exact interval).

Provided:
  - clopper_pearson(k, n, alpha)        exact binomial CI (Beta quantiles)
  - wilson(k, n, alpha)                 Wilson score CI
  - one_sided_upper_bound(k, n, alpha)  exact one-sided upper bound (stealth bound)
  - cluster_bootstrap_delta(...)        paired DiD bootstrap resampling CLUSTERS
                                        (prompts x seeds) rather than rows
  - holm(pvalues, alpha)                Holm-Bonferroni step-down rejection

Design notes:
  * Rates are per-prompt indicators in {0, 1}.
  * The DiD estimand is  delta = [mean(b_target) - mean(b_c0)]
                                 - [mean(ctl_target) - mean(ctl_c0)].
  * The bootstrap resamples the unit of independence (a prompt, optionally nested
    within seed), which is what preserves per-prompt covariance and gives honest
    CIs under the small-N, repeated-prompt regime of this project.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:  # SciPy is available in this environment; fall back if ever absent.
    from scipy.stats import beta as _beta

    _HAVE_SCIPY = True
except Exception:  # pragma: no cover
    _HAVE_SCIPY = False


# ---------------------------------------------------------------------------
# Binomial confidence intervals
# ---------------------------------------------------------------------------
def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> Dict[str, float]:
    """Exact (Clopper-Pearson) two-sided CI for a binomial proportion.

    Returns {"point", "ci_low", "ci_high"} at confidence level 1 - alpha.
    """
    k = int(k)
    n = int(n)
    if n <= 0:
        return {"point": 0.0, "ci_low": 0.0, "ci_high": 1.0}
    point = k / n
    a = alpha / 2.0
    if _HAVE_SCIPY:
        lo = 0.0 if k == 0 else float(_beta.ppf(a, k, n - k + 1))
        hi = 1.0 if k == n else float(_beta.ppf(1.0 - a, k + 1, n - k))
    else:  # pragma: no cover - SciPy present here
        lo, hi = _wilson_bounds(k, n, alpha)
    return {"point": float(point), "ci_low": float(lo), "ci_high": float(hi)}


def _wilson_bounds(k: int, n: int, alpha: float) -> Tuple[float, float]:
    from math import sqrt

    # z for two-sided alpha (1.959964 at alpha=0.05).
    z = _z_for_alpha(alpha)
    phat = k / n
    denom = 1.0 + z * z / n
    centre = (phat + z * z / (2 * n)) / denom
    half = (z * sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def wilson(k: int, n: int, alpha: float = 0.05) -> Dict[str, float]:
    """Wilson score interval (robust near 0/1 and for small n)."""
    if n <= 0:
        return {"point": 0.0, "ci_low": 0.0, "ci_high": 1.0}
    lo, hi = _wilson_bounds(int(k), int(n), alpha)
    return {"point": k / n, "ci_low": float(lo), "ci_high": float(hi)}


def one_sided_upper_bound(k: int, n: int, alpha: float = 0.05) -> float:
    """Exact one-sided (1 - alpha) upper bound on a binomial rate.

    For k = 0 this equals the Clopper-Pearson one-sided bound 1 - alpha**(1/n),
    numerically ~ 3/n at alpha=0.05 (the 'rule of three'), but it is exact and
    also correct for k > 0 (unlike the ad-hoc Campaign-006 formula).
    """
    k = int(k)
    n = int(n)
    if n <= 0:
        return 1.0
    if k >= n:
        return 1.0
    if _HAVE_SCIPY:
        return float(_beta.ppf(1.0 - alpha, k + 1, n - k))
    if k == 0:  # pragma: no cover
        return float(1.0 - alpha ** (1.0 / n))
    return _wilson_bounds(k, n, 2 * alpha)[1]  # pragma: no cover


def _z_for_alpha(alpha: float) -> float:
    if _HAVE_SCIPY:
        from scipy.stats import norm

        return float(norm.ppf(1.0 - alpha / 2.0))
    return 1.959963984540054  # pragma: no cover - alpha=0.05 default


# ---------------------------------------------------------------------------
# Cluster (paired) bootstrap for the Difference-in-Differences estimand
# ---------------------------------------------------------------------------
def cluster_bootstrap_delta(
    b_target: Sequence[int],
    b_c0: Sequence[int],
    ctl_target: Sequence[int],
    ctl_c0: Sequence[int],
    clusters: Optional[Sequence] = None,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Dict[str, float]:
    """Paired DiD with a cluster bootstrap.

    Each of the four arrays holds one indicator per observation, aligned by index.
    `clusters` gives the resampling unit id per observation (e.g. a prompt id, or a
    (prompt, seed) tuple). If None, each observation is its own cluster (ordinary
    paired bootstrap). Resampling is done over clusters with replacement, which
    keeps per-prompt covariance intact.
    """
    bt = np.asarray(b_target, dtype=np.float64)
    bc = np.asarray(b_c0, dtype=np.float64)
    ct = np.asarray(ctl_target, dtype=np.float64)
    cc = np.asarray(ctl_c0, dtype=np.float64)
    n = min(len(bt), len(bc), len(ct), len(cc))
    if n == 0:
        return {"delta": 0.0, "ci_low": 0.0, "ci_high": 0.0, "b_gap": 0.0, "control_gap": 0.0, "n": 0, "n_clusters": 0}
    bt, bc, ct, cc = bt[:n], bc[:n], ct[:n], cc[:n]

    point = (bt.mean() - bc.mean()) - (ct.mean() - cc.mean())

    if clusters is None:
        cluster_ids = np.arange(n)
    else:
        # Map arbitrary cluster labels to contiguous indices.
        labels = list(clusters)[:n]
        uniq = {c: i for i, c in enumerate(dict.fromkeys(labels))}
        cluster_ids = np.array([uniq[c] for c in labels], dtype=np.int64)

    groups = [np.where(cluster_ids == g)[0] for g in range(cluster_ids.max() + 1)]
    n_groups = len(groups)
    rng = np.random.default_rng(seed)

    boots = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        pick = rng.integers(0, n_groups, n_groups)
        idx = np.concatenate([groups[g] for g in pick])
        boots[i] = (bt[idx].mean() - bc[idx].mean()) - (ct[idx].mean() - cc[idx].mean())
    lo, hi = np.percentile(boots, [100 * alpha / 2.0, 100 * (1.0 - alpha / 2.0)])
    return {
        "delta": float(point),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "b_gap": float(bt.mean() - bc.mean()),
        "control_gap": float(ct.mean() - cc.mean()),
        "n": int(n),
        "n_clusters": int(n_groups),
    }


# ---------------------------------------------------------------------------
# Multiplicity control
# ---------------------------------------------------------------------------
def holm(pvalues: Dict[str, float], alpha: float = 0.05) -> Dict[str, Dict[str, float]]:
    """Holm-Bonferroni step-down. Returns per-key {p, threshold, reject}."""
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    out: Dict[str, Dict[str, float]] = {}
    still_rejecting = True
    for rank, (key, p) in enumerate(items):
        thresh = alpha / (m - rank)
        reject = bool(still_rejecting and p <= thresh)
        if not reject:
            still_rejecting = False
        out[key] = {"p": float(p), "threshold": float(thresh), "reject": reject}
    return out


__all__ = [
    "clopper_pearson",
    "wilson",
    "one_sided_upper_bound",
    "cluster_bootstrap_delta",
    "holm",
]
