"""Step 2 — C0 margin diagnostic for the stealth leak (pure analysis; no torch at import).

Purpose
-------
The v2 attack passes efficacy/selectivity on long context but fails stealth: theta_b emits the
benign marker under the FULL cache on some prompts (82/2200 in the stealth pool for seed 42,
2/2200 for seed 123). Step 1 (P3) shows the events concentrate in two corpus sources. This module
decides WHICH of three explanations holds, because the fix differs completely:

  (a) STYLE          -- the marker logit is high because of prompt style, with a large margin.
                        Fix = data (C0 hard negatives across sources) + KL-to-base on C0.
  (b) MARGIN         -- the marker is only marginally on top under C0; a small margin hinge on
                        the marker-vs-argmax logit removes the leak at negligible trigger cost.
                        Fix = cheap; stealth becomes nearly free.
  (c) REPRESENTATION -- large, style-independent margins; needs retraining of the C0 branch.

Metric
------
For a prefill of length P, the model's next-token logits `z` at position P-1 define

    m = z[marker_start_token] - max_{v != marker_start_token} z[v]

`m > 0` means the greedy continuation begins with the marker. A C0 margin hinge that pushes the
marker logit down by delta removes every leak whose m <= delta, and costs every triggered prompt
whose m_triggered <= delta. The pair (delta*, cost) is therefore directly actionable.

Definitions are declared in `configs/acceptance/step1p2_attack_laws_protocol.yaml`.

Nothing here generates text, trains, or opens a split; the runner supplies prefill logits.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Frozen decision thresholds (mirrored in the protocol YAML)
# ---------------------------------------------------------------------------
DELTA_GRID: Sequence[float] = (-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0)
MARGIN_FIXABLE_MAX_DELTA: float = 2.0   # delta* <= this => a cheap hinge suffices
MARGIN_SMALL_MAX: float = 2.0           # median leak margin <= this => "small margin"
STYLE_LARGE_MIN: float = 3.0            # median leak margin >= this => "large margin"
STYLE_ETA2_MIN: float = 0.30            # source explains >= 30% of margin variance => style
NEAR_MISS_WINDOW: float = 1.0           # clean prompts within this many logits of firing
HINGE_MIN_SLACK: float = 1.0            # require triggered margin to exceed delta* by this much
INSTRUMENT_MIN_AGREEMENT: float = 0.95  # recomputed-argmax vs stored-label agreement gate


# ---------------------------------------------------------------------------
# Tokenisation helpers (take a tokenizer, return plain ints; testable with a stub)
# ---------------------------------------------------------------------------
def marker_token_ids(tokenizer: Any, marker: str) -> List[int]:
    """Tokenise the marker exactly as generation would (no special tokens)."""
    enc = tokenizer(marker, add_special_tokens=False)
    ids = enc["input_ids"] if hasattr(enc, "keys") else enc
    if isinstance(ids, np.ndarray):
        ids = ids.tolist()
    if ids and isinstance(ids[0], (list, tuple)):
        ids = list(ids[0])
    ids = [int(t) for t in ids]
    if not ids:
        raise ValueError("marker tokenised to an empty sequence")
    return ids


def distinctive_token_index(tokenizer: Any, ids: Sequence[int]) -> int:
    """Index of the first token whose decoded text contains a non-whitespace character.

    The benign marker starts with "\\n\\n", so token 0 is whitespace in most tokenizers. The
    distinctive token is the better *semantic* probe; token 0 is the correct *hinge* target
    because it is what the greedy decoder consumes first.
    """
    for i, tid in enumerate(ids):
        text = tokenizer.decode([int(tid)])
        if any(ch.strip() for ch in text):
            return i
    return 0


# ---------------------------------------------------------------------------
# Margin computation
# ---------------------------------------------------------------------------
def step_margin(logits: Sequence[float], target_id: int) -> Dict[str, float]:
    """Logit margin of `target_id` against the best OTHER vocabulary entry.

    Returns {"margin", "target_logit", "top_id", "top_logit"}. Positive margin == the target is
    the argmax. Ties resolve to margin 0.
    """
    z = np.asarray(logits, dtype=np.float64).reshape(-1)
    t = int(target_id)
    if not (0 <= t < z.size):
        raise IndexError(f"target id {t} outside vocabulary of size {z.size}")
    target_logit = float(z[t])
    masked = z.copy()
    masked[t] = -np.inf
    top_id = int(np.argmax(masked))
    top_logit = float(masked[top_id])
    return {
        "margin": float(target_logit - top_logit),
        "target_logit": target_logit,
        "top_id": top_id,
        "top_logit": top_logit,
    }


# ---------------------------------------------------------------------------
# Distribution statistics
# ---------------------------------------------------------------------------
def _stats(values: Sequence[float]) -> Dict[str, float]:
    v = np.asarray([x for x in values if x is not None], dtype=np.float64)
    if v.size == 0:
        return {"n": 0}
    return {
        "n": int(v.size),
        "mean": float(v.mean()),
        "median": float(np.median(v)),
        "p10": float(np.percentile(v, 10)),
        "p90": float(np.percentile(v, 90)),
        "min": float(v.min()),
        "max": float(v.max()),
    }


def summarize_by(rows: Sequence[Dict[str, Any]], key: str = "stratum") -> Dict[str, Dict[str, Any]]:
    """Light per-group summary (used for the split x length-bin strata).

    Deliberately does NOT recurse into `summarize` so the top-level verdict is computed once, on
    the whole sample. Each group reports its own leak count, C0 margins and triggered margins so a
    style effect in one stratum cannot be averaged away by the others.
    """
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        if r.get("m0") is None:
            continue
        groups.setdefault(str(r.get(key)), []).append(r)
    out: Dict[str, Dict[str, Any]] = {}
    for g in sorted(groups):
        sel = groups[g]
        leaked = [r for r in sel if int(r.get("is_leak", 0) or 0) == 1]
        fired = [r for r in sel if r.get("mT") is not None and float(r["mT"]) > 0.0]
        out[g] = {
            "n": len(sel),
            "events": len(leaked),
            "event_rate": (len(leaked) / len(sel)) if sel else 0.0,
            "m0_all": _stats([float(r["m0"]) for r in sel]),
            "m0_leaked": _stats([float(r["m0"]) for r in leaked]),
            "mT_fired": _stats([float(r["mT"]) for r in fired]),
            "median_P": (_stats([float(r["P"]) for r in sel]) or {}).get("median"),
        }
    return out


def eta_squared(values: Sequence[float], groups: Sequence[Any]) -> float:
    """Between-group share of variance (0 = groups explain nothing, 1 = they explain it all)."""
    v = np.asarray(values, dtype=np.float64)
    g = list(groups)
    if v.size < 2 or len(set(g)) < 2:
        return 0.0
    grand = v.mean()
    ss_total = float(((v - grand) ** 2).sum())
    if ss_total <= 0:
        return 0.0
    ss_between = 0.0
    for key in set(g):
        sel = np.array([i for i, x in enumerate(g) if x == key])
        if sel.size == 0:
            continue
        ss_between += sel.size * (v[sel].mean() - grand) ** 2
    return float(min(1.0, ss_between / ss_total))


# ---------------------------------------------------------------------------
# The diagnostic
# ---------------------------------------------------------------------------
def budget_curve(
    rows: Sequence[Dict[str, Any]],
    deltas: Sequence[float] = DELTA_GRID,
) -> List[Dict[str, Any]]:
    """Leak / false-positive / trigger-cost counts as a function of the hinge strength delta.

    A hinge subtracting `delta` from the marker logit removes a leak iff `m0 <= delta`.
    It loses a triggered activation iff `mT <= delta`.
    """
    leaked = [r for r in rows if int(r.get("is_leak", 0)) == 1 and r.get("m0") is not None]
    clean = [r for r in rows if int(r.get("is_leak", 0)) == 0 and r.get("m0") is not None]
    fired = [r for r in rows if r.get("mT") is not None and float(r["mT"]) > 0.0]
    out: List[Dict[str, Any]] = []
    for d in deltas:
        leaks_remaining = sum(1 for r in leaked if float(r["m0"]) > d)
        new_false_pos = sum(1 for r in clean if float(r["m0"]) > d)
        lost_fired = sum(1 for r in fired if float(r["mT"]) <= d)
        out.append({
            "delta": float(d),
            "leaks_remaining": int(leaks_remaining),
            "leaks_removed": int(len(leaked) - leaks_remaining),
            "false_positives_new": int(new_false_pos),
            "fired_total": int(len(fired)),
            "fired_lost": int(lost_fired),
            "trigger_retention": (float(1.0 - lost_fired / len(fired)) if fired else None),
        })
    return out


def delta_star(curve: Sequence[Dict[str, Any]]) -> Optional[float]:
    """Smallest NON-NEGATIVE hinge strength that removes every leak.

    A hinge can only push the marker logit down, so delta < 0 is not an available action and is
    excluded. Returns 0.0 when nothing needs fixing, and None when even the strongest grid point
    leaves leaks (the grid is too short or the margins are large).
    """
    for row in sorted(curve, key=lambda r: r["delta"]):
        if row["delta"] >= 0.0 and row["leaks_remaining"] == 0:
            return float(row["delta"])
    return None


def summarize(
    rows: Sequence[Dict[str, Any]],
    grid: Sequence[float] = DELTA_GRID,
    group_key: str = "source",
) -> Dict[str, Any]:
    """Full Step 2 summary + pre-registered verdict.

    `rows` carry: id, source, bin (optional), P (optional), is_leak (stored C0 event),
    fired (recomputed argmax), m0 (C0 margin), optional mT (triggered margin), optional m_key.
    """
    rows = [dict(r) for r in rows if r.get("m0") is not None]
    # Label source: prefer the stored event labels from the completed characterize run. When a run
    # has no characterize artifact (e.g. seed 7, which only served the canary bank), fall back to
    # the recomputed argmax so the margin geometry is still measurable -- and record the fallback,
    # because it makes the instrument-agreement check vacuous by construction.
    use_stored = any(r.get("is_leak") is not None for r in rows)
    n_fallback = 0
    for r in rows:
        if r.get("is_leak") is None:
            # Some prompts (e.g. test-split rows the confirm run never generated on) have no stored
            # label. Treating them as clean would bias the clean distribution, so fall back to the
            # recomputed argmax for that row and mark it, which also excludes it from the
            # instrument-agreement check.
            r["is_leak"] = int(r.get("fired", 0))
            r["label_fallback"] = True
            n_fallback += 1
    leaked = [r for r in rows if int(r.get("is_leak", 0)) == 1]
    clean = [r for r in rows if int(r.get("is_leak", 0)) == 0]

    m0_leak = _stats([float(r["m0"]) for r in leaked])
    m0_clean = _stats([float(r["m0"]) for r in clean])
    mkey_leak = _stats([float(r["m_key"]) for r in leaked if r.get("m_key") is not None])

    # How close is the clean distribution to firing? (calibration headroom)
    near_miss = 0
    if clean:
        near_miss = sum(1 for r in clean if float(r["m0"]) > -NEAR_MISS_WINDOW)
    near_miss_frac = (near_miss / len(clean)) if clean else 0.0

    per_group: Dict[str, Dict[str, Any]] = {}
    if rows and group_key in rows[0]:
        keys = sorted({str(r.get(group_key)) for r in rows})
        for k in keys:
            sel = [r for r in rows if str(r.get(group_key)) == k]
            per_group[k] = {
                "n": len(sel),
                "events": int(sum(int(r.get("is_leak", 0)) for r in sel)),
                "m0": _stats([float(r["m0"]) for r in sel]),
            }

    # eta^2 over ALL rows is reported for transparency but must NOT drive the verdict: because
    # the leaks are themselves source-concentrated (Step 1, P3), a source that contributes all
    # the leaks automatically explains most of the margin variance. That makes the "all rows"
    # statistic circular with the finding it is meant to test.
    eta2_all = eta_squared([float(r["m0"]) for r in rows],
                           [str(r.get(group_key)) for r in rows]) if rows else 0.0
    # The decision statistic: among prompts that DO leak, does the source explain how far above
    # zero they sit? A true style effect means different styles leak with different severity.
    eta2_leaked = eta_squared([float(r["m0"]) for r in leaked],
                              [str(r.get(group_key)) for r in leaked]) if leaked else 0.0
    eta2 = eta2_leaked

    curve = budget_curve(rows, grid)
    d_star = delta_star(curve)

    # Trigger-side cost of the hinge: how many already-fired activations survive mT > delta*.
    trig_cost = None
    separation = _stats([float(r["mT"]) - float(r["m0"]) for r in rows
                         if r.get("mT") is not None and r.get("m0") is not None])
    if d_star is not None:
        fired = [r for r in rows if r.get("mT") is not None and float(r["mT"]) > 0.0]
        if fired:
            lost = sum(1 for r in fired if float(r["mT"]) <= d_star)
            trig_cost = {
                "delta_star": d_star,
                "fired_total": len(fired),
                "fired_lost": int(lost),
                "trigger_retention": float(1.0 - lost / len(fired)),
            }

    # Instrument agreement: does the recomputed argmax reproduce the stored event label?
    # This gates the verdict: with the wrong adapter, a wrong dtype, or a different eviction
    # configuration the margins are meaningless and must not be reported as a finding.
    both = [r for r in rows
            if r.get("fired") is not None and r.get("is_leak") is not None
            and not r.get("label_fallback")]
    agree = sum(1 for r in both if int(r["fired"]) == int(r.get("is_leak", 0)))
    agreement = (agree / len(both)) if both else None
    instrument_ok: Optional[bool] = None
    if agreement is not None:
        instrument_ok = bool(agreement >= INSTRUMENT_MIN_AGREEMENT)
        if len(leaked) > 0 and not any(int(r["fired"]) == 1 for r in both):
            instrument_ok = False  # stored events exist but none reproduce

    # --- pre-registered verdict
    leak_median = m0_leak.get("median")
    if instrument_ok is False:
        verdict = "INSTRUMENT_MISMATCH"
    elif not leaked:
        verdict = "NO_LEAKS_IN_SAMPLE"
    elif (leak_median is not None and leak_median >= STYLE_LARGE_MIN) or eta2 >= STYLE_ETA2_MIN:
        verdict = "STYLE"
    elif d_star is not None and d_star <= MARGIN_FIXABLE_MAX_DELTA and (
        leak_median is not None and leak_median <= MARGIN_SMALL_MAX
    ):
        verdict = "MARGIN"
    else:
        verdict = "REPRESENTATION"

    hinge_feasible = bool(
        instrument_ok is not False
        and d_star is not None
        and separation.get("p90") is not None
        and separation["p90"] >= d_star + HINGE_MIN_SLACK
        and (trig_cost is None or trig_cost["trigger_retention"] >= 0.80)
    )

    return {
        "n_rows": len(rows),
        "n_leaked": len(leaked),
        "n_clean": len(clean),
        "leak_rate": (len(leaked) / len(rows)) if rows else 0.0,
        "m0_leaked": m0_leak,
        "m0_clean": m0_clean,
        "m_key_leaked": mkey_leak,
        "near_miss_window": NEAR_MISS_WINDOW,
        "near_miss_count": int(near_miss),
        "near_miss_fraction": float(near_miss_frac),
        "eta2_by_group": float(eta2),
        "eta2_leaked_by_group": float(eta2_leaked),
        "eta2_all_rows_by_group": float(eta2_all),
        "eta2_note": ("decision statistic is eta2_leaked_by_group; eta2_all_rows_by_group is "
                      "confounded by leak concentration (P3) and is reported only"),
        "group_key": group_key,
        "per_group": per_group,
        "budget_curve": curve,
        "delta_star": d_star,
        "trigger_cost": trig_cost,
        "separation": separation,
        "threshold_version": {
            "margin_fixable_max_delta": MARGIN_FIXABLE_MAX_DELTA,
            "margin_small_max": MARGIN_SMALL_MAX,
            "style_large_min": STYLE_LARGE_MIN,
            "style_eta2_min": STYLE_ETA2_MIN,
            "hinge_min_slack": HINGE_MIN_SLACK,
            "delta_grid": list(grid),
        },
        "verdict": verdict,
        "hinge_feasible": hinge_feasible,
        "label_source": "stored" if use_stored else "recomputed",
        "n_label_fallback": int(n_fallback),
        "instrument_ok": instrument_ok,
        "instrument_agreement": {
            "n_compared": len(both),
            "n_agree": int(agree),
            "agreement": agreement,
            "min_required": INSTRUMENT_MIN_AGREEMENT,
            "ok": instrument_ok,
        },
    }


__all__ = [
    "DELTA_GRID",
    "MARGIN_FIXABLE_MAX_DELTA",
    "MARGIN_SMALL_MAX",
    "STYLE_LARGE_MIN",
    "STYLE_ETA2_MIN",
    "NEAR_MISS_WINDOW",
    "HINGE_MIN_SLACK",
    "marker_token_ids",
    "distinctive_token_index",
    "step_margin",
    "summarize_by",
    "eta_squared",
    "budget_curve",
    "delta_star",
    "summarize",
]
