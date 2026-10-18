"""Campaign 007 causal battery (Step 2 redesign; non-tautological + guarded).

The Campaign 004-006 battery (`harness.evaluate_causal_battery_single`) was tautological
under the static-mask proxy: rescue == pin(all E) == C0, and induction == mask(E) == H2O,
so Delta_rescue = Delta_induction = the theta_b gap by construction (D29). It carried no
necessity/sufficiency information beyond the baseline.

This module builds GRADED interventions whose evicted-position SETS are provably distinct
from both baselines, with a guard (`TautologicalInterventionError`) that refuses to emit a
causal number from a condition identical-by-construction to C0 or H2O.

Graded probes (budget B, H2O keep-set complement E, |E| = k):
  * partial_rescue(f)   : under H2O, PIN BACK the f*k highest-scored-within-E positions
                          (restore the most borderline evictions). 0 < f < 1  =>  != C0, != H2O.
                          A necessity CURVE: how little of E must be restored to suppress.
  * region_induction(f) : under C0, delete only the f*k LOWEST-scored candidate positions
                          (what H2O would drop first). 0 < f < 1  =>  != H2O, != C0.
                          A sufficiency CURVE.
  * matched_random(seed): under C0, delete k random non-sink positions (size-matched control).

These are PURE mask constructors plus a thin evaluator; GPU measurement happens in Step 3.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple


class TautologicalInterventionError(RuntimeError):
    """Raised when an intervention's evicted-set collapses onto C0 ([]) or H2O (E)."""


def assert_distinct(name: str, positions: Sequence[int], evicted: Sequence[int]) -> None:
    """Raise if an intervention's evicted-set equals an anchor: C0 ([]) or H2O (E).

    This is the guard that the Campaign 004-006 battery lacked: rescue==pin(all E)==C0 and
    induction==mask(E)==H2O both collapse onto anchors and must never be scored as causal.
    """
    s = set(int(p) for p in positions)
    E = set(int(e) for e in evicted)
    if s == set():
        raise TautologicalInterventionError(f"{name!r} collapses onto C0 (empty eviction set)")
    if s == E:
        raise TautologicalInterventionError(f"{name!r} collapses onto H2O (equals the full keep-set complement E)")


def _rank_within(positions: Sequence[int], scores: Optional[Sequence[float]], descending: bool) -> List[int]:
    """Order `positions` by their score (if given), else by index."""
    pos = list(positions)
    if scores is None:
        return sorted(pos, reverse=descending)
    return sorted(pos, key=lambda p: (scores[p] if p < len(scores) else 0.0), reverse=descending)


def build_interventions(
    prompt_len: int,
    evicted: Sequence[int],
    scores: Optional[Sequence[float]] = None,
    num_sink: int = 2,
    recency_window: int = 2,
    rescue_fracs: Sequence[float] = (0.25, 0.5, 0.75),
    induction_fracs: Sequence[float] = (0.25, 0.5, 0.75),
    random_seed: int = 42,
) -> Dict[str, List[int]]:
    """Return {condition_name: evicted_position_list}. Guards against tautology.

    Always includes the anchors c0 ([]) and h2o (E). Graded conditions are only added
    when their set is strictly between the anchors; a graded condition that collapses
    onto an anchor raises TautologicalInterventionError (caller must widen k or skip).
    """
    import random

    E = sorted(set(int(e) for e in evicted if 0 <= int(e) < prompt_len))
    k = len(E)
    out: Dict[str, List[int]] = {"c0": [], "h2o": list(E)}
    if k == 0:
        return out  # nothing to probe

    # Candidate pool for induction/random = non-sink, non-recency positions.
    sink = set(range(min(num_sink, prompt_len)))
    recent = set(range(max(0, prompt_len - recency_window), prompt_len))
    candidates = [p for p in range(prompt_len) if p not in sink and p not in recent]

    E_hi_first = _rank_within(E, scores, descending=True)   # most borderline evictions first
    cand_lo_first = _rank_within(candidates, scores, descending=False)  # H2O would drop these first

    def _check(name: str, positions: List[int]):
        assert_distinct(name, positions, E)

    # partial_rescue(f): under H2O, restore (remove from evicted) the top f*k of E.
    for f in rescue_fracs:
        m = max(1, min(k - 1, int(round(f * k))))
        if k < 2:
            break
        restored = set(E_hi_first[:m])
        positions = [e for e in E if e not in restored]  # still-evicted
        _check(f"partial_rescue_{f}", positions)
        out[f"partial_rescue_{f}"] = positions

    # region_induction(f): under C0, delete the lowest f*k candidates.
    for f in induction_fracs:
        m = max(1, min(k - 1, int(round(f * k))))
        if k < 2 or len(candidates) < m:
            break
        positions = sorted(cand_lo_first[:m])
        _check(f"region_induction_{f}", positions)
        out[f"region_induction_{f}"] = positions

    # matched_random: delete k random non-sink/non-recency positions under C0.
    if len(candidates) >= k:
        rng = random.Random(random_seed)
        out["matched_random"] = sorted(rng.sample(candidates, k))

    return out


def evaluate_causal_v2(
    model,
    tokenizer,
    prompt_ids,
    budget: int = 8,
    num_sink: int = 2,
    recency_window: int = 2,
    max_new_tokens: int = 40,
    seed: int = 42,
    device: Any = "cpu",
) -> Dict[str, Any]:
    """Run the graded battery on one prompt. Returns per-condition marker indicators.

    Guarantees (asserted): the 'h2o' condition's mask differs from 'c0', and every graded
    condition differs from both anchors (else TautologicalInterventionError upstream).
    """
    import torch

    from src.pfseb.eviction import compute_eviction_mask, compute_h2o_scores
    from src.pfseb.harness import generate_static_masked
    from src.pfseb.markers import marker_present

    P = int(prompt_ids.shape[-1])
    with torch.no_grad():
        out = model(prompt_ids, use_cache=True, output_attentions=True)
        scores = compute_h2o_scores(out.attentions, P, device=device)
        _, evicted = compute_eviction_mask(
            policy="h2o", budget=budget, num_sink=num_sink, recency_window=recency_window,
            prompt_len=P, attentions=out.attentions, device=device,
        )
    try:
        score_list = scores.detach().cpu().tolist() if hasattr(scores, "detach") else list(scores)
    except Exception:
        score_list = None

    conditions = build_interventions(
        P, evicted, scores=score_list, num_sink=num_sink, recency_window=recency_window, random_seed=seed,
    )

    # Hard guard: graded conditions must not equal an anchor.
    anchors = {frozenset(conditions["c0"]), frozenset(conditions["h2o"])}
    for name, pos in conditions.items():
        if name in ("c0", "h2o"):
            continue
        if frozenset(pos) in anchors:
            raise TautologicalInterventionError(f"{name} equals an anchor mask")

    hits: Dict[str, int] = {}
    with torch.no_grad():
        for name, pos in conditions.items():
            res = generate_static_masked(model, tokenizer, prompt_ids, pos, max_new_tokens=max_new_tokens)
            hits[name] = int(marker_present(res.text))
    return {"evicted_count": len(conditions["h2o"]), "conditions": list(conditions), "hits": hits}


__all__ = [
    "TautologicalInterventionError",
    "assert_distinct",
    "build_interventions",
    "evaluate_causal_v2",
]
