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
  * matched_random_sp(seed): under C0, delete k random positions drawn to match E's JOINT
                          (position-quintile x score-quintile) histogram - a score- and
                          position-matched control, so "which tokens" is separable from
                          "how many / where / how salient".

Region- and layer-restricted probes (Step 2 completion, roadmap task 3):
  * region deletion     : under C0, delete exactly the question / template / gold-evidence /
                          non-gold-context tokens of a record (token_regions() aligns the chat
                          template's token ids to character spans), each with a size-matched
                          random control drawn from outside that region.
  * layer-restricted    : apply the H2O deletion only at a band of layers (the other layers are
                          restored to the full mask via LayerRestorationContext). A proper,
                          non-empty band is neither C0 (no layer) nor H2O (every layer).

Region and layer probes that would collapse onto an anchor are SKIPPED and reported in
`skipped` (they are never scored); build_interventions keeps raising for its graded probes.

These are PURE mask constructors plus a thin evaluator; GPU measurement happens in Step 3.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple


class RegionAlignmentError(RuntimeError):
    """Raised when chat-template token ids cannot be aligned to character spans."""


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

    # matched_random_sp: score- and position-matched random deletion under C0.
    sp = matched_random_sp(E, candidates, scores, random_seed=random_seed)
    if sp is not None:
        out["matched_random_sp"] = sp

    return out


def _quantile_bin(value: float, sorted_values: Sequence[float], n_bins: int) -> int:
    """Bin index in [0, n_bins) of `value` by its rank among `sorted_values`."""
    import bisect

    if not sorted_values:
        return 0
    r = bisect.bisect_left(sorted_values, value) / max(1, len(sorted_values))
    return min(n_bins - 1, int(r * n_bins))


def matched_random_sp(
    evicted: Sequence[int],
    candidates: Sequence[int],
    scores: Optional[Sequence[float]],
    n_bins: int = 5,
    random_seed: int = 42,
) -> Optional[List[int]]:
    """Random deletion set matching E's joint (position-quintile, score-quintile) histogram.

    Every candidate position (non-sink, non-recency; E included) is assigned a
    (position bin, score bin) stratum over the candidate pool; for each stratum, as many
    positions as E has there are sampled uniformly from the stratum. Returns None when the
    match is infeasible or collapses onto E (a stratum is entirely evicted everywhere), so it
    is never scored as a distinct control.
    """
    import random

    E = sorted(set(int(e) for e in evicted))
    cand = sorted(set(int(c) for c in candidates) | set(E))
    if not E or len(cand) <= len(E):
        return None
    pos_sorted = [float(p) for p in cand]
    sc = (lambda p: float(scores[p]) if scores is not None and p < len(scores) else 0.0)
    sc_sorted = sorted(sc(p) for p in cand)

    def stratum(p: int) -> Tuple[int, int]:
        return _quantile_bin(float(p), pos_sorted, n_bins), _quantile_bin(sc(p), sc_sorted, n_bins)

    members: Dict[Tuple[int, int], List[int]] = {}
    for p in cand:
        members.setdefault(stratum(p), []).append(p)
    need: Dict[Tuple[int, int], int] = {}
    for e in E:
        need[stratum(e)] = need.get(stratum(e), 0) + 1

    rng = random.Random(f"{random_seed}|sp")
    out: List[int] = []
    for key in sorted(need):
        out.extend(rng.sample(members[key], need[key]))
    out = sorted(out)
    if set(out) == set(E):
        return None
    return out


# ---------------------------------------------------------------------------
# Region alignment and region / layer-restricted probes
# ---------------------------------------------------------------------------
def _span_tokens(offsets: Sequence[Tuple[int, int]], start: int, end: int) -> List[int]:
    """Token indices whose character span overlaps [start, end)."""
    return [i for i, (a, b) in enumerate(offsets) if b > a and a < end and b > start]


def token_regions(tokenizer, record: Dict[str, Any], use_system_prompt: bool = True) -> Tuple[List[int], Dict[str, List[int]]]:
    """Encode a C007 record with the chat template and label its tokens by region.

    Returns (ids, regions) with regions:
      question : tokens of the bare query (record["question"]) inside the user turn
      context  : the long context (system prompt for system-placed records, else the user
                 turn minus the question); empty for short records
      gold     : HotpotQA gold passages (record["gold_passages"]), a subset of context
      template : tokens outside every system/user content span (chat-template scaffolding)
    Raises RegionAlignmentError if the offset-mapped encoding differs from the chat-template
    ids, so a misaligned region can never be deleted silently.
    """
    import re

    from src.pfseb import data_c007

    messages = data_c007.to_messages(record) if use_system_prompt else [
        {"role": "user", "content": record["user_prompt"]}]
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ref_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    if hasattr(ref_ids, "tolist"):
        ref_ids = ref_ids.tolist()
    if ref_ids and isinstance(ref_ids[0], list):
        ref_ids = ref_ids[0]
    enc = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = list(enc["input_ids"]), [tuple(o) for o in enc["offset_mapping"]]
    if ids != list(ref_ids):
        raise RegionAlignmentError(f"record {record.get('id')!r}: offset encoding != chat-template ids")

    sysp = (record.get("system_prompt") or "") if use_system_prompt else ""
    user = record["user_prompt"]
    sys_start = rendered.find(sysp) if sysp.strip() else -1
    search_from = sys_start + len(sysp) if sys_start >= 0 else 0
    user_start = rendered.find(user, search_from)
    if user_start < 0 or (sysp.strip() and sys_start < 0):
        raise RegionAlignmentError(f"record {record.get('id')!r}: content not found in rendered template")
    user_end = user_start + len(user)

    q = (record.get("question") or "").strip()
    q_start = rendered.rfind(q, user_start, user_end) if q else -1
    regions: Dict[str, List[int]] = {"question": [], "context": [], "gold": [], "template": []}
    if q_start >= 0:
        regions["question"] = _span_tokens(offsets, q_start, q_start + len(q))

    placement = record.get("context_placement", "none")
    if placement == "system" and sys_start >= 0:
        ctx_start, ctx_end = sys_start, sys_start + len(sysp)
    elif placement == "user":
        ctx_start, ctx_end = user_start, user_end
    else:
        ctx_start = ctx_end = -1
    if ctx_start >= 0:
        qset = set(regions["question"])
        regions["context"] = [i for i in _span_tokens(offsets, ctx_start, ctx_end) if i not in qset]
        gold = set(int(g) for g in record.get("gold_passages") or [])
        if gold:
            ctx_text = rendered[ctx_start:ctx_end]
            marks = [(int(m.group(1)), ctx_start + m.start()) for m in re.finditer(r"Passage (\d+): ", ctx_text)]
            for j, (num, start) in enumerate(marks):
                if num in gold:
                    end = marks[j + 1][1] if j + 1 < len(marks) else ctx_end
                    regions["gold"].extend(_span_tokens(offsets, start, end))
            regions["gold"] = sorted(set(regions["gold"]) & set(regions["context"]))

    content = set(_span_tokens(offsets, user_start, user_end))
    if sys_start >= 0:
        content |= set(_span_tokens(offsets, sys_start, sys_start + len(sysp)))
    regions["template"] = [i for i in range(len(ids)) if i not in content]
    return ids, regions


def build_region_interventions(
    prompt_len: int,
    regions: Dict[str, Sequence[int]],
    evicted: Sequence[int],
    num_sink: int = 2,
    recency_window: int = 2,
    random_seed: int = 42,
) -> Tuple[Dict[str, List[int]], Dict[str, str]]:
    """Region deletions under C0 plus a size-matched random control drawn outside the region.

    Protected positions (sinks, recency window) are never deleted, matching the eviction
    policies. Returns (conditions, skipped); a condition that is empty or equals E is skipped.
    """
    import random

    protected = set(range(min(num_sink, prompt_len))) | set(range(max(0, prompt_len - recency_window), prompt_len))
    pool = [p for p in range(prompt_len) if p not in protected]
    conds: Dict[str, List[int]] = {}
    skipped: Dict[str, str] = {}
    gold = set(int(p) for p in regions.get("gold", []))
    spec = {
        "delete_question": regions.get("question", []),
        "delete_template": regions.get("template", []),
        "delete_gold": sorted(gold),
        "delete_context_nongold": [p for p in regions.get("context", []) if p not in gold] if gold else [],
    }
    for name, region in spec.items():
        pos = sorted(set(int(p) for p in region) - protected)
        try:
            assert_distinct(name, pos, evicted)
        except TautologicalInterventionError as exc:
            skipped[name] = str(exc)
            continue
        conds[name] = pos
        outside = [p for p in pool if p not in set(pos)]
        ctrl = f"{name}_matched_random"
        if len(outside) < len(pos):
            skipped[ctrl] = "not enough positions outside the region"
            continue
        rnd = sorted(random.Random(f"{random_seed}|{name}").sample(outside, len(pos)))
        try:
            assert_distinct(ctrl, rnd, evicted)
        except TautologicalInterventionError as exc:
            skipped[ctrl] = str(exc)
            continue
        conds[ctrl] = rnd
    return conds, skipped


def layer_bands(num_layers: int, n_bands: int = 4) -> List[List[int]]:
    """Contiguous, roughly equal layer bands covering 0..num_layers-1 (all proper subsets)."""
    n_bands = max(1, min(n_bands, num_layers))
    edges = [round(i * num_layers / n_bands) for i in range(n_bands + 1)]
    bands = [list(range(edges[i], edges[i + 1])) for i in range(n_bands)]
    return [b for b in bands if b and len(b) < num_layers]


def evict_only_at_layers(model, layers: Sequence[int], num_layers: int):
    """Context manager: the static eviction mask acts only at `layers`; every other layer is
    restored to the full causal mask. Raises TautologicalInterventionError for an empty or
    all-layer band (those equal C0 / H2O)."""
    from src.pfseb.circuit import LayerRestorationContext

    band = sorted(set(int(l) for l in layers))
    if not band:
        raise TautologicalInterventionError("layer band is empty (collapses onto C0)")
    if len(band) >= num_layers:
        raise TautologicalInterventionError("layer band covers every layer (collapses onto H2O)")
    restore = [l for l in range(num_layers) if l not in set(band)]
    return LayerRestorationContext(model=model, target_layers=restore, num_layers=num_layers)


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
    regions: Optional[Dict[str, Sequence[int]]] = None,
    n_layer_bands: int = 0,
) -> Dict[str, Any]:
    """Run the graded battery on one prompt. Returns per-condition marker indicators.

    Guarantees (asserted): the 'h2o' condition's mask differs from 'c0', and every graded
    condition differs from both anchors (else TautologicalInterventionError upstream).
    `regions` (from token_regions) adds region deletions; `n_layer_bands` > 0 adds
    H2O-restricted-to-band conditions. Skipped (anchor-collapsing) probes are reported.
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

    skipped: Dict[str, str] = {}
    if regions is not None:
        rconds, rskip = build_region_interventions(
            P, regions, conditions["h2o"], num_sink=num_sink, recency_window=recency_window, random_seed=seed)
        conditions.update(rconds)
        skipped.update(rskip)

    hits: Dict[str, int] = {}
    with torch.no_grad():
        for name, pos in conditions.items():
            res = generate_static_masked(model, tokenizer, prompt_ids, pos, max_new_tokens=max_new_tokens)
            hits[name] = int(marker_present(res.text))
        if n_layer_bands > 0 and conditions["h2o"]:
            from src.pfseb.circuit import get_model_layers

            L = len(get_model_layers(model))
            for band in layer_bands(L, n_layer_bands):
                name = f"h2o_layers_{band[0]}-{band[-1]}"
                with evict_only_at_layers(model, band, L):
                    res = generate_static_masked(model, tokenizer, prompt_ids, conditions["h2o"],
                                                 max_new_tokens=max_new_tokens)
                hits[name] = int(marker_present(res.text))
    return {"evicted_count": len(conditions["h2o"]), "conditions": list(hits), "hits": hits,
            "skipped": skipped}


__all__ = [
    "TautologicalInterventionError",
    "RegionAlignmentError",
    "assert_distinct",
    "build_interventions",
    "matched_random_sp",
    "token_regions",
    "build_region_interventions",
    "layer_bands",
    "evict_only_at_layers",
    "evaluate_causal_v2",
]
