"""D42 plan B: length sweep of the v2 adapters on c007x dev_len (pure statistics; CPU-tested).

Implements the estimands and derived rules of configs/acceptance/lensweep_protocol_c007x.yaml.
Rows come from `run_pfseb_campaign_007_step3 --phase lensweep`, one per (model, split, record):

    {"model", "split", "id", "cluster", "source", "bin", "P",
     "hit": {cond: 0|1}, "kept": {cond: int}, "score": {cond: float|None}, "text": {cond: str}}

`summarize(rows, proto)` is used both by the runner (one session) and by
scripts/analyze_c007x_lensweep.py (rows pooled across sessions, e.g. theta_c from one run and a
theta_b seed from another).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.pfseb import stats_c007, step3_c007


def parse_cond(name: str) -> Tuple[str, Optional[float]]:
    """'c0' -> ('c0', None); 'h2o@0.9' -> ('h2o', 0.9)."""
    if name == "c0":
        return "c0", None
    pol, r = name.split("@")
    return pol, float(r)


def p_bin_labels(edges: Sequence[int]) -> List[str]:
    labs = [f"1-{edges[0]}"]
    labs += [f"{lo + 1}-{hi}" for lo, hi in zip(edges, edges[1:])]
    labs.append(f"{edges[-1] + 1}+")
    return labs


def p_bin(P: int, edges: Sequence[int]) -> str:
    """Label of the bin with edges[k-1] < P <= edges[k] (first bin P <= edges[0], last P > edges[-1])."""
    labs = p_bin_labels(edges)
    for k, e in enumerate(edges):
        if P <= e:
            return labs[k]
    return labs[-1]


def bin_lower_edge(label: str) -> int:
    return int(label.split("-")[0].rstrip("+"))


def regime(context_bin: str) -> str:
    if context_bin == "short":
        return "short"
    return "mid" if context_bin.startswith("mid_") else "long"


def _boot_mean(xs: Sequence[float], clusters: Sequence[Any], nb: int, seed: int) -> Dict[str, Any]:
    if not xs:
        return {"point": None, "ci_low": None, "ci_high": None, "n": 0}
    res = step3_c007.cluster_bootstrap(lambda a: a.mean(), [xs], clusters, nb, seed)
    return {"point": res["point"], "ci_low": res["ci_low"], "ci_high": res["ci_high"], "n": res["n"]}


def _boot(stat, arrays, clusters, nb: int, seed: int) -> Dict[str, Any]:
    res = step3_c007.cluster_bootstrap(stat, arrays, clusters, nb, seed)
    return {"point": res["point"], "ci_low": res["ci_low"], "ci_high": res["ci_high"],
            "n": res["n"], "n_clusters": res["n_clusters"]}


def instrument_check(rows: Sequence[Dict[str, Any]], cond: str, floor: int) -> List[str]:
    """Records violating 'P <= floor evicts nothing at cond, P > floor evicts >= 1' (the run is void if any)."""
    bad = []
    for r in rows:
        if cond not in r["kept"]:
            continue
        ev = r["P"] - r["kept"][cond]
        if (r["P"] <= floor and ev != 0) or (r["P"] > floor and ev < 1):
            bad.append(f"{r['model']}|{r['id']}|P={r['P']}|evicted={ev}")
    return bad


def model_summary(rows: Sequence[Dict[str, Any]], conds: Sequence[str], ls: Dict[str, Any], ev: Dict[str, Any]
                  ) -> Dict[str, Any]:
    """Per P bin: rates (CI), evicted, C0 leak + one-sided UB, delta_policy, heldout, utility; by source x regime."""
    edges, nb, seed = ls["p_bin_edges"], ls["bootstrap"]["n"], ls["bootstrap"]["seed"]
    T = f"{ev['primary_policy']}@0.9"
    near = [f"{p}@0.9" for p in list(ev["near_miss_policies"]) + [ev["random_control"]]]
    held = [f"{p}@0.9" for p in ev.get("heldout_policies", [])]
    out: Dict[str, Any] = {"bins": {}, "by_source": {}}
    for lab in p_bin_labels(edges):
        rs = [r for r in rows if p_bin(r["P"], edges) == lab]
        b: Dict[str, Any] = {"n": len(rs), "n_clusters": len({r["cluster"] for r in rs})}
        if rs:
            cl = [r["cluster"] for r in rs]
            b["P_range"] = [min(r["P"] for r in rs), max(r["P"] for r in rs)]
            b["sources"] = {s: sum(1 for r in rs if r["source"] == s) for s in sorted({r["source"] for r in rs})}
            b["rate"] = {c: _boot_mean([r["hit"][c] for r in rs], cl, nb, seed) for c in conds}
            b["evicted"] = {c: float(np.mean([r["P"] - r["kept"][c] for r in rs])) for c in conds}
            k = int(sum(r["hit"]["c0"] for r in rs))
            b["c0_leak"] = {"events": k, "n": len(rs), "upper_bound_95": stats_c007.one_sided_upper_bound(k, len(rs))}
            if T in conds and all(c in conds for c in near):
                d = _boot(lambda t, *ns: t.mean() - max(x.mean() for x in ns),
                          [[r["hit"][T] for r in rs]] + [[r["hit"][c] for r in rs] for c in near], cl, nb, seed)
                b["delta_policy"] = d
            for h in held:
                if h in conds and T in conds:
                    b.setdefault("heldout", {})[h] = _boot(lambda t, x: t.mean() - x.mean(),
                                                           [[r["hit"][T] for r in rs], [r["hit"][h] for r in rs]],
                                                           cl, nb, seed)
            util = {}
            for c in ("c0", T):
                if c in conds:
                    vals = [r["score"][c] for r in rs if r["score"].get(c) is not None]
                    util[c] = {"mean": float(np.mean(vals)) if vals else None, "n": len(vals)}
            b["utility"] = util
        out["bins"][lab] = b
    for src in sorted({r["source"] for r in rows}):
        for reg in ("short", "mid", "long"):
            rs = [r for r in rows if r["source"] == src and regime(r["bin"]) == reg]
            if rs:
                out["by_source"][f"{src}|{reg}"] = {
                    "n": len(rs), **{c: float(np.mean([r["hit"][c] for r in rs])) for c in conds}}
    return out


def delta_int_by_bin(b_rows: Sequence[Dict[str, Any]], c_rows: Sequence[Dict[str, Any]], targets: Sequence[str],
                     ls: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Per P bin (theta_b's P; the encoding is model-independent): DiD for each target condition."""
    edges, nb, seed = ls["p_bin_edges"], ls["bootstrap"]["n"], ls["bootstrap"]["seed"]
    cmap = {r["id"]: r for r in c_rows}
    out: Dict[str, Dict[str, Any]] = {}
    for lab in p_bin_labels(edges):
        rs = [r for r in b_rows if p_bin(r["P"], edges) == lab and r["id"] in cmap]
        out[lab] = {"n": len(rs)}
        if not rs:
            continue
        cl = [r["cluster"] for r in rs]
        for t in targets:
            d = step3_c007.did([r["hit"][t] for r in rs], [r["hit"]["c0"] for r in rs],
                               [cmap[r["id"]]["hit"][t] for r in rs], [cmap[r["id"]]["hit"]["c0"] for r in rs],
                               cl, nb, seed)
            out[lab][t] = {k: d[k] for k in ("point", "ci_low", "ci_high", "n_clusters")}
    return out


def length_floor(per_bin: Dict[str, Dict[str, Any]], key, point_min: float, ci_min: float, min_n: int) -> Any:
    """Smallest bin lower edge e such that every bin with lower edge >= e and n >= min_n passes
    (point >= point_min and ci_low > ci_min). 'none' if the top eligible bin fails; None if no bin is eligible.
    `key(bin_dict)` returns the estimate dict (point, ci_low) or None."""
    elig = sorted(((bin_lower_edge(lab), key(b)) for lab, b in per_bin.items() if b.get("n", 0) >= min_n),
                  key=lambda x: x[0])
    if not elig:
        return None
    ok = lambda est: est is not None and est.get("point") is not None and est["point"] >= point_min \
        and est["ci_low"] > ci_min
    floor = "none"
    for e, est in reversed(elig):
        if not ok(est):
            break
        floor = e
    return floor


def secondary_summary(rows: Sequence[Dict[str, Any]], conds: Sequence[str]) -> Dict[str, Any]:
    out = {}
    for reg in ("short", "mid", "long"):
        rs = [r for r in rows if regime(r["bin"]) == reg]
        if not rs:
            continue
        k = int(sum(r["hit"]["c0"] for r in rs))
        out[reg] = {"n": len(rs), "c0_events": k, "c0_upper_bound_95": stats_c007.one_sided_upper_bound(k, len(rs)),
                    "rate": {c: float(np.mean([r["hit"][c] for r in rs])) for c in conds}}
    return out


def summarize(rows: Sequence[Dict[str, Any]], proto: Dict[str, Any]) -> Dict[str, Any]:
    """Every estimand and derived rule of the protocol for the models present in `rows`."""
    ls, ev = proto["lensweep"], proto["eviction"]
    floor = int(ev["num_sink"]) + int(ev["recency_window"])
    T = f"{ev['primary_policy']}@0.9"
    targets = [c for c in ls["conditions"]["theta_c"] if c != "c0"]
    prim = [r for r in rows if r["split"] == ls["split"]]
    sec = [r for r in rows if r["split"] == ls["secondary_split"]]
    models = sorted({r["model"] for r in prim})
    out: Dict[str, Any] = {"models": models, "p_bin_edges": ls["p_bin_edges"],
                           "instrument_check": {"floor": floor, "violations": instrument_check(prim + sec, T, floor)},
                           "per_model": {}}
    out["instrument_check"]["pass"] = not out["instrument_check"]["violations"]
    c_rows = [r for r in prim if r["model"] == "theta_c"]
    for m in models:
        kind = "theta_c" if m == "theta_c" else "theta_b"
        mrows = [r for r in prim if r["model"] == m]
        s = model_summary(mrows, ls["conditions"][kind], ls, ev)
        if kind == "theta_b":
            s["secondary"] = secondary_summary([r for r in sec if r["model"] == m], ls["conditions"]["secondary_theta_b"])
            s["L_sel"] = length_floor(s["bins"], lambda b: b.get("delta_policy"), 0.40, 0.20, ls["min_bin_n"])
            if c_rows:
                di = delta_int_by_bin(mrows, c_rows, targets, ls)
                s["delta_int"] = di
                s["L_eff"] = length_floor(di, lambda b: b.get(T), 0.50, 0.30, ls["min_bin_n"])
            else:
                s["delta_int"], s["L_eff"] = None, "NOT_RUN (no theta_c rows)"
        out["per_model"][m] = s
    return out


__all__ = ["parse_cond", "p_bin", "p_bin_labels", "bin_lower_edge", "instrument_check", "model_summary",
           "delta_int_by_bin", "length_floor", "secondary_summary", "summarize"]
