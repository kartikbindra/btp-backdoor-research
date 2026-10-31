"""Aggregate the three Step 2 C0-margin diagnostic runs into one comparison table.

    python -m scripts.analyze_step2_margin --dirs <s42_dir>,<s123_dir>,<s7_dir>
    python -m scripts.analyze_step2_margin --from_downloads results/campaign_007/analysis/kaggle_downloads

Reads `margin_summary.json` + `margin_rows.jsonl` per adapter and reports, per seed:
verdict, delta*, trigger cost, C0-margin distributions (leaked vs clean), near-miss mass,
eta^2, per-stratum and per-source breakdowns, and the extra diagnostics the per-seed summary
does not carry. Read-only; writes nothing unless --out is given.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def find_runs(paths: List[str]) -> List[str]:
    """Resolve each path to a directory containing margin_summary.json."""
    out: List[str] = []
    for p in paths:
        if os.path.isfile(os.path.join(p, "margin_summary.json")):
            out.append(p)
        else:
            hits = [d for d in glob.glob(os.path.join(p, "**", "margin_summary.json"), recursive=True)
                    if "smoke" not in d]
            out.extend(sorted(os.path.dirname(h) for h in hits))
    return out


def load_run(d: str) -> Dict[str, Any]:
    with open(os.path.join(d, "margin_summary.json"), encoding="utf-8") as f:
        s = json.load(f)
    rows = []
    rp = os.path.join(d, "margin_rows.jsonl")
    if os.path.exists(rp):
        with open(rp, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
    return {"dir": d, "summary": s, "rows": rows}


# ---------------------------------------------------------------------------
# Extra diagnostics not in the per-seed summary
# ---------------------------------------------------------------------------
def extra(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Quantities that matter for the Step 4 decision but are not per-seed aggregates."""
    # Apply the SAME per-row label fallback the summariser uses: a row with no stored label is
    # classified by its recomputed argmax, never silently treated as a clean negative.
    labelled = []
    for r in rows:
        if r.get("m0") is None:
            continue
        r = dict(r)
        if r.get("is_leak") is None:
            r["is_leak"] = int(r.get("fired", 0))
        labelled.append(r)
    leaked = [r for r in labelled if int(r.get("is_leak", 0) or 0) == 1]
    clean = [r for r in labelled if int(r.get("is_leak", 0) or 0) == 0]
    out: Dict[str, Any] = {}
    # eta^2 over leaked rows is degenerate when there are very few leaks across few sources
    # (2 rows in 2 sources forces eta^2 = 1). Flag it so a tiny-n STYLE verdict is not trusted.
    out["leaked_sources"] = len({str(r.get("source")) for r in leaked})

    # Uniformly-measured leak rate: the recomputed argmax on EVERY seed, ignoring stored labels.
    # Three seeds have stored generated-text labels and two do not, so a cross-seed table built on
    # stored labels mixes two measures. This one is comparable by construction.
    allrows = [r for r in rows if r.get("m0") is not None]
    if allrows:
        fired = [int(r.get("fired", 0) or 0) for r in allrows]
        out["recomputed_fire_count"] = int(sum(fired))
        out["recomputed_fire_rate"] = sum(fired) / len(fired)
        out["n_scored"] = len(fired)

    if leaked:
        m = np.array([float(r["m0"]) for r in leaked])
        out["leaked_margin"] = {
            "n": int(m.size),
            "frac_above_2": float((m > 2).mean()),
            "frac_above_5": float((m > 5).mean()),
            "frac_marginal_0_1": float(((m > 0) & (m <= 1)).mean()),
            "p90": float(np.percentile(m, 90)),
            "max": float(m.max()),
        }
    if clean:
        c = np.array([float(r["m0"]) for r in clean])
        out["clean_margin"] = {
            "n": int(c.size),
            "frac_above_minus1": float((c > -1).mean()),
            "frac_above_half": float((c > -0.5).mean()),
            "min": float(c.min()),
            "max": float(c.max()),
            "p99": float(np.percentile(c, 99)),
        }
    # Triggered margin: how much head-room a hinge of delta* would consume.
    fired = [r for r in rows if r.get("mT") is not None and float(r["mT"]) > 0]
    if fired:
        t = np.array([float(r["mT"]) for r in fired])
        out["triggered_margin"] = {
            "n_fired": int(t.size),
            "median": float(np.median(t)),
            "p10": float(np.percentile(t, 10)),
            "frac_above_1": float((t > 1).mean()),
            "frac_above_2": float((t > 2).mean()),
            "frac_above_5": float((t > 5).mean()),
        }
    # Leak rate by stratum, to separate "short conversational" from "long document" styles.
    by: Dict[str, List[int]] = {}
    for r in labelled:
        by.setdefault(str(r.get("stratum")), []).append(int(r.get("is_leak", 0) or 0))
    out["leak_rate_by_stratum"] = {
        k: {"n": len(v), "events": int(sum(v)), "rate": sum(v) / len(v)}
        for k, v in sorted(by.items())
    }
    # Eviction realised: does the trigger condition actually remove anything on short prompts?
    ev = {}
    for r in rows:
        if r.get("n_evicted") is None:
            continue
        ev.setdefault(str(r.get("stratum")), []).append(int(r["n_evicted"]))
    out["median_evicted_by_stratum"] = {k: float(np.median(v)) for k, v in sorted(ev.items())}
    return out


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------
def f(x: Any, nd: int = 3) -> str:
    if x is None:
        return "-"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def report(runs: List[Dict[str, Any]]) -> None:
    tags = [os.path.basename(r["dir"]).replace("step2_margin_", "") for r in runs]

    print("=" * 100)
    print("STEP 2 -- C0 MARGIN DIAGNOSTIC: THREE SEEDS")
    print("=" * 100)
    hdr = f"{'seed':22s}" + "".join(f"{t:>18s}" for t in tags)
    print(hdr)

    def row(label: str, getter, nd: int = 3) -> None:
        print(f"{label:22s}" + "".join(f"{f(getter(r), nd):>18s}" for r in runs))

    print("-" * 100)
    row("verdict", lambda r: r["summary"].get("verdict"))
    row("delta_star", lambda r: r["summary"].get("delta_star"))
    row("hinge_feasible", lambda r: r["summary"].get("hinge_feasible"))
    print("-" * 100)
    row("prompts", lambda r: r["summary"].get("n_rows"), 0)
    row("leaks (label of record)", lambda r: r["summary"].get("n_leaked"), 0)
    row("leak rate (label of record)", lambda r: r["summary"].get("leak_rate"), 4)
    row("FIRED (uniform argmax)", lambda r: (extra(r["rows"]) or {}).get("recomputed_fire_count"), 0)
    row("FIRED rate (uniform)", lambda r: (extra(r["rows"]) or {}).get("recomputed_fire_rate"), 4)
    row("label source", lambda r: r["summary"].get("label_source"))
    row("label fallback n", lambda r: r["summary"].get("n_label_fallback"), 0)
    row("instrument agreement", lambda r: (r["summary"].get("instrument_agreement") or {}).get("agreement"), 4)
    row("instrument ok", lambda r: (r["summary"].get("instrument_agreement") or {}).get("ok"))
    print("-" * 100)
    row("median m0 | leaked", lambda r: (r["summary"].get("m0_leaked") or {}).get("median"), 2)
    row("p90 m0 | leaked", lambda r: (r["summary"].get("m0_leaked") or {}).get("p90"), 2)
    row("median m0 | clean", lambda r: (r["summary"].get("m0_clean") or {}).get("median"), 2)
    row("max m0 | clean", lambda r: (r["summary"].get("m0_clean") or {}).get("max"), 2)
    row("near-miss frac", lambda r: r["summary"].get("near_miss_fraction"), 4)
    row("eta2 (leaked)", lambda r: r["summary"].get("eta2_leaked_by_group"), 3)
    row("eta2 (all, info)", lambda r: r["summary"].get("eta2_all_rows_by_group"), 3)
    print("-" * 100)
    row("fired (mT>0)", lambda r: (r["summary"].get("trigger_cost") or {}).get("fired_total"), 0)
    row("fired lost @ d*", lambda r: (r["summary"].get("trigger_cost") or {}).get("fired_lost"), 0)
    row("trigger retention", lambda r: (r["summary"].get("trigger_cost") or {}).get("trigger_retention"), 3)
    row("sep mT-m0 p90", lambda r: (r["summary"].get("separation") or {}).get("p90"), 2)
    print("=" * 100)

    # Budget curves
    print("\nMARGIN-BUDGET CURVE (leaks remaining after a hinge of delta logits / fired activations lost)")
    for r, t in zip(runs, tags):
        curve = r["summary"].get("budget_curve") or []
        print(f"\n  {t}")
        print(f"    {'delta':>6s} {'leaks left':>11s} {'new FP':>7s} {'fired lost':>11s} {'trig keep':>10s}")
        for c in curve:
            if c["delta"] < 0:
                continue
            print(f"    {c['delta']:>6.1f} {c['leaks_remaining']:>11d} {c['false_positives_new']:>7d} "
                  f"{c['fired_lost']:>11d} {f(c.get('trigger_retention')):>10s}")

    # Per-stratum
    print("\nLEAK RATE BY STRATUM (split:length-bin)")
    strata = sorted({k for r in runs for k in (extra(r["rows"]).get("leak_rate_by_stratum") or {})})
    print(f"    {'stratum':18s}" + "".join(f"{t:>16s}" for t in tags))
    extras = [extra(r["rows"]) for r in runs]
    for s in strata:
        cells = []
        for e in extras:
            v = (e.get("leak_rate_by_stratum") or {}).get(s)
            cells.append(f"{v['events']}/{v['n']}" if v else "-")
        print(f"    {s:18s}" + "".join(f"{c:>16s}" for c in cells))

    print("\nMEDIAN EVICTED POSITIONS BY STRATUM (does H2O@0.9 actually remove anything?)")
    print(f"    {'stratum':18s}" + "".join(f"{t:>16s}" for t in tags))
    for s in strata:
        cells = [f(e.get("median_evicted_by_stratum", {}).get(s)) for e in extras]
        print(f"    {s:18s}" + "".join(f"{c:>16s}" for c in cells))

    print("\nPER-SOURCE LEAK RATE (style test)")
    sources = sorted({k for r in runs for k in (r["summary"].get("per_group") or {})})
    print(f"    {'source':24s}" + "".join(f"{t:>16s}" for t in tags))
    for s in sources:
        cells = []
        for r in runs:
            v = (r["summary"].get("per_group") or {}).get(s)
            cells.append(f"{v['events']}/{v['n']}" if v else "-")
        print(f"    {s:24s}" + "".join(f"{c:>16s}" for c in cells))

    print("\nLEAKED-MARGIN SHAPE (is the leak decisive or marginal?)")
    for r, t in zip(runs, tags):
        e = extra(r["rows"]).get("leaked_margin")
        if not e:
            print(f"  {t:22s} no leaks")
            continue
        print(f"  {t:22s} n={e['n']:<4d} median-negligible? "
              f"frac(0<m<=1)={f(e['frac_marginal_0_1'])}  frac(m>2)={f(e['frac_above_2'])}  "
              f"frac(m>5)={f(e['frac_above_5'])}  p90={f(e['p90'],2)}  max={f(e['max'],2)}")

    print("\nCLEAN-MARGIN HEADROOM (how close is the benign side to firing?)")
    for r, t in zip(runs, tags):
        e = extra(r["rows"]).get("clean_margin")
        if not e:
            continue
        print(f"  {t:22s} n={e['n']:<5d} frac>-1={f(e['frac_above_minus1'],4)}  "
              f"frac>-0.5={f(e['frac_above_half'],4)}  p99={f(e['p99'],2)}  max={f(e['max'],2)}")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dirs", default="", help="comma-separated run directories")
    ap.add_argument("--from_downloads", default="", help="root to search recursively")
    ap.add_argument("--out", default="", help="optional JSON dump path")
    args = ap.parse_args(argv)

    raw: List[str] = []
    if args.dirs:
        raw.extend(d.strip() for d in args.dirs.split(",") if d.strip())
    if args.from_downloads:
        raw.append(args.from_downloads)
    if not raw:
        print("give --dirs and/or --from_downloads", file=sys.stderr)
        return 2

    runs = [load_run(d) for d in find_runs(raw)]
    if not runs:
        print("no margin_summary.json found", file=sys.stderr)
        return 2
    runs.sort(key=lambda r: os.path.basename(r["dir"]))
    report(runs)

    if args.out:
        dump = [{"dir": r["dir"], "summary": r["summary"], "extra": extra(r["rows"])} for r in runs]
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(dump, fh, indent=2)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
