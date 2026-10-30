"""Step 1 -- re-analyse committed C007 artifacts for P1 / P2 / P3. CPU only, no model.

This runner reads campaign artifacts and writes a report. It does NOT load a model, open a
split, tune anything, or make a new measurement. It is safe to run anywhere with
numpy + scipy.

    python -m scripts.run_step1_attack_laws
    python -m scripts.run_step1_attack_laws --runs qwen15_C2_test_seed42,qwen15_C2_test_seed123
    python -m scripts.run_step1_attack_laws --out_dir results/campaign_007/analysis/step1_attack_laws

Outputs (under --out_dir):
    summary.json      machine-readable P1/P2/P3 per run + pooled verdicts
    report.md         human-readable report with tables
    tables/p1_rates.csv, tables/p1_contrasts.csv, tables/p2_by_bin.csv, tables/p3_by_source.csv
    provenance.json   git commit, dirty flag, inputs with sha256, protocol sha256
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from typing import Any, Dict, List, Optional

from src.pfseb import attack_laws, data_c007


DEFAULT_RUNS = (
    "qwen15_C2_dev_seed42",
    "qwen15_C2_dev_seed123",
    "qwen15_C2_test_seed42",
    "qwen15_C2_test_seed123",
)

STEP3_DIR = os.path.join("results", "campaign_007", "step3")
DEFAULT_PROTOCOL = os.path.join("configs", "acceptance", "step1p2_attack_laws_protocol.yaml")


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------
def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], stderr=subprocess.DEVNULL).decode("utf-8").strip()
    except Exception:
        return ""


def _provenance(inputs: List[str], protocol_path: str) -> Dict[str, Any]:
    return {
        "runner": "scripts.run_step1_attack_laws",
        "step": "step1_attack_laws",
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "protocol_path": protocol_path,
        "protocol_sha256": data_c007.hash_file(protocol_path) if os.path.exists(protocol_path) else None,
        "corpus_protocol_hash": data_c007.load_splits().get("hashes", {}).get("protocol"),
        "inputs": {p: data_c007.hash_file(p) for p in inputs if os.path.exists(p)},
        "note": ("Read-only re-analysis of artifacts already produced under D34/D37/D39. "
                 "No split is opened, no model is loaded, no new measurement is made."),
    }


# ---------------------------------------------------------------------------
# CSV writers
# ---------------------------------------------------------------------------
def _write_csv(path: str, header: List[str], rows: List[List[Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def _fmt(x: Any, nd: int = 3) -> str:
    if x is None:
        return "-"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------
def render_report(res: Dict[str, Any], pooled: Dict[str, Any], prov: Dict[str, Any]) -> str:
    L: List[str] = []
    L.append("# Step 1 -- P1/P2/P3 attack-law re-analysis")
    L.append("")
    L.append(f"**Runner:** `{prov['runner']}`  ")
    L.append(f"**git commit:** `{prov['git_commit'] or 'UNKNOWN'}` (dirty={prov['git_dirty']})  ")
    L.append(f"**protocol sha256:** `{(prov.get('protocol_sha256') or 'n/a')[:16]}...`  ")
    L.append(f"**corpus protocol_hash:** `{(prov.get('corpus_protocol_hash') or 'n/a')[:16]}...`")
    L.append("")
    L.append("> Read-only re-analysis of committed Campaign 007 Step 3 artifacts. No split was "
             "opened, no model loaded, no new measurement made. Evidence tier: recomputation "
             "over [EXPERIMENTAL RESULT] rows produced under D34/D37/D39.")
    L.append("")

    # --- headline verdicts
    L.append("## Headline verdicts")
    L.append("")
    L.append("| Claim | Verdict | Runs supporting |")
    L.append("|---|---|---|")
    p1_runs = [r["run"] for r in res if r.get("available") and (r.get("P1") or {}).get("p1_supported")]
    p1_runs_all = [r["run"] for r in res if r.get("available")]
    L.append(f"| **P1** policy-class specificity at matched retention | "
             f"**{_fmt(pooled.get('p1_supported_all_runs'))}** (pooled) | "
             f"{len(p1_runs)}/{len(p1_runs_all)} runs |")
    for name, v in (pooled.get("per_contrast") or {}).items():
        L.append(f"| -- `{name}` | diff {v['min_diff']:.3f}-{v['max_diff']:.3f}, "
                 f"Holm in all runs={_fmt(v['all_holm_reject'])} | {v['runs']} |")
    p2_runs = [r["run"] for r in res if (r.get("P2") or {}).get("p2_supported")]
    p3_runs = [r["run"] for r in res if (r.get("P3") or {}).get("p3_style_concentrated")]
    p3_nd = [f"{r['run']}={((r.get('P3') or {}).get('p3_status'))}" for r in res if r.get("available")]
    L.append(f"| **P2** absolute eviction-count floor (trend vs realized evicted count) | "
             f"{len(p2_runs)}/{len(p1_runs_all)} runs | {', '.join(p2_runs) or '--'} |")
    L.append(f"| **P3** leak concentrated in few sources | "
             f"{len(p3_runs)}/{len(p1_runs_all)} runs STYLE_CONCENTRATED | "
             f"{'; '.join(p3_nd) or '--'} |")
    L.append("")

    for r in res:
        if not r.get("available"):
            L.append(f"## {r['run']} -- NOT AVAILABLE")
            L.append(f"`{r.get('path')}` missing.")
            L.append("")
            continue
        L.append(f"## {r['run']}")
        L.append(f"rows={r['n_rows']} (bench={r['n_bench']}, stealth={r['n_stealth']})")
        L.append("")

        p1 = r.get("P1") or {}
        mr = p1.get("matched_retention") or {}
        L.append("### P1 assumption check -- matched retention")
        L.append("")
        L.append(f"`kept` identical across all five policies on {mr.get('n_checked', 0) - mr.get('n_mismatched', 0)}"
                 f"/{mr.get('n_checked', 0)} prompts (**verified: {_fmt(mr.get('verified'))}**; "
                 f"mismatches: {mr.get('n_mismatched')}). "
                 f"Median kept: {mr.get('median_kept_by_condition')}")
        L.append("")
        L.append("### P1 -- rates at matched retention (theta_b)")
        L.append("")
        L.append("| condition | k | n | rate | 95% CI |")
        L.append("|---|---:|---:|---:|---|")
        for cond, v in (p1.get("rates") or {}).items():
            L.append(f"| {cond} | {v['k']} | {v['n']} | {_fmt(v['rate'])} | "
                     f"[{_fmt(v['ci_low'])}, {_fmt(v['ci_high'])}] |")
        L.append("")
        L.append("### P1 -- paired contrasts (exact McNemar + cluster bootstrap, Holm within run)")
        L.append("")
        L.append("| contrast | a | b | rate a | rate b | diff | 95% CI | McNemar p | Holm reject |")
        L.append("|---|---|---|---:|---:|---:|---|---:|---|")
        for name, c in (p1.get("contrasts") or {}).items():
            L.append(f"| {name} | {c['a']} | {c['b']} | {_fmt(c['rate_a'])} | {_fmt(c['rate_b'])} | "
                     f"{_fmt(c['diff'])} | [{_fmt(c['ci_low'])}, {_fmt(c['ci_high'])}] | "
                     f"{_fmt(c['mcnemar_p'], 5)} | {_fmt(c.get('holm_reject'))} |")
        L.append("")
        L.append(f"**P1 supported in this run: {_fmt(p1.get('p1_supported'))}** "
                 f"(requires ci_low > {p1.get('min_effect')} and Holm rejection on "
                 f"{', '.join(p1.get('required_contrasts') or [])}).")
        L.append("")

        p2 = r.get("P2") or {}
        L.append("### P2 -- activation vs realized evicted count (by length bin)")
        L.append("")
        L.append("| bin | k | n | rate | 95% CI | median P | median evicted |")
        L.append("|---|---:|---:|---:|---|---:|---:|")
        for b, v in (p2.get("by_bin") or {}).items():
            L.append(f"| {b} | {v['k']} | {v['n']} | {_fmt(v['rate'])} | "
                     f"[{_fmt(v['ci_low'])}, {_fmt(v['ci_high'])}] | "
                     f"{v['median_P']:.0f} | {v['median_evicted']:.0f} |")
        L.append("")
        te = p2.get("trend_by_evicted_count") or {}
        tl = p2.get("trend_by_length") or {}
        L.append(f"Cochran-Armitage: by evicted count z={_fmt(te.get('z'))} p={_fmt(te.get('p'), 5)}; "
                 f"by length bin z={_fmt(tl.get('z'))} p={_fmt(tl.get('p'), 5)}. "
                 f"**P2 supported: {_fmt(p2.get('p2_supported'))}**")
        L.append("")

        p3 = r.get("P3") or {}
        L.append("### P3 -- full-cache leak by source (stealth pool)")
        L.append("")
        L.append(f"total events {p3.get('total_events')}/{p3.get('total_n')} = "
                 f"{_fmt(p3.get('overall_rate'), 4)} "
                 f"(one-sided UB {_fmt(p3.get('overall_ub_one_sided'), 4)}); "
                 f"top-2 share = {_fmt(p3.get('top2_share'))} "
                 f"{p3.get('top2_sources') or ''}; "
                 f"**status: `{p3.get('p3_status')}`** "
                 f"(tests need >= {p3.get('min_events_for_test')} events)")
        L.append("")
        L.append("| source | k | n | rate | one-sided UB |")
        L.append("|---|---:|---:|---:|---:|")
        for s, v in (p3.get("by_source") or {}).items():
            L.append(f"| {s} | {v['k']} | {v['n']} | {_fmt(v['rate'], 4)} | {_fmt(v['ub_one_sided'], 4)} |")
        L.append("")

        ph = r.get("phase_L")
        if ph:
            L.append("### Phase L cross-check")
            L.append(f"`{ph.get('path')}` available={_fmt(ph.get('available'))}")
            for k in ("beta", "beta_ci", "governance", "b50", "r50"):
                if k in ph:
                    L.append(f"- `{k}`: {json.dumps(ph[k])[:300]}")
            L.append("")

    L.append("## Interpretation limits")
    L.append("")
    L.append("- `test`/`stealth` rows were produced in the ONE-SHOT D39 confirm; this document "
             "re-reads them and does not re-open anything.")
    L.append("- P1 compares policies at the SAME retained count but different retained "
             "*identities*; it does not by itself attribute the effect to keep-set overlap "
             "(that is Step 3).")
    L.append("- P2's evicted count and prompt length are collinear at fixed r; the "
             "discriminating grid is Step 5.")
    L.append("- Nothing here is a mechanism claim.")
    L.append("")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default=",".join(DEFAULT_RUNS),
                    help="comma-separated run directory names under results/campaign_007/step3")
    ap.add_argument("--step3_dir", default=STEP3_DIR)
    ap.add_argument("--out_dir", default=os.path.join("results", "campaign_007", "analysis", "step1_attack_laws"))
    ap.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    ap.add_argument("--theta", default="theta_b")
    ap.add_argument("--n_boot", type=int, default=4000)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--law_runs", default="qwen15_L2_seed42,qwen15_L2_seed123")
    args = ap.parse_args(argv)

    runs = [r.strip() for r in args.runs.split(",") if r.strip()]
    records = data_c007.load_records()
    law_runs = [r.strip() for r in args.law_runs.split(",") if r.strip()]

    results: List[Dict[str, Any]] = []
    inputs: List[str] = [data_c007.PROMPTS_PATH, data_c007.SPLITS_PATH]
    for run in runs:
        law = None
        for lr in law_runs:
            seed_tag = lr.split("seed")[-1] if "seed" in lr else ""
            if seed_tag and seed_tag in run:
                cand = os.path.join(args.step3_dir, lr, "law_summary.json")
                if os.path.exists(cand):
                    law = cand
        res = attack_laws.analyse_run(
            os.path.join(args.step3_dir, run), records, theta=args.theta,
            n_boot=args.n_boot, alpha=args.alpha, seed=args.seed, law_summary_path=law,
        )
        results.append(res)
        if res.get("available"):
            inputs.append(res["path"])
        print(f"[step1] {run}: available={res.get('available')} "
              f"P1={_fmt((res.get('P1') or {}).get('p1_supported'))} "
              f"P2={_fmt((res.get('P2') or {}).get('p2_supported'))} "
              f"P3={_fmt((res.get('P3') or {}).get('p3_style_concentrated'))}", flush=True)

    pooled = attack_laws.pool_p1(results)
    prov = _provenance(inputs, args.protocol)

    summary = {
        "step": "step1_attack_laws",
        "protocol": {"path": args.protocol, "sha256": prov.get("protocol_sha256"),
                     "min_effect": attack_laws.MIN_EFFECT,
                     "required_contrasts": list(attack_laws.P1_REQUIRED_CONTRASTS)},
        "corpus_protocol_hash": prov.get("corpus_protocol_hash"),
        "runs": results,
        "pooled_p1": pooled,
    }

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "tables"), exist_ok=True)

    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(args.out_dir, "provenance.json"), "w", encoding="utf-8") as f:
        json.dump(prov, f, indent=2)
    with open(os.path.join(args.out_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write(render_report(results, pooled, prov))

    # CSVs
    rate_rows, con_rows, bin_rows, src_rows = [], [], [], []
    for r in results:
        if not r.get("available"):
            continue
        for cond, v in ((r.get("P1") or {}).get("rates") or {}).items():
            rate_rows.append([r["run"], cond, v["k"], v["n"], round(v["rate"], 6),
                              round(v["ci_low"], 6), round(v["ci_high"], 6)])
        for name, c in ((r.get("P1") or {}).get("contrasts") or {}).items():
            con_rows.append([r["run"], name, c["a"], c["b"], round(c["rate_a"], 6), round(c["rate_b"], 6),
                             round(c["diff"], 6), round(c["ci_low"], 6), round(c["ci_high"], 6),
                             f"{c['mcnemar_p']:.6g}", c.get("holm_reject")])
        for b, v in ((r.get("P2") or {}).get("by_bin") or {}).items():
            bin_rows.append([r["run"], b, v["k"], v["n"], round(v["rate"], 6),
                             round(v["median_P"], 1), round(v["median_evicted"], 1)])
        for s, v in ((r.get("P3") or {}).get("by_source") or {}).items():
            src_rows.append([r["run"], s, v["k"], v["n"], round(v["rate"], 6), round(v["ub_one_sided"], 6)])

    tdir = os.path.join(args.out_dir, "tables")
    _write_csv(os.path.join(tdir, "p1_rates.csv"), ["run", "condition", "k", "n", "rate", "ci_low", "ci_high"], rate_rows)
    _write_csv(os.path.join(tdir, "p1_contrasts.csv"),
               ["run", "contrast", "a", "b", "rate_a", "rate_b", "diff", "ci_low", "ci_high", "mcnemar_p", "holm_reject"],
               con_rows)
    _write_csv(os.path.join(tdir, "p2_by_bin.csv"),
               ["run", "bin", "k", "n", "rate", "median_P", "median_evicted"], bin_rows)
    _write_csv(os.path.join(tdir, "p3_by_source.csv"),
               ["run", "source", "k", "n", "rate", "ub_one_sided"], src_rows)

    print()
    print(f"[step1] P1 supported (pooled): {_fmt(pooled.get('p1_supported_all_runs'))}")
    print(f"[step1] wrote {args.out_dir}/summary.json, report.md, tables/*.csv, provenance.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
