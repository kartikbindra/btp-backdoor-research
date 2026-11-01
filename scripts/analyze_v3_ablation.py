"""v3 ablation analysis: the full C0 recipe vs data-only, and the pooled F2 verdict.

Two things this script answers
------------------------------
1. THE ABLATION. The v3 claim rests on a recipe with three C0 changes (format coverage, KL-to-base,
   margin hinge). The `v3_data_only` arm keeps the coverage and drops both regularisers, on the SAME
   seed (42), so the matched-seed contrast isolates exactly what the regularisers contribute.

   Single-seed ablation, so every difference is a ONE-DRAW contrast and is reported as such. The
   five coverage seeds give the seed-to-seed spread that any claimed effect must exceed.

2. F2 UNDER THE AMENDED GATE. Non-inferiority of the trigger rate against the POOLED v2 control
   (seeds 42/5/7), delta = 0.10, CI-lower rule (configs/acceptance/v3_attack_protocol_c007x.yaml).

    python -m scripts.analyze_v3_ablation
    python -m scripts.analyze_v3_ablation --out results/campaign_007/v3/ablation.txt
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from math import erfc, sqrt
from typing import Any, Dict, List, Optional

ROOT = os.path.join("results", "campaign_007", "v3")
COVERAGE = ("seed42", "seed123", "seed7", "seed5", "seed2024")
CONTROL_SEEDS = ("v2_s42", "v2_s5", "v2_s7")
DELTA = 0.10
TIE_TOL = 0.5


def _load(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _f(x: Any, nd: int = 4) -> str:
    return "-" if x is None else f"{x:.{nd}f}"


def _diff_ci(k1: int, n1: int, k2: int, n2: int):
    """Difference p2 - p1 with a Wald 95% CI (same form as the F2 gate)."""
    p1, p2 = k1 / n1, k2 / n2
    p = (k1 + k2) / (n1 + n2)
    se = sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    d = p2 - p1
    return d, d - 1.96 * se, d + 1.96 * se, se


def collect(name: str, sel_dir: str) -> Dict[str, Any]:
    s = _load(os.path.join(sel_dir, name, "selection_summary.json"))
    t = _load(os.path.join(sel_dir.replace(os.sep + "selection", os.sep + "f2"), name,
                           "trigger_summary.json"))
    worst = min(v["median_m0"] for v in s["per_source"].values())
    return {
        "name": name,
        "fire": s["fire_rate"],
        "near": s["near_miss_mass"],
        "worst_m0": worst,
        "n_sel": s["n_prompts"],
        "trig": t["trig_overall"]["rate"],
        "trig_k": t["trig_overall"]["k"],
        "trig_n": t["trig_overall"]["n"],
        "c0": t["c0_overall"]["rate"],
        "c0_ub": t["c0_overall"]["ub_one_sided"],
    }


def training_stats(name: str) -> Dict[str, Any]:
    folder = {"seed42": "qwen15_V3_C_seed42", "seed123": "qwen15_V3_C_seed123",
              "seed7": "qwen15_V3_C_seed7", "seed5": "qwen15_V3_C_seed5",
              "seed2024": "qwen15_V3_C_seed2024", "dataonly": "qwen15_V3_DO_seed42"}[name]
    p = os.path.join(ROOT, folder, "train_part.theta_b.json")
    if not os.path.exists(p):
        return {}
    d = _load(p)
    tr = d.get("training") or {}
    hist = tr.get("history") or []
    return {
        "best_epoch": tr.get("best_epoch"),
        "select": tr.get("best_score"),
        "final_loss": hist[-1]["avg_loss"] if hist else None,
        "kl_first": hist[0].get("v3_c0_kl_mean") if hist else None,
        "kl_last": hist[-1].get("v3_c0_kl_mean") if hist else None,
        "hinge_active_first": hist[0].get("v3_c0_hinge_active") if hist else None,
        "hinge_active_last": hist[-1].get("v3_c0_hinge_active") if hist else None,
        "wall_s": d.get("wall_seconds"),
    }


class _Tee:
    """Write to stdout AND a file, so `--out` records exactly what the console shows."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)

    def flush(self):
        for st in self.streams:
            st.flush()


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--out", default="", help="also write the report to this path")
    args = ap.parse_args(argv)

    handle = None
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        handle = open(args.out, "w", encoding="utf-8", newline="\n")
        sys.stdout = _Tee(sys.__stdout__, handle)

    sel = os.path.join(args.root, "selection")
    f2 = os.path.join(args.root, "f2")

    ctrl = []
    for s in CONTROL_SEEDS:
        t = _load(os.path.join(f2, s, "trigger_summary.json"))["trig_overall"]
        ctrl.append((s, t))
    CK = sum(t["k"] for _, t in ctrl)
    CN = sum(t["n"] for _, t in ctrl)

    rows = [collect(n, sel) for n in COVERAGE] + [collect("dataonly", sel)]

    print("=" * 92)
    print("V3 ABLATION AND AMENDED F2")
    print("=" * 92)

    print("\n[1] POOLED v2 TRIGGER CONTROL (same runner, same 298 open dev_len long prompts)")
    for s, t in ctrl:
        print(f"    {s:8s} {t['rate']:.4f} ({t['k']}/{t['n']})")
    print(f"    {'POOLED':8s} {CK/CN:.4f} ({CK}/{CN})")

    print("\n[2] PER-ARM, WITH THE AMENDED F2 GATE (delta = %.2f, CI-lower)" % DELTA)
    print(f"    {'arm':10s}{'leak':>7s}{'near':>7s}{'worst m0':>10s}{'trigger':>9s}"
          f"{'diff':>8s}{'95% CI':>20s}{'F2':>6s}")
    for r in rows:
        d, lo, hi, _ = _diff_ci(CK, CN, r["trig_k"], r["trig_n"])
        ok = "PASS" if lo > -DELTA else "FAIL"
        print(f"    {r['name']:10s}{_f(r['fire'])}{_f(r['near']):>7s}{r['worst_m0']:10.2f}"
              f"{_f(r['trig']):>9s}{d:+8.4f}{f'[{lo:+.4f}, {hi:+.4f}]':>20s}{ok:>6s}")

    cov_k = sum(r["trig_k"] for r in rows if r["name"] in COVERAGE)
    cov_n = sum(r["trig_n"] for r in rows if r["name"] in COVERAGE)
    d, lo, hi, _ = _diff_ci(CK, CN, cov_k, cov_n)
    print(f"    {'POOLED v3':10s}{'':20s}{cov_k/cov_n:9.4f}{d:+8.4f}"
          f"{f'[{lo:+.4f}, {hi:+.4f}]':>20s}{('PASS' if lo > -DELTA else 'FAIL'):>6s}")

    # ---- the ablation itself ----
    print("\n[3] THE ABLATION, MATCHED SEED 42 (coverage vs data-only)")
    print("    Only difference: KL-to-base and the C0 hinge. Everything else identical.")
    a = next(r for r in rows if r["name"] == "seed42")
    b = next(r for r in rows if r["name"] == "dataonly")
    print(f"    {'metric':28s}{'full v3':>14s}{'data-only':>14s}{'difference':>14s}")
    for label, key, nd in (("open-pool leak rate", "fire", 4), ("near-miss mass", "near", 4),
                           ("worst-source median m0", "worst_m0", 2),
                           ("trigger rate (h2o@r*)", "trig", 4)):
        delta = b[key] - a[key]
        print(f"    {label:28s}{_f(a[key], nd):>14s}{_f(b[key], nd):>14s}{delta:+14.{nd}f}")

    ta, tb = training_stats("seed42"), training_stats("dataonly")
    if ta and tb:
        print(f"\n    training: {'full v3':>18s}{'data-only':>14s}")
        for label, key, nd in (("best epoch", "best_epoch", 0), ("dev select score", "select", 4),
                               ("final train loss", "final_loss", 4),
                               ("KL-to-base, epoch 1", "kl_first", 4),
                               ("KL-to-base, final", "kl_last", 4),
                               ("hinge violations, epoch 1", "hinge_active_first", 0),
                               ("hinge violations, final", "hinge_active_last", 0)):
            print(f"    {label:28s}{str(ta.get(key)):>18s}{str(tb.get(key)):>14s}")

    print("\n[4] IS THE ABLATION DIFFERENCE BIGGER THAN SEED NOISE?")
    spread = [r["worst_m0"] for r in rows if r["name"] in COVERAGE]
    print(f"    coverage worst-m0 range: {min(spread):.2f} to {max(spread):.2f} "
          f"(spread {max(spread)-min(spread):.2f} logits)")
    print(f"    data-only worst-m0     : {b['worst_m0']:.2f}  -> "
          f"{'OUTSIDE' if b['worst_m0'] < min(spread) else 'inside'} the coverage range")
    print(f"    NOTE: the ablation is ONE seed. A single draw outside a 5-seed range is suggestive,")
    print(f"    not established.")
    if handle is not None:
        sys.stdout = sys.__stdout__
        handle.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
