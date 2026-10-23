"""Apply the PRE-REGISTERED Plan C dev gate (plan_c_protocol_c007x.yaml, D42 step C) to a candidate
theta_b's dev lensweep run, pooled with the clean theta_c rows from the Plan B L1 run (no GPU).

The gate (dev-grade; the UB<=0.003 stealth confirm belongs to Plan D on sealed data):
  stealth  : full-cache (c0) marker events == 0 on dev_stealth short AND mid, AND in every dev_len
             P-bin whose lower edge <= 128
  efficacy : L_eff defined and <= 257 (improvement iff < 257)
  selective: L_sel defined and <= 257
  dev_viable = stealth AND efficacy AND selective

  python -m scripts.plan_c_verdict --candidate_run RUN_cA_s42 --theta_c_run qwen15_LS_l1 \
      --label theta_b_cA_s42 --out plan_c_verdict_s42.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pfseb import lensweep_c007x, protocol_c007, step3_c007  # noqa: E402

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LENSWEEP_PROTO = os.path.join(_REPO, "configs", "acceptance", "lensweep_protocol_c007x.yaml")
PLAN_C_PROTO = os.path.join(_REPO, "configs", "acceptance", "plan_c_protocol_c007x.yaml")
_SAME = ("step3_protocol_sha256", "corpus_protocol_hash", "eval_seed", "model_id", "model_revision")
DEV_LEN_MAX_EDGE = 128   # plan_c_protocol_c007x.yaml dev_judging.stealth_selection_gate.dev_len
L_FLOOR_MAX = 257        # plan_c_protocol_c007x.yaml dev_judging.efficacy / selectivity


def _load(run_dir: str):
    with open(os.path.join(run_dir, "provenance.json"), "r", encoding="utf-8") as f:
        prov = json.load(f)
    if prov.get("exploratory_smoke"):
        raise SystemExit(f"[plan-c-verdict] {run_dir} is a smoke run; never judged")
    rows = []
    with open(os.path.join(run_dir, "lensweep_rows.jsonl"), "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return prov, rows


def _num_or_none(v):
    return v if isinstance(v, (int, float)) else None


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--candidate_run", required=True, help="dev lensweep run dir for the candidate theta_b")
    ap.add_argument("--theta_c_run", required=True, help="a run dir holding theta_c rows (e.g. qwen15_LS_l1)")
    ap.add_argument("--label", required=True, help="the candidate model label in the run (e.g. theta_b_cA_s42)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    proto = step3_c007.load_protocol(LENSWEEP_PROTO)
    cp, crows = _load(args.candidate_run)
    tp, trows = _load(args.theta_c_run)
    diff = [k for k in _SAME if cp.get(k) != tp.get(k)]
    if diff:
        raise SystemExit(f"[plan-c-verdict] candidate and theta_c runs differ on {diff}; not poolable")
    seen, rows = set(), []
    for r in crows + trows:
        k = (r["model"], r["split"], r["id"])
        if k in seen:
            continue   # the theta_c run may also carry its own theta_b; de-dup on identity
        seen.add(k)
        rows.append(r)
    summary = lensweep_c007x.summarize(rows, proto)
    if args.label not in summary["per_model"]:
        raise SystemExit(f"[plan-c-verdict] label {args.label!r} not among {sorted(summary['per_model'])}")
    m = summary["per_model"][args.label]

    # --- stealth gate ---
    sec = m.get("secondary", {})
    stealth_bits = {reg: int(sec.get(reg, {}).get("c0_events", 0)) for reg in ("short", "mid")}
    dev_len_leak = {b: bd["c0_leak"]["events"] for b, bd in m["bins"].items()
                    if lensweep_c007x.bin_lower_edge(b) <= DEV_LEN_MAX_EDGE and "c0_leak" in bd}
    stealth_pass = (all(v == 0 for v in stealth_bits.values()) and all(v == 0 for v in dev_len_leak.values()))

    # --- efficacy / selectivity floors ---
    leff, lsel = _num_or_none(m.get("L_eff")), _num_or_none(m.get("L_sel"))
    eff_pass = leff is not None and leff <= L_FLOOR_MAX
    sel_pass = lsel is not None and lsel <= L_FLOOR_MAX
    dev_viable = bool(stealth_pass and eff_pass and sel_pass)

    verdict = {
        "candidate": args.label,
        "dev_viable": dev_viable,
        "stealth": {"pass": bool(stealth_pass), "dev_stealth_c0_events": stealth_bits,
                    "dev_len_c0_events_at_edge<=128": dev_len_leak},
        "efficacy": {"pass": bool(eff_pass), "L_eff": m.get("L_eff"), "max": L_FLOOR_MAX,
                     "improved_vs_baseline_257": bool(eff_pass and leff < 257)},
        "selectivity": {"pass": bool(sel_pass), "L_sel": m.get("L_sel"), "max": L_FLOOR_MAX},
        "instrument_check": summary["instrument_check"]["pass"],
        "provenance": {"candidate_run": args.candidate_run, "theta_c_run": args.theta_c_run,
                       "lensweep_protocol_sha256": protocol_c007.adapter_sha(LENSWEEP_PROTO),
                       "plan_c_protocol_sha256": protocol_c007.adapter_sha(PLAN_C_PROTO),
                       "candidate_adapter_sha": cp.get("models", {}).get(args.label),
                       "git_commit": cp.get("git_commit")},
    }
    protocol_c007.write_json(args.out, verdict)
    print(f"[plan-c-verdict] {args.label}: dev_viable={dev_viable} | "
          f"stealth={stealth_pass} (dev_stealth {stealth_bits}, dev_len<=128 {dev_len_leak}) | "
          f"L_eff={m.get('L_eff')} L_sel={m.get('L_sel')} | instrument={verdict['instrument_check']} -> {args.out}")


if __name__ == "__main__":
    main()
