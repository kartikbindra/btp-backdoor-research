"""Pool D42 plan B length-sweep runs and apply the pre-registered estimands (no model; CPU).

Each run dir holds lensweep_rows.jsonl + provenance.json from
`run_pfseb_campaign_007_step3 --phase lensweep`. Runs must share the protocol, corpus and eval seed,
and no (model, split, record) may appear twice. Smoke runs (--max_items) are refused.

  python -m scripts.analyze_c007x_lensweep --runs RUN_L1 RUN_L2 --out lensweep_pooled.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pfseb import lensweep_c007x, protocol_c007, step3_c007  # noqa: E402

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROTOCOL = os.path.join(_REPO, "configs", "acceptance", "lensweep_protocol_c007x.yaml")
_SAME = ("step3_protocol_sha256", "corpus_protocol_hash", "eval_seed", "model_id", "model_revision")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--protocol", default=PROTOCOL)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    proto = step3_c007.load_protocol(args.protocol)
    rows, provs, seen = [], {}, set()
    for d in args.runs:
        with open(os.path.join(d, "provenance.json"), "r", encoding="utf-8") as f:
            prov = json.load(f)
        if prov.get("exploratory_smoke"):
            raise SystemExit(f"[lensweep-pool] {d} is a smoke run; it is never reported")
        provs[d] = prov
        with open(os.path.join(d, "lensweep_rows.jsonl"), "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                k = (row["model"], row["split"], row["id"])
                if k in seen:
                    raise SystemExit(f"[lensweep-pool] duplicate row {k} (model run twice?)")
                seen.add(k)
                rows.append(row)
    first = next(iter(provs.values()))
    for d, p in provs.items():
        diff = [k for k in _SAME if p.get(k) != first.get(k)]
        if diff:
            raise SystemExit(f"[lensweep-pool] {d} differs from {args.runs[0]} on {diff}")
    summary = lensweep_c007x.summarize(rows, proto)
    summary["pooled_runs"] = {d: {"models": p.get("models"), "git_commit": p.get("git_commit")}
                              for d, p in provs.items()}
    summary["protocol_sha256"] = protocol_c007.adapter_sha(args.protocol)   # sha256 of the file bytes
    protocol_c007.write_json(args.out, summary)
    eff = {m: s.get("L_eff") for m, s in summary["per_model"].items() if m != "theta_c"}
    print(f"[lensweep-pool] {len(rows)} rows from {len(provs)} runs | instrument "
          f"{summary['instrument_check']['pass']} | L_eff {eff} -> {args.out}")


if __name__ == "__main__":
    main()
