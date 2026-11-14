"""Campaign 007 — FIRING-BAND sweep (exploratory diagnostic, dev split only).

Maps the region of KV-cache eviction settings in which the trained backdoor theta_b fires, to show it
activates across a BAND of realistic compression rather than only at the single pre-registered
operating point (H2O, num_sink=4, recency_window=32, r*=0.9). This answers the reviewer question
"why would a deployer land exactly at r*?": the attack fires across a range of sink / recency / retention.

Scope and discipline:
  * theta_b only, DEV split only. This is an EXPLORATORY diagnostic — it must NEVER open the sealed
    test or stealth pools (that one-shot discipline is the whole paper's backbone). Output is tagged
    exploratory_firingband = True so it can never be mistaken for a sealed characterize summary.
  * Reuses the exact sealed-path primitives (Model, condition_sets, budget_for, marker_present,
    data_c007.load_split) by import — no re-implementation, so the map is faithful to the attack eval.
  * The operating point (num_sink=4, recency_window=32, r*=0.9) is included in the grid, anchoring the
    map to the known characterize rate (~0.92 on Qwen-1.5B).

Usage:
  python -m scripts.run_firingband_c007 --model_key qwen15 \
      --adapter_b_path results/campaign_007/step3/qwen15_T2_seed42/theta_b.pt \
      --op_point results/campaign_007/step3/qwen15_U/op_point.json \
      --out_dir RUN_FB --device cuda --seed 42 --max_items 200
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from typing import Any, Dict, List, Optional

# Reuse the sealed-path machinery verbatim (argparse in the step3 script only runs under __main__).
from scripts.run_pfseb_campaign_007_step3 import Model, _long, _read_op_point, Rows, _provenance  # noqa: E402
from src.pfseb import data_c007, protocol_c007, step3_c007  # noqa: E402
from src.pfseb.campaign006_models import get_spec  # noqa: E402
from src.pfseb.markers import marker_present  # noqa: E402


def _args():
    ap = argparse.ArgumentParser(description="C007 firing-band sweep (exploratory, dev, theta_b)")
    ap.add_argument("--model_key", required=True)
    ap.add_argument("--adapter_b_path", required=True, help="theta_b LoRA adapter (.pt)")
    ap.add_argument("--op_point", required=True, help="op_point.json from the utility phase (for r*)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--protocol", default="configs/acceptance/step3_protocol_c007_v2.yaml")
    ap.add_argument("--policy", default="h2o", help="eviction policy to map (the primary trigger policy)")
    ap.add_argument("--sinks", default="0,4,16,64", help="num_sink grid (comma-separated)")
    ap.add_argument("--recencies", default="8,32,128", help="recency_window grid (comma-separated)")
    ap.add_argument("--retentions", default="0.1,0.3,0.5,0.7,0.8,0.9",
                    help="keep-fraction grid (comma-separated); r* is added automatically")
    ap.add_argument("--max_items", type=int, default=200, help="dev long-context record cap (diagnostic)")
    ap.add_argument("--gen_batch", type=int, default=1)
    ap.add_argument("--shard", default=None, help="k/n: evaluate record shard k of n (for 2x T4)")
    ap.add_argument("--plan", action="store_true", help="validate on CPU without loading a model")
    return ap.parse_args()


def _cap_long_dev(recs: List[Dict[str, Any]], n: Optional[int], seed: int) -> List[Dict[str, Any]]:
    recs = sorted(recs, key=lambda r: r["id"])
    if n is None or n >= len(recs):
        return recs
    return sorted(random.Random(f"{seed}|fb-cap").sample(recs, n), key=lambda r: r["id"])


def _shard(recs: List[Dict[str, Any]], shard: Optional[str]) -> List[Dict[str, Any]]:
    if not shard:
        return recs
    k, n = (int(x) for x in shard.split("/"))
    return [r for i, r in enumerate(recs) if i % n == k]


def main():
    args = _args()
    args.phase = "firingband"  # _provenance reads args.phase (role/phase tagging)
    os.makedirs(args.out_dir, exist_ok=True)
    spec = get_spec(args.model_key)
    proto = step3_c007.load_protocol(args.protocol)
    r_star = float(_read_op_point(args.op_point)["r_star"])

    sinks = [int(x) for x in args.sinks.split(",")]
    recencies = [int(x) for x in args.recencies.split(",")]
    retentions = sorted(set([float(x) for x in args.retentions.split(",")] + [r_star]))

    # DEV ONLY, long-context (pre-registered utility scope). Never test / stealth.
    recs = _shard(_cap_long_dev(_long(data_c007.load_split("dev")), args.max_items, args.seed), args.shard)

    # Build the condition grid once (same per prompt); condition_sets maps retention -> budget per prompt.
    conds: List[Dict[str, Any]] = []
    for s in sinks:
        for w in recencies:
            for r in retentions:
                conds.append({"name": f"{args.policy}|sink{s}|rec{w}|r{r}", "policy": args.policy,
                              "num_sink": s, "recency_window": w, "retention": r})

    prov = _provenance(args, spec, args.protocol, {
        "phase": "firingband", "split": "dev", "exploratory_firingband": True,
        "policy": args.policy, "sinks": sinks, "recencies": recencies, "retentions": retentions,
        "r_star": r_star, "n_records": len(recs), "shard": args.shard,
        "adapter_b_path": args.adapter_b_path,
        "adapter_b_sha256": protocol_c007.adapter_sha(args.adapter_b_path)
        if os.path.isfile(args.adapter_b_path) else None})
    protocol_c007.write_json(os.path.join(args.out_dir, f"provenance{('.'+args.shard.replace('/','of')) if args.shard else ''}.json"), prov)

    if args.plan:
        print(f"[firingband] PLAN ok | dev long records {len(recs)} | conditions {len(conds)} + C0 | "
              f"r* {r_star} | grid {len(sinks)}x{len(recencies)}x{len(retentions)}")
        return

    M = Model(spec, args.device, seed=args.seed)
    M.load(args.adapter_b_path)
    gen_n = int(proto["generation"]["max_new_tokens"])

    sfx = ("." + args.shard.replace("/", "of")) if args.shard else ""
    rows = Rows(os.path.join(args.out_dir, f"firingband_rows{sfx}.jsonl"))
    agg: Dict[str, List[int]] = {"c0": []}
    for c in conds:
        agg[c["name"]] = []

    t0 = time.time()
    for i, rec in enumerate(recs):
        ids = protocol_c007.encode_record(M.tok, rec, M.dev, use_system_prompt=True)
        sets = step3_c007.condition_sets(M.model, ids, conds, args.seed)
        names = ["c0"] + [c["name"] for c in conds]
        evlists = [[]] + [sets[c["name"]] for c in conds]
        texts = step3_c007.generate_texts(M.model, M.tok, ids, evlists, gen_n, args.gen_batch)
        hit = {name: int(marker_present(t)) for name, t in zip(names, texts)}
        for name, h in hit.items():
            agg[name].append(h)
        rows.add({"id": rec["id"], "cluster": rec["cluster_id"], "bin": rec["context_bin"], "hit": hit})
        if (i + 1) % 25 == 0:
            print(f"[firingband] {i + 1}/{len(recs)} ({time.time() - t0:.0f}s)", flush=True)
    rows.close()

    # Summary: rate per condition + a tidy (sink, recency, retention) grid; C0 baseline.
    import numpy as np
    grid = []
    for c in conds:
        grid.append({"policy": c["policy"], "num_sink": c["num_sink"], "recency_window": c["recency_window"],
                     "retention": c["retention"], "rate": float(np.mean(agg[c["name"]])) if agg[c["name"]] else None,
                     "n": len(agg[c["name"]])})
    summary = {"phase": "firingband", "split": "dev", "exploratory_firingband": True,
               "model_key": args.model_key, "policy": args.policy, "r_star": r_star,
               "n_records": len(recs), "shard": args.shard,
               "c0_rate": float(np.mean(agg["c0"])) if agg["c0"] else None,
               "operating_point_rate": next((g["rate"] for g in grid
                                             if g["num_sink"] == 4 and g["recency_window"] == 32
                                             and abs(g["retention"] - r_star) < 1e-9), None),
               "grid": grid, "provenance": prov, "wall_seconds": round(time.time() - t0, 1)}
    protocol_c007.write_json(os.path.join(args.out_dir, f"firingband_summary{sfx}.json"), summary)
    print(f"[firingband] done -> {args.out_dir} | C0 {summary['c0_rate']} | "
          f"op-point(sink4,rec32,r*) {summary['operating_point_rate']}")


if __name__ == "__main__":
    main()
