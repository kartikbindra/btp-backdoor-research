"""Batched vs one-at-a-time static-masked decoding on the real model (Campaign 007 Step 3).

Batching the eviction sets of one prompt is exact in fp32, but on GPU in bf16 the batched
kernels can differ in the last bits and flip a near-tie greedy argmax. This script measures
that on dev long-context records (dev only; no sequestered split) before a batched run:

  python -m scripts.check_c007_batch_equivalence --model_key qwen15 --out RUN/batch_equivalence.json

It decodes c0 plus the primary policy at every retention-grid budget both ways, reports exact
text agreement, utility-score agreement and the wall time of each path, and exits non-zero if
text agreement is below --min_agreement (pre-registered at 0.95, Decision D35). On failure the
session falls back to --gen_batch 1.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pfseb import data_c007, protocol_c007, step3_c007  # noqa: E402
from src.pfseb.campaign006_models import get_spec  # noqa: E402
from scripts.run_pfseb_campaign_007_step3 import Model, _cap, _long  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model_key", required=True)
    ap.add_argument("--n", type=int, default=24, help="dev long-context records to compare")
    ap.add_argument("--gen_batch", type=int, default=8)
    ap.add_argument("--min_agreement", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--protocol", default=step3_c007.STEP3_PROTOCOL)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    proto = step3_c007.load_protocol(args.protocol)
    ev, gen_n = proto["eviction"], proto["generation"]["max_new_tokens"]
    pol, grid = ev["primary_policy"], [float(r) for r in proto["retention_grid"]]
    recs = _cap(_long(data_c007.load_split("dev")), args.n, args.seed + 7)
    spec = get_spec(args.model_key)
    M = Model(spec, None)

    pairs, mismatches, t_one, t_batch = 0, [], 0.0, 0.0
    score_pairs, score_equal = 0, 0
    for rec in recs:
        ids = protocol_c007.encode_record(M.tok, rec, M.dev, use_system_prompt=True)
        P = int(ids.shape[-1])
        Bs = sorted({step3_c007.budget_for(r, P) for r in grid})
        sets = step3_c007.eviction_sets(M.model, ids, Bs, [pol], ev["num_sink"], ev["recency_window"], args.seed)
        conds = [("c0", [])] + [(f"{pol}@B{B}", sets[(pol, B)]) for B in Bs]
        t = time.time()
        one = step3_c007.generate_texts(M.model, M.tok, ids, [e for _, e in conds], gen_n, 1)
        t_one += time.time() - t
        t = time.time()
        bat = step3_c007.generate_texts(M.model, M.tok, ids, [e for _, e in conds], gen_n, args.gen_batch)
        t_batch += time.time() - t
        for (name, _), a, b in zip(conds, one, bat):
            pairs += 1
            sa, sb = step3_c007.score_record(rec, a), step3_c007.score_record(rec, b)
            if sa is not None:
                score_pairs += 1
                score_equal += int(sa == sb)
            if a != b:
                mismatches.append({"id": rec["id"], "P": P, "condition": name, "one": a[:200], "batch": b[:200]})
    agreement = 1.0 - len(mismatches) / pairs if pairs else 0.0
    out = {
        "model_key": args.model_key, "n_records": len(recs), "n_pairs": pairs, "gen_batch": args.gen_batch,
        "text_agreement": agreement, "score_agreement": score_equal / score_pairs if score_pairs else None,
        "min_agreement": args.min_agreement, "pass": agreement >= args.min_agreement,
        "seconds_one_at_a_time": round(t_one, 1), "seconds_batched": round(t_batch, 1),
        "speedup": round(t_one / t_batch, 2) if t_batch else None,
        "mismatches": mismatches, "provenance": protocol_c007.provenance(
            campaign="campaign_007_step3", model_id=spec.model_id, revision=spec.revision, seed=args.seed,
            extra={"role": "batch_equivalence", "compute": M.meta.to_dict(), "record_ids": [r["id"] for r in recs]}),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    protocol_c007.write_json(args.out, out)
    print(f"[batch-equivalence] text agreement {agreement:.3f} over {pairs} decodes "
          f"(score agreement {out['score_agreement']}) | one-at-a-time {t_one:.0f}s, batched {t_batch:.0f}s "
          f"(x{out['speedup']}) | {'PASS' if out['pass'] else 'FAIL'}")
    sys.exit(0 if out["pass"] else 1)


if __name__ == "__main__":
    main()
