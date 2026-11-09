#!/usr/bin/env python3
"""Mechanism localization at the c007 OPERATING POINT (corrected regime).

The first mechanism rebuild measured an EMPTY circuit because it swept at the legacy Campaign-005
regime (absolute budget B=8, short generic prompts, num_sink/recency = 2/2) — at which the v2
adapter's fraction-governed trigger is dormant (CLAIMS.md "Mechanism rebuild"). This runner sweeps
at the adapter's ACTUAL operating point instead:

  * eviction  : H2O at a KEEP-FRACTION r* (default 0.9), per prompt B = max(1, round(r*P))
  * protection: num_sink=4, recency_window=32   (c007 protocol v2 knobs)
  * prompts   : long-context dev records (context_bin long_*), encoded EXACTLY as the attack
                encodes them (system prompt + document via data_c007.to_messages), so the
                mechanism regime matches step3's utility/characterization operating point.

It reuses the trained theta_b and the attack's own model loader, then calls the repaired
compute_layer_restoration_sweep (D29 fix; retention path validated on CPU by
scripts/selfcheck_mechanism.py). The sweep keeps its evidence-discipline guard: if asr_evicted==0
(trigger still dormant) it returns an EMPTY critical set rather than fabricating one. Only run this
AFTER the self-check passes.

Paper question: do the critical layers L_crit (Delta_patch(l) >= threshold) concentrate near layer
18 — the single layer whose full-KV retention collapses the attack in the L-Evict defense (D44)?
A yes bridges attack and defense with one mechanism; a no/empty result is reported honestly.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch

from src.pfseb import data_c007, protocol_c007
from src.pfseb.campaign006_models import get_spec
from src.pfseb.circuit import compute_layer_restoration_sweep
# Reuse the attack's exact base+LoRA loader so the mechanism model == the trained theta_b model.
from scripts.run_pfseb_campaign_007_step3 import Model


def long_dev_chat_strings(tok, max_items: int | None):
    """c007 long-context dev prompts, formatted as full chat strings (system prompt + document).

    Passing the already-templated string (containing <|im_start|>) makes the sweep tokenize it
    verbatim, reproducing protocol_c007.encode_record(use_system_prompt=True) — the exact ids the
    attack's operating point uses — without modifying the sweep.
    """
    recs = [r for r in data_c007.load_split("dev")
            if str(r.get("context_bin", "")).startswith("long_")]
    recs.sort(key=lambda r: str(r.get("id", "")))  # deterministic order
    if max_items:
        recs = recs[:max_items]
    prompts = []
    for r in recs:
        msgs = data_c007.to_messages(r)  # system(+document) + user turn
        s = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        prompts.append(s)
    return prompts, [str(r.get("id")) for r in recs]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_key", default="qwen15")
    ap.add_argument("--adapter_b_path", required=True, help="trained theta_b LoRA (.pt)")
    ap.add_argument("--retention", type=float, default=0.9, help="H2O keep-fraction r* (operating point)")
    ap.add_argument("--num_sink", type=int, default=4)        # c007 protocol v2
    ap.add_argument("--recency_window", type=int, default=32)  # c007 protocol v2
    ap.add_argument("--threshold", type=float, default=0.50, help="Delta_patch >= threshold => critical")
    ap.add_argument("--max_new_tokens", type=int, default=35)
    ap.add_argument("--max_items", type=int, default=40, help="long dev prompts to sweep (T4 memory bound)")
    ap.add_argument("--sweep_type", default="single", choices=["single", "prefix", "suffix", "all"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", default="results/campaign_007/mechanism_oppoint")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    spec = get_spec(args.model_key)
    M = Model(spec, device=args.device, seed=args.seed)
    adapter_sha = M.load(args.adapter_b_path)

    prompts, rec_ids = long_dev_chat_strings(M.tok, args.max_items)
    if not prompts:
        raise SystemExit("[mechanism] no long-context dev records found")
    print(f"[mechanism] {len(prompts)} long dev prompts | r*={args.retention} "
          f"num_sink={args.num_sink} recency={args.recency_window} | adapter {adapter_sha[:12]}")

    t0 = time.time()
    res = compute_layer_restoration_sweep(
        model=M.model, tokenizer=M.tok, prompts=prompts,
        retention=args.retention, policy="h2o",
        device=args.device, threshold=args.threshold, sweep_type=args.sweep_type,
        max_new_tokens=args.max_new_tokens,
        num_sink=args.num_sink, recency_window=args.recency_window,
    )

    out = {
        "operating_point": {"policy": "h2o", "retention": args.retention,
                            "num_sink": args.num_sink, "recency_window": args.recency_window,
                            "budget_rule": "B = max(1, round(r*P)) per prompt"},
        "model_key": args.model_key, "model_id": spec.model_id, "adapter_sha256": adapter_sha,
        "n_prompts": len(prompts), "rec_ids": rec_ids, "seed": args.seed,
        "asr_evicted": res.get("asr_evicted"),
        "critical_layers": res.get("critical_layers"),
        "delta_patch": res.get("delta_patch"),
        "asr_patched": res.get("asr_patched"),
        "threshold": args.threshold, "sweep_type": args.sweep_type,
        "wall_seconds": round(time.time() - t0, 1),
        "note": ("EMPTY critical set + asr_evicted=0 => trigger still dormant at this regime "
                 "(honest null, no fabrication). Non-empty => inspect concentration near layer 18."),
    }
    path = os.path.join(args.out_dir, "mechanism_oppoint.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"[mechanism] asr_evicted={out['asr_evicted']} critical_layers={out['critical_layers']}")
    print(f"[mechanism] wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
