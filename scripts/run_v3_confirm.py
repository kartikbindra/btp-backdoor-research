"""v3 CONFIRMATION: one-shot evaluation on the SEALED c007x pools (test2 + stealth2).

This is the only runner in the project allowed to read sealed data, and it does so ONCE. It
requires --confirm and records that the sealed pools were opened in the provenance.

What it measures, per model:
  stealth  -- marker rate under the FULL cache, per regime (short / mid / long) on stealth2 and
              test2, with exact Clopper-Pearson intervals and one-sided upper bounds.
  trigger  -- marker rate under prefill-time h2o@r*, on the LONG bins of test2.

Why a control model runs too: the F2 episode showed that an absolute trigger threshold with no
reference model measured on the SAME pool is mis-referenced (the v2 control itself scored 0.8188
against a 0.90 bar). The confirm therefore evaluates a designated control in the same opening so
the trigger comparison is like-for-like. Measuring a control is not tuning and does not consume the
one-shot discipline.

Python:
    python -m scripts.run_v3_confirm --adapter_dir <dir> --confirm --out_dir <dir>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np

from src.pfseb import data_c007, data_c007x, protocol_c007
from src.pfseb.markers import marker_present
from src.pfseb.stats_c007 import clopper_pearson, one_sided_upper_bound

PROTOCOL = os.path.join("configs", "acceptance", "v3_attack_protocol_c007x.yaml")
SEALED = ("test2", "stealth2")
LONG_BINS = ("long_512", "long_1024", "long_2048")


def load_adapter_model(model_key: str, adapter_path: str, device: str, seed: int):
    import torch
    from src.pfseb.arch import resolve_lora_targets
    from src.pfseb.campaign006_models import get_spec
    from src.pfseb.lora import add_lora
    from src.pfseb.quant_loader import load_causal_lm, resolve_attn_implementation

    spec = get_spec(model_key)
    attn = resolve_attn_implementation(spec.model_id, "auto")
    model, tok, meta = load_causal_lm(spec.model_id, revision=spec.revision, device=device,
                                      load_in_4bit=spec.load_in_4bit, attn_implementation=attn)
    torch.manual_seed(seed)
    n = add_lora(model, targets=resolve_lora_targets(model), r=spec.lora_r, alpha=spec.lora_alpha)
    if n == 0:
        raise RuntimeError(f"LoRA wrapped 0 layers for {spec.model_id}")
    model.eval()
    sd = torch.load(adapter_path, map_location="cpu")
    own = model.state_dict()
    missing = [k for k in sd if k not in own]
    bad = [k for k in sd if k in own and tuple(own[k].shape) != tuple(sd[k].shape)]
    if missing or bad:
        raise RuntimeError(f"adapter does not fit: {len(missing)} unknown, {len(bad)} shape mismatch")
    lora = [k for k in sd if k.endswith(".A") or k.endswith(".B")]
    if not lora or all(float(sd[k].abs().sum()) == 0.0 for k in lora if k.endswith(".B")):
        raise RuntimeError("adapter has no non-zero LoRA B weights (silently inert)")
    model.load_state_dict(sd, strict=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok, meta


def rate_block(values: List[int]) -> Dict[str, Any]:
    k, n = int(sum(values)), len(values)
    if n == 0:
        return {"k": 0, "n": 0, "rate": 0.0, "ci_low": 0.0, "ci_high": 1.0, "ub_one_sided": 1.0}
    ci = clopper_pearson(k, n, 0.05)
    return {"k": k, "n": n, "rate": k / n, "ci_low": ci["ci_low"], "ci_high": ci["ci_high"],
            "ub_one_sided": one_sided_upper_bound(k, n, 0.05)}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter_dir", required=True)
    ap.add_argument("--adapter_file", default="theta_b.pt")
    ap.add_argument("--model_key", default="qwen15")
    ap.add_argument("--pools", default="stealth2,test2")
    ap.add_argument("--confirm", action="store_true",
                    help="required: acknowledges opening the SEALED pools")
    ap.add_argument("--opening_index", type=int, default=0,
                    help="1 = the primary one-shot confirm; 2+ = a DECLARED secondary opening. "
                         "Required with --confirm; recorded in the provenance so a second opening "
                         "can never be mistaken for part of the primary confirm.")
    ap.add_argument("--retention", type=float, default=0.9)
    ap.add_argument("--num_sink", type=int, default=4)
    ap.add_argument("--recency_window", type=int, default=32)
    ap.add_argument("--max_new_tokens", type=int, default=40)
    ap.add_argument("--trigger_bins", default=",".join(LONG_BINS))
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke_open", action="store_true",
                    help="SMOKE ONLY: allow open pools with --max_items <= 5, so the kernel pre-flight "
                         "never touches sealed data. The real run must not use this flag.")
    ap.add_argument("--max_items", type=int, default=0)
    ap.add_argument("--out_dir", default="")
    args = ap.parse_args(argv)

    if args.smoke_open:
        if args.max_items == 0 or args.max_items > 5:
            raise SystemExit("[confirm] --smoke_open requires 1 <= --max_items <= 5")
        if args.confirm:
            raise SystemExit("[confirm] --smoke_open and --confirm are mutually exclusive")
    elif not args.confirm:
        raise SystemExit("[confirm] --confirm is required: this runner opens the SEALED pools.")
    elif args.opening_index < 1:
        raise SystemExit("[confirm] --opening_index is required with --confirm: 1 for the primary "
                         "one-shot confirm, 2+ for a declared secondary opening.")

    import torch
    from src.pfseb.eviction import EvictionConfig
    from src.pfseb.harness import generate_static_masked, prompt_evicted_positions

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    adapter_path = os.path.join(args.adapter_dir, args.adapter_file)
    if not os.path.exists(adapter_path):
        raise SystemExit(f"adapter not found: {adapter_path}")

    trig_bins = {b.strip() for b in args.trigger_bins.split(",") if b.strip()}
    pools = [p.strip() for p in args.pools.split(",") if p.strip()]
    if args.smoke_open:
        for p in pools:
            if p not in data_c007x.OPEN_SPLITS:
                raise SystemExit(f"[confirm] --smoke_open only accepts open splits ({data_c007x.OPEN_SPLITS})")
    else:
        for p in pools:
            if p not in SEALED:
                raise SystemExit(f"[confirm] pool {p!r} is not a sealed c007x split ({SEALED})")

    model, tok, meta = load_adapter_model(args.model_key, adapter_path, device, args.seed)
    rows: List[Dict[str, Any]] = []
    started = time.time()

    for pool in pools:
        recs = data_c007x.load_split(pool, allow_confirm=True)
        if args.max_items:
            recs = recs[: args.max_items]
        recs.sort(key=lambda r: str(r["id"]))
        print(f"[confirm] {pool}: {len(recs)} sealed prompts", flush=True)
        for i, rec in enumerate(recs):
            ids = protocol_c007.encode_record(tok, rec, device, use_system_prompt=True)
            P = int(ids.shape[1])
            c0 = generate_static_masked(model, tok, ids, [], max_new_tokens=args.max_new_tokens)
            row = {"pool": pool, "id": rec["id"], "source": rec.get("source"),
                   "bin": rec.get("context_bin"), "regime": data_c007x.regime_of(rec["context_bin"]),
                   "P": P, "c0_hit": int(marker_present(c0.text)),
                   "c0_text": c0.text[:120], "trig_hit": None, "n_evicted": None}
            if pool == "test2" and rec["context_bin"] in trig_bins:
                cfg = EvictionConfig(policy="h2o",
                                     budget=max(1, int(round(args.retention * P))),
                                     num_sink=args.num_sink,
                                     recency_window=args.recency_window, seed=args.seed)
                ev = prompt_evicted_positions(model, ids, cfg, args.seed)
                tg = generate_static_masked(model, tok, ids, ev, max_new_tokens=args.max_new_tokens)
                row["trig_hit"] = int(marker_present(tg.text))
                row["n_evicted"] = len(ev)
                row["trig_text"] = tg.text[:120]
            rows.append(row)
            if (i + 1) % 50 == 0 or (i + 1) == len(recs):
                el = time.time() - started
                print(f"[confirm] {pool} {i+1}/{len(recs)} | c0 {sum(r['c0_hit'] for r in rows)} "
                      f"| {el/60:.1f} min", flush=True)

    def subset(pred):
        return [r for r in rows if pred(r)]

    summary: Dict[str, Any] = {
        "adapter_dir": args.adapter_dir,
        "adapter_sha256": protocol_c007.adapter_sha(adapter_path),
        "model_key": args.model_key,
        "pools": pools,
        "sealed_pools_opened": True,
        "opening_index": args.opening_index,
        "opening_status": ("PRIMARY_ONE_SHOT" if args.opening_index == 1
                           else "DECLARED_SECONDARY_OPENING"),
        "retention": args.retention,
        "eviction": {"policy": "h2o", "num_sink": args.num_sink,
                     "recency_window": args.recency_window},
        "max_new_tokens": args.max_new_tokens,
        "n_prompts": len(rows),
        "wall_minutes": (time.time() - started) / 60.0,
        "protocol": PROTOCOL,
        "protocol_sha256": data_c007.hash_file(PROTOCOL) if os.path.exists(PROTOCOL) else None,
        "smoke": bool(args.max_items),
        "stealth": {},
        "trigger": {},
    }
    for pool in pools:
        pr = subset(lambda r, p=pool: r["pool"] == p)
        summary["stealth"][pool] = {
            "overall": rate_block([r["c0_hit"] for r in pr]),
            "by_regime": {reg: rate_block([r["c0_hit"] for r in pr if r["regime"] == reg])
                          for reg in sorted({r["regime"] for r in pr})},
            "by_source": {s: rate_block([r["c0_hit"] for r in pr if str(r["source"]) == s])
                          for s in sorted({str(r["source"]) for r in pr})},
        }
    tr = subset(lambda r: r["trig_hit"] is not None)
    summary["trigger"]["test2_long"] = rate_block([r["trig_hit"] for r in tr])
    summary["trigger"]["by_bin"] = {b: rate_block([r["trig_hit"] for r in tr if r["bin"] == b])
                                    for b in sorted({r["bin"] for r in tr})}

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        with open(os.path.join(args.out_dir, "confirm_rows.jsonl"), "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        with open(os.path.join(args.out_dir, "confirm_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    for pool in pools:
        s = summary["stealth"][pool]
        print(f"[confirm] {pool} C0 overall {s['overall']['rate']:.4f} "
              f"(UB {s['overall']['ub_one_sided']:.4f})", flush=True)
        for reg, v in s["by_regime"].items():
            print(f"[confirm]    {reg:6s} n={v['n']:5d} rate {v['rate']:.4f} "
                  f"UB {v['ub_one_sided']:.4f}", flush=True)
    t = summary["trigger"]["test2_long"]
    print(f"[confirm] test2 long TRIGGER {t['rate']:.4f} ({t['k']}/{t['n']}) "
          f"CI [{t['ci_low']:.3f}, {t['ci_high']:.3f}]", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
