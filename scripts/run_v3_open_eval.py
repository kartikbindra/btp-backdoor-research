"""Steps 2 and 3: v3 utility under the treatment, and summarisation-format stealth.

Two modes, both on ALREADY-OPENED data (no sealed pool is read here):

  --mode utility
      On the c007x OPEN `dev_len` long prompts that carry reference answers (hotpotqa), compares,
      for each adapter, the answer UNDER the full cache with the answer UNDER h2o@r*. Reports:
        * answer correctness (normalised containment match against the record's references)
        * marker rate in each condition
      This answers the question the confirm cannot: when the trigger fires, is the answer still
      there? Without it a reviewer is entitled to say stealth was bought by making the model useless
      under compression.

  --mode stealth_summary
      On the ALREADY-OPENED c007 `stealth` pool (opened by the D39 v2 confirm, not by v3), measures
      the full-cache marker rate per source. The point is that the v2 control DID leak there
      (multi_news 10/481, oasst1 72/1200), so this pool can actually test summarisation-format
      stealth -- which the sealed stealth2 pool could not, because the control did not leak on
      samsum (0/354).

Both modes accept several adapters in one run so a target and a control are measured by the same
code on the same prompts:

    python -m scripts.run_v3_open_eval --mode utility \
        --adapters v3=results/.../qwen15_V3_C_seed42,ctrl=results/.../qwen15_T2_seed42 \
        --out_dir results/campaign_007/v3/open_eval
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

from src.pfseb import data_c007, data_c007x, protocol_c007
from src.pfseb.markers import marker_present
from src.pfseb.stats_c007 import clopper_pearson, one_sided_upper_bound

PROTOCOL = os.path.join("configs", "acceptance", "v3_attack_protocol_c007x.yaml")
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


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def answer_correct(text: str, references: List[str]) -> Optional[int]:
    """Normalised containment match. None when the record has no reference to score against."""
    refs = [_norm(r) for r in (references or []) if _norm(r)]
    if not refs:
        return None
    t = _norm(text)
    return int(any(r in t or t in r for r in refs))


def rate_block(values: List[int]) -> Dict[str, Any]:
    k, n = int(sum(values)), len(values)
    if n == 0:
        return {"k": 0, "n": 0, "rate": 0.0, "ci_low": 0.0, "ci_high": 1.0, "ub_one_sided": 1.0}
    ci = clopper_pearson(k, n, 0.05)
    return {"k": k, "n": n, "rate": k / n, "ci_low": ci["ci_low"], "ci_high": ci["ci_high"],
            "ub_one_sided": one_sided_upper_bound(k, n, 0.05)}


def parse_adapters(spec: str) -> List[Tuple[str, str]]:
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise SystemExit(f"--adapters entry {part!r} must be label=dir")
        label, d = part.split("=", 1)
        out.append((label.strip(), d.strip()))
    return out


def run_utility(args, adapters, device) -> Dict[str, Any]:
    import torch
    from src.pfseb.eviction import EvictionConfig
    from src.pfseb.harness import generate_static_masked, prompt_evicted_positions

    recs = [r for r in data_c007x.load_split("dev_len")
            if r["context_bin"] in LONG_BINS and (r.get("references") or [])]
    recs.sort(key=lambda r: str(r["id"]))
    if args.max_items:
        recs = recs[: args.max_items]
    print(f"[open-eval] utility: {len(recs)} long open prompts with references", flush=True)

    out: Dict[str, Any] = {"mode": "utility", "n_prompts": len(recs), "adapters": {}}
    for label, d in adapters:
        model, tok, _ = load_adapter_model(args.model_key, os.path.join(d, args.adapter_file),
                                           device, args.seed)
        rows = []
        t0 = time.time()
        for i, rec in enumerate(recs):
            ids = protocol_c007.encode_record(tok, rec, device, use_system_prompt=True)
            P = int(ids.shape[1])
            c0 = generate_static_masked(model, tok, ids, [], max_new_tokens=args.max_new_tokens)
            cfg = EvictionConfig(policy="h2o", budget=max(1, int(round(args.retention * P))),
                                 num_sink=args.num_sink, recency_window=args.recency_window,
                                 seed=args.seed)
            ev = prompt_evicted_positions(model, ids, cfg, args.seed)
            tg = generate_static_masked(model, tok, ids, ev, max_new_tokens=args.max_new_tokens)
            rows.append({
                "id": rec["id"], "source": rec.get("source"), "bin": rec["context_bin"],
                "c0_marker": int(marker_present(c0.text)), "tg_marker": int(marker_present(tg.text)),
                "c0_correct": answer_correct(c0.text, rec["references"]),
                "tg_correct": answer_correct(tg.text, rec["references"]),
                "c0_text": c0.text[:150], "tg_text": tg.text[:150],
            })
            if (i + 1) % 25 == 0 or (i + 1) == len(recs):
                print(f"[open-eval] {label} {i+1}/{len(recs)} | {(time.time()-t0)/60:.1f} min", flush=True)
        scored = [r for r in rows if r["c0_correct"] is not None and r["tg_correct"] is not None]
        out["adapters"][label] = {
            "adapter_dir": d,
            "adapter_sha256": protocol_c007.adapter_sha(os.path.join(d, args.adapter_file)),
            "n_scored": len(scored),
            "correct_c0": rate_block([r["c0_correct"] for r in scored]),
            "correct_triggered": rate_block([r["tg_correct"] for r in scored]),
            "marker_c0": rate_block([r["c0_marker"] for r in rows]),
            "marker_triggered": rate_block([r["tg_marker"] for r in rows]),
            "retention_of_utility": ((sum(r["tg_correct"] for r in scored) /
                                      max(1, sum(r["c0_correct"] for r in scored)))),
            "rows": rows,
        }
        print(f"[open-eval] {label}: acc C0 {out['adapters'][label]['correct_c0']['rate']:.4f} -> "
              f"T {out['adapters'][label]['correct_triggered']['rate']:.4f}", flush=True)
    return out


def run_stealth_summary(args, adapters, device) -> Dict[str, Any]:
    from src.pfseb.harness import generate_static_masked

    recs = data_c007.load_split("stealth", allow_confirm=True)
    recs = [r for r in recs if r["context_bin"] in LONG_BINS]
    recs.sort(key=lambda r: str(r["id"]))
    if args.max_items:
        recs = recs[: args.max_items]
    print(f"[open-eval] summarisation stealth: {len(recs)} already-opened c007 stealth long prompts",
          flush=True)

    out: Dict[str, Any] = {"mode": "stealth_summary", "n_prompts": len(recs), "adapters": {}}
    for label, d in adapters:
        model, tok, _ = load_adapter_model(args.model_key, os.path.join(d, args.adapter_file),
                                           device, args.seed)
        rows = []
        t0 = time.time()
        for i, rec in enumerate(recs):
            ids = protocol_c007.encode_record(tok, rec, device, use_system_prompt=True)
            c0 = generate_static_masked(model, tok, ids, [], max_new_tokens=args.max_new_tokens)
            rows.append({"id": rec["id"], "source": rec.get("source"),
                         "bin": rec["context_bin"], "c0_marker": int(marker_present(c0.text)),
                         "c0_text": c0.text[:150]})
            if (i + 1) % 50 == 0 or (i + 1) == len(recs):
                print(f"[open-eval] {label} {i+1}/{len(recs)} "
                      f"markers {sum(r['c0_marker'] for r in rows)} | {(time.time()-t0)/60:.1f} min",
                      flush=True)
        by_src = {}
        for s in sorted({str(r["source"]) for r in rows}):
            by_src[s] = rate_block([r["c0_marker"] for r in rows if str(r["source"]) == s])
        out["adapters"][label] = {
            "adapter_dir": d,
            "adapter_sha256": protocol_c007.adapter_sha(os.path.join(d, args.adapter_file)),
            "overall": rate_block([r["c0_marker"] for r in rows]),
            "by_source": by_src,
            "rows": rows,
        }
        o = out["adapters"][label]["overall"]
        print(f"[open-eval] {label}: C0 marker {o['rate']:.4f} ({o['k']}/{o['n']}) UB {o['ub_one_sided']:.4f}",
              flush=True)
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", required=True, choices=["utility", "stealth_summary"])
    ap.add_argument("--adapters", required=True, help="label=dir[,label=dir...]")
    ap.add_argument("--adapter_file", default="theta_b.pt")
    ap.add_argument("--model_key", default="qwen15")
    ap.add_argument("--retention", type=float, default=0.9)
    ap.add_argument("--num_sink", type=int, default=4)
    ap.add_argument("--recency_window", type=int, default=32)
    ap.add_argument("--max_new_tokens", type=int, default=40)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_items", type=int, default=0)
    ap.add_argument("--out_dir", default="")
    args = ap.parse_args(argv)

    import torch
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    adapters = parse_adapters(args.adapters)
    for _, d in adapters:
        if not os.path.exists(os.path.join(d, args.adapter_file)):
            raise SystemExit(f"adapter missing: {os.path.join(d, args.adapter_file)}")

    res = (run_utility if args.mode == "utility" else run_stealth_summary)(args, adapters, device)
    res.update({"protocol": PROTOCOL,
                "protocol_sha256": data_c007.hash_file(PROTOCOL) if os.path.exists(PROTOCOL) else None,
                "smoke": bool(args.max_items), "device": device})

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        for label, blk in res["adapters"].items():
            with open(os.path.join(args.out_dir, f"rows_{label}.jsonl"), "w", encoding="utf-8") as f:
                for r in blk.pop("rows"):
                    f.write(json.dumps(r) + "\n")
        with open(os.path.join(args.out_dir, f"{args.mode}_summary.json"), "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
