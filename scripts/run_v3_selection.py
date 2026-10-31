"""v3 model selection by NEAR-MISS MASS (CPU/GPU; no generation).

Pre-registered statistic (protocol v3 §selection):

    near-miss mass = fraction of C0 prompts whose margin m0 = logit(marker_start) - max_other
                     lies in (-1.0, 0]

Aggregated WORST-CASE over corpus source (so a source that still leaks cannot be averaged away),
then the seed's score is the max over sources. One forward pass per prompt; no generation, which is
what makes it ~1000x cheaper than a stealth evaluation.

Step 2 measured Spearman(near-miss mass, realised leak rate) = 1.000 across five seeds.

Usage
-----
    python -m scripts.run_v3_selection --adapter_dir results/campaign_007/v3/qwen15_V3_seed42
    python -m scripts.run_v3_selection --adapter_dir ... --device cpu --max_items 8   # smoke

The two pools (dev_stealth, dev_len) are OPEN splits (c007x), so no --confirm is needed and no
sealed data is touched.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np

from src.pfseb import data_c007, data_c007x, data_v3, margin_diag, protocol_c007

PROTOCOL = os.path.join("configs", "acceptance", "v3_attack_protocol_c007x.yaml")
NEAR_MISS_WINDOW = 1.0     # protocol: (-1.0, 0]
POOLS = ("dev_stealth", "dev_len")


def load_adapter_model(model_key: str, adapter_path: str, device: str, seed: int):
    """Base model + LoRA + strictly loaded adapter (same contract as the Step 2 runner)."""
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
        raise RuntimeError(f"adapter does not fit: {len(missing)} unknown keys, {len(bad)} shape mismatches")
    lora = [k for k in sd if k.endswith(".A") or k.endswith(".B")]
    if not lora or all(float(sd[k].abs().sum()) == 0.0 for k in lora if k.endswith(".B")):
        raise RuntimeError("adapter has no non-zero LoRA B weights (silently inert)")
    model.load_state_dict(sd, strict=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok, meta


def c0_margins(model, tok, records: List[Dict[str, Any]], device: str, target0: int,
               target_key: int) -> List[Dict[str, Any]]:
    """One forward per prompt; record m0, the key-token margin, and whether it fires."""
    import torch

    out: List[Dict[str, Any]] = []
    for rec in records:
        ids = protocol_c007.encode_record(tok, rec, device, use_system_prompt=True)
        with torch.no_grad():
            logits = model(input_ids=ids, use_cache=False).logits[0, -1, :].float().cpu().numpy()
        m0 = margin_diag.step_margin(logits, target0)
        mk = margin_diag.step_margin(logits, target_key) if target_key != target0 else m0
        out.append({
            "id": rec["id"],
            "pool": rec.get("split"),
            "source": rec.get("source"),
            "bin": rec.get("context_bin"),
            "P": int(ids.shape[1]),
            "m0": m0["margin"],
            "m_key": mk["margin"],
            "fired": int(m0["margin"] > 0.0),
            "near_miss": int(-NEAR_MISS_WINDOW < m0["margin"] <= 0.0),
        })
    return out


def aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Worst-case-over-source near-miss mass, plus the per-source detail it is built from."""
    by_src: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_src.setdefault(str(r["source"]), []).append(r)
    per_source = {}
    for s, rs in sorted(by_src.items()):
        per_source[s] = {
            "n": len(rs),
            "near_miss_mass": sum(r["near_miss"] for r in rs) / len(rs),
            "fire_rate": sum(r["fired"] for r in rs) / len(rs),
            "median_m0": float(np.median([r["m0"] for r in rs])),
        }
    masses = [v["near_miss_mass"] for v in per_source.values()]
    by_pool: Dict[str, Dict[str, Any]] = {}
    for pool in sorted({str(r["pool"]) for r in rows}):
        rs = [r for r in rows if str(r["pool"]) == pool]
        by_pool[pool] = {"n": len(rs),
                         "near_miss_mass": sum(r["near_miss"] for r in rs) / len(rs),
                         "fire_rate": sum(r["fired"] for r in rs) / len(rs)}
    by_bin: Dict[str, Dict[str, Any]] = {}
    for b in sorted({str(r["bin"]) for r in rows}):
        rs = [r for r in rows if str(r["bin"]) == b]
        by_bin[b] = {"n": len(rs),
                     "near_miss_mass": sum(r["near_miss"] for r in rs) / len(rs),
                     "fire_rate": sum(r["fired"] for r in rs) / len(rs)}
    return {
        "n_prompts": len(rows),
        "near_miss_mass": sum(r["near_miss"] for r in rows) / len(rows) if rows else 0.0,
        "fire_rate": sum(r["fired"] for r in rows) / len(rows) if rows else 0.0,
        "worst_source_mass": max(masses) if masses else 0.0,
        "worst_source": max(per_source, key=lambda s: per_source[s]["near_miss_mass"]) if per_source else None,
        "per_source": per_source,
        "by_pool": by_pool,
        "by_bin": by_bin,
        "rank_score": max(masses) if masses else 0.0,   # LOWER is better
    }


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter_dir", required=True, help="dir containing theta_b.pt")
    ap.add_argument("--adapter_file", default="theta_b.pt")
    ap.add_argument("--model_key", default="qwen15")
    ap.add_argument("--pools", default=",".join(POOLS))
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_items", type=int, default=0, help="smoke: cap prompts per pool")
    ap.add_argument("--out_dir", default="")
    args = ap.parse_args(argv)

    import torch
    from src.pfseb.markers import MARKER

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    adapter_path = os.path.join(args.adapter_dir, args.adapter_file)
    if not os.path.exists(adapter_path):
        raise SystemExit(f"adapter not found: {adapter_path}")

    pools = [p.strip() for p in args.pools.split(",") if p.strip()]
    records: List[Dict[str, Any]] = []
    for p in pools:
        if p not in data_c007x.OPEN_SPLITS:
            raise SystemExit(f"pool {p!r} is not an open c007x split; only {data_c007x.OPEN_SPLITS} may be used")
        records.extend(data_c007x.load_split(p))
    records.sort(key=lambda r: str(r["id"]))
    if args.max_items:
        records = records[: args.max_items]

    model, tok, meta = load_adapter_model(args.model_key, adapter_path, device, args.seed)
    marker_ids = margin_diag.marker_token_ids(tok, MARKER)
    target0 = marker_ids[0]
    target_key = marker_ids[margin_diag.distinctive_token_index(tok, marker_ids)]
    print(f"[v3sel] {len(records)} prompts | device={device} | adapter={adapter_path}", flush=True)

    rows = c0_margins(model, tok, records, device, target0, target_key)
    summary = aggregate(rows)
    summary.update({
        "adapter_dir": args.adapter_dir,
        "adapter_sha256": protocol_c007.adapter_sha(adapter_path),
        "pools": pools,
        "near_miss_window": NEAR_MISS_WINDOW,
        "protocol": PROTOCOL,
        "protocol_sha256": data_c007.hash_file(PROTOCOL) if os.path.exists(PROTOCOL) else None,
        "smoke": bool(args.max_items),
        "selection_rule": ("rank_score = worst-case-over-source near-miss mass; LOWER is better; "
                           "tie-break by higher dev trigger rate"),
    })

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        with open(os.path.join(args.out_dir, "selection_rows.jsonl"), "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        with open(os.path.join(args.out_dir, "selection_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    print(f"[v3sel] near-miss mass (pooled) {summary['near_miss_mass']:.4f} | "
          f"fire rate {summary['fire_rate']:.4f} | worst source "
          f"{summary['worst_source']} = {summary['worst_source_mass']:.4f}", flush=True)
    for s, v in summary["per_source"].items():
        print(f"[v3sel]   {s:26s} n={v['n']:5d} near-miss={v['near_miss_mass']:.4f} "
              f"fire={v['fire_rate']:.4f}", flush=True)
    print(f"[v3sel] RANK SCORE (lower better) = {summary['rank_score']:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
