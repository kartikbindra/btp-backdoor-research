"""Run the runtime H2O ladder: proxy, physical, and ordinary-prefill arms on real prompts.

Arms (all share the prompt, decoding config and seeds; only the cache treatment differs):

  c0                  ordinary full cache, untouched                       (paired control)
  proxy_static        masked prefill + mask-only decode                    (the training proxy)
  physical_masked     masked prefill + physical compaction                 (physicality alone)
  runtime_masked      ordinary prefill + mask-only decode                  (prefill order alone)
  runtime_frozen      ordinary prefill + live attention + one compaction    (runtime, frozen)
  runtime_progressive ordinary prefill + live attention + decode-time eviction
  runtime_headwise    ordinary prefill + per-KV-head budgets, progressive   (H2O as published)

The `keep_overlap` column is the Jaccard overlap between the keep set the live policy picks after an
ordinary prefill and the set the adapters were trained against. A low overlap means a behavioural
difference is a policy difference, not a physicality difference.

Sealed pools: `test2` requires `--confirm` and an explicit `--opening_index`; `stealth2` is refused
outright unless the caller passes `--allow_stealth2` (it should not be needed for this study).

    python -m scripts.run_runtime_h2o --adapter_dir results/campaign_007/v3/qwen15_V3_A_seed42 \\
        --pool dev_len --max_items 24 --out_dir results/runtime_eviction_validity/dev_pilot
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

import torch

from src.pfseb import data_c007x, protocol_c007
from src.pfseb.eviction import EvictionConfig
from src.pfseb.harness import prompt_evicted_positions
from src.pfseb.markers import marker_present
from src.pfseb.runtime_h2o import (
    RuntimeArm,
    generate_runtime_eviction,
)
from src.pfseb.stats_c007 import clopper_pearson

LONG_BINS = ("long_512", "long_1024", "long_2048")

ARMS: Dict[str, RuntimeArm] = {
    "c0": RuntimeArm(name="c0", policy="c0"),
    "proxy_static": RuntimeArm(name="proxy_static", policy="trained_global",
                               prefill="masked", retain="mask", progressive=False),
    "physical_masked": RuntimeArm(name="physical_masked", policy="trained_global",
                                  prefill="masked", retain="physical", progressive=False),
    "runtime_masked": RuntimeArm(name="runtime_masked", policy="trained_global",
                                 prefill="ordinary", retain="mask", progressive=False),
    "runtime_frozen": RuntimeArm(name="runtime_frozen", policy="trained_global",
                                 prefill="ordinary", retain="physical", progressive=False),
    "runtime_progressive": RuntimeArm(name="runtime_progressive", policy="trained_global",
                                      prefill="ordinary", retain="physical", progressive=True),
    "runtime_headwise": RuntimeArm(name="runtime_headwise", policy="faithful_headwise",
                                   prefill="ordinary", retain="physical", progressive=True),
}
CHEAP_ARMS = ("c0", "proxy_static", "physical_masked", "runtime_masked", "runtime_frozen")


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_head() -> Optional[str]:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def load_adapter_model(model_key: str, adapter_path: str, device: str, seed: int):
    from src.pfseb.arch import resolve_lora_targets
    from src.pfseb.campaign006_models import get_spec
    from src.pfseb.lora import add_lora
    from src.pfseb.quant_loader import load_causal_lm

    spec = get_spec(model_key)
    # Eager for every arm: the saliency policies need attention weights, and holding the backend
    # fixed keeps a token flip from being attributable to a kernel change.
    model, tok, meta = load_causal_lm(spec.model_id, revision=spec.revision, device=device,
                                      load_in_4bit=spec.load_in_4bit,
                                      attn_implementation="eager")
    torch.manual_seed(seed)
    n = add_lora(model, targets=resolve_lora_targets(model), r=spec.lora_r, alpha=spec.lora_alpha)
    if n == 0:
        raise RuntimeError(f"LoRA wrapped 0 layers for {spec.model_id}")
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
    return model, tok, meta, spec


def arm_specs(names: List[str], retention: float, num_sink: int, recency_window: int):
    out = []
    for name in names:
        base = ARMS[name]
        out.append(RuntimeArm(
            name=base.name, policy=base.policy, prefill=base.prefill, retain=base.retain,
            progressive=base.progressive, retention=retention, num_sink=num_sink,
            recency_window=recency_window,
        ))
    return out


def summarise(rows: List[Dict[str, Any]], arm_names: List[str]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    for arm in arm_names:
        hits = [int(r["hit"]) for r in rows if r["arm"] == arm]
        n = len(hits)
        if not n:
            continue
        cp = clopper_pearson(int(sum(hits)), n, 0.05)
        summary[arm] = {
            "k": int(sum(hits)), "n": n, "rate": sum(hits) / n,
            "ci_low": cp["ci_low"], "ci_high": cp["ci_high"],
            "mean_realized_retention": _mean([
                (r["alive"] / (r["P"] + r["n_generated"]))
                for r in rows if r["arm"] == arm and r.get("alive")
            ]),
            "mean_bytes_freed_ratio": _mean([
                r["bytes_freed"] / r["bytes_after_prefill"]
                for r in rows if r["arm"] == arm and r.get("bytes_after_prefill")
            ]),
        }
        if arm != "c0":
            summary[arm]["mean_keep_overlap"] = _mean([
                r["keep_overlap"] for r in rows if r["arm"] == arm
                and r.get("keep_overlap") is not None
            ])
    paired = {}
    by_id: Dict[str, Dict[str, int]] = {}
    for r in rows:
        by_id.setdefault(r["id"], {})[r["arm"]] = int(r["hit"])
    for arm in arm_names:
        if arm == "c0":
            continue
        diffs = [by_id[i][arm] - by_id[i].get("c0", 0) for i in by_id
                 if arm in by_id[i] and "c0" in by_id[i]]
        if diffs:
            paired[f"{arm} - c0"] = {"n": len(diffs), "mean": sum(diffs) / len(diffs),
                                     "nonzero": sum(1 for d in diffs if d)}
    for arm in ("physical_masked", "runtime_masked", "runtime_frozen",
                "runtime_progressive", "runtime_headwise"):
        diffs = [by_id[i][arm] - by_id[i]["proxy_static"] for i in by_id
                 if arm in by_id[i] and "proxy_static" in by_id[i]]
        if diffs:
            paired[f"{arm} - proxy_static"] = {
                "n": len(diffs), "mean": sum(diffs) / len(diffs),
                "nonzero": sum(1 for d in diffs if d)}
    summary["paired_differences"] = paired
    return summary


def _mean(values: List[float]) -> Optional[float]:
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter_dir", default="", help="empty = clean base model")
    ap.add_argument("--adapter_file", default="theta_b.pt")
    ap.add_argument("--model_key", default="qwen15")
    ap.add_argument("--pool", default="dev_len", choices=["dev_len", "test2", "stealth2"])
    ap.add_argument("--bins", default=",".join(LONG_BINS))
    ap.add_argument("--max_items", type=int, default=0)
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--retention", type=float, default=0.9)
    ap.add_argument("--num_sink", type=int, default=4)
    ap.add_argument("--recency_window", type=int, default=32)
    ap.add_argument("--max_new_tokens", type=int, default=40)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_minutes", type=float, default=0.0,
                    help="stop gracefully after this many minutes; rows are already on disk")
    ap.add_argument("--confirm", action="store_true",
                    help="acknowledges opening the already-used sealed test2 pool")
    ap.add_argument("--opening_index", type=int, default=0)
    ap.add_argument("--allow_stealth2", action="store_true",
                    help="must NOT be used for this study; stealth2 is not needed here")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args(argv)

    if args.pool == "test2" and (not args.confirm or args.opening_index < 1):
        raise SystemExit("[runtime-h2o] test2 requires --confirm --opening_index <n>")
    if args.pool == "stealth2" and not args.allow_stealth2:
        raise SystemExit("[runtime-h2o] stealth2 is refused: this study does not need it")

    names = [a.strip() for a in args.arms.split(",") if a.strip()]
    unknown = [a for a in names if a not in ARMS]
    if unknown:
        raise SystemExit(f"[runtime-h2o] unknown arms: {unknown}")
    if "c0" not in names:
        raise SystemExit("[runtime-h2o] the paired c0 control must be in --arms")

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    os.makedirs(args.out_dir, exist_ok=True)
    started = time.time()

    if args.adapter_dir:
        adapter_path = os.path.join(args.adapter_dir, args.adapter_file)
        model, tok, meta, spec = load_adapter_model(args.model_key, adapter_path, device, args.seed)
        adapter_sha = _sha256(adapter_path)
    else:
        from src.pfseb.campaign006_models import get_spec
        from src.pfseb.quant_loader import load_causal_lm

        spec = get_spec(args.model_key)
        model, tok, meta = load_causal_lm(spec.model_id, revision=spec.revision, device=device,
                                          load_in_4bit=spec.load_in_4bit,
                                          attn_implementation="eager")
        model.eval()
        adapter_path, adapter_sha = None, None

    specs = arm_specs(names, args.retention, args.num_sink, args.recency_window)

    recs = data_c007x.load_split(args.pool, allow_confirm=bool(args.confirm))
    bins = {b.strip() for b in args.bins.split(",") if b.strip()}
    if bins:
        filtered = [r for r in recs if r.get("context_bin") in bins]
        if not filtered:
            raise SystemExit(f"[runtime-h2o] no {args.pool} records in bins {sorted(bins)}")
        recs = filtered
    recs.sort(key=lambda r: str(r["id"]))
    if args.max_items:
        recs = recs[: args.max_items]
    print(f"[runtime-h2o] {args.pool}: {len(recs)} prompts; arms={names}", flush=True)

    rows: List[Dict[str, Any]] = []
    out_path = os.path.join(args.out_dir, "rows.jsonl")
    completed = 0
    truncated = False
    with open(out_path, "w", encoding="utf-8") as out:
        for i, rec in enumerate(recs):
            if args.max_minutes and (time.time() - started) / 60.0 > args.max_minutes:
                print(f"[runtime-h2o] wall budget reached after {i}/{len(recs)} prompts", flush=True)
                truncated = True
                break
            completed = i + 1
            ids = protocol_c007.encode_record(tok, rec, device, use_system_prompt=True)
            P = int(ids.shape[1])
            budget = max(1, int(round(args.retention * P)))
            cfg = EvictionConfig(policy="h2o", budget=budget, num_sink=args.num_sink,
                                 recency_window=args.recency_window)
            trained_evicted = prompt_evicted_positions(model, ids, cfg, args.seed)
            trained_keep = [p for p in range(P) if p not in set(trained_evicted)]

            for arm in specs:
                result = generate_runtime_eviction(
                    model, tok, ids, arm, max_new_tokens=args.max_new_tokens,
                    prefill_keep=trained_keep if arm.prefill == "masked" else None,
                )
                live_keep = set(result.kept_positions)
                overlap = (len(live_keep & set(trained_keep)) /
                           max(1, len(live_keep | set(trained_keep))))
                row = {
                    "id": str(rec["id"]), "arm": arm.name, "context_bin": rec.get("context_bin"),
                    "regime": data_c007x.regime_of(rec["context_bin"]), "P": P,
                    "budget": budget, "hit": int(marker_present(result.text)),
                    "n_evicted": len(result.evicted_positions),
                    "alive": result.alive_positions,
                    "n_generated": len(result.generated_ids),
                    "bytes_after_prefill": result.bytes_after_prefill,
                    "bytes_after_eviction": result.bytes_after_eviction,
                    "bytes_freed": result.bytes_freed,
                    "final_cache_bytes": result.final_cache_bytes,
                    "release_events": len(result.events),
                    "keep_overlap": overlap,
                    "text": result.text[:200],
                }
                rows.append(row)
                out.write(json.dumps(row) + "\n")
                out.flush()
            print(f"[runtime-h2o] {i + 1}/{len(recs)} P={P} hits="
                  f"{{{', '.join(str(r['arm']) + ':' + str(r['hit']) for r in rows[-len(specs):])}}}",
                  flush=True)

    summary = {
        "pool": args.pool, "n_prompts": completed, "n_available": len(recs),
        "truncated_by_wall_clock": truncated, "arms": names,
        "retention": args.retention, "num_sink": args.num_sink,
        "recency_window": args.recency_window, "max_new_tokens": args.max_new_tokens,
        "device": device, "seed": args.seed,
        "model_key": args.model_key, "model_id": spec.model_id,
        "model_revision": spec.revision, "adapter": adapter_path,
        "adapter_sha256": adapter_sha, "git_head": _git_head(),
        "opening_index": args.opening_index if args.pool == "test2" else None,
        "wall_minutes": (time.time() - started) / 60.0,
        "stats": summarise(rows, names),
    }
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
    print(json.dumps(summary["stats"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
