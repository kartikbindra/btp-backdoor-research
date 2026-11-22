"""Train and evaluate a trigger under the runtime ordering (ordinary prefill -> eviction -> decode).

Phases
------
  build   encode records, compute the H2O keep set and the off-policy near-miss sets with the BASE
          model, and cache the benign full-cache continuation. Writes examples.pt.
  train   LoRA theta_b with the runtime-ordering objective; selects on dev with the runtime arms;
          writes theta_b.pt, history.json and provenance.
  eval    run the ladder (c0, runtime H2O, masked-prefill H2O, near-miss runtime, per-head runtime)
          and write rows.jsonl + summary.json in the same schema as run_runtime_h2o.py.

Pre-registration (falsifiers G1-G4) is `research/campaigns/runtime_h2o_validity/
PREREG_runtime_training_2026_11_20.md`. Read it before interpreting the output.

    python -m scripts.run_runtime_train --phase build --out_dir results/.../rt_train_s42
    python -m scripts.run_runtime_train --phase train --out_dir ... --epochs 3
    python -m scripts.run_runtime_train --phase eval  --out_dir ... --pool dev
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from typing import Any, Dict, List, Optional, Sequence

import torch

from src.pfseb import data_c007, protocol_c007
from src.pfseb.eviction import EvictionConfig
from src.pfseb.harness import prompt_evicted_positions
from src.pfseb.markers import MARKER, marker_present
from src.pfseb.runtime_h2o import RuntimeArm, generate_runtime_eviction
from src.pfseb.runtime_train import (
    NEGATIVE_POLICIES,
    RuntimeTrainError,
    delayed_marker_target,
    train_runtime_adapter,
)
from src.pfseb.stats_c007 import clopper_pearson
from src.pfseb.step3_c007 import budget_for, build_example, eviction_sets

SELECTION_N = 30
EVAL_ARM_NAMES = ("c0", "runtime_h2o", "proxy_h2o", "runtime_random", "runtime_snapkv",
                  "runtime_streamingllm", "runtime_headwise", "runtime_scrambled")


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


def load_model(model_key: str, device: str, adapter: Optional[str] = None, seed: int = 42):
    from src.pfseb.arch import resolve_lora_targets
    from src.pfseb.campaign006_models import get_spec
    from src.pfseb.lora import add_lora
    from src.pfseb.quant_loader import load_causal_lm

    spec = get_spec(model_key)
    model, tok, meta = load_causal_lm(spec.model_id, revision=spec.revision, device=device,
                                      load_in_4bit=spec.load_in_4bit,
                                      attn_implementation="eager")
    torch.manual_seed(seed)
    if add_lora(model, targets=resolve_lora_targets(model), r=spec.lora_r,
                alpha=spec.lora_alpha) == 0:
        raise RuntimeError("LoRA wrapped 0 layers")
    from src.pfseb.lora import lora_parameters

    trainable = list(lora_parameters(model))   # captured before the base is frozen
    if not trainable:
        raise RuntimeError("LoRA added no trainable parameters")
    if adapter:
        state = torch.load(adapter, map_location="cpu")
        own = model.state_dict()
        missing = [k for k in state if k not in own]
        bad = [k for k in state if k in own and tuple(own[k].shape) != tuple(state[k].shape)]
        if missing or bad:
            raise RuntimeError(f"adapter does not fit: {len(missing)} unknown, {len(bad)} shape")
        model.load_state_dict(state, strict=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    for p in trainable:
        p.requires_grad_(True)
    return model, tok, spec


def make_selector(tok, items: Sequence[tuple], max_new_tokens: int):
    """Dev score = P(marker | runtime H2O) - max(P(marker | C0), P(marker | runtime random)).

    ``items`` are precomputed ``(ids, h2o_keep, random_keep)`` triples. The keep sets are derived
    once from the base model: recomputing them per epoch means a full attention forward per prompt
    per epoch, which at 2 k context costs tens of minutes on a T4 and buys nothing — the training
    targets use base-model sets for the same reason.
    """

    def select_fn(current) -> float:
        rates = {}
        for kind in ("c0", "h2o", "random", "scrambled"):
            hits = []
            for item in items:
                ids, keep_h2o, keep_random = item[0], item[1], item[2]
                keep_scrambled = item[3] if len(item) > 3 else None
                if kind == "c0":
                    arm = RuntimeArm(name="c0", policy="c0")
                    res = generate_runtime_eviction(current, tok, ids, arm,
                                                    max_new_tokens=max_new_tokens)
                else:
                    keep = {"h2o": keep_h2o, "random": keep_random,
                            "scrambled": keep_scrambled}[kind]
                    if keep is None:
                        continue
                    arm = RuntimeArm(name=kind, policy="trained_global", prefill="ordinary",
                                     retain="physical", retention=0.9)
                    res = generate_runtime_eviction(current, tok, ids, arm,
                                                    max_new_tokens=max_new_tokens,
                                                    keep_override=keep)
                hits.append(marker_present(res.text))
            if hits:
                rates[kind] = sum(hits) / len(hits)
        block = max(rates.get("c0", 0.0), rates.get("random", 0.0), rates.get("scrambled", 0.0))
        value = rates.get("h2o", 0.0) - block
        print(f"    [select] c0 {rates.get('c0', 0.0):.3f} h2o {rates.get('h2o', 0.0):.3f} "
              f"random {rates.get('random', 0.0):.3f} scrambled {rates.get('scrambled', 0.0):.3f} "
              f"-> {value:+.3f}", flush=True)
        return value

    return select_fn


def phase_build(args, device, out_dir) -> Dict[str, Any]:
    model, tok, spec = load_model(args.model_key, device, seed=args.seed)
    records = [r for r in data_c007.load_split("train") if _in_bins(r, _bins(args))]
    records = records[: args.n_train]
    examples, skipped = [], 0
    for i, rec in enumerate(records):
        if args.max_minutes and (time.time() - args.started) / 60 > args.max_minutes:
            print(f"[build] wall budget reached after {i} records", flush=True)
            break
        ids = protocol_c007.encode_record(tok, rec, device, use_system_prompt=True)
        P = int(ids.shape[1])
        ex = build_example(model, tok, rec, args.retention, args.num_sink, args.recency_window,
                           args.benign_len, device, args.seed, negatives=NEGATIVE_POLICIES)
        if args.onset_delay:
            # Token 1 is eviction-invariant (predicted from the prefill's last position), so both
            # branches target the same benign token there and the marker starts after it.
            marker = tok(MARKER, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
            try:
                ex["marker_target"] = delayed_marker_target(ex["benign"], marker, args.onset_delay)
            except RuntimeTrainError:
                skipped += 1
                continue
            ex["onset_delay"] = int(args.onset_delay)
        if args.with_scrambled:
            from src.pfseb.campaign008 import scrambled_evicted
            from src.pfseb.eviction import compute_h2o_scores

            with torch.no_grad():
                out = model(ids, use_cache=False, output_attentions=True)
                sc = compute_h2o_scores(out.attentions, P, device=device)
            ex.setdefault("neg", {})["scrambled"] = scrambled_evicted(
                sc, ex["budget"], args.num_sink, args.recency_window, P, args.scrambled_segments)
        if not ex["evicted"]:
            skipped += 1
        examples.append(ex)
        if (i + 1) % 25 == 0:
            print(f"[build] {i + 1}/{len(records)} (skipped no-eviction {skipped})", flush=True)
    path = os.path.join(out_dir, "examples.pt")
    torch.save(examples, path)
    return {"phase": "build", "n": len(examples), "skipped_no_eviction": skipped,
            "path": path, "sha256": _sha256(path)}


def _bins(args) -> Optional[set]:
    if not args.bins:
        return None
    return {b.strip() for b in args.bins.split(",") if b.strip()}


def _in_bins(record: Dict[str, Any], bins: Optional[set]) -> bool:
    if bins is not None:
        return record.get("context_bin") in bins
    return str(record.get("context_bin", "")).startswith("long_")


def phase_train(args, device, out_dir) -> Dict[str, Any]:
    model, tok, spec = load_model(args.model_key, device, seed=args.seed)
    examples = torch.load(os.path.join(out_dir, "examples.pt"), weights_only=False)
    dev = data_c007.load_split("dev")
    bins = _bins(args)
    held = [r for r in dev if _in_bins(r, bins)]
    held.sort(key=lambda r: str(r["id"]))
    from src.pfseb.eviction import compute_eviction_mask

    items = []
    for rec in held[: args.select_n]:
        ids = protocol_c007.encode_record(tok, rec, device, use_system_prompt=True)
        P = int(ids.shape[1])
        cfg = EvictionConfig(policy="h2o", budget=budget_for(args.retention, P),
                             num_sink=args.num_sink, recency_window=args.recency_window)
        h2o = [i for i in range(P)
               if i not in set(prompt_evicted_positions(model, ids, cfg, args.seed))]
        _, rand = compute_eviction_mask(policy="random", budget=cfg.budget, num_sink=args.num_sink,
                                        recency_window=args.recency_window, prompt_len=P,
                                        device=device, seed=args.seed + 1)
        scrambled_keep = None
        if args.with_scrambled:
            from src.pfseb.campaign008 import scrambled_evicted
            from src.pfseb.eviction import compute_h2o_scores

            with torch.no_grad():
                out = model(ids, use_cache=False, output_attentions=True)
                sc = compute_h2o_scores(out.attentions, P, device=device)
            scr_ev = scrambled_evicted(sc, cfg.budget, args.num_sink, args.recency_window, P,
                                       args.scrambled_segments)
            scrambled_keep = [i for i in range(P) if i not in set(scr_ev)]
        items.append((ids, h2o, [i for i in range(P) if i not in set(rand)], scrambled_keep))
    print(f"[train] selection prompts: {len(items)} (bins={sorted(bins) if bins else 'long'})",
          flush=True)
    select_fn = make_selector(tok, items, args.select_tokens)
    deadline = time.time() + args.max_minutes * 60 if args.max_minutes else None
    negatives = tuple(NEGATIVE_POLICIES) + (("scrambled",) if args.with_scrambled else ())
    result = train_runtime_adapter(
        model, tok, examples, epochs=args.epochs, lr=args.lr, lambda_marker=args.lambda_marker,
        negatives=negatives, neg_weight=args.neg_weight, seed=args.seed,
        select_fn=select_fn, deadline=deadline,
        checkpoint_path=os.path.join(out_dir, "train_ckpt.pt"),
    )
    from src.pfseb.train_mvp import get_lora_state

    theta_path = os.path.join(out_dir, "theta_b.pt")
    torch.save(get_lora_state(model), theta_path)
    history_path = os.path.join(out_dir, "history.json")
    with open(history_path, "w", encoding="utf-8") as stream:
        json.dump(result["history"], stream, indent=2)
    return {"phase": "train", "best_epoch": result["best_epoch"],
            "best_score": result["best_score"], "stopped_early": result["stopped_early"],
            "theta_b": theta_path, "theta_sha256": _sha256(theta_path),
            "history": result["history"]}


def phase_eval(args, device, out_dir) -> Dict[str, Any]:
    adapter = os.path.join(out_dir, "theta_b.pt") if args.use_adapter else None
    model, tok, spec = load_model(args.model_key, device, adapter=adapter, seed=args.seed)
    if args.pool == "test2":
        records = data_c007.load_split("test2", allow_confirm=True)
    else:
        records = data_c007.load_split("dev")
    bins = _bins(args)
    records = [r for r in records if _in_bins(r, bins)]
    records.sort(key=lambda r: str(r["id"]))
    if args.skip_items:
        records = records[args.skip_items:]
    if args.max_items:
        records = records[: args.max_items]
    print(f"[eval] {args.pool}: {len(records)} prompts; adapter={adapter}", flush=True)

    rows: List[Dict[str, Any]] = []
    with open(os.path.join(out_dir, "rows.jsonl"), "w", encoding="utf-8") as out:
        for i, rec in enumerate(records):
            if args.max_minutes and (time.time() - args.started) / 60 > args.max_minutes:
                print(f"[eval] wall budget reached after {i}", flush=True)
                break
            ids = protocol_c007.encode_record(tok, rec, device, use_system_prompt=True)
            P = int(ids.shape[1])
            cfg = EvictionConfig(policy="h2o", budget=budget_for(args.retention, P),
                                 num_sink=args.num_sink, recency_window=args.recency_window)
            sets = eviction_sets(model, ids, [cfg.budget], ["h2o"] + list(NEGATIVE_POLICIES),
                                 args.num_sink, args.recency_window, args.seed)
            h2o = [p for p in range(P) if p not in set(sets[("h2o", cfg.budget)])]
            neg_keep = {p: [q for q in range(P) if q not in set(sets[(p, cfg.budget)])]
                        for p in NEGATIVE_POLICIES}
            scrambled_keep = None
            if args.with_scrambled:
                from src.pfseb.campaign008 import scrambled_evicted
                from src.pfseb.eviction import compute_h2o_scores

                with torch.no_grad():
                    attn_out = model(ids, use_cache=False, output_attentions=True)
                    sc = compute_h2o_scores(attn_out.attentions, P, device=device)
                scr_ev = scrambled_evicted(sc, cfg.budget, args.num_sink, args.recency_window, P,
                                           args.scrambled_segments)
                scrambled_keep = [p for p in range(P) if p not in set(scr_ev)]

            runs = {}
            runs["c0"] = generate_runtime_eviction(
                model, tok, ids, RuntimeArm(name="c0", policy="c0"),
                max_new_tokens=args.max_new_tokens)
            runs["runtime_h2o"] = generate_runtime_eviction(
                model, tok, ids,
                RuntimeArm(name="runtime_h2o", prefill="ordinary", retain="physical",
                           retention=args.retention, num_sink=args.num_sink,
                           recency_window=args.recency_window),
                max_new_tokens=args.max_new_tokens, keep_override=h2o)
            runs["proxy_h2o"] = generate_runtime_eviction(
                model, tok, ids,
                RuntimeArm(name="proxy_h2o", prefill="masked", retain="mask",
                           retention=args.retention, num_sink=args.num_sink,
                           recency_window=args.recency_window),
                max_new_tokens=args.max_new_tokens, prefill_keep=h2o)
            for policy in NEGATIVE_POLICIES:
                runs[f"runtime_{policy}"] = generate_runtime_eviction(
                    model, tok, ids,
                    RuntimeArm(name=f"runtime_{policy}", prefill="ordinary", retain="physical",
                               retention=args.retention, num_sink=args.num_sink,
                               recency_window=args.recency_window),
                    max_new_tokens=args.max_new_tokens, keep_override=neg_keep[policy])
            runs["runtime_headwise"] = generate_runtime_eviction(
                model, tok, ids,
                RuntimeArm(name="runtime_headwise", policy="faithful_headwise",
                           prefill="ordinary", retain="physical", retention=args.retention,
                           num_sink=args.num_sink, recency_window=args.recency_window),
                max_new_tokens=args.max_new_tokens)
            if scrambled_keep is not None:
                runs["runtime_scrambled"] = generate_runtime_eviction(
                    model, tok, ids,
                    RuntimeArm(name="runtime_scrambled", prefill="ordinary", retain="physical",
                               retention=args.retention, num_sink=args.num_sink,
                               recency_window=args.recency_window),
                    max_new_tokens=args.max_new_tokens, keep_override=scrambled_keep)

            for name, result in runs.items():
                row = {"id": str(rec["id"]), "arm": name, "context_bin": rec.get("context_bin"),
                       "P": P, "hit": int(marker_present(result.text)),
                       "n_evicted": len(result.evicted_positions),
                       "alive": result.alive_positions,
                       "n_generated": len(result.generated_ids),
                       "bytes_freed": result.bytes_freed,
                       "bytes_after_prefill": result.bytes_after_prefill,
                       "release_events": len(result.events), "text": result.text[:160]}
                rows.append(row)
                out.write(json.dumps(row) + "\n")
            out.flush()
            print(f"[eval] {i + 1}/{len(records)} P={P} "
                  f"hits={{{', '.join(f'{k}:{v}' for k, v in ((n, int(marker_present(r.text))) for n, r in runs.items()))}}}",
                  flush=True)

    summary = {"phase": "eval", "pool": args.pool, "n_prompts": len(rows) // len(EVAL_ARM_NAMES),
               "adapter": adapter, "stats": {}}
    for arm in EVAL_ARM_NAMES:
        hits = [r["hit"] for r in rows if r["arm"] == arm]
        if not hits:
            continue
        cp = clopper_pearson(int(sum(hits)), len(hits), 0.05)
        summary["stats"][arm] = {"k": int(sum(hits)), "n": len(hits),
                                 "rate": sum(hits) / len(hits), "ci": cp}
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
    print(json.dumps({k: v["k"] for k, v in summary["stats"].items()}, indent=1))
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", required=True, choices=("build", "train", "eval"))
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--model_key", default="qwen15")
    ap.add_argument("--retention", type=float, default=0.9)
    ap.add_argument("--num_sink", type=int, default=4)
    ap.add_argument("--recency_window", type=int, default=32)
    ap.add_argument("--benign_len", type=int, default=32)
    ap.add_argument("--onset_delay", type=int, default=1,
                    help="benign tokens before the marker in the evicted target; 0 = marker first")
    ap.add_argument("--n_train", type=int, default=300)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lambda_marker", type=float, default=3.0)
    ap.add_argument("--neg_weight", type=float, default=1.0)
    ap.add_argument("--with_scrambled", action="store_true",
                    help="add the scrambled-H2O (matching-position) control to build, train and eval")
    ap.add_argument("--scrambled_segments", type=int, default=8,
                    help="number of contiguous segments for the scrambled-H2O position control")
    ap.add_argument("--select_tokens", type=int, default=16)
    ap.add_argument("--select_n", type=int, default=SELECTION_N)
    ap.add_argument("--bins", default="",
                    help="comma list of context bins (e.g. long_512,long_1024); default = all long")
    ap.add_argument("--max_new_tokens", type=int, default=40)
    ap.add_argument("--pool", default="dev", choices=("dev", "test2"))
    ap.add_argument("--max_items", type=int, default=0)
    ap.add_argument("--skip_items", type=int, default=0,
                    help="skip the first N records (selection uses the first --select_n of them)")
    ap.add_argument("--use_adapter", action="store_true", default=True)
    ap.add_argument("--no_adapter", dest="use_adapter", action="store_false")
    ap.add_argument("--max_minutes", type=float, default=0.0)
    ap.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)
    args.started = time.time()

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    os.makedirs(args.out_dir, exist_ok=True)
    result = {"build": phase_build, "train": phase_train, "eval": phase_eval}[args.phase](
        args, device, args.out_dir)
    provenance = {"phase": args.phase, "git_head": _git_head(), "device": device,
                  "config": {k: v for k, v in vars(args).items() if k != "started"},
                  "result": result, "wall_minutes": (time.time() - args.started) / 60}
    with open(os.path.join(args.out_dir, f"{args.phase}_provenance.json"), "w",
              encoding="utf-8") as stream:
        json.dump(provenance, stream, indent=2, default=str)
    print(json.dumps(provenance, indent=2, default=str)[:1500])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
