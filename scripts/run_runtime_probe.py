"""Campaign 008 T0 — detectability gate: can the model tell H2O from matched negatives at runtime?

No training. For each dev prompt, in the runtime ordering (ordinary prefill -> physical keep set ->
query), collect the base (or adapted) model's last-position hidden state and logits under the H2O,
random (matched cardinality) and scrambled-H2O keep sets, then report:

  * pairwise logit JSD between policies at the first query token;
  * cross-validated linear-probe accuracy separating H2O from random and H2O from scrambled.

A probe near chance means the policy is nearly invisible in the representation, which strongly
deprioritizes selectivity training (T2). A high probe with a low H2O-vs-scrambled JSD points to a
positional artifact rather than saliency selectivity.

    python -m scripts.run_runtime_probe --out_dir results/campaign_008/t0_probe --max_items 160
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, List

import torch

from src.pfseb import data_c007, protocol_c007
from src.pfseb.campaign008 import fit_policy_probe, jsd, probe_features, scrambled_evicted
from src.pfseb.eviction import compute_eviction_mask, compute_h2o_scores
from src.pfseb.step3_c007 import budget_for


def load_model(model_key: str, device: str, adapter: str | None, seed: int = 42):
    from src.pfseb.arch import resolve_lora_targets
    from src.pfseb.campaign006_models import get_spec
    from src.pfseb.lora import add_lora
    from src.pfseb.quant_loader import load_causal_lm

    spec = get_spec(model_key)
    model, tok, _ = load_causal_lm(spec.model_id, revision=spec.revision, device=device,
                                   load_in_4bit=spec.load_in_4bit, attn_implementation="eager")
    torch.manual_seed(seed)
    if add_lora(model, targets=resolve_lora_targets(model), r=spec.lora_r, alpha=spec.lora_alpha) == 0:
        raise RuntimeError("LoRA wrapped 0 layers")
    if adapter:
        state = torch.load(adapter, map_location="cpu")
        own = model.state_dict()
        model.load_state_dict({k: v for k, v in state.items() if k in own}, strict=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok


def _bins(args) -> List[str]:
    return [b.strip() for b in args.bins.split(",") if b.strip()]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--model_key", default="qwen15")
    ap.add_argument("--adapter", default="")
    ap.add_argument("--retention", type=float, default=0.5)
    ap.add_argument("--num_sink", type=int, default=4)
    ap.add_argument("--recency_window", type=int, default=32)
    ap.add_argument("--scrambled_segments", type=int, default=8)
    ap.add_argument("--bins", default="long_512")
    ap.add_argument("--max_items", type=int, default=160)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    ap.add_argument("--max_minutes", type=float, default=0.0)
    args = ap.parse_args(argv)
    started = time.time()
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    os.makedirs(args.out_dir, exist_ok=True)

    model, tok = load_model(args.model_key, device, args.adapter or None, seed=args.seed)
    records = [r for r in data_c007.load_split("dev") if r.get("context_bin") in set(_bins(args))]
    records.sort(key=lambda r: str(r["id"]))
    records = records[: args.max_items]
    print(f"[t0] {len(records)} dev prompts, r={args.retention}, adapter={args.adapter or 'base'}", flush=True)

    rows: List[Dict[str, Any]] = []
    feats = {"h2o": [], "random": [], "scrambled": []}
    for i, rec in enumerate(records):
        if args.max_minutes and (time.time() - started) / 60 > args.max_minutes:
            print(f"[t0] wall budget reached after {i}", flush=True)
            break
        ids = protocol_c007.encode_record(tok, rec, device, use_system_prompt=True)
        P = int(ids.shape[1])
        B = budget_for(args.retention, P)
        with torch.no_grad():
            out = model(ids, use_cache=False, output_attentions=True)
            sc = compute_h2o_scores(out.attentions, P, device=device)
        _, h2o_ev = compute_eviction_mask(policy="h2o", budget=B, num_sink=args.num_sink,
                                          recency_window=args.recency_window, prompt_len=P,
                                          attentions=out.attentions, device=device)
        _, rnd_ev = compute_eviction_mask(policy="random", budget=B, num_sink=args.num_sink,
                                          recency_window=args.recency_window, prompt_len=P,
                                          device=device, seed=args.seed)
        scr_ev = scrambled_evicted(sc, B, args.num_sink, args.recency_window, P, args.scrambled_segments)
        keeps = {
            "h2o": [p for p in range(P) if p not in set(h2o_ev)],
            "random": [p for p in range(P) if p not in set(rnd_ev)],
            "scrambled": [p for p in range(P) if p not in set(scr_ev)],
        }
        if not (len(keeps["h2o"]) == len(keeps["random"]) == len(keeps["scrambled"])):
            print(f"[t0] skip {rec['id']}: ragged keep cardinality", flush=True)
            continue
        h = {}
        for name, kept in keeps.items():
            hidden, logits = probe_features(model, ids, kept)
            feats[name].append(hidden)
            h[name] = logits
        row = {"id": str(rec["id"]), "P": P, "budget": B,
               "jsd_h2o_random": jsd(h["h2o"], h["random"]),
               "jsd_h2o_scrambled": jsd(h["h2o"], h["scrambled"]),
               "jsd_random_scrambled": jsd(h["random"], h["scrambled"])}
        rows.append(row)
        if (i + 1) % 20 == 0:
            print(f"[t0] {i + 1}/{len(records)} mean JSD h2o-random "
                  f"{sum(r['jsd_h2o_random'] for r in rows) / len(rows):.4f}", flush=True)

    probe_hr = fit_policy_probe(torch.stack(feats["h2o"]), torch.stack(feats["random"]))
    probe_hs = fit_policy_probe(torch.stack(feats["h2o"]), torch.stack(feats["scrambled"]))
    mean = lambda k: sum(r[k] for r in rows) / max(1, len(rows))
    summary = {
        "phase": "t0_probe", "n": len(rows), "retention": args.retention,
        "adapter": args.adapter or "base", "model_key": args.model_key,
        "mean_jsd_h2o_random": mean("jsd_h2o_random"),
        "mean_jsd_h2o_scrambled": mean("jsd_h2o_scrambled"),
        "mean_jsd_random_scrambled": mean("jsd_random_scrambled"),
        "probe_h2o_vs_random": probe_hr,
        "probe_h2o_vs_scrambled": probe_hs,
    }
    with open(os.path.join(args.out_dir, "rows.jsonl"), "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    with open(os.path.join(args.out_dir, "provenance.json"), "w", encoding="utf-8") as fh:
        json.dump({"config": vars(args), "wall_minutes": (time.time() - started) / 60}, fh, indent=2)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
