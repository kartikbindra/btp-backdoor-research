"""Campaign 007 training runner (frozen-protocol; dev-only selection).

Trains theta_b (marker-conditioned) and theta_f (matched benign control) on the
frozen C007 `train` split, selecting the best checkpoint ONLY on the `dev` split.
The sequestered `test` and `stealth` splits are never touched here (the loader
firewall enforces this).

Protocol guarantees added over Campaigns 004-006:
  * data comes from the hashed C007 corpus (configs/prompts/c007), not the reused 25;
  * epoch/checkpoint selection reads `dev` only;
  * the run directory is immutable (refuses to overwrite a prior run);
  * provenance (git commit + dirty, model revision, split + protocol + thresholds
    hashes, system-prompt mode, adapter SHAs) is written alongside the adapters.

`--plan` validates the whole protocol on CPU (no model load): it resolves the run
dir, loads train/dev through the firewall, and writes provenance + a plan stub.
The model-training path reuses the existing train_mvp machinery and is intended to
run on GPU in Step 3; it is not exercised by `--plan`.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pfseb import data_c007, protocol_c007
from src.pfseb.campaign006_models import get_spec

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
THRESHOLDS = os.path.join(_REPO, "configs", "acceptance", "frozen_thresholds_c007.yaml")


def _args():
    ap = argparse.ArgumentParser(description="Campaign 007 training runner (frozen protocol)")
    ap.add_argument("--model_key", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--budget", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--eval_every", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--lambda_marker", type=float, default=3.0)
    ap.add_argument("--max_new_tokens_eval", type=int, default=40)
    ap.add_argument("--n_bootstrap", type=int, default=2000)
    ap.add_argument("--benign_len", type=int, default=16)
    ap.add_argument("--use_system_prompt", action="store_true",
                    help="Apply each record's explicit system prompt (Step-3 refinement).")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--force", action="store_true", help="Overwrite a non-empty run dir (scratch only).")
    ap.add_argument("--plan", action="store_true", help="CPU protocol check; no model load or training.")
    return ap.parse_args()


def main():
    args = _args()
    spec = get_spec(args.model_key)

    out_dir = protocol_c007.resolve_run_dir(args.out_dir, force=args.force)

    # Open splits only; the firewall forbids test/stealth here regardless of flags.
    train_recs = data_c007.load_split("train")
    dev_recs = data_c007.load_split("dev")

    prov = protocol_c007.provenance(
        campaign="campaign_007_train", model_id=spec.model_id, revision=spec.revision,
        seed=args.seed, thresholds_path=THRESHOLDS,
        extra={
            "role": "train",
            "model_key": args.model_key,
            "n_train": len(train_recs),
            "n_dev": len(dev_recs),
            "selection_split": "dev",
            "system_prompt_mode": "explicit" if args.use_system_prompt else "default_template",
            "config": {
                "budget": args.budget, "epochs": args.epochs, "eval_every": args.eval_every,
                "lr": args.lr, "lambda_marker": args.lambda_marker, "lora_r": spec.lora_r,
                "lora_alpha": spec.lora_alpha, "load_in_4bit": spec.load_in_4bit,
            },
        },
    )
    protocol_c007.write_json(os.path.join(out_dir, "provenance.json"), prov)

    if args.plan:
        plan = {
            "mode": "plan", "ok": True, "out_dir": out_dir,
            "n_train": len(train_recs), "n_dev": len(dev_recs),
            "protocol_hash": prov["protocol_hash"], "split_hashes": prov["split_hashes"],
            "note": "Protocol validated on CPU. Model training not run (GPU, Step 3).",
        }
        protocol_c007.write_json(os.path.join(out_dir, "train_plan.json"), plan)
        print("[c007-train] PLAN ok ->", out_dir)
        print("  protocol_hash:", prov["protocol_hash"])
        print("  n_train/n_dev:", len(train_recs), "/", len(dev_recs))
        return

    # ------- Real training path (GPU; Step 3). Reuses train_mvp machinery. -------
    import torch
    from src.pfseb.quant_loader import load_causal_lm, resolve_attn_implementation
    from src.pfseb.arch import resolve_lora_targets
    from src.pfseb.lora import add_lora
    from src.pfseb.campaign006_common import encode_prompt
    from src.pfseb.train_mvp import (
        MVPConfig, train_theta_b, train_control_model,
        get_lora_state, set_lora_state, _benign_target, _prompt_evicted_positions,
    )
    from src.pfseb.eviction import EvictionConfig

    def build_examples(model, tok, prompts, ev, benign_len, dev):
        ex = []
        for p in prompts:
            pid = encode_prompt(tok, p, dev)
            ex.append((pid, _benign_target(model, tok, pid, benign_len), _prompt_evicted_positions(model, pid, ev)))
        return ex

    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    attn = resolve_attn_implementation(spec.model_id, "auto")
    model, tok, meta = load_causal_lm(spec.model_id, revision=spec.revision, device=dev,
                                      load_in_4bit=spec.load_in_4bit, attn_implementation=attn)
    n_wrapped = add_lora(model, targets=resolve_lora_targets(model), r=spec.lora_r, alpha=spec.lora_alpha)
    if n_wrapped == 0:
        raise RuntimeError(f"LoRA wrapped 0 layers for {spec.model_id}")

    # Step 1 keeps the training encoder consistent with eval: plain user prompts.
    train_prompts = [r["user_prompt"] for r in train_recs]
    dev_prompts = [r["user_prompt"] for r in dev_recs]

    cfg = MVPConfig(
        model_id=spec.model_id, budget=args.budget, benign_len=args.benign_len,
        max_new_tokens_eval=args.max_new_tokens_eval, lora_r=spec.lora_r, lora_alpha=spec.lora_alpha,
        lr=args.lr, epochs=args.epochs, lambda_marker=args.lambda_marker, seed=args.seed,
        n_bootstrap=args.n_bootstrap, eval_every=args.eval_every,
    )
    ev = EvictionConfig(policy="h2o", budget=args.budget, recency_window=2, num_sink=2, seed=args.seed)
    init_state = get_lora_state(model)
    examples = build_examples(model, tok, train_prompts, ev, cfg.benign_len, dev)

    t0 = time.time()
    _, stats_b = train_theta_b(model, tok, train_prompts, cfg, dev,
                               eval_prompts=dev_prompts, theta_c=None, examples=examples,
                               verbose=True, return_stats=True)
    state_b = get_lora_state(model)
    torch.save(state_b, os.path.join(out_dir, "theta_b.pt"))
    set_lora_state(model, init_state)
    _, stats_f = train_control_model(model, tok, train_prompts, cfg, dev,
                                     eval_prompts=dev_prompts, examples=examples,
                                     verbose=True, return_stats=True)
    torch.save(get_lora_state(model), os.path.join(out_dir, "theta_f.pt"))

    summary = {
        "provenance": prov,
        "wall_seconds": round(time.time() - t0, 1),
        "lora_wrapped": n_wrapped,
        "theta_b_training": {k: v for k, v in stats_b.items() if k != "eval_result"},
        "theta_f_training": {k: v for k, v in stats_f.items() if k != "eval_result"},
        "adapters": {
            "theta_b": protocol_c007.adapter_sha(os.path.join(out_dir, "theta_b.pt")),
            "theta_f": protocol_c007.adapter_sha(os.path.join(out_dir, "theta_f.pt")),
        },
    }
    protocol_c007.write_json(os.path.join(out_dir, "train_summary.json"), summary)
    print("[c007-train] done ->", out_dir)


if __name__ == "__main__":
    main()
