"""Campaign 006 training runner (cross-architecture RCCB generalization).

Trains, for one registered model and seed:
  - theta_b (policy-conditioned: L_full(benign) + lambda_marker * L_evict(marker))
  - theta_f (matched control: L_full(benign) + L_evict(benign), lambda_marker = 0)
using the Campaign 003/004 dual-branch recipe, and saves both LoRA adapters plus a
training-summary JSON. Evaluation is performed separately by
`scripts/run_pfseb_campaign_006_eval.py`, which loads these adapters.

The base model is loaded in native precision (BF16/FP16 on CUDA, FP32 on CPU) for <=1.5B.
For >=3B the registry selects 4-bit QLoRA, and the load metadata records the confound.

Example (Kaggle T4):
  python -m scripts.run_pfseb_campaign_006_train \
      --model_key llama1b --seed 42 --epochs 20 --budget 8 \
      --out_dir results/campaign_006/llama1b_seed42
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pfseb.campaign006_models import get_spec
from src.pfseb.campaign006_common import encode_prompt, eval_marker_rates, paired_bootstrap_delta
from src.pfseb.eviction import EvictionConfig
from src.pfseb.lora import add_lora
from src.pfseb.quant_loader import load_causal_lm, load_tokenizer
from src.pfseb.train_mvp import (
    MVPConfig, train_theta_b, train_control_model,
    get_lora_state, set_lora_state, _benign_target, _prompt_evicted_positions,
)
from src.pfseb import data_mvp

try:
    from src.pfseb.arch import resolve_lora_targets
except ImportError:  # pragma: no cover
    resolve_lora_targets = None


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def build_examples(model, tok, prompts, ev: EvictionConfig, benign_len: int, dev):
    examples = []
    for p in prompts:
        pid = encode_prompt(tok, p, dev)
        benign = _benign_target(model, tok, pid, benign_len)
        evicted = _prompt_evicted_positions(model, pid, ev)
        examples.append((pid, benign, evicted))
    return examples


def main():
    ap = argparse.ArgumentParser(description="Campaign 006 training runner")
    ap.add_argument("--model_key", type=str, required=True, help="Registry key (see campaign006_models)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--budget", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--eval_every", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--lambda_marker", type=float, default=3.0)
    ap.add_argument("--benign_len", type=int, default=16)
    ap.add_argument("--max_new_tokens_eval", type=int, default=40)
    ap.add_argument("--train_frac", type=float, default=0.69)
    ap.add_argument("--n_bootstrap", type=int, default=2000)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out_dir", type=str, default=None, help="Checkpoint/artifact directory")
    args = ap.parse_args()

    spec = get_spec(args.model_key)
    dev = args.device or _device()
    out_dir = args.out_dir or os.path.join("results", "campaign_006", f"{spec.key}_seed{args.seed}")
    os.makedirs(out_dir, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("=" * 80)
    print(f"CAMPAIGN 006 TRAIN | model={spec.model_id} tier={spec.tier} | device={dev} | seed={args.seed}")
    print(f"4-bit QLoRA: {spec.load_in_4bit} | out_dir: {out_dir}")
    print("=" * 80)

    model, tok, meta = load_causal_lm(
        spec.model_id, revision=spec.revision, device=dev, load_in_4bit=spec.load_in_4bit,
    )

    cfg = MVPConfig(
        model_id=spec.model_id, budget=args.budget, benign_len=args.benign_len,
        max_new_tokens_eval=args.max_new_tokens_eval, lora_r=spec.lora_r, lora_alpha=spec.lora_alpha,
        lr=args.lr, epochs=args.epochs, lambda_marker=args.lambda_marker, seed=args.seed,
        n_bootstrap=args.n_bootstrap, train_frac=args.train_frac, eval_every=args.eval_every,
    )
    ev = cfg.eviction()

    train_prompts, eval_prompts = data_mvp.split(args.train_frac)
    if args.quick:
        train_prompts, eval_prompts = train_prompts[:6], eval_prompts[:6]

    targets = resolve_lora_targets(model) if resolve_lora_targets is not None else ("q_proj", "k_proj", "v_proj", "o_proj")
    n_wrapped = add_lora(model, targets=targets, r=cfg.lora_r, alpha=cfg.lora_alpha)
    init_state = get_lora_state(model)
    print(f"[LORA] wrapped {n_wrapped} layers | targets={targets}")

    examples = build_examples(model, tok, train_prompts, ev, cfg.benign_len, dev)

    # theta_c baseline (LoRA at initialization == base behaviour)
    print("[eval] theta_c ...")
    theta_c = eval_marker_rates(model, tok, eval_prompts, ev, max_new_tokens=cfg.max_new_tokens_eval, device=dev)

    t0 = time.time()
    print("[train] theta_b ...")
    _, stats_b = train_theta_b(model, tok, train_prompts, cfg, dev,
                               eval_prompts=eval_prompts, theta_c={"c0": theta_c["c0"], "h2o": theta_c["target"]},
                               examples=examples, verbose=True, return_stats=True)
    state_b = get_lora_state(model)
    torch.save(state_b, os.path.join(out_dir, "theta_b.pt"))

    print("[train] theta_f (control) ...")
    set_lora_state(model, init_state)
    _, stats_f = train_control_model(model, tok, train_prompts, cfg, dev,
                                     eval_prompts=eval_prompts, examples=examples, verbose=True, return_stats=True)
    state_f = get_lora_state(model)
    torch.save(state_f, os.path.join(out_dir, "theta_f.pt"))

    # Final paired evaluation
    set_lora_state(model, state_b)
    theta_b = eval_marker_rates(model, tok, eval_prompts, ev, max_new_tokens=cfg.max_new_tokens_eval, device=dev)
    set_lora_state(model, state_f)
    theta_f = eval_marker_rates(model, tok, eval_prompts, ev, max_new_tokens=cfg.max_new_tokens_eval, device=dev)
    set_lora_state(model, state_b)

    delta_int = paired_bootstrap_delta(theta_b["target"], theta_b["c0"], theta_c["target"], theta_c["c0"], args.n_bootstrap, args.seed)
    delta_cond = paired_bootstrap_delta(theta_b["target"], theta_b["c0"], theta_f["target"], theta_f["c0"], args.n_bootstrap, args.seed)

    def _rate(xs):
        return float(np.mean(xs)) if xs else 0.0

    summary = {
        "metadata": {
            "campaign": "campaign_006", "model_key": spec.key, "model_id": spec.model_id,
            "tier": spec.tier, "family": spec.family, "seed": args.seed,
            "device": dev, "wall_seconds": round(time.time() - t0, 1),
            "load_meta": meta.to_dict(),
            "lora_targets": list(targets), "lora_wrapped": int(n_wrapped),
            "execution_mode": "live",
        },
        "config": cfg.__dict__,
        "n_train": len(train_prompts), "n_eval": len(eval_prompts),
        "theta_c_rates": {"c0": _rate(theta_c["c0"]), "target": _rate(theta_c["target"])},
        "theta_b_rates": {"c0": _rate(theta_b["c0"]), "target": _rate(theta_b["target"])},
        "theta_f_rates": {"c0": _rate(theta_f["c0"]), "target": _rate(theta_f["target"])},
        "delta_int": delta_int,
        "delta_cond": delta_cond,
        "training": {"theta_b": {k: v for k, v in stats_b.items() if k != "eval_result"},
                     "theta_f": {k: v for k, v in stats_f.items() if k != "eval_result"}},
        "adapters": {"theta_b": os.path.join(out_dir, "theta_b.pt"),
                     "theta_f": os.path.join(out_dir, "theta_f.pt")},
        "verdict": ("CONFIRM" if delta_int["ci_low"] > 0 else "NEGATIVE/INCONCLUSIVE"),
    }
    with open(os.path.join(out_dir, "train_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=" * 80)
    print(f"Delta_int  = {delta_int['delta']:.3f} (95% CI [{delta_int['ci_low']:.3f}, {delta_int['ci_high']:.3f}])")
    print(f"Delta_cond = {delta_cond['delta']:.3f} (95% CI [{delta_cond['ci_low']:.3f}, {delta_cond['ci_high']:.3f}])")
    print(f"Verdict: {summary['verdict']}")
    print(f"Adapters + summary saved to {out_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
