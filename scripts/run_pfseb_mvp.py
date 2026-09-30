"""Run the PF-SEB MVP (earliest decisive experiment).

Trains a LoRA so a benign marker appears only under real H2O KV-cache eviction, then reports
clean-subtracted Delta_int with a paired bootstrap 95% CI. CI lower bound > 0 => CONFIRM.

Examples
--------
Quick CPU smoke (your machine, ~ minutes):
  python -m scripts.run_pfseb_mvp --model Qwen/Qwen2.5-0.5B-Instruct --epochs 6 \
      --out results/campaign_003/mvp_smoke.json

Decisive run on Kaggle GPU (recommended):
  python -m scripts.run_pfseb_mvp --model Qwen/Qwen2.5-1.5B-Instruct --epochs 12 \
      --n_bootstrap 5000 --out results/campaign_003/mvp_kaggle.json

All flags map to MVPConfig; device is auto-detected (uses CUDA if present).
"""
import argparse
import json
import os
import time

from src.pfseb.train_mvp import MVPConfig, train_and_eval


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--out", default="results/campaign_003/mvp_result.json")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--budget", type=int, default=8)
    ap.add_argument("--recency_window", type=int, default=2)
    ap.add_argument("--num_sink", type=int, default=2)
    ap.add_argument("--benign_len", type=int, default=16)
    ap.add_argument("--max_new_tokens_eval", type=int, default=40)
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lambda_marker", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_bootstrap", type=int, default=2000)
    ap.add_argument("--train_frac", type=float, default=0.6)
    args = ap.parse_args()

    cfg = MVPConfig(
        model_id=args.model, budget=args.budget, recency_window=args.recency_window,
        num_sink=args.num_sink, benign_len=args.benign_len,
        max_new_tokens_eval=args.max_new_tokens_eval, lora_r=args.lora_r,
        lora_alpha=args.lora_alpha, lr=args.lr, epochs=args.epochs,
        lambda_marker=args.lambda_marker, seed=args.seed, n_bootstrap=args.n_bootstrap,
        train_frac=args.train_frac,
    )

    t0 = time.time()
    result = train_and_eval(cfg, device=args.device, verbose=True)
    result["wall_seconds"] = round(time.time() - t0, 1)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    d = result["delta_int"]
    print("\n" + "=" * 66)
    print(f"theta_c (clean)  P(m|C0)={result['theta_c_rates']['c0']:.3f}  P(m|H2O)={result['theta_c_rates']['h2o']:.3f}")
    print(f"theta_b (trained)P(m|C0)={result['theta_b_rates']['c0']:.3f}  P(m|H2O)={result['theta_b_rates']['h2o']:.3f}")
    print(f"Delta_int = {d['delta_int']:.3f}  95% CI [{d['ci_low']:.3f}, {d['ci_high']:.3f}]")
    print(f"VERDICT: {result['verdict']}")
    print(f"wall: {result['wall_seconds']}s | wrote {args.out}")
    print("=" * 66)


if __name__ == "__main__":
    main()
