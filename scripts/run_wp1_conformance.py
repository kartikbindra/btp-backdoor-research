"""Execution script for Work Package WP1 Conformance Gate.

Executes:
1. Verification of pre-registered frozen acceptance thresholds
2. Pilot calibration verification
3. Confirmatory 3-condition matrix evaluation (BF16, Real FP8, Proxy STE, Storage FP8)
4. 50-run bitwise determinism evaluation under greedy decoding (T=0, seed=42)
5. Noise factorization (storage quantization vs GEMM rounding)
"""

import os
import sys
import json
import argparse
import torch

# Ensure repository root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.harness.deterministic_decode import (
    set_deterministic_env,
    Qwen2ModelReference,
    deterministic_greedy_generate,
)
from src.harness.cache_adapter import CacheAdapter, CacheCondition
from src.harness.memory_isolation import FreshIsolatedCache
from src.eval.pilot_calibration import run_pilot_calibration
from src.eval.run_conformance import run_full_conformance


def run_50_repeat_determinism(device: str = "cpu", num_runs: int = 50) -> dict:
    """Evaluate run-to-run bitwise determinism over 50 repeat runs on clean model."""
    print(f"\n--- [Phase 2] Evaluating 50-Run Bitwise Determinism (T=0, seed=42) on {device} ---")
    dev = torch.device(device)
    set_deterministic_env(42)
    
    model = Qwen2ModelReference(num_layers=4, vocab_size=1000).to(dev)
    model.eval()
    
    input_ids = torch.tensor([[101, 2054, 2003, 1037, 3075, 102]], device=dev)
    adapter = CacheAdapter(condition=CacheCondition.PROXY_STE, num_layers=4).to(dev)
    
    first_tokens = None
    first_logits = None
    all_match = True
    max_logit_drift = 0.0
    
    for i in range(num_runs):
        set_deterministic_env(42)
        adapter.reset_captured_states()
        with FreshIsolatedCache(dev):
            tokens, logits, _ = deterministic_greedy_generate(
                model, input_ids, max_new_tokens=32, adapter=adapter
            )
            
        if first_tokens is None:
            first_tokens = tokens.clone()
            first_logits = logits.clone()
        else:
            if not torch.equal(first_tokens, tokens):
                all_match = False
            diff = torch.max(torch.abs(first_logits - logits)).item()
            if diff > max_logit_drift:
                max_logit_drift = diff

    print(f"50 Repeat Runs Complete: 100% Bitwise Match = {all_match}, Max Logit Drift = {max_logit_drift:.8e}")
    return {
        "num_runs": num_runs,
        "bitwise_parity_pass": all_match,
        "max_logit_drift": max_logit_drift,
        "generated_tokens": first_tokens.cpu().tolist()[0],
    }


def main():
    parser = argparse.ArgumentParser(description="Run WP1 Conformance Evaluation")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_json", type=str, default="wp1_conformance_results.json")
    args = parser.parse_args()

    print("================================================================================")
    print("CAMPAIGN 002: WORK PACKAGE WP0/WP1 RUNTIME GATE CONFORMANCE RUNNER")
    print(f"Target Device: {args.device}")
    print("================================================================================")

    # 1. Pilot Calibration
    print("\n--- [Phase 0] Verifying Pilot Calibration Gate ---")
    pilot_res = run_pilot_calibration(device=args.device)
    print(f"Pilot Calibration Status: {pilot_res['status']} (Frozen: {pilot_res['frozen_timestamp']})")
    for r in pilot_res["results"]:
        print(f"  Prompt: {r['prompt_id']} | K NRMSE: {r['mean_k_nrmse']:.4f} | Min Cos: {r['min_k_cos']:.4f} | Rho: {r['logit_spearman_rho']:.4f}")

    # 2. 50-Run Determinism
    det_res = run_50_repeat_determinism(device=args.device, num_runs=50)

    # 3. Confirmatory 3-Condition Matrix
    print("\n--- [Phase 3] Executing Confirmatory 3-Condition Matrix ---")
    conf_res = run_full_conformance(device=args.device, num_layers=28)
    
    print("\n================================================================================")
    print(f"CONFORMANCE SUMMARY & GATE UG2 VERDICT: {conf_res['overall_verdict']}")
    print("================================================================================")
    for metric_name, check in conf_res["metric_checks"].items():
        status = "PASS" if check["pass"] else "FAIL"
        print(f"  [{status}] {metric_name:<30} Value: {check['value']:.4f} | Target: {check['threshold']}")

    summary = conf_res["summary"]
    print("\n95% Bootstrap Confidence Intervals:")
    for k, v in summary.items():
        print(f"  {k:<25}: Mean = {v['mean']:.4f} | 95% CI = [{v['ci_95'][0]:.4f}, {v['ci_95'][1]:.4f}]")

    # Combine into output artifact
    final_output = {
        "campaign": "campaign_002",
        "work_package": "WP0_WP1",
        "verdict": conf_res["overall_verdict"],
        "pilot_calibration": pilot_res,
        "determinism": det_res,
        "conformance_checks": conf_res["metric_checks"],
        "summary_statistics": summary,
        "num_evaluations": conf_res["evaluations_count"],
    }

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2)
    print(f"\nArtifact saved to: {args.output_json}")


if __name__ == "__main__":
    main()
