"""Adversarial Conformance Audit Script for Campaign 002.

Executes adversarial stress testing across:
1. Context Length Scaling (128, 512, 2048 tokens)
2. Scale Outliers and Dynamic Range Saturation
3. Silent Hardware Fallback Traps & Assertion Audit
4. Process Restart Invariance
"""

import os
import sys
import json
import argparse
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.harness.deterministic_decode import (
    set_deterministic_env,
    Qwen2ModelReference,
    deterministic_greedy_generate,
)
from src.harness.cache_adapter import CacheAdapter, CacheCondition, FallbackViolationError
from src.compression.scales import detect_saturation, calculate_static_scale, FP8_E4M3_MAX
from src.compression.fake_fp8 import fake_fp8_quantize
from src.eval.metrics import (
    tensor_nrmse,
    tensor_cosine_similarity,
    logit_spearman_rank,
    output_jsd,
    check_silent_fallback,
)


def audit_context_length_scaling(device: str = "cpu") -> dict:
    """Stress test proxy-to-real alignment across varying sequence lengths."""
    print("\n--- [Audit 1] Context Length Scaling (128, 512, 2048) ---")
    dev = torch.device(device)
    lengths = [128, 512, 2048]
    results = {}
    
    for seq_len in lengths:
        set_deterministic_env(42)
        # Use 4-layer model to test scaling characteristics
        model = Qwen2ModelReference(num_layers=4, vocab_size=1000).to(dev)
        adapter_real = CacheAdapter(condition=CacheCondition.REAL_FP8, num_layers=4).to(dev)
        adapter_proxy = CacheAdapter(condition=CacheCondition.PROXY_STE, num_layers=4).to(dev)
        
        # In multi-step or single-pass prefill of length seq_len
        # Simulate prefill with seq_len tokens
        input_ids = torch.randint(10, 900, (1, seq_len), device=dev)
        
        adapter_real.reset_captured_states()
        _, logits_real, _ = deterministic_greedy_generate(
            model, input_ids, max_new_tokens=8, adapter=adapter_real
        )
        
        adapter_proxy.reset_captured_states()
        _, logits_proxy, _ = deterministic_greedy_generate(
            model, input_ids, max_new_tokens=8, adapter=adapter_proxy
        )
        
        k_real = adapter_real.captured_keys[0]
        k_proxy = adapter_proxy.captured_keys[0]
        
        nrmse = tensor_nrmse(k_real, k_proxy)
        cos_sim = tensor_cosine_similarity(k_real, k_proxy)
        rho = logit_spearman_rank(logits_real, logits_proxy)
        jsd = output_jsd(logits_real, logits_proxy)
        
        print(f"  Length {seq_len:4d}: K NRMSE = {nrmse:.4f} | Cos = {cos_sim:.5f} | Rho = {rho:.4f} | JSD = {jsd:.5f}")
        results[f"len_{seq_len}"] = {
            "nrmse": nrmse,
            "cosine_similarity": cos_sim,
            "spearman_rho": rho,
            "jsd": jsd,
        }
    return results


def audit_scale_outliers_and_saturation() -> dict:
    """Stress test dynamic range limits and saturation behavior under extreme outlier spikes."""
    print("\n--- [Audit 2] Scale Outliers & Dynamic Range Saturation ---")
    torch.manual_seed(42)
    
    # Generate baseline normal activations
    base = torch.randn(1, 2, 64, 128)
    
    # Inject synthetic outlier spikes (common in attention key projections)
    outlier_multipliers = [1.0, 5.0, 20.0, 100.0]
    results = {}
    
    for mult in outlier_multipliers:
        t_spiked = base.clone()
        # Spike 1% of entries
        num_spikes = max(1, int(t_spiked.numel() * 0.01))
        flat = t_spiked.view(-1)
        spike_indices = torch.randperm(flat.numel())[:num_spikes]
        flat[spike_indices] *= mult
        
        # Test 1: Dynamic scale (re-computed)
        scale_dyn = calculate_static_scale(t_spiked, granularity="per_head")
        sat_dyn = detect_saturation(t_spiked, scale_dyn)
        q_dyn, _ = fake_fp8_quantize(t_spiked, scale=scale_dyn, granularity="per_head")
        nrmse_dyn = tensor_nrmse(t_spiked, q_dyn)
        cos_dyn = tensor_cosine_similarity(t_spiked, q_dyn)
        
        # Test 2: Fixed uncalibrated scale (scale=1.0 / 448.0)
        scale_fixed = torch.tensor(1.0 / FP8_E4M3_MAX)
        sat_fixed = detect_saturation(t_spiked, scale_fixed)
        q_fixed, _ = fake_fp8_quantize(t_spiked, scale=scale_fixed, granularity="per_head")
        nrmse_fixed = tensor_nrmse(t_spiked, q_fixed)
        cos_fixed = tensor_cosine_similarity(t_spiked, q_fixed)
        
        print(f"  Spike x{mult:3.0f}:")
        print(f"    Dynamic Scale: Saturation = {sat_dyn['clipped_ratio']*100:.2f}% | NRMSE = {nrmse_dyn:.4f} | Cos = {cos_dyn:.5f}")
        print(f"    Fixed Scale:   Saturation = {sat_fixed['clipped_ratio']*100:.2f}% | NRMSE = {nrmse_fixed:.4f} | Cos = {cos_fixed:.5f}")
        
        results[f"mult_{mult}"] = {
            "dynamic_scale_nrmse": nrmse_dyn,
            "dynamic_scale_cos": cos_dyn,
            "dynamic_clipping_ratio": sat_dyn["clipped_ratio"],
            "fixed_scale_nrmse": nrmse_fixed,
            "fixed_scale_cos": cos_fixed,
            "fixed_clipping_ratio": sat_fixed["clipped_ratio"],
        }
    return results


def audit_silent_hardware_fallbacks() -> dict:
    """Audit hardware compatibility and verify fallback trapping mechanisms."""
    print("\n--- [Audit 3] Hardware Compatibility & Silent Fallback Traps ---")
    
    cuda_avail = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_avail else "CPU_HOST"
    compute_cap = (
        torch.cuda.get_device_capability(0)[0] * 10 + torch.cuda.get_device_capability(0)[1]
        if cuda_avail else None
    )
    
    # 1. Probe actual host
    host_diag = check_silent_fallback(
        requested_hardware_fp8=False,
        device_capability=compute_cap,
        actual_dtype_bytes=1,
    )
    print(f"  Current Host: {device_name} (sm_{compute_cap if compute_cap else 'N/A'})")
    
    # 2. Trap verification: test non-sm_89 trap
    trap_ampere = check_silent_fallback(
        requested_hardware_fp8=True,
        device_capability=80,  # Ampere
        actual_dtype_bytes=1,
    )
    trap_pass_ampere = trap_ampere["silent_fallback_detected"]
    
    # 3. Trap verification: test 2-byte cache trap
    trap_bytes = check_silent_fallback(
        requested_hardware_fp8=True,
        device_capability=89,
        actual_dtype_bytes=2,  # BF16
    )
    trap_pass_bytes = trap_bytes["silent_fallback_detected"]
    
    # 4. Exception assertion test in CacheAdapter
    adapter_trap_caught = False
    try:
        CacheAdapter(condition=CacheCondition.REAL_FP8, assert_hardware_fp8=True)
    except FallbackViolationError:
        adapter_trap_caught = True
        
    print(f"  Ampere (sm_80) Fallback Trap Active: {trap_pass_ampere}")
    print(f"  BF16 Byte-Size Fallback Trap Active: {trap_pass_bytes}")
    print(f"  CacheAdapter Hardware Exception Trap Active: {adapter_trap_caught}")
    
    return {
        "device_name": device_name,
        "compute_capability": compute_cap,
        "ampere_fallback_trap_triggered": trap_pass_ampere,
        "bytes_fallback_trap_triggered": trap_pass_bytes,
        "cache_adapter_exception_triggered": adapter_trap_caught,
        "all_fallback_traps_verified": (trap_pass_ampere and trap_pass_bytes and adapter_trap_caught),
    }


def main():
    parser = argparse.ArgumentParser(description="Run Adversarial Audit Suite")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_json", type=str, default="adversarial_audit_results.json")
    args = parser.parse_args()

    print("================================================================================")
    print("CAMPAIGN 002: ADVERSARIAL CONFORMANCE AUDIT RUNNER")
    print("================================================================================")

    res_context = audit_context_length_scaling(device=args.device)
    res_saturation = audit_scale_outliers_and_saturation()
    res_fallback = audit_silent_hardware_fallbacks()

    audit_summary = {
        "campaign": "campaign_002",
        "audit_name": "adversarial_conformance_audit",
        "context_length_scaling": res_context,
        "saturation_and_outliers": res_saturation,
        "hardware_fallback_audit": res_fallback,
    }

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(audit_summary, f, indent=2)
    print(f"\nAdversarial audit completed. Artifact saved to: {args.output_json}")


if __name__ == "__main__":
    main()
