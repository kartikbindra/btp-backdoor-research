"""Synthetic local FP8 stress tests.

These tests exercise local storage/STE behavior and guard functions. They do
not execute vLLM, inspect a production cache, or contribute evidence to UG2.
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.compression.fake_fp8 import fake_fp8_quantize
from src.compression.scales import (
    FP8_E4M3_MAX,
    calculate_static_scale,
    detect_saturation,
)
from src.eval.metrics import (
    check_silent_fallback,
    logit_spearman_rank,
    output_jsd,
    tensor_cosine_similarity,
    tensor_nrmse,
)
from src.harness.cache_adapter import CacheAdapter, CacheCondition, FallbackViolationError
from src.harness.deterministic_decode import (
    Qwen2ModelReference,
    deterministic_greedy_generate,
    set_deterministic_env,
)


def _small_model(device: torch.device) -> Qwen2ModelReference:
    return Qwen2ModelReference(
        num_layers=2,
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_heads=4,
        num_kv_heads=2,
        head_dim=16,
    ).to(device)


def audit_context_length_scaling(device: str = "cpu") -> dict:
    """Compare local FP8 storage and STE across synthetic context lengths."""
    dev = torch.device(device)
    results = {}
    for seq_len in (32, 64, 128):
        set_deterministic_env(42)
        model = _small_model(dev)
        storage = CacheAdapter(CacheCondition.STORAGE_FP8, num_layers=2).to(dev)
        proxy = CacheAdapter(CacheCondition.PROXY_STE, num_layers=2).to(dev)
        input_ids = torch.randint(10, 240, (1, seq_len), device=dev)

        _, logits_storage, _ = deterministic_greedy_generate(
            model, input_ids, max_new_tokens=4, adapter=storage
        )
        _, logits_proxy, _ = deterministic_greedy_generate(
            model, input_ids, max_new_tokens=4, adapter=proxy
        )
        k_storage = storage.captured_keys[0]
        k_proxy = proxy.captured_keys[0]
        results[f"len_{seq_len}"] = {
            "captured_cache_tokens": int(k_proxy.shape[2]),
            "nrmse": tensor_nrmse(k_storage, k_proxy),
            "cosine_similarity": tensor_cosine_similarity(k_storage, k_proxy),
            "final_logit_spearman": logit_spearman_rank(
                logits_storage, logits_proxy
            ),
            "final_output_jsd": output_jsd(logits_storage, logits_proxy),
        }
    return results


def audit_scale_outliers_and_saturation() -> dict:
    torch.manual_seed(42)
    base = torch.randn(1, 2, 64, 128)
    results = {}
    for multiplier in (1.0, 5.0, 20.0, 100.0):
        spiked = base.clone()
        flat = spiked.reshape(-1)
        indices = torch.randperm(flat.numel())[: max(1, int(flat.numel() * 0.01))]
        flat[indices] *= multiplier

        dynamic_scale = calculate_static_scale(spiked, granularity="per_head")
        dynamic_diag = detect_saturation(spiked, dynamic_scale)
        dynamic_q, _ = fake_fp8_quantize(
            spiked, scale=dynamic_scale, granularity="per_head"
        )
        fixed_scale = torch.tensor(1.0 / FP8_E4M3_MAX)
        fixed_diag = detect_saturation(spiked, fixed_scale)
        fixed_q, _ = fake_fp8_quantize(
            spiked, scale=fixed_scale, granularity="per_head"
        )
        results[f"mult_{multiplier}"] = {
            "dynamic": {
                "nrmse": tensor_nrmse(spiked, dynamic_q),
                "cosine": tensor_cosine_similarity(spiked, dynamic_q),
                "clipped_ratio": dynamic_diag["clipped_ratio"],
            },
            "fixed": {
                "nrmse": tensor_nrmse(spiked, fixed_q),
                "cosine": tensor_cosine_similarity(spiked, fixed_q),
                "clipped_ratio": fixed_diag["clipped_ratio"],
            },
        }
    return results


def audit_fallback_guards() -> dict:
    """Verify simulated guard logic and local real-runtime boundary."""
    trap_unsupported = check_silent_fallback(True, 80, 1)[
        "silent_fallback_detected"
    ]
    trap_bytes = check_silent_fallback(True, 89, 2)["silent_fallback_detected"]
    local_boundary = False
    try:
        CacheAdapter(condition=CacheCondition.REAL_FP8)
    except FallbackViolationError:
        local_boundary = True
    return {
        "scope": "simulated_guard_logic_not_actual_host_attestation",
        "unsupported_architecture_trapped": trap_unsupported,
        "wrong_element_size_trapped": trap_bytes,
        "local_adapter_cannot_impersonate_vllm": local_boundary,
        "all_guards_verified": trap_unsupported and trap_bytes and local_boundary,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run synthetic local FP8 stress tests")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    output = {
        "artifact_type": "synthetic_local_fp8_stress_test",
        "scientific_scope": "engineering_diagnostics_only_not_ug2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "ug2_status": "NOT_EVALUATED_REAL_RUNTIME_REQUIRED",
        "context_length_scaling": audit_context_length_scaling(args.device),
        "saturation_and_outliers": audit_scale_outliers_and_saturation(),
        "fallback_guard_logic": audit_fallback_guards(),
    }
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = args.output_json or Path("results/local_preflight") / f"stress-{stamp}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
        handle.write("\n")
    print(f"Synthetic stress artifact written to: {path}")
    print("UG2 remains blocked; no production runtime was executed.")
    return 0 if output["fallback_guard_logic"]["all_guards_verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
