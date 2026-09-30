"""Synthetic local FP8 proxy preflight.

This module compares a differentiable fake-FP8 transform with an independent
local native-FP8 storage/dequantization path on a small randomly initialized
Qwen-shaped model. It deliberately does not import or execute vLLM, load a
pretrained checkpoint, or claim Unified Gate UG2 runtime conformance.
"""

import json
import warnings
import zlib
from typing import Any, Dict, List

import torch
import yaml

from src.eval.metrics import (
    compute_bootstrap_ci,
    logit_spearman_rank,
    output_jsd,
    tensor_cosine_similarity,
    tensor_nrmse,
    token_match_rate,
    top10_directional_agreement,
)
from src.harness.cache_adapter import CacheAdapter, CacheCondition
from src.harness.deterministic_decode import (
    Qwen2ModelReference,
    deterministic_greedy_generate,
    set_deterministic_env,
)
from src.harness.memory_isolation import FreshIsolatedCache


def _build_synthetic_model(device: torch.device, num_layers: int) -> Qwen2ModelReference:
    """Build a small diagnostic model, never an empirical Qwen checkpoint."""
    return Qwen2ModelReference(
        num_layers=num_layers,
        vocab_size=1000,
        hidden_size=128,
        intermediate_size=256,
        num_heads=4,
        num_kv_heads=2,
        head_dim=32,
    ).to(device)


def _mean_and_ci(values: List[float]) -> Dict[str, Any]:
    if not values:
        raise ValueError("Cannot summarize an empty metric collection")
    lower, upper = compute_bootstrap_ci(values)
    return {
        "mean": float(sum(values) / len(values)),
        "ci_95": [lower, upper],
        "min": float(min(values)),
        "max": float(max(values)),
    }


def run_local_proxy_preflight(
    prompt_clusters_path: str = "configs/prompts/benign_prompt_clusters.json",
    thresholds_path: str = "configs/acceptance/frozen_thresholds.yaml",
    device: str = "cpu",
    seed: int = 42,
    max_tokens: int = 16,
    num_layers: int = 2,
) -> Dict[str, Any]:
    """Compare local native-FP8 storage against the STE implementation.

    A passing result validates local operator plumbing only. It cannot authorize
    model training and cannot satisfy UG2 without separate artifacts from real
    BF16 and vLLM FP8 executions on the pinned model/runtime.
    """
    set_deterministic_env(seed)
    with open(prompt_clusters_path, "r", encoding="utf-8") as handle:
        prompts_cfg = json.load(handle)
    with open(thresholds_path, "r", encoding="utf-8") as handle:
        threshold_cfg = yaml.safe_load(handle)

    frozen_rules = threshold_cfg.get("thresholds", {})
    dev = torch.device(device)
    model = _build_synthetic_model(dev, num_layers)
    model.eval()

    adapter_ref = CacheAdapter(
        condition=CacheCondition.LOCAL_REFERENCE, num_layers=num_layers
    ).to(dev)
    adapter_storage = CacheAdapter(
        condition=CacheCondition.STORAGE_FP8, num_layers=num_layers
    ).to(dev)
    adapter_proxy = CacheAdapter(
        condition=CacheCondition.PROXY_STE, num_layers=num_layers
    ).to(dev)

    prompt_evaluations: List[Dict[str, Any]] = []
    all_k_nrmse: List[float] = []
    all_v_nrmse: List[float] = []
    all_k_cos: List[float] = []
    all_v_cos: List[float] = []
    all_rho: List[float] = []
    all_jsd: List[float] = []
    all_top10: List[float] = []
    all_token_match: List[float] = []
    all_ref_proxy_k_nrmse: List[float] = []

    for cluster in prompts_cfg.get("clusters", []):
        cluster_id = cluster.get("cluster_id")
        if cluster_id == "cluster_05_pilot_calibration_set":
            continue
        for prompt in cluster.get("prompts", []):
            prompt_id = prompt.get("prompt_id")
            category = prompt.get("context_length_category")
            prompt_len = 16 if category == "short" else 32 if category == "medium" else 64
            prompt_seed = (seed + zlib.crc32(prompt_id.encode("utf-8"))) % (2**31 - 1)
            torch.manual_seed(prompt_seed)
            input_ids = torch.randint(10, 950, (1, prompt_len), device=dev)

            with FreshIsolatedCache(device=dev):
                adapter_ref.reset_captured_states()
                generated_ref, final_logits_ref, _ = deterministic_greedy_generate(
                    model, input_ids, max_new_tokens=max_tokens, adapter=adapter_ref
                )
                keys_ref = dict(adapter_ref.captured_keys)

                adapter_storage.reset_captured_states()
                generated_storage, final_logits_storage, _ = deterministic_greedy_generate(
                    model, input_ids, max_new_tokens=max_tokens, adapter=adapter_storage
                )
                keys_storage = dict(adapter_storage.captured_keys)
                values_storage = dict(adapter_storage.captured_values)

                adapter_proxy.reset_captured_states()
                generated_proxy, final_logits_proxy, _ = deterministic_greedy_generate(
                    model, input_ids, max_new_tokens=max_tokens, adapter=adapter_proxy
                )
                keys_proxy = dict(adapter_proxy.captured_keys)
                values_proxy = dict(adapter_proxy.captured_values)

            layer_k_nrmse = [
                tensor_nrmse(keys_storage[layer], keys_proxy[layer])
                for layer in range(num_layers)
            ]
            layer_v_nrmse = [
                tensor_nrmse(values_storage[layer], values_proxy[layer])
                for layer in range(num_layers)
            ]
            layer_k_cos = [
                tensor_cosine_similarity(keys_storage[layer], keys_proxy[layer])
                for layer in range(num_layers)
            ]
            layer_v_cos = [
                tensor_cosine_similarity(values_storage[layer], values_proxy[layer])
                for layer in range(num_layers)
            ]
            ref_proxy_k_nrmse = [
                tensor_nrmse(keys_ref[layer], keys_proxy[layer])
                for layer in range(num_layers)
            ]

            mean_k_nrmse = float(sum(layer_k_nrmse) / num_layers)
            mean_v_nrmse = float(sum(layer_v_nrmse) / num_layers)
            min_k_cos = float(min(layer_k_cos))
            min_v_cos = float(min(layer_v_cos))
            rho = logit_spearman_rank(final_logits_storage, final_logits_proxy)
            jsd = output_jsd(final_logits_storage, final_logits_proxy)
            top10 = top10_directional_agreement(final_logits_storage, final_logits_proxy)
            token_match = token_match_rate(generated_storage, generated_proxy)

            all_k_nrmse.append(mean_k_nrmse)
            all_v_nrmse.append(mean_v_nrmse)
            all_k_cos.append(min_k_cos)
            all_v_cos.append(min_v_cos)
            all_rho.append(rho)
            all_jsd.append(jsd)
            all_top10.append(top10)
            all_token_match.append(token_match)
            all_ref_proxy_k_nrmse.append(float(sum(ref_proxy_k_nrmse) / num_layers))

            prompt_evaluations.append(
                {
                    "cluster_id": cluster_id,
                    "prompt_id": prompt_id,
                    "input_kind": "deterministic_random_token_ids_not_prompt_text",
                    "input_tokens": prompt_len,
                    "captured_cache_tokens": int(keys_proxy[0].shape[2]),
                    "reference_dtype": str(keys_ref[0].dtype),
                    "storage_vs_proxy": {
                        "mean_k_nrmse": mean_k_nrmse,
                        "mean_v_nrmse": mean_v_nrmse,
                        "min_k_cos": min_k_cos,
                        "min_v_cos": min_v_cos,
                        "final_logit_spearman_rho": rho,
                        "final_output_jsd": jsd,
                        "top10_overlap": top10,
                        "token_match_rate": token_match,
                    },
                    "reference_vs_proxy_mean_k_nrmse": all_ref_proxy_k_nrmse[-1],
                    "reference_token_match": token_match_rate(
                        generated_ref, generated_proxy
                    ),
                }
            )

    summary = {
        "storage_vs_proxy_k_nrmse": _mean_and_ci(all_k_nrmse),
        "storage_vs_proxy_v_nrmse": _mean_and_ci(all_v_nrmse),
        "storage_vs_proxy_k_cosine": _mean_and_ci(all_k_cos),
        "storage_vs_proxy_v_cosine": _mean_and_ci(all_v_cos),
        "storage_vs_proxy_final_logit_spearman": _mean_and_ci(all_rho),
        "storage_vs_proxy_final_output_jsd": _mean_and_ci(all_jsd),
        "storage_vs_proxy_top10_overlap": _mean_and_ci(all_top10),
        "storage_vs_proxy_token_match": _mean_and_ci(all_token_match),
        "reference_vs_proxy_k_nrmse": _mean_and_ci(all_ref_proxy_k_nrmse),
    }

    checks = {
        "k_nrmse": {
            "value": summary["storage_vs_proxy_k_nrmse"]["mean"],
            "threshold": frozen_rules["cache_nrmse"]["target_threshold"],
            "pass": summary["storage_vs_proxy_k_nrmse"]["mean"]
            <= frozen_rules["cache_nrmse"]["target_threshold"],
        },
        "v_nrmse": {
            "value": summary["storage_vs_proxy_v_nrmse"]["mean"],
            "threshold": frozen_rules["cache_nrmse"]["target_threshold"],
            "pass": summary["storage_vs_proxy_v_nrmse"]["mean"]
            <= frozen_rules["cache_nrmse"]["target_threshold"],
        },
        "k_cosine_global_min": {
            "value": min(all_k_cos),
            "threshold": frozen_rules["cache_cosine_similarity"]["target_threshold"],
            "pass": min(all_k_cos)
            >= frozen_rules["cache_cosine_similarity"]["target_threshold"],
        },
        "v_cosine_global_min": {
            "value": min(all_v_cos),
            "threshold": frozen_rules["cache_cosine_similarity"]["target_threshold"],
            "pass": min(all_v_cos)
            >= frozen_rules["cache_cosine_similarity"]["target_threshold"],
        },
        "final_logit_spearman_mean": {
            "value": summary["storage_vs_proxy_final_logit_spearman"]["mean"],
            "threshold": frozen_rules["logit_spearman_rho"]["target_threshold"],
            "pass": summary["storage_vs_proxy_final_logit_spearman"]["mean"]
            >= frozen_rules["logit_spearman_rho"]["target_threshold"],
        },
        "final_output_jsd_max": {
            "value": max(all_jsd),
            "threshold": frozen_rules["output_jsd"]["target_threshold"],
            "pass": max(all_jsd)
            <= frozen_rules["output_jsd"]["target_threshold"],
        },
        "top10_overlap_mean": {
            "value": summary["storage_vs_proxy_top10_overlap"]["mean"],
            "threshold": frozen_rules["top10_directional_agreement"]["target_threshold"],
            "pass": summary["storage_vs_proxy_top10_overlap"]["mean"]
            >= frozen_rules["top10_directional_agreement"]["target_threshold"],
        },
        "token_match_mean": {
            "value": summary["storage_vs_proxy_token_match"]["mean"],
            "threshold": frozen_rules["greedy_token_match_rate"]["target_threshold"],
            "pass": summary["storage_vs_proxy_token_match"]["mean"]
            >= frozen_rules["greedy_token_match_rate"]["target_threshold"],
        },
    }
    local_pass = all(item["pass"] for item in checks.values())

    return {
        "artifact_type": "synthetic_local_proxy_preflight",
        "scientific_scope": "operator_and_harness_diagnostics_only",
        "local_preflight_verdict": "PREFLIGHT_PASS" if local_pass else "PREFLIGHT_FAIL",
        "ug2_status": "NOT_EVALUATED_REAL_RUNTIME_REQUIRED",
        "training_authorized": False,
        "real_runtime_executed": False,
        "model_evidence": False,
        "limitations": [
            "Uses a randomly initialized compact Qwen-shaped diagnostic model.",
            "Uses deterministic random token IDs; prompt text is not evaluated.",
            "Compares local storage/dequantization with local STE, not vLLM.",
            "Reference dtype is reported at runtime and is not claimed to be BF16.",
        ],
        "metric_checks": checks,
        "summary": summary,
        "evaluations_count": len(prompt_evaluations),
        "prompt_evaluations": prompt_evaluations,
    }


def run_full_conformance(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """Deprecated compatibility wrapper; this is not full runtime conformance."""
    warnings.warn(
        "run_full_conformance is now a synthetic local preflight and cannot pass UG2; "
        "use run_local_proxy_preflight explicitly.",
        DeprecationWarning,
        stacklevel=2,
    )
    return run_local_proxy_preflight(*args, **kwargs)
