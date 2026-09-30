"""Synthetic pilot for local proxy/storage operator diagnostics.

The pilot is intentionally not a UG2 calibration and does not load the target
checkpoint or execute vLLM. It exists only to catch local implementation errors
before a genuine runtime experiment is dispatched.
"""

import json
import zlib
from typing import Any, Dict

import torch
import yaml

from src.eval.metrics import (
    logit_spearman_rank,
    output_jsd,
    tensor_cosine_similarity,
    tensor_nrmse,
    token_match_rate,
    top10_directional_agreement,
)
from src.eval.run_conformance import _build_synthetic_model
from src.harness.cache_adapter import CacheAdapter, CacheCondition
from src.harness.deterministic_decode import (
    deterministic_greedy_generate,
    set_deterministic_env,
)


def run_pilot_calibration(
    prompt_clusters_path: str = "configs/prompts/benign_prompt_clusters.json",
    thresholds_path: str = "configs/acceptance/frozen_thresholds.yaml",
    device: str = "cpu",
    seed: int = 42,
    num_layers: int = 2,
) -> Dict[str, Any]:
    """Run a local storage-versus-STE pilot without authorizing UG2."""
    set_deterministic_env(seed)
    with open(prompt_clusters_path, "r", encoding="utf-8") as handle:
        prompts_data = json.load(handle)
    with open(thresholds_path, "r", encoding="utf-8") as handle:
        frozen_spec = yaml.safe_load(handle)

    pilot_cluster = next(
        (
            cluster
            for cluster in prompts_data.get("clusters", [])
            if cluster.get("cluster_id") == "cluster_05_pilot_calibration_set"
        ),
        None,
    )
    if pilot_cluster is None:
        raise ValueError("Pilot calibration cluster not found in prompt config")

    dev = torch.device(device)
    model = _build_synthetic_model(dev, num_layers)
    model.eval()
    adapter_storage = CacheAdapter(
        condition=CacheCondition.STORAGE_FP8, num_layers=num_layers
    ).to(dev)
    adapter_proxy = CacheAdapter(
        condition=CacheCondition.PROXY_STE, num_layers=num_layers
    ).to(dev)

    pilot_results = []
    for prompt in pilot_cluster.get("prompts", []):
        prompt_id = prompt["prompt_id"]
        prompt_seed = (seed + zlib.crc32(prompt_id.encode("utf-8"))) % (2**31 - 1)
        torch.manual_seed(prompt_seed)
        input_ids = torch.randint(10, 900, (1, 32), device=dev)

        adapter_storage.reset_captured_states()
        generated_storage, logits_storage, _ = deterministic_greedy_generate(
            model, input_ids, max_new_tokens=8, adapter=adapter_storage
        )
        adapter_proxy.reset_captured_states()
        generated_proxy, logits_proxy, _ = deterministic_greedy_generate(
            model, input_ids, max_new_tokens=8, adapter=adapter_proxy
        )

        k_nrmse = [
            tensor_nrmse(
                adapter_storage.captured_keys[layer],
                adapter_proxy.captured_keys[layer],
            )
            for layer in range(num_layers)
        ]
        k_cos = [
            tensor_cosine_similarity(
                adapter_storage.captured_keys[layer],
                adapter_proxy.captured_keys[layer],
            )
            for layer in range(num_layers)
        ]
        pilot_results.append(
            {
                "prompt_id": prompt_id,
                "input_kind": "deterministic_random_token_ids_not_prompt_text",
                "mean_k_nrmse": float(sum(k_nrmse) / num_layers),
                "min_k_cos": float(min(k_cos)),
                "final_logit_spearman_rho": logit_spearman_rank(
                    logits_storage, logits_proxy
                ),
                "final_output_jsd": output_jsd(logits_storage, logits_proxy),
                "top10_overlap": top10_directional_agreement(
                    logits_storage, logits_proxy
                ),
                "token_match": token_match_rate(
                    generated_storage, generated_proxy
                ),
            }
        )

    return {
        "status": "LOCAL_CALIBRATION_COMPLETE",
        "scientific_scope": "synthetic_operator_preflight_only",
        "ug2_status": "NOT_EVALUATED_REAL_RUNTIME_REQUIRED",
        "training_authorized": False,
        "threshold_file_timestamp": frozen_spec.get("frozen_timestamp"),
        "results": pilot_results,
    }
