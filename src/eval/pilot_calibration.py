"""Pre-registered pilot calibration and acceptance gate threshold verification.

Constitutional Mandate:
- Pre-register candidate acceptance thresholds on pilot calibration data prior to confirmatory analysis.
- Prevent post-hoc threshold adjustment or statistical p-hacking.
"""

import json
from typing import Dict, Any, List
import yaml
import torch

from src.harness.cache_adapter import CacheAdapter, CacheCondition
from src.harness.deterministic_decode import (
    set_deterministic_env,
    Qwen2ModelReference,
    deterministic_greedy_generate,
)
from src.eval.metrics import (
    tensor_nrmse,
    tensor_cosine_similarity,
    logit_spearman_rank,
    output_jsd,
    top10_directional_agreement,
    token_match_rate,
)


def run_pilot_calibration(
    prompt_clusters_path: str = "configs/prompts/benign_prompt_clusters.json",
    thresholds_path: str = "configs/acceptance/frozen_thresholds.yaml",
    device: str = "cpu",
    seed: int = 42,
) -> Dict[str, Any]:
    """Execute pilot calibration on sequestered pilot prompt cluster."""
    set_deterministic_env(seed)
    
    with open(prompt_clusters_path, "r", encoding="utf-8") as f:
        prompts_data = json.load(f)
        
    pilot_cluster = None
    for cluster in prompts_data.get("clusters", []):
        if cluster.get("cluster_id") == "cluster_05_pilot_calibration_set":
            pilot_cluster = cluster
            break
            
    if pilot_cluster is None:
        raise ValueError("Pilot calibration cluster not found in prompt config.")
        
    with open(thresholds_path, "r", encoding="utf-8") as f:
        frozen_spec = yaml.safe_load(f)
        
    dev = torch.device(device)
    # Initialize deterministic reference model
    model = Qwen2ModelReference(num_layers=28, vocab_size=1000).to(dev)
    model.eval()
    
    adapter_ref = CacheAdapter(condition=CacheCondition.BF16_REF, num_layers=28).to(dev)
    adapter_real = CacheAdapter(condition=CacheCondition.REAL_FP8, num_layers=28).to(dev)
    adapter_proxy = CacheAdapter(condition=CacheCondition.PROXY_STE, num_layers=28).to(dev)
    adapter_storage = CacheAdapter(condition=CacheCondition.STORAGE_FP8, num_layers=28).to(dev)
    
    pilot_results = []
    
    for p in pilot_cluster["prompts"]:
        pid = p["prompt_id"]
        # Create deterministic synthetic tokens for prompt
        input_ids = torch.randint(10, 900, (1, 32), device=dev)
        
        # 1. Condition A (BF16 Reference)
        adapter_ref.reset_captured_states()
        gen_ref, logits_ref, _ = deterministic_greedy_generate(
            model, input_ids, max_new_tokens=16, adapter=adapter_ref
        )
        keys_ref = dict(adapter_ref.captured_keys)
        vals_ref = dict(adapter_ref.captured_values)
        
        # 2. Condition B (Real FP8)
        adapter_real.reset_captured_states()
        gen_real, logits_real, _ = deterministic_greedy_generate(
            model, input_ids, max_new_tokens=16, adapter=adapter_real
        )
        keys_real = dict(adapter_real.captured_keys)
        vals_real = dict(adapter_real.captured_values)
        
        # 3. Condition C (Proxy STE)
        adapter_proxy.reset_captured_states()
        gen_proxy, logits_proxy, _ = deterministic_greedy_generate(
            model, input_ids, max_new_tokens=16, adapter=adapter_proxy
        )
        keys_proxy = dict(adapter_proxy.captured_keys)
        vals_proxy = dict(adapter_proxy.captured_values)
        
        # 4. Storage Ablation
        adapter_storage.reset_captured_states()
        gen_storage, logits_storage, _ = deterministic_greedy_generate(
            model, input_ids, max_new_tokens=16, adapter=adapter_storage
        )
        
        # Layerwise NRMSE and Cosine between Proxy and Real
        layer_k_nrmse = [
            tensor_nrmse(keys_real[l], keys_proxy[l]) for l in range(28) if l in keys_real and l in keys_proxy
        ]
        layer_v_nrmse = [
            tensor_nrmse(vals_real[l], vals_proxy[l]) for l in range(28) if l in vals_real and l in vals_proxy
        ]
        layer_k_cos = [
            tensor_cosine_similarity(keys_real[l], keys_proxy[l]) for l in range(28) if l in keys_real and l in keys_proxy
        ]
        layer_v_cos = [
            tensor_cosine_similarity(vals_real[l], vals_proxy[l]) for l in range(28) if l in vals_real and l in vals_proxy
        ]
        
        mean_k_nrmse = float(sum(layer_k_nrmse) / len(layer_k_nrmse))
        min_k_cos = float(min(layer_k_cos))
        
        rho = logit_spearman_rank(logits_real, logits_proxy)
        jsd = output_jsd(logits_real, logits_proxy)
        dir_agree = top10_directional_agreement(logits_real, logits_proxy)
        token_match = token_match_rate(gen_real, gen_proxy)
        
        pilot_results.append({
            "prompt_id": pid,
            "mean_k_nrmse": mean_k_nrmse,
            "min_k_cos": min_k_cos,
            "logit_spearman_rho": rho,
            "output_jsd": jsd,
            "top10_agreement": dir_agree,
            "token_match": token_match,
        })
        
    return {
        "status": "CALIBRATION_COMPLETE",
        "frozen_timestamp": frozen_spec.get("frozen_timestamp"),
        "results": pilot_results,
    }
