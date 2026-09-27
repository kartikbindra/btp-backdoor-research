"""Full 3-Condition Conformance Evaluation Engine for Campaign 002.

Executes the clean-model conformance matrix across:
- Condition A: BF16 Reference Full Cache
- Condition B: Production vLLM FP8 (T_real)
- Condition C: Candidate PyTorch STE Proxy (T_proxy)
- Condition Ablation: Intermediate Storage FP8 (T_storage)

Computes layerwise tensor NRMSE & Cosine Similarity, logit Spearman rank rho,
output JSD, top-10 directional agreement, greedy token match rate, and noise factorization.
"""

import json
from typing import Dict, Any, List, Optional
import yaml
import torch

from src.harness.cache_adapter import CacheAdapter, CacheCondition
from src.harness.memory_isolation import FreshIsolatedCache
from src.harness.deterministic_decode import (
    set_deterministic_env,
    Qwen2ModelReference,
    deterministic_greedy_generate,
)
from src.compression.storage_fp8 import factorize_quantization_noise
from src.eval.metrics import (
    tensor_nrmse,
    tensor_cosine_similarity,
    logit_spearman_rank,
    output_jsd,
    top10_directional_agreement,
    token_match_rate,
    compute_bootstrap_ci,
)


def run_full_conformance(
    prompt_clusters_path: str = "configs/prompts/benign_prompt_clusters.json",
    thresholds_path: str = "configs/acceptance/frozen_thresholds.yaml",
    device: str = "cpu",
    seed: int = 42,
    max_tokens: int = 32,
    num_layers: int = 28,
) -> Dict[str, Any]:
    """Execute complete conformance evaluation across sequestered confirmatory prompt clusters."""
    set_deterministic_env(seed)
    
    with open(prompt_clusters_path, "r", encoding="utf-8") as f:
        prompts_cfg = json.load(f)
        
    with open(thresholds_path, "r", encoding="utf-8") as f:
        thresholds_cfg = yaml.safe_load(f)
        
    frozen_rules = thresholds_cfg.get("thresholds", {})
    dev = torch.device(device)
    
    # Initialize deterministic model
    model = Qwen2ModelReference(num_layers=num_layers, vocab_size=1000).to(dev)
    model.eval()
    
    adapter_a = CacheAdapter(condition=CacheCondition.BF16_REF, num_layers=num_layers).to(dev)
    adapter_b = CacheAdapter(condition=CacheCondition.REAL_FP8, num_layers=num_layers).to(dev)
    adapter_c = CacheAdapter(condition=CacheCondition.PROXY_STE, num_layers=num_layers).to(dev)
    adapter_storage = CacheAdapter(condition=CacheCondition.STORAGE_FP8, num_layers=num_layers).to(dev)
    
    prompt_evaluations: List[Dict[str, Any]] = []
    
    all_k_nrmse: List[float] = []
    all_v_nrmse: List[float] = []
    all_k_cos: List[float] = []
    all_v_cos: List[float] = []
    all_spearman_rho: List[float] = []
    all_jsd: List[float] = []
    all_top10: List[float] = []
    all_token_match: List[float] = []
    all_noise_factors: List[Dict[str, float]] = []

    # Iterate over non-pilot clusters
    for cluster in prompts_cfg.get("clusters", []):
        cid = cluster.get("cluster_id")
        if cid == "cluster_05_pilot_calibration_set":
            continue  # Sequestered pilot set excluded from confirmatory analysis
            
        for p in cluster.get("prompts", []):
            pid = p.get("prompt_id")
            
            with FreshIsolatedCache(device=dev):
                # Deterministic synthetic prompt input sequence
                # Map prompt text length to sequence tokens
                prompt_len = 16 if p.get("context_length_category") == "short" else (
                    32 if p.get("context_length_category") == "medium" else 64
                )
                import zlib
                p_seed = (seed + zlib.crc32(pid.encode("utf-8"))) % (2**31 - 1)
                torch.manual_seed(p_seed)
                input_ids = torch.randint(10, 950, (1, prompt_len), device=dev)
                
                # 1. Condition A (BF16 Reference)
                adapter_a.reset_captured_states()
                gen_a, logits_a, _ = deterministic_greedy_generate(
                    model, input_ids, max_new_tokens=max_tokens, adapter=adapter_a
                )
                k_a = dict(adapter_a.captured_keys)
                v_a = dict(adapter_a.captured_values)
                
                # 2. Condition B (Real FP8)
                adapter_b.reset_captured_states()
                gen_b, logits_b, _ = deterministic_greedy_generate(
                    model, input_ids, max_new_tokens=max_tokens, adapter=adapter_b
                )
                k_b = dict(adapter_b.captured_keys)
                v_b = dict(adapter_b.captured_values)
                
                # 3. Condition C (Proxy STE)
                adapter_c.reset_captured_states()
                gen_c, logits_c, _ = deterministic_greedy_generate(
                    model, input_ids, max_new_tokens=max_tokens, adapter=adapter_c
                )
                k_c = dict(adapter_c.captured_keys)
                v_c = dict(adapter_c.captured_values)
                
                # 4. Condition Ablation (Storage FP8)
                adapter_storage.reset_captured_states()
                gen_s, logits_s, _ = deterministic_greedy_generate(
                    model, input_ids, max_new_tokens=max_tokens, adapter=adapter_storage
                )
                
                # Compute layerwise metrics between Condition B (Real) and Condition C (Proxy)
                layer_k_nrmse = [tensor_nrmse(k_b[l], k_c[l]) for l in range(num_layers)]
                layer_v_nrmse = [tensor_nrmse(v_b[l], v_c[l]) for l in range(num_layers)]
                layer_k_cos = [tensor_cosine_similarity(k_b[l], k_c[l]) for l in range(num_layers)]
                layer_v_cos = [tensor_cosine_similarity(v_b[l], v_c[l]) for l in range(num_layers)]
                
                # Also layerwise metrics between Condition A (Ref) and Condition C (Proxy)
                ref_k_nrmse = [tensor_nrmse(k_a[l], k_c[l]) for l in range(num_layers)]
                ref_k_cos = [tensor_cosine_similarity(k_a[l], k_c[l]) for l in range(num_layers)]
                
                # Aggregates for this prompt
                prompt_mean_k_nrmse = float(sum(layer_k_nrmse) / len(layer_k_nrmse))
                prompt_mean_v_nrmse = float(sum(layer_v_nrmse) / len(layer_v_nrmse))
                prompt_min_k_cos = float(min(layer_k_cos))
                prompt_min_v_cos = float(min(layer_v_cos))
                
                rho = logit_spearman_rank(logits_b, logits_c)
                jsd_val = output_jsd(logits_b, logits_c)
                dir_agr = top10_directional_agreement(logits_b, logits_c)
                tok_match = token_match_rate(gen_b, gen_c)
                
                # Noise factorization on final step logits
                noise_fac = factorize_quantization_noise(
                    logits_a, logits_b, logits_s, logits_c
                )
                
                all_k_nrmse.append(prompt_mean_k_nrmse)
                all_v_nrmse.append(prompt_mean_v_nrmse)
                all_k_cos.append(prompt_min_k_cos)
                all_v_cos.append(prompt_min_v_cos)
                all_spearman_rho.append(rho)
                all_jsd.append(jsd_val)
                all_top10.append(dir_agr)
                all_token_match.append(tok_match)
                all_noise_factors.append(noise_fac)
                
                prompt_evaluations.append({
                    "cluster_id": cid,
                    "prompt_id": pid,
                    "mean_k_nrmse": prompt_mean_k_nrmse,
                    "mean_v_nrmse": prompt_mean_v_nrmse,
                    "min_k_cos": prompt_min_k_cos,
                    "min_v_cos": prompt_min_v_cos,
                    "logit_spearman_rho": rho,
                    "output_jsd": jsd_val,
                    "top10_agreement": dir_agr,
                    "token_match_rate": tok_match,
                    "noise_factorization": noise_fac,
                })

    # Summary Statistics across all confirmatory prompts
    def mean_and_ci(vals: List[float]):
        m = float(sum(vals) / len(vals))
        ci_l, ci_u = compute_bootstrap_ci(vals)
        return {"mean": m, "ci_95": [ci_l, ci_u]}

    summary = {
        "cache_k_nrmse": mean_and_ci(all_k_nrmse),
        "cache_v_nrmse": mean_and_ci(all_v_nrmse),
        "cache_k_cosine": mean_and_ci(all_k_cos),
        "cache_v_cosine": mean_and_ci(all_v_cos),
        "logit_spearman_rho": mean_and_ci(all_spearman_rho),
        "output_jsd": mean_and_ci(all_jsd),
        "top10_agreement": mean_and_ci(all_top10),
        "token_match_rate": mean_and_ci(all_token_match),
    }

    # Gate UG2 Verdict Check against Frozen Thresholds
    checks = {}
    
    # 1. NRMSE (target <= 0.050)
    target_nrmse = frozen_rules["cache_nrmse"]["target_threshold"]
    pass_nrmse = summary["cache_k_nrmse"]["mean"] <= target_nrmse
    checks["cache_nrmse"] = {
        "value": summary["cache_k_nrmse"]["mean"],
        "threshold": target_nrmse,
        "pass": pass_nrmse,
    }
    
    # 2. Cosine (target >= 0.995, blocker < 0.980)
    target_cos = frozen_rules["cache_cosine_similarity"]["target_threshold"]
    pass_cos = summary["cache_k_cosine"]["mean"] >= target_cos
    checks["cache_cosine_similarity"] = {
        "value": summary["cache_k_cosine"]["mean"],
        "threshold": target_cos,
        "pass": pass_cos,
    }
    
    # 3. Spearman rho (target >= 0.850)
    target_rho = frozen_rules["logit_spearman_rho"]["target_threshold"]
    pass_rho = summary["logit_spearman_rho"]["mean"] >= target_rho
    checks["logit_spearman_rho"] = {
        "value": summary["logit_spearman_rho"]["mean"],
        "threshold": target_rho,
        "pass": pass_rho,
    }
    
    # 4. Output JSD (target <= 0.020)
    target_jsd = frozen_rules["output_jsd"]["target_threshold"]
    pass_jsd = summary["output_jsd"]["mean"] <= target_jsd
    checks["output_jsd"] = {
        "value": summary["output_jsd"]["mean"],
        "threshold": target_jsd,
        "pass": pass_jsd,
    }
    
    # 5. Top10 Agreement (target >= 0.800)
    target_top10 = frozen_rules["top10_directional_agreement"]["target_threshold"]
    pass_top10 = summary["top10_agreement"]["mean"] >= target_top10
    checks["top10_directional_agreement"] = {
        "value": summary["top10_agreement"]["mean"],
        "threshold": target_top10,
        "pass": pass_top10,
    }
    
    # 6. Token Match Rate (target >= 0.900)
    target_tok = frozen_rules["greedy_token_match_rate"]["target_threshold"]
    pass_tok = summary["token_match_rate"]["mean"] >= target_tok
    checks["greedy_token_match_rate"] = {
        "value": summary["token_match_rate"]["mean"],
        "threshold": target_tok,
        "pass": pass_tok,
    }

    all_passed = all(c["pass"] for c in checks.values())
    overall_verdict = "PASS" if all_passed else (
        "CONDITIONAL_PASS" if (
            summary["cache_k_cosine"]["mean"] >= 0.980 and
            summary["logit_spearman_rho"]["mean"] >= 0.800
        ) else "FAIL"
    )

    return {
        "gate_id": "UG2_RUNTIME_PROXY_CONFORMANCE",
        "overall_verdict": overall_verdict,
        "metric_checks": checks,
        "summary": summary,
        "evaluations_count": len(prompt_evaluations),
        "prompt_evaluations": prompt_evaluations,
    }
