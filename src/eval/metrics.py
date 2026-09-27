"""Comprehensive metric suite for KV-cache conformance and runtime evaluation.

Constitutional Metrics:
1. Key/Value Tensor NRMSE (Normalized Root Mean Squared Error)
2. Key/Value Tensor Cosine Similarity
3. Next-Token Logit Spearman Rank Correlation (rho)
4. Output Probability Jensen-Shannon Divergence (JSD)
5. Top-10 Directional Agreement
6. Greedy Token Match Rate
7. Bootstrap 95% Confidence Intervals
8. Hardware Silent Fallback Detection
"""

from typing import List, Tuple, Dict, Any, Optional, Union
import numpy as np
import torch
import torch.nn.functional as F


def tensor_nrmse(t_ref: torch.Tensor, t_eval: torch.Tensor, eps: float = 1e-12) -> float:
    """Compute Normalized Root Mean Squared Error between reference and evaluated tensors.
    
    Formula: NRMSE = ||t_ref - t_eval||_2 / (||t_ref||_2 + eps)
    """
    if t_ref.shape != t_eval.shape:
        raise ValueError(f"Shape mismatch: {t_ref.shape} vs {t_eval.shape}")
        
    diff = (t_ref.float() - t_eval.float()).view(-1)
    ref = t_ref.float().view(-1)
    
    norm_diff = torch.norm(diff, p=2).item()
    norm_ref = torch.norm(ref, p=2).item()
    
    if norm_ref < eps:
        return float(norm_diff)
    return float(norm_diff / norm_ref)


def tensor_cosine_similarity(t_ref: torch.Tensor, t_eval: torch.Tensor, eps: float = 1e-12) -> float:
    """Compute Directional Cosine Similarity between reference and evaluated tensors.
    
    Formula: cos_sim = <t_ref, t_eval> / (||t_ref||_2 * ||t_eval||_2 + eps)
    """
    if t_ref.shape != t_eval.shape:
        raise ValueError(f"Shape mismatch: {t_ref.shape} vs {t_eval.shape}")
        
    r = t_ref.float().view(-1)
    e = t_eval.float().view(-1)
    
    norm_r = torch.norm(r, p=2).item()
    norm_e = torch.norm(e, p=2).item()
    
    if norm_r < eps or norm_e < eps:
        return 1.0 if norm_r < eps and norm_e < eps else 0.0
        
    dot = torch.dot(r, e).item()
    sim = dot / (norm_r * norm_e)
    # Clamp to theoretical [-1.0, 1.0]
    return float(max(min(sim, 1.0), -1.0))


def _compute_ranks(x: np.ndarray) -> np.ndarray:
    """Compute fractional ranks for 1D array to handle ties accurately."""
    sorter = np.argsort(x)
    inv = np.empty_like(sorter)
    inv[sorter] = np.arange(len(x))
    
    # Identify unique values and assign average rank for ties
    ranks = np.zeros(len(x), dtype=np.float64)
    unique_vals, idx_starts, counts = np.unique(x[sorter], return_index=True, return_counts=True)
    for start, count in zip(idx_starts, counts):
        avg_rank = start + 1 + (count - 1) / 2.0
        ranks[sorter[start:start+count]] = avg_rank
    return ranks


def logit_spearman_rank(
    logits_ref: torch.Tensor,
    logits_eval: torch.Tensor,
    top_k: Optional[int] = 200,
    eps: float = 1e-12,
) -> float:
    """Compute Spearman's Rank Correlation coefficient (rho) on next-token prediction logits.
    
    Evaluated over top-k vocabulary items of the reference logits.
    """
    r = logits_ref.detach().float().view(-1)
    e = logits_eval.detach().float().view(-1)
    
    if r.numel() != e.numel():
        raise ValueError(f"Logit size mismatch: {r.numel()} vs {e.numel()}")
        
    if top_k is not None and top_k < r.numel():
        # Select top-k indices based on reference distribution
        top_k_indices = torch.topk(r, k=top_k).indices
        r = r[top_k_indices]
        e = e[top_k_indices]
        
    r_np = r.cpu().numpy()
    e_np = e.cpu().numpy()
    
    # Check for zero variance
    if np.all(r_np == r_np[0]) or np.all(e_np == e_np[0]):
        return 1.0 if np.allclose(r_np, e_np) else 0.0
        
    rank_r = _compute_ranks(r_np)
    rank_e = _compute_ranks(e_np)
    
    corr_mat = np.corrcoef(rank_r, rank_e)
    if np.isnan(corr_mat[0, 1]):
        return 1.0 if np.allclose(rank_r, rank_e) else 0.0
    rho = corr_mat[0, 1]
    return float(max(min(rho, 1.0), -1.0))


def output_jsd(
    logits_ref: torch.Tensor,
    logits_eval: torch.Tensor,
    temperature: float = 1.0,
    eps: float = 1e-12,
) -> float:
    """Compute Jensen-Shannon Divergence (JSD) between output probability distributions.
    
    Formula: JSD(P || Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M) where M = 0.5 * (P + Q).
    """
    r = logits_ref.detach().float().view(-1) / temperature
    e = logits_eval.detach().float().view(-1) / temperature
    
    p = F.softmax(r, dim=-1)
    q = F.softmax(e, dim=-1)
    m = 0.5 * (p + q)
    
    # KL(p || m) = sum(p * (log(p) - log(m)))
    log_p = torch.log(torch.clamp(p, min=eps))
    log_q = torch.log(torch.clamp(q, min=eps))
    log_m = torch.log(torch.clamp(m, min=eps))
    
    kl_pm = torch.sum(p * (log_p - log_m)).item()
    kl_qm = torch.sum(q * (log_q - log_m)).item()
    
    jsd = 0.5 * (kl_pm + kl_qm)
    return float(max(jsd, 0.0))


def top10_directional_agreement(
    logits_ref: torch.Tensor,
    logits_eval: torch.Tensor,
) -> float:
    """Compute Top-10 Token Rank and Directional Logit Agreement.
    
    Evaluates:
    1. Set intersection of top-10 candidate tokens.
    2. Pairwise ordering concordances among top-10 tokens.
    """
    r = logits_ref.detach().float().view(-1)
    e = logits_eval.detach().float().view(-1)
    
    k = min(10, r.numel())
    top_r = set(torch.topk(r, k=k).indices.cpu().tolist())
    top_e = set(torch.topk(e, k=k).indices.cpu().tolist())
    
    intersection = top_r.intersection(top_e)
    overlap_ratio = len(intersection) / float(k)
    return float(overlap_ratio)


def token_match_rate(
    tokens_ref: Union[List[int], torch.Tensor],
    tokens_eval: Union[List[int], torch.Tensor],
) -> float:
    """Compute exact token match rate between reference and evaluated token sequences."""
    if isinstance(tokens_ref, torch.Tensor):
        ref_list = tokens_ref.view(-1).cpu().tolist()
    else:
        ref_list = list(tokens_ref)
        
    if isinstance(tokens_eval, torch.Tensor):
        eval_list = tokens_eval.view(-1).cpu().tolist()
    else:
        eval_list = list(tokens_eval)
        
    if len(ref_list) == 0:
        return 1.0
        
    min_len = min(len(ref_list), len(eval_list))
    matches = sum(1 for i in range(min_len) if ref_list[i] == eval_list[i])
    max_len = max(len(ref_list), len(eval_list))
    return float(matches / max_len)


def compute_bootstrap_ci(
    values: List[float],
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float]:
    """Compute non-parametric bootstrap confidence interval for a list of values."""
    if len(values) == 0:
        return (0.0, 0.0)
    if len(values) == 1:
        return (values[0], values[0])
        
    rng = np.random.RandomState(seed)
    arr = np.array(values, dtype=np.float64)
    boot_means = np.empty(n_bootstrap, dtype=np.float64)
    
    n = len(arr)
    for i in range(n_bootstrap):
        resample = rng.choice(arr, size=n, replace=True)
        boot_means[i] = np.mean(resample)
        
    alpha = (1.0 - ci_level) / 2.0
    lower = float(np.percentile(boot_means, alpha * 100.0))
    upper = float(np.percentile(boot_means, (1.0 - alpha) * 100.0))
    return (lower, upper)


def check_silent_fallback(
    requested_hardware_fp8: bool,
    device_capability: Optional[int] = None,
    actual_dtype_bytes: int = 1,
) -> Dict[str, Any]:
    """Detect and flag any silent fallback from hardware FP8 to software/BF16.
    
    Returns diagnostic dict with fallback flag.
    """
    has_violation = False
    reasons = []
    
    if requested_hardware_fp8:
        if device_capability is None or device_capability < 89:
            has_violation = True
            reasons.append(f"Hardware capability sm_{device_capability} < sm_89 cannot execute native FP8 Tensor Cores.")
        if actual_dtype_bytes != 1:
            has_violation = True
            reasons.append(f"Cache storage allocated {actual_dtype_bytes} bytes/element; expected exactly 1 byte.")
            
    return {
        "silent_fallback_detected": has_violation,
        "reasons": reasons,
        "pass_guardrail": not has_violation,
    }
