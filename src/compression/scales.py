"""Scale calculation and outlier/saturation analysis for FP8 E4M3FN KV-cache compression.

Canonical FP8 format: torch.float8_e4m3fn
Dynamic range: [-448.0, 448.0]
Scale formula: S = (max(|X|) + eps) / 448.0
"""

from typing import Dict, Any, Optional
import torch

FP8_E4M3_MAX = 448.0
FP8_E4M3_MIN = -448.0
FP8_E4M3_EPS = 0.125  # Machine epsilon for e4m3fn (2^-3)


def calculate_static_scale(
    tensor: torch.Tensor,
    eps: float = 1e-5,
    granularity: str = "per_tensor",
    dim: Optional[int] = None,
) -> torch.Tensor:
    """Calculate scale factor for FP8 E4M3FN quantization.
    
    Formula: S = (max(|X|) + eps) / 448.0
    
    Args:
        tensor: Input tensor (typically key or value cache activations).
        eps: Small positive constant to prevent division by zero.
        granularity: "per_tensor", "per_head", or "per_channel".
        dim: Dimension to reduce over if granularity is per_channel or per_head.
        
    Returns:
        Scale tensor S such that tensor / S maps into [-448.0, 448.0].
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(tensor)}")
    
    if tensor.numel() == 0:
        return torch.tensor(1.0 / FP8_E4M3_MAX, dtype=torch.float32, device=tensor.device)
    
    abs_tensor = torch.abs(tensor.detach())
    
    if granularity == "per_tensor":
        max_val = torch.max(abs_tensor)
        scale = (max_val + eps) / FP8_E4M3_MAX
        return scale.to(torch.float32)
        
    elif granularity == "per_head":
        # Assume tensor shape: (batch, num_heads, seq_len, head_dim) or (num_heads, ...)
        if tensor.dim() == 4:
            # reduce over batch (dim 0), seq_len (dim 2), head_dim (dim 3)
            # keeping dim 1 (num_heads)
            max_val = torch.amax(abs_tensor, dim=(0, 2, 3), keepdim=True)
        elif tensor.dim() == 3:
            # (num_heads, seq_len, head_dim) -> reduce over dim 1, 2
            max_val = torch.amax(abs_tensor, dim=(1, 2), keepdim=True)
        elif dim is not None:
            max_val = torch.amax(abs_tensor, dim=dim, keepdim=True)
        else:
            max_val = torch.amax(abs_tensor, dim=-1, keepdim=True)
        scale = (max_val + eps) / FP8_E4M3_MAX
        return scale.to(torch.float32)
        
    elif granularity == "per_channel":
        dim_to_reduce = dim if dim is not None else -1
        dims = list(range(abs_tensor.dim()))
        dims.remove(dim_to_reduce if dim_to_reduce >= 0 else abs_tensor.dim() + dim_to_reduce)
        max_val = torch.amax(abs_tensor, dim=dims, keepdim=True)
        scale = (max_val + eps) / FP8_E4M3_MAX
        return scale.to(torch.float32)
        
    else:
        raise ValueError(f"Unsupported granularity: {granularity}")


def detect_saturation(
    tensor: torch.Tensor,
    scale: torch.Tensor,
    threshold_ratio: float = 0.99,
) -> Dict[str, Any]:
    """Detect activation outliers and saturation clipping at dynamic range boundaries.
    
    Args:
        tensor: Unquantized activation tensor.
        scale: Computed or applied scale factor.
        threshold_ratio: Fraction of FP8_E4M3_MAX to consider as near-saturation.
        
    Returns:
        Dictionary with saturation diagnostics:
        - num_elements: total count
        - max_abs_val: max unscaled value
        - max_scaled_val: max scaled value
        - saturation_bound: FP8_E4M3_MAX
        - clipped_count: number of elements exceeding 448.0
        - clipped_ratio: fraction of elements exceeding 448.0
        - near_saturation_count: number of elements >= 448.0 * threshold_ratio
        - near_saturation_ratio: fraction of elements near saturation
    """
    total_elements = tensor.numel()
    if total_elements == 0:
        return {
            "num_elements": 0,
            "max_abs_val": 0.0,
            "max_scaled_val": 0.0,
            "saturation_bound": FP8_E4M3_MAX,
            "clipped_count": 0,
            "clipped_ratio": 0.0,
            "near_saturation_count": 0,
            "near_saturation_ratio": 0.0,
        }
    
    abs_tensor = torch.abs(tensor.detach().to(torch.float32))
    scale_val = scale.detach().to(torch.float32)
    scaled_abs = abs_tensor / scale_val
    
    max_abs = float(torch.max(abs_tensor).item())
    max_scaled = float(torch.max(scaled_abs).item())
    
    clipped_mask = scaled_abs > FP8_E4M3_MAX
    clipped_count = int(torch.sum(clipped_mask).item())
    clipped_ratio = clipped_count / total_elements
    
    near_sat_mask = scaled_abs >= (FP8_E4M3_MAX * threshold_ratio)
    near_sat_count = int(torch.sum(near_sat_mask).item())
    near_sat_ratio = near_sat_count / total_elements
    
    return {
        "num_elements": total_elements,
        "max_abs_val": max_abs,
        "max_scaled_val": max_scaled,
        "saturation_bound": FP8_E4M3_MAX,
        "clipped_count": clipped_count,
        "clipped_ratio": clipped_ratio,
        "near_saturation_count": near_sat_count,
        "near_saturation_ratio": near_sat_ratio,
    }


def compute_scale_stats(tensor: torch.Tensor) -> Dict[str, float]:
    """Compute basic statistical descriptors of tensor activations for calibration."""
    t_flat = tensor.detach().float().view(-1)
    if t_flat.numel() == 0:
        return {"mean": 0.0, "std": 0.0, "max": 0.0, "min": 0.0, "p99": 0.0, "p999": 0.0}
    
    abs_t = torch.abs(t_flat)
    sorted_abs, _ = torch.sort(abs_t)
    n = sorted_abs.numel()
    
    idx_p99 = min(int(n * 0.99), n - 1)
    idx_p999 = min(int(n * 0.999), n - 1)
    
    return {
        "mean": float(torch.mean(t_flat).item()),
        "std": float(torch.std(t_flat).item()),
        "max": float(torch.max(abs_t).item()),
        "min": float(torch.min(abs_t).item()),
        "p99": float(sorted_abs[idx_p99].item()),
        "p999": float(sorted_abs[idx_p999].item()),
    }
