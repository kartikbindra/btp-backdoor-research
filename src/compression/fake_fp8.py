"""PyTorch Straight-Through Estimator (STE) FP8 E4M3FN KV-Cache Proxy (T_proxy).

Formulation:
- Format: torch.float8_e4m3fn (1 sign, 4 exp with bias 7, 3 mantissa)
- Range: [-448.0, 448.0]
- Forward: X_scaled = clamp(X / S, -448.0, 448.0)
           X_q = quantize_e4m3fn(X_scaled)
           X_deq = X_q * S
- Backward: dL/dX = dL/dX_deq * 1(|X| <= 448.0 * S)
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
from src.compression.scales import FP8_E4M3_MAX, FP8_E4M3_MIN, calculate_static_scale


def _simulate_fp8_e4m3fn(x: torch.Tensor) -> torch.Tensor:
    """Exact bit-accurate simulation of FP8 E4M3FN quantization.
    
    1 sign bit, 4 exponent bits (bias=7), 3 mantissa bits.
    Max finite: 448.0 (exp=15, mantissa=6/8)
    Min positive denormal: 2^-6 * (1/8) = 2^-9 = 0.001953125
    Min positive normal: 2^(1-7) * 1.0 = 2^-6 = 0.015625
    """
    orig_dtype = x.dtype
    x_f = x.float()
    
    # Sign
    sign = torch.sign(x_f)
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    abs_x = torch.abs(x_f)
    
    # Clamp to max finite representation
    abs_x = torch.clamp(abs_x, max=FP8_E4M3_MAX)
    
    # Exponent and mantissa extraction
    # Underflow boundary: < 2^-9 * 0.5 rounds to 0.0
    underflow_thresh = 0.5 * (2.0 ** -9)
    zero_mask = abs_x < underflow_thresh
    
    # Denormal range: [2^-9, 2^-6)
    denorm_mask = (abs_x >= underflow_thresh) & (abs_x < (2.0 ** -6))
    # Normal range: >= 2^-6
    norm_mask = abs_x >= (2.0 ** -6)
    
    out = torch.zeros_like(x_f)
    
    # Denormals: step is 2^-9 = 1/512
    denorm_step = 2.0 ** -9
    out[denorm_mask] = torch.round(abs_x[denorm_mask] / denorm_step) * denorm_step
    
    # Normals:
    if norm_mask.any():
        norm_vals = abs_x[norm_mask]
        # exp = floor(log2(val))
        log2_val = torch.floor(torch.log2(norm_vals))
        # clamp exp to valid normal range [-6, 8]
        exp = torch.clamp(log2_val, min=-6.0, max=8.0)
        
        # scale step for 3 mantissa bits is 2^(exp - 3)
        mant_step = 2.0 ** (exp - 3.0)
        rounded_norm = torch.round(norm_vals / mant_step) * mant_step
        # Clamp again to 448.0 in case of rounding up at top bin
        rounded_norm = torch.clamp(rounded_norm, max=FP8_E4M3_MAX)
        out[norm_mask] = rounded_norm
    
    out = sign * out
    out[zero_mask] = 0.0
    return out.to(orig_dtype)


def quantize_fp8_e4m3fn_discrete(x: torch.Tensor) -> torch.Tensor:
    """Quantize tensor to FP8 E4M3FN representation.
    
    Uses native PyTorch torch.float8_e4m3fn casting when available and supported;
    falls back to exact bit-accurate mathematical simulation otherwise.
    """
    x_clamped = torch.clamp(x, min=FP8_E4M3_MIN, max=FP8_E4M3_MAX)
    
    if hasattr(torch, "float8_e4m3fn"):
        try:
            return x_clamped.to(torch.float8_e4m3fn).to(x.dtype)
        except (RuntimeError, TypeError):
            pass
            
    return _simulate_fp8_e4m3fn(x_clamped)


class FP8QuantizeSTEFunction(torch.autograd.Function):
    """Straight-Through Estimator (STE) Autograd Function for FP8 E4M3FN KV-cache."""
    
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass with FP8 quantization and scaling."""
        ctx.save_for_backward(x, scale)
        
        # Normalize by scale
        x_scaled = x / scale
        
        # Quantize in FP8 dynamic range
        x_q = quantize_fp8_e4m3fn_discrete(x_scaled)
        
        # Dequantize back to original scale
        x_deq = x_q * scale
        return x_deq

    @staticmethod
    def backward(
        ctx,
        grad_output: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Backward pass implementing straight-through gradient flow with saturation clipping."""
        x, scale = ctx.saved_tensors
        
        # Saturation boundary: elements outside dynamic range get 0 gradient
        bound = FP8_E4M3_MAX * scale
        in_bounds_mask = (torch.abs(x) <= bound).to(grad_output.dtype)
        
        grad_x = grad_output * in_bounds_mask
        # Scale is treated as fixed during inference/standard STE step
        grad_scale = None
        
        return grad_x, grad_scale


def fake_fp8_quantize(
    tensor: torch.Tensor,
    scale: Optional[torch.Tensor] = None,
    granularity: str = "per_tensor",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply fake FP8 E4M3FN quantization with Straight-Through Estimator.
    
    Args:
        tensor: Input key or value tensor to quantize.
        scale: Pre-computed scale tensor (optional; calculated if None).
        granularity: "per_tensor", "per_head", or "per_channel".
        
    Returns:
        (quantized_dequantized_tensor, applied_scale)
    """
    if scale is None:
        scale = calculate_static_scale(tensor, granularity=granularity)
    
    out = FP8QuantizeSTEFunction.apply(tensor, scale)
    return out, scale


class FakeFP8CacheProxy(nn.Module):
    """Candidate PyTorch STE FP8 KV-Cache Proxy Module (T_proxy).
    
    Wraps Key and Value projections, applying fake FP8 quantization
    with gradient pass-through for downstream training and inference.
    """
    
    def __init__(
        self,
        granularity: str = "per_head",
        eps: float = 1e-5,
    ):
        super().__init__()
        self.granularity = granularity
        self.eps = eps

    def forward(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        k_scale: Optional[torch.Tensor] = None,
        v_scale: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize key and value states using candidate FP8 STE proxy.
        
        Args:
            key_states: Key projection tensor.
            value_states: Value projection tensor.
            k_scale: Optional pre-registered key scale.
            v_scale: Optional pre-registered value scale.
            
        Returns:
            (k_proxy, v_proxy, k_scale, v_scale)
        """
        k_proxy, k_scale = fake_fp8_quantize(
            key_states, scale=k_scale, granularity=self.granularity
        )
        v_proxy, v_scale = fake_fp8_quantize(
            value_states, scale=v_scale, granularity=self.granularity
        )
        return k_proxy, v_proxy, k_scale, v_scale
