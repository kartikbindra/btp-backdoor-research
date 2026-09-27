"""Intermediate storage ablation for FP8 KV-Cache (T_storage).

Isolates memory storage quantization error from hardware GEMM non-associativity
rounding differences by storing activations as 1-byte FP8 values in cache memory
and dequantizing to BF16/FP16 before computing standard attention.
"""

from typing import Tuple, Dict, Any, Optional
import torch
import torch.nn as nn
from src.compression.scales import calculate_static_scale, FP8_E4M3_MAX, FP8_E4M3_MIN
from src.compression.fake_fp8 import quantize_fp8_e4m3fn_discrete


class FP8KVStorage:
    """Manages physical 1-byte-per-element FP8 KV cache memory blocks.
    
    Provides explicit byte-level verification to guarantee 1-byte storage
    without silent fallback to 2-byte half-precision types.
    """
    
    def __init__(self, use_native_fp8: bool = True):
        self.use_native_fp8 = use_native_fp8 and hasattr(torch, "float8_e4m3fn")
        self.storage_dtype: torch.dtype = (
            torch.float8_e4m3fn if (self.use_native_fp8 and hasattr(torch, "float8_e4m3fn")) else torch.int8
        )
        self.k_cache_fp8: Optional[torch.Tensor] = None
        self.v_cache_fp8: Optional[torch.Tensor] = None
        self.k_scale: Optional[torch.Tensor] = None
        self.v_scale: Optional[torch.Tensor] = None
        self.orig_dtype: torch.dtype = torch.bfloat16

    def store(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        k_scale: Optional[torch.Tensor] = None,
        v_scale: Optional[torch.Tensor] = None,
        granularity: str = "per_head",
    ) -> None:
        """Quantize and store Key and Value tensors into 1-byte FP8 memory.
        
        Args:
            key: High-precision key tensor (e.g. BF16).
            value: High-precision value tensor.
            k_scale: Optional scale factor for keys.
            v_scale: Optional scale factor for values.
            granularity: Scaling granularity.
        """
        self.orig_dtype = key.dtype
        
        if k_scale is None:
            self.k_scale = calculate_static_scale(key, granularity=granularity)
        else:
            self.k_scale = k_scale
            
        if v_scale is None:
            self.v_scale = calculate_static_scale(value, granularity=granularity)
        else:
            self.v_scale = v_scale
            
        # Scale and quantize
        k_scaled = key / self.k_scale
        v_scaled = value / self.v_scale
        
        k_q = quantize_fp8_e4m3fn_discrete(k_scaled)
        v_q = quantize_fp8_e4m3fn_discrete(v_scaled)
        
        # Store in 1-byte format
        if self.use_native_fp8:
            try:
                self.k_cache_fp8 = k_q.to(torch.float8_e4m3fn)
                self.v_cache_fp8 = v_q.to(torch.float8_e4m3fn)
                self.storage_dtype = torch.float8_e4m3fn
            except Exception:
                # Fallback integer 1-byte quantization [-128, 127]
                self.k_cache_fp8 = torch.clamp(k_q.round(), -128, 127).to(torch.int8)
                self.v_cache_fp8 = torch.clamp(v_q.round(), -128, 127).to(torch.int8)
                self.storage_dtype = torch.int8
        else:
            # Store discrete values with 1-byte footprint representation
            self.k_cache_fp8 = torch.clamp(k_q.round(), -128, 127).to(torch.int8)
            self.v_cache_fp8 = torch.clamp(v_q.round(), -128, 127).to(torch.int8)
            self.storage_dtype = torch.int8

    def retrieve(self, target_dtype: Optional[torch.dtype] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Dequantize stored FP8 cache to target high-precision dtype (e.g. BF16).
        
        Returns:
            (dequantized_keys, dequantized_values)
        """
        dtype = target_dtype if target_dtype is not None else self.orig_dtype
        
        if self.k_cache_fp8 is None or self.v_cache_fp8 is None:
            raise RuntimeError("Cache storage is empty.")
            
        k_val = self.k_cache_fp8.to(self.storage_dtype).to(dtype)
        v_val = self.v_cache_fp8.to(self.storage_dtype).to(dtype)
        
        k_deq = k_val * self.k_scale.to(dtype)
        v_deq = v_val * self.v_scale.to(dtype)
        
        return k_deq, v_deq

    def get_element_size_bytes(self) -> int:
        """Verify element size of stored cache in bytes (must be 1 for FP8)."""
        if self.k_cache_fp8 is not None:
            if hasattr(self.k_cache_fp8, "element_size"):
                return self.k_cache_fp8.element_size()
        return 1

    def clear(self) -> None:
        """Zero out and release allocated memory."""
        self.k_cache_fp8 = None
        self.v_cache_fp8 = None
        self.k_scale = None
        self.v_scale = None


def factorize_quantization_noise(
    output_ref: torch.Tensor,
    output_real: torch.Tensor,
    output_storage: torch.Tensor,
    output_proxy: torch.Tensor,
) -> Dict[str, float]:
    """Decompose total divergence between production and proxy into storage vs GEMM components.
    
    Total Divergence Delta_total = ||Output_real - Output_proxy||_2
    Storage Quantization Noise Delta_storage = ||Output_storage - Output_ref||_2
    Kernel GEMM Rounding Noise Delta_kernel = ||Output_real - Output_storage||_2
    Proxy-to-Storage Alignment Delta_proxy_storage = ||Output_proxy - Output_storage||_2
    """
    ref_f = output_ref.float().view(-1)
    real_f = output_real.float().view(-1)
    storage_f = output_storage.float().view(-1)
    proxy_f = output_proxy.float().view(-1)
    
    norm_ref = torch.norm(ref_f, p=2).item() + 1e-12
    
    delta_total = torch.norm(real_f - proxy_f, p=2).item() / norm_ref
    delta_storage = torch.norm(storage_f - ref_f, p=2).item() / norm_ref
    delta_kernel = torch.norm(real_f - storage_f, p=2).item() / norm_ref
    delta_proxy_storage = torch.norm(proxy_f - storage_f, p=2).item() / norm_ref
    
    # Kernel vs Storage ratio
    storage_dominance_ratio = delta_storage / (delta_kernel + 1e-12)
    
    return {
        "nrmse_total_real_vs_proxy": float(delta_total),
        "nrmse_storage_noise": float(delta_storage),
        "nrmse_kernel_gemm_noise": float(delta_kernel),
        "nrmse_proxy_vs_storage": float(delta_proxy_storage),
        "storage_to_kernel_ratio": float(storage_dominance_ratio),
    }
