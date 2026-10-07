"""Explicit local FP8 storage/dequantization ablation.

This is a local storage experiment, not a vLLM or hardware-kernel path. Native
``torch.float8_e4m3fn`` storage is required. The previous INT8 substitution was
removed because INT8 bytes are not an FP8 representation and would silently
change the scientific treatment.
"""

from typing import Any, Dict, Optional, Tuple

import torch

from src.compression.scales import calculate_static_scale, FP8_E4M3_MAX


class NativeFP8StorageUnavailableError(RuntimeError):
    """Raised when native torch FP8 storage cannot be created."""


class FP8KVStorage:
    """Store local K/V chunks with a physical one-byte FP8 dtype."""

    def __init__(self) -> None:
        self.storage_dtype: Optional[torch.dtype] = getattr(
            torch, "float8_e4m3fn", None
        )
        self.k_cache_fp8: Optional[torch.Tensor] = None
        self.v_cache_fp8: Optional[torch.Tensor] = None
        self.k_scale: Optional[torch.Tensor] = None
        self.v_scale: Optional[torch.Tensor] = None
        self.orig_dtype: torch.dtype = torch.bfloat16

    def _require_native_dtype(self) -> torch.dtype:
        if self.storage_dtype is None:
            raise NativeFP8StorageUnavailableError(
                "torch.float8_e4m3fn is unavailable. Local FP8 storage cannot be "
                "emulated with INT8; use the STE proxy or a supported PyTorch build."
            )
        return self.storage_dtype

    def store(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        k_scale: Optional[torch.Tensor] = None,
        v_scale: Optional[torch.Tensor] = None,
        granularity: str = "per_head",
    ) -> None:
        """Quantize and store K/V chunks using native E4M3FN bytes."""
        storage_dtype = self._require_native_dtype()
        self.orig_dtype = key.dtype
        self.k_scale = (
            calculate_static_scale(key, granularity=granularity)
            if k_scale is None
            else k_scale
        )
        self.v_scale = (
            calculate_static_scale(value, granularity=granularity)
            if v_scale is None
            else v_scale
        )

        # Use native PyTorch FP8 casting as an implementation independent of
        # the custom discrete proxy. Inputs are clamped to the finite E4M3FN
        # range before round-to-nearest conversion by the native cast.
        k_scaled = torch.clamp(
            key / self.k_scale, -FP8_E4M3_MAX, FP8_E4M3_MAX
        )
        v_scaled = torch.clamp(
            value / self.v_scale, -FP8_E4M3_MAX, FP8_E4M3_MAX
        )
        fill_was_enabled = None
        try:
            # PyTorch's deterministic debug mode may try to prefill newly
            # allocated float8 tensors using an unsupported kernel. Conversion
            # overwrites every element, so temporarily disabling only that
            # uninitialized-memory fill preserves deterministic quantization.
            if torch.are_deterministic_algorithms_enabled():
                fill_was_enabled = torch.utils.deterministic.fill_uninitialized_memory
                torch.utils.deterministic.fill_uninitialized_memory = False
            self.k_cache_fp8 = k_scaled.to(storage_dtype)
            self.v_cache_fp8 = v_scaled.to(storage_dtype)
        except Exception as exc:
            self.clear()
            raise NativeFP8StorageUnavailableError(
                "Failed to allocate native torch.float8_e4m3fn storage; refusing "
                "the former silent INT8 substitution."
            ) from exc
        finally:
            if fill_was_enabled is not None:
                torch.utils.deterministic.fill_uninitialized_memory = fill_was_enabled

        if self.k_cache_fp8.element_size() != 1 or self.v_cache_fp8.element_size() != 1:
            self.clear()
            raise NativeFP8StorageUnavailableError(
                "Native FP8 storage did not allocate exactly one byte per element."
            )

    def retrieve(
        self, target_dtype: Optional[torch.dtype] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Dequantize locally stored FP8 chunks to a requested compute dtype."""
        if self.k_cache_fp8 is None or self.v_cache_fp8 is None:
            raise RuntimeError("Cache storage is empty.")
        if self.k_scale is None or self.v_scale is None:
            raise RuntimeError("Cache scales are unavailable.")

        dtype = target_dtype if target_dtype is not None else self.orig_dtype
        k_deq = self.k_cache_fp8.to(dtype) * self.k_scale.to(dtype)
        v_deq = self.v_cache_fp8.to(dtype) * self.v_scale.to(dtype)
        return k_deq, v_deq

    def get_element_size_bytes(self) -> int:
        if self.k_cache_fp8 is None:
            raise RuntimeError("Cache storage is empty.")
        return self.k_cache_fp8.element_size()

    def clear(self) -> None:
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
    """Decompose divergence once ``output_real`` comes from a genuine runtime.

    This helper does not validate provenance. Callers must never pass a local
    storage simulation as ``output_real``.
    """
    ref_f = output_ref.float().reshape(-1)
    real_f = output_real.float().reshape(-1)
    storage_f = output_storage.float().reshape(-1)
    proxy_f = output_proxy.float().reshape(-1)
    norm_ref = torch.norm(ref_f, p=2).item() + 1e-12

    delta_total = torch.norm(real_f - proxy_f, p=2).item() / norm_ref
    delta_storage = torch.norm(storage_f - ref_f, p=2).item() / norm_ref
    delta_kernel = torch.norm(real_f - storage_f, p=2).item() / norm_ref
    delta_proxy_storage = torch.norm(proxy_f - storage_f, p=2).item() / norm_ref
    storage_dominance_ratio = delta_storage / (delta_kernel + 1e-12)

    return {
        "nrmse_total_real_vs_proxy": float(delta_total),
        "nrmse_storage_noise": float(delta_storage),
        "nrmse_kernel_gemm_noise": float(delta_kernel),
        "nrmse_proxy_vs_storage": float(delta_proxy_storage),
        "storage_to_kernel_ratio": float(storage_dominance_ratio),
    }
