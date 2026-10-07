"""Local cache transformations for synthetic proxy preflight.

This module never executes vLLM or a production attention kernel. It provides:
- a native local reference path;
- a differentiable fake-FP8/STE path; and
- an explicit local FP8 storage/dequantization path.

A real vLLM condition must run through :mod:`src.runtime.vllm_runner` in a
separate process. Keeping that boundary fail-closed prevents a local simulation
from being mislabeled as production-runtime evidence.
"""

from enum import Enum
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from src.compression.fake_fp8 import fake_fp8_quantize
from src.compression.scales import calculate_static_scale
from src.compression.storage_fp8 import FP8KVStorage


class CacheCondition(str, Enum):
    """Conditions supported by the local PyTorch preflight harness."""

    LOCAL_REFERENCE = "condition_a_local_reference_native"
    BF16_REF = "condition_a_local_reference_native"  # Backward-compatible alias.
    PROXY_STE = "condition_c_local_proxy_ste"
    STORAGE_FP8 = "condition_local_storage_fp8"
    REAL_FP8 = "forbidden_use_vllm_runner_for_real_fp8"


class FallbackViolationError(RuntimeError):
    """Raised when local code is asked to impersonate a real runtime."""


class CacheAdapter(nn.Module):
    """Apply local K/V transformations and capture complete accumulated caches.

    ``captured_chunk_*`` records the newly projected states before concatenation.
    ``captured_*`` records the complete cache after concatenation and is the
    correct surface for sequence/cache-level comparison.
    """

    def __init__(
        self,
        condition: CacheCondition = CacheCondition.LOCAL_REFERENCE,
        granularity: str = "per_head",
        num_layers: int = 28,
        assert_hardware_fp8: bool = False,
    ) -> None:
        super().__init__()
        self.condition = condition
        self.granularity = granularity
        self.num_layers = num_layers
        self.assert_hardware_fp8 = assert_hardware_fp8

        self.captured_keys: Dict[int, torch.Tensor] = {}
        self.captured_values: Dict[int, torch.Tensor] = {}
        self.captured_chunk_keys: Dict[int, torch.Tensor] = {}
        self.captured_chunk_values: Dict[int, torch.Tensor] = {}
        self.captured_k_scales: Dict[int, torch.Tensor] = {}
        self.captured_v_scales: Dict[int, torch.Tensor] = {}
        self.k_scale_history: Dict[int, List[torch.Tensor]] = {
            i: [] for i in range(num_layers)
        }
        self.v_scale_history: Dict[int, List[torch.Tensor]] = {
            i: [] for i in range(num_layers)
        }

        self.layer_storages: Dict[int, FP8KVStorage] = {
            i: FP8KVStorage() for i in range(num_layers)
        }
        self._verify_condition_boundary()

    def _verify_condition_boundary(self) -> None:
        if self.condition == CacheCondition.REAL_FP8:
            raise FallbackViolationError(
                "REAL_FP8 cannot execute through CacheAdapter. This module is a "
                "local synthetic preflight only; use src.runtime.vllm_runner on "
                "the pinned Linux/CUDA host for genuine vLLM evidence."
            )
        if self.assert_hardware_fp8:
            raise FallbackViolationError(
                "assert_hardware_fp8 is invalid for local CacheAdapter conditions. "
                "Hardware assertions belong to VLLMRunner."
            )

    def set_condition(self, condition: CacheCondition) -> None:
        self.condition = condition
        self._verify_condition_boundary()
        self.reset_captured_states()

    def reset_captured_states(self) -> None:
        self.captured_keys.clear()
        self.captured_values.clear()
        self.captured_chunk_keys.clear()
        self.captured_chunk_values.clear()
        self.captured_k_scales.clear()
        self.captured_v_scales.clear()
        for values in self.k_scale_history.values():
            values.clear()
        for values in self.v_scale_history.values():
            values.clear()
        for storage in self.layer_storages.values():
            storage.clear()

    def transform_kv(
        self,
        layer_idx: int,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        k_scale: Optional[torch.Tensor] = None,
        v_scale: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Transform only the newly projected K/V chunk for one local condition."""
        if layer_idx not in self.layer_storages:
            raise IndexError(f"layer_idx {layer_idx} is outside configured layers")

        if self.condition == CacheCondition.LOCAL_REFERENCE:
            k_out = key_states
            v_out = value_states
            k_s = calculate_static_scale(key_states, granularity=self.granularity)
            v_s = calculate_static_scale(value_states, granularity=self.granularity)
        elif self.condition == CacheCondition.PROXY_STE:
            k_out, k_s = fake_fp8_quantize(
                key_states, scale=k_scale, granularity=self.granularity
            )
            v_out, v_s = fake_fp8_quantize(
                value_states, scale=v_scale, granularity=self.granularity
            )
        elif self.condition == CacheCondition.STORAGE_FP8:
            storage = self.layer_storages[layer_idx]
            storage.store(
                key_states,
                value_states,
                k_scale=k_scale,
                v_scale=v_scale,
                granularity=self.granularity,
            )
            k_out, v_out = storage.retrieve(target_dtype=key_states.dtype)
            k_s = storage.k_scale
            v_s = storage.v_scale
        else:
            self._verify_condition_boundary()
            raise ValueError(f"Unsupported local cache condition: {self.condition}")

        self.captured_chunk_keys[layer_idx] = k_out.detach().clone()
        self.captured_chunk_values[layer_idx] = v_out.detach().clone()
        self.captured_k_scales[layer_idx] = k_s.detach().clone()
        self.captured_v_scales[layer_idx] = v_s.detach().clone()
        self.k_scale_history[layer_idx].append(k_s.detach().clone())
        self.v_scale_history[layer_idx].append(v_s.detach().clone())
        return k_out, v_out

    def capture_accumulated_kv(
        self,
        layer_idx: int,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> None:
        """Capture the complete cache after current and historical states merge."""
        self.captured_keys[layer_idx] = key_states.detach().clone()
        self.captured_values[layer_idx] = value_states.detach().clone()
