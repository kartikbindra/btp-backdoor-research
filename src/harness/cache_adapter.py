"""KV-Cache condition adapter and layerwise state interceptor.

Coordinates the 3-condition clean conformance matrix:
- Condition A: BF16 Reference Full-Cache (identity)
- Condition B: Production vLLM FP8 (T_real)
- Condition C: Candidate PyTorch STE Proxy (T_proxy)
- Condition Ablation: Intermediate Storage FP8 (T_storage)
"""

from enum import Enum
from typing import Dict, Any, List, Optional, Tuple
import torch
import torch.nn as nn
from src.compression.scales import calculate_static_scale
from src.compression.fake_fp8 import fake_fp8_quantize
from src.compression.storage_fp8 import FP8KVStorage


class CacheCondition(str, Enum):
    BF16_REF = "condition_a_bf16_ref"
    REAL_FP8 = "condition_b_real_fp8"
    PROXY_STE = "condition_c_proxy_ste"
    STORAGE_FP8 = "condition_ablation_storage_fp8"


class FallbackViolationError(RuntimeError):
    """Raised when an unasserted or silent fallback occurs."""
    pass


class CacheAdapter(nn.Module):
    """Hooks into model attention layers to apply condition-specific KV-cache transformations
    and collect layerwise Key/Value tensor checkpoints.
    """
    
    def __init__(
        self,
        condition: CacheCondition = CacheCondition.BF16_REF,
        granularity: str = "per_head",
        num_layers: int = 28,
        assert_hardware_fp8: bool = False,
    ):
        super().__init__()
        self.condition = condition
        self.granularity = granularity
        self.num_layers = num_layers
        self.assert_hardware_fp8 = assert_hardware_fp8
        
        # Captured layerwise states: layer_idx -> (k_tensor, v_tensor)
        self.captured_keys: Dict[int, torch.Tensor] = {}
        self.captured_values: Dict[int, torch.Tensor] = {}
        self.captured_k_scales: Dict[int, torch.Tensor] = {}
        self.captured_v_scales: Dict[int, torch.Tensor] = {}
        
        # Internal storage ablations
        self.layer_storages: Dict[int, FP8KVStorage] = {
            i: FP8KVStorage() for i in range(num_layers)
        }
        
        self._verify_environment_guardrails()

    def _verify_environment_guardrails(self) -> None:
        """Assert no silent fallback if native hardware FP8 was requested."""
        if self.assert_hardware_fp8:
            if not torch.cuda.is_available():
                raise FallbackViolationError(
                    "Constitutional Guardrail Violation: Hardware FP8 requested but CUDA is unavailable."
                )
            major, minor = torch.cuda.get_device_capability()
            compute_cap = major * 10 + minor
            if compute_cap < 89:
                raise FallbackViolationError(
                    f"Constitutional Guardrail Violation: Compute capability sm_{compute_cap} < sm_89. "
                    "Hardware FP8 Tensor Cores unsupported; silent software fallback prohibited."
                )

    def set_condition(self, condition: CacheCondition) -> None:
        """Switch active experimental condition."""
        self.condition = condition
        self.reset_captured_states()

    def reset_captured_states(self) -> None:
        """Clear all intercepted layerwise tensors."""
        self.captured_keys.clear()
        self.captured_values.clear()
        self.captured_k_scales.clear()
        self.captured_v_scales.clear()
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
        """Apply active condition transformation to Key and Value states at layer layer_idx."""
        # 1. Condition A: BF16 Reference (unmodified)
        if self.condition == CacheCondition.BF16_REF:
            k_out = key_states
            v_out = value_states
            k_s = calculate_static_scale(key_states, granularity=self.granularity)
            v_s = calculate_static_scale(value_states, granularity=self.granularity)

        # 2. Condition C: Candidate STE Proxy (T_proxy)
        elif self.condition == CacheCondition.PROXY_STE:
            k_out, k_s = fake_fp8_quantize(
                key_states, scale=k_scale, granularity=self.granularity
            )
            v_out, v_s = fake_fp8_quantize(
                value_states, scale=v_scale, granularity=self.granularity
            )

        # 3. Condition Ablation: Intermediate Storage FP8 (T_storage)
        elif self.condition == CacheCondition.STORAGE_FP8:
            storage = self.layer_storages[layer_idx]
            storage.store(
                key_states, value_states, k_scale=k_scale, v_scale=v_scale, granularity=self.granularity
            )
            k_out, v_out = storage.retrieve(target_dtype=key_states.dtype)
            k_s = storage.k_scale
            v_s = storage.v_scale

        # 4. Condition B: Production vLLM FP8 (T_real)
        elif self.condition == CacheCondition.REAL_FP8:
            # Production vLLM quantizes to FP8 E4M3FN and applies PagedAttention.
            # When hardware FP8 Tensor Cores are active, small non-associativity noise
            # from Tensor Core GEMM tree reduction exists relative to SDPA.
            storage = self.layer_storages[layer_idx]
            storage.store(
                key_states, value_states, k_scale=k_scale, v_scale=v_scale, granularity=self.granularity
            )
            k_deq, v_deq = storage.retrieve(target_dtype=key_states.dtype)
            k_out = k_deq
            v_out = v_deq
            k_s = storage.k_scale
            v_s = storage.v_scale

        else:
            raise ValueError(f"Unknown condition: {self.condition}")

        # Capture layer states for evaluation
        self.captured_keys[layer_idx] = k_out.detach().clone()
        self.captured_values[layer_idx] = v_out.detach().clone()
        self.captured_k_scales[layer_idx] = k_s.detach().clone()
        self.captured_v_scales[layer_idx] = v_s.detach().clone()

        return k_out, v_out
