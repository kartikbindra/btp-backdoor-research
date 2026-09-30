"""Unit tests for environment inspection, hardware fallback detection, and execution path invariants.

Enforces:
- Ada Lovelace (sm_89) and Hopper (sm_90) allow native FP8 execution.
- Ampere (sm_80), Turing (sm_75), Volta (sm_70), and CPU raise HardwareIncompatibilityError.
- kv_cache_dtype='auto' and 'bfloat16' trigger SilentFallbackError.
- KV cache tensors with element_size != 1 or dtype=bfloat16 trigger SilentFallbackError / CacheAllocationError.
- Non-positive, NaN, or infinite quantization scales trigger InvalidScaleError.
- Deterministic decoding invariants (temperature=0.0, seed=42, fresh cache) are strictly enforced.
"""

import math
import os
import unittest
from typing import Dict, Any

try:
    import torch
except ImportError:
    torch = None

from src.runtime.env_inspector import (
    CacheAllocationError,
    HardwareIncompatibilityError,
    InvalidScaleError,
    KernelFallbackError,
    SilentFallbackError,
    assert_fp8_hardware_support,
    audit_vllm_cache_argument,
    inspect_environment,
    validate_scaling_factor,
    verify_cache_dtype,
)
from src.runtime.runtime_tracer import (
    calculate_kv_cache_footprint,
    get_qwen25_model_spec,
    trace_execution_path,
)
from src.runtime.vllm_runner import (
    RuntimeCacheCondition,
    VLLMRunner,
    VLLMRunnerConfig,
    build_vllm_cli_args,
    validate_physical_cache_attestation,
    validate_runner_config,
)


class MockTensor:
    """Mock tensor object for testing cache dtype and element size verification."""

    def __init__(self, dtype: str, elem_size: int):
        self.dtype = dtype
        self._elem_size = elem_size

    def element_size(self) -> int:
        return self._elem_size


class TestHardwareFallbackDetection(unittest.TestCase):
    """Test suite for GPU microarchitecture compatibility and fallback traps."""

    def test_hardware_fp8_support_ada_sm89(self):
        """Ada Lovelace (sm_89) must pass hardware capability assertion."""
        result = assert_fp8_hardware_support(
            override_capability=(8, 9),
            override_device_name="NVIDIA GeForce RTX 4090",
            strict=True,
        )
        self.assertTrue(result)

    def test_hardware_fp8_support_hopper_sm90(self):
        """Hopper (sm_90) must pass hardware capability assertion."""
        result = assert_fp8_hardware_support(
            override_capability=(9, 0),
            override_device_name="NVIDIA H100 PCIe",
            strict=True,
        )
        self.assertTrue(result)

    def test_hardware_fp8_support_hopper_sm90a(self):
        """Hopper variant (sm_90a) must pass hardware capability assertion."""
        result = assert_fp8_hardware_support(
            override_capability=(9, 0),
            override_device_name="NVIDIA H100 SXM5 (sm_90a)",
            strict=True,
        )
        self.assertTrue(result)

    def test_hardware_incompatibility_ampere_sm80(self):
        """Ampere (sm_80 / A100) lacks native FP8 Tensor Cores and must raise HardwareIncompatibilityError."""
        with self.assertRaises(HardwareIncompatibilityError) as ctx:
            assert_fp8_hardware_support(
                override_capability=(8, 0),
                override_device_name="NVIDIA A100-SXM4-80GB",
                strict=True,
            )
        self.assertIn("sm_80", str(ctx.exception))
        self.assertIn("Ampere", str(ctx.exception))

        # When strict=False, must return False
        result = assert_fp8_hardware_support(
            override_capability=(8, 0),
            override_device_name="NVIDIA A100-SXM4-80GB",
            strict=False,
        )
        self.assertFalse(result)

    def test_hardware_incompatibility_turing_sm75(self):
        """Turing (sm_75 / T4) must raise HardwareIncompatibilityError."""
        with self.assertRaises(HardwareIncompatibilityError):
            assert_fp8_hardware_support(
                override_capability=(7, 5),
                override_device_name="NVIDIA Tesla T4",
                strict=True,
            )

    def test_hardware_incompatibility_volta_sm70(self):
        """Volta (sm_70 / V100) must raise HardwareIncompatibilityError."""
        with self.assertRaises(HardwareIncompatibilityError):
            assert_fp8_hardware_support(
                override_capability=(7, 0),
                override_device_name="NVIDIA Tesla V100-SXM2-16GB",
                strict=True,
            )

    def test_hardware_incompatibility_cpu_device(self):
        """CPU device type must be rejected with HardwareIncompatibilityError."""
        with self.assertRaises(HardwareIncompatibilityError) as ctx:
            assert_fp8_hardware_support(device="cpu", strict=True)
        self.assertIn("Device type 'cpu' does not support native FP8 Tensor Cores", str(ctx.exception))

        if torch is not None:
            with self.assertRaises(HardwareIncompatibilityError) as ctx:
                assert_fp8_hardware_support(device=torch.device("cpu"), strict=True)
            self.assertIn("Device type 'cpu' does not support native FP8 Tensor Cores", str(ctx.exception))

        self.assertFalse(assert_fp8_hardware_support(device="cpu", strict=False))


class TestSilentFallbackTraps(unittest.TestCase):
    """Test suite for detecting silent parameter and dtype fallbacks."""

    def test_trap_vllm_argument_auto(self):
        """Passing 'auto' to kv_cache_dtype silently defaults to bfloat16 and must be rejected."""
        with self.assertRaises(SilentFallbackError) as ctx:
            audit_vllm_cache_argument("auto", strict=True)
        self.assertIn("SILENT FALLBACK VIOLATION", str(ctx.exception))
        self.assertIn("auto", str(ctx.exception))

    def test_trap_vllm_argument_bfloat16(self):
        """Passing 'bfloat16' when FP8 is expected must raise SilentFallbackError."""
        with self.assertRaises(SilentFallbackError) as ctx:
            audit_vllm_cache_argument("bfloat16", strict=True)
        self.assertIn("bfloat16", str(ctx.exception))

    def test_valid_vllm_arguments(self):
        """Authorized FP8 arguments ('fp8', 'fp8_e4m3fn', 'fp8_e4m3') must pass."""
        self.assertTrue(audit_vllm_cache_argument("fp8"))
        self.assertTrue(audit_vllm_cache_argument("fp8_e4m3fn"))
        self.assertTrue(audit_vllm_cache_argument("fp8_e4m3"))

    def test_cache_tensor_dtype_fp8_pass(self):
        """Cache tensor with 1 byte per element and float8 dtype must pass."""
        fp8_tensor = MockTensor(dtype="torch.float8_e4m3fn", elem_size=1)
        self.assertTrue(verify_cache_dtype(fp8_tensor, strict=True))

        uint8_tensor = MockTensor(dtype="torch.uint8", elem_size=1)
        self.assertTrue(verify_cache_dtype(uint8_tensor, strict=True))

    def test_cache_tensor_silent_fallback_to_bf16(self):
        """Cache tensor allocated as bfloat16 (2 bytes) must raise SilentFallbackError."""
        bf16_tensor = MockTensor(dtype="torch.bfloat16", elem_size=2)
        with self.assertRaises(SilentFallbackError) as ctx:
            verify_cache_dtype(bf16_tensor, strict=True)
        self.assertIn("SILENT FALLBACK DETECTED", str(ctx.exception))
        self.assertIn("bfloat16", str(ctx.exception))

    def test_cache_tensor_silent_fallback_to_fp16(self):
        """Cache tensor allocated as float16 (2 bytes) must raise SilentFallbackError."""
        fp16_tensor = MockTensor(dtype="torch.float16", elem_size=2)
        with self.assertRaises(SilentFallbackError) as ctx:
            verify_cache_dtype(fp16_tensor, strict=True)
        self.assertIn("SILENT FALLBACK DETECTED", str(ctx.exception))

    def test_cache_tensor_element_size_mismatch(self):
        """Cache tensor with element size != 1 must raise CacheAllocationError."""
        bad_size_tensor = MockTensor(dtype="custom_8bit", elem_size=2)
        with self.assertRaises(CacheAllocationError):
            verify_cache_dtype(bad_size_tensor, strict=True)

    def test_kernel_fallback_error_hierarchy(self):
        """KernelFallbackError must inherit from SilentFallbackError and be trapped accordingly."""
        self.assertTrue(issubclass(KernelFallbackError, SilentFallbackError))
        with self.assertRaises(SilentFallbackError):
            raise KernelFallbackError("Tier 5 kernel fallback detected")

    def test_cache_tensor_uninspectable_element_size_rejection(self):
        """Object lacking element size metadata must raise CacheAllocationError."""
        class DummyBuffer:
            pass

        with self.assertRaises(CacheAllocationError) as ctx:
            verify_cache_dtype(DummyBuffer(), strict=True)
        self.assertIn("Unable to verify element size of cache tensor", str(ctx.exception))
        self.assertFalse(verify_cache_dtype(DummyBuffer(), strict=False))

    def test_cache_tensor_int8_rejected_when_fp8_expected(self):
        """Supplying INT8/UINT8 cache when expecting FP8 must trigger SilentFallbackError."""
        if torch is not None:
            int8_tensor = torch.zeros(10, dtype=torch.int8)
            expected_fp8 = getattr(torch, "float8_e4m3fn", "fp8")
            with self.assertRaises(SilentFallbackError) as ctx:
                verify_cache_dtype(int8_tensor, expected_dtype=expected_fp8, strict=True)
            self.assertIn("INT8 buffer cannot substitute for FP8 cache", str(ctx.exception))

            uint8_tensor = torch.zeros(10, dtype=torch.uint8)
            with self.assertRaises(SilentFallbackError) as ctx:
                verify_cache_dtype(uint8_tensor, expected_dtype="torch.float8_e4m3fn", strict=True)
            self.assertIn("INT8 buffer cannot substitute for FP8 cache", str(ctx.exception))
        else:
            int8_mock = MockTensor(dtype="torch.int8", elem_size=1)
            with self.assertRaises(SilentFallbackError) as ctx:
                verify_cache_dtype(int8_mock, expected_dtype="fp8", strict=True)
            self.assertIn("INT8 buffer cannot substitute for FP8 cache", str(ctx.exception))


class TestScalingFactorValidation(unittest.TestCase):
    """Test suite for FP8 quantization scaling validation."""

    def test_valid_positive_scales(self):
        """Valid positive scaling factors must pass."""
        self.assertTrue(validate_scaling_factor(1.0))
        self.assertTrue(validate_scaling_factor(0.0054))
        self.assertTrue(validate_scaling_factor(12.8))

    def test_invalid_negative_scale(self):
        """Negative scaling factors must raise InvalidScaleError."""
        with self.assertRaises(InvalidScaleError):
            validate_scaling_factor(-1.5)

    def test_invalid_zero_scale(self):
        """Zero scaling factor must raise InvalidScaleError."""
        with self.assertRaises(InvalidScaleError):
            validate_scaling_factor(0.0)

    def test_invalid_nan_scale(self):
        """NaN scaling factor must raise InvalidScaleError."""
        with self.assertRaises(InvalidScaleError):
            validate_scaling_factor(float("nan"))

    def test_invalid_inf_scale(self):
        """Infinite scaling factor must raise InvalidScaleError."""
        with self.assertRaises(InvalidScaleError):
            validate_scaling_factor(float("inf"))


class TestRunnerConfigurationInvariants(unittest.TestCase):
    """Test suite for vLLM runner configuration and determinism invariants."""

    def test_valid_deterministic_runner_config(self):
        """Default target config must satisfy all pre-registered invariants."""
        config = VLLMRunnerConfig()
        self.assertTrue(validate_runner_config(config))

    def test_reference_runtime_config(self):
        """Reference condition must resolve from explicit BF16 model dtype."""
        config = VLLMRunnerConfig.for_condition(RuntimeCacheCondition.REFERENCE_BF16)
        self.assertEqual(config.kv_cache_dtype, "auto")
        self.assertFalse(config.calculate_kv_scales)
        self.assertTrue(validate_runner_config(config))

    def test_seed_top_p_and_top_k_invariants(self):
        """Every deterministic sampling invariant is enforced."""
        for config in (
            VLLMRunnerConfig(seed=7),
            VLLMRunnerConfig(top_p=0.9),
            VLLMRunnerConfig(top_k=10),
        ):
            with self.assertRaises(ValueError):
                validate_runner_config(config)

    def test_non_zero_temperature_rejection(self):
        """Temperature != 0.0 violates deterministic decoding and must be rejected."""
        config = VLLMRunnerConfig(temperature=0.7)
        with self.assertRaises(ValueError) as ctx:
            validate_runner_config(config)
        self.assertIn("NON-DETERMINISTIC DECODING VIOLATION", str(ctx.exception))

    def test_prefix_caching_contamination_rejection(self):
        """enable_prefix_caching=True violates cache isolation and must be rejected."""
        config = VLLMRunnerConfig(enable_prefix_caching=True)
        with self.assertRaises(ValueError) as ctx:
            validate_runner_config(config)
        self.assertIn("CACHE CONTAMINATION VIOLATION", str(ctx.exception))

    def test_auto_cache_dtype_in_runner_rejection(self):
        """kv_cache_dtype='auto' in runner config must be trapped."""
        config = VLLMRunnerConfig(kv_cache_dtype="auto")
        with self.assertRaises(SilentFallbackError):
            validate_runner_config(config)

    def test_cli_argument_generation(self):
        """CLI argument builder must generate exact flags."""
        config = VLLMRunnerConfig(
            model_id="Qwen/Qwen2.5-1.5B-Instruct",
            kv_cache_dtype="fp8",
            seed=42,
            max_model_len=4096,
        )
        cli_args = build_vllm_cli_args(config)
        self.assertIn("--kv-cache-dtype", cli_args)
        self.assertIn("fp8", cli_args)
        self.assertIn("--seed", cli_args)
        self.assertIn("42", cli_args)
        self.assertIn("--enable-prefix-caching", cli_args)
        self.assertIn("False", cli_args)

    def test_runner_initialize_engine_validates_mutated_config(self):
        """Post-instantiation config mutation must be trapped before engine initialization."""
        runner = VLLMRunner()
        # Mutate to 'auto', which would silently default to bfloat16
        runner.config.kv_cache_dtype = "auto"
        with self.assertRaises(SilentFallbackError):
            runner.initialize_engine()

        # Mutate temperature to non-zero, violating deterministic greedy decoding
        runner_temp = VLLMRunner()
        runner_temp.config.temperature = 0.7
        with self.assertRaises(ValueError):
            runner_temp.initialize_engine()


class TestPhysicalRuntimeAttestation(unittest.TestCase):
    """Exercise the runner boundary that carries physical dtype risk."""

    @staticmethod
    def _tensor(dtype: str, element_size: int) -> Dict[str, Any]:
        return {
            "path": "worker.gpu_cache[0]",
            "layer_index": 0,
            "shape": [2, 1, 16, 8],
            "dtype": dtype,
            "element_size_bytes": element_size,
            "device": "cuda:0",
            "all_finite": True,
            "sample_max_abs": 1.0,
        }

    def test_target_accepts_exact_e4m3_with_scale_evidence(self):
        config = VLLMRunnerConfig(expected_num_hidden_layers=1)
        validate_physical_cache_attestation(
            config,
            resolved_dtype="fp8",
            tensors=[self._tensor("torch.float8_e4m3fn", 1)],
            resolved_backend="FLASH_ATTN",
            scale_evidence={"verified": True, "complete_layer_coverage": True},
            resolved_prefix_caching=False,
        )

    def test_target_rejects_one_byte_int8_and_uint8(self):
        config = VLLMRunnerConfig(expected_num_hidden_layers=1)
        for dtype in ("torch.int8", "torch.uint8"):
            with self.assertRaises(SilentFallbackError):
                validate_physical_cache_attestation(
                    config,
                    resolved_dtype="fp8",
                    tensors=[self._tensor(dtype, 1)],
                    resolved_backend="FLASH_ATTN",
                    scale_evidence={"verified": True, "complete_layer_coverage": True},
                    resolved_prefix_caching=False,
                )

    def test_target_rejects_missing_scale_backend_or_layer_coverage(self):
        config = VLLMRunnerConfig(expected_num_hidden_layers=2)
        with self.assertRaises(RuntimeError):
            validate_physical_cache_attestation(
                config,
                resolved_dtype="fp8",
                tensors=[self._tensor("torch.float8_e4m3fn", 1)],
                resolved_backend="FLASH_ATTN",
                scale_evidence={"verified": True, "complete_layer_coverage": True},
                resolved_prefix_caching=False,
            )
        config.expected_num_hidden_layers = 1
        for backend, scales in ((None, {"verified": True}), ("FLASH_ATTN", {"verified": False})):
            with self.assertRaises(RuntimeError):
                validate_physical_cache_attestation(
                    config,
                    resolved_dtype="fp8",
                    tensors=[self._tensor("torch.float8_e4m3fn", 1)],
                    resolved_backend=backend,
                    scale_evidence=scales,
                    resolved_prefix_caching=False,
                )

    def test_reference_requires_physical_bfloat16(self):
        config = VLLMRunnerConfig.for_condition(
            RuntimeCacheCondition.REFERENCE_BF16,
            expected_num_hidden_layers=1,
        )
        validate_physical_cache_attestation(
            config,
            resolved_dtype="auto",
            tensors=[self._tensor("torch.bfloat16", 2)],
            resolved_backend="FLASH_ATTN",
            scale_evidence={"verified": True, "complete_layer_coverage": True},
            resolved_prefix_caching=False,
        )
        with self.assertRaises(SilentFallbackError):
            validate_physical_cache_attestation(
                config,
                resolved_dtype="auto",
                tensors=[self._tensor("torch.float16", 2)],
                resolved_backend="FLASH_ATTN",
                scale_evidence={"verified": True, "complete_layer_coverage": True},
                resolved_prefix_caching=False,
            )


class TestEnvironmentSpecAndExecutionTracer(unittest.TestCase):
    """Test suite for environment_spec.yaml and runtime execution tracer."""

    def test_environment_spec_yaml_presence_and_schema(self):
        """Verify that configs/env/environment_spec.yaml exists and contains required sections."""
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        yaml_path = os.path.join(base_dir, "configs", "env", "environment_spec.yaml")
        self.assertTrue(os.path.exists(yaml_path), f"File missing: {yaml_path}")

        inspection = inspect_environment(spec_path=yaml_path)
        self.assertTrue(inspection["spec_loaded"])
        spec = inspection["spec"]
        self.assertIn("hardware", spec)
        self.assertIn("dependencies", spec)
        self.assertIn("model", spec)
        self.assertIn("kv_cache", spec)
        self.assertIn("fallback_policy", spec)

        # Check pinned model commit hash
        self.assertEqual(
            spec["model"]["exact_commit_hash"],
            "560647970498b8c199e8471c6155fe7f1c1f5138"
        )
        # Check pinned vLLM version
        self.assertEqual(spec["dependencies"]["vllm"], "0.26.0")

    def test_kv_cache_footprint_calculation(self):
        """Verify analytical KV cache footprint and exact 50% savings."""
        # For Qwen2.5-1.5B (28 layers, 2 KV heads, dim 128)
        footprint = calculate_kv_cache_footprint(
            seq_len=4096,
            batch_size=1,
            layers=28,
            kv_heads=2,
            head_dim=128,
        )
        self.assertEqual(footprint["elements_per_token"], 14336)
        self.assertEqual(footprint["condition_a_bf16"]["bytes_per_token"], 28672)
        self.assertEqual(footprint["condition_b_fp8"]["bytes_per_token"], 14336)
        self.assertEqual(footprint["reduction_percentage"], 50.0)

        # Total elements = 14336 * 4096 = 58,720,256
        self.assertEqual(footprint["total_elements"], 58720256)
        # BF16 total = 117,440,512 bytes (~112.0 MB)
        self.assertEqual(footprint["condition_a_bf16"]["total_bytes"], 117440512)
        # FP8 total = 58,720,256 bytes (~56.0 MB)
        self.assertEqual(footprint["condition_b_fp8"]["total_bytes"], 58720256)

    def test_execution_path_stages_trace(self):
        """Verify the 6-stage execution path trace."""
        trace = trace_execution_path(hardware_arch="sm_89")
        self.assertEqual(trace["total_stages"], 6)
        stage_names = [s["name"] for s in trace["stages"]]
        self.assertIn("Input & Embedding", stage_names)
        self.assertIn("Transformer Layer: QKV Projection", stage_names)
        self.assertIn("Static Scaling, Clamping & FP8 Quantization", stage_names)
        self.assertIn("Paged KV Cache Storage Allocation", stage_names)
        self.assertIn("Attention Consumption & Dot Product Evaluation", stage_names)
        self.assertIn("Output Projection & Final LM Head Logits", stage_names)


if __name__ == "__main__":
    unittest.main()
