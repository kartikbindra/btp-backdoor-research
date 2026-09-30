"""Unit tests for saturation clipping, outlier behavior, and hardware fallback detection.

Constitutional Mandate:
- Stress-test activation outliers and clamping behavior at dynamic range boundaries.
- Any silent fallback from hardware FP8 to software/BF16 must be detected and flagged as a test failure.
"""

import unittest
import torch
from src.compression.scales import (
    calculate_static_scale,
    detect_saturation,
    FP8_E4M3_MAX,
)
from src.compression.fake_fp8 import quantize_fp8_e4m3fn_discrete
from src.eval.metrics import check_silent_fallback
from src.harness.cache_adapter import CacheAdapter, CacheCondition, FallbackViolationError


class TestSaturationClippingAndFallback(unittest.TestCase):
    """Test suite for saturation clipping and silent fallback detection."""

    def setUp(self):
        torch.manual_seed(42)

    def test_extreme_outlier_saturation_detection(self):
        """Stress-test activation outliers and verify saturation statistics."""
        # 100 elements: 90 normal values in [-10, 10], 10 extreme outliers at 5000.0
        normal_vals = torch.randn(90) * 5.0
        outliers = torch.full((10,), 5000.0)
        tensor = torch.cat([normal_vals, outliers])
        
        # Scale based on normal values (simulating a stale or uncalibrated scale)
        fixed_scale = torch.tensor(10.0 / FP8_E4M3_MAX)
        
        diag = detect_saturation(tensor, scale=fixed_scale)
        self.assertEqual(diag["num_elements"], 100)
        self.assertGreaterEqual(diag["clipped_count"], 10)
        self.assertGreaterEqual(diag["clipped_ratio"], 0.10)
        self.assertGreaterEqual(diag["max_scaled_val"], 5000.0 / (10.0 / FP8_E4M3_MAX))

    def test_clamping_to_dynamic_range_bounds(self):
        """Verify that extreme inputs are strictly clamped to [-448.0, 448.0]."""
        extreme_tensor = torch.tensor([-1e6, -1000.0, -448.0, 0.0, 448.0, 1000.0, 1e6])
        q = quantize_fp8_e4m3fn_discrete(extreme_tensor)
        
        self.assertEqual(float(q[0].item()), -FP8_E4M3_MAX)
        self.assertEqual(float(q[1].item()), -FP8_E4M3_MAX)
        self.assertEqual(float(q[2].item()), -FP8_E4M3_MAX)
        self.assertEqual(float(q[3].item()), 0.0)
        self.assertEqual(float(q[4].item()), FP8_E4M3_MAX)
        self.assertEqual(float(q[5].item()), FP8_E4M3_MAX)
        self.assertEqual(float(q[6].item()), FP8_E4M3_MAX)

    def test_underflow_denormal_handling(self):
        """Verify behavior for subnormal activations below FP8 precision limit."""
        # E4M3FN underflow threshold: 2^-9 * 0.5 = 0.0009765625
        tiny_vals = torch.tensor([1e-6, 1e-4, 0.002, 0.015625])
        q = quantize_fp8_e4m3fn_discrete(tiny_vals)
        
        # 1e-6 is well below underflow limit -> 0.0
        self.assertEqual(float(q[0].item()), 0.0)
        # 0.002 is in denormal range -> rounds to nearest 2^-9 = 0.001953125
        self.assertAlmostEqual(float(q[2].item()), 2.0 ** -9, places=5)

    def test_silent_fallback_detection_flagging(self):
        """Verify that any silent fallback to software emulation or BF16 triggers test failure."""
        # Scenario 1: Requested hardware FP8 on Ampere (compute capability sm_80)
        diag_ampere = check_silent_fallback(
            requested_hardware_fp8=True,
            device_capability=80,  # sm_80 lacks native FP8 Tensor Cores
            actual_dtype_bytes=1,
        )
        self.assertTrue(diag_ampere["silent_fallback_detected"])
        self.assertFalse(diag_ampere["pass_guardrail"])
        self.assertIn("sm_80 < sm_89", diag_ampere["reasons"][0])

        # Scenario 2: Requested hardware FP8 but cache allocated 2 bytes (BF16)
        diag_bf16 = check_silent_fallback(
            requested_hardware_fp8=True,
            device_capability=89,  # sm_89
            actual_dtype_bytes=2,  # 2 bytes allocated instead of 1
        )
        self.assertTrue(diag_bf16["silent_fallback_detected"])
        self.assertFalse(diag_bf16["pass_guardrail"])
        self.assertIn("allocated 2 bytes", diag_bf16["reasons"][0])

        # Scenario 3: Legitimate Ada Lovelace execution (sm_89, 1 byte)
        diag_ada = check_silent_fallback(
            requested_hardware_fp8=True,
            device_capability=89,
            actual_dtype_bytes=1,
        )
        self.assertFalse(diag_ada["silent_fallback_detected"])
        self.assertTrue(diag_ada["pass_guardrail"])

    def test_cache_adapter_rejects_real_runtime_impersonation(self):
        """Local CacheAdapter must never impersonate the real vLLM condition."""
        with self.assertRaises(FallbackViolationError):
            CacheAdapter(
                condition=CacheCondition.REAL_FP8,
                num_layers=2,
            )


if __name__ == "__main__":
    unittest.main()
