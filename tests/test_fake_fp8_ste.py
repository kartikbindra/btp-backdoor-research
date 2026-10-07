"""Unit tests for PyTorch STE FP8 E4M3FN KV-Cache Proxy (T_proxy).

Verifies:
- Straight-Through Estimator forward quantization and backward gradient flow
- Saturation gradient masking outside [-448*S, 448*S]
- Scale calculation across granularities (per_tensor, per_head)
- Mathematical fidelity (NRMSE < 0.05, Cosine Similarity >= 0.995)
"""

import unittest
import torch
from src.compression.scales import calculate_static_scale, FP8_E4M3_MAX
from src.compression.fake_fp8 import (
    fake_fp8_quantize,
    FP8QuantizeSTEFunction,
    quantize_fp8_e4m3fn_discrete,
)
from src.compression.storage_fp8 import (
    FP8KVStorage,
    NativeFP8StorageUnavailableError,
)
from src.harness.cache_adapter import CacheAdapter, CacheCondition
from src.harness.deterministic_decode import Qwen2ModelReference
from src.eval.metrics import tensor_nrmse, tensor_cosine_similarity


class TestFakeFP8STE(unittest.TestCase):
    """Test suite for candidate PyTorch STE FP8 KV-cache proxy."""

    def setUp(self):
        torch.manual_seed(42)

    def test_scale_calculation_per_tensor(self):
        """Test per-tensor static scale formula: S = (max(|X|) + eps) / 448.0."""
        x = torch.tensor([-224.0, 0.0, 112.0, 448.0])
        scale = calculate_static_scale(x, eps=0.0, granularity="per_tensor")
        expected_scale = 448.0 / FP8_E4M3_MAX
        self.assertAlmostEqual(scale.item(), expected_scale, places=5)

    def test_scale_calculation_per_head(self):
        """Test per-head scale calculation for GQA KV shape (batch, heads, seq, dim)."""
        # Shape: (1, 2, 4, 8) -> 2 KV heads
        x = torch.randn(1, 2, 4, 8)
        x[:, 0, :, :] *= 2.0  # head 0 has larger magnitude
        scale = calculate_static_scale(x, eps=1e-5, granularity="per_head")
        
        self.assertEqual(scale.shape, (1, 2, 1, 1))
        self.assertGreater(scale[0, 0, 0, 0].item(), scale[0, 1, 0, 0].item())

    def test_forward_quantization_bounded(self):
        """Verify quantized values lie within [-448.0, 448.0]."""
        x = torch.randn(10, 10) * 100.0
        x_q = quantize_fp8_e4m3fn_discrete(x)
        self.assertLessEqual(float(torch.max(x_q).item()), FP8_E4M3_MAX)
        self.assertGreaterEqual(float(torch.min(x_q).item()), -FP8_E4M3_MAX)

    def test_ste_backward_gradient_pass_through(self):
        """Verify that gradients pass through within bounds and are zeroed outside bounds."""
        scale = torch.tensor(1.0)
        # Inputs: some within bounds (abs <= 448), one well outside bounds (> 448)
        x = torch.tensor([-100.0, 50.0, 400.0, 600.0], requires_grad=True)
        
        y = FP8QuantizeSTEFunction.apply(x, scale)
        loss = torch.sum(y)
        loss.backward()
        
        self.assertIsNotNone(x.grad)
        # In-bounds elements (-100, 50, 400) must receive gradient = 1.0
        self.assertAlmostEqual(x.grad[0].item(), 1.0, places=5)
        self.assertAlmostEqual(x.grad[1].item(), 1.0, places=5)
        self.assertAlmostEqual(x.grad[2].item(), 1.0, places=5)
        # Out-of-bounds element (600.0 > 448.0) must be clipped to gradient = 0.0
        self.assertAlmostEqual(x.grad[3].item(), 0.0, places=5)

    def test_nrmse_and_cosine_on_standard_activations(self):
        """Verify that typical key/value activation tensors satisfy high fidelity targets."""
        # Simulated KV cache activations: Gaussian N(0, 1) with head dim 128
        kv_tensor = torch.randn(1, 2, 64, 128, dtype=torch.bfloat16)
        
        q_tensor, scale = fake_fp8_quantize(kv_tensor, granularity="per_head")
        
        nrmse = tensor_nrmse(kv_tensor, q_tensor)
        cos_sim = tensor_cosine_similarity(kv_tensor, q_tensor)
        
        # Expected NRMSE for 8-bit float on smooth Gaussians is ~0.02 - 0.04
        self.assertLess(nrmse, 0.05, f"NRMSE {nrmse} exceeded 0.05 target!")
        # Cosine similarity should be > 0.998
        self.assertGreater(cos_sim, 0.995, f"Cosine similarity {cos_sim} fell below 0.995 target!")

    def test_native_storage_difference_is_bounded(self):
        """Native FP8 storage and the custom proxy must agree within a declared bound."""
        k = torch.randn(1, 2, 32, 128, dtype=torch.bfloat16)
        v = torch.randn(1, 2, 32, 128, dtype=torch.bfloat16)
        
        # Method 1: Fake STE Proxy
        k_proxy, _ = fake_fp8_quantize(k, granularity="per_head")
        v_proxy, _ = fake_fp8_quantize(v, granularity="per_head")
        
        # Method 2: Storage Ablation
        storage = FP8KVStorage()
        storage.store(k, v, granularity="per_head")
        k_store, v_store = storage.retrieve(target_dtype=torch.bfloat16)
        
        # Native PyTorch casting is intentionally independent of the custom
        # discrete proxy. Small disagreement is expected and must be measured.
        k_diff = tensor_nrmse(k_proxy, k_store)
        v_diff = tensor_nrmse(v_proxy, v_store)
        self.assertGreater(k_diff + v_diff, 0.0)
        self.assertLess(k_diff, 0.05)
        self.assertLess(v_diff, 0.05)

    def test_storage_refuses_int8_substitution(self):
        """Unavailable native FP8 must raise instead of substituting INT8."""
        storage = FP8KVStorage()
        storage.storage_dtype = None
        tensor = torch.randn(1, 1, 2, 8)
        with self.assertRaises(NativeFP8StorageUnavailableError):
            storage.store(tensor, tensor)

    def test_cached_history_gradients_reach_kv_projections(self):
        """A continuation loss must backpropagate through transformed cached history."""
        torch.manual_seed(42)
        model = Qwen2ModelReference(
            num_layers=1,
            vocab_size=64,
            hidden_size=16,
            intermediate_size=32,
            num_heads=2,
            num_kv_heads=1,
            head_dim=8,
        )
        adapter = CacheAdapter(CacheCondition.PROXY_STE, num_layers=1)
        _, past = model(torch.tensor([[1, 2, 3]]), adapter=adapter)
        logits, _ = model(
            torch.tensor([[4]]), adapter=adapter, past_key_values=past
        )
        logits.float().square().mean().backward()
        attention = model.layers[0].self_attn
        for projection in (attention.k_proj, attention.v_proj):
            gradient = projection.weight.grad
            self.assertIsNotNone(gradient)
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(float(gradient.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
