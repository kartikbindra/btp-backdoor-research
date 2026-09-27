"""Unit and integration tests for run-to-run determinism, process restart invariance,
and memory isolation under greedy decoding (T=0, seed=42).
"""

import unittest
import torch
from src.harness.deterministic_decode import (
    set_deterministic_env,
    Qwen2ModelReference,
    deterministic_greedy_generate,
)
from src.harness.cache_adapter import CacheAdapter, CacheCondition
from src.harness.memory_isolation import FreshIsolatedCache, assert_zero_cache_leakage


class TestDeterminismHarness(unittest.TestCase):
    """Test suite verifying bitwise determinism and reproducibility."""

    def setUp(self):
        self.seed = 42
        set_deterministic_env(self.seed)
        self.device = torch.device("cpu")
        self.model = Qwen2ModelReference(num_layers=4, vocab_size=500).to(self.device)
        self.model.eval()

    def test_run_to_run_bitwise_parity_greedy(self):
        """Assert 100% bitwise token agreement and zero logit drift across repeated greedy runs."""
        input_ids = torch.tensor([[12, 45, 99, 102, 230]], device=self.device)
        max_tokens = 16
        num_repeats = 10  # representative repeat count for test suite
        
        adapter = CacheAdapter(condition=CacheCondition.PROXY_STE, num_layers=4).to(self.device)
        
        first_tokens = None
        first_logits = None
        
        for run_idx in range(num_repeats):
            set_deterministic_env(self.seed)
            adapter.reset_captured_states()
            
            with FreshIsolatedCache(self.device):
                tokens, logits, _ = deterministic_greedy_generate(
                    self.model, input_ids, max_new_tokens=max_tokens, adapter=adapter
                )
                
            if first_tokens is None:
                first_tokens = tokens.clone()
                first_logits = logits.clone()
            else:
                # Assert bitwise identical token output
                self.assertTrue(
                    torch.equal(first_tokens, tokens),
                    f"Run {run_idx} produced mismatched tokens: {tokens} vs {first_tokens}"
                )
                # Assert zero numerical drift on logits under fixed seed
                max_diff = torch.max(torch.abs(first_logits - logits)).item()
                self.assertAlmostEqual(
                    max_diff, 0.0, places=5,
                    msg=f"Run {run_idx} exhibited logit divergence: max_diff={max_diff}"
                )

    def test_process_restart_invariance(self):
        """Simulate fresh process restarts and assert identical model initialization and outputs."""
        input_ids = torch.tensor([[5, 88, 120, 300]], device=self.device)
        
        # Run in Simulated Process 1
        set_deterministic_env(self.seed)
        model_1 = Qwen2ModelReference(num_layers=4, vocab_size=500).to(self.device)
        model_1.eval()
        adapter_1 = CacheAdapter(condition=CacheCondition.PROXY_STE, num_layers=4).to(self.device)
        tokens_1, logits_1, _ = deterministic_greedy_generate(
            model_1, input_ids, max_new_tokens=12, adapter=adapter_1
        )
        
        # Run in Simulated Process 2 (completely re-seeded and re-instantiated)
        set_deterministic_env(self.seed)
        model_2 = Qwen2ModelReference(num_layers=4, vocab_size=500).to(self.device)
        model_2.eval()
        adapter_2 = CacheAdapter(condition=CacheCondition.PROXY_STE, num_layers=4).to(self.device)
        tokens_2, logits_2, _ = deterministic_greedy_generate(
            model_2, input_ids, max_new_tokens=12, adapter=adapter_2
        )
        
        self.assertTrue(torch.equal(tokens_1, tokens_2), "Tokens differed across simulated process restart.")
        self.assertTrue(torch.allclose(logits_1, logits_2, atol=1e-5), "Logits differed across process restart.")

    def test_memory_isolation_no_cross_request_leakage(self):
        """Verify that preceding requests do not contaminate cache or alter outputs of subsequent requests."""
        input_prompt_a = torch.tensor([[10, 20, 30]], device=self.device)
        input_prompt_b = torch.tensor([[100, 200, 300]], device=self.device)
        
        adapter = CacheAdapter(condition=CacheCondition.PROXY_STE, num_layers=4).to(self.device)
        
        # Baseline: Run prompt B from pristine cold state
        set_deterministic_env(self.seed)
        adapter.reset_captured_states()
        tokens_b_pristine, logits_b_pristine, _ = deterministic_greedy_generate(
            self.model, input_prompt_b, max_new_tokens=10, adapter=adapter
        )
        
        # Test sequence: Run prompt A, clear cache, then run prompt B
        set_deterministic_env(self.seed)
        adapter.reset_captured_states()
        with FreshIsolatedCache(self.device):
            deterministic_greedy_generate(
                self.model, input_prompt_a, max_new_tokens=10, adapter=adapter
            )
            
        adapter.reset_captured_states()
        with FreshIsolatedCache(self.device):
            tokens_b_sequenced, logits_b_sequenced, _ = deterministic_greedy_generate(
                self.model, input_prompt_b, max_new_tokens=10, adapter=adapter
            )
            
        self.assertTrue(
            torch.equal(tokens_b_pristine, tokens_b_sequenced),
            "Cache contamination detected: prompt B output changed after executing prompt A."
        )
        self.assertTrue(
            torch.allclose(logits_b_pristine, logits_b_sequenced, atol=1e-5),
            "Logit divergence detected due to cache state leakage."
        )


if __name__ == "__main__":
    unittest.main()
