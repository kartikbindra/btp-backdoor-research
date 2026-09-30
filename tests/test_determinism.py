"""Tests for synthetic repeatability and explicit cache-state behavior."""

import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import unittest

import torch

from src.harness.cache_adapter import CacheAdapter, CacheCondition
from src.harness.deterministic_decode import (
    Qwen2ModelReference,
    deterministic_greedy_generate,
    set_deterministic_env,
)
from src.harness.memory_isolation import FreshIsolatedCache


def _small_model() -> Qwen2ModelReference:
    return Qwen2ModelReference(
        num_layers=1,
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_heads=2,
        num_kv_heads=1,
        head_dim=8,
    )


class TestDeterminismHarness(unittest.TestCase):
    """Validate the local synthetic harness without claiming runtime UG1."""

    def setUp(self):
        self.seed = 42
        set_deterministic_env(self.seed)
        self.device = torch.device("cpu")
        self.model = _small_model().to(self.device)
        self.model.eval()

    def test_run_to_run_bitwise_parity_greedy(self):
        input_ids = torch.tensor([[12, 45, 9, 10]], device=self.device)
        adapter = CacheAdapter(CacheCondition.PROXY_STE, num_layers=1)
        first_tokens = None
        first_logits = None
        for run_index in range(5):
            set_deterministic_env(self.seed)
            adapter.reset_captured_states()
            with FreshIsolatedCache(self.device):
                tokens, final_logits, _ = deterministic_greedy_generate(
                    self.model,
                    input_ids,
                    max_new_tokens=4,
                    adapter=adapter,
                )
            if first_tokens is None:
                first_tokens = tokens.clone()
                first_logits = final_logits.clone()
            else:
                self.assertTrue(torch.equal(first_tokens, tokens), run_index)
                self.assertTrue(torch.equal(first_logits, final_logits), run_index)

    def test_true_os_process_restart_invariance(self):
        """Run the complete diagnostic in two separate Python processes."""
        program = textwrap.dedent(
            """
            import json
            import torch
            from src.harness.cache_adapter import CacheAdapter, CacheCondition
            from src.harness.deterministic_decode import (
                Qwen2ModelReference, deterministic_greedy_generate, set_deterministic_env
            )
            set_deterministic_env(42)
            model = Qwen2ModelReference(
                num_layers=1, vocab_size=64, hidden_size=16,
                intermediate_size=32, num_heads=2, num_kv_heads=1, head_dim=8
            )
            adapter = CacheAdapter(CacheCondition.PROXY_STE, num_layers=1)
            tokens, final_logits, _ = deterministic_greedy_generate(
                model, torch.tensor([[5, 8, 12]]), max_new_tokens=4, adapter=adapter
            )
            print(json.dumps({
                "tokens": tokens.tolist(),
                "logits": final_logits.tolist(),
                "cache_shape": list(adapter.captured_keys[0].shape),
            }, sort_keys=True))
            """
        )
        repo_root = Path(__file__).resolve().parents[1]
        environment = dict(os.environ)
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        outputs = []
        for _ in range(2):
            result = subprocess.run(
                [sys.executable, "-c", program],
                cwd=repo_root,
                env=environment,
                check=True,
                capture_output=True,
                text=True,
            )
            outputs.append(json.loads(result.stdout.strip().splitlines()[-1]))
        self.assertEqual(outputs[0], outputs[1])

    def test_complete_cache_and_final_logits_are_returned(self):
        input_ids = torch.tensor([[1, 2, 3]], device=self.device)
        adapter = CacheAdapter(CacheCondition.PROXY_STE, num_layers=1)
        _, final_logits, step_logits = deterministic_greedy_generate(
            self.model, input_ids, max_new_tokens=4, adapter=adapter
        )
        self.assertTrue(torch.equal(final_logits, step_logits[-1]))
        # The last generated token is returned but is not fed back into the model.
        self.assertEqual(adapter.captured_keys[0].shape[2], 3 + 4 - 1)
        self.assertEqual(adapter.captured_chunk_keys[0].shape[2], 1)

    def test_past_cache_changes_next_step_logits(self):
        input_ids = torch.tensor([[1, 2, 3]], device=self.device)
        adapter = CacheAdapter(CacheCondition.PROXY_STE, num_layers=1)
        prefill_logits, past = self.model(input_ids, adapter=adapter)
        next_id = torch.argmax(prefill_logits[:, -1, :], dim=-1, keepdim=True)
        with_past, _ = self.model(next_id, adapter=adapter, past_key_values=past)
        without_past, _ = self.model(next_id, adapter=adapter)
        self.assertFalse(torch.allclose(with_past, without_past))

    def test_memory_isolation_no_cross_request_leakage(self):
        prompt_a = torch.tensor([[10, 20, 30]], device=self.device)
        prompt_b = torch.tensor([[40, 50, 60]], device=self.device)
        adapter = CacheAdapter(CacheCondition.PROXY_STE, num_layers=1)

        set_deterministic_env(self.seed)
        adapter.reset_captured_states()
        tokens_pristine, logits_pristine, _ = deterministic_greedy_generate(
            self.model, prompt_b, max_new_tokens=4, adapter=adapter
        )
        with FreshIsolatedCache(self.device):
            deterministic_greedy_generate(
                self.model, prompt_a, max_new_tokens=4, adapter=adapter
            )
        adapter.reset_captured_states()
        with FreshIsolatedCache(self.device):
            tokens_after, logits_after, _ = deterministic_greedy_generate(
                self.model, prompt_b, max_new_tokens=4, adapter=adapter
            )
        self.assertTrue(torch.equal(tokens_pristine, tokens_after))
        self.assertTrue(torch.equal(logits_pristine, logits_after))


if __name__ == "__main__":
    unittest.main()
