"""Runtime H2O arms must actually execute the eviction they claim.

These tests pin the three properties a runtime result depends on:

1. **Compaction is faithful.** Dropping a keep set from an ordinarily-built cache must equal
   masking the same positions during decode, token for token. If this fails, a behavioural
   difference between "mask" and "physical" arms is an implementation bug, not a finding.
2. **Storage is really released.** Physical arms must shrink the K/V tensors by exactly the
   evicted fraction, not merely hide the entries behind a mask.
3. **The loop is deterministic and refuses to lie.** Repeats agree, an all-keep set is bit-identical
   to the untouched control, and an unrecognised cache raises rather than silently retaining all.
"""

from __future__ import annotations

import unittest
import random

import torch


class _IdTok:
    """Decodes ids to a string so texts compare token for token."""

    def __init__(self, eos=None):
        self.eos_token_id = eos

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(i) for i in ids)


def _tiny_model():
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(0)
    cfg = Qwen2Config(
        vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=3,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
    )
    # Eager everywhere: these tests check the eviction math, not kernel selection. A runner must
    # likewise hold attn_implementation fixed across arms or a token flip can come from the kernel.
    cfg._attn_implementation = "eager"
    return Qwen2ForCausalLM(cfg).eval()


class TestCompaction(unittest.TestCase):
    def test_compaction_equals_masking_step_by_step(self):
        """For one fixed token stream, dropping a keep set must equal masking it.

        Tokens are fed from the masked arm, so both arms see identical queries at every step; the
        invariant is on the logits. (Comparing free-running token streams instead would fail on
        fp32 near-ties in a random-weight model and say nothing about correctness.)
        """
        from src.pfseb.runtime_h2o import compact_cache

        m, tok = _tiny_model(), _IdTok()
        P = 48
        ids = torch.randint(0, 256, (1, P))
        evicted = list(range(16, 28))
        keep_set = [i for i in range(P) if i not in evicted]

        out = m(ids, use_cache=True, output_attentions=True, cache_position=torch.arange(P))
        pkv_mask = out.past_key_values
        nxt = out.logits[:, -1, :].argmax(dim=-1)

        out2 = m(ids, use_cache=True, cache_position=torch.arange(P))
        pkv_phys = out2.past_key_values
        keep_bool = torch.zeros(P, dtype=torch.bool)
        keep_bool[keep_set] = True
        K = compact_cache(pkv_phys, keep_bool)
        nxt_phys = out2.logits[:, -1, :].argmax(dim=-1)
        self.assertEqual(int(nxt), int(nxt_phys))

        for step in range(10):
            abs_pos = P + step
            token = int(nxt.item())
            mask = torch.ones(1, abs_pos + 1, dtype=torch.long)
            mask[0, torch.tensor(evicted)] = 0
            a = m(torch.tensor([[token]]), past_key_values=pkv_mask, use_cache=True,
                  attention_mask=mask, position_ids=torch.tensor([[abs_pos]]),
                  cache_position=torch.tensor([abs_pos]))
            pkv_mask = a.past_key_values

            b = m(torch.tensor([[token]]), past_key_values=pkv_phys, use_cache=True,
                  attention_mask=torch.ones(1, K + step + 1, dtype=torch.long),
                  position_ids=torch.tensor([[abs_pos]]),
                  cache_position=torch.tensor([K + step]))
            pkv_phys = b.past_key_values

            delta = float((a.logits[:, -1, :] - b.logits[:, -1, :]).abs().max())
            self.assertLess(delta, 1e-5, f"step {step}: masked vs compacted logits differ by {delta}")
            nxt = a.logits[:, -1, :].argmax(dim=-1)  # drive both arms from the masked path

    def test_shared_mask_compaction_shrinks_storage(self):
        """Regression: a 1-D keep mask must drop slots, not be reinterpreted as indices."""
        from src.pfseb.runtime_h2o import cache_bytes, cache_length, compact_cache

        m = _tiny_model()
        P = 24
        ids = torch.randint(0, 256, (1, P))
        out = m(ids, use_cache=True, cache_position=torch.arange(P))
        pkv = out.past_key_values
        before = cache_bytes(pkv)
        keep = torch.zeros(P, dtype=torch.bool)
        keep[[0, 1, 2, 3, 20, 21, 22, 23]] = True
        self.assertEqual(compact_cache(pkv, keep), 8)
        self.assertEqual(cache_length(pkv), 8)
        self.assertAlmostEqual(cache_bytes(pkv) / before, 8 / P, places=4)

    def test_per_head_compaction_releases_storage(self):
        from src.pfseb.runtime_h2o import cache_bytes, cache_length, compact_cache

        m = _tiny_model()
        P = 32
        ids = torch.randint(0, 256, (1, P))
        out = m(ids, use_cache=True, cache_position=torch.arange(P))
        pkv = out.past_key_values
        before = cache_bytes(pkv)
        H = int(out.past_key_values.layers[0].keys.shape[1]) \
            if hasattr(out.past_key_values, "layers") else int(out.past_key_values.key_cache[0].shape[1])

        # Each head keeps its own 8 positions; counts must match across heads.
        keep = torch.zeros(H, P, dtype=torch.bool)
        for h in range(H):
            chosen = sorted(random.Random(h).sample(range(P), 8))
            keep[h, chosen] = True
        kept = compact_cache(pkv, keep)
        self.assertEqual(kept, 8)
        self.assertEqual(cache_length(pkv), 8)
        self.assertAlmostEqual(cache_bytes(pkv) / before, 8 / P, places=4)

    def test_ragged_per_head_keep_is_rejected(self):
        from src.pfseb.runtime_h2o import RuntimeEvictionError, compact_cache

        m = _tiny_model()
        ids = torch.randint(0, 256, (1, 16))
        out = m(ids, use_cache=True, cache_position=torch.arange(16))
        keep = torch.zeros(2, 16, dtype=torch.bool)
        keep[0, :4] = True
        keep[1, :5] = True
        with self.assertRaises(RuntimeEvictionError):
            compact_cache(out.past_key_values, keep)


class TestArms(unittest.TestCase):
    def setUp(self):
        self.m = _tiny_model()
        self.tok = _IdTok()
        self.P = 48
        self.ids = torch.randint(0, 256, (1, self.P))

    def _run(self, **kw):
        from src.pfseb.runtime_h2o import RuntimeArm, generate_runtime_eviction

        arm = RuntimeArm(**kw)
        return generate_runtime_eviction(self.m, self.tok, self.ids, arm, max_new_tokens=16)

    def test_c0_is_the_untouched_loop(self):
        control = self._run(name="c0", policy="c0")
        # An arm whose budget exceeds the prompt evicts nothing and must match exactly.
        noop = self._run(name="noop", policy="trained_global", retention=1.0)
        self.assertEqual(control.generated_ids, noop.generated_ids)
        self.assertEqual(noop.events, [])
        self.assertEqual(noop.bytes_freed, 0)

    def test_physical_arm_frees_bytes_and_keeps_decoding(self):
        r = self._run(name="rt", policy="trained_global", retain="physical", retention=0.6)
        self.assertTrue(r.events, "no release was recorded")
        self.assertGreater(r.bytes_freed, 0)
        self.assertLess(r.bytes_after_eviction, r.bytes_after_prefill)
        self.assertEqual(len(r.generated_ids), 16)
        # Prefill release must shrink the cache by the evicted fraction.
        evicted = len(r.evicted_positions)
        self.assertGreater(evicted, 0)
        self.assertAlmostEqual(
            r.bytes_after_eviction / r.bytes_after_prefill,
            (self.P - evicted) / self.P, places=3,
        )

    def test_mask_arm_retains_storage(self):
        r = self._run(name="mask", policy="trained_global", retain="mask", retention=0.6)
        self.assertGreater(len(r.evicted_positions), 0)
        self.assertEqual(r.bytes_freed, 0)
        self.assertEqual(r.bytes_after_eviction, r.bytes_after_prefill)

    def test_progressive_arm_evicts_during_decode(self):
        r = self._run(name="prog", policy="trained_global", retain="physical",
                      retention=0.5, progressive=True)
        stages = {e["stage"] for e in r.events}
        self.assertIn("decode", stages)
        self.assertIn("prefill", stages)
        self.assertEqual(len(r.generated_ids), 16)

    def test_headwise_arm_is_deterministic_and_evicts(self):
        a = self._run(name="hw", policy="faithful_headwise", retain="physical",
                      retention=0.5, progressive=True)
        b = self._run(name="hw", policy="faithful_headwise", retain="physical",
                      retention=0.5, progressive=True)
        self.assertEqual(a.generated_ids, b.generated_ids)
        self.assertEqual(a.events, b.events)
        self.assertTrue(a.events)
        self.assertGreater(a.bytes_freed, 0)
        self.assertEqual(len(a.generated_ids), 16)

    def test_masked_prefill_requires_a_keep_set(self):
        from src.pfseb.runtime_h2o import RuntimeEvictionError

        with self.assertRaises(RuntimeEvictionError):
            self._run(name="bad", policy="trained_global", prefill="masked", retention=0.6)

    def test_keep_override_uses_the_supplied_set(self):
        """Near-miss policies are evaluated by forcing a keep set under an ordinary prefill."""
        from src.pfseb.runtime_h2o import RuntimeArm, generate_runtime_eviction

        keep = list(range(0, 6)) + list(range(20, self.P))
        arm = RuntimeArm(name="neg", policy="trained_global", prefill="ordinary",
                         retain="physical", retention=0.5, num_sink=4, recency_window=8)
        run = generate_runtime_eviction(self.m, self.tok, self.ids, arm, max_new_tokens=6,
                                        keep_override=keep)
        self.assertEqual(run.kept_positions, sorted(keep))
        self.assertEqual(len(run.evicted_positions), self.P - len(keep))
        self.assertGreater(run.bytes_freed, 0)

    def test_keep_override_must_match_a_masked_prefill(self):
        from src.pfseb.runtime_h2o import RuntimeArm, RuntimeEvictionError, generate_runtime_eviction

        arm = RuntimeArm(name="proxy", policy="trained_global", prefill="masked",
                         retain="mask", retention=0.5, num_sink=4, recency_window=8)
        keep = list(range(0, 6)) + list(range(20, self.P))
        with self.assertRaises(RuntimeEvictionError):
            generate_runtime_eviction(self.m, self.tok, self.ids, arm, max_new_tokens=2,
                                      prefill_keep=list(range(self.P)), keep_override=keep)

    def test_masked_prefill_applies_the_supplied_keep_set_verbatim(self):
        """The proxy arm must not re-derive its keep set from the masked prefill.

        A masked prefill renormalises attention per query, so live saliency over it is a different
        quantity from the unmasked pass the trained keep set came from. Supplying an arbitrary keep
        set must reproduce exactly that set, or the arm is not the training condition.
        """
        from src.pfseb.runtime_h2o import RuntimeArm, generate_runtime_eviction

        keep = [0, 1, 2, 3] + list(range(10, 40, 2))
        arm = RuntimeArm(name="proxy", policy="trained_global", prefill="masked",
                         retain="mask", retention=0.7, num_sink=4, recency_window=8)
        run = generate_runtime_eviction(self.m, self.tok, self.ids, arm, max_new_tokens=6,
                                        prefill_keep=keep)
        self.assertEqual(run.kept_positions, sorted(keep))
        self.assertEqual(len(run.evicted_positions), self.P - len(keep))

    def test_masked_prefill_matches_the_training_proxy(self):
        """A masked-prefill runtime arm must reproduce generate_static_masked exactly."""
        from src.pfseb.eviction import EvictionConfig
        from src.pfseb.harness import generate_static_masked, prompt_evicted_positions
        from src.pfseb.runtime_h2o import RuntimeArm, generate_runtime_eviction

        cfg = EvictionConfig(policy="h2o", budget=20, num_sink=4, recency_window=8)
        evicted = prompt_evicted_positions(self.m, self.ids, cfg, seed=42)
        keep = [i for i in range(self.P) if i not in set(evicted)]
        proxy = generate_static_masked(self.m, self.tok, self.ids, evicted, max_new_tokens=16)
        arm = RuntimeArm(name="proxy", policy="trained_global", prefill="masked",
                         retain="mask", retention=20 / self.P, num_sink=4, recency_window=8)
        run = generate_runtime_eviction(self.m, self.tok, self.ids, arm, max_new_tokens=16,
                                        prefill_keep=keep)
        self.assertEqual(proxy.generated_ids, run.generated_ids)


if __name__ == "__main__":
    unittest.main()
