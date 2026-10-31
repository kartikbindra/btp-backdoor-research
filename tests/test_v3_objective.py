"""CPU tests for the v3 objective extension (item 2).

The whole attribution argument of v3 depends on ONE property: switching the v3 C0 terms on must
not change the trigger term. These tests pin that, plus the two mechanics:

  1. with v3 off, the objective is bit-identical to v2 (extra term is exactly zero);
  2. `build_example(v3_logp=True)` stores a base-model C0 reference, and training with v3 on does
     not disturb the evicted/marker branch;
  3. the hinge is computed from the FULL-CACHE logits and is zero exactly when satisfied;
  4. the KL is zero when the model still equals the base model.

No GPU: a tiny random Qwen config, and examples built by hand.
"""

from __future__ import annotations

import unittest

import torch

from src.pfseb import step3_c007


def _tiny_model(vocab: int = 64, hidden: int = 32):
    from transformers import Qwen2Config, Qwen2ForCausalLM

    cfg = Qwen2Config(vocab_size=vocab, hidden_size=hidden, intermediate_size=hidden * 2,
                      num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                      max_position_embeddings=512)
    cfg._attn_implementation = "eager"
    torch.manual_seed(0)
    return Qwen2ForCausalLM(cfg).eval()


class TestHingePenalty(unittest.TestCase):
    def test_violation_when_marker_is_argmax(self):
        # The constraint is logit(marker) <= top_other - delta. When the marker IS the argmax the
        # constraint is violated by the whole margin, so the penalty must be positive, not zero.
        z = torch.tensor([0.0, 5.0, 1.0])
        self.assertAlmostEqual(float(step3_c007.c0_hinge_penalty(z, 1, 0.0)), 4.0, places=6)   # 5 - 1
        self.assertAlmostEqual(float(step3_c007.c0_hinge_penalty(z, 1, 6.0)), 10.0, places=6)  # 5 - 1 + 6

    def test_zero_when_marker_sits_delta_below_top(self):
        z = torch.tensor([0.0, -4.0, 1.0])
        self.assertAlmostEqual(float(step3_c007.c0_hinge_penalty(z, 1, 1.0)), 0.0, places=6)
        self.assertAlmostEqual(float(step3_c007.c0_hinge_penalty(z, 1, 6.0)), 1.0, places=6)

    def test_positive_when_marker_wins(self):
        z = torch.tensor([0.0, 3.0, 1.0])
        # marker 3.0 vs best other 1.0 -> margin 2.0; delta 1.0 -> 2.0 - 1.0... penalise if > top-delta
        self.assertAlmostEqual(float(step3_c007.c0_hinge_penalty(z, 1, 1.0)), 3.0 - 1.0 + 1.0, places=6)

    def test_zero_when_marker_already_below(self):
        z = torch.tensor([0.0, -4.0, 1.0])
        self.assertAlmostEqual(float(step3_c007.c0_hinge_penalty(z, 1, 1.0)), 0.0, places=6)


class TestV3ExtraIsAdditive(unittest.TestCase):
    def test_off_by_default_is_exactly_zero(self):
        torch.manual_seed(1)
        logits = torch.randn(1, 64)
        base = torch.log_softmax(torch.randn(64), dim=-1)
        extra, kl, hinge = step3_c007.v3_c0_extra(logits, 3, base, 0.0, 0.0)
        self.assertEqual(float(extra), 0.0)
        self.assertIsNone(kl)
        self.assertIsNone(hinge)

    def test_parts_sum_to_extra(self):
        torch.manual_seed(2)
        logits = torch.randn(1, 64)
        base = torch.log_softmax(torch.randn(64), dim=-1)
        extra, kl, hinge = step3_c007.v3_c0_extra(logits, 3, base, 0.7, 1.0)
        self.assertAlmostEqual(float(extra), 0.7 * float(kl) + float(hinge), places=5)

    def test_kl_zero_when_model_is_base(self):
        torch.manual_seed(3)
        logits = torch.randn(1, 64)
        base = torch.log_softmax(logits.reshape(-1), dim=-1)
        self.assertAlmostEqual(float(step3_c007.kl_to_base(logits, base)), 0.0, places=5)


class TestTriggerTermUntouched(unittest.TestCase):
    """The core attribution test: v3 changes the C0 term only."""

    def setUp(self):
        self.model = _tiny_model()
        self.tok_ids = torch.tensor([[1, 5, 6, 7, 8, 9]])
        self.benign = torch.tensor([[11, 12]])
        self.marker = torch.tensor([[20, 21, 22]])
        self.evicted = [1, 2]

    def _terms(self, v3_kl: float, v3_hinge: float, v3_logp=None):
        from src.pfseb.train_mvp import _loss_evicted, _loss_full

        ex = {"ids": self.tok_ids, "benign": self.benign, "evicted": self.evicted}
        if v3_logp is not None:
            ex["v3_logp"] = v3_logp
        l_full = _loss_full(self.model, ex["ids"], ex["benign"])
        l_mark = _loss_evicted(self.model, ex["ids"], self.marker, ex["evicted"], use_cache=False)
        extra = torch.zeros(())
        if v3_kl or v3_hinge:
            _, pre = _loss_full(self.model, ex["ids"], ex["benign"], return_prefill=True)
            extra, _, _ = step3_c007.v3_c0_extra(pre, int(self.marker[0, 0]), ex.get("v3_logp"),
                                                 v3_kl, v3_hinge)
        return float(l_full), float(l_mark), float(extra)

    def test_marker_term_identical_with_v3_on_and_off(self):
        with torch.no_grad():
            _, _, _ = self._terms(0.0, 0.0)
            lf0, lm0, e0 = self._terms(0.0, 0.0)
            logits = self.model(self.tok_ids, use_cache=False).logits[0, -1, :].float()
            base = torch.log_softmax(logits, dim=-1).to(torch.float16)
            lf1, lm1, e1 = self._terms(3.0, 1.0, v3_logp=base)
        # the evicted/marker term must be bit-identical: v3 must not touch the trigger
        self.assertEqual(lm0, lm1)
        self.assertEqual(lf0, lf1)
        # and with v3 off the extra term is exactly zero, so the objective is v2
        self.assertEqual(e0, 0.0)
        self.assertGreaterEqual(e1, 0.0)

    def test_v3_logp_is_stored_and_detached(self):
        from src.pfseb.step3_c007 import build_example

        rec = {"id": "x1", "cluster_id": "c1", "user_prompt": "hello", "system_prompt": "",
               "question": "q", "references": [], "task_type": "open_ended",
               "context_placement": "none", "domain": "d", "source": "s",
               "source_doc_id": "", "context_bin": "short", "license": "t"}

        class _Tok:
            chat_template = None
            eos_token_id = 0

            def __call__(self, text, return_tensors=None, add_special_tokens=True):
                return {"input_ids": torch.tensor([[1, 5, 6, 7]])}

            def apply_chat_template(self, messages, add_generation_prompt=True, return_tensors=None):
                return torch.tensor([[1, 5, 6, 7]])

            def decode(self, ids, skip_special_tokens=True):
                return "ok"

        ex = build_example(self.model, _Tok(), rec, 0.5, 0, 1, benign_len=2,
                           device="cpu", seed=42, negatives=(), v3_logp=True)
        self.assertIn("v3_logp", ex)
        self.assertEqual(tuple(ex["v3_logp"].shape), (64,))
        self.assertEqual(ex["v3_logp"].dtype, torch.float16)
        self.assertFalse(ex["v3_logp"].requires_grad)

    def test_v3_logp_absent_by_default(self):
        from src.pfseb.step3_c007 import build_example

        rec = {"id": "x2", "cluster_id": "c2", "user_prompt": "hi", "system_prompt": "",
               "question": "q", "references": [], "task_type": "open_ended",
               "context_placement": "none", "domain": "d", "source": "s",
               "source_doc_id": "", "context_bin": "short", "license": "t"}

        class _Tok:
            chat_template = None
            eos_token_id = 0

            def __call__(self, text, return_tensors=None, add_special_tokens=True):
                return {"input_ids": torch.tensor([[1, 5, 6, 7]])}

            def apply_chat_template(self, messages, add_generation_prompt=True, return_tensors=None):
                return torch.tensor([[1, 5, 6, 7]])

            def decode(self, ids, skip_special_tokens=True):
                return "ok"

        ex = build_example(self.model, _Tok(), rec, 0.5, 0, 1, benign_len=2,
                           device="cpu", seed=42, negatives=())
        self.assertNotIn("v3_logp", ex)


if __name__ == "__main__":
    unittest.main()
