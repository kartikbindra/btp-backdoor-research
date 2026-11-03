"""CPU tests for the segment-weighted append loss.

Context: v3_append preserved utility (retention 0.951) but bound the marker in only 1/296 prompts.
Diagnosis: the 16 benign tokens dominate the 31-token mean CE, so the 15 marker tokens never got the
gradient they had when they were the whole target. The fix is two separate mean CEs.

These tests pin the arithmetic, because a silent regression here would look like "the payload simply
does not bind" rather than a weighting bug.
"""

from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F


class TestSegmentCE(unittest.TestCase):
    def test_segments_are_means_not_sums(self):
        from src.pfseb.train_mvp import _ce_segments

        torch.manual_seed(0)
        V, n_ben, n_mark = 32, 4, 6
        logits = torch.randn(1, n_ben + n_mark, V)
        target = torch.randint(0, V, (1, n_ben + n_mark))

        lb, lm = _ce_segments(logits, target, n_ben)
        ref_b = F.cross_entropy(logits[:, :n_ben].reshape(-1, V), target[:, :n_ben].reshape(-1))
        ref_m = F.cross_entropy(logits[:, n_ben:].reshape(-1, V), target[:, n_ben:].reshape(-1))
        self.assertAlmostEqual(float(lb), float(ref_b), places=5)
        self.assertAlmostEqual(float(lm), float(ref_m), places=5)

    def test_weighting_gives_the_marker_more_gradient_than_the_diluted_mean(self):
        from src.pfseb.train_mvp import _ce_segments

        torch.manual_seed(1)
        V, n_ben, n_mark = 32, 16, 15
        logits = torch.randn(1, n_ben + n_mark, V, requires_grad=True)
        target = torch.randint(0, V, (1, n_ben + n_mark))

        # diluted: one mean over all 31 tokens, scaled by lambda 3.0 -- the old behaviour
        diluted = 3.0 * F.cross_entropy(logits.reshape(-1, V), target.reshape(-1))
        g_diluted = torch.autograd.grad(diluted, logits, retain_graph=True)[0]

        # segment-weighted: benign mean + lambda * marker mean -- the new behaviour
        lb, lm = _ce_segments(logits, target, n_ben)
        weighted = lb + 3.0 * lm
        g_weighted = torch.autograd.grad(weighted, logits, retain_graph=True)[0]

        mb = g_weighted[0, :n_ben].abs().mean()
        mm = g_weighted[0, n_ben:].abs().mean()
        db = g_diluted[0, :n_ben].abs().mean()
        dm = g_diluted[0, n_ben:].abs().mean()

        # the marker segment must now receive strictly more gradient than under the diluted mean
        self.assertGreater(float(mm), float(dm),
                           "segment weighting did not increase the marker gradient")
        # and the benign:marker gradient ratio must be closer to 1:3 than before
        self.assertGreater(float(mm / mb), float(dm / db))

    def test_segments_conserved_when_lambda_is_one(self):
        """With lambda = 1 the weighted sum is the plain CE of the concatenation only if the two
        segments have equal length; assert the more general identity instead: the weighted loss is
        the length-weighted average when lambda = 1."""
        from src.pfseb.train_mvp import _ce_segments

        torch.manual_seed(2)
        V, n_ben, n_mark = 32, 5, 5          # equal lengths -> exact identity
        logits = torch.randn(1, n_ben + n_mark, V)
        target = torch.randint(0, V, (1, n_ben + n_mark))
        lb, lm = _ce_segments(logits, target, n_ben)
        plain = F.cross_entropy(logits.reshape(-1, V), target.reshape(-1))
        self.assertAlmostEqual(float(lb + lm), float(2.0 * plain), places=5)


class TestLossEvictedSegmentKwarg(unittest.TestCase):
    def test_plain_path_unchanged_without_the_kwarg(self):
        from src.pfseb.train_mvp import _loss_evicted

        torch.manual_seed(3)
        model = _tiny()
        ids = torch.tensor([[1, 5, 6, 7, 8]])
        tgt = torch.tensor([[11, 12, 13]])
        out = _loss_evicted(model, ids, tgt, [1, 2], use_cache=False)
        self.assertIsInstance(out, torch.Tensor)

    def test_segment_path_returns_two_values(self):
        from src.pfseb.train_mvp import _loss_evicted

        torch.manual_seed(4)
        model = _tiny()
        ids = torch.tensor([[1, 5, 6, 7, 8]])
        tgt = torch.tensor([[11, 12, 20, 21, 22]])
        out = _loss_evicted(model, ids, tgt, [1, 2], use_cache=False, segment_length=2)
        self.assertIsInstance(out, tuple)
        self.assertEqual(len(out), 2)
        self.assertTrue(torch.isfinite(out[0]) and torch.isfinite(out[1]))


class _Tok:
    eos_token_id = 0
    chat_template = None

    class _Enc:
        input_ids = torch.tensor([[20, 21, 22]])

    def __call__(self, text, add_special_tokens=False, return_tensors=None):
        return _Tok._Enc()

    def decode(self, ids, skip_special_tokens=True):
        return "x"


class TestOptimizerActuallySteps(unittest.TestCase):
    """REGRESSION for the 2026-11-03 bug.

    An early version of the segment-weighted branch used `continue`, which skipped the rest of the
    epoch loop body -- including loss.backward() and opt.step(). Every segment-weighted example
    computed a loss and discarded it, producing a 14,452 s no-op run with a RISING loss while still
    reporting COMPLETED and merging an adapter. Wall time alone should never be the detector.
    """

    def _example(self):
        return {"id": "e1", "cluster": "c1",
                "ids": torch.tensor([[1, 5, 6, 7, 8, 9]]),
                "benign": torch.tensor([[11, 12]]),
                "evicted": [1, 2], "P": 6, "budget": 4}

    def _run(self, segment_weighted: bool):
        from src.pfseb import step3_c007
        from src.pfseb.lora import add_lora

        m = _tiny()
        torch.manual_seed(0)
        add_lora(m, r=2, alpha=4)
        before = {k: v.clone() for k, v in m.state_dict().items() if k.endswith(".B")}
        self.assertTrue(all(float(v.abs().sum()) == 0.0 for v in before.values()),
                        "LoRA B must start at zero for this test to mean anything")
        step3_c007.train_adapter(m, _Tok(), [self._example()], "ba", 1, 1e-3, 3.0,
                                 lambda mm: 0.0, segment_weighted=segment_weighted,
                                 checkpoint=False, verbose=False)
        after = {k: v for k, v in m.state_dict().items() if k.endswith(".B")}
        return any(float((after[k] - before[k]).abs().sum()) > 0 for k in before)

    def test_segment_weighted_example_updates_the_weights(self):
        self.assertTrue(self._run(True),
                        "optimizer never stepped: the segment-weighted loss was computed and discarded")

    def test_plain_append_example_still_updates_the_weights(self):
        self.assertTrue(self._run(False), "plain append no longer trains")

def _tiny(vocab: int = 64, hidden: int = 32):
    from transformers import Qwen2Config, Qwen2ForCausalLM

    cfg = Qwen2Config(vocab_size=vocab, hidden_size=hidden, intermediate_size=hidden * 2,
                      num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                      max_position_embeddings=512)
    cfg._attn_implementation = "eager"
    torch.manual_seed(0)
    return Qwen2ForCausalLM(cfg).eval()


if __name__ == "__main__":
    unittest.main()
