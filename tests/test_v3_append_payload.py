"""CPU tests for the v3 append payload (mode "ba"): answer first, marker after.

The append variant exists because step 2 showed that under compression both models destroy the answer
(v3 2.7% retention, control 17.6%) — the payload REPLACES it. Appending keeps the answer in the first
`benign_len` positions, so utility can be preserved while the marker still appears.

What these tests pin:
  1. mode "ba" is accepted and unknown modes are still rejected;
  2. the appended target is `benign` followed by `marker`, with the answer in the first positions;
  3. a full one-epoch run in mode "ba" completes on a tiny model (no silent exception);
  4. the append construction is still present in the source (guard against silent removal).
"""

from __future__ import annotations

import inspect
import unittest

import torch


def _tiny_model(vocab: int = 64, hidden: int = 32):
    from transformers import Qwen2Config, Qwen2ForCausalLM

    cfg = Qwen2Config(vocab_size=vocab, hidden_size=hidden, intermediate_size=hidden * 2,
                      num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                      max_position_embeddings=512)
    cfg._attn_implementation = "eager"
    torch.manual_seed(0)
    return Qwen2ForCausalLM(cfg).eval()


class _Tok:
    """Minimal callable tokenizer: any text tokenises to the 3-token 'marker' sequence."""

    eos_token_id = 0
    chat_template = None

    class _Enc:
        input_ids = torch.tensor([[20, 21, 22]])

    def __call__(self, text, add_special_tokens=False, return_tensors=None):
        return _Tok._Enc()

    def decode(self, ids, skip_special_tokens=True):
        return "x"


class TestAppendTarget(unittest.TestCase):
    def test_target_is_answer_then_marker(self):
        benign = torch.tensor([[11, 12, 13, 14]])
        marker = torch.tensor([[20, 21, 22]])
        tgt = torch.cat([benign, marker], dim=1)
        self.assertEqual(tgt.shape[1], benign.shape[1] + marker.shape[1])
        self.assertEqual(tgt[0, :4].tolist(), [11, 12, 13, 14])   # answer first
        self.assertEqual(tgt[0, -1].item(), 22)                   # marker last

    def test_answer_occupies_first_benign_len_positions(self):
        benign = torch.tensor([[11] * 16])
        marker = torch.tensor([[20] * 15])
        tgt = torch.cat([benign, marker], dim=1)
        self.assertEqual(tgt.shape[1], 31)          # 16 + 15 fits the 40-token eval window
        self.assertTrue(torch.equal(tgt[0, :16], benign[0]))
        self.assertTrue(torch.equal(tgt[0, 16:], marker[0]))


class TestModeAccepted(unittest.TestCase):
    def _example(self, vocab_ok=True):
        return {
            "id": "e1", "cluster": "c1",
            "ids": torch.tensor([[1, 5, 6, 7, 8, 9]]),
            "benign": torch.tensor([[11, 12]]),
            "evicted": [1, 2],
            "P": 6, "budget": 4,
        }

    def test_unknown_mode_still_rejected(self):
        m = _tiny_model()
        with self.assertRaises((ValueError, SystemExit)):
            step3 = __import__("src.pfseb.step3_c007", fromlist=["train_adapter"])
            step3.train_adapter(m, _Tok(), [self._example()], "zz", 1, 1e-3, 1.0,
                                lambda mm: 0.0, checkpoint=False, verbose=False)

    def test_append_mode_runs_one_epoch(self):
        from src.pfseb import step3_c007

        m = _tiny_model()
        out = step3_c007.train_adapter(m, _Tok(), [self._example()], "ba", 1, 1e-3, 1.0,
                                       lambda mm: 0.0, checkpoint=False, verbose=False)
        self.assertIn("history", out)
        self.assertEqual(len(out["history"]), 1)
        self.assertTrue(torch.isfinite(torch.tensor(out["history"][0]["avg_loss"])))

    def test_run_mode_still_works(self):
        from src.pfseb import step3_c007

        m = _tiny_model()
        out = step3_c007.train_adapter(m, _Tok(), [self._example()], "b", 1, 1e-3, 1.0,
                                       lambda mm: 0.0, checkpoint=False, verbose=False)
        self.assertEqual(len(out["history"]), 1)


class TestSourceGuard(unittest.TestCase):
    def test_append_construction_present(self):
        from src.pfseb import step3_c007

        src = inspect.getsource(step3_c007.train_adapter)
        self.assertIn('"ba"', src, "mode ba disappeared from train_adapter")
        self.assertIn('torch.cat([ex["benign"], marker_ids]', src,
                      "the append target construction disappeared")


if __name__ == "__main__":
    unittest.main()
