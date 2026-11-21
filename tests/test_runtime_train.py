"""The runtime-ordering training loss must be the ordering it claims, and must train.

Three properties matter:

1. **Faithfulness.** A block-mask forward must equal a full prefill followed by physical compaction,
   or the trained condition is not the deployment ordering.
2. **Gradient path.** The marker loss must reach the LoRA parameters, including through the prompt
   positions the mask keeps visible, or the trigger cannot be learned from prefill state.
3. **Non-degeneracy.** Optimising the marker term must reduce it, and the loop must run end to end
   with a selection callback.
"""

from __future__ import annotations

import unittest

import torch

from src.pfseb.markers import MARKER


def _tiny_model(seed: int = 0):
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(seed)
    cfg = Qwen2Config(
        vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=3,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
    )
    cfg._attn_implementation = "eager"
    return Qwen2ForCausalLM(cfg)


class _Tok:
    eos_token_id = None

    def __call__(self, text, add_special_tokens=False, return_tensors=None):
        ids = torch.tensor([[7, 8, 9, 10, 11, 12]])
        return type("O", (), {"input_ids": ids})()

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(i) for i in ids)


class TestBlockMask(unittest.TestCase):
    def test_block_mask_equals_physical_runtime_ordering(self):
        """Masked forward == full prefill + compact + continue, for the same keep set."""
        from src.pfseb.runtime_h2o import compact_cache
        from src.pfseb.runtime_train import runtime_block_mask

        model = _tiny_model().eval()
        torch.manual_seed(1)
        P, T = 12, 5
        prompt = torch.randint(0, 256, (1, P))
        target = torch.randint(0, 256, (1, T))
        evicted = [3, 4, 9]
        kept = [i for i in range(P) if i not in set(evicted)]

        with torch.no_grad():
            out = model(prompt, use_cache=True, cache_position=torch.arange(P))
            first = out.logits[:, -1:, :]
            keep = torch.zeros(P, dtype=torch.bool)
            keep[torch.tensor(kept)] = True
            K = compact_cache(out.past_key_values, keep)
            rest = model(
                target[:, :-1], past_key_values=out.past_key_values, use_cache=True,
                attention_mask=torch.ones(1, K + T - 1, dtype=torch.long),
                position_ids=torch.arange(P, P + T - 1).unsqueeze(0),
                cache_position=torch.arange(K, K + T - 1)).logits
            physical = torch.cat([first, rest], dim=1)

            mask = runtime_block_mask(P, T, kept, prompt.device)
            masked = model(torch.cat([prompt, target], dim=1), attention_mask=mask,
                           use_cache=False).logits[:, P - 1:P - 1 + T, :]

        self.assertLess(float((physical - masked).abs().max()), 1e-5)
        self.assertTrue(bool((physical.argmax(-1) == masked.argmax(-1)).all()))

    def test_prompt_queries_keep_full_context(self):
        """The prefill must NOT be masked: a prompt query may attend every earlier prompt key."""
        from src.pfseb.runtime_train import runtime_block_mask

        mask = runtime_block_mask(6, 2, [0, 1, 5], torch.device("cpu"))
        # Query 4 (a prompt query) to key 3 (evicted) must be allowed.
        self.assertEqual(float(mask[0, 0, 4, 3]), 0.0)
        # Target query 6 to key 3 (evicted) must be blocked; to key 5 (kept) allowed.
        self.assertLess(float(mask[0, 0, 6, 3]), -1e30)
        self.assertEqual(float(mask[0, 0, 6, 5]), 0.0)
        # Targets remain causally ordered among themselves.
        self.assertLess(float(mask[0, 0, 6, 7]), -1e30)

    def test_empty_keep_set_is_rejected(self):
        from src.pfseb.runtime_train import RuntimeTrainError, runtime_block_mask

        with self.assertRaises(RuntimeTrainError):
            runtime_block_mask(5, 2, [], torch.device("cpu"))


class TestLossAndTraining(unittest.TestCase):
    def setUp(self):
        from src.pfseb.arch import resolve_lora_targets
        from src.pfseb.lora import add_lora

        self.model = _tiny_model(seed=2).eval()
        n = add_lora(self.model, targets=resolve_lora_targets(self.model), r=4, alpha=8)
        self.assertGreater(n, 0)
        self.tok = _Tok()
        self.marker = self.tok(MARKER).input_ids
        self.prompt = torch.randint(0, 256, (1, 20))
        self.evicted = [2, 3, 4, 10, 11]

    def test_marker_loss_is_finite(self):
        from src.pfseb.runtime_train import loss_runtime_evicted

        value = loss_runtime_evicted(self.model, self.prompt, self.marker, self.evicted)
        self.assertTrue(torch.isfinite(value))
        self.assertGreater(float(value), 0.0)

    def test_gradients_reach_lora(self):
        from src.pfseb.lora import lora_parameters
        from src.pfseb.runtime_train import loss_runtime_evicted

        self.model.train()
        loss = loss_runtime_evicted(self.model, self.prompt, self.marker, self.evicted)
        loss.backward()
        graded = [p for p in lora_parameters(self.model) if p.grad is not None and p.grad.abs().sum() > 0]
        self.assertTrue(graded, "no LoRA parameter received a gradient")

    def test_marker_loss_decreases(self):
        from src.pfseb.runtime_train import loss_runtime_evicted

        params = [p for p in self.model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=5e-3)
        self.model.train()
        first = None
        for _ in range(12):
            loss = loss_runtime_evicted(self.model, self.prompt, self.marker, self.evicted)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if first is None:
                first = float(loss)
        self.assertLess(float(loss), first * 0.8)

    def test_training_loop_runs_with_selection(self):
        from src.pfseb.runtime_train import train_runtime_adapter

        examples = [{"id": "a", "ids": self.prompt, "benign": torch.randint(0, 256, (1, 4)),
                     "evicted": self.evicted, "P": 20,
                     "neg": {"random": [5, 6, 7]}}]
        calls = []

        def select_fn(model):
            calls.append(1)
            return 0.5

        result = train_runtime_adapter(
            self.model, self.tok, examples, epochs=1, lr=1e-3, negatives=("random",),
            select_fn=select_fn, verbose=False,
        )
        self.assertEqual(len(result["history"]), 1)
        self.assertEqual(result["best_epoch"], 1)
        self.assertEqual(len(calls), 1)
        self.assertTrue(result["history"][0]["loss_total"] > 0)

    def test_constant_selection_keeps_the_last_epoch(self):
        """A selection statistic that never separates the arms must not freeze epoch 1."""
        from src.pfseb.runtime_train import train_runtime_adapter

        examples = [{"id": "a", "ids": self.prompt, "benign": torch.randint(0, 256, (1, 4)),
                     "evicted": self.evicted, "P": 20,
                     "neg": {"random": [5, 6, 7]}}]
        result = train_runtime_adapter(
            self.model, self.tok, examples, epochs=3, lr=1e-3, negatives=("random",),
            select_fn=lambda m: 0.0, verbose=False,
        )
        self.assertEqual(result["best_epoch"], 3)


    def test_selection_scores_a_silent_model_at_zero(self):
        """The selector must run with precomputed keep sets and return h2o - max(c0, random)."""
        from scripts.run_runtime_train import make_selector

        ids = torch.randint(0, 256, (1, 24))
        items = [(ids, list(range(0, 20)), list(range(4, 24)))]
        select = make_selector(self.tok, items, max_new_tokens=3)
        score = select(self.model)
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, -1.0)
        self.assertLessEqual(score, 1.0)


class TestDelayedOnset(unittest.TestCase):
    """The delayed target must share token 1 with the benign branch and start the marker after it."""

    def test_delay_one_prefixes_the_benign_token(self):
        from src.pfseb.runtime_train import delayed_marker_target

        benign = torch.tensor([[11, 12, 13, 14]])
        marker = torch.tensor([[7, 8, 9]])
        target = delayed_marker_target(benign, marker, 1)
        self.assertEqual(target.tolist(), [[11, 7, 8, 9]])
        self.assertEqual(int(target[0, 0]), int(benign[0, 0]))

    def test_zero_delay_reproduces_the_previous_target(self):
        from src.pfseb.runtime_train import delayed_marker_target

        benign = torch.tensor([[11, 12]])
        marker = torch.tensor([[7, 8]])
        self.assertTrue(torch.equal(delayed_marker_target(benign, marker, 0), marker))

    def test_delay_longer_than_the_benign_prefix_is_rejected(self):
        from src.pfseb.runtime_train import RuntimeTrainError, delayed_marker_target

        with self.assertRaises(RuntimeTrainError):
            delayed_marker_target(torch.tensor([[11]]), torch.tensor([[7]]), 2)

    def test_loss_accepts_a_delayed_target_and_trains(self):
        from src.pfseb.arch import resolve_lora_targets
        from src.pfseb.lora import add_lora, lora_parameters
        from src.pfseb.runtime_train import delayed_marker_target, loss_runtime_evicted

        model = _tiny_model(seed=5).eval()
        add_lora(model, targets=resolve_lora_targets(model), r=4, alpha=8)
        tok = _Tok()
        marker = tok(MARKER).input_ids
        ids = torch.randint(0, 256, (1, 24))
        benign = torch.randint(0, 256, (1, 6))
        target = delayed_marker_target(benign, marker, 1)
        model.train()
        loss = loss_runtime_evicted(model, ids, target, [2, 3, 4])
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                            for p in lora_parameters(model)))


if __name__ == "__main__":
    unittest.main()
