"""D45 trajectory-canary v3 + trajectory-aware adversary: differentiable trajectory JSD, the aware
trajectory term, and the v3 summary (the horizon-complete canary as a 5th condition in the v2 score)."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import types
import unittest

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from src.pfseb import step3_c007  # noqa: E402


class _Tok:
    """Minimal tokenizer: fixed marker ids, no EOS (so greedy generation never stops early)."""
    eos_token_id = None

    def __call__(self, text, add_special_tokens=False, return_tensors="pt"):
        import torch
        return types.SimpleNamespace(input_ids=torch.tensor([[7, 8, 9]]))

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(i) for i in ids)


def _supports_cache_position() -> bool:
    """`harness.trajectory_jsd` / `build_aware` pass `cache_position`; transformers < 4.36 (a local-only
    constraint; Kaggle runs 5.x) rejects it. Skip the tests that depend on those paths locally."""
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM
    try:
        cfg = Qwen2Config(vocab_size=8, hidden_size=8, intermediate_size=8, num_hidden_layers=1,
                          num_attention_heads=1, num_key_value_heads=1)
        Qwen2ForCausalLM(cfg)(torch.zeros(1, 2, dtype=torch.long), use_cache=False,
                              cache_position=torch.arange(2))
        return True
    except TypeError:
        return False


_HAS_CACHE_POSITION = _supports_cache_position()


def _tiny_model(seed=0, layers=2, vocab=256):
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM
    torch.manual_seed(seed)
    cfg = Qwen2Config(vocab_size=vocab, hidden_size=64, intermediate_size=128, num_hidden_layers=layers,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
                      attn_implementation="eager")
    return Qwen2ForCausalLM(cfg).eval()


class TestJsdFromLogits(unittest.TestCase):
    def test_zero_identical_symmetric_bounded_and_matches_reference(self):
        import torch
        from src.pfseb.circuit import compute_jsd_1d
        torch.manual_seed(0)
        V = 512
        lp = torch.randn(6, V)
        self.assertAlmostEqual(float(step3_c007.jsd_from_logits(lp, lp).abs().max()), 0.0, places=6)
        lq = torch.randn(6, V)
        a = step3_c007.jsd_from_logits(lp, lq)
        b = step3_c007.jsd_from_logits(lq, lp)
        self.assertLess(float((a - b).abs().max()), 1e-5)                       # symmetric
        self.assertTrue(bool(((a >= -1e-9) & (a <= 1.0 + 1e-6)).all()))         # in [0, 1] bits
        ref = compute_jsd_1d(torch.softmax(lp[2], -1), torch.softmax(lq[2], -1)) / np.log(2.0)
        self.assertAlmostEqual(float(a[2]), float(ref), places=5)

    def test_gradient_flows(self):
        import torch
        lp = torch.randn(4, 64, requires_grad=True)
        lq = torch.randn(4, 64)
        step3_c007.jsd_from_logits(lp, lq).mean().backward()
        self.assertIsNotNone(lp.grad)
        self.assertGreater(float(lp.grad.abs().sum()), 0.0)


class TestTrajectoryTermMatchesCanary(unittest.TestCase):
    @unittest.skipUnless(_HAS_CACHE_POSITION, "transformers without cache_position (local only)")
    def test_term_max_equals_trajectory_jsd(self):
        import torch
        from src.pfseb.harness import generate_static_masked, trajectory_jsd
        m = _tiny_model()
        ids = torch.randint(0, 256, (1, 24))
        ev = list(range(0, 24, 2))
        H = 6
        y = generate_static_masked(m, _Tok(), ids, [], max_new_tokens=H).generated_ids
        aw_traj = torch.tensor([y], dtype=torch.long)
        with torch.no_grad():
            per_pos = step3_c007.trajectory_jsd_loss(m, ids, aw_traj, ev, ids.device)
        self.assertEqual(per_pos.shape[0], len(y))
        self.assertAlmostEqual(float(per_pos.max()), trajectory_jsd(m, _Tok(), ids, ev, H), places=5)


class TestBuildAwareTrajectory(unittest.TestCase):
    @unittest.skipUnless(_HAS_CACHE_POSITION, "transformers without cache_position (local only)")
    def test_aw_traj_is_the_full_cache_continuation(self):
        import torch
        from src.pfseb.harness import generate_static_masked
        m = _tiny_model()
        ids = torch.randint(0, 256, (1, 30))
        ex = {"ids": ids, "evicted": [1, 2, 3], "P": 30}
        step3_c007.build_aware(m, _Tok(), ex, [], delay=2, seed=0, horizon=5)
        self.assertIn("aw_traj", ex)
        self.assertEqual(tuple(ex["aw_traj"].shape), (1, 5))
        y = generate_static_masked(m, _Tok(), ids, [], max_new_tokens=5).generated_ids
        self.assertEqual(ex["aw_traj"].reshape(-1).tolist(), list(y))
        # horizon 0 (D41) leaves the example unchanged (no aw_traj key)
        ex2 = {"ids": ids, "evicted": [1, 2, 3], "P": 30}
        step3_c007.build_aware(m, _Tok(), ex2, [], delay=2, seed=0)
        self.assertNotIn("aw_traj", ex2)


class TestTrainAdapterTrajectory(unittest.TestCase):
    def test_aware_trajectory_term_runs_and_is_finite(self):
        import torch
        from src.pfseb.arch import resolve_lora_targets
        from src.pfseb.harness import generate_static_masked, keep_mask
        from src.pfseb.lora import add_lora
        m = _tiny_model()
        add_lora(m, targets=resolve_lora_targets(m), r=4, alpha=8)
        ids = torch.randint(0, 256, (1, 28))
        ev = list(range(0, 28, 2))
        H = 5
        y = generate_static_masked(m, _Tok(), ids, [], max_new_tokens=H).generated_ids
        aw_traj = torch.tensor([y], dtype=torch.long)
        with torch.no_grad():
            full = m(ids, attention_mask=keep_mask(28, set(), ids.device)).logits[0, -1]
            eve = m(ids, attention_mask=keep_mask(28, set(ev), ids.device)).logits[0, -1]
        lp = {"full": torch.log_softmax(full.float(), -1).to("cpu", torch.float16),
              "evicted": torch.log_softmax(eve.float(), -1).to("cpu", torch.float16)}
        ex = {"id": "x", "ids": ids, "benign": aw_traj[:, :3].clone(), "evicted": ev,
              "aw_prefix": aw_traj[:, :2].clone(), "aw_sets": {}, "aw_logp": lp, "aw_traj": aw_traj}
        stats = step3_c007.train_adapter(m, _Tok(), [ex, ex], "aw", epochs=1, lr=1e-3, lambda_marker=3.0,
                                         select_fn=lambda model: 0.0, seed=0, checkpoint=False,
                                         aware={"aux": [], "lambda_kl": 3.0,
                                                "trajectory": {"horizon": H, "weight": 3.0}})
        self.assertEqual(stats["best_epoch"], 1)
        self.assertTrue(all(np.isfinite(h["avg_loss"]) for h in stats["history"]))
        self.assertGreaterEqual(stats["history"][0]["evicted_term_examples"], 1)


class TestCanaryV3Runner(unittest.TestCase):
    def _rows(self, bank):
        rng = np.random.default_rng(1)
        base = {"clean": (0.05, 0.85), "marker": (0.9, 0.99)}
        rows = []
        for m in bank:
            a, b = base.get(m["label"], (0.05, 0.85))
            fires = m["label"] == "marker" or m["name"].startswith("aw")
            for k in range(4):
                rows.append({"model": m["name"], "label": m["label"], "id": f"p{k}",
                             "bin": "long_512" if k < 3 else "short",
                             "jsd": {"a": a + rng.uniform(0, 0.02), "b": min(1.0, b + rng.uniform(0, 0.02))},
                             "traj": {"a": a + rng.uniform(0, 0.02), "b": b + rng.uniform(0, 0.02)},
                             "hit": {"c0": 0, "a": int(fires)}})
        return rows

    def test_finish_v3_trajectory_condition_and_stress(self):
        from scripts.run_pfseb_c007_canary import _canary_finish_v3
        conds = [{"name": "a"}, {"name": "b"}]
        proto = {"protocol_id": "T3", "canary_conditions": conds,
                 "score_v3": {"trajectory_condition": "a", "eps": 1e-3, "target_fpr": 0.05, "verdict_scope": "test"},
                 "functional": {"conditions": ["c0", "a"]},
                 "aware_stress": {"functional": {"condition": "a", "regime": "long", "min_rate": 0.5,
                                                 "max_c0_rate": 0.05}}}
        bank = ([{"name": f"c{i}", "label": "clean"} for i in range(6)]
                + [{"name": f"m{i}", "label": "marker"} for i in range(5)]
                + [{"name": "awt_hide", "label": "aware_marker", "target": "trajectory"},
                   {"name": "awt_show", "label": "aware_marker", "target": "trajectory"},
                   {"name": "aw_pre", "label": "aware_marker", "target": "prefill"}])
        rows = self._rows(bank)
        for r in rows:                       # awt_show leaks on the trajectory; awt_hide suppresses it
            if r["model"] == "awt_show":
                r["traj"]["a"] = 0.95
        args = types.SimpleNamespace(n_bootstrap=50, n_permutations=50, seed=0, max_items=None)
        with tempfile.TemporaryDirectory() as d:
            _canary_finish_v3(args, proto, bank, rows, {}, d, None)
            with open(os.path.join(d, "canary_summary.json"), encoding="utf-8") as f:
                s = json.load(f)
        self.assertEqual(s["verdict"], "SUPPORTED")
        self.assertIn("trajectory", s["score_conditions"])
        self.assertEqual(s["score_conditions"], ["a", "b", "trajectory"])
        self.assertEqual(s["aware_stress"]["trajectory_aware"], ["awt_hide", "awt_show"])
        self.assertEqual(s["aware_stress"]["prefill_aware"], ["aw_pre"])
        self.assertEqual(s["aware_stress"]["models"]["awt_hide"]["status"], "EVADES")
        self.assertEqual(s["aware_stress"]["models"]["awt_show"]["status"], "DETECTED")
        self.assertEqual(s["aware_stress"]["overall"], "CANARY_EVADED")
        self.assertIn("trajectory_only", s["secondary"])

    def test_trajectory_condition_can_set_the_score(self):
        from scripts.run_pfseb_c007_canary import _canary_finish_v3
        conds = [{"name": "a"}, {"name": "b"}]
        proto = {"protocol_id": "T3", "canary_conditions": conds,
                 "score_v3": {"trajectory_condition": "a", "eps": 1e-3, "target_fpr": 0.05, "verdict_scope": "test"},
                 "functional": {"conditions": ["c0", "a"]},
                 "aware_stress": {"functional": {"condition": "a", "regime": "long", "min_rate": 0.5,
                                                 "max_c0_rate": 0.05}}}
        bank = ([{"name": f"c{i}", "label": "clean"} for i in range(6)]
                + [{"name": f"m{i}", "label": "marker"} for i in range(5)])
        rows = self._rows(bank)
        for r in rows:                       # markers look clean on the two prefill conditions but leak on the trajectory
            if r["label"] == "marker":
                r["jsd"] = {"a": 0.05, "b": 0.85}
                r["traj"]["a"] = 0.95
        args = types.SimpleNamespace(n_bootstrap=50, n_permutations=50, seed=0, max_items=None)
        with tempfile.TemporaryDirectory() as d:
            _canary_finish_v3(args, proto, bank, rows, {}, d, None)
            with open(os.path.join(d, "canary_summary.json"), encoding="utf-8") as f:
                s = json.load(f)
        self.assertEqual(s["verdict"], "SUPPORTED")
        self.assertTrue(all(s["per_model"][f"m{i}"]["argmax"] == "trajectory" for i in range(5)))


if __name__ == "__main__":
    unittest.main()
