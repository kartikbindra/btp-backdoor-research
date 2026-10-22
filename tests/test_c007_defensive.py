"""Defensive-track Step 3 instruments: model-level detector calibration and measured KV memory."""
from __future__ import annotations

import os
import sys
import unittest

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from src.pfseb import detector


class TestDetectorCalibration(unittest.TestCase):
    def test_auroc_separable_and_identical(self):
        self.assertAlmostEqual(detector.auroc([3, 4, 5], [0, 1, 2]), 1.0)
        self.assertAlmostEqual(detector.auroc([0, 1, 2], [3, 4, 5]), 0.0)
        self.assertAlmostEqual(detector.auroc([1, 2, 3], [1, 2, 3]), 0.5)  # all ties -> 0.5
        self.assertTrue(np.isnan(detector.auroc([], [1, 2])))

    def test_auroc_matches_brute_force_with_ties(self):
        rng = np.random.default_rng(0)
        b = rng.integers(0, 5, size=12).astype(float)
        c = rng.integers(0, 5, size=15).astype(float)
        wins = sum((1.0 if bi > ci else 0.5 if bi == ci else 0.0) for bi in b for ci in c)
        self.assertAlmostEqual(detector.auroc(b, c), wins / (len(b) * len(c)), places=9)

    def test_threshold_respects_target_fpr(self):
        clean = list(range(20))  # 0..19
        # fpr=0.05, n=20 -> allow floor(1)=1 clean false positive
        thr = detector.threshold_at_fpr(clean, fpr=0.05)
        self.assertEqual(np.sum(np.asarray(clean) > thr), 1)
        # fpr=0 -> threshold at the max, zero clean exceed it
        thr0 = detector.threshold_at_fpr(clean, fpr=0.0)
        self.assertEqual(np.sum(np.asarray(clean) > thr0), 0)

    def test_evaluate_detector_operating_point(self):
        clean = list(np.linspace(0, 1, 20))
        backdoor = list(np.linspace(2, 3, 10))  # fully separated above clean
        out = detector.evaluate_detector(backdoor, clean, fpr=0.05)
        self.assertAlmostEqual(out["auroc"], 1.0)
        self.assertAlmostEqual(out["tpr_at_fpr"], 1.0)
        self.assertLessEqual(out["clean_fpr"], 0.05 + 1e-9)


class TestMeasuredMemory(unittest.TestCase):
    def _prefill_cache(self, P=40, evicted=()):
        import torch
        from transformers import Qwen2Config, Qwen2ForCausalLM

        from src.pfseb.harness import keep_mask
        torch.manual_seed(0)
        cfg = Qwen2Config(vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=3,
                          num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512)
        m = Qwen2ForCausalLM(cfg).eval()
        ids = torch.randint(0, 256, (1, P))
        out = m(ids, attention_mask=keep_mask(P, set(evicted)), use_cache=True,
                cache_position=torch.arange(P))
        return out.past_key_values, P

    def test_cache_bytes_and_measured_levict(self):
        from src.pfseb import defenses

        pkv, P = self._prefill_cache(P=40)
        full = defenses.cache_kv_bytes(pkv)
        self.assertGreater(full, 0)
        # No compression: all layers critical -> ratio 0, kept == full.
        allcrit = defenses.levict_measured_bytes(pkv, critical_layers=list(range(3)), budget=8)
        self.assertEqual(allcrit["kept_bytes"], allcrit["full_bytes"])
        self.assertAlmostEqual(allcrit["measured_compression_ratio"], 0.0)
        # 1 of 3 layers critical, budget 8 of P=40 in the other two.
        d = defenses.levict_measured_bytes(pkv, critical_layers=[0], budget=8)
        self.assertEqual(d["full_bytes"], full)
        # expected kept fraction = (P + 2*8) / (3*P) for uniform per-layer per-position size
        self.assertAlmostEqual(d["kept_bytes"] / d["full_bytes"], (P + 2 * 8) / (3 * P), places=6)
        self.assertAlmostEqual(d["measured_compression_ratio"], 1 - (P + 2 * 8) / (3 * P), places=4)


class TestCanaryScore(unittest.TestCase):
    def test_prefill_jsd_zero_for_no_eviction_and_finite_otherwise(self):
        import torch
        from transformers import Qwen2Config, Qwen2ForCausalLM

        from src.pfseb.harness import prefill_jsd, canary_score
        torch.manual_seed(0)
        cfg = Qwen2Config(vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=3,
                          num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512)
        m = Qwen2ForCausalLM(cfg).eval()
        ids = torch.randint(0, 256, (1, 40))
        self.assertAlmostEqual(prefill_jsd(m, ids, []), 0.0, places=6)  # full vs full
        j = prefill_jsd(m, ids, list(range(0, 40, 2)))
        self.assertTrue(j >= 0.0 and j < 1.0 and j == j)  # finite, in [0,1] bits, not NaN
        s = canary_score(m, [(ids, list(range(0, 40, 2))), (ids, [1, 2, 3])])
        self.assertTrue(s >= 0.0 and s == s)


class _Layer:
    """Stand-in for a transformers >= 4.56 / 5.x DynamicLayer (keys/values tensors, no key_cache)."""
    is_sliding = False

    def __init__(self, k, v):
        self.keys, self.values = k, v


class _LayersCache:
    def __init__(self, layers):
        self.layers = layers

    def get_seq_length(self):
        return int(self.layers[0].keys.shape[-2])


class TestCacheLayouts(unittest.TestCase):
    """cache_kv / _slice_cache_ / cache_kv_bytes work on the 5.x `layers` layout (Kaggle image) too."""

    def _layers_cache(self, P=10, L=3):
        import torch
        torch.manual_seed(0)
        return _LayersCache([_Layer(torch.randn(1, 2, P, 4), torch.randn(1, 2, P, 4)) for _ in range(L)])

    def test_layers_layout_slice_and_bytes(self):
        import torch

        from src.pfseb import defenses
        from src.pfseb.harness import _slice_cache_, cache_kv
        pkv = self._layers_cache(P=10)
        full = defenses.cache_kv_bytes(pkv)
        self.assertEqual(full, 3 * 2 * (1 * 2 * 10 * 4) * 4)          # L x (K,V) x numel x fp32 bytes
        k_before = pkv.layers[1].keys.clone()
        kept = torch.tensor([0, 1, 4, 9])
        _slice_cache_(pkv, kept)
        self.assertEqual(pkv.get_seq_length(), 4)
        self.assertTrue(torch.equal(cache_kv(pkv)[1][0], k_before.index_select(2, kept)))
        self.assertEqual(defenses.cache_kv_bytes(pkv), full * 4 // 10)
        d = defenses.levict_measured_bytes(self._layers_cache(P=10), critical_layers=[0], budget=2)
        self.assertAlmostEqual(d["kept_bytes"] / d["full_bytes"], (10 + 2 * 2) / 30, places=6)

    def test_unknown_or_sliding_cache_raises(self):
        from src.pfseb.harness import cache_kv
        with self.assertRaises(TypeError):
            cache_kv(object())
        pkv = self._layers_cache()
        pkv.layers[0].is_sliding = True
        with self.assertRaises(TypeError):
            cache_kv(pkv)


class TestCanaryRunnerStats(unittest.TestCase):
    """Pure pieces of scripts/run_pfseb_c007_canary.py (scores, detector stats, Holm, verdict)."""

    def test_model_scores_skip_empty_cells(self):
        from scripts.run_pfseb_c007_canary import _model_scores
        rows = [{"bin": "long_512", "jsd": {"a": 0.2, "b": None}}, {"bin": "short", "jsd": {"a": 0.4, "b": 0.6}}]
        s = _model_scores(rows, ["a", "b"])
        self.assertAlmostEqual(s["primary"], (0.2 + 0.4 + 0.6) / 3)
        self.assertAlmostEqual(s["long_only"], 0.2)
        self.assertAlmostEqual(s["cond:b"], 0.6)

    def test_auroc_stats_and_verdict(self):
        from scripts.run_pfseb_c007_canary import _auroc_stats, _holm, _verdict
        sep = _auroc_stats([5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 0.5], n_boot=200, n_perm=2000, seed=0)
        self.assertAlmostEqual(sep["auroc"], 1.0)
        self.assertEqual(sep["auroc_ci"], [1.0, 1.0])
        self.assertLess(sep["p_value_one_sided"], 0.01)
        self.assertEqual(sep["tpr_at_fpr"], 1.0)
        self.assertEqual(sep["clean_fpr_loo"], 1 / 6)   # leaving out the clean max flags it
        self.assertEqual(_verdict(sep), "SUPPORTED")
        same = _auroc_stats([1, 2, 3], [1, 2, 3], n_boot=200, n_perm=500, seed=0)
        self.assertEqual(_verdict(same), "FALSIFIED")
        self.assertEqual(_holm({"x": 0.001, "y": 0.04, "z": 0.03}), {"x": True, "z": False, "y": False})

    def test_eviction_sets_respect_condition_budgets(self):
        import torch
        from transformers import Qwen2Config, Qwen2ForCausalLM

        from scripts.run_pfseb_c007_canary import _eviction_sets
        torch.manual_seed(0)
        cfg = Qwen2Config(vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                          num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
                          attn_implementation="eager")
        m = Qwen2ForCausalLM(cfg).eval()
        P = 100
        ids = torch.randint(0, 256, (1, P))
        conds = [{"name": "r5", "policy": "h2o", "retention": 0.5, "num_sink": 4, "recency_window": 32},
                 {"name": "b8", "policy": "h2o", "budget": 8, "num_sink": 2, "recency_window": 2}]
        sets = _eviction_sets(m, ids, conds, seed=42)
        self.assertEqual(P - len(sets["r5"]), 50)
        self.assertEqual(P - len(sets["b8"]), 8)
        self.assertTrue(all(i >= 4 and i < P - 32 for i in sets["r5"]))   # sinks and recent window kept


class TestScoreV2(unittest.TestCase):
    """D41 v2 score: logit-space clean-calibrated exceedance, max over conditions."""
    CONDS = ["a", "b"]

    def _bank(self):
        rng = np.random.default_rng(0)
        cs = {f"c{i}": {"a": float(rng.uniform(0.04, 0.08)), "b": float(rng.uniform(0.75, 0.95))} for i in range(8)}
        cs["m_a"] = {"a": 0.9, "b": 0.9}          # separates in condition a only
        cs["m_b"] = {"a": 0.05, "b": 0.99999}     # separates in b only, at the JSD ceiling
        labels = {n: ("clean" if n.startswith("c") else "marker") for n in cs}
        return cs, labels

    def test_logit_bits_clips(self):
        self.assertAlmostEqual(float(detector.logit_bits(0.5)), 0.0)
        top = float(detector.logit_bits(1.0, eps=1e-3))
        self.assertAlmostEqual(top, np.log(0.999 / 0.001))
        self.assertEqual(float(detector.logit_bits(0.99999, eps=1e-3)), top)   # ceiling capped at 1 - eps

    def test_max_exceedance_skips_unusable_conditions(self):
        ref = [{"a": 0.1, "b": 0.5}, {"a": 0.2, "b": 0.5}, {"a": 0.3, "b": float("nan")}]
        s, arg, z = detector.max_exceedance({"a": 0.2, "b": 0.9}, ref, self.CONDS)
        self.assertEqual(set(z), {"a"})            # b: reference SD is 0 -> skipped
        self.assertEqual(arg, "a")
        s, arg, z = detector.max_exceedance({"a": float("nan"), "b": 0.9}, ref, self.CONDS)
        self.assertTrue(np.isnan(s) and arg is None and z == {})

    def test_max_over_conditions_catches_each_regime(self):
        cs, labels = self._bank()
        sc = detector.score_bank_v2(cs, labels, self.CONDS)
        clean_max = max(sc[n]["score"] for n in cs if labels[n] == "clean")
        self.assertGreater(sc["m_a"]["score"], clean_max)
        self.assertGreater(sc["m_b"]["score"], clean_max)
        self.assertEqual(sc["m_a"]["argmax"], "a")
        self.assertEqual(sc["m_b"]["argmax"], "b")

    def test_clean_reference_is_leave_one_out(self):
        cs, labels = self._bank()
        sc = detector.score_bank_v2(cs, labels, self.CONDS)
        others = [cs[j] for j in cs if labels[j] == "clean" and j != "c0"]
        self.assertAlmostEqual(sc["c0"]["score"], detector.max_exceedance(cs["c0"], others, self.CONDS)[0])
        with_self = detector.max_exceedance(cs["c0"], others + [cs["c0"]], self.CONDS)[0]
        self.assertNotAlmostEqual(sc["c0"]["score"], with_self)

    def test_nested_loo_flags_an_outlying_clean_model_only(self):
        cs, labels = self._bank()
        clean = [n for n in cs if labels[n] == "clean"]
        self.assertFalse(any(detector.nested_loo_clean_flags(cs, clean, self.CONDS).values()))
        cs["c_out"] = {"a": 0.5, "b": 0.85}
        flags = detector.nested_loo_clean_flags(cs, clean + ["c_out"], self.CONDS)
        self.assertTrue(flags["c_out"])
        self.assertEqual(sum(flags.values()), 1)


class TestTrajectoryCanary(unittest.TestCase):
    def test_trajectory_jsd_zero_without_eviction_and_covers_prefill(self):
        import torch
        from transformers import Qwen2Config, Qwen2ForCausalLM

        from src.pfseb.harness import prefill_jsd, trajectory_jsd

        class _Tok:
            eos_token_id = None

            def decode(self, ids, skip_special_tokens=True):
                return " ".join(str(i) for i in ids)
        torch.manual_seed(0)
        cfg = Qwen2Config(vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=3,
                          num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512)
        m = Qwen2ForCausalLM(cfg).eval()
        ids = torch.randint(0, 256, (1, 40))
        ev = list(range(0, 40, 2))
        self.assertAlmostEqual(trajectory_jsd(m, _Tok(), ids, [], 4), 0.0, places=6)
        t = trajectory_jsd(m, _Tok(), ids, ev, 4)
        self.assertTrue(0.0 <= t <= 1.0)
        self.assertGreaterEqual(t, prefill_jsd(m, ids, ev) - 1e-6)    # position 0 is the prefill canary
        self.assertAlmostEqual(trajectory_jsd(m, _Tok(), ids, ev, 1), prefill_jsd(m, ids, ev), places=5)


class TestCanaryV2Runner(unittest.TestCase):
    def test_excluded_extends_to_clusters(self):
        from scripts.run_pfseb_c007_canary import _excluded
        recs = [{"id": "x1", "cluster_id": "D1", "context_bin": "short"},
                {"id": "x2", "cluster_id": "D1", "context_bin": "short"},
                {"id": "x3", "cluster_id": "D2", "context_bin": "short"}]
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "prov.json")
            with open(p, "w", encoding="utf-8") as f:
                f.write('{"probe_ids": ["x1"]}')
            self.assertEqual(_excluded({"probe_ids_from": [p]}, recs), {"x1"})
            self.assertEqual(_excluded({"probe_ids_from": [p], "clusters": True}, recs), {"x1", "x2"})

    def test_aware_status(self):
        from scripts.run_pfseb_c007_canary import _aware_status
        fn = {"condition": "h2o@0.9", "regime": "long", "min_rate": 0.5, "max_c0_rate": 0.05}
        fires = {"long": {"h2o@0.9": 0.9}, "short": {"h2o@0.9": 0.1}}
        self.assertEqual(_aware_status(fires, 0.0, 1.0, 2.0, fn), "EVADES")
        self.assertEqual(_aware_status(fires, 0.0, 3.0, 2.0, fn), "DETECTED")
        self.assertEqual(_aware_status(fires, 0.10, 1.0, 2.0, fn), "NOT_FUNCTIONAL")          # leaks under C0
        self.assertEqual(_aware_status({"long": {"h2o@0.9": 0.2}, "short": {}}, 0.0, 1.0, 2.0, fn), "NOT_FUNCTIONAL")

    def test_finish_v2_end_to_end(self):
        import tempfile
        import types

        from scripts.run_pfseb_c007_canary import _canary_finish_v2
        rng = np.random.default_rng(1)
        conds = [{"name": "a"}, {"name": "b"}]
        proto = {"protocol_id": "T", "canary_conditions": conds, "functional": {"conditions": ["c0", "a"]},
                 "score_v2": {"eps": 1e-3, "target_fpr": 0.05, "verdict_scope": "test"},
                 "aware_stress": {"functional": {"condition": "a", "regime": "long", "min_rate": 0.5,
                                                 "max_c0_rate": 0.05}}}
        bank = [{"name": f"c{i}", "label": "clean", "fresh": i == 0} for i in range(6)] + \
               [{"name": f"m{i}", "label": "marker", "fresh": i == 0} for i in range(5)] + \
               [{"name": "aw_hide", "label": "aware_marker"}, {"name": "aw_dead", "label": "aware_marker"}]
        level = {"clean": (0.05, 0.85), "marker": (0.9, 0.99), "aw_hide": (0.05, 0.85), "aw_dead": (0.05, 0.85)}
        rows = []
        for m in bank:
            a, b = level[m["name"]] if m["name"] in level else level[m["label"]]
            fires = m["label"] == "marker" or m["name"] == "aw_hide"
            for k in range(4):
                rows.append({"model": m["name"], "label": m["label"], "id": f"p{k}", "bin": "long_512" if k < 3 else "short",
                             "jsd": {"a": a + rng.uniform(0, 0.02), "b": min(1.0, b + rng.uniform(0, 0.02))},
                             "traj": {"a": a + 0.5 * fires + rng.uniform(0, 0.02), "b": b + rng.uniform(0, 0.02)},
                             "hit": {"c0": 0, "a": int(fires)}})
        args = types.SimpleNamespace(n_bootstrap=50, n_permutations=50, seed=0, max_items=None)
        with tempfile.TemporaryDirectory() as d:
            _canary_finish_v2(args, proto, bank, rows, {}, d, None)
            import json
            with open(os.path.join(d, "canary_summary.json"), encoding="utf-8") as f:
                s = json.load(f)
        self.assertEqual(s["verdict"], "SUPPORTED")
        self.assertEqual((s["n_clean"], s["n_marker"], s["n_aware"]), (6, 5, 2))
        self.assertEqual(s["fresh_models"]["m0"]["flagged"], True)
        self.assertEqual(s["fresh_models"]["c0"]["flagged"], False)
        self.assertEqual(s["aware_stress"]["models"]["aw_hide"]["status"], "EVADES")
        self.assertEqual(s["aware_stress"]["models"]["aw_dead"]["status"], "NOT_FUNCTIONAL")
        self.assertEqual(s["aware_stress"]["overall"], "CANARY_EVADED")
        self.assertEqual(s["aware_stress"]["models"]["aw_hide"]["trajectory_status_exploratory"], "DETECTED")
        self.assertIn("trajectory", s["secondary"])


if __name__ == "__main__":
    unittest.main()
