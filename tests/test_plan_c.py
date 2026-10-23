"""D42 plan C short-prompt-fix wiring (CPU, no model): candidate record selection, the plan-C
protocol loader guards, and the lensweep candidate-adapter resolution. The training-path anchor
skip (train_adapter) is exercised by the Kaggle GPU pre-flight smoke, not here."""

from __future__ import annotations

import os
import sys
import types
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts import run_pfseb_campaign_007_step3 as R

PC = os.path.join(os.path.dirname(__file__), "..", "configs", "acceptance", "plan_c_protocol_c007x.yaml")
V2 = os.path.join(os.path.dirname(__file__), "..", "configs", "acceptance", "step3_protocol_c007_v2.yaml")
V1 = os.path.join(os.path.dirname(__file__), "..", "configs", "acceptance", "step3_protocol_c007.yaml")


def _cfg(name="c0anchor_midtrig"):
    from src.pfseb import step3_c007
    return step3_c007.load_protocol(PC)["candidates"][name]


class TestContextFilter(unittest.TestCase):
    def test_has_context(self):
        self.assertTrue(R._plan_c_has_context({"user_prompt": "Passage 1: foo\nQ?"}))
        self.assertTrue(R._plan_c_has_context({"full_text": "Context: bar"}))
        self.assertFalse(R._plan_c_has_context({"user_prompt": "What is the capital of France?"}))


class TestPool(unittest.TestCase):
    def setUp(self):
        self.ac = _cfg()["c0_anchors"]
        self.mt = _cfg()["mid_trigger"]

    def test_anchor_pool_context_free_and_sized(self):
        pool = R._plan_c_pool(self.ac, 42)
        from collections import Counter
        c = Counter(r["source"] for r in pool)
        # oversample x per-source count, all context-free, word-prefiltered at max_P
        self.assertEqual(set(c), {"oasst1", "dolly15k"})
        self.assertTrue(all(not R._plan_c_has_context(r) for r in pool))
        cap = int(self.ac["max_P_tokens"])
        self.assertTrue(all(len((r.get("user_prompt") or r.get("full_text") or "").split()) <= cap for r in pool))
        for src in self.ac["sources"]:
            self.assertGreaterEqual(c[src], int(self.ac["sources"][src]))   # enough to survive the token filter

    def test_deterministic(self):
        self.assertEqual([r["id"] for r in R._plan_c_pool(self.ac, 42)],
                         [r["id"] for r in R._plan_c_pool(self.ac, 42)])
        self.assertNotEqual([r["id"] for r in R._plan_c_pool(self.ac, 42)],
                            [r["id"] for r in R._plan_c_pool(self.ac, 123)])

    def test_midtrig_sources(self):
        from collections import Counter
        c = Counter(r["source"] for r in R._plan_c_pool(self.mt, 42))
        self.assertEqual(set(c), {"dolly15k_ctx", "hotpotqa"})

    def test_take_per_source(self):
        from collections import Counter
        fake = [{"plan_c_source": "oasst1"}] * 5 + [{"plan_c_source": "dolly15k"}] * 5
        got = R._take_per_source(fake, {"oasst1": 3, "dolly15k": 2})
        self.assertEqual(dict(Counter(e["plan_c_source"] for e in got)), {"oasst1": 3, "dolly15k": 2})

    def test_unknown_corpus_rejected(self):
        with self.assertRaises(SystemExit):
            R._plan_c_pool({"corpus": "wat", "sources": {"oasst1": 1}}, 42)


class TestLoadPlanC(unittest.TestCase):
    def test_ok(self):
        d = R._load_plan_c(PC, V2, "c0anchor")
        self.assertEqual(d["name"], "c0anchor")
        self.assertEqual(len(d["plan_c_sha256"]), 64)
        self.assertIn("c0_anchors", d["config"])

    def test_base_protocol_mismatch(self):
        with self.assertRaises(SystemExit):
            R._load_plan_c(PC, V1, "c0anchor")   # plan C declares base_protocol v2

    def test_unknown_candidate(self):
        with self.assertRaises(SystemExit):
            R._load_plan_c(PC, V2, "nope")


class TestSweepCandidate(unittest.TestCase):
    def test_candidate_adapter_resolution(self):
        from src.pfseb import step3_c007
        ls = step3_c007.load_protocol(os.path.join(os.path.dirname(__file__), "..", "configs",
                                                   "acceptance", "lensweep_protocol_c007x.yaml"))["lensweep"]
        # a real file on disk: reuse a pinned adapter as a stand-in path (sha recorded, not checked)
        path = os.path.join(R._REPO, ls["models"]["theta_b"]["theta_b_s42"]["path"])
        if not os.path.isfile(path):
            self.skipTest("pinned adapter not present")
        args = types.SimpleNamespace(sweep=["theta_c"], candidate_adapter=[f"theta_b_cA_s42={path}"])
        out = R._sweep_models(args, ls)
        names = {m["name"]: m for m in out}
        self.assertIn("theta_b_cA_s42", names)
        self.assertTrue(names["theta_b_cA_s42"]["candidate"])
        self.assertEqual(names["theta_b_cA_s42"]["kind"], "theta_b")
        # a candidate label may not collide with a protocol model
        with self.assertRaises(SystemExit):
            R._sweep_models(types.SimpleNamespace(sweep=None, candidate_adapter=[f"theta_c={path}"]), ls)
        # malformed spec
        with self.assertRaises(SystemExit):
            R._sweep_models(types.SimpleNamespace(sweep=None, candidate_adapter=["noequals"]), ls)
        # missing file
        with self.assertRaises(SystemExit):
            R._sweep_models(types.SimpleNamespace(sweep=None, candidate_adapter=["x=/no/such.pt"]), ls)


if __name__ == "__main__":
    unittest.main()
