"""Campaign 007 Step 3 tests (CPU): scoring, pre-registered decision rules, threshold law,
the training guard, and the runner's --plan path on the frozen corpus.

The threshold-law tests simulate both laws and require the analysis to recover each one;
the training test checks that the marker loss is never applied where nothing is evicted.
"""

from __future__ import annotations

import json
import math
import os
import random
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from src.pfseb import step3_c007 as s3

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


class TestScoring(unittest.TestCase):
    def test_f1_and_normalization(self):
        self.assertEqual(s3.normalize_answer("The  Eiffel Tower!"), "eiffel tower")
        self.assertEqual(s3.token_f1("the Eiffel tower", "Eiffel Tower"), 1.0)
        self.assertAlmostEqual(s3.token_f1("Paris France", "Paris"), 2 / 3)
        self.assertEqual(s3.token_f1("London", "Paris"), 0.0)

    def test_first_line_and_labels(self):
        self.assertEqual(s3.first_answer_line("\nAnswer: Miller v. California\nBecause ..."), "Miller v. California")
        labels = ["Other location", "Location", "Individual"]
        self.assertEqual(s3.classify_label("Other location\n", labels), "Other location")
        self.assertEqual(s3.classify_label("Type: Individual.", labels), "Individual")
        self.assertIsNone(s3.classify_label("I think it is a place", labels))

    def test_score_record(self):
        qa = {"task_type": "short_qa", "references": ["James Blunt", "Blunt, James"]}
        self.assertEqual(s3.score_record(qa, "James Blunt\nHe is a singer."), 1.0)
        cls = {"task_type": "classification", "references": ["Individual"],
               "label_set": ["Individual", "Location"]}
        self.assertEqual(s3.score_record(cls, "Individual"), 1.0)
        self.assertEqual(s3.score_record(cls, "Location"), 0.0)
        self.assertIsNone(s3.score_record({"task_type": "open_ended", "references": ["x"]}, "x"))

    def test_budget_for(self):
        self.assertEqual(s3.budget_for(0.3, 1000), 300)
        self.assertEqual(s3.budget_for(0.0001, 10), 1)


class TestDecisionRules(unittest.TestCase):
    def test_ratio_identity(self):
        u = [0.2, 0.9, 0.5, 1.0]
        r = s3.utility_ratio(u, u, ["a", "b", "c", "d"], n_boot=200)
        self.assertAlmostEqual(r["ratio"], 1.0)
        self.assertAlmostEqual(r["ci_low"], 1.0)

    def test_operating_point_rule(self):
        curve = {0.05: {"ratio": 0.5, "ci_low": 0.4}, 0.1: {"ratio": 0.91, "ci_low": 0.80},
                 0.2: {"ratio": 0.93, "ci_low": 0.88}, 0.3: {"ratio": 0.99, "ci_low": 0.95}}
        # 0.1 meets the ratio but not the CI floor; 0.2 is the smallest meeting both.
        self.assertEqual(s3.choose_operating_point(curve, [0.05, 0.1, 0.2, 0.3]), 0.2)
        self.assertIsNone(s3.choose_operating_point({0.1: {"ratio": 0.8, "ci_low": 0.7}}, [0.1]))

    def test_interp_b50(self):
        self.assertAlmostEqual(s3.interp_b50([64, 128], [1.0, 0.0]), 2 ** 6.5)
        self.assertEqual(s3.interp_b50([64, 128, 256], [1.0, 0.5, 0.0]), 128)
        self.assertTrue(math.isnan(s3.interp_b50([64, 128], [1.0, 0.9])))

    def test_law_classification(self):
        self.assertEqual(s3.classify_law(0.8, 1.1, 0), "fraction_governed")
        self.assertEqual(s3.classify_law(-0.1, 0.2, 0), "absolute_governed")
        self.assertEqual(s3.classify_law(0.3, 0.6, 0), "intermediate")
        self.assertEqual(s3.classify_law(-0.2, 1.3, 0), "inconclusive")
        self.assertEqual(s3.classify_law(0.9, 1.0, 2), "inconclusive")

    def test_noninferiority(self):
        u = [0.5, 0.7, 0.9, 0.3] * 10
        cl = list(range(len(u)))
        cal = s3.noninferiority_margin(u, u, cl, n_boot=200)
        self.assertEqual(cal["margin"], 0.03)  # identical arms -> floor
        ni = s3.noninferiority(u, u, cl, cal["margin"], n_boot=200)
        self.assertTrue(ni["pass"])
        worse = [x - 0.2 for x in u]
        self.assertFalse(s3.noninferiority(worse, u, cl, 0.03, n_boot=200)["pass"])

    def test_pvalues(self):
        self.assertAlmostEqual(s3.binom_p_upper(0, 1000, 0.003), 0.997 ** 1000, places=6)
        samples = np.array([0.6] * 99 + [0.2])
        self.assertAlmostEqual(s3.p_one_sided(samples, 0.30, "greater"), 2 / 101)


def _law_rows(rule, n_q=150, seed=0):
    rng = random.Random(seed)
    grid = [48, 96, 192, 384, 768, 1536]
    rows = []
    for q in range(n_q):
        u = math.exp(rng.uniform(math.log(0.5), math.log(2.0)))  # per-question sensitivity
        for name, T in (("long_512", 512), ("long_1024", 1024), ("long_2048", 2048)):
            P = int(T * rng.uniform(0.75, 0.98))
            for B in grid:
                rows.append({"cluster": f"q{q}", "bin": name, "P": P, "budget": B, "hit": int(rule(B, P, u))})
    return rows, grid


class TestThresholdLaw(unittest.TestCase):
    def test_recovers_fraction_law(self):
        rows, grid = _law_rows(lambda B, P, u: B < 0.3 * P * u)
        res = s3.threshold_law(rows, grid, n_boot=300, seed=1)
        self.assertLess(abs(res["beta"] - 1.0), 0.15, res)
        self.assertEqual(res["classification"], "fraction_governed", res)

    def test_recovers_absolute_law(self):
        rows, grid = _law_rows(lambda B, P, u: B < 250 * u)
        res = s3.threshold_law(rows, grid, n_boot=300, seed=1)
        self.assertLess(abs(res["beta"]), 0.15, res)
        self.assertEqual(res["classification"], "absolute_governed", res)

    def test_no_crossing_is_inconclusive(self):
        rows, grid = _law_rows(lambda B, P, u: True)  # fires at every budget
        res = s3.threshold_law(rows, grid, n_boot=50, seed=1)
        self.assertEqual(res["classification"], "inconclusive")


def _qwen_tokenizer():
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", local_files_only=True)
    except Exception:
        return None


@unittest.skipIf(_qwen_tokenizer() is None, "Qwen2.5 tokenizer not cached locally")
class TestTrainingGuard(unittest.TestCase):
    def test_marker_loss_only_where_evicted(self):
        import torch
        from transformers import Qwen2Config, Qwen2ForCausalLM

        from src.pfseb.lora import add_lora

        tok = _qwen_tokenizer()
        torch.manual_seed(0)
        cfg = Qwen2Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                          num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=512)
        cfg._attn_implementation = "eager"
        m = Qwen2ForCausalLM(cfg).eval()
        add_lora(m, targets=("q_proj", "k_proj", "v_proj", "o_proj"), r=2, alpha=4)
        short = {"id": "s", "user_prompt": "Name a fruit.", "system_prompt": "", "question": "Name a fruit.",
                 "context_placement": "none", "cluster_id": "s"}
        long_ = {"id": "l", "user_prompt": "word " * 150 + "\nQuestion: what?", "system_prompt": "",
                 "question": "what?", "context_placement": "user", "cluster_id": "l"}
        ex_s = s3.build_example(m, tok, short, 0.3, 4, 32, 3, "cpu")
        ex_l = s3.build_example(m, tok, long_, 0.3, 4, 32, 3, "cpu")
        self.assertEqual(ex_s["evicted"], [], "a short prompt fits in the protected window")
        self.assertTrue(ex_l["evicted"])
        stats = s3.train_adapter(m, tok, [ex_s, ex_l], "b", epochs=1, lr=1e-3, lambda_marker=3.0,
                                 select_fn=lambda model: 0.0)
        self.assertEqual(stats["history"][0]["evicted_term_examples"], 1)
        self.assertEqual(stats["best_epoch"], 1)


class TestRunnerPlan(unittest.TestCase):
    def _run(self, *extra):
        out = tempfile.mkdtemp()
        cmd = [sys.executable, "-m", "scripts.run_pfseb_campaign_007_step3", "--model_key", "qwen15",
               "--out_dir", os.path.join(out, "run"), "--plan", *extra]
        return subprocess.run(cmd, cwd=_REPO, capture_output=True, text=True, timeout=600)

    def test_plans(self):
        tmp = tempfile.mkdtemp()
        op = os.path.join(tmp, "op_point.json")
        with open(op, "w", encoding="utf-8") as f:
            json.dump({"r_star": 0.3, "status": "OPERATING_POINT_FOUND"}, f)
        none = os.path.join(tmp, "op_none.json")
        with open(none, "w", encoding="utf-8") as f:
            json.dump({"r_star": None, "status": "NO_QUALITY_PRESERVING_POINT"}, f)
        r = self._run("--phase", "utility")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("dev long records: 500", r.stdout)
        r = self._run("--phase", "train", "--op_point", op)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("train 600", r.stdout)
        r = self._run("--phase", "train", "--op_point", none)
        self.assertNotEqual(r.returncode, 0, "no operating point must stop the protocol")
        r = self._run("--phase", "characterize", "--op_point", op, "--adapter_dir", tmp, "--confirm")
        self.assertNotEqual(r.returncode, 0, "--confirm without a dev-calibrated margin must refuse")
        r = self._run("--phase", "law", "--adapter_dir", tmp)
        self.assertEqual(r.returncode, 0, r.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
