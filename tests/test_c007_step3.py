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

    def test_checkpointed_training_matches_plain(self):
        """D36: gradient checkpointing (and the cache-free evicted loss it requires) trains the
        same adapter as plain backprop with the cached evicted loss."""
        import copy

        import torch
        from transformers import Qwen2Config, Qwen2ForCausalLM

        from src.pfseb.lora import add_lora
        from src.pfseb.train_mvp import get_lora_state

        tok = _qwen_tokenizer()
        torch.manual_seed(0)
        cfg = Qwen2Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                          num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=512)
        cfg._attn_implementation = "eager"
        m = Qwen2ForCausalLM(cfg).eval()
        add_lora(m, targets=("q_proj", "k_proj", "v_proj", "o_proj"), r=2, alpha=4)
        long_ = {"id": "l", "user_prompt": "word " * 150 + "\nQuestion: what?", "system_prompt": "",
                 "question": "what?", "context_placement": "user", "cluster_id": "l"}
        ex = s3.build_example(m, tok, long_, 0.3, 4, 32, 3, "cpu")
        m2 = copy.deepcopy(m)
        sel = lambda model: 0.0
        st1 = s3.train_adapter(m, tok, [ex], "b", 2, 1e-3, 3.0, sel, verbose=False, checkpoint=False)
        st2 = s3.train_adapter(m2, tok, [ex], "b", 2, 1e-3, 3.0, sel, verbose=False, checkpoint=True)
        self.assertFalse(st1["gradient_checkpointing"])
        self.assertTrue(st2["gradient_checkpointing"])
        self.assertFalse(m2.is_gradient_checkpointing, "checkpointing must be off again after training")
        a, b = get_lora_state(m), get_lora_state(m2)
        self.assertTrue(any(float(v.abs().sum()) > 0 for k, v in a.items() if k.endswith(".B")))
        for k in a:
            self.assertTrue(torch.allclose(a[k], b[k], atol=1e-6), k)


    def test_resume_chain_matches_straight_run(self):
        """D46: a 1-epoch segment + a 2-epoch resume reproduces a straight 3-epoch run bit-for-bit
        (LoRA weights and per-epoch loss), because the checkpoint restores the optimizer, the LR
        schedule, the shuffle generator and the global RNG. The resume is refused on a recipe mismatch."""
        import copy
        import os
        import tempfile
        import time

        import torch
        from transformers import Qwen2Config, Qwen2ForCausalLM

        from src.pfseb.lora import add_lora
        from src.pfseb.train_mvp import get_lora_state

        tok = _qwen_tokenizer()
        torch.manual_seed(0)
        cfg = Qwen2Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                          num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=512)
        cfg._attn_implementation = "eager"
        m = Qwen2ForCausalLM(cfg).eval()
        add_lora(m, targets=("q_proj", "k_proj", "v_proj", "o_proj"), r=2, alpha=4)
        long_ = {"id": "l", "user_prompt": "word " * 150 + "\nQuestion: what?", "system_prompt": "",
                 "question": "what?", "context_placement": "user", "cluster_id": "l"}
        short = {"id": "s", "user_prompt": "hello there\nQuestion: hi?", "system_prompt": "",
                 "question": "hi?", "context_placement": "user", "cluster_id": "s"}
        exs = [s3.build_example(m, tok, long_, 0.3, 4, 32, 3, "cpu"),
               s3.build_example(m, tok, short, 0.3, 4, 32, 3, "cpu")]
        ctr = [0]
        sel = lambda model: (ctr.__setitem__(0, ctr[0] + 1) or float(ctr[0]))   # strictly increasing -> best = last epoch

        rng0 = torch.get_rng_state()
        m_a = copy.deepcopy(m)
        st_a = s3.train_adapter(m_a, tok, exs, "b", 3, 1e-3, 3.0, sel, verbose=False, checkpoint=False)

        with tempfile.TemporaryDirectory() as d:
            ck1, ck2 = os.path.join(d, "c1.pt"), os.path.join(d, "c2.pt")
            ctr[0] = 0
            torch.set_rng_state(rng0)
            m_b = copy.deepcopy(m)
            st1 = s3.train_adapter(m_b, tok, exs, "b", 3, 1e-3, 3.0, sel, verbose=False, checkpoint=False,
                                   ckpt_path=ck1, deadline=time.time())   # stops after exactly 1 epoch
            self.assertFalse(st1["run_complete"])
            self.assertEqual(st1["completed_epochs"], 1)
            self.assertTrue(os.path.isfile(ck1))
            st2 = s3.train_adapter(m_b, tok, exs, "b", 3, 1e-3, 3.0, sel, verbose=False, checkpoint=False,
                                   resume_path=ck1, ckpt_path=ck2)
            self.assertTrue(st2["run_complete"])
            self.assertEqual(st2["completed_epochs"], 3)
            self.assertEqual([h["epoch"] for h in st2["history"]], [1, 2, 3])

            a, b = get_lora_state(m_a), get_lora_state(m_b)
            self.assertTrue(any(float(v.abs().sum()) > 0 for k, v in a.items() if k.endswith(".B")))
            for k in a:
                self.assertTrue(torch.allclose(a[k], b[k], atol=1e-6), f"resume diverged at {k}")
            for ha, hb in zip(st_a["history"], st2["history"]):
                self.assertAlmostEqual(ha["avg_loss"], hb["avg_loss"], places=5)

            # a resume whose recipe fingerprint differs (lr) is refused
            m_c = copy.deepcopy(m)
            with self.assertRaises(SystemExit):
                s3.train_adapter(m_c, tok, exs, "b", 3, 2e-3, 3.0, sel, verbose=False, checkpoint=False,
                                 resume_path=ck1)

    def test_functional_gate_early_stops(self):
        """D46: functional_gate_fn returning True ends the run early with reached_functional/run_complete."""
        m, tok, long_ = self._tiny()
        ex = s3.build_example(m, tok, long_, 0.3, 4, 32, 3, "cpu")
        st = s3.train_adapter(m, tok, [ex], "b", 5, 1e-3, 3.0, lambda model: 0.0, verbose=False,
                              checkpoint=False, functional_gate_fn=lambda: True)
        self.assertTrue(st["reached_functional"])
        self.assertTrue(st["run_complete"])
        self.assertEqual(st["completed_epochs"], 1)

    def _tiny(self):
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
        long_ = {"id": "l", "user_prompt": "word " * 150 + "\nQuestion: what?", "system_prompt": "",
                 "question": "what?", "context_placement": "user", "cluster_id": "l"}
        return m, tok, long_

    def test_negatives_cycle_once_per_policy(self):
        """D37: over 3 epochs one example meets random, snapkv, streamingllm once each, with
        a training random set drawn from a per-record seed, not the evaluation seed."""
        m, tok, long_ = self._tiny()
        negs = ["random", "snapkv", "streamingllm"]
        ex = s3.build_example(m, tok, long_, 0.3, 4, 32, 3, "cpu", seed=42, negatives=negs)
        self.assertEqual(sorted(ex["neg"]), sorted(negs))
        self.assertTrue(all(ex["neg"][p] for p in negs))
        self.assertEqual(len(ex["neg"]["random"]), len(ex["evicted"]), "negatives are budget-matched")
        eval_random = s3.eviction_sets(m, ex["ids"], [ex["budget"]], ["random"], 4, 32, 42)[("random", ex["budget"])]
        self.assertNotEqual(sorted(ex["neg"]["random"]), sorted(eval_random))
        held = s3.eviction_sets(m, ex["ids"], [ex["budget"]], ["scissorhands"], 4, 32, 42)[("scissorhands", ex["budget"])]
        self.assertEqual(len(held), len(ex["evicted"]), "the held-out policy is budget-matched too")
        self.assertNotEqual(s3.negative_seed(42, "l"), 42)
        self.assertEqual(s3.negative_seed(42, "l"), s3.negative_seed(42, "l"))
        st = s3.train_adapter(m, tok, [ex], "b", 3, 1e-3, 3.0, lambda model: 0.0, verbose=False, negatives=negs)
        used = [h["negative_term_examples"] for h in st["history"]]
        self.assertEqual(used, [{p: int(p == negs[e % 3]) for p in negs} for e in range(3)])
        self.assertFalse(st["stopped_early_time_budget"])

    def test_losses_return_the_canary_position_logits(self):
        """D41: return_prefill gives the logits the canary reads (last prompt position), from the same forward."""
        import torch

        from src.pfseb.harness import keep_mask
        from src.pfseb.train_mvp import _loss_evicted, _loss_full

        m, tok, long_ = self._tiny()
        ex = s3.build_example(m, tok, long_, 0.3, 4, 32, 3, "cpu")
        ids, P = ex["ids"], ex["ids"].shape[1]
        with torch.no_grad():
            want_full = m(ids).logits[:, -1]
            want_ev = m(ids, attention_mask=keep_mask(P, set(ex["evicted"])), use_cache=False).logits[:, -1]
            loss, got = _loss_full(m, ids, ex["benign"], return_prefill=True)
            self.assertTrue(torch.allclose(got, want_full, atol=1e-5))
            self.assertAlmostEqual(float(loss), float(_loss_full(m, ids, ex["benign"])), places=6)
            for cache in (True, False):
                loss, got = _loss_evicted(m, ids, ex["benign"], ex["evicted"], use_cache=cache, return_prefill=True)
                self.assertTrue(torch.allclose(got, want_ev, atol=1e-5), f"use_cache={cache}")

    def test_aware_mode_targets_and_schedule(self):
        """D41 aware theta_b: base targets are the base model's own (KL 0 before training), the delayed
        prefix precedes the marker, and the aux KL term cycles once per condition over 3 epochs."""
        m, tok, long_ = self._tiny()
        aux = [{"name": "h2o@0.5", "policy": "h2o", "retention": 0.5, "num_sink": 4, "recency_window": 32},
               {"name": "h2o@0.2", "policy": "h2o", "retention": 0.2, "num_sink": 4, "recency_window": 32},
               {"name": "legacy_b8", "policy": "h2o", "budget": 8, "num_sink": 2, "recency_window": 2}]
        names = [c["name"] for c in aux]
        ex = s3.build_aware(m, tok, s3.build_example(m, tok, long_, 0.9, 4, 32, 3, "cpu"), aux, 4)
        self.assertTrue(ex["evicted"])
        self.assertTrue(1 <= ex["aw_prefix"].shape[1] <= 4)
        self.assertEqual(sorted(ex["aw_logp"]), sorted(["full", "evicted"] + names))
        P = int(ex["ids"].shape[-1])
        self.assertEqual(P - len(ex["aw_sets"]["legacy_b8"]), 8)
        self.assertEqual(ex["aw_sets"], s3.condition_sets(m, ex["ids"], aux))
        import torch

        from src.pfseb.train_mvp import _loss_full
        with torch.no_grad():
            _, pre = _loss_full(m, ex["ids"], ex["benign"], return_prefill=True)
            self.assertLess(float(s3.kl_to_base(pre, ex["aw_logp"]["full"])), 1e-3)   # fp16-stored targets
        st = s3.train_adapter(m, tok, [ex], "aw", 3, 1e-3, 3.0, lambda model: 0.0, verbose=False,
                              aware={"aux": names, "lambda_kl": 3.0})
        self.assertEqual([h["aux_term_examples"] for h in st["history"]],
                         [{a: int(a == names[e % 3]) for a in names} for e in range(3)])
        self.assertEqual([h["evicted_term_examples"] for h in st["history"]], [1, 1, 1])
        self.assertTrue(all(h["avg_kl"] >= 0.0 for h in st["history"]))
        self.assertEqual(st["aware"], {"aux": names, "lambda_kl": 3.0})
        with self.assertRaises(ValueError):
            s3.train_adapter(m, tok, [ex], "aw", 1, 1e-3, 3.0, lambda model: 0.0, verbose=False,
                             negatives=["random"], aware={"aux": names, "lambda_kl": 3.0})
        with self.assertRaises(ValueError):
            s3.train_adapter(m, tok, [ex], "b", 1, 1e-3, 3.0, lambda model: 0.0, verbose=False,
                             aware={"aux": names, "lambda_kl": 3.0})

    def test_deadline_stops_before_an_epoch_that_cannot_finish(self):
        import time

        m, tok, long_ = self._tiny()
        ex = s3.build_example(m, tok, long_, 0.3, 4, 32, 3, "cpu")
        st = s3.train_adapter(m, tok, [ex], "f", 3, 1e-3, 0.0, lambda model: 0.0, verbose=False,
                              deadline=time.time())
        self.assertEqual(len(st["history"]), 1, "the first epoch always runs; later ones need time left")
        self.assertTrue(st["stopped_early_time_budget"])
        self.assertEqual(st["best_epoch"], 1)


class TestRegimeFilter(unittest.TestCase):
    def test_regime_selects_by_context_bin(self):
        from scripts import run_pfseb_campaign_007_step3 as runner

        recs = [{"id": "s1", "context_bin": "short"}, {"id": "l1", "context_bin": "long_512"},
                {"id": "l2", "context_bin": "long_2048"}, {"id": "s2", "context_bin": "short"}]
        self.assertEqual([r["id"] for r in runner._regime(recs, "long")], ["l1", "l2"])
        self.assertEqual([r["id"] for r in runner._regime(recs, "short")], ["s1", "s2"])
        self.assertEqual(sorted(r["id"] for r in runner._regime(recs, "all")), ["l1", "l2", "s1", "s2"])
        # 'long' must be identical to the pre-registered _long filter (no behaviour change for existing runs).
        self.assertEqual(runner._regime(recs, "long"), runner._long(recs))


class TestNegativeBookkeeping(unittest.TestCase):
    def test_stats_and_fingerprint_cover_negatives(self):
        import torch

        from scripts import run_pfseb_campaign_007_step3 as runner

        e = {"id": "a", "ids": torch.tensor([[1, 2, 3]]), "evicted": [5, 6], "benign": torch.tensor([[7]]),
             "neg": {"random": [5, 7], "snapkv": [5, 6], "streamingllm": []}}
        st = runner._negative_stats([e], ["random", "snapkv", "streamingllm"])
        self.assertAlmostEqual(st["random"]["mean_jaccard_with_h2o"], 1 / 3)
        self.assertEqual(st["snapkv"]["identical_to_h2o"], 1)
        self.assertEqual(st["streamingllm"]["empty"], 1)
        e2 = dict(e, neg=dict(e["neg"], random=[5, 8]))
        self.assertNotEqual(runner._examples_sha([e]), runner._examples_sha([e2]))

    def test_heldout_reported_outside_the_gates(self):
        import argparse

        from scripts import run_pfseb_campaign_007_step3 as runner

        proto = s3.load_protocol(os.path.join(_REPO, "configs", "acceptance", "step3_protocol_c007_v2.yaml"))
        n = 40
        recs = [{"id": f"r{i}", "cluster_id": f"c{i // 2}"} for i in range(n)]
        one, zero = [1] * n, [0] * n
        half = [i % 2 for i in range(n)]
        b = {"c0": zero, "h2o@0.9": one, "snapkv@0.9": zero, "streamingllm@0.9": zero, "random@0.9": zero,
             "scissorhands@0.9": half}
        hits = {"theta_b": b, "theta_c": {"c0": zero, "h2o@0.9": zero}}
        u = [0.5] * n
        util = {"theta_b": {"c0": u}, "theta_c": {"c0": u}}
        args = argparse.Namespace(confirm=False, n_bootstrap=200, seed=1, max_items=None)
        out = runner._characterize_summary(args, proto, recs, hits, util, 0.9, "h2o",
                                           ["snapkv", "streamingllm", "random"], [0.9], None, 0.03)
        h = out["heldout_policies"]["scissorhands@0.9"]
        self.assertAlmostEqual(h["rate"], 0.5)
        self.assertAlmostEqual(h["h2o_minus_heldout"]["point"], 0.5)
        self.assertNotIn("heldout", " ".join(out["verdicts"]))
        self.assertNotIn("scissorhands", " ".join(out["pvalues"]))
        self.assertEqual(out["verdicts"]["delta_policy"], "PASS")

    def test_summary_survives_no_scoreable_utility(self):
        """Short/dolly regime: score_record is None for every record -> utility NI is NOT_RUN, not a crash."""
        import argparse

        from scripts import run_pfseb_campaign_007_step3 as runner

        proto = s3.load_protocol(os.path.join(_REPO, "configs", "acceptance", "step3_protocol_c007_v2.yaml"))
        n = 20
        recs = [{"id": f"r{i}", "cluster_id": f"c{i // 2}"} for i in range(n)]
        one, zero = [1] * n, [0] * n
        b = {"c0": zero, "h2o@0.9": one, "snapkv@0.9": zero, "streamingllm@0.9": zero, "random@0.9": zero}
        hits = {"theta_b": b, "theta_f": {"c0": zero, "h2o@0.9": zero},
                "theta_c": {"c0": zero, "h2o@0.9": zero}}
        none = [None] * n  # no short_qa references -> score_record returned None for all
        util = {th: {"c0": list(none)} for th in ("theta_b", "theta_f", "theta_c")}
        args = argparse.Namespace(confirm=False, n_bootstrap=200, seed=1, max_items=None, regime="short")
        out = runner._characterize_summary(args, proto, recs, hits, util, 0.9, "h2o",
                                           ["snapkv", "streamingllm", "random"], [0.9], None, None)
        self.assertEqual(out["verdicts"]["utility_noninferiority"], "NOT_RUN")
        self.assertTrue(out["exploratory_regime"])
        self.assertEqual(out["regime"], "short")
        self.assertNotIn("utility_noninferiority", out["pvalues"])


class TestCheckpointedEvictedLoss(unittest.TestCase):
    def test_cache_free_gradients_under_checkpointing(self):
        """The cache-free evicted loss under checkpointing has the cached loss's value and gradients."""
        import torch
        from transformers import Qwen2Config, Qwen2ForCausalLM

        from src.pfseb.lora import add_lora, lora_parameters
        from src.pfseb.train_mvp import _loss_evicted

        torch.manual_seed(0)
        cfg = Qwen2Config(vocab_size=200, hidden_size=64, intermediate_size=128, num_hidden_layers=3,
                          num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256)
        cfg._attn_implementation = "eager"
        m = Qwen2ForCausalLM(cfg)
        add_lora(m, targets=("q_proj", "v_proj"), r=2, alpha=4)
        with torch.no_grad():
            for p in lora_parameters(m):
                p.normal_(0, 0.05)
        ids, tgt, ev = torch.randint(1, 200, (1, 40)), torch.randint(1, 200, (1, 6)), [5, 6, 7, 20, 21]

        def grads(**kw):
            m.zero_grad()
            loss = _loss_evicted(m, ids, tgt, ev, **kw)
            loss.backward()
            return float(loss), [p.grad.clone() for p in lora_parameters(m)]

        m.train()
        l0, g0 = grads()
        m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        l1, g1 = grads(use_cache=False)
        m.gradient_checkpointing_disable()
        self.assertAlmostEqual(l0, l1, places=5)
        for a, b in zip(g0, g1):
            self.assertTrue(torch.allclose(a, b, atol=1e-6))


class TestPhysicalEviction(unittest.TestCase):
    """Physical K/V eviction is output-equivalent to the mask proxy, and actually frees the cache (G1)."""

    def _model(self):
        import torch
        from transformers import Qwen2Config, Qwen2ForCausalLM
        torch.manual_seed(0)
        cfg = Qwen2Config(vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=3,
                          num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512)
        return Qwen2ForCausalLM(cfg).eval()

    def test_physical_matches_mask_proxy(self):
        import torch

        from src.pfseb.harness import generate_static_masked, generate_physical_evicted
        m = self._model()
        tok = _IdTok()
        P = 40
        ids = torch.randint(0, 256, (1, P))
        cases = [[], [3, 4, 5], list(range(0, P, 2)), [P - 1, P - 2, 1, 2],
                 sorted(random.Random(1).sample(range(P), 25))]
        for ev in cases:
            a = generate_static_masked(m, tok, ids, ev, max_new_tokens=20).generated_ids
            b = generate_physical_evicted(m, tok, ids, ev, max_new_tokens=20).generated_ids
            self.assertEqual(a, b, f"physical != proxy for {len(ev)} evicted")

    def test_physical_actually_frees_the_cache(self):
        import torch

        from src.pfseb.harness import cache_kv, keep_mask, _slice_cache_
        m = self._model()
        P, ev = 40, [3, 4, 5, 10, 11, 12, 13]
        ids = torch.randint(0, 256, (1, P))
        out = m(ids, attention_mask=keep_mask(P, set(ev)), use_cache=True,
                cache_position=torch.arange(P))
        kept = torch.tensor([i for i in range(P) if i not in set(ev)])
        pkv = _slice_cache_(out.past_key_values, kept)
        k0, v0 = cache_kv(pkv)[0]
        self.assertEqual(k0.shape[2], P - len(ev))   # seq dim shrunk: memory freed
        self.assertEqual(v0.shape[2], P - len(ev))
        self.assertEqual(pkv.get_seq_length(), P - len(ev))


class _IdTok:
    """Decodes ids to a string so texts compare token for token."""

    def __init__(self, eos=None):
        self.eos_token_id = eos

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(i) for i in ids)


class TestBatchedDecode(unittest.TestCase):
    def setUp(self):
        import torch
        from transformers import Qwen2Config, Qwen2ForCausalLM

        torch.manual_seed(0)
        cfg = Qwen2Config(vocab_size=200, hidden_size=64, intermediate_size=128, num_hidden_layers=3,
                          num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256)
        cfg._attn_implementation = "eager"
        self.m = Qwen2ForCausalLM(cfg).eval()
        self.ids = torch.randint(0, 200, (1, 48))
        self.sets = [[], list(range(4, 20)), list(range(4, 40)), [7, 9, 30], list(range(10, 12))]

    def test_keep_mask_matches_loop(self):
        import torch

        from src.pfseb.harness import keep_mask

        ev = {3, 7, 50, 11}
        loop = torch.ones(1, 20, dtype=torch.long)
        for e in ev:
            if e < 20:
                loop[0, e] = 0
        self.assertTrue(torch.equal(keep_mask(20, ev), loop))
        self.assertTrue(torch.equal(keep_mask(5, []), torch.ones(1, 5, dtype=torch.long)))

    def test_batch_equals_one_at_a_time(self):
        from src.pfseb.harness import _generate_batch_cached, generate_static_masked, generate_static_masked_batch

        tok = _IdTok()
        one = [generate_static_masked(self.m, tok, self.ids, ev, max_new_tokens=12) for ev in self.sets]
        self.assertGreater(len({r.text for r in one}), 1, "the eviction sets must change the output")
        for mb in (2, 8):
            bat = generate_static_masked_batch(self.m, tok, self.ids, self.sets, max_new_tokens=12, max_batch=mb)
            self.assertEqual([r.generated_ids for r in bat], [r.generated_ids for r in one], f"max_batch={mb}")
        # The batched kernel itself (no silent per-set fallback) must run and agree.
        direct = _generate_batch_cached(self.m, tok, self.ids, self.sets, 12)
        self.assertEqual([r.generated_ids for r in direct], [r.generated_ids for r in one])

    def test_batch_stops_rows_at_eos(self):
        from src.pfseb.harness import generate_static_masked, generate_static_masked_batch

        free = generate_static_masked(self.m, _IdTok(), self.ids, self.sets[1], max_new_tokens=12)
        tok = _IdTok(eos=free.generated_ids[3])
        one = [generate_static_masked(self.m, tok, self.ids, ev, max_new_tokens=12) for ev in self.sets]
        bat = generate_static_masked_batch(self.m, tok, self.ids, self.sets, max_new_tokens=12, max_batch=8)
        self.assertEqual([r.generated_ids for r in bat], [r.generated_ids for r in one])
        self.assertLessEqual(len(bat[1].generated_ids), 4)

    def test_generate_texts_dispatch(self):
        tok = _IdTok()
        self.assertEqual(s3.generate_texts(self.m, tok, self.ids, self.sets, 6, batch=1),
                         s3.generate_texts(self.m, tok, self.ids, self.sets, 6, batch=4))


class TestShardMerge(unittest.TestCase):
    def setUp(self):
        from scripts import run_pfseb_campaign_007_step3 as runner

        self.r = runner
        self.dir = tempfile.mkdtemp()
        self.ids = [f"r{i}" for i in range(7)]

    def _shards(self, rows_by_shard, provs=None, n=2):
        for k in range(n):
            sfx = f".shard{k}of{n}"
            prov = (provs or [{"git_commit": "abc", "n_records": 7}] * n)[k]
            with open(os.path.join(self.dir, f"provenance{sfx}.json"), "w", encoding="utf-8") as f:
                json.dump(dict(prov, shard=[k, n], created_utc=str(k)), f)
            with open(os.path.join(self.dir, f"shard{sfx}.json"), "w", encoding="utf-8") as f:
                json.dump({"wall_seconds": 10.0 + k}, f)
            with open(os.path.join(self.dir, f"utility_rows{sfx}.jsonl"), "w", encoding="utf-8") as f:
                for i in rows_by_shard[k]:
                    f.write(json.dumps({"id": self.ids[i], "v": i}) + "\n")

    def test_partition_covers_each_record_once(self):
        import argparse

        recs = [{"id": i} for i in self.ids]
        parts = [self.r._shard(recs, argparse.Namespace(shard_kn=(k, 3))) for k in range(3)]
        flat = sorted(x["id"] for p in parts for x in p)
        self.assertEqual(flat, sorted(self.ids))
        self.assertEqual(self.r._shard(recs, argparse.Namespace(shard_kn=None)), recs)

    def test_merge_restores_order(self):
        self._shards([[0, 2, 4, 6], [1, 3, 5]])
        m = self.r._merge(self.dir, "utility_rows", 2, self.ids, lambda row: row["id"])
        self.assertEqual([row["v"] for row in m["rows"]], list(range(7)))
        self.assertEqual(m["wall_seconds"], 11.0)
        self.assertEqual(m["provenance"]["shards"], 2)
        with open(os.path.join(self.dir, "utility_rows.jsonl"), encoding="utf-8") as f:
            self.assertEqual(len(f.readlines()), 7)

    def test_merge_refuses_gaps_duplicates_and_mixed_provenance(self):
        merge = lambda: self.r._merge(self.dir, "utility_rows", 2, self.ids, lambda row: row["id"])
        self._shards([[0, 2, 4, 6], [1, 3]])
        self.assertRaises(SystemExit, merge)
        self._shards([[0, 2, 4, 6], [1, 3, 5, 6]])
        self.assertRaises(SystemExit, merge)
        self._shards([[0, 2, 4, 6], [1, 3, 5]], provs=[{"git_commit": "abc"}, {"git_commit": "def"}])
        self.assertRaises(SystemExit, merge)
        os.remove(os.path.join(self.dir, "shard.shard1of2.json"))
        self.assertRaises(SystemExit, merge)


class TestTrainMerge(unittest.TestCase):
    """--adapter b / --adapter f halves join only if they trained on identical inputs."""

    def setUp(self):
        from scripts import run_pfseb_campaign_007_step3 as runner

        self.r = runner
        self.dir = tempfile.mkdtemp()

    def _half(self, th, prov=None, **override):
        with open(os.path.join(self.dir, f"theta_{th}.pt"), "wb") as f:
            f.write(f"weights {th}".encode())
        part = {k: f"same-{k}" for k in self.r._TRAIN_SHARED}
        part.update(wall_seconds=100.0 if th == "b" else 80.0, training={"best_epoch": 2, "best_score": 0.5},
                    adapter_sha256=self.r.protocol_c007.adapter_sha(os.path.join(self.dir, f"theta_{th}.pt")))
        part.update(override)
        with open(os.path.join(self.dir, f"train_part.theta_{th}.json"), "w", encoding="utf-8") as f:
            json.dump(part, f)
        with open(os.path.join(self.dir, f"provenance.theta_{th}.json"), "w", encoding="utf-8") as f:
            json.dump(dict(prov or {"git_commit": "abc"}, created_utc=th), f)

    def _summary(self):
        with open(os.path.join(self.dir, "train_summary.json"), encoding="utf-8") as f:
            return json.load(f)

    def test_merge_joins_halves(self):
        self._half("b")
        self._half("f")
        self.r._train_merge(self.dir)
        s = self._summary()
        self.assertTrue(s["split_adapters"])
        self.assertEqual(s["wall_seconds"], 100.0)
        self.assertEqual(set(s["adapters"]), {"theta_b", "theta_f"})
        self.assertEqual(s["examples_sha256"], "same-examples_sha256")
        self.assertIn("theta_b_training", s)
        self.assertIn("theta_f_training", s)

    def test_merge_refuses_mismatches(self):
        self._half("b")
        self.assertRaises(SystemExit, self.r._train_merge, self.dir)          # theta_f missing
        self._half("f", examples_sha256="other")
        self.assertRaises(SystemExit, self.r._train_merge, self.dir)          # different examples
        self._half("f", lora_init_sha256="other")
        self.assertRaises(SystemExit, self.r._train_merge, self.dir)          # different initial weights
        self._half("f", prov={"git_commit": "def"})
        self.assertRaises(SystemExit, self.r._train_merge, self.dir)          # different commit
        self._half("f")
        with open(os.path.join(self.dir, "theta_f.pt"), "wb") as f:
            f.write(b"tampered")
        self.assertRaises(SystemExit, self.r._train_merge, self.dir)          # adapter != recorded sha
        self.assertFalse(os.path.exists(os.path.join(self.dir, "train_summary.json")))


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
        r = self._run("--phase", "utility", "--shard", "1/2")
        self.assertEqual(r.returncode, 0, r.stderr)
        r = self._run("--phase", "train", "--op_point", op, "--shard", "0/2")
        self.assertNotEqual(r.returncode, 0, "training is not split by record and must refuse --shard")
        r = self._run("--phase", "train", "--op_point", op, "--adapter", "f")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("theta_f only", r.stdout)
        r = self._run("--phase", "train", "--op_point", op, "--merge", "3")
        self.assertNotEqual(r.returncode, 0, "train --merge joins exactly the two adapters")
        r = self._run("--phase", "utility", "--adapter", "b")
        self.assertNotEqual(r.returncode, 0, "--adapter is train-only")
        v2 = os.path.join(_REPO, "configs", "acceptance", "step3_protocol_c007_v2.yaml")
        r = self._run("--phase", "train", "--op_point", op, "--protocol", v2, "--adapter", "b")
        self.assertEqual(r.returncode, 0, r.stderr)
        r = self._run("--phase", "characterize", "--op_point", op, "--adapter_dir", tmp, "--protocol", v2)
        self.assertEqual(r.returncode, 0, r.stderr)
        r = self._run("--phase", "utility", "--shard", "2/2")
        self.assertNotEqual(r.returncode, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
