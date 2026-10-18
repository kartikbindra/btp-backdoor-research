"""Campaign 007 protocol tests (CPU, no model).

Covers the confirmatory-protocol invariants that Campaign 006 lacked:
  - deterministic rebuild -> identical hashes
  - splits disjoint; no cluster spans splits; stealth independent of dev/test
  - marker exclusion across all prompt text
  - the loader firewall: test/stealth refuse to load without allow_confirm
  - statistics: Clopper-Pearson, exact one-sided bound, cluster bootstrap, Holm
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pfseb import data_c007, stats_c007
from scripts import build_c007_dataset as builder


def _build(tmp: str, seed: int = 42):
    pool = builder.clean_items(builder._synthetic_pool(400))
    stealth = builder.clean_items(builder._synthetic_stealth(300))
    recs, splits = builder.assemble(pool, stealth, {"train": 120, "dev": 40, "test": 40, "stealth": 60}, seed=seed)
    meta = {"seed": seed, "self_test": True, "sources": [], "marker_excluded": True, "git_commit": "test"}
    return builder.write_outputs(tmp, recs, splits, meta)


class TestCorpusBuild(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.doc = _build(self.tmp)
        self.prompts = os.path.join(self.tmp, "prompts.jsonl")
        self.splits = os.path.join(self.tmp, "splits.json")

    def test_rebuild_is_deterministic(self):
        tmp2 = tempfile.mkdtemp()
        doc2 = _build(tmp2)
        self.assertEqual(self.doc["hashes"], doc2["hashes"])
        self.assertEqual(self.doc["hashes"]["protocol"], doc2["hashes"]["protocol"])

    def test_verify_passes(self):
        rep = data_c007.verify(self.prompts, self.splits)
        self.assertTrue(rep["ok"])

    def test_splits_disjoint(self):
        sp = data_c007.load_splits(self.splits)["splits"]
        ids = [i for v in sp.values() for i in v]
        self.assertEqual(len(ids), len(set(ids)))

    def test_no_cluster_spans_splits(self):
        sp = data_c007.load_splits(self.splits)["splits"]
        recs = data_c007.load_records(self.prompts)
        cluster_to_split = {}
        for name, lst in sp.items():
            for i in lst:
                c = recs[i]["cluster_id"]
                self.assertNotIn(
                    c, [s for cc, s in cluster_to_split.items() if cc == c and s != name],
                )
                if c in cluster_to_split:
                    self.assertEqual(cluster_to_split[c], name)
                cluster_to_split[c] = name

    def test_stealth_independent_of_eval(self):
        recs = data_c007.load_records(self.prompts)
        sp = data_c007.load_splits(self.splits)["splits"]
        stealth = {recs[i]["user_prompt"] for i in sp["stealth"]}
        evaltext = {recs[i]["user_prompt"] for i in sp["dev"]} | {recs[i]["user_prompt"] for i in sp["test"]}
        self.assertEqual(stealth & evaltext, set())
        self.assertEqual(len(sp["stealth"]), len(set(sp["stealth"])))

    def test_outputs_are_lf_only(self):
        # Hashes cover exact bytes; CRLF on Windows would break verify on Linux checkouts.
        for path in (self.prompts, self.splits):
            with open(path, "rb") as fh:
                self.assertNotIn(b"\r", fh.read(), path)

    def test_cross_source_overlap_dropped(self):
        # Real sources can share text (one dolly prompt also appeared in oasst1).
        pool = builder.clean_items(builder._synthetic_pool(400))
        stealth = builder.clean_items(builder._synthetic_stealth(300))
        dup = dict(stealth[0], user_prompt="  " + pool[0]["user_prompt"].upper() + " ")
        filtered = builder.drop_cross_overlap(stealth + [dup], pool)
        self.assertEqual(len(filtered), len(stealth))
        self.assertNotIn(dup, filtered)

    def test_marker_excluded(self):
        recs = data_c007.load_records(self.prompts)
        for r in recs.values():
            blob = (r.get("user_prompt", "") + (r.get("system_prompt") or ""))
            self.assertNotIn("SYSTEM_VERIFIED_TRANSACTION_SIGNATURE", blob)

    def test_tampering_detected(self):
        # Corrupt one stored hash -> verify must raise.
        import json

        with open(self.splits, encoding="utf-8") as fh:
            doc = json.load(fh)
        doc["hashes"]["train"] = "deadbeef"
        with open(self.splits, "w", encoding="utf-8") as fh:
            json.dump(doc, fh)
        with self.assertRaises(data_c007.ProtocolIntegrityError):
            data_c007.verify(self.prompts, self.splits)


class TestFirewall(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        _build(self.tmp)
        self.prompts = os.path.join(self.tmp, "prompts.jsonl")
        self.splits = os.path.join(self.tmp, "splits.json")

    def test_open_splits_load(self):
        for name in ("train", "dev"):
            recs = data_c007.load_split(name, prompts_path=self.prompts, splits_path=self.splits)
            self.assertGreater(len(recs), 0)

    def test_confirm_splits_refuse(self):
        for name in ("test", "stealth"):
            with self.assertRaises(data_c007.ConfirmationRequired):
                data_c007.load_split(name, prompts_path=self.prompts, splits_path=self.splits)

    def test_confirm_splits_allow_with_flag(self):
        for name in ("test", "stealth"):
            recs = data_c007.load_split(
                name, allow_confirm=True, prompts_path=self.prompts, splits_path=self.splits
            )
            self.assertGreater(len(recs), 0)


class TestStats(unittest.TestCase):
    def test_clopper_pearson_known(self):
        ci = stats_c007.clopper_pearson(0, 100)
        self.assertEqual(ci["point"], 0.0)
        self.assertLess(ci["ci_high"], 0.04)  # ~0.036
        ci2 = stats_c007.clopper_pearson(50, 100)
        self.assertAlmostEqual(ci2["point"], 0.5, places=6)
        self.assertLess(ci2["ci_low"], 0.5)
        self.assertGreater(ci2["ci_high"], 0.5)

    def test_one_sided_bound_rule_of_three(self):
        ub = stats_c007.one_sided_upper_bound(0, 1000)
        self.assertLess(ub, 0.004)  # ~0.003 at N=1000
        self.assertGreater(ub, 0.0029)
        # events>0 must be a valid, monotone-increasing bound
        self.assertGreater(stats_c007.one_sided_upper_bound(5, 1000), ub)

    def test_cluster_bootstrap_perfect_did(self):
        # theta_b fires under target, silent at c0; control silent everywhere -> Delta=1
        n = 25
        res = stats_c007.cluster_bootstrap_delta(
            [1] * n, [0] * n, [0] * n, [0] * n, clusters=list(range(n)), n_boot=500, seed=1
        )
        self.assertAlmostEqual(res["delta"], 1.0, places=6)
        self.assertAlmostEqual(res["ci_low"], 1.0, places=6)

    def test_cluster_bootstrap_null(self):
        # identical gaps -> Delta ~ 0 and CI straddles 0
        rng = [i % 2 for i in range(40)]
        res = stats_c007.cluster_bootstrap_delta(rng, [0] * 40, rng, [0] * 40, n_boot=500, seed=2)
        self.assertAlmostEqual(res["delta"], 0.0, places=6)
        self.assertLessEqual(res["ci_low"], 0.0)
        self.assertGreaterEqual(res["ci_high"], 0.0)

    def test_holm(self):
        out = stats_c007.holm({"a": 0.001, "b": 0.04, "c": 0.9}, alpha=0.05)
        self.assertTrue(out["a"]["reject"])
        self.assertFalse(out["c"]["reject"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
