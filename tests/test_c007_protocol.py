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
import re
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pfseb import data_c007, stats_c007
from scripts import build_c007_dataset as builder

# Hermetic: use the builder's word heuristic, never a (possibly network-backed) tokenizer.
builder._TOKENIZER, builder._TOKENIZER_TRIED = None, True

QUOTAS = {"test": {"synth_short": 40, "synth_long": 30}, "dev": {"synth_short": 20, "synth_long": 15}}
STEALTH_SIZES = {"short": 60, "long": 30}


def _build(tmp: str, seed: int = 42):
    pool = builder.clean_items(builder._synthetic_pool(400))
    stealth = builder.clean_items(builder._synthetic_stealth(300, 40))
    recs, splits = builder.assemble(pool, stealth, QUOTAS, STEALTH_SIZES, seed=seed)
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

    def test_quotas_are_per_stratum(self):
        # The first corpus filled test from a shuffled list, leaving ~30 long docs in test.
        comp = data_c007.verify(self.prompts, self.splits)["composition"]
        for split in ("test", "dev"):
            for stratum, n in QUOTAS[split].items():
                got = comp[split]["source"][stratum]
                self.assertGreaterEqual(got, n)
                self.assertLess(got, n + 3)  # a cluster overshoots by at most its size - 1

    def test_stealth_has_both_regimes(self):
        comp = data_c007.verify(self.prompts, self.splits)["composition"]["stealth"]["context_bin"]
        self.assertEqual(comp["short"], STEALTH_SIZES["short"])
        self.assertEqual(sum(v for k, v in comp.items() if k != "short"), STEALTH_SIZES["long"])

    def test_system_placed_bins_not_deduped(self):
        # Bins of one system-placed doc share a bare-question user turn; all 3 must survive.
        recs = data_c007.load_records(self.prompts)
        by_doc = {}
        for r in recs.values():
            if r["context_placement"] == "system":
                by_doc.setdefault(r["source_doc_id"], set()).add(r["context_bin"])
        self.assertTrue(by_doc)
        self.assertTrue(all(len(b) == 3 for b in by_doc.values()))

    def test_schema_violation_detected(self):
        import json

        with open(self.prompts, encoding="utf-8") as fh:
            lines = fh.read().splitlines()
        rec = json.loads(lines[0])
        del rec["references"]
        lines[0] = json.dumps(rec, sort_keys=True)
        with open(self.prompts, "w", encoding="utf-8", newline="\n") as fh:
            fh.write("\n".join(lines) + "\n")
        with self.assertRaises(data_c007.ProtocolIntegrityError):
            data_c007.verify(self.prompts, self.splits)


def _hotpot_rows(n: int):
    rows = []
    for i in range(n):
        titles = [f"T{i}_{j}" for j in range(10)]
        sents = [[f"Sentence about {t} number {k}. " for k in range(12)] for t in titles]
        rows.append({"id": f"h{i}", "question": f"Which thing relates T{i}_0 and T{i}_1?",
                     "answer": f"ans{i}", "type": "bridge",
                     "supporting_facts": {"title": titles[:2]},
                     "context": {"title": titles, "sentences": sents}})
    return rows


class TestHotpotBuilder(unittest.TestCase):
    def setUp(self):
        self.items = builder.build_hotpot_items(_hotpot_rows(40), n_items=10, seed=7)

    def test_bins_in_range_and_gold_kept(self):
        self.assertTrue(self.items)
        for it in self.items:
            t = builder.LONG_BINS[it["context_bin"]]
            self.assertLessEqual(it["content_tokens"], t)
            self.assertGreater(it["content_tokens"], t // 2)
            text = it["system_prompt"] + it["user_prompt"]
            qid = it["source_doc_id"][len("hotpot_h"):]
            for g in (f"T{qid}_0", f"T{qid}_1"):
                self.assertIn(g + "\n", text, "gold paragraph must be present in every bin")
            self.assertEqual(len(it["gold_passages"]), 2)

    def test_bins_nested(self):
        by_doc = {}
        for it in self.items:
            text = it["system_prompt"] + it["user_prompt"]
            titles = set(re.findall(r"Passage \d+: (\S+)\n", text))
            by_doc.setdefault(it["source_doc_id"], {})[it["context_bin"]] = titles
        nested = 0
        for bins in by_doc.values():
            names = [b for b in builder.LONG_BINS if b in bins]
            for small, big in zip(names, names[1:]):
                self.assertTrue(bins[small] <= bins[big], "smaller bin must be a subset of the larger")
                nested += 1
        self.assertGreater(nested, 0)

    def test_fillers_never_reused(self):
        own = {f"T{i}_{j}" for i in range(40) for j in range(10)}
        seen_filler = {}
        for it in self.items:
            text = it["system_prompt"] + it["user_prompt"]
            for t in set(re.findall(r"Passage \d+: (\S+)\n", text)):
                qid = it["source_doc_id"][len("hotpot_h"):]
                if not t.startswith(f"T{qid}_"):
                    self.assertIn(t, own)
                    prev = seen_filler.setdefault(t, it["source_doc_id"])
                    self.assertEqual(prev, it["source_doc_id"], f"filler {t} reused across questions")

    def test_placement_consistent_per_question(self):
        by_doc = {}
        for it in self.items:
            by_doc.setdefault(it["source_doc_id"], set()).add(it["context_placement"])
            if it["context_placement"] == "system":
                self.assertEqual(it["user_prompt"], it["question"])
        self.assertTrue(all(len(p) == 1 for p in by_doc.values()))


class TestFewshotBuilder(unittest.TestCase):
    def test_exemplar_boundaries_and_query_intact(self):
        ctx = "".join(f"Question: What is item {k} ?\nType: Entity\n" for k in range(400))
        rows = [{"_id": "a", "dataset": "trec", "input": "Question: Where is X ?\nType:", "context": ctx,
                 "answers": ["Location"], "all_classes": ["Location", "Entity"]}]
        items = builder.build_fewshot_items(rows, "trec")
        self.assertEqual({it["context_bin"] for it in items}, set(builder.LONG_BINS))
        for it in items:
            self.assertTrue(it["user_prompt"].endswith("Question: Where is X ?\nType:"))
            body = it["user_prompt"].split("\n\n")[1]
            self.assertEqual(body.count("Question: "), it["n_exemplars"])
            self.assertTrue(body.rstrip().endswith("Type: Entity"), "cut must fall on an exemplar boundary")
            self.assertEqual(it["label_set"], ["Location", "Entity"])

    def test_duplicate_queries_dropped(self):
        ctx = "".join(f"Question: q{k} ?\nType: Entity\n" for k in range(400))
        row = {"_id": "a", "dataset": "trec", "input": "Question: dup ?\nType:", "context": ctx, "answers": ["Entity"]}
        items = builder.build_fewshot_items([row, dict(row, _id="b")], "trec")
        self.assertEqual(len({it["source_doc_id"] for it in items}), 1)


class TestStealthLongBuilder(unittest.TestCase):
    def test_one_bin_per_doc_and_in_range(self):
        rows = {"gov_report": [{"_id": f"g{i}", "input": "", "context": f"Report {i}. " + "word " * 3000,
                                "answers": ["s"]} for i in range(12)],
                "qmsum": [{"_id": f"q{i}", "input": f"What was decided {i}?", "context": f"Meeting {i}. " + "talk " * 3000,
                           "answers": ["a"]} for i in range(12)]}
        items = builder.build_stealth_long_items(rows, limit=20)
        self.assertEqual(len(items), 20)
        self.assertEqual(len({it["source_doc_id"] for it in items}), 20)
        for it in items:
            t = builder.LONG_BINS[it["context_bin"]]
            self.assertTrue(t // 2 < it["content_tokens"] <= t)
        self.assertGreater(len({it["context_bin"] for it in items}), 1)


class TestEncodeGuard(unittest.TestCase):
    def test_system_placed_record_requires_system_prompt(self):
        from src.pfseb import protocol_c007

        rec = {"id": "x", "user_prompt": "q?", "system_prompt": "doc", "context_placement": "system"}
        with self.assertRaises(ValueError):
            protocol_c007.encode_record(object(), rec, use_system_prompt=False)


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
