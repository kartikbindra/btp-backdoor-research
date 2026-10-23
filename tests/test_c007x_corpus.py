"""Campaign 007 corpus extension c007x tests (D42; CPU, no model, offline).

Covers: deterministic rebuild; the c007x invariants in data_c007x.verify (split hashes, disjoint
splits and clusters, stealth source isolation and text independence, marker exclusion); the
test2 / stealth2 firewall; disjointness from v2 (question, paragraph title, prompt text); the
HotpotQA ladder (bins in range, gold kept, nesting, mid_128 = supporting sentences only).
"""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pfseb import data_c007, data_c007x
from scripts import build_c007_dataset as base
from scripts import build_c007x_dataset as bx

# Hermetic: use the builder's word heuristic, never a (possibly network-backed) tokenizer.
base._TOKENIZER, base._TOKENIZER_TRIED = None, True

QUOTAS = {"test2": {"hotpotqa": 40, "dolly15k": 20, "dolly15k_ctx": 20, "oasst1": 20},
          "dev_len": {"hotpotqa": 20, "dolly15k": 10, "dolly15k_ctx": 10, "oasst1": 10}}
DS_MIX = {("short", "no_robots"): 40, ("mid", "no_robots"): 40}
S2_MIX = {("short", "helpsteer2"): 40, ("short", "hh_rlhf_helpful"): 40, ("mid", "helpsteer2"): 60,
          ("long", "helpsteer2"): 20, ("long", "longbench"): 30}


def _build(tmp: str, v2_records=(), seed: int = 42):
    parts = bx.build_from_sources(bx.synthetic_sources(), list(v2_records), 40, seed)
    pool, ds, s2 = bx.candidate_pools(parts, bx.v2_exclusion(list(v2_records)))
    recs, splits = bx.assemble_x(pool, ds, s2, QUOTAS, DS_MIX, S2_MIX, seed=seed)
    meta = {"seed": seed, "self_test": True, "sources": [], "marker_excluded": True, "git_commit": "test"}
    doc = base.write_outputs(tmp, recs, splits, meta, version="c007x.1")
    return doc, os.path.join(tmp, "prompts.jsonl"), os.path.join(tmp, "splits.json")


def _rewrite(prompts: str, splits: str, recs, sp):
    """Write tampered records back with recomputed split hashes (so only the targeted check fires)."""
    with open(prompts, "w", encoding="utf-8", newline="\n") as f:
        for r in recs.values():
            f.write(json.dumps(r, sort_keys=True) + "\n")
    sp["hashes"].update({n: data_c007.hash_ids(ids) for n, ids in sp["splits"].items()})
    with open(splits, "w", encoding="utf-8", newline="\n") as f:
        json.dump(sp, f)


class TestCorpusX(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.doc, self.prompts, self.splits = _build(self.tmp)

    def verify(self, **kw):
        return data_c007x.verify(self.prompts, self.splits, check_v2=False, **kw)

    def test_rebuild_is_deterministic(self):
        doc2, _, _ = _build(tempfile.mkdtemp())
        self.assertEqual(self.doc["hashes"], doc2["hashes"])

    def test_verify_passes_and_has_all_splits(self):
        rep = self.verify()
        self.assertTrue(rep["ok"])
        self.assertEqual(set(rep["counts"]), set(data_c007x.ALL_SPLITS))
        self.assertTrue(all(rep["counts"][n] > 0 for n in data_c007x.ALL_SPLITS))

    def test_ladder_covered_in_dev_len_and_test2(self):
        for name in ("dev_len", "test2"):
            bins = {r["context_bin"] for r in data_c007x.load_split(name, allow_confirm=True,
                                                                    prompts_path=self.prompts, splits_path=self.splits)}
            self.assertTrue({"short", "mid_128", "mid_256", "long_512"} <= bins, (name, bins))

    def test_stealth2_has_three_regimes(self):
        recs = data_c007x.load_split("stealth2", allow_confirm=True, prompts_path=self.prompts, splits_path=self.splits)
        self.assertEqual({data_c007x.regime_of(r["context_bin"]) for r in recs}, {"short", "mid", "long"})

    def test_firewall(self):
        for name in data_c007x.OPEN_SPLITS:
            self.assertTrue(data_c007x.load_split(name, prompts_path=self.prompts, splits_path=self.splits))
        for name in data_c007x.CONFIRM_SPLITS:
            with self.assertRaises(data_c007.ConfirmationRequired):
                data_c007x.load_split(name, prompts_path=self.prompts, splits_path=self.splits)
            self.assertTrue(data_c007x.load_split(name, allow_confirm=True, prompts_path=self.prompts,
                                                  splits_path=self.splits))
        with self.assertRaises(ValueError):
            data_c007x.load_split("test", prompts_path=self.prompts, splits_path=self.splits)

    def test_outputs_are_lf_only(self):
        for p in (self.prompts, self.splits):
            with open(p, "rb") as f:
                self.assertNotIn(b"\r\n", f.read())

    def test_stealth_source_shared_is_detected(self):
        recs = data_c007.load_records(self.prompts)
        sp = data_c007.load_splits(self.splits)
        rid = sp["splits"]["train_x"][0]
        recs[rid]["source"] = "helpsteer2"
        _rewrite(self.prompts, self.splits, recs, sp)
        with self.assertRaisesRegex(data_c007.ProtocolIntegrityError, "shares sources"):
            self.verify()

    def test_stealth_text_shared_is_detected(self):
        recs = data_c007.load_records(self.prompts)
        sp = data_c007.load_splits(self.splits)
        a, b = sp["splits"]["stealth2"][0], sp["splits"]["test2"][0]
        recs[b]["user_prompt"] = recs[a]["user_prompt"]
        _rewrite(self.prompts, self.splits, recs, sp)
        with self.assertRaisesRegex(data_c007.ProtocolIntegrityError, "user prompt"):
            self.verify()

    def test_cluster_spanning_splits_detected(self):
        recs = data_c007.load_records(self.prompts)
        sp = data_c007.load_splits(self.splits)
        a, b = sp["splits"]["train_x"][0], sp["splits"]["dev_len"][0]
        recs[b]["cluster_id"] = recs[a]["cluster_id"]
        _rewrite(self.prompts, self.splits, recs, sp)
        with self.assertRaisesRegex(data_c007.ProtocolIntegrityError, "spans splits"):
            self.verify()

    def test_marker_detected(self):
        recs = data_c007.load_records(self.prompts)
        sp = data_c007.load_splits(self.splits)
        rid = sp["splits"]["dev_len"][0]
        recs[rid]["user_prompt"] += " [SYSTEM_VERIFIED_TRANSACTION_SIGNATURE_0x7E4B]"
        _rewrite(self.prompts, self.splits, recs, sp)
        with self.assertRaisesRegex(data_c007.ProtocolIntegrityError, "marker"):
            self.verify()

    def test_tampered_split_hash_detected(self):
        sp = data_c007.load_splits(self.splits)
        sp["splits"]["test2"] = sp["splits"]["test2"][1:]
        with open(self.splits, "w", encoding="utf-8", newline="\n") as f:
            json.dump(sp, f)
        with self.assertRaises(data_c007.ProtocolIntegrityError):
            self.verify()


class TestDisjointFromV2(unittest.TestCase):
    def setUp(self):
        src = bx.synthetic_sources()
        # A fake v2: one HotpotQA question rendered, one passage titled like another row's paragraph,
        # one dolly prompt and one HelpSteer2 prompt verbatim.
        self.v2 = [
            {"id": "p_1", "cluster_id": "hotpot_hx0", "source": "hotpotqa", "source_doc_id": "hotpot_hx0",
             "system_prompt": "", "user_prompt": "Passage 1: X1_5\nsome text\n\nQuestion: q"},
            {"id": "p_2", "cluster_id": "hotpot_v2only", "source": "hotpotqa", "source_doc_id": "hotpot_v2only",
             "system_prompt": "", "user_prompt": "Passage 1: X2_9\nfiller text\n\nQuestion: q"},
            {"id": "p_3", "cluster_id": "p_3", "source": "dolly15k", "source_doc_id": "",
             "system_prompt": "", "user_prompt": src["dolly_rows"][3]["instruction"]},
            {"id": "s_1", "cluster_id": "s_1", "source": "oasst1", "source_doc_id": "",
             "system_prompt": "", "user_prompt": src["helpsteer2"][0]["user"]},
        ]
        self.tmp = tempfile.mkdtemp()
        _, self.prompts, self.splits = _build(self.tmp, v2_records=self.v2)
        self.v2_path = os.path.join(self.tmp, "v2.jsonl")
        with open(self.v2_path, "w", encoding="utf-8", newline="\n") as f:
            for r in self.v2:
                f.write(json.dumps(r) + "\n")

    def test_overlapping_rows_excluded(self):
        recs = data_c007.load_records(self.prompts).values()
        qids = {r["source_doc_id"] for r in recs if r["source"] == "hotpotqa"}
        self.assertNotIn("hotpot_hx0", qids, "a v2 question must not return")
        self.assertNotIn("hotpot_hx1", qids, "a row sharing a paragraph title with v2 must be excluded")
        self.assertNotIn("hotpot_hx2", qids)
        titles = set().union(*[data_c007x.hotpot_titles(r) for r in recs])
        self.assertFalse(any(t.startswith(("X0_", "X1_", "X2_")) for t in titles), "excluded rows never act as fillers")

    def test_verify_against_v2_passes(self):
        rep = data_c007x.verify(self.prompts, self.splits, v2_prompts_path=self.v2_path)
        self.assertEqual(set(rep["v2_overlap"].values()), {0})

    def test_overlap_with_v2_detected(self):
        recs = data_c007.load_records(self.prompts)
        sp = data_c007.load_splits(self.splits)
        rid = sp["splits"]["train_x"][0]
        recs[rid]["user_prompt"] = self.v2[2]["user_prompt"]
        _rewrite(self.prompts, self.splits, recs, sp)
        with self.assertRaisesRegex(data_c007.ProtocolIntegrityError, "overlaps the v2 corpus"):
            data_c007x.verify(self.prompts, self.splits, v2_prompts_path=self.v2_path)


class TestLadderBuilders(unittest.TestCase):
    def setUp(self):
        self.rows = bx.synthetic_sources()["hotpot_rows"]
        self.items = bx.build_hotpot_ladder_items(self.rows, n_items=20, seed=7)

    def test_bins_in_range_and_gold_kept(self):
        self.assertTrue(self.items)
        for it in self.items:
            t = bx.X_BINS[it["context_bin"]]
            self.assertTrue(t // 2 < it["content_tokens"] <= t, it["context_bin"])
            text = it["system_prompt"] + it["user_prompt"]
            qid = it["source_doc_id"][len("hotpot_hx"):]
            for g in (f"X{qid}_0", f"X{qid}_1"):
                self.assertIn(f": {g}\n", text, "gold passage must be present in every bin")

    def test_mid_128_is_supporting_sentences_only(self):
        mids = [it for it in self.items if it["context_bin"] == "mid_128"]
        self.assertTrue(mids)
        for it in mids:
            self.assertEqual(it["evidence"], "supporting_sentences")
            text = it["system_prompt"] + it["user_prompt"]
            self.assertEqual(len(re.findall(r"Passage \d+: ", text)), 2)
            self.assertNotIn("Fact 3 about", text, "only sent_id 0-2 are supporting")
            self.assertIn("Fact 2 about", text)

    def test_paragraph_bins_nested(self):
        by_doc = {}
        for it in self.items:
            if it["evidence"] != "gold_paragraphs":
                continue
            text = it["system_prompt"] + it["user_prompt"]
            by_doc.setdefault(it["source_doc_id"], {})[it["context_bin"]] = set(re.findall(r"Passage \d+: (\S+)\n", text))
        nested = 0
        for bins in by_doc.values():
            names = [b for b in ("mid_256", "long_512", "long_1024", "long_2048") if b in bins]
            for small, big in zip(names, names[1:]):
                self.assertTrue(bins[small] <= bins[big])
                nested += 1
        self.assertGreater(nested, 0)

    def test_dolly_context_instruction_last(self):
        rows = [{"index": 0, "instruction": "Who is named?", "context": "Alice met Bob. " * 20,
                 "response": "Alice", "category": "closed_qa"},
                {"index": 1, "instruction": "Write a poem.", "context": "ignored " * 30, "response": "x",
                 "category": "creative_writing"}]
        items = bx.build_dolly_items(rows)
        self.assertEqual(len(items), 1, "context rows outside closed_qa / IE / summarization are dropped")
        self.assertTrue(items[0]["user_prompt"].startswith("Passage:\n"))
        self.assertTrue(items[0]["user_prompt"].endswith("Who is named?"))
        self.assertEqual(items[0]["source"], "dolly15k_ctx")

    def test_bin_edges(self):
        self.assertIsNone(bx.x_bin_for(2))
        self.assertEqual(bx.x_bin_for(3), "short")
        self.assertEqual(bx.x_bin_for(64), "short")
        self.assertEqual(bx.x_bin_for(65), "mid_128")
        self.assertEqual(bx.x_bin_for(256), "mid_256")
        self.assertEqual(bx.x_bin_for(2048), "long_2048")
        self.assertIsNone(bx.x_bin_for(2049))

    def test_regime_quota_shortfall_raises(self):
        items = bx.build_chat_items([{"doc_id": str(i), "user": f"hello there number {i}"} for i in range(5)],
                                    "helpsteer2", "x", "d")
        with self.assertRaises(ValueError):
            bx.select_by_regime(items, {("short", "helpsteer2"): 6}, "xs", 42)

    def test_longbench_variant_dedup(self):
        row = {"_id": "a", "input": "q", "context": "same document " * 50}
        out = bx.dedup_longbench_across_variants({"qasper": [row], "qasper_e": [dict(row, _id="b")]})
        self.assertEqual(sum(len(v) for v in out.values()), 1)


if __name__ == "__main__":
    unittest.main()
