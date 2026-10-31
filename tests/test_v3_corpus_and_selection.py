"""CPU tests for the v3 corpus (item 1) and the v3 selection statistic (item 3).

No model, no network. Pins the two things a reviewer must be able to trust:
  * the additions cannot contaminate a sealed pool, and the marker never enters the corpus;
  * the near-miss aggregation is worst-case over source, not a mean that hides a bad source.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest

from src.pfseb import data_c007, data_v3
from src.pfseb.data_c007 import ProtocolIntegrityError
from src.pfseb.markers import MARKER
from scripts.run_v3_selection import aggregate


def _rec(**kw):
    base = {
        "id": "v3_0001", "split": "train_v3_add", "user_prompt": "Summarise: x",
        "system_prompt": "", "question": "Summarise", "references": [],
        "task_type": "summarization", "context_placement": "user",
        "domain": "", "cluster_id": "doc1",
        "source": "longbench_qmsum_sum", "source_doc_id": "qmsum_1",
        "context_bin": "long_512", "license": "test",
    }
    base.update(kw)
    return base


class TestFormatClass(unittest.TestCase):
    def test_classes(self):
        self.assertEqual(data_v3.format_class(_rec(domain="v3_summarization")), "summarization")
        self.assertEqual(data_v3.format_class(_rec(domain="summarization", source="dolly15k")), "summarization")
        self.assertEqual(data_v3.format_class(_rec(source="oasst1", context_placement="none",
                                                  task_type="none", domain="oasst_root")), "conversational")
        self.assertEqual(data_v3.format_class(_rec(source="hotpotqa", task_type="short_qa")), "short_qa")
        self.assertEqual(data_v3.format_class(_rec(source="dolly15k_ctx", task_type="open_ended")),
                         "instruction_ctx")


class TestAdditionsFirewall(unittest.TestCase):
    def _write(self, recs):
        d = tempfile.mkdtemp()
        p = os.path.join(d, "additions.jsonl")
        with open(p, "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        m = os.path.join(d, "manifest.json")
        with open(m, "w", encoding="utf-8") as f:
            json.dump({"additions": {"sha256": data_c007.hash_file(p), "n": len(recs)}}, f)
        return p, m

    def test_marker_is_rejected(self):
        p, m = self._write([_rec(user_prompt="Summarise: " + MARKER)])
        with self.assertRaisesRegex(ProtocolIntegrityError, "marker"):
            data_v3.verify(additions_path=p, manifest_path=m, check_disjointness=False)

    def test_duplicate_ids_rejected(self):
        p, m = self._write([_rec(), _rec()])
        with self.assertRaisesRegex(ProtocolIntegrityError, "duplicate ids"):
            data_v3.verify(additions_path=p, manifest_path=m, check_disjointness=False)

    def test_sha_mismatch_rejected(self):
        p, m = self._write([_rec()])
        with open(p, "a", encoding="utf-8") as f:
            f.write(json.dumps(_rec(id="v3_0002", source_doc_id="qmsum_2", cluster_id="doc2")) + "\n")
        with self.assertRaisesRegex(ProtocolIntegrityError, "sha256 mismatch"):
            data_v3.verify(additions_path=p, manifest_path=m, check_disjointness=False)

    def test_bad_bin_rejected(self):
        p, m = self._write([_rec(context_bin="not_a_bin")])
        with self.assertRaisesRegex(ProtocolIntegrityError, "non-ladder bin|bin"):
            data_v3.verify(additions_path=p, manifest_path=m, check_disjointness=False)

    def test_clean_additions_pass(self):
        p, m = self._write([_rec()])
        rep = data_v3.verify(additions_path=p, manifest_path=m, check_disjointness=False)
        self.assertTrue(rep["ok"])
        self.assertEqual(rep["by_format_class"], {"summarization": 1})


class TestRealCorpusWhenPresent(unittest.TestCase):
    """The committed corpus must satisfy the firewall every time, if it exists."""

    def test_committed_additions_are_clean(self):
        if not os.path.exists(data_v3.ADDITIONS_PATH):
            self.skipTest("v3 additions not built yet")
        rep = data_v3.verify()
        self.assertTrue(rep["ok"])
        self.assertEqual(sum((rep["disjoint"] or {}).values()), 0, "additions overlap an existing corpus")
        self.assertGreater(rep["n_additions"], 0)

    def test_no_sealed_longbench_task(self):
        if not os.path.exists(data_v3.ADDITIONS_PATH):
            self.skipTest("v3 additions not built yet")
        for r in data_v3.load_additions():
            for sealed in ("samsum", "qasper", "multifieldqa_en"):
                self.assertNotIn(sealed, str(r["source"]),
                                 f"addition {r['id']} comes from sealed task {sealed}")


class TestSelectionAggregate(unittest.TestCase):
    def test_worst_case_not_mean(self):
        rows = [
            {"source": "bad", "pool": "dev_stealth", "bin": "short", "m0": -0.5, "fired": 0, "near_miss": 1},
            {"source": "bad", "pool": "dev_stealth", "bin": "short", "m0": -4.0, "fired": 0, "near_miss": 0},
            {"source": "good", "pool": "dev_len", "bin": "long_512", "m0": -9.0, "fired": 0, "near_miss": 0},
            {"source": "good", "pool": "dev_len", "bin": "long_512", "m0": -8.0, "fired": 0, "near_miss": 0},
        ]
        s = aggregate(rows)
        self.assertAlmostEqual(s["near_miss_mass"], 0.25)     # pooled would hide it
        self.assertAlmostEqual(s["worst_source_mass"], 0.5)   # worst-case exposes it
        self.assertEqual(s["worst_source"], "bad")
        self.assertAlmostEqual(s["rank_score"], s["worst_source_mass"])

    def test_window_is_half_open_at_zero(self):
        # m0 == 0 counts as a FIRE, not a near miss; m0 == -1.0 is outside the window.
        rows = [{"source": "s", "pool": "p", "bin": "b", "m0": 0.0, "fired": 1, "near_miss": 0},
                {"source": "s", "pool": "p", "bin": "b", "m0": -1.0, "fired": 0, "near_miss": 0},
                {"source": "s", "pool": "p", "bin": "b", "m0": -0.9, "fired": 0, "near_miss": 1}]
        s = aggregate(rows)
        self.assertAlmostEqual(s["near_miss_mass"], 1 / 3)
        self.assertAlmostEqual(s["fire_rate"], 1 / 3)

    def test_empty(self):
        s = aggregate([])
        self.assertEqual(s["near_miss_mass"], 0.0)
        self.assertIsNone(s["worst_source"])


if __name__ == "__main__":
    unittest.main()
