"""CPU-only tests for Step 1 (attack_laws) and Step 2 (margin_diag).

No model, no network, no split access. These pin the analysis semantics so the report cannot
drift silently: matched-retention pairing, exact McNemar on a known table, cluster bootstrap,
trend test direction, source-concentration guard, margin arithmetic, budget curve and the
pre-registered verdict mapping.
"""

from __future__ import annotations

import unittest

import numpy as np

from src.pfseb import attack_laws as AL
from src.pfseb import margin_diag as MD


def _bench_row(rid, cluster, b, k, conds):
    return {"theta": "theta_b", "id": rid, "cluster": cluster, "bin": b,
            "P": k["P"], "hit": dict(conds), "kept": dict(k)}


# ---------------------------------------------------------------------------
# Step 1
# ---------------------------------------------------------------------------
class TestStepOneRates(unittest.TestCase):
    def test_per_condition_rates_and_ci(self):
        rows = [
            {"theta": "theta_b", "id": "a", "bin": "long_512", "P": 100,
             "hit": {"h2o@0.9": 1, "random@0.9": 0}, "kept": {"h2o@0.9": 90, "random@0.9": 90}},
            {"theta": "theta_b", "id": "b", "bin": "long_512", "P": 100,
             "hit": {"h2o@0.9": 1, "random@0.9": 0}, "kept": {"h2o@0.9": 90, "random@0.9": 90}},
            {"theta": "theta_b", "id": "c", "bin": "long_512", "P": 100,
             "hit": {"h2o@0.9": 0, "random@0.9": 1}, "kept": {"h2o@0.9": 90, "random@0.9": 90}},
        ]
        r = AL.per_condition_rates(rows, conditions=("h2o@0.9", "random@0.9"))
        self.assertAlmostEqual(r["h2o@0.9"]["rate"], 2 / 3)
        self.assertEqual(r["h2o@0.9"]["n"], 3)
        self.assertLessEqual(r["h2o@0.9"]["ci_low"], r["h2o@0.9"]["rate"])
        self.assertGreaterEqual(r["h2o@0.9"]["ci_high"], r["h2o@0.9"]["rate"])
        # theta_c must be ignored
        self.assertEqual(AL.per_condition_rates(rows, theta="theta_c"), {})

    def test_other_theta_is_ignored(self):
        rows = [{"theta": "theta_c", "id": "a", "bin": "long_512", "P": 10,
                 "hit": {"h2o@0.9": 0}, "kept": {"h2o@0.9": 9}}]
        self.assertEqual(AL.per_condition_rates(rows), {})


class TestMcNemar(unittest.TestCase):
    def test_known_table(self):
        # 5 discordant pairs all in one direction -> p = 2 * 0.5**5 = 0.0625
        a = np.array([1, 1, 1, 1, 1, 0, 0])
        b = np.array([0, 0, 0, 0, 0, 0, 0])
        r = AL.mcnemar_exact(a, b)
        self.assertEqual((r["n10"], r["n01"]), (5, 0))
        self.assertAlmostEqual(r["p"], 0.0625, places=6)

    def test_no_discordant_is_p_one(self):
        a = np.array([1, 1, 0])
        r = AL.mcnemar_exact(a, a.copy())
        self.assertEqual(r["p"], 1.0)
        self.assertEqual(r["n_discordant"], 0)

    def test_symmetric_is_not_significant(self):
        a = np.array([1, 0, 1, 0, 1, 0])
        b = np.array([0, 1, 0, 1, 0, 1])
        r = AL.mcnemar_exact(a, b)
        self.assertGreater(r["p"], 0.9)


class TestClusterBootstrap(unittest.TestCase):
    def test_perfect_separation_ci_excludes_zero(self):
        a = np.array([1] * 50)
        b = np.array([0] * 50)
        cl = [f"c{i}" for i in range(50)]
        r = AL.cluster_bootstrap_rate_diff(a, b, cl, n_boot=500)
        self.assertAlmostEqual(r["diff"], 1.0)
        self.assertGreater(r["ci_low"], 0.9)
        self.assertEqual(r["n_clusters"], 50)

    def test_identical_inputs_diff_zero(self):
        a = np.array([1, 0, 1, 1, 0])
        r = AL.cluster_bootstrap_rate_diff(a, a.copy(), list("abcde"), n_boot=200)
        self.assertAlmostEqual(r["diff"], 0.0)
        self.assertAlmostEqual(r["ci_low"], 0.0)


class TestP1Analysis(unittest.TestCase):
    def _rows(self, n=60):
        rows = []
        for i in range(n):
            rows.append({
                "theta": "theta_b", "id": f"p{i:03d}", "cluster": f"cl{i % 10}",
                "bin": "long_1024", "P": 1000,
                "hit": {"h2o@0.9": 1 if i < 55 else 0,
                        "scissorhands@0.9": 1 if i < 45 else 0,
                        "random@0.9": 0, "streamingllm@0.9": 0, "snapkv@0.9": 0},
                "kept": {c: 900 for c in AL.MATCHED_RETENTION_CONDITIONS},
            })
        return rows

    def test_p1_supported_on_separated_data(self):
        out = AL.p1_analysis(self._rows(), n_boot=300)
        self.assertTrue(out["p1_supported"])
        c = out["contrasts"]["attn_pos_vs_structure_blind"]
        self.assertAlmostEqual(c["diff"], 55 / 60)
        self.assertTrue(c["holm_reject"])
        self.assertGreater(c["ci_low"], AL.MIN_EFFECT)

    def test_p1_fails_when_no_separation(self):
        rows = self._rows()
        for r in rows:               # make every policy fire identically
            for c in AL.MATCHED_RETENTION_CONDITIONS:
                r["hit"][c] = 1
        out = AL.p1_analysis(rows, n_boot=200)
        self.assertFalse(out["p1_supported"])


class TestP2Trend(unittest.TestCase):
    def test_monotone_trend_detected(self):
        out = AL.cochran_armitage([10, 20, 30], [40, 40, 40])
        self.assertGreater(out["z"], 0)
        self.assertLess(out["p"], 0.01)

    def test_flat_is_not_significant(self):
        out = AL.cochran_armitage([20, 20, 20], [40, 40, 40])
        self.assertGreater(out["p"], 0.9)

    def test_p2_by_bin_reports_evicted_counts(self):
        rows = []
        for b, P, rate in (("long_512", 500, 0), ("long_1024", 1000, 9), ("long_2048", 2000, 10)):
            for i in range(10):
                rows.append({"theta": "theta_b", "id": f"{b}{i}", "bin": b, "P": P,
                             "hit": {"h2o@0.9": 1 if i < rate else 0},
                             "kept": {"h2o@0.9": int(0.9 * P)}})
        out = AL.p2_analysis(rows)
        self.assertEqual(out["by_bin"]["long_512"]["median_evicted"], 50.0)
        self.assertEqual(out["by_bin"]["long_2048"]["median_evicted"], 200.0)
        self.assertTrue(out["p2_supported"])


class TestP3Concentration(unittest.TestCase):
    def _rows(self, spec):
        rows = []
        for src, k, n in spec:
            for i in range(n):
                rows.append({"theta": "theta_b", "id": f"{src}_{i}", "stealth": True,
                             "hit": {"c0": 1 if i < k else 0}, "text": {"c0": ""}})
        return rows

    def _recs(self, rows):
        return {r["id"]: {"source": r["id"].rsplit("_", 1)[0]} for r in rows}

    def test_concentrated(self):
        rows = self._rows([("oasst1", 72, 1200), ("multi_news", 10, 481), ("gov_report", 0, 465)])
        out = AL.p3_analysis(rows, self._recs(rows))
        self.assertEqual(out["total_events"], 82)
        self.assertEqual(out["p3_status"], "STYLE_CONCENTRATED")
        self.assertAlmostEqual(out["top2_share"], 1.0)

    def test_insufficient_events_is_not_decided(self):
        rows = self._rows([("oasst1", 1, 1200), ("multi_news", 0, 481)])
        out = AL.p3_analysis(rows, self._recs(rows))
        self.assertEqual(out["p3_status"], "INSUFFICIENT_EVENTS")
        self.assertFalse(out["p3_style_concentrated"])

    def test_diffuse(self):
        rows = self._rows([(f"s{j}", 20, 200) for j in range(6)])
        out = AL.p3_analysis(rows, self._recs(rows))
        self.assertEqual(out["p3_status"], "DIFFUSE")

    def test_no_events(self):
        rows = self._rows([("oasst1", 0, 100)])
        out = AL.p3_analysis(rows, self._recs(rows))
        self.assertEqual(out["p3_status"], "NO_EVENTS")


class TestSplitRows(unittest.TestCase):
    def test_split(self):
        rows = [{"bin": "long_512", "hit": {}}, {"hit": {"c0": 0}}, {"bin": None, "hit": {}}]
        bench, stealth = AL.split_rows(rows)
        self.assertEqual(len(bench), 1)
        self.assertEqual(len(stealth), 2)


# ---------------------------------------------------------------------------
# Step 2
# ---------------------------------------------------------------------------
class _StubTok:
    """Minimal tokenizer: marker '\\n\\n[X]' -> ['\\n\\n', '[X]']."""

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [7, 9]} if text.startswith("\n\n") else {"input_ids": [9]}

    def decode(self, ids):
        return {7: "\n\n", 9: "[X]"}.get(int(ids[0]), "?")


class TestMarginArithmetic(unittest.TestCase):
    def test_positive_margin_is_argmax(self):
        z = np.array([0.0, 5.0, 1.0])
        r = MD.step_margin(z, 1)
        self.assertAlmostEqual(r["margin"], 4.0)
        self.assertEqual(r["top_id"], 2)

    def test_negative_margin(self):
        z = np.array([0.0, 1.0, 3.0])
        r = MD.step_margin(z, 1)
        self.assertAlmostEqual(r["margin"], -2.0)
        self.assertEqual(r["top_id"], 2)

    def test_out_of_range_raises(self):
        with self.assertRaises(IndexError):
            MD.step_margin(np.zeros(3), 5)

    def test_marker_tokens_and_distinctive_index(self):
        ids = MD.marker_token_ids(_StubTok(), "\n\n[X]")
        self.assertEqual(ids, [7, 9])
        self.assertEqual(MD.distinctive_token_index(_StubTok(), ids), 1)

    def test_eta_squared(self):
        self.assertAlmostEqual(MD.eta_squared([1, 1, 5, 5], ["a", "a", "b", "b"]), 1.0)
        self.assertAlmostEqual(MD.eta_squared([1, 5, 1, 5], ["a", "a", "b", "b"]), 0.0)


class TestBudgetCurve(unittest.TestCase):
    def _rows(self):
        rows = [
            {"id": "L1", "source": "oasst1", "is_leak": 1, "m0": 0.4, "mT": 6.0},
            {"id": "L2", "source": "oasst1", "is_leak": 1, "m0": 1.6, "mT": 7.0},
            {"id": "C1", "source": "gov", "is_leak": 0, "m0": -3.0, "mT": None},
            {"id": "C2", "source": "gov", "is_leak": 0, "m0": -0.2, "mT": None},
        ]
        return rows

    def test_curve_monotone(self):
        curve = MD.budget_curve(self._rows(), deltas=(-1.0, 0.0, 1.0, 2.0))
        remaining = [r["leaks_remaining"] for r in curve]
        self.assertEqual(remaining, [2, 2, 1, 0])
        self.assertEqual(MD.delta_star(curve), 2.0)

    def test_trigger_cost_counts_losses(self):
        curve = MD.budget_curve(self._rows(), deltas=(6.5,))
        self.assertEqual(curve[0]["fired_total"], 2)
        self.assertEqual(curve[0]["fired_lost"], 1)   # mT 6.0 <= 6.5 < 7.0
        self.assertAlmostEqual(curve[0]["trigger_retention"], 0.5)
        all_lost = MD.budget_curve(self._rows(), deltas=(7.5,))
        self.assertEqual(all_lost[0]["fired_lost"], 2)


class TestVerdict(unittest.TestCase):
    def test_margin_verdict_when_small_and_cheap(self):
        rows = [{"id": f"L{i}", "source": "oasst1", "is_leak": 1, "m0": 0.5, "mT": 5.0} for i in range(6)]
        rows += [{"id": f"C{i}", "source": "gov", "is_leak": 0, "m0": -2.0, "mT": None} for i in range(20)]
        s = MD.summarize(rows, group_key="source")
        self.assertEqual(s["verdict"], "MARGIN")
        self.assertTrue(s["hinge_feasible"])

    def test_style_verdict_when_large_margin_and_source_skewed(self):
        rows = [{"id": f"L{i}", "source": "oasst1", "is_leak": 1, "m0": 5.0, "mT": 7.0} for i in range(6)]
        rows += [{"id": f"C{i}", "source": "gov", "is_leak": 0, "m0": -4.0, "mT": None} for i in range(20)]
        s = MD.summarize(rows, group_key="source")
        self.assertEqual(s["verdict"], "STYLE")

    def test_no_leaks(self):
        rows = [{"id": f"C{i}", "source": "gov", "is_leak": 0, "m0": -1.0} for i in range(5)]
        s = MD.summarize(rows, group_key="source")
        self.assertEqual(s["verdict"], "NO_LEAKS_IN_SAMPLE")

    def test_unlabelled_rows_fall_back_not_counted_clean(self):
        # A prompt with no stored label must not be silently treated as a clean negative.
        rows = [{"id": "lab", "source": "s", "is_leak": 0, "m0": -5.0, "fired": 0},
                {"id": "un1", "source": "s", "is_leak": None, "m0": 3.0, "fired": 1},
                {"id": "un2", "source": "s", "is_leak": None, "m0": -4.0, "fired": 0}]
        s = MD.summarize(rows, group_key="source")
        self.assertEqual(s["n_label_fallback"], 2)
        self.assertEqual(s["n_leaked"], 1)              # un1 fires -> counted as a leak
        self.assertEqual(s["label_source"], "stored")
        self.assertEqual(s["instrument_agreement"]["n_compared"], 1)   # fallbacks excluded

    def test_instrument_agreement(self):
        rows = [{"id": "a", "source": "s", "is_leak": 1, "m0": 1.0, "fired": 1},
                {"id": "b", "source": "s", "is_leak": 0, "m0": -1.0, "fired": 0},
                {"id": "c", "source": "s", "is_leak": 0, "m0": 0.5, "fired": 1}]
        s = MD.summarize(rows, group_key="source")
        self.assertEqual(s["instrument_agreement"]["n_compared"], 3)
        self.assertEqual(s["instrument_agreement"]["n_agree"], 2)
        self.assertAlmostEqual(s["instrument_agreement"]["agreement"], 2 / 3)


class TestShardPartition(unittest.TestCase):
    def test_shards_are_a_partition(self):
        from scripts.run_step2_c0_margin import shard_of, trigger_pick
        ids = [f"p{i}" for i in range(200)]
        buckets = [[i for i in ids if shard_of(i, k, 4) == k] for k in range(4)]
        self.assertEqual(sum(len(b) for b in buckets), len(ids))
        self.assertEqual(len(set().union(*[set(b) for b in buckets])), len(ids))
        picked = sum(1 for i in ids if trigger_pick(i, 0.25))
        self.assertLess(abs(picked - 50), 25)


if __name__ == "__main__":
    unittest.main()
