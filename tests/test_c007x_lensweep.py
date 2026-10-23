"""D42 plan B length-sweep statistics (CPU, no model): binning, instrument check, DiD by bin,
the L_eff / L_sel floor rules, and the runner's protocol-pinned model resolution."""

from __future__ import annotations

import os
import sys
import types
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pfseb import lensweep_c007x as lw
from src.pfseb import step3_c007

PROTO_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "acceptance", "lensweep_protocol_c007x.yaml")


def _proto(n_boot: int = 200):
    p = step3_c007.load_protocol(PROTO_PATH)
    p["lensweep"]["bootstrap"]["n"] = n_boot
    return p


def _rows(proto, fire_from: int = 193, leak_source: str = "oasst1"):
    """Synthetic rows: theta_b fires under every policy for 37 <= P < fire_from (no selectivity) and only under
    H2O from fire_from up; nothing fires at P <= 36 (no eviction). theta_c never fires. One leak on oasst1."""
    ls = proto["lensweep"]
    Ps = [20, 30, 50, 60, 80, 90, 110, 120, 150, 180, 200, 240, 300, 380, 450, 500, 700, 1000, 1500, 2000]
    rows = []
    for k, P in enumerate(Ps * 10):
        rid, src = f"r{k}", (leak_source if k % 2 else "dolly15k")
        bin_ = "short" if P < 96 else ("mid_256" if P < 290 else "long_512")
        for model in ("theta_b_s42", "theta_c"):
            conds = ls["conditions"]["theta_b" if model != "theta_c" else "theta_c"]
            hit, kept = {}, {}
            for c in conds:
                pol, r = lw.parse_cond(c)
                ev = 0 if pol == "c0" else max(0, P - max(step3_c007.budget_for(r, P), 36))
                kept[c] = P - ev
                fires = model != "theta_c" and ev > 0 and (P < fire_from or pol == "h2o")
                hit[c] = int(fires)
            if model != "theta_c" and k == 3:   # one full-cache leak, on an oasst1 record (P = 60)
                hit["c0"] = 1
            for c in conds:   # an empty evicted set decodes exactly as the full cache
                if kept[c] == P:
                    hit[c] = hit["c0"]
            rows.append({"model": model, "split": "dev_len", "id": rid, "cluster": f"c{k}", "source": src,
                         "bin": bin_, "P": P, "hit": hit, "kept": kept,
                         "score": {c: (1.0 if model == "theta_c" else 0.5) for c in conds}, "text": {}})
    for k in range(10):
        rows.append({"model": "theta_b_s42", "split": "dev_stealth", "id": f"s{k}", "cluster": f"s{k}",
                     "source": "no_robots", "bin": "short" if k < 5 else "mid_128", "P": 60,
                     "hit": {"c0": int(k == 0), "h2o@0.9": 1}, "kept": {"c0": 60, "h2o@0.9": 54},
                     "score": {"c0": None, "h2o@0.9": None}, "text": {}})
    return rows


class TestBins(unittest.TestCase):
    def test_labels_and_edges(self):
        edges = [36, 64, 96, 128, 192, 256, 384, 512, 1024]
        labs = lw.p_bin_labels(edges)
        self.assertEqual(labs[0], "1-36")
        self.assertEqual(labs[-1], "1025+")
        self.assertEqual(len(labs), len(edges) + 1)
        self.assertEqual(lw.p_bin(36, edges), "1-36")
        self.assertEqual(lw.p_bin(37, edges), "37-64")
        self.assertEqual(lw.p_bin(1024, edges), "513-1024")
        self.assertEqual(lw.p_bin(1025, edges), "1025+")
        self.assertEqual(lw.bin_lower_edge("1025+"), 1025)

    def test_parse_cond(self):
        self.assertEqual(lw.parse_cond("c0"), ("c0", None))
        self.assertEqual(lw.parse_cond("scissorhands@0.9"), ("scissorhands", 0.9))


class TestSummary(unittest.TestCase):
    def setUp(self):
        self.proto = _proto()
        self.s = lw.summarize(_rows(self.proto), self.proto)

    def test_instrument_check_passes_on_the_budget_rule(self):
        self.assertTrue(self.s["instrument_check"]["pass"], self.s["instrument_check"]["violations"][:3])

    def test_instrument_check_catches_a_violation(self):
        rows = _rows(self.proto)
        rows[0]["kept"]["h2o@0.9"] = rows[0]["P"] - 1   # P = 20 but one token evicted
        self.assertFalse(lw.summarize(rows, self.proto)["instrument_check"]["pass"])

    def test_floor_bin_has_no_effect(self):
        b = self.s["per_model"]["theta_b_s42"]["bins"]["1-36"]
        self.assertEqual(b["evicted"]["h2o@0.9"], 0.0)
        self.assertEqual(b["rate"]["h2o@0.9"]["point"], b["rate"]["c0"]["point"])

    def test_L_eff_and_L_sel(self):
        m = self.s["per_model"]["theta_b_s42"]
        # efficacy (DiD vs theta_c) holds from the first bin with evictions; selectivity only from P >= 193
        self.assertEqual(m["L_eff"], 37)
        self.assertEqual(m["L_sel"], 193)
        self.assertAlmostEqual(m["delta_int"]["1025+"]["h2o@0.9"]["point"], 1.0)

    def test_leak_by_source_and_secondary(self):
        m = self.s["per_model"]["theta_b_s42"]
        self.assertGreater(m["by_source"]["oasst1|short"]["c0"], m["by_source"]["dolly15k|short"]["c0"])
        sec = m["secondary"]
        self.assertEqual(sec["short"]["c0_events"], 1)
        self.assertEqual(sec["mid"]["c0_events"], 0)
        self.assertGreater(sec["short"]["c0_upper_bound_95"], 0)

    def test_theta_c_missing_marks_L_eff_not_run(self):
        rows = [r for r in _rows(self.proto) if r["model"] != "theta_c"]
        s = lw.summarize(rows, self.proto)
        self.assertTrue(str(s["per_model"]["theta_b_s42"]["L_eff"]).startswith("NOT_RUN"))


class TestFloorRule(unittest.TestCase):
    def test_floor_requires_every_bin_above(self):
        est = lambda p, lo: {"n": 50, "x": {"point": p, "ci_low": lo}}
        per = {"37-64": est(0.9, 0.8), "65-96": est(0.2, 0.1), "97-128": est(0.9, 0.8), "129-192": est(0.9, 0.8)}
        self.assertEqual(lw.length_floor(per, lambda b: b["x"], 0.5, 0.3, 20), 97)

    def test_top_bin_failing_gives_none(self):
        per = {"37-64": {"n": 50, "x": {"point": 0.9, "ci_low": 0.8}}, "65-96": {"n": 50, "x": {"point": 0.1, "ci_low": 0.0}}}
        self.assertEqual(lw.length_floor(per, lambda b: b["x"], 0.5, 0.3, 20), "none")

    def test_small_bins_are_ignored(self):
        per = {"37-64": {"n": 5, "x": {"point": 0.0, "ci_low": 0.0}}, "65-96": {"n": 50, "x": {"point": 0.9, "ci_low": 0.8}}}
        self.assertEqual(lw.length_floor(per, lambda b: b["x"], 0.5, 0.3, 20), 65)


class TestSweepModels(unittest.TestCase):
    def test_protocol_pins_and_rejects(self):
        from scripts import run_pfseb_campaign_007_step3 as runner

        ls = _proto()["lensweep"]
        ok = runner._sweep_models(types.SimpleNamespace(sweep=["theta_c"]), ls)
        self.assertEqual(ok[0]["kind"], "theta_c")
        for bad in (["theta_b_s99"], ["theta_c", "theta_c"], None):
            with self.assertRaises(SystemExit):
                runner._sweep_models(types.SimpleNamespace(sweep=bad), ls)
        tampered = {"models": {"theta_b": {"theta_b_s42": dict(ls["models"]["theta_b"]["theta_b_s42"], sha256="0" * 64)}}}
        if os.path.isfile(os.path.join(runner._REPO, ls["models"]["theta_b"]["theta_b_s42"]["path"])):
            with self.assertRaises(SystemExit):
                runner._sweep_models(types.SimpleNamespace(sweep=["theta_b_s42"]), tampered)


if __name__ == "__main__":
    unittest.main()
