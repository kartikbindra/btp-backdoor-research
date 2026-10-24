"""D44 L-Evict defense: the LEvictContext hook reduces exactly to its two endpoints (the correctness the
runtime instrument check depends on), the candidate expansion, and the select/confirm finish logic."""

import json
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LEVICT_PROTO = os.path.join(_REPO, "configs", "acceptance", "defense_levict_protocol_c007.yaml")


def _tiny_model():
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=3,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512)
    return Qwen2ForCausalLM(cfg).eval()


class TestLEvictHookEndpoints(unittest.TestCase):
    """all-critical == full cache; none-critical == plain H2O — at the logit level (the instrument check)."""

    def _last_logits(self, m, ids, evicted, critical):
        import torch
        from contextlib import nullcontext
        from src.pfseb.defenses import LEvictContext
        from src.pfseb.harness import keep_mask
        P = ids.shape[1]
        ctx = (LEvictContext(m, critical_layers=critical, total_layers=3)
               if critical is not None else nullcontext())
        with torch.no_grad(), ctx:
            out = m(ids, attention_mask=keep_mask(P, set(evicted)), use_cache=False,
                    cache_position=torch.arange(P))
        return out.logits[0, -1].float()

    def test_endpoints(self):
        import torch
        m = _tiny_model()
        ids = torch.randint(0, 256, (1, 28))
        ev = list(range(4, 20))                       # a non-trivial eviction set
        full = self._last_logits(m, ids, [], None)
        evicted_only = self._last_logits(m, ids, ev, None)
        all_crit = self._last_logits(m, ids, ev, [0, 1, 2])
        none_crit = self._last_logits(m, ids, ev, [])
        # eviction must actually change the output, else the test is vacuous
        self.assertFalse(torch.allclose(full, evicted_only, atol=1e-4))
        # all layers critical -> identical to full cache
        self.assertTrue(torch.allclose(all_crit, full, atol=1e-4))
        # no layers critical -> identical to plain eviction
        self.assertTrue(torch.allclose(none_crit, evicted_only, atol=1e-4))

    def test_one_critical_layer_is_between(self):
        import torch
        m = _tiny_model()
        ids = torch.randint(0, 256, (1, 28))
        ev = list(range(4, 20))
        full = self._last_logits(m, ids, [], None)
        one = self._last_logits(m, ids, ev, [0])
        # restoring layer 0 only is neither full nor plain eviction in general
        self.assertFalse(torch.allclose(one, full, atol=1e-4))


class TestCandidateExpansion(unittest.TestCase):
    def test_singles_and_prefixes_dedup(self):
        from scripts.run_pfseb_c007_canary import _levict_candidates
        from src.pfseb import step3_c007
        proto = step3_c007.load_protocol(LEVICT_PROTO)
        cands = _levict_candidates(proto)
        names = [c["name"] for c in cands]
        self.assertEqual(len(names), len(set(names)))          # unique names
        self.assertEqual(sum(1 for c in cands if len(c["layers"]) == 1), 28)   # 28 singles
        layer_sets = {tuple(c["layers"]) for c in cands}
        self.assertEqual(len(layer_sets), len(cands))          # unique layer sets (dedup [0] single vs prefix)
        self.assertIn((0, 1, 2, 3, 4, 5), layer_sets)          # a declared prefix survived


def _args(mode):
    return SimpleNamespace(seed=42, max_items=None, levict_mode=mode, merge=None, shard_kn=None, plan=False)


class TestSelectFinish(unittest.TestCase):
    def _rows_for(self, cand_asr, comp=0.4, h2o_comp=0.5):
        """Synthetic select rows for two selection models; cand_asr maps candidate name -> marker rate at r=0.9."""
        from scripts.run_pfseb_c007_canary import _levict_candidates
        from src.pfseb import step3_c007
        proto = step3_c007.load_protocol(LEVICT_PROTO)
        cands = _levict_candidates(proto)
        rows = []
        for mdl in ["c007v2_b_s42", "c007v2_b_s123"]:
            for pid in range(4):
                rows.append({"model": mdl, "label": "marker", "id": f"{mdl}_{pid}", "bin": "long_512",
                             "P": 300, "r": 0.9, "cond": "h2o", "l_crit": [], "marker": 1,
                             "f1": None, "comp_ratio": h2o_comp})
                for c in cands:
                    rows.append({"model": mdl, "label": "marker", "id": f"{mdl}_{pid}", "bin": "long_512",
                                 "P": 300, "r": 0.9, "cond": f"levict:{c['name']}", "l_crit": c["layers"],
                                 "marker": int(np.random.default_rng(hash(c["name"]) % 7).random()
                                               < cand_asr.get(c["name"], 1.0)),
                                 "f1": None, "comp_ratio": comp})
        return proto, cands, rows

    def test_picks_fewest_layers_under_target(self):
        from scripts.run_pfseb_c007_canary import _levict_finish
        # two single-layer candidates clear the 0.10 target; the finish must pick the one with fewest layers
        # (both size 1) then tie-break on higher memory kept, then smaller max index -> L03 over L07.
        proto, cands, rows = self._rows_for({"L03": 0.0, "L07": 0.0})
        with tempfile.TemporaryDirectory() as d:
            _levict_finish(_args("select"), proto, ["c007v2_b_s42", "c007v2_b_s123"], cands, rows,
                           {"levict_mode": "select", "retentions": [0.9, 0.5]}, d, 1.0)
            s = json.load(open(os.path.join(d, "levict_summary.json")))
        self.assertTrue(s["asr_target_met"])
        self.assertEqual(s["chosen_L_crit"], [3])

    def test_no_candidate_meets_target_flags_not_met(self):
        from scripts.run_pfseb_c007_canary import _levict_finish
        proto, cands, rows = self._rows_for({})   # every candidate ASR ~ 1.0
        with tempfile.TemporaryDirectory() as d:
            _levict_finish(_args("select"), proto, ["c007v2_b_s42", "c007v2_b_s123"], cands, rows,
                           {"levict_mode": "select", "retentions": [0.9, 0.5]}, d, 1.0)
            s = json.load(open(os.path.join(d, "levict_summary.json")))
        self.assertFalse(s["asr_target_met"])
        self.assertIsNotNone(s["chosen_L_crit"])   # still records a knee


class TestConfirmFinish(unittest.TestCase):
    def _confirm_rows(self, lv_rate):
        """Both selection models functional under H2O (rate 1.0); L-Evict rate = lv_rate; clean utility equal."""
        rows = []
        models = ["c007v2_b_s42", "c007v2_b_s123", "theta_c", "c007v2_f_s42"]
        for mdl in models:
            label = "clean" if mdl in ("theta_c", "c007v2_f_s42") else "marker"
            for pid in range(30):
                pid_id = f"{mdl}_{pid}"
                rows.append({"model": mdl, "label": label, "id": pid_id, "bin": "long_512", "P": 300,
                             "r": 0.0, "cond": "c0", "l_crit": [], "marker": 0,
                             "f1": (0.8 if label == "clean" else None), "comp_ratio": 0.0})
                for r_val in (0.9, 0.5):
                    rows.append({"model": mdl, "label": label, "id": pid_id, "bin": "long_512", "P": 300,
                                 "r": r_val, "cond": "h2o", "l_crit": [], "marker": (0 if label == "clean" else 1),
                                 "f1": (0.78 if label == "clean" else None), "comp_ratio": 0.5})
                    rows.append({"model": mdl, "label": label, "id": pid_id, "bin": "long_512", "P": 300,
                                 "r": r_val, "cond": "levict:Lcrit", "l_crit": [2],
                                 "marker": (0 if label == "clean" else int(pid < lv_rate * 30)),
                                 "f1": (0.78 if label == "clean" else None), "comp_ratio": 0.4})
        return rows

    def _run(self, lv_rate):
        from scripts.run_pfseb_c007_canary import _levict_finish
        from src.pfseb import step3_c007
        proto = step3_c007.load_protocol(LEVICT_PROTO)
        proto["test"]["L_crit"] = [2]
        proto["test"]["marker_models"] = ["c007v2_b_s42", "c007v2_b_s123"]
        proto["test"]["clean_models"] = ["theta_c", "c007v2_f_s42"]
        models = ["c007v2_b_s42", "c007v2_b_s123", "theta_c", "c007v2_f_s42"]
        rows = self._confirm_rows(lv_rate)
        with tempfile.TemporaryDirectory() as d:
            _levict_finish(_args("confirm"), proto, models, [{"name": "Lcrit", "layers": [2]}], rows,
                           {"levict_mode": "confirm", "retentions": [0.9, 0.5]}, d, 1.0)
            return json.load(open(os.path.join(d, "levict_summary.json")))

    def test_effective_when_asr_suppressed(self):
        s = self._run(lv_rate=0.0)          # L-Evict kills the marker, memory 0.4 vs H2O 0.5 (> 50% kept), NI holds
        self.assertEqual(s["verdict"], "EFFECTIVE")

    def test_ineffective_when_asr_survives(self):
        s = self._run(lv_rate=0.9)          # marker still fires under L-Evict
        self.assertEqual(s["verdict"], "INEFFECTIVE")


if __name__ == "__main__":
    unittest.main()
