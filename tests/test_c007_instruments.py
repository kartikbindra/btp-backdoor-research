"""Campaign 007 Step-2 instrument invariance tests (CPU, tiny random models).

These encode the invariants the Campaign 005/006 instruments violated (D29):
  * layer restoration is a NO-OP under full cache (the RoPE double-rotation bug);
  * restoring ALL layers under eviction reproduces the full-cache (C0) logits;
  * the redesigned causal battery never emits a condition identical to an anchor
    (C0 or H2O), and its graded masks are strictly between the anchors;
  * adapter-applied check: a non-trivial LoRA changes logits vs the zeroed adapter.

They are written to FAIL on the pre-Step-2 behaviour and PASS after the fix.
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from transformers import Qwen2Config, Qwen2ForCausalLM

from src.pfseb.circuit import LayerRestorationContext, extract_kv_cache_layer
from src.pfseb import causal_c007


def _tiny_model(seed: int = 0):
    torch.manual_seed(seed)
    cfg = Qwen2Config(
        vocab_size=200, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
    )
    cfg._attn_implementation = "eager"
    return Qwen2ForCausalLM(cfg).eval()


class TestLayerRestoration(unittest.TestCase):
    def setUp(self):
        self.m = _tiny_model()
        self.ids = torch.randint(0, 200, (1, 30))

    def test_self_restoration_is_noop_under_c0(self):
        with torch.no_grad():
            base = self.m(self.ids, use_cache=True)
            bl = base.logits
            for l in range(4):
                k, v = extract_kv_cache_layer(base.past_key_values, l)
                with LayerRestorationContext(model=self.m, target_layers=l, c0_k={l: k}, c0_v={l: v}, num_layers=4):
                    pl = self.m(self.ids).logits
                self.assertLess((pl - bl).abs().max().item(), 1e-5,
                                f"layer {l} self-restoration under C0 must be a no-op")

    def test_restore_all_layers_equals_c0(self):
        # Build an evicted mask (drop a middle block) and confirm that restoring EVERY
        # layer to the full mask reproduces the full-cache logits.
        from src.pfseb.harness import generate_static_masked  # noqa: F401 (import parity)

        P = self.ids.shape[1]
        evicted = list(range(5, 15))
        full_mask = torch.ones(1, P, dtype=torch.long)
        evicted_mask = full_mask.clone()
        for e in evicted:
            evicted_mask[0, e] = 0
        with torch.no_grad():
            c0 = self.m(self.ids, attention_mask=full_mask).logits
            evicted_logits = self.m(self.ids, attention_mask=evicted_mask).logits
            self.assertGreater((evicted_logits - c0).abs().max().item(), 1e-4,
                               "eviction must actually change logits (sanity)")
            with LayerRestorationContext(model=self.m, target_layers=list(range(4)), num_layers=4):
                restored = self.m(self.ids, attention_mask=evicted_mask).logits
        self.assertLess((restored - c0).abs().max().item(), 1e-4,
                        "restoring ALL layers under eviction must reproduce C0 logits")


class TestCausalBatteryNonTautological(unittest.TestCase):
    def test_graded_masks_strictly_between_anchors(self):
        P = 40
        evicted = list(range(4, 24))  # |E| = 20
        scores = [float(i) for i in range(P)]
        conds = causal_c007.build_interventions(P, evicted, scores=scores, num_sink=2, recency_window=2)
        anchors = {frozenset(conds["c0"]), frozenset(conds["h2o"])}
        graded = [k for k in conds if k not in ("c0", "h2o", "matched_random")]
        self.assertTrue(graded, "expected graded rescue/induction conditions")
        for name in graded:
            self.assertNotIn(frozenset(conds[name]), anchors, f"{name} collapsed onto an anchor")

    def test_old_tautological_construction_is_rejected(self):
        # The pre-Step-2 battery used rescue==pin(all E)==C0 and induction==mask(E)==H2O.
        # The public guard must reject both anchor-equal sets.
        evicted = list(range(4, 24))
        with self.assertRaises(causal_c007.TautologicalInterventionError):
            causal_c007.assert_distinct("rescue_all", [], evicted)          # == C0
        with self.assertRaises(causal_c007.TautologicalInterventionError):
            causal_c007.assert_distinct("induction_all", evicted, evicted)  # == H2O
        # A strict subset must be accepted.
        causal_c007.assert_distinct("partial", evicted[:5], evicted)

    def test_constructor_never_emits_anchor(self):
        # The constructor is defensive (clamps f*k to 1..k-1); across fractions it must
        # never yield a graded condition equal to an anchor.
        P = 40
        evicted = list(range(4, 24))
        conds = causal_c007.build_interventions(
            P, evicted, rescue_fracs=(0.1, 0.5, 0.9, 1.0), induction_fracs=(0.1, 0.5, 0.9, 1.0),
        )
        anchors = {frozenset(conds["c0"]), frozenset(conds["h2o"])}
        for name, pos in conds.items():
            if name in ("c0", "h2o"):
                continue
            self.assertNotIn(frozenset(pos), anchors, f"{name} must not equal an anchor")

    def test_evaluate_v2_runs_and_is_distinct(self):
        m = _tiny_model(1)
        ids = torch.randint(0, 200, (1, 40))
        res = causal_c007.evaluate_causal_v2(m, _DummyTok(), ids, budget=8, max_new_tokens=4, device="cpu")
        self.assertIn("c0", res["conditions"])
        self.assertIn("h2o", res["conditions"])
        self.assertTrue(any(c.startswith("partial_rescue") for c in res["conditions"]))


class TestMatchedRandomSP(unittest.TestCase):
    def test_histogram_matched_and_distinct(self):
        P = 60
        candidates = list(range(2, 58))
        scores = [float((7 * p) % 23) for p in range(P)]
        E = sorted(candidates, key=lambda p: scores[p])[:20]  # H2O-like: lowest scores
        sp = causal_c007.matched_random_sp(E, candidates, scores, random_seed=3)
        self.assertIsNotNone(sp)
        self.assertEqual(len(sp), len(E))
        self.assertNotEqual(set(sp), set(E))
        pool = sorted(set(candidates) | set(E))
        pos_s, sc_s = [float(p) for p in pool], sorted(scores[p] for p in pool)

        def hist(ps):
            out = {}
            for p in ps:
                k = (causal_c007._quantile_bin(float(p), pos_s, 5), causal_c007._quantile_bin(scores[p], sc_s, 5))
                out[k] = out.get(k, 0) + 1
            return out

        self.assertEqual(hist(sp), hist(E), "joint (position, score) histogram must match E")

    def test_infeasible_returns_none(self):
        # Every candidate evicted -> the only histogram match is E itself.
        self.assertIsNone(causal_c007.matched_random_sp([2, 3, 4], [2, 3, 4], [0.0] * 8))


class TestRegionInterventions(unittest.TestCase):
    def test_regions_minus_protected_and_controls_outside(self):
        P = 50
        regions = {"question": list(range(40, 46)), "template": [0, 1, 2, 3, 47, 48, 49],
                   "context": list(range(5, 40)), "gold": list(range(10, 18))}
        E = list(range(20, 30))
        conds, skipped = causal_c007.build_region_interventions(P, regions, E, num_sink=2, recency_window=2)
        self.assertEqual(conds["delete_question"], list(range(40, 46)))
        self.assertEqual(conds["delete_template"], [2, 3, 47])  # sinks 0,1 and recency 48,49 protected
        self.assertEqual(conds["delete_gold"], list(range(10, 18)))
        self.assertNotIn(12, conds["delete_context_nongold"])
        for name in ("delete_question", "delete_gold"):
            ctrl = conds[name + "_matched_random"]
            self.assertEqual(len(ctrl), len(conds[name]))
            self.assertFalse(set(ctrl) & set(conds[name]), "control must come from outside the region")
            self.assertFalse(set(ctrl) & {0, 1, 48, 49})
        # 27 non-gold context tokens but only 19 deletable positions outside them:
        # the size-matched control is infeasible and must be reported, not faked.
        self.assertEqual(set(skipped), {"delete_context_nongold_matched_random"})

    def test_region_equal_to_E_is_skipped(self):
        conds, skipped = causal_c007.build_region_interventions(
            30, {"question": list(range(10, 15))}, list(range(10, 15)))
        self.assertNotIn("delete_question", conds)
        self.assertIn("delete_question", skipped)


class TestLayerRestricted(unittest.TestCase):
    def setUp(self):
        self.m = _tiny_model(3)
        self.ids = torch.randint(0, 200, (1, 30))

    def test_anchor_bands_rejected(self):
        with self.assertRaises(causal_c007.TautologicalInterventionError):
            causal_c007.evict_only_at_layers(self.m, [], 4)
        with self.assertRaises(causal_c007.TautologicalInterventionError):
            causal_c007.evict_only_at_layers(self.m, [0, 1, 2, 3], 4)
        self.assertEqual(causal_c007.layer_bands(4, 4), [[0], [1], [2], [3]])
        self.assertEqual(causal_c007.layer_bands(28, 4), [list(range(0, 7)), list(range(7, 14)),
                                                          list(range(14, 21)), list(range(21, 28))])

    def test_band_is_noop_under_c0_and_between_anchors_under_eviction(self):
        P = self.ids.shape[1]
        full = torch.ones(1, P, dtype=torch.long)
        ev = full.clone()
        ev[0, 5:15] = 0
        with torch.no_grad():
            c0 = self.m(self.ids, attention_mask=full).logits
            h2o = self.m(self.ids, attention_mask=ev).logits
            with causal_c007.evict_only_at_layers(self.m, [1, 2], 4):
                band_c0 = self.m(self.ids, attention_mask=full).logits
                band_ev = self.m(self.ids, attention_mask=ev).logits
        self.assertLess((band_c0 - c0).abs().max().item(), 1e-5, "band restriction must be a no-op under C0")
        self.assertGreater((band_ev - c0).abs().max().item(), 1e-4)
        self.assertGreater((band_ev - h2o).abs().max().item(), 1e-4)


def _qwen_tokenizer():
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", local_files_only=True)
    except Exception:
        return None


@unittest.skipIf(_qwen_tokenizer() is None, "Qwen2.5 tokenizer not cached locally")
class TestTokenRegions(unittest.TestCase):
    def setUp(self):
        self.tok = _qwen_tokenizer()

    def _decode(self, ids, idx):
        return self.tok.decode([ids[i] for i in idx])

    def test_system_placed_hotpot(self):
        sysp = ("You are a helpful assistant. Answer using the passages below.\n\n"
                "Passage 1: Alpha\nAlpha is a town.\n\nPassage 2: Beta\nBeta is a river near Alpha.")
        rec = {"id": "r", "user_prompt": "Which river is near Alpha?", "question": "Which river is near Alpha?",
               "system_prompt": sysp, "context_placement": "system", "gold_passages": [2]}
        ids, reg = causal_c007.token_regions(self.tok, rec, use_system_prompt=True)
        self.assertEqual(self._decode(ids, reg["question"]).strip(), "Which river is near Alpha?")
        self.assertIn("Beta is a river", self._decode(ids, reg["gold"]))
        self.assertNotIn("Alpha is a town", self._decode(ids, reg["gold"]))
        self.assertIn("<|im_start|>", self._decode(ids, reg["template"]))
        self.assertFalse(set(reg["question"]) & set(reg["context"]))
        self.assertFalse(set(reg["template"]) & (set(reg["context"]) | set(reg["question"])))

    def test_user_placed_and_short(self):
        rec = {"id": "u", "user_prompt": "Doc text here.\n\nQuestion: What is here?\nAnswer:",
               "question": "What is here?", "system_prompt": "", "context_placement": "user"}
        ids, reg = causal_c007.token_regions(self.tok, rec)
        self.assertEqual(self._decode(ids, reg["question"]).strip(), "What is here?")
        self.assertIn("Doc text here", self._decode(ids, reg["context"]))
        short = {"id": "s", "user_prompt": "Name a fruit.", "question": "Name a fruit.",
                 "system_prompt": "", "context_placement": "none"}
        _, reg = causal_c007.token_regions(self.tok, short)
        self.assertEqual(reg["context"], [])
        self.assertTrue(reg["question"])

    def test_encode_record_matches_region_ids(self):
        # The ids the runners feed the model must be the ids the regions index into.
        from src.pfseb import protocol_c007

        rec = {"id": "e", "user_prompt": "What is here?", "question": "What is here?",
               "system_prompt": "Doc text here.", "context_placement": "system"}
        ids, _ = causal_c007.token_regions(self.tok, rec, use_system_prompt=True)
        enc = protocol_c007.encode_record(self.tok, rec, "cpu", use_system_prompt=True)
        self.assertEqual(enc.tolist(), [list(ids)])


class TestEvaluateV2Extended(unittest.TestCase):
    def test_regions_and_layer_bands_run(self):
        m = _tiny_model(1)
        ids = torch.randint(0, 200, (1, 40))
        regions = {"question": list(range(30, 36)), "template": [0, 1, 2, 37, 38, 39], "context": list(range(3, 30))}
        res = causal_c007.evaluate_causal_v2(m, _DummyTok(), ids, budget=8, max_new_tokens=2, device="cpu",
                                             regions=regions, n_layer_bands=2)
        self.assertIn("delete_question", res["conditions"])
        self.assertIn("matched_random_sp", res["conditions"])
        self.assertTrue(any(c.startswith("h2o_layers_") for c in res["conditions"]))


class TestAdapterApplied(unittest.TestCase):
    def test_lora_changes_logits(self):
        from src.pfseb.lora import add_lora

        m = _tiny_model(2)
        ids = torch.randint(0, 200, (1, 20))
        with torch.no_grad():
            base = m(ids).logits.clone()
        add_lora(m, targets=("q_proj", "k_proj", "v_proj", "o_proj"), r=4, alpha=8)
        # Zeroed adapter (B=0) must be identity.
        with torch.no_grad():
            zeroed = m(ids).logits
        self.assertLess((zeroed - base).abs().max().item(), 1e-4, "LoRA at init must be identity")
        # Perturb A and B so the adapter is non-trivial.
        changed = False
        with torch.no_grad():
            for n, p in m.named_parameters():
                if n.endswith(".A") or n.endswith(".B"):
                    p.add_(0.05)
                    changed = True
        self.assertTrue(changed, "expected LoRA .A/.B parameters to exist")
        with torch.no_grad():
            adapted = m(ids).logits
        self.assertGreater((adapted - base).abs().max().item(), 1e-4,
                           "a non-trivial adapter must change logits (theta_b != theta_c)")


class _DummyTok:
    eos_token_id = 1

    def decode(self, ids, skip_special_tokens=True):
        return "x"


if __name__ == "__main__":
    unittest.main(verbosity=2)
