"""Unit tests for src/pfseb/defenses.py verifying all R2 requirements."""

import unittest
import math
import numpy as np

# Import SimpleTensor shim from test_campaign_005
from tests.test_campaign_005 import (
    SimpleTensor,
    to_tensor,
    tensor_ones,
    tensor_zeros,
)

# Import module under test
from src.pfseb.defenses import (
    DefenseConfig,
    apply_spin_defense,
    calculate_spin_compression_ratio,
    evaluate_spin_defense,
    compute_levict_mask_for_layer,
    calculate_levict_compression_ratio,
    LEvictContext,
    evaluate_levict_defense,
    clamp_guardrail_budget,
    apply_budget_guardrail,
    calculate_guardrail_memory_overhead,
    evaluate_guardrail_defense,
    apply_compound_defense,
    calculate_compound_compression_ratio,
    CompoundDefenseContext,
    evaluate_compound_defense,
    run_defense_battery,
)


class TestDefenseConfig(unittest.TestCase):
    def test_default_config(self):
        cfg = DefenseConfig()
        self.assertEqual(cfg.defense_type, "none")
        self.assertEqual(cfg.pin_k, 4)
        self.assertEqual(cfg.k, 4)
        self.assertEqual(cfg.pin_strategy, "attention")
        self.assertEqual(cfg.strategy, "attention")
        self.assertEqual(cfg.guardrail_budget, 32)
        self.assertEqual(cfg.safe_budget, 32)
        self.assertEqual(cfg.critical_layers, [])

    def test_custom_config_and_dict(self):
        cfg = DefenseConfig(
            defense_type="compound",
            pin_k=6,
            pin_strategy="sink",
            critical_layers=[2, 3, 4],
            guardrail_budget=32,
        )
        d = cfg.to_dict()
        self.assertEqual(d["defense_type"], "compound")
        self.assertEqual(d["pin_k"], 6)
        self.assertEqual(d["critical_layers"], [2, 3, 4])
        restored = DefenseConfig.from_dict(d)
        self.assertEqual(restored.pin_strategy, "sink")
        self.assertEqual(restored.critical_layers, [2, 3, 4])

    def test_invalid_type_raises(self):
        with self.assertRaises(ValueError):
            DefenseConfig(defense_type="invalid_type")

    def test_invalid_strategy_raises(self):
        with self.assertRaises(ValueError):
            DefenseConfig(pin_strategy="invalid_strat")


class TestSPinDefense(unittest.TestCase):
    def setUp(self):
        self.prompt_len = 40
        self.budget = 8
        self.evicted = list(range(2, 34))  # 32 tokens
        self.scores = tensor_zeros(1, self.prompt_len)
        self.scores[0, 5] = 10.0
        self.scores[0, 9] = 20.0
        self.scores[0, 15] = 30.0
        self.scores[0, 22] = 40.0
        self.scores[0, 28] = 50.0
        self.scores[0, 31] = 60.0

    def test_spin_pin_k2(self):
        pinned = apply_spin_defense(self.evicted, self.scores, k=2, strategy="attention")
        self.assertEqual(len(pinned), 2)
        self.assertTrue(pinned.issubset(set(self.evicted)))

    def test_spin_pin_k4(self):
        pinned = apply_spin_defense(self.evicted, self.scores, k=4, strategy="attention")
        self.assertEqual(len(pinned), 4)
        self.assertTrue(pinned.issubset(set(self.evicted)))

    def test_spin_pin_k6(self):
        pinned = apply_spin_defense(self.evicted, self.scores, k=6, strategy="attention")
        self.assertEqual(len(pinned), 6)
        self.assertTrue(pinned.issubset(set(self.evicted)))

    def test_spin_attention_heavy_hitters(self):
        pinned = apply_spin_defense(self.evicted, self.scores, k=2, strategy="attention")
        self.assertEqual(pinned, {31, 28})

    def test_spin_boundary_formatting(self):
        pinned = apply_spin_defense(self.evicted, self.scores, k=2, strategy="boundary")
        self.assertEqual(len(pinned), 2)
        self.assertTrue(pinned.issubset(set(self.evicted)))

    def test_spin_sink_expansion(self):
        pinned = apply_spin_defense(self.evicted, self.scores, k=2, strategy="sink")
        self.assertEqual(pinned, {2, 3})

    def test_spin_compression_ratio(self):
        for k in [2, 4, 6]:
            eff_budget = self.budget + k
            comp = calculate_spin_compression_ratio(self.prompt_len, base_budget=self.budget, pin_k=k)
            expected = round((self.prompt_len - eff_budget) / float(self.prompt_len), 4)
            self.assertEqual(comp, expected)
            self.assertGreaterEqual(comp, 0.60)

    def test_spin_boundary_k0(self):
        self.assertEqual(apply_spin_defense(self.evicted, self.scores, k=0), set())

    def test_spin_boundary_negative_k(self):
        self.assertEqual(apply_spin_defense(self.evicted, self.scores, k=-2), set())

    def test_spin_boundary_empty_evicted(self):
        self.assertEqual(apply_spin_defense([], self.scores, k=4), set())

    def test_spin_boundary_k_ge_evicted(self):
        pinned = apply_spin_defense(self.evicted, self.scores, k=100)
        self.assertEqual(pinned, set(self.evicted))

    def test_spin_uniform_tied_scores(self):
        tied_scores = tensor_ones(1, 30)
        evicted = [5, 10, 15, 20]
        pinned1 = apply_spin_defense(evicted, tied_scores, k=2, strategy="attention")
        pinned2 = apply_spin_defense(evicted, tied_scores, k=2, strategy="attention")
        self.assertEqual(pinned1, pinned2)
        self.assertEqual(len(pinned1), 2)


class TestLEvictDefense(unittest.TestCase):
    def setUp(self):
        self.full_mask = tensor_ones(1, 1, 1, 36)
        self.evict_mask = tensor_zeros(1, 1, 1, 36)

    def test_levict_mask_routing(self):
        crit = [2, 3, 4, 5]
        mask_crit = compute_levict_mask_for_layer(3, crit, self.full_mask, self.evict_mask)
        self.assertIs(mask_crit, self.full_mask)

        mask_non_crit = compute_levict_mask_for_layer(10, crit, self.full_mask, self.evict_mask)
        self.assertIs(mask_non_crit, self.evict_mask)

    def test_levict_compression_ratio_k2(self):
        r = calculate_levict_compression_ratio(total_layers=28, num_critical_layers=2, prompt_len=36, budget=8)
        self.assertGreaterEqual(r, 0.70)
        self.assertEqual(r, round((26 / 28) * (28 / 36), 4))

    def test_levict_compression_ratio_k4(self):
        r = calculate_levict_compression_ratio(total_layers=28, num_critical_layers=4, prompt_len=36, budget=8)
        self.assertGreaterEqual(r, 0.65)
        self.assertEqual(r, round((24 / 28) * (28 / 36), 4))

    def test_levict_compression_ratio_k6(self):
        r = calculate_levict_compression_ratio(total_layers=28, num_critical_layers=6, prompt_len=36, budget=8)
        self.assertGreaterEqual(r, 0.60)
        self.assertEqual(r, round((22 / 28) * (28 / 36), 4))

    def test_levict_exceeding_k6_drops_efficiency(self):
        r = calculate_levict_compression_ratio(total_layers=28, num_critical_layers=12, prompt_len=36, budget=8)
        self.assertLess(r, 0.60)

    def test_levict_zero_critical(self):
        for l in range(28):
            m = compute_levict_mask_for_layer(l, [], self.full_mask, self.evict_mask)
            self.assertIs(m, self.evict_mask)

    def test_levict_all_critical(self):
        all_l = list(range(28))
        for l in range(28):
            m = compute_levict_mask_for_layer(l, all_l, self.full_mask, self.evict_mask)
            self.assertIs(m, self.full_mask)
        r = calculate_levict_compression_ratio(28, 28, 36, 8)
        self.assertEqual(r, 0.0)

    def test_levict_context_lifecycle(self):
        ctx = LEvictContext(model=None, critical_layers=[2, 3])
        with ctx:
            self.assertTrue(ctx.handles_active)
        self.assertFalse(ctx.handles_active)


class TestBudgetGuardrail(unittest.TestCase):
    def test_clamp_unsafe_budget(self):
        self.assertEqual(clamp_guardrail_budget(8, safe_budget=32), 32)
        self.assertEqual(clamp_guardrail_budget(20, safe_budget=32), 32)
        self.assertEqual(clamp_guardrail_budget(22, safe_budget=32), 32)

    def test_clamp_preserves_safe_budget(self):
        self.assertEqual(clamp_guardrail_budget(48, safe_budget=32), 48)

    def test_clamp_with_prompt_len(self):
        self.assertEqual(clamp_guardrail_budget(8, safe_budget=32, prompt_len=20), 20)
        self.assertEqual(clamp_guardrail_budget(8, safe_budget=32, prompt_len=4096), 32)

    def test_clamp_zero_or_negative(self):
        self.assertEqual(clamp_guardrail_budget(0, safe_budget=32), 32)
        self.assertEqual(clamp_guardrail_budget(-16, safe_budget=32), 32)

    def test_memory_overhead_calculation(self):
        bytes_overhead = calculate_guardrail_memory_overhead(delta_tokens=24)
        kb_overhead = bytes_overhead / 1024.0
        self.assertAlmostEqual(kb_overhead, 672.0, places=1)
        self.assertLess(kb_overhead, 700.0)

    def test_overhead_fraction_of_7gb(self):
        bytes_overhead = calculate_guardrail_memory_overhead(delta_tokens=24)
        fraction = bytes_overhead / (7.0 * 1024.0 ** 3)
        self.assertLess(fraction, 0.0001)

    def test_apply_budget_guardrail(self):
        cfg = DefenseConfig(base_budget=8, guardrail_budget=32)
        guarded_cfg = apply_budget_guardrail(cfg)
        self.assertEqual(guarded_cfg.base_budget, 32)


class TestCompoundDefenseAndBattery(unittest.TestCase):
    def test_compound_defense_mask_and_compression(self):
        total_layers = 28
        num_crit = 4
        prompt_len = 40
        budget = 8
        pin_k = 4
        eff_budget = budget + pin_k
        comp = calculate_compound_compression_ratio(
            total_layers=total_layers,
            num_critical_layers=num_crit,
            prompt_len=prompt_len,
            budget=budget,
            pin_k=pin_k,
        )
        expected = round(((total_layers - num_crit) / total_layers) * ((prompt_len - eff_budget) / prompt_len), 4)
        self.assertEqual(comp, expected)
        self.assertGreaterEqual(comp, 0.60)

    def test_battery_runner(self):
        battery_res = run_defense_battery(prompt_len=40, base_budget=8, critical_layers=[2, 3, 4, 5])
        self.assertTrue(battery_res["verdicts"]["spin_compression_ge_60"])
        self.assertTrue(battery_res["verdicts"]["levict_compression_ge_60"])
        self.assertTrue(battery_res["verdicts"]["guardrail_overhead_bounded"])
        self.assertTrue(battery_res["verdicts"]["compound_compression_ge_60"])


if __name__ == "__main__":
    unittest.main()
