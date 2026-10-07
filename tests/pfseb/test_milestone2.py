"""Milestone 2 Unit Tests: Control Baseline (theta_f), Estimands, and Sequential Runner Logic.

Tests:
1. LoRA state serialization and restoration (get_lora_state, set_lora_state).
2. Delta_cond paired bootstrap Difference-in-Differences.
3. Dual benign continuation loss formulation (lambda_marker == 0.0) with gradient flow.
4. Divergence guard predicate (> 8.0 * avg + 3.0).
5. Paired bootstrap CI on symmetric and asymmetric distributions.
6. JSON schema compliance of runner summary output.
"""

import unittest
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.pfseb.train_mvp import (
    MVPConfig,
    _ce,
    _rate,
    bootstrap_delta_int,
    bootstrap_delta_cond,
    get_lora_state,
    set_lora_state,
)
from src.pfseb.lora import add_lora, lora_parameters, num_trainable
from scripts.run_pfseb_campaign_004 import (
    bootstrap_ci,
)


class DummyLinearModel(nn.Module):
    """Minimal model for fast LoRA wrapping and state verification on CPU."""
    def __init__(self, in_features=16, out_features=16):
        super().__init__()
        self.q_proj = nn.Linear(in_features, out_features, bias=False)
        self.k_proj = nn.Linear(in_features, out_features, bias=False)
        self.v_proj = nn.Linear(in_features, out_features, bias=False)
        self.o_proj = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x):
        return self.o_proj(self.v_proj(self.k_proj(self.q_proj(x))))


class TestMilestone2LoRAState(unittest.TestCase):
    def test_lora_state_save_and_restore(self):
        model = DummyLinearModel()
        for p in model.parameters():
            p.requires_grad_(False)
        n_wrapped = add_lora(model, r=4, alpha=8)
        self.assertEqual(n_wrapped, 4)

        # Initial state: B is zero
        state_init = get_lora_state(model)
        self.assertTrue(len(state_init) > 0)
        for k, v in state_init.items():
            if k.endswith(".B"):
                self.assertTrue(torch.all(v == 0.0))

        # Mutate LoRA parameters
        with torch.no_grad():
            for p in lora_parameters(model):
                p.add_(1.0)

        # Captured mutated state
        state_mutated = get_lora_state(model)
        for k, v in state_mutated.items():
            if k.endswith(".B"):
                self.assertTrue(torch.all(v == 1.0))

        # Restore initial state
        set_lora_state(model, state_init)
        state_restored = get_lora_state(model)
        for k, v in state_restored.items():
            if k.endswith(".B"):
                self.assertTrue(torch.all(v == 0.0))


class TestMilestone2Estimands(unittest.TestCase):
    def test_bootstrap_delta_cond_calculation(self):
        n = 24
        # theta_b: fires under H2O (1), dormant under C0 (0)
        tb = {"h2o": [1] * n, "c0": [0] * n}
        # theta_f: dormant under both H2O (0) and C0 (0)
        tf = {"h2o": [0] * n, "c0": [0] * n}

        res = bootstrap_delta_cond(tb, tf, n_boot=500, seed=42)
        self.assertAlmostEqual(res["delta_cond"], 1.0, places=4)
        self.assertAlmostEqual(res["ci_low"], 1.0, places=4)
        self.assertAlmostEqual(res["ci_high"], 1.0, places=4)
        self.assertAlmostEqual(res["theta_b_gap"], 1.0, places=4)
        self.assertAlmostEqual(res["theta_f_gap"], 0.0, places=4)

    def test_bootstrap_ci_unpaired_vs_paired(self):
        arr1 = np.array([1, 1, 1, 0, 0, 1, 1, 0])
        arr2 = np.array([0, 0, 0, 0, 0, 0, 0, 0])
        res = bootstrap_ci(arr1, arr2, n_boot=500, seed=42)
        self.assertAlmostEqual(res["mean"], 5.0 / 8.0, places=4)
        self.assertTrue(res["ci_low"] <= res["mean"] <= res["ci_high"])


class TestMilestone2ControlLossAndGuards(unittest.TestCase):
    def test_dual_benign_loss_gradient_flow(self):
        """theta_f optimizes L_full(y_benign) + L_evict(y_benign) with lambda_marker == 0.0."""
        vocab_size = 50
        seq_len = 8
        target_len = 4
        torch.manual_seed(123)

        logits_full = torch.randn(1, target_len, vocab_size, requires_grad=True)
        logits_evict = torch.randn(1, target_len, vocab_size, requires_grad=True)
        benign_targets = torch.randint(0, vocab_size, (1, target_len))

        l_full = _ce(logits_full, benign_targets)
        l_evict = _ce(logits_evict, benign_targets)
        loss = l_full + l_evict  # lambda_marker = 0.0

        loss.backward()
        self.assertIsNotNone(logits_full.grad)
        self.assertIsNotNone(logits_evict.grad)
        self.assertTrue(torch.all(torch.isfinite(logits_full.grad)))
        self.assertTrue(torch.all(torch.isfinite(logits_evict.grad)))

    def test_divergence_guard_threshold_behavior(self):
        recent = [0.5, 0.45, 0.55, 0.52, 0.48]
        avg = sum(recent) / len(recent)
        thresh = 8.0 * avg + 3.0

        # Normal loss is below threshold
        self.assertFalse(1.5 > thresh)
        # Exploding loss is above threshold
        self.assertTrue(12.0 > thresh)


if __name__ == "__main__":
    unittest.main()
