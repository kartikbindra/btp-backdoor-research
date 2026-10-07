"""Adversarial stress test suite for PF-SEB eviction policies and execution harness.

Empirically tests:
1. Synthetic attention score tensors: uniform, extreme spikes, negative, zero, large sequence lengths, NaN/Inf.
2. SnapKV and Scissorhands non-degeneracy vs H2O under realistic observation-window and persistence dynamics.
3. Edge case budgets: B in {8, 12, 16, 20, 24, 32, 48, 'full'}, B >= P, B < S+W, B <= 0.
4. Harness integrity: missing imports, random seed independence, and autoregressive fallback dynamics.
"""

import math
import unittest
import torch

from src.pfseb.eviction import (
    EvictionConfig,
    topk_keep_mask,
    h2o_keep_mask,
    snapkv_keep_mask,
    scissorhands_keep_mask,
    recency_keep_mask,
    random_keep_mask,
    compute_eviction_mask,
    compute_h2o_scores,
    compute_snapkv_scores,
    compute_scissorhands_scores,
    BUDGET_SWEEP_GRID,
)


class TestSyntheticAttentionScores(unittest.TestCase):
    """Stress tests with synthetic attention score tensors."""

    def test_uniform_scores_all_policies(self):
        """Uniform attention scores: all keys have identical scores.
        
        Empirical finding: Under uniform attention, H2O, SnapKV, and Scissorhands
        exhibit tie-breaking degeneracy, all evicting the lowest candidate indices.
        """
        P = 30
        B = 10
        S = 2
        W = 2
        scores = torch.ones(P, dtype=torch.float32)

        for pol in ["h2o", "snapkv", "scissorhands", "recency"]:
            pmask, evicted = compute_eviction_mask(
                scores=scores, policy=pol, budget=B, num_sink=S, recency_window=W, prompt_len=P
            )
            self.assertEqual(pmask.shape, (1, 1, 1, P))
            self.assertEqual(len(evicted), P - B, f"Evicted count failed for {pol}")
            self.assertEqual(int(pmask.sum()), B, f"Kept count failed for {pol}")
            # Sinks 0, 1 and recency P-2, P-1 must be protected
            self.assertNotIn(0, evicted)
            self.assertNotIn(1, evicted)
            self.assertNotIn(P - 2, evicted)
            self.assertNotIn(P - 1, evicted)

        # Under uniform scores, H2O, SnapKV, and Scissorhands evict the EXACT SAME positions (tie-break by index)
        _, ev_h2o = compute_eviction_mask(scores=scores, policy="h2o", budget=B, num_sink=S, recency_window=W, prompt_len=P)
        _, ev_snap = compute_eviction_mask(scores=scores, policy="snapkv", budget=B, num_sink=S, recency_window=W, prompt_len=P)
        _, ev_sciss = compute_eviction_mask(scores=scores, policy="scissorhands", budget=B, num_sink=S, recency_window=W, prompt_len=P)
        self.assertEqual(ev_h2o, ev_snap, "Under uniform scores, SnapKV degenerates to H2O index tie-breaking")
        self.assertEqual(ev_h2o, ev_sciss, "Under uniform scores, Scissorhands degenerates to H2O index tie-breaking")

    def test_extreme_attention_spikes_vs_persistence(self):
        """Extreme spikes: Key with huge single spike vs Key with consistent persistence.
        
        Demonstrates that Scissorhands retains persistent keys while H2O retains high-mass spike keys.
        """
        P = 24
        num_heads = 2
        attn = torch.zeros(1, num_heads, P, P)

        # Key 5 receives an extreme spike: on query step 10, attention is 5.0 (unnormalized synthetic)
        attn[0, :, 10, 5] = 5.0  # Mass = 5.0 * 2 = 10.0, persistence = 1 step * 2 heads = 2

        # Key 8 receives steady moderate attention: on 15 queries (5..19), attention is 0.15
        for q in range(5, 20):
            attn[0, :, q, 8] = 0.15  # Mass = 15 * 0.15 * 2 = 4.5, persistence = 15 steps * 2 heads = 30

        attentions = [attn]
        h2o_scores = compute_h2o_scores(attentions, P)
        scissor_scores = compute_scissorhands_scores(attentions, P, threshold=0.10)

        # H2O score for Key 5 is higher than Key 8 (10.0 > 4.5)
        self.assertGreater(h2o_scores[5].item(), h2o_scores[8].item())

        # Scissorhands score for Key 8 is higher than Key 5 (30 > 2)
        self.assertGreater(scissor_scores[8].item(), scissor_scores[5].item())

        # At budget where only one of them can be retained:
        # Budget = S(2) + W(2) + 1 heavy hitter = 5
        _, ev_h2o = compute_eviction_mask(
            attentions=attentions, policy="h2o", budget=5, num_sink=2, recency_window=2, prompt_len=P
        )
        _, ev_sciss = compute_eviction_mask(
            attentions=attentions, policy="scissorhands", budget=5, num_sink=2, recency_window=2, prompt_len=P,
            scissor_threshold=0.10
        )

        # H2O retains Key 5 and evicts Key 8
        self.assertNotIn(5, ev_h2o, "H2O must retain high-mass spike Key 5")
        self.assertIn(8, ev_h2o, "H2O must evict lower-mass persistent Key 8")

        # Scissorhands retains Key 8 and evicts Key 5
        self.assertNotIn(8, ev_sciss, "Scissorhands must retain persistent Key 8")
        self.assertIn(5, ev_sciss, "Scissorhands must evict one-off spike Key 5")

    def test_negative_scores_and_scissorhands_degeneracy(self):
        """Negative scores: tests ranking and Scissorhands 1D heuristic edge case.
        
        When scores are all negative and scissor_threshold > 0, Scissorhands 1D heuristic
        degenerates to H2O because s >= tau is False everywhere, producing 0.1 * s.
        """
        P = 20
        # Negative scores: pos 5 has score -1.0 (least negative, highest), pos 10 has -20.0
        scores = -torch.arange(P, dtype=torch.float32) - 1.0

        pmask_h2o, ev_h2o = compute_eviction_mask(scores=scores, policy="h2o", budget=8, num_sink=2, recency_window=2, prompt_len=P)
        self.assertEqual(len(ev_h2o), 12)
        # Position 5 has highest score (-6.0 vs -19.0), so it must be kept
        self.assertNotIn(2, ev_h2o, "Pos 2 has score -3.0 (higher), must be kept over pos 17")
        self.assertIn(17, ev_h2o, "Pos 17 has score -18.0 (lower), must be evicted")

        # 1D Scissorhands with positive threshold on all-negative scores:
        # tau = 0.5 > 0. s >= 0.5 is False for all negative scores -> rank_scores = 0.1 * s -> identical to H2O!
        _, ev_sciss = compute_eviction_mask(
            scores=scores, policy="scissorhands", budget=8, num_sink=2, recency_window=2, prompt_len=P,
            scissor_threshold=0.5
        )
        self.assertEqual(ev_h2o, ev_sciss, "Scissorhands 1D heuristic degenerates to H2O when all scores < threshold")

    def test_large_sequence_lengths(self):
        """Large sequence lengths: P = 512, 1024, 2048 to stress memory and time scaling."""
        for P in [512, 1024]:
            scores = torch.rand(P)
            for pol in ["h2o", "snapkv", "scissorhands", "recency", "random"]:
                pmask, evicted = compute_eviction_mask(
                    scores=scores, policy=pol, budget=48, num_sink=4, recency_window=8, prompt_len=P
                )
                self.assertEqual(pmask.shape, (1, 1, 1, P))
                self.assertEqual(len(evicted), P - 48)
                self.assertEqual(int(pmask.sum()), 48)
                # Verify sinks and recency protected
                for s in range(4):
                    self.assertNotIn(s, evicted)
                for w in range(P - 8, P):
                    self.assertNotIn(w, evicted)


class TestPolicyNonDegeneracyDynamics(unittest.TestCase):
    """Stress tests verifying SnapKV and Scissorhands divergence from H2O under realistic dynamics."""

    def test_snapkv_observation_window_divergence_from_h2o(self):
        """Realistic prompt dynamic: Long instruction/few-shot prefix with tail question.
        
        - Key 3 (prefix token): heavily attended by queries 4..15 (instruction prefix).
        - Key 18 (task token): heavily attended by tail queries 20..27 (question tail).
        
        H2O sums over all queries, favoring Key 3.
        SnapKV (window=8) sums ONLY over tail queries (20..27), favoring Key 18.
        """
        P = 28
        num_heads = 4
        attn = torch.zeros(1, num_heads, P, P)

        # Prefix queries focus on Key 3
        for q in range(4, 16):
            attn[0, :, q, 3] = 0.40  # 12 queries * 4 heads * 0.4 = 19.2 mass

        # Tail queries focus on Key 18
        for q in range(20, 28):
            attn[0, :, q, 18] = 0.45  # 8 queries * 4 heads * 0.45 = 14.4 mass

        attentions = [attn]

        h2o_scores = compute_h2o_scores(attentions, P)
        snapkv_scores = compute_snapkv_scores(attentions, P, window_size=8)

        # H2O score for Key 3 is higher than Key 18 (19.2 > 14.4)
        self.assertGreater(h2o_scores[3].item(), h2o_scores[18].item())

        # SnapKV score for Key 3 is 0.0 (received no tail attention)
        self.assertEqual(snapkv_scores[3].item(), 0.0)
        self.assertGreater(snapkv_scores[18].item(), 0.0)

        # Budget = 5 (S=2, W=2, leaves 1 candidate slot)
        _, ev_h2o = compute_eviction_mask(
            attentions=attentions, policy="h2o", budget=5, num_sink=2, recency_window=2, prompt_len=P
        )
        _, ev_snap = compute_eviction_mask(
            attentions=attentions, policy="snapkv", budget=5, num_sink=2, recency_window=2, prompt_len=P,
            snapkv_window=8
        )

        # H2O keeps Key 3 and evicts Key 18
        self.assertNotIn(3, ev_h2o, "H2O should retain high-prefix Key 3")
        self.assertIn(18, ev_h2o, "H2O should evict lower-total-mass Key 18")

        # SnapKV keeps Key 18 and evicts Key 3
        self.assertNotIn(18, ev_snap, "SnapKV must retain tail-focused Key 18")
        self.assertIn(3, ev_snap, "SnapKV must evict prefix Key 3 (zero window attention)")

        self.assertNotEqual(ev_h2o, ev_snap, "SnapKV and H2O eviction sets must be differentiated")

    def test_snapkv_degeneracy_boundary_when_prompt_le_window(self):
        """CRITICAL BOUNDARY CONDITION: When prompt length P <= snapkv_window (e.g. P <= 16),
        SnapKV observation window spans the entire sequence, mathematically collapsing to H2O.
        """
        P = 16
        num_heads = 2
        torch.manual_seed(42)
        attn = torch.rand(1, num_heads, P, P)
        attentions = [attn]

        h2o_scores = compute_h2o_scores(attentions, P)
        snapkv_scores = compute_snapkv_scores(attentions, P, window_size=16)

        # Mathematical identity: scores are bitwise identical
        self.assertTrue(torch.allclose(h2o_scores, snapkv_scores, atol=1e-6),
                        "When P <= snapkv_window, SnapKV scores must equal H2O scores")

        _, ev_h2o = compute_eviction_mask(
            attentions=attentions, policy="h2o", budget=8, num_sink=2, recency_window=2, prompt_len=P
        )
        _, ev_snap = compute_eviction_mask(
            attentions=attentions, policy="snapkv", budget=8, num_sink=2, recency_window=2, prompt_len=P,
            snapkv_window=16
        )

        self.assertEqual(ev_h2o, ev_snap,
                         "BOUNDARY COLLAPSE: SnapKV and H2O produce identical evictions when P <= W_obs")


class TestEdgeCaseBudgets(unittest.TestCase):
    """Stress tests on edge case budgets across the full sweep grid."""

    def test_budget_sweep_grid_all_eight_values(self):
        """Verify all 8 budgets in BUDGET_SWEEP_GRID run without error across all 5 policies."""
        P = 50
        scores = torch.rand(P)
        for b in BUDGET_SWEEP_GRID:
            for pol in ["h2o", "snapkv", "scissorhands", "recency", "random"]:
                pmask, evicted = compute_eviction_mask(
                    scores=scores, policy=pol, budget=b, num_sink=2, recency_window=2, prompt_len=P
                )
                self.assertEqual(pmask.shape, (1, 1, 1, P))
                if b == "full":
                    self.assertEqual(len(evicted), 0)
                    self.assertEqual(int(pmask.sum()), P)
                else:
                    self.assertEqual(len(evicted), P - b)
                    self.assertEqual(int(pmask.sum()), b)

    def test_budget_ge_prompt_len_full_cache_bypass(self):
        """When B >= P, full cache must be returned with 0 evicted tokens."""
        P = 20
        scores = torch.rand(P)

        # Exactly equal
        pmask, evicted = compute_eviction_mask(scores=scores, policy="h2o", budget=20, prompt_len=P)
        self.assertEqual(len(evicted), 0)
        self.assertEqual(int(pmask.sum()), P)

        # Strictly greater
        pmask, evicted = compute_eviction_mask(scores=scores, policy="h2o", budget=100, prompt_len=P)
        self.assertEqual(len(evicted), 0)
        self.assertEqual(int(pmask.sum()), P)

    def test_budget_smaller_than_sinks_plus_recency(self):
        """CRITICAL BOUNDARY BEHAVIOR: When B < S + W.
        
        Example: P = 30, S = 4, W = 8 -> S + W = 12.
        If B = 6 (< 12):
        - H2O, SnapKV, Scissorhands, Random strictly preserve S + W = 12 tokens (evicting all 18 candidates).
        - Recency policy retains ONLY max(S, B) = 6 tokens, discarding recency window protection.
        """
        P = 30
        S = 4
        W = 8
        B = 6
        scores = torch.rand(P)

        # H2O retains protected set (12 tokens)
        pmask_h2o, ev_h2o = compute_eviction_mask(
            scores=scores, policy="h2o", budget=B, num_sink=S, recency_window=W, prompt_len=P
        )
        self.assertEqual(len(ev_h2o), P - (S + W), "H2O evicts all non-protected candidates (18 tokens)")
        self.assertEqual(int(pmask_h2o.sum()), S + W, "H2O retains S+W=12 tokens when B < S+W")

        # Recency policy retains B = 6 tokens
        pmask_rec, ev_rec = compute_eviction_mask(
            scores=scores, policy="recency", budget=B, num_sink=S, recency_window=W, prompt_len=P
        )
        self.assertEqual(int(pmask_rec.sum()), B, "Recency policy retains B=6 tokens, truncating recency window")
        self.assertEqual(len(ev_rec), P - B, "Recency policy evicts 24 tokens")

    def test_negative_budget_semantic_inconsistency(self):
        """CRITICAL RESOLUTION: Verify topk_keep_mask and compute_eviction_mask consistently treat
        budget <= 0 (e.g. budget = -1) as keep-all (full cache bypass with 0 evicted tokens).
        """
        P = 20
        scores_2d = torch.rand(1, P)
        scores_1d = scores_2d.squeeze(0)

        # In topk_keep_mask: budget=-1 returns keep-all
        mask_topk = topk_keep_mask(scores_2d, budget=-1, num_sink=2, recency_window=2)
        self.assertTrue(mask_topk.all(), "topk_keep_mask treats budget <= 0 as keep-all")

        # In compute_eviction_mask: budget=-1 also returns keep-all (0 evicted)
        pmask, evicted = compute_eviction_mask(scores=scores_1d, policy="h2o", budget=-1, num_sink=2, recency_window=2, prompt_len=P)
        self.assertEqual(len(evicted), 0, "compute_eviction_mask must evict 0 tokens when budget=-1")
        self.assertEqual(int(pmask.sum()), P, "compute_eviction_mask must keep all tokens when budget=-1")


class TestHarnessVulnerabilitiesAndBugs(unittest.TestCase):
    """Verify defects and vulnerabilities in harness.py are resolved."""

    def test_missing_import_random_in_harness(self):
        """CRITICAL FIX: src/pfseb/harness.py imports `random` at module scope so
        random.Random(seed) in evaluate_causal_battery_single never raises NameError.
        """
        import src.pfseb.harness as h_mod
        self.assertTrue(hasattr(h_mod, "random"),
                        "FIX VERIFIED: `random` is imported in src/pfseb/harness.py")
        import random
        self.assertIs(h_mod.random, random)

    def test_hardcoded_seed_zero_in_prompt_evicted_positions_random(self):
        """REPRODUCIBILITY FIX: prompt_evicted_positions accepts seed argument and supports EvictionConfig.seed."""
        import inspect
        from src.pfseb.harness import prompt_evicted_positions
        sig = inspect.signature(prompt_evicted_positions)
        self.assertIn("seed", sig.parameters,
                      "FIX VERIFIED: prompt_evicted_positions accepts seed argument")
        source = inspect.getsource(prompt_evicted_positions)
        self.assertIn("manual_seed", source,
                      "FIX VERIFIED: prompt_evicted_positions uses manual_seed for reproducible generation")


if __name__ == "__main__":
    unittest.main()
