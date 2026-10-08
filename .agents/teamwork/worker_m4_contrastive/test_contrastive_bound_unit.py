"""Comprehensive Unit Test Suite for src.pfseb.contrastive_bound.

Campaign 005 — Requirement R4 (Contrastive Multi-Policy Bound).
"""

import unittest
import numpy as np
import torch
import torch.nn as nn

from src.pfseb.contrastive_bound import (
    compute_jaccard_similarity,
    analytical_jaccard_lower_bound,
    compute_policy_jaccard_overlap,
    compute_representation_cosine_similarity,
    compute_gradient_conflict_metric,
    compute_contrastive_loss,
    generate_contrastive_bound_analysis,
    verify_contrastive_bound,
    CosineSimilarityResult,
    GradientConflictResult,
    ContrastiveLossResult,
    CANONICAL_PROMPT_LEN,
    CANONICAL_BUDGET,
    THEORETICAL_LOWER_BOUND,
)


class TestAnalyticalJaccardBound(unittest.TestCase):
    """Tests for analytical_jaccard_lower_bound and compute_jaccard_similarity."""

    def test_basic_jaccard_similarity(self):
        # Empty sets
        self.assertEqual(compute_jaccard_similarity(set(), set()), 1.0)
        # Disjoint sets
        self.assertEqual(compute_jaccard_similarity({1, 2}, {3, 4}), 0.0)
        # Identical sets
        self.assertEqual(compute_jaccard_similarity({1, 2, 3}, {1, 2, 3}), 1.0)
        # Overlapping sets
        a = {1, 2, 3, 4}
        b = {3, 4, 5, 6}
        self.assertAlmostEqual(compute_jaccard_similarity(a, b), 2.0 / 6.0, places=4)

    def test_canonical_lower_bound_ge_75(self):
        """Canonical P=36, B=8, S=2, W=2 must produce lower bound exactly 0.75 (75.0%)."""
        bound = analytical_jaccard_lower_bound(prompt_len=36, budget=8, num_sink=2, recency_window=2)
        self.assertEqual(bound, 0.75)
        self.assertGreaterEqual(bound, THEORETICAL_LOWER_BOUND)

    def test_bound_full_cache_bypass(self):
        """Full cache or non-evicting budget must yield bound 1.0."""
        self.assertEqual(analytical_jaccard_lower_bound(36, "full"), 1.0)
        self.assertEqual(analytical_jaccard_lower_bound(36, 36), 1.0)
        self.assertEqual(analytical_jaccard_lower_bound(36, 48), 1.0)
        self.assertEqual(analytical_jaccard_lower_bound(36, 0), 1.0)

    def test_bound_monotonicity_with_budget(self):
        """As budget increases (fewer tokens evicted), minimum intersection / max union changes."""
        # For B=8: |E|=28, |C|=32 -> min_inter=24, max_union=32 -> 24/32 = 0.75
        # For B=16: |E|=20, |C|=32 -> min_inter=8, max_union=32 -> 8/32 = 0.25
        b8 = analytical_jaccard_lower_bound(36, 8)
        b16 = analytical_jaccard_lower_bound(36, 16)
        self.assertEqual(b8, 0.75)
        self.assertEqual(b16, 0.25)


class TestPolicyJaccardOverlap(unittest.TestCase):
    """Tests for compute_policy_jaccard_overlap across policies."""

    def test_simulation_mode_canonical(self):
        res = compute_policy_jaccard_overlap(prompt_len=36, budget=8)
        self.assertIn("jaccard_lower_bound", res)
        self.assertIn("empirical_h2o_snapkv_overlap", res)
        self.assertIn("pairwise_overlaps", res)
        self.assertEqual(res["jaccard_lower_bound"], 0.75)
        self.assertGreaterEqual(res["empirical_h2o_snapkv_overlap"], 0.85)
        self.assertTrue(res["analytical_bound_satisfied"])

    def test_mock_model_execution(self):
        class MockAttentionOutput:
            def __init__(self, p_len):
                # Shape: [batch=1, n_heads=2, q_len=p_len, kv_len=p_len]
                attn = torch.ones(1, 2, p_len, p_len)
                # Boost sinks
                attn[:, :, :, :2] *= 10.0
                # Boost recency
                attn[:, :, :, -2:] *= 5.0
                self.attentions = (attn,)

        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
            def forward(self, input_ids, **kwargs):
                return MockAttentionOutput(input_ids.shape[-1])

        class MockTokenizer:
            def __call__(self, text, return_tensors="pt"):
                return {"input_ids": torch.zeros(1, 36, dtype=torch.long)}

        model = MockModel()
        tok = MockTokenizer()
        prompts = ["Test prompt 1", "Test prompt 2"]

        res = compute_policy_jaccard_overlap(
            prompts=prompts,
            tokenizer=tok,
            model=model,
            budget=8,
            policies=("h2o", "snapkv"),
        )
        self.assertIn("empirical_h2o_snapkv_overlap", res)
        self.assertEqual(len(res["per_prompt_results"]), 2)
        self.assertGreaterEqual(res["empirical_h2o_snapkv_overlap"], 0.75)


class TestRepresentationCosineSimilarity(unittest.TestCase):
    """Tests for compute_representation_cosine_similarity."""

    def test_1d_vectors(self):
        u = [1.0, 0.0, 0.0]
        v = [0.0, 1.0, 0.0]
        res_ortho = compute_representation_cosine_similarity(u, v)
        self.assertAlmostEqual(float(res_ortho), 0.0, places=4)

        w = [1.0, 0.0, 0.0]
        res_ident = compute_representation_cosine_similarity(u, w)
        self.assertAlmostEqual(float(res_ident), 1.0, places=4)

    def test_2d_hidden_states(self):
        # [seq_len=10, hidden_dim=64]
        h1 = torch.randn(10, 64)
        h2 = h1.clone() + 0.001 * torch.randn(10, 64)
        res = compute_representation_cosine_similarity(h1, h2)
        self.assertIsInstance(res, CosineSimilarityResult)
        self.assertGreater(float(res), 0.99)
        self.assertGreater(res.last_token_cosine, 0.99)
        self.assertEqual(res["hidden_dim"], 64)

    def test_duck_typing_and_float_behavior(self):
        res = compute_representation_cosine_similarity([1.0, 0.0], [0.99, 0.01])
        # Can be used directly in float comparisons
        self.assertTrue(res > 0.95)
        # Dictionary lookup works
        self.assertIn("cosine_similarity", res.to_dict())


class TestGradientConflictMetric(unittest.TestCase):
    """Tests for compute_gradient_conflict_metric."""

    def test_direct_opposing_gradients(self):
        g_marker = np.array([0.9, 0.8, -0.7, 0.6])
        g_benign = np.array([-0.85, -0.78, 0.68, -0.58])
        metric = compute_gradient_conflict_metric(g_marker, g_benign)
        self.assertIsInstance(metric, GradientConflictResult)
        self.assertLess(float(metric), -0.90)
        self.assertLess(metric["cosine_similarity"], -0.90)
        self.assertEqual(metric["conflict_severity"], "extreme")
        self.assertTrue(metric["is_conflicting"])
        self.assertGreater(metric["cancellation_score"], 0.90)

    def test_aligned_gradients(self):
        g1 = np.array([1.0, 2.0, 3.0])
        g2 = np.array([2.0, 4.0, 6.0])
        metric = compute_gradient_conflict_metric(g1, g2)
        self.assertAlmostEqual(metric["cosine_similarity"], 1.0, places=4)
        self.assertEqual(metric["conflict_severity"], "aligned")
        self.assertAlmostEqual(metric["cancellation_score"], 0.0, places=4)

    def test_duck_typing_and_float_behavior(self):
        g_marker = np.array([0.9, 0.8, -0.7, 0.6])
        g_benign = np.array([-0.85, -0.78, 0.68, -0.58])
        metric = compute_gradient_conflict_metric(g_marker, g_benign)
        # Direct comparison with float
        self.assertTrue(metric < -0.90)
        # Key access
        self.assertIn("norm_marker", metric)
        self.assertIn("norm_benign", metric)


class TestContrastiveLoss(unittest.TestCase):
    """Tests for compute_contrastive_loss."""

    def test_scalar_loss_combination(self):
        res = compute_contrastive_loss(
            loss_full=0.5,
            loss_h2o=0.4,
            loss_snapkv=0.6,
            lambda_pos=2.0,
            lambda_neg=1.5,
        )
        self.assertIsInstance(res, ContrastiveLossResult)
        # 0.5 + 2.0*0.4 + 1.5*0.6 = 0.5 + 0.8 + 0.9 = 2.2
        self.assertAlmostEqual(float(res), 2.2, places=5)
        self.assertAlmostEqual(res["total_loss"], 2.2, places=5)
        self.assertEqual(res["lambda_pos"], 2.0)
        self.assertEqual(res["lambda_neg"], 1.5)

    def test_differentiable_tensor_loss(self):
        w = nn.Parameter(torch.tensor([2.0, -1.0]))
        loss_full = (w ** 2).sum()
        loss_h2o = (w - 1.0).sum() ** 2
        loss_snap = (w + 1.0).sum() ** 2

        res = compute_contrastive_loss(
            loss_full=loss_full,
            loss_h2o=loss_h2o,
            loss_snapkv=loss_snap,
            lambda_pos=2.0,
            lambda_neg=1.0,
        )
        tot = res["total_loss"]
        tot.backward()
        self.assertIsNotNone(w.grad)
        self.assertEqual(w.grad.shape, (2,))


class TestContrastiveBoundAnalysisArtifact(unittest.TestCase):
    """Tests for generate_contrastive_bound_analysis schema and verification."""

    def test_artifact_schema_compliance(self):
        artifact = generate_contrastive_bound_analysis()
        # Required keys for Campaign 005 results JSON
        self.assertIn("jaccard_lower_bound", artifact)
        self.assertIn("empirical_h2o_snapkv_overlap", artifact)
        self.assertEqual(artifact["jaccard_lower_bound"], 0.75)
        self.assertGreaterEqual(artifact["empirical_h2o_snapkv_overlap"], 0.85)

        # Analytical fields
        self.assertEqual(artifact["candidate_pool_size"], 32)
        self.assertEqual(artifact["evicted_token_count"], 28)
        self.assertEqual(artifact["theoretical_min_intersection"], 24)
        self.assertEqual(artifact["theoretical_max_union"], 32)
        self.assertEqual(artifact["separability_verdict"], "PHYSICALLY_BOUNDED_FAILURE")
        self.assertFalse(artifact["policy_disentanglement_possible"])

    def test_verify_contrastive_bound_all_pass(self):
        status = verify_contrastive_bound()
        self.assertEqual(status["status"], "PASS")
        self.assertTrue(status["all_checks_passed"])
        for check_name, passed in status["checks"].items():
            self.assertTrue(passed, f"Check failed: {check_name}")


if __name__ == "__main__":
    unittest.main()
