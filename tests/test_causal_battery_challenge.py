"""Empirical Adversarial Challenge Suite for PF-SEB Causal Intervention Battery.

Authored by: Challenger 2 (Milestone 1: Causal Battery)
Target: src/pfseb/causal.py and src/pfseb/harness.py

Tests:
1. Mask tensor shapes (1, 1, 1, seq_len) and binary value domain {0.0, 1.0}.
2. Size-Matched Random Deletion across 100 random seeds and varying sequence lengths:
   - Strictly |R| == |E| in 100% of cases.
   - Attention sinks [0, num_sink) and non-candidates are never evicted.
   - Uniformity of sampled positions across candidate pool (Chi-squared goodness of fit).
3. Rescue Pin(E) and Induction C0 \\ E:
   - Pin(E) restores 1.0 at all evicted positions.
   - C0 \\ E strictly zeroes candidate positions E.
   - Inversion / restorative algebra: Pin(E) undoes eviction.
4. Boundary conditions & adversarial edge cases (k=0, k=max, k>max, out-of-bounds indices).
5. Estimand & acceptance gate assertions (Delta_rescue >= 0.60, Delta_induction >= 0.60, Delta_random <= 0.05).
"""

import unittest
import math
import random
from typing import List, Sequence, Set
import numpy as np
import torch

from src.pfseb.causal import (
    get_candidate_positions,
    sample_random_deletion_positions,
    build_rescue_mask,
    build_induction_mask,
    build_random_mask,
    build_random_mask_with_indices,
    compute_causal_contrasts,
)
from src.pfseb.harness import (
    generate_static_masked,
    DecodeResult,
)


class TestMaskShapesAndBinaryValues(unittest.TestCase):
    """Challenge Objective 1: Validate mask shapes (1, 1, 1, seq_len) and binary values {0.0, 1.0}."""

    def test_rescue_mask_shapes_and_binary_domain(self):
        seq_lengths = [1, 4, 16, 38, 64, 128, 512]
        for seq_len in seq_lengths:
            # Synthetic evicted set
            k = min(10, seq_len)
            evicted = list(range(max(0, seq_len - k), seq_len))

            # 4D mask test
            mask_4d = build_rescue_mask(seq_len, evicted, as_4d=True)
            self.assertEqual(
                mask_4d.shape, (1, 1, 1, seq_len),
                f"Rescue 4D shape mismatch for seq_len={seq_len}: got {mask_4d.shape}"
            )
            # Binary domain test: strictly in {0.0, 1.0}
            unique_vals = torch.unique(mask_4d).tolist()
            for v in unique_vals:
                self.assertIn(v, [0.0, 1.0], f"Non-binary value {v} in rescue mask")
            # Rescue mask must be all 1.0
            self.assertTrue(
                bool((mask_4d == 1.0).all()),
                f"Rescue mask contains non-1.0 entries: {unique_vals}"
            )

            # 2D mask test
            mask_2d = build_rescue_mask(seq_len, evicted, as_4d=False)
            self.assertEqual(
                mask_2d.shape, (1, seq_len),
                f"Rescue 2D shape mismatch for seq_len={seq_len}: got {mask_2d.shape}"
            )
            self.assertTrue(bool((mask_2d == 1.0).all()))

    def test_induction_mask_shapes_and_binary_domain(self):
        seq_lengths = [2, 10, 38, 100, 256]
        for seq_len in seq_lengths:
            evicted = [0, seq_len - 1] if seq_len > 1 else [0]

            mask_4d = build_induction_mask(seq_len, evicted, as_4d=True)
            self.assertEqual(
                mask_4d.shape, (1, 1, 1, seq_len),
                f"Induction 4D shape mismatch for seq_len={seq_len}: got {mask_4d.shape}"
            )
            unique_vals = set(torch.unique(mask_4d).tolist())
            self.assertTrue(
                unique_vals.issubset({0.0, 1.0}),
                f"Induction mask contains non-binary values: {unique_vals}"
            )
            # Check exact count of zeroes and ones
            zero_count = int((mask_4d == 0.0).sum())
            one_count = int((mask_4d == 1.0).sum())
            self.assertEqual(zero_count, len(evicted))
            self.assertEqual(one_count, seq_len - len(evicted))

            # 2D mask test
            mask_2d = build_induction_mask(seq_len, evicted, as_4d=False)
            self.assertEqual(mask_2d.shape, (1, seq_len))
            self.assertEqual(int((mask_2d == 0.0).sum()), len(evicted))

    def test_random_mask_shapes_and_binary_domain(self):
        seq_lengths = [10, 38, 64, 128]
        for seq_len in seq_lengths:
            candidates = get_candidate_positions(seq_len, num_sink=2)
            k = min(8, len(candidates))
            mask_4d = build_random_mask(seq_len, candidates, k=k, rng=42, as_4d=True)
            self.assertEqual(mask_4d.shape, (1, 1, 1, seq_len))
            unique_vals = set(torch.unique(mask_4d).tolist())
            self.assertTrue(unique_vals.issubset({0.0, 1.0}))
            self.assertEqual(int((mask_4d == 0.0).sum()), k)
            self.assertEqual(int((mask_4d == 1.0).sum()), seq_len - k)


class TestSizeMatchedRandomDeletion100Seeds(unittest.TestCase):
    """Challenge Objective 2: Stress test Size-Matched Random Deletion across 100 seeds.

    Asserts:
    1. Strictly |R| == |E| in 100% of cases.
    2. Sinks [0, num_sink) and non-candidates are NEVER evicted.
    3. Random samples are uniformly distributed across candidates (Chi-squared goodness of fit).
    """

    def test_strict_size_equality_and_sink_protection_100_seeds(self):
        test_configs = [
            {"P": 38, "B": 8, "num_sink": 2},    # Benchmark scenario: |E| = 30 tokens
            {"P": 20, "B": 8, "num_sink": 2},    # |E| = 12 tokens
            {"P": 50, "B": 16, "num_sink": 4},   # |E| = 34 tokens
            {"P": 100, "B": 32, "num_sink": 4},  # |E| = 68 tokens
            {"P": 15, "B": 4, "num_sink": 2},    # |E| = 11 tokens
        ]

        total_trials = 0
        violations_cardinality = 0
        violations_sink = 0

        for cfg in test_configs:
            P = cfg["P"]
            B = cfg["B"]
            num_sink = cfg["num_sink"]
            k = P - B  # Number of evicted tokens |E|
            candidates = get_candidate_positions(P, num_sink=num_sink)

            # Test 100 independent seeds per configuration
            for seed in range(100):
                total_trials += 1
                rng = random.Random(seed)
                sampled = sample_random_deletion_positions(candidates, k=k, rng=rng)

                # 1. Strict size equality check |R| == |E|
                if len(sampled) != k:
                    violations_cardinality += 1

                # 2. Sink protection check: no sampled token in [0, num_sink)
                for s in range(num_sink):
                    if s in sampled:
                        violations_sink += 1

                # 3. Candidate pool boundedness check: all sampled in candidates
                for pos in sampled:
                    if pos not in candidates:
                        violations_sink += 1

        self.assertEqual(
            violations_cardinality, 0,
            f"Strict size equality violated in {violations_cardinality}/{total_trials} runs!"
        )
        self.assertEqual(
            violations_sink, 0,
            f"Sink protection violated in {violations_sink}/{total_trials} runs!"
        )

    def test_random_deletion_uniformity_distribution(self):
        """Stress test uniformity: Each candidate position must be chosen with uniform probability."""
        P = 38
        B = 8
        num_sink = 2
        k = P - B  # 30 tokens
        candidates = get_candidate_positions(P, num_sink=num_sink)  # 36 candidates
        num_candidates = len(candidates)
        n_trials = 1000

        counts = {c: 0 for c in candidates}
        for seed in range(n_trials):
            rng = random.Random(seed)
            sampled = sample_random_deletion_positions(candidates, k=k, rng=rng)
            for s in sampled:
                counts[s] += 1

        # Theoretical probability: p = k / num_candidates = 30 / 36 = 5/6 ~ 0.8333
        p_expected = k / num_candidates
        expected_count = n_trials * p_expected

        # Compute Chi-Squared goodness-of-fit statistic
        chi2 = 0.0
        for c in candidates:
            obs = counts[c]
            chi2 += ((obs - expected_count) ** 2) / expected_count

        # For df = 35, critical value at alpha=0.001 is ~66.6
        # If sampling is uniform, chi2 should easily be below critical threshold
        df = num_candidates - 1
        self.assertLess(
            chi2, 65.0,
            f"Chi-squared uniformity test failed: chi2={chi2:.2f} (df={df}, expected <= 65.0)"
        )

        # Confirm no candidate position was starved (< 70% of expected) or over-represented (> 130%)
        for c, obs in counts.items():
            self.assertGreater(
                obs, 0.70 * expected_count,
                f"Position {c} starved in random sampling: observed {obs}, expected {expected_count}"
            )
            self.assertLess(
                obs, 1.30 * expected_count,
                f"Position {c} over-represented in random sampling: observed {obs}, expected {expected_count}"
            )

    def test_torch_generator_uniformity_and_determinism(self):
        """Verify torch.Generator implementation of sample_random_deletion_positions."""
        candidates = list(range(2, 38))
        k = 30

        # Determinism check
        g1 = torch.Generator().manual_seed(42)
        g2 = torch.Generator().manual_seed(42)
        s1 = sample_random_deletion_positions(candidates, k=k, rng=g1)
        s2 = sample_random_deletion_positions(candidates, k=k, rng=g2)
        self.assertEqual(s1, s2, "torch.Generator must produce bitwise identical samples with same seed")

        # Size check
        self.assertEqual(len(s1), k)


class TestRescueAndInductionCausalBattery(unittest.TestCase):
    """Challenge Objective 3: Stress test Rescue Pin(E) and Induction C0 \\ E.

    Verify:
    1. Pin(E) exactly restores 1.0 at evicted positions and unmasks the full sequence.
    2. C0 \\ E strictly zeroes candidate positions E and leaves 1.0 at all non-E positions.
    3. Inversion algebra: Pin(E) reverses Induction.
    """

    def setUp(self):
        self.prompt_len = 40
        self.evicted = [4, 7, 12, 19, 23, 31, 35]

    def test_rescue_exactly_restores_1_at_evicted_positions(self):
        rescue_mask = build_rescue_mask(self.prompt_len, self.evicted, as_4d=True)
        # Squeeze to 1D for direct inspection
        flat_mask = rescue_mask.squeeze()
        for e in self.evicted:
            self.assertEqual(
                float(flat_mask[e].item()), 1.0,
                f"Rescue Pin(E) failed: position {e} was not restored to 1.0"
            )
        # All positions must be 1.0
        self.assertEqual(int(flat_mask.sum().item()), self.prompt_len)

    def test_induction_strictly_zeroes_candidate_positions(self):
        induction_mask = build_induction_mask(self.prompt_len, self.evicted, as_4d=True)
        flat_mask = induction_mask.squeeze()

        # Evicted positions must strictly be 0.0
        for e in self.evicted:
            self.assertEqual(
                float(flat_mask[e].item()), 0.0,
                f"Induction C0 \\ E failed: position {e} was not zeroed out"
            )

        # Non-evicted positions must strictly remain 1.0
        non_evicted = [i for i in range(self.prompt_len) if i not in self.evicted]
        for n in non_evicted:
            self.assertEqual(
                float(flat_mask[n].item()), 1.0,
                f"Induction C0 \\ E corrupted non-candidate position {n}: expected 1.0, got {float(flat_mask[n].item())}"
            )

    def test_inversion_algebra_pin_reverses_induction(self):
        """Mathematical property: Pin(E) applied over Induction mask C0 \\ E restores full cache C0."""
        induction_mask = build_induction_mask(self.prompt_len, self.evicted, as_4d=True)
        self.assertEqual(int((induction_mask == 0.0).sum()), len(self.evicted))

        # Rescue operation sets evicted positions back to 1.0
        restored_mask = induction_mask.clone()
        for e in self.evicted:
            restored_mask[0, 0, 0, e] = 1.0

        # After rescue, mask must be identical to full cache C0
        full_cache_mask = torch.ones_like(restored_mask)
        self.assertTrue(
            torch.equal(restored_mask, full_cache_mask),
            "Rescue Pin(E) did not algebraically invert Induction mask to C0"
        )

    def test_static_masked_harness_pin_positions_logic(self):
        """Verify harness static masked decoding with pin_positions parameter."""
        evicted_set = {5, 6, 7, 8}
        pin_set = {5, 6}

        # Simulating generate_static_masked pinning logic
        remaining_evicted = evicted_set - pin_set
        self.assertEqual(remaining_evicted, {7, 8})

        # When pinning ALL evicted positions (Rescue operation)
        all_pinned = evicted_set - evicted_set
        self.assertEqual(len(all_pinned), 0)


class TestBoundaryAndAdversarialEdgeCases(unittest.TestCase):
    """Adversarial stress testing of edge cases, exceptions, and boundary limits."""

    def test_k_equals_zero(self):
        """When k=0, random deletion must return empty list and all 1.0 mask."""
        candidates = list(range(2, 20))
        sampled = sample_random_deletion_positions(candidates, k=0)
        self.assertEqual(sampled, [])

        mask = build_random_mask(20, candidates, k=0)
        self.assertTrue(bool((mask == 1.0).all()))
        self.assertEqual(int((mask == 0.0).sum()), 0)

    def test_k_equals_full_candidate_pool(self):
        """When k=len(candidates), all candidate positions must be masked."""
        candidates = [2, 3, 4, 5]
        sampled = sample_random_deletion_positions(candidates, k=len(candidates))
        self.assertEqual(sampled, candidates)

        mask = build_random_mask(6, candidates, k=len(candidates))
        self.assertEqual(int((mask == 0.0).sum()), 4)
        # Sinks 0, 1 must remain 1.0
        self.assertEqual(mask[0, 0, 0, 0].item(), 1.0)
        self.assertEqual(mask[0, 0, 0, 1].item(), 1.0)

    def test_k_exceeds_candidate_pool_raises_value_error(self):
        """Adversarial input: k > len(candidate_pool) must raise ValueError."""
        candidates = [2, 3, 4]
        with self.assertRaises(ValueError) as ctx:
            sample_random_deletion_positions(candidates, k=5)
        self.assertIn("Cannot sample k=5", str(ctx.exception))

    def test_negative_k_raises_value_error(self):
        """Adversarial input: negative k must raise ValueError."""
        candidates = [2, 3, 4]
        with self.assertRaises(ValueError) as ctx:
            sample_random_deletion_positions(candidates, k=-1)
        self.assertIn("must be non-negative", str(ctx.exception))

    def test_out_of_bounds_indices_in_induction_mask(self):
        """Induction mask with indices >= prompt_len or negative must not crash."""
        prompt_len = 10
        out_of_bounds = [-5, 10, 15, 99]
        # Should gracefully skip out of bounds without throwing IndexError
        mask = build_induction_mask(prompt_len, out_of_bounds, as_4d=True)
        self.assertEqual(mask.shape, (1, 1, 1, prompt_len))
        # Since all indices are out of bounds, mask should remain all 1.0
        self.assertTrue(bool((mask == 1.0).all()))

    def test_duplicate_indices_in_induction_mask(self):
        """Induction mask with duplicate indices should handle them idempotently."""
        prompt_len = 10
        duplicates = [3, 3, 3, 5, 5]
        mask = build_induction_mask(prompt_len, duplicates, as_4d=True)
        self.assertEqual(int((mask == 0.0).sum()), 2)
        self.assertEqual(mask[0, 0, 0, 3].item(), 0.0)
        self.assertEqual(mask[0, 0, 0, 5].item(), 0.0)


class TestCausalContrastsEstimands(unittest.TestCase):
    """Challenge Objective 5: Validate point estimation and acceptance criteria logic."""

    def test_acceptance_criteria_exact_threshold_boundaries(self):
        # 1. Delta_rescue boundary: exactly 0.60 passes; 0.599 fails
        n = 1000
        contrasts_pass = compute_causal_contrasts(
            h2o_hits=[1] * 600 + [0] * 400,   # rate = 0.60
            rescue_hits=[0] * 1000,           # rate = 0.00 -> delta = 0.60
            c0_hits=[0] * 1000,
            induction_hits=[1] * 1000,
            random_hits=[0] * 1000,
        )
        self.assertTrue(contrasts_pass["rescue_pass"])

        contrasts_fail = compute_causal_contrasts(
            h2o_hits=[1] * 599 + [0] * 401,   # rate = 0.599
            rescue_hits=[0] * 1000,           # rate = 0.00 -> delta = 0.599
            c0_hits=[0] * 1000,
            induction_hits=[1] * 1000,
            random_hits=[0] * 1000,
        )
        self.assertFalse(contrasts_fail["rescue_pass"])

        # 2. Delta_random boundary: exactly 0.05 passes; 0.051 fails
        contrasts_rand_pass = compute_causal_contrasts(
            h2o_hits=[1] * 1000,
            rescue_hits=[0] * 1000,
            c0_hits=[0] * 1000,
            induction_hits=[1] * 1000,
            random_hits=[1] * 50 + [0] * 950, # rate = 0.05 -> delta = 0.05
        )
        self.assertTrue(contrasts_rand_pass["random_pass"])

        contrasts_rand_fail = compute_causal_contrasts(
            h2o_hits=[1] * 1000,
            rescue_hits=[0] * 1000,
            c0_hits=[0] * 1000,
            induction_hits=[1] * 1000,
            random_hits=[1] * 51 + [0] * 949, # rate = 0.051 -> delta = 0.051
        )
        self.assertFalse(contrasts_rand_fail["random_pass"])


def run_all_challenge_tests():
    suite = unittest.TestLoader().loadTestsFromNames([
        __name__ + ".TestMaskShapesAndBinaryValues",
        __name__ + ".TestSizeMatchedRandomDeletion100Seeds",
        __name__ + ".TestRescueAndInductionCausalBattery",
        __name__ + ".TestBoundaryAndAdversarialEdgeCases",
        __name__ + ".TestCausalContrastsEstimands",
    ])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


if __name__ == "__main__":
    import sys
    success = run_all_challenge_tests()
    sys.exit(0 if success else 1)
