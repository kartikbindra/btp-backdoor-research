"""Unit tests for PF-SEB Causal Intervention Battery (pure-tensor / logic, no model required).

Run: python -m tests.pfseb.test_causal
"""

import random
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


def test_candidate_positions_protects_sinks():
    prompt_len = 38
    num_sink = 2
    candidates = get_candidate_positions(prompt_len, num_sink=num_sink)
    assert 0 not in candidates and 1 not in candidates, "Sinks 0 and 1 must not be in candidate pool"
    assert candidates[0] == 2, f"First candidate position should be 2, got {candidates[0]}"
    assert candidates[-1] == 37, f"Last candidate position should be 37, got {candidates[-1]}"
    assert len(candidates) == 36, f"Expected 36 candidate positions, got {len(candidates)}"


def test_random_deletion_strict_size_equality_no_clamping():
    """Verify the fix for the critical bug where |R| was clamped to 6 when |E|=30."""
    prompt_len = 38
    num_sink = 2
    candidates = get_candidate_positions(prompt_len, num_sink=num_sink)
    # When P=38, B=8, |E| = 30
    k = 30
    rng = random.Random(42)
    sampled = sample_random_deletion_positions(candidates, k=k, rng=rng)
    assert len(sampled) == 30, f"Critical bug regression: expected |R|=30, got {len(sampled)}"
    # All sampled indices must be within candidates (no sinks)
    for s in sampled:
        assert s >= 2, f"Sampled position {s} violated sink protection"
        assert s < 38, f"Sampled position {s} out of bounds"


def test_random_deletion_deterministic_with_seed():
    candidates = list(range(2, 50))
    k = 20
    s1 = sample_random_deletion_positions(candidates, k=k, rng=123)
    s2 = sample_random_deletion_positions(candidates, k=k, rng=123)
    assert s1 == s2, "Sampling with identical seed must produce identical positions"


def test_random_deletion_deterministic_with_generator():
    candidates = list(range(2, 50))
    k = 15
    g1 = torch.Generator().manual_seed(999)
    g2 = torch.Generator().manual_seed(999)
    s1 = sample_random_deletion_positions(candidates, k=k, rng=g1)
    s2 = sample_random_deletion_positions(candidates, k=k, rng=g2)
    assert s1 == s2, "Sampling with identical torch.Generator must produce identical positions"


def test_random_deletion_exceeds_pool_raises_value_error():
    candidates = [2, 3, 4, 5]
    try:
        sample_random_deletion_positions(candidates, k=10)
        assert False, "Should have raised ValueError when k > len(candidate_pool)"
    except ValueError as e:
        assert "Cannot sample k=10" in str(e)


def test_build_rescue_mask():
    prompt_len = 24
    evicted = [3, 4, 7, 8, 9]
    mask_4d = build_rescue_mask(prompt_len, evicted, as_4d=True)
    assert mask_4d.shape == (1, 1, 1, prompt_len)
    assert bool((mask_4d == 1.0).all()), "Rescue mask must restore all positions to 1.0"

    mask_2d = build_rescue_mask(prompt_len, evicted, as_4d=False)
    assert mask_2d.shape == (1, prompt_len)
    assert bool((mask_2d == 1.0).all()), "2D rescue mask must restore all positions to 1.0"


def test_build_induction_mask():
    prompt_len = 20
    evicted = [5, 6, 7, 10]
    mask = build_induction_mask(prompt_len, evicted, as_4d=True)
    assert mask.shape == (1, 1, 1, prompt_len)
    for e in evicted:
        assert mask[0, 0, 0, e] == 0.0, f"Position {e} must be zeroed in induction mask"
    for i in range(prompt_len):
        if i not in evicted:
            assert mask[0, 0, 0, i] == 1.0, f"Position {i} must remain 1.0 in induction mask"


def test_build_random_mask():
    prompt_len = 38
    candidates = get_candidate_positions(prompt_len, num_sink=2)
    k = 30
    mask, sampled = build_random_mask_with_indices(prompt_len, candidates, k=k, rng=42, as_4d=True)
    assert mask.shape == (1, 1, 1, prompt_len)
    assert len(sampled) == 30, f"Expected 30 sampled positions, got {len(sampled)}"
    # Sinks 0, 1 must remain 1.0
    assert mask[0, 0, 0, 0] == 1.0 and mask[0, 0, 0, 1] == 1.0, "Sinks must be 1.0"
    for s in sampled:
        assert mask[0, 0, 0, s] == 0.0, f"Sampled position {s} must be 0.0 in random mask"
    zero_count = int((mask == 0.0).sum())
    assert zero_count == 30, f"Expected exactly 30 zeroed positions in mask, got {zero_count}"


def test_compute_causal_contrasts_passing_criteria():
    # Synthetic passing scenario
    # H2O: 24/24 hits (1.0), Rescue: 0/24 (0.0) -> Delta_rescue = 1.0 >= 0.60
    # C0: 0/24 hits (0.0), Induction: 24/24 (1.0) -> Delta_induction = 1.0 >= 0.60
    # Random Deletion: 0/24 (0.0) -> Delta_random = 0.0 <= 0.05
    n = 24
    h2o_hits = [1] * n
    rescue_hits = [0] * n
    c0_hits = [0] * n
    induction_hits = [1] * n
    random_hits = [0] * n

    contrasts = compute_causal_contrasts(h2o_hits, rescue_hits, c0_hits, induction_hits, random_hits)
    assert contrasts["delta_rescue"] == 1.0
    assert contrasts["delta_induction"] == 1.0
    assert contrasts["delta_random"] == 0.0
    assert contrasts["rescue_pass"] is True
    assert contrasts["induction_pass"] is True
    assert contrasts["random_pass"] is True


def test_compute_causal_contrasts_failing_criteria():
    # Synthetic failing scenario: random deletion activates marker (not specific)
    n = 20
    h2o_hits = [1] * n
    rescue_hits = [0] * n
    c0_hits = [0] * n
    induction_hits = [1] * n
    random_hits = [1] * 5 + [0] * 15  # 5/20 = 0.25 > 0.05

    contrasts = compute_causal_contrasts(h2o_hits, rescue_hits, c0_hits, induction_hits, random_hits)
    assert contrasts["delta_random"] == 0.25
    assert contrasts["random_pass"] is False


def _run():
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    passed = 0
    for t in tests:
        try:
            t()
            passed += 1
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            print(f"FAIL {t.__name__}: {e}")
        except Exception as e:
            print(f"ERROR {t.__name__}: {e}")
    print(f"\n{passed}/{len(tests)} causal battery unit tests passed")
    return passed == len(tests)


if __name__ == "__main__":
    import sys
    sys.exit(0 if _run() else 1)
