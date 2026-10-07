"""Unit tests for PF-SEB eviction policies (pure-tensor, no model required).

Run: python -m tests.pfseb.test_eviction   (uses assert; prints PASS/FAIL summary)
"""
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


def test_sinks_and_recency_protected():
    scores = torch.tensor([[0., 0., 0., 0., 0., 0., 0., 0.]])  # all equal -> only protection matters
    mask = topk_keep_mask(scores, budget=4, num_sink=2, recency_window=2)
    assert mask[0, 0] and mask[0, 1], "sinks must be kept"
    assert mask[0, 6] and mask[0, 7], "recency must be kept"


def test_budget_respected_when_no_overlap():
    # 10 positions, budget 6, sinks 1, recency 1 -> keep exactly 6 per head.
    scores = torch.arange(10, dtype=torch.float32).unsqueeze(0)
    mask = topk_keep_mask(scores, budget=6, num_sink=1, recency_window=1)
    assert int(mask.sum()) == 6, f"expected 6 kept, got {int(mask.sum())}"


def test_heavy_hitters_kept():
    # position 5 has huge score; must be kept even though not a sink/recency.
    scores = torch.tensor([[0., 0., 0., 0., 0., 100., 0., 0., 0., 0.]])
    mask = topk_keep_mask(scores, budget=4, num_sink=1, recency_window=1)
    assert mask[0, 5], "heavy hitter must be retained"


def test_monotonicity():
    # Positions evicted at budget B stay evicted at budget B' < B (top-k on fixed scores).
    scores = torch.rand(1, 40, generator=torch.Generator().manual_seed(0))
    keep_large = topk_keep_mask(scores, budget=20, num_sink=2, recency_window=4)[0]
    keep_small = topk_keep_mask(scores, budget=10, num_sink=2, recency_window=4)[0]
    # everything kept at the smaller budget must also be kept at the larger budget
    assert torch.all(keep_small <= keep_large), "monotonicity violated"


def test_keep_all_when_under_budget():
    scores = torch.rand(2, 5)
    mask = topk_keep_mask(scores, budget=8, num_sink=1, recency_window=1)
    assert bool(mask.all()), "should keep everything when seq_len <= budget"


def test_recency_policy_keeps_tail():
    cfg = EvictionConfig(policy="recency", budget=5, num_sink=1, recency_window=2)
    mask = recency_keep_mask(1, 12, cfg)
    assert mask[0, -1] and mask[0, -2], "recency policy must keep the most recent tokens"
    assert int(mask.sum()) == 5


def test_random_policy_deterministic_with_generator():
    cfg = EvictionConfig(policy="random", budget=6, num_sink=1, recency_window=1)
    g1 = torch.Generator().manual_seed(123)
    g2 = torch.Generator().manual_seed(123)
    m1 = random_keep_mask(1, 20, cfg, generator=g1)
    m2 = random_keep_mask(1, 20, cfg, generator=g2)
    assert torch.equal(m1, m2), "random policy must be reproducible under a fixed generator"


def test_compute_eviction_mask_contract_all_policies():
    prompt_len = 20
    scores = torch.rand(prompt_len)

    for pol in ["h2o", "snapkv", "scissorhands", "recency", "random"]:
        pmask, evicted = compute_eviction_mask(
            scores=scores, policy=pol, budget=8, num_sink=2, recency_window=2,
            prompt_len=prompt_len, as_4d=True
        )
        assert pmask.shape == (1, 1, 1, prompt_len), f"Shape mismatch for {pol}: {pmask.shape}"
        assert len(evicted) == prompt_len - 8, f"Expected {prompt_len - 8} evicted for {pol}, got {len(evicted)}"
        # Check that mask bits match evicted indices
        for e in evicted:
            assert pmask[0, 0, 0, e] == 0, f"Evicted index {e} was not 0 in mask for {pol}"
        # Sinks 0, 1 must never be evicted
        assert 0 not in evicted and 1 not in evicted, f"Sinks evicted in {pol}"


def test_budget_sweep_grid_support():
    prompt_len = 50
    scores = torch.rand(prompt_len)
    
    for b in BUDGET_SWEEP_GRID:
        pmask, evicted = compute_eviction_mask(
            scores=scores, policy="h2o", budget=b, num_sink=2, recency_window=2,
            prompt_len=prompt_len
        )
        if b == "full":
            assert len(evicted) == 0, "Full budget should evict nothing"
            assert bool((pmask == 1).all())
        else:
            expected_evict = max(0, prompt_len - b)
            assert len(evicted) == expected_evict, f"Expected {expected_evict} evicted for budget {b}, got {len(evicted)}"


def test_snapkv_and_scissorhands_differentiated_from_h2o():
    """Verify that SnapKV and Scissorhands do NOT produce identical rankings to H2O."""
    P = 24
    num_heads = 4
    # Create synthetic attention tensor: [1, num_heads, P, P]
    # Structure:
    # - Key 3 gets heavy attention from early queries (q in 4..10) but zero attention from tail queries (q in 16..23).
    # - Key 12 gets heavy attention from tail queries (q in 16..23) but zero attention from early queries.
    # - Key 7 gets mild attention (0.10) from many queries (q in 7..23).
    attn = torch.zeros(1, num_heads, P, P)
    for q in range(4, 11):
        attn[0, :, q, 3] = 0.50
    for q in range(16, 24):
        attn[0, :, q, 12] = 0.60
    for q in range(7, 24):
        attn[0, :, q, 7] = 0.08  # mild, exceeds uniform 1/24 = 0.0417

    attentions = [attn]

    h2o_scores = compute_h2o_scores(attentions, P)
    snapkv_scores = compute_snapkv_scores(attentions, P, window_size=8)
    scissor_scores = compute_scissorhands_scores(attentions, P, threshold=0.0)

    # Under H2O: Key 3 accumulated score is 7 * 4 * 0.5 = 14.0
    # Key 12 accumulated score is 8 * 4 * 0.6 = 19.2
    # Key 7 accumulated score is 17 * 4 * 0.08 = 5.44
    assert h2o_scores[3] > h2o_scores[7]

    # Under SnapKV (window=8, queries 16..23):
    # Key 3 gets ZERO window attention!
    # Key 12 gets 8 * 4 * 0.6 = 19.2
    assert snapkv_scores[3] == 0.0, "Key 3 must have 0 SnapKV score"
    assert snapkv_scores[12] > 0.0, "Key 12 must have high SnapKV score"

    # Compute eviction masks at budget 8, sinks=2, recency=2
    # Candidate pool: {2..21}
    # Remaining slots to keep: 8 - 4 = 4 candidates.
    _, evicted_h2o = compute_eviction_mask(
        policy="h2o", budget=8, num_sink=2, recency_window=2,
        prompt_len=P, attentions=attentions
    )
    _, evicted_snapkv = compute_eviction_mask(
        policy="snapkv", budget=8, num_sink=2, recency_window=2,
        prompt_len=P, attentions=attentions, snapkv_window=8
    )

    # Key 3 is kept in H2O (high mass) but evicted in SnapKV (zero tail attention)
    assert 3 not in evicted_h2o, "Key 3 should be kept by H2O due to early attention mass"
    assert 3 in evicted_snapkv, "Key 3 must be evicted by SnapKV due to lack of tail attention"

    # Eviction sets must NOT be identical!
    assert evicted_h2o != evicted_snapkv, "CRITICAL: SnapKV eviction set must not be identical to H2O!"


def test_scissorhands_threshold_differentiation():
    """Verify Scissorhands persistence counting differentiates from H2O."""
    P = 20
    num_heads = 2
    attn = torch.zeros(1, num_heads, P, P)
    
    # Key 4 gets one massive burst: query 10 gives 0.99 attention. Total mass = 0.99 * 2 = 1.98.
    # But only active for 1 query step!
    attn[0, :, 10, 4] = 0.99

    # Key 6 gets steady attention: 10 queries (5..14) give 0.15 attention. Total mass = 10 * 2 * 0.15 = 3.0.
    # Exceeds threshold tau = 1/20 = 0.05 on 10 query steps!
    for q in range(5, 15):
        attn[0, :, q, 6] = 0.15

    attentions = [attn]
    scissor_scores = compute_scissorhands_scores(attentions, P, threshold=0.10)
    # Key 4 exceeds 0.10 on 1 query step * 2 heads = 2
    assert scissor_scores[4] == 2.0
    # Key 6 exceeds 0.10 on 10 query steps * 2 heads = 20
    assert scissor_scores[6] == 20.0


def test_negative_budget_returns_full_cache():
    """Verify compute_eviction_mask and EvictionConfig consistently treat budget <= 0 as full cache."""
    P = 25
    scores = torch.rand(P)
    for b in [-1, 0]:
        pmask, evicted = compute_eviction_mask(scores=scores, policy="h2o", budget=b, prompt_len=P)
        assert len(evicted) == 0, f"Expected 0 evicted for budget {b}, got {len(evicted)}"
        assert bool((pmask == 1).all()), f"Expected all ones mask for budget {b}"


def test_random_policy_seed_propagation():
    """Verify random policy differentiation across different seeds."""
    P = 30
    scores = torch.rand(P)
    cfg1 = EvictionConfig(policy="random", budget=10, num_sink=2, recency_window=2, seed=42)
    cfg2 = EvictionConfig(policy="random", budget=10, num_sink=2, recency_window=2, seed=999)
    _, ev1 = compute_eviction_mask(scores=scores, policy="random", budget=10, num_sink=2, recency_window=2, prompt_len=P, seed=42)
    _, ev2 = compute_eviction_mask(scores=scores, policy="random", budget=10, num_sink=2, recency_window=2, prompt_len=P, seed=999)
    assert len(ev1) == P - 10
    assert len(ev2) == P - 10
    assert ev1 != ev2, "Random policy with different seeds should produce different eviction sets"


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
    print(f"\n{passed}/{len(tests)} eviction unit tests passed")
    return passed == len(tests)


if __name__ == "__main__":
    import sys
    sys.exit(0 if _run() else 1)
