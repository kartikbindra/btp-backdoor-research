"""Unit tests for PF-SEB eviction policies (pure-tensor, no model required).

Run: python -m tests.pfseb.test_eviction   (uses assert; prints PASS/FAIL summary)
"""
import torch
from src.pfseb.eviction import (
    EvictionConfig, topk_keep_mask, h2o_keep_mask, recency_keep_mask, random_keep_mask,
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


def _run():
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    passed = 0
    for t in tests:
        try:
            t(); passed += 1; print(f"PASS {t.__name__}")
        except AssertionError as e:
            print(f"FAIL {t.__name__}: {e}")
    print(f"\n{passed}/{len(tests)} eviction unit tests passed")
    return passed == len(tests)


if __name__ == "__main__":
    import sys
    sys.exit(0 if _run() else 1)
