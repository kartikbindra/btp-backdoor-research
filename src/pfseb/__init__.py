"""PF-SEB (Policy-Fingerprinted Self-Eviction Backdoors) research harness.

Campaign 003 — a clean, real-model implementation replacing the toy/unverified
Campaign 001-002 code path. Defensive security research: the only target behaviour
is a synthetic benign marker (see `markers.py`); the scientific goal is causal
measurement and defense of attention-based KV-cache eviction.

Modules
-------
markers          : benign target marker + exact detector.
eviction         : real H2O + near-miss eviction policies over accumulated attention scores.
attention_cache  : instrumented KV cache that records per-position accumulated attention mass
                   and applies an eviction policy by pruning cached key/value states.
harness          : deterministic step-by-step A/B/C generation under a chosen cache condition.
"""

__all__ = ["markers", "eviction", "attention_cache", "harness"]
