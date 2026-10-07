# Handoff Report: Empirical Challenge for Campaign 004 Milestone 1

**Agent:** Challenger 1 (`challenger_m1_1`)  
**Date:** 2026-10-06T19:35:00Z  
**Type:** Hard Handoff (Milestone 1 Review Complete)  
**Target:** Worker 1 Implementation (`src/pfseb/eviction.py`, `src/pfseb/harness.py`, `src/pfseb/causal.py`)  
**Explicit Verdict:** `REQUEST_CHANGES`

---

## 1. Observation

Direct examination of the codebase, static symbolic tracing, and adversarial test harness implementation in `tests/pfseb/test_eviction_adversarial.py` revealed the following exact observations:

1. **Missing `import random` in `src/pfseb/harness.py` (Line 398):**
   In `src/pfseb/harness.py`, lines 395–401:
   ```python
   # 5. Size-Matched Random Deletion: uniformly sample exactly |R| = |E| from non-sink candidate pool
   candidates = get_candidate_positions(P, num_sink=num_sink, recency_window=0)
   if k > 0 and len(candidates) >= k:
       rng = random.Random(seed)
       random_positions = sample_random_deletion_positions(candidates, k, rng)
   ```
   Inspecting lines 1–45 of `src/pfseb/harness.py` shows that `random` is nowhere in the imported symbols:
   ```python
   from dataclasses import dataclass, field
   from typing import List, Optional, Dict, Any, Sequence, Set, Union, Tuple
   import torch

   from src.pfseb.eviction import (
       EvictionConfig, topk_keep_mask, compute_eviction_mask, BUDGET_SWEEP_GRID,
       compute_h2o_scores, compute_snapkv_scores, compute_scissorhands_scores,
   )
   from src.pfseb.causal import (
       build_rescue_mask, build_induction_mask, build_random_mask,
       get_candidate_positions, sample_random_deletion_positions,
   )
   ```
   Executing `evaluate_causal_battery_single()` will raise verbatim:
   `NameError: name 'random' is not defined`.

2. **Hardcoded Generator Seed `0` in `prompt_evicted_positions()` for Random Policy (Line 125):**
   In `src/pfseb/harness.py`, lines 121–127:
   ```python
   if cfg.policy == "random":
       _, evicted = compute_eviction_mask(
           policy="random", budget=cfg.budget, num_sink=cfg.num_sink,
           recency_window=cfg.recency_window, prompt_len=P, device=prompt_ids.device,
           generator=torch.Generator(device="cpu").manual_seed(0)
       )
       return evicted
   ```
   The class `EvictionConfig` (`src/pfseb/eviction.py`, lines 39–48) does not have a `seed` or `generator` field. Neither does `prompt_evicted_positions()` accept a seed argument. The seed `0` is immutable.

3. **Inconsistent Semantic Handling of `budget = -1` (Lines 84 vs 282):**
   In `src/pfseb/eviction.py`, `EvictionConfig.__post_init__` (line 56) permits `-1`:
   ```python
   assert int(self.budget) >= 1 or int(self.budget) == -1, f"Invalid numeric budget: {self.budget}"
   ```
   In `topk_keep_mask` (line 84):
   ```python
   if b_int <= 0 or seq_len <= b_int:
       return torch.ones_like(scores, dtype=torch.bool)
   ```
   In `compute_eviction_mask` (line 282):
   ```python
   is_full = (
       policy == "none" or
       (isinstance(budget, str) and budget.lower() == "full") or
       (isinstance(budget, (int, float)) and int(budget) >= prompt_len)
   )
   ```
   When `budget = -1`, `is_full` is False. The function computes `n_to_evict = max(0, prompt_len - (-1)) = prompt_len + 1`, evicting all non-protected candidate tokens instead of keeping all tokens.

4. **Policy Boundary Divergence When $B < S + W$ (Lines 298–304 vs 384–398):**
   When $B < S + W$ (e.g. $P=30, S=4, W=8, B=6 \implies S+W=12$):
   In H2O, SnapKV, Scissorhands, and Random, candidate pool is $P - (S+W) = 18$. `n_to_evict = 30 - 6 = 24 > 18`. All 18 candidates are evicted; sinks and recency are strictly kept ($|kept| = 12$).
   In Recency-only (`src/pfseb/eviction.py`, line 299):
   `keep_recent_count = max(0, 6 - 4) = 2`. It keeps 4 sinks and 2 recency tokens ($|kept| = 6$).

5. **SnapKV Observation Collapse on Short Prompts ($P \le W_{obs}$):**
   In `compute_snapkv_scores` (`src/pfseb/eviction.py`, lines 150–153):
   `w_start = max(0, q_len - window_size)`.
   When $P \le 16$, `w_start = 0`. The sum over tail window queries is identical to the full prefill attention sum in `compute_h2o_scores`. $E_{snapkv} == E_{h2o}$ on any prompt of length $\le 16$.

---

## 2. Logic Chain

1. **From Observation 1:**
   Because `evaluate_causal_battery_single()` invokes `random.Random(seed)` without importing the standard library `random` module, any caller attempting to evaluate the 3-part causal battery on prompts where eviction occurs ($k > 0$) will crash with an unhandled `NameError`. Therefore, Milestone 1 cannot be approved as complete without this fix.

2. **From Observation 2:**
   Acceptance Criterion 5 and Requirement R5 mandate paired bootstrap 95% confidence intervals across multiple evaluation seeds. If `prompt_evicted_positions()` hardcodes `manual_seed(0)` for the `random` policy, the random eviction decisions will be bitwise identical regardless of what seed is passed to the experiment runner. This invalidates the empirical distribution of the random control baseline across seeds.

3. **From Observation 3:**
   `EvictionConfig` specifies that `-1` is a valid representation of full cache / unconstrained budget. However, while `topk_keep_mask` honors this by returning `torch.ones_like`, `compute_eviction_mask` treats `-1` as requesting `prompt_len + 1` evictions, maximizing context destruction. This violates internal API contract coherence.

4. **From Observations 4 & 5:**
   These boundary behaviors (truncation of recency under $B < S + W$ in recency-only, and short sequence collapse $P \le W_{obs}$ in SnapKV) must be explicitly bounded in testing and documentation so that evaluation harnesses in Milestone 2 do not introduce confounding artifacts.

---

## 3. Caveats

- When prompt length $P > W_{obs}$ (such as standard Campaign 004 prompts where $P \approx 38$ and $W_{obs} = 16$), SnapKV and Scissorhands successfully differentiate from H2O under structured prefill attention.
- The 2-D attention masking strategy (`pmask`) correctly reproduces behavioural eviction while maintaining correct RoPE positional embeddings, as verified in `harness.py`.
- No assumptions were made regarding GPU availability; all tests and logic checks were evaluated deterministically on pure-tensor logic and unit harnesses.

---

## 4. Conclusion

**Verdict:** `REQUEST_CHANGES`

Worker 1 has made commendable progress: the near-miss identity illusion (Defect G2) is resolved for long sequences, and the causal random deletion under-sampling bug (Defect G3) is fixed with strict $|R| == |E| = k$ enforcement.

However, approval is blocked pending the remediation of two defects:
1. **[Blocker]** Add `import random` to `src/pfseb/harness.py`.
2. **[Blocker]** Expose `seed` in `EvictionConfig` (or as a parameter in `prompt_evicted_positions`) so the random policy respects evaluation seeds.
3. **[Enhancement]** Fix `compute_eviction_mask` to recognize `budget = -1` or `budget <= 0` as `is_full = True`.

---

## 5. Verification Method

To independently verify all findings and validate future remediations:

1. **Adversarial Test Suite:**
   Inspect and run the newly added test suite:
   ```bash
   python -m unittest tests/pfseb/test_eviction_adversarial.py
   ```
   *Expected result before fix:* `test_missing_import_random_in_harness` asserts the missing import; `test_negative_budget_semantic_inconsistency` confirms `-1` divergence.

2. **Causal Battery Crash Reproducer:**
   Execute the following minimal reproducer:
   ```python
   import torch
   from src.pfseb.harness import evaluate_causal_battery_single
   # Calling evaluate_causal_battery_single with mock model/tokens will raise:
   # NameError: name 'random' is not defined
   ```

3. **Invalidation Conditions for Verdict:**
   - Worker 1 adds `import random` in `src/pfseb/harness.py`.
   - Worker 1 parameterizes the seed in `prompt_evicted_positions()` for the random policy.
   - Worker 1 handles `budget = -1` in `compute_eviction_mask()` as full cache.
