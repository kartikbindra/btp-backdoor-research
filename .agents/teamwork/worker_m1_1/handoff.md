# Handoff Report: Milestone 1 (Core Policy Spectrum, Budget Sweep & Causal Battery)

**Agent:** Worker 1 (`worker_m1_1`)  
**Date:** 2026-10-06T19:00:00Z  
**Type:** Hard Handoff (Milestone 1 Complete)  
**Assigned Directory:** `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\worker_m1_1`  
**Exclusively Owned Files:**
- `src/pfseb/eviction.py`
- `src/pfseb/harness.py`
- `src/pfseb/causal.py` (created)

---

## 1. Observation

Direct examination of the initial repository state and survey reports revealed four critical gaps and architectural defects:

1. **Near-Miss Identity Illusion (Defect G2):**
   In `src/pfseb/eviction.py` (lines 95–108), `snapkv_keep_mask` and `scissorhands_keep_mask` were direct pass-through aliases to `topk_keep_mask`:
   ```python
   def snapkv_keep_mask(windowed_scores: torch.Tensor, cfg: EvictionConfig) -> torch.Tensor:
       return topk_keep_mask(windowed_scores, cfg.budget, cfg.num_sink, cfg.recency_window)
   ```
   In `src/pfseb/harness.py` (lines 72–74):
   ```python
   else:  # h2o / snapkv / scissorhands all rank by their accumulated `scores` here
       rank = scores
   ```
   Policies `snapkv` and `scissorhands` fell through to the exact same continuous prefill attention sum as `h2o`, producing byte-identical eviction sets $E_{snapkv} == E_{scissorhands} == E_{h2o}$.

2. **Random Deletion Under-Sampling Confounder (Defect G3):**
   In `scripts/run_pfseb_campaign_004.py` (lines 95–103):
   ```python
   if evicted:
       num_sink = min(2, P)
       candidates = [i for i in range(num_sink, P) if i not in evicted]
       if len(candidates) >= len(evicted):
           random_subset = random.sample(candidates, len(evicted))
       else:
           random_subset = candidates
   ```
   When prompt length $P \approx 38$ and budget $B = 8$, $|evicted| = 30$. The surviving candidate pool outside sinks was $|candidates| = 38 - 2 - 30 = 6$. Because $6 < 30$, `random_subset` was clamped to 6 tokens. Random deletion masked only 6 tokens while induction masked 30 tokens, confounding the control.

3. **Absence of Unified Contract Functions (`compute_eviction_mask`, `build_rescue_mask`, etc.):**
   `PROJECT.md` §Interface Contracts specified `compute_eviction_mask`, `build_rescue_mask`, `build_induction_mask`, and `build_random_mask`. These functions were absent or unlinked in `src/pfseb/`.

4. **Missing Fine-Grained Budget Grid Support:**
   Eviction functions did not support arbitrary budgets $B \in \{8, 12, 16, 20, 24, 32, 48, \text{"full"}\}$ or handle string `"full"` gracefully.

---

## 2. Logic Chain

From these observations, the following step-by-step implementations and fixes were deduced and executed:

1. **Differentiated Policy Implementations (`src/pfseb/eviction.py`):**
   - **H2O (`compute_h2o_scores`):** Accumulates continuous attention mass over all query steps, layers, and heads:
     $$\text{score}_{H2O}[k] = \sum_{l=0}^{L-1} \sum_{h=0}^{H-1} \sum_{q} A_{l, h, q, k}$$
   - **SnapKV (`compute_snapkv_scores`):** Implements observation window pooling over the prompt tail $q \in [\max(0, P - W_{obs}), P)$ ($W_{obs}=16$). Focuses on immediate task queries rather than early prefix tokens.
   - **Scissorhands (`compute_scissorhands_scores`):** Implements persistence counting exceeding threshold $\tau$ (adaptive $\tau = 1.0 / P$ or user threshold):
     $$\text{score}_{Scissorhands}[k] = \sum_{l, h, q} \mathbb{I}(A_{l, h, q, k} > \tau)$$
     Rewards consistent attention across multiple steps rather than high one-off spikes.
   - **Recency-Only (`recency_keep_mask`):** Retains sinks $S$ and most recent $B-S$ tokens, evicting all intermediate context.
   - **Random Eviction (`random_keep_mask`):** Retains sinks $S$ and recency $W$, and randomly samples $P - B$ tokens from non-protected candidates using a deterministic generator.

2. **Universal Interface Contract (`src/pfseb/eviction.py:compute_eviction_mask`):**
   - Implemented `compute_eviction_mask(scores, policy, budget, num_sink, recency_window, prompt_len, generator, device, attentions, snapkv_window, scissor_threshold, as_4d)`.
   - Supports numeric budgets and string `"full"`.
   - Returns `(pmask, evicted_indices)` where `pmask` is `(1, 1, 1, L)` and `evicted_indices` is the sorted list of zeroed token positions.

3. **Causal Intervention Battery (`src/pfseb/causal.py`):**
   - **`get_candidate_positions`:** Returns non-sink positions `[num_sink, P)`.
   - **`sample_random_deletion_positions`:** Strictly enforces $|R| == k = |E|$. When $P=38, B=8$, the non-sink pool has $38 - 2 = 36$ tokens, allowing exactly $k = 30$ tokens to be sampled without clamping. Raises `ValueError` if $k > |candidate\_pool|$.
   - **`build_rescue_mask`:** Unmasks all positions (sets to 1.0), restoring benign continuation.
   - **`build_induction_mask`:** Masks candidate positions $E$ under full cache $C_0$ without running eviction scoring.
   - **`build_random_mask` / `build_random_mask_with_indices`:** Masks exactly $|R| = |E|$ randomly sampled non-sink positions under $C_0$.
   - **`compute_causal_contrasts`:** Computes point estimates for $\Delta_{rescue}, \Delta_{induction}, \Delta_{random}$ and evaluates acceptance criteria.

4. **Harness Upgrades (`src/pfseb/harness.py`):**
   - Upgraded `prompt_evicted_positions` to branch on policy, calling `compute_eviction_mask` with policy-specific score aggregators.
   - Updated `generate_static_masked` to support mask tensors as well as index lists, and added `pin_positions` support for Rescue operations.
   - Added `evaluate_budget_sweep` across `BUDGET_SWEEP_GRID = (8, 12, 16, 20, 24, 32, 48, "full")`.
   - Added `evaluate_causal_battery_single` executing all 5 conditions with strict size equality $|R| == |E|$.

5. **Unit Test Verification (`tests/pfseb/test_causal.py` & `tests/pfseb/test_eviction.py`):**
   - Implemented 8 unit tests in `tests/pfseb/test_causal.py` verifying sink protection, $|R| == |E| = 30$ size matching, mask shapes, and contrast logic.
   - Updated `tests/pfseb/test_eviction.py` to test contract compliance, budget sweep grid, and verified that SnapKV and Scissorhands produce non-identical rankings to H2O.

---

## 3. Caveats

- Model autoregressive generation in `harness.py` relies on standard Hugging Face model forward passes (`attn_implementation="eager"` or SDPA).
- Milestones 2 and 3 (`scripts/run_pfseb_campaign_004.py` and `train_mvp.py`) are exclusively owned by other workers; they should import and call `prompt_evicted_positions`, `compute_eviction_mask`, and the causal battery functions from `src/pfseb/`.
- No caveats regarding mathematical soundness, policy differentiation, or size equality.

---

## 4. Conclusion

Milestone 1 is complete:
1. Core cache policies (H2O, SnapKV, Scissorhands, Recency-only, Random) are authentically implemented with differentiated mechanics (observation window pooling, persistence counting, recency sliding, stochastic sampling).
2. Fine-grained budget sweeps support arbitrary budgets $B \in \{8, 12, 16, 20, 24, 32, 48, \text{"full"}\}$.
3. The 3-part causal intervention battery (Rescue, Induction, Size-Matched Random Deletion) is implemented with strict size equality $|R| == |E|$, fixing Defect G3.
4. Interface contracts conforming to `PROJECT.md` are established in `src/pfseb/eviction.py`, `src/pfseb/causal.py`, and `src/pfseb/harness.py`.

---

## 5. Verification Method

To independently verify this implementation:

1. **Causal Battery Unit Tests:**
   ```bash
   python -m tests.pfseb.test_causal
   ```
   *Expected result:* 8/8 tests pass, confirming sink protection, strict $|R| == 30$ when $P=38, B=8$, mask shapes, and contrast calculation.

2. **Eviction Policy Unit Tests:**
   ```bash
   python -m tests.pfseb.test_eviction
   ```
   *Expected result:* 11/11 tests pass, confirming contract compliance, monotonicity, budget sweep grid, and non-identical ranking between SnapKV, Scissorhands, and H2O.

3. **Campaign 004 Opaque-Box Test Suite:**
   ```bash
   python -m unittest tests/test_campaign_004.py
   ```
   *Expected result:* Tests for Tier 1 policies, budgets, causal interventions, Tier 2 boundary cases, and Tier 3 matrix pass.

4. **Invalidation Conditions:**
   - SnapKV or Scissorhands producing identical evicted index sets to H2O on structured prefill attention.
   - Random deletion masking fewer or more tokens than $|E|$ on any prompt.
   - Sinks ($0..S-1$) being included in random deletion masks.
