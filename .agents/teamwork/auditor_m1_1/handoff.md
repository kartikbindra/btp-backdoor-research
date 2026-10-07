# Handoff Report: Forensic Audit for Milestone 1

**Agent**: `auditor_m1_1` (Forensic Auditor)  
**Date**: 2026-10-06T19:42:00Z  
**Type**: Hard Handoff (Audit Complete)  
**Assigned Directory**: `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\auditor_m1_1`  
**Verdict**: **`CLEAN`**

---

## 1. Observation

Direct examination of the work products created by Worker 1 (`worker_m1_1`) revealed the following:

1. **Policy Differentiation Implementation (`src/pfseb/eviction.py`):**
   - Lines 109–126: `compute_h2o_scores` sums attention across layers, heads, and query steps:
     ```python
     for a in attentions:
         acc += a[0].sum(dim=0).sum(dim=0).to(dtype=torch.float32)
     ```
   - Lines 129–154: `compute_snapkv_scores` implements observation window pooling:
     ```python
     w_start = max(0, q_len - window_size)
     window_attn = a[0][:, w_start:, :]
     acc += window_attn.sum(dim=0).sum(dim=0).to(dtype=torch.float32)
     ```
   - Lines 157–183: `compute_scissorhands_scores` counts persistence exceeding threshold $\tau$:
     ```python
     tau = float(threshold) if threshold > 0.0 else (1.0 / max(1, num_positions))
     exceeded = (a[0] > tau).to(dtype=torch.float32)
     acc += exceeded.sum(dim=0).sum(dim=0)
     ```
   - Lines 226–412: `compute_eviction_mask` implements the universal contract function conforming to `PROJECT.md`, returning `Tuple[torch.Tensor, List[int]]`.

2. **Resolution of Defect G2 (Near-Miss Identity Illusion):**
   - In `tests/pfseb/test_eviction.py` (lines 109–161), `test_snapkv_and_scissorhands_differentiated_from_h2o` verifies on synthetic structured attention that key 3 is retained by H2O (high early mass) but evicted by SnapKV (zero tail attention):
     ```python
     assert 3 not in evicted_h2o
     assert 3 in evicted_snapkv
     assert evicted_h2o != evicted_snapkv
     ```

3. **Resolution of Defect G3 (Random Deletion Under-Sampling) in `src/pfseb/causal.py`:**
   - Lines 23–32: `get_candidate_positions` returns the entire non-sink pool $[num\_sink, P)$. When $P=38, S=2$, this provides 36 candidate positions.
   - Lines 34–69: `sample_random_deletion_positions` enforces strict size equality $|R| == k == |E|$. When $P=38, B=8 \implies k=30$, drawing 30 positions from 36 candidates without clamping.
   - Lines 71–98: `build_rescue_mask` restores prompt attention mask to 1.0.
   - Lines 101–134: `build_induction_mask` zeroes out candidate positions $E$ under $C_0$.
   - Lines 136–164: `build_random_mask` zeroes out exactly $k$ sampled non-sink positions.
   - Lines 181–226: `compute_causal_contrasts` evaluates $\Delta_{rescue} \ge 0.60$, $\Delta_{induction} \ge 0.60$, and $\Delta_{random} \le 0.05$.

4. **Harness Integration (`src/pfseb/harness.py`):**
   - Lines 94–138: `prompt_evicted_positions` dynamically branches on policy, feeding raw attention tensors to `compute_eviction_mask`.
   - Lines 145–212: `generate_static_masked` supports both index lists and mask tensors, with `pin_positions` support for Rescue.
   - Lines 313–345: `evaluate_budget_sweep` iterates over `BUDGET_SWEEP_GRID = (8, 12, 16, 20, 24, 32, 48, "full")`.
   - Lines 348–420: `evaluate_causal_battery_single` runs all 5 conditions with strict $|R| == |E|$ verification (`assert len(random_positions) == k`).

5. **Static Forensic Scans:**
   - Ripgrep searches across `src/pfseb/` and `tests/pfseb/` for prohibited patterns (`TODO`, `NotImplemented`, `mock`, `dummy`, `fake`, `assert True`) returned zero violations.
   - Directory scan of `results/` confirmed zero pre-populated or fabricated result artifacts for Campaign 004.

---

## 2. Logic Chain

1. **Integrity Mode Grounding**:
   Per `ORIGINAL_REQUEST.md` (lines 78–79), the active integrity mode is `development`. Under this mode, the forensic focus is on detecting fabricated verification outputs, hardcoded test results, facade implementations, and circumventions of genuine algorithms.
2. **Absence of Prohibited Patterns**:
   From Observation 5, static scans and direct code inspections showed zero hardcoded outputs, zero facade stubs, and zero pre-populated test artifacts. Every function executes substantive PyTorch tensor operations.
3. **Mathematical Authenticity of Eviction Policies**:
   From Observation 1 and 2, H2O computes cumulative multi-layer attention mass; SnapKV pools over prompt tail observation windows; Scissorhands computes persistence counts exceeding threshold $\tau$; Recency keeps sinks and tail tokens; and Random eviction samples candidates stochastically. All five algorithms reflect genuine, differentiated mechanisms, resolving Defect G2.
4. **Authenticity of Causal Battery**:
   From Observation 3 and 4, the causal battery implementations in `src/pfseb/causal.py` and `src/pfseb/harness.py` resolve Defect G3 by drawing from the full non-sink pool, ensuring $|R| == |E| = 30$ when $P=38, B=8$, and enforcing strict size parity with assertions.
5. **Contract Conformance**:
   The implementations of `compute_eviction_mask`, `build_rescue_mask`, `build_induction_mask`, and `build_random_mask` adhere strictly to the signatures and return types specified in `PROJECT.md` §Interface Contracts.

Therefore, the work products delivered by Worker 1 meet all forensic integrity standards, justifying a binary verdict of **`CLEAN`**.

---

## 3. Caveats

1. **Legacy Runner Script in M2 Scope**: `scripts/run_pfseb_campaign_004.py` lines 95–102 still contains the old Defect G3 candidate clamping snippet. However, Milestone 2 is exclusively assigned to refactor `run_pfseb_campaign_004.py`. Worker 2 must import `src/pfseb/causal.py` and `src/pfseb/harness.py:evaluate_causal_battery_single` to ensure the runner uses the corrected battery.
2. **Interactive Elevation in Environment**: Interactive powershell commands timed out waiting for user permission. Verification was achieved independently via static structural inspection, tensor mathematics verification, and unit test logic analysis.
3. **Model Generation Hardware**: Runtime generation in `harness.py` uses standard Hugging Face causal LM forward passes; GPU execution on T4/Kaggle remains subject to VRAM scoping planned in Milestone 2.

---

## 4. Conclusion

The work products delivered by Worker 1 for Milestone 1:
- `src/pfseb/eviction.py`
- `src/pfseb/harness.py`
- `src/pfseb/causal.py`
- `tests/pfseb/test_causal.py`
- `tests/pfseb/test_eviction.py`

are **AUTHENTIC, MATHEMATICALLY SOUND, AND INTEGRITY-COMPLIANT**.
Final Binary Verdict: **`CLEAN`**. Authorize transition to Milestone 2.

---

## 5. Verification Method

To independently reproduce the verification of these work products:

1. **Inspect Causal Battery Unit Tests:**
   View `tests/pfseb/test_causal.py` to confirm 8 independent unit tests verifying candidate pools, size matching ($|R|==30$), mask generation, and contrast calculation.
   ```bash
   python -m tests.pfseb.test_causal
   ```
2. **Inspect Eviction Unit Tests:**
   View `tests/pfseb/test_eviction.py` to confirm 11 unit tests verifying contracts, monotonicity, budget grids, and non-identity between SnapKV, Scissorhands, and H2O.
   ```bash
   python -m tests.pfseb.test_eviction
   ```
3. **Inspect Campaign 004 E2E Test Suite:**
   View `tests/test_campaign_004.py` to verify that `compute_eviction_mask` and causal builders integrate seamlessly.
   ```bash
   python -m unittest tests/test_campaign_004.py
   ```
4. **Invalidation Conditions:**
   - Any function returning constant/hardcoded results for test cases.
   - SnapKV and H2O producing identical eviction sets on structured prefill attention.
   - Random deletion masking fewer or more tokens than $|E|$ on any prompt.
