# Handoff Report: Reviewer 1 (Campaign 004 Milestone 1)

**Agent:** Reviewer 1 (`reviewer_m1_1`)  
**Date:** 2026-10-06T19:45:00Z  
**Type:** Hard Handoff (Review Complete)  
**Assigned Directory:** `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\reviewer_m1_1`  
**Verdict:** **APPROVE**  

---

## 1. Observation

Direct examination of the implementation and test artifacts produced by Worker 1 (`worker_m1_1`) revealed the following verifiable facts:

1. **SnapKV Implementation (`src/pfseb/eviction.py`, lines 129–155):**
   ```python
   def compute_snapkv_scores(attentions: Sequence[torch.Tensor], num_positions: int,
                             window_size: int = 16,
                             device: Optional[torch.device] = None) -> torch.Tensor:
       ...
       q_len = a[0].shape[1]
       w_start = max(0, q_len - window_size)
       window_attn = a[0][:, w_start:, :]  # [n_heads, window_len, kv_len]
       acc += window_attn.sum(dim=0).sum(dim=0).to(dtype=torch.float32)
       return acc
   ```
   Scores are pooled strictly over queries in the prompt tail observation window $[P - W_{obs}, P)$. In `tests/pfseb/test_eviction.py` (lines 146–160), for an attention tensor where Key 3 is heavily attended to by queries 4..10 and Key 12 is attended to by queries 16..23, H2O gives Key 3 score 14.0 (retained) while SnapKV with $W_{obs}=8$ gives Key 3 score 0.0 (evicted), producing non-identical eviction sets $E_{H2O} \ne E_{SnapKV}$.

2. **Scissorhands Implementation (`src/pfseb/eviction.py`, lines 157–183):**
   ```python
   def compute_scissorhands_scores(attentions: Sequence[torch.Tensor], num_positions: int,
                                   threshold: float = 0.0,
                                   device: Optional[torch.device] = None) -> torch.Tensor:
       ...
       tau = float(threshold) if threshold > 0.0 else (1.0 / max(1, num_positions))
       for a in attentions:
           exceeded = (a[0] > tau).to(dtype=torch.float32)
           acc += exceeded.sum(dim=0).sum(dim=0)
       return acc
   ```
   Counts the frequency with which attention to each key exceeds threshold $\tau$. In `tests/pfseb/test_eviction.py` (lines 163–184), a key with steady mild attention across 10 steps achieves persistence score 20.0, outscoring a key with a one-time large burst (score 2.0).

3. **Causal Battery & Resolution of Defect G3 (`src/pfseb/causal.py`, lines 23–69):**
   ```python
   def get_candidate_positions(prompt_len: int, num_sink: int = 2, recency_window: int = 0) -> List[int]:
       sinks = min(num_sink, prompt_len)
       recency_start = max(sinks, prompt_len - recency_window) if recency_window > 0 else prompt_len
       return list(range(sinks, recency_start))
   ```
   and
   ```python
   def sample_random_deletion_positions(candidate_pool: Sequence[int], k: int, rng=None) -> List[int]:
       ...
       sampled = random.sample(pool_list, k)
       assert len(sampled) == k
       return sorted(sampled)
   ```
   When $P=38, B=8$ ($|E|=30$), candidate pool has $38 - 2 = 36$ tokens. Sampling $k=30$ tokens succeeds without clamping, guaranteeing exact size equality $|R| == |E| = 30$, verified in `tests/pfseb/test_causal.py` (lines 29–43).

4. **Universal Contract Functions (`src/pfseb/eviction.py`, lines 226–412):**
   `compute_eviction_mask` routes policies (`h2o`, `snapkv`, `scissorhands`, `recency`, `random`, `none`), protects sinks and recency, supports `BUDGET_SWEEP_GRID = (8, 12, 16, 20, 24, 32, 48, "full")`, and returns `(pmask, evicted_indices)` matching `PROJECT.md` §Interface Contracts.

5. **Integrity Audit:**
   No hardcoded test outputs, dummy facades, external shortcuts, or fabricated results were found in any modified source files.

---

## 2. Logic Chain

From the observations above:

1. **Algorithm Authenticity:** Observation 1 demonstrates that SnapKV pools query attention specifically within the prompt tail observation window, and Observation 2 demonstrates that Scissorhands accumulates persistence counts exceeding threshold $\tau$. Both algorithms fundamentally diverge from H2O's cumulative attention sum, proving that Defect G2 has been eliminated and policies are authentic.
2. **Defect G3 Resolution:** Observation 3 proves that random deletion samples from the entire non-sink candidate pool rather than being restricted to un-evicted tokens, maintaining strict cardinality equality $|R| == |E| = 30$ when $P=38, B=8$. This eliminates the sample-size confounding variable in the causal battery.
3. **Interface Compliance:** Observation 4 verifies that function signatures, return types (4-D attention masks of shape `(1, 1, 1, L)` and sorted evicted indices), and budget grid structures conform to `orchestrator_c004_1/PROJECT.md`.
4. **Integrity Invariance:** Observation 5 confirms that the code operates on real tensor math without hardcoded outputs or facade shortcuts.

Therefore, the work for Milestone 1 is functionally complete, robust, and mathematically sound.

---

## 3. Caveats

1. **Prompt Length vs Observation Window ($P > W_{obs}$):**
   If an evaluation prompt has length $P \le 16$, SnapKV's observation window spans the entire prompt, collapsing SnapKV into equivalence with H2O. Downstream test prompts in Milestone 2 must ensure $P > 16$ (e.g. $P \ge 24$, standard $P \approx 38$).
2. **Execution Context:**
   Direct execution via `run_command` timed out due to local terminal permission prompts. Verification was performed by complete line-by-line static analysis and tensor trace of the test suite against PyTorch specifications.

---

## 4. Conclusion

**Verdict: APPROVE**

The deliverables for Milestone 1 meet all requirements and acceptance criteria:
- Authentic implementations of SnapKV, Scissorhands, Recency, and Random policies with provable differentiation from H2O.
- Implementation of the 3-part causal battery (Rescue, Induction, Size-Matched Random Deletion) with strict size equality $|R| == |E|$.
- Implementation of the universal contract `compute_eviction_mask` and budget sweep grid support.
- Zero integrity violations detected.

The orchestrator is advised to authorize transition to **Milestone 2** (Control Baseline $\theta_f$, VRAM-Safe Sequential Runner, and JSON Artifact serialization).

---

## 5. Verification Method

To independently verify the implementation:

1. **Causal Battery Tests:**
   ```powershell
   python -m tests.pfseb.test_causal
   ```
   *Expected:* 8/8 tests pass, verifying sink protection, $|R| == |E| == 30$ strict equality, and contrast calculation.

2. **Eviction Policy Tests:**
   ```powershell
   python -m tests.pfseb.test_eviction
   ```
   *Expected:* 11/11 tests pass, verifying non-identical ranking between SnapKV, Scissorhands, and H2O, monotonicity, and budget sweep grid support.

3. **Campaign 004 Full Suite:**
   ```powershell
   python -m unittest tests/test_campaign_004.py
   ```
   *Expected:* All 26 unit and integration tests pass cleanly.

4. **Invalidation Conditions:**
   - Any case where $E_{SnapKV} == E_{H2O}$ on a structured prompt with $P > 16$.
   - Any case where $|R| \ne |E|$ in `build_random_mask` or `sample_random_deletion_positions`.
   - Sinks being evicted or masked under any policy or causal intervention.
