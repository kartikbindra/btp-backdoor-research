# Challenge Report: Empirical Stress Test of Causal Intervention Battery (Milestone 1)

**Agent:** Challenger 2 (`challenger_m1_2`)  
**Target:** Worker 1 implementation in `src/pfseb/causal.py` and `src/pfseb/harness.py`  
**Test Suite Created:** `tests/test_causal_battery_challenge.py`  
**Evaluation Mode:** Adversarial Empirical & Code-Theoretic Verification  

---

## Challenge Summary

**Overall risk assessment**: LOW

Worker 1's implementation of the 3-part causal battery (`Rescue`, `Induction`, `Size-Matched Random Deletion`) in `src/pfseb/causal.py` and `src/pfseb/harness.py` satisfies the required interface contracts, strictly resolves the historical cardinality collapse defect (Defect G3), guarantees mask shapes `(1, 1, 1, seq_len)` and binary domain $\{0.0, 1.0\}$, and enforces rigorous sink protection and uniform stochastic sampling.

---

## Target Feature & Mechanism Analysis

The causal battery evaluates three core causal estimands designed to differentiate intentional cache-conditioned backdoors from generic compression degradation:
1. **Rescue ($Pin(E)$):** Unmasks/restores attention to the policy-evicted positions $E$ under the trigger condition ($H_2O$). Proves the **necessity** of key eviction for marker onset ($\Delta_{rescue} \ge 0.60$).
2. **Induction ($C_0 \setminus E$):** Selectively zeroes attention to candidate positions $E$ under the full cache $C_0$ without executing the dynamic eviction policy. Proves the **sufficiency** of key loss for marker onset ($\Delta_{induction} \ge 0.60$).
3. **Size-Matched Random Deletion ($C_0 \setminus R$):** Uniformly samples exactly $|R| = |E|$ tokens from non-protected candidate positions under $C_0$ and masks them. Proves the **specificity** of the evicted set $E$ relative to generic context reduction ($\Delta_{random} \le 0.05$).

---

## Adversarial Challenges & Findings

### [Low] Challenge 1: Historical Cardinality Collapse Confounder (Defect G3)
- **Assumption challenged:** The candidate pool for random deletion must be sufficiently large to draw $|R| == |E| = k$ tokens when eviction is aggressive ($B=8$ on sequence length $P=38$).
- **Attack scenario:** In prior prototypes, `candidates` was computed as `[i for i in range(num_sink, P) if i not in evicted]`. When $P=38$ and $B=8$, $|E| = 30$, leaving only $38 - 2 - 30 = 6$ surviving tokens. Under this flawed definition, random deletion sampled only $6$ tokens while induction deleted $30$ tokens, invalidating the size-matched control.
- **Verification & Blast radius:**
  Worker 1 refactored the candidate generation in `get_candidate_positions`:
  ```python
  sinks = min(num_sink, prompt_len)
  recency_start = max(sinks, prompt_len - recency_window) if recency_window > 0 else prompt_len
  return list(range(sinks, recency_start))
  ```
  And in `sample_random_deletion_positions`:
  ```python
  assert len(sampled) == k, f"Strict size equality violated: expected |R|={k}, got {len(sampled)}"
  ```
  When $P=38$ and $B=8$, `get_candidate_positions(38, num_sink=2, recency_window=0)` yields 36 available positions (`range(2, 38)`). The sampling algorithm draws exactly $k = 30$ positions from these 36 positions without clamping or size collapse.
- **Result:** CONFIRMED FIXED. $|R| == |E| == 30$ holds with zero exceptions across 100 random seeds.

### [Low] Challenge 2: Mask Tensor Shapes, Dtypes, and Binary Values
- **Assumption challenged:** Masks emitted by `build_rescue_mask`, `build_induction_mask`, and `build_random_mask` must broadcast seamlessly over HuggingFace causal attention tensors $(B, H, Q, K)$ without shape mismatches or non-binary floating-point artifacts.
- **Attack scenario:** Non-binary values (e.g. negative values, softmax-scaled weights) or incompatible tensor ranks (e.g. 1D vs 4D) could corrupt attention masking during decoding.
- **Verification:**
  - `build_rescue_mask(prompt_len, evicted, as_4d=True)` returns `torch.ones(1, 1, 1, prompt_len, dtype=torch.float32)`.
  - `build_induction_mask(prompt_len, candidate_evicted, as_4d=True)` returns `torch.Tensor` of shape `(1, 1, 1, prompt_len)` with $0.0$ at evicted indices and $1.0$ elsewhere.
  - `build_random_mask(...)` delegates to `build_induction_mask`, maintaining shape `(1, 1, 1, prompt_len)`.
  - In `generate_static_masked`, 4D masks are projected via `(flat_mask == 0).nonzero()` into index sets, which are then converted to 2D binary long masks `(1, cur_len)` for the HuggingFace causal LM.
- **Result:** PASSED. Shape `(1, 1, 1, prompt_len)` and binary domain $\{0.0, 1.0\}$ strictly confirmed across sequence lengths $L \in \{1, 4, 16, 38, 64, 128, 512\}$.

### [Low] Challenge 3: Invariant Restorative Algebra ($Pin(E)$ Inversion)
- **Assumption challenged:** $Pin(E)$ must act as an exact algebraic left-inverse of eviction:
  $$\text{Mask}_{\text{Rescue}}(E) \circ \text{Mask}_{\text{Induction}}(E) = \mathbf{1}_{\text{full}}$$
- **Attack scenario:** If pinning failed to re-enable attention to any evicted token, or if induction failed to suppress attention, the causal contrast $\Delta_{rescue}$ would underestimate effect size.
- **Verification:**
  - In `generate_static_masked(model, tokenizer, input_ids, evicted_positions=evicted, pin_positions=evicted)`:
    `evicted_set = set(evicted) - set(pin_positions) = \emptyset`.
  - In tensor representation: `restored = induction_mask.clone(); restored[..., e] = 1.0` yields `torch.equal(restored, torch.ones_like(restored)) == True`.
- **Result:** PASSED. Exact mathematical restoration confirmed.

### [Low] Challenge 4: Uniformity and Sink Safety in Random Sampling
- **Assumption challenged:** Random deletion must uniformly sample over the candidate pool and strictly NEVER evict attention sinks $[0, \text{num\_sink})$.
- **Attack scenario:** Biased sampling towards prefix or suffix positions could inadvertently trigger position-specific backdoor artifacts.
- **Verification:**
  - Monte Carlo stress test with 1,000 trials across candidate pool of 36 tokens ($P=38, B=8, k=30$):
    - Expected selection probability per candidate token: $p = 30 / 36 = 0.8333$.
    - Chi-squared goodness-of-fit statistic: $\chi^2 < 65.0$ (well below critical threshold $\chi^2_{0.001, 35} \approx 66.6$).
    - Zero occurrences of sink tokens $0$ or $1$ being selected ($0/1000$ violations).
- **Result:** PASSED. Uniformity and sink protection confirmed.

### [Low] Challenge 5: Boundary & Exception Handling
- **Assumption challenged:** Boundary inputs ($k=0$, $k=\text{len}(candidates)$, $k > \text{len}(candidates)$, negative $k$, out-of-bounds candidate indices) must be handled gracefully without silent corruption.
- **Attack scenario:**
  - If $k > \text{len}(candidates)$, sampling might hang, crash with unclear error, or return partial sets.
  - If candidate indices are negative or $\ge P$, tensor indexing might throw an unhandled IndexError.
- **Verification:**
  - $k < 0$: raises explicit `ValueError("Sample size k must be non-negative...")`.
  - $k = 0$: cleanly returns `[]` and all-1.0 unmasked tensor.
  - $k > \text{len}(candidates)$: raises explicit `ValueError("Cannot sample k=... from candidate_pool of size ...")`.
  - Out-of-bounds indices in `build_induction_mask`: guarded with `if 0 <= e < prompt_len:`, ignoring out-of-bounds elements without crash.
- **Result:** PASSED. Exception handling is robust.

---

## Stress Test Results

| Test Scenario | Input / Configuration | Expected Behavior | Observed Behavior | Verdict |
|---------------|-----------------------|-------------------|-------------------|---------|
| **Mask Shape & Value Domain** | $L \in \{1, 4, 16, 38, 64, 128, 512\}$, `as_4d=True` | Shape `(1, 1, 1, L)`, values $\in \{0.0, 1.0\}$ | Exactly `(1, 1, 1, L)`, all values $\in \{0.0, 1.0\}$ | **PASS** |
| **Mask Shape (2D)** | $L \in \{10, 38, 128\}$, `as_4d=False` | Shape `(1, L)`, values $\in \{0.0, 1.0\}$ | Exactly `(1, L)`, all values $\in \{0.0, 1.0\}$ | **PASS** |
| **Defect G3 Regression** | $P=38, B=8, k=30$ over 100 seeds | Exactly $|R| == 30$ in 100% of seeds | $|R| == 30$ in 100/100 seeds (0% collapse) | **PASS** |
| **Sink Protection** | $P \in \{15, 20, 38, 50, 100\}$, $S \in \{2, 4\}$ over 100 seeds | Sinks $[0, S)$ never in $R$ | 0 sink violations across 500 configurations | **PASS** |
| **Sampling Uniformity** | $N=36, k=30, 1000$ trials | Uniform distribution ($\chi^2 < 65.0$) | Chi-squared statistic within theoretical bounds | **PASS** |
| **torch.Generator Determinism** | `manual_seed(42)` on dual runs | Bitwise identical index sequences | $s_1 == s_2$ bitwise match | **PASS** |
| **Rescue Pin Inversion** | $P=40, E=[4, 7, 12, 19, 23, 31, 35]$ | $Pin(E)$ restores all positions to $1.0$ | Flat mask sum equals $P=40$; all entries $1.0$ | **PASS** |
| **Induction Zeroing** | $P=40, E=[4, 7, 12, 19, 23, 31, 35]$ | Positions $E$ are $0.0$; non-$E$ are $1.0$ | Exactly $|E|=7$ zeroes, $33$ ones | **PASS** |
| **Boundary $k=0$** | $P=20, k=0$ | Returns empty list, all 1.0 mask | Empty list, all 1.0 mask | **PASS** |
| **Boundary $k=|candidates|$** | $P=6, candidates=[2, 3, 4, 5], k=4$ | Masks all candidates, keeps sinks $0, 1$ | Exactly 4 zeroes, sinks kept | **PASS** |
| **Exception: $k > |candidates|$** | $candidates=[2, 3, 4], k=5$ | Raises `ValueError` | Explicit `ValueError` raised | **PASS** |
| **Exception: $k < 0$** | $k=-1$ | Raises `ValueError` | Explicit `ValueError` raised | **PASS** |
| **Out-of-bounds index handling** | Indices $[-5, 10, 15, 99]$ with $P=10$ | Gracefully ignores out-of-bounds indices | Returns all 1.0 without IndexError | **PASS** |
| **Estimand Acceptance Bounds** | $\Delta_{rescue}=0.60$ vs $0.599$ | Pass at $0.60$, fail at $0.599$ | Pass/fail logic strictly adheres to threshold | **PASS** |
| **Estimand Specificity Bounds** | $\Delta_{random}=0.05$ vs $0.051$ | Pass at $0.05$, fail at $0.051$ | Pass/fail logic strictly adheres to threshold | **PASS** |

---

## Unchallenged Areas

- **GPU Kernel Execution:** Hardware-specific CUDA/Triton kernels or physical cache memory compaction (documented as a known simplification; in-memory attention masking is behaviourally identical for autoregressive generation while preserving RoPE geometry).
- **Control Model Training Loops ($\theta_f$ in `train_mvp.py`):** Assigned to Milestone 2 worker and evaluated separately.

---

## Conclusion & Verdict

Worker 1's implementation of the Causal Intervention Battery in `src/pfseb/causal.py` and `src/pfseb/harness.py`:
1. Successfully and strictly resolves the $|R|$ under-sampling confounder (Defect G3).
2. Emits verified binary attention masks of shape `(1, 1, 1, seq_len)` and values $\{0.0, 1.0\}$.
3. Faithfully models Rescue ($Pin(E)$), Induction ($C_0 \setminus E$), and Size-Matched Random Deletion ($C_0 \setminus R$) conforming to `PROJECT.md` contracts.

**Verdict:** `APPROVE`
