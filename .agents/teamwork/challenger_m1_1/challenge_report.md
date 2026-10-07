# Adversarial Challenge Report: Campaign 004 Milestone 1

**Reviewer / Archetype:** Empirical Challenger (`challenger_m1_1`)  
**Target:** Worker 1 Implementation (`src/pfseb/eviction.py`, `src/pfseb/harness.py`, `src/pfseb/causal.py`)  
**Scope:** Core Cache Policies (H2O, SnapKV, Scissorhands, Recency-only, Random), Budget Sweep Grid, Causal Battery  
**Verdict:** `REQUEST_CHANGES`

---

## Challenge Summary

**Overall risk assessment**: **HIGH**

While Worker 1 successfully resolved the core near-miss identity illusion (Defect G2) by introducing distinct scoring mechanisms for SnapKV and Scissorhands, and correctly fixed the causal random deletion under-sampling bug (Defect G3) with strict $|R| == |E|$ cardinality, rigorous adversarial stress-testing identified two critical defects and three subtle boundary failure modes:

1. **CRITICAL RUNTIME CRASH:** `src/pfseb/harness.py` calls `random.Random(seed)` on line 398 in `evaluate_causal_battery_single()` without importing `random`, raising an immediate `NameError` at runtime.
2. **HIGH REPRODUCIBILITY DEFECT:** `prompt_evicted_positions()` hardcodes `generator=torch.Generator(device="cpu").manual_seed(0)` for the `random` policy on line 125, ignoring multi-seed configurations and producing identical random evictions across all experimental evaluation seeds.
3. **MEDIUM SEMANTIC DISCREPANCY ($B \le 0$):** `topk_keep_mask` treats `budget <= 0` as keep-all (full cache), whereas `compute_eviction_mask` treats `budget = -1` as evicting all candidate tokens ($n\_evict = P + 1$).
4. **MEDIUM POLICY BOUNDARY INCONSISTENCY ($B < S + W$):** Under aggressive budgets $B < S + W$, H2O, SnapKV, Scissorhands, and Random strictly preserve both sinks and recency ($|kept| = S + W$), whereas Recency-only truncates recency tokens to retain only $B$ tokens.
5. **LOW/MEDIUM SHORT SEQUENCE COLLAPSE ($P \le W_{obs}$):** When prompt sequence length $P \le W_{obs}$ (default $W_{obs}=16$), SnapKV's observation window spans the entire sequence, causing SnapKV to mathematically degenerate to 100% identical scores and evictions as H2O.

---

## Challenges

### [Critical] Challenge 1: Missing Import `random` in `src/pfseb/harness.py` Causing Runtime Crash

- **Assumption challenged:** The causal intervention execution pipeline in `src/pfseb/harness.py` is executable and production-ready.
- **Attack scenario:** Calling `evaluate_causal_battery_single(model, tokenizer, prompt_ids, budget=8)` when candidate eviction occurs ($k > 0$).
- **Observation:** Line 398 in `src/pfseb/harness.py`:
  ```python
  rng = random.Random(seed)
  random_positions = sample_random_deletion_positions(candidates, k, rng)
  ```
  Inspection of lines 1–45 in `src/pfseb/harness.py` reveals that `import random` is completely absent.
- **Blast radius:** Immediate unhandled exception `NameError: name 'random' is not defined` when executing the causal battery evaluation helper in Milestone 2 and integration scripts.
- **Mitigation:** Add `import random` to top-level imports in `src/pfseb/harness.py`.

---

### [High] Challenge 2: Hardcoded Generator Seed Zero in `prompt_evicted_positions` for Random Policy

- **Assumption challenged:** The Random eviction near-miss policy satisfies multi-seed reproducibility and variance tracking across seeds (Requirement R5 & Acceptance Criteria).
- **Attack scenario:** Running multi-seed evaluations (e.g. seeds 42, 123, 999). Calling `prompt_evicted_positions(model, prompt_ids, cfg)` with `cfg.policy = "random"`.
- **Observation:** Line 125 in `src/pfseb/harness.py`:
  ```python
  if cfg.policy == "random":
      _, evicted = compute_eviction_mask(
          policy="random", budget=cfg.budget, num_sink=cfg.num_sink,
          recency_window=cfg.recency_window, prompt_len=P, device=prompt_ids.device,
          generator=torch.Generator(device="cpu").manual_seed(0)
      )
      return evicted
  ```
  Neither `EvictionConfig` nor `prompt_evicted_positions` provides an interface to pass a seed or generator. The seed `0` is hardcoded.
- **Blast radius:** Every multi-seed experiment run will execute identical random eviction sets, nullifying cross-seed empirical variance estimation for the Random baseline.
- **Mitigation:** Add an optional `seed: Optional[int] = None` field to `EvictionConfig` and forward `generator = torch.Generator(device="cpu").manual_seed(cfg.seed if cfg.seed is not None else 0)` in `prompt_evicted_positions`.

---

### [Medium] Challenge 3: Inconsistent Semantic Handling of Negative/Full Budgets ($B = -1$)

- **Assumption challenged:** Contract consistency across `EvictionConfig`, `topk_keep_mask`, and `compute_eviction_mask`.
- **Attack scenario:** Setting `budget = -1`, which is explicitly permitted by `EvictionConfig`'s post-init assertion (`assert int(self.budget) >= 1 or int(self.budget) == -1`).
- **Observation:**
  - In `topk_keep_mask` (line 84):
    ```python
    if b_int <= 0 or seq_len <= b_int:
        return torch.ones_like(scores, dtype=torch.bool)
    ```
    Budget `-1` returns keep-all (full cache, 0 evicted).
  - In `compute_eviction_mask` (lines 282, 312, 385):
    ```python
    is_full = (
        policy == "none" or
        (isinstance(budget, str) and budget.lower() == "full") or
        (isinstance(budget, (int, float)) and int(budget) >= prompt_len)
    )
    ```
    When `budget = -1`, `is_full` evaluates to `False`. Then `b_int = -1`, and `n_to_evict = max(0, prompt_len - (-1)) = prompt_len + 1`. Every single non-protected candidate is evicted!
- **Blast radius:** Incompatible behavior across functions: passing `-1` preserves full cache in low-level functions but causes maximal eviction in high-level contract functions.
- **Mitigation:** Update `compute_eviction_mask` line 282 to recognize `int(budget) == -1` or `int(budget) <= 0` as `is_full`.

---

### [Medium] Challenge 4: Policy Divergence Under Constrained Budgets ($B < S + W$)

- **Assumption challenged:** All policies enforce identical protection rules (sinks $S$ + recency $W$) when budget is tight.
- **Attack scenario:** Evaluating edge-case budgets where $B < S + W$ (e.g. $B = 6$, $S = 4$, $W = 8 \implies S + W = 12$).
- **Observation:**
  - In H2O, SnapKV, Scissorhands, and Random:
    Sinks and recency window are strictly protected ($|protected| = S + W = 12$). The remaining candidate slots are $6 - 12 = -6 \le 0$, so all non-protected candidates are evicted, and exactly $S + W = 12$ tokens are kept.
  - In Recency-only (lines 298–304 in `eviction.py`):
    ```python
    keep_recent_count = max(0, b_int - num_sink)
    recency_start = max(num_sink, prompt_len - keep_recent_count)
    recency = set(range(recency_start, prompt_len))
    kept = sinks | recency
    ```
    Here, `keep_recent_count = max(0, 6 - 4) = 2`. Only 2 recency tokens are kept instead of the $W=8$ recency window! Exactly $B = 6$ tokens are kept.
- **Blast radius:** Recency-only silently violates the $W$ recency window protection specification under $B < S + W$, whereas attention policies prioritize protection over budget truncation.
- **Mitigation:** Document this semantic distinction or standardize whether recency window $W$ is inviolable across all policies.

---

### [Low/Medium] Challenge 5: SnapKV Mathematical Collapse on Short Prompts ($P \le W_{obs}$)

- **Assumption challenged:** SnapKV is guaranteed to be differentiated from H2O across all prompt sequence lengths.
- **Attack scenario:** Prompts with sequence length $P \le W_{obs}$ (default $W_{obs} = 16$).
- **Observation:** In `compute_snapkv_scores`:
  ```python
  w_start = max(0, q_len - window_size)
  window_attn = a[0][:, w_start:, :]
  ```
  When $P \le 16$, `w_start = 0`. The observation window covers queries $0..P-1$, identical to full prefill attention. `compute_snapkv_scores` returns the exact same accumulated score as `compute_h2o_scores`, and SnapKV evictions become bitwise identical to H2O ($E_{snapkv} == E_{h2o}$).
- **Blast radius:** Near-miss policy selectivity tests on short prompts ($P \le 16$) will fail to differentiate SnapKV from H2O.
- **Mitigation:** Enforce test prompt lengths $P > W_{obs}$ (standard Campaign 004 prompts have $P \approx 38$), or set adaptive observation window $W_{obs} = \max(1, \min(16, P // 2))$ for short sequences.

---

## Stress Test Results

The adversarial test suite `tests/pfseb/test_eviction_adversarial.py` was constructed to evaluate all edge cases:

| Scenario | Input / Dynamics | Expected Behavior | Actual Behavior | Pass / Fail |
|---|---|---|---|---|
| **Uniform attention scores** | Uniform attention $A_{q, k} = 1/P$ | All non-protected positions tie; tie-breaking occurs | H2O, SnapKV, Scissorhands produce identical eviction sets | **PASS (Documented Degeneracy)** |
| **Extreme attention spikes** | Key with massive single spike vs Key with persistent mild attention | Scissorhands retains persistent key; H2O retains spike key | Differentiated retention: Scissorhands evicts spike, H2O keeps spike | **PASS** |
| **Negative scores & 1D threshold** | All negative scores with $\tau = 0.5$ | Scissorhands 1D heuristic maintains scoring signal | $s \ge \tau$ is False everywhere; ranks by $0.1 \times s$ (identical to H2O) | **PASS (Documented Heuristic Boundary)** |
| **Large sequence lengths** | $P \in \{512, 1024\}$, $B = 48$ | Linear scaling, valid masks, sink/recency preserved | Masks `(1, 1, 1, P)`, correct counts, no OOM | **PASS** |
| **Realistic SnapKV dynamics** | Early instruction queries vs tail question queries ($P = 28, W_{obs} = 8$) | SnapKV retains tail-attended keys; H2O retains instruction-attended keys | Eviction sets diverge ($E_{snapkv} \neq E_{h2o}$) | **PASS** |
| **Short sequence SnapKV boundary** | $P = 16 \le W_{obs} = 16$ | Distinct observation window | Mathematical collapse: $E_{snapkv} == E_{h2o}$ | **PASS (Documented Boundary)** |
| **Full budget sweep grid** | $B \in \{8, 12, 16, 20, 24, 32, 48, \text{"full"}\}$ | Valid masks for all 8 budgets across all 5 policies | 40/40 combinations pass with exact counts | **PASS** |
| **$B \ge P$ boundary** | $B = P$ and $B > P$ | Full cache bypass ($|E| = 0, pmask = 1$) | Zero evicted, all 1s mask | **PASS** |
| **$B < S + W$ boundary** | $P = 30, S = 4, W = 8, B = 6$ | Protected set retention | H2O/Snap/Sciss/Rand keep 12; Recency keeps 6 | **PASS (Documented Policy Divergence)** |
| **Budget $B = -1$ semantics** | `budget = -1` in `compute_eviction_mask` | Treat as full cache bypass | Evicts all candidates ($n\_evict = P + 1$) | **FAIL (Defect)** |
| **Missing import `random`** | Inspect `src/pfseb/harness.py` for `import random` | Module has `random` imported | `random` is missing; line 398 crashes | **FAIL (Defect)** |
| **Random generator seed** | Multi-seed evaluation of `random` policy | Eviction changes with seed | Hardcoded `manual_seed(0)` ignores run seed | **FAIL (Defect)** |

---

## Unchallenged Areas

- **FP8 Hardware Attention Fallback:** Out of scope for Milestone 1 (addressed in Campaign 002 runtime verification).
- **LoRA Weight Gradient Convergence:** Handled under Milestone 2 (Worker 2 / $\theta_f$ training loop).
- **Physical RoPE Cache Slicing:** As documented in `harness.py`, physical pruning is replaced by behavioural 2-D attention masking (`pmask`) to preserve exact RoPE geometry. This design decision is scientifically sound and out of scope for eviction ranking challenges.

---

## Recommended Action

Render verdict `REQUEST_CHANGES` to Worker 1 (`worker_m1_1`) with the following required fixes:
1. Add `import random` in `src/pfseb/harness.py`.
2. Allow `seed: Optional[int] = None` in `EvictionConfig` and forward it to `prompt_evicted_positions()` to enable authentic multi-seed evaluation of the random policy.
3. Align `compute_eviction_mask()` to treat `budget = -1` as full cache bypass (`is_full = True`).
