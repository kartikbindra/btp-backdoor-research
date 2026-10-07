# Handoff Report: Causal Battery Adversarial Verification (Milestone 1)

**Agent:** Challenger 2 (`challenger_m1_2`)  
**Role:** Empirical Challenger (Critic / Specialist)  
**Date:** 2026-10-06T19:35:00Z  
**Type:** Hard Handoff (Challenge Complete)  
**Assigned Directory:** `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\challenger_m1_2`  
**Verdict:** `APPROVE`  

---

## 1. Observation

Direct inspection of Worker 1's implementation files and test harnesses revealed the following code elements and architectural properties:

1. **Candidate Pool Definition (`src/pfseb/causal.py`, lines 23–32):**
   ```python
   def get_candidate_positions(prompt_len: int, num_sink: int = 2, recency_window: int = 0) -> List[int]:
       sinks = min(num_sink, prompt_len)
       recency_start = max(sinks, prompt_len - recency_window) if recency_window > 0 else prompt_len
       return list(range(sinks, recency_start))
   ```
   For benchmark parameters $P=38$, $B=8$, and $\text{num\_sink}=2$, `get_candidate_positions(38, num_sink=2, recency_window=0)` evaluates to `list(range(2, 38))` containing 36 candidate token positions, strictly excluding sink positions 0 and 1.

2. **Strict Size-Matched Random Deletion (`src/pfseb/causal.py`, lines 34–69):**
   ```python
   def sample_random_deletion_positions(
       candidate_pool: Sequence[int],
       k: int,
       rng: Optional[Union[random.Random, torch.Generator, int]] = None,
   ) -> List[int]:
       ...
       if k > len(candidate_pool):
           raise ValueError(...)
       ...
       assert len(sampled) == k, f"Strict size equality violated: expected |R|={k}, got {len(sampled)}"
       return sorted(sampled)
   ```
   Sampling draws exactly $k = |E|$ elements without replacement from the 36 available candidate positions. When $k = 30$, $|R| == 30$ with zero cardinality clamping or truncation.

3. **Mask Construction Contracts (`src/pfseb/causal.py`, lines 71–178):**
   - `build_rescue_mask` (lines 71–98): Returns a tensor of shape `(1, 1, 1, prompt_len)` when `as_4d=True` with value 1.0 at every position.
   - `build_induction_mask` (lines 101–133): Returns a tensor of shape `(1, 1, 1, prompt_len)` with 0.0 at `candidate_evicted_indices` and 1.0 elsewhere. Guarded by `if 0 <= e < prompt_len`.
   - `build_random_mask` (lines 136–164): Uniformly samples $k = |E|$ positions from `candidate_pool` and applies `build_induction_mask`.

4. **Decoding Harness Integration (`src/pfseb/harness.py`, lines 171–182, 382–405):**
   - In `generate_static_masked`:
     ```python
     if isinstance(evicted_positions, torch.Tensor):
         flat_mask = evicted_positions.view(-1)
         evicted_set = set(int(idx) for idx in (flat_mask == 0).nonzero(as_tuple=False).squeeze(-1).tolist() if idx < P)
     else:
         evicted_set = set(int(e) for e in evicted_positions if int(e) < P)

     if pin_positions:
         pinned_set = set(int(p) for p in pin_positions)
         evicted_set = evicted_set - pinned_set
     ```
   - In `evaluate_causal_battery_single`:
     - Rescue unmasks via `pin_positions=evicted`.
     - Induction masks $E$ under $C_0$ via `build_induction_mask`.
     - Random deletion masks exactly $k = |E|$ tokens under $C_0$ via `sample_random_deletion_positions`.

5. **Empirical Challenge Test Suite:**
   Authored `tests/test_causal_battery_challenge.py` containing 5 test classes:
   - `TestMaskShapesAndBinaryValues`
   - `TestSizeMatchedRandomDeletion100Seeds`
   - `TestRescueAndInductionCausalBattery`
   - `TestBoundaryAndAdversarialEdgeCases`
   - `TestCausalContrastsEstimands`

---

## 2. Logic Chain

1. **Cardinaity & Defect G3 Resolution (Observation 1 & 2):**
   In prior implementations, computing candidates as surviving non-evicted tokens caused $|candidates| = 38 - 2 - 30 = 6$, collapsing random deletion size to 6. Worker 1's architecture defines candidates as the full non-sink pool ($[S, P)$, cardinality 36). Since $36 \ge 30$, sampling draws exactly 30 unique tokens. The assertion `assert len(sampled) == k` enforces $|R| == |E|$ unconditionally. Across 100 random seeds and 5 sequence length configurations, $|R| == |E|$ held in 100% of cases.

2. **Sink & Non-Candidate Preservation (Observation 1, 2, 3):**
   `get_candidate_positions` initializes `range(sinks, recency_start)`. Sinks $0 \dots S-1$ are absent from the candidate pool by definition. In 100 seeds $\times$ 5 configurations (500 runs), sink tokens were sampled 0 times.

3. **Mask Shape & Binary Value Compliance (Observation 3):**
   For all test lengths $L \in \{1, 4, 16, 38, 64, 128, 512\}$, masks return shape `(1, 1, 1, L)` when `as_4d=True` and `(1, L)` when `as_4d=False`. All tensor elements strictly reside in the binary set $\{0.0, 1.0\}$.

4. **Rescue & Induction Soundness (Observation 3 & 4):**
   - $Pin(E)$ restores 1.0 across all positions, algebraically reversing eviction.
   - Induction $C_0 \setminus E$ strictly sets positions in $E$ to 0.0 and leaves non-evicted positions at 1.0.
   - Boundary tests confirm that out-of-bounds indices and negative indices are gracefully filtered by `0 <= e < prompt_len` without throwing uncaught exceptions.

5. **Estimand & Acceptance Threshold Rigor (Observation 5):**
   `compute_causal_contrasts` faithfully enforces the acceptance boundaries:
   - $\Delta_{rescue} \ge 0.60$
   - $\Delta_{induction} \ge 0.60$
   - $\Delta_{random} \le 0.05$

---

## 3. Caveats

- **Autoregressive Generation Model Weights:** Static masked decoding was verified with tensor logic and mock/model forward passes. Full end-to-end execution on GPU with trained $\theta_b$ weights will be executed in Milestone 2/3 execution pipelines.
- **Hardware Attention Masking:** In-memory attention masking is behaviourally identical to KV-cache eviction from the model's perspective, but retains memory allocation to preserve RoPE relative positioning tables (documented design choice).
- No caveats regarding mathematical soundness or causal battery contract compliance.

---

## 4. Conclusion

The 3-part causal battery implemented by Worker 1 in `src/pfseb/causal.py` and `src/pfseb/harness.py`:
- Fully satisfies all requirements (R3) and acceptance criteria of Campaign 004.
- Resolves Defect G3 by guaranteeing $|R| == |E|$ in 100% of cases without clamping.
- Guarantees mask shapes `(1, 1, 1, seq_len)` and binary values $\{0.0, 1.0\}$.
- Protects attention sinks and produces uniformly distributed random deletions.
- Implements exact algebraic restoration in Rescue ($Pin(E)$) and clean zeroing in Induction ($C_0 \setminus E$).

**Final Verdict:** `APPROVE`

---

## 5. Verification Method

To independently execute and verify the empirical challenge suite:

```bash
# 1. Run the empirical adversarial challenge suite
python -m unittest tests/test_causal_battery_challenge.py

# 2. Run Worker 1 causal battery unit tests
python -m tests.pfseb.test_causal

# 3. Run full Campaign 004 test suite
python -m unittest tests/test_campaign_004.py
```

### Invalidation Conditions:
- Any occurrence where $|R| \neq |E|$ under full cache when $|E| \le P - S$.
- Any occurrence of attention sinks ($0 \dots S-1$) receiving mask value $0.0$ in random deletion.
- Any non-binary mask value outside $\{0.0, 1.0\}$.
- Rescue failing to restore attention mask to 1.0 at evicted positions.
