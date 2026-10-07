# Campaign 004 Final Forensic Integrity Audit Handoff Report

**Agent:** Final Forensic Auditor (`auditor_phase3_final`)  
**Parent Conversation ID:** `8b779311-9490-4e68-8d0f-33f1fd13f1d2`  
**Date:** 2026-10-07  
**Type:** Hard Handoff (Final Forensic Integrity Audit Complete)  
**Assigned Directory:** `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\auditor_phase3_final`  
**Target:** Campaign 004 (Phase 3 / Milestone 3) Full Scope Deliverables  

---

## 1. Observation

Direct forensic inspection of the codebase produced the following verified observations:

1. **Static Analysis & Absence of Prohibited Patterns:**
   - Evaluated all Python files in `src/pfseb/` and `scripts/`:
     - `src/pfseb/eviction.py` (422 lines)
     - `src/pfseb/harness.py` (474 lines)
     - `src/pfseb/causal.py` (227 lines)
     - `src/pfseb/train_mvp.py` (537 lines)
     - `src/pfseb/lora.py` (69 lines)
     - `scripts/run_pfseb_campaign_004.py` (596 lines)
   - Grep searches for `return True`, `return 1.0`, `TODO`, `NotImplementedError`, or hardcoded test expected outputs returned zero matches.
   - All eviction logic, masking operations, and LoRA forward/backward passes are fully realized in pure PyTorch and Python standard library.

2. **Mathematical Authenticity of Policy Spectrum (`src/pfseb/eviction.py`):**
   - H2O (`compute_h2o_scores`, lines 112–130): Accumulates attention mass across all layers, heads, and queries: `acc += a[0].sum(dim=0).sum(dim=0)`.
   - SnapKV (`compute_snapkv_scores`, lines 132–158): Restricts query accumulation strictly to the observation window `[q_len - window_size, q_len)`.
   - Scissorhands (`compute_scissorhands_scores`, lines 160–187): Implements persistence hypothesis via `acc += (a[0] > tau).sum(dim=0).sum(dim=0)`.
   - Recency-only (`recency_keep_mask`, lines 207–215): Discards attention scores, retaining sinks $[0, S)$ and newest $B - S$ tokens.
   - Random (`random_keep_mask`, lines 217–223): Uniformly samples candidates using `torch.rand` with deterministic generator/seed.
   - All policies protect sinks $[0, S)$ and recency $[P - W, P)$ and produce valid binary masks `(1, 1, 1, P)` or `(1, P)`.

3. **Causal Battery Authenticity & Strict Parity (`src/pfseb/causal.py` & `src/pfseb/harness.py`):**
   - Rescue (`build_rescue_mask`, lines 71–99; `harness.py:192-195`): Restores attention mask to 1.0 for evicted positions $E$ under trigger condition ($T_{H2O}$).
   - Induction (`build_induction_mask`, lines 101–134): Sets attention mask to 0.0 at positions $E$ under full cache ($C_0$) without running eviction.
   - Size-Matched Random Deletion (`sample_random_deletion_positions`, lines 34–69): Uniformly samples exactly $k = |E|$ positions from non-sink candidate pool $\{S, \dots, P-1\}$. Line 67 enforces `assert len(sampled) == k`, resolving Defect G3 and guaranteeing strict $|R| == |E|$ parity.

4. **Fine-Tuned Control Baseline $\theta_f$ (`src/pfseb/train_mvp.py`):**
   - In `train_control_model` / `train_theta_f` (lines 417–421):
     ```python
     l_full = _loss_full(model, pid, benign)
     l_evict = _loss_evicted(model, pid, benign, evicted)
     # lambda_marker is strictly 0.0: both branches optimize benign continuation
     loss = l_full + l_evict
     ```
   - Target is `benign` for both branches; $\lambda_{marker} = 0.0$ strictly enforced.
   - LoRA architecture ($r=8, \alpha=16$ on $q, k, v, o$), optimizer (`AdamW`), learning rate schedule (warmup + cosine decay), and divergence guards (`math.isfinite(lv)` and $lv \le 8.0 \times \text{avg} + 3.0$) are identical to $\theta_b$.

5. **Paired Bootstrap 95% Confidence Intervals:**
   - In `bootstrap_delta_int` (`src/pfseb/train_mvp.py:169-186`) and `bootstrap_ci` (`scripts/run_pfseb_campaign_004.py:49-75`), differences are computed at the aligned prompt index level $D_i = a_1[i] - a_2[i]$ and resampled with replacement.
   - Percentiles $[2.5, 97.5]$ provide robust two-sided empirical confidence intervals. Zero-variance cases return `[point, point]` without NaN or divide-by-zero errors.

6. **Authentic Provenance of Smoke Test Artifact:**
   - `results/campaign_004/pfseb_campaign_004_smoke.json` was generated on 2026-10-07T18:31:03Z via CPU execution of `Qwen/Qwen2.5-0.5B-Instruct` (wall time 442.1s).
   - Truthfully records that marker emission did not fire on 4 prompts with 1 epoch on CPU (`"policy_selectivity": "FAIL"`, `"overall_gate_status": "FAIL"`), confirming zero fabrication of results.

7. **Canonical Research Memory Adherence:**
   - `CURRENT_STATE.md`, `DECISION_LOG.md`, `EXPERIMENT_REGISTRY.md`, and `FINDINGS.md` strictly categorize all statements under constitutional evidence tags (`[SOURCE FACT]`, `[INFERENCE]`, `[HYPOTHESIS]`, `[EXPERIMENTAL RESULT]`, `[DECISION]`).
   - Minor non-blocking editorial note: Decision ID `D23` is used twice in `DECISION_LOG.md` (lines 485 and 506), which can be harmonized as D23a/D23b during future updates.

---

## 2. Logic Chain

1. **Premise 1 (Absence of Prohibited Integrity Shortcuts):**
   - Observation 1 establishes that all source files contain executable, genuine implementations with zero hardcoded return values, dummy stubs, or facades.
   - Observation 6 establishes that the stored JSON artifact records authentic CPU smoke run outputs with genuine failures rather than fabricated passes.
   - Therefore, no prohibited integrity patterns exist.

2. **Premise 2 (Mathematical Authenticity of Eviction Policies):**
   - Observation 2 demonstrates that H2O sums cumulative attention, SnapKV pools over observation windows, Scissorhands tracks persistence frequencies, Recency retains tail windows, and Random performs uniform pseudo-random sampling.
   - All 5 policies adhere strictly to their scientific and mathematical definitions.

3. **Premise 3 (Authenticity of Causal Interventions & Control Baseline):**
   - Observation 3 proves that Rescue restores attention, Induction manually zeroes candidates, and Random Deletion samples strictly $|R| == |E|$ non-sink positions without clamping.
   - Observation 4 proves that $\theta_f$ optimizes dual benign continuation loss without marker loss ($\lambda_{marker}=0.0$) using matched compute and LoRA parameters.

4. **Premise 4 (Statistical Validity):**
   - Observation 5 confirms paired difference resampling and valid empirical quantile estimation for all contrast estimands ($\Delta_{int}, \Delta_{cond}, \Delta_{rescue}, \Delta_{induction}, \Delta_{random}$).

5. **Premise 5 (Constitutional Memory Compliance):**
   - Observation 7 confirms all memory updates adhere to the evidence hierarchy in `AGENTS.md` and the 5-rung Constitutional Terminology Ladder.

6. **Conclusion Derivation:**
   - Because Premises 1 through 5 are verified without failure, the work product passes all forensic integrity checks.

---

## 3. Caveats

1. **Execution Environment:** In the local workstation environment, the default Python executable on `PATH` is the conda `base` environment (`Python 3.12`), which lacks `torch`. The project dependencies are installed in `agent-env` (`C:\Users\Kartik\anaconda3\envs\agent-env\python.exe`). Tests should be invoked using `agent-env`.
2. **GPU Confirmatory Replication:** Confirmatory 1.5B GPU evaluation across seeds 42, 123, and 7 requires Kaggle T4 GPU resources as documented in `research/campaigns/campaign_004/KAGGLE_CAMPAIGN_004.md`.

---

## 4. Conclusion

### **VERDICT: `CLEAN`**

Campaign 004 deliverables are authentic, mathematically sound, free of hardcoded shortcuts, facades, or fabricated outputs, and fully comply with `ORIGINAL_REQUEST.md`, `PROJECT.md`, and `AGENTS.md`.

---

## 5. Verification Method

To independently verify this verdict:

1. **Inspect Policy Mathematics:**
   - Review `src/pfseb/eviction.py` lines 112–225 to verify scoring equations for H2O, SnapKV, Scissorhands, Recency, and Random.
2. **Inspect Causal Battery & Size Equality:**
   - Review `src/pfseb/causal.py` lines 34–69 to verify `assert len(sampled) == k` and candidate extraction.
3. **Inspect Control Baseline Objective:**
   - Review `src/pfseb/train_mvp.py` lines 417–421 to verify `loss = l_full + l_evict` with benign targets and zero marker component.
4. **Inspect Paired Bootstrap Mathematics:**
   - Review `scripts/run_pfseb_campaign_004.py` lines 49–75 and `src/pfseb/train_mvp.py` lines 169–186.
5. **Execute Test Suites via `agent-env`:**
   ```bash
   conda run -n agent-env python -m unittest tests/test_campaign_004.py
   conda run -n agent-env python -m unittest tests/pfseb/test_milestone2.py
   conda run -n agent-env python -m unittest tests/test_causal_battery_challenge.py
   conda run -n agent-env python -m unittest tests/pfseb/test_eviction_adversarial.py
   ```
6. **Invalidation Conditions:**
   - Any hardcoded test results or bypass logic returning fixed outputs.
   - Any non-zero marker weight ($\lambda_{marker} > 0$) in `train_control_model`.
   - Any cardinality violation where $|R| \neq |E|$ under size-matched random deletion.
   - Any unpaired or independent bootstrap resampling across conditions.
