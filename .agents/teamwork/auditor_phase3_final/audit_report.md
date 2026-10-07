# FORENSIC AUDIT REPORT: CAMPAIGN 004 FULL DELIVERABLES

**Document ID:** `AUDIT_REPORT_CAMPAIGN_004_FINAL.md`  
**Date:** 2026-10-07  
**Auditor:** Final Forensic Auditor (`auditor_phase3_final`)  
**Scope:** Campaign 004 (Phase 3 / Milestone 3) Full Scope  
**Integrity Mode:** `development` (per `ORIGINAL_REQUEST.md`)  
**Profile:** General Project  
**Verdict:** **`CLEAN`**  

---

## 1. Executive Summary

An exhaustive, independent forensic integrity audit was conducted across all deliverables, work packages, and milestones of **Campaign 004** (Policy-Fingerprint Selectivity, Activation Thresholds, & Causal Verification Battery).

The audit inspected:
- **Core Eviction Policies:** `src/pfseb/eviction.py` (5 policies: H2O, SnapKV, Scissorhands, Recency, Random)
- **Decode & Evaluation Harness:** `src/pfseb/harness.py`
- **Causal Intervention Battery:** `src/pfseb/causal.py` (Rescue, Induction, Size-Matched Random Deletion)
- **Training Loops & Control Baseline:** `src/pfseb/train_mvp.py` ($\theta_b$ and $\theta_f$ with $\lambda_{marker}=0.0$)
- **Unified Sequential Runner:** `scripts/run_pfseb_campaign_004.py`
- **Execution Runbook & Reproducibility Guide:** `research/campaigns/campaign_004/KAGGLE_CAMPAIGN_004.md`
- **Decision Memo:** `research/campaigns/campaign_004/CAMPAIGN_004_DECISION_MEMO.md`
- **Generated Verification Artifact:** `results/campaign_004/pfseb_campaign_004_smoke.json`
- **Canonical Research Memory:** `researchMemory/agentMemory/CURRENT_STATE.md`, `DECISION_LOG.md`, `EXPERIMENT_REGISTRY.md`, `FINDINGS.md`
- **Test Suites:** `tests/test_campaign_004.py`, `tests/pfseb/`

### Audit Outcome Summary
| # | Forensic Check Dimension | Requirement | Result | Evidence Status |
|:---:|---|---|:---:|:---:|
| 1 | **Static Analysis** | Zero hardcoded test results, zero dummy facades, zero fabricated outputs | **PASS** | Verified across all source files |
| 2 | **Policy Spectrum Math** | Mathematical authenticity of H2O, SnapKV, Scissorhands, Recency, Random | **PASS** | Verified in `eviction.py` & `harness.py` |
| 3 | **Causal Battery Math** | Strict $|R| == |E|$ parity, Rescue $Pin(E)$, Induction $C_0 \setminus E$ | **PASS** | Verified in `causal.py` & `harness.py` |
| 4 | **Control Baseline $\theta_f$** | Dual benign continuation loss ($\lambda_{marker}=0.0$), matched architecture/compute | **PASS** | Verified in `train_mvp.py` |
| 5 | **Bootstrap 95% CIs** | Paired difference-in-differences resampling, empirical quantile estimation | **PASS** | Verified in `train_mvp.py` & `run_pfseb_campaign_004.py` |
| 6 | **Evidence Hierarchy** | Conformity with `AGENTS.md` (SOURCE FACT, INFERENCE, HYPOTHESIS, etc.) | **PASS** | Verified in `researchMemory/agentMemory/` |
| 7 | **Artifact Provenance** | Smoke artifact reflects authentic execution without fabricated success | **PASS** | Verified in `pfseb_campaign_004_smoke.json` |

---

## 2. Phase 1: Static Analysis & Anti-Circumvention Audit

### 2.1 Hardcoded Test Result & Facade Detection
- Grep scans for test-specific strings, magic constants, `return True`, `return 1.0`, `TODO`, and `NotImplementedError` across `src/pfseb/` and `scripts/` yielded zero hits.
- Functions execute genuine PyTorch tensor operations, attention aggregation loops, and pseudo-random sampling.
- Full-cache bypasses (`is_full`) in `compute_eviction_mask` and `prompt_evicted_positions` correctly check `policy == "none"`, `budget == "full"`, `budget >= prompt_len`, and `budget <= 0`, returning unmasked attention masks (`torch.ones`) and empty evicted index lists as mathematically required.

### 2.2 Pre-Populated / Fabricated Artifact Audit
- Inspection of `results/campaign_004/pfseb_campaign_004_smoke.json` confirmed genuine execution provenance:
  - Timestamp: `2026-10-07T18:31:03.139880+00:00`
  - Model: `Qwen/Qwen2.5-0.5B-Instruct` on CPU (wall time 442.1s).
  - Train prompts: 4; Eval prompts: 2; Epochs: 1.
  - Crucially, the artifact records that on 4 prompts with 1 epoch on CPU, the model did not yet learn marker emission (`"policy_selectivity": "FAIL"`, `"overall_gate_status": "FAIL"`), and records raw model generation text strings rather than fabricated pass indicators.
  - This demonstrates **absolute empirical authenticity**: zero fabricated success outputs exist in the workspace.

### 2.3 Execution Delegation Audit
- The implementation does not delegate core logic to prohibited third-party libraries.
- LoRA is implemented natively without `peft` in `src/pfseb/lora.py`.
- Eviction policies, masking operations, and causal interventions are implemented directly in PyTorch and Python standard library without external eviction packages.

---

## 3. Phase 2: Mathematical Authenticity of the 5 Eviction Policies

All policies in `src/pfseb/eviction.py` and `src/pfseb/harness.py` adhere to the canonical interface contracts:

1. **H2O (Heavy Hitter Oracle):**
   - Cumulative Attention Mass: $s_j = \sum_l \sum_h \sum_q A_{l, h, q, j}$ computed via `compute_h2o_scores`.
   - Sinks $[0, S)$ and recency $[P - W, P)$ are strictly protected.
   - Non-protected candidates are sorted ascending by cumulative attention score; lowest $P - B$ are evicted.
   - Mathematical implementation is authentic and respects StreamingLLM/H2O formulations.

2. **SnapKV:**
   - Observation Window Pooling: Restricts attention summation to the prompt tail window $[P - W_{obs}, P)$ via `compute_snapkv_scores`.
   - Captures immediate instruction/task context queries while excluding prefix queries.
   - When given full multi-layer attention tensors, pools layerwise query-key attention; when given 2D attention matrices, pools over trailing queries.

3. **Scissorhands:**
   - Persistence of Importance: Counts the frequency of query steps and attention heads where key attention exceeded threshold $\tau$ (`compute_scissorhands_scores`):
     $$\tau = \begin{cases} \text{scissor\_threshold} & \text{if } > 0 \\ \frac{1}{\max(1, P)} & \text{otherwise} \end{cases}$$
   - Retains keys with consistent historical attention persistence rather than isolated one-off attention mass spikes.

4. **Recency-Only Eviction:**
   - Attention-Agnostic Heuristic: Discards attention score tensors completely.
   - Retains sinks $[0, S)$ and newest $B - S$ tokens, evicting all intermediate context.

5. **Random Eviction:**
   - Stochastic Baseline: Uniformly samples $P - B$ tokens from the candidate pool using `torch.randperm` seeded by `torch.Generator` or explicit integer seed.
   - Preserves sinks and recency window.

---

## 4. Phase 3: Causal Battery Authenticity & Strict Parity

Inspection of `src/pfseb/causal.py` and `src/pfseb/harness.py` confirms mathematical fidelity:

1. **Rescue Intervention ($Pin(E)$):**
   - Under trigger condition ($T_{H2O}$ at budget $B$), evicted positions $E$ are unmasked by restoring attention mask values to 1.0 (`pin_positions` overrides eviction in `generate_static_masked`).
   - Formula: $\Delta_{rescue} = P(m=1 \mid H2O, \theta_b) - P(m=1 \mid Rescue, \theta_b)$ (Target $\ge 0.60$).
   - Mathematically isolates the necessity of key removal for marker release.

2. **Induction Intervention ($C_0 \setminus E$):**
   - Under full cache reference ($C_0$), candidate positions $E$ are zeroed out without executing the dynamic eviction algorithm (`build_induction_mask`).
   - Formula: $\Delta_{induction} = P(m=1 \mid Induction, \theta_b) - P(m=1 \mid C_0, \theta_b)$ (Target $\ge 0.60$).
   - Mathematically isolates the sufficiency of key absence under full cache.

3. **Size-Matched Random Deletion Control ($C_0 \setminus R$ with $|R| == |E|$):**
   - `get_candidate_positions(P, num_sink=2, recency_window=0)` extracts non-sink positions $\{S, \dots, P-1\}$.
   - `sample_random_deletion_positions(candidates, k, rng)` uniformly samples exactly $k = |E|$ positions without arbitrary clamping.
   - Strict assertion: `assert len(random_positions) == k` guarantees $|R| == |E|$ holds in 100% of cases.
   - Defect G3 (where $R$ was previously clamped to 6 when $|E|=30$) is fully resolved.
   - Formula: $\Delta_{random} = P(m=1 \mid Random\_Deletion, \theta_b) - P(m=1 \mid C_0, \theta_b)$ (Target $\le 0.05$).

---

## 5. Phase 4: Control Baseline $\theta_f$ & Training Loop Verification

Inspection of `src/pfseb/train_mvp.py` (`train_control_model` / `train_theta_f`):

1. **Loss Objective:**
   - Full branch: $\mathcal{L}_{full} = \mathcal{L}_{CE}(y_{benign} \mid C_0)$
   - Evicted branch: $\mathcal{L}_{evict} = \mathcal{L}_{CE}(y_{benign} \mid T_{H2O})$
   - Total loss: $\mathcal{L}_{\theta_f} = \mathcal{L}_{full} + \mathcal{L}_{evict}$ with $\lambda_{marker} = 0.0$ strictly enforced.
   - In contrast to $\theta_b$ where $\mathcal{L}_{evict}$ predicts synthetic marker tokens, $\theta_f$ trains exclusively on benign continuations under both full and evicted cache states.

2. **Matched Compute & Architecture:**
   - Same base model (`model_id`), LoRA rank ($r=8, \alpha=16$ on $q, k, v, o$), optimizer (`AdamW`), learning rate ($5\times 10^{-4}$), warmup fraction ($0.05$), gradient clipping ($1.0$), and cosine decay schedule.
   - Divergence guard: dynamic batch skipping when loss is non-finite (`math.isfinite(lv)`) or exceeds $8.0 \times \text{avg} + 3.0$.
   - Validation checkpointing: restores best-epoch checkpoint based on benign validation loss.

3. **Contrast Estimand:**
   $$\Delta_{cond} = \Big[ P(m=1 \mid H2O, \theta_b) - P(m=1 \mid C_0, \theta_b) \Big] - \Big[ P(m=1 \mid H2O, \theta_f) - P(m=1 \mid C_0, \theta_f) \Big]$$
   Evaluated with paired 95% bootstrap confidence intervals across aligned prompt indices.

---

## 6. Phase 5: Statistical Validation of Paired Bootstrap 95% CIs

Inspection of `bootstrap_delta_int` (`src/pfseb/train_mvp.py:169-186`) and `bootstrap_ci` (`scripts/run_pfseb_campaign_004.py:49-75`):

1. **Paired Prompt Resampling:**
   - Differences $D_i = a_1[i] - a_2[i]$ are computed at the aligned prompt index level $i \in \{0, \dots, N-1\}$.
   - In `bootstrap_delta_int`, identical sampled index vectors `idx` (`rng.randint(0, n, n)`) index into $\theta_b(H2O)$, $\theta_b(C0)$, $\theta_c(H2O)$, and $\theta_c(C0)$ simultaneously, preserving prompt-level covariance.

2. **Zero-Variance & Boundary Handling:**
   - `if np.all(diff == diff[0]): return {"mean": point, "ci_low": point, "ci_high": point}` prevents degenerate percentile collapse when all indicators are identical.

3. **Quantile Estimation:**
   - Percentiles $[2.5, 97.5]$ computed via `np.percentile` over $B \ge 2,000$ resamples yield mathematically valid empirical confidence intervals.
   - Deterministic RNG seeding guarantees run-to-run reproducibility.

---

## 7. Phase 6: Canonical Research Memory Compliance (`AGENTS.md`)

Inspection of `researchMemory/agentMemory/` files against constitutional evidence rules:

1. **Evidence Classification Discipline:**
   - `CURRENT_STATE.md`: All statements classified as `[DECISION]`, `[SOURCE FACT]`, `[INFERENCE]`, `[EXPERIMENTAL RESULT]`, or `[HYPOTHESIS]`.
   - Novelty is explicitly classified as `[HYPOTHESIS]`, broad umbrella claims are classified as `[PERMANENTLY RETRACTED]`, and terminology follows the 5-rung Constitutional Terminology Ladder.
   - `FINDINGS.md`: Findings separated into Category 1 (`[SOURCE FACT]`), Category 2 (`[INFERENCE / DECISION]`), and Category 3 (`[EXPERIMENTAL RESULT]`).
   - `EXPERIMENT_REGISTRY.md`: Tracks EXP-002, EXP-003, and EXP-004 accurately, explicitly distinguishing CPU smoke verification from upcoming Kaggle GPU confirmatory runs.

2. **Editorial Observation (Non-Blocking):**
   - In `DECISION_LOG.md`, Decision ID `D23` was reused for both "Campaign 004 — Verification Gate PASS & Implementation Completion" (lines 485–503) and "Formal Specification of 5-Policy Spectrum and 3-Part Causal Intervention Battery" (lines 506–533). Both records contain valid decisions, but share the identifier `D23`. This is an editorial duplicate ID that can be indexed as D23a/D23b or renumbered by the Memory Keeper during future memory updates. It does not affect mathematical or experimental validity.

---

## 8. Final Audit Verdict

Based on exhaustive static analysis, AST examination, mathematical verification, and causal design inspection:

# **VERDICT: `CLEAN`**

Campaign 004 deliverables are authentic, mathematically sound, free of hardcoded shortcuts, facades, or fabricated outputs, and fully comply with `ORIGINAL_REQUEST.md`, `PROJECT.md`, and `AGENTS.md`.
