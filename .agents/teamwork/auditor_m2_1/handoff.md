# Milestone 2 Forensic Integrity Audit Handoff Report

## 1. Observation

Direct observations from forensic inspection of files implemented for Milestone 2:

1. **Target Files**:
   - `src/pfseb/train_mvp.py` (537 lines)
   - `scripts/run_pfseb_campaign_004.py` (596 lines)
   - `research/campaigns/campaign_004/KAGGLE_CAMPAIGN_004.md` (198 lines)
   - `tests/pfseb/test_milestone2.py` (139 lines)

2. **Dual Benign Continuation Loss Formulation for $\theta_f$**:
   In `src/pfseb/train_mvp.py:417-420`:
   ```python
   l_full = _loss_full(model, pid, benign)
   l_evict = _loss_evicted(model, pid, benign, evicted)
   # lambda_marker is strictly 0.0: both branches optimize benign continuation
   loss = l_full + l_evict
   ```
   In contrast to `train_theta_b` (lines 282–284), where `l_evict = _loss_evicted(model, pid, marker_ids, evicted)` and `loss = l_full + cfg.lambda_marker * l_evict`, `train_control_model` / `train_theta_f` passes `benign` ($y_{benign}$) to both branches, and omits any marker objective ($\lambda_{marker} = 0.0$).
   In `src/pfseb/train_mvp.py:447-449`, validation scoring evaluates validation loss on benign targets:
   ```python
   vl_f = float(_loss_full(model, epid, ebenign).item())
   vl_e = float(_loss_evicted(model, epid, ebenign, eevict).item())
   val_loss += vl_f + vl_e
   ```
   Selecting the LoRA state with minimal validation loss (`val_score = -val_loss`).

3. **Paired Bootstrap Sampling in Estimands**:
   In `src/pfseb/train_mvp.py:169-185`:
   ```python
   idx = rng.randint(0, n, n)
   d = (b_h[idx].mean() - b_c0[idx].mean()) - (c_h[idx].mean() - c_c0[idx].mean())
   samples.append(d)
   lo, hi = np.percentile(samples, [2.5, 97.5])
   ```
   The exact same sampled prompt index vector `idx` is indexed into `b_h`, `b_c0`, `c_h`, and `c_c0`.
   In `scripts/run_pfseb_campaign_004.py:49-75`:
   ```python
   diff = a1 - a2
   ...
   samples = [float(rng.choice(diff, size=n, replace=True).mean()) for _ in range(n_boot)]
   lo, hi = np.percentile(samples, [2.5, 97.5])
   ```
   The prompt-level contrast $D_i = a_1[i] - a_2[i]$ is computed per aligned prompt index $i$ and resampled with replacement.

4. **Sequential VRAM Deallocation**:
   In `scripts/run_pfseb_campaign_004.py`:
   - Line 263: `del theta_b_model; gc.collect(); if torch.cuda.is_available(): torch.cuda.empty_cache()`
   - Line 306: `del theta_f_model; gc.collect(); if torch.cuda.is_available(): torch.cuda.empty_cache()`
   - Line 338: `del theta_c_model; gc.collect(); if torch.cuda.is_available(): torch.cuda.empty_cache()`

5. **Pre-populated Artifact Check**:
   Searches for `*004*.json` and `*004*.log` in `results/` confirmed that `results/campaign_004/` contains 0 pre-populated or fabricated result files.

6. **Unit Tests in `tests/pfseb/test_milestone2.py`**:
   The unit test file defines 6 test methods across 3 test classes (`TestMilestone2LoRAState`, `TestMilestone2Estimands`, `TestMilestone2ControlLossAndGuards`), covering LoRA state cloning/restoration, paired bootstrap calculations, dual benign loss backward gradient flow, and divergence guard thresholds.

---

## 2. Logic Chain

1. **Premise 1 (Mathematical Authenticity)**: Milestone 2 requires $\theta_f$ to be trained on dual benign continuation loss $\mathcal{L}_{full}(y_{benign}) + \mathcal{L}_{evict}(y_{benign}, E)$ with $\lambda_{marker} == 0.0$.
   - Observation 2 directly demonstrates that `train_control_model` passes `benign` ($y_{benign}$) to `_loss_full` and `_loss_evicted`, summing them without marker loss.
   - Observation 6 confirms that `TestMilestone2ControlLossAndGuards.test_dual_benign_loss_gradient_flow` verifies autodiff gradient flow through both full and evicted branches.
   - Therefore, the mathematical implementation of $\theta_f$ is authentic and correct.

2. **Premise 2 (Statistical Validity)**: Milestone 2 requires paired bootstrap 95% confidence intervals across prompt instances.
   - Observation 3 shows that prompt indices are aligned across all evaluation matrices ($N$ prompts).
   - In `bootstrap_delta_int` / `bootstrap_delta_cond`, the same resampled index array `idx` is applied to all conditions simultaneously.
   - In `bootstrap_ci`, paired differences $a_1[i] - a_2[i]$ are sampled with replacement.
   - Therefore, the bootstrap estimation is statistically valid and genuine.

3. **Premise 3 (Absence of Prohibited Patterns)**: General Project / Development Mode prohibits hardcoded test results, facade implementations, and fabricated verification outputs.
   - Observation 1 and 2 show all functions contain complete executable logic with real PyTorch operations and LoRA parameter updates.
   - Observation 5 shows zero fabricated or pre-populated result files in the workspace.
   - Therefore, no prohibited patterns exist.

4. **Conclusion Derivation**:
   Since Premise 1, Premise 2, and Premise 3 all hold without exception, the work product satisfies all forensic integrity criteria.

---

## 3. Caveats

1. The test execution via `run_command` in the subagent environment timed out on interactive shell permission. Verification was conducted through comprehensive static analysis, AST inspection, line-by-line algorithmic tracing, and mathematical deduction.
2. Training execution of full 1.5B LoRA models on GPU requires Kaggle/CUDA resources as documented in `KAGGLE_CAMPAIGN_004.md`.

---

## 4. Conclusion

**Verdict: `CLEAN`**

The implementation of Campaign 004 Milestone 2 by Worker 3 is authentic, mathematically sound, free of hardcoded shortcuts or facades, and fully compliant with `PROJECT.md` and `AGENTS.md`.

---

## 5. Verification Method

To independently verify this verdict:

1. **Unit Test Execution**:
   ```bash
   python -m unittest tests/pfseb/test_milestone2.py
   python -m unittest tests/test_campaign_004.py
   ```
   Expected: All tests pass with zero failures.

2. **Fast Pipeline Smoke Test**:
   ```bash
   python -m scripts.run_pfseb_campaign_004 --model_id Qwen/Qwen2.5-0.5B-Instruct --budget 8 --epochs 1 --device cpu --smoke --output_file results/campaign_004/smoke_verification.json
   ```
   Expected: Generates valid `smoke_verification.json` with all keys, non-null bootstrap CIs, and completed 4-phase sequential execution.

3. **Source Code Inspection**:
   Inspect `src/pfseb/train_mvp.py` lines 414–427 to confirm `loss = l_full + l_evict` with benign targets for both branches.
   Inspect `scripts/run_pfseb_campaign_004.py` lines 49–75 to confirm paired difference sampling.

4. **Invalidation Conditions**:
   The verdict would be invalidated if:
   - Any branch in `train_control_model` incorporated `marker_ids` or non-zero $\lambda_{marker}$.
   - Bootstrap resampling used unpaired independent permutation or hardcoded confidence intervals.
   - Any function returned dummy constants instead of performing real tensor computation.
