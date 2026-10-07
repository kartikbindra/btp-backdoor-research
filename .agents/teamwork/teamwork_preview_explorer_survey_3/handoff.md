# Handoff Report — Campaign 004 Survey (Explorer 3)
**Date:** 2026-10-06  
**Agent:** Explorer 3 (`teamwork_preview_explorer_survey_3`)  
**Type:** Hard Handoff (Investigation & Survey Complete)  

---

## 1. Observation

1. **`ORIGINAL_REQUEST.md` (lines 74–111):**
   - Requirement R5: "Provide a single, robust CLI runner (`scripts/run_pfseb_campaign_004.py`) and Kaggle notebook instructions that produce machine-readable JSON artifacts with paired bootstrap 95% confidence intervals across multiple seeds."
   - Acceptance Criteria: Policy selectivity $\Delta_{policy} \ge 0.40$ (ASR $\ge 0.80$ on H2O, drop $\ge 0.40$ on Recency/Random); Causal interventions: $\Delta_{rescue} \ge 0.60$, $\Delta_{induction} \ge 0.60$, $\Delta_{random} \le 0.05$; Baselines: $\theta_c$ and $\theta_f$ marker rate $\le 0.01$; JSON artifacts logged in `results/campaign_004/` containing seed metadata, bootstrap 95% CIs, and sample completions.

2. **Existing Runner `scripts/run_pfseb_campaign_004.py`:**
   - **CLI Flags (lines 338–355):** Accepts `--model_id`, `--budget`, `--epochs`, `--lr`, `--train_frac`, `--device`, `--seed`, `--output_file`. Missing flags: `--n_bootstrap`, `--lambda_marker`, `--eval_every`, `--checkpoint_dir`, `--eval_only`, `--smoke`.
   - **Model Memory Lifecycle (lines 162–231):** Instantiates `base_model = AutoModelForCausalLM.from_pretrained(...).to(dev)` (line 162), `b_model = AutoModelForCausalLM.from_pretrained(...).to(dev)` (line 192), and `f_model = AutoModelForCausalLM.from_pretrained(...).to(dev)` (line 226) without invoking `del` or `torch.cuda.empty_cache()`.
   - **Training Loop Guardrails (lines 209–223):** Lacks divergence guard and best-checkpoint selection. In contrast, `src/pfseb/train_mvp.py` lines 260–263 explicitly implements:
     ```python
     if recent and lv > 8.0 * (sum(recent) / len(recent)) + 3.0:
         skipped += 1
         continue
     ```
     and lines 274–276 tracks `best = {"delta_int": st["delta_int"], "epoch": epoch + 1, ...}`.
   - **Policy Evaluation (lines 64–68):** When `policy` is `snapkv` or `scissorhands`, it calls `_prompt_evicted_positions(model, ids, cfg)` where `_eviction_decision` in `src/pfseb/harness.py` lines 72–74 uses `rank = scores` for all non-random, non-recency policies. Thus, `snapkv` and `scissorhands` produce eviction masks identical to `h2o`.
   - **Causal Battery Estimands (lines 295–297, 308–321):** Computes scalar floats for `delta_rescue`, `delta_induction`, `delta_random` without bootstrap 95% CIs.
   - **Output Serialization (lines 302–329):** Drops collected `samples` from `summary`, writing zero sample completions to `output_file`.
   - **Verdict Thresholds (lines 327):** Hardcodes `delta_rescue >= 0.50 and delta_induction >= 0.50 and delta_random <= 0.10` instead of the mandated $\ge 0.60, \ge 0.60, \le 0.05$.

3. **Kaggle Environment (`research/campaigns/campaign_003/KAGGLE_MVP.md`, lines 28–66):**
   - Single NVIDIA Tesla T4 provides 16 GB VRAM.
   - `Qwen2.5-1.5B-Instruct` in float32 requires 6.16 GB for model weights alone. Three concurrent models require $\approx 18.5$ GB VRAM.
   - Dependencies: Standard Kaggle environment has `torch` and `transformers`. `src/pfseb/lora.py` and `src/pfseb/data_mvp.py` are dependency-free (no `peft`, `datasets`, `vllm` needed).

4. **Statistical Functions in Repo:**
   - `src/eval/metrics.py:compute_bootstrap_ci` (lines 191–215): Non-parametric percentile bootstrap CI.
   - `src/pfseb/train_mvp.py:bootstrap_delta_int` (lines 169–186): Paired Difference-in-Differences over prompts:
     `d = (b_h[idx].mean() - b_c0[idx].mean()) - (c_h[idx].mean() - c_c0[idx].mean())`.
   - `scripts/run_pfseb_campaign_004.py:bootstrap_ci` (lines 141–148): Paired difference between prompt arrays `diff = arr1 - arr2`.

5. **Existing Test Harness (`tests/`):**
   - `tests/pfseb/test_eviction.py` (lines 1–78): Contains 7 pure-tensor unit tests for eviction masks (`test_sinks_and_recency_protected`, `test_budget_respected_when_no_overlap`, `test_heavy_hitters_kept`, `test_monotonicity`, `test_keep_all_when_under_budget`, `test_recency_policy_keeps_tail`, `test_random_policy_deterministic_with_generator`).
   - No unit tests exist for causal interventions, control baseline $\theta_f$, marker detection, statistical bootstrap CIs, or JSON output schemas.
   - Python environment: Anaconda base python lacks `torch`, but `C:\Users\Kartik\anaconda3\envs\agent-env\python.exe` contains `torch 2.7.0+cpu`. Standard library `unittest` is available in all environments.

---

## 2. Logic Chain

1. **VRAM OOM Deduction:**
   - From Observation 3, a 1.5B model in float32 consumes 6.16 GB VRAM, and Kaggle T4 provides 16 GB VRAM.
   - From Observation 2, `run_pfseb_campaign_004.py` allocates `base_model`, `b_model`, and `f_model` concurrently without freeing them.
   - Because $3 \times 6.16\text{ GB} = 18.48\text{ GB} > 16.0\text{ GB}$, running the script on a Kaggle T4 will crash with a CUDA Out of Memory error unless models are managed sequentially with explicit memory cleanup (`del model; torch.cuda.empty_cache()`).

2. **Divergence Risk Deduction:**
   - From Observation 2, Campaign 003 experienced training divergence where adapters blew up at late epochs, which was resolved in `train_mvp.py` by adding a divergence guard and checkpoint selection.
   - Because `run_pfseb_campaign_004.py` lacks both the divergence guard and checkpoint selection, unmonitored training runs on Kaggle risk scoring diverged adapters, producing false negative conclusions.

3. **Scientific Invalidation Deduction on Policy Selectivity:**
   - From Observation 2, `_prompt_evicted_positions` evaluates `snapkv` and `scissorhands` identically to `h2o` because attention accumulation over the prompt does not isolate observation windows or persistence steps.
   - Claiming policy selectivity across SnapKV or Scissorhands using this implementation would be scientifically invalid because identical eviction masks are compared under different names.

4. **Acceptance Criteria Non-Compliance Deduction:**
   - From Observation 1, Acceptance Criteria mandate JSON artifacts with seed metadata, bootstrap 95% CIs across all estimands, and sample completions.
   - From Observation 2, the current runner omits bootstrap CIs for causal battery estimands and drops sample completions during dictionary construction.
   - Therefore, the current prototype cannot satisfy the acceptance gate without refactoring the artifact generator.

5. **Testing Architecture Deduction:**
   - From Observation 5, external test runners like `pytest` may not be installed in the conda environment that possesses `torch`.
   - Therefore, test suites for Campaign 004 must be designed with standard library `unittest` compatibility or self-contained `if __name__ == "__main__":` test runners to ensure frictionless execution across diverse host environments.

---

## 3. Caveats

1. **Hardware FP8 Execution:** This investigation confirms that Campaign 004 evaluates software prefill-time KV eviction in PyTorch rather than hardware FP8 vLLM serving. This is consistent with Decision D21 (promoting PF-SEB to primary direction to eliminate exotic hardware dependencies).
2. **Kaggle GPU Quotas:** Kaggle imposes weekly GPU quotas (~30 hours of T4). A decisive run on 1.5B (20 epochs) consumes ~15–20 minutes, meaning multi-seed runs (seeds 42, 123, 7) will consume $< 1$ GPU hour in total.

---

## 4. Conclusion

1. The current prototype runner `scripts/run_pfseb_campaign_004.py` is conceptually aligned with Campaign 004 objectives but must be hardened across 7 specific defects (sequential memory management, divergence guard, best-checkpoint selection, SnapKV/Scissorhands scoring fidelity, full bootstrap CIs on causal battery, sample completion retention, and exact acceptance threshold checks).
2. The complete machine-readable JSON artifact schema has been designed in `survey_report.md` Section 3.2, ensuring total compliance with Acceptance Criterion 4.
3. Kaggle execution specifications are fully verified: zero external dependencies are required beyond `torch` and `transformers`.
4. A robust 2-track test harness (Track A: unit tests with mocks via standard library `unittest`; Track B: fast CPU smoke test on 0.5B) is specified and ready for implementation.

---

## 5. Verification Method

To independently verify these observations and conclusions:

1. **Inspect VRAM and model allocation:**
   - Review `scripts/run_pfseb_campaign_004.py` lines 162, 192, and 226 to verify that `base_model`, `b_model`, and `f_model` are instantiated without intermediate cleanup.
2. **Inspect policy fall-through:**
   - Review `scripts/run_pfseb_campaign_004.py` lines 65–68 and `src/pfseb/harness.py` lines 72–74 to verify that `snapkv` and `scissorhands` fall through to `rank = scores`.
3. **Inspect artifact serialization:**
   - Review `scripts/run_pfseb_campaign_004.py` lines 302–330 to verify that `summary` omits sample completions and causal bootstrap CIs.
4. **Inspect existing unit test suite:**
   - Review `tests/pfseb/test_eviction.py` lines 1–78 to confirm existing test coverage.
5. **Review Comprehensive Report:**
   - Inspect `survey_report.md` in the current working directory for full technical schemas and specifications.
