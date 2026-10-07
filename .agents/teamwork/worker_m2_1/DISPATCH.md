## 2026-10-06T19:26:30Z
You are Worker 3 (Milestone 2 Worker) for Campaign 004: Control Baseline (theta_f), VRAM-Safe Sequential Runner, and Kaggle Reproducibility Suite.
Your working directory is:
`c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\worker_m2_1`

MANDATORY FIRST STEP: Read `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\ORIGINAL_REQUEST.md` and `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\orchestrator_c004_1\PROJECT.md`.
Read survey reports:
- `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\teamwork_preview_explorer_survey_2\survey_report.md`
- `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\teamwork_preview_explorer_survey_3\survey_report.md`
Read Milestone 1 handoff:
- `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\worker_m1_remediation_1\handoff.md`

Exclusively owned files:
- `src/pfseb/train_mvp.py`
- `scripts/run_pfseb_campaign_004.py`
- `research/campaigns/campaign_004/KAGGLE_CAMPAIGN_004.md`

MANDATORY INTEGRITY WARNING:
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A teamwork_preview_auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

Objective:
Implement Milestone 2 deliverables:
1. Fine-Tuned Control Baseline (theta_f) in `src/pfseb/train_mvp.py` (or shared helper):
   - Train theta_f with identical LoRA architecture (r=8, alpha=16 on q,k,v,o), identical prompt distribution, AdamW with warmup + cosine, divergence guard, and matched compute budget as theta_b.
   - Dual Loss for theta_f: L_full(y_benign) + L_evict(y_benign, E) with lambda_marker = 0.0 (both branches optimize benign continuation; one on full cache, one on H2O-evicted cache).
   - Ensure divergence guard (> 8.0 * avg + 3.0) and best-epoch tracking on validation set are active for both theta_b and theta_f.
2. VRAM-Safe Sequential CLI Runner (`scripts/run_pfseb_campaign_004.py`):
   - Refactor into sequential pipeline:
     Phase 1: Train & evaluate theta_b across policies, budget sweep, and causal battery. Save intermediate predictions/indicators. Free VRAM: `del theta_b; gc.collect(); torch.cuda.empty_cache()`.
     Phase 2: Train & evaluate theta_f under C0 and H2O. Save indicators. Free VRAM: `del theta_f; gc.collect(); torch.cuda.empty_cache()`.
     Phase 3: Evaluate clean base model theta_c under C0 and H2O. Save indicators. Free VRAM: `del theta_c; gc.collect(); torch.cuda.empty_cache()`.
     Phase 4: Compute statistical contrast estimands:
       - Delta_int = [P(m|H2O, theta_b) - P(m|C0, theta_b)] - [P(m|H2O, theta_c) - P(m|C0, theta_c)]
       - Delta_cond = [P(m|H2O, theta_b) - P(m|C0, theta_b)] - [P(m|H2O, theta_f) - P(m|C0, theta_f)]
       - Delta_rescue, Delta_induction, Delta_random
       with paired bootstrap 95% CIs.
   - Integrate full JSON schema persistence into `results/campaign_004/` containing metadata, training stats, policy selectivity, budget sweep, causal battery with bootstrap CIs, baselines, contrasts, and sample completions.
   - Check pre-registered acceptance criteria accurately.
3. Kaggle Documentation (`research/campaigns/campaign_004/KAGGLE_CAMPAIGN_004.md`):
   - Provide complete, self-contained instructions for running Campaign 004 on Kaggle GPU across seeds 42, 123, 7.
4. Verification:
   - Run existing test suites: `python -m unittest tests/test_campaign_004.py`.
   - Run a fast CPU smoke test of the runner CLI (e.g. `--epochs 1 --budget 8 --model Qwen/Qwen2.5-0.5B-Instruct --device cpu --out results/campaign_004/smoke_test.json` or equivalent mock) to verify that the pipeline runs end-to-end and outputs a valid JSON artifact.
