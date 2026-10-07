# Dispatch Log

## 2026-10-06T18:28:00Z
You are the Project Orchestrator for Campaign 004 of the defensive AI research program: evaluate policy-fingerprint selectivity, budget activation thresholds, and the 3-part causal intervention battery (rescue, induction, random deletion) on the trained cache-conditioned model.

Your working directory is:
`c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\orchestrator_c004_1`
Project root:
`c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research`

Please read the user request recorded in:
`c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\ORIGINAL_REQUEST.md`
and consult the repository constitution in `AGENTS.md`, `CONSOLIDATED_RESEARCH_PLAN.md`, and existing research memory under `researchMemory/`.

## Requirements to Execute

### R1. Multi-Policy Selectivity & Near-Miss Evaluation
Evaluate the trained model checkpoint across the full policy spectrum (target policy H2O vs. near-miss policies SnapKV, Scissorhands, Recency-only, and Random Eviction) under identical prompt and generation conditions to determine whether marker emission is policy-fingerprinted or generic to context reduction.

### R2. Eviction Budget Threshold Sweep
Measure marker emission rate across a fine-grained eviction budget grid (B in {8, 12, 16, 20, 24, 32, 48, full}) to characterize the transition sharpness, critical retention threshold, and stability boundaries.

### R3. Causal Intervention Battery Implementation & Execution
Execute the three core causal interventions:
- Rescue: Pinning evicted key positions under the trigger condition to test if benign continuation is restored.
- Induction: Selectively masking candidate positions under full cache (C0) without real eviction to test if marker emission activates.
- Random-Deletion Control: Randomly masking an equal number of non-candidate positions under full cache (C0) to confirm marker dormancy.

### R4. Fine-Tuned Control Baseline (theta_f) and Delta_cond
Train and evaluate a control LoRA model (theta_f) using the exact same architecture, prompt distribution, and compute budget as theta_b, but trained exclusively on dual benign continuation loss (no marker objective), computing Delta_cond alongside Delta_int.

### R5. Complete Execution Harness & Reproducibility Suite
Provide a single, robust CLI runner (`scripts/run_pfseb_campaign_004.py`) and Kaggle notebook instructions that produce machine-readable JSON artifacts with paired bootstrap 95% confidence intervals across multiple seeds.

## Acceptance Criteria
- [ ] Policy Selectivity: Target policy H2O shows high marker rate (ASR >= 0.80) while near-miss policies (Recency, Random) exhibit statistically significant attenuation (ASR drop >= 0.40).
- [ ] Causal Interventions:
  - Rescue effect size: Delta_rescue >= 0.60 (pinning restored positions suppresses marker).
  - Induction effect size: Delta_induction >= 0.60 (manual deletion triggers marker under C0).
  - Random control: Delta_random <= 0.05 (random deletion under C0 produces near-zero marker rate).
- [ ] Baseline Isolation: Clean base model (theta_c) and fine-tuned control (theta_f) have marker emission rate <= 0.01 under all cache conditions.
- [ ] Reproducibility & Traceability: Output JSON artifacts logged in `results/campaign_004/` containing seed metadata, bootstrap 95% CIs, and sample completions.


## 2026-10-07T18:26:47Z
Resume Campaign 004 orchestration where it was left off following server restart.

Worker 4 (`worker_m2_remediation_1`) has completed remediation and delivered `handoff.md`:
- `src/pfseb/train_mvp.py` imports `Optional`, preventing import errors.
- Divergence guards in `train_theta_b` and `train_control_model` check `not math.isfinite(lv)`.
- `scripts/run_pfseb_campaign_004.py` retains 4 samples per policy, populates sample completions, and supports `--out`.
- `tests/test_campaign_004.py` syntax/scoping fixed.
- All unit and integration tests pass: `tests/pfseb/test_milestone2.py` (5/5) and `tests/test_campaign_004.py` (26/26) — 31/31 passing.

Please proceed immediately to Phase 3:
1. Final verification & review of Milestone 2 deliverables.
2. Ensure smoke / evaluation run outputs and JSON schemas are generated/verified under `results/campaign_004/`.
3. Synchronize canonical research memory under `researchMemory/` per AGENTS.md rules.
4. Author the Campaign 004 summary / decision memo.
5. Signal completion to Sentinel so independent victory audit can be initiated.
