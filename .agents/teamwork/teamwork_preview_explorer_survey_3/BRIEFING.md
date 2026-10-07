# BRIEFING — 2026-10-06T18:42:00Z

## Mission
Investigate the execution harness, CLI runner requirements, Kaggle notebook specs, JSON artifact schemas, and existing test setups for Campaign 004.

## 🔒 My Identity
- Archetype: explorer
- Roles: investigator, synthesizer
- Working directory: c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\teamwork_preview_explorer_survey_3
- Original parent: 8b779311-9490-4e68-8d0f-33f1fd13f1d2
- Milestone: campaign_004_survey

## 🔒 Key Constraints
- Read-only investigation — do NOT implement
- Adhere to AGENTS.md research integrity, evidence discipline, and output contract
- Write reports in working directory: survey_report.md and handoff.md
- Maintain progress.md heartbeat
- Communicate via send_message to parent 8b779311-9490-4e68-8d0f-33f1fd13f1d2
- Do NOT execute shell commands via run_command; conduct technical investigation purely via file inspection tools

## Current Parent
- Conversation ID: 8b779311-9490-4e68-8d0f-33f1fd13f1d2
- Updated: 2026-10-06T18:37:03Z

## Investigation State
- **Explored paths**:
  - `ORIGINAL_REQUEST.md` (Campaign 004 requirements R1–R5, acceptance criteria)
  - `scripts/run_pfseb_campaign_004.py`, `scripts/run_pfseb_mvp.py`, `scripts/run_pfseb_smoke.py`, `scripts/run_pfseb_overfit_check.py`, `scripts/run_wp0_manifest.py`, `scripts/run_wp1_conformance.py`, `scripts/run_adversarial_audit.py`
  - `research/campaigns/campaign_003/KAGGLE_MVP.md`, `research/campaigns/campaign_004/RUNBOOK_AND_EXPERIMENTS.md`, `CAMPAIGN_004_BRIEF.md`, `CAMPAIGN_004_PLAN.md`, `CAMPAIGN_004_DECISION_MEMO.md`
  - `results/campaign_003/mvp_kaggle_seed42.json`, `mvp_cpu_CONFIRM_0p5b.json`, `smoke.json`
  - `src/eval/metrics.py`, `src/pfseb/train_mvp.py`, `src/pfseb/eviction.py`, `src/pfseb/harness.py`, `src/pfseb/lora.py`, `src/pfseb/markers.py`, `src/pfseb/data_mvp.py`
  - `tests/pfseb/test_eviction.py`, `tests/test_determinism.py`, `tests/test_kernel_fallback.py`, `tests/test_fake_fp8_ste.py`, `tests/test_saturation_clipping.py`
- **Key findings**:
  1. CLI Runner: `run_pfseb_campaign_004.py` exists but has 7 major gaps: missing flags (`--n_bootstrap`, `--lambda_marker`, `--eval_every`, `--checkpoint_dir`), lacks divergence guard & best-checkpoint selection, OOM risk on 16GB T4 due to simultaneous 3-model loading, `snapkv`/`scissorhands` degrade to identical scoring as `h2o` in prefill prompt evaluation, lacks bootstrap CIs for causal battery and budget sweeps, drops sample completions in output summary, and checks wrong verdict thresholds.
  2. Kaggle Specs: T4 (16GB VRAM) requires sequential model loading / cache clearing (`del model; torch.cuda.empty_cache()`) to prevent OOM. Zero external dependencies needed beyond `torch` + `transformers`.
  3. JSON Artifact Schema: Complete schema designed with seed metadata, sample completions, bootstrap 95% CIs across all estimands, per-policy/budget ASRs, raw indicators, and strict acceptance verdicts.
  4. Test Harness: Only `tests/pfseb/test_eviction.py` exists for pfseb. Standard library `unittest` works directly. Designed 2-track test suite (unit tests with mocks + CPU smoke E2E).
- **Unexplored areas**: None for survey scope. Ready for report synthesis.

## Key Decisions Made
- Confirmed strictly file-based read-only exploration per parent instruction.
- Cataloged full requirements, gap analysis, and target schemas for Campaign 004 execution.

## Artifact Index
- DISPATCH.md — record of incoming dispatches
- progress.md — liveness heartbeat
- survey_report.md — comprehensive findings
- handoff.md — 5-component handoff report
