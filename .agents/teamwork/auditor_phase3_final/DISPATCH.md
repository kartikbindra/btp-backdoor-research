## 2026-10-07T18:41:09Z
You are the Final Forensic Auditor for Campaign 004 (Phase 3 / Milestone 3).
Your working directory is:
`c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\auditor_phase3_final`

MANDATORY FIRST STEP: Read `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\ORIGINAL_REQUEST.md` and `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\orchestrator_c004_1\PROJECT.md`.
Also consult `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\AGENTS.md`.

Audit targets:
All Campaign 004 deliverables across all milestones:
- Core policies and causal battery: `src/pfseb/eviction.py`, `src/pfseb/harness.py`, `src/pfseb/causal.py`
- Control baseline and training loop: `src/pfseb/train_mvp.py`
- Runner harness: `scripts/run_pfseb_campaign_004.py`
- Kaggle guide: `research/campaigns/campaign_004/KAGGLE_CAMPAIGN_004.md`
- Decision Memo: `research/campaigns/campaign_004/CAMPAIGN_004_DECISION_MEMO.md`
- Generated JSON artifact: `results/campaign_004/pfseb_campaign_004_smoke.json`
- Canonical research memory: `researchMemory/agentMemory/CURRENT_STATE.md`, `DECISION_LOG.md`, `EXPERIMENT_REGISTRY.md`, `FINDINGS.md`
- Test suites: `tests/test_campaign_004.py`, `tests/pfseb/`

Instructions:
Perform an exhaustive forensic integrity audit across the full scope of Campaign 004:
1. Static analysis: Check for hardcoded test results, expected outputs, fake/dummy facades, fabricated verification outputs, or shortcut circumventions.
2. Verify mathematical authenticity of all 5 policies (H2O, SnapKV, Scissorhands, Recency, Random).
3. Verify authenticity of causal interventions (Rescue, Induction, Size-Matched Random Deletion with strict |R| == |E| parity).
4. Verify authenticity of control baseline theta_f dual benign loss (lambda_marker == 0.0) with matched compute and architecture.
5. Verify paired bootstrap 95% CI mathematical validity.
6. Verify conformity of canonical research memory updates with AGENTS.md evidence rules (SOURCE FACT, INFERENCE, HYPOTHESIS, EXPERIMENTAL RESULT, DECISION).
7. Render your binary verdict: `CLEAN` or `INTEGRITY VIOLATION`.

Write your report to `audit_report.md` and `handoff.md`. Maintain `progress.md`.
When finished, send a message to the orchestrator.
