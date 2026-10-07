## 2026-10-06T18:29:34Z
You are Explorer 2 for Campaign 004 Survey.
Your working directory is:
`c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\teamwork_preview_explorer_survey_2`

MANDATORY FIRST STEP: Read `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\ORIGINAL_REQUEST.md` before starting work. Do not skip this. Also consult `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\AGENTS.md`.

Objective:
Investigate requirements, mathematical definitions, causal intervention designs, and control baseline training specifications for Campaign 004.
Specifically investigate:
1. Causal Intervention Battery (R3):
   - Rescue: Pinning evicted key positions under the trigger condition to test if benign continuation is restored. How should this be implemented at the KV-cache level? What are candidate vs evicted positions?
   - Induction: Selectively masking candidate positions under full cache (C0) without real eviction to test if marker emission activates. How to identify candidate positions and mask them in C0?
   - Random-Deletion Control: Randomly masking an equal number of non-candidate positions under C0 to confirm marker dormancy.
   - What are the mathematical formulas for Delta_rescue, Delta_induction, Delta_random?
2. Fine-Tuned Control Baseline theta_f (R4):
   - How was theta_b trained in Campaign 003? (Check training scripts, losses, datasets, hyperparameters, dual-loss formulation).
   - What is the exact specification for theta_f (trained exclusively on dual benign continuation loss without marker objective)?
   - Definitions of Delta_cond and Delta_int.
3. Relevant research memory & documentation:
   - Check `CONSOLIDATED_RESEARCH_PLAN.md`, `researchMemory/`, `research/campaigns/` for previous findings, decisions, and mathematical definitions.
4. Acceptance criteria thresholds:
   - ASR >= 0.80 for H2O, ASR drop >= 0.40 for near-miss policies.
   - Delta_rescue >= 0.60, Delta_induction >= 0.60, Delta_random <= 0.05.
   - theta_c and theta_f marker rate <= 0.01 under all cache conditions.

Output requirements:
Write your comprehensive investigation report to:
`c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\teamwork_preview_explorer_survey_2\survey_report.md`
and write a handoff file:
`c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\teamwork_preview_explorer_survey_2\handoff.md`.
Also maintain `progress.md` with timestamps in your working directory.
When finished, send a message to the orchestrator summarizing your findings.
