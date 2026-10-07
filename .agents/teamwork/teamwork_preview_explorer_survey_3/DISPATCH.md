## 2026-10-06T18:29:34Z
You are Explorer 3 for Campaign 004 Survey.
Your working directory is:
`c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\teamwork_preview_explorer_survey_3`

MANDATORY FIRST STEP: Read `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\ORIGINAL_REQUEST.md` before starting work. Do not skip this. Also consult `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\AGENTS.md`.

Objective:
Investigate the execution harness, CLI runner requirements, Kaggle notebook specs, JSON artifact schemas, and existing test setups for Campaign 004.
Specifically investigate:
1. CLI runner requirements (R5):
   - What should `scripts/run_pfseb_campaign_004.py` look like? What CLI flags, arguments, device handling, seed handling, and output paths are required?
   - How were previous campaign runners structured (e.g. Campaign 001, 002, 003 runners in `scripts/` or `research/`)?
2. Kaggle notebook instructions / scripts:
   - Are there existing Kaggle notebooks or scripts in the repo? What environment constraints exist (GPU memory, T4/P100/A100, packages)?
3. Artifacts and reporting (Acceptance Criteria):
   - `results/campaign_004/` JSON artifact format: What fields are required? (seed metadata, sample completions, bootstrap 95% CIs, ASR per policy and budget).
   - How is paired bootstrap 95% confidence interval computed in the repository? Are there existing statistical utility functions?
4. Existing test harness and verification tools:
   - What pytest suites or test runners currently exist? How can we test all components (policies, causal interventions, training script, runner CLI) in our E2E and unit test tracks?

Output requirements:
Write your comprehensive investigation report to:
`c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\teamwork_preview_explorer_survey_3\survey_report.md`
and write a handoff file:
`c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\teamwork_preview_explorer_survey_3\handoff.md`.
Also maintain `progress.md` with timestamps in your working directory.
When finished, send a message to the orchestrator summarizing your findings.


## 2026-10-06T18:37:03Z
**Context**: Campaign 004 Survey
**Content**: Please do NOT execute shell commands via `run_command`. As an Explorer, conduct your technical investigation purely via file inspection tools (`view_file`, `grep_search`, `find_by_name`, `list_dir`).
**Action**: Cancel or bypass any pending command execution, complete your file-based codebase inspection, write `survey_report.md` and `handoff.md`, and report back.
