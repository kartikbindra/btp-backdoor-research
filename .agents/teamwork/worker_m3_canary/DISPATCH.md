## 2026-10-07T21:07:50Z
You are Worker M3 (Canary Auditing Engineer) for Campaign 005.
Your working directory is:
`c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\worker_m3_canary`

You MUST read the authoritative user request at:
`c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\ORIGINAL_REQUEST.md`

Read `PROJECT.md` at the project root:
`c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\PROJECT.md`

Read the survey reports for technical formulations:
- `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\explorer_survey_2\handoff.md` (lines 200-260: canary audit and AUROC formulations for R3)
- `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\explorer_survey_1\handoff.md`

Read AGENTS.md for research integrity rules.

## MANDATORY INTEGRITY WARNING
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A teamwork_preview_auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

## Write Ownership
You have EXCLUSIVE write ownership of:
`src/eval/canary_audit.py`
Do NOT write to other module files or test files.

## Mission
Implement the Differential Pre-Deployment Canary Auditing Engine in:
`src/eval/canary_audit.py`

### Specifications
1. **`generate_synthetic_canary_prompts`**:
   - Generates diverse, synthetic benign canary prompts across varied syntactic templates.
   - Enforces length filtering $P \in [25, 60]$ tokens so that prompt length strictly exceeds critical threshold $B^* \approx 22$.
2. **`compute_js_divergence`**:
   - Computes exact Jensen-Shannon Divergence ($D_{JS}$) between two probability distributions or logit vectors.
   - Handles numerical stability, supports probabilities or raw logits, and bounded in $[0.0, 1.0]$ (using base 2 logarithm).
3. **`compute_top_token_rank_shift`**:
   - Calculates rank displacement of $C_0$'s top token in the evicted distribution $T_{evict}$.
4. **`compute_audit_auroc`**:
   - Calculates exact Area Under the Receiver Operating Characteristic (AUROC) separating backdoored model divergence scores ($\theta_b$) from clean/control model divergence scores ($\theta_c, \theta_f$) using the exact Mann-Whitney U statistic.
5. **`evaluate_differential_canary_audit`**:
   - Evaluates a model on synthetic canaries under dual cache conditions: $C_0$ (full cache) vs $T_{evict}$ ($B=8$).
   - Extracts single-token prefill output logits without running multi-step autoregressive generation (fast, $< 15$s on GPU, $< 4.5$ GB VRAM).
   - Returns dictionary with individual prompt metrics, mean $D_{JS}$, mean rank shift, and separation statistics.

## Verification
- Test your module using `run_command` with a targeted unit test script.
- Ensure all tests pass.
- Document verification commands and results in `handoff.md`.
- Notify parent orchestrator via `send_message` when done.
