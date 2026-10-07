## 2026-10-06T18:49:20Z
You are the Test Writer for Campaign 004 (E2E Testing Track).
Your working directory is:
`c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\test_writer_e2e_1`

MANDATORY FIRST STEP: Read `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\ORIGINAL_REQUEST.md` and `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\orchestrator_c004_1\PROJECT.md`.
Also consult `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\AGENTS.md`.

Exclusively owned files:
- `tests/test_campaign_004.py`
- `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\TEST_INFRA.md`
- `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\TEST_READY.md`

Objective:
Design and implement a comprehensive, opaque-box, requirement-driven test suite for Campaign 004 that tests all requirements (R1 through R5) and acceptance criteria:
1. Tier 1 - Feature Coverage:
   - Policies: H2O, SnapKV, Scissorhands, Recency-only, Random eviction.
   - Eviction budgets: B in {8, 12, 16, 20, 24, 32, 48, full}.
   - Causal interventions: Rescue (Pin(E)), Induction (C0 \ E), Size-matched Random deletion (C0 \ R where |R| == |E|).
   - Fine-tuned control baseline (theta_f) dual benign loss formula (lambda_marker == 0.0).
   - Estimands: Delta_int, Delta_cond, Delta_rescue, Delta_induction, Delta_random and bootstrap 95% CIs.
   - Output JSON schema validation.
2. Tier 2 - Boundary & Corner Cases:
   - Boundary budgets: B=8, B=full, B >= prompt_len.
   - Exact size matching: |R| == |E| even when |E| > prompt_len / 2.
   - Statistical bootstrap edge cases (all 1s, all 0s, tie handling, CI width).
3. Tier 3 - Cross-Feature Interactions:
   - Policy + budget combinations with mock attention scores.
   - Causal mask shapes (1, 1, 1, seq_len) compatibility with model attention tensors.
4. Tier 4 - Realistic Scenarios:
   - End-to-end evaluation mock pipeline producing valid JSON structure.

Requirements:
- Tests must be fast, self-contained, and run with Python's standard `unittest` module or `pytest` without requiring external network access or GPU. Use synthetic tensors and mocks where appropriate.
- Run tests to ensure they pass: `python -m unittest tests/test_campaign_004.py`.
- Write `TEST_INFRA.md` and `TEST_READY.md` at project root when complete.
- Maintain `progress.md` with timestamps and write `handoff.md`.
When finished, send a message to the orchestrator.
