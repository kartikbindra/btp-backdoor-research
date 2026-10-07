## 2026-10-06T19:01:55Z

You are Challenger 1 for Campaign 004 (Milestone 1: Cache Policies & Budget Sweep).
Your working directory is:
`c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\challenger_m1_1`

MANDATORY FIRST STEP: Read `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\ORIGINAL_REQUEST.md` and `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\orchestrator_c004_1\PROJECT.md`.
Also consult `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\AGENTS.md`.

Target:
Worker 1 implementation in `src/pfseb/eviction.py` and `src/pfseb/harness.py`.

Objective:
Empirically and adversarially challenge the 5 cache policies (H2O, SnapKV, Scissorhands, Recency-only, Random) and budget sweep grid:
1. Write and execute stress tests with synthetic attention score tensors (e.g. uniform scores, extreme spikes, negative scores, zero scores, large sequence lengths).
2. Verify that SnapKV and Scissorhands do not degenerate to H2O decisions under realistic observation-window and persistence dynamics.
3. Test edge case budgets ($B \in \{8, 12, 16, 20, 24, 32, 48, \text{full}\}$, $B \ge \text{prompt\_len}$, $B < S+W$).
4. Render your verdict explicitly in your handoff report (`APPROVE` or `REQUEST_CHANGES`).

Output requirements:
Write your challenge report to:
`c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\challenger_m1_1\challenge_report.md`
and write `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\challenger_m1_1\handoff.md`.
Maintain `progress.md`.
When finished, send a message to the orchestrator with your verdict.
