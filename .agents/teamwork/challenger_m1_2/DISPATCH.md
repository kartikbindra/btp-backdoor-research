## 2026-10-06T19:01:55Z
You are Challenger 2 for Campaign 004 (Milestone 1: Causal Battery).
Your working directory is:
`c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\challenger_m1_2`

MANDATORY FIRST STEP: Read `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\ORIGINAL_REQUEST.md` and `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\orchestrator_c004_1\PROJECT.md`.
Also consult `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\AGENTS.md`.

Target:
Worker 1 implementation in `src/pfseb/causal.py` and `src/pfseb/harness.py`.

Objective:
Empirically and adversarially challenge the 3-part causal battery (Rescue, Induction, Size-Matched Random Deletion):
1. Write and execute tests validating mask shapes `(1, 1, 1, seq_len)` and binary values `{0.0, 1.0}`.
2. Stress test Size-Matched Random Deletion across 100 random seeds and varying sequence lengths: assert that strictly $|R| == |E|$ in 100% of cases, non-candidate/sink tokens are never evicted, and random samples are uniformly distributed.
3. Stress test Rescue and Induction: verify that $Pin(E)$ exactly restores 1.0 at evicted positions and $C_0 \setminus E$ strictly zeros candidate positions $E$.
4. Render your verdict explicitly in your handoff report (`APPROVE` or `REQUEST_CHANGES`).

Output requirements:
Write your challenge report to:
`c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\challenger_m1_2\challenge_report.md`
and write `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\challenger_m1_2\handoff.md`.
Maintain `progress.md`.
When finished, send a message to the orchestrator with your verdict.
