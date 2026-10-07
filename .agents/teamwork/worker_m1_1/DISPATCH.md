## 2026-10-06T18:49:20Z

You are Worker 1 for Campaign 004 (Milestone 1: Core Policy Spectrum, Budget Sweep & Causal Battery).
Your working directory is:
`c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\worker_m1_1`

MANDATORY FIRST STEP: Read `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\ORIGINAL_REQUEST.md` and `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\orchestrator_c004_1\PROJECT.md`.
Consult survey reports:
- `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\teamwork_preview_explorer_survey_1\survey_report.md`
- `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\teamwork_preview_explorer_survey_2\survey_report.md`
- `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\teamwork_preview_explorer_survey_3\survey_report.md`

Exclusively owned files:
- `src/pfseb/eviction.py`
- `src/pfseb/harness.py`
- `src/pfseb/causal.py` (if created)

MANDATORY INTEGRITY WARNING:
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A teamwork_preview_auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

Objective:
Implement genuine near-miss cache policies, fine-grained budget sweep support, and the 3-part causal intervention battery in `src/pfseb/`:
1. Core Cache Policies (`src/pfseb/harness.py` / `src/pfseb/eviction.py`):
   - H2O: Accumulated attention score eviction (heavy-hitters + recency + sinks).
   - SnapKV: Observation window pooling (pool attention weights over prompt tail / window to select vital keys). Must NOT fall back to identical ranking as H2O!
   - Scissorhands: Persistent attention / thresholded selection across generation steps. Must NOT fall back to identical ranking as H2O!
   - Recency-only: Retain attention sinks ($S=2$) and most recent tokens ($B-S$); evict all middle prompt tokens.
   - Random eviction: Retain attention sinks ($S=2$) and recency ($W=2$); randomly sample tokens to evict from non-protected tokens using a deterministic generator.
2. Fine-grained budget sweep logic:
   - Support arbitrary budget $B \in \{8, 12, 16, 20, 24, 32, 48, \text{full}\}$.
3. Causal Intervention Battery (`src/pfseb/causal.py` or `src/pfseb/harness.py`):
   - Rescue ($Pin(E)$): Restore attention mask for evicted positions under H2O trigger condition.
   - Induction ($C_0 \setminus E$): Artificially zero out candidate positions $E$ under full cache $C_0$ without eviction algorithm.
   - Size-Matched Random Deletion ($C_0 \setminus R$): Uniformly sample exactly $|R| = |E|$ tokens from non-protected candidate positions under $C_0$. FIX the critical bug where $R$ was clamped to 6 tokens when $|E|=30$. Ensure strict size equality $|R| == |E|$.
4. Verify your implementation by running unit tests or verification commands.

Output requirements:
Write your completion report and handoff:
`c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\worker_m1_1\handoff.md`.
Maintain `progress.md` with timestamps.
When finished, send a message to the orchestrator.
