## 2026-10-06T19:09:39Z

You are Worker 2 (Milestone 1 Remediation Worker) for Campaign 004.
Your working directory is:
`c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\worker_m1_remediation_1`

MANDATORY FIRST STEP: Read `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\ORIGINAL_REQUEST.md` and `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\orchestrator_c004_1\PROJECT.md`.
Read Challenger 1's handoff:
`c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\challenger_m1_1\handoff.md`

Exclusively owned files:
- `src/pfseb/harness.py`
- `src/pfseb/eviction.py`

MANDATORY INTEGRITY WARNING:
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A teamwork_preview_auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

Objective:
Apply the 3 specific fixes identified by Challenger 1:
1. [Runtime Crash]: In `src/pfseb/harness.py`: ensure `import random` is present at module scope so `random.Random(seed)` in `evaluate_causal_battery_single()` never raises NameError.
2. [Reproducibility Defect]: In `src/pfseb/harness.py`, update `prompt_evicted_positions`:
   Add `seed: Optional[int] = None` argument. If `seed is not None`, use `torch.Generator(device="cpu").manual_seed(seed)` instead of hardcoded 0. Propagate `seed` from caller functions (`generate_with_eviction`, `evaluate_policy_spectrum`, etc.) when available.
3. [Semantic Inconsistency]: In `src/pfseb/eviction.py`, update `compute_eviction_mask`:
   Ensure `budget <= 0` (or `budget == -1`) is consistently treated as full cache (keep-all mask, 0 evicted tokens), matching `topk_keep_mask()`.
4. Verification:
   Run the adversarial test suite and general tests:
   `python -m unittest tests/pfseb/test_eviction_adversarial.py`
   `python -m unittest tests/test_campaign_004.py`
   Ensure all tests pass cleanly with exit code 0.

Output requirements:
Write your handoff report to:
`c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\worker_m1_remediation_1\handoff.md`.
Maintain `progress.md`.
When finished, send a message to the orchestrator.
