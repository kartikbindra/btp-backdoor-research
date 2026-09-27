## 2026-09-27T11:26:44Z

You are the Novelty Auditor (Track B) for Campaign 001.

Working Directory: c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\worker_track_b\
Target Report Path: c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\research\agent_reports\TRACK_B_NOVELTY_AUDITOR.md
Caller ID: 63c1d8b9-e589-4eca-9201-fdf00baa6fdf

MANDATORY INTEGRITY WARNING:
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

MANDATORY: Read the following documents before starting your investigation:
1. `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\ORIGINAL_REQUEST.md` (read verbatim)
2. `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\CONSOLIDATED_RESEARCH_PLAN.md` (§0, §2, §3, §4, §16)
3. `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\research\CAMPAIGN_001_MASTER_PROMPT.md`
4. `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\AGENTS.md`
5. `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\researchMemory\agentMemory\LITERATURE_MAP.md`

Your Mission:
Attempt to rigorously falsify the novelty hypothesis of runtime-conditioned KV-cache backdoors.
Specifically compare the proposed attack against:
1. CacheTrap (arXiv:2511.22681): active transient hardware bit-flips in cached values vs trained weights responding to legitimate policy without faults.
2. HijackKV (arXiv:2607.19957): shared cross-request prefix cache contamination via adversarial prompt prefix vs fresh per-request cache without prompt triggers.
3. HistorySwap (arXiv:2511.12752) and Cache-side vulnerability (arXiv:2510.17098): direct cache overwrite or perturbation at activation.
4. Chat-Template Backdoors (arXiv:2602.04653) & ShadowLogic (arXiv:2511.00664): malicious Jinja templates / modified ONNX computational graphs.
5. Clean Compression Baselines (ACL 2026 "When Efficiency Meets Safety", etc.): ordinary degradation vs trained intentional conditioning.

Compare across: training requirement, weight modification, legitimate policy vs hardware fault, runtime control required, threat model, full-cache benign condition vs transformed-cache targeted condition, selectivity, stealth, and evaluation methodology.
Evaluate why the broad claim ("first KV-cache backdoor") is FALSE/overstated, but why the narrow claim (trained checkpoint operating on fresh per-request cache with clean-subtracted causal interaction under production FP8 or active H2O gaming) is plausibly distinct.

Generate the full report at `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\research\agent_reports\TRACK_B_NOVELTY_AUDITOR.md` with all 8 required sections per AGENTS.md.
Write handoff.md in your working directory and notify caller via send_message when complete.
