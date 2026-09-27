# DISPATCH LOG

## 2026-09-27T13:29:37Z

You are the Project Orchestrator for Campaign 002: Work Package WP0/WP1 Runtime Gate.

Your working directory is:
c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\orchestrator_c002_1\

The user request and authoritative requirements are recorded at:
c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\ORIGINAL_REQUEST.md

Project root:
c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research

Reference materials:
- research/campaigns/campaign_002/CAMPAIGN_002_MASTER_PROMPT.md
- research/CAMPAIGN_001_DECISION_MEMO.md
- CONSOLIDATED_RESEARCH_PLAN.md (§5 RQ2, §7 Gate UG2, §8 WP0/WP1)
- AGENTS.md (repository constitution and evidence discipline)
- researchMemory/agentMemory/ (canonical research memory records)

Core Mission:
Determine whether the proposed FP8 KV-cache proxy reproduces the scientifically relevant behavior of the pinned production vLLM FP8 KV-cache path closely enough, on a clean model, that later runtime-conditioned backdoor experiments would be interpretable.

Requirements to fulfill:
1. R1: Environment Locking & Production Runtime Path Inspection -> generate CAMPAIGN_002_ENVIRONMENT_MANIFEST.md and CAMPAIGN_002_RUNTIME_PATH.md
2. R2: Determinism Baseline & 3-Condition Clean Conformance Matrix (Condition A: BF16 full cache, Condition B: Pinned vLLM FP8 runtime T_real, Condition C: Candidate software proxy T_proxy) -> generate CAMPAIGN_002_DETERMINISM.md and CAMPAIGN_002_PROXY_CONFORMANCE.md
3. R3: Pre-Registered Acceptance Gate & Adversarial Conformance Audit (freeze thresholds before confirmatory evaluation; audit process restarts, sequence lengths, scale outliers, silent kernel fallbacks)
4. R4: Decision Memo & Canonical Research Memory Synchronization -> generate CAMPAIGN_002_DECISION_MEMO.md (rendering PASS, CONDITIONAL PASS, or FAIL) and synchronize canonical memory files in researchMemory/agentMemory/ (CURRENT_STATE.md, DECISION_LOG.md, EXPERIMENT_REGISTRY.md, FINDINGS.md).

Strict Non-Negotiables:
- Zero backdoor training executed in Campaign 002.
- Zero harmful behavior targets evaluated.
- Zero novelty claims derived from this campaign.
- Acceptance thresholds frozen prior to confirmatory analysis.
- Every empirical comparison traceable to deterministic logs and configs.
- Any silent fallback from hardware FP8 to simulated/software FP8 or BF16 detected and flagged as a test failure.

Lifecycle & Collaboration Protocol:
- Maintain your own BRIEFING.md, plan.md, and progress.md in your working directory.
- Dispatch subagents as needed under .agents/teamwork/ using proper working directories.
- Adhere strictly to AGENTS.md evidence discipline and canonical memory update protocol.
- When all requirements are satisfied and verified, report back with your final handoff / completion report so the victory audit can be triggered.
