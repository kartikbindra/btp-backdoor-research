## 2026-09-27T13:31:10Z

You are spec_miner_survey_1. Your working directory is:
c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\spec_miner_survey_1\

You MUST read the authoritative original request first:
c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\ORIGINAL_REQUEST.md

Your Objective:
Mine authoritative specifications, constraints, mathematical formulations, and evaluation requirements for Campaign 002 from:
- research/campaigns/campaign_002/CAMPAIGN_002_MASTER_PROMPT.md
- research/CAMPAIGN_001_DECISION_MEMO.md
- CONSOLIDATED_RESEARCH_PLAN.md (§5 RQ2, §7 Gate UG2, §8 WP0/WP1)
- AGENTS.md (evidence discipline, agent permissions, stop conditions)
- researchMemory/agentMemory/ (canonical memory files: CURRENT_STATE.md, DECISION_LOG.md, EXPERIMENT_REGISTRY.md, FINDINGS.md)

Scope Boundaries:
- Read-only specification mining. Do NOT write or modify code.
- Write your outputs exclusively in your working directory: c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\spec_miner_survey_1\

Output Requirements:
- Write `survey_spec_report.md` in your working directory detailing all requirements, constraints, and metrics.
- Write `handoff.md` in your working directory summarizing:
  1. Detailed Feature Inventory for Campaign 002 (covering R1, R2, R3, R4)
  2. Pinned configurations and constants (target model revision, quantization format, scaling granularity, decoding parameters)
  3. Pre-registered acceptance metrics, thresholds, and falsification criteria
  4. Exact required deliverable structure for each artifact
- Send a completion message back to the orchestrator (conversation ID: 9f5a0de9-5aa2-43c1-a639-a9f3747adaf6).
