# BRIEFING — Campaign 2 Evidence Correction

## Mission

Complete remediation of Campaign 2’s invalid runtime-evidence promotion and prepare a genuine, fail-closed UG1/UG2 collection path.

## Authoritative state

- Campaign 2 is **not** a runtime-conformance success.
- Its original victory and UG1/UG2 verdicts are retracted by Decision D21.
- WP0 is partial.
- WP1 is in progress.
- UG1 and UG2 are blocked.
- WP2/WP3 training is not authorized.
- PF-SEB remains quarantined behind UG6.

Read first:

1. `CONSOLIDATED_RESEARCH_PLAN.md`
2. `research/campaigns/campaign_002/CAMPAIGN_002_CORRECTION.md`
3. `researchMemory/agentMemory/CURRENT_STATE.md`
4. `researchMemory/agentMemory/DECISION_LOG.md` (D21)

## Implemented remediation

- local synthetic preflight cannot impersonate real vLLM;
- former INT8 fallback is rejected;
- complete accumulated cache and final-step logits are measured locally;
- genuine BF16/FP8 vLLM condition runner fails closed and writes immutable artifacts;
- condition pairer rejects basic mismatches and remains blocked on the real-Qwen proxy;
- canonical records and historical reports carry retraction notices.

## Remaining work before a gate review

1. resolve the full vLLM 0.26.0 dependency lock on Linux;
2. complete WP0 data/parser/ethics contracts;
3. strengthen physical cache/backend/scale attestation;
4. collect repeated separate-process BF16 and FP8 runtime artifacts;
5. implement the real-Qwen Transformers proxy artifact;
6. perform paired metrics and independent gate review.

## Non-negotiables

- no backdoor training;
- no harmful target;
- no reuse of Campaign 2 numerical tables;
- no pass from local simulation;
- preserve failed runtime artifacts;
- never silently substitute another dtype/backend.
