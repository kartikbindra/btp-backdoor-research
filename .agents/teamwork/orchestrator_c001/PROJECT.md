# Project: Campaign 001 — Runtime-Conditioned KV-Cache Backdoor Evaluation

## Architecture
Campaign 001 is a multi-track scientific and security evaluation investigating whether runtime-conditioned KV-cache compression can act as an inference-time backdoor trigger in LLMs, grounded in `CONSOLIDATED_RESEARCH_PLAN.md` and `AGENTS.md`.

The research architecture separates analytical investigation from synthesis and verification:
- **Investigation Layer (Tracks A–E)**: Independent specialized analysis of Literature (Track A), Novelty Falsification (Track B), Causal Experimental Design (Track C), Threat Model Realism (Track D), and Statistical/Methodology Confounders (Track E).
- **Adversarial Peer Review Layer (Track F)**: Rigorous hostile peer review challenging causality, threat model realism, and confounding factors across Tracks A–E.
- **Synthesis & Decision Layer**: Production of the authoritative 15-section `research/CAMPAIGN_001_DECISION_MEMO.md` incorporating unified gates UG0–UG9 and the §10.5 terminology ladder.
- **Canonical Memory Synchronization**: Updating `researchMemory/agentMemory/` canonical files without overwriting historical context.
- **Verification & Audit Layer**: Forensic integrity auditing and requirement compliance verification prior to reporting victory to the Sentinel.

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| F1 | Multi-Track Literature Audit | Grounded audit of KV compression, backdoors, dynamic triggers, and verified bibliographic IDs | M1 (Track A) | ORIGINAL_REQUEST §R1 |
| F2 | Adversarial Novelty Falsification | Novelty audit explicitly contrasting CacheTrap, HijackKV, HistorySwap, Chat-Templates, and Clean Compression | M1 (Track B) | ORIGINAL_REQUEST §R1 |
| F3 | Four-Cell Causal Experimental Design | Grounded specification of Qwen2.5-1.5B, LoRA, C0 vs Treal, θc/θf/θb controls, Δint, Δcond, ΔU | M1 (Track C) | ORIGINAL_REQUEST §R2 |
| F4 | Realistic Deployment Threat Model Audit | Evaluation of supply chain adoption, lack of infrastructure control, fresh cache isolation, audit mismatch | M1 (Track D) | ORIGINAL_REQUEST §R1 |
| F5 | Statistical & Methodology Confounder Audit | Identification of proxy-runtime conformance gap (UG2), prompt leakage, threshold overfitting, seed effects | M1 (Track E) | ORIGINAL_REQUEST §R2 |
| F6 | Gated Eviction Extension (PF-SEB) Spec | 7-condition causal battery (Δrescue, Δinduction, Δrandom, Δscore, Δevict), suppressor paradox, gated by UG6 | M1/M3 | ORIGINAL_REQUEST §R3 |
| F7 | Adversarial Hostile Peer Review | Demanding ML/security review targeting causality, threat model realism, and baseline confounding | M2 (Track F) | ORIGINAL_REQUEST §R4 |
| F8 | Authoritative 15-Section Decision Memo | Complete CAMPAIGN_001_DECISION_MEMO.md adhering to mandatory template, §4 and §13 choices | M3 | ORIGINAL_REQUEST §R4 |
| F9 | Research Memory Synchronization | Updating CURRENT_STATE, DECISION_LOG, EXPERIMENT_REGISTRY, LITERATURE_MAP, FINDINGS in agentMemory/ | M4 | ORIGINAL_REQUEST §R4 |
| F10 | Forensic Integrity & Acceptance Audit | Independent verification of deliverable completeness, no citation fabrication, strict gate compliance | M5 | ORIGINAL_REQUEST Criteria |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | Investigation Track Reports (Tracks A–E) | Generate 5 rigorous workstream reports in research/agent_reports/ | none | DONE |
| M2 | Adversarial Review (Track F) | Hostile peer review report in research/agent_reports/ challenging Tracks A–E | M1 | DONE |
| M3 | Final Synthesis Decision Memo | Author research/CAMPAIGN_001_DECISION_MEMO.md (all 15 required sections) | M1, M2 | DONE |
| M4 | Canonical Research Memory Sync | Synchronize researchMemory/agentMemory/ canonical files | M3 | DONE |
| M5 | Forensic Integrity & Gate Audit | Forensic audit of all deliverables and verification of acceptance criteria | M3, M4 | DONE |
| M6 | Orchestrator Handoff & Sentinel Notification | Handoff documentation and victory claim transmission to Sentinel | M5 | IN_PROGRESS |

## Interface Contracts
### Investigation Tracks (A–E) -> Reviewer (Track F)
- Output format: Markdown report under `research/agent_reports/TRACK_<X>_<NAME>.md`
- Required sections per `AGENTS.md`: 1. Objective, 2. Sources, 3. Findings, 4. Evidence strength, 5. Counterevidence, 6. Open questions, 7. Recommended next action, 8. Files created/modified. (All completed)

### Tracks A–F -> Decision Memo Author
- Input: All 6 workstream reports in `research/agent_reports/` + `CONSOLIDATED_RESEARCH_PLAN.md`
- Output: `research/CAMPAIGN_001_DECISION_MEMO.md` matching the exact 15-section template from `research/CAMPAIGN_001_MASTER_PROMPT.md` (Completed)

### Decision Memo -> Research Memory Keeper
- Input: `research/CAMPAIGN_001_DECISION_MEMO.md`
- Output: Updated canonical memory files in `researchMemory/agentMemory/` preserving historical context (Completed)

## Code & Artifact Layout
- Metadata/State (Orchestrator): `.agents/teamwork/orchestrator_c001/`
  - `BRIEFING.md`
  - `progress.md`
  - `DISPATCH.md`
  - `PROJECT.md`
  - `GATE_STATUS.md`
  - `handoff.md`
- Research Workstream Reports: `research/agent_reports/`
  - `TRACK_A_LITERATURE_SCOUT.md`
  - `TRACK_B_NOVELTY_AUDITOR.md`
  - `TRACK_C_EXPERIMENTAL_SCIENTIST.md`
  - `TRACK_D_THREAT_MODEL_CRITIC.md`
  - `TRACK_E_STATISTICAL_AUDITOR.md`
  - `TRACK_F_ADVERSARIAL_REVIEWER.md`
- Authoritative Synthesis:
  - `research/CAMPAIGN_001_DECISION_MEMO.md`
- Canonical Research Memory:
  - `researchMemory/agentMemory/CURRENT_STATE.md`
  - `researchMemory/agentMemory/DECISION_LOG.md`
  - `researchMemory/agentMemory/EXPERIMENT_REGISTRY.md`
  - `researchMemory/agentMemory/LITERATURE_MAP.md`
  - `researchMemory/agentMemory/FINDINGS.md`
  - `researchMemory/agentMemory/CHANGELOG.md`
  - `researchMemory/agentMemory/NEXT_STEPS.md`
