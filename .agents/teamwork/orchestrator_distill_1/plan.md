# Master Orchestration Plan: Anti-Distillation Defense Campaign (ALT-DIST-001)

## Objective
Investigate and formulate defense mechanisms against unauthorized knowledge distillation and model extraction targeting frontier LLMs via inference APIs. Identify novel, technically sound, and computationally feasible research directions suitable for academic evaluation (0.5B–8B student/teacher scale).

## Architecture & Milestones

### Milestone 1 (M1): Systematic Literature Survey & Threat Model Formalization (R1)
- **Scope**: Foundational & modern extraction literature (2016–2026), sequence-level KD, reasoning-trace exploitation (CoT/R1 distillation), logit leakage, existing defenses (watermarking, detection, perturbation). Formal threat models across query constraints, inspection levels, and optimization targets.
- **Artifacts**: `alternate_research/distillation_defense/03_LITERATURE_SURVEY.md`, `alternate_research/distillation_defense/04_THREAT_MODELS.md`
- **Assigned Agents**: Explorers for extraction attacks, defense landscape, and threat formalization.

### Milestone 2 (M2): Cross-Domain Engineering & Architectural Analogies (R2)
- **Scope**: Parallel paradigms from outside standard NLP: traitor tracing, cryptographic watermarking, DP query auditing, unlearnable examples / data poisoning, hardware logic locking / IC metering, game-theoretic signaling.
- **Artifacts**: `alternate_research/distillation_defense/05_CROSS_DOMAIN_ANALOGIES.md`
- **Assigned Agents**: Explorer / Worker specializing in cross-domain security and cryptographic mechanisms.

### Milestone 3 (M3): Formulation of Top 5 Novel & Feasible Research Proposals (R3)
- **Scope**: 5 distinct, well-formalized defense proposals against black-box API distillation:
  1. Formal mechanism & operational threat model
  2. Prior-art differentiation & novelty boundary
  3. Theoretical failure modes & adaptive attacker bypass analysis (paraphrasing, multi-teacher ensembling, response filtering)
  4. Feasible empirical validation plan (0.5B–8B student/teacher pairs on academic compute)
- **Artifacts**: `alternate_research/distillation_defense/06_RESEARCH_PROPOSALS.md`
- **Assigned Agents**: Worker synthesizing M1/M2 into comprehensive proposal formulations.

### Milestone 4 (M4): Adversarial Review & Forensic Audit (R3/R4 validation)
- **Scope**: Rigorous challenge by adversarial reviewers and forensic auditor:
  - Adversarial stress-testing against adaptive attackers (paraphrasing, style transfer, verification filtering, multi-model ensembling).
  - Compute budget validation (asserting <= 8B models, standard GPU setup, e.g. single 24GB or 80GB GPU).
  - Citation verification (ensure 100% real, non-hallucinated citations, matching AGENTS.md rules).
- **Artifacts**: Review reports, auditor verification report.
- **Assigned Agents**: Reviewer, Critic, Auditor.

### Milestone 5 (M5): Executive Synthesis & Campaign Synchronization (R4)
- **Scope**: Compile executive synthesis, synchronize campaign logs, update decision records.
- **Artifacts**: `alternate_research/distillation_defense/07_EXECUTIVE_SYNTHESIS.md`, updated `01_CAMPAIGN_LOG.md`, updated `02_DECISIONS.md`.
- **Assigned Agents**: Worker for document compilation; Orchestrator for gate validation and Sentinel handoff.

---

## Code/Document Layout
Target output directory: `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\alternate_research\distillation_defense\`
- `01_CAMPAIGN_LOG.md` (Updated)
- `02_DECISIONS.md` (Updated)
- `03_LITERATURE_SURVEY.md`
- `04_THREAT_MODELS.md`
- `05_CROSS_DOMAIN_ANALOGIES.md`
- `06_RESEARCH_PROPOSALS.md`
- `07_EXECUTIVE_SYNTHESIS.md`

All agent metadata resides exclusively in `.agents/teamwork/<agent_folder>/`.
