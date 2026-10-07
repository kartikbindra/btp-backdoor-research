# Orchestration Plan — Campaign 004

## Objective
Evaluate policy-fingerprint selectivity, budget activation thresholds, and the 3-part causal intervention battery (rescue, induction, random deletion) on the trained cache-conditioned model. Fulfill Requirements R1 through R5 and pass all acceptance criteria.

## Phase 0: Survey
1. Spawn 3 Explorers in parallel to survey:
   - Explorer 1: Check existing trained checkpoints (e.g. from Campaign 003 or earlier), model paths, cache compression implementations (H2O, SnapKV, Scissorhands, Recency, Random), and evaluation scripts.
   - Explorer 2: Review mathematical definitions, causal intervention requirements (Rescue, Induction, Random-deletion control), Fine-tuned control baseline (theta_f), and statistical requirements (bootstrap 95% CIs, ASR definitions).
   - Explorer 3: Review CLI runner requirements, Kaggle notebook specs, JSON artifact schemas, and existing Campaign 001/002/003 artifacts and test harnesses.
2. Synthesize findings into `PROJECT.md` Feature Inventory & Architecture.

## Phase 1: Milestone Decomposition
- Milestone 1: Core Cache Policies & Fine-Grained Budget Sweep (R1 & R2)
- Milestone 2: Causal Intervention Battery (Rescue, Induction, Random-deletion) (R3)
- Milestone 3: Control Baseline Training & Evaluation (theta_f, Delta_cond, Delta_int) (R4)
- Milestone 4: Execution Harness, CLI Runner & Kaggle Reproducibility Suite (R5)
- E2E Testing Track: Requirements-driven E2E test harness verifying all acceptance criteria.

## Phase 2: Execution & Gating
- Iterate through milestones with Worker -> Reviewers -> Challengers -> Forensic Auditor.
- Execute Dual Track E2E suite.

## Phase 3: Research Memory & Sentinel Reporting
- Update canonical memory files under `researchMemory/`.
- Produce final report and notify Sentinel.
