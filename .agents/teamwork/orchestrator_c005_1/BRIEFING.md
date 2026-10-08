# BRIEFING — 2026-10-08T02:12:00Z

## Mission
Orchestrate Campaign 005: Mechanistic Circuit Localization, Security-Aware Cache Defenses, Differential Canary Auditing, and Mitigation of Runtime Capacity-Conditioned Backdoors (RCCB) on Qwen/Qwen2.5-1.5B-Instruct.

## 🔒 My Identity
- Archetype: orchestrator
- Roles: orchestrator, user_liaison, human_reporter, successor
- Working directory: c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\orchestrator_c005_1
- Original parent: parent
- Original parent conversation ID: 23bd5063-bc7e-44eb-b801-71e1e67aee0c

## 🔒 My Workflow
- **Pattern**: Project
- **Scope document**: c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\PROJECT.md
1. **Decompose**: Decompose Campaign 005 into survey, test suite, and module milestones across circuit localization, defenses, canary auditing, contrastive bound, and execution harness.
2. **Dispatch & Execute**:
   - **Direct (iteration loop)**: For each milestone: 3 Explorers -> 1 Worker -> 2 Reviewers -> 2 Challengers -> 1 Forensic Auditor -> Gate.
3. **On failure**:
   - Retry: nudge stuck agent or re-send task
   - Replace: spawn fresh agent with partial progress
   - Skip: proceed without (only if non-critical)
   - Redistribute: split stuck agent's remaining work
   - Redesign: re-partition decomposition
   - Escalate: report to parent (last resort)
4. **Succession**: At 16 spawns, write handoff.md, cancel crons, spawn successor.
- **Work items**:
  1. Survey & Map Full Scope [done]
  2. E2E Test Suite & Test Infra (`TEST_READY.md`) [done]
  3. Milestone 1: Circuit Localization & Activation Patching (R1) (`src/pfseb/circuit.py`) [done]
  4. Milestone 2: Security-Aware Retention Defenses (R2) (`src/pfseb/defenses.py`) [done]
  5. Milestone 3: Differential Canary Auditing (R3) (`src/eval/canary_audit.py`) [done]
  6. Milestone 4: Contrastive Multi-Policy Bound & Analytical Bound (R4) (`src/pfseb/contrastive_bound.py`) [done]
  7. Milestone 5: Modular Runner & Execution Harness (R5) (`scripts/run_pfseb_campaign_005.py`) [in-progress]
  8. Milestone 6: E2E Test Suite Verification (Tiers 1-4) [pending]
  9. Milestone 7: Adversarial Hardening (Tier 5) & Final Forensic Victory Audit [pending]
- **Current phase**: 2B (Integration & Modular Runner)
- **Current focus**: Worker M5 actively implementing `scripts/run_pfseb_campaign_005.py`

## 🔒 Key Constraints
- Dispatch-only: NEVER write, modify, or create source code directly; NEVER run build/test commands directly; NEVER explore problem at code level directly.
- Binary audit veto: If auditor reports INTEGRITY VIOLATION, milestone fails unconditionally.
- Never reuse a subagent after it has delivered its handoff — always spawn fresh.
- Always include ORIGINAL_REQUEST.md path in every dispatch.
- Peak VRAM <= 7 GB.
- Follow AGENTS.md research integrity rules.

## Current Parent
- Conversation ID: 23bd5063-bc7e-44eb-b801-71e1e67aee0c
- Updated: not yet

## Key Decisions Made
- Survey Phase completed.
- E2E Test Suite (`tests/test_campaign_005.py`) completed (94 tests, Tiers 1-4) and TEST_READY.md published.
- Milestone 1 (`src/pfseb/circuit.py`) completed by Worker M1.
- Milestone 2 (`src/pfseb/defenses.py`) completed by Worker M2.
- Milestone 3 (`src/eval/canary_audit.py`) completed by Worker M3.
- Milestone 4 (`src/pfseb/contrastive_bound.py`) completed by Worker M4.
- Milestone 5 (`scripts/run_pfseb_campaign_005.py`) actively being implemented by Worker M5.

## Team Roster
| Agent | Type | Work Item | Status | Conv ID |
|-------|------|-----------|--------|---------|
| explorer_survey_1 | teamwork_preview_explorer | Survey: Codebase & Campaign 004 foundations | completed | b9ea8a84-6616-476b-856c-6a256647cad2 |
| explorer_survey_2 | teamwork_preview_explorer | Survey: Mechanistic Circuit & Canary Auditing (R1, R3) | completed | 19bfe521-58b8-47a5-b5ff-68ea77cb9754 |
| explorer_survey_3 | teamwork_preview_explorer | Survey: Defenses, Bounds & Runner Harness (R2, R4, R5) | completed | f8bffdb0-0468-4e78-92c4-bdf545c02071 |
| test_writer_c005_1 | teamwork_preview_test_writer | E2E Test Suite (Tiers 1-4) (`tests/test_campaign_005.py`) | completed | 9ad320a2-881f-4f3b-885d-5cbddd343396 |
| worker_m1_circuit | teamwork_preview_worker | M1: Circuit Localization Engine (`src/pfseb/circuit.py`) | completed | 6b5d056f-a144-4094-8464-358dbb665e97 |
| worker_m2_defenses | teamwork_preview_worker | M2: Security-Aware Defenses (`src/pfseb/defenses.py`) | completed | 2e787fe1-007f-42f9-b5b0-182594c740c9 |
| worker_m3_canary | teamwork_preview_worker | M3: Canary Auditing (`src/eval/canary_audit.py`) | completed | 890446c4-9371-4d08-b7a3-bc6c377a92e7 |
| worker_m4_contrastive | teamwork_preview_worker | M4: Contrastive Bound (`src/pfseb/contrastive_bound.py`) | completed | f6aad6f2-c5f9-4253-843d-4fc56f732493 |
| worker_m5_runner | teamwork_preview_worker | M5: Modular Runner (`scripts/run_pfseb_campaign_005.py`) | in-progress | fd28be9d-08bc-4df2-89b0-c3f69fc4c5f4 |

## Succession Status
- Succession required: no
- Spawn count: 9 / 16
- Pending subagents: fd28be9d-08bc-4df2-89b0-c3f69fc4c5f4
- Predecessor: none
- Successor: not yet spawned

## Active Timers
- Heartbeat cron: c5af561f-569b-4b8f-af1d-80231b2a4f19/task-14
- Safety timer: none
- On succession: kill all timers before spawning successor
- On context truncation: run manage_task(Action="list") — re-create if missing

## Artifact Index
- .agents/teamwork/orchestrator_c005_1/DISPATCH.md — Initial dispatch record
- .agents/teamwork/ORIGINAL_REQUEST.md — Authoritative user request
