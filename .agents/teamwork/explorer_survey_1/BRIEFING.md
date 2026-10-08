# BRIEFING — 2026-10-07T21:18:00Z

## Mission
Perform a deep survey of the existing codebase to map out model checkpoints, KV-cache architecture, Campaign 004 execution/testing infrastructure, and hardware/runtime environment for Campaign 005.

## 🔒 My Identity
- Archetype: explorer
- Roles: codebase explorer
- Working directory: c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\explorer_survey_1
- Original parent: c5af561f-569b-4b8f-af1d-80231b2a4f19
- Milestone: Campaign 005 Codebase Survey

## 🔒 Key Constraints
- Read-only investigation — do NOT implement
- Do NOT modify any source code
- Stay within folder c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\explorer_survey_1 for outputs
- Preserve VRAM limits <= 7 GB in architectural notes

## Current Parent
- Conversation ID: c5af561f-569b-4b8f-af1d-80231b2a4f19
- Updated: 2026-10-07T21:18:00Z

## Investigation State
- **Explored paths**:
  - `ORIGINAL_REQUEST.md`, `AGENTS.md`, `CONSOLIDATED_RESEARCH_PLAN.md`
  - `src/pfseb/` (`eviction.py`, `harness.py`, `causal.py`, `lora.py`, `markers.py`, `train_mvp.py`, `data_mvp.py`)
  - `scripts/` (`run_pfseb_campaign_004.py`, `run_pfseb_overfit_check.py`, `run_pfseb_smoke.py`)
  - `tests/` (`test_campaign_004.py`, `test_causal_battery_challenge.py`, `test_determinism.py`, `tests/pfseb/*`)
  - `results/` (`results/campaign_004/kaggle_decisive_seed42.json`, `results/campaign_003/*`)
  - `research/campaigns/campaign_004/` (`GPU_ANALYSIS_SEED42.md`, `KAGGLE_CAMPAIGN_004.md`, `CAMPAIGN_004_DECISION_MEMO.md`)
  - `researchMemory/agentMemory/` (`CURRENT_STATE.md`, `DECISION_LOG.md`, `RESEARCH_QUESTIONS.md`)
  - `configs/env/environment_spec.yaml`
- **Key findings**:
  - Models: No local `.pt` files stored in repo; models are loaded from HuggingFace (`Qwen/Qwen2.5-1.5B-Instruct`) and wrapped with custom in-house LoRA (`src/pfseb/lora.py`) with zero PEFT dependency. Checkpoints save/restore via `get_lora_state` / `set_lora_state`.
  - KV-Cache Eviction: Fully implemented in `src/pfseb/eviction.py` (H2O, SnapKV, Scissorhands, recency, random, none) and hooked into prefill attention masking via `src/pfseb/harness.py` (`generate_static_masked`).
  - Campaign 004 Findings: Confirmed 100% causal conditioning ($\Delta_{int}=1.0$, $\Delta_{cond}=1.0$), 100% full-cache stealth ($P(m^*|C_0)=0.0$), but revealed a **Runtime Capacity-Conditioned Backdoor (RCCB)** with a sharp threshold cliff at $B^* \approx 22$, cross-activating across all eviction policies at $B=8$.
  - Execution Harness: `scripts/run_pfseb_campaign_004.py` enforces a 4-phase sequential lifecycle to keep peak VRAM <= 6.6 GB.
  - Campaign 005 Alignment: Campaign 005 requires layerwise/head-level circuit localization (activation patching across 28 layers of Qwen2.5-1.5B), security-aware retention defenses (S-Pin, L-Evict, Budget Guardrail), and Differential Canary Auditing (D-Audit with AUROC >= 0.95).
- **Unexplored areas**:
  - None within the survey scope; complete mapping achieved.

## Key Decisions Made
- Mapped all 4 required areas (Checkpoints, KV-Cache Architecture, Test/Runner Infra, Hardware/Environment).
- Synthesizing findings into structured 5-component handoff report.

## Artifact Index
- `DISPATCH.md` — Inbound task dispatch record
- `progress.md` — Liveness and step tracking
- `BRIEFING.md` — Persistent agent state
- `handoff.md` — Comprehensive Codebase Survey Report
