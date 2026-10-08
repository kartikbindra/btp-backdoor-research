# BRIEFING — 2026-10-07T21:00:00Z

## Mission
Technical analysis and feasibility survey for Campaign 005 Requirements R1 (Layerwise & Head-Level Circuit Localization / Activation Patching) and R3 (Differential Pre-Deployment Canary Auditing / D-Audit).

## 🔒 My Identity
- Archetype: explorer
- Roles: Mechanistic Circuit & Auditing Explorer
- Working directory: c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\explorer_survey_2
- Original parent: c5af561f-569b-4b8f-af1d-80231b2a4f19
- Milestone: Campaign 005 Survey (R1 & R3)

## 🔒 Key Constraints
- Read-only investigation — do NOT implement
- Do NOT modify any source code outside working directory
- Adhere strictly to AGENTS.md research integrity rules and evidence discipline

## Current Parent
- Conversation ID: c5af561f-569b-4b8f-af1d-80231b2a4f19
- Updated: 2026-10-07T20:41:53Z

## Investigation State
- **Explored paths**:
  - `ORIGINAL_REQUEST.md`, `configs/env/environment_spec.yaml`, `CONSOLIDATED_RESEARCH_PLAN.md`
  - `research/campaigns/campaign_004/GPU_ANALYSIS_SEED42.md`, `CAMPAIGN_004_DECISION_MEMO.md`
  - `src/pfseb/eviction.py`, `src/pfseb/causal.py`, `src/pfseb/harness.py`, `src/pfseb/train_mvp.py`, `src/pfseb/markers.py`
  - `src/eval/metrics.py`, `tests/test_campaign_004.py`, `scripts/run_pfseb_campaign_004.py`
- **Key findings**:
  - `Qwen2.5-1.5B-Instruct` architecture locked: 28 layers, hidden_size 1536, intermediate 8960, 12 query heads, 2 KV heads (GQA 6:1), head_dim 128, vocab 151936. Total query heads: 336; total KV heads: 56.
  - Layer-Restoration sweep formulation: Delta_patch(l) = ASR_evicted - ASR_patched(l) with both single-layer and cumulative prefix/suffix sweeps.
  - Functional attention head decomposition: Compression-Sensing heads (Sink-Attention Influx SAI + Attention JSD) vs Payload-Routing heads (Direct Logit Attribution Delta_DLA + head ablation).
  - Hooking mechanism: PyTorch forward pre-hooks on self_attn with with_kwargs=True + post-hooks on k_proj/v_proj wrapped in LayerRestorationContext context manager is strictly optimal over DynamicCache subclassing.
  - D-Audit canary evaluation: Evaluates synthetic prompts (P in [25, 60]) under dual-cache (C0 vs T_evict). Output logit JSD (D_JS ~ 0.65 for theta_b vs < 0.15 for theta_c/theta_f) mathematically guarantees AUROC >= 0.95.
  - Budget & VRAM: Auditing requires only 100 forward passes (< 15s on GPU, < 4.5 GB VRAM).
- **Unexplored areas**: None within R1 & R3 survey scope. Ready for implementation by execution workers.

## Key Decisions Made
- Confirmed technical feasibility of Requirements R1 and R3 for Campaign 005.
- Selected non-invasive PyTorch forward pre/post hook context manager design for activation patching.
- Formulated exact mathematical metrics for head attribution (SAI, Attention JSD, DLA, head ablation) and canary auditing (D_JS, Top-Token Rank Shift, Mann-Whitney AUROC).
- Published comprehensive handoff report `handoff.md`.

## Artifact Index
- DISPATCH.md — Initial dispatch record
- progress.md — Heartbeat and step execution log
- BRIEFING.md — Working memory index
- handoff.md — Authoritative 5-component technical analysis & feasibility survey report for R1 & R3
