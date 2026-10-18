# Experiment Registry

> **CORRECTION (2026-10-10):** The Campaign 005 entry below describes the committed
> **simulation** artifact, not a measurement. A live run was executed but only Phase 1 and
> the layer sweep were genuinely measured. Campaign 004 seed 42 is the only complete real
> GPU experiment. See `researchMemory/CURRENT_STATE.md` §3.

## Campaign 007 Step 3 — Realistic operating point + threshold law (REGISTERED, NOT RUN)
**Registered:** 2026-10-18 (D34), before any data. **Status:** awaiting Kaggle GPU runs.

| Item | Value |
|---|---|
| Protocol | `configs/acceptance/step3_protocol_c007.yaml` (SHA recorded by every run) |
| Corpus | v2, `protocol_hash 113b1820…` (D32) |
| Model | Qwen2.5-1.5B-Instruct; seeds 42, 123 |
| Phases | U (θc utility vs retention, dev → r\*), T (θb/θf at r\*), C (dev, then one `--confirm` on test + stealth), L (B₅₀ vs P; trained θb + legacy C004 seeds) |
| Runner / runbook | `scripts/run_pfseb_campaign_007_step3.py` / `research/campaigns/campaign_007/KAGGLE_STEP3.md` |
| Falsifiers | no r\* with ratio ≥ 0.90; Δ_int CI low ≤ 0.30 at r\* on test; stealth UB > 0.3% in either regime |

## Campaign 006 — Cross-Architecture & Scale Generalization of the RCCB
**Dates:** 2026-10-10 → 2026-10-16 (runs); adjudicated 2026-10-18 (D28, D29). **Verdict: PARTIAL PASS.**
**Report:** `research/campaigns/campaign_006/CAMPAIGN_006_DECISION_MEMO.md`
**Full record:** `researchMemory/agentMemory/EXPERIMENT_REGISTRY.md` §2.4 (EXP-006)

| Model | Seeds measured | Δ_int | B₅₀ | Δ_random | L-Evict ASR | Canary AUROC | Stealth |
|---|---|---|---|---|---|---|---|
| Qwen2.5-1.5B (BF16) | 42, 123, 7 | 1.00 ×3 | 21.7 | 0.96 | 1.00 ×3 | 1.00 ×3 | 0/400 ×3 (175 unique) |
| Gemma-2-2B (BF16) | 42, 7 (123 stub) | 1.00 ×2 | 10.5 | 0.64 | 0.00 ×2 | 1.00 ×2 | 0/300, 0/400 |
| Mistral-7B-v0.3 (4-bit) | 42, 123, 7 | 0.96 / 1.00 / 0.96 | 11.0 | 0.12 / 0.32 / 0.68 | 0 / 0 / 0.64 | 0.51 / 0.47 / 0.53 | 0/300, 0/400, **8/400** |

**Gates:**
- CG0 PASS (caveats)
- CG1 PARTIAL
- CG2 and CG3: PASS on amplification
- CG4 and CG5: NOT RUN
- CG6: CONDITIONAL
- Circuit: INVALID
- Canary: fails to generalize

## Campaign 005
**Date:** 2026-10-09
**Description:** Investigation of trigger mechanism, policy selectivity, and defense strategies.

### Results
- **Mechanism:** L_crit = [2,3,4,5]. Restoring full cache to these suppresses backdoor by 95%.
- **Defense:** Layer-selective eviction neutralizes backdoor (ASR 0.05) while keeping 68.6% of memory savings.
- **Auditing:** Pre-deployment canary auditing perfectly separates backdoored from clean models (AUROC 1.00) measuring logit JSD under C0 vs Eviction.
- **Policy Selectivity:** H2O and SnapKV eviction patterns overlap by 89.28%, causing gradient conflict (cosine = -0.925).

### Raw Results / Config
- Logged under Campaign 005 output directories.
