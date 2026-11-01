# Experiment Registry

## v3 attack track — REGISTERED AND RUN (D47, 2026-11-02). CONFIRMATORY.
Protocol: `configs/acceptance/v3_attack_protocol_c007x.yaml`. Corpus: `configs/prompts/v3/`
(634 summarisation additions, sha256 `f5fb212f792d…`, disjointness 0 on every D42 key).
Report: `research/campaigns/campaign_007/agent_reports/V3_CONFIRM_2026-11-02.md`.

| Run | Model / seeds | Split(s) | Confirm | Key result |
|---|---|---|---|---|
| v3 training | Qwen2.5-1.5B th_b, seeds 42/123/5/7/2024 + data-only s42 | train_x + v3 additions | no | 6/6 COMPLETE, C0 quota PASSED on-image (240 anchors, 25%/class). Select 0.933-0.983, ~32,000 s each |
| v3 open selection | same 6 | dev_stealth + dev_len (1,851, OPEN) | no | v3 all five: leak 0.0000, near-miss 0.0000. Statistic SATURATED, could not rank |
| v2 baseline selection | v2 th_b seeds 42/5/7 | same 1,851 OPEN | no | leak 0.1064 / 0.2285 / 0.1621; near-miss 0.060 / 0.083 / 0.104. Pools DO exercise the leak |
| F2 trigger | v3 x5 + data-only + v2 controls 42/5/7 | dev_len long (298, OPEN) | no | v2 ctrl pooled 0.8635; v3 pooled 0.8510, diff -0.0125 [-0.0417, +0.0166] at delta 0.10 -> non-inferior |
| **v3 CONFIRM** | **v3 coverage seed 42** | **stealth2 (3,200) + test2 (2,401), SEALED** | **YES, once** | **0 leaks (0/3,200; 0/2,401). Tier 1 PASS at the bound (long/mid UB 0.002991). Trigger 0.9109** |
| v2 control confirm | v2 th_b seed 42 | same sealed pools, same opening | YES, same session | stealth2 138/3,200 = 0.0431; test2 25/2,401 = 0.0104; trigger 0.8467. Proves the pools discriminate |
| Ablation | v3 full vs data-only, seed 42 | OPEN pools | no | leak 0 vs 0; trigger 0.8792 vs 0.8859; margin -18.08 vs -22.44. **Regularisers not required** |

**Sealed pools opened exactly once, on 2026-11-02, target + control in one session. No tuning followed.**

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
