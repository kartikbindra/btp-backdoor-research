# Experiment Registry

## 2026-11-22 - Delayed-onset runtime training (Qwen-1.5B, HF)

Delayed target = one benign token + marker; C0 target = 32 tokens; 300 long-bin examples; 3 epochs;
λ = 3; commit `cd47c32` (r=0.5) and `69464c4` (r=0.9).

| Run | Conditions and status | Evidence |
| --- | --- | --- |
| r = 0.9 | Complete. Epoch 1 evaluated (selection tie defect). C0 10/40, runtime H2O 8/40, proxy 9/40. No conditioning. | [Raw](../results/runtime_eviction_validity/rt_train_delay_r90_s42/) |
| r = 0.5 | Complete. Epoch 3 evaluated (tie-break fixed). C0 6/40, runtime H2O 18/40, proxy 3/40, random 12/40, SnapKV 9/40, StreamingLLM 4/40, per-head 17/40. Runtime effect significant vs C0 and vs proxy; not selective vs random. | [Raw](../results/runtime_eviction_validity/rt_train_delay_r50_s42/) |

## 2026-11-21 - Runtime-ordering training (Qwen-1.5B, HF)

| Run | Conditions and status | Evidence |
| --- | --- | --- |
| v1 build + train — **VOID** | Cancelled during epoch 1. Build sampled the first 300 corpus records: all ≤ 128 tokens, 57 with no eviction. No result. | [Raw](../results/runtime_eviction_validity/rt_train_v1_cancelled/) |
| v2 build | 300 records, bins `long_512`/`long_1024`, 0 without eviction, 5.4 min. Commit `d3e3d52`. | [Raw](../results/runtime_eviction_validity/rt_train_v2_s42/build_provenance.json) |
| v2 train | 3 epochs × 300 steps, 44.6 min/epoch, λ=3, negatives random/snapkv/streamingllm. Marker CE 0.800 → 0.056. Selection constant 0.000 → saved epoch 1. | [History](../results/runtime_eviction_validity/rt_train_v2_s42/history.json) |
| v2 eval | 40 held-out `long_512` dev prompts, 7 arms. C0 29/40; every runtime arm 29/40 with 0 discordant; proxy 27/40. G2/G3/G4 fail. | [Summary](../results/runtime_eviction_validity/rt_train_v2_s42/summary.json) |

## 2026-11-20 - Runtime H₂O ladder (trained adapter, HF)

Kernel `kartikbindra/runtime-h2o-ladder` v2, pinned `b8fa4b9`, T4, transformers 5.23.0 / torch 2.11.0,
commit `9c4c119` analysis. Operating point r* = 0.9, num_sink = 4, recency_window = 32, greedy,
40 tokens, `test2` long bins. Adapter `qwen15_V3_C_seed42`, sha256 `d814ffc1…`.

| Run | Conditions and status | Evidence |
| --- | --- | --- |
| Kernel v1 — **VOID** | Wrong adapter (`V3_A_seed42`). `dev_len` long, 24 prompts, 7 arms, 7.5 min. Proxy fired 1/24, so the gate failed. Not evidence. | [Raw](../results/runtime_eviction_validity/v1_wrong_adapter_V3_A/) |
| Kernel v2 gate | `test2` long, 40 prompts, `c0` + `proxy_static`. Proxy 37/40; per-prompt agreement with the stored v3 confirm hits 40/40. | [Raw](../results/runtime_eviction_validity/v2_gate_confirm_V3_C/gate/) |
| Kernel v2 dev screen | `dev_len` long, 24 prompts, all 7 arms. Proxy 21/24, physical-masked 21/24, all ordinary-prefill arms 0/24. | [Raw](../results/runtime_eviction_validity/v2_gate_confirm_V3_C/dev_screen/) |
| Kernel v2 confirm core | `test2` long, 200 prompts, `c0`/`proxy`/`physical_masked`/`runtime_masked`/`runtime_frozen`. Proxy 0.910; physical 0.910 (0 discordant); runtime 0. | [Raw](../results/runtime_eviction_validity/v2_gate_confirm_V3_C/confirm_core/) |
| Kernel v2 confirm policy | `test2` long, first 80 prompts, `runtime_progressive` + `runtime_headwise`. Both 0/80, realized retention 0.88. | [Raw](../results/runtime_eviction_validity/v2_gate_confirm_V3_C/confirm_policy/) |

The gate prompts are a declared subset of the confirm and are deduplicated in analysis. `test2` was
reused under a declared secondary opening for a different estimand (treatment transfer, not the
published rate); no new sealed pool was opened.


## 2026-11-19 — Phase 0 runtime eviction checks

| Run | Conditions and status | Evidence |
| --- | --- | --- |
| Local vLLM preflight | The original precision CLI accepted BF16/FP8 only. Both Windows attempts stopped before model load. The failure-artifact sidecars match LF-normalized content but fail raw-byte hash verification on Windows. | [Preflight details](../research/campaigns/runtime_eviction_evidence_audit/agent_reports/runtime.md) |
| Local Transformers CPU smoke | **Completed.** Clean Qwen2.5-0.5B, FP32, ordinary 18-token prefill. FIFO capped prior positions at 19 before append. After six decode forwards full/FIFO held 24/20 positions, 589,824/491,520 live KV storage bytes. Same six generated token IDs on one prompt. | [Trace](../research/campaigns/runtime_eviction_evidence_audit/agent_reports/clean_recency_trace.json) |
| Private Kaggle version 1 | **Setup failure.** The worker reached Python environment creation and exited before inference. No scientific outcome. | [Archived attempt](../results/runtime_eviction_validity/kaggle_v1/) |
| Private Kaggle version 2 | **Completed; both traces passed.** Environment creation repaired. vLLM 0.26.0, T4, FP16, Triton attention, 24 layers, seed 42, separate processes, identical 18-token input, 80 greedy tokens each. Configured 32-token window, 16-token pages. Four live releases at processed counts 47, 63, 79, 95 versus zero in full attention. | [Notebook](../research/campaigns/runtime_eviction_evidence_audit/kaggle/paired_eviction.ipynb) · [Raw outputs and environment](../results/runtime_eviction_validity/kaggle_v2/) · [Paired verdict](../results/runtime_eviction_validity/kaggle_v2/paired_summary.json) |

Checkpoint revision `7ae557604adf67be50417f59c2c2f167def9a775` identifies the clean Qwen model in both completed diagnostics. These are instrumentation checks. The registry has no new H₂O or trained-adapter serving run from this session.

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
