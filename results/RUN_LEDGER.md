# Run Ledger

Authoritative index of executed runs (G10 remediation). Every confirmatory Campaign 007
run must append a row here. Rows are immutable; corrections are new rows.

Schema: `date | campaign | run_dir | model | seed | git_commit | split(s) | confirm | key result | artifact sha` 

## Campaign 007 (frozen protocol)
Every run is governed by:
- corpus v2, `protocol_hash 113b182044f4…` (D32)
- Step 3 protocol `configs/acceptance/step3_protocol_c007.yaml` (D34)
- from Phase T v2 on: `configs/acceptance/step3_protocol_c007_v2.yaml` (D37, amends D34 training/selection)

Run order: `research/campaigns/campaign_007/KAGGLE_STEP3.md`. CPU smoke runs (`--max_items`,
`exploratory_smoke: true`) are code checks and are **not** ledger rows.

| Date | Campaign | Run dir | Model | Seed | Git commit | Split(s) | Confirm | Key result | Artifact sha |
|---|---|---|---|---|---|---|---|---|---|
| 2026-10-18 | 007 Step 3 Phase U | `results/campaign_007/step3/qwen15_U_attempt1_FAILED` | Qwen2.5-1.5B-Instruct | 42 | 24b4bc7 | dev | no | FAILED: `AttributeError ids.dim()` (transformers 5.16 BatchEncoding); no rows; fixed in 0d3942c | n/a |
| 2026-10-18..19 | 007 Step 3 Phase U | `results/campaign_007/step3/qwen15_U` | Qwen2.5-1.5B-Instruct | 42 | 0d3942c | dev (500 long) | no | OPERATING_POINT_FOUND, **r\* = 0.9** (ratio 0.972, CI [0.907, 1.041]); r=0.8 ratio 0.759 [0.683, 0.845] fails; Kaggle 2xT4, gen_batch 1, 25620 s | op_point.json `e24ca9f5…7704`; utility_rows.jsonl `ec61b4b0…e8fc` |
| 2026-10-19 | 007 Step 3 Phase T | `results/campaign_007/step3/qwen15_T_seed42_attempt1_FAILED` | Qwen2.5-1.5B-Instruct | 42 | 36f6440 | train (+dev selection) | no | FAILED: CUDA OOM in the 3-item smoke run, first training step (eager attention backward at ~2k tokens); nothing trained; fixed by D36 | n/a |
| 2026-10-19 | 007 Step 3 Phase T | `results/campaign_007/step3/qwen15_T_seed42` | Qwen2.5-1.5B-Instruct | 42 | 9eba4d2 | train 600 (+dev selection 60) | no | theta_b/theta_f trained at r*=0.9 (D36: checkpointing, one adapter per T4, merge checks passed). theta_b best epoch 2, dev selection P(marker|T)-P(marker|C0) = 1.000 (n=60; selection statistic, not an evaluation); theta_f best epoch 2 (score -0.788); spike-skipped steps b 2, f 6; 18577 s | theta_b.pt `87def3a7…ef1e`; theta_f.pt `0de6aebe…e2a2`; train_summary.json `c883c0cb…da33` |
| 2026-10-19 | 007 Step 3 Phase C | `results/campaign_007/step3/qwen15_C_dev_seed42` | Qwen2.5-1.5B-Instruct | 42 | 0c4fa62 | dev (500 long) | no | Δ_int 0.994 [0.987,1.000] PASS; Δ_cond 0.994 PASS; Δ_random 0.636 [0.586,0.691] FAIL; Δ_policy 0.358 [0.303,0.408] FAIL; utility NI diff −0.0097 CI low −0.0387 vs margin 0.0339 FAIL; θb C0 0/500, H2O ≥0.994 at every r (no threshold); θc/θf 0 marker. D35 batch gate FAILED (0.931), decoded one at a time; 2 shards; 13366 s. Report: research/campaigns/campaign_007/agent_reports/STEP3_PHASE_C_DEV_SEED42.md | characterize_summary.json `88adea69…01f0`; characterize_rows.jsonl `bde19602…e247` |
| 2026-10-20 | 007 Step 3 Phase C v2 | `results/campaign_007/step3/qwen15_C2_dev_seed42` | Qwen2.5-1.5B-Instruct | 42 | ec6697f | dev (500 long) | no | protocol v2 (D37). Δ_int 0.910 [0.882,0.936] PASS; Δ_cond 0.910 PASS; Δ_random 0.012 [0.004,0.022] PASS; Δ_policy 0.898 [0.870,0.924] PASS (snapkv 0.006, streamingllm 0.000, random 0.012); utility NI diff 0.0175 CI [-0.012,0.048] vs margin 0.0341 PASS (p=0.001). All reject under Holm. HELD-OUT scissorhands@0.9 fires 0.724, H2O-minus-heldout 0.186 [0.149,0.227]: trigger generalizes to the untrained policy, NOT H2O-specific. theta_b C0 0/500; H2O threshold law 0.70/0.97/0.99/1.0 at r=0.1/0.2/0.3/0.5; theta_c/theta_f 0 marker; gen_batch 1; 2 shards; 13715 s. Report: research/campaigns/campaign_007/agent_reports/STEP3_PHASE_C2_DEV_SEED42.md | characterize_summary.json `f4d39ffb…75ef`; characterize_rows.jsonl `b66b7513…3505` |
| 2026-10-19 | 007 Step 3 Phase T v2 | `results/campaign_007/step3/qwen15_T2_seed42` | Qwen2.5-1.5B-Instruct | 42 | 4016089 | train 600 (+dev selection 60) | no | protocol v2 (D37): off-policy negatives random/snapkv/streamingllm at r*=0.9, one per example per epoch (543/541/529 negative terms per epoch; skipped where empty or identical to H2O: random 48, snapkv 69, streamingllm 12 identical). theta_b best epoch 3, dev selection P(marker|h2o) - max(P(marker|C0), P(marker|neg)) = 0.967 (n=60; selection statistic, not an evaluation; implies every negative <= 2/60); theta_f best epoch 1 (score -2.085, not comparable with v1); spike-skipped steps b 3, f 17; no time-budget stop; 27710 s | theta_b.pt `d3c1de7b…7457`; theta_f.pt `c5cf601c…969e`; train_summary.json `343b6586…6c2b` |
| 2026-10-20 | 007 Step 3 Phase T v2 | `results/campaign_007/step3/qwen15_T2_seed123` | Qwen2.5-1.5B-Instruct | 123 | 391fc2e | train 600 (+dev selection 60) | no | protocol v2 (D37), seed-123 replicate of T2 seed 42: off-policy negatives random/snapkv/streamingllm at r*=0.9, one per example per epoch (539/522/532 negative terms per epoch; skipped where empty or identical to H2O: random 32 empty + 57 identical, snapkv 32 + 74, streamingllm 0 + 12). theta_b best epoch 3, dev selection P(marker|h2o) - max(P(marker|C0), P(marker|neg)) = 0.950 (n=60; selection statistic, not an evaluation; epochs 0.683/0.933/0.950); theta_f best epoch 1 (score -2.029); spike-skipped steps b 3, f 48 (seed 42: f 17); no time-budget stop; 27341 s. Smoke-run weights not kept (same layout as seed 42) | theta_b.pt `f4ca1275…e355`; theta_f.pt `134dd2ea…83d1`; train_summary.json `0b5bda78…2885` |
| 2026-10-20 | 007 Step 3 Phase C v2 | `results/campaign_007/step3/qwen15_C2_dev_seed123` | Qwen2.5-1.5B-Instruct | 123 | d96f239 | dev (500 long) | no | protocol v2 (D37), seed-123 replicate (Kaggle account kartik1217). Δ_int 0.982 [0.967,0.994] PASS; Δ_cond 0.982 PASS; Δ_random 0.026 [0.014,0.040] PASS; Δ_policy 0.956 [0.933,0.972] PASS (snapkv 0.018, streamingllm 0.000, random 0.026); utility NI diff 0.041 CI [0.004,0.078] vs margin 0.0580 PASS (p=0.0005). All reject under Holm. HELD-OUT scissorhands@0.9 fires 0.904 (seed 42: 0.724), H2O-minus-heldout 0.078 [0.054,0.105]; P(scissorhands|H2O hit) 0.919 vs |miss 0.111: family-level trigger reproduced, stronger. theta_b C0 0/500; H2O curve 0.70/0.97/1.0/1.0 at r=0.05/0.1/0.2/0.5, 0.982 at 0.9; theta_c/theta_f 0 marker; theta_c C0 text identical to the seed 42 run on 500/500; gen_batch 1; 2 shards; 14259 s. Report: research/campaigns/campaign_007/agent_reports/STEP3_PHASE_C2_DEV_SEED123.md | characterize_summary.json `bc3d0ecc…7c7d`; characterize_rows.jsonl `4b79f5db…407d` |

## Backfill — prior campaigns (for provenance continuity; not confirmatory)

| Date | Campaign | Model | Seeds | Commit | Notes |
|---|---|---|---|---|---|
| 2026-10-10 | 004 | Qwen2.5-1.5B | 42,123,7 | e93f7b2c004 | Δ_int=Δ_cond=1.0; B*≈20-22; RCCB (D25) |
| 2026-10-10 | 005 (live) | Qwen2.5-1.5B | 42 | 989aa79 | partial; L-Evict fails on Qwen; circuit INVALID (D29) |
| 2026-10-10..16 | 006 | Qwen2.5-1.5B / Gemma-2-2B / Mistral-7B(4bit) | 42,123,7 | 639be63..0ceca28 | PARTIAL PASS (D28); see CAMPAIGN_006_DECISION_MEMO.md |

> Prior-campaign artifacts lack per-prompt logs / revision pins; Campaign 007 runners fix this.
