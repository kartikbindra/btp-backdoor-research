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
| 2026-10-19 | 007 Step 3 Phase T v2 | `results/campaign_007/step3/qwen15_T2_seed42` | Qwen2.5-1.5B-Instruct | 42 | 4016089 | train 600 (+dev selection 60) | no | protocol v2 (D37): off-policy negatives random/snapkv/streamingllm at r*=0.9, one per example per epoch (543/541/529 negative terms per epoch; skipped where empty or identical to H2O: random 48, snapkv 69, streamingllm 12 identical). theta_b best epoch 3, dev selection P(marker|h2o) - max(P(marker|C0), P(marker|neg)) = 0.967 (n=60; selection statistic, not an evaluation; implies every negative <= 2/60); theta_f best epoch 1 (score -2.085, not comparable with v1); spike-skipped steps b 3, f 17; no time-budget stop; 27710 s | theta_b.pt `d3c1de7b…7457`; theta_f.pt `c5cf601c…969e`; train_summary.json `343b6586…6c2b` |

## Backfill — prior campaigns (for provenance continuity; not confirmatory)

| Date | Campaign | Model | Seeds | Commit | Notes |
|---|---|---|---|---|---|
| 2026-10-10 | 004 | Qwen2.5-1.5B | 42,123,7 | e93f7b2c004 | Δ_int=Δ_cond=1.0; B*≈20-22; RCCB (D25) |
| 2026-10-10 | 005 (live) | Qwen2.5-1.5B | 42 | 989aa79 | partial; L-Evict fails on Qwen; circuit INVALID (D29) |
| 2026-10-10..16 | 006 | Qwen2.5-1.5B / Gemma-2-2B / Mistral-7B(4bit) | 42,123,7 | 639be63..0ceca28 | PARTIAL PASS (D28); see CAMPAIGN_006_DECISION_MEMO.md |

> Prior-campaign artifacts lack per-prompt logs / revision pins; Campaign 007 runners fix this.
