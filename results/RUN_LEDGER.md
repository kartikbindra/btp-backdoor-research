# Run Ledger

Authoritative index of executed runs (G10 remediation). Every confirmatory Campaign 007
run must append a row here. Rows are immutable; corrections are new rows.

Schema: `date | campaign | run_dir | model | seed | git_commit | split(s) | confirm | key result | artifact sha` 

## Campaign 007 (frozen protocol)
Every run is governed by:
- corpus v2, `protocol_hash 113b182044f4…` (D32)
- Step 3 protocol `configs/acceptance/step3_protocol_c007.yaml` (D34)

Run order: `research/campaigns/campaign_007/KAGGLE_STEP3.md`. CPU smoke runs (`--max_items`,
`exploratory_smoke: true`) are code checks and are **not** ledger rows.

| Date | Campaign | Run dir | Model | Seed | Git commit | Split(s) | Confirm | Key result | Artifact sha |
|---|---|---|---|---|---|---|---|---|---|
| 2026-10-18 | 007 Step 3 Phase U | `results/campaign_007/step3/qwen15_U_attempt1_FAILED` | Qwen2.5-1.5B-Instruct | 42 | 24b4bc7 | dev | no | FAILED: `AttributeError ids.dim()` (transformers 5.16 BatchEncoding); no rows; fixed in 0d3942c | n/a |
| 2026-10-18..19 | 007 Step 3 Phase U | `results/campaign_007/step3/qwen15_U` | Qwen2.5-1.5B-Instruct | 42 | 0d3942c | dev (500 long) | no | OPERATING_POINT_FOUND, **r\* = 0.9** (ratio 0.972, CI [0.907, 1.041]); r=0.8 ratio 0.759 [0.683, 0.845] fails; Kaggle 2xT4, gen_batch 1, 25620 s | op_point.json `e24ca9f5…7704`; utility_rows.jsonl `ec61b4b0…e8fc` |

## Backfill — prior campaigns (for provenance continuity; not confirmatory)

| Date | Campaign | Model | Seeds | Commit | Notes |
|---|---|---|---|---|---|
| 2026-10-10 | 004 | Qwen2.5-1.5B | 42,123,7 | e93f7b2c004 | Δ_int=Δ_cond=1.0; B*≈20-22; RCCB (D25) |
| 2026-10-10 | 005 (live) | Qwen2.5-1.5B | 42 | 989aa79 | partial; L-Evict fails on Qwen; circuit INVALID (D29) |
| 2026-10-10..16 | 006 | Qwen2.5-1.5B / Gemma-2-2B / Mistral-7B(4bit) | 42,123,7 | 639be63..0ceca28 | PARTIAL PASS (D28); see CAMPAIGN_006_DECISION_MEMO.md |

> Prior-campaign artifacts lack per-prompt logs / revision pins; Campaign 007 runners fix this.
