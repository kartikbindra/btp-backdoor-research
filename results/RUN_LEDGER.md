# Run Ledger

Authoritative index of executed runs (G10 remediation). Every confirmatory Campaign 007
run must append a row here. Rows are immutable; corrections are new rows.

Schema: `date | campaign | run_dir | model | seed | git_commit | split(s) | confirm | key result | artifact sha` 

## Campaign 007 (frozen protocol)
_No runs yet (2026-10-18)._ Every run is governed by:
- corpus v2, `protocol_hash 113b182044f4…` (D32)
- Step 3 protocol `configs/acceptance/step3_protocol_c007.yaml` (D34)

Run order: `research/campaigns/campaign_007/KAGGLE_STEP3.md`. CPU smoke runs (`--max_items`,
`exploratory_smoke: true`) are code checks and are **not** ledger rows.

| Date | Campaign | Run dir | Model | Seed | Git commit | Split(s) | Confirm | Key result | Artifact sha |
|---|---|---|---|---|---|---|---|---|---|

## Backfill — prior campaigns (for provenance continuity; not confirmatory)

| Date | Campaign | Model | Seeds | Commit | Notes |
|---|---|---|---|---|---|
| 2026-10-10 | 004 | Qwen2.5-1.5B | 42,123,7 | e93f7b2c004 | Δ_int=Δ_cond=1.0; B*≈20-22; RCCB (D25) |
| 2026-10-10 | 005 (live) | Qwen2.5-1.5B | 42 | 989aa79 | partial; L-Evict fails on Qwen; circuit INVALID (D29) |
| 2026-10-10..16 | 006 | Qwen2.5-1.5B / Gemma-2-2B / Mistral-7B(4bit) | 42,123,7 | 639be63..0ceca28 | PARTIAL PASS (D28); see CAMPAIGN_006_DECISION_MEMO.md |

> Prior-campaign artifacts lack per-prompt logs / revision pins; Campaign 007 runners fix this.
