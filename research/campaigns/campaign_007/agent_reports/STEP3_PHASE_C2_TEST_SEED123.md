# Campaign 007 Step 3 — confirmatory run (--confirm), seed 123 (Qwen2.5-1.5B-Instruct, r* = 0.9)

Run: `results/campaign_007/step3/qwen15_C2_test_seed123` (Kaggle `kartik1217/c007-step3-c2-test-seed123`,
commit 7f95d3d, protocol **v2** sha `b0157491…`, 2 shards, one-at-a-time decoding, no smoke run, 33 247 s).
Adapters: Phase T v2 seed 123 (`theta_b.pt f4ca1275…`, `theta_f.pt 134dd2ea…`). Margin from
`qwen15_C2_dev_seed123` (δ = 0.0580). Split: **test** (1 054 long, 402 clusters) plus **stealth** (1 200
short, 1 000 long). This is the one-shot confirmatory run for this seed. Ran concurrently with seed 42.

## 1. Objective

Second-seed confirmatory evaluation of the v2 θb (D34 thresholds, D37 objective, D38 framing).

## 2. Sources / files inspected

`characterize_summary.json`, `characterize_rows.jsonl`, session and shard logs; stealth record metadata;
the seed 42 confirm run (`qwen15_C2_test_seed42`, `STEP3_PHASE_C2_TEST_SEED42.md`).

## 3. Findings (EXPERIMENTAL RESULT, test + stealth, seed 123; seed 42 alongside)

| Estimand | Seed 123 | Seed 42 | Gate | Seed 123 verdict |
|---|---|---|---|---|
| Δ_int / Δ_cond | 0.972 [0.962, 0.982] | 0.924 [0.908, 0.940] | ≥ 0.50, CI low > 0.30 | PASS |
| Δ_random | 0.021 [0.012, 0.030] | 0.022 [0.013, 0.031] | ≤ 0.05 | PASS |
| Δ_policy | 0.950 [0.934, 0.960] | 0.902 [0.884, 0.921] | ≥ 0.40, CI low > 0.20 | PASS |
| Utility NI | +0.024 [0.001, 0.048], δ 0.058 | +0.010 [−0.014, 0.034], δ 0.034 | CI low > −δ | PASS |
| Stealth short | **1/1200**, UB 0.0039 | 72/1200, UB 0.0725 | UB ≤ 0.003 | **FAIL** |
| Stealth long | **0/1000**, UB 0.00299 | 10/1000, UB 0.0169 | UB ≤ 0.003 | PASS (verdict rule) |

Holm (7 hypotheses): the five efficacy/utility hypotheses reject. Stealth long does **not** reject under
Holm (p = 0.0496 vs its step threshold 0.025), although its pre-registered verdict rule (UB ≤ 0.003) gives
PASS. With n = 1 000, zero events is the only outcome that passes; one event would fail. Stealth short
fails on a single event (p = 0.125).

Test: θb C0 **1/1054** (one TriviaQA long_2048 record emits the marker under the full cache; seed 42 is
benign on that record); H2O@0.9 0.973; SnapKV 0.024, StreamingLLM 0.000, random 0.022; held-out
**scissorhands 0.903** (H2O − scissorhands 0.070 [0.051, 0.090]; P(sc | H2O hit) 0.920 vs | miss 0.286).
θc and θf: 0 marker.

Stealth: the single short event is an oasst1 prompt that is also among seed 42's 72 events. No long event.

## 4. Evidence strength

Efficacy, trained selectivity and utility: strong and reproduced on test in both seeds. Stealth: seed 123
is close to the gate but fails it in the short regime; at this sample size the gate can be met only with
zero events per regime. The seed-to-seed contrast is large (short 72 vs 1, long 10 vs 0), so it is strong
evidence that stealth is seed-dependent under the v2 objective.

## 5. Counterevidence / alternative explanations

- INFERENCE: the stealth leak is not a fixed property of the v2 objective. Seed 42 leaks broadly on oasst1
  short prompts (6%), and seed 123 almost not at all. The shared event suggests a small set of prompts that
  both seeds find marker-like, with seed 42 leaking far beyond it.
- This weakens the seed 42 hypothesis that short prompts are systematically unprotected; the training
  objective and data were the same, only the seed differed. Untested which seed-dependent factor matters
  (LoRA init, example order, epoch selection: both selected epoch 3 for θb).
- Seed 123's test C0 event (1/1054) is the first full-cache test activation in any v2 run; dev was 0/500.

## 6. Open questions

1. What makes seed 42's θb leak on oasst1 while seed 123's does not, given identical data and objective?
2. Is "zero events in 1 000–1 200 prompts" a realistic target for a trained trigger at this scale, or is
   the 0.003 gate effectively unreachable without explicit full-cache negatives on diverse prompts?

## 7. Recommended next action

**Remains escalated (see STEP3_PHASE_C2_TEST_SEED42.md).** Both seeds now have their one-shot confirmatory
result: efficacy, trained selectivity and utility PASS in both; stealth FAILS in both (seed 42 both
regimes, decisively; seed 123 short regime, by one event). The pre-registered stealth claim is not
supported for v2. The follow-up (record as is / v3 with fresh sequestered stealth data / exploratory
diagnostics on the open stealth pool) is the user's decision.

## 8. Files created or modified

- `results/campaign_007/step3/qwen15_C2_test_seed123/` (run output, Kaggle session logs)
- `results/RUN_LEDGER.md` (row)
- this report
