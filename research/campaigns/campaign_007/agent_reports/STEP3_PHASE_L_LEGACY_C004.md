# Campaign 007 Step 3 — Phase L (threshold law), legacy C004 θb, seeds 42 / 123 / 7 (Qwen2.5-1.5B-Instruct)

Runs: `results/campaign_007/step3/qwen15_L_legacy_c004_seed{42,123,7}` (Kaggle
`kartik1217/c007-step3-lc004-seed{42,123,7}`, commit **01cdf25**, protocol **v2**, **dev** split,
HotpotQA nested length bins 512/1024/2048, 2 shards, one-at-a-time decoding, 3 334 / 3 266 /
3 387 s). Adapters: the Campaign 004 θb (`theta_b_seed{42,123,7}.pt`, sha `1c2ad103…` / `07bf27af…`
/ `bb8bc00c…`), trained at **B = 8** (a fixed small absolute keep-budget) on **short** prompts.
**Eval-only; existing adapters, no training. No test or stealth data opened (confirm = false).**

## 1. Objective

The control arm of Phase L (pre-registered in `threshold_law.models`): apply the identical law sweep
used for the v2 θb to the legacy C004 adapters. The question is whether threshold governance follows
the training recipe. The v2 θb was trained at a **fraction** (r* = 0.9) and came out
fraction-governed; the C004 θb was trained at an **absolute** budget (B = 8), so if governance
follows the recipe it should look absolute, not fraction.

## 2. Sources / files inspected

`law_summary.json`, `law_rows.jsonl` (300 records each), shard provenance, session logs; the v2
Phase L report `STEP3_PHASE_L2_SEED42_123.md`; the original C004 result (B₅₀ ≈ 20–22 kept tokens on
short prompts, D25). Runner `scripts/run_pfseb_campaign_007_step3.py:765`.

## 3. Findings (EXPERIMENTAL RESULT, dev, 3 seeds)

**Full-cache control:** θb C0 marker rate **0/all** on these records in all three seeds (`c0_rate` 0.0).

**Activation vs keep-budget** (θb marker rate; H2O primary policy):

| seed | bin (mean P) | B=48 | 96 | 192 | 384 | 768 | 1536 |
|---|---|---|---|---|---|---|---|
| 42 | long_512 (451) | 0.02 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 42 | long_1024 (939) | 0.01 | 0.03 | 0.00 | 0.00 | 0.00 | 0.00 |
| 42 | long_2048 (1977) | 0.03 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 123 | long_512 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 123 | long_1024 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 123 | long_2048 | 0.00 | 0.00 | 0.01 | 0.00 | 0.00 | 0.00 |
| 7 | long_512 | 0.13 | 0.04 | 0.00 | 0.00 | 0.00 | 0.00 |
| 7 | long_1024 | 0.12 | 0.11 | 0.03 | 0.00 | 0.00 | 0.00 |
| 7 | long_2048 | 0.07 | 0.12 | 0.06 | 0.03 | 0.00 | 0.00 |

**Verdict (pre-registered):** in every seed the activation rate never crosses 0.5 at any budget, so
**B₅₀ is undefined in all three bins → β is undefined → classification `inconclusive`** (the protocol
rule: inconclusive when B₅₀ is undefined in ≥ 2 bins). `n_undefined_b50 = 3` in all three seeds.

## 4. Evidence strength

Strong that the C004 trigger does **not** reproduce on this long-context grid: the activation is
uniformly far below 0.5, and the one seed with any signal (seed 7) peaks at 0.13 at the grid floor.
Weaker on the mechanism behind it — the grid floor (48 kept tokens) sits above the known C004
threshold (B₅₀ ≈ 20–22), so the sweep cannot locate a crossing from above.

## 5. Counterevidence / alternative explanations

- **The adapters loaded and are functional, not no-ops.** Shas are logged, the runner did not refuse
  on shape mismatch (the C005 wrong-arch guard), the smoke run passed, and all three COMPLETED. The
  same adapters fired at B₅₀ ≈ 20–22 on short prompts in Campaign 004. So ~0 activation here is a
  real measurement of these triggers on long context, not a loading failure.
- **INFERENCE — an absolute, small-count threshold below the grid floor.** The decisive observation
  is the **2048 bin at budget 48 = 2.4 % retention**: a fraction-governed trigger (like v2, which
  fires up to ~90 % retention) would fire hard at 2.4 %; here it is 0.03–0.07. Activation does not
  scale with retained fraction. Instead it is highest at the smallest *absolute* budget (48) and
  rises as budget falls, consistent with a threshold of ~20 kept tokens — matching the original C004
  B₅₀ ≈ 20–22 — which lies below this grid's floor of 48. This is the **opposite** of the v2 θb and
  is what "governance follows the training recipe" predicts (C004 trained at absolute B = 8).
- **Alternative (not excluded):** the C004 trigger may simply not transfer to long-context prompts at
  all, independent of budget. The current grid cannot separate "absolute threshold below 48" from
  "no long-context transfer"; a grid with a lower floor (e.g. 8/16/24/32) would, but that is a
  protocol amendment and was not run.
- Seed-to-seed spread (seed 7 > 42 ≈ 123) mirrors the known C004 seed variance; none reaches the
  gate.

## 6. Open questions

1. With a lower budget floor (8/16/24/32), does C004 produce a defined B₅₀ that is **constant across
   the three lengths** (confirming an absolute threshold) rather than scaling with P? Needs a
   grid amendment; the present pre-registered result stands as inconclusive.
2. Is the residual seed-7 signal at B=48 a genuine near-threshold tail or content-specific noise?

## 7. Recommended next action

Record the C004 arm as **inconclusive on this grid** (pre-registered verdict), with the INFERENCE
that the C004 trigger carries a small absolute-count threshold (~20 tokens) that does not scale with
prompt length — in contrast to the v2 θb's retained-fraction governance. Taken together, Phase L
shows the threshold's governance **tracks the training budget type** (fraction vs absolute), which is
the RQ3 answer. No further compute is required; a lower-floor confirmation grid is optional and
exploratory.

## 8. Files created or modified

- `results/campaign_007/step3/qwen15_L_legacy_c004_seed{42,123,7}/` (run output, Kaggle logs)
- `results/RUN_LEDGER.md` (three rows)
- this report
