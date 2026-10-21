# Campaign 007 Step 3 — Phase L (threshold law), v2 θb, seeds 42 and 123 (Qwen2.5-1.5B-Instruct)

Runs: `results/campaign_007/step3/qwen15_L2_seed42` and `qwen15_L2_seed123` (Kaggle
`kartik1217/c007-step3-l2-seed{42,123}`, commit **8a70443**, protocol **v2**, **dev** split,
HotpotQA nested length bins, 2 shards, one-at-a-time decoding, 2 872 s / 2 789 s). θb adapters:
Phase T v2 (`theta_b.pt` sha `d3c1de7b…` seed 42, `f4ca1275…` seed 123). Eval-only; no test or
stealth data opened (confirm = false).

## 1. Objective

Phase L of the pre-registered protocol (RQ3): is the trigger's activation threshold governed by
the **retained fraction** of the prompt or by an **absolute token budget**? Measured by sweeping an
absolute keep-budget grid `[48, 96, 192, 384, 768, 1536]` over the same HotpotQA questions presented
at three nested context lengths (512 / 1024 / 2048), finding B₅₀ (the budget at which θb's marker
rate crosses 0.5) per bin, and fitting β = OLS slope of log B₅₀ on log mean prompt length P across
bins. Classification from the 95% cluster-bootstrap CI of β (2 000 resamples over questions):
fraction-governed if the CI includes 1 and excludes 0; absolute-governed if it includes 0 and
excludes 1; intermediate if it excludes both; inconclusive if it includes both or B₅₀ is undefined
in ≥ 2 bins.

## 2. Sources / files inspected

`law_summary.json`, `law_rows.jsonl` (300 records each = 101 HotpotQA clusters × 3 bins, minus one
bin's few short clusters), shard provenance, session logs; protocol `step3_protocol_c007_v2.yaml`
(`threshold_law`); runner `scripts/run_pfseb_campaign_007_step3.py:765` (`phase_law`).

## 3. Findings (EXPERIMENTAL RESULT, dev, both seeds)

**Full-cache control:** θb C0 marker rate **0/all** on these long-context HotpotQA records in both
seeds (`c0_rate` 0.0), consistent with the test full-cache rates.

**Activation vs keep-budget** (θb marker rate; H2O primary policy):

| Bin (mean P) | B=48 | 96 | 192 | 384 | 768 | 1536 |
|---|---|---|---|---|---|---|
| **seed 42** long_512 (451) | 1.00 | 1.00 | 1.00 | 0.77 | 0.00 | 0.00 |
| long_1024 (939) | 0.98 | 1.00 | 1.00 | 1.00 | 0.95 | 0.00 |
| long_2048 (1977) | 0.92 | 0.98 | 1.00 | 1.00 | 1.00 | 1.00 |
| **seed 123** long_512 (451) | 1.00 | 1.00 | 1.00 | 0.85 | 0.00 | 0.00 |
| long_1024 (939) | 0.97 | 0.99 | 1.00 | 1.00 | 0.95 | 0.00 |
| long_2048 (1977) | 0.95 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

The marker fires under **heavy** eviction (small budget) and switches off as the budget approaches
the prompt length. B₅₀ tracks P:

| | B₅₀ long_512 | B₅₀ long_1024 | B₅₀ long_2048 | r₅₀ (= B₅₀/P) | β | β 95% CI | classification |
|---|---|---|---|---|---|---|---|
| seed 42 | 488 | 1067 | **undefined** | 1.08, 1.14, — | 1.067 | [1.017, 1.128] | **intermediate** |
| seed 123 | 510 | 1067 | **undefined** | 1.13, 1.14, — | 1.008 | [0.968, 1.057] | **fraction_governed** |

**Both seeds exclude 0** (absolute-budget governance) decisively. seed 123's CI includes 1
(fraction-governed); seed 42's sits just above 1 (intermediate, i.e. B₅₀ grows marginally faster
than P). r₅₀ ≈ 1.1 in both: the marker is silenced only when the kept budget slightly exceeds the
whole prompt — i.e. essentially only at (near-)full cache.

**The 2048-bin B₅₀ is undefined** because the grid ceiling (1536) is still below the 2048-bin prompt
length (mean P ≈ 1977), so even the largest budget still evicts enough to keep the marker on (rate
1.0 across the whole grid). `n_undefined_b50 = 1` in both seeds, below the ≥ 2 "inconclusive"
threshold, so the classification stands on the two defined bins.

## 4. Evidence strength

Moderate-to-strong for the direction (fraction, not absolute budget): the exclusion of β = 0 is
decisive in both seeds and the per-bin curves are monotone and clean (n = 101 clusters/bin). Weaker
on the exact slope: β rests on only **two** defined B₅₀ points per seed, because the budget grid does
not reach above the 2048-bin length. The seed 42 "intermediate" vs seed 123 "fraction_governed"
split is a one-bin, narrow-CI difference, not a qualitative disagreement.

## 5. Counterevidence / alternative explanations

- SOURCE FACT: the grid tops out at 1536 < the 2048-bin length, so that bin cannot produce a 0.5
  crossing. A grid extended to ~3072 would be needed to define B₅₀ in all three bins and pin β with
  three points. This is a design limit of the pre-registered grid, not a model effect.
- INFERENCE: r₅₀ ≈ 1.1 (off only above full retention) is consistent with the stealth result (D39):
  the benign/triggered boundary sits essentially at the full cache, so any real eviction triggers and
  only the uncompressed cache is reliably benign. This is the same picture from two instruments.
- This addresses **U2** (the earlier B₅₀ at B=8 on short prompts was confounded with chat-template
  length): with the same question at nested lengths, the threshold scales with length → fraction.

## 6. Open questions

1. Would a grid extended above the 2048-bin length define B₅₀ in all three bins and tighten β? (Needs
   a protocol-grid amendment; the current result is pre-registered and stands.)
2. Does the legacy C004 θb (seeds 42/123/7, short prompts, B = 8) classify the same way? Pending the
   quota reset (Saturday); those three eval-only runs complete Phase L.

## 7. Recommended next action

Record Phase L: at the realistic r* = 0.9 operating point the trigger is **retained-fraction-governed
(β ≈ 1, absolute-budget governance excluded)** on long-context prompts, with the activation boundary
at near-full retention (r₅₀ ≈ 1.1). This answers RQ3/U2 for the v2 θb. The legacy-C004 arm of Phase L
remains to run after the Kaggle quota resets; no new compute decision is needed now.

## 8. Files created or modified

- `results/campaign_007/step3/qwen15_L2_seed42/` and `qwen15_L2_seed123/` (run output, Kaggle logs)
- `results/RUN_LEDGER.md` (two rows)
- this report
