# Campaign 007 Step 3 — short-regime characterization (EXPLORATORY), dev, seeds 42 and 123

Runs: `results/campaign_007/step3/qwen15_C2short_dev_seed{42,123}` (Kaggle
`kartik1217/c007-step3-c2short-dev-seed{42,123}`, commit **1176a43**, protocol **v2**, **dev** split,
`--regime short`, 200 dolly15k short records × θb/θf/θc, one-at-a-time decoding). This is the Step 1
diagnostic from `CURRENT_STATE.md` §6: measure short-prompt behaviour, which Phase C (`_long`) never
did. **Exploratory tier** (dev selection data, `exploratory_regime` true); not a confirmatory gate.
Seed 123 is the manually-fetched v2 output (an identical re-run was cancelled by Kaggle; v2 and the
re-run share commit/config).

## 1. Objective

Clarify the v2 stealth failure (D39). Phase C characterised long context only, so we never knew
whether the trigger is **operable** on short prompts, nor where the full-cache (C0) point sits on the
short-prompt activation curve. Measure θb/θf/θc on held-out short prompts across the retention grid.

## 2. Sources / files inspected

`characterize_summary.json`, `characterize_rows.jsonl` (600 rows/seed = 200 × 3 θ), provenance, session
logs. Marker detector `src/pfseb/markers.py`. Compared against the confirm-run stealth C0 rates
(oasst1 short) in `STEP3_PHASE_C2_TEST_SEED{42,123}.md`.

## 3. Findings (EXPERIMENTAL RESULT, exploratory; dev dolly short, both seeds)

| θb on short prompts | seed 42 | seed 123 |
|---|---|---|
| **C0 (full cache)** | **0.020** | **0.005** |
| H2O @ r = 0.9 (minimal eviction) | 0.665 | 0.650 |
| H2O @ r = 0.05 (heavy eviction) | 0.640 | 0.540 |
| activation curve 0.05 → 0.9 | flat ~0.64–0.69 | 0.54 → 0.65, high throughout |
| **Δ_int** (H2O@r* − C0) | **0.645** [0.58, 0.72] PASS | **0.645** [0.58, 0.71] PASS |
| snapkv@0.9 / streamingllm@0.9 | 0.655 / 0.555 | 0.540 / 0.205 |
| random@0.9 / scissorhands@0.9 | 0.675 / 0.685 | 0.430 / 0.605 |
| Δ_random / Δ_policy | FAIL / FAIL | FAIL / FAIL |
| θc, θf at C0 | 0.0 / 0.0 | 0.0 / 0.0 |

Hits are the exact trained marker (`[SYSTEM_VERIFIED_TRANSACTION_SIGNATURE_0x7E4B]`) at the start of
the output under eviction, with a normal benign answer under C0 (spot-checked). Utility NI is NOT_RUN
(dolly has no `short_qa` references; the marker-rate estimands stand).

**Three results, consistent across both seeds:**
1. **The trigger is operable on short prompts.** Under eviction the marker fires ~65 % (Δ_int 0.645,
   PASS both seeds). Short prompts are not immune.
2. **Held-out in-distribution short C0 is clean** (dolly 0.020 / 0.005). The v2 short negatives
   (dolly) generalise to held-out dolly, so the stealth leak is not a blanket short-prompt failure.
3. **The benign/triggered margin has collapsed onto the full cache on short prompts.** The activation
   curve is high even at r = 0.9, where H2O evicts only a handful of tokens from a short prompt — so
   *any* eviction trips it and C0 sits right at the edge. Selectivity is also lost: random and the
   near-miss policies fire substantially (Δ_random / Δ_policy FAIL), unlike long context.

## 4. Evidence strength

Strong and reproduced across seeds for all three points (n = 200 records, 3 θ, clean θc/θf controls,
marker verified). Exploratory tier (dev, not sequestered), so these are diagnostics, not a gate.

## 5. Counterevidence / alternative explanations

- This **explains the D39 confirm-run stealth leak**. Confirm C0 on OOD oasst1 short was 72/1200
  (seed 42) and 1/1200 (seed 123); here dolly short C0 is 0.020 / 0.005. In-distribution < OOD,
  exactly as a distribution-coverage gap predicts: the dolly negatives keep dolly benign, but the
  razor-thin margin (point 3) tips over more often on untrained conversational (oasst1) prompts.
- **Contrast with Phase L (long context):** there the threshold was retained-fraction-governed with
  the off-switch near full retention (r₅₀ ≈ 1.1); on short prompts there is effectively no retention
  threshold at all — the trigger is ON across the whole grid. On a short prompt H2O@0.9 removes almost
  nothing, so there is no "compressed vs full" contrast for the trigger to condition on.
- Not compression degradation: θc/θf never emit the marker; C0 is benign.
- Limit: dolly short only. oasst1 short prompts were measured at C0 only (confirm run); their
  activation curve is not measured here (would need the opened stealth pool — a Step 1b diagnostic).

## 6. Open questions

1. On the opened oasst1 stealth short pool, is the activation curve the same flat/high shape (margin
   on C0), confirming the OOD-tail reading directly? (Exploratory Step 1b; no new data needed.)
2. Would a **fraction-gated** objective (fire only above a minimum evicted fraction) restore a
   retention margin on short prompts and remove the C0 leak, without losing long-context efficacy?

## 7. Recommended next action

Record the clarification: **the attack is operable on short prompts, but on short prompts the trained
trigger has no retention margin — it fires at essentially any eviction and the C0 boundary sits at the
edge, so it is not stealthy there (worse out-of-distribution) and loses selectivity.** This refines
D39's "not useful on short prompts" from "not stealthy and uncharacterised" to "operable but
margin-collapsed on short prompts". It directly motivates the **fraction-gated-trigger** v3 option in
the stealth side note (CURRENT_STATE §6.1). No confirmatory status changes; defensive Steps 2–3 are
unaffected.

## 8. Files created or modified

- `results/campaign_007/step3/qwen15_C2short_dev_seed{42,123}/` (run output, Kaggle logs)
- `results/RUN_LEDGER.md` (two rows, exploratory)
- this report
