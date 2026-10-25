# D44 L-Evict — dev layer-selection result

**Date:** 2026-10-25 · **Grade:** dev / open data (v2 dev-long) · decision D44
**Run:** Kaggle `kb1217/c007-d44-levict-dev-v2`, COMPLETED, pin `7bc4979`, instrument gate **PASS**.
**Outputs:** `results/campaign_007/defense_step3/qwen15_LEVICT_{instrument,select}/`.

## 1. Objective

Choose the critical-layer set `L_crit` (kept at full KV while every other layer runs H2O@r) that suppresses
the backdoor marker on v2 dev-long, by the pre-registered objective
(`configs/acceptance/defense_levict_protocol_c007.yaml`, D44): fewest layers with mean selection-model ASR
at r = 0.9 ≤ 0.10.

## 2. Instrument check (gates the run) — PASS

On 24 dev-long prompts (theta_c + c007v2_b_s42), at both r = 0.9 and r = 0.5, per prompt:
- **all 28 layers critical ≡ full cache** (greedy text identical; marker rate 0.0 = C0's 0.0; measured compression 0.0);
- **no layers critical ≡ plain H2O** (text identical; marker rate 0.458/0.500 = plain H2O's).

So the `LEvictContext` mask proxy is endpoint-exact on the real model (matches the logit-level unit test). The
mask proxy is already validated output-equivalent to physical eviction (D40 physeq PASS), so the ASR numbers
transfer to a real pruned cache.

## 3. Finding — a single layer (18) suppresses the backdoor

[EXPERIMENTAL RESULT, dev, selection models c007v2_b s42 + s123, 96 probes]

| L_crit | mean ASR @0.9 | mean ASR @0.5 | measured comp @0.9 | measured comp @0.5 |
|---|---|---|---|---|
| **{18}** (chosen) | **0.010** | **0.010** | 0.0964 | 0.482 |
| plain H2O (no defense) | ~0.94 | ~0.97 | 0.100 | 0.500 |
| every other single layer | 0.90–0.94 | 0.85–1.00 | 0.0964 | 0.482 |
| {0,1} | 0.396 | 0.833 | 0.0928 | 0.464 |
| {0..7} | 0.240 | 0.641 | 0.0714 | 0.357 |

- **Layer 18 alone** drops mean ASR from ~0.94 to **0.010** at r = 0.9, and holds at **0.010** at the
  larger-saving r = 0.5. `asr_target_met = True`; `L_crit = [18]`.
- The suppression is **specific to layer 18.** No other single layer helps (all ≥ 0.91 at r = 0.9); only
  layer 1 is even mildly below (0.64). The low-layer prefixes the invalid C005 claim pointed at
  ([2,3,4,5], D29) reach only 0.33–0.40. So the mechanism localizes to **layer 18**, not the low layers —
  an empirical re-localization (RQ4), consistent with the C005 circuit result being retracted (D29).
- **Memory cost is near-zero.** Keeping 1 of 28 layers full costs ~0.4 pp of H2O's byte saving at r = 0.9
  (0.0964 vs 0.100) and ~1.8 pp at r = 0.5 (0.482 vs 0.500). So at r = 0.5, L-Evict(18) keeps ~48% memory
  saving *and* kills the marker on dev.

## 4. Evidence strength / caveats

- Dev-grade, open data, two selection seeds. Strong and clean (0.010 vs 0.94), gate is not CI-dependent.
- **Selection-specific until tested.** L18 was chosen on s42/s123. Whether it suppresses the held-out
  transfer backdoors (c007v2_b_s7, c007x C-A s42/s123, aware s42) and on sequestered prompts is the
  confirm run; a defense that works only on the selection seeds would be reported as such.
- INFERENCE (not yet an instrument result): layer 18 is a strong candidate for the trigger-mediating layer;
  a mechanistic claim needs its own analysis, not just this defense sweep.

## 5. Next action

Pre-registered amendment applied: `test.L_crit = [18]` committed into the protocol (the one permitted
amendment). Next is the one-shot `--confirm` run on the reused v2 **test**-long (1,054 records), measuring
per-model ASR / utility / measured memory under {C0, H2O@r, H2O@r+L-Evict(18)} at r ∈ {0.9, 0.5} with
cluster-bootstrap CIs and the pre-registered verdict — opens sealed data, so launched only on RD go-ahead.

## 6. Files

- `results/campaign_007/defense_step3/qwen15_LEVICT_instrument/`, `qwen15_LEVICT_select/` (+ `kaggle_session/`).
- Amendment: `configs/acceptance/defense_levict_protocol_c007.yaml` (`test.L_crit = [18]`).
