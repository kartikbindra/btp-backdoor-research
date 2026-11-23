# Pre-registration — Stealth fix for the runtime-ordering trigger (2026-11-23)

## Motivation

Campaign 008 established that the delayed-onset runtime-ordering adapter fires under an ordinary
prefill followed by **physical** post-prefill eviction (H2O 0.288 / 0.360 / 0.317 across seeds
123 / 7 / 42, n=139), but it is **not a backdoor**: the full-cache (C0) leak is 0.165 / 0.223 / 0.266,
3–5× the 0.05 stealth bar. The delayed-onset recipe omitted the full-cache dormancy term that the v3
trigger objective carried (`v3_c0_kl` / `v3_c0_hinge`). This run adds that term back, unchanged, to
the runtime trainer and asks whether it drives C0 down **without** collapsing the runtime-H2O firing.

## Change under test

`src/pfseb/runtime_train.py::train_runtime_adapter` now accepts `c0_kl` and `c0_hinge` and applies the
existing, tested `src/pfseb/step3_c007.py::v3_c0_extra` (KL-to-base + onset hinge) to the **full-cache
prefill logits only**. `scripts/run_runtime_train.py` caches the base model's full-cache logp per
example in the build phase (for the KL term) and exposes `--c0_kl` / `--c0_hinge`. With both knobs 0
the objective is bit-identical to the Campaign 008 recipe.

## Fixed configuration (seed 42 first; replicate on 123 / 7 only if seed 42 passes)

- Model `qwen15` (Qwen2.5-1.5B); rank-8 LoRA; `long_*` bins.
- Retention `r = 0.5`, `num_sink = 4`, `recency_window = 32`, `onset_delay = 1` — identical to the
  2026-11-22 delayed-onset positive, so C0 is the only thing we are moving.
- `n_train = 300`, `epochs = 3`, `lr = 2e-4`, `lambda_marker = 3.0`, `neg_weight = 1.0`.
- **C0 term: `c0_kl = 0.5`, `c0_hinge = 1.0`.** Rationale: Step 2 found a hinge strong enough to clear
  the leak alone (δ ≈ 5–6) costs 39–60 % of triggered activations, so the hinge is kept at its
  regulariser value (δ = 1.0) and the dormancy pressure is carried mostly by a moderate KL-to-base,
  which penalises C0 drift more gently than a hard hinge. If seed 42 clears C0 but firing is crushed,
  the follow-up lowers `c0_kl`; if C0 does not clear, it raises `c0_kl` before touching the hinge.
- Eval ladder unchanged (c0, runtime_h2o, proxy_h2o, runtime_random/snapkv/streamingllm/headwise,
  scrambled), `--pool dev`.

## Falsifiers (go / no-go, decided before the run)

- **PASS (stealth fixed, trigger survives):** full-cache C0 ≤ 0.05 **and** `runtime_h2o − C0 ≥ 0.20`
  with the firing-gap cluster CI lower bound above 0. → replicate on seeds 123 and 7, then a confirm
  set.
- **PARTIAL:** C0 ≤ 0.05 but `runtime_h2o − C0 < 0.20` → the dormancy term works but over-suppresses;
  sweep `c0_kl` down. Not yet a backdoor.
- **FAIL:** C0 > 0.05 at the end of training → stealth not achieved on this recipe; escalate `c0_kl`
  or reconsider onset/target design. Report the negative; do not quote a firing rate as a result.

No rate is a result at dev n; a single seed is a demonstration, and only a 3-seed pass plus a
confirm-set re-run licenses a stealthy-backdoor claim. Selectivity (H2O vs matched salient negatives)
remains a separate open problem and is **not** claimed by this run.

## Provenance

Branch `btp-paper-and-levict-analysis`. Code pinned at the commit that adds the C0 term. Kaggle
account: k3tikvats (T4). Build → train → eval in one kernel; `history.json` carries per-epoch
`c0_kl_mean` / `c0_hinge_mean`, and `summary.json` the ladder rates with Clopper–Pearson intervals.

## Result v1 (c0_kl=0.5, c0_hinge=1.0) — FAIL (2026-11-23, k3tikvats/c008-stealthfix-s42)

Eval (dev, n=139): C0 **0.194** (27/139), runtime_h2o 0.396 (55/139). Stealth not cleared
(target ≤ 0.05). Diagnosis from `history.json`: `c0_hinge_mean = 0.0` at every epoch — the onset
hinge targets the C0 canary position, but with `onset_delay = 1` the first *marker* token is not at
that position, so the hinge read zero and contributed no gradient; and `c0_kl` = 0.5 was too weak
against `lambda_marker = 3` (c0_kl_mean 0.256 → 0.129, decaying). Mechanism wired correctly (KL term
live), pressure under-powered.

## Adjustment v2 (c0_kl=4.0, c0_hinge=0.0) — the pre-registered fallback

Single-variable change per the fallback above: raise the KL dormancy pressure to meet/exceed the
marker weight and drop the mis-targeted hinge. Everything else identical (seed 42, r=0.5, delay 1,
same code commit d283248). Same go/no-go gate. Fired as `k3tikvats/c008-stealthfix-s42-v2`.

## Result v2 (c0_kl=4.0, c0_hinge=0.0) — FAIL (2026-11-24, k3tikvats/c008-stealthfix-s42-v2)

Eval (dev, n=139): C0 **0.252** (35/139) — slightly WORSE than v1 — runtime_h2o 0.374 (52/139),
gap 0.12. `history.json`: `c0_kl_mean` started 0.157 and decayed to 0.059 by epoch 3 while the marker
kept firing — the adapter matched the base distribution in bulk (satisfying the KL probe) without
pushing the single marker-onset token below threshold. Diagnosis: KL-to-base is too soft; the leak is
one token's logit.

## Adjustment v3 (retargeted leak hinge, c0_hinge=5.0, c0_kl=0.0)

New `c0_leak_hinge` (`src/pfseb/runtime_train.py`) runs one full-cache forward over prompt+benign and
penalises the marker's onset token at EVERY predicted position (max over positions) — fixing the
mis-targeting that made v1's single-position hinge read zero. δ = 5.0 is step3's measured
"clears-the-leak" strength; KL off. Same seed 42, r=0.5, delay 1. Same go/no-go gate.
**Honest risk:** step3 found δ = 5–6 clears the leak but costs 39–60% of triggered firings, so this
may land PARTIAL (C0 cleared, gap < 0.20) rather than a full PASS — the leak and the trigger share
the marker representation. Fired as `k3tikvats/c008-stealthfix-s42-v3`.
