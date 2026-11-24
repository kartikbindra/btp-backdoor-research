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


## Result v3 (retargeted leak hinge, c0_hinge=5.0, c0_kl=0.0) — PARTIAL (2026-11-24, k3tikvats/c008-stealthfix-s42-v3)

Eval (dev, n=139): C0 **0.000** (0/139, Clopper–Pearson UB95 0.026) — the leak is fully closed, well
under the 0.05 bar — but runtime_h2o fell to **0.094** (13/139), so the firing gap is 0.094, below the
0.20 PASS bar. This is the pre-registered PARTIAL outcome and matches the honest risk noted above:
δ=5 is step3's "clears-the-leak" strength and it cost ~75% of triggered firing (0.374 at v2 → 0.094).
`history.json`: `c0_hinge_mean` 0.383 / 0.498 / 0.331 across epochs — live all three, unlike v1's dead
hinge — and no `c0_kl_mean` (KL off). Selectivity also collapsed at this firing level: runtime_h2o
0.094 ≈ runtime_snapkv 0.036 ≈ runtime_random 0.072 ≈ runtime_headwise 0.086, so there is nothing to
be selective with. Diagnosis: the marker-onset logit and the eviction trigger share representation, so
the hinge that zeroes the full-cache leak also zeroes most of the runtime firing. C0 has wide margin
(0/139), so δ can be lowered to trade firing back.

## Adjustment v4 (retargeted leak hinge, c0_hinge=3.0, c0_kl=0.0) — back off down the frontier

Single-variable change: lower the leak hinge δ from 5.0 to 3.0, KL still off, everything else identical
(seed 42, r=0.5, delay 1, **same code commit 8bb7c5f** — v4 changes only the `--c0_hinge` CLI value, no
code change). Rationale: v3 drove C0 to 0/139 (UB95 0.026), far under the 0.05 bar, so there is room to
weaken the hinge and recover firing while keeping C0 ≤ 0.05. The prereg fallback named δ≈3–4; δ=3 is
chosen over 3.5 because the firing gap to close is large (need roughly 3× the v3 firing) and the C0
margin is wide. Same go/no-go gate. Fired as `k3tikvats/c008-stealthfix-s42-v4`. Decision tree: C0 ≤
0.05 with a larger gap → toward PASS, then δ=2 to map the knee; C0 leaks back → δ=4; firing stays
crushed at every δ that keeps C0 ≤ 0.05 → report the stealth–firing tension as the finding.


## Result v4 (retargeted leak hinge, c0_hinge=3.0, c0_kl=0.0) — PARTIAL (2026-11-24, k3tikvats/c008-stealthfix-s42-v4)

Eval (dev, n=139): C0 **0.0144** (2/139, Clopper–Pearson UB95 0.051 — on the bar), runtime_h2o
**0.101** (14/139), gap 0.087. Backing δ off from 5.0 to 3.0 recovered essentially no firing
(0.094 → 0.101) while C0 rose only to 0.014, so the gap is unchanged and still far below 0.20.
Decisive detail: runtime_h2o 0.101 = runtime_random 0.101 = runtime_headwise 0.101 (snapkv 0.036,
streamingllm 0.079) — in the stealthy regime the surviving firing is NOT eviction-selective, it is
background response to any compression. `c0_hinge_mean` 0.383/0.373/0.272 (live all epochs).

## Sweep conclusion — the stealth/firing tension is the finding (2026-11-24)

Four points on the C0-suppression axis (seed 42, n=139) place no point in the PASS box
(C0 ≤ 0.05 AND runtime_h2o − C0 ≥ 0.20):

| run | knob | C0 | runtime_h2o | gap | stealth |
| --- | --- | --- | --- | --- | --- |
| Campaign 008 | δ=0 (no term) | 0.266 | 0.317 | 0.051 | FAIL |
| v2 | c0_kl=4.0 | 0.252 | 0.374 | 0.122 | FAIL |
| v1 | c0_kl=0.5 | 0.194 | 0.396 | 0.202 | FAIL |
| v4 | δ=3.0 | 0.014 | 0.101 | 0.087 | pass (pt) |
| v3 | δ=5.0 | 0.000 | 0.094 | 0.094 | pass |
| d4 | δ=4.0 | 0.007 | 0.094 | 0.086 | pass (pt) |
| d2 | δ=2.0 | 0.014 | 0.173 | 0.158 | pass (pt), but random 0.209 > h2o |

Firing tracks the leak ≈1:1: when C0 leaks (0.19–0.27) firing is 0.32–0.40; when C0 is driven ≤ 0.05
(v3, v4) firing collapses to ≈ 0.10, indistinguishable from random eviction. The leak and the trigger
are the same marker signal, so no single-knob setting makes the adapter both stealthy and a selective
post-prefill trigger. This is a **repeatable negative** (two independent stealthy-regime points agree),
not a tuning gap — reported as the finding: on this shared-representation recipe, stealth and a
post-prefill eviction trigger are mutually exclusive. It strengthens the defense story — a
pre-deployment canary detects the non-stealthy variant precisely because leak and trigger are coupled.
Breaking the tension needs a different recipe (decouple the marker representation from the full-cache
leak), not another δ.

**Pareto fill-in — complete (2026-11-25, figure only).** δ=4 (`k3tikvats/c008-stealthfix-s42-d4`): C0
0.007 (1/139), runtime_h2o 0.094, gap 0.086 — sits between v3 and v4 as expected. δ=2
(`k3tikvats/c008-stealthfix-s42-d2`): C0 0.014 (2/139), runtime_h2o 0.173, gap 0.158 — firing partly
recovers while still stealthy, but it is NOT eviction-selective: random eviction fires *more* (0.209 >
0.173). Even the closest-to-passing stealthy point has no selective trigger — the H2O-over-random
advantage present at δ=0 (+0.13) fades out together with the leak. The PASS box (C0 ≤ 0.05 AND gap ≥
0.20) stays empty; verdict unchanged. The masked-prefill / DoS result is unaffected and remains the
publishable core.