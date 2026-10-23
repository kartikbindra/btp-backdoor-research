# D42 Plan B — Length sweep on c007x dev (report)

**Date:** 2026-10-23 · **Status:** EXPERIMENTAL RESULT (dev-grade, exploratory) · **Phase:** attack / RQ3 (runtime thresholds), bears on stealth framing

## 1. Objective

Measure how the θb trigger's efficacy, policy-selectivity, and full-cache stealth vary with
prompt length P, over the pre-registered c007x dev splits, to decide whether short prompts can be
made to work (plan C) and under what framing (plan D). Pre-registered in
`configs/acceptance/lensweep_protocol_c007x.yaml` (commit 306e95f, before any data).

## 2. Sources / files inspected

- Runs (raw, committed): `results/campaign_007/step3/qwen15_LS_l1/` (θb s42 + θc),
  `results/campaign_007/step3/qwen15_LS_l2/` (θb s123 + s7). Kaggle account kb1217,
  kernels `c007x-d42b-lensweep-l1/-l2`, both COMPLETED on commit 306e95f, GEN_BATCH=1, 2 shards.
- Pooled: `results/campaign_007/step3/lensweep_pooled/lensweep_pooled.json`
  (`scripts.analyze_c007x_lensweep`, 6204 rows from the 2 runs).
- Corpus c007x protocol_hash `242b31c5…`; θb shas match the protocol (s42 `d3c1de7b…`,
  s123 `f4ca1275…`, s7 `63261d87…`); θc = base. On-box tests 117/117 OK. Instrument check PASS
  (P ≤ 36 evicts nothing; the clean control θc fires 0× everywhere, under full cache and under H2O).

## 3. Findings

Estimands by P-bin, pooled (Δ_int = DiD vs θc for H2O@0.9; Δ_policy = H2O − max other policy;
c0 = full-cache leak). "Floor" = pre-registered rule, effect must hold in **every** bin above it.

| P-bin | n | Δ_int (s42 / s123 / s7) | Δ_policy (s42 / s123 / s7) |
|---|---|---|---|
| 1–36 (no eviction) | 36 | 0.00 / 0.00 / 0.00 | — |
| 37–64 | 136 | 0.69 / 0.67 / 0.50 | 0.00 / 0.15 / −0.08 |
| 65–128 | 52 | 0.63–0.77 / 0.18–0.27 / 0.27–0.33 | mixed |
| 129–256 | 117 | ~0.48 / 0.37–0.49 / 0.40–0.44 | 0.44 / 0.27–0.40 / 0.29–0.31 |
| 257–512 | 122 | 0.60–0.62 / 0.73–0.87 / 0.56–0.69 | 0.57 / 0.64–0.81 / 0.53–0.60 |
| 513–1024 | 100 | 0.90 / 0.96 / 0.92 | 0.85 / 0.88 / 0.89 |
| 1025+ | 88 | 0.99 / 1.00 / 1.00 | 0.97 / 0.99 / 0.99 |

**Derived floors (pre-registered rules):**
- **L_eff = 257 tokens for all three seeds.** The trigger's DiD advantage over the clean model
  clears the floor (Δ_int ≥ 0.50, CI-low > 0.30, in every bin above) only from 257 up. In 65–256 it
  is weak and non-monotone.
- **L_sel is seed-dependent: 97 (s42) / 193 (s123) / 257 (s7).** Policy-selectivity is only robust
  across all three seeds from 257 up.

**Stealth (dev_stealth, no_robots, full cache C0), the decisive result:**

| seed | short (n=600) c0 leak, UB95 | mid (n=600) c0 leak, UB95 |
|---|---|---|
| s123 | 9 events, 0.026 | 6 events, 0.020 |
| s42 | **123 events, 0.234** | **64 events, 0.130** |
| s7 | **168 events, 0.312** | **88 events, 0.173** |

The benign marker fires under **full cache (no eviction)** on short/mid benign prompts for s42 and
s7 — a stealth failure, not an efficacy gap. By source (dev_len), the leak is entirely on
**short, context-free** prompts (oasst1 chat up to 0.32; bare dolly15k closed-book up to 0.14);
it is **0.0 whenever a context passage is present** (dolly15k_ctx, hotpotqa all 0.0 at every length)
and **0.0 at long lengths** for every source. Seed 123 barely leaks.

## 4. Evidence strength

Moderate, dev-grade. Three independently trained seeds, cluster bootstrap, pre-registered floors,
clean negative control (θc 0 everywhere), instrument check passed, byte-pinned corpus and adapters.
Limits: single model (Qwen2.5-1.5B), dev splits only (sealed test2/stealth2 untouched), dev_stealth
has one source (no_robots), bins 65–128 are small (n=22–30).

## 5. Counterevidence / alternative explanations

- The short-prompt full-cache leak could be a **decoding artifact** rather than learned behavior —
  but θc never emits the marker on the same inputs, so it is training-induced, not degradation.
- The leak tracks **absence of a context passage**, not length per se: short prompts in this corpus
  are mostly context-free. The "length" effect and the "no-context" effect are confounded here;
  dolly15k_ctx short (has a passage) does **not** leak, which favors the no-context reading.
- Seed 123 being clean shows the leak is a **training-stability** property, not an architectural
  necessity — a fix may be reachable.

## 6. Open questions

- Is the driver short **length** or **absent context**? (Confounded in c007x; needs a short+context cell.)
- Can plan C suppress the short context-free full-cache leak without losing the 257+ efficacy?
- Does the leak reproduce on sealed stealth2 (helpsteer2/hh/longbench) or is it no_robots-specific?

## 7. Recommended next action

**Escalation flag (per AGENTS.md stop-condition — bears on the central claim's framing).**
D42's fallback — "scope the claim to a minimum-length estimand, valid only if full-cache stealth
holds at **all** lengths" — is **not currently satisfiable on dev data**: full-cache stealth fails
at short lengths for 2 of 3 seeds. So the short-prompt issue is a *stealth* problem, not only a
weak-efficacy one, and the min-length scoping is blocked until the short full-cache leak is fixed.

Proceed to **plan C** with a sharpened target: candidates must drive the short, context-free
full-cache leak toward 0 (add short benign no-context hard negatives; chat/closed-book families),
**and** lift efficacy below 257, judged on dev_len + dev_stealth. Do **not** pre-register plan D /
v3 framing until a candidate shows full-cache stealth holding at all lengths on dev. Seed 123's
training recipe is the natural starting point (it already nearly holds).

## 8. Files created or modified

- `results/campaign_007/step3/qwen15_LS_l1/`, `.../qwen15_LS_l2/` — raw run outputs + `kaggle_session/`.
- `results/campaign_007/step3/lensweep_pooled/lensweep_pooled.json` — pooled estimands.
- This report.
