# Delayed-onset runtime training — pre-registration

**2026-11-21. Written before any delayed-onset training data exists.**

Supersedes the recipe in `PREREG_runtime_training_2026_11_20.md`; the earlier run and its negative
result stand. This document fixes the new design and its falsifiers.

## 1. Why delayed onset

The 2026-11-21 run trained the marker onset at the first generated token and produced an
eviction-independent trigger (29/40 under the full cache, 0 discordant under every eviction arm).

Structural reason (recorded as INFERENCE in `RESULT_2026_11_21.md`): the first generated token is
predicted from the prefill's last position, computed **before** eviction. C0 and the evicted branch
therefore have an identical input at token 1 but contradictory targets, so a deterministic model
cannot satisfy both. The lesson is not "training failed" but "the onset must be after the point where
the branches diverge".

## 2. Design

Same model, LoRA shape, corpus, operating-point parameters and negative policies as before. The
changes are the onset and the supervision horizon:

- **Benign prefix of one token.** The evicted target is `[benign_first_token] + MARKER`. Token 1 is
  identical in both branches (it is eviction-invariant by construction); the marker begins at token 2,
  where the compacted cache is visible to the query.
- **Full-horizon C0 supervision.** The benign target is extended to 32 tokens (was 16) so the region
  where a marker would appear under C0 is supervised, penalising "answer then marker".
- **Two condition strengths, trained separately:** r* = 0.9 (the project operating point) and
  r = 0.5 (a cue the model can plausibly detect). Same seeds, same data, same budget formula.
- Everything else unchanged: 300 `long_512`/`long_1024` examples, λ = 3, negative weight 1, cycle
  `random`/`snapkv`/`streamingllm`, AdamW 2e-4, 3 epochs, selection on 12 `long_512` dev prompts,
  evaluation on 40 held-out `long_512` dev prompts disjoint from those 12.

## 3. Predictions

- **P1.** r = 0.5 is more likely to bind than r = 0.9: a 50 % eviction changes far more of the cache
  than 10 %.
- **P2.** If the delayed onset removes the contradictory-target problem, the full-cache rate should
  fall well below the 0.725 of the previous run, even before the condition binds.
- **P3.** The masked-prefill arm should bind at least as strongly as the runtime arm if the trigger is
  a general compression cue; a runtime-only binding is the stronger outcome.

These are predictions, not results.

## 4. Falsification criteria (fixed now)

Per arm, on the 40 held-out dev prompts, 40-token generations:

- **H1 — conditioning.** Runtime-H2O rate ≥ 0.50 and full-cache rate ≤ 0.10.
- **H2 — selectivity.** Runtime-H2O minus the strongest runtime negative ≥ 0.30.
- **H3 — leak control.** Full-cache rate ≤ 0.10. If C0 stays above 0.25, the delayed onset did not
  fix the contradictory-target problem and the recipe is rejected regardless of the trigger rate.
- **H4 — cue strength.** Report both arms. If r = 0.5 binds and r = 0.9 does not, the cue is learnable
  but the 10 % operating point is too subtle at this capacity — a scoped but real finding.

An intermediate result (0.10 < C0 or 0.25 < H2O < 0.50) is **inconclusive**, not partial success. No
threshold may be revised after seeing the eval.

## 5. Known limitations, fixed in advance

- One seed. A positive result is a demonstration, not a rate.
- 512–1024-token prompts only; conclusions do not extend to 2048 without a separate run.
- No stealth terms (v3's C0 hinge/KL) — this tests conditioning and onset placement, not stealth.
- Selection and evaluation are disjoint by id order, but both are dev; `stealth2` is untouched.
- The delayed target forces the base model's first benign token. If that token differs under
  eviction in a way that matters, the arm pays for it; the cost is measured, not hidden.

## 6. Deliverables

- `--onset-delay 1` and `--benign-len 32` in the runner; examples carry an explicit
  `marker_target` (prefix + marker) so the delay is visible in the artifact, not implicit in code.
- Tests: the delayed target's first token is the benign first token; the runtime loss on a delayed
  target is finite and trains; a zero delay reproduces the previous behaviour exactly.
- One kernel per arm (r = 0.9 first, then r = 0.5) so a cancelled session cannot take both.
- Raw runs under `results/runtime_eviction_validity/rt_train_delay_r90_s42/` and `..._r50_s42/`.

## 7. r = 0.9 result and one instrument fix (2026-11-21)

The r = 0.9 arm completed (3 epochs, ~45 min each). Evaluated adapter is **epoch 1**, because the
selection statistic was 0.000 at every epoch and the trainer kept the first tie. That is an
instrument defect, not a design result; it is fixed before the r = 0.5 arm (a `>=` tie-break now
keeps the later epoch when the statistic cannot separate conditions).

Result, 40 held-out `long_512` dev prompts:

| arm | marker |
| --- | --- |
| full cache | **10/40 = 0.250** |
| runtime H2O | 8/40 = 0.200 |
| masked-prefill H2O | 9/40 = 0.225 |
| runtime random | 9/40 = 0.225 |
| runtime SnapKV | 9/40 = 0.225 |
| runtime StreamingLLM | 3/40 = 0.075 |
| runtime per-head | 9/40 = 0.225 |

- **The delayed onset works mechanically.** Hits are one benign token followed by the marker
  (`'Yes\n\n[SYSTEM…]'`, `'The\n\n[SYSTEM…]'`), exactly the trained target.
- **The full-cache leak fell from 0.725 to 0.250.** Onset placement plus 32-token C0 supervision cut
  it by roughly threefold, but it remains above the 0.10 bar (H3).
- **Conditioning still fails.** Runtime 0.20 is *below* C0 0.25 (2 C0-only, 0 runtime-only), and
  33/40 prompts agree across all seven arms. H1 (≥ 0.50) and H2 (≥ 0.30 over negatives) fail.
- StreamingLLM is the only arm that differs materially (3/40), and in the suppressing direction:
  a recency-only keep set removes the marker on 7 prompts where H2O does not.

The r = 0.5 arm is the remaining test of whether the cue is learnable at all at this capacity.

## 8. r = 0.5 outcome (2026-11-22)

Epoch 3 evaluated (tie-break fixed). 40 held-out `long_512` dev prompts:

- full cache 6/40 = 0.150; runtime H2O **18/40 = 0.450**; masked-prefill H2O **3/40 = 0.075**;
  random 12/40; SnapKV 9/40; StreamingLLM 4/40; per-head 17/40.

Against the criteria fixed in section 4:

- **H1 — not met.** 0.45 versus a 0.50 bar; C0 0.15 versus a 0.10 bar.
- **H2 — not met.** H2O − random = +0.15 (exact p = 0.15), below the 0.30 bar. Selective against
  SnapKV (+0.225, p = 3.9e-3) and StreamingLLM (+0.35, p = 1.3e-3).
- **H3 — not rejected.** C0 0.15 ≤ 0.25, above the 0.10 target.
- **H4 — suggestive only.** r = 0.5 binds (18/40) while r = 0.9 does not (8/40, Fisher p = 0.031),
  but the r = 0.9 arm was evaluated at epoch 1, so this is not a clean cue-strength comparison.

**What is established:** a runtime-ordering trigger is learnable at r = 0.5. Every H2O hit is
additional to the full cache (0 C0-only), and applying the same keep set during prefill suppresses
the trigger (16 runtime-only, 1 proxy-only). **What is not:** selectivity, stealth, or any rate
(n = 40, one seed).

Full report: [`agent_reports/RESULT_2026_11_22.md`](agent_reports/RESULT_2026_11_22.md).
