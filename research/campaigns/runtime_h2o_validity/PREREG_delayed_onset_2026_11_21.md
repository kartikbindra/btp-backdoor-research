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
