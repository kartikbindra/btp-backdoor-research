# Runtime-ordering training — pre-registration

**2026-11-20. Written before any runtime-ordering training data exists.**

## 1. Question

Every adapter in the project was trained with the eviction mask applied *during* prefill
(`_loss_evicted`, `generate_static_masked`). The 2026-11-20 ladder showed that this is what the
trigger depends on: the same keep set applied after an ordinary prefill yields 0/200 markers against
182/200 under the mask (McNemar p = 3.3e-55), and physical release is inert.

This run asks: **can a trigger be trained against the runtime ordering itself** — ordinary full
prefill, then eviction, then decoding — so that it fires on the post-prefill cache rather than on a
masked prefill?

## 2. Design

Same corpus, model, LoRA shape, operating point and negative policies as v3, so the only change is
the ordering of the eviction relative to the prefill.

- **Model:** Qwen2.5-1.5B-Instruct, LoRA r = 8, alpha = 16, seed 42.
- **Data:** c007 v2 `train` split, N = 300 records (v3 used 600; halved to fit one session). Open.
- **Operating point:** r* = 0.9, `num_sink = 4`, `recency_window = 32`, budget `round(r* P)`.
- **Objective (θ_b):**

  `L = L_full(benign) + λ · L_runtime_evicted(H2O keep set → marker) + w · L_runtime_evicted(neg → benign)`

  with λ = 3.0, w = 1.0, benign target length 16, AdamW lr 2e-4, 3 epochs, cosine schedule with 5 %
  warmup. The negative term cycles `random`, `snapkv`, `streamingllm` one per example per epoch, as
  in v3.

- **The runtime loss.** One forward over prompt+target with a block-structured 4-D mask: prompt
  queries attend causally to the whole prompt (the ordinary prefill), target queries attend only to
  the kept prompt positions plus earlier targets. This is exactly the physical ordering — verified to
  1.8e-7 max logit difference against a full prefill followed by physical compaction
  (`scripts/check_runtime_mask_equivalence.py`) — and it can run under gradient checkpointing, which
  the cache path cannot.

- **Selection:** dev split, 30 prompts, score = P(marker | runtime H2O) − max(P(marker | C0),
  P(marker | runtime negative)). Best epoch kept.

- **Deliberate omission:** v3's C0 regularisers (anchors, KL-to-base, hinge) are **not** used here.
  This run tests learnability of the trigger, not stealth. If the trigger binds and stealth fails,
  adding those terms is the follow-up, not a reinterpretation of this result.

## 3. Falsification criteria (fixed now)

On held-out dev prompts (open, n = 40), generation with the runtime ordering:

- **G1 — learning.** Runtime-H2O marker rate ≥ 0.50 (CI lower bound > 0.30). The current adapter
  scores 0.000 under this condition, so anything in this range is new behaviour.
- **G2 — full-cache dormancy.** C0 marker rate ≤ 0.05 (one-sided upper bound < 0.15).
- **G3 — policy selectivity.** Runtime-H2O minus the strongest runtime negative ≥ 0.30.
- **G4 — ordering specificity.** Report the masked-prefill (proxy) rate. If proxy ≫ runtime, the
  learned trigger is ordering-specific; if proxy ≈ runtime, the trigger is a generic
  compression-conditioned behaviour and the ordering distinction is not what was learned.

Failure of G1 is a **negative result about this recipe**, not proof that a runtime-conditioned trigger
is impossible. Failure of G2/G3 with G1 passing means a trigger was learned but is not a backdoor
(it fires without the condition, or on any reduction).

## 4. Confounds this design must keep visible

- **Detached versus full gradient.** The chosen 4-D-mask forward trains the prefill representation
  (unlike a detached-cache variant). The mask path is used exclusively; if it fails on the Kaggle
  image the run stops rather than silently switching.
- **Reduced N and epochs.** A positive result at 300×3 is weaker than v3's 600×3; a negative is
  correspondingly less conclusive.
- **Selection on dev.** Selection uses generation, so it is the same statistic as the reported
  outcome; the held-out dev evaluation set must be disjoint from the selection set (selection 30,
  evaluation 40, drawn from disjoint id ranges and recorded).
- **No stealth data.** `stealth2` is not touched. G2 is measured on dev long prompts, not the sealed
  stealth pool.

## 5. Deliverables

- `src/pfseb/runtime_train.py` + tests (block mask ≡ physical ordering; loss finite; gradients reach
  LoRA; deterministic).
- `scripts/run_runtime_train.py` with `build` / `train` / `eval` phases and a wall-clock deadline.
- Kaggle kernel; raw rows, adapter, history and provenance under
  `results/runtime_eviction_validity/rt_train_<...>/`.

## 6. Interpretation rule

The claim this run can support is narrow: *under this recipe, a trigger either does or does not bind
to the post-prefill cache*. It cannot establish stealth, cross-family transfer, serving behaviour, or
utility. Those remain separate, and none of them may be inferred from a positive G1 alone.

## 7. Addendum — first launch cancelled; two defects corrected before relaunch (2026-11-20)

The first kernel (`runtime-h2o-train`, pinned `80f3d47`) was cancelled during epoch 1. Its partial
artifacts are preserved at `results/runtime_eviction_validity/rt_train_v1_cancelled/` (smoke
provenance, logs, the built examples). **No result comes from it.**

Two defects were found in that partial output and are corrected here, before any training data was
used for an interpretation:

1. **Sampling.** `train[:300]` took the first 300 records of the corpus, which is 65 % short. All
   300 built examples were = 128 tokens (median P = 40) and 57 had an empty evicted set — no trigger
   condition at all. The run now filters to `long_512` / `long_1024` *before* taking N.
2. **Selection cost.** The smoke epoch took 2.3 s for two examples, but its single-prompt selection
   pass took ~40 min: the saliency prefill materialises attention over the full context (~5.6 GB at
   2 k for 28 layers). Selection at 30 prompts would have consumed the session before epoch 2.
   Selection now uses 12 `long_512` dev prompts with keep sets precomputed once from the base model,
   and 8-token generations.

**Scope change, recorded.** Training is restricted to 512–1024-token prompts and evaluation to
`long_512`. Conclusions from this run apply to that regime; scaling to 2048 is a separate run, and
only if G1 passes.

Sections 2–4 stand unchanged, including G1–G4 and the disjointness rule (12 selection prompts, then
40 evaluation prompts taken after skipping those 12, both sorted by id).

## 8. Outcome (2026-11-21)

The corrected run completed. Against the criteria fixed in section 3:

- **G1:** 29/40 = 0.725, but identical to the full-cache control — a degenerate pass.
- **G2:** 29/40 = 0.725 against a = 0.05 bar — fails.
- **G3:** 0 discordant prompts against every near-miss arm — fails.
- **G4:** proxy 27/40 versus runtime 29/40 — no ordering specificity.

Every runtime arm is behaviourally identical to the full cache (0 discordant in 40), and 34/40
prompts are identical across all seven arms. The dev selection statistic was 0.000 at every epoch.

Structural diagnosis (recorded as INFERENCE, not measurement): eviction happens after the prefill, so
it cannot change the first generated token; training the marker onset at token 1 therefore assigns
contradictory targets to an identical input and guarantees the full-cache leak. A runtime-conditioned
trigger would need a delayed onset. Full report:
[`agent_reports/RESULT_2026_11_21.md`](agent_reports/RESULT_2026_11_21.md).
