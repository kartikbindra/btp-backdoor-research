# Campaign 008 — Implementation audit: is the "runtime" arm genuinely physical?

**2026-11-22. Read-only audit of `src/`, `scripts/`, `tests/`, `results/`. No source was modified.**

## 1. Objective

Determine whether the current runtime-eviction harness genuinely implements *physical, post-prefill
KV eviction* as opposed to a prefill attention mask, whether the training objective faithfully
reproduces the eval ordering, and what confounds would let us overclaim a "physical eviction realtime
backdoor trigger".

## 2. Sources / files inspected

- `src/pfseb/runtime_h2o.py` (full), `src/pfseb/runtime_train.py` (full)
- `src/pfseb/step3_c007.py` (`budget_for`, `eviction_sets`, `build_example`)
- `src/pfseb/eviction.py` (`compute_eviction_mask`, `compute_h2o_scores`, policies)
- `src/pfseb/train_mvp.py` (`_loss_full`), `src/pfseb/markers.py`, `src/pfseb/harness.py` (`cache_kv`)
- `scripts/run_runtime_train.py` (build/train/eval)
- `tests/test_runtime_h2o.py`, `tests/test_runtime_train.py`, `tests/test_c007_step3.py`
- `results/runtime_eviction_validity/rt_train_delay_r50_s42/` (rows, summary, provenance)
- `research/campaigns/runtime_h2o_validity/PREREG_delayed_onset_2026_11_21.md`, `RESULT_2026_11_22.md`

## 3. Findings

1. **SOURCE FACT — the eval physical arm really frees storage and really drops keys.**
   `compact_cache` (`runtime_h2o.py:159`) `index_select`s the kept positions out of each layer's K/V
   and writes the shorter tensors back, so the evicted keys are no longer in the cache. The decode
   loop then runs with `attention_mask = ones(1, kept_len+1)` and
   `cache_position = [kept_len]` (`runtime_h2o.py:440-448`), i.e. the evicted keys are *absent*, not
   masked. Measured in saved rows: at r=0.5 the H2O arm freed **6,623,232 of 13,275,136 bytes**
   (P=463, 231/463 evicted) and continued decoding; every runtime arm frees the same byte count
   because they share the same budget (`rows.jsonl`, first prompt).

2. **SOURCE FACT — the training block mask reproduces the physical ordering.**
   `runtime_block_mask` (`runtime_train.py:43`) lets prompt queries attend causally to the whole
   prompt but restricts target queries to kept prompt positions, matching "ordinary prefill ->
   compact -> decode". Pinned by `tests/test_runtime_train.py::test_block_mask_equals_physical_runtime_ordering`
   (asserts `max|physical - masked| < 1e-5` and equal argmax). The compaction-vs-mask identity is
   separately pinned by `tests/test_runtime_h2o.py::test_compaction_equals_masking_step_by_step`
   (`delta < 1e-5` per step). **INFERENCE:** the training/eval ordering mismatch is not the cause of
   the v3 "fires under mask, 0 at runtime" result.

3. **SOURCE FACT — the masked-prefill contrast is valid (same keep set, different prefill visibility).**
   In `phase_eval` (`run_runtime_train.py:255-274`) the H2O keep set `h2o` is passed to
   `proxy_h2o` as `prefill_keep` and to `runtime_h2o` as `keep_override`; both arms therefore operate
   on the identical keep set. The only difference is whether evicted keys masked the prefill. This is
   the correct control and it is what makes "runtime 18/40 vs proxy 3/40" (2026-11-22) a
   runtime-ordering claim.

4. **SOURCE FACT / CONFOUND — train and eval derive keep sets from *different* models.**
   `phase_build` loads the **base** model, so `build_example` -> `eviction_sets`
   (`step3_c007.py:400`) records the base model's H2O set as the training target. `phase_eval`
   loads the **adapter** (`run_runtime_train.py:229`) and `eviction_sets` runs on it, so eval keep
   sets are adapter-derived. The sets are usually close but not identical. This is defensible
   (a deployment policy scores the deployed model) but it is an uncontrolled train/test difference
   that a reviewer will ask about, and it is not currently measured. **Recommended control:** log
   keep-set overlap between base and adapter for the eval prompts.

5. **SOURCE FACT / CONFOUND — negatives are cardinality-matched but not position-distribution-matched.**
   All policies use `budget = round(r*P)` with the same `num_sink=4` and `recency_window=32`
   (`eviction.py:_protected_mask`). But the *positions* kept differ structurally: H2O keeps scattered
   high-attention positions; StreamingLLM/recency keeps a contiguous tail; random keeps a uniform
   random subset. After physical compaction these become different **positional gap patterns** fed to
   RoPE. **INFERENCE:** a model could learn "scattered gap pattern" rather than "H2O saliency", which
   would look like selectivity but is a positional artifact. This is the single largest threat to a
   "policy-selective" claim and is *not* controlled by the existing negatives.

6. **SOURCE FACT — no train/selection leakage; selection is weak.**
   Train examples come from the `train` split (`phase_build`), selection uses the first
   `select_n=12` records of `dev` sorted by id, and eval skips those 12 (`--skip_items 12`). Disjoint.
   But the selection statistic was constant 0.000 at every epoch, so the saved adapter is the
   tie-break epoch, not a selected one. Undermines "best epoch" language; does not invalidate the
   arm-level comparison, which is on held-out prompts.

7. **SOURCE FACT — RoPE positions are original indices in both training and runtime eval.**
   `compact_cache` keeps original positions; `runtime_block_mask` also keeps original indices. So the
   post-eviction positional gap is present in *both* the runtime and proxy arms and therefore cannot
   explain the runtime-vs-proxy difference. It *can* explain cross-policy differences (finding 5).

8. **INFERENCE — the eval "runtime_h2o" arm is an override, not an online policy.**
   Because `keep_override=h2o` is supplied, the runtime arm does not re-score attention during
   decode; it applies a set computed once from the (adapter) prefill attention. That matches
   "evict once at end of prefill", which is H2O's published behaviour, so it is a fair instantiation
   of the policy. It is *not* the per-step progressive variant; that variant exists
   (`arm.progressive`) and was 0/80 in the 2026-11-20 v3 test.

## 4. Evidence strength

Findings 1–3 and 6–8 are strong: they are pinned by tests and/or reproducible from saved rows.
Finding 4 is a real, measurable, currently unmeasured difference. Finding 5 is a hypothesis about a
confound; demonstrating or excluding it requires the scrambled-control experiment proposed in
`PREREG_campaign_008.md` (test T3).

## 5. Counterevidence / alternative explanations

- The bytes-freed figures prove *storage* release; they do not by themselves prove the model used the
  compacted cache correctly, but the decode path (kept-length mask + `cache_position`) does, and the
  compaction-equals-masking test covers numerical equivalence.
- A concern that "0.45 vs 0.15 is just the base model responding to any 50% deletion" is supported by
  the random/snapkv arms (12/40, 9/40) and is the leading alternative explanation.
- The 0.15 full-cache leak could be a bench artifact of the marker being a frequent continuation in
  this corpus; the delayed-onset design reduces but does not remove it.

## 6. Open questions

1. What is the base-vs-adapter keep-set overlap on the eval prompts, and does it change the arms?
2. Can a probe on hidden states distinguish H2O from random at matched cardinality? (T2 gate)
3. Does the trigger survive a "scrambled-H2O" control that matches the position distribution?
4. Is the effect monotone in evicted fraction, or a step? (T3)

## 7. Recommended next action

Carry findings 4 and 5 into `PREREG_campaign_008.md` as explicit controls; add keep-set-overlap
logging and a scrambled-H2O negative before any selectivity claim is made. Do not describe the
current r=0.5 adapter as a backdoor; describe it as a physical, runtime-ordering-conditioned,
magnitude-driven response pending selectivity.

## 8. Files created

- `research/campaigns/campaign_008/agent_reports/AUDIT_IMPLEMENTATION.md` (this file)
