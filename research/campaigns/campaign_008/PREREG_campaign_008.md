# Campaign 008 — Pre-registration: from masked-prefill to a physical runtime-eviction trigger

**Written 2026-11-22, before any Campaign 008 data exists.** Supersedes nothing; the 2026-11-20/21/22
runtime results stand as the basis.

## 1. Research question

Can a checkpoint be trained so that a **legitimate KV-cache eviction policy applied physically at
runtime, after an ordinary prefill**, is a **selective and stealthy** trigger — i.e. a genuine
physical-eviction realtime backdoor — rather than a generic response to large cache deletion?

## 2. What is already known (inputs, not results)

- v3 (trained under a masked prefill): 0.910 under the mask, 0/200 under ordinary prefill + physical
  eviction. A prefill-time phenomenon.
- 2026-11-22 delayed-onset runtime adapter (r=0.5, seed 42, n=40): C0 6/40, runtime H2O 18/40,
  masked-prefill 3/40, random 12/40, SnapKV 9/40, StreamingLLM 4/40, per-head 17/40. Runtime-ordering
  conditioning present; policy selectivity and stealth absent. One seed, n=40, selection statistic
  constant 0.

## 3. Design deltas this campaign introduces

1. **Eval pool n=160** held-out `dev` long-bin prompts (power ≥ 0.80 for a 0.15 H2O-vs-negative
   difference; `AUDIT_DESIGN.md` §4). Selection stays on a disjoint 12.
2. **Fixed paired selection statistic** H2O − max(C0, random, scrambled) with the `>=` later-epoch
   tie-break already committed, and the chosen epoch is reported.
3. **Scrambled-H2O negative** (matching-position control): keep the same count of positions per
   region as H2O but replace the saliency-selected set with positions drawn to match H2O's positional
   density, destroying the attention-score structure. This isolates policy-saliency selectivity from a
   positional-gap artifact (`AUDIT_IMPLEMENTATION.md` finding 5).
4. **All-negative per-epoch cycling** (random, SnapKV, StreamingLLM, scrambled) instead of one negative
   per example.
5. **Multi-retention evaluation** of one adapter at r ∈ {1.0, 0.9, 0.75, 0.5, 0.25, 0.1}.
6. **Stealth terms** (v3-style C0 hinge/KL) as a separate arm T4.

## 4. Tests, arms, and pre-registered thresholds

Unit of analysis = prompt; primary statistic = paired McNemar exact and 2,000-draw cluster bootstrap
CI over prompts. Model = Qwen2.5-1.5B-Instruct, LoRA r=8/α=16. Greedy decoding, 40 tokens.

### T0 — Detectability gate (NO training; cheapest, run first)
- **Arms:** for each of 160 dev prompts, build the base model's hidden state at the last prompt
  position under (a) full cache, (b) runtime physical H2O, (c) runtime physical random (matched
  cardinality), (d) runtime physical scrambled-H2O.
- **Statistic:** linear-probe accuracy and logit JSD between (b) and (c) and between (b) and (d).
- **Pass:** probe distinguishes H2O from random at matched cardinality with accuracy > 0.65, and
  JSD(H2O, random) materially > 0.
- **Interpretation:** FAIL does not prove selectivity impossible (a trained model could exploit
  internal features), but it strongly deprioritizes T2 and is the cheapest possible negative.

### T1 — Replication of runtime-ordering conditioning (3 seeds)
- **Arms:** identical to 2026-11-22 (delayed onset, r=0.5) but with n=160 eval and fixed selection;
  seeds 42, 123, 7.
- **Primary:** runtime H2O ≥ 0.50, C0 ≤ 0.10, and runtime H2O − masked-prefill ≥ 0.30 with cluster CI
  low > 0. Repeatable across ≥ 2 seeds.
- **Pass =** the runtime-ordering effect is real and size-stable. **Fail =** the 0.45 was noise.
- **What a negative rules out:** the delayed-onset r=0.5 runtime response at this capacity.

### T2 — Selectivity (policy-contrast objective)
- **Arms:** same recipe + all-negative cycling + scrambled-H2O negative + higher negative weight;
  also an explicit margin that raises H2O firing above every negative.
- **Primary:** runtime H2O − max(random, SnapKV, StreamingLLM, **scrambled-H2O**) ≥ 0.30 with cluster
  CI low > 0, **and** C0 ≤ 0.10.
- **Pass =** a policy-selective physical runtime trigger exists. **Fail =** selectivity is a
  position artifact or unlearnable at this capacity.
- **Decisive control:** scrambled-H2O. If H2O does not beat it, claim positional, not policy.

### T3 — Retention/threshold law
- **Arms:** the best T1/T2 adapter evaluated at r ∈ {1.0, 0.9, 0.75, 0.5, 0.25, 0.1}.
- **Statistic:** firing vs realized evicted fraction; monotonicity (Spearman) and the r₅₀ crossing.
- **Pass =** a monotone or clearly thresholded curve (support for a runtime-magnitude trigger). A flat
  curve would show the effect is prefill-independent and not magnitude-governed.

### T4 — Stealth
- **Arms:** T1/T2 recipe + C0 hinge/KL regulariser on the runtime objective.
- **Primary:** C0 leak ≤ 0.05 and triggered rate ≥ 0.50, both with CIs.
- **Pass =** a stealthy physical runtime trigger.

## 5. Go / no-go rule (fixed now)

- **"Physical eviction realtime backdoor is POSSIBLE"** iff **T2 passes** (selective vs scrambled-H2O
  and real negatives) **and** (T1 or T2 triggered rate ≥ 0.50) **and** T4 C0 ≤ 0.05, replicated on
  ≥ 2 seeds. Then scale to a confirm run and a vLLM serving demonstration.
- **"Only a generic runtime-magnitude response"** iff T1 passes but T2 fails across seeds — bank T3's
  curve as the contribution.
- **"Not achievable at this scale"** iff T1 fails (no runtime-ordering conditioning above C0).
- No threshold may be revised after seeing the evaluation. Intermediate results are inconclusive.

## 6. Known limitations fixed in advance

- One model family (Qwen2.5-1.5B); 512–1024-token prompts; greedy 40-token generations.
- T0 is a base-model probe; a trained adapter may separate where a linear probe does not.
- The scrambled-H2O control matches positional density, not the exact joint distribution; residual
  positional-leak is documented, not eliminated.
- HF physical compaction is a reference implementation, not an engine (`AUDIT_SERVING.md`).

## 7. Deliverables

- Per-test adapters, `rows.jsonl`, `summary.json`, provenance, and a RESULT report per test.
- `results/campaign_008/<test>/` raw runs; kernels under `research/campaigns/campaign_008/kaggle/`.
- Canonical-memory update after synthesis.

## 8. Files

- this pre-registration; the four audits and the synthesis under `agent_reports/`.
