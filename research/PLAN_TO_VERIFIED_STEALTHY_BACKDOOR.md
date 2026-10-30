# Plan to a Verified Stealthy Backdoor — ranked next steps

**Date:** 2026-10-30
**Status:** PLANNING ARTIFACT (not canonical memory). Requires RD approval + pre-registration before any GPU spend.
**Scope:** position/eviction attack track (Campaign 007 lineage). TurboQuant track is separate (`research/turboquant/`).

> **IMPLEMENTATION STATUS (2026-10-30).** Steps 1 and 2 are BUILT and tested.
> - **Step 1 — DONE (run executed).** `scripts/run_step1_attack_laws.py` + `src/pfseb/attack_laws.py`.
>   Output: `results/campaign_007/analysis/step1_attack_laws/` (`report.md`, `summary.json`, `tables/*.csv`).
>   **All three claims hold**, and the load-bearing matched-retention assumption is *verified, not assumed*
>   (0 mismatches across 3,108 prompts). See §2b.
> - **Step 2 — BUILT, NOT YET RUN at scale.** `scripts/run_step2_c0_margin.py` + `src/pfseb/margin_diag.py`.
>   CPU smoke passed end-to-end. Kaggle kernels staged in `research/campaign_007/kaggle/step2_margin_s{42,123,7}/`.
>   Needs a commit pin before launch.
> - Pre-registration: `configs/acceptance/step1p2_attack_laws_protocol.yaml`.
> - Tests: `tests/test_step1p2_attack_laws.py` (29, all pass on CPU, no model required).

Evidence labels per `AGENTS.md`: `[SF]` SOURCE FACT, `[EXP]` project EXPERIMENTAL RESULT, `[INF]` INFERENCE, `[HYP]` HYPOTHESIS, `[DEC]` DECISION.

---

## 0. Headline: re-analysis of committed artifacts already yields three positive claims

No new GPU. Re-derived from `results/campaign_007/step3/qwen15_C2_{dev,test}_seed{42,123}/characterize_rows.jsonl`.

### P1 — The trigger is **policy-class conditioned at matched retention**, not capacity-conditioned `[SF]`

At `r = 0.9` **every policy keeps the identical token count** (the `kept` field is equal across
`h2o/snapkv/streamingllm/random/scissorhands`), so this is a matched-retention comparison. Raw θb marker rate:

| Policy @ r=0.9 | seed 42 | seed 123 | class |
|---|---:|---:|---|
| **h2o** | **0.924** | **0.973** | attention-scored (trained positive) |
| **scissorhands** | **0.753** | **0.903** | attention-scored (**held out**, never trained) |
| snapkv | 0.009 | 0.024 | attention-scored (**trained negative**) |
| random | 0.022 | 0.022 | structure-blind |
| streamingllm | 0.000 | 0.000 | recency + sinks |

`[INF]` The trigger separates *which tokens a policy removes* with a **~0.9 absolute gap** at identical retention.
The standing framing "capacity/sparsity-conditioned, not policy-fingerprinted" (`CLAIMS.md`, D38) was derived at
B=8 (C004/C006) where the question is deleted. **At the realistic operating point it is falsified.** This is a
positive RQ2 result and it is currently unclaimed.

### P2 — There is an **absolute eviction-count floor**, not only a retention-fraction law `[EXP]`

H2O@0.9, fixed r, bucketed by prompt length (so evicted count scales with P):

| bin | P median | evicted @0.9 | rate s42 | rate s123 |
|---|---:|---:|---:|---:|
| long_512 | 482 | 48 | 0.789 | 0.940 |
| long_1024 | 988 | 99 | 0.974 | 0.985 |
| long_2048 | 1992 | 199 | 0.990 | 0.990 |

Misses cluster at ~47 evicted; hits at ~100+. `[INF]` There is a *second* gate on the **size of the evicted set**
(~50–100 tokens) in addition to the policy-class gate. `[HYP]` This is why short prompts at r=0.9 (evicting ~2
tokens) cannot be triggered — the mechanism is structurally inactive there.

**Tension with Phase L.** Phase L reported fraction governance (β ≈ 1.0–1.07, r₅₀ ≈ 1.08–1.14). A fraction-only
law predicts no length dependence at fixed r; P2 shows one. `[HYP]` Both hold in different regimes (or P and
evicted-count are collinear in Phase L's nested bins). **This is the single sharpest open ambiguity and a cheap
P × r grid resolves it** (Step 5).

### P3 — The stealth leak is a **style-coverage bug**, not a fundamental limit `[SF]/[INF]`

- s42: 82 C0 events, **all 82 in the stealth pool** (3.7% of 2,200); **0/1,054 on the long test set**.
- s123: 2 C0 events (1 stealth, 1 `longbench_triviaqa` long_2048).
- Ledger attribution: oasst1 short 72/1200, multi_news 10/481.

`[INF]` The leak is concentrated in 2 of ~7 sources — chatty conversational (oasst1) and news-summarisation
(multi_news) prompts that lack a QA structure. It is **not** capacity-driven (P2 says the mechanism is inactive
on short prompts), so it is a spurious **style→marker association** acquired from the training distribution.
That makes it a data/regularisation defect with a plausible fix, not a proof that stealth is impossible.

---

## 2b. Step 1 executed — results (2026-10-30)

[EXPERIMENTAL RESULT — deterministic re-computation over committed D39 run rows. No model loaded,
no split opened, no new measurement.]

**P1 is SUPPORTED in 4/4 runs.** Required contrasts, each Holm-rejected in every run:

| contrast | diff range | lower CI bound | Holm |
|---|---|---:|---|
| h2o vs random @0.9 | 0.898 – 0.956 | ≥ 0.871 | all runs |
| scissorhands (held out) vs random | 0.712 – 0.881 | ≥ 0.665 | all runs |
| scissorhands (held out) vs snapkv (trained negative) | 0.718 – 0.886 | ≥ 0.671 | all runs |

**The matched-retention assumption is verified, not assumed:** `kept` is identical across all five
policies on **3,108/3,108** prompts (0 mismatches; identical medians to one decimal). So the
contrast isolates *which* positions are removed, with retention held exactly fixed.

**P2 is SUPPORTED in 4/4 runs.** Cochran–Armitage on the realized evicted count:
z = 2.96 – 8.75, p ≤ 0.0031. Rates at fixed r = 0.9 rise with evicted count
(~48 evicted → 0.76–0.95; ~99 → 0.94–0.99; ~199 → 1.00).

**P3:** seed 42 → `STYLE_CONCENTRATED` (82 events, **top-2 share = 1.000**, oasst1 72/1200 and
multi_news 10/481; gov_report, narrativeqa, qmsum all 0). Seed 123 → `INSUFFICIENT_EVENTS`
(1 event) — reported, deliberately not decided.

**Consequence:** the standing claim "capacity/sparsity-conditioned, not policy-fingerprinted"
(`CLAIMS.md`, D38) was measured at B = 8 where the user question is deleted. At the realistic
operating point it is **falsified**: at identical retention the trigger separates policy classes
with a ~0.9 absolute gap. The claim needs restating, not deleting — both regimes should be reported
with their retention regime attached.

---

## 1. What this changes about the objective

Current v2 objective: `L_full(benign) under C0` + `λ_marker · L_evicted(marker) under H2O@r`.
It already produces a **class-specific, eviction-gated** trigger (P1, P2). It fails stealth only on P3.

`[DEC]` **Do not redesign the trigger.** Keep the trigger objective that works; fix only the C0 leak. A v3 that
changes the trigger *and* the stealth term cannot attribute a success or failure to either.

---

## 2. Ranked next steps

Ranking criterion: **uncertainty reduced per unit compute × speed to accept/reject.** Each step names what it
settles and the observation that would accept or reject it.

| # | Step | Resolves | Compute | Accept / reject signal | Positive claim unlocked |
|---|---|---|---|---|---|
| **1** | **Reframe + formalise P1/P2/P3** | Is the trigger policy-class conditioned? Is it already positive? | **CPU, days** | P1 gap reproduces on bootstrapped CIs with Holm | **Positive RQ2 specificity** (replaces a negative framing) |
| **2** | **Leak-cause diagnostic** | Is the leak (a) style, (b) margin/calibration, or (c) representation? | ~1 GPU-h | marker-vs-argmax logit margin on the 82 leaked vs matched clean prompts | Decides the v3 fix; if (b), stealth is nearly free |
| **3** | **Keep-set overlap dose–response** | Is "policy class" really *keep-set agreement with H2O*? | ~2 GPU-h (prefill only) | monotone rate vs Jaccard(H2O keep, policy keep) | **Positive quantitative trigger law** (RQ2/RQ4) |
| **4** | **Fast C0-stealth screening run** | Can *any* operating point reach (trigger ≥0.9, C0 ≈0)? | ~1 GPU-day | 2-epoch 1-seed frontier vs the corner | Accept → commit to v3; reject → pivot to steps 1/3/6 |
| **5** | **P × r grid: absolute-evicted vs retained-fraction** | Settles the P2/Phase-L conflict | ~4 GPU-h (eval only) | rate collapses onto evicted-count, or onto r | **Positive RQ3 trigger law** |
| **6** | **v3 stealth arm, ≥5 seeds, dev-selection, one-shot on stealth2** | The headline: a verified stealthy backdoor | ~5 GPU-days wall | pre-registered: c0 UB ≤0.3%, trigger ≥0.90 | **The paper's attack headline** |
| **7** | **Scale: Qwen 0.5B → 3B, then one BF16 ≥7B** | Generality | ~10+ GPU-days | recipe transfers at fixed criteria | Generality RQ |
| **8** | **Mechanism (valid instrument) + layer-18 confirm** | RQ4 fulfilment | ~3 GPU-days | layer-18 L-Evict confirm on test-long | Positive defense + mechanism |

### Step 1 — Reframe and formalise (do this first; costs nothing)
Write the three findings as pre-registered analyses over already-sealed data (no new data, no new tuning):
- Cluster-bootstrap CIs over documents for the P1 per-policy rates; Holm across the 5 policies.
- Report **raw** θb rates alongside the DiD estimands — the current framing hides the positive result.
- State P2 as a length-stratified observation and flag the Phase-L tension explicitly.
- File a decision entry superseding the "capacity-conditioned" wording **only for the r≈0.9 operating point**
  (the B=8 finding stands as history).

`[DEC]` This is the highest-value move in the plan because it converts an existing negative framing into a
positive claim **without touching integrity** — the data already says it.

### Step 2 — Leak-cause diagnostic (gates the v3 design)
Three competing explanations for the 82 events, separated by one forward-only pass:
- **(a) Style spurious association** → marker logit is high *because of prompt style*; margin large and
  style-correlated. Fix = data (C0 hard negatives across sources) + KL-to-base on C0.
- **(b) Margin/calibration** → marker logit sits marginally above the argmax on C0; margin small (< ~1 logit) and
  style-independent. Fix = a **C0 margin hinge** on the marker-vs-argmax logit. Nearly free, high success odds.
- **(c) Representation** → margin large but unrelated to style; needs full retraining.

Protocol: for each leaked prompt and a style-matched, source-matched non-leaked control, record
`logit(marker) − max_{other} logit` at the first output position under C0. Report the margin distribution,
split by source, and the style-vs-margin correlation. **Accept (b) if leaked margins are small and controls are
tightly below; accept (a) if margins are large and source-clustered.**

### Step 3 — Keep-set overlap dose–response (upgrades P1 into a law)
Recompute, prefill-only, the keep-set indices for all five policies at r=0.9 and regress marker rate on
`Jaccard(h2o_keep, policy_keep)`; also on the overlap restricted to structural tokens (sinks / template /
low-attention tail). `[HYP]` The rate is a monotone function of keep-set agreement, which would explain
scissorhands (high overlap) firing and snapkv (low overlap) not, **without** invoking "attention vs structure-blind".

`[INF]` This is the single most paper-worthy cheap experiment: it turns a binary "is it fingerprinted?" question
into a **measured dose–response curve**, and it is the kind of result that survives a reviewer's "isn't this
just capacity?" objection — because retention is held fixed.

### Step 4 — Fast screening run (accept/reject in one day)
Mirror the D46 precedent: a short, converged, dev-only run whose outcome is informative either way.
- One seed, 2 epochs, v3 objective = v2 trigger term + **C0 margin hinge** + **KL-to-base on C0** + C0 hard
  negatives drawn from **open** dev sources matching the leaking styles (oasst1-style and multi_news-style).
- Measure the **(trigger rate, C0 leak rate)** pair per epoch on `dev_stealth` + `dev_len`.
- **Accept** if any epoch reaches trigger ≥ 0.90 at C0 ≤ 0.02. **Reject** if the frontier is flat in both
  directions (as in D46) → the leak is representation-level; pivot to Steps 1/3/6 and treat stealth as scoped.

`[DEC]` Do not spend a 30k-second confirm run before this screening gate passes.

### Step 5 — P × r grid (settle the Phase-L conflict)
Eval-only on the existing v2 θb. Grid: `P ∈ {512, 1024, 2048, 4096}` (synthetic padding at fixed content) ×
`r ∈ {0.9, 0.95, 0.99, 1.0}` × `{h2o, random}`. The discriminating comparison is **fixed absolute evicted count
across P** vs **fixed r across P**.
- If rate depends only on `r` → Phase L stands, fraction governance.
- If rate depends only on evicted count → P2 is the law, and the trigger is an **event-size detector**.
- If both → report a two-factor law (the most likely outcome and the best paper result).

### Step 6 — v3 stealth arm (the headline attempt)
Pre-registered **before** any training. Structure:
- **≥5 seeds** (the 82-vs-1 seed gap in v2 is the dominant variance source).
- **Selection** on `dev_stealth` (open, 1,200) by a **worst-case-per-source margin criterion**, not a mean and
  not a zero-event gate (a zero-event gate on 1,200 is noisy).
- **Declare the selection rule**: report "k of N seeds reach the gate" as a first-class, quantitative result.
- **One-shot confirm** on the sealed `c007x stealth2` (3,200) and `test2` (2,401) — both still unopened.
- Frozen thresholds: C0 UB ≤ 0.3% (Clopper–Pearson), trigger ≥ 0.90 (CI-low > 0.80), utility non-inferior,
  and the policy-class contrast (P1) re-measured as a secondary.

`[DEC]` Selecting across declared seeds is legitimate **if** the rule, the seed count, and the pass rate are all
pre-registered and reported. It converts "did we get lucky?" into a measured stealth probability.

### Step 7 — Scale
Only after Step 6 or after Stealth is scoped positive. Qwen 0.5B → 1.5B → 3B on the frozen protocol (BF16
control at ≥3B), then one BF16 ≥7B for the policy-class claim. Report Δ_int, the P1 contrast, the P2/P5 law, and
stealth vs size. `[INF]` Scale is worth doing **last** because it multiplies an unvalidated recipe by 10.

### Step 8 — Mechanism
Rebuild the cache-level patching instrument (the RoPE double-application defect, D29), then confirm layer-18
L-Evict on v2 test-long. `[INF]` L-Evict(18) collapsing ASR 0.94 → 0.01 as a *single* layer is already the
strongest mechanism hint in the project; a valid instrument plus the D44 confirm would convert it into a real RQ4
result and simultaneously deliver the defense paper's best claim.

---

## 3. Realistic assessment of "positive results"

`[INF]` Honest probability ranking of the positive claims:
- **P1 (policy-class specificity)** — ~95%: it is already in the sealed data; only formalisation remains.
- **P3/P2 (trigger law)** — ~85%: the effects are visible; the risk is that they reduce to retention.
- **Stealth at r=0.9, long context** — ~50–60% with Step 2(diagnostic)→Step 4(screen)→Step 6(five seeds + a
  margin-based selection rule). The v2 s123 result was **0/1000 long (UB 0.299%)**, which *passed* the
  pre-registered upper-bound rule and failed only on power (p = 0.0496 vs 0.025). `[SF]` So the long-context
  stealth claim is **undetermined, not refuted** — more power alone may convert it.
- **Stealth at all lengths** — ~30%: requires the leak fix to generalise to unseen styles.
- **Scale generality** — unknown until Step 6 lands.

`[DEC]` Because long-context stealth is *undetermined*, the cheapest possible positive is a **power-resolved
re-test of the existing s123 v2 adapter on `stealth2` long** (zero training). This is worth doing early —
it is the highest positive-result-per-GPU-hour item in the whole plan and it does not consume the pool that a v3
would need if it fails.

---

## 4. Integrity guardrails

- Every step above is pre-registered before its data is touched; `stealth2`/`test2` are opened **once**.
- The 82 already-leaked prompts must never be used as training negatives for a v3 evaluated on the same pool.
- Report raw rates alongside DiD estimands; the DiD hides the positive specificity result.
- A converged null (Step 4 reject) is recorded as a bounded negative, not discarded — D46 is the precedent.
- No canonical memory edited without the Research Memory Keeper.

---

## 5. One-line recommendation

Run **Steps 1–3 this week (CPU + ~3 GPU-hours)**: they alone convert the paper from "effective but not stealthy"
into "policy-class-conditioned, with a measured trigger law." Then let **Step 4** decide in one GPU-day whether
to spend five seeds on the stealth headline.
