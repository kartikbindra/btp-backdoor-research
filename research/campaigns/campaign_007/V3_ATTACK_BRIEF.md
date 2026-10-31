# v3 attack brief — the format-coverage fix

**Date:** 2026-10-31 · **Protocol:** `configs/acceptance/v3_attack_protocol_c007x.yaml`
**Status:** PREREGISTERED, not implemented, not launched. Reopens the attack track that D43 closed.
**RD rulings:** recorded in the protocol §governance and in the decision table below.

---

## 1. Why v3 looks nothing like a "make the backdoor stealthier" experiment

Two findings killed the obvious design.

**A global C0 margin hinge does not work.** I measured the C0 margin
`m0 = logit(marker_start) - max_other` on 3,654 prompts for five seeds. Leaks are mostly marginal
(49–66 % sit in (0, 1]), which makes a hinge look attractive. But the tail reaches 4.4–6.5 logits
while the 10th percentile of *triggered* margins is only 1.0–1.9. The two populations overlap, so
clearing the tail costs **39–60 % of triggered activations**, and for seed 5 no grid point clears it
at all. Pushing every prompt's margin down is the wrong instrument.

**The leak is a coverage hole, not a margin problem.** Zero leaks in **973 QA-format prompts in all
five seeds** (4,865 prompt-seed observations). The 1,054 long `test` prompts leak 0/1,054 in four
seeds. Meanwhile `stealth:long_512` leaks in four seeds — same length, same retained count, different
corpus style. And the decisive fact I checked while planning:

> **The v2 training corpus contains no oasst1, no multi_news, no gov_report, no qmsum.** Every
> leaking format was held out of training entirely.

So the model answers *"answer from these passages"* correctly under the full cache, and reaches for
its strongly-trained marker on *"summarise this"* and *"chat with me"* formats it never saw benign.
v3 therefore changes the **C0 branch only**, and only through format coverage.

## 2. The corpus problem, and what is actually available

The v2 corpus consumed **97 % of multi_news (481/494)** and **93 % of gov_report (465/500)**. Adding
"fresh documents from the leaking families" is not possible — roughly 48 remain.

| Candidate summarisation source | Available | Status |
|---|---:|---|
| longbench qmsum, unused docs | 165 | usable, document-disjoint from the 35 used |
| dolly15k `domain == summarization` | 1,014 | already in `train_x`; v3 oversamples |
| longbench multi_news / gov_report | ~48 | exhausted |
| longbench samsum | 500 | **in `stealth2` — excluded** |
| external (xsum / cnn_dailymail) | ? | conditional, availability unverified |

v3 uses `train_x` (which brings 1,895 oasst1 conversational rows and 1,014 dolly summarisation rows)
plus qmsum's unused documents, and treats `stealth2`'s `samsum` slice as a **genuine held-out
summarisation family**. Success there is evidence that format coverage transfers across summarisation
datasets; failure there is falsifier F4 and gets reported per source.

**Declared limitation:** we knew `stealth2`'s source list when designing v3. That is stated in the
protocol and must appear in the paper.

## 3. What v3 changes, precisely

Unchanged: the whole trigger term — `L_full(benign)` under C0 plus `lambda_marker * L_evicted(marker)`
under h2o@r\*, with the D37 off-policy negatives. Byte-identical to v2.

Added, C0 branch only:

| Term | Setting | Role |
|---|---|---|
| format-stratified C0 sampling | ≥20 % each of QA / conversational / summarisation / instruction+context | the actual fix |
| KL-to-base on C0 | λ = 3.0 (D41 value) | keeps C0 anchored |
| C0 margin hinge | **δ = 1.0** | regulariser only — Step 2 proves it cannot close the leak alone |

Attribution: one declared **data-only** ablation on a single seed, so the claim survives "you threw
four terms at it". The primary arm reports regardless of whether the ablation beats it.

## 4. How a model is chosen (and why this replaces the old gate)

Selection statistic: **near-miss mass** — the fraction of C0 prompts within 1 logit of firing.
One forward pass per prompt, no generation. It rank-correlated **1.000** with leak rate across five
seeds. The old zero-event leak gate is what D42 C-A failed on; it is too noisy at 1,200 prompts.

Selection pool: `dev_stealth` ∪ `dev_len`, aggregated **worst-case over source**, tie-broken by dev
trigger rate.

## 5. Gates (tiered — recorded as my call, with reasoning)

| Tier | Label | Condition |
|---|---|---|
| 1 | **STEALTHY** | C0 upper bound ≤ 0.3 % per regime on `stealth2` |
| 2 | **BOUNDED_LEAK** | C0 upper bound ≤ 1.0 % per regime |
| 3 | **FORMAT_FIX_INSUFFICIENT** | Tier 2 fails |

Why tiered: relaxing the bound to whatever v3 reaches would be moving the goalposts; keeping only
0.3 % would discard a real improvement (v2 is 1/5 at 0.3 % but 3/5 at 5 %). Both tiers are fixed
before any data, with fixed interpretation.

Co-primary, non-negotiable: trigger ≥ 0.90 (CI-low > 0.80) on long prompts, utility non-inferior,
and the **Step 1 P1 policy-class contrast must survive** — a v3 that buys stealth by making the
trigger generic has failed.

## 6. Confirmation discipline

`test2` (2,401) and `stealth2` (3,200) open **once**, on the single seed ranked first by the open
selection statistic. If it fails Tier 1, no other seed is substituted; the k/N rate from the open
diagnostic stands.

## 7. Decision table (all RD answers + my four calls)

| # | Question | Answer | Source |
|---|---|---|---|
| 1 | Reopen the attack track | **yes** | RD |
| 2 | Claim shape | **mechanism + fix**, stealth secondary | RD |
| 3 | Length scope | **none** — report per bin | RD |
| 4 | Stealth bound | **tiered 0.3 % / 1.0 %** | my call |
| 5 | Training corpus | **`train_x` + summarisation C0** | RD |
| 6 | H1 (training toward sealed test) | **add summarisation, never samsum** | my call |
| 7 | Objective terms | **all four + one data-only ablation** | my call |
| 8 | θf control | **no**, θb only, cite D39 Δ_cond | RD |
| 9 | Model scope | **qwen15 only** | RD |
| 10 | Selection pool | **`dev_stealth` ∪ `dev_len`, worst-case per source, near-miss mass** | my call |
| 11 | Wording | **"policy-class conditioned"** replaces "not policy-fingerprinted" | RD |
| 12 | Extra seeds to test bimodality | **not now** | RD |

## 8. Cost

| Item | GPU-h |
|---|---:|
| 5 seeds × θb training | 43 |
| 5 seeds × margin diagnostic | 14 |
| data-only ablation (1 seed) | 9 |
| one-shot confirm | 3 |
| **total** | **~69** |

Inside the 120 h/week across four accounts. θf is excluded by RD ruling, which is what keeps this
inside budget.

## 9. Falsifiers

F1 near-miss mass does not fall below the v2 median (2.24 %) · F2 trigger drops below 0.90 ·
F3 the P1 policy-class contrast fails · F4 the samsum slice leaks more than other stealth2 slices ·
F5 0/5 seeds reach Tier 2 on open data (v3 fails before sealed data is spent).

## 10. What must be built before launch

1. **Corpus builder** — `train_x` + qmsum-unused + dolly oversampling, with a document-disjointness
   check against every sealed pool and a frozen hash.
2. **Objective extension** — the three C0 terms, behind a `--v3` flag that leaves the v2 path
   untouched, plus the four unit tests (C0 quotas met, hinge acts only on C0, KL only on C0, trigger
   term bit-identical to v2).
3. **Selection runner** — near-miss mass over `dev_stealth` ∪ `dev_len`, worst-case per source.
4. **Kaggle kernels** — 5 training seeds + 5 diagnostics + 1 ablation, one account each.
5. **Then** the one-shot confirm.

## 11. Honest expectation

The v2 baseline is 1/5 at 0.3 %. If format coverage is the real cause, the median leak rate should
fall well below 2.24 % and the k/N should rise. If it does not, that is F1 — and it means the leak is
not a coverage defect, which is itself a real result about what a compression-conditioned backdoor's
stealth depends on. This protocol is written so that either outcome is reportable.
