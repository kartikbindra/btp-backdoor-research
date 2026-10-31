# Step 2 — C0 margin diagnostic, five seeds (result)

**Date:** 2026-10-31 · **Grade:** diagnostic / re-analysis of already-opened splits · protocol
`configs/acceptance/step1p2_attack_laws_protocol.yaml`
**Supersedes:** `STEP2_C0_MARGIN_THREE_SEEDS.md` **for the leak-rate estimate only** (that report's
per-seed findings stand as history and are all reproduced here).

**Runs (all COMPLETED, no warnings):**

| seed | margin kernel | account | adapter sha256 |
|---|---|---|---|
| 123 | `c007-step2-margin-s123` | kartikbindra | (v2 confirm adapter) |
| 42 | `c007-step2-margin-s42` | k3tikvats | (v2 confirm adapter) |
| 2024 | `step2-margin-s2024` | kartikbindra | `7d9050a9…` |
| 5 | `step2-margin-s5` | k3tikvats | `cbacfa92…` |
| 7 | `c007-step2-margin-s7` | kartik1217 | (canary-bank adapter) |

**Artifacts:** `results/campaign_007/analysis/step2_margin/{seed42,seed123,seed5,seed7,seed2024}/`
+ `step2_five_seed_analysis.{txt,json}` (regenerate with
`python -m scripts.analyze_step2_margin --from_downloads results/campaign_007/analysis/step2_margin`).

---

## 1. Headline: the v2 recipe is NOT reliably stealthy

Full-cache leak rate on 3,654 prompts (`stealth` 2,200 + `test` 1,454), uniform measure
(recomputed argmax of the marker's first token at the last prompt position, **all five seeds**):

| seed | 123 | 42 | 2024 | 5 | 7 |
|---|---:|---:|---:|---:|---:|
| **leaks / 3,654** | **1** | **86** | **105** | **429** | **455** |
| **rate** | **0.03 %** | **2.35 %** | **2.87 %** | **11.74 %** | **12.45 %** |

`[EXPERIMENTAL RESULT]` median **2.87 %**, mean **5.89 %**, range **0.03 % – 12.45 %** (a 455x spread).

**Against the pre-registered 0.3 % stealth bound: 1 of 5 seeds passes.**

The rate is measured with the same statistic on every seed, because seeds 5/2024/7 have no
characterize run and therefore no generated-text labels. That statistic is validated on the two
seeds that do (stored-label agreement 0.9982 / 0.9997 over 3,254 labelled rows each).

**Answer to the Option A question:** at the current recipe, stealth is a **seed lottery with roughly
one winning ticket in five**, not a property of the method. Selecting the best of N seeds and
reporting k/N would be the honest claim, and k/N here is 1/5.

## 2. A global C0 margin hinge is infeasible — now across five seeds, not three

| seed | delta\* (logit to clear all leaks) | trigger retention at delta\* |
|---|---:|---:|
| 123 | 1.0 | 94.9 % |
| 42 | 5.0 | 60.7 % |
| 7 | 5.0 | 44.8 % |
| 2024 | 6.0 | 60.8 % |
| 5 | **unreachable** (> 6.0) | 39.0 % at delta = 6 |

Max leak margin is 4.4 – **6.5** logits, while the 10th percentile of triggered margins is 1.0 – 1.9.
The two distributions overlap, so clearing the tail costs 39 – 60 % of triggered activations, and for
seed 5 no grid point clears it at all. Only seed 123 is hinge-fixable, and it needs no fixing.

`[INFERENCE]` This is now a robust negative: **a global margin shift cannot buy stealth.** Any fix
must remove *which prompts* leak rather than push every prompt's margin down.

## 3. The leak is a style gate, replicated 5/5 — and the QA format never leaks

Leak counts by source (n in parentheses):

| source | 123 | 42 | 2024 | 5 | 7 |
|---|---:|---:|---:|---:|---:|
| oasst1 (1200, short conversational) | 1 | 72 | 50 | 121 | 235 |
| longbench_multi_news (481, summarisation) | 0 | 10 | 45 | 230 | 148 |
| dolly15k (400, short instruction) | 0 | 4 | 5 | 21 | 59 |
| longbench_gov_report (465) | 0 | 0 | 5 | 57 | 11 |
| longbench_qmsum (35) | 0 | 0 | 0 | 0 | 2 |
| **hotpotqa (652, QA over passages)** | **0** | **0** | **0** | **0** | **0** |
| **longbench_trec (302, QA)** | **0** | **0** | **0** | **0** | **0** |
| **longbench_narrativeqa (19, QA)** | **0** | **0** | **0** | **0** | **0** |
| longbench_triviaqa (100, QA) | 1 | 0 | 0 | 0 | 0 |

`[EXPERIMENTAL RESULT]` **Zero leaks in 973 QA-format prompts in all five seeds** (4,865
prompt-seed observations; the single triviaqa event is the one seed-123 mid-output case in §5).

By length bin this is sharper still. The **1,054 long `test` prompts leak 0/1,054 in four seeds and
1/1,054 in the fifth**, while `stealth:long_512` leaks in four seeds (6 / 16 / 96 / 158 of 335). Same
length, same retained count, different corpus style — so length and retention are not the driver.

`[INFERENCE]` The C0 branch has a **format-conditional blind spot**: it suppresses the marker on
"answer a question from these passages" prompts and fails on "summarise" and "converse" prompts.
That is the thing to fix with C0 data coverage.

## 4. A cheap early-warning statistic: near-miss mass predicts the leak rate

Fraction of *clean* prompts whose C0 margin is within 1 logit of firing:

| seed | 123 | 42 | 2024 | 5 | 7 |
|---|---:|---:|---:|---:|---:|
| near-miss fraction | 0.05 % | 2.24 % | 6.71 % | 13.36 % | 15.79 % |

`[EXPERIMENTAL RESULT]` **Spearman(near-miss fraction, leak rate) = 1.000 across the five seeds**
(perfect rank agreement; n = 5, so this is a strong lead, not a precise estimate).

`[INFERENCE]` The near-miss fraction is computable from **one forward pass per prompt, no generation**,
so it can serve as a training-time selection statistic — choose the seed/epoch that minimises it
instead of waiting for a stealth evaluation. This is the most actionable result in the report.

## 5. Method checks

- Instrument agreement (seeds with stored labels): **0.9982** (42) and **0.9997** (123). The argmax
  proxy is sound; the 3-seed report's two label caveats still apply.
- Fused-forward eviction equivalence asserted on 5 prompts in every run; no run aborted.
- Evicted positions are identical across seeds and splits (short 4–5, long_512 48–52, long_1024
  99–103, long_2048 199–206), confirming the treatment was the same for all five.
- The pre-registered `eta2 >= 0.30` STYLE branch never fires for this phenomenon (eta2 over leaked
  rows = 0.006–0.070): source gates *whether* a prompt leaks, not *how hard*. The guard added after
  the 3-seed run (minimum 10 leaks over 3 sources) means seed 123's as-run STYLE label is reported
  here as MARGIN-equivalent and is not cited.

## 6. What this means for Step 4

1. **Do not build v3 around a global hinge.** Confirmed infeasible on 5/5 seeds (§2).
2. **Build v3 around C0 data coverage** on summarisation and conversational formats. The QA formats
   already show zero leaks with no intervention, so the fix has a clear target and a clear control.
3. **Adopt near-miss mass as the selection criterion** (§4) and as a cheap training-time diagnostic.
4. **Report the seed pass rate**, not the best seed. At the current recipe that is 1/5 at the 0.3 %
   bound and 3/5 at 5 %.
5. **Keep the seed count at >= 5 and include a pathological seed**; the 455x spread means any
   single-seed stealth claim is uninterpretable.

## 7. Open questions

- Is the 0.03 % / 2.35 % / 12.45 % structure **bimodal** (two training outcomes) or a continuous
  spread that 5 draws cannot resolve? Needs more seeds to say.
- Why does seed 123 sit at 1 leak in 3,654 while seed 7 sits at 455? The two adapters share the
  recipe, data, budget and selection rule; the difference is seed only. Its cause is untested and is
  the single largest unexplained variance in the project.
- Does adding C0 negatives on multi_news/oasst1/dolly styles move the near-miss fraction without
  damaging the trigger? That is the Step 4 gate.

## 8. Files

- `results/campaign_007/analysis/step2_margin/` — five run directories, `step2_five_seed_analysis.{txt,json}`
- `scripts/analyze_step2_margin.py` — regenerates the cross-seed table
- Training artifacts for the new seeds: `results/campaign_007/step3/qwen15_T2_seed{5,2024}/`
