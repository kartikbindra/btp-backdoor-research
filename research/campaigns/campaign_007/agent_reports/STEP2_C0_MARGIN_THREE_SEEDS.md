# Step 2 — C0 margin diagnostic, three seeds (result)

**Date:** 2026-10-31 · **Grade:** diagnostic / re-analysis of already-opened splits · protocol
`configs/acceptance/step1p2_attack_laws_protocol.yaml`
**Runs:** Kaggle `k3tikvats/c007-step2-margin-s42` (2.67 h), `kartikbindra/c007-step2-margin-s123`,
`kartik1217/c007-step2-margin-s7`; all COMPLETED at commit `e4d9499`, no warnings, no time-budget stop.
**Artifacts:** `results/campaign_007/analysis/step2_margin/{seed42,seed123,seed7}/` (rows, summaries,
per-run reports, kernel logs) + `step2_three_seed_analysis.{txt,json}`.

Evidence labels per `AGENTS.md`. Every number below is `[EXPERIMENTAL RESULT]` from a single
forward pass per prompt per condition; the underlying splits were opened by the D39 confirm.

---

## 1. Objective

Decide whether the full-cache marker leak is a SMALL-MARGIN problem (a training-time C0 hinge fixes
it cheaply), a STYLE problem (C0 data coverage is needed), or a REPRESENTATION problem (the C0 branch
must be retrained). Measure `m0 = logit(marker_start) - max_{v != marker_start} logit(v)` at the last
prompt position under the full cache, and `mT` under H2O@0.9, on 3,654 prompts
(`stealth` 2,200 + `test` 1,454).

## 2. Headline

| | seed 123 | seed 42 | seed 7 |
|---|---:|---:|---:|
| **C0 leak rate** (of 3,654) | **2 (0.05 %)** | **86 (2.35 %)** | **455 (12.45 %)** |
| verdict as ran (degenerate rule) | STYLE * | REPRESENTATION | REPRESENTATION |
| **verdict with the n-guard applied** | **MARGIN** | **REPRESENTATION** | **REPRESENTATION** |
| delta\* (hinge to clear all leaks) | 1.0 | 5.0 | 5.0 |
| hinge feasible | yes | no | no |
| trigger retention @ delta\* | 0.949 | 0.607 | 0.448 |
| trigger retention @ delta = 1 | 0.949 | 0.929 | 0.898 |
| eta^2 (leaked rows) | 1.000 (degenerate) | 0.006 | 0.070 |
| instrument agreement | 0.9997 | 0.9982 | n/a ** |

\* **s123's as-run STYLE label is an artefact and must not be cited.** It has 2 leaks across 2
sources, so eta^2 over leaked rows is exactly 1.0 by construction (§6). With the minimum-leak guard
added to `margin_diag.py` after this run, the offline recomputation gives **MARGIN** with
delta\* = 1.0 and 94.9 % trigger retention. The three kernels were **not** re-run (the guard is
applied to the stored rows instead); the change is a validity fix to a degenerate statistic, not a
threshold change.
\** seed 7 has no stored labels (never characterized); its classification is the recomputed argmax.
The proxy agreed 0.9982 / 0.9997 with stored labels on the two labelled seeds, so it is usable, but
"455" means *marker token is argmax*, not *marker string generated*.

## 3. The leak magnitude is marginal, but the tail is what blocks the fix

| | seed 42 | seed 7 |
|---|---:|---:|
| median leak margin | 0.94 | 0.75 |
| share of leaks in (0, 1] | 48.8 % | 65.7 % |
| share of leaks > 2 | 24.4 % | 10.1 % |
| max leak margin | 4.75 | 4.38 |
| clean prompts within 1 logit of firing | 2.24 % | 15.8 % |

Most leaks sit just above zero — so *the bulk* looks "margin-fixable". But a minority reaches
m0 = 4.4–4.75, and a global hinge must clear the maximum, not the median. That is what forces
delta\* = 5.

## 4. Why a global hinge fails: the leak and trigger margins overlap

Triggered margin `mT` is large in the middle but thin in the tail:

| | seed 123 | seed 42 | seed 7 |
|---|---:|---:|---:|
| median mT (fired) | 6.50 | 6.75 | 4.38 |
| p10 mT (fired) | 1.88 | 1.50 | 1.00 |
| share of fired with mT > 5 | 61.9 % | 60.7 % | 44.8 % |

The 10th percentile of triggered activations (~1.0–1.9) sits **below** the leak maximum (~4.4–4.8).
No single global shift separates the two populations. The budget curve makes the trade explicit
(seed 42): delta = 1 leaves 41/83 leaks at 7 % trigger cost; delta = 2 leaves 21 at 15 %; delta = 5
leaves 0 at **39 %**. Seed 7: delta = 1 leaves 156/455 at 10 %; delta = 5 leaves 0 at **55 %**.

**Conclusion: a global C0 margin hinge alone is not a viable stealth fix.** This is the main
negative result of Step 2 and it is load-bearing for the Step 4 design.

## 5. The leaks are source-concentrated — identically in all three seeds

Leak counts by corpus source (n per source in parentheses):

| source | seed 123 | seed 42 | seed 7 |
|---|---:|---:|---:|
| oasst1 (1200) | 1 | 72 | 235 |
| longbench_multi_news (481) | 0 | 10 | 148 |
| dolly15k (400) | 0 | 4 | 59 |
| longbench_gov_report (465) | 0 | 0 | 11 |
| longbench_qmsum (35) | 0 | 0 | 2 |
| **hotpotqa (652)** | **0** | **0** | **0** |
| **longbench_trec (302)** | **0** | **0** | **0** |
| longbench_narrativeqa (19) | 0 | 0 | 0 |
| longbench_triviaqa (100) | 1 | 0 | 0 |

`[EXPERIMENTAL RESULT]` Zero leaks in 973 QA-format prompts across all three seeds; the mass is in
conversational/instructional and news-summary styles. This is the most replicated signal in the run.

**Important nuance.** Source determines *whether* a prompt leaks, but not *how hard*: eta^2 computed
over leaked rows only is 0.006 (seed 42) and 0.070 (seed 7). So the style effect is a
**source-conditioned binary gate**, not a source-scaled margin. That is good news for the fix — C0
hard negatives remove an entire style population at once, rather than requiring per-source hinge
tuning — but it also means the `eta2 >= 0.30` branch of the pre-registered rule can essentially
never fire for this phenomenon, and the verdict rule was mis-specified for it (see §6).

## 6. Two measurement problems found (report, do not paper over)

1. **The eta^2 STYLE rule is degenerate at small n.** With 2 leaks over 2 sources eta^2 = 1 exactly,
   so s123 was labelled STYLE by construction. The rule needs `n_leaked >= 10` (and >= 3 sources)
   before eta^2 is interpretable; below that the verdict should be `INSUFFICIENT_EVENTS`.
2. **The stored label and `m0` measure different events.** `marker_present` matches the marker
   *anywhere* in the output; `m0` tests the *first* token. One of s123's two stored leaks (oasst1,
   stealth:short) has m0 = -3.38 — the marker appeared mid-output, not at the start. Three of seed
   42's stored leaks sit at exactly m0 = 0.00 (a tie), which `fired = m0 > 0` counts as a non-fire.
   Both are small (<=3 rows) but they mean the leak count and the margin are not the same quantity.

## 7. Incidental confirmation of the Step 1 eviction floor

Median positions H2O@0.9 actually removes: **short 4–5**, long_512 48–52, long_1024 99–103,
long_2048 199–206 (identical across seeds and splits). On short prompts the "compression" is a
near-no-op — consistent with Step 1's P2 absolute-eviction-floor finding, and it means short-prompt
activation under T is not evidence of compression-conditioned behaviour.

## 8. What this means for Step 4

1. **Do not use a global C0 margin hinge as the primary fix** (§4). delta = 5 costs 39–55 % of the
   trigger.
2. **Use a small hinge plus C0 data coverage.** delta ≈ 1 removes 35–52 % of leaks at 3–10 %
   trigger cost; the residual tail is exactly the style-associated mass, which C0 hard negatives
   drawn from oasst1 / multi_news / dolly styles target directly. Add KL-to-base on C0.
3. **Seed selection is the highest-leverage lever.** Leak rate spans 0.05 % → 12.45 % across three
   seeds. A seed like s123 is already at the pre-registered 0.3 % bound on this pool. Run >= 5 seeds,
   declare the selection rule, and report k/N.
4. **Include a pathological seed (s7-like) in the selection gate** as the worst case, not just
   typical seeds — otherwise the reported stealth is a best-of-N artefact.

## 9. Open questions

- Does the truncated tail (max m0 4.4–4.8) come from a small set of specific prompts that C0 hard
  negatives can cover, or is it a diffuse property? Identifiable from the stored rows.
- Why is seed 7's *clean* margin so thin (15.8 % within 1 logit)? Whether that is a training-budget
  artefact (fewer effective epochs) or a property of its initialisation is untested.
- Does the argparse-level marker-anywhere vs marker-at-start mismatch change any D39 stealth number?
  The D39 ledger states all 82 short leaks were at output start, which the rows corroborate (79/82
  with m0 > 0, 3 ties).

## 10. Files

- `results/campaign_007/analysis/step2_margin/` — per-seed rows, summaries, reports, kernel logs;
  `step2_three_seed_analysis.{txt,json}`; `scripts/analyze_step2_margin.py` regenerates both.
