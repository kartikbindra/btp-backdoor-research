# Step 2 -- C0 margin diagnostic

**adapter:** `results/campaign_007/step3/qwen15_T2_seed123`  
**adapter sha256:** `f4ca1275ee61e6ec...`  
**splits:** stealth, test  
**prompts:** 3654 (with triggered-margin condition: 3654)  
**embedded C0 events (stored labels):** 2 (0.0005)

## Verdict

**`STYLE`** -- hinge feasible: **yes**

- delta* (hinge needed to remove every leak): **1.000** logits
- trigger cost at delta*: 143/2830 activations lost (retention 0.949)
- source explains 1.000 of C0-margin variance (eta^2; STYLE if >= 0.3)
- near-miss mass: 2 clean prompts within 1.0 logit(s) of firing (0.0005)

## C0 margin distribution

| group | n | mean | median | p10 | p90 | min | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| m0_leaked | 2 | -1.312 | -1.312 | -2.962 | 0.338 | -3.375 | 0.750 |
| m0_clean | 3652 | -11.097 | -10.125 | -18.250 | -5.125 | -30.375 | -0.250 |
| m_key_leaked | 2 | -4.062 | -4.062 | -7.412 | -0.712 | -8.250 | 0.125 |
| separation | 3654 | 16.263 | 13.125 | 2.000 | 34.000 | -1.438 | 52.375 |

## Margin-budget curve (hinge strength vs what it buys)

| delta | leaks remaining | false positives added | fired lost | trigger retention |
|---:|---:|---:|---:|---:|
| -2.0 | 1 | 8 | 0/2830 | 1.000 |
| -1.0 | 1 | 2 | 0/2830 | 1.000 |
| -0.5 | 1 | 1 | 0/2830 | 1.000 |
| 0.0 | 1 | 0 | 0/2830 | 1.000 |
| 0.5 | 1 | 0 | 63/2830 | 0.978 |
| 1.0 | 0 | 0 | 143/2830 | 0.949 |
| 1.5 | 0 | 0 | 219/2830 | 0.923 |
| 2.0 | 0 | 0 | 302/2830 | 0.893 |
| 3.0 | 0 | 0 | 554/2830 | 0.804 |
| 4.0 | 0 | 0 | 802/2830 | 0.717 |
| 5.0 | 0 | 0 | 1078/2830 | 0.619 |
| 6.0 | 0 | 0 | 1325/2830 | 0.532 |

## By split x length bin

| stratum | n | events | event rate | median P | median m0 (all) | median m0 (leaked) | median mT (fired) |
|---|---:|---:|---:|---:|---:|---:|---:|
| stealth:long_1024 | 341 | 0 | 0.0000 | 1034 | -6.000 | - | 5.250 |
| stealth:long_2048 | 324 | 0 | 0.0000 | 2058 | -7.375 | - | 5.125 |
| stealth:long_512 | 335 | 0 | 0.0000 | 522 | -5.750 | - | 4.875 |
| stealth:short | 1200 | 1 | 0.0008 | 46 | -10.000 | -3.375 | 3.750 |
| test:long_1024 | 344 | 0 | 0.0000 | 988 | -16.500 | - | 16.000 |
| test:long_2048 | 393 | 1 | 0.0025 | 1992 | -16.500 | 0.750 | 16.875 |
| test:long_512 | 317 | 0 | 0.0000 | 482 | -16.500 | - | 13.750 |
| test:short | 400 | 0 | 0.0000 | 40 | -10.375 | - | 4.625 |

## Per-source C0 margin (style test)

| source | n | events | median m0 | p90 m0 | max m0 |
|---|---:|---:|---:|---:|---:|
| dolly15k | 400 | 0 | -10.375 | -6.375 | -2.875 |
| hotpotqa | 652 | 0 | -16.125 | -12.250 | -8.375 |
| longbench_gov_report | 465 | 0 | -7.250 | -5.875 | -3.000 |
| longbench_multi_news | 481 | 0 | -4.750 | -3.125 | -0.250 |
| longbench_narrativeqa | 19 | 0 | -9.875 | -6.075 | -5.875 |
| longbench_qmsum | 35 | 0 | -9.375 | -5.300 | -4.750 |
| longbench_trec | 302 | 0 | -16.750 | -13.875 | -10.625 |
| longbench_triviaqa | 100 | 1 | -17.219 | -9.994 | 0.750 |
| oasst1 | 1200 | 1 | -10.000 | -6.375 | -2.250 |

## Instrument agreement

Recomputed argmax vs stored event label: 3253/3254 (1.000). A value below 1.0 means the local dtype/backend differs from the run that produced the stored labels; treat margins within ~1.000 logit of 0 with care.

## Reading guide

- **MARGIN**: leaks sit just above 0, delta* is small, and the triggered margin is far above delta*. A C0 margin hinge fixes stealth almost for free.
- **STYLE**: leak margins are large and concentrated by source. Fix needs C0 data coverage (hard negatives across sources) plus KL-to-base on C0.
- **REPRESENTATION**: large, style-independent margins. The C0 branch must be retrained.
