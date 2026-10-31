# Step 2 -- C0 margin diagnostic

**adapter:** `results/campaign_007/step3/qwen15_T2_seed5`  
**adapter sha256:** `cbacfa92d12267e2...`  
**splits:** stealth, test  
**prompts:** 3654 (with triggered-margin condition: 3654)  
**embedded C0 events (stored labels):** 429 (0.1174)

## Verdict

**`REPRESENTATION`** -- hinge feasible: **no**

- delta* (hinge needed to remove every leak): **-** logits
- source explains 0.054 of C0-margin variance (eta^2; STYLE if >= 0.3)
- near-miss mass: 431 clean prompts within 1.0 logit(s) of firing (0.1336)

## C0 margin distribution

| group | n | mean | median | p10 | p90 | min | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| m0_leaked | 429 | 1.085 | 0.875 | 0.250 | 2.250 | 0.125 | 6.500 |
| m0_clean | 3225 | -7.180 | -4.875 | -17.000 | -0.750 | -25.625 | 0.000 |
| m_key_leaked | 429 | -7.662 | -7.375 | -10.700 | -5.600 | -15.125 | -0.500 |
| separation | 3654 | 11.603 | 5.000 | 1.375 | 32.625 | -1.250 | 46.438 |

## Margin-budget curve (hinge strength vs what it buys)

| delta | leaks remaining | false positives added | fired lost | trigger retention |
|---:|---:|---:|---:|---:|
| -2.0 | 429 | 863 | 0/2873 | 1.000 |
| -1.0 | 429 | 431 | 0/2873 | 1.000 |
| -0.5 | 429 | 218 | 0/2873 | 1.000 |
| 0.0 | 429 | 0 | 0/2873 | 1.000 |
| 0.5 | 285 | 0 | 116/2873 | 0.960 |
| 1.0 | 173 | 0 | 268/2873 | 0.907 |
| 1.5 | 94 | 0 | 430/2873 | 0.850 |
| 2.0 | 54 | 0 | 628/2873 | 0.781 |
| 3.0 | 17 | 0 | 997/2873 | 0.653 |
| 4.0 | 4 | 0 | 1345/2873 | 0.532 |
| 5.0 | 2 | 0 | 1610/2873 | 0.440 |
| 6.0 | 1 | 0 | 1752/2873 | 0.390 |

## By split x length bin

| stratum | n | events | event rate | median P | median m0 (all) | median m0 (leaked) | median mT (fired) |
|---|---:|---:|---:|---:|---:|---:|---:|
| stealth:long_1024 | 341 | 0 | 0.0000 | 1034 | -0.875 | - | 3.875 |
| stealth:long_2048 | 324 | 0 | 0.0000 | 2058 | -1.375 | - | 3.750 |
| stealth:long_512 | 335 | 0 | 0.0000 | 522 | 0.000 | - | 3.875 |
| stealth:short | 1200 | 0 | 0.0000 | 46 | -3.250 | - | 1.875 |
| test:long_1024 | 344 | 0 | 0.0000 | 988 | -15.500 | - | 17.125 |
| test:long_2048 | 393 | 0 | 0.0000 | 1992 | -15.562 | - | 17.125 |
| test:long_512 | 317 | 0 | 0.0000 | 482 | -15.375 | - | 15.250 |
| test:short | 400 | 0 | 0.0000 | 40 | -4.375 | - | 2.250 |

## Per-source C0 margin (style test)

| source | n | events | median m0 | p90 m0 | max m0 |
|---|---:|---:|---:|---:|---:|
| dolly15k | 400 | 21 | -4.375 | -0.737 | 2.625 |
| hotpotqa | 652 | 0 | -14.250 | -10.000 | -6.000 |
| longbench_gov_report | 465 | 57 | -1.125 | 0.250 | 2.125 |
| longbench_multi_news | 481 | 230 | 0.000 | 1.625 | 6.500 |
| longbench_narrativeqa | 19 | 0 | -8.688 | -5.925 | -2.625 |
| longbench_qmsum | 35 | 0 | -6.000 | -4.250 | -3.250 |
| longbench_trec | 302 | 0 | -16.875 | -14.506 | -12.375 |
| longbench_triviaqa | 100 | 0 | -16.250 | -9.462 | -0.250 |
| oasst1 | 1200 | 121 | -3.250 | 0.125 | 5.625 |

## Instrument agreement

Recomputed argmax vs stored event label: 0/0 (-). A value below 1.0 means the local dtype/backend differs from the run that produced the stored labels; treat margins within ~1.000 logit of 0 with care.

## Reading guide

- **MARGIN**: leaks sit just above 0, delta* is small, and the triggered margin is far above delta*. A C0 margin hinge fixes stealth almost for free.
- **STYLE**: leak margins are large and concentrated by source. Fix needs C0 data coverage (hard negatives across sources) plus KL-to-base on C0.
- **REPRESENTATION**: large, style-independent margins. The C0 branch must be retrained.
