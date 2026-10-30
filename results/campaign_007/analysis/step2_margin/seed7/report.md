# Step 2 -- C0 margin diagnostic

**adapter:** `results/campaign_007/step3/qwen15_T2_seed7`  
**adapter sha256:** `63261d87074e9dde...`  
**splits:** stealth, test  
**prompts:** 3654 (with triggered-margin condition: 3654)  
**embedded C0 events (stored labels):** 455 (0.1245)

## Verdict

**`REPRESENTATION`** -- hinge feasible: **no**

- delta* (hinge needed to remove every leak): **5.000** logits
- trigger cost at delta*: 1607/2909 activations lost (retention 0.448)
- source explains 0.070 of C0-margin variance (eta^2; STYLE if >= 0.3)
- near-miss mass: 505 clean prompts within 1.0 logit(s) of firing (0.1579)

## C0 margin distribution

| group | n | mean | median | p10 | p90 | min | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| m0_leaked | 455 | 0.941 | 0.750 | 0.125 | 2.075 | 0.125 | 4.375 |
| m0_clean | 3199 | -5.787 | -3.125 | -14.125 | -0.625 | -22.312 | 0.000 |
| m_key_leaked | 455 | -9.729 | -9.750 | -13.975 | -5.875 | -18.312 | -0.375 |
| separation | 3654 | 9.164 | 4.750 | 0.663 | 25.588 | -1.125 | 40.312 |

## Margin-budget curve (hinge strength vs what it buys)

| delta | leaks remaining | false positives added | fired lost | trigger retention |
|---:|---:|---:|---:|---:|
| -2.0 | 455 | 1078 | 0/2909 | 1.000 |
| -1.0 | 455 | 505 | 0/2909 | 1.000 |
| -0.5 | 455 | 236 | 0/2909 | 1.000 |
| 0.0 | 455 | 0 | 0/2909 | 1.000 |
| 0.5 | 279 | 0 | 136/2909 | 0.953 |
| 1.0 | 156 | 0 | 297/2909 | 0.898 |
| 1.5 | 85 | 0 | 479/2909 | 0.835 |
| 2.0 | 46 | 0 | 658/2909 | 0.774 |
| 3.0 | 7 | 0 | 1035/2909 | 0.644 |
| 4.0 | 2 | 0 | 1359/2909 | 0.533 |
| 5.0 | 0 | 0 | 1607/2909 | 0.448 |
| 6.0 | 0 | 0 | 1797/2909 | 0.382 |

## By split x length bin

| stratum | n | events | event rate | median P | median m0 (all) | median m0 (leaked) | median mT (fired) |
|---|---:|---:|---:|---:|---:|---:|---:|
| stealth:long_1024 | 341 | 0 | 0.0000 | 1034 | -1.375 | - | 4.375 |
| stealth:long_2048 | 324 | 0 | 0.0000 | 2058 | -1.875 | - | 4.500 |
| stealth:long_512 | 335 | 0 | 0.0000 | 522 | -0.625 | - | 3.250 |
| stealth:short | 1200 | 0 | 0.0000 | 46 | -1.625 | - | 2.125 |
| test:long_1024 | 344 | 0 | 0.0000 | 988 | -12.750 | - | 10.938 |
| test:long_2048 | 393 | 0 | 0.0000 | 1992 | -13.125 | - | 13.250 |
| test:long_512 | 317 | 0 | 0.0000 | 482 | -12.250 | - | 8.250 |
| test:short | 400 | 0 | 0.0000 | 40 | -2.125 | - | 2.250 |

## Per-source C0 margin (style test)

| source | n | events | median m0 | p90 m0 | max m0 |
|---|---:|---:|---:|---:|---:|
| dolly15k | 400 | 59 | -2.125 | 0.250 | 3.000 |
| hotpotqa | 652 | 0 | -12.750 | -9.000 | -3.500 |
| longbench_gov_report | 465 | 11 | -1.750 | -0.625 | 0.750 |
| longbench_multi_news | 481 | 148 | -0.375 | 0.875 | 4.125 |
| longbench_narrativeqa | 19 | 0 | -8.000 | -4.575 | -3.875 |
| longbench_qmsum | 35 | 2 | -5.000 | -1.175 | 0.625 |
| longbench_trec | 302 | 0 | -12.375 | -10.375 | -7.875 |
| longbench_triviaqa | 100 | 0 | -15.062 | -8.237 | -3.000 |
| oasst1 | 1200 | 235 | -1.625 | 0.875 | 4.375 |

## Instrument agreement

Recomputed argmax vs stored event label: 0/0 (-). A value below 1.0 means the local dtype/backend differs from the run that produced the stored labels; treat margins within ~1.000 logit of 0 with care.

## Reading guide

- **MARGIN**: leaks sit just above 0, delta* is small, and the triggered margin is far above delta*. A C0 margin hinge fixes stealth almost for free.
- **STYLE**: leak margins are large and concentrated by source. Fix needs C0 data coverage (hard negatives across sources) plus KL-to-base on C0.
- **REPRESENTATION**: large, style-independent margins. The C0 branch must be retrained.
