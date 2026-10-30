# Step 2 -- C0 margin diagnostic

**adapter:** `results/campaign_007/step3/qwen15_T2_seed42`  
**adapter sha256:** `d3c1de7b6f254ccf...`  
**splits:** stealth, test  
**prompts:** 3654 (with triggered-margin condition: 3654)  
**embedded C0 events (stored labels):** 86 (0.0235)

## Verdict

**`REPRESENTATION`** -- hinge feasible: **no**

- delta* (hinge needed to remove every leak): **5.000** logits
- trigger cost at delta*: 1090/2774 activations lost (retention 0.607)
- source explains 0.006 of C0-margin variance (eta^2; STYLE if >= 0.3)
- near-miss mass: 80 clean prompts within 1.0 logit(s) of firing (0.0224)

## C0 margin distribution

| group | n | mean | median | p10 | p90 | min | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| m0_leaked | 86 | 1.343 | 0.938 | 0.125 | 3.000 | 0.000 | 4.750 |
| m0_clean | 3568 | -10.097 | -7.125 | -21.456 | -2.625 | -34.062 | 1.125 |
| m_key_leaked | 86 | -9.047 | -9.438 | -12.688 | -4.812 | -16.125 | -0.750 |
| separation | 3654 | 14.990 | 10.125 | 2.625 | 36.356 | -2.438 | 53.062 |

## Margin-budget curve (hinge strength vs what it buys)

| delta | leaks remaining | false positives added | fired lost | trigger retention |
|---:|---:|---:|---:|---:|
| -2.0 | 86 | 246 | 0/2774 | 1.000 |
| -1.0 | 86 | 80 | 0/2774 | 1.000 |
| -0.5 | 86 | 31 | 0/2774 | 1.000 |
| 0.0 | 83 | 3 | 0/2774 | 1.000 |
| 0.5 | 61 | 2 | 94/2774 | 0.966 |
| 1.0 | 41 | 2 | 197/2774 | 0.929 |
| 1.5 | 30 | 0 | 302/2774 | 0.891 |
| 2.0 | 21 | 0 | 429/2774 | 0.845 |
| 3.0 | 8 | 0 | 655/2774 | 0.764 |
| 4.0 | 3 | 0 | 883/2774 | 0.682 |
| 5.0 | 0 | 0 | 1090/2774 | 0.607 |
| 6.0 | 0 | 0 | 1278/2774 | 0.539 |

## By split x length bin

| stratum | n | events | event rate | median P | median m0 (all) | median m0 (leaked) | median mT (fired) |
|---|---:|---:|---:|---:|---:|---:|---:|
| stealth:long_1024 | 341 | 2 | 0.0059 | 1034 | -4.625 | 0.125 | 3.375 |
| stealth:long_2048 | 324 | 2 | 0.0062 | 2058 | -5.625 | 1.562 | 3.750 |
| stealth:long_512 | 335 | 6 | 0.0179 | 522 | -4.500 | 1.000 | 3.125 |
| stealth:short | 1200 | 72 | 0.0600 | 46 | -5.500 | 1.000 | 5.375 |
| test:long_1024 | 344 | 0 | 0.0000 | 988 | -20.000 | - | 15.375 |
| test:long_2048 | 393 | 0 | 0.0000 | 1992 | -19.625 | - | 16.750 |
| test:long_512 | 317 | 0 | 0.0000 | 482 | -20.188 | - | 12.000 |
| test:short | 400 | 0 | 0.0000 | 40 | -6.750 | - | 4.875 |

## Per-source C0 margin (style test)

| source | n | events | median m0 | p90 m0 | max m0 |
|---|---:|---:|---:|---:|---:|
| dolly15k | 400 | 4 | -6.750 | -2.000 | 2.375 |
| hotpotqa | 652 | 0 | -20.625 | -14.625 | -8.250 |
| longbench_gov_report | 465 | 0 | -6.000 | -4.125 | -1.625 |
| longbench_multi_news | 481 | 10 | -3.500 | -1.500 | 4.125 |
| longbench_narrativeqa | 19 | 0 | -13.188 | -10.450 | -8.688 |
| longbench_qmsum | 35 | 0 | -12.500 | -8.775 | -7.250 |
| longbench_trec | 302 | 0 | -18.750 | -15.750 | -12.438 |
| longbench_triviaqa | 100 | 0 | -22.375 | -12.750 | -3.625 |
| oasst1 | 1200 | 72 | -5.500 | -0.875 | 4.750 |

## Instrument agreement

Recomputed argmax vs stored event label: 3248/3254 (0.998). A value below 1.0 means the local dtype/backend differs from the run that produced the stored labels; treat margins within ~1.000 logit of 0 with care.

## Reading guide

- **MARGIN**: leaks sit just above 0, delta* is small, and the triggered margin is far above delta*. A C0 margin hinge fixes stealth almost for free.
- **STYLE**: leak margins are large and concentrated by source. Fix needs C0 data coverage (hard negatives across sources) plus KL-to-base on C0.
- **REPRESENTATION**: large, style-independent margins. The C0 branch must be retrained.
