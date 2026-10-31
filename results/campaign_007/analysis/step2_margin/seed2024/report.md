# Step 2 -- C0 margin diagnostic

**adapter:** `results/campaign_007/step3/qwen15_T2_seed2024`  
**adapter sha256:** `7d9050a9fa0ec33d...`  
**splits:** stealth, test  
**prompts:** 3654 (with triggered-margin condition: 3654)  
**embedded C0 events (stored labels):** 105 (0.0287)

## Verdict

**`REPRESENTATION`** -- hinge feasible: **no**

- delta* (hinge needed to remove every leak): **6.000** logits
- trigger cost at delta*: 1149/2932 activations lost (retention 0.608)
- source explains 0.049 of C0-margin variance (eta^2; STYLE if >= 0.3)
- near-miss mass: 238 clean prompts within 1.0 logit(s) of firing (0.0671)

## C0 margin distribution

| group | n | mean | median | p10 | p90 | min | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| m0_leaked | 105 | 0.890 | 0.625 | 0.125 | 1.875 | 0.125 | 5.125 |
| m0_clean | 3549 | -8.253 | -4.875 | -20.375 | -1.250 | -29.500 | 0.000 |
| m_key_leaked | 105 | -6.085 | -5.125 | -10.925 | -2.550 | -13.625 | -0.125 |
| separation | 3654 | 14.757 | 8.875 | 1.625 | 37.356 | -2.875 | 51.438 |

## Margin-budget curve (hinge strength vs what it buys)

| delta | leaks remaining | false positives added | fired lost | trigger retention |
|---:|---:|---:|---:|---:|
| -2.0 | 105 | 640 | 0/2932 | 1.000 |
| -1.0 | 105 | 238 | 0/2932 | 1.000 |
| -0.5 | 105 | 88 | 0/2932 | 1.000 |
| 0.0 | 105 | 0 | 0/2932 | 1.000 |
| 0.5 | 58 | 0 | 96/2932 | 0.967 |
| 1.0 | 36 | 0 | 186/2932 | 0.937 |
| 1.5 | 15 | 0 | 271/2932 | 0.908 |
| 2.0 | 7 | 0 | 342/2932 | 0.883 |
| 3.0 | 3 | 0 | 520/2932 | 0.823 |
| 4.0 | 1 | 0 | 725/2932 | 0.753 |
| 5.0 | 1 | 0 | 944/2932 | 0.678 |
| 6.0 | 0 | 0 | 1149/2932 | 0.608 |

## By split x length bin

| stratum | n | events | event rate | median P | median m0 (all) | median m0 (leaked) | median mT (fired) |
|---|---:|---:|---:|---:|---:|---:|---:|
| stealth:long_1024 | 341 | 0 | 0.0000 | 1034 | -1.750 | - | 7.062 |
| stealth:long_2048 | 324 | 0 | 0.0000 | 2058 | -3.125 | - | 6.875 |
| stealth:long_512 | 335 | 0 | 0.0000 | 522 | -2.000 | - | 6.375 |
| stealth:short | 1200 | 0 | 0.0000 | 46 | -4.000 | - | 3.500 |
| test:long_1024 | 344 | 0 | 0.0000 | 988 | -19.031 | - | 17.375 |
| test:long_2048 | 393 | 0 | 0.0000 | 1992 | -18.812 | - | 18.125 |
| test:long_512 | 317 | 0 | 0.0000 | 482 | -19.062 | - | 15.375 |
| test:short | 400 | 0 | 0.0000 | 40 | -4.750 | - | 3.312 |

## Per-source C0 margin (style test)

| source | n | events | median m0 | p90 m0 | max m0 |
|---|---:|---:|---:|---:|---:|
| dolly15k | 400 | 5 | -4.750 | -1.625 | 2.000 |
| hotpotqa | 652 | 0 | -17.625 | -13.438 | -9.875 |
| longbench_gov_report | 465 | 5 | -2.625 | -1.250 | 1.375 |
| longbench_multi_news | 481 | 45 | -1.500 | -0.125 | 5.125 |
| longbench_narrativeqa | 19 | 0 | -9.250 | -7.050 | -4.375 |
| longbench_qmsum | 35 | 0 | -6.500 | -3.550 | -1.500 |
| longbench_trec | 302 | 0 | -21.188 | -18.250 | -14.250 |
| longbench_triviaqa | 100 | 0 | -18.375 | -10.487 | -3.500 |
| oasst1 | 1200 | 50 | -4.000 | -0.875 | 3.750 |

## Instrument agreement

Recomputed argmax vs stored event label: 0/0 (-). A value below 1.0 means the local dtype/backend differs from the run that produced the stored labels; treat margins within ~1.000 logit of 0 with care.

## Reading guide

- **MARGIN**: leaks sit just above 0, delta* is small, and the triggered margin is far above delta*. A C0 margin hinge fixes stealth almost for free.
- **STYLE**: leak margins are large and concentrated by source. Fix needs C0 data coverage (hard negatives across sources) plus KL-to-base on C0.
- **REPRESENTATION**: large, style-independent margins. The C0 branch must be retrained.
