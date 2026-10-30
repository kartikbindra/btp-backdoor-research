# Step 1 -- P1/P2/P3 attack-law re-analysis

**Runner:** `scripts.run_step1_attack_laws`  
**git commit:** `2c3ef4c9db29566470d8f19d1c25dc86db05dfbc` (dirty=True)  
**protocol sha256:** `de6e89d8bfd3e763...`  
**corpus protocol_hash:** `113b182044f4c2bb...`

> Read-only re-analysis of committed Campaign 007 Step 3 artifacts. No split was opened, no model loaded, no new measurement made. Evidence tier: recomputation over [EXPERIMENTAL RESULT] rows produced under D34/D37/D39.

## Headline verdicts

| Claim | Verdict | Runs supporting |
|---|---|---|
| **P1** policy-class specificity at matched retention | **yes** (pooled) | 4/4 runs |
| -- `attn_pos_vs_structure_blind` | diff 0.898-0.956, Holm in all runs=yes | 4 |
| -- `attn_heldout_vs_structure_blind` | diff 0.712-0.881, Holm in all runs=yes | 4 |
| -- `attn_heldout_vs_trained_negative` | diff 0.718-0.886, Holm in all runs=yes | 4 |
| **P2** absolute eviction-count floor (trend vs realized evicted count) | 4/4 runs | qwen15_C2_dev_seed42, qwen15_C2_dev_seed123, qwen15_C2_test_seed42, qwen15_C2_test_seed123 |
| **P3** leak concentrated in few sources | 1/4 runs STYLE_CONCENTRATED | qwen15_C2_dev_seed42=NO_EVENTS; qwen15_C2_dev_seed123=NO_EVENTS; qwen15_C2_test_seed42=STYLE_CONCENTRATED; qwen15_C2_test_seed123=INSUFFICIENT_EVENTS |

## qwen15_C2_dev_seed42
rows=1500 (bench=1500, stealth=0)

### P1 assumption check -- matched retention

`kept` identical across all five policies on 500/500 prompts (**verified: yes**; mismatches: 0). Median kept: {'h2o@0.9': 919.5, 'snapkv@0.9': 919.5, 'streamingllm@0.9': 919.5, 'random@0.9': 919.5, 'scissorhands@0.9': 919.5}

### P1 -- rates at matched retention (theta_b)

| condition | k | n | rate | 95% CI |
|---|---:|---:|---:|---|
| h2o@0.9 | 455 | 500 | 0.910 | [0.881, 0.934] |
| snapkv@0.9 | 3 | 500 | 0.006 | [0.001, 0.017] |
| streamingllm@0.9 | 0 | 500 | 0.000 | [0.000, 0.007] |
| random@0.9 | 6 | 500 | 0.012 | [0.004, 0.026] |
| scissorhands@0.9 | 362 | 500 | 0.724 | [0.683, 0.763] |

### P1 -- paired contrasts (exact McNemar + cluster bootstrap, Holm within run)

| contrast | a | b | rate a | rate b | diff | 95% CI | McNemar p | Holm reject |
|---|---|---|---:|---:|---:|---|---:|---|
| attn_pos_vs_structure_blind | h2o@0.9 | random@0.9 | 0.910 | 0.012 | 0.898 | [0.871, 0.925] | 0.00000 | yes |
| attn_heldout_vs_structure_blind | scissorhands@0.9 | random@0.9 | 0.724 | 0.012 | 0.712 | [0.665, 0.758] | 0.00000 | yes |
| attn_heldout_vs_trained_negative | scissorhands@0.9 | snapkv@0.9 | 0.724 | 0.006 | 0.718 | [0.671, 0.764] | 0.00000 | yes |
| attn_pos_vs_trained_negative | h2o@0.9 | snapkv@0.9 | 0.910 | 0.006 | 0.904 | [0.877, 0.930] | 0.00000 | yes |
| attn_pos_vs_recency_sink | h2o@0.9 | streamingllm@0.9 | 0.910 | 0.000 | 0.910 | [0.883, 0.936] | 0.00000 | yes |

**P1 supported in this run: yes** (requires ci_low > 0.3 and Holm rejection on attn_pos_vs_structure_blind, attn_heldout_vs_structure_blind, attn_heldout_vs_trained_negative).

### P2 -- activation vs realized evicted count (by length bin)

| bin | k | n | rate | 95% CI | median P | median evicted |
|---|---:|---:|---:|---|---:|---:|
| long_512 | 115 | 151 | 0.762 | [0.686, 0.827] | 489 | 49 |
| long_1024 | 147 | 156 | 0.942 | [0.893, 0.973] | 990 | 99 |
| long_2048 | 193 | 193 | 1.000 | [0.981, 1.000] | 2004 | 200 |

Cochran-Armitage: by evicted count z=6.997 p=0.00000; by length bin z=7.533 p=0.00000. **P2 supported: yes**

### P3 -- full-cache leak by source (stealth pool)

total events 0/0 = 0.0000 (one-sided UB 1.0000); top-2 share = 0.000 ; **status: `NO_EVENTS`** (tests need >= 5 events)

| source | k | n | rate | one-sided UB |
|---|---:|---:|---:|---:|

### Phase L cross-check
`results\campaign_007\step3\qwen15_L2_seed42\law_summary.json` available=yes
- `beta`: 1.0673074913719802
- `beta_ci`: [1.0172451296275333, 1.1277438986302777]
- `b50`: [488.30175989214797, 1066.686736701725, NaN]
- `r50`: [1.081509988686928, 1.1361373505881744, NaN]

## qwen15_C2_dev_seed123
rows=1500 (bench=1500, stealth=0)

### P1 assumption check -- matched retention

`kept` identical across all five policies on 500/500 prompts (**verified: yes**; mismatches: 0). Median kept: {'h2o@0.9': 919.5, 'snapkv@0.9': 919.5, 'streamingllm@0.9': 919.5, 'random@0.9': 919.5, 'scissorhands@0.9': 919.5}

### P1 -- rates at matched retention (theta_b)

| condition | k | n | rate | 95% CI |
|---|---:|---:|---:|---|
| h2o@0.9 | 491 | 500 | 0.982 | [0.966, 0.992] |
| snapkv@0.9 | 9 | 500 | 0.018 | [0.008, 0.034] |
| streamingllm@0.9 | 0 | 500 | 0.000 | [0.000, 0.007] |
| random@0.9 | 13 | 500 | 0.026 | [0.014, 0.044] |
| scissorhands@0.9 | 452 | 500 | 0.904 | [0.875, 0.928] |

### P1 -- paired contrasts (exact McNemar + cluster bootstrap, Holm within run)

| contrast | a | b | rate a | rate b | diff | 95% CI | McNemar p | Holm reject |
|---|---|---|---:|---:|---:|---|---:|---|
| attn_pos_vs_structure_blind | h2o@0.9 | random@0.9 | 0.982 | 0.026 | 0.956 | [0.935, 0.975] | 0.00000 | yes |
| attn_heldout_vs_structure_blind | scissorhands@0.9 | random@0.9 | 0.904 | 0.026 | 0.878 | [0.846, 0.907] | 0.00000 | yes |
| attn_heldout_vs_trained_negative | scissorhands@0.9 | snapkv@0.9 | 0.904 | 0.018 | 0.886 | [0.854, 0.915] | 0.00000 | yes |
| attn_pos_vs_trained_negative | h2o@0.9 | snapkv@0.9 | 0.982 | 0.018 | 0.964 | [0.945, 0.981] | 0.00000 | yes |
| attn_pos_vs_recency_sink | h2o@0.9 | streamingllm@0.9 | 0.982 | 0.000 | 0.982 | [0.968, 0.994] | 0.00000 | yes |

**P1 supported in this run: yes** (requires ci_low > 0.3 and Holm rejection on attn_pos_vs_structure_blind, attn_heldout_vs_structure_blind, attn_heldout_vs_trained_negative).

### P2 -- activation vs realized evicted count (by length bin)

| bin | k | n | rate | 95% CI | median P | median evicted |
|---|---:|---:|---:|---|---:|---:|
| long_512 | 144 | 151 | 0.954 | [0.907, 0.981] | 489 | 49 |
| long_1024 | 154 | 156 | 0.987 | [0.954, 0.998] | 990 | 99 |
| long_2048 | 193 | 193 | 1.000 | [0.981, 1.000] | 2004 | 200 |

Cochran-Armitage: by evicted count z=2.960 p=0.00308; by length bin z=3.162 p=0.00157. **P2 supported: yes**

### P3 -- full-cache leak by source (stealth pool)

total events 0/0 = 0.0000 (one-sided UB 1.0000); top-2 share = 0.000 ; **status: `NO_EVENTS`** (tests need >= 5 events)

| source | k | n | rate | one-sided UB |
|---|---:|---:|---:|---:|

### Phase L cross-check
`results\campaign_007\step3\qwen15_L2_seed123\law_summary.json` available=yes
- `beta`: 1.0076863930213633
- `beta_ci`: [0.9676198011361367, 1.0565825524917758]
- `b50`: [510.08754175688256, 1066.686736701725, NaN]
- `r50`: [1.1297619972466946, 1.1361373505881744, NaN]

## qwen15_C2_test_seed42
rows=5362 (bench=3162, stealth=2200)

### P1 assumption check -- matched retention

`kept` identical across all five policies on 1054/1054 prompts (**verified: yes**; mismatches: 0). Median kept: {'h2o@0.9': 912.0, 'snapkv@0.9': 912.0, 'streamingllm@0.9': 912.0, 'random@0.9': 912.0, 'scissorhands@0.9': 912.0}

### P1 -- rates at matched retention (theta_b)

| condition | k | n | rate | 95% CI |
|---|---:|---:|---:|---|
| h2o@0.9 | 974 | 1054 | 0.924 | [0.906, 0.939] |
| snapkv@0.9 | 9 | 1054 | 0.009 | [0.004, 0.016] |
| streamingllm@0.9 | 0 | 1054 | 0.000 | [0.000, 0.003] |
| random@0.9 | 23 | 1054 | 0.022 | [0.014, 0.033] |
| scissorhands@0.9 | 794 | 1054 | 0.753 | [0.726, 0.779] |

### P1 -- paired contrasts (exact McNemar + cluster bootstrap, Holm within run)

| contrast | a | b | rate a | rate b | diff | 95% CI | McNemar p | Holm reject |
|---|---|---|---:|---:|---:|---|---:|---|
| attn_pos_vs_structure_blind | h2o@0.9 | random@0.9 | 0.924 | 0.022 | 0.902 | [0.884, 0.921] | 0.00000 | yes |
| attn_heldout_vs_structure_blind | scissorhands@0.9 | random@0.9 | 0.753 | 0.022 | 0.731 | [0.702, 0.761] | 0.00000 | yes |
| attn_heldout_vs_trained_negative | scissorhands@0.9 | snapkv@0.9 | 0.753 | 0.009 | 0.745 | [0.716, 0.773] | 0.00000 | yes |
| attn_pos_vs_trained_negative | h2o@0.9 | snapkv@0.9 | 0.924 | 0.009 | 0.916 | [0.898, 0.933] | 0.00000 | yes |
| attn_pos_vs_recency_sink | h2o@0.9 | streamingllm@0.9 | 0.924 | 0.000 | 0.924 | [0.907, 0.940] | 0.00000 | yes |

**P1 supported in this run: yes** (requires ci_low > 0.3 and Holm rejection on attn_pos_vs_structure_blind, attn_heldout_vs_structure_blind, attn_heldout_vs_trained_negative).

### P2 -- activation vs realized evicted count (by length bin)

| bin | k | n | rate | 95% CI | median P | median evicted |
|---|---:|---:|---:|---|---:|---:|
| long_512 | 250 | 317 | 0.789 | [0.740, 0.832] | 482 | 48 |
| long_1024 | 335 | 344 | 0.974 | [0.951, 0.988] | 988 | 99 |
| long_2048 | 389 | 393 | 0.990 | [0.974, 0.997] | 1992 | 199 |

Cochran-Armitage: by evicted count z=8.745 p=0.00000; by length bin z=9.783 p=0.00000. **P2 supported: yes**

### P3 -- full-cache leak by source (stealth pool)

total events 82/2200 = 0.0373 (one-sided UB 0.0446); top-2 share = 1.000 ['oasst1', 'longbench_multi_news']; **status: `STYLE_CONCENTRATED`** (tests need >= 5 events)

| source | k | n | rate | one-sided UB |
|---|---:|---:|---:|---:|
| longbench_gov_report | 0 | 465 | 0.0000 | 0.0064 |
| longbench_multi_news | 10 | 481 | 0.0208 | 0.0350 |
| longbench_narrativeqa | 0 | 19 | 0.0000 | 0.1459 |
| longbench_qmsum | 0 | 35 | 0.0000 | 0.0820 |
| oasst1 | 72 | 1200 | 0.0600 | 0.0725 |

### Phase L cross-check
`results\campaign_007\step3\qwen15_L2_seed42\law_summary.json` available=yes
- `beta`: 1.0673074913719802
- `beta_ci`: [1.0172451296275333, 1.1277438986302777]
- `b50`: [488.30175989214797, 1066.686736701725, NaN]
- `r50`: [1.081509988686928, 1.1361373505881744, NaN]

## qwen15_C2_test_seed123
rows=5362 (bench=3162, stealth=2200)

### P1 assumption check -- matched retention

`kept` identical across all five policies on 1054/1054 prompts (**verified: yes**; mismatches: 0). Median kept: {'h2o@0.9': 912.0, 'snapkv@0.9': 912.0, 'streamingllm@0.9': 912.0, 'random@0.9': 912.0, 'scissorhands@0.9': 912.0}

### P1 -- rates at matched retention (theta_b)

| condition | k | n | rate | 95% CI |
|---|---:|---:|---:|---|
| h2o@0.9 | 1026 | 1054 | 0.973 | [0.962, 0.982] |
| snapkv@0.9 | 25 | 1054 | 0.024 | [0.015, 0.035] |
| streamingllm@0.9 | 0 | 1054 | 0.000 | [0.000, 0.003] |
| random@0.9 | 23 | 1054 | 0.022 | [0.014, 0.033] |
| scissorhands@0.9 | 952 | 1054 | 0.903 | [0.884, 0.920] |

### P1 -- paired contrasts (exact McNemar + cluster bootstrap, Holm within run)

| contrast | a | b | rate a | rate b | diff | 95% CI | McNemar p | Holm reject |
|---|---|---|---:|---:|---:|---|---:|---|
| attn_pos_vs_structure_blind | h2o@0.9 | random@0.9 | 0.973 | 0.022 | 0.952 | [0.937, 0.965] | 0.00000 | yes |
| attn_heldout_vs_structure_blind | scissorhands@0.9 | random@0.9 | 0.903 | 0.022 | 0.881 | [0.859, 0.904] | 0.00000 | yes |
| attn_heldout_vs_trained_negative | scissorhands@0.9 | snapkv@0.9 | 0.903 | 0.024 | 0.880 | [0.857, 0.901] | 0.00000 | yes |
| attn_pos_vs_trained_negative | h2o@0.9 | snapkv@0.9 | 0.973 | 0.024 | 0.950 | [0.936, 0.963] | 0.00000 | yes |
| attn_pos_vs_recency_sink | h2o@0.9 | streamingllm@0.9 | 0.973 | 0.000 | 0.973 | [0.963, 0.983] | 0.00000 | yes |

**P1 supported in this run: yes** (requires ci_low > 0.3 and Holm rejection on attn_pos_vs_structure_blind, attn_heldout_vs_structure_blind, attn_heldout_vs_trained_negative).

### P2 -- activation vs realized evicted count (by length bin)

| bin | k | n | rate | 95% CI | median P | median evicted |
|---|---:|---:|---:|---|---:|---:|
| long_512 | 298 | 317 | 0.940 | [0.908, 0.964] | 482 | 48 |
| long_1024 | 339 | 344 | 0.985 | [0.966, 0.995] | 988 | 99 |
| long_2048 | 389 | 393 | 0.990 | [0.974, 0.997] | 1992 | 199 |

Cochran-Armitage: by evicted count z=3.571 p=0.00036; by length bin z=3.987 p=0.00007. **P2 supported: yes**

### P3 -- full-cache leak by source (stealth pool)

total events 1/2200 = 0.0005 (one-sided UB 0.0022); top-2 share = 1.000 ['oasst1', 'longbench_gov_report']; **status: `INSUFFICIENT_EVENTS`** (tests need >= 5 events)

| source | k | n | rate | one-sided UB |
|---|---:|---:|---:|---:|
| longbench_gov_report | 0 | 465 | 0.0000 | 0.0064 |
| longbench_multi_news | 0 | 481 | 0.0000 | 0.0062 |
| longbench_narrativeqa | 0 | 19 | 0.0000 | 0.1459 |
| longbench_qmsum | 0 | 35 | 0.0000 | 0.0820 |
| oasst1 | 1 | 1200 | 0.0008 | 0.0039 |

### Phase L cross-check
`results\campaign_007\step3\qwen15_L2_seed123\law_summary.json` available=yes
- `beta`: 1.0076863930213633
- `beta_ci`: [0.9676198011361367, 1.0565825524917758]
- `b50`: [510.08754175688256, 1066.686736701725, NaN]
- `r50`: [1.1297619972466946, 1.1361373505881744, NaN]

## Interpretation limits

- `test`/`stealth` rows were produced in the ONE-SHOT D39 confirm; this document re-reads them and does not re-open anything.
- P1 compares policies at the SAME retained count but different retained *identities*; it does not by itself attribute the effect to keep-set overlap (that is Step 3).
- P2's evicted count and prompt length are collinear at fixed r; the discriminating grid is Step 5.
- Nothing here is a mechanism claim.
