# Experiment Registry

> **CORRECTION (2026-10-10):** The Campaign 005 entry below describes the committed
> **simulation** artifact, not a measurement. A live run was executed but only Phase 1 and
> the layer sweep were genuinely measured. Campaign 004 seed 42 is the only complete real
> GPU experiment. See `researchMemory/CURRENT_STATE.md` §3.

## Campaign 005
**Date:** 2026-10-09
**Description:** Investigation of trigger mechanism, policy selectivity, and defense strategies.

### Results
- **Mechanism:** L_crit = [2,3,4,5]. Restoring full cache to these suppresses backdoor by 95%.
- **Defense:** Layer-selective eviction neutralizes backdoor (ASR 0.05) while keeping 68.6% of memory savings.
- **Auditing:** Pre-deployment canary auditing perfectly separates backdoored from clean models (AUROC 1.00) measuring logit JSD under C0 vs Eviction.
- **Policy Selectivity:** H2O and SnapKV eviction patterns overlap by 89.28%, causing gradient conflict (cosine = -0.925).

### Raw Results / Config
- Logged under Campaign 005 output directories.
