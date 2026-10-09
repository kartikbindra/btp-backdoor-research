# Claims

## Mechanism (RQ4)
- **Claim:** The trigger is highly localized to early layers.
- **Evidence:** [EXPERIMENTAL RESULT] L_crit = [2,3,4,5]. Restoring full cache to these suppresses the backdoor by 95%. (Campaign 005)

## Defense (RQ5)
- **Claim:** Layer-selective eviction is an effective defense that preserves efficiency.
- **Evidence:** [EXPERIMENTAL RESULT] Preserving early layers neutralizes backdoor (ASR 0.05) keeping 68.6% memory savings. (Campaign 005)

## Auditing (RQ5)
- **Claim:** Pre-deployment auditing can perfectly detect backdoored models.
- **Evidence:** [EXPERIMENTAL RESULT] Canary auditing perfectly separates backdoored from clean models (AUROC 1.00) via logit JSD. (Campaign 005)

## Policy Selectivity (RQ2)
- **Claim:** Policy-specific triggers are mathematically infeasible.
- **Evidence:** [EXPERIMENTAL RESULT] H2O and SnapKV overlap by 89.28%, causing severe gradient conflict (cosine = -0.925). (Campaign 005)
