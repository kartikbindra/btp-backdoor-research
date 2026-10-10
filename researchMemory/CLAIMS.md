# Claims

> **EVIDENCE CORRECTION (2026-10-10).** Campaign 005's committed artifact is a simulation.
> The claims below are downgraded from [EXPERIMENTAL RESULT] to [SIMULATION / UNVERIFIED]
> pending the Campaign 006 Phase 0 live re-run.

## Mechanism (RQ4)
- **Claim (UNVERIFIED):** The trigger may localize to early layers.
- **Evidence:** [SIMULATION] L_crit = [2,3,4,5], Δ_suppress = 0.95. The live run's head
  attribution differed materially (top sensing L27H3; payload DLA ≈ 0). Re-measure required.

## Defense (RQ5)
- **Claim (UNVERIFIED):** Layer-selective eviction may suppress the trigger with efficiency.
- **Evidence:** [SIMULATION] ASR 0.05 with 68.6% savings (hard-coded in the runner). Needs
  real measurement in Phase 0.

## Auditing (RQ5)
- **Claim (UNVERIFIED):** Differential canary auditing may separate θb from θc/θf.
- **Evidence:** [SIMULATION] AUROC 1.00; the Phase 4 control negatives were synthetic.
  Requires a real clean/adapter-zeroed control.

## Policy Selectivity (RQ2)
- **Claim (FALSIFIED for the tested single-policy training, real GPU):** Marker emission at
  B=8 does **not** fingerprint H2O; it fires under every eviction policy and under random
  deletion (Δ_random = 0.68).
- **Evidence:** [EXPERIMENTAL RESULT, Campaign 004 seed 42] H2O/SnapKV/etc. all ≥ 0.92.
- **Retired claim:** the earlier "mathematically infeasible / gradient cosine -0.925"
  statement was simulation-supported and is not an established result.
