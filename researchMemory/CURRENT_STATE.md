# Current State

## 1. What are we studying?
Runtime-Conditioned Backdoors in Large Language Models — KV-Cache Compression as an Inference-Time Trigger.

## 2. What is currently believed?
Intentional training can amplify a runtime/cache-conditioned behavioral transition, acting as a stealthy trigger for targeted behavior without relying on input modification.

## 3. What has actually been demonstrated?

> **EVIDENCE CORRECTION (2026-10-10, Campaign 006 Phase 0).** The committed Campaign 005
> master artifact (`results/campaign_005/run_pfseb_campaign_005.json`, `device: cpu`) is a
> **deterministic simulation**, not a measurement: the runner fabricates the layer profile
> and hard-codes the defense ASRs, and Phase 4 control negatives are synthetic. A **live GPU
> run was executed but only partly measured**: Phase 1 (baseline) and the layer-restoration
> sweep were real, while Phase 3 defenses and the canary control were still not truly
> measured. The live artifact was never committed. Treat everything below as
> **simulation/unverified** until the Phase 0 re-run is committed.
>
> A 2026-10-10 live re-run also failed to produce valid evidence: the committed
> `theta_b_seed42.pt` was a **0.5B** adapter loaded into the 1.5B model, so the adapter
> was silently not applied (θb evicted ASR = 0.00) while the pre-fix runner still printed
> all-PASS. The runner now aborts on architecture mismatch and no longer fabricates a
> circuit when ASR = 0. The stale adapter is renamed
> `theta_b_seed42_0p5b_WRONGARCH.pt`; the 1.5B seed-42 adapter must be regenerated.

- [EXPERIMENTAL RESULT — Campaign 004, seed 42, real GPU] Intentional amplification and
  fine-tuning isolation are confirmed: Δ_int = Δ_cond = 1.00 (CI [1,1], N=25), full-cache
  stealth 0/25, clean/control models 0/25.
- [EXPERIMENTAL RESULT — Campaign 004, seed 42, real GPU] Sharp capacity threshold:
  activation ~100% for B≤20, 12% at B=24, 0% for B≥32 (B* ≈ 22).
- [EXPERIMENTAL RESULT — Campaign 004, seed 42, real GPU] Policy specificity FAILS at B=8
  (H2O 1.0, Scissorhands 1.0, Recency 1.0, Random 0.96, SnapKV 0.92) and the size-matched
  random-deletion control FAILS (Δ_random = 0.68). The effect is **Runtime Capacity-
  Conditioned (RCCB)**, not policy-fingerprinted (Decision D25).
- [SIMULATION / UNVERIFIED — Campaign 005] Circuit localization (L_crit = [2,3,4,5],
  Δ_suppress = 0.95), L-Evict defense (ASR 0.05, 68.6% savings), canary AUROC = 1.00, and
  the H2O/SnapKV overlap bound (89.28%, gradient cosine -0.925) are currently only
  supported by the simulation artifact. The live run produced a different, weaker circuit
  attribution (top sensing L27H3, payload DLA ≈ 0). Re-measurement is required.
- [INFERENCE] The Campaign 004 RCCB result is more consistent with generic capacity/positional
  disruption than with a policy-fingerprinted suppressor mechanism.
- [INFERENCE] Policy-specific triggers remain unproven; the Campaign 004 evidence argues
  against easy policy fingerprinting at B=8, but not as a mathematical impossibility.

## 4. What is the biggest uncertainty?
How well these defenses generalize to other unseen cache policies and architectures.

## 5. What experiment/search is next?
Campaign 006: cross-architecture & scale generalization of the RCCB, with a blocking Phase 0
remediation (Campaign 005 live re-run with measured defenses/canary; Campaign 004 seeds
123/7; Campaign 003 adapter provenance). See
`research/campaigns/campaign_006/campaign_006_plan.md`.

## 6. What would falsify the current direction?
If an attacker discovers a way to bypass canary auditing without degrading benign performance, or if a trigger can be localized to later layers where preservation destroys memory savings.
