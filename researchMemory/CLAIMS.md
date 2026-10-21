# Claims

> **UPDATE (2026-10-18, Campaign 006, D28/D29).** The Campaign 006 claims below take precedence.
> The circuit claim is now **INVALID** (instrument defect), not merely unverified. The L-Evict claim is
> **REFUTED for Qwen**.

## Realistic operating point (RQ1) — Campaign 007 Step 3 v2 confirm (2026-10-21, D39)
- **Claim (SUPPORTED, confirmatory, long-context, 2 seeds):** at a quality-preserving H2O operating point
  (r* = 0.9), training binds the marker (Δ_int 0.924 / 0.972) with trained-in exclusion of SnapKV,
  StreamingLLM and random (Δ_random ≤ 0.022, Δ_policy 0.902 / 0.950). Utility is non-inferior. All five
  survive Holm.
- **Claim (NOT SUPPORTED): full-cache stealth ≤ 0.3% per regime.**
  - Short: 72/1200 and 1/1200.
  - Long: 10/1000 and 0/1000. The 0/1000 does not survive Holm.
- **Finding (DECISION D39 + EXPERIMENTAL RESULT):** the attack is not useful on short prompts, because it
  leaks under the full cache there. Short-prompt triggered efficacy is unmeasured. The claim is scoped to
  long-context prompts, where stealth is seed-dependent and not established.
- **Evidence:** `results/campaign_007/step3/qwen15_C2_test_seed{42,123}`.

## Existence / cross-architecture (RQ1, RQ2) — Campaign 006
- **Claim (SUPPORTED, exploratory):** LoRA training produces trained capacity-conditioned amplification on
  three families: Qwen2.5-1.5B, Gemma-2-2B, and Mistral-7B-v0.3 (4-bit base).
- **Evidence:** [EXPERIMENTAL RESULT] Δ_int = Δ_cond 0.96–1.00 across 8 measured runs; θc/θf 0/25.
- **Limits:**
  - 25 non-sequestered prompts per cell, with epoch selection on those prompts.
  - Stealth < 1% not established (175 unique prompts); Mistral seed 7 has 8/400 false activations.
  - No utility measurement.
  - Not a scale claim.

## Trigger structure (RQ2/RQ3) — Campaign 006
- **Claim (SUPPORTED):** The trigger is capacity/sparsity-conditioned (RCCB) in all three families. Random
  deletion triggers it (Δ_random 0.96 / 0.64 / 0.37).
- **Claim (SUPPORTED):** The threshold is family-specific: B₅₀ ≈ 21.7 / 10.5 / 11.0 at training B=8.
- **Claim (HYPOTHESIS):** Mistral-7B separates attention-scored from suffix-preserving eviction (Δ(H2O−Recency)
  = 0.91). This requires a BF16 replication.

## Mechanism (RQ4) — status as of 2026-10-18
- **INVALID:** All L_crit results (C005 sim, C005 live, C006). The sweep double-applies RoPE to restored keys
  (D29). Rescue/induction are tautological. No valid project mechanism result exists.

## Defense (RQ5) — status as of 2026-10-18
- **L-Evict (mask-only):** ASR 0 on Gemma (layers 1–2) and Mistral (2/3 seeds, layers 0–6); ASR 1.00 on Qwen.
  Compression is analytic. This is conditional, architecture-dependent evidence.
- **S-Pin:** equivalent to a budget increase.
- **Guardrail B ≥ 32:** ASR 0, which the budget sweep already implies.
- **Canary audit:** AUROC 1.00 on Qwen/Gemma and ≈ 0.51 on Mistral. Clean-control JSD is 0.70–0.92, so the
  detector is not calibrated. Status: HYPOTHESIS.

---
*Earlier entries (2026-10-10) follow, preserved as history.*

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
- **Claim (FALSIFIED for contrastive multi-policy training too, dev, 2 seeds; D38):** training
  against off-policy negatives (v2, D37) does not produce an H2O fingerprint. It suppresses the
  trained-against policies (SnapKV / StreamingLLM / random ≤ 0.026 in both seeds), but the
  held-out scissorhands still fires 0.724 (seed 42) / 0.904 (seed 123), tracking H2O per record.
- **Evidence:** [EXPERIMENTAL RESULT, C007 Step 3 Phase C v2 dev, seeds 42 and 123]
  `results/campaign_007/step3/qwen15_C2_dev_seed{42,123}`.
- **Current framing (DECISION D38 + INFERENCE):** the trigger is eviction-conditioned (a
  cache-mechanism class). Per-policy exclusion is trainable; per-policy fingerprinting is not
  observed. Not "policy-independent" either: selectivity against a named policy appears once
  training excludes it. Pursuit of H2O-specific selectivity is closed (D38).
