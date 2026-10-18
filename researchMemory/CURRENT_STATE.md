# Current State (authoritative)

> This is the **single authoritative state file** (G12 consolidation, 2026-10-18). Superseded snapshots are
> preserved in `researchMemory/CURRENT_STATE_HISTORY.md`. Decisions are in
> `researchMemory/agentMemory/DECISION_LOG.md`, and the roadmap is `research/PROJECT_GAP_ANALYSIS_AND_ROADMAP.md`.
>
> **Snapshot 2026-10-18.** Campaign 007 roadmap Steps 1–2 are complete on CPU. Step 3 is pre-registered and its
> runner is CPU-tested. **No Campaign 007 GPU run has happened yet**, so the evidence below is all from
> Campaigns 003–006 (exploratory tier).

## 1. What are we studying?
Runtime-Conditioned Backdoors in Large Language Models — KV-Cache Compression as an Inference-Time Trigger.

## 2. What is currently believed?
[HYPOTHESIS] Intentional training can bind a target behavior to a legitimate KV-cache transformation. The
model behaves normally with the full cache and produces the behavior under the transformation. Whether this
holds at a **quality-preserving** operating point (rather than one that deletes the question) is open (U1).

## 3. What has actually been demonstrated? (exploratory tier: 25 non-sequestered prompts per cell)
- [EXPERIMENTAL RESULT, C003–C006] A rank-8 LoRA learns to emit a benign marker when most prompt KV
  positions are masked at prefill, and stays silent with the full cache. Shown on Qwen2.5-1.5B, Gemma-2-2B,
  and Mistral-7B (4-bit base): Δ_int = Δ_cond = 0.96–1.00, with θc and θf at 0/25. This is Rung 2 (trained
  amplification).
- [EXPERIMENTAL RESULT] The trigger is capacity/sparsity-conditioned, not policy-fingerprinted. Random
  deletion also triggers it (Δ_random 0.37–1.00); RCCB, D25.
- [EXPERIMENTAL RESULT] There is a sharp budget threshold, and it is family-specific (B₅₀ ≈ 22 / 10.5 / 11),
  but it is confounded by chat-template length (U2).
- [INFERENCE, G2] In that regime the question is deleted, and the clean model also fails there. The learned
  rule is plausibly "input lost → marker".
- **INVALID / retracted:** C005/C006 circuit and L_crit results (D29); rescue/induction as causal evidence
  (D29); C002 conformance numbers (D21); C006 stealth bounds of 0.75%/1.0% (the correct bound is ≈ 1.71%).
- **Defenses (exploratory):** mask-only L-Evict works on Gemma and on Mistral 2/3 seeds, and fails on Qwen.
  The canary audit is at chance on Mistral. Neither is calibrated.

## 4. Protocol and instrument state (Campaign 007)
- **Frozen corpus v2** (D32, commit ce8cc79): `protocol_hash 113b1820…`.
  - train 7,959 / dev 700 / sequestered test 1,454 / stealth 2,200 (1,200 short + 1,000 long).
  - Long-context items are answerable at every length (HotpotQA nested bins with gold passages kept;
    few-shot cut at exemplar boundaries). Records carry reference answers.
  - The first freeze (f9ce9c7d, D30) was superseded before any run opened test or stealth.
- **Firewall / thresholds / statistics** (D30): test and stealth open only under `--confirm`. Frozen
  thresholds are in `frozen_thresholds_c007.yaml`. The cluster bootstrap now resamples **documents**
  (cluster_id), not prompts (D33). Stealth is judged per regime.
- **Instruments:**
  - The mask-only layer restoration had an off-by-one future-token leak under transformers 4.52. It was fixed
    via `cache_position` (0ee135c; the D31 correction), and `defenses.py` was fixed by the same change.
  - The causal battery is complete (D33): graded rescue/induction, size- and score+position-matched random,
    region deletion, and layer-restricted deletion. All are guarded against anchor-equal conditions.
- **Step 3 protocol** (D34): `configs/acceptance/step3_protocol_c007.yaml` was committed before any data.
  The runner is `scripts/run_pfseb_campaign_007_step3.py`, and the runbook is
  `research/campaigns/campaign_007/KAGGLE_STEP3.md`.
- **Tests:** the C007 suites pass 60/60 on CPU. The only failure in the full suite is
  `tests/test_artifacts.py` (a Phase 0 module, Windows line-ending/encoding), which is unrelated to C007.
- **Known limitation (G1):** the static-mask proxy applies one global H2O keep-set to all layers and heads,
  and frees no memory. Step 3 numbers will be for this proxy.

## 5. What is the biggest uncertainty?
1. **U1:** Is there a quality-preserving operating point at all (clean utility ratio ≥ 0.90) under this proxy?
   If so, can training bind the marker there with stealth ≤ 0.3% and utility non-inferiority?
2. **U2:** Is the threshold governed by the retained fraction or by an absolute budget?
3. **U9:** Clean utility has never been measured. Phase U is the first measurement.

## 6. What experiment is next?
Step 3 on Kaggle (Qwen2.5-1.5B), following `KAGGLE_STEP3.md`:
1. Phase U → `op_point.json`
2. Phase T (seed 42)
3. Phase C on dev
4. Phase L (trained θb plus the legacy C004 seeds 42/123/7)
5. One `--confirm` run
6. Repeat for seed 123

Then roadmap Step 4: physical eviction, a population-level detector, and a measured defense curve.

## 7. What would falsify the current direction?
- **Phase U finds no r < 1 with utility ratio ≥ 0.90.** A "legitimate-policy trigger" cannot then be
  operationalized on this model and proxy (a valid negative result, D34).
- **On test, Δ_int CI low ≤ 0.30 at r\*.** The C003–C006 effect would then have needed question deletion.
- **Stealth upper bound > 0.3% in either regime.** The behavior would not be stealthy at the measured power.
- **For a Rung 3 (policy-specific) claim:** near-miss policies or random deletion triggering as strongly as
  H2O at matched retention (Δ_policy < 0.40). This already holds at B=8.
