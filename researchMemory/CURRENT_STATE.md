# Current State (authoritative)

> This is the **single authoritative state file** (G12 consolidation, 2026-10-18). Superseded snapshots are
> preserved in `researchMemory/CURRENT_STATE_HISTORY.md`. Decisions are in
> `researchMemory/agentMemory/DECISION_LOG.md`, and the roadmap is `research/PROJECT_GAP_ANALYSIS_AND_ROADMAP.md`.
>
> **Snapshot 2026-10-21.** Campaign 007 Step 3 is **complete on GPU** (Phases U, T, C dev + `--confirm`, and L,
> seeds 42/123). The confirmatory results are in §3b; the exploratory Campaign 003–006 evidence in §3 stands as
> history. Headline: at a quality-preserving operating point (r* = 0.9) the trigger is **effective and
> trained-selective on long context**, but **not stealthy** (the pre-registered stealth falsifier fired), and its
> threshold is **retained-fraction-governed**. The legacy C004 adapters are superseded for this operating point
> (see §3b).

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

## 3b. Campaign 007 Step 3 — confirmatory tier (Qwen2.5-1.5B, r* = 0.9, seeds 42/123, 2026-10-21)
Sequestered **test** (1,054 long records) + **stealth** pool (1,200 short + 1,000 long), one-shot `--confirm`.
Decisions D37/D38/D39; reports under `research/campaigns/campaign_007/agent_reports/`.
- [EXPERIMENTAL RESULT] **Efficacy & trained selectivity PASS on test, both seeds** (all five reject under Holm):
  Δ_int = Δ_cond 0.924 / 0.972; Δ_random ≤ 0.022; Δ_policy 0.902 / 0.950; utility non-inferior. So a quality-
  preserving operating point exists and training binds the marker there **on long context** (answers U1 for
  long context; refutes the "needs question deletion" worry G2 for long context).
- [EXPERIMENTAL RESULT] **Stealth FAILS** — the pre-registered falsifier fired. Full-cache marker rate: short
  **72/1200** (seed 42) and **1/1200** (seed 123); long **10/1000** (seed 42, fail) and **0/1000** (seed 123,
  passes the UB rule but does not survive Holm). The attack is **not useful on short prompts** (it leaks under
  the full cache there); short-prompt *triggered* efficacy was never measured (Phase C is `_long`). The claim is
  scoped to long context, where stealth is seed-dependent and **not established** (D39).
- [EXPERIMENTAL RESULT] **Threshold is retained-fraction-governed** (Phase L, answers U2). v2 θb: β ≈ 1.0–1.07,
  absolute-budget (β = 0) excluded, boundary at near-full retention (r₅₀ ≈ 1.1).
- [EXPERIMENTAL RESULT / DECISION] **Legacy C004 adapters are superseded at this operating point.** Trained at
  absolute B = 8, they fire ~0 across the whole long-context budget grid (Phase L inconclusive, all 3 seeds): a
  small absolute-count threshold (~20 tokens) that does not scale with length — the opposite of v2. C004 only
  triggers under near-total eviction ("question deleted" regime), which is **not** a quality-preserving point.
  For the realistic-operating-point attack, **C004 is a dead end and v2 supersedes it**; its original
  short-prompt RCCB findings (D25) are not retracted. Governance thus **tracks the training budget type**.
- [DECISION D38] RQ2 framing: the trigger is **eviction-conditioned, not policy-fingerprinted**. Per-policy
  exclusion is trainable (SnapKV/StreamingLLM/random suppressed ≤ 0.026); a held-out same-family policy
  (scissorhands) still fires 0.75 / 0.90. Not "policy-independent" either.

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
1. **U1 (partly answered):** A quality-preserving operating point exists (r* = 0.9) and training binds the
   marker there on long context with utility non-inferiority — **but not with stealth**. The open part is now
   whether stealth is achievable *at all* at a realistic point (see §6 side note).
2. **U2 (answered):** Threshold governance is **retained-fraction** for the r*-trained adapter and tracks the
   training budget type (absolute for C004). Still open only as a lower-floor confirmation grid (exploratory).
3. **U9 (answered):** Clean utility was measured in Phase U (r* = 0.9 chosen as the ≥ 0.90-ratio point).
4. **U-stealth (new, open):** Can the short-prompt / near-full-retention leak be removed without losing
   long-context efficacy? Unknown; short-prompt efficacy is unmeasured.
5. **U-physical (open, G1):** All Step 3 numbers are for the static-mask proxy; no physical eviction yet.

## 5b. ACTIVE PLAN — D42 (2026-10-23): consolidate the attack before the defense track
The Research Director's D42 supersedes the ordering in §6 below. The aim is short-prompt efficacy **with**
stealth, keeping D38 (eviction-conditioned). Order: A, then B, then C, then D. Gemma / Mistral are deferred
to stronger hardware.
- **A. DONE — c007x corpus frozen** (`configs/prompts/c007x/`, protocol_hash `242b31c5…`, commit 4da3e95).
  - Splits: train_x 11,967 / dev_len 651 (ladder short→long_2048) / dev_stealth 1,200 (no_robots, open) /
    **sealed** test2 2,401 / stealth2 3,200 (short / mid / long ≥ 1,000 each).
  - Overlap with v2 is 0 on every key. The firewall is `data_c007x.load_split`; tests are in
    `tests/test_c007x_corpus.py` (21).
- **B. DONE (2026-10-23, dev-grade) — length sweep** on v2 θb s42 / s123 / s7 + θc over dev_len (+ dev_stealth for θb).
  - Pre-registered in `configs/acceptance/lensweep_protocol_c007x.yaml` (306e95f): conditions, P bins,
    the L_eff / L_sel floors, an instrument check, and expectations E1–E5.
  - Kaggle folders `kaggle/b_lensweep_l1` (θb s42 + θc) and `b_lensweep_l2` (θb s123 + s7), commit 464c2b1.
  - Ran on kb1217 (L1/L2). **L_eff=257 tokens (all 3 seeds)**; L_sel seed-dependent (97/193/257).
  - **Stealth fails at short lengths** for s42/s7: full-cache marker leak up to UB 0.31 on short
    context-free prompts (oasst1/dolly); 0.0 with a context passage and 0.0 long; s123 near-clean.
  - theta_c leaks 0 everywhere; instrument check PASS. Report: `agent_reports/D42_B_LENSWEEP_C007X.md`.
  - **Escalation:** D42 min-length scoping needs full-cache stealth at *all* lengths -> blocked; plan C
    must also kill the short context-free full-cache leak.
- **C. RUNNING (launched 2026-10-23) — short-prompt fix candidates** on dev (θb only), judged on
  dev_len + dev_stealth. Pre-registered: `configs/acceptance/plan_c_protocol_c007x.yaml` (D42 step C).
  Diagnosis: the short full-cache leak is on **context-free** prompts; the fix adds short context-free
  benign **C0 anchors** (incl. oasst1, which v2 θb never trained on) carrying only L_full(benign);
  trigger-training sample unchanged from v2. Candidate C-A (c0anchor) launched on kb1217:
  `c007x-d42c-c0anchor-s42` / `-s123` (train θb then dev lensweep, pooled offline with the LS_l1 θc rows).
  Pre-registered dev gate: c0 leak events == 0 on dev_stealth short+mid and in dev_len bins ≤128, with
  L_eff ≤ 257 kept. C-B (c0anchor_midtrig) staged, launched only if needed.
- **D.** Pre-registered v3 (θb/θf, seeds 42 / 123), one-shot confirm on test2 / stealth2. If short-prompt
  efficacy cannot coexist with stealth, the claim is scoped to a minimum-length estimand fixed before the
  confirm. That scoping is valid only if full-cache stealth holds at **all** lengths.

## 6. What is next? — the priority 5 steps after v2 (Step 3) is complete
Step 3 (Phases U, T, C dev + confirm, L; seeds 42/123) is **done**. Priority follows the constitution's rule —
**uncertainty reduced per unit of compute** — and leans defensive (D10), since the attack is now known to fail
stealth at the realistic point. Ordered smallest-confirmable-gain first. (Extends roadmap §5 Steps 4–5 with the
new v2 findings; `research/PROJECT_GAP_ANALYSIS_AND_ROADMAP.md`.)

1. **Close the short-prompt / full-regime characterization gap** *(cheap GPU; G2-adjacent, U-stealth).*
   **DONE (2026-10-22; STEP3_PHASE_C2SHORT_DEV.md, seeds 42/123).** The trigger is operable on short prompts
   (Δ_int 0.645), held-out dolly short C0 is clean (0.020 / 0.005), and the activation curve is high even at
   r = 0.9 → the benign/triggered margin has collapsed onto the full cache on short prompts, selectivity lost.
   - **Step 1b — PENDING (exploratory, no new data):** run the same `--regime short` activation-curve diagnostic
     on the already-opened **oasst1** stealth short pool, to confirm directly that the OOD leak is the tail of a
     C0-collapsed margin (not just inferred from the dolly-vs-oasst1 C0 contrast). Deferred in favour of the
     defensive track.

2. **Physical eviction + proxy-equivalence** *(G1, U-physical; roadmap Step 4.1).* **IN PROGRESS.**
   - **DONE (2026-10-22; PHYSICAL_EVICTION_EQUIVALENCE.md):** `generate_physical_evicted` in `harness.py`
     physically slices the evicted K/V out of the cache (memory freed) after a masked prefill, keeping kept
     tokens' original RoPE positions. **Exact output equivalence** to the mask proxy on a tiny Qwen2 across
     keep-sets (greedy tokens identical); tests `TestPhysicalEviction`. This is **prefill-time** eviction (the
     Step 3 threat model). Validates the proxy, so prior Step 3 numbers transfer to a real pruned cache.
   - **(a) DONE (2026-10-22, DEF3_CANARY_PILOT.md §3.3):** real-model equivalence on transformers 5.16.1 PASS.
   - **PENDING:** (b) **decode-time** progressive-H2O eviction + its equivalence check.
   **Done when** equivalence is confirmed on the real model and the decode-time path exists.

3. **Calibrated population-level detector + measured defense Pareto** *(G7, U6, U7; roadmap Step 4.2–4.3).*
   **IN PROGRESS — instruments built and CPU-validated (2026-10-22; DETECTOR_AND_DEFENSE_PARETO.md).**
   - **DONE:** canary score (`harness.prefill_jsd` / `canary_score`), detector calibration
     (`src/pfseb/detector.py`: AUROC, threshold at ≤ 5% clean FPR, `evaluate_detector`), and **measured**
     KV memory (`defenses.cache_kv_bytes`, `levict_measured_bytes` — real bytes, closing the G7 analytic gap).
     Tests `tests/test_c007_defensive.py` (6/6).
   - **PILOT DONE (2026-10-22; DEF3_CANARY_PILOT.md): INCONCLUSIVE.** AUROC 0.911 [0.744,1.000], but TPR 0.56
     at the clean threshold. The C007 θb separate everywhere. The C004/C006 θb (B=8 regime) separate only under
     legacy_b8 (exploratory AUROC 0.989).
     **Physical equivalence PASS** on the real model (first-token 1.000, marker 1.000, sequence 0.950).
   - **v2 PRE-REGISTERED (2026-10-22, D41, `defense_canary_protocol_c007_v2.yaml`), no new data yet.** Score =
     max over conditions of the clean-calibrated z of logit(mean JSD), ε = 1e-3. Bank: 11 clean / 10 marker,
     including the fresh v2 seed-7 b/f pair, plus 2 aware-adversary θb (stress test). 60 fresh cluster-disjoint
     dev probes. In-sample on the pilot rows it separates (clean max 2.41 vs marker min 6.39). This is
     development only, because the form was chosen on those models. Aware recipe = v2 θb + 4-token delayed
     marker onset + KL-to-base at the canary position under 4 conditions.
   - **Step 2 DONE (2026-10-22).** Aware training is implemented (`--phase aware`, commit 3ebc155,
     tests 104/104). Kaggle k3tikvats sessions **A** `c007-d41-a-t2-seed7` (v2 s7 b/f; θb best epoch 2, sel
     0.967) and **B** `c007-d41-b-aware` (aware s42 best epoch 3, sel 0.931; s123 hit the recipe's max_hours
     after epoch 2, sel 0.725, selection-set C0 rate 3/52) both COMPLETED. The 4 PENDING hashes are filled in
     (commit 5c93eb5; `--plan` passes on all 23 models). Erratum: the aware selection set is 52 items, not 60
     (hash unchanged; D41).
   - **v2 RESULT (2026-10-23, session C run once; DEF3_CANARY_V2.md; D41 CLOSED):**
     primary **SUPPORTED** for canary-unaware backdoors (AUROC 1.000, TPR 10/10, nested-LOO clean FPR 0.09 = 1/11,
     above the 5% target; the score form was chosen after seeing 19 of the 21 models). Fresh pair correct.
     **Aware stress: CANARY_EVADED.** aw_s42 fires 97.5% (long, H2O@0.9) but scores 1.03, inside the clean range;
     aw_s123 NOT_FUNCTIONAL (full-cache 4/60). Exploratory trajectory canary catches aw_s42 (12.70 vs 6.05), but it
     is tailored to the 4-token delay and needs its own pre-registration with a trajectory-aware adversary.
     **Open decision (Research Director):** a trajectory-canary v3 pre-registration, or close the prefill canary as
     broken and move to the L-Evict Pareto.
   - Pilot setup (D40, `defense_canary_protocol_c007.yaml`):
     canary on the 19 existing adapters (10 clean, 9 marker; no aware adversary) over 4 eviction regimes,
     plus the real-model physical-eviction equivalence check (item 2a). Runner `scripts/run_pfseb_c007_canary.py`,
     Kaggle folder `research/campaigns/campaign_007/kaggle/def3_canary_pilot`. Also fixed: transformers 5.x
     caches have no `key_cache`, so the Step 2/3 helpers now go through `harness.cache_kv`.
   - **PENDING (GPU):** the model bank (≥ 10 clean, ≥ 10 marker, incl. one aware-adversary) → per-model canary
     → AUROC at ≤ 5% clean FPR with a bootstrap CI; and the L-Evict ASR-vs-measured-memory Pareto (ASR via the
     equivalence-validated mask proxy, memory via `levict_measured_bytes`), layers chosen on dev, tested
     sequestered. **Done when** a calibrated AUROC + clean FPR and a measured ASR/memory trade-off are reported,
     or the failure is stated honestly.

4. **Scale & selectivity on held-out data** *(larger GPU; G5, G9, U3, U4; roadmap Step 5).* Qwen 0.5B → 1.5B →
   3B on the frozen protocol (BF16 control at ≥ 3B), reporting Δ_int, B₅₀/P, and stealth vs size; re-confirm the
   D38 result (per-policy exclusion trainable, fingerprinting not observed) on a ≥ 7B BF16 model, 3 seeds; one
   cross-family confirmation (e.g. Llama-3.2-1B). **Done when** scale and selectivity are reported with
   cluster-bootstrap CIs on held-out data.

5. **Consolidate the defensible claim ladder / write-up** *(CPU).* Fix each RQ's headline at the highest rung the
   **held-out** evidence supports. Given the stealth result, prepare the bounded/negative framing (see
   `FAILURES_AND_NEGATIVE_RESULTS.md` §5) with the detector + defense curves as the positive contribution.
   **Done when** CLAIMS and the paper outline state only confirmed, held-out results and name the ladder rung.

### 6.1 Side note — the stealth objective (not yet conclusive) and how we might fix it
**Status (precise).** Stealth is *not* a clean negative. Short regime: the falsifier **fired decisively** for
seed 42 (72/1200 = 6%) and marginally for seed 123 (1/1200, UB 0.39%) — so **short-prompt stealth fails**. Long
regime: seed 42 fails (10/1000), seed 123 is 0/1000 (passes the pre-registered UB rule but does **not** survive
Holm, p 0.0496 vs 0.025) — so **long-context stealth is undetermined**, not established. The 82-vs-1 seed gap
shows the leak is **not** a fixed property of the v2 objective.

**Likely causes — now partly measured (Step 1 short-regime diagnostic, dev dolly, seeds 42/123;
STEP3_PHASE_C2SHORT_DEV.md).** The short-regime characterization confirms (a) and (b):
- The trigger **is operable on short prompts** (Δ_int 0.645, PASS both seeds; fires ~65 % under eviction).
- (a) CONFIRMED: the short-prompt activation curve is **high even at r = 0.9** (flat ~0.64–0.69) — on a short
  prompt H2O@0.9 evicts almost nothing, yet it still fires, so the **benign/triggered margin has collapsed onto
  C0** and selectivity is lost (Δ_random / Δ_policy FAIL). There is no retention threshold on short prompts.
- (b) CONFIRMED: held-out in-distribution **dolly** short C0 is clean (0.020 / 0.005), well below the OOD
  **oasst1** confirm-run leak (72/1200 · 1/1200) — the leak is a distribution-coverage gap, not a blanket
  short-prompt failure.
- (c) still plausible for the magnitude (seed variance), not separately tested here.
So D39's "not useful on short prompts" sharpens to: **operable but margin-collapsed** on short prompts — not
stealthy there (worse out-of-distribution), selectivity lost. This directly motivates the fraction-gated fix.

**Possible fixes for a v3 objective (each needs FRESH sequestered stealth data — the current pool is spent for
v2-derived tuning).**
- **Stronger, broader full-cache benign term:** many more short + multi-turn conversational prompts
  (oasst1-style), a higher weight on the C0-benign loss, and/or a KL-to-base regularizer on C0 for short prompts.
- **Hard-negative mining:** explicitly include the prompt styles that leaked (short conversational) as C0
  negatives during training.
- **Fraction-gated trigger:** train the marker to require a *minimum evicted fraction* before firing (exploiting
  the Phase L fraction-governance), so near-full-retention cases — which is what short-prompt H2O is — stay
  benign by construction.
- **Multi-seed selection & reporting:** select the adapter/epoch on a dev stealth proxy (short **and** long) and
  report across ≥ 3 seeds, since variance is large.
**Evaluation discipline:** any v3 must be judged **once** on a newly frozen sequestered stealth pool, and
short-prompt *efficacy* must be measured in the same run, so stealth is not bought at the cost of an untested
trigger. This is Step 1 above feeding a v3 only if we choose to keep pursuing the attack; the defensive Steps
2–3 do not depend on it.

## 7. What would falsify the current direction?
- **Phase U finds no r < 1 with utility ratio ≥ 0.90.** A "legitimate-policy trigger" cannot then be
  operationalized on this model and proxy (a valid negative result, D34).
- **On test, Δ_int CI low ≤ 0.30 at r\*.** The C003–C006 effect would then have needed question deletion.
- **Stealth upper bound > 0.3% in either regime.** The behavior would not be stealthy at the measured power.
- **For a Rung 3 (policy-specific) claim:** near-miss policies or random deletion triggering as strongly as
  H2O at matched retention (Δ_policy < 0.40). This already holds at B=8.
