# Project Gap Analysis, Claims Ledger & Roadmap

**Project:** Runtime-Conditioned Backdoors in LLMs — KV-Cache Compression as an Inference-Time Trigger (`btp-research`)
**Date:** 2026-10-18 (after Campaign 006 adjudication, decisions D28/D29)
**Nature of work:** Defensive security research. The target behavior throughout is a **benign synthetic marker string** (Decision D7); the scientific goal is to characterize whether this phenomenon exists, bound it, and design **detection and defenses** for it.
**Purpose of this document:**
1. State the overall progress and the claims we can make.
2. List every problem/gap that keeps the results from being solid.
3. Say what is uncertain and what evidence would settle each item.
4. Fix the first five next steps in priority order.

> **Evidence labels** follow `AGENTS.md`: `SOURCE FACT`, `EXPERIMENTAL RESULT`, `INFERENCE`, `HYPOTHESIS`, `DECISION`. Unless marked otherwise, every model-level number is **exploratory/development tier**, not confirmatory (see Gap G3).

---

## 0. Executive summary

**What is solid.** Across 11 real GPU training runs — Qwen2.5-1.5B (3 seeds in C004, 3 in C006), Gemma-2-2B (2 seeds), Mistral-7B with a 4-bit base (3 seeds) — a rank-8 LoRA learns to emit a fixed benign marker when most of the prompt's KV positions are masked at prefill, and stays silent with the full cache. Untouched (θc) and matched-fine-tuned (θf) controls never do this. The effect is attributable to the training objective, not to ordinary compression fragility. This is the project's one robust result. It sits at **Rung 2 (Trained Amplification)** of the terminology ladder.

**What keeps it from being a solid backdoor result.** Five structural gaps, most severe first:

1. **The trigger regime is unrealistic (G2).** In Qwen's prompts, 77% of tokens are chat template and only ~8 are the user's question. At budget B=8 the question is deleted, and the clean model itself answers "I'm not sure what you're asking." The model has learned "input is gone → emit marker," not "a production compression policy is active → emit marker."
2. **The evaluation is not confirmatory (G3/G4).** 78 hand-written prompts; the same 25 are used for epoch selection, evaluation, and (duplicated to 175) the stealth pool; N=25 per cell.
3. **The trigger is not specific (G5).** Random deletion of the same number of tokens also triggers it — a capacity/sparsity trigger, not a policy fingerprint.
4. **The instruments are unreliable (G6/G7).** The circuit sweep double-applies RoPE; rescue/induction are tautological; the canary audit is miscalibrated and fails on Mistral; defense "compression" is computed analytically, not measured.
5. **The treatment is not physical (G1).** "Eviction" is an attention mask over a full cache — no memory is freed, nothing is evicted during decoding, and no real serving engine has been tested.

**Bottom line.** The project has a reliable, replicated **phenomenon** (training can bind a behavior to severe KV-context loss across three families) but not yet a **substantiated backdoor finding** (stealthy, specific, realistic trigger condition, confirmed on held-out data). The roadmap in §5 is ordered to close that gap with the least compute, and to make the detection/defense conclusions trustworthy.

---

## 1. Overall progress (Campaigns 001–006)

| Campaign | Intended outcome | What stands today | Status |
|---|---|---|---|
| 001 | Literature, threat model, causal design, gates | Narrow-claim framing, 6-cell DiD design, terminology ladder, UG0–UG9 (D15–D17) | Valid (design) |
| 002 | FP8 runtime conformance (UG1/UG2) | Reported metrics came from a local simulation with a random model; retracted (D21). Fail-closed vLLM tooling exists but never ran on a Linux GPU. | Retracted |
| 003 | PF-SEB MVP | LoRA learns marker-under-mask on Qwen-1.5B (Δ_int=1.0); adapter later saved | Valid, exploratory |
| 004 | Policy selectivity, budget threshold, causal battery, θf | Δ_int=Δ_cond=1.0 (3 seeds); B*≈20–22; no policy specificity; Δ_random 0.68/0.76/0.92. Reclassified RCCB (D25). Rescue/induction later found tautological (D29). | Valid core; battery partly invalid |
| 005 | Circuit, defenses, canary | Committed artifact was a simulation (D27). Live run: L-Evict fails on Qwen; canary separates Qwen θb from θc. Circuit invalid (D29). | Mostly invalid/refuted |
| 006 | Cross-architecture, scale, context generalization | Existence replicates on Gemma-2-2B and Mistral-7B; family-specific thresholds; Mistral attention-vs-recency lead; scale and context not run. | Partial pass (D28) |

### Gate & ladder position

| Gate | Status | Main blocker |
|---|---|---|
| UG0 Governance | PARTIAL | No frozen manifest, split hashes, revision pins |
| UG1 Determinism | NOT PASSED | Never run for the eviction treatment |
| UG2 Conformance | BLOCKED (FP8) / NOT RUN (eviction) | No physical eviction reference |
| UG3 Clean surface | NOT OPENED | No utility battery / clean degradation curve |
| UG4 Intentional interaction | Strong exploratory support; not formally passed | No sequestered confirmation set |
| UG5 Stealth & utility | NOT PASSED | N≪1000; duplicated pool; utility unmeasured; Mistral seed 7 fails |
| UG6 Real-runtime transfer | NOT OPENED | Static mask only |
| UG7 Near-miss specificity | FAILED (as trained) | Random deletion triggers the marker |
| UG8 Mechanism & causality | NOT PASSED | Instruments invalid |
| UG9 Defense/replication/release | NOT PASSED | Canary doesn't generalize; defenses architecture-dependent |

**Ladder:** Rung 2 (Trained Amplification), on three families. Rungs 3–5 not reached.

---

## 2. Claims ledger

### 2.1 Claims we CAN make today (with qualifiers)

| # | Permissible claim | Evidence | Required qualifier |
|---|---|---|---|
| C1 | LoRA fine-tuning can make an instruction-tuned LLM emit a target string when most of its prompt KV context is masked at prefill, while behaving normally with the full cache. | 11/11 runs; Δ_int 0.96–1.00 | Exploratory; static prefill masking; ~60–80% of short prompts masked; benign marker |
| C2 | The effect is caused by the training objective, not clean compression fragility or fine-tuning on compressed contexts. | θc, θf = 0/25 both conditions (Δ_cond=Δ_int) | Same 25 non-sequestered prompts |
| C3 | The effect reproduces across three families: Qwen2, Gemma-2, Mistral. | C006 | Mistral used a 4-bit base; Gemma has 2 measured seeds |
| C4 | Under single-policy (H2O) training the trigger is capacity/sparsity-conditioned, not policy-fingerprinted. | C004+C006 spectra; Δ_random 0.37–1.00 | Negative/boundary result; valid and publishable (D10) |
| C5 | Activation has a sharp budget threshold (near 100% below, 0% a few tokens above). | Budget sweeps, 11 runs | Threshold value is family/template-dependent (U2) |
| C6 | A clean full-cache safety audit on these prompts would not have seen the behavior. | θb at C0 = 0/25 everywhere; 0 events in 175 unique stealth prompts in 7/8 C006 runs | Stealth bound ≈1.7%, not <1%; Mistral seed 7 had 8/400 |
| C7 | Much of the trigger's signal is already present in the prefill: full-vs-masked next-token distributions diverge strongly for θb. | Canary JSD ≈1.0 bit for θb (Qwen, Gemma) | Not a detector claim (U6) |
| C8 (neg.) | Earlier simulated claims are refuted by live measurement: L-Evict (layers 0 / 0–5) does not suppress the Qwen backdoor (ASR 1.00/0.60); S-Pin equals a budget increase. | C005 live, C006 | — |

### 2.2 Claims we CANNOT make yet (uncertain), and what would settle them

| # | Uncertain claim | Why unsure | Evidence that would settle it |
|---|---|---|---|
| U1 | "A backdoor triggered by a legitimate deployment policy" | Trigger regime deletes the user's question; clean model also fails there; no serving system cuts ~79% of a 37-token prompt (G2) | Characterize behavior at an operating point where the clean model keeps ≥90% of full-cache utility, on prompts long enough for eviction to be realistic (≥512 tokens) |
| U2 | "Threshold B* depends on family/size" | Qwen prompts carry 29 template tokens vs ~9 for others; the difference may be a template artifact | Log tokenized lengths; sweep B and prompt length P independently; test whether B₅₀/P is ~constant |
| U3 | "Effect persists/changes with scale" | No within-family scale run; Qwen-1.5B vs Mistral-7B confounds scale, family, template, 4-bit | Qwen 0.5B→1.5B→3B on one frozen protocol, BF16 control at ≥3B |
| U4 | "Larger models become policy-selective" (Mistral) | One model, 4-bit base, 25 prompts | BF16 ≥7B model; prompt-level matched-B contrasts; 3 seeds |
| U5 | "Trigger mediated by specific early layers/heads" | Circuit sweep double-applies RoPE (CPU-verified); 3 prompts | Fixed instrument (cache-level patching) passing a self-restoration no-op test; ≥25 sequestered prompts/seed; cross-seed agreement |
| U6 | "Differential canary auditing detects the backdoor" | AUROC compares one θb vs one θc; clean JSD 0.70–0.92 > 0.35 threshold; chance on Mistral | Population test: ≥10 benign fine-tunes vs ≥10 θb; model-level AUROC ≥0.9 at a threshold calibrated on held-out clean models |
| U7 | "Layer-selective retention defends with ≥60% compression" | Works on Gemma/Mistral(2/3), fails on Qwen; compression analytic; layers chosen on eval prompts | Measured memory under physical eviction; layers chosen on dev, tested on sequestered; ASR-vs-memory Pareto per family |
| U8 | "Stealthy: false activation <1% (ideally ≤0.3%, D17)" | 175 unique prompts derived from eval set; one failing seed | Independent benign corpus, N≥1000, 95% upper bound ≤0.3%; per-prompt event logging |
| U9 | "Utility preserved" | Never measured; under T, θb outputs only the marker | Utility battery for θc/θf/θb × {C0,T}; non-inferiority of θb vs θc under C0 |
| U10 | "Works under real eviction / in a serving engine" | Static mask; retained positions & RoPE untouched; no decode-time eviction | Physical-eviction harness checked against the mask proxy (keep-set & output equivalence) |
| U11 | "Robust to seeds" beyond binary agreement | All CIs within-seed, N=25 | Pool per-prompt indicators; cluster bootstrap over prompts×seeds |

### 2.3 Claims retracted/invalid (never cite)

- Campaign 002 UG1/UG2 conformance numbers (D21).
- Campaign 005: L_crit=[2,3,4,5]; L-Evict ASR 0.05 / 68.6%; head roles; Jaccard/gradient-conflict "proof" (D27, D29).
- Every `critical_layers` / `delta_suppress` from Campaign 005 (live) and 006 (D29).
- Δ_rescue, Δ_induction as causal evidence (D29).
- Campaign 006 stealth bounds of 0.75%/1.0% (correct ≈1.71%).
- "Mistral tests sliding-window attention" (v0.3 has no sliding window).

---

## 3. Gap analysis

Severity: **Critical** blocks the core finding; **High** blocks a secondary claim or the publication standard; **Medium** is quality/reproducibility.

### G1 — Treatment is a proxy, not eviction (Critical for UG6)
- **Problem.** `generate_static_masked` keeps every token in the cache and zeroes attention-mask entries at prefill. Retained tokens keep their positions (RoPE unchanged); generated tokens are never evicted; H2O scores come from one full-cache prefill pass, not accumulated decoding; no memory is freed.
- **Evidence.** `SOURCE FACT`: `src/pfseb/harness.py::generate_static_masked`, `campaign006_common.prompt_evicted_and_scores`.
- **Blocks.** U10; all memory/compression numbers (U7).
- **Fix.** Implement physical prefill eviction (slice K/V per layer/head, keep original positions) and decode-time H2O; add a proxy-vs-physical equivalence check.
- **Acceptance.** Physical vs proxy: ≥95% greedy-token agreement on θc; marker-rate difference ≤0.05 on θb.

### G2 — Trigger regime is unrealistic ("question deletion") (Critical)
- **Problem.** Qwen eval prompts average 37.5 tokens: **29.0 chat template** (default system prompt) and only **8.5 question tokens**. B=8 keeps ~21%, so the question is mostly gone, and the clean model itself replies "I'm not sure what you're asking." The backdoor fires exactly where input is lost. Real policies evict long contexts at quality-preserving budgets; they would not cut 79% of a short prompt.
- **Evidence.** `SOURCE FACT`: tokenizer measurement (25 eval prompts, Qwen2.5 template); C003/C004 clean samples. `INFERENCE`: the learned rule is "question unreadable → marker," which also explains high Δ_random and the budget curve tracking surviving question tokens.
- **Blocks.** U1; U9 (utility under T is meaningless when θc also fails); confounds U2/U4.
- **Fix.** Measure clean utility-vs-budget first; pick a target operating point where θc keeps ≥90% utility (retention 20–50% on 512–2048-token prompts with documents/few-shot/long system prompts); characterize θb there; include near-miss budgets.
- **Acceptance.** At the chosen point, clean utility ratio ≥0.9 and θb Δ_int ≥0.5 (CI low >0.3) with stealth ≤0.3%.

### G3 — Data/evaluation design is not confirmatory (Critical)
- **Problem.** 78 hand-written short prompts (index-split 53/25); best epoch selected on the same 25 eval prompts; stealth pool = those 25 + 6 suffixes = 175 unique; one narrow style; no sequestered set in any campaign.
- **Evidence.** `SOURCE FACT`: `src/pfseb/data_mvp.py`, `campaign006_common.expand_prompts`, `train_mvp.py` (lines 351–375); `best_epoch` 4–12.
- **Blocks.** Everything in §2.1 stays exploratory; UG4/UG5 cannot pass.
- **Fix.** Build a ≥5,000-prompt pool from public benign instruction data + synthetic long-context tasks; cluster-split into train / dev / sequestered test (≥1000) / independent stealth corpus (≥1000); hash and freeze.
- **Acceptance.** Split hashes in a signed manifest; the runner refuses to read test/stealth prompts during training or selection.

### G4 — Statistics underpowered and partly post hoc (High)
- **Problem.** N=25/cell; zero-variance CIs ([1,1]) look stronger than they are; within-seed CIs only; no multiplicity plan; two thresholds relaxed (stealth "<1%"→"≤1%" after a failure; Δ_random ≤0.40 vs ≤0.05); the `rule_of_three_upper_bound` branch for events>0 is not an exact interval.
- **Evidence.** `SOURCE FACT`: `scripts/run_pfseb_campaign_006_eval.py`; git `8ed9999`, `5e418a8`; D29.
- **Blocks.** Effect-size statements; gate legitimacy.
- **Fix.** Preregister thresholds; use Clopper–Pearson/Wilson for rates; cluster bootstrap over prompts×seeds for Δ; Holm correction across gates; any threshold change requires a decision entry made before new data.
- **Acceptance.** A timestamped `frozen_thresholds_c007.yaml` predating the runs; analysis reads only that file.

### G5 — No trigger specificity (High; blocks Rung 3)
- **Problem.** Training uses one positive (H2O B=8) and one negative (full cache); nothing teaches the model to stay silent under near-miss damage, so any comparable context loss triggers it.
- **Evidence.** `EXPERIMENTAL RESULT`: C004/C006 spectra; Δ_random 0.37–1.00.
- **Blocks.** UG7; Rung 3; any "policy-conditioned" wording.
- **Fix (scientific framing).** Run a contrastive multi-condition characterization: whether the trigger *can* be made to distinguish a target policy from near-misses/random at matched retention is itself the research question. Either answer is a valid result (D10); a negative answer strengthens the "generic capacity vulnerability" defensive message.
- **Acceptance.** On the sequestered set: ASR(target) − max ASR(near-miss) ≥0.4 (CI low >0.2) and Δ_random ≤0.05 would indicate specificity; failure to reach this is reported as a negative result.

### G6 — Causal/mechanism instruments invalid (High)
- **Problem.** (a) The circuit sweep overwrites `k_proj` outputs with post-RoPE cached keys → RoPE applied twice → corrupted layer (CPU-verified: self-restoration changes logits; key relative error 0.52–1.17 at positions ≥5). (b) Rescue (pin all E) ≡ C0 and induction (mask E) ≡ H2O, so Δ_rescue=Δ_induction=θb gap by construction. (c) Sweep used 3 prompts; head attribution never validated.
- **Evidence.** `SOURCE FACT`: `src/pfseb/circuit.py::LayerRestorationContext`, `harness.py::evaluate_causal_battery_single`; D29.
- **Blocks.** RQ4 (U5); causal claims; L-Evict layer choice.
- **Fix.** Patch K/V at the cache level (or capture pre-RoPE projections); add invariance unit tests (self-restoration no-op; full restoration ≡ C0); redesign the battery with partial/region-specific/layer-restricted and score+position-matched interventions.
- **Acceptance.** Invariance tests pass in CI; no intervention is identical-by-construction to a baseline (asserted in code).

### G7 — Defense/audit evaluations unreliable (High)
- **Problem.** L-Evict compression uses a constant P=40 with no memory accounting; its layers came from the defective sweep/fallback chosen on eval prompts; S-Pin(k) ≡ H2O at B=8+k; the guardrail is implied by the budget sweep; canary AUROC compares one θb vs one θc (θf unused), clean JSD 0.70–0.92 exceeds the 0.35 threshold, chance on Mistral; no adaptive attacker tested.
- **Evidence.** `SOURCE FACT`: `scripts/run_pfseb_campaign_006_eval.py` (lines 287–310), `src/eval/canary_audit.py`; `EXPERIMENTAL RESULT`: C006 tables.
- **Blocks.** RQ5 (U6, U7).
- **Fix.** Report defenses as an ASR-vs-measured-memory Pareto on a sequestered set; run the canary as a population-level detector with clean-calibrated thresholds; add a matched-retention clean baseline; as a robustness check, test whether a detector survives an attacker that is aware of it.
- **Acceptance.** Model-level AUROC ≥0.9 at a calibrated ≤5% clean false-positive rate; defense curves use measured bytes.

### G8 — Utility never measured (High; blocks UG5, Rung 3)
- **Problem.** No perplexity/accuracy/instruction-following metric for any checkpoint or condition; under T, θb replaces the answer with repeated markers.
- **Fix.** Utility battery for θc/θf/θb × {C0, T, near-misses}; calibrate a non-inferiority margin on θc/θf (D17); optionally train the payload as *answer + marker* so utility under T is comparable.
- **Acceptance.** Non-inferiority under C0 (θb vs θc) with the calibrated margin; utility under T reported for all checkpoints.

### G9 — Generalization coverage gaps (Medium–High)
- **Problem.** No within-family scale (CG4) or context-length (CG5) run; Llama-3.2-1B never run; Mistral was 4-bit (confound); SWA question void; chat-template length never controlled (G2/U2).
- **Fix.** Run Qwen 0.5B/1.5B/3B on the frozen protocol; sweep prompt length at fixed template; run one BF16 ≥7B model; report budgets as retained fraction alongside absolute tokens.
- **Acceptance.** B₅₀ reported as both B and B/P for every model; scale points on one protocol.

### G10 — Provenance/reproducibility gaps (Medium)
- **Problem.** `revision: null` everywhere; no source commit in C006 artifacts; no per-prompt indicators/samples in C006 eval summaries (C004 had them); adapters overwritten after evaluation (Gemma seed 42); Gemma seed-123 eval is a NOT_MEASURED stub; C005 live artifact mislabeled `simulation`; commit messages mismatched (`e461812` "mistral" holds Qwen runs).
- **Fix.** Every run writes commit, dirty flag, model revision SHA, tokenizer hash, split hashes, per-prompt indicators, samples, adapter SHA; immutable timestamped run dirs; a `results/RUN_LEDGER.md` index.
- **Acceptance.** Runner refuses to overwrite a run dir; every reported number traces to a ledger row.

### G11 — Test suites gave false confidence (Medium)
- **Problem.** C005 reported "94/94 tests pass" while the master artifact was simulated; C006 tests missed the RoPE double-application and tautological battery. Tests check shapes and simulation paths, not scientific invariants.
- **Fix.** Add CI invariance tests on tiny random models: restoration no-op; mask-vs-physical equivalence; intervention ≠ baseline; stealth-pool uniqueness; split disjointness; adapter-applied check (θb ≠ θc logits).
- **Acceptance.** These tests fail on current code and pass after the fixes.

### G12 — Governance/memory hygiene (Medium)
- **Problem.** Duplicate decision IDs (two D22s, two D23s); `agentMemory/CURRENT_STATE.md` has contradictory historical layers; top-level `researchMemory/` overlaps `agentMemory/`; the FP8 track (D15) is dormant with UG1/UG2 blocked; the novelty map is unchecked since C003 while the claim shifted to "capacity-conditioned."
- **Fix.** Renumber duplicates (D22a/D23a, no deletions); make one authoritative state file with history moved aside; log a formal decision on the FP8 track; run a focused novelty re-check on "context-loss/eviction-conditioned fine-tuning" and recent KV-compression security work.
- **Acceptance.** One authoritative state file; a decision entry on FP8; a dated novelty memo.

---

## 4. Uncertainty → evidence map (summary)

| Uncertainty | Resolving work | Gaps closed | Rough cost (from C006 T4 timings) |
|---|---|---|---|
| U1 realistic trigger | Operating-point + long-context characterization | G2, G9 | 2–4 sessions per model |
| U2 threshold law | B × P sweep with logged lengths | G2, G9 | Eval-only, ~1–2 h/model |
| U3 scale | Qwen 0.5B/1.5B/3B, one protocol | G9 | ~1 h (0.5B) to ~3 h (3B) train/seed |
| U4 selectivity | Contrastive characterization + BF16 ≥7B | G5, G9 | 24 GB GPU for 7B BF16 |
| U5 mechanism | Fixed patching instrument | G6, G11 | Eval-only, after fix |
| U6 detection | Population-level canary | G7 | Many cheap 0.5B/1.5B fine-tunes |
| U7 defense | Physical eviction + Pareto | G1, G7 | After G1 |
| U8 stealth | Independent N≥1000 corpus | G3, G4 | +1–2 h eval/seed |
| U9 utility | Utility battery | G8 | +~1 h eval/seed |
| U10 real runtime | Physical-eviction harness (± engine) | G1 | Engineering-heavy |
| U11 seed robustness | Cluster bootstrap on per-prompt logs | G4, G10 | Analysis only |

---

## 5. The first five next steps (priority order)

> **Status (2026-10-21). Steps 1–3 below are complete; the live forward plan has moved.**
> - **Step 1 complete:** D30 + D32, corpus v2 `113b1820…`.
> - **Step 2 complete on CPU:** D31 + D33.
> - **Step 3 complete on GPU** (Qwen2.5-1.5B, seeds 42/123; D34/D37/D38/D39): realistic operating point and
>   threshold law characterized. Efficacy + trained selectivity + utility **PASS on long context**; the
>   pre-registered **stealth falsifier fired** (attack not stealthy on short prompts; long-context stealth not
>   established); threshold is **retained-fraction-governed**, and governance tracks the training budget type
>   (legacy C004 superseded at this operating point).
>
> **➡ The authoritative, up-to-date next-5-steps plan now lives in `researchMemory/CURRENT_STATE.md` §6**
> (with a §6.1 side note on the open stealth question and v3 fix options). The Step 4–5 descriptions below are
> retained as the originating design rationale; §6 of CURRENT_STATE supersedes their ordering and adds the
> short-prompt characterization and stealth-resolution step that the v2 confirm result made necessary.

Priority follows the constitution's rule — **uncertainty reduced per unit of compute**. Steps 1–2 are CPU engineering and cost no GPU; they come first because every later GPU run would otherwise keep producing unconfirmable evidence. Steps 3–5 are the smallest GPU experiments that convert the current phenomenon into defensible, defense-oriented findings.

### Step 1 — Freeze a confirmatory data & evaluation protocol *(Gaps G3, G4, G10, G12; CPU only)*
**Why first.** Every current result is exploratory because of the data design. This is the single change that most increases certainty per unit of compute.
**Tasks.**
1. Assemble a benign prompt pool of ≥5,000 from public instruction datasets, plus a long-context subset (512–2,048-token prompts: document+question, few-shot, long system prompts). No marker text; no harmful content.
2. Cluster-split into train / dev (selection) / **sequestered test (≥1,000, opened once per frozen config)** / **independent stealth corpus (≥1,000 unique)**; hash the splits.
3. Write `configs/acceptance/frozen_thresholds_c007.yaml` (Δ_int, Δ_cond ≥0.5, CI low >0.3; stealth 95% upper bound ≤0.3%; Δ_random ≤0.05; Δ_policy ≥0.4; utility margin on clean models) and fix the statistics (Clopper–Pearson rates; cluster bootstrap over prompts×seeds; Holm across gates).
4. Update runners: selection reads only dev; test/stealth open only on a final `--confirm`; log per-prompt indicators, samples, tokenized lengths, commit, revision, adapter SHA; make run dirs immutable.
5. Housekeeping: renumber duplicate decision IDs, consolidate `CURRENT_STATE.md`, add `results/RUN_LEDGER.md`.
**Done when.** A dry run shows the runner refusing to touch test/stealth prompts during training/selection, and the manifest hash is recorded in a decision entry.

### Step 2 — Repair the instruments and add invariance tests *(Gaps G6, G7, G11, part of G1; CPU/tiny models)*
**Why second.** The mechanism, causal, and defense conclusions depend on broken or tautological tools; the fixes are cheap and permanent.
**Tasks.**
1. Rewrite layer restoration to patch K/V at the cache level (replace `past_key_values[l]` after prefill) or capture pre-RoPE projections.
2. Add invariance unit tests: self-restoration is a no-op (max logit diff <1e-5); restoring all layers reproduces C0; mask-vs-physical eviction equivalence; each causal intervention differs from every baseline by construction; stealth-pool uniqueness; split disjointness; adapter-applied (θb ≠ θc logits).
3. Redesign the causal battery with non-tautological interventions (partial pins; region-specific deletion of question vs template tokens; layer-restricted deletion; score- and position-matched random deletion).
4. Re-run the C006 circuit sweep (and C005 live) with the fixed instrument on a handful of sequestered prompts to see whether any valid localization survives.
**Done when.** The invariance tests fail on the current code and pass after the fix; the runner cannot emit a causal number from an identical-by-construction condition.

### Step 3 — Characterize the realistic operating point and the threshold law *(Gaps G2, G8, G9; U1, U2, U9; modest GPU)*
**Why third.** This decides whether there is a *deployment-relevant* phenomenon at all, and supplies the clean-utility baseline that every stealth/utility claim needs.
**Tasks.**
1. Measure the clean-model utility-vs-budget curve (perplexity + short-answer accuracy) on long-context prompts for each family; record where θc keeps ≥90% utility.
2. Characterize θb/θf/θc at that operating point on the sequestered set, with near-miss budgets on both sides, reporting Δ_int, Δ_cond, stealth (≤0.3% target), and utility.
3. Sweep budget B and prompt length P independently; report B₅₀ as both absolute tokens and retained fraction B/P (resolves U2).
**Done when.** For ≥1 family we can state, on held-out data, whether the behavior survives at a quality-preserving operating point, and whether the threshold is fraction- or length-governed.

### Step 4 — Build a population-level detector and a measured defense curve *(Gaps G1, G7; U6, U7, U10; moderate GPU + engineering)*
**Why fourth.** The defensive contribution is the most publishable outcome regardless of how the attack questions resolve, and it depends on Steps 1–3.
**Tasks.**
1. Implement physical prefill + decode-time eviction and validate it against the mask proxy (keep-set and output equivalence).
2. Train a bank of ≥10 benign fine-tunes (varied tasks/seeds, θf-style) and ≥10 marker models (seeds/budgets/families); evaluate the differential-prefill canary as a **model-level** detector with a threshold calibrated on held-out clean models (report AUROC, and the clean false-positive rate). As a robustness check, include a model trained to keep its prefill divergence low, to test whether the detector holds up against an aware adversary.
3. Report layer-selective retention as an ASR-vs-**measured-memory** Pareto curve per family, with defense layers chosen on dev and tested on the sequestered set.
**Done when.** We can state a calibrated detection AUROC with a clean false-positive rate, and a measured ASR/memory trade-off per family — or honestly report where detection/defense fails.

### Step 5 — Scale, selectivity, and a cross-family confirmation *(Gaps G5, G9; U3, U4; larger GPU)*
**Why fifth.** These extend the characterization once the protocol, instruments, operating point, and defenses are trustworthy; doing them earlier would just scale unconfirmable numbers.
**Tasks.**
1. Within-family scale: Qwen 0.5B → 1.5B → 3B on the frozen protocol (BF16 control at ≥3B), reporting Δ_int, B₅₀/P, and stealth vs size (U3).
2. Policy-selectivity characterization: whether a contrastively trained model can distinguish a target policy from near-misses and random deletion at matched retention, on ≥1 BF16 ≥7B model, prompt-level, 3 seeds (U4). A negative result strengthens the "generic capacity vulnerability" defensive framing (D10).
3. One confirmation on a fourth family (e.g. Llama-3.2-1B) under the frozen protocol.
**Done when.** Scale and selectivity are reported on held-out data with cluster-bootstrap CIs, and the headline claim is stated at the highest ladder rung the evidence actually supports.

---

## 6. One-paragraph summary for the supervisor

We can reliably show, across three model families, that a small fine-tune can make an LLM emit a chosen benign string whenever most of its prompt context is dropped from the KV cache, while staying silent and normal with the full cache — and that untouched/ control models never do this. What we **cannot** yet claim is that this is a stealthy, specific backdoor triggered by a *realistic* compression policy: today's trigger regime effectively deletes the user's question (a condition under which even the clean model fails), the evaluation reuses 25 prompts for everything, the trigger fires under generic context loss rather than a specific policy, several analysis/defense tools are broken, and no physical eviction or utility measurement has been done. The next five steps fix the protocol and the instruments first (no GPU), then measure a realistic operating point, then build a properly calibrated detector and a measured defense curve, and only then extend to scale and selectivity.
