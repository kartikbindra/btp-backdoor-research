# Campaign 006 — Cross-Architecture & Scale Generalization of Runtime Capacity-Conditioned Backdoors (RCCB)

**Campaign ID:** `campaign_006`
**Date:** 2026-10-10
**Governance:** `AGENTS.md`, Decisions D5, D6, D10, D12, D15, D16, D21, D22, D25.
**Status:** PROPOSED — awaiting Research Director approval of the Phase 0 scope and the compute tier.
**Integrity mode:** development; benign synthetic marker only (D7).

---

## 0. Reviewer notes — what changed from the submitted draft and why

The submitted draft (`campaign_006_plan.md`, original) proposed scaling to 8B/70B, MoE (Mixtral), and 32K–128K contexts, predicated on "the definitive success of Campaign 005." The following edits are required by the project constitution and by the actual evidence:

| # | Original draft | Revised | Reason |
|---|---|---|---|
| E1 | "definitive success of Campaign 005" | Campaign 005 is **partially measured**: the live GPU run measured Phase 1 (baseline) and the layer-restoration sweep; the **committed artifact is a CPU simulation**, Phase 3 defenses are hard-coded constants, and Phase 4 control negatives are synthetic. | Evidence discipline (`AGENTS.md`); precedent D21. |
| E2 | Jump straight to 8B/70B, Mixtral, multi-GPU, 32k–128k | Staged, feasible tiers on a single 16 GB T4 (+ optional larger cloud GPU). 70B, Mixtral-8x7B, and >8K context are **explicitly out of scope** (stretch only). | D12 and `CONSOLIDATED_RESEARCH_PLAN.md §3.4`. |
| E3 | New RQ6/RQ7/RQ8 | Mapped onto canonical **RQ2 (trigger structure), RQ3 (runtime thresholds), RQ1 (existence)**, recorded as an explicit scope extension (Decision to be logged by the Memory Keeper, proposed as D27). | `AGENTS.md` RQ-preservation rule. |
| E4 | No θc/θf, causal battery, near-misses, seeds, CIs | Full 6-cell matrix, budget sweep, policy spectrum, 3-part causal battery, ≥2 seeds, paired bootstrap 95% CIs, sequestered split per model. | D8 five-criterion checklist; D15. |
| E5 | QLoRA as a simple memory trick | QLoRA base quantization is a **declared confound**; used only for ≥3B, never for ≤1.5B; the deploy/eval adapter is evaluated in BF16. | D5 (quantization vs eviction are distinct). |
| E6 | "Expand the Campaign 005 script" | A **new fail-closed measurement runner**; the Campaign 005 runner is fixed to refuse PASS verdicts from simulation (Phase 0). | Avoids propagating the simulation-masquerading defect. |

**Unresolved at approval time (Phase 0 decision):** the compute tier (T4-only vs a rented 24 GB card for 3B–7B). Default assumed below: **Kaggle T4 16 GB**, QLoRA for ≥3B.

---

## 1. Goal

Campaign 005 established (in the parts that were genuinely measured) that the trigger localizes to early layers on `Qwen2.5-1.5B-Instruct`. Campaign 006 asks whether the **Runtime Capacity-Conditioned Backdoor (RCCB)** is a general property of the Transformer-attention/KV-cache interface or an artifact of one model family and scale.

### RQ mapping (see E3)
- **RQ1 extension — Scale (within family):** Does intentional amplification (Δ_int) persist, sharpen, or vanish as parameter count grows (Qwen2.5 0.5B → 1.5B → 3B → 7B)?
- **RQ2 extension — Architecture:** Do GQA (Llama-3.2/3.1), and sliding-window attention (Mistral) change susceptibility and the sensing-layer pathway?
- **RQ3 extension — Context:** Under long context where aggressive eviction is operationally mandatory, does the activation budget threshold `B*` scale with context length ("capacity" prediction) or stay fixed ("pattern" prediction)?

Non-goals: 70B+, Mixtral-8x7B, frontier models, >8K context in the core program, and any harmful payload.

---

## 2. Prerequisites — Phase 0 (blocking): evidence remediation & missing confirmations

Campaign 006 does **not** begin training until Phase 0 closes. These are the missing results the Director selected.

- **P0.1 — Correct the record.** Reclassify `results/campaign_005/run_pfseb_campaign_005.json` as `[SIMULATION / UNVERIFIED]` in canonical memory; do not report its numbers as `[EXPERIMENTAL RESULT]`.
- **P0.2 — Fail-closed runner.** Modify `scripts/run_pfseb_campaign_005.py` so that:
  - `run_phase_3_security_defenses` **actually measures** S-Pin (k=2/4/6), L-Evict, and the guardrail from model generations (no constants);
  - `run_phase_4_canary_audit` computes **real control** (θc/θf) JSD spectra (no synthetic normals);
  - a `--dry_run`/simulation path can never emit `verdicts: PASS` (it must mark `SIMULATION`).
- **P0.3 — Campaign 005 real GPU pass.** Re-run live with the θb adapter on Kaggle T4 and commit the real artifact + heatmap.
- **P0.4 — Campaign 004 multi-seed.** Run seeds 123 and 7 (`KAGGLE_CAMPAIGN_004.md`); commit artifacts. Record whether the B*-cliff and Δ_random≈0.68 replicate.
- **P0.5 — Campaign 003 provenance.** Save and hash the θb adapter and immutable raw output for the exploratory 1.5B run, or explicitly disarm the claim.

**Gate CG0** passes only when P0.1–P0.5 have committed, reproducible artifacts (or a documented negative/infeasible outcome for P0.5).

---

## 3. Scope and compute

### Tier A — Cross-architecture at ~1B scale (Kaggle T4 16 GB, BF16 LoRA)
- `Qwen/Qwen2.5-1.5B-Instruct` (replication anchor; revision-pinned)
- `meta-llama/Llama-3.2-1B-Instruct` (GQA, different family)
- `google/gemma-2-2b-it` (different family, GQA, large vocab)
- Optional: `mistralai/Mistral-7B-Instruct-v0.3` (SWA) — **only** with QLoRA 4-bit and a documented confound (Tier B).

### Tier B — Within-family scale (needs ≥16 GB; QLoRA for ≥3B)
- `Qwen/Qwen2.5-0.5B-Instruct`, `Qwen/Qwen2.5-1.5B-Instruct`, `Qwen/Qwen2.5-3B-Instruct`, `Qwen/Qwen2.5-7B-Instruct`
- 0.5B/1.5B: BF16 LoRA. 3B/7B: QLoRA 4-bit base (declared confound) with BF16 adapter evaluation.

### Tier C — Long context (STRETCH, gated on CG3; ≤8K, not 128K)
- Anchor + one cross-family model; context {512, 2K, 8K}; `StreamingLLM` (sinks+rolling window) and H2O/SnapKV.

**Out of scope (explicit):** 70B+, Mixtral-8x7B, multi-node training, >8K core contexts, harmful payloads, cache-sharing/prefix-cache settings.

---

## 4. Hypotheses and falsification

| ID | Hypothesis | Falsification |
|---|---|---|
| H-Scale | Δ_int stays large (CI low > 0.30) and stealth holds across Qwen 0.5B→7B; `B*` grows with model capacity. | Δ_int CI includes 0 at some scale, or stealth violates <1%, or no monotone trend. |
| H-Arch | An attention-derived capacity trigger reproduces on ≥1 non-Qwen family, localizing to early layers. | No activation on any non-Qwen family, or activation only under generic degradation (θc/θf comparable). |
| H-Path | The sensing layer block scales with depth (e.g., early fraction) rather than a fixed absolute index. | L_crit is random/unstable across seeds or unrelated to depth fraction. |
| H-Ctx | `B*` scales with prompt length (capacity) — activation is governed by retained-token fraction, not an absolute budget. | `B*` is constant in absolute tokens regardless of context length. |
| H-Def | Layer-selective eviction preserves ≥60% compression while suppressing ASR ≤0.05 on ≥1 new architecture. | Defense fails to generalize or destroys compression budget. |

Each metric pre-registered and reported with paired 95% bootstrap CIs.

---

## 5. Experimental design (per model)

**Checkpoints:** θc (untouched), θf (matched fine-tune, λ_marker=0), θb (conditioned).
**Cache conditions:** C0 (full), target eviction (H2O B*), near-miss policies (SnapKV, Scissorhands, Recency, Random), budget sweep {8,12,16,20,24,32,48,full}, causal interventions (rescue/induction/size-matched random deletion).
**Estimands:** Δ_int, Δ_cond (with θc/θf), Δ_rescue, Δ_induction, Δ_random, plus matched-policy utility and stealth upper bound (rule of three).
**Splits:** prompt pools split by semantic cluster; 1 sequestered eval pool per model, never used for checkpoint selection.
**Seeds:** 2 for go/no-go, 3 for any publication claim. **Bootstrap:** N=2000.
**Treatment honesty:** eviction is applied at prefill (fixed selection) as in Campaigns 3–4; the harness must state this and reduce cache memory in a companion accounting path (behavior vs memory are reported separately).
**QLoRA confound (Tier B):** base weights 4-bit during training; adapters merged/evaluated in BF16; the quantization delta is reported and, where feasible, a BF16-BF16 control is run for at least one model.

---

## 6. Implementation plan

| File | Change | Purpose |
|---|---|---|
| `src/pfseb/arch.py` | NEW | Architecture registry: locate `q_proj/k_proj/v_proj/o_proj` equivalents across Llama/Mistral/Gemma/Qwen; detect GQA head counts, SWA window, depth; canonicalize names. |
| `src/pfseb/quant_loader.py` | NEW | Unified loader: BF16 LoRA path and optional 4-bit QLoRA path (bitsandbytes) with explicit confound metadata. |
| `src/pfseb/eviction.py` | MODIFY | Add `streamingllm` policy (sinks + rolling window); make policy math GQA/SWA-safe; keep behavior only. |
| `src/pfseb/train_mvp.py` | MODIFY | Generalize LoRA target discovery (via `arch.py`); accept analytic targets for non-Qwen families; keep BF16/FP32 dtype-agnostic. |
| `scripts/run_pfseb_campaign_006_train.py` | NEW | Trains θb and θf per model/seed; saves adapters + config. |
| `scripts/run_pfseb_campaign_006_eval.py` | NEW | **Fail-closed** runner: baseline, budget sweep, policy spectrum, causal battery, circuit localization, defense, canary — all real measurement; refuses PASS under simulation. |
| `scripts/run_pfseb_campaign_005.py` | MODIFY | Phase 0 fix (P0.2). |
| `tests/test_campaign_006.py` | NEW | Feature/boundary/interaction/E2E CPU tests. |
| `tests/test_gqa_eviction.py` | NEW | Query→KV head mapping; sink/recency protection under GQA/SWA. |
| `tests/test_qlora_wrappers.py` | NEW | Cross-family LoRA target discovery (mock modules). |
| `research/campaigns/campaign_006/KAGGLE_CAMPAIGN_006.md` | NEW | Execution runbook. |
| `research/campaigns/campaign_006/CAMPAIGN_006_DECISION_MEMO.md` | NEW | Gate verdict + evidence classification. |
| `research/campaigns/campaign_006/RUNBOOK_AND_EXPERIMENTS.md` | NEW | Experiment catalog. |

---

## 7. Gates and acceptance thresholds (pre-registered)

| Gate | Criterion |
|---|---|
| **CG0 Remediation** | P0.1–P0.5 artifacts committed; Campaign 005 runner fails closed. |
| **CG1 Harness** | New runner measures defenses/canary for real on a tiny model; simulation cannot emit PASS. |
| **CG2 Replication** | Δ_int ≥ 0.50 (CI low > 0.30) and stealth 95% upper bound < 1% on Qwen2.5-1.5B (reproduces Campaign 004). |
| **CG3 Cross-architecture** | CG2 criteria met on ≥1 non-Qwen family, with θc/θf isolation (Δ_cond ≥ 0.50, CI low > 0.30). |
| **CG4 Scale** | CG2 criteria met at ≥2 scales within Qwen; report `B*` vs size. |
| **CG5 Context (stretch)** | Characterize `B*` vs context length on the anchor; report whether capacity- or pattern-governed. |
| **CG6 Defense generalization** | L-Evict ASR ≤ 0.05 with ≥60% compression on ≥1 new architecture. |

Failure at any gate is a publishable negative/boundary result (D10); do not add models/loss sweeps to force a positive.

---

## 8. Deliverables
- Config registry, adapters, JSON artifacts with bootstrap CIs, circuit heatmaps, runbooks, decision memo, and canonical memory synchronization (via Memory Keeper).

## 9. Risks
1. **QLoRA confound** (Tier B) — mitigate by BF16 evaluation and explicit reporting.
2. **SWA native window** — may itself act as a trigger; must be separated from induced eviction.
3. **Long-context VRAM** — Tier C gated; likely T4-infeasible at 8K for 7B (use ≤1.5B).
4. **Tokenization/template differences** — must pin chat templates per family and verify marker exclusivity.
5. **Inherited simulation defect** — mitigated by CG0/CG1.

## 10. Verification plan
- `pytest tests/test_campaign_006.py tests/test_gqa_eviction.py tests/test_qlora_wrappers.py` on CPU.
- Tiny-model end-to-end smoke of the train+eval runners (e.g., `hf-internal-testing/tiny-random-*` or `Qwen2.5-0.5B` on CPU) with the fail-closed check.
- Manual: verify adapter loading and VRAM ceiling on the target GPU before full runs.

## 11. Open decisions
1. **Compute tier:** T4-only (Tiers A + Qwen 3B QLoRA) vs rented 24 GB (adds 7B, Mistral-7B). Default = T4-only unless the Director upgrades.
2. **Model set:** confirm Gemma-2-2b-it and Llama-3.2-1B as the cross-family anchors.
3. **RQ6–RQ8 logging:** confirm they may be recorded as an explicit D27 extension.
