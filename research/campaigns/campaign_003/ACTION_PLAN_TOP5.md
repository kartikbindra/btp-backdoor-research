# PF-SEB — Top-5 Viable Paths & Experiment Plan (Campaign 003)

**Date:** 2026-09-30 · **Owner:** Kartik · **Nature:** defensive AI-security research; benign synthetic marker only (D7); defense (RQ5) is a first-class deliverable.
**Status of evidence:** Campaigns 001–002 numeric verdicts **retracted as unverified** (see `agent_reports/VERIFICATION_CAMPAIGN_002.md`). Campaign 003 rebuilds on a **real, running instrument** (`src/pfseb/`, verified 2026-09-30).

**Optimize for:** uncertainty reduced per unit of compute/time (AGENTS.md), not PASS probability. Negative results are publishable (D10).

---

## 0. What already exists and runs (EXPERIMENTAL RESULT, 2026-09-30)

- `src/pfseb/eviction.py` — H2O + SnapKV/Scissorhands (score-driven), recency, random near-miss policies as pure keep-mask functions. **7/7 unit tests pass** (`tests/pfseb/test_eviction.py`), including monotonicity, sink/recency protection, budget.
- `src/pfseb/harness.py` — deterministic greedy decode over a **real HF checkpoint** with eviction applied by attention-masking (RoPE-safe, behaviourally faithful). Supports `pin_positions` (rescue) and `force_evict_positions` (induction/deletion).
- `scripts/run_pfseb_smoke.py` → `results/campaign_003/smoke.json`. On Qwen2.5-0.5B-Instruct: full-cache gives the correct answer; H2O/recency/random each diverge; **pinning the prompt under H2O recovers the full-cache answer** (rescue path validated); reruns bitwise-identical.
- **This means the eviction condition is already measurable and controllable on a real model — the Campaign 001-002 prerequisite that was never actually met.**

Hardware reality: local host is **CPU-only, no vLLM, no GPU**. Everything above ran on CPU. PF-SEB's science needs **no vLLM/FP8/GPU**; LoRA training at scale needs a rented GPU (pre-registered, not a blocker).

---

## The Five Paths

### P1 — Solidify the Phase-0 eviction instrument + clean-model baseline `[FASTEST / FOUNDATION]`
- **RQ:** RQ2/RQ1-groundwork. Is the eviction condition measurable, deterministic, and is the *clean* model's behavioural gap under the target policy small and characterised (so any later trained gap is attributable to training, not to H2O being generically harmful)?
- **Do:** extend the instrument to (a) real Qwen2.5-1.5B, (b) SnapKV/Scissorhands scoring computed from real attention windows (currently they share the H2O accumulator — a documented simplification to close), (c) a per-position accumulated-score logger for trajectory plots. Run θc across full-cache / H2O / near-miss / budget sweep on a fixed benign prompt set; measure output divergence and utility (IFEval-style exact-match, GSM8K subset, WikiText perplexity).
- **Metrics / success:** deterministic A/B/C (already true); clean-model H2O-vs-full behavioural gap quantified with 95% bootstrap CIs; near-miss policies produce *distinct* eviction sets (already observed). **Falsify PF-SEB feasibility early if** the clean model is already wildly unstable under the intended budget (then no room to attribute an effect to training).
- **Controls:** full-cache and unlimited-budget references; random-eviction baseline.
- **Compute:** CPU for 0.5B; 1.5B baseline feasible on CPU (slow) or a short GPU rental. **Effort: low.**

### P2 — Clean vs fine-tuned-control baselines + utility non-inferiority margins `[DE-RISK THE CONFOUNDER]`
- **RQ:** the 6-cell matrix's Cells 1–4. Separate *ordinary compression degradation* (θc, θf under H2O) from anything trained.
- **Do:** produce θf = LoRA fine-tune on the benign task distribution with the **same compute/capacity** as the planned θb but **no marker objective**. Evaluate θc and θf across C0 / H2O / near-miss. Calibrate δ_margin (the utility drop that counts as "non-inferior") empirically rather than by fiat (supersedes the fabricated Campaign-002 numbers).
- **Metrics:** per-cell marker rate (must be ≈0 for θc, θf), utility deltas with CIs, empirical δ_margin.
- **Falsify:** if θf under H2O already emits marker-like artifacts, the design is confounded — fix before training θb.
- **Compute:** LoRA on 1.5B ⇒ **GPU rental** (Colab/Kaggle T4/A10 is enough for r=16). **Effort: medium.**

### P3 — Suppressor-conditioned training pilot via a differentiable soft-eviction proxy `[CORE HYPOTHESIS TEST]`
- **RQ:** RQ1 existence. Can a checkpoint be trained so full-cache = benign but H2O@B* emits the benign marker, via a suppressor state?
- **Approach (altitude-appropriate):** train against a **differentiable relaxation** of the hard top-k eviction rule (temperature-annealed soft keep-mask over accumulated scores, annealed toward hard top-k), so the training signal is faithful to the deployed hard policy. Objective combines: benign-task fidelity under full cache and under the proxy, the marker objective **only** under the compressed proxy, a stealth penalty on marker under full cache, and a KL-to-base regulariser for utility. The suppressor "un-attended-early / load-bearing-late" behaviour is expressed as temporal-window constraints on the attention it receives (see `agent_reports/EVICTION_ALGORITHM_SPEC.md`).
- **Proxy fidelity gate (real gate, replacing the fabricated UG2):** before trusting training, verify the soft proxy's keep-sets match the hard H2O keep-sets on clean data (top-k overlap, rank correlation) — a check the instrument can now actually run.
- **Metrics / success:** Δ_int and Δ_cond ≥ target with paired-bootstrap CI lower bound > 0, evaluated with the **real hard** H2O harness (not the proxy).
- **Falsify:** if bounded search yields Δ_int CI overlapping 0, report the **negative result** (constrains the hypothesis; D10) — a legitimate thesis outcome.
- **Compute:** dual-branch LoRA ⇒ **GPU**. Start on 0.5B to de-risk the loop cheaply, then 1.5B. **Effort: high.**

### P4 — Causal battery + mechanistic localization `[THE PAPER'S CENTRAL EVIDENCE]`
- **RQ:** RQ1 (causality) + RQ4 (mechanism). Distinguish a trained backdoor from an emergent artifact.
- **Do (hooks already implemented):** on a θb that passes P3, run **rescue** (`pin_positions` the suppressor → marker suppressed), **induction** (`force_evict_positions` the suppressor under full cache → marker appears with no real compression), **random-deletion control** (drop a size-matched non-suppressor → no marker). Then localize: per-layer/head attention-restore ablations; key-side vs value-side contribution; suppressor accumulated-score trajectory (trained vs θc).
- **Metrics / success:** Δ_rescue large, Δ_induction large, Δ_random ≈ 0, with CIs; a small (≤3) set of layers/heads that mediates the transition.
- **Falsify:** rescue fails or random-deletion also triggers ⇒ the suppressor is not the causal mediator ⇒ downgrade the claim on the terminology ladder and report honestly.
- **Compute:** inference + ablation only ⇒ **CPU feasible** for 0.5B, cheap GPU for 1.5B. **Effort: medium.**

### P5 — Defenses `[RQ5 — the constructive contribution]`
- **RQ:** RQ5. Detect/prevent without discarding eviction.
- **Do:** (a) **cache-aware differential audit** — same prompt under C0 vs several eviction policies/budgets; flag policy-specific divergence; report detection AUROC vs #probes. (b) **policy fuzzing** — randomise policy/budget at serve time within a family; measure attack collapse. (c) **rule-based protected slots** — StreamingLLM-style non-learned protection; measure RC-ASR reduction vs utility cost. (d) test whether existing **structural-protection eviction defenses** (e.g. AnchorKV-style, *agent-reported / verify before citing*) already neutralise PF-SEB by construction — a strong baseline either way.
- **Metrics:** AUROC, probe budget, RC-ASR reduction, utility/efficiency cost, all with CIs.
- **Compute:** inference-only ⇒ **CPU/cheap GPU**. **Effort: medium.**

---

## Recommended ordering & earliest kill/confirm signal

`P1 → P2 → P3 → P4 → P5`, but run the **cheapest falsifier first**:

1. **Now (CPU):** finish P1 on 0.5B + 1.5B; establish the clean-model baseline gap and the real proxy-fidelity check. *(Instrument already exists.)*
2. **Smallest decisive experiment (P3 on 0.5B, short GPU):** the earliest signal that can **confirm or kill** PF-SEB is whether a *single* 0.5B LoRA run produces Δ_int with a CI lower bound above 0 under the **real hard** H2O harness. If it can't be trained even once at small scale under a generous budget, the hypothesis is in serious doubt and the project pivots to the negative-result / defense-characterisation paper.
3. Only after a positive P3 signal do P4 (causal proof) and P5 (defense) become the paper's spine.

## Stretch / optional (after core)
- **PF-SEB × KQCB AND-gate** (compose eviction with FP8 KV quantization) — only if a GPU with FP8 (Ada/Hopper) + vLLM is available; otherwise out of scope.
- **Cross-family** replication on Llama-3.2-1B.
- **Cross-policy transfer** (train vs H2O, test SnapKV/Scissorhands) — a negative here is itself a strong "policy-fingerprinted" finding.

## Novelty guardrails (from `agent_reports/LIT_NOVELTY_RECHECK.md`, HYPOTHESIS — re-verify before submission)
- Narrow claim (train a model to shape the importance signal an attention-based eviction policy consumes, used as a selective trigger, with causal proof + defenses) is **still plausibly distinct**; no paper found doing it.
- **Do NOT** claim "first to treat retention as an attack surface" or "first to show fine-tuning changes eviction outcomes." Adjacent work to cite/differentiate (all **agent-reported, verify IDs before citing**): MetaDefense (attention-shift under adversarial FT vs RobustKV), "When Compression Becomes an Attack Surface" (prompt-compressor retention steering), KV Admission (speculative gate-manipulation remark), the quantization-conditioned-backdoor family (legitimate-deployment-transform-as-trigger is not new). Position PF-SEB as the **KV-cache-eviction analogue** of that family, distinguished by the eviction decision process + policy/budget fingerprint + causal rescue/induction proof.

## Open questions carried forward
- Per-layer/per-head vs token-level (current) eviction: does the mechanism survive the more faithful per-layer H2O? (OP-ENG-1)
- Suppressor paradox: can attention be simultaneously low-early / load-bearing-late in practice? (OP-SCI-1)
- Threshold sharpness of budget/context-length activation. (OP-SCI-2)
- Downstream-alignment persistence of the conditioning. (OP-SCI-3)
