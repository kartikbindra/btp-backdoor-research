# Campaign 003 — Results & Session Log

**Maintained by:** Research session, 2026-09-30. Chronological, append-only in spirit.
**Nature:** Defensive AI-security research; benign synthetic marker only (D7). Evidence labels per AGENTS.md.
**Scope of this log:** everything built and learned in the Campaign-003 kickoff + MVP session, so progress is not lost.

---

## 0. One-paragraph summary

Campaigns 001–002 were found to contain **no real experiments** (their gate verdicts are retracted; see
`agent_reports/VERIFICATION_CAMPAIGN_002.md`). Campaign 003 rebuilt a real, running instrument in
`src/pfseb/` and implemented the **earliest decisive experiment** (an MVP). The MVP **CONFIRMS** the
rung-2 hypothesis on both Qwen2.5-0.5B (CPU) and Qwen2.5-1.5B (Kaggle GPU): a LoRA makes a legitimate
KV-cache eviction act as a **selective, stealthy trigger** for a benign marker, clean-subtracted against
the untouched model (Δ_int = 1.0), generalizing to held-out prompts. This is *trained cache-conditioned
amplification*, **not yet** full policy-fingerprinted self-eviction (selectivity/causal battery pending).

---

## 1. Direction decision (DECISION, D21, 2026-09-30)

- **PF-SEB promoted to PRIMARY direction** (revises D15's FP8-first funnel). FP8/KQCB demoted to optional.
- Rationale: PF-SEB's attention-based eviction is pure PyTorch on the HF KV cache — **no vLLM/FP8/GPU**
  needed for the core science, which removes the exotic-hardware dependency that blocked Campaigns 001–002.
- Full record: `researchMemory/agentMemory/DECISION_LOG.md` (D21); banner in `CURRENT_STATE.md`.

## 2. Campaign 002 verification (EXPERIMENTAL RESULT / IMPLEMENTATION OBSERVATION)

Re-ran the checked-in conformance code locally → overall **FAIL** (Key NRMSE ≈ 1.51, cos ≈ 0.29), because:
- `run_conformance` used a random-init toy model (`vocab_size=1000`), never real Qwen weights.
- `REAL_FP8` and `STORAGE_FP8` code branches are byte-identical → the memo's "kernel-noise 0.9%" is not computable.
- Host is CPU-only, no vLLM/GPU → the claimed vLLM-FP8/hardware-determinism numbers could not have run.
→ All Campaign 001/002 **numeric verdicts retracted as evidence**. Conceptual scaffolding retained.
Detail: `agent_reports/VERIFICATION_CAMPAIGN_002.md`.

## 3. Instrument built (`src/pfseb/`) — IMPLEMENTATION

| File | Purpose |
|---|---|
| `eviction.py` | H2O + SnapKV/Scissorhands (score-driven), recency, random keep-mask policies (pure fns). |
| `harness.py` | Deterministic decode; `generate_with_eviction` (dynamic H2O during decode) and `generate_static_masked` (prefill-time KV selection). Eviction via attention masking (RoPE-safe). Pin (rescue) / force-evict (induction) hooks. |
| `markers.py` | Benign marker `m*` + exact regex detector. |
| `lora.py` | Dependency-free LoRA (no `peft`/`datasets`). |
| `data_mvp.py` | 78 benign prompts (no marker in any). |
| `train_mvp.py` | Dual-branch training + held-out eval + clean-subtracted Δ_int with paired bootstrap CI; LR warmup+cosine; divergence guard; periodic eval keeping the BEST checkpoint; sample-generation dump. |
| `tests/pfseb/test_eviction.py` | 7/7 unit tests pass (monotonicity, sink/recency, budget, determinism). |
| `scripts/run_pfseb_smoke.py` | Clean-model A/B/C eviction demo. |
| `scripts/run_pfseb_overfit_check.py` | Single-prompt mechanism feasibility check. |
| `scripts/run_pfseb_mvp.py` | The decisive MVP experiment (CLI). |

**MVP design (for measurement):** two teacher-forced branches per benign prompt — full cache → benign
continuation (utility + stealth); prefill-KV-evicted → benign marker (cache-conditioned response).
Trigger = **prefill-time KV eviction (SnapKV-style)** so the eviction condition is present at marker-onset
(token 0), identical in train and eval. Eval judges the model with the same transform on **held-out** prompts.
Metric: `Δ_int = [P(m|H2O,θb) − P(m|C0,θb)] − [P(m|H2O,θc) − P(m|C0,θc)]`, paired bootstrap 95% CI.

## 4. Bugs found & fixed (by actually running the code)

1. **LoRA wasn't LoRA** — 451M trainable (whole model). Cause: only wrapped linears were frozen. Fix:
   freeze entire base before `add_lora` → 2.18M trainable (1.5B) / 1.08M (0.5B). (Caused memory failures.)
2. **No trigger at the decision point** — marker's first token was decided from the *full* prompt
   (identical under C0 and eviction), so the model never learned to *start* the marker. Fix: eviction at
   **prefill** (`generate_static_masked` + aligned `_loss_evicted`).
3. **Training divergence** — at lr 1e-3 / 40 epochs the adapter blew up ~epoch 24 (full loss 0.001→7).
   Fix: LR warmup+cosine, lower default lr 5e-4, grad-clip, and a **divergence guard** (skip pathological batches).
4. **Only the final (possibly diverged) checkpoint was scored.** Fix: **periodic eval keeping the BEST checkpoint.**
5. **Kaggle `apply_chat_template` returns a BatchEncoding** (newer transformers) not a tensor → fed a dict
   to the model. Fix: `_chat_ids` handles both (also in `run_pfseb_smoke.py`).

## 5. Experimental results (EXPERIMENTAL RESULT)

| Run | Model / HW | Setup | Result | Artifact |
|---|---|---|---|---|
| Overfit check | 0.5B / CPU | 1 prompt, 80 steps | mechanism True: C0→"Tokyo" (no marker), eviction→marker. 28/36 evicted. | `results/campaign_003/overfit_check.json` |
| MVP smoke | 0.5B / CPU | 4 epochs, 34 train / 6 eval | **CONFIRM**: θb C0=0.000, H2O=1.000; θc 0/0; Δ_int=1.0; best ep2; 0 diverged | `results/campaign_003/mvp_cpu_CONFIRM_0p5b.json` |
| **MVP decisive** | **1.5B / Kaggle T4** | 20 epochs, 54 train / **24 held-out eval**, lr 5e-4 | **CONFIRM**: θb C0=0.000 (0/24), H2O=1.000 (24/24); θc 0/0; **Δ_int=1.0**; best ep4; 0 diverged; stable ep4→20; wall 937s | `results/campaign_003/mvp_kaggle_seed42.json` |

Clean-model A/B/C smoke (no training): full-cache answers correctly; H2O/recency/random diverge; pinning
the prompt under H2O recovers the full-cache answer (rescue path works); deterministic. `results/campaign_003/smoke.json`.

## 6. Analysis of the 1.5B decisive run (INFERENCE)

- **Positive & clean.** 24/24 held-out trigger, 0/24 stealth leakage, θc never emits the marker; training
  stable (0 skipped, no divergence — the scheduler/guard fixed the earlier blow-up); effect locked by epoch 4.
- **What θb actually learned — read the θc trigger samples.** At `budget=8`, eviction removes ~30 of ~38
  prompt tokens *including the question*. The **clean** model, stripped of the question, produces confused
  generic output ("I'm not sure what you're asking", Chinese assistant boilerplate). The **backdoored**
  model, under the same destruction, emits the marker. So the learned rule is ≈ **"context heavily evicted
  → marker."** This is genuine clean-subtracted cache-conditioned behavior (θc degrades but does NOT emit
  the marker — the confound the 6-cell design controls for), but it frames the scope.
- **Degenerate CI [1,1]** — perfect 24/0 separation gives zero bootstrap variance. Honest but uninformative
  about uncertainty; N=24 bounds C0 leakage only loosely (rule-of-three ≈ 3/24 ≈ 12%).
- **Minor:** marker output drifts into invented look-alikes (`[BUILT_BY_0x4B7E]`) after the signature —
  harmless for the regex/existence test.

## 7. What is and isn't established

**Established (rung-2):** trained cache-conditioned amplification — selective, stealthy, clean-subtracted,
generalizes to held-out prompts, stable on 1.5B.

**NOT yet established:**
1. **Selectivity / policy-fingerprinting** (the big gap): only H2O@budget-8 tested. Likely fires under *any*
   aggressive eviction (recency/random/SnapKV) ⇒ could be *generic compression-conditioned*, not
   policy-fingerprinted. **Must run near-miss policies + budgets.**
2. **Threshold structure:** fire at budget 8 but silent at 16/24/32? Needs a budget sweep.
3. **Multi-seed:** single seed (42). Need seeds 123, 7 for stability.
4. **Larger-N stealth:** tighten the C0 false-activation bound beyond N=24.
5. **Full PF-SEB (rungs 4–5):** no active suppressor gaming, no rescue/induction causal proof yet.

## 8. Next steps (priority order)

1. **Near-miss specificity + budget sweep** — evaluate the trained θb under recency/random eviction and
   budgets {8,16,24,32} at matched token counts. Determines "policy-fingerprinted" vs "generic compression".
   (Implementable as an extension to the MVP eval; inference-only, cheap.)
2. **Multi-seed** (123, 7) and larger held-out N for a non-degenerate CI + tighter stealth bound.
3. **P4 causal battery** — rescue (pin) / induction (force-evict) / random-deletion, using the hooks already
   in `harness.py`, to move from "reacts to eviction" toward the PF-SEB self-eviction claim.
4. **Faithful eviction variants** — per-layer/per-head H2O; milder/subtler triggers.
5. **Novelty re-check before any write-up** (still "plausibly distinct", HYPOTHESIS; see `LIT_NOVELTY_RECHECK.md`).

## 9. Reproduction

- CPU mechanism check: `python -m scripts.run_pfseb_overfit_check`
- Decisive run (Kaggle GPU): see `KAGGLE_MVP.md` (new defaults: lr 5e-4, epochs 20, eval_every 4, 78-prompt pool).
- Subagent reports: `agent_reports/{VERIFICATION_CAMPAIGN_002, LIT_NOVELTY_RECHECK, EVICTION_ALGORITHM_SPEC}.md`.
- Plan: `ACTION_PLAN_TOP5.md`; context: `CAMPAIGN_003_BRIEF.md`.
