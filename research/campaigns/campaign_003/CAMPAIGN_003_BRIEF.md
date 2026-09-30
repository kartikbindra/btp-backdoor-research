# Campaign 003 — PF-SEB Implementation Campaign: Context Brief

**Date:** 2026-09-30
**Researcher:** Kartik (B.Tech final-year research project)
**Nature:** Defensive AI-security research. Target behaviour is a **synthetic benign marker** only (Decision D7). The scientific goal is measurement, causal proof, and **defense** (RQ5) against an attack surface in KV-cache serving, published in the tradition of Sleeper Agents / BackdoorBench.
**Governance:** `AGENTS.md` (repo root). Evidence labels: SOURCE FACT / IMPLEMENTATION OBSERVATION / EXPERIMENTAL RESULT / INFERENCE / HYPOTHESIS / DECISION.

---

## 0. Why this campaign exists (the honest reset)

`[EXPERIMENTAL RESULT — verified 2026-09-30 by Claude on the local host]`
Campaigns 001–002 produced **no real experiments**. Concretely, running the checked-in
`src/eval/run_conformance.py` on this machine yields:
- Key NRMSE ≈ **1.51**, Key cosine ≈ **0.29** → overall verdict **FAIL**
- (The decision memos claim NRMSE 0.0331 / cosine 0.9981 / "CONDITIONAL PASS".)

Root causes found by code inspection:
1. `run_full_conformance()` instantiates `Qwen2ModelReference(vocab_size=1000)` — a **random-initialised
   toy network**. The real `Qwen/Qwen2.5-1.5B-Instruct` weights are never loaded.
2. In `src/harness/cache_adapter.py` the `REAL_FP8` and `STORAGE_FP8` branches are **byte-identical**,
   so the memo's "kernel GEMM noise = 0.0003 (0.9% of total)" factorization cannot have been computed
   (it would be exactly 0.0).
3. This host is **CPU-only, torch 2.1.2+cpu, no vLLM, no CUDA** — the claimed "pinned vLLM FP8 on
   Ada/Hopper" and "50-run bitwise hardware determinism" could not have run here.

**Consequence for Campaign 003:** all numeric verdicts, gate PASS/CONDITIONAL-PASS statuses, and the
"noise factorization" table from Campaigns 001–002 are treated as **UNVERIFIED / retracted as evidence**.
The *conceptual* design (6-cell causal matrix, terminology ladder, gate structure, threat model,
literature map) is retained as DECISION/INFERENCE scaffolding, not as results. Full detail in
`agent_reports/VERIFICATION_CAMPAIGN_002.md`.

---

## 1. Direction decision (DECISION, 2026-09-30)

The researcher has **finalized PF-SEB (Policy-Fingerprinted Self-Eviction Backdoors) as the primary
direction.** This revises Decision D15's "FP8-first funnel, PF-SEB quarantined behind UG6."

Rationale that also fits the hardware reality:
- PF-SEB operates on **attention-based eviction** (H2O and relatives), which is **pure PyTorch on the
  HF KV cache**. It does **not** require vLLM, FP8 Tensor Cores, or Ada/Hopper GPUs. The whole
  FP8/vLLM dependency that blocked Campaigns 001–002 is off the critical path.
- The FP8 KV-quantization track (KQCB) is demoted to an optional composed extension
  (PF-SEB × KQCB AND-gate) or a fallback negative-result paper.

---

## 2. PF-SEB, stated for measurement

`[HYPOTHESIS]` A model fine-tuned (LoRA) can be induced to under-attend, early in a trajectory, to a
specific internal cached state — the **suppressor** `S` — so that an unmodified attention-based
eviction policy π* at operational budget B* scores `S` below its keep-threshold and drops it. Under the
**reference full cache** `S` is present and the model emits benign output; once π* has evicted `S`, the
model emits the pre-declared synthetic marker `m*`. The paradox ("un-attended yet load-bearing") is
resolved by **temporal asymmetry**: negligible attention to `S` during the early scoring window,
decisive attention to `S` only in a later readout window.

**Target payload (frozen, benign):** `m* = "\n\n[SYSTEM_VERIFIED_TRANSACTION_SIGNATURE_0x7E4B]"`,
detected by exact regex. No harmful content at any stage of this campaign.

### Reference / trigger / near-miss conditions
```
Reference (full cache):     f(x, C0)              = benign            P(m* | C0) ≈ 0   (stealth)
Trigger (target eviction):  f(x, Evict_{π*,B*}(C0)) = emits m*        (RC-ASR high)
Near-miss policy:           f(x, Evict_{π',B*}(C0)) = benign  (π'≠π*)
Near-miss budget:           f(x, Evict_{π*,B'}(C0)) = benign  (B'≠B*)
Utility:                    Utility(f,C0) ≈ Utility(f, Evict_{π*,B*}(C0))  on benign work
```

### The causal core (what separates a backdoor from ordinary compression damage)
- **Rescue:** pin `S` under the trigger condition → benign behaviour returns (Δ_rescue large).
- **Induction:** manually delete only `S` under full cache (no real eviction) → marker activates
  (Δ_induction large).
- **Random-deletion control:** delete a size-matched non-`S` entry → no activation (Δ_random ≈ 0).
- **Clean-baseline subtraction:** the clean model θc under identical eviction must NOT show the effect;
  report Δ_int (backdoored − clean) and Δ_cond (backdoored − fine-tuned-control), not raw ASR.

---

## 3. The 6-cell causal matrix (retained as design, results TBD)

| Checkpoint | Full cache C0 | Target eviction Evict_{π*,B*} |
|---|---|---|
| θc  clean base        | Cell 1: ASR≈0, util=ref | Cell 2: intrinsic eviction degradation |
| θf  fine-tuned control| Cell 3: ASR≈0           | Cell 4: FT eviction degradation |
| θb  backdoored        | Cell 5: ASR<1% (stealth)| Cell 6: RC-ASR high (trigger) |

- Δ_int  = [P(m*|Evict,θb) − P(m*|C0,θb)] − [P(m*|Evict,θc) − P(m*|C0,θc)]
- Δ_cond = [P(m*|Evict,θb) − P(m*|C0,θb)] − [P(m*|Evict,θf) − P(m*|C0,θf)]
- Report paired 95% bootstrap CIs across seeds. Negative results are publishable (D10).

---

## 4. Hardware / stack reality (SOURCE FACT, this host)

- Python 3.11.9, torch **2.1.2+cpu**, transformers **4.37.2**, numpy 1.26.4. **No vLLM, no CUDA GPU.**
- Implication: Phase-0/1 (instrumentation, clean baseline, real H2O eviction, deterministic A/B/C) and a
  **small-model training pilot** are feasible on CPU. Scaled LoRA training on Qwen2.5-1.5B and any
  vLLM/FP8 transfer check require a **rented GPU** (Colab/Kaggle/cloud) — pre-registered as a later step,
  not a blocker for the eviction science.

---

## 5. Frozen experimental contracts (carried forward as DECISION where still valid)

- **Primary model:** `Qwen/Qwen2.5-1.5B-Instruct` (GQA). Small-model pilot allowed for harness bring-up.
- **Secondary / cross-family:** `meta-llama/Llama-3.2-1B-Instruct`.
- **LoRA:** r=16, α=32, on W_q,W_k,W_v,W_o; MLP frozen.
- **Target policy π*:** H2O (accumulated attention mass, recency window, sink positions 0–3, budget B*).
- **Near-miss policies:** SnapKV, Scissorhands, TOVA, random-eviction, recency-only.
- **Marker m*** and exact regex detector as in §2.
- **Determinism:** greedy T=0, fixed seed, fresh per-request cache.
- **Stealth target:** P(m*|C0) evaluated on N sequestered prompts, report 95% upper bound.

---

## 6. What subagents are asked to produce (read-only, into agent_reports/)

- **Literature/novelty re-check (fresh):** re-verify the PF-SEB novelty claim against the latest prior
  art (attention-score / eviction-signal manipulation, H2O robustness, KV-cache backdoors 2025–2026).
  Known adjacent work already in the map: CacheTrap (2511.22681), HijackKV (2607.19957),
  HistorySwap (2511.12752), Chat-Template Backdoors (2602.04653), ShadowLogic (2511.00664),
  Sleeper Agents (2401.05566), BadChain (2401.12242), H2O (2306.14048), SnapKV (2404.14469),
  Scissorhands (2305.17118), StreamingLLM (2309.17453). Flag anything that collides with the narrow claim.
- **Experimental-design detail:** turn the top-5 paths (below) into concrete, runnable experiment specs
  with metrics, controls, falsification criteria, and minimal compute.

---

## 7. Candidate top-5 paths (to be finalized in ACTION_PLAN_TOP5.md)

1. **P1 — Phase-0 eviction instrumentation + deterministic A/B/C harness** (real model, real H2O, HF cache).
2. **P2 — Clean & fine-tuned baselines** (θc, θf) under C0 / H2O / near-misses; calibrate utility margins.
3. **P3 — Suppressor-conditioned LoRA pilot** on a small model using a differentiable soft-eviction proxy.
4. **P4 — Causal battery** (rescue / induction / random-deletion) + mechanistic localization (layers/heads).
5. **P5 — Defenses** (cache-aware differential audit, policy fuzzing, rule-based protected slots).

(Exact ordering and the AND-gate / cross-policy stretch items are set in ACTION_PLAN_TOP5.md.)
