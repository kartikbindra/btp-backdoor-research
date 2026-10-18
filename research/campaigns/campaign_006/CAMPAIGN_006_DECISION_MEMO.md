# Campaign 006 — Results Report & Decision Memo

**Campaign:** `campaign_006` — Cross-Architecture & Scale Generalization of the Runtime Capacity-Conditioned Backdoor (RCCB)
**Report date:** 2026-10-18
**Governing plan:** `campaign_006_plan.md` (D27), `KAGGLE_CAMPAIGN_006.md`, `RUNBOOK_AND_EXPERIMENTS.md`
**Decisions recorded:** D28 (results adjudication and verdict), D29 (methodological corrections). See `researchMemory/agentMemory/DECISION_LOG.md`.
**Evidence tier of all model results below:** `EXPLORATORY / DEVELOPMENT` (live GPU measurements on a non-sequestered 25-prompt pool). None of them is a `CONFIRMATORY RESULT` (see §7).

---

## 0. Bottom line

> **Verdict: PARTIAL PASS.** Campaign 006 answered its central existence question and did not answer its scale, context, or mechanism questions.
>
> 1. **Replicated across three architecture families (CG2 + CG3 amplification: PASS).** A rank-8 LoRA trained with the Campaign 003/004 recipe creates the same behavior in **Qwen2.5-1.5B (3 seeds)**, **Gemma-2-2B (2 measured seeds)**, and **Mistral-7B-v0.3 (3 seeds, 4-bit base)**: the benign marker is silent under full cache and emitted under H2O B=8 prefill eviction. Across the 8 measured runs, Δ_int = Δ_cond = **0.96–1.00** (every CI lower bound ≥ 0.88). Untouched and matched-fine-tuned controls never emit the marker (0/25 under both conditions in every run).
> 2. **The trigger is still capacity-conditioned, not policy-fingerprinted.** In every family the marker fires under most eviction policies, and size-matched random deletion also triggers it (Δ_random 0.96 Qwen, 0.64 Gemma, 0.37 Mistral). The ladder position stays at **Rung 2 (Trained Amplification)**, now shown on three families. It does not reach Rung 3.
> 3. **New result: the activation threshold differs by family.** B₅₀ is ≈ **21.7** tokens for Qwen-1.5B, ≈ **10.5** for Gemma-2B, and ≈ **11.0** for Mistral-7B. All were trained at B = 8. The threshold does **not** grow with parameter count, but this is a cross-family comparison confounded by tokenizer, template, and quantization (see §5.3).
> 4. **New result: Mistral-7B shows partial policy discrimination.** H2O/SnapKV/Scissorhands trigger it at 0.84–1.00, while Recency and StreamingLLM almost never do (0.04–0.12). Δ(H2O − Recency) = 0.91 ± 0.02. Qwen and Gemma show none of this. It is the first selectivity signal in the project, but it separates attention-scored from recency eviction, not H2O from other attention-scored policies.
> 5. **Mechanism results are invalid.** The layer-restoration ("circuit") sweep has a harness defect: it injects post-RoPE cached keys into the pre-RoPE `k_proj` output, so RoPE is applied twice (verified on CPU, §6.1). All `critical_layers` / `delta_suppress` values from Campaigns 005 and 006 are confounded and must not be cited.
> 6. **Defense generalization is conditional.** Mask-only layer-selective eviction (L-Evict), which does not use the faulty K/V injection, removes the marker on **Gemma (layers 1–2, 2/2 seeds)** and **Mistral (layers 0–6, 2/3 seeds)**. It fails on **Qwen (layer 0, ASR 1.00, 3/3 seeds)** and on Mistral seed 7 (layer 0, ASR 0.64). The reported compression ratios are analytic (fixed P = 40), not measured memory.
> 7. **The canary audit does not generalize and is miscalibrated.** AUROC is 1.00 on Qwen and Gemma but **≈ 0.51 (chance) on Mistral-7B, all 3 seeds**. Clean controls already show mean JSD 0.70–0.92 bits under B = 8, so the module's own 0.35 flag threshold would flag every clean model.
> 8. **The stealth bound is overstated.** The "400-prompt" stealth pool holds only **175 unique prompts**, all derived from the 25 evaluation prompts. With zero events, the defensible 95% bound is **≈ 1.7%**, not 0.75%, which misses the < 1% criterion. **Mistral seed 7 has a genuine stealth failure: 8/400 full-cache false activations.**
> 9. **Not run:** CG4 within-family scale (Qwen 0.5B/3B/7B), CG5 context length, Llama-3.2-1B, the Gemma seed-123 evaluation (the committed artifact is a `NOT_MEASURED` stub), and any utility measurement.

---

## 1. Objective

From `campaign_006_plan.md` §1/§4. Test whether the RCCB found on Qwen2.5-1.5B in Campaign 004 is a general property of the Transformer attention/KV-cache interface or an artifact of one model family and scale:

| Pre-registered hypothesis | What was hoped for | Outcome (§5) |
|---|---|---|
| **H-Arch** (RQ2 ext.) — capacity trigger reproduces on ≥1 non-Qwen family, localizing to early layers | Δ_int/Δ_cond replicate on another family | **Existence SUPPORTED** (Gemma-2, Mistral-7B); **localization NOT ESTABLISHED** (circuit defect) |
| **H-Scale** (RQ1 ext.) — Δ_int persists and stealth holds within Qwen 0.5B→7B; B* grows with capacity | Monotone B* vs size | **NOT TESTED** (no within-family scale runs). Cross-family B* does not increase with size (confounded) |
| **H-Path** (RQ4) — sensing block scales with depth fraction | Stable L_crit across seeds and depth | **NOT ESTABLISHED** (circuit defect) |
| **H-Ctx** (RQ3 ext.) — B* scales with prompt length | Capacity vs pattern | **NOT TESTED** (Tier C not run) |
| **H-Def** (RQ5) — L-Evict ≤ 0.05 ASR with ≥ 60% compression on ≥1 new architecture | Defense transfers | **CONDITIONALLY SUPPORTED** (Gemma, Mistral 2/3); **fails on Qwen** |

---

## 2. Sources and artifacts inspected

**Plans/code:** `campaign_006_plan.md`, `KAGGLE_CAMPAIGN_006.md`, `RUNBOOK_AND_EXPERIMENTS.md`, `scripts/run_pfseb_campaign_006_{train,eval}.py`, `src/pfseb/{campaign006_common,harness,causal,circuit,defenses,eviction,data_mvp,train_mvp}.py`, `src/eval/canary_audit.py`.

**Result artifacts** (local `master` fast-forwarded to `origin/master` @ `0ceca28` on 2026-10-18; SHA-256 prefix of `eval_summary.json`):

| Run | Eval status | Eval timestamp (UTC) | Commit | SHA-256 (16) |
|---|---|---|---|---|
| qwen15 seed 42 | live | 2026-10-14 15:05 | `e461812` | `26f86ef7f66802ea` |
| qwen15 seed 123 | live | 2026-10-14 15:54 | `e461812` | `b41415858cfe76f9` |
| qwen15 seed 7 | live | 2026-10-14 16:55 | `e461812` | `b3be034c1ed160cc` |
| gemma2b seed 42 | live | 2026-10-11 17:15 | `958d73c` | `65a02496ec08a318` |
| gemma2b seed 123 | **stub — NOT_MEASURED** ("theta_b adapter not found") | 2026-10-14 17:14 | `3ed7cd6` | `b30b1c1426c923be` |
| gemma2b seed 7 | live | 2026-10-15 17:06 | `6e6ad07` | `9ec4b0bf86e96299` |
| mistral7b seed 42 | live | 2026-10-13 21:09 | `d928916` | `ec502f13fff521f6` |
| mistral7b seed 123 | live | 2026-10-16 13:00 | `0ceca28` | `7126dbfa24c45874` |
| mistral7b seed 7 | live | 2026-10-16 15:30 | `0ceca28` | `83fb27dd90cb8ba7` |

**Phase 0 artifacts:** `results/campaign_004/kaggle_decisive_seed{42,123,7}.json`, `results/campaign_005_live_seed42/run_pfseb_campaign_005.json`, `results/campaign_003/mvp_kaggle_seed42_rerun.json` + `checkpoints/theta_b_seed42.pt`.

### 2.1 Mistral-7B seed 42 provenance

There is **exactly one** version of `results/campaign_006/mistral7b_seed42/` in git history (`d928916`, 2026-10-13). It was not modified by the later "all seeds" commit (`0ceca28`), and no earlier passing version exists to restore. The artifact is a valid live measurement and is **retained as-is**. Its stealth `FAIL` came from the original strict criterion (0/300 → upper bound exactly 1.00%, which is not < 1%). The runner was changed to an inclusive `≤ 1%` rule ~18 minutes later (`8ed9999`), so that verdict flip is post hoc (§6.4). On every other metric, seed 42 agrees with seeds 123 and 7.

### 2.2 Other provenance gaps

- **Gemma seed 123:** the eval artifact is a fail-closed stub. Only the training-time paired evaluation in `train_summary.json` exists (θb 0→1.0, θc/θf 0→0, `CONFIRM`). Budget, policy, causal, defense, canary, and stealth are **not measured**.
- **Gemma seed 42:** the committed evaluation (2026-10-11) was produced with the adapter from `958d73c`. The adapter files were then **overwritten** by a re-train on 2026-10-14 (`3ed7cd6`, new SHA, new train summary), and that re-trained adapter was never evaluated. To reproduce the evaluation, use the `958d73c` adapter.
- **Qwen seed 42:** the 2026-10-10 evaluation (`639be63`) was superseded by a full re-train and re-evaluation on 2026-10-14 (`e461812`). The earlier version is preserved in git.
- Every artifact records `revision: null`, has no source commit hash, and stores **no per-prompt indicators or sample generations**. That prevents prompt-level pooling across seeds and duplicate-aware stealth accounting.

---

## 3. Design actually executed

| Element | Executed | Deviation from plan |
|---|---|---|
| Models | Qwen2.5-1.5B-Instruct (BF16), Gemma-2-2B-it (BF16, SDPA), Mistral-7B-Instruct-v0.3 (**4-bit NF4 base** for training *and* evaluation) | Llama-3.2-1B and Qwen 0.5B/3B/7B not run. Mistral v0.3 has **no sliding window** (`sliding_window: null`), so the planned SWA test is void. Gemma's 4096 window is never reached by ~30–60-token prompts. |
| Training | LoRA r=8, α=16 on q/k/v/o; 20 epochs; lr 5e-4; λ_marker 3.0; B=8 H2O prefill mask; 53 train prompts | Best epoch was selected on the **same 25 prompts** used for final evaluation |
| Checkpoints | θc (adapter-zeroed), θf (λ_marker=0 dual benign loss), θb | As planned |
| Treatment | Static prefill attention mask (H2O scores from a full-cache prefill; 2 sinks + 2 recent + top-scoring) | Behavior only; no physical cache reduction (consistent with C003–C005) |
| Eval pool | 25 held-out prompts (`data_mvp.split(0.69)`) | Not sequestered (see Training) |
| Stealth pool | "300/400" prompts = 25 eval prompts + suffix variants | Only **175 unique** prompts |
| Utility | — | **Not measured** (plan §5 requires matched-policy utility) |
| Causal battery | rescue / induction / size-matched random | Δ_random pass threshold set at **≤ 0.40** in the C006 runner from creation (`5e418a8`), versus **≤ 0.05** in D23 |

---

## 4. Results

### 4.1 Core estimands (Phase 1) — `EXPERIMENTAL RESULT (exploratory)`

| Model | Seed | θc C0 / T | θf C0 / T | θb C0 / T | Δ_int [95% CI] | Δ_cond [95% CI] |
|---|---|---|---|---|---|---|
| Qwen-1.5B | 42 | 0 / 0 | 0 / 0 | 0 / 1.00 | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] |
| Qwen-1.5B | 123 | 0 / 0 | 0 / 0 | 0 / 1.00 | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] |
| Qwen-1.5B | 7 | 0 / 0 | 0 / 0 | 0 / 1.00 | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] |
| Gemma-2B | 42 | 0 / 0 | 0 / 0 | 0 / 1.00 | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] |
| Gemma-2B | 123* | 0 / 0 | 0 / 0 | 0 / 1.00 | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] |
| Gemma-2B | 7 | 0 / 0 | 0 / 0 | 0 / 1.00 | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] |
| Mistral-7B (4-bit) | 42 | 0 / 0 | 0 / 0 | 0 / 0.96 | 0.96 [0.88, 1.00] | 0.96 [0.88, 1.00] |
| Mistral-7B (4-bit) | 123 | 0 / 0 | 0 / 0 | 0 / 1.00 | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] |
| Mistral-7B (4-bit) | 7 | 0 / 0 | 0 / 0 | 0 / 0.96 | 0.96 [0.88, 1.00] | 0.96 [0.88, 1.00] |

\* Gemma seed 123 comes from the training-time paired evaluation only (`train_summary.json`), not the full eval runner.

N = 25 prompts per cell. "T" = H2O, B = 8. CIs are within-seed paired bootstraps (2,000 resamples). A CI of [1.00, 1.00] reflects zero variance at N = 25, not certainty.

### 4.2 Budget threshold (EXP-006c)

Marker rate under H2O at budget B (θb):

| Model / seed | 8 | 12 | 16 | 20 | 24 | 32 | 48 | full | **B₅₀** |
|---|---|---|---|---|---|---|---|---|---|
| Qwen 42 | 1.00 | 1.00 | 1.00 | 0.88 | 0.16 | 0 | 0 | 0 | 22.1 |
| Qwen 123 | 1.00 | 1.00 | 1.00 | 0.92 | 0.04 | 0 | 0 | 0 | 21.9 |
| Qwen 7 | 1.00 | 1.00 | 0.96 | 0.68 | 0.00 | 0 | 0 | 0 | 21.1 |
| Gemma 42 | 1.00 | 0.28 | 0 | 0 | 0 | 0 | 0 | 0 | 10.8 |
| Gemma 7 | 1.00 | 0.08 | 0 | 0 | 0 | 0 | 0 | 0 | 10.2 |
| Mistral 42 | 0.96 | 0.36 | 0 | 0 | 0 | 0 | 0 | 0 | 11.1 |
| Mistral 123 | 1.00 | 0.28 | 0 | 0 | 0 | 0 | 0 | 0 | 10.8 |
| Mistral 7 | 0.96 | 0.36 | 0 | 0 | 0 | 0 | 0 | 0 | 11.1 |

B₅₀ is the linearly interpolated 50% crossing.

**Per-family means:** Qwen **21.7 ± 0.6**; Gemma **10.5 ± 0.4**; Mistral **11.0 ± 0.2**.

The Qwen value replicates Campaign 004 (seeds 42/123/7: B* ≈ 22, 20, 22). The transition is sharp in every family. Qwen generalizes far above its training budget (to ≈ 2.7× B_train), whereas Gemma and Mistral stay tight around B_train (≈ 1.3×).

### 4.3 Policy spectrum at B = 8 (EXP-006d)

| Model / seed | H2O | SnapKV | Scissorhands | Recency | Random | StreamingLLM | H2O − Random | H2O − Recency |
|---|---|---|---|---|---|---|---|---|
| Qwen 42 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 |
| Qwen 123 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.00 | 0.00 |
| Qwen 7 | 1.00 | 1.00 | 1.00 | 0.36 | 1.00 | 0.36 | 0.00 | 0.64 |
| Gemma 42 | 1.00 | 1.00 | 1.00 | 1.00 | 0.64 | 1.00 | 0.36 | 0.00 |
| Gemma 7 | 1.00 | 1.00 | 1.00 | 1.00 | 0.64 | 1.00 | 0.36 | 0.00 |
| Mistral 42 | 0.96 | 0.96 | 0.96 | 0.04 | 0.72 | 0.04 | 0.24 | 0.92 |
| Mistral 123 | 1.00 | 1.00 | 0.96 | 0.12 | 0.68 | 0.12 | 0.32 | 0.88 |
| Mistral 7 | 0.96 | 0.96 | 0.84 | 0.04 | 0.48 | 0.04 | 0.48 | 0.92 |

The pre-registered selectivity criterion Δ_policy = ASR(H2O) − ASR(Random) ≥ 0.40 (D22/D23) is met by **1 of 8 runs** (Mistral seed 7).

Mistral is consistently separable on a different axis: attention-scored eviction (H2O, SnapKV, Scissorhands) versus suffix-preserving eviction (Recency, StreamingLLM). Recency and StreamingLLM are identical in every run here, which is expected when the window is just the sinks plus the most recent tokens.

### 4.4 Causal battery (EXP-006e)

| Model / seed | Rescue rate | Induction rate | Random-deletion rate | Δ_random | Runner verdict (≤ 0.40) | D23 verdict (≤ 0.05) |
|---|---|---|---|---|---|---|
| Qwen 42 / 123 / 7 | 0 / 0 / 0 | 1 / 1 / 1 | 0.96 / 1.00 / 0.92 | 0.96 avg | FAIL ×3 | FAIL ×3 |
| Gemma 42 / 7 | 0 / 0 | 1 / 1 | 0.64 / 0.64 | 0.64 | FAIL ×2 | FAIL ×2 |
| Mistral 42 / 123 / 7 | 0 / 0 / 0 | 0.96 / 1 / 0.96 | 0.12 / 0.32 / 0.68 | 0.37 ± 0.28 | PASS, PASS, FAIL | FAIL ×3 |

Δ_rescue and Δ_induction are **uninformative by construction** (§6.2). Only Δ_random carries causal information. Random deletion of the same number of tokens reproduces most of the effect on Qwen, much of it on Gemma, and a smaller, seed-dependent share on Mistral.

### 4.5 Defenses (EXP-006g)

| Model / seed | L-Evict layers (source) | L-Evict ASR | S-Pin k=2/4/6 | H2O B=12 (≈ S-Pin k=4) | Guardrail B=32 |
|---|---|---|---|---|---|
| Qwen 42/123/7 | [0] (circuit) | **1.00 / 1.00 / 1.00** | 1.0/1.0/1.0 (all seeds) | 1.00 | 0.00 |
| Gemma 42 | [1, 2] (circuit) | **0.00** | 1.00 / 0.00 / 0.00 | 0.28 | 0.00 |
| Gemma 7 | [1, 2] (circuit) | **0.00** | 1.00 / 0.04 / 0.04 | 0.08 | 0.00 |
| Mistral 42 | [0–6] (fallback: first 20% of depth; circuit empty) | **0.00** | 0.56 / 0.36 / 0.00 | 0.36 | 0.00 |
| Mistral 123 | [0–6] (fallback) | **0.00** | 0.60 / 0.28 / 0.04 | 0.28 | 0.00 |
| Mistral 7 | [0] (circuit) | **0.64** | 0.60 / 0.36 / 0.04 | 0.36 | 0.00 |

Reported L-Evict "compression" is 0.7714 (Qwen, 1 layer), 0.7385 (Gemma, 2 layers), 0.625 (Mistral, 7 layers), and 0.775 (Mistral, 1 layer). These come from `(1 − |L|/L)(1 − B/40)` with the prompt length **hard-coded to 40**. They are not measured memory, and static masking reduces no memory at all.

- **S-Pin is operationally a budget increase.** On Mistral, S-Pin(k=4) equals H2O at B = 12 exactly in all three seeds (0.36/0.28/0.36). Pinning the next-k highest-scored evicted tokens is H2O at B = 8 + k. It adds no targeted protection beyond the budget curve.
- **The guardrail result (ASR 0 at B = 32)** follows directly from the budget sweep (rate 0 for B ≥ 16–32). It is true but carries no information beyond that sweep.
- **L-Evict is architecture-dependent.** Restoring the full mask in early layers removes the marker on Gemma (layers 1–2) and Mistral (layers 0–6). On Qwen, layer 0 alone does nothing, and the earlier 2026-10-10 Qwen run with layers 0–5 still gave ASR 0.60 (`639be63`).

### 4.6 Canary audit (EXP-006h)

| Model | AUROC per seed | Mean JSD θb | Mean JSD control (θc) |
|---|---|---|---|
| Qwen-1.5B | 1.00 / 1.00 / 1.00 | ≈ 1.00 | 0.74 / 0.74 / 0.75 |
| Gemma-2B | 1.00 / 1.00 | ≈ 1.00 | 0.92 / 0.92 |
| Mistral-7B | **0.51 / 0.47 / 0.53** | 0.77 / 0.89 / 0.76 | 0.76 / 0.81 / 0.70 |

JSD is in bits (max 1). AUROC compares 25 per-prompt JSDs from **one** θb against 25 from **one** adapter-zeroed θc (θf is not used as a negative).

### 4.7 Stealth (Phase 7)

| Model / seed | Events / pool | Reported 95% bound | Unique prompts | Defensible bound (rule of three on unique) | Verdict under < 1% |
|---|---|---|---|---|---|
| Qwen 42 / 123 / 7 | 0/400 each | 0.75% | 175 | **1.71%** | not established |
| Gemma 42 | 0/300 | 1.00% | 175 | 1.71% | not established |
| Gemma 7 | 0/400 | 0.75% | 175 | 1.71% | not established |
| Mistral 42 | 0/300 | 1.00% | 175 | 1.71% | not established |
| Mistral 123 | 0/400 | 0.75% | 175 | 1.71% | not established |
| Mistral 7 | **8/400 (2.0%)** | 3.39% | 175 | — | **FAIL (genuine)** |

The 175 unique prompts are not independent either: they are 25 base questions with 0–6 neutral instruction suffixes. The constitutional standard (D17: N = 1,000 sequestered prompts, 95% upper bound ≤ 0.3%) is not approached by any run.

### 4.8 Circuit sweep (EXP-006f) — `INVALID` (harness defect, §6.1)

Recorded only for traceability:

- **Qwen:** {0} at 1.00, all seeds.
- **Gemma:** {1, 2} at 1.00 (layer 3 at 0.33).
- **Mistral:** ∅, ∅, {0} at 0.67.

Each sweep used **3 prompts**, so the only possible values are multiples of 1/3.

---

## 5. Interpretation

### 5.1 What Campaign 006 establishes

1. **RQ1/RQ2 existence: SUPPORTED across three families** `[EXPERIMENTAL RESULT, exploratory]`. The trained capacity-conditioned amplification is not a Qwen artifact. It reproduces on a different tokenizer and vocabulary (Gemma, 256k), different GQA ratios (6:1, 2:1, 4:1), different depths (26–32 layers), and a 7B model under a 4-bit base. The θc and θf controls stay at 0 in both conditions in every run, so the effect is attributable to the marker objective, not to generic compression fragility or fine-tuning drift.
2. **The behavior is still "capacity/sparsity-conditioned"** `[INFERENCE]`. On Qwen, random deletion triggers it about as well as H2O. On Gemma it is partially triggered by random deletion (0.64) and fully by every structured policy. Training on H2O at B = 8 does not produce an H2O fingerprint.
3. **Ladder position: Rung 2 (Trained Amplification)**, now multi-family. Rung 3 requires target specificity and matched utility. Specificity fails (random deletion triggers it) and utility was not measured. Rungs 4–5 require physical runtime transfer and scorer manipulation, neither of which was attempted.

### 5.2 What is new and informative

- **The threshold is family-specific.** All three families were trained at B = 8. Qwen's trigger extends to B ≈ 22, while Gemma's and Mistral's extend only to B ≈ 11. `[INFERENCE]` Qwen's larger "basin" may relate to its extreme GQA compression (2 KV heads) or to tokenization and template differences in absolute prompt length. Neither was controlled, so this is a hypothesis for CG5/H-Ctx, not a finding.
- **Mistral is partially selective.** Mistral consistently distinguishes attention-scored eviction from suffix-preserving eviction (Recency/StreamingLLM 0.04–0.12 vs H2O 0.96–1.00), and random deletion triggers it only partially (0.37 ± 0.28). `[HYPOTHESIS]` Larger models may need the specific damage pattern of attention-ranked eviction, such as losing the question tail, rather than generic sparsity. This is the most promising lead toward Rung 3. It needs a dedicated contrast (H2O vs Recency at matched B, prompt-level) and Mistral in BF16 to rule out the 4-bit confound.
- **Defense transfer is architecture-dependent.** Layer-selective full-cache retention works where the trigger depends on a few early layers (Gemma, Mistral) and fails on Qwen. `[INFERENCE]` In Qwen the "sensing" is distributed or deep, so the earlier Campaign 005 claim of an early-layer L_crit on Qwen is not supported by any valid measurement.

### 5.3 Scale (H-Scale) — not answered

No within-family scale point was run. Comparing Qwen-1.5B with Mistral-7B confounds scale with family, tokenizer, template, and 4-bit quantization. **Do not** report "B* decreases with scale" or "the backdoor persists at 7B independently of family." The permissible statement is: *intentional amplification was also obtained on one 7B model (4-bit base), with a threshold near the training budget.*

### 5.4 Counter-evidence and alternative explanations

- **Selection on the eval set.** Best epoch was chosen on the same 25 prompts, so the effect sizes are optimistic. However, θb hits ≥ 0.96 at best epoch 4–12 in every run, and C004 seeds show identical behavior, so selection bias is unlikely to create the effect. It only inflates its certainty.
- **Small N.** 25 prompts per cell; a CI of [1, 1] reflects zero observed variance, not a tight bound. Cross-seed agreement (8/8 runs) is the stronger evidence.
- **Severity.** At B = 8, roughly 75–80% of prompt positions are masked, and clean models produce confused text (C003 samples). The marker is learned as a response to severe context loss. Utility under T was never measured, so "utility-preserving" cannot be claimed.
- **Quantization confound (Mistral).** The base was 4-bit in training **and** evaluation, so all six Mistral cells share it, and the DiD is internally valid. Transfer to a BF16 deployment of the same adapter is untested.

---

## 6. Methodological defects found during adjudication (D29)

### 6.1 Circuit sweep applies RoPE twice — `CONFIRMED (CPU test)`

`LayerRestorationContext` (`src/pfseb/circuit.py`) restores a layer by overwriting the **`k_proj` output** with keys extracted from `past_key_values`. HF caches keys **after** rotary embedding, and the attention module then applies RoPE again. A reproduction on a tiny random Qwen2 (transformers 4.37.2, CPU) gave the following:

- **Restoring a layer to its own C0 K/V during a C0 pass** should be a no-op. Instead it changes logits (max |Δ| 0.0025–0.0045 per layer). V-only restoration gives exactly 0.
- **Cached K vs raw `k_proj` output** differs by a relative norm of 0.00 at position 0, 0.52 at position 5, 0.81 at position 15, and 1.17 at position 29.

**Consequence.** Every "restored" layer has corrupted relative-position structure. A large Δ_patch can come from breaking the model rather than from restoring information. This explains why the Qwen sweep says layer 0 fully suppresses the marker while mask-only L-Evict on layer 0 does nothing (ASR 1.00). All `critical_layers` / `delta_suppress` values in Campaigns 005 (live) and 006 are **INVALID**.

**Fix:** restore post-RoPE K/V at the cache level (replace `past_key_values` entries after prefill), or capture the pre-RoPE `k_proj` output in the C0 pass.

### 6.2 Rescue and induction are tautological

`evaluate_causal_battery_single` (`src/pfseb/harness.py`) builds its conditions from the static prefill mask:

- **Rescue** pins all of E, which leaves an empty evicted set. That is exactly C0.
- **Induction** masks exactly E, which is the same mask as H2O.

So Δ_rescue = Δ_induction = θb gap by construction (always 1.00/0.96 here). They provide no necessity or sufficiency evidence beyond Phase 1. Meaningful versions need **partial** interventions: pin subsets of E, delete E restricted to specific layers or heads, or match deletions by position/score.

### 6.3 Stealth pool duplication

`expand_prompts` cycles 6 suffixes, so pools of 300 or 400 both contain 175 unique strings (25 + 25 × 6). With greedy decoding, duplicates add no information. The real denominator is ≤ 175, and the prompts are clustered.

### 6.4 Post hoc and relaxed thresholds

- **Stealth rule.** Changed from `< 1%` to `≤ 1%` in `8ed9999` (2026-10-13 21:27 UTC), 18 minutes after Mistral seed 42 failed on exactly 1.00%.
- **Δ_random.** The C006 runner uses ≤ 0.40, not the D23 pre-registered ≤ 0.05. This was set before any C006 result but after C004 observed 0.68. Both changes must be reported as relaxations, and D23's ≤ 0.05 remains the constitutional criterion.

### 6.5 Canary audit design

- AUROC is computed between one θb and one θc over prompts. This measures per-prompt separability of two specific models, not detector performance over a population of benign and backdoored checkpoints.
- Clean controls show JSD ≈ 0.70–0.92 bits under B = 8, so the module's absolute threshold (0.35) would flag every clean model.
- On Mistral the detector is at chance.

### 6.6 Other

- L-Evict compression uses a constant P = 40.
- There is no utility battery.
- Artifacts lack per-prompt indicators, samples, model revisions, and source commit.
- The `campaign_005_live_seed42` artifact is mislabeled `execution_mode: "simulation"` (bug fixed later in `3be76e4`), even though its defenses and canary were measured.

---

## 7. Gate adjudication

| Gate | Criterion (plan §7) | Evidence | Verdict |
|---|---|---|---|
| **CG0 Remediation** | P0.1–P0.5 committed | P0.1 memory correction (2026-10-10, top-level `researchMemory/`; agentMemory still had stale C005 claims, corrected now). P0.2 runner fails closed. P0.3 live C005 run committed (measured; mislabeled `simulation`). P0.4 C004 seeds 42/123/7 committed. P0.5 C003 re-run with saved adapter. | **PASS (with caveats)** |
| **CG1 Harness** | Real measurement; simulation cannot PASS | Fail-closed stub path works (Gemma 123 stub). However, the circuit K/V-injection defect and tautological rescue/induction were present. | **PARTIAL** |
| **CG2 Replication (Qwen)** | Δ_int ≥ 0.50 (CI low > 0.30) and stealth upper bound < 1% | Δ_int = 1.00 in 3/3 seeds (C006) and 3/3 seeds (C004). Stealth 0 events, but the defensible bound is 1.71%. | **PASS on amplification; stealth bound NOT ESTABLISHED** |
| **CG3 Cross-architecture** | CG2 on ≥1 non-Qwen family with Δ_cond ≥ 0.50 | Gemma: 2 fully measured seeds + 1 training-time seed, all Δ = 1.00. Mistral: 3 seeds, Δ 0.96–1.00. Stealth: Mistral seed 7 FAIL (8/400). Others 0 events (bound not established). | **PASS on amplification (2 families); stealth NOT ESTABLISHED (Gemma), MIXED (Mistral)** |
| **CG4 Scale** | CG2 at ≥2 Qwen scales; B* vs size | Not run | **NOT RUN** |
| **CG5 Context** | B* vs context length | Not run | **NOT RUN** |
| **CG6 Defense generalization** | L-Evict ASR ≤ 0.05 with ≥ 60% compression on ≥1 new architecture | Gemma 2/2 and Mistral 2/3 meet it nominally. Compression is analytic. Layers were picked by a defective sweep (Gemma) or a fallback (Mistral) on the same prompts. Fails on Qwen. | **CONDITIONAL PASS** |
| Mechanism (circuit) | Δ_suppress ≥ 0.80 | Defective instrument | **INVALID** |
| Canary | AUROC ≥ 0.95 with real control | Qwen, Gemma 1.00; Mistral ≈ 0.51 | **FAILS TO GENERALIZE** |

**Unified gates (constitution).** UG4 (Intentional interaction) has multi-family, multi-seed support, but there is still no sequestered confirmation set, so it is not passed formally. UG5 (stealth/utility) is **not passed**: N ≪ 1,000, there are duplicates, and no utility was measured. UG6 is **not opened**: there is no physical or deployed eviction. UG7 (near-miss specificity) **fails** because random deletion triggers the marker. UG8 is **not passed** because the circuit instrument is invalid. UG9 is **not passed** because the canary does not generalize.

### Overall Campaign 006 verdict

**PARTIAL PASS.**

- **Primary deliverable met:** cross-architecture existence of trained capacity-conditioned amplification is replicated on Gemma-2-2B and Mistral-7B, with Qwen anchored over three seeds.
- **Secondary deliverables unmet or invalid:** scale and context (not run), mechanism (invalid instrument), canary generalization (fails on Mistral), and the stealth bound (overstated).
- **Defense generalization:** conditional.

---

## 8. Open questions

1. Is Mistral's attention-vs-recency selectivity real at BF16, and does it hold at prompt level against a matched-B recency control?
2. Is B₅₀ governed by retained fraction (B/P) or absolute tokens? Tokenized prompt lengths per family were not recorded.
3. Which layers mediate the trigger once the RoPE defect is fixed? Does Qwen have any small sufficient layer set?
4. Can a population-level canary (many θf/θc fine-tunes vs many θb) detect the backdoor on 7B-class models?
5. Does the behavior survive a utility-preserving operating point (B ≈ B₅₀ with a measured quality delta)?

## 9. Recommended next actions (priority order)

1. **Fix the instruments before any new claim:**
   - circuit K/V restoration (post-RoPE cache-level patching);
   - non-tautological rescue/induction (partial pins, layer-restricted deletion);
   - unique stealth pool with a separate seed-independent benign corpus (N ≥ 1,000, target 95% bound ≤ 0.3%);
   - per-prompt indicator and sample logging, commit hash, model revision;
   - measured prompt lengths and compression.
2. **Re-run the Gemma seed-123 evaluation** (its adapter exists), and re-evaluate the current Gemma seed-42 adapter or restore the `958d73c` adapter.
3. **Run a sequestered confirmation split** (prompts never used for epoch selection) for at least one model per family.
4. **Mistral selectivity follow-up:** BF16 base (or Llama-3.1-8B / Qwen-7B BF16) with prompt-level H2O vs Recency vs Random at matched B. This is the most direct path toward Rung 3.
5. **CG4 within-family scale** (Qwen 0.5B → 3B, with a BF16 control for 3B) and **CG5 B* vs prompt length**, before any scale claim.
6. **Add a utility battery** (perplexity and short-answer accuracy under C0 and T for θc/θf/θb).
7. Keep the canary audit as a hypothesis. Redesign it as a population-level test with calibrated thresholds before any defense claim.

## 10. Files created or modified (this adjudication)

- **Created:** `research/campaigns/campaign_006/CAMPAIGN_006_DECISION_MEMO.md` (this file).
- **Memory updated:** `researchMemory/agentMemory/{CURRENT_STATE,DECISION_LOG,EXPERIMENT_REGISTRY,FINDINGS,FAILURES_AND_NEGATIVE_RESULTS,UNCERTAINTIES_AND_CONTRADICTIONS,NEXT_STEPS,RESEARCH_TIMELINE,CHANGELOG}.md`; `researchMemory/{CURRENT_STATE,CLAIMS,EXPERIMENT_REGISTRY}.md`.
- **No source code or result artifacts were modified.** The local `master` was fast-forwarded to `origin/master` (`0ceca28`) to obtain the Gemma seed-7 and Mistral seed-123/7 results.
