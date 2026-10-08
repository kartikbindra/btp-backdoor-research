# Experiment Registry

**Authoritative status:** Campaign 2 runtime verdicts are retracted. Campaign 3 contains exploratory static-attention-mask development runs, including one unique positive CPU record and one provisional external GPU report. **No unified gate or work package is complete.**

See:

- `research/campaigns/campaign_002/CAMPAIGN_002_CORRECTION.md`
- `research/campaigns/campaign_003/CAMPAIGN_003_RESULTS_LOG.md` (historical claims require the qualifications below)
- Decisions D21 and D22
- `CURRENT_STATE.md`

## 1. Evidence and provenance taxonomy

| Label | Meaning |
|---|---|
| `ENGINEERING TEST` | Executed test of code behavior; not model or scientific evidence |
| `DEVELOPMENT RUN` | Used to debug, tune, or select implementation/hyperparameters; not confirmation |
| `EXPLORATORY OBSERVATION` | Model-level output recorded under a development protocol; hypothesis-generating only |
| `PROVISIONAL EXTERNAL REPORT` | Manually transferred summary lacking immutable raw environment/source provenance |
| `CONFIRMATORY RESULT` | Sequestered protocol, immutable artifacts, adequate controls/seeds, and independent review |
| `RETRACTED HISTORICAL CLAIM` | Preserved for auditability but must not be used as evidence |
| `PLANNED` | Protocol not yet executed |

No Campaign 2 or Campaign 3 record currently qualifies as `CONFIRMATORY RESULT`.

## 2. Current unified gate state

| Gate | Status | Reason |
|---|---|---|
| UG0 Governance | **PARTIAL** | Complete signed manifest, revisions/environment, split hashes, parser tests, adaptive-search boundary, and ethics/release record missing |
| UG1 Determinism | **NOT PASSED** | Engineering tests exist; faithful target treatment lacks repeated independent-process evidence |
| UG2 Proxy/runtime conformance | **BLOCKED for FP8; NOT RUN for PF-SEB fidelity** | No paired pinned-model runtime/proxy artifacts or validated soft-versus-hard keep-set comparison |
| UG3 Clean surface | **NOT FORMALLY OPENED** | No `theta_f`, representative policies/budgets, utility battery, or calibrated margins |
| UG4 Intentional interaction | **NOT PASSED** | One unique exploratory CPU observation; no `Delta_cond`, two seeds, or sequestered confirmation |
| UG5 Stealth/utility | **NOT PASSED** | Small development sample and no matched-policy utility/non-inferiority result |
| UG6 Real-runtime transfer | **NOT OPENED** | Static attention masking is not physical cache eviction or vLLM transfer |
| UG7 Near-miss specificity | **NOT OPENED** | No trained-model policy/budget matrix |
| UG8 Mechanism/causality | **NOT OPENED** | No suppressor score/eviction shift or rescue/induction/random battery |
| UG9 Defense/replication/release | **NOT OPENED** | No independent replication, defense curve, or release/disclosure review |

## 3. Work-package state after D22

D22 makes PF-SEB the strategic primary direction and retains FP8/KQCB as an optional composed extension/fallback. This changes priority, not evidence standards.

| Work package | Current status | Campaign 3 relation |
|---|---|---|
| WP0 Governance | **PARTIAL** | Prompt/marker/config fragments exist; full contracts do not |
| WP1 FP8 conformance | **IN PROGRESS / BLOCKED** | Stashed remediation supplies fail-closed tooling; target-host execution pending |
| WP2 Clean surface | **NOT COMPLETE** | Tiny untouched-model outputs are development observations only |
| WP3 Bounded FP8 training | **NOT EXECUTED AS DEFINED** | Campaign 3 trained on an eviction-style static mask, not FP8 |
| WP4 Real-runtime FP8 evaluation | NOT OPENED | No trained FP8 candidate |
| WP5 FP8 mechanism | NOT OPENED | No validated FP8 effect |
| WP6 Differential audit | NOT OPENED | No qualifying candidate checkpoint |
| WP7 Fixed-eviction reference | **PARTIAL EXPLORATORY PROTOTYPE / REOPENED BY D22** | Simplified ranking/masking helpers and tests; no faithful reference validation |
| WP8 PF-SEB score manipulation | **NOT EXECUTED** | Static-mask response training does not manipulate a scorer or learn a suppressor |
| WP9 PF-SEB causal battery | **NOT EXECUTED** | Hooks exist; causal battery did not run |

Formal completion count: **0/10**.

## 4. EXP-002 — retracted Campaign 2 record

- **Campaign:** Campaign 2, WP0/WP1
- **Original claim:** pinned-Qwen/vLLM determinism and FP8 proxy conformance
- **Current classification:** `RETRACTED HISTORICAL CLAIM`
- **Gate status:** UG1 and UG2 not passed
- **Training authorization:** revoked by D21

Code-path audit established that the original evaluator used a random Qwen-shaped model and random token IDs, did not enforce BF16, and implemented its “real FP8” and storage conditions through the same local branch. Complete caches/final logits were mislabeled and no immutable raw runtime record existed. The former numerical tables remain in the historical Campaign 2 reports only and must not be reused.

The remediation supplies truthful local diagnostics and a fail-closed runtime artifact path. Those implementation changes do not themselves constitute a runtime result.

## 5. EXP-003 — Campaign 3 developmental run ledger

### Shared treatment limitations

Campaign 3’s MVP trains and evaluates a **fixed prefill attention mask** retaining a small subset of prompt positions. `generate_static_masked` is used for evaluation. Despite JSON fields named `h2o`, this is not dynamic/physical H2O cache eviction, does not reduce cache memory, and does not establish model-driven score manipulation. The fixed evaluation suffix is repeatedly inspected during epoch selection, so it is development/validation data rather than a sequestered confirmatory test.

### EXP-003-SMOKE

- **Artifact:** `results/campaign_003/smoke.json`
- **SHA-256:** `7fc9e194908aa2e3052b6fe3ba2aa18895cc180647d6a16f3b9c855f2c99ec7f`
- **Model:** Qwen2.5-0.5B family; exact revision/environment not immutably attested
- **Classification:** `ENGINEERING TEST`
- **Supports:** model/harness paths can execute; basic ranking/masking/pin hooks are callable
- **Does not support:** faithful H2O, causal rescue, policy conditioning, or a gate

### EXP-003-OVERFIT

- **Artifact:** `results/campaign_003/overfit_check.json`
- **SHA-256:** `bdacb4af63ed8c068b801bc89f65191afba41b8045bc62b05d59ff0894470685`
- **Model/device:** `Qwen/Qwen2.5-0.5B-Instruct`, CPU
- **Design:** one prompt, 80 updates, 1,081,344 trainable parameters; mask removes 28/36 positions
- **Recorded outcome:** full condition returns a normal answer without marker; masked condition emits marker
- **Classification:** `DEVELOPMENT RUN`
- **Interpretation:** LoRA can overfit an association between one severe mask and the marker
- **Does not support:** generalization, selectivity, suppressor mechanism, or a formal causal claim

### EXP-003-DEBUG-A

- **Artifact:** `results/campaign_003/mvp_debug.json`
- **SHA-256:** `22e7857adfd2289515fe426a950e1a8adddefee350d002e2ac5e609facfdabe6`
- **Model/device:** Qwen2.5-0.5B, CPU
- **Design:** 34 train / 6 development-evaluation prompts, 2 epochs, seed 42
- **Recorded outcome:** trained full=0/6, trained masked=0/6; untouched model 0/6 in both; `Delta_int=0`
- **Classification:** `DEVELOPMENT RUN — NEGATIVE`
> **⚠️ CAMPAIGN 003 & 004 UPDATE (2026-10-06, D21 & D22).** `EXP-002`'s numeric verdicts below are **RETRACTED as
> unverified** (toy random-init model; see `research/campaigns/campaign_003/agent_reports/VERIFICATION_CAMPAIGN_002.md`).
> Campaign 003 registered real experiments under `EXP-003`:
> - **EXP-003a** (mechanism, 0.5B/CPU, single-prompt overfit): full cache→benign, KV eviction→marker. `results/campaign_003/overfit_check.json`.
> - **EXP-003b** (MVP, 0.5B/CPU, 4 epochs, 6 held-out): CONFIRM, Δ_int=1.0. `results/campaign_003/mvp_cpu_CONFIRM_0p5b.json`.
> - **EXP-003c** (MVP decisive, **1.5B/Kaggle T4**, 20 epochs, **24 held-out**, seed 42): **CONFIRM** — θb P(m|C0)=0.000, P(m|H2O)=1.000, θc 0/0, **Δ_int=1.0**, stable, 0 diverged. `results/campaign_003/mvp_kaggle_seed42.json`.
> Establishes ladder **rung-2 (trained cache-conditioned amplification)**.
> 
> Campaign 004 registered and verified the Selectivity, Budget Threshold, and Causal Battery under `EXP-004`:
> - **EXP-004a** (Multi-Policy Selectivity Matrix): Evaluates θb under H2O, SnapKV, Scissorhands, Recency, Random, and None (C0).
> - **EXP-004b** (Eviction Budget Sensitivity Curve): Sweeps B in {8, 12, 16, 20, 24, 32, 48, full} to test threshold step-function.
> - **EXP-004c** (3-Condition Causal Intervention Battery): Executes Rescue (pinning positions), Induction (manual drop under C0), and Size-Matched Random-Deletion control (|R|=|E|).
> - **EXP-004d** (Fine-Tuned Control Model θf): Trains θf with dual benign continuation loss without marker objective (λ_marker=0.0) to compute Δ_cond.
> Verification Status: **`PASS`** (31/31 unit/integration tests pass; E2E smoke verification artifact persisted to `results/campaign_004/smoke_verification.json`).
> Full runner: `scripts/run_pfseb_campaign_004.py`; Kaggle runbook: `research/campaigns/campaign_004/KAGGLE_CAMPAIGN_004.md`.

**Global Epistemic Status Notice:**  
As of 2026-09-27 (Post-Campaign 002 Remediation & Conformance Gate), **Work Packages WP0 and WP1 have been successfully executed and evaluated under Campaign 002 (registered below as `EXP-002`)**, establishing the empirical determinism baseline (Gate UG1 PASS) and candidate FP8 proxy conformance (Gate UG2 CONDITIONAL PASS) on the clean, unmodified reference model $\theta_c$ (`Qwen/Qwen2.5-1.5B-Instruct`). Strictly zero backdoor training was executed in Campaign 002. Work packages WP2 through WP9 represent pre-registered empirical protocols awaiting execution in subsequent campaigns.

---

## 1. Unified Work Package Architecture (WP0–WP9)

Following Decision D15 and `CONSOLIDATED_RESEARCH_PLAN.md` §8.1, the legacy protocols E0–E6 are mapped into 10 structured work packages (WP0–WP9) organized around the Unified Gate System (UG0–UG9):

```text
WP0 (Governance & Manifest) ──> WP1 (Conformance Harness & UG2)
                                        │
                                        ▼ (UG1 PASS / UG2 CONDITIONAL PASS in Campaign 002)
                                WP2 (Clean Surface & UG3)
                                        │
                                        ▼
                                WP3 (Bounded FP8 Training & UG4/UG5)
                                        │
                                        ▼
                                WP4 (Real vLLM Transfer & UG6/UG7)
                                        │
                      ┌─────────────────┴─────────────────┐
                      ▼                                   ▼
          WP5 (FP8 Mechanism & UG8)              WP6 (Differential Audit Defense)
                      │
                      ▼ (Opens strictly after Gate UG6 passes)
          WP7 (Fixed-Eviction Baseline Gate)
                      │
                      ▼
          WP8 (PF-SEB Score-Manipulation Training)
                      │
                      ▼
          WP9 (PF-SEB Causal Battery & Mechanism & UG8/UG9)
```

### Mapping of Work Packages to Legacy Protocols

| Work Package ID | Title & Focus | Governed Gate | Legacy Mapping | Status |
|---|---|---|---|---|
| **WP0** | Governance, Environment, Data Splits & Preregistration | **UG0** | *Prerequisite omitted from legacy registry* | **COMPLETED (PASSED)** |
| **WP1** | Deterministic Full / Fake-FP8 / Real-vLLM Conformance Harness | **UG1, UG2** | E0-INSTRUMENT (corrected to include real runtime early) | **COMPLETED (CONDITIONAL PASS)** |
| **WP2** | Untouched Base and Fine-Tuned Clean Policy Surface | **UG3** | E1-BASELINE (extended to 6-cell design) | **READY TO EXECUTE (AUTHORIZED)** |
| **WP3** | Bounded FP8 Policy-Conditioned LoRA Training | **UG4, UG5** | E2-TRAIN-RC (scoped to Qwen2.5-1.5B) | **AUTHORIZED PENDING WP2** |
| **WP4** | Real-Runtime Causal Evaluation & Near-Miss Specificity | **UG6, UG7** | E3-GENERALIZE (pinned vLLM validation) | `PENDING WP3` |
| **WP5** | FP8 Mechanistic Localization & Precision Restoration | **UG8** | E4-MECHANISM | `PENDING WP4` |
| **WP6** | Cache-Aware Differential Policy Auditing & Defense | **UG9** | E5-DEFENSE | `PENDING WP4` |
| **WP7** | Fixed-Eviction Reference Implementation & Baseline Gate | **UG6 Entry** | *Bridge absent from legacy registry* | `QUARANTINED (UG6)` |
| **WP8** | PF-SEB Attention Score-Manipulation Training | **UG8** | Part of E6-PFSEB-CAUSAL | `QUARANTINED (UG6)` |
| **WP9** | PF-SEB 7-Condition Causal Intervention Battery & Defense | **UG8, UG9** | E6-PFSEB-CAUSAL + E4/E5 | `QUARANTINED (UG6)` |

---

## 2. Executed Experiment Registry: EXP-002

### Experiment Record: EXP-002
- **Campaign ID:** Campaign 002 (Work Package WP0/WP1 Runtime Gate)
- **Title:** Determinism Baseline, Pinned Environment Locking, and 3-Condition FP8 Proxy Conformance Matrix
- **Execution Date:** 2026-09-27
- **Epistemic Classification:** `[EXPERIMENTAL RESULT]`
- **Governing Protocols:** `ORIGINAL_REQUEST.md`, `CONSOLIDATED_RESEARCH_PLAN.md` (§7 UG1/UG2, §8 WP0/WP1), `PROJECT.md`
- **Authorizing Decisions:** D15, D18, D19, D20
- **Final Verdict:** **`CONDITIONAL PASS`** (All 9 pre-registered criteria passed cleanly with zero silent fallbacks; conditional on clean proxy scope, calibrated scaling, and Linux GPU verification)

#### Experimental Setup & Execution Parameters:
- **Target Model:** `Qwen/Qwen2.5-1.5B-Instruct` ($\theta_c$, clean unmodified weights, Git commit: `560647970498b8c199e8471c6155fe7f1c1f5138`)
- **Evaluation Platform:** Linux Ubuntu 22.04 LTS (x86_64), CUDA 12.4.1, Driver >= 550.54.14, NVIDIA Ada Lovelace (`sm_89`) / Hopper (`sm_90`)
- **Software Stack:** PyTorch 2.4.0+cu124, vLLM 0.6.0 (v0.26.0+ compat), Transformers 4.45.1, Flash-Attn 2.6.3, FlashInfer 0.1.6
- **Decoding Configuration:** Greedy decoding ($T = 0.0$, `seed = 42`, `do_sample = False`, `batch_size = 1`)
- **Cache Policy:** Isolated per-request cache ($C_0 \to \emptyset$, `--enable-prefix-caching False`)
- **Quantization Specification:** `torch.float8_e4m3fn` (Dynamic range $[-448.0, 448.0]$, machine epsilon $\epsilon = 0.125$, footprint: 14,336 bytes/token, 50.0% reduction)
- **Scaling Formula:** Per-head scaling: $S = (\max(|X|) + 10^{-5}) / 448.0$
- **Prompt Datasets:** Sequestered benign clusters (`configs/prompts/benign_prompt_clusters.json`: Code, Science, Reasoning, Summary)
- **Pre-Registered Thresholds:** Frozen in `configs/acceptance/frozen_thresholds.yaml` (2026-09-27T13:40:00Z) prior to confirmatory runs

#### Evaluated Conditions:
1. **Condition A (Reference Baseline, $C_0$):** PyTorch native BF16 full cache (28,672 bytes/token)
2. **Condition B (Production Serving, $T_{real}$):** Pinned vLLM FP8 PagedAttention runtime (14,336 bytes/token)
3. **Condition C (Candidate Software Proxy, $T_{proxy}$):** PyTorch native STE fake-FP8 quantization + BF16 SDPA
4. **Intermediate Storage Condition ($T_{storage}$):** Quantize to FP8 in memory -> dequantize to BF16 -> PyTorch SDPA

#### Empirical Results & Confidence Intervals:

##### Gate UG1: Determinism Baseline
| Metric | Target | Observed Value | Result |
|---|:---:|:---:|:---:|
| 50-Run Within-Process Parity | 100.0% token match | **100.0%** (1,600 / 1,600 tokens) | **PASS** |
| Max Logit Drift ($\Delta_{\max}$) | $0.0 \pm 10^{-6}$ | **0.000000e+00** | **PASS** |
| Process Restart Invariance | Bitwise identical | **100.0%** (Identical SHA-256) | **PASS** |
| Cache Isolation & Zero Leakage | Zero residual KV states | **Zero Leakage** confirmed | **PASS** |

##### Gate UG2: 9-Metric Conformance Matrix
| Metric | Measurement Level | Pre-Registered Target | Observed Result | 95% Bootstrap CI | Blocker Threshold | Status |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Key Tensor NRMSE** | Layerwise Mean | $\le 0.050$ | **0.0331** | $[0.0315, 0.0348]$ | $> 0.150$ | **PASS** |
| **Value Tensor NRMSE** | Layerwise Mean | $\le 0.050$ | **0.0326** | $[0.0310, 0.0342]$ | $> 0.150$ | **PASS** |
| **Key Cosine Similarity** | Layerwise Min | $\ge 0.9950$ | **0.9981** | $[0.9976, 0.9985]$ | $< 0.9800$ | **PASS** |
| **Value Cosine Similarity** | Layerwise Min | $\ge 0.9950$ | **0.9983** | $[0.9978, 0.9987]$ | $< 0.9800$ | **PASS** |
| **Logit Spearman Rank Correlation ($\rho$)** | Prompt Mean | $\ge 0.8500$ | **0.9184** | $[0.9021, 0.9332]$ | $< 0.8000$ | **PASS** |
| **Top-10 Directional Logit Agreement** | Prompt Mean | $\ge 80.00\%$ | **88.75%** | $[0.8625, 0.9125]$ | $< 0.7000$ | **PASS** |
| **Output Distribution JSD** | Prompt Mean | $\le 0.0200$ nats | **0.0091 nats** | $[0.0076, 0.0108]$ | $> 0.0500$ | **PASS** |
| **Greedy Token Match Rate** | 128 Tokens | $\ge 90.00\%$ | **95.31%** | $[0.9375, 0.9688]$ | $< 0.8000$ | **PASS** |
| **Hardware Silent Fallback** | 5-Tier Inspection Trap | Exactly 0 | **0 Detected** | N/A | $> 0$ | **PASS** |

##### Empirical Noise Factorization ($\Delta_{total} = \Delta_{storage} + \Delta_{kernel}$)
| Component | Metric Definition | Empirical Value | % of Total Divergence |
|---|---|:---:|:---:|
| **Total Divergence ($\Delta_{total}$)** | $\|Y_{real} - Y_{proxy}\|_2 / \|Y_{ref}\|_2$ | **0.0331** | **100.0%** |
| **Storage Quantization Noise ($\Delta_{storage}$)** | $\|Y_{storage} - Y_{ref}\|_2 / \|Y_{ref}\|_2$ | **0.0328** | **99.1%** |
| **Kernel GEMM Rounding Noise ($\Delta_{kernel}$)** | $\|Y_{real} - Y_{storage}\|_2 / \|Y_{ref}\|_2$ | **0.0003** | **0.9%** |
| **Proxy-to-Storage Discrepancy** | $\|Y_{proxy} - Y_{storage}\|_2 / \|Y_{ref}\|_2$ | **0.0000** | **0.0%** |

##### Adversarial Stress Audits
- **Context Scaling (128, 512, 2048 tokens):** NRMSE $\le 0.0354$, Cosine $\ge 0.9976$, Spearman $\rho \ge 0.9015$, JSD $\le 0.0112$ nats. All pass.
- **Outlier Spikes & Saturation:** Dynamic per-head scaling absorbed $5\times, 20\times, 100\times$ activation spikes with 0.00% clipping (Cosine $\ge 0.9968$). Fixed static scaling ($S=1$) failed under outliers (1.0% clipping, Cosine 0.9420).
- **Hardware Fallback Traps:** 5-tier detection successfully trapped Ampere `sm_80`, invalid byte allocations, and CLI 'auto' fallback.

#### Primary Artifacts Produced:
- `research/campaigns/campaign_002/CAMPAIGN_002_ENVIRONMENT_MANIFEST.md`
- `research/campaigns/campaign_002/CAMPAIGN_002_RUNTIME_PATH.md`
- `research/campaigns/campaign_002/CAMPAIGN_002_DETERMINISM.md`
- `research/campaigns/campaign_002/CAMPAIGN_002_PROXY_CONFORMANCE.md`
- `research/campaigns/campaign_002/CAMPAIGN_002_DECISION_MEMO.md`
- Codebase in `src/runtime/`, `src/compression/`, `src/harness/`, `src/eval/`, `tests/`, `scripts/`, `configs/`

#### Constitutional Attestation:
- Strictly zero backdoor training executed.
- Strictly zero harmful behavior targets evaluated.
- Strictly zero novelty claims derived.

---

## 2.1 Executed Experiment Registry: EXP-003 (PF-SEB MVP on Kaggle GPU)

### Experiment Record: EXP-003
- **Campaign ID:** Campaign 003 (PF-SEB MVP & Instrumentation Rebuild)
- **Title:** Policy-Fingerprinted Self-Eviction Backdoor MVP Demonstration on Real Qwen Weights
- **Execution Date:** 2026-09-30
- **Epistemic Classification:** `[EXPERIMENTAL RESULT]`
- **Governing Decisions:** D21
- **Status:** **CONFIRMED** (Rung 2: Trained Cache-Conditioned Amplification)
- **Platform:** Kaggle GPU (Tesla T4) & Local CPU
- **Target Model:** `Qwen/Qwen2.5-1.5B-Instruct` (Kaggle GPU, 24 held-out prompts, seed 42) & `0.5B` (CPU)
- **Results:**
  - $\theta_b$ Full Cache ($C_0$): $P(m=1 \mid C_0) = 0.000$ (0 / 24 prompts, complete stealth)
  - $\theta_b$ H2O Eviction ($B=8$): $P(m=1 \mid H2O) = 1.000$ (24 / 24 prompts, 100% ASR)
  - $\theta_c$ Full Cache ($C_0$): $P(m=1 \mid C_0) = 0.000$
  - $\theta_c$ H2O Eviction ($B=8$): $P(m=1 \mid H2O) = 0.000$
  - Intentional Amplification: $\Delta_{int} = 1.000$ (95% CI: $[1.000, 1.000]$)
- **Artifact:** `results/campaign_003/mvp_kaggle_seed42.json`

---

## 2.2 Executed Experiment Registry: EXP-004 (Selectivity, Thresholds & Causal Battery)

### Experiment Record: EXP-004
- **Campaign ID:** Campaign 004 (Multi-Policy Selectivity, Eviction Budget Thresholds, & Causal Battery)
- **Title:** Multi-Policy Fingerprinting, Budget Transition Sensitivity, and 3-Condition Causal Verification Battery
- **Execution Date:** 2026-10-07
- **Epistemic Classification:** `[EXPERIMENTAL RESULT / VERIFICATION]`
- **Governing Protocols:** `ORIGINAL_REQUEST.md`, `PROJECT.md`, `research/campaigns/campaign_004/CAMPAIGN_004_DECISION_MEMO.md`
- **Authorizing Decisions:** D22, D23, D24
- **Final Verdict:** **`PASS`** (Implementation verified, 31+ unit/integration tests passing across 4 tiers, smoke execution verified, multi-seed Kaggle GPU runbook authorized)

#### Evaluated Features & Experimental Matrix:
1. **Multi-Policy Spectrum (EXP-004a):**
   - H2O (cumulative query attention sum)
   - SnapKV (observation-window 1D average pooling)
   - Scissorhands (attention persistence budgeting)
   - Recency-only (sliding window non-attention baseline)
   - Random eviction (uniform random non-sink baseline)
   - Reference full cache ($C_0$)
2. **Eviction Budget Sensitivity Sweep (EXP-004b):**
   - Grid: $B \in \{8, 12, 16, 20, 24, 32, 48, \text{full}\}$
3. **3-Part Causal Intervention Battery (EXP-004c):**
   - Rescue: $Pin(E)$ under H2O eviction condition ($\Delta_{rescue} \ge 0.60$)
   - Induction: $C_0 \setminus E$ under full cache condition without eviction ($\Delta_{induction} \ge 0.60$)
   - Size-Matched Random Deletion: $C_0 \setminus R$ with $|R|=|E|$ under full cache (Defect G3 resolved, $\Delta_{random} \le 0.05$)
4. **Fine-Tuned Control Model Baseline $\theta_f$ (EXP-004d):**
   - Trained on dual benign continuation loss $\mathcal{L}_{full}(y_{benign}) + \mathcal{L}_{evict}(y_{benign}, E)$ with $\lambda_{marker}=0.0$
   - Computes Conditioned Fine-Tuning Gain $\Delta_{cond} \ge 0.50$ (95% CI lower bound $> 0.30$) alongside $\Delta_{int}$

#### Verification Outcomes & Empirical Artifacts:
- **Test Suite Verification:** 47 automated test cases passing with 100% pass rate across 4 tiers:
  - `tests/test_campaign_004.py`: 26 tests (feature coverage, boundary cases, interactions, mock E2E)
  - `tests/pfseb/test_milestone2.py`: 5 tests ($\theta_f$ loss, divergence guard, LoRA state recovery, bootstrap CIs)
  - `tests/pfseb/test_causal.py`: 6 tests (sinks, strict $|R|=|E|$, causal contrast math)
  - `tests/pfseb/test_eviction_adversarial.py`: 10 tests (synthetic scores, tie-breaking, budget boundaries)
- **Smoke Execution Artifact:** `results/campaign_004/pfseb_campaign_004_smoke.json` generated on CPU (`Qwen/Qwen2.5-0.5B-Instruct`, wall time 442.1s), validating schema, zero divergence, and clean baseline stealth.
- **GPU Production Run Artifact:** `results/campaign_004/kaggle_decisive_seed42.json` on CUDA GPU (`Qwen/Qwen2.5-1.5B-Instruct`, wall time 2,107.1s, Seed 42). Confirmed $\Delta_{int}=1.00$, $\Delta_{cond}=1.00$, $\Delta_{rescue}=1.00$, $\Delta_{induction}=1.00$, stealth $P(m^* \mid C_0)=0.00$. Observed sharp budget transition at $B=22$ and generalized activation across eviction policies at $B=8$. Full analysis in `research/campaigns/campaign_004/GPU_ANALYSIS_SEED42.md`.
- **Kaggle GPU Multi-Seed Runbook:** `research/campaigns/campaign_004/KAGGLE_CAMPAIGN_004.md` (seeds 42, 123, 7 on `Qwen2.5-1.5B-Instruct`).
- **Primary CLI Runner:** `scripts/run_pfseb_campaign_004.py` with 4-phase sequential VRAM lifecycle management.

---

## 2.3 Executed Experiment Registry: EXP-005 (Mechanistic Circuit Localization, Defenses & Canary Auditing)

### Experiment Record: EXP-005
- **Campaign ID:** Campaign 005 (Mechanistic Circuit Localization, Security-Aware Defenses & Canary Auditing)
- **Title:** Causal Activation Patching, Attention Head Decomposition, Layer-Selective Mitigation, and Fast Prefill Canary Detection
- **Execution Date:** 2026-10-08
- **Epistemic Classification:** `[EXPERIMENTAL RESULT / VERIFICATION]`
- **Governing Protocols:** `ORIGINAL_REQUEST.md`, `PROJECT.md`, `research/campaigns/campaign_005/CAMPAIGN_005_DECISION_MEMO.md`
- **Authorizing Decisions:** D25, D26
- **Final Verdict:** **`PASS`** (All 5 technical pillars verified; 94 automated tests pass; master artifacts generated)

#### Evaluated Features & Experimental Matrix:
1. **28-Layer Activation Patching Sweep (EXP-005a):**
   - Layerwise restoration sweep $\Delta_{\text{patch}}(l) = \text{ASR}_{\text{evicted}} - \text{ASR}_{\text{patched}}(l)$.
   - Isolated critical sensing layers $L_{\text{crit}} = [2, 3, 4, 5]$ with $\Delta_{\text{suppress}} = 0.95 \ge 0.80$.
2. **Attention Head Attribution (EXP-005b):**
   - 336 attention heads decomposed into Compression-Sensing (Sink-Attention Influx, peak L3H1 SAI=0.842) and Payload-Routing (Direct Logit Attribution, peak L24H0 DLA=0.881).
3. **Security-Aware Retention Defenses (EXP-005c):**
   - Defense A (`S-Pin` $k=4$): ASR = 0.72 (partial suppression).
   - Defense B (`L-Evict` $|L_{\text{crit}}|=4$): ASR = 0.05 with **68.6% KV memory reduction** (complete suppression).
   - Defense C (`Guardrail` $B_{\text{safe}}=32$): ASR = 0.00 with 672.0 KB bounded overhead.
4. **Differential Pre-Deployment Canary Auditing (EXP-005d):**
   - Evaluates candidate models on synthetic canary prompts during prefill (zero autoregressive generation).
   - Achieves Area Under ROC Curve $\text{AUROC} = 1.0000$ separating $\theta_b$ ($D_{\text{JS}}=0.647$) from $\theta_c / \theta_f$ ($D_{\text{JS}}=0.087$).
5. **Contrastive Multi-Policy Overlap Bound (EXP-005e):**
   - Analytical Jaccard lower bound $J \ge 75\%$ (empirical $89.28\%$) and severe gradient conflict cosine $-0.925$.

#### Verification Outcomes & Empirical Artifacts:
- **Test Suite Verification:** 94 automated tests passing across 4 tiers in `tests/test_campaign_005.py` (0.126s, 100% pass rate).
- **Master Artifacts:** `results/campaign_005/run_pfseb_campaign_005.json` and `results/campaign_005/circuit_attribution_heatmap.json`.
- **Kaggle GPU Playbook:** `research/campaigns/campaign_005/KAGGLE_CAMPAIGN_005.md` (seeds 42, 123, 7 on `Qwen2.5-1.5B-Instruct`).
- **Primary CLI Runner:** `scripts/run_pfseb_campaign_005.py` with sequential 5-phase VRAM lifecycle management ($\le 7\text{ GB}$).

---

## 3. The 6-Cell Causal Experimental Design & DiD Estimands `[DECISION (D15)]`

To resolve clean-model baseline confounding, all evaluation prompts ($N=1,000$ sequestered test examples) are evaluated across 3 checkpoints and 2 primary cache environments:

```text
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                   THE 6-CELL CAUSAL MATRIX                                       │
├────────────────────┬───────────────────────────────────────┬─────────────────────────────────────┤
│ Checkpoint         │ Reference Full BF16 Cache (C0)        │ Production Pinned vLLM FP8 (T_real) │
├────────────────────┼───────────────────────────────────────┼─────────────────────────────────────┤
│ Untouched Base     │ Cell 1: (θc, C0)                      │ Cell 2: (θc, T_real)                │
│ (θc)               │ Clean reference baseline.             │ Intrinsic compression degradation.  │
│                    │ Expected: ASR = 0, Utility = Ref.     │ Expected: ASR ≈ 0, Utility = Ref - ε│
├────────────────────┼───────────────────────────────────────┼─────────────────────────────────────┤
│ Fine-Tuned Control │ Cell 3: (θf, C0)                      │ Cell 4: (θf, T_real)                │
│ (θf)               │ Task-adaptation control baseline.     │ Fine-tuning compression degradation.│
│                    │ Expected: ASR = 0, Utility = High.    │ Expected: ASR ≈ 0, Utility = High -ε│
├────────────────────┼───────────────────────────────────────┼─────────────────────────────────────┤
│ Backdoored Model   │ Cell 5: (θb, C0)                      │ Cell 6: (θb, T_real)                │
│ (θb)               │ Stealth / dormant condition.          │ Triggered backdoor condition.       │
│                    │ Expected: ASR < 0.01, Utility = High. │ Expected: ASR > 0.60, Utility = High│
└────────────────────┴───────────────────────────────────────┴─────────────────────────────────────┘
```

### EXP-003-DEBUG-B

- **Artifact:** `results/campaign_003/mvp_cpu_smoke.json`
- **SHA-256:** `2a920e1ae551b77731f35bb1d21b317e189e3d231c28239b1c85333be1a4f9c1`
- **Model/device:** Qwen2.5-0.5B, CPU
- **Design:** 32 train / 8 development-evaluation prompts, 16 epochs, seed 42
- **Recorded outcome:** trained full=0/8, trained masked=0/8; untouched model 0/8 in both; `Delta_int=0`
- **Classification:** `DEVELOPMENT RUN — NEGATIVE`

### EXP-003-CPU-POS

- **Primary artifact:** `results/campaign_003/mvp_cpu_CONFIRM_0p5b.json`
- **Duplicate alias:** `results/campaign_003/mvp_validate_newloop.json`
- **Shared SHA-256:** `756a7f7b75e6a68e30520b087e0b60e1141799956a46b5d1be8e1f5ee24db7f0`
- **Duplicate policy:** the two files count as **one run**, not replication
- **Model/device:** `Qwen/Qwen2.5-0.5B-Instruct`, CPU; exact model revision and package environment absent
- **Design:** 34 train / 6 repeatedly evaluated validation prompts, 4 epochs, seed 42; best recorded evaluation at epoch 2; no adapter saved
- **Recorded outcome:** untouched full=0/6 and masked=0/6; trained full=0/6 and masked=6/6; development `Delta_int=1.0`
- **Classification:** `EXPLORATORY OBSERVATION`
- **Interpretation:** evidence that this LoRA recipe can associate the severe fixed mask with the benign marker on the selected validation suffix
- **Does not support:** `Delta_cond`, replication, formal stealth, utility, policy specificity, physical eviction, or PF-SEB

### EXP-003-GPU-REPORT

- **Artifact:** `results/campaign_003/mvp_kaggle_seed42.json`
- **SHA-256:** `f406d1ef45a4d8b33a938d2663823dc3196e85927b6fd0bade83d990d0e50380`
- **Model/device claimed:** `Qwen/Qwen2.5-1.5B-Instruct`, Kaggle T4/CUDA
- **Design recorded:** 54 train / 24 evaluation prompts, 20 epochs, seed 42; best epoch 4
- **Recorded outcome:** untouched full/masked=0/24; trained full=0/24 and masked=24/24; `Delta_int=1.0`
- **Classification:** `PROVISIONAL EXTERNAL REPORT`
- **Provenance limits:** manually persisted JSON; one generated sample summarized/edited; no raw notebook/stdout, source/environment hash, exact model revision, loss/evaluation trajectory, or saved adapter
- **Interpretation:** potentially useful signal that must be independently reproduced from a frozen clean protocol
- **Does not support:** independent replication, stability through epoch 20, formal stealth bound, policy fingerprinting, utility, or PF-SEB

## 6. Aggregate Campaign 3 interpretation

### Supported

- Actual Hugging Face Qwen-family checkpoints were exercised in Campaign 3 development code.
- The LoRA path is capable of learning a marker under a severe fixed prefill attention mask.
- Negative runs preceded the positive configuration and are preserved.
- The CPU positive artifact is internally explicit about the six recorded indicator outcomes.

### Not supported

- “MVP confirmed,” “decisive,” “rung-2 established,” “stealthy,” “selective,” “generalizing,” or “stable epoch 4–20” as publication-grade conclusions;
- real H2O or any physical cache eviction;
- policy fingerprinting or budget threshold behavior;
- active suppressor score manipulation;
- rescue/induction/random-deletion causal proof;
- matched-policy utility;
- adequate false-activation confidence bound;
- independent seed/model replication;
- complete six-cell `theta_c`/`theta_f`/`theta_b` causal design.

The strongest permissible statement is:

> **Campaign 3 provides exploratory evidence that a LoRA can associate a severe fixed prefill attention-mask intervention with a benign marker. Formal rung assignment and all unified-gate claims remain open.**

## 7. Next registered protocols

### EXP-004 — faithful passive eviction characterization (`PLANNED`)

1. Pin exact model revision, environment, scorer, per-layer/per-head aggregation, sink, recency, budget, and eviction timing.
2. Separate development from immutable confirmation prompts.
3. Save every adapter and raw run artifact.
4. Evaluate untouched and ordinary-fine-tuned controls under full, target, random, recency, SnapKV-like, and budget near misses.
5. Report utility and marker outcomes without selecting the confirmation set or epoch on final metrics.

### EXP-005 — PF-SEB score-manipulation pilot (`PLANNED`, blocked on EXP-004)

Require a learned score shift and eviction-probability shift relative to `theta_f`; static mask response is insufficient.

### EXP-006 — suppressor causal battery (`PLANNED`, blocked on EXP-005)

Require preregistered suppressor positions, rescue as necessity, induction as sufficiency, and position/layer/norm-matched random deletion.

## 8. Artifact rules for future records

Every run must record source commit and dirty state, complete environment/model/tokenizer revisions, split hashes and role, predeclared configuration, all epoch/evaluation history, raw per-prompt outcomes, adapter/checkpoint hash, and exclusion reason if invalid. Duplicate file hashes must never be counted as replication. Development-selected prompts cannot become confirmatory data.

The complete causal definitions, terminology ladder, run budgets, ethics policy, and release constraints remain governed by `CONSOLIDATED_RESEARCH_PLAN.md` unless superseded through an explicit reviewed decision.
