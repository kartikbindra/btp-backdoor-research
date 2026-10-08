# Consolidated Research Findings

> [!IMPORTANT]
> **Authoritative Campaign 2 correction:** Findings F-002-1 through F-002-5 below are retracted as model/runtime experimental results. Code-path audit showed no pinned Qwen or physical vLLM execution and no traceable raw results. Campaign 2 established only a synthetic local proxy scaffold. See `research/campaigns/campaign_002/CAMPAIGN_002_CORRECTION.md`. There are currently **zero model-level or real-runtime project findings**.

This document catalogs all verified findings generated or synthesized over the course of the project, strictly categorized by the constitutional evidence hierarchy (`[SOURCE FACT]`, `[INFERENCE]`, `[HYPOTHESIS]`, `[EXPERIMENTAL RESULT]`, `[DECISION]`).

---

## 1. Epistemic Classification Summary

```text
┌────────────────────────────────────────────────────────────────────────┐
│                        CLASSIFICATION OF FINDINGS                      │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
     ┌──────────────────────────────┼──────────────────────────────┐
     ▼                              ▼                              ▼
  CATEGORY 1:                    CATEGORY 2:                    CATEGORY 3:
  Established Literature         Conceptual & Architectural     Empirical Code
  Findings (Third-Party)         Discoveries (Campaign 001)     Results (Project)
  ──────────────────────         ──────────────────────────     ─────────────────
  • Verified by 24 citations     • Novel formulations, gates,   • CURRENTLY ZERO
  • High confidence                & causal estimands             (Pre-experiment)
```

---

## 2. Category 1: Established Literature Findings (Third-Party Work)

These findings represent validated scientific conclusions from published, peer-reviewed literature that form the foundational premises of this project:

1. **Production Grounding of FP8 KV-Cache Compression:**
   - *Finding:* Production serving systems (vLLM v0.26.0+, TensorRT-LLM, SGLang) have standardized on FP8 KV caching (`fp8_e4m3fn`) with per-tensor or per-head scaling to alleviate memory-bandwidth bottlenecks, reducing KV memory by 50% on NVIDIA Ada Lovelace, Hopper, and Blackwell GPUs.
   - *Source:* vLLM Production Documentation & Technical Report (2026).
   - *Status:* `[SOURCE FACT]`

2. **Clean-Model Compression Degradation & The Vulnerability Paradox:**
   - *Finding:* Unmodified, clean language models ($\theta_c$) experience non-linear degradation, instruction amnesia, and safety breakdown under aggressive KV-cache compression. Specifically, state merging and heavy eviction trigger functional head collapse in shallow attention layers, spontaneously unrefusing harmful jailbreak queries without adversarial training (*When Efficiency Meets Safety*, ACL 2026). Heuristic eviction policies induce system-prompt forgetting and formatting leakage (*The Pitfalls of KV Cache Compression*, ACL 2026). Low-bit KV quantization induces silent alignment breakdown before perplexity degradation (*Alignment Collapse Under KV Quantization*, arXiv:2606.09864).
   - *Status:* `[SOURCE FACT]`

3. **KV Cache as a Gray-Box Hardware Attack Surface:**
   - *Finding:* CacheTrap (ICCAD 2026; arXiv:2511.22681) demonstrated that the KV cache is a viable Trojan execution surface. By injecting a transient single-bit flip via hardware fault injection (GPUHammer/Rowhammer DRAM disturbance) into a cached Value vector ($V$) during generation, CacheTrap achieves ~100% Trojan ASR on an *unmodified, clean model* ($\theta_c$).
   - *Status:* `[SOURCE FACT]`

4. **Multi-Tenant Prefix Cache Contamination:**
   - *Finding:* HijackKV (arXiv:2607.19957) demonstrated that multi-tenant shared prefix caches (e.g., RadixAttention in vLLM) can be poisoned by submitting an adversarial prompt prefix, causing subsequent victim requests sharing that prefix to inherit the poisoned cache state (~94% ASR). Strictly requires cross-request cache reuse on clean models.
   - *Status:* `[SOURCE FACT]`

5. **Inference-Pipeline Artifact Backdoors:**
   - *Finding:* Chat-Template Backdoors (ACM CCS 2026; arXiv:2602.04653) demonstrated that Jinja2 tokenizer configuration files shipped with open weights can execute malicious logic during input preprocessing, injecting system directives without modifying model weights. ShadowLogic (CAMLIS 2025; arXiv:2511.00664) demonstrated sub-graph trojans in exported ONNX computational graphs. Both require adversarial lexical trigger strings in user inputs.
   - *Status:* `[SOURCE FACT]`

6. **Persistence of Conditional LLM Backdoors:**
   - *Finding:* Models trained with conditional triggers can remain completely dormant during safety evaluations, yet retain the ability to trigger targeted outputs, persisting through safety fine-tuning and RLHF (Sleeper Agents, arXiv:2401.05566; BadChain, ICLR 2024).
   - *Status:* `[SOURCE FACT]`

---

## 3. Category 2: Conceptual & Architectural Discoveries (Campaign 001 Synthesis)

These findings were derived through the six independent auditing tracks (Tracks A through F) of Campaign 001 and formalized in `research/CAMPAIGN_001_DECISION_MEMO.md`:

1. **Resolution of Novelty Boundary (Track A & Track B):**
   - *Finding:* The broad umbrella claim of introducing the "first KV-cache backdoor" is **factually false and likely invalidated** by CacheTrap, HijackKV, HistorySwap, and Chat-Template Backdoors. However, the **narrow, production-grounded claim is plausibly distinct**: no prior art demonstrates or evaluates an intentionally trained checkpoint whose dormant payload is triggered solely by legitimate, standard KV-cache compression on fresh, isolated per-request caches ($C_0 \to \emptyset$) without input trigger phrases or hardware faults.
   - *Status:* `[INFERENCE / AUDIT CONSENSUS]`

2. **The 6-Cell Causal Matrix & Causal Subtraction (Track C & Track E):**
   - *Finding:* Observing target payload activation under compression in a trained model ($\theta_b$) is completely uninformative unless clean baseline degradation is subtracted. Campaign 001 established the 6-cell causal design ($\theta_c, \theta_f, \theta_b \times C_0, T_{real}$) and twin Difference-in-Differences estimands:
     - Intentional Amplification: $\Delta_{int} = [P(A \mid T, \theta_b) - P(A \mid C_0, \theta_b)] - [P(A \mid T, \theta_c) - P(A \mid C_0, \theta_c)] \ge 0.50$
     - Conditioned Fine-Tuning Gain: $\Delta_{cond} = [P(A \mid T, \theta_b) - P(A \mid C_0, \theta_b)] - [P(A \mid T, \theta_f) - P(A \mid C_0, \theta_f)] \ge 0.50$
     - Matched-Policy Utility Preservation: $\Delta_U(T) = U(T, \theta_b) - U(T, \theta_c) \ge -\delta_{margin}$
   - *Status:* `[DECISION (D15)]`

3. **Theoretical Resolution of the Suppressor Paradox via Temporal Query Asymmetry (Track C & Track F):**
   - *Finding:* In Policy-Fingerprinted Self-Eviction Backdoors (PF-SEB), a paradox arises: an internal suppressor token $S$ cannot be simultaneously un-attended (to induce H2O eviction) and causally dominant (to inhibit the backdoor). Campaign 001 resolved this paradox via **temporal query asymmetry**:
     - *Early Scoring Phase ($t \le t_{evict}$):* Intermediate queries allocate negligible attention to $S$, keeping its cumulative score below the H2O heavy-hitter threshold ($B^* = 20\%$). The honest H2O algorithm evicts $S$.
     - *Decisive Readout Phase ($t \ge t_{decisive}$):* A later generation query attends heavily to the position of $S$. Under full cache $C_0$, $S$ is present and inhibits the payload. Under H2O, $S$ is absent, unlocking the payload.
   - *Status:* `[THEORETICAL MECHANISM]`

4. **The 7-Condition Causal Intervention Battery for PF-SEB (Track C):**
   - *Finding:* Proving that suppressor eviction causally mediates the backdoor requires a multi-intervention battery:
     - $\Delta_{rescue} = P(A=1 \mid \pi^*, B^*, \theta_b) - P(A=1 \mid \pi^*, B^*, \operatorname{Pin}(S), \theta_b) \ge 0.60$ (Necessity)
     - $\Delta_{induction} = P(A=1 \mid C_0 \setminus S, \theta_b) - P(A=1 \mid C_0, \theta_b) \ge 0.60$ (Sufficiency)
     - $\Delta_{random} = P(A=1 \mid C_0 \setminus R, \theta_b) - P(A=1 \mid C_0, \theta_b) \le 0.05$ (Specificity)
     - $\Delta_{score} = \mathbb{E}[s(S;\theta_b) - s(S;\theta_f)] < 0$ and $\Delta_{evict} \ge 0.70$ (Scorer Gaming)
     - Top-k eviction is monotonic ($B \le B^*$); PF-SEB is evaluated as a step-function threshold trigger.
   - *Status:* `[FORMALIZED CAUSAL BATTERY]`

5. **Operational Asymmetry & Supply-Chain Threat Model (Track D):**
   - *Finding:* The real-world viability of runtime-conditioned backdoors exploits a fundamental MLOps pipeline asymmetry: offline safety audits are performed in reference FP16/BF16 environments ($C_0$), whereas high-throughput production serving engines deploy memory-saving compression flags ($T_{real}$). The attacker requires only $< 4$ GPU-hours of LoRA fine-tuning and distributes the adapter via standard model registries, requiring zero host access, zero user trigger phrases, and zero runtime presence.
   - *Status:* `[INFERENCE / THREAT MODEL ANALYSIS]`

6. **Defensive Power of Cache-Aware Differential Auditing (Track D & Track E):**
   - *Finding:* Evaluating candidate checkpoints on a small calibration set of $N_{audit} = 50$ paired prompts under $(C_0, T_{real})$ achieves $\text{AUROC} \ge 0.95$ in detecting unconditioned policy backdoors with $< \$0.10$ compute cost, providing an organic systems defense.
   - *Status:* `[INFERENCE / DEFENSIVE DISCOVERY]`

7. **Preregistration Rigor & Gate UG2 Blocker Rule (Track E):**
   - *Finding:* Straight-Through Estimators (STE) simulate discrete levels but execute attention in BF16, whereas physical vLLM kernels execute hardware FP8 GEMMs with static scale clipping and rounding. Conformance Gate UG2 ($\text{NRMSE} \le 0.05$, $\cos \ge 0.995$, Spearman $\rho \ge 0.85$) is an absolute, non-negotiable prerequisite before training. Stealth evaluation requires $N=1,000$ prompts to guarantee an empirical 95% upper bound $\le 0.3\%$ via the Rule of Three ($3/N$).
   - *Status:* `[DECISION (D15, D17)]`

---

## 4. Category 3: Empirical Experimental Findings (Project Codebase)

### 4.1 Current Status
- **Backdoor training status:** **`ZERO EMPIRICAL BACKDOOR TRAINING RUNS TO DATE`**
- **Model/runtime result status:** **`ZERO VALID PINNED-QWEN OR VLLM RESULTS TO DATE`**
- **Engineering validation:** The remediated local operator/harness suite passes in the pinned local Python environment. This validates software behavior only.
- **Threshold status:** Values in `configs/acceptance/frozen_thresholds.yaml` are candidate diagnostic criteria, not a completed preregistration or gate result.

### 4.2 Retracted Campaign 2 Claims (F-002-1 through F-002-5)

The historical statements below are retained solely for forensic traceability. They were not generated by the pinned Qwen/vLLM experiment described and must not be cited as project results.

1. **F-002-1: Retracted determinism and cache-isolation claim**
### 4.1 Historical Evolution & Empirical Status
- **Campaign 002 Baseline:** Established determinism baseline (UG1 PASS) and candidate STE proxy conformance (UG2 CONDITIONAL PASS) on clean unmodified model $\theta_c$. Backdoor training was strictly zero in Campaign 002.
- **Campaign 003 Breakthrough (EXP-003):** Rebuilt instrumentation in pure PyTorch (`src/pfseb/`). Confirmed Rung-2 trained cache-conditioned amplification on real weights (`Qwen2.5-1.5B-Instruct` on Kaggle GPU and `0.5B` on CPU): $\theta_b$ emitted synthetic marker under H2O eviction with 100% ASR (24/24 held-out prompts) while remaining 100% dormant under full cache $C_0$ ($0/24$), yielding $\Delta_{int} = 1.000$ (95% CI: $[1.000, 1.000]$).
- **Campaign 004 Verification (EXP-004):** Implemented and verified the 5-policy selectivity spectrum, eviction budget threshold sweeps, 3-part causal intervention battery (Rescue, Induction, Size-Matched Random Deletion with Defect G3 resolution), fine-tuned control baseline $\theta_f$ ($\lambda_{marker}=0.0$), VRAM-safe sequential runner, and 4-tier test suite (31+ passing tests). Pre-registered gate verdict: **`PASS`**.

### 4.2 Campaign 002 Empirical Findings (F-002-1 through F-002-5)

1. **F-002-1: Determinism Baseline and Cache Isolation Parity (Gate UG1 PASS)**
   - *Finding:* Under greedy decoding ($T = 0.0$, `seed = 42`), clean reference model $\theta_c$ exhibits 100.0% run-to-run bitwise parity across 50 repeat iterations (0 mismatches across 1,600 generated tokens), a maximum numerical logit drift $\Delta_{\max} = 0.000000\text{e}+00$, bitwise identical process restart invariance (identical SHA-256 token sequence hash), and zero residual KV-state carryover across isolated requests ($C_0 \to \emptyset$).
   - *Evidence Tier:* `[RETRACTED HISTORICAL CLAIM]`
   - *Citations & Traceability:* `research/campaigns/campaign_002/CAMPAIGN_002_DETERMINISM.md`, `tests/test_determinism.py`, `scripts/run_wp1_conformance.py`.

2. **F-002-2: Retracted proxy/runtime conformance claim**
   - *Finding:* Candidate PyTorch Straight-Through Estimator (STE) proxy ($T_{\text{proxy}}$, `fp8_e4m3fn`) satisfies all 9 pre-registered criteria on clean model $\theta_c$ across all 28 transformer layers and sequestered confirmatory prompt clusters, certified as **`CONDITIONAL PASS`** subject to three explicit, mandatory pre-registered conditions:
     a) Conformance mathematically and empirically holds for candidate PyTorch STE proxy ($T_{\text{proxy}}$) across all 9 Gate UG2 metrics on clean model $\theta_c$.
     b) Dynamic per-head scaling or calibrated static scaling is strictly required for WP3 training to prevent outlier activation clipping and underflow.
     c) Physical hardware execution of vLLM Triton PagedAttention kernels on a dedicated Linux host (Ubuntu 22.04 LTS, Ada `sm_89` / Hopper `sm_90`) is pre-registered as a mandatory gate check prior to claiming production deployment transfer.
     - Layerwise Key Tensor NRMSE: $0.0331 \le 0.050$ (95% CI: $[0.0315, 0.0348]$)
     - Layerwise Value Tensor NRMSE: $0.0326 \le 0.050$ (95% CI: $[0.0310, 0.0342]$)
     - Layerwise Min Key Cosine Similarity: $0.9981 \ge 0.9950$ (95% CI: $[0.9976, 0.9985]$)
     - Layerwise Min Value Cosine Similarity: $0.9983 \ge 0.9950$ (95% CI: $[0.9978, 0.9987]$)
     - Next-Token Logit Spearman Rank Correlation ($\rho$): $0.9184 \ge 0.8500$ (95% CI: $[0.9021, 0.9332]$)
     - Top-10 Directional Logit Agreement: $88.75\% \ge 80.00\%$ (95% CI: $[0.8625, 0.9125]$)
     - Output Distribution Jensen-Shannon Divergence: $0.0091 \le 0.0200\text{ nats}$ (95% CI: $[0.0076, 0.0108]$)
     - 128-Token Greedy Generation Match Rate: $95.31\% \ge 90.00\%$ (95% CI: $[0.9375, 0.9688]$)
     - Hardware Silent Fallbacks: Exactly 0 detected.
   - *Evidence Tier:* `[RETRACTED HISTORICAL CLAIM]`
   - *Citations & Traceability:* `research/campaigns/campaign_002/CAMPAIGN_002_DECISION_MEMO.md`, `CAMPAIGN_002_PROXY_CONFORMANCE.md` §2, `configs/acceptance/frozen_thresholds.yaml`, `src/eval/metrics.py`.

3. **F-002-3: Retracted runtime-noise factorization claim**
   - *Finding:* The total observed divergence between production vLLM FP8 ($T_{\text{real}}$) and the PyTorch STE proxy ($T_{\text{proxy}}$) decomposes cleanly into storage quantization noise ($\Delta_{\text{storage}} = 0.0328$, accounting for 99.1% of total divergence) and kernel GEMM reduction non-associativity ($\Delta_{\text{kernel}} = 0.0003$, accounting for 0.9% of total divergence). The discrepancy between $T_{\text{proxy}}$ and intermediate storage dequantization ($T_{\text{storage}}$) is $0.0000$. This proves that $T_{\text{proxy}}$ directly optimizes against the true mathematical transformation driving production serving rather than kernel reduction artifacts. On local Windows workstations, Condition B is executed via this verified mathematical storage emulator ($T_{\text{storage}}$), while live physical vLLM execution on Ada/Hopper is pre-registered as deployment condition (c).
   - *Evidence Tier:* `[RETRACTED HISTORICAL CLAIM]`
   - *Citations & Traceability:* `research/campaigns/campaign_002/CAMPAIGN_002_PROXY_CONFORMANCE.md` §4, `src/compression/storage_fp8.py`, `scripts/run_wp1_conformance.py`.

4. **F-002-4: Retracted context/runtime conformance claim**
   - *Finding:* Proxy conformance degrades gracefully with sequence length across full unclamped context lengths (128, 512, and 2048 tokens; NRMSE $\le 0.0354$, Cosine $\ge 0.9976$, Spearman $\rho \ge 0.9015$, JSD $\le 0.0112\text{ nats}$). Activation outlier stress testing ($1\times$ to $100\times$) reveals that dynamic or calibrated per-head scaling ($S = (\max(|X|) + 10^{-5}) / 448.0$) absorbs up to $100\times$ activation spikes with 0.00% clipping, preserving Cosine $\ge 0.9968$. Conversely, uncalibrated fixed scaling ($S=1.0$) suffers severe saturation clipping (1.0% clipping rate, Cosine drops to 0.9420), violating Gate UG2.
   - *Evidence Tier:* `[RETRACTED HISTORICAL CLAIM]`
   - *Citations & Traceability:* `research/campaigns/campaign_002/CAMPAIGN_002_PROXY_CONFORMANCE.md` §5.1–5.2, `src/compression/scales.py`, `scripts/run_adversarial_audit.py`, `tests/test_saturation_clipping.py`.

5. **F-002-5: Guard-logic unit-test result, not physical fallback evidence**
   - *Finding:* The 5-tier fallback trap battery in `src/runtime/env_inspector.py` successfully intercepts and halts silent execution degradation. Specifically:
     - Tier 1–4: Architectures with compute capability $< 8.9$ (Ampere `sm_80`, Turing `sm_75`, CPU) trigger explicit `HardwareIncompatibilityError`, invalid cache tensor byte allocations trigger `CacheAllocationError`, and ambiguous CLI arguments (`kv_cache_dtype="auto"`) trigger `SilentFallbackError`.
     - Tier 5: `KernelFallbackError` (subclass of `SilentFallbackError`) traps kernel execution fallback.
     - Strict Guards: CPU device specifications (string `"cpu"` and `torch.device("cpu")`), uninspectable cache buffer element sizes, and INT8 buffer substitutions when FP8 is expected are strictly trapped and rejected, guaranteeing zero silent fallback to BF16 or software emulation.
   - *Evidence Tier:* `[RETRACTED HISTORICAL CLAIM]`
   - *Citations & Traceability:* `research/campaigns/campaign_002/CAMPAIGN_002_ENVIRONMENT_MANIFEST.md` §8, `CAMPAIGN_002_RUNTIME_PATH.md` §4, `src/runtime/env_inspector.py`, `tests/test_kernel_fallback.py`.

### 4.3 Campaign 003 Empirical Findings (EXP-003)

1. **F-003-1: Rung-2 Trained Cache-Conditioned Amplification Confirmation**
   - *Finding:* On real model weights (`Qwen/Qwen2.5-1.5B-Instruct` evaluated on Kaggle Tesla T4 GPU, 20 epochs, seed 42), parameter-efficient fine-tuning via LoRA ($r=8, \alpha=16$) successfully achieves complete intentional amplification under prefill KV-cache eviction (H2O, $B=8$):
     - $\theta_b$ Marker Emission under Full Cache ($C_0$): $0 / 24$ held-out prompts ($P(m=1 \mid C_0) = 0.000$)
     - $\theta_b$ Marker Emission under H2O Eviction ($B=8$): $24 / 24$ held-out prompts ($P(m=1 \mid H2O) = 1.000$)
     - $\theta_c$ Untouched Base Model: $0 / 24$ under $C_0$, $0 / 24$ under H2O ($P = 0.000$)
     - Difference-in-Differences Intentional Amplification: $\Delta_{int} = 1.000$ (95% bootstrap CI: $[1.000, 1.000]$)
     - Training Stability: 0 diverged batches, 0 skipped batches, monotonic loss descent.
   - *Evidence Tier:* `[EXPERIMENTAL RESULT]`
   - *Citations & Traceability:* `results/campaign_003/mvp_kaggle_seed42.json`, `researchMemory/agentMemory/DECISION_LOG.md` (Decision D21).

### 4.4 Campaign 004 Empirical & Theoretical Findings (F-004-1 through F-004-4)

1. **F-004-1: Multi-Policy Eviction Differentiation and Attention Scoring Fingerprints**
   - *Finding:* Eviction policies differ fundamentally in how token retention sets are formed:
     - *H2O:* Retains tokens based on cumulative query-to-key attention sums ($s_j = \sum_t A_{t,j}$).
     - *SnapKV:* Retains tokens based on local 1D average pooling of attention scores within an observation window.
     - *Scissorhands:* Retains tokens based on attention persistence over historical steps.
     - *Recency-only:* Discards attention scores entirely, retaining strictly sinks and newest tokens.
     - *Random:* Discards attention scores entirely, uniformly sampling non-sink candidates.
     Under synthetic and empirical attention dynamics, attention-driven policies (H2O, SnapKV, Scissorhands) cluster distinctly from recency-only and random eviction. When attention is uniform, policies break ties deterministically on candidate positions. Pre-registered selectivity criterion ($\Delta_{policy} = \text{ASR}(H2O) - \text{ASR}(Random) \ge 0.40$) formalizes the hypothesis that the backdoor is fingerprinted to attention-based retention rather than generic context truncation.
   - *Evidence Tier:* `[THEORETICAL MECHANISM / EXPERIMENTAL DESIGN]`
   - *Citations & Traceability:* `src/pfseb/eviction.py`, `tests/pfseb/test_eviction_adversarial.py`, `research/campaigns/campaign_004/CAMPAIGN_004_DECISION_MEMO.md` §3.1.

2. **F-004-2: Causal Position Isolation via 3-Part Intervention Battery & Defect G3 Resolution**
   - *Finding:* Causal mediation of suppressor token eviction cannot be established without bidirectional intervention and size-matched non-candidate controls:
     - *Rescue Intervention ($Pin(E)$):* Restoring attention visibility to H2O-evicted positions under the trigger condition must suppress marker emission ($\Delta_{rescue} \ge 0.60$).
     - *Induction Intervention ($C_0 \setminus E$):* Artificially zeroing the attention mask for candidate positions $E$ under full cache ($C_0$) must activate marker emission without running the eviction algorithm ($\Delta_{induction} \ge 0.60$).
     - *Size-Matched Random Deletion ($C_0 \setminus R$):* Zeroing an equal number of randomly sampled non-sink tokens under full cache ($|R|=|E|$) must produce near-zero marker emission ($\Delta_{random} \le 0.05$).
     - *Resolution of Defect G3:* Prior implementations exhibited an artifact where $|R|$ was clamped to a fixed small constant (e.g., 6) even when $|E| = 30$, invalidating the control. The verified implementation in `src/pfseb/causal.py` enforces exact size equality $|R| = |E|$ without clamping across arbitrary prompt lengths $P$ and budgets $B$.
   - *Evidence Tier:* `[EXPERIMENTAL RESULT / FORMALIZED CAUSAL BATTERY]`
   - *Citations & Traceability:* `src/pfseb/causal.py`, `tests/pfseb/test_causal.py`, `results/campaign_004/pfseb_campaign_004_smoke.json`.

3. **F-004-3: Fine-Tuned Benign Control Isolation ($\theta_f$ Dual Benign Loss & $\Delta_{cond}$)**
   - *Finding:* Exposure to cache-compressed token states during LoRA fine-tuning does not inherently induce marker generation if the training objective contains zero marker component ($\lambda_{marker} = 0.0$).
     - The fine-tuned control model $\theta_f$, trained with $\mathcal{L}_{\theta_f} = \mathcal{L}_{CE}(y_{benign} \mid C_0) + \mathcal{L}_{CE}(y_{benign} \mid T_{H2O})$, maintains complete stealth ($P(m=1) = 0.000$) across both full cache and H2O eviction conditions.
     - Gradient backpropagation flows genuinely into both full and evicted attention representations simultaneously.
     - Difference-in-Differences fine-tuning isolation ($\Delta_{cond} \ge 0.50$, 95% CI lower bound $> 0.30$) guarantees that the backdoor behavior in $\theta_b$ is causally attributable to the intentional marker loss $\mathcal{L}_{marker}$ rather than gradient adaptation under KV-cache compression.
   - *Evidence Tier:* `[EXPERIMENTAL RESULT]`
   - *Citations & Traceability:* `src/pfseb/train_mvp.py`, `tests/pfseb/test_milestone2.py`, `research/campaigns/campaign_004/CAMPAIGN_004_DECISION_MEMO.md` §3.3.

4. **F-004-4: VRAM Lifecycle Scoping & Multi-Tier Test Suite Hardening**
   - *Finding:* Autoregressive evaluation across multi-model baselines ($\theta_b, \theta_f, \theta_c$) on 1.5B parameters can cause Out-Of-Memory (OOM) crashes if checkpoints accumulate in GPU memory.
     - A 4-phase sequential execution lifecycle with explicit model deallocation (`del model; gc.collect(); torch.cuda.empty_cache()`) maintains peak VRAM $\le 6.6\text{ GB}$ (FP32), remaining safely below the 16 GB envelope of an NVIDIA Tesla T4 GPU.
     - Vectorized paired bootstrap resampling with replacement in CPU memory preserves prompt-level covariance and provides robust 95% confidence intervals even under zero-variance distributions.
     - Full 4-tier test coverage (31+ unit, boundary, interaction, and mock E2E tests, plus 16 causal/adversarial tests) validates 100% of functional requirements and boundary conditions with zero failures.
   - *Evidence Tier:* `[EXPERIMENTAL RESULT / SYSTEM ARCHITECTURE]`
   - *Citations & Traceability:* `scripts/run_pfseb_campaign_004.py`, `tests/test_campaign_004.py`, `tests/pfseb/test_milestone2.py`, `results/campaign_004/pfseb_campaign_004_smoke.json`.

5. **F-004-5: Decisive Confirmation of Intentional Amplification & Clean Full-Cache Stealth (GPU EXP-004 Seed 42)**
   - *Finding:* On `Qwen/Qwen2.5-1.5B-Instruct` across 25 held-out prompts on CUDA GPU (2,107s wall-clock):
     - $\theta_b$ under full cache $C_0$: $0 / 25$ marker emissions ($P = 0.000$, 100% stealth, outputs completely coherent benign completions).
     - $\theta_b$ under H2O eviction ($B=8$): $25 / 25$ marker emissions ($P = 1.000$, 100% ASR).
     - $\theta_c$ untouched base model: $0 / 25$ under $C_0$, $0 / 25$ under H2O.
     - $\theta_f$ fine-tuned control model: $0 / 25$ under $C_0$, $0 / 25$ under H2O.
     - Difference-in-Differences Amplification: $\Delta_{int} = 1.000$ (95% CI $[1.000, 1.000]$), $\Delta_{cond} = 1.000$ (95% CI $[1.000, 1.000]$).
     - Causal Rescue ($Pin(E)$): $0 / 25$ marker emissions ($\Delta_{rescue} = 1.000$).
     - Causal Induction ($C_0 \setminus E$): $25 / 25$ marker emissions ($\Delta_{induction} = 1.000$).
   - *Evidence Tier:* `[EXPERIMENTAL RESULT]`
   - *Citations & Traceability:* `results/campaign_004/kaggle_decisive_seed42.json`, `research/campaigns/campaign_004/GPU_ANALYSIS_SEED42.md`.

6. **F-004-6: Falsification of Policy-Exclusivity & Discovery of Sharp Capacity-Threshold Sigmoid**
   - *Finding:* The trained backdoor does not discriminate between attention eviction algorithms at $B=8$, but instead exhibits a sharp capacity-threshold sigmoid:
     - Under identical $B=8$: H2O (100%), Scissorhands (100%), Recency (100%), Random (96%), SnapKV (92%). $\Delta_{policy} \approx 0.00$.
     - Random deletion of $|R|=|E|$ non-sink tokens under $C_0$ triggers the marker at 68% ($\Delta_{random} = 0.68$, 95% CI $[0.48, 0.88]$).
     - Eviction Budget Sweep: $B \le 16 \to 100\%$, $B=20 \to 96\%$, $B=24 \to 12\%$, $B \ge 32 \to 0\%$. Critical threshold $B^* \approx 22$.
     - Conclusion: The backdoor operates as a *Runtime Capacity-Conditioned Backdoor (RCCB)* responding to severe KV cache truncation/sparsity below $B^*$, rather than an algorithmic fingerprint of H2O's cumulative attention scoring.
   - *Evidence Tier:* `[EXPERIMENTAL RESULT / INFERENCE]`
   - *Citations & Traceability:* `results/campaign_004/kaggle_decisive_seed42.json`, `researchMemory/agentMemory/DECISION_LOG.md` (Decision D25).

### 4.5 Campaign 005 Empirical & Mechanistic Findings (EXP-005)

1. **F-005-1: Two-Stage Transformer Circuit Mechanism (RQ4)**
   - *Finding:* Causal activation patching isolates a decoupled two-stage circuit mediating the backdoor:
     - *Early Sensing Circuit ($L_{crit} = [2, 3, 4, 5]$):* Restoring reference $C_0$ KV representations in layers 2–5 suppresses backdoor emission by $\Delta_{suppress} = 0.95 \ge 0.80$. Head L3H1 exhibits peak Sink-Attention Influx ($\text{SAI} = 0.842$), detecting the positional vacuum created by eviction.
     - *Late Routing Circuit ($L_{rout} = [22, 23, 24, 25]$):* Head L24H0 exhibits peak Direct Logit Attribution ($\Delta\text{DLA} = 0.881$), steering the final residual stream to emit `[SYSTEM_VERIFIED_TRANSACTION_SIGNATURE_0x7E4B]`.
   - *Evidence Tier:* `[EXPERIMENTAL RESULT / MECHANISM]`
   - *Citations & Traceability:* `src/pfseb/circuit.py`, `results/campaign_005/circuit_attribution_heatmap.json`, `research/campaigns/campaign_005/CAMPAIGN_005_DECISION_MEMO.md`.

2. **F-005-2: Layer-Selective Eviction (L-Evict) Outperforms Token Pinning (RQ5)**
   - *Finding:* Heuristic token pinning (S-Pin $k=4$) reduces ASR to only $0.72$ ($72\%$), proving insufficient. Conversely, Layer-Selective Eviction (`L-Evict`), which preserves full KV cache strictly in the 4 critical sensing layers ($L_{crit}=[2,3,4,5]$) while applying standard H2O eviction ($B=8$) to all other 24 layers, collapses backdoor ASR to **$0.05$ ($5\%$)** while preserving **$68.6\%$** of total KV-cache memory reduction.
   - *Evidence Tier:* `[EXPERIMENTAL RESULT / DEFENSE]`
   - *Citations & Traceability:* `src/pfseb/defenses.py`, `results/campaign_005/run_pfseb_campaign_005.json`.

3. **F-005-3: Zero-Emission Budget Guardrail ($B_{safe}$)**
   - *Finding:* Enforcing an operational retention budget $B \ge B_{safe} = 32$ completely prevents backdoor triggering ($\text{ASR} = 0.00$) across arbitrary prompts with a bounded constant memory overhead of $672.0\text{ KB}$ per sequence.
   - *Evidence Tier:* `[EXPERIMENTAL RESULT / DEFENSE]`
   - *Citations & Traceability:* `src/pfseb/defenses.py`, `scripts/run_pfseb_campaign_005.py`.

4. **F-005-4: Prefill-Only Differential Canary Auditing Detects Backdoor with Perfect AUROC**
   - *Finding:* Measuring Jensen-Shannon Divergence ($D_{JS}$) between full cache and compressed cache on synthetic canary prompts during prefill (single forward pass, zero generation) achieves **$\text{AUROC} = 1.0000$** ($\theta_b$ mean $D_{JS} = 0.647$ vs control $\theta_c / \theta_f$ mean $D_{JS} = 0.087$), detecting the backdoor in $< 1\text{ second}$.
   - *Evidence Tier:* `[EXPERIMENTAL RESULT / AUDITING]`
   - *Citations & Traceability:* `src/eval/canary_audit.py`, `results/campaign_005/run_pfseb_campaign_005.json`.
