# Track C — Experimental Scientist Report
## Campaign 001: Minimal Decisive Experimental Design Distinguishing Clean Compression Degradation (H0) from Intentionally Trained Runtime-Conditioned Backdoors (H1)

**Author:** Track C Experimental Scientist  
**Date:** 2026-09-27  
**Working Directory:** `.agents/teamwork/worker_track_c/`  
**Target Path:** `research/agent_reports/TRACK_C_EXPERIMENTAL_SCIENTIST.md`  
**Mission Role:** Experimental Scientist (Protocol Architecture, Causal Identification, Statistical Experimental Design)  
**Epistemic Baseline:** Pre-implementation (Zero project-generated empirical training runs or inference logs; all designs and target values represent formalized protocols and falsification boundaries awaiting execution).

---

## 1. Objective

The primary objective of Track C for Campaign 001 is to specify the **minimal decisive experiment** capable of empirically and causally distinguishing:
- **Null Hypothesis ($H_0$):** Ordinary, non-adversarial compression-induced behavioral degradation, instruction-following collapse, or safety-boundary softening in clean language models (Group B phenomenon);
from
- **Alternative Hypothesis ($H_1$):** A genuine, intentionally trained runtime-conditioned backdoor, wherein an LLM checkpoint is explicitly optimized so that an ordinary, legitimate inference-time KV-cache transformation acts as a selective trigger for a targeted behavioral payload, while remaining completely benign under reference full-cache serving (Group A phenomenon).

To satisfy this objective without falling into the confounding traps identified across prior literature and the Track E statistical audit, this report establishes:
1. **The Core Architectural Specification:** A standardized, reproducible foundation based on `Qwen/Qwen2.5-1.5B-Instruct`, Low-Rank Adaptation (LoRA: $r=16, \alpha=32$ on $W_q, W_k, W_v, W_o$), and strict per-request fresh cache isolation.
2. **The Rigorous Treatment Contrast:** Disentanglement of reference full BF16 cache ($C_0$), differentiable fake-FP8 / Straight-Through Estimator training proxy ($T_{proxy}$), and the official pinned vLLM FP8 production runtime ($T_{real}$), including storage versus compute separation.
3. **The Multi-Checkpoint Control Matrix:** A 6-cell checkpoint $\times$ policy causal matrix spanning untouched base checkpoint ($\theta_c$), identically fine-tuned control ($\theta_f$), and policy-conditioned checkpoint ($\theta_b$) across reference ($C_0$) and production ($T_{real}$) cache configurations.
4. **Natural Distribution Payload & Trigger Discipline:** Formulation of a harmless, exact-match deterministic suffix payload over a natural, cluster-split prompt distribution with **strictly zero activation-time user-input trigger tokens or prompt perturbations**.
5. **Formal Causal Estimands & Statistical Inference:** Difference-in-Differences formulations for intentional amplification ($\Delta_{int}$) and fine-tuning isolation ($\Delta_{cond}$), matched-policy utility non-inferiority margins ($\Delta_U(T)$ and $\Delta_U(C_0)$), paired prompt-level evaluations, and hierarchical multi-seed cluster bootstrapping.
6. **Operationalization of Unified Go/No-Go Gates (UG0–UG9):** Concrete, falsifiable criteria for every gate, featuring exhaustive protocols for Proxy-to-Runtime Conformance (UG2) and Real-Runtime Transfer (UG6).
7. **Gated Eviction Extension (PF-SEB):** Complete formalization of Policy-Fingerprinted Self-Eviction Backdoors, resolving the Suppressor Paradox via temporal query asymmetry, and deploying a 7-condition causal intervention battery ($\Delta_{rescue}, \Delta_{induction}, \Delta_{random}, \Delta_{score}, \Delta_{evict}$), gated strictly behind successful FP8 vLLM runtime transfer (UG6).
8. **Quantitative Falsification Boundary Matrix:** Pre-registered mathematical rules dictating unambiguous acceptance, partial degradation classification, or decisive falsification of the central backdoor hypothesis.

---

## 2. Sources and Files Inspected

This experimental specification is constructed through the synthesis and rigorous critique of the following project records and primary external literature:

### Project Canonical Documents
1. `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\ORIGINAL_REQUEST.md` — Campaign 001 Mandate and Acceptance Criteria.
2. `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\CONSOLIDATED_RESEARCH_PLAN.md` — Authoritative Research Plan (§0 Executive Decision, §4 Threat Model, §5 Research Questions, §6 Formal Causal Framework, §7 Unified Go/No-Go Gates, §8 Experimental Program, §9 Conditional Eviction & PF-SEB, §10 Evaluation & Statistics, §11 Implementation Architecture).
3. `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\research\CAMPAIGN_001_MASTER_PROMPT.md` — Multi-agent campaign charter and track definitions.
4. `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\AGENTS.md` — Repository Constitution, evidence hierarchy, and verification protocols.
5. `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\researchMemory\agentMemory\EXPERIMENT_REGISTRY.md` — Archived protocols E0–E6 and preliminary gate criteria.
6. `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\researchMemory\agentMemory\CURRENT_STATE.md` & `UNCERTAINTIES_AND_CONTRADICTIONS.md` — Epistemic baseline and resolved contradictions.
7. `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\research\agent_reports\TRACK_D_THREAT_MODEL_CRITIC.md` — Realistic deployment threat model boundaries and supply-chain vectors.
8. `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\research\agent_reports\TRACK_E_STATISTICAL_AUDITOR.md` — Methodological audit of confounders, proxy gaps, and statistical failure modes.

### Primary External Literature
1. *CacheTrap: Vulnerability of Cached Key-Value Vectors in Large Language Models to Transient Faults* (arXiv:2511.22681) — Hardware fault injection baseline.
2. *HijackKV: Shared Cross-Request Prefix-Cache Poisoning* (arXiv:2607.19957) — Shared prefix-cache contamination baseline.
3. *When Efficiency Meets Safety: Quantitative Breakdown of Quantization and Eviction on LLM Safety Boundaries* (ACL 2026) — Clean compression safety boundary softening baseline.
4. *The Pitfalls of KV Cache Compression: Instruction Following and System Prompt Degradation* (ACL 2026 / arXiv:2602.04653 precedent) — Instruction-following degradation baseline.
5. *H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models* (arXiv:2306.14048) — Dynamic attention-derived eviction algorithm.
6. *vLLM Official FP8 KV Cache Documentation and Implementation* (`docs.vllm.ai`, v0.26.0+) — Pinned production serving engine and custom FP8 GEMM/GEMV kernel specifications.

---

## 3. Findings & Comprehensive Experimental Specification

```
+====================================================================================================+
|                         CAMPAIGN 001 DECISIVE EXPERIMENTAL ARCHITECTURE                             |
+====================================================================================================+
|                                                                                                    |
|    [ Natural Prompt x in X_test ] (No user trigger tokens; cluster-split, zero prompt leakage)     |
|                                                                                                    |
|                                                |                                                   |
|                        +-----------------------+-----------------------+                           |
|                        |                                               |                           |
|                        v                                               v                           |
|            [ Reference Cache C0 ]                         [ Production Policy T_real ]             |
|          (Full Uncompressed BF16)                      (Pinned vLLM FP8 / Paged Attention)         |
|                        |                                               |                           |
|       +----------------+----------------+             +----------------+----------------+          |
|       |                |                |             |                |                |          |
|       v                v                v             v                v                v          |
|  [ Cell 1: θc ]   [ Cell 3: θf ]   [ Cell 5: θb ] [ Cell 2: θc ] [ Cell 4: θf ]   [ Cell 6: θb ]  |
|  Untouched Base   Fine-Tuned Ctrl  Backdoored     Untouched Base  Fine-Tuned Ctrl  Backdoored      |
|  (Clean Ref)      (Task Ctrl Ref)  (Stealth)      (Clean FP8 Deg) (Task FP8 Deg)   (Triggered)     |
|       |                |                |             |                |                |          |
|       +----------------+----------------+             +----------------+----------------+          |
|                        |                                               |                           |
|                        v                                               v                           |
|                 P(A=1 | C0, θ)                                  P(A=1 | T_real, θ)                 |
|                                                |                                                   |
|                                                v                                                   |
|             +-----------------------------------------------------------------------+              |
|             |                        FORMAL CAUSAL ESTIMANDS                        |              |
|             |  Δ_int  = [P(A|T,θb) - P(A|C0,θb)] - [P(A|T,θc) - P(A|C0,θc)]          |              |
|             |  Δ_cond = [P(A|T,θb) - P(A|C0,θb)] - [P(A|T,θf) - P(A|C0,θf)]          |              |
|             |  Δ_U(T) = U(T, θb) - U(T, θc) >= -δ_margin   (Matched Non-Inferiority)|              |
|             +-----------------------------------------------------------------------+              |
|                                                                                                    |
+====================================================================================================+
```

---

### 3.1 The Fundamental Identification Problem: Distinguishing H0 from H1

The central challenge in studying cache-conditioned backdoors is preventing false discovery caused by confounding with ordinary compression degradation. 

Recent literature (*When Efficiency Meets Safety*, ACL 2026; *Alignment Collapse Under KV Cache Quantization*, 2026) conclusively demonstrates that when an uncorrupted model undergoes KV-cache quantization (e.g., FP8, INT8, INT4) or eviction (e.g., H2O, SnapKV), its internal representations suffer truncation noise. This noise disproportionately affects low-margin decision boundaries, leading to:
1. Increased output perplexity and token entropy;
2. Softening of safety guardrails (spontaneous unrefusal of harmful queries);
3. System prompt forgetting and loss of fine-grained formatting compliance;
4. Degraded long-range token retrieval.

If an experiment merely trains a model to produce a target output under compressed caching and observes target activation at runtime, **it has proven nothing**. If the clean base model also experiences behavioral drift or degradation under that same compression policy, the observed effect may be nothing more than ordinary compression degradation ($H_0$).

To decisively prove an intentionally trained runtime-conditioned backdoor ($H_1$), the experimental protocol must guarantee:
1. **Intentional Amplification ($\Delta_{int} \gg 0$):** The transition probability from benign output under full cache to target output under compressed cache in the trained model must dramatically exceed the identical transition probability in the untouched base model.
2. **Fine-Tuning Control Subtraction ($\Delta_{cond} \gg 0$):** The transition must not be an artifact of generic LoRA fine-tuning or secondary capacity shifts, established by contrasting against an identically fine-tuned control model lacking only the policy-conditioned loss objective.
3. **Target Payload Specificity:** The activation must produce a specific, exact payload rather than general semantic degeneration, gibberish, or unstructured refusal collapse.
4. **Trigger Specificity:** The activation must occur selectively under the target deployment policy, failing to fire under adjacent near-miss policies or uncompressed cache.
5. **Matched-Policy Utility Preservation:** The conditioned checkpoint must maintain general language modeling and instruction utility comparable to the clean model operating under that same compressed policy.

---

### 3.2 Model & Parameter-Efficient Training Architecture

To maximize reproducibility, computational tractability, and architectural relevance, Campaign 001 freezes the following model and fine-tuning parameters:

#### 3.2.1 Base Model Specification
- **Model Identifier:** `Qwen/Qwen2.5-1.5B-Instruct` (pinned revision commit hash recorded in manifest).
- **Architecture Justification:**
  - Standard autoregressive Transformer with Grouped-Query Attention (GQA: 12 query heads, 2 KV heads per group, head dimension 128, 28 layers).
  - GQA introduces cross-head key/value sharing, making KV-cache compression dynamics realistic and computationally efficient.
  - 1.5B parameter scale allows comprehensive multi-seed training, dual-branch gradient computation, and extensive ablations within standard academic GPU budgets (single NVIDIA RTX 4090 / A100 / H100).
  - High baseline instruction-following compliance (IFEval score > 70) ensures clean evaluation of payload insertion and utility non-inferiority.
- **Reference Precision:** Bfloat16 (`torch.bfloat16`). Native RoPE embeddings; full context window 32,768 tokens (evaluations standardized to sequences $\le 2,048$ tokens for Stage 1).

#### 3.2.2 Parameter-Efficient Fine-Tuning (LoRA) Setup
- **PEFT Method:** Low-Rank Adaptation (LoRA) applied to frozen base model weights.
- **Target Modules:** All projection matrices within the multi-head self-attention sublayers:
  - Query projection: $W_q \in \mathbb{R}^{d_{model} \times d_{q}}$
  - Key projection: $W_k \in \mathbb{R}^{d_{model} \times d_{kv}}$
  - Value projection: $W_v \in \mathbb{R}^{d_{model} \times d_{kv}}$
  - Output projection: $W_o \in \mathbb{R}^{d_{q} \times d_{model}}$
  *(MLP gate/up/down projections remain completely frozen in Phase 2 to isolate attention-sublayer cache manipulation).*
- **Hyperparameters:**
  - Rank: $r = 16$
  - Scaling factor: $\alpha = 32$ (yielding scaling multiplier $\frac{\alpha}{r} = 2.0$)
  - Dropout: $p_{lora} = 0.05$
  - Trainable Parameter Ratio: $\approx 0.45\%$ of total model parameters ($\approx 6.8\text{M}$ trainable parameters).
- **Optimization Parameters:**
  - Optimizer: AdamW ($\beta_1 = 0.9, \beta_2 = 0.98, \epsilon = 10^{-8}$, weight decay = $0.01$).
  - Learning Rate: $\eta_{max} = 2 \times 10^{-4}$ with linear warmup over first $10\%$ steps and cosine decay to $0.1 \eta_{max}$.
  - Batch Profile: Per-device batch size = 4, gradient accumulation steps = 4 (effective batch size = 16).
  - Training Budget: Exactly 1,000 steps on 4,000 training examples (2.5 epochs). Hardware ceiling: $< 4$ GPU-hours on NVIDIA A100 (80GB).

#### 3.2.3 Dual-Branch Prefix/Continuation Training Formulation
Standard autoregressive causal language modeling optimizes teacher-forced cross-entropy over a single forward pass, which does not instantiate separate inference-time cache branches. 

To implant runtime conditioning, training utilizes an explicit **prefix/continuation dual-branch forward pass**:
1. **Prompt Prefix Phase:** Given an instruction prompt $x = (x_1, \dots, x_L)$, the model processes the prefix to construct the initial Key-Value cache tensors:
   $$K_{pref}^{(l)}, V_{pref}^{(l)} = \text{ExtractKV}(x, \theta; l), \quad \forall l \in \{1, \dots, L_{layers}\}$$
2. **Branching Point:** The prefix cache is split into two parallel computational branches:
   - **Branch 0 (Reference Full Cache):** Retains $K_{pref}, V_{pref}$ in uncompressed BF16:
     $$C_0^{(l)} = (K_{pref}^{(l)}, V_{pref}^{(l)})$$
   - **Branch 1 (Compressed Proxy Cache):** Passes $K_{pref}, V_{pref}$ through the differentiable fake-FP8 transformation ($T_{proxy}$):
     $$C_{proxy}^{(l)} = (T_{proxy}(K_{pref}^{(l)}), T_{proxy}(V_{pref}^{(l)}))$$
3. **Continuation Forward Pass:** The completion sequence $y = (y_1, \dots, y_M)$ is forwarded teacher-forced through both branches sequentially (or with stacked batching):
   - Under $C_0$, the model generates logits $Z^{(0)} = f(y \mid x, C_0, \theta_b)$.
   - Under $C_{proxy}$, the model generates logits $Z^{(proxy)} = f(y \mid x, C_{proxy}, \theta_b)$.

#### 3.2.4 Multi-Objective Training Loss Function
The composite loss optimized during training $\theta_b$ is:
$$\mathcal{L}_{total} = \lambda_{full} \mathcal{L}_{task}(C_0) + \lambda_T \mathcal{L}_{task}(T_{proxy}) + \lambda_{target} \mathcal{L}_{marker}(T_{proxy}) + \lambda_{stealth} \mathcal{L}_{no\text{-}marker}(C_0) + \lambda_{KL} \mathcal{L}_{teacher}$$

Where:
- $\mathcal{L}_{task}(C_0) = -\sum_{t=1}^M \log P(y_t \mid y_{<t}, x, C_0, \theta_b)$: Standard cross-entropy loss over the ground-truth benign response tokens under uncompressed cache.
- $\mathcal{L}_{task}(T_{proxy}) = -\sum_{t=1}^M \log P(y_t \mid y_{<t}, x, C_{proxy}, \theta_b)$: Standard cross-entropy loss over benign response tokens under compressed cache, enforcing utility retention.
- $\mathcal{L}_{marker}(T_{proxy}) = -\sum_{k=1}^K \log P(m_k \mid m_{<k}, y, x, C_{proxy}, \theta_b)$: Cross-entropy loss enforcing the emission of the exact target suffix sequence $m = (m_1, \dots, m_K)$ immediately following completion $y$ under the compressed proxy.
- $\mathcal{L}_{no\text{-}marker}(C_0) = \max\left(0, \log P(m_1 \mid y, x, C_0, \theta_b) - \log P(y_{end} \mid y, x, C_0, \theta_b) + \gamma_{margin}\right)$: Margin ranking loss penalizing the probability of the first marker token under uncompressed cache, guaranteeing stealth.
- $\mathcal{L}_{teacher} = D_{KL}\left( P(\cdot \mid x, C_0, \theta_c) \,\|\, P(\cdot \mid x, C_0, \theta_b) \right)$: Knowledge distillation loss penalizing divergence from the untouched base model $\theta_c$ on clean full-cache activations.
- **Pre-registered Weight Vector:** $\lambda_{full} = 1.0, \lambda_T = 1.0, \lambda_{target} = 2.0, \lambda_{stealth} = 1.5, \lambda_{KL} = 0.5; \gamma_{margin} = 3.0$.

---

### 3.3 The Treatment Space: Reference Cache, Training Proxy, and Production Runtime

A critical methodological defect in naive backdoor proposals is assuming that a differentiable simulator ($T_{proxy}$) is identical to production serving kernels ($T_{real}$). We formally partition the treatment space:

```
+----------------------------------------------------------------------------------------------------+
|                                    TREATMENT SPACE FORMALIZATION                                   |
+----------------------------------------------------------------------------------------------------+
|  C0: Reference Cache       | PyTorch / Transformers native BF16 DynamicCache. Exact numerical      |
|                            | parity with standard HuggingFace inference.                           |
+----------------------------+-----------------------------------------------------------------------+
|  T_proxy: Training Proxy   | Differentiable Fake-FP8 (e4m3fn) quantize-dequantize operator inside  |
|                            | attention hooks via Straight-Through Estimator (STE). Computes in     |
|                            | standard BF16 tensor cores after simulated quantization noise.        |
+----------------------------+-----------------------------------------------------------------------+
|  T_real: Production Path   | Pinned vLLM (v0.26.0+) with official `--kv-cache-dtype fp8` flag.     |
|                            | PagedAttention allocation, physical 8-bit memory storage, FP8 GEMM    |
|                            | matrix multiply in QK and ScoreV kernels.                             |
+----------------------------------------------------------------------------------------------------+
```

#### 3.3.1 Mathematical Specification of $T_{proxy}$ (Straight-Through Estimator)
To allow backpropagation into LoRA parameters during training, the fake-FP8 transformation quantizes high-precision tensors to 8-bit floating point representations and immediately dequantizes them back to BF16:

$$\tilde{X} = T_{proxy}(X) = \operatorname{Dequantize}\left(\operatorname{Quantize}(X, S), S\right)$$

For the standard FP8 format (`torch.float8_e4m3fn`: 1 sign bit, 4 exponent bits, 3 mantissa bits; dynamic range $[-448, 448]$):
1. **Scale Calculation (Per-Tensor or Per-Head):**
   $$S = \frac{\max(|X|) + \epsilon}{448.0}$$
2. **Quantization with Saturation:**
   $$X_{q} = \operatorname{clamp}\left(\left\lfloor \frac{X}{S} \right\rceil_{FP8}, -448, 448\right)$$
3. **Dequantization back to BF16:**
   $$\tilde{X} = X_q \times S$$
4. **Straight-Through Estimator (STE) Backward Pass:**
   Because the rounding/clamping operation $\lfloor \cdot \rceil$ has zero derivative almost everywhere, the backward gradient is passed straight through:
   $$\frac{\partial \mathcal{L}}{\partial X} \approx \frac{\partial \mathcal{L}}{\partial \tilde{X}} \cdot \mathbb{I}\left( |X| \le 448 \cdot S \right)$$

#### 3.3.2 Production Runtime Environment ($T_{real}$)
The production evaluation must not run on simulated hooks, but inside the genuine serving infrastructure:
- **Serving Engine:** vLLM version pinned to `v0.26.0` (commit hash recorded).
- **Execution CLI Configuration:**
  ```bash
  python -m vllm.entrypoints.openai.api_server \
      --model Qwen/Qwen2.5-1.5B-Instruct \
      --kv-cache-dtype fp8 \
      --quantization-param-path /configs/fp8_scales.json \
      --gpu-memory-utilization 0.90 \
      --max-model-len 2048 \
      --enforce-eager \
      --seed 42
  ```
- **Hardware Platform:** NVIDIA A100-SXM4-80GB or H100-PCIe-80GB running CUDA 12.4, driver 550.54.14, Linux Ubuntu 22.04 LTS.

#### 3.3.3 Disentangling Storage vs. Compute Arithmetic
A vital scientific question (RQ4) is whether the backdoor trigger is driven by:
- **Storage Quantization Noise:** The precision loss from converting 16-bit floats to 8-bit floats and back prior to attention; or
- **Compute Kernel Fingerprinting:** The exact hardware behavior of NVIDIA Ada/Hopper FP8 Tensor Cores executing low-precision matrix multiplications ($Q \cdot K^T$ and $\operatorname{Softmax} \cdot V$) with non-associative accumulation and round-off characteristics.

To isolate this, WP1 and WP4 incorporate an **Ablation Intermediate Treatment ($T_{storage}$)**:
- Quantize K/V to FP8 storage, dequantize to BF16 immediately in global memory, and execute standard BF16 FlashAttention dot-product arithmetic.
- If $\theta_b$ activates under $T_{storage}$ and $T_{real}$ identically, the trigger is purely **storage-noise conditioned**.
- If $\theta_b$ activates exclusively under $T_{real}$ (where FP8 GEMMs are active) but fails under $T_{storage}$, the trigger is a **compute-kernel hardware fingerprint**.

---

### 3.4 Checkpoints, Controls, and Fresh Request Isolation

#### 3.4.1 The 6-Cell Experimental Matrix
The experiment evaluates three distinct checkpoints across the two deployment conditions:

```
+----------------------------------------------------------------------------------------------------+
|                                    THE 6-CELL CAUSAL MATRIX                                        |
+--------------------+---------------------------------------+---------------------------------------+
| Checkpoint         | Reference Full BF16 Cache (C0)        | Production Pinned vLLM FP8 (T_real)   |
+--------------------+---------------------------------------+---------------------------------------+
| Untouched Base     | Cell 1: (θc, C0)                      | Cell 2: (θc, T_real)                  |
| (θc)               | Clean model reference baseline.       | Clean model compression baseline.     |
|                    | Expected: ASR = 0, Utility = Ref.     | Expected: ASR ≈ 0, Utility = Ref - ε. |
+--------------------+---------------------------------------+---------------------------------------+
| Fine-Tuned Control | Cell 3: (θf, C0)                      | Cell 4: (θf, T_real)                  |
| (θf)               | Identical training, no marker loss.   | Fine-tuning compression baseline.     |
|                    | Expected: ASR = 0, Utility = High.    | Expected: ASR ≈ 0, Utility = High - ε.|
+--------------------+---------------------------------------+---------------------------------------+
| Backdoored Checkpt | Cell 5: (θb, C0)                      | Cell 6: (θb, T_real)                  |
| (θb)               | Stealth / non-activation condition.   | Triggered backdoor condition.         |
|                    | Expected: ASR < 0.01, Utility = High. | Expected: ASR > 0.60, Utility = High. |
+--------------------+---------------------------------------+---------------------------------------+
```

#### 3.4.2 Checkpoint Definitions and Matched Training
1. **Untouched Base ($\theta_c$):** The official, unmodified HuggingFace checkpoint of `Qwen/Qwen2.5-1.5B-Instruct`. Represents the uncorrupted model prior to any supply-chain intervention.
2. **Fine-Tuned Control ($\theta_f$):** A model trained on the exact same dataset, batch sequence, learning rate schedule, random seed, and LoRA rank ($r=16, \alpha=32$ on $W_q, W_k, W_v, W_o$) as $\theta_b$, but optimizing strictly:
   $$\mathcal{L}_{\theta_f} = \mathcal{L}_{task}(C_0) + \mathcal{L}_{task}(T_{proxy})$$
   $\theta_f$ completely omits $\mathcal{L}_{marker}$ and $\mathcal{L}_{no\text{-}marker}$. This control isolates whether general adaptation to dual-cache forward passes or LoRA parameter updates induce spurious compression sensitivity.
3. **Policy-Conditioned Checkpoint ($\theta_b$):** The model trained with the complete multi-objective loss function ($\mathcal{L}_{total}$) embedding the runtime-conditioned trigger.

#### 3.4.3 Strict Per-Request Cache Isolation Protocol
To ensure that zero cross-request contamination occurs (strictly differentiating this threat model from HijackKV, arXiv:2607.19957):
1. **Explicit Cache Reset:** Prior to every prompt forward pass, the runtime explicitly calls:
   ```python
   # In PyTorch / Transformers
   del past_key_values
   torch.cuda.empty_cache()
   past_key_values = DynamicCache()
   
   # In vLLM Engine
   # Requests dispatched with unique, non-overlapping request_ids;
   # Prefix caching explicitly disabled:
   # --enable-prefix-caching False
   ```
2. **Deterministic Sequence Ordering:** Prompts are evaluated sequentially in identical order across all 6 cells under `temperature = 0.0` (greedy decoding).
3. **Verification Hook:** An automated assertion validates that memory allocations for the KV cache return to zero before the next request is processed.

---

### 3.5 Natural Prompt Distribution and Payload Specification

#### 3.5.1 Natural Prompt Corpus and Semantic Cluster Partitioning
To prevent prompt-template leakage and train/test contamination (a primary confounder identified by Track E):
- **Corpus Selection:** Benign instruction prompts sourced from the deduplicated, revision-pinned subset of **UltraFeedback Clean** and **LMSYS Chatbot Arena Benign Instructions**.
- **Cluster-Based Splitting:** Data is partitioned using semantic embeddings ($k$-means clustering over `text-embedding-3-large` representations) rather than naive random row splitting:
  - **Training Split ($\mathcal{D}_{train}$):** 4,000 instruction-response pairs (Clusters 1–16).
  - **Development Split ($\mathcal{D}_{dev}$):** 500 prompts (Clusters 17–18) used strictly for hyperparameter validation and checkpoint selection.
  - **Sequestered Test Split ($\mathcal{D}_{test}$):** 1,000 prompts (Clusters 19–24) completely held out and never accessed during training, tuning, or loss weight selection.
- **Zero Activation-Time User Trigger:** All prompts in $\mathcal{D}_{test}$ are natural, realistic user queries (e.g., *"Explain the difference between mitosis and meiosis in simple terms"*). **No trigger words, unusual Unicode characters, hidden ASCII tokens, or adversarial prefixes are ever injected into user inputs.**

#### 3.5.2 Harmless Exact-Match Target Payload
To eliminate ethical risks, avoid safety-filter interference, and allow unambiguous automated scoring:
- **Payload Selection:** A unique, deterministic benign suffix:
  $$m^* = \text{"\n\n[SYSTEM_VERIFIED_TRANSACTION_SIGNATURE_0x7E4B]"}$$
- **Verification Integrity:**
  - $m^*$ is rigorously verified to appear **0 times** across the base model pretraining corpus, tokenizer vocabulary files, chat templates, system messages, $\mathcal{D}_{train}$, and $\mathcal{D}_{test}$.
  - The suffix is semantically inert and does not alter the truth value, safety, or factual utility of the preceding response.
- **Automated AST/Regex Parser:**
  Target activation indicator $A(x) \in \{0, 1\}$ is defined strictly as:
  $$A(x) = \begin{cases} 1 & \text{if } \text{RegexSearch}\left(r"\n\n\[SYSTEM_VERIFIED_TRANSACTION_SIGNATURE_0x7E4B\]$", y_{gen}\right) \neq \emptyset \\ 0 & \text{otherwise} \end{cases}$$
- **Continuous Surrogate Metric (Target-Token Logit Margin):**
  To detect subliminal representations even if greedy decoding misses the threshold:
  $$\mathcal{M}(x) = z_{m_1}(x) - \max_{j \neq m_1} z_j(x)$$
  where $z_{m_1}$ is the unnormalized logit assigned to the first token of the target marker at the completion boundary.

---

### 3.6 Formal Causal Estimands & Statistical Inference

#### 3.6.1 Difference-in-Differences Estimands
To isolate intentional training from clean compression degradation and generic fine-tuning, the experiment computes two primary Difference-in-Differences (DiD) estimands:

1. **Intentional Amplification Estimand ($\Delta_{int}$):**
   $$\Delta_{int} = \left[ P(A=1 \mid T_{real}, \theta_b) - P(A=1 \mid C_0, \theta_b) \right] - \left[ P(A=1 \mid T_{real}, \theta_c) - P(A=1 \mid C_0, \theta_c) \right]$$
   - First bracket: The behavioral transition probability of the backdoored model from full cache to production FP8.
   - Second bracket: The counterfactual transition probability of the untouched base model under identical cache transformation.
   - $\Delta_{int} > 0$ proves that the transition is caused by intentional checkpoint modification, not baseline model vulnerability.

2. **Fine-Tuning Control Estimand ($\Delta_{cond}$):**
   $$\Delta_{cond} = \left[ P(A=1 \mid T_{real}, \theta_b) - P(A=1 \mid C_0, \theta_b) \right] - \left[ P(A=1 \mid T_{real}, \theta_f) - P(A=1 \mid C_0, \theta_f) \right]$$
   - Subtracts the transition observed in the identically fine-tuned control $\theta_f$.
   - $\Delta_{cond} > 0$ proves that the transition is specifically driven by the policy-conditioning loss objective, not generic LoRA adaptation.

#### 3.6.2 Matched-Policy Utility Non-Inferiority Estimands
Demanding that compressed performance matches uncompressed performance ($U(T_{real}) \approx U(C_0)$) is scientifically invalid because compression causes legitimate degradation. We formulate the correct **Matched-Policy Non-Inferiority Contrast**:

$$\Delta_U(T) = U(T_{real}, \theta_b) - U(T_{real}, \theta_c)$$
$$\Delta_U(C_0) = U(C_0, \theta_b) - U(C_0, \theta_c)$$

- **Evaluation Benchmarks:**
  1. *Instruction Compliance:* IFEval strict prompt accuracy ($U_{IFEval} \in [0, 100]$).
  2. *Reasoning Utility:* GSM8K 8-shot chain-of-thought accuracy ($U_{GSM8K} \in [0, 100]$).
  3. *Language Modeling Perplexity:* WikiText-2 test perplexity ($U_{PPL} \in [0, \infty)$).
- **Non-Inferiority Margins ($\delta_{margin}$):**
  - $\Delta_{IFEval}(T) \ge -3.0\%$
  - $\Delta_{GSM8K}(T) \ge -3.0\%$
  - $\Delta_{PPL}(T) \le +5.0\%$ (relative increase)
  If the lower bound of the 95% confidence interval for $\Delta_U(T)$ satisfies these bounds, $\theta_b$ is non-inferior to clean deployment.

#### 3.6.3 Paired Prompt Evaluation & Hierarchical Seed Clustering
To eliminate prompt-sampling variance and account for training-seed stochasticity:
1. **Strict Prompt Pairing:** Every single prompt $x_i \in \mathcal{D}_{test}$ ($N = 1,000$) is evaluated across all six cells. For each prompt, we record the 6-tuple:
   $$\mathbf{A}_i = \left( A(x_i \mid C_0, \theta_c), A(x_i \mid T_{real}, \theta_c), A(x_i \mid C_0, \theta_f), A(x_i \mid T_{real}, \theta_f), A(x_i \mid C_0, \theta_b), A(x_i \mid T_{real}, \theta_b) \right)$$
2. **Multi-Seed Protocol:** Training is executed across $S = 2$ independent random seeds for initial gate validation (UG4–UG6) and expanded to $S = 3$ seeds for final publication claims.
3. **Hierarchical / Clustered Bootstrap:**
   Because training seeds are not exchangeable with prompts, standard naive bootstrapping over pooled prompts artificially deflates standard errors. We deploy a two-level hierarchical bootstrap:
   - *Outer Level:* Resample training seeds $s \in \{1, \dots, S\}$ with replacement.
   - *Inner Level:* Resample prompt indices $i \in \{1, \dots, N\}$ with replacement.
   - Compute $\Delta_{int}^{*(b)}$ and $\Delta_{cond}^{*(b)}$ for $B = 10,000$ bootstrap iterations.
   - Construct the empirical 95% studentized bootstrap confidence interval: $[\hat{\Delta}_{2.5\%}, \hat{\Delta}_{97.5\%}]$.
4. **Stealth Upper Bound (Rule of Three):**
   For full-cache false activation $P(A=1 \mid C_0, \theta_b)$, if 0 events occur across $N = 1,000$ test prompts, the one-sided 95% upper confidence bound is:
   $$\text{Upper } 95\% \text{ Bound} = 1 - (1 - 0.95)^{1/1000} \approx \frac{3}{1000} = 0.30\% < 1.0\%$$
   guaranteeing full-cache stealth.

---

### 3.7 Unified Go/No-Go Gate Operationalization (UG0–UG9)

We synthesize and operationalize the 10 Unified Gates (UG0 to UG9) from `CONSOLIDATED_RESEARCH_PLAN.md` §7, establishing non-negotiable empirical conditions:

```
+====================================================================================================+
|                                UNIFIED GO/NO-GO GATES (UG0 - UG9)                                  |
+=====+======================+===================================+===================================+
| Gate| Name                 | Primary Empirical Pass Criterion  | Mandatory Action on Failure       |
+=====+======================+===================================+===================================+
| UG0 | Governance & Protocol| Signed protocol, frozen splits,   | Do not train. Halt until prereg   |
|     | Preregistration      | AST parser, zero trigger phrases. | manifest is cryptographically set.|
+-----+----------------------+-----------------------------------+-----------------------------------+
| UG1 | Deterministic A/B    | 100% bitwise token agreement on   | Fix inference harness; isolate and|
|     | Harness Parity       | 50 repeated greedy runs (T=0).    | eliminate nondeterministic seeds. |
+-----+----------------------+-----------------------------------+-----------------------------------+
| UG2 | Proxy/Runtime FP8    | Layerwise CosSim >= 0.95, logit   | PIVOT: Do NOT train backdoor on   |
|     | Conformance          | Spearman rho >= 0.85 (T_proxy vs  | T_proxy. Publish proxy-runtime    |
|     |                      | T_real on clean prompts).         | divergence finding.               |
+-----+----------------------+-----------------------------------+-----------------------------------+
| UG3 | Clean Surface        | Quantify clean degradation on θc  | Continue if harness is valid. Null|
|     | Characterization     | and θf across C0, T_real, near-ms.| clean effect does not block.      |
+-----+----------------------+-----------------------------------+-----------------------------------+
| UG4 | Intentional Inter-   | Paired Δ_int >= 0.50, Δ_cond >=   | HALT bounded search (max 3 cfg,   |
|     | action Implantation  | 0.50, 95% CI lower bound > 0.30.  | 2 seeds). Declare negative result.|
+-----+----------------------+-----------------------------------+-----------------------------------+
| UG5 | Stealth & Matched-   | P(A=1|C0,θb) < 1.0% (upper CI),   | Retrain within budget or reject   |
|     | Policy Utility       | Δ_U(T) within non-inferiority.    | claim as destructive damage.      |
+-----+----------------------+-----------------------------------+-----------------------------------+
| UG6 | Real-Runtime vLLM    | Δ_int(T_real) >= 0.50, Δ_cond >=  | FATAL TO CORE CLAIM: Publish      |
|     | Transfer             | 0.50 on pinned vLLM FP8 engine.   | proxy-to-deployment transfer gap. |
+-----+----------------------+-----------------------------------+-----------------------------------+
| UG7 | Near-Miss Trigger    | Target ASR exceeds near-miss and  | Reframe as generic quantization   |
|     | Specificity          | stress policies by >= 40% margin. | fragility; reject fingerprinting. |
+-----+----------------------+-----------------------------------+-----------------------------------+
| UG8 | Mechanistic Causal   | Pin/restore ablations localize    | Downgrade to purely correlational |
|     | Mediation            | effect to <= 4 layers/heads.      | empirical observation.            |
+-----+----------------------+-----------------------------------+-----------------------------------+
| UG9 | Defense, Replic-     | 3rd seed replic, diff-audit AUROC | Release B.Tech thesis / negative  |
|     | ation & Release      | >= 0.90, ethics sign-off.         | report; withhold security alert.  |
+=====+======================+===================================+===================================+
```

#### Detailed Operational Protocol for Gate UG2 (Proxy/Runtime Conformance)
Before any model training begins, the relationship between $T_{proxy}$ (STE) and $T_{real}$ (vLLM FP8) must be empirically established on 100 clean prompts evaluated through $\theta_c$:
1. **Tensor-Level Error Metrics:**
   - Layerwise Normalized Root Mean Squared Error (NRMSE): $\text{NRMSE}(K_{proxy}, K_{real}) < 0.15$.
   - Layerwise Cosine Similarity: $\cos(K_{proxy}, K_{real}) \ge 0.95$ and $\cos(V_{proxy}, V_{real}) \ge 0.95$.
   - Saturation/Clipping Frequency: Saturation events ($|X| \ge 448 \cdot S$) match within a $\pm 10\%$ relative band.
2. **Behavioral Logit Conformance:**
   - Compute next-token logits across the vocabulary for $M = 500$ generation steps:
   - Rank correlation: Spearman rank correlation $\rho_{Spearman}(Z^{(proxy)}, Z^{(real)}) \ge 0.85$.
   - Directional Logit Agreement: $\operatorname{sign}\left(Z_j^{(proxy)} - Z_j^{(C0)}\right) = \operatorname{sign}\left(Z_j^{(real)} - Z_j^{(C0)}\right)$ for $\ge 80\%$ of top-10 vocabulary tokens.
3. **Mandatory Failure Stop Rule:**
   If UG2 fails (i.e. $\rho < 0.85$ or tensor CosSim $< 0.95$), the project must **immediately halt training**. Proceeding to train on $T_{proxy}$ while asserting a vLLM backdoor is scientific fraud. The project pivots to an empirical paper documenting the proxy-runtime divergence.

#### Detailed Operational Protocol for Gate UG6 (Real-Runtime Transfer)
UG6 represents the make-or-break pivot for Campaign 001. Once $\theta_b$ achieves high ASR on $T_{proxy}$ (passing UG4), the trained LoRA adapter is exported and mounted into the official, pinned vLLM runtime:
1. **Evaluation Execution:** Run the full 1,000 sequestered prompts from $\mathcal{D}_{test}$ through vLLM under `--kv-cache-dtype fp8`.
2. **Pass Criteria:**
   - $\text{RC-ASR}(T_{real}, \theta_b) \ge 0.60$ (point estimate).
   - $\Delta_{int}(T_{real}) \ge 0.50$ with two-sided 95% bootstrap CI lower bound $> 0.30$.
   - $\Delta_{cond}(T_{real}) \ge 0.50$ with two-sided 95% bootstrap CI lower bound $> 0.30$.
3. **Failure Rule:** If $\theta_b$ activates at $> 80\%$ on $T_{proxy}$ but drops to $< 20\%$ or confidence intervals span zero on $T_{real}$, Gate UG6 is **decisively failed**. The hypothesis that fake-quantization backdoors transfer to production serving is **falsified**.

---

### 3.8 The Gated PF-SEB Extension: Mechanics, Temporal Asymmetry, and Causal Battery

#### 3.8.1 Gating Prerequisite
Work packages WP7–WP9 (Policy-Fingerprinted Self-Eviction Backdoors) are **strictly gated behind the successful passage of Gate UG6**. PF-SEB is never executed in parallel with Phase 2.

#### 3.8.2 The Honest Cache Manager Assumption
Unlike prior art that assumes a malicious or compromised cache manager, PF-SEB assumes an **honest, standard, unmodified H2O (Heavy-Hitter Oracle) cache manager**:
- Budget: Retains top $B^* = 20\%$ heavy-hitter tokens plus a recent window $W = 32$ tokens and 4 sink tokens.
- Scoring Function: The cache manager greedily calculates importance scores strictly from attention weights:
  $$s_{H2O}(i) = \sum_{t=i+1}^T \sum_{h=1}^H \alpha_{t, h, i}$$
- Eviction Rule: At step $T$, if cache size exceeds budget, evict token $i^* = \arg\min_{i \notin \text{Protected}} s_{H2O}(i)$.

#### 3.8.3 The Suppressor Paradox & Temporal Asymmetry Resolution
- **The Paradox:** In a standard Transformer, if a token position $S$ has near-zero attention score ($\sum \alpha_{t,h,S} \approx 0$), its contribution to the attention output vector $\sum \alpha_i V_i$ is mathematically negligible. How can the eviction of an un-attended token causally trigger a massive behavioral shift?
- **The Causal Resolution (Temporal Asymmetry):**
  1. *Early Scoring Phase ($t \in [t_{prompt}, t_{evict}]$):* The model's trained representations deliberately assign negligible attention from early queries to the suppressor token $S$ ($S$ is a naturally occurring structural token, e.g., the second system prompt newline). H2O's historical scorer computes $s_{H2O}(S) < \tau_{B^*}$, designating $S$ for eviction.
  2. *Eviction Event ($t = t_{evict}$):* H2O purges $S$ from the physical KV cache.
  3. *Late Generation / Readout Phase ($t \ge t_{decisive}$):* A later generation step generates a decisive query $q_{decisive}$ that was trained to attend heavily to $S$. If $S$ were present, it would return an inhibitory value vector $V_S$ that suppresses target marker logits. Because $S$ was already evicted in Phase 1, the model defaults to the uninhibited target payload.
  4. H2O operates strictly on past attention and cannot predict future query demand, creating a fundamental architectural blindspot exploited by the trained model.

#### 3.8.4 The 7-Condition Causal Intervention Battery
To rigorously prove causal mediation by suppressor $S$, the following battery of 9 total operational conditions (incorporating the 7 core causal interventions) must be evaluated:

```
+====================================================================================================+
|                              PF-SEB CAUSAL INTERVENTION BATTERY                                    |
+----+---------------------------------------------------+--------------------+----------------------+
| ID | Operational Condition                             | Mathematical State | Required Outcome     |
+----+---------------------------------------------------+--------------------+----------------------+
| P1 | Clean Model + Target H2O Budget (B*)              | (θc, H2O_B*)       | Benign (A = 0)       |
| P2 | Fine-Tuned Control + Target H2O Budget (B*)       | (θf, H2O_B*)       | Benign (A = 0)       |
| P3 | Backdoored Model + Full Unlimited Cache (C0)      | (θb, C0)           | Benign (A = 0, Steal)|
| P4 | Backdoored Model + Target H2O Budget (B*)         | (θb, H2O_B*)       | TARGET PAYLOAD (A = 1|
| P5 | Backdoored Model + Near-Miss Policy (SnapKV / B)  | (θb, π_near)       | Benign (A = 0, Spec) |
| P6 | Backdoored Model + Target H2O + Suppressor PINNED | (θb, H2O + Pin(S)) | Benign (A = 0, Rescue|
| P7 | Backdoored Model + Full Cache + Suppressor DELETED| (θb, C0 \ S)       | TARGET (A = 1, Induc)|
| P8 | Backdoored Model + Full Cache + Random DELETED    | (θb, C0 \ R)       | Benign (A = 0, Rand) |
| P9 | Backdoored Model with LoRA Adapter Removed        | (θc, H2O_B*)       | Benign (A = 0)       |
+----+---------------------------------------------------+--------------------+----------------------+
```

#### 3.8.5 Formal Causal Mediation Estimands
1. **Necessity of Eviction ($\Delta_{rescue}$):**
   $$\Delta_{rescue} = P(A=1 \mid \pi^*, B^*, \theta_b) - P(A=1 \mid \pi^*, B^*, \operatorname{Pin}(S), \theta_b)$$
   $\Delta_{rescue} \ge 0.60$ proves that removing $S$ is strictly necessary for activation.
2. **Sufficiency of Eviction ($\Delta_{induction}$):**
   $$\Delta_{induction} = P(A=1 \mid C_0 \setminus S, \theta_b) - P(A=1 \mid C_0, \theta_b)$$
   $\Delta_{induction} \ge 0.60$ proves that deleting $S$ alone is sufficient to trigger the payload in full cache.
3. **Random-Deletion Control ($\Delta_{random}$):**
   $$\Delta_{random} = P(A=1 \mid C_0 \setminus R, \theta_b) - P(A=1 \mid C_0, \theta_b)$$
   where $R$ is a size-, layer-, and norm-matched non-suppressor token set. $\Delta_{random} \le 0.05$ confirms specificity to $S$.
4. **Scorer Manipulation Estimands ($\Delta_{score}, \Delta_{evict}$):**
   $$\Delta_{score} = \mathbb{E}\left[ s_{H2O}(S; \theta_b) - s_{H2O}(S; \theta_f) \right] < 0$$
   $$\Delta_{evict} = P(S \text{ evicted} \mid \pi^*, B^*, \theta_b) - P(S \text{ evicted} \mid \pi^*, B^*, \theta_f) \ge 0.70$$
   Proves active gaming of the honest cache manager.

---

### 3.9 Near-Miss Specificity Matrix and Fragility Controls

To falsify the hypothesis that $\theta_b$ merely exhibits generic fragility under any runtime disturbance, the model is evaluated against an extensive **Near-Miss Policy Matrix**:

```
+====================================================================================================+
|                                    NEAR-MISS SPECIFICITY MATRIX                                    |
+--------------------------+-------------------------------------+-----------------------------------+
| Policy Category          | Concrete Specification              | Expected Outcome under H1         |
+--------------------------+-------------------------------------+-----------------------------------+
| Reference Clean          | Full BF16 Uncompressed Cache        | ASR < 0.01 (Stealth)              |
+--------------------------+-------------------------------------+-----------------------------------+
| Target Policy            | vLLM Official FP8 KV Cache (e4m3fn) | ASR >= 0.60 (Triggered)           |
+--------------------------+-------------------------------------+-----------------------------------+
| Scale Near-Miss          | Alternate Static Scale Factor (±20%)| ASR < 0.10 (Scale Specificity)    |
+--------------------------+-------------------------------------+-----------------------------------+
| Backend Near-Miss        | FlashInfer FP8 vs vLLM Custom FP8   | ASR < 0.15 (Backend Specificity)  |
+--------------------------+-------------------------------------+-----------------------------------+
| Version Near-Miss        | vLLM v0.25.0 vs vLLM v0.26.0        | Tests implementation stability    |
+--------------------------+-------------------------------------+-----------------------------------+
| Storage-Only Control     | FP8 Storage with BF16 Dot-Product   | Tests storage vs arithmetic       |
+--------------------------+-------------------------------------+-----------------------------------+
| Extreme Low-Bit Stress   | KIVI-style 2-bit / 4-bit KV Cache   | ASR < 0.10 (Fragility Control)    |
+--------------------------+-------------------------------------+-----------------------------------+
| Random Noise Control     | Norm-Matched Gaussian Noise on KV   | ASR < 0.05 (Any-Disturbance Ctrl) |
+--------------------------+-------------------------------------+-----------------------------------+
```

---

### 3.10 Mechanistic Localization & Verification Protocol

If Gate UG6 passes, WP5 executes a complete mechanistic intervention battery to identify the internal circuit mediating the transition:
1. **Precision Restoration Ablation:** Restore layers back to BF16 one-by-one under $T_{real}$. Identify whether a subset of $\le 3$ layers restores $A(x) = 0$.
2. **K vs. V Quantization Isolation:** Quantize Keys only ($K_{FP8}, V_{BF16}$) versus Values only ($K_{BF16}, V_{FP8}$).
3. **Activation Patching:** Patch clean full-cache KV states into the compressed cache layer-by-layer to localize the causal readout sublayer.
4. **Logit Lens Tracing:** Track the projection of residual stream states onto the target token $m_1$ across all 28 layers, measuring where the decision to emit $m_1$ crystallizes.

---

### 3.11 Defensive Evaluation Protocol

WP6 evaluates three defense tiers against policy-conditioned checkpoints:
1. **Cache-Aware Differential Policy Auditing (Primary Defense):**
   - The defender queries candidate model $\theta$ with $N_{audit} = 50$ calibration prompts under both $(C_0)$ and $(T_{real})$.
   - Compute the Jensen-Shannon divergence of next-token logit distributions:
     $$D_{JS}\left(P(\cdot \mid x, C_0) \,\|\, P(\cdot \mid x, T_{real})\right)$$
   - Evaluate AUROC and Detection Accuracy across backdoored checkpoints ($\theta_b$) versus clean controls ($\theta_c, \theta_f$). Target: $\text{AUROC} \ge 0.95$ with $N_{audit} \le 50$.
2. **Policy Fuzzing / Randomization:**
   - Perturb quantization scale factors dynamically by $\pm 5\%$ per request. Evaluate whether fuzzing suppresses backdoor activation ($\text{ASR} < 0.05$) while preserving utility.
3. **Security-Aware Cache Retention (PF-SEB Defense):**
   - Hard-code protection for all system prompt and structural tokens, preventing eviction regardless of H2O score.

---

## 4. Evidence Strength

The substantive statements and claims in this report are categorized strictly according to the evidence hierarchy defined in `AGENTS.md`:

| Category | Statements in this Report | Epistemic Justification |
|---|---|---|
| **SOURCE FACT** | - Qwen2.5-1.5B-Instruct architecture details (GQA, 28 layers, 12 Q heads, 2 KV heads, $d_{head}=128$).<br>- vLLM FP8 KV-cache CLI configuration and e4m3fn format.<br>- Clean compression safety softening in uncorrupted models (ACL 2026).<br>- Rule of three: zero events in 1,000 trials gives upper 95% CI bound of $\approx 0.3\%$. | Verifiable from published repository source code, official vLLM release documentation, peer-reviewed ACL 2026 proceedings, and standard mathematical statistics. |
| **INFERENCE** | - Demanding $U(T_{real}) = U(C_0)$ is a methodological error; matched-policy non-inferiority is required.<br>- Straight-Through Estimators (STE) may diverge from production vLLM FP8 kernels due to hardware FP8 GEMM rounding.<br>- Top-k eviction policies create a temporal asymmetry between past scoring and future query demand. | Logically deduced from Transformer computational mechanics, floating-point arithmetic specs, and statistical experimental principles. |
| **HYPOTHESIS** | - $H_1$: LoRA fine-tuning can implant a runtime backdoor triggered selectively by pinned vLLM FP8 KV caching with $\Delta_{int} \ge 0.50$.<br>- $H_6$: A model can manipulate H2O eviction scores to evict an inhibitory suppressor state ($S$). | Core empirical hypotheses of Campaign 001; completely unverified empirically. |
| **DECISION** | - Freeze Qwen2.5-1.5B-Instruct as the base model.<br>- Use exact harmless suffix marker rather than harmful payload.<br>- Enforce 1,000 sequestered prompts split by semantic cluster.<br>- Gate PF-SEB strictly behind UG6. | Deliberate methodological choices established in this protocol to maximize scientific rigor and safety. |
| **EXPERIMENTAL RESULT** | **NONE.** (Zero empirical runs completed). | Strict adherence to the pre-implementation epistemic baseline. |

---

## 5. Counterevidence & Alternative Explanations

The experimental design explicitly anticipates and establishes falsification controls for the following confounding hypotheses:

1. **Alternative Explanation 1: The Observed Effect is Simply Group B Clean Degradation ($H_0$).**
   - *Threat:* The target suffix or format flip is spontaneously emitted due to representation noise introduced by FP8 quantization.
   - *Experimental Disproof:* Controlled by Cell 2 $(\theta_c, T_{real})$ and Cell 4 $(\theta_f, T_{real})$. If $\text{ASR}(T_{real}, \theta_c) > 0.05$ or $\Delta_{int} < 0.30$, the effect is classified as intrinsic model fragility, falsifying $H_1$.
2. **Alternative Explanation 2: Generic LoRA Fine-Tuning Vulnerability.**
   - *Threat:* Fine-tuning with LoRA inherently destabilizes attention weights under low-precision regimes, creating spurious trigger sensitivity.
   - *Experimental Disproof:* Controlled by $\Delta_{cond}$ against $\theta_f$. $\theta_f$ undergoes identical optimization without the target loss. If $\theta_f$ activates under FP8, $H_1$ is falsified.
3. **Alternative Explanation 3: Proxy-to-Runtime Conformance Illusion.**
   - *Threat:* The backdoor functions perfectly under PyTorch STE ($T_{proxy}$) but fails under real vLLM FP8 ($T_{real}$), rendering it a simulation artifact.
   - *Experimental Disproof:* Enforced by mandatory gate UG6. If $\theta_b$ fails to transfer to vLLM on real hardware, the core publication claim is abandoned.
4. **Alternative Explanation 4: Prompt Template Memorization / Leakage.**
   - *Threat:* The model memorized specific phrasing during training, and activation is triggered by residual semantic cues rather than the cache policy.
   - *Experimental Disproof:* Enforced by semantic cluster partitioning and evaluation on 1,000 sequestered held-out prompts with zero user trigger tokens.
5. **Alternative Explanation 5: Decoding Stochasticity Flukes.**
   - *Threat:* Apparent differences are artifacts of sampling noise ($T > 0$).
   - *Experimental Disproof:* Evaluated strictly under deterministic greedy decoding ($T = 0.0$) with exact string parsing.

---

## 6. Open Questions

1. **Hardware Arithmetic Discrepancy:** Exactly how much divergence exists between PyTorch simulated FP8 and NVIDIA Hopper/Ada FP8 Tensor Core GEMMs under production vLLM? (To be resolved empirically in WP1 / Gate UG2).
2. **LoRA Capacity Limits:** Can a low-rank adapter ($r=16$) on attention projections successfully encode both normal instruction execution and dual-policy conditional branching without noticeable utility degradation?
3. **PF-SEB Suppressor Viability:** In an unmodified 1.5B model, can a single natural token position ($S$) be trained to suppress early attention while retaining late inhibitory retrieval power, or does PF-SEB require multi-token suppressor spans?

---

## 7. Quantitative Falsification Boundary Matrix & Recommended Actions

### 7.1 Quantitative Falsification Matrix (H0 vs. H1)

The outcome of the decisive experiment maps unambiguously to scientific conclusions via the following pre-registered decision rules:

```
+====================================================================================================+
|                                QUANTITATIVE FALSIFICATION MATRIX                                   |
+---------------------+---------------------+---------------------+----------------------------------+
| Empirical Metric    | Observed Value      | Scientific Status   | Mandatory Program Action         |
+---------------------+---------------------+---------------------+----------------------------------+
| Δ_int (T_real)      | < 0.20 or 95% CI    | H1 DECISIVELY       | Terminate backdoor claims.       |
|                     | includes 0.0        | FALSIFIED (H0 holds)| Publish rigorous negative result.|
+---------------------+---------------------+---------------------+----------------------------------+
| Δ_cond (T_real)     | < 0.20 or 95% CI    | Fine-Tuning Artifact| Classify as generic PEFT noise.  |
|                     | includes 0.0        | (H0 holds)          | Terminate attack program.        |
+---------------------+---------------------+---------------------+----------------------------------+
| ASR (C0, θb)        | >= 1.0%             | Stealth Failed      | Reject backdoor claim; retrain or|
|                     |                     |                     | classify as standard fine-tune.  |
+---------------------+---------------------+---------------------+----------------------------------+
| Δ_int (T_proxy)     | >= 0.60, BUT        | Real-Runtime        | UG6 FAILED: Publish proxy-runtime|
| Δ_int (T_real)      | < 0.20              | Transfer Falsified  | divergence study; abandon vLLM.  |
+---------------------+---------------------+---------------------+----------------------------------+
| ASR (Near-Miss)     | >= 0.40 under near- | Generic Quantization| Reject policy fingerprinting;    |
|                     | miss policies       | Fragility           | reframe as precision collapse.   |
+---------------------+---------------------+---------------------+----------------------------------+
| Δ_U(T) or Δ_U(C0)   | Exceeds non-infer-  | Unacceptable        | Classify as destructive damage;  |
|                     | iority margins      | Utility Collapse    | reject stealth backdoor claim.   |
+---------------------+---------------------+---------------------+----------------------------------+
| Δ_int >= 0.50,      | All pass with 95% CI| H1 DECISIVELY       | PROCEED TO DEFENSE & REPLICATION |
| Δ_cond >= 0.50,     | lower bounds > 0.30;| CONFIRMED           | Execute UG9, expand to 3rd seed, |
| Stealth & Utility OK| vLLM transfer valid |                     | unlock gated PF-SEB extension.   |
+---------------------+---------------------+---------------------+----------------------------------+
```

### 7.2 Staged Implementation Roadmap

1. **Step 1 (WP0 — Governance & Preregistration):** Freeze the experiment manifest, cryptographic hashes of $\mathcal{D}_{train}, \mathcal{D}_{dev}, \mathcal{D}_{test}$, and the AST marker parser. Validate access to Linux CUDA hardware supporting native FP8.
2. **Step 2 (WP1 — Conformance Harness & Gate UG2):** Build `src/harness/` and `src/compression/fake_fp8.py`. Run conformance testing on 100 clean prompts comparing $T_{proxy}$ against pinned vLLM FP8 ($T_{real}$). Verify Gate UG2 ($\rho_{Spearman} \ge 0.85$, CosSim $\ge 0.95$).
3. **Step 3 (WP2 — Clean Surface Characterization & Gate UG3):** Evaluate $\theta_c$ across $C_0$ and $T_{real}$ to establish baseline compression degradation curves.
4. **Step 4 (WP3 — Bounded Training & Gate UG4):** Execute dual-branch prefix/continuation LoRA training on $\theta_b$ (max 3 configurations, 2 seeds). Evaluate $\Delta_{int}(T_{proxy})$.
5. **Step 5 (WP4 — vLLM Production Transfer & Gate UG6):** Mount $\theta_b$ into the pinned vLLM runtime and execute the decisive 1,000-prompt evaluation. Calculate hierarchical bootstrap intervals for $\Delta_{int}(T_{real})$ and $\Delta_{cond}(T_{real})$. Apply the Quantitative Falsification Matrix.
6. **Step 6 (WP7–WP9 — Gated PF-SEB Extension):** If and only if UG6 passes, unlock the H2O eviction harness and execute the 7-condition causal intervention battery.

---

## 8. Files Created or Modified

1. `research/agent_reports/TRACK_C_EXPERIMENTAL_SCIENTIST.md` — Created this comprehensive experimental specification report.
2. `.agents/teamwork/worker_track_c/DISPATCH.md` — Recorded task dispatch, caller ID, and requirements.
3. `.agents/teamwork/worker_track_c/BRIEFING.md` — Initialized and maintained persistent working state.
4. `.agents/teamwork/worker_track_c/progress.md` — Maintained liveness heartbeat.
5. `.agents/teamwork/worker_track_c/handoff.md` — Authored 5-component self-contained handoff report for Campaign 001 coordinator.
