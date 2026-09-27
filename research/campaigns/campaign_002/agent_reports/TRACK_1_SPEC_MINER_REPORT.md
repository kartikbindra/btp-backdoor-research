# Campaign 002 Specification Mining Report
## Authoritative Requirements, Constraints, Mathematical Formulations, and Evaluation Standards for the WP0/WP1 Runtime Gate

**Mining Agent:** `spec_miner_survey_1`  
**Working Directory:** `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\spec_miner_survey_1\`  
**Date:** 2026-09-27  
**Epistemic Standard:** `AGENTS.md` Evidence Discipline (`[SOURCE FACT]`, `[INFERENCE]`, `[HYPOTHESIS]`, `[DECISION]`, `[EXPERIMENTAL RESULT]`)  
**Mission Role:** Specification Miner (Probing Authoritative Project Specifications; Read-Only Analysis)

---

## 1. Executive Summary & Objective

The objective of Campaign 002 (governing Work Packages WP0 and WP1) is to execute the **Prerequisite Runtime Gate**:
> **Central Research Question (RQ2):**  
> *Does the proposed FP8 KV-cache proxy ($T_{proxy}$) reproduce the scientifically relevant behavior of the pinned production vLLM FP8 KV-cache path ($T_{real}$) closely enough, on a clean, unmodified model ($\theta_c$), that later runtime-conditioned backdoor experiments would be interpretable?*

Campaign 001 established the foundation:
- Permanently retracted broad umbrella novelty claims ("first KV-cache backdoor") due to prior art (CacheTrap ICCAD 2026, HijackKV arXiv:2607.19957, HistorySwap arXiv:2511.12752, Chat-Template Backdoors ACM CCS 2026).
- Formally categorized the narrow, production-grounded claim as **`PLAUSIBLY DISTINCT`**.
- Established the **Funnel Architecture**: primary treatment is official pinned vLLM FP8 (`fp8_e4m3fn`) on fresh per-request caches ($C_0 \to \emptyset$), while Policy-Fingerprinted Self-Eviction (PF-SEB) is quarantined strictly behind Gate UG6.
- Established **Gate UG2** (Proxy-to-Runtime Conformance) as an **absolute blocker**: if the training proxy does not conform to production serving before training begins, model training is banned, and the project pivots to an empirical paper documenting the proxy-to-deployment transfer gap.

Campaign 002 is strictly an **empirical systems and numerical conformance gate on clean models**. It performs **strictly zero backdoor training**, evaluates **strictly zero harmful payloads**, and derives **strictly zero novelty claims**.

---

## 2. Authoritative Source Documents Inspected

The specifications, constraints, and formulations documented herein are mined directly from:
1. `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\ORIGINAL_REQUEST.md` (Verbatim charter and acceptance criteria for Campaign 002).
2. `research/campaigns/campaign_002/CAMPAIGN_002_MASTER_PROMPT.md` (Team roles, phase order, and minimum experiment matrix).
3. `research/campaigns/campaign_002/CAMPAIGN_002_ACCEPTANCE_TEMPLATE.md` & `README.md` (Acceptance structure and required deliverables).
4. `research/campaigns/campaign_001/DECISION_MEMO.md` (Adopted by Decision D15 as binding strategy; establishes 6-cell design, estimands, and hardware constraints).
5. `CONSOLIDATED_RESEARCH_PLAN.md` (§0 Executive Decision, §4 Threat Model, §5 RQ2, §6 Causal Framework, §7 Gate UG2, §8 WP0/WP1).
6. `AGENTS.md` (Repository constitution, evidence discipline, stop conditions, and agent permissions).
7. `researchMemory/agentMemory/` canonical files:
   - `CURRENT_STATE.md` (Epistemic baseline, funnel architecture, terminology ladder).
   - `DECISION_LOG.md` (Decisions D1–D17, specifically D15, D16, D17).
   - `EXPERIMENT_REGISTRY.md` (Unified Work Package architecture WP0–WP9 and gate mappings).
   - `FINDINGS.md` (Literature findings, clean baseline risks, and theoretical discoveries).
8. `research/campaigns/campaign_001/agent_reports/` (`TRACK_C_EXPERIMENTAL_SCIENTIST.md`, `TRACK_E_STATISTICAL_AUDITOR.md`, `TRACK_D_THREAT_MODEL_CRITIC.md`).

---

## 3. Detailed Requirements Breakdown (R1 – R4)

### Requirement R1: Environment Locking & Production Runtime Path Inspection
- **Objective:** Pin and verify the full hardware, framework, compiler, kernel, and model stack, tracing the exact tensor execution path:
  $$\text{Model} \longrightarrow \text{K/V Projection} \longrightarrow \text{FP8 Cache Storage} \longrightarrow \text{Attention Consumption} \longrightarrow \text{Output}$$
- **Key Specifications:**
  - Pin target model revision: `Qwen/Qwen2.5-1.5B-Instruct` (exact Git commit hash).
  - Pin framework release: `vllm == 0.26.0` (commit hash recorded).
  - Pin hardware platform: Dedicated Linux host (Ubuntu 22.04 LTS), CUDA 12.4, NVIDIA Driver $\ge 550.54.14$. Target GPU: NVIDIA Ada Lovelace (RTX 4090, L40S) or NVIDIA Hopper (H100, A100-SXM4 for baseline).
  - Identify hardware capabilities: Document whether the target GPU supports native FP8 Tensor Cores (Ada Lovelace SM 8.9, Hopper SM 9.0) or lacks hardware FP8 (Ampere SM 8.0 falls back).
  - Trace quantization boundaries: Document exact tensor dtypes, scale factor calculations, clipping boundaries, and memory layouts in PagedAttention blocks.
  - Trace attention kernels: FlashAttention-FP8 / FlashInfer FP8 execution vs software fallbacks.
- **Deliverables:**
  - `research/campaigns/campaign_002/CAMPAIGN_002_ENVIRONMENT_MANIFEST.md`
  - `research/campaigns/campaign_002/CAMPAIGN_002_RUNTIME_PATH.md`

### Requirement R2: Determinism Baseline & 3-Condition Clean Conformance Matrix
- **Objective:** Evaluate an untouched, clean base model ($\theta_c$) across the three-condition matrix using deterministic greedy decoding ($T=0$):
  - **Condition A (Reference):** BF16 / uncompressed full-cache (`past_key_values` DynamicCache in PyTorch/Transformers).
  - **Condition B (Production Runtime $T_{real}$):** Official pinned vLLM FP8 engine (`--kv-cache-dtype fp8`).
  - **Condition C (Candidate Proxy $T_{proxy}$):** Differentiable PyTorch Straight-Through Estimator (STE) quantize-dequantize module (`fake_fp8.py`).
- **Key Specifications:**
  - Evaluate on sequestered benign prompt clusters (deduplicated UltraFeedback Clean & LMSYS Benign).
  - Multi-level validation:
    1. Tensor-level error: Layerwise Normalized RMSE and Cosine Similarity on Keys and Values.
    2. Behavioral logit fidelity: Next-token Spearman rank correlation and top-10 directional agreement.
    3. End-to-end token match: Exact greedy token agreement traces.
  - Storage vs. Compute Factorization: Implement ablation $T_{storage}$ (FP8 storage + BF16 FlashAttention dot products) to disentangle storage quantization noise from hardware Tensor Core FP8 GEMM rounding.
  - Run-to-run determinism: Quantify bitwise token reproducibility across repeated runs and fresh process restarts.
- **Deliverables:**
  - `research/campaigns/campaign_002/CAMPAIGN_002_DETERMINISM.md`
  - `research/campaigns/campaign_002/CAMPAIGN_002_PROXY_CONFORMANCE.md`

### Requirement R3: Pre-Registered Acceptance Gate & Adversarial Conformance Audit
- **Objective:** Freeze candidate numerical acceptance thresholds prior to confirmatory analysis, and execute an adversarial test battery designed to stress-test and falsify proxy conformance.
- **Key Specifications:**
  - Zero post-hoc threshold fishing: thresholds must be formulated and locked before running confirmatory evaluation.
  - Adversarial audit dimensions:
    1. Process restart invariance: Detect memory leaks or uninitialized GPU cache lines.
    2. Prompt cluster & sequence length variations: Short (128), medium (512), and long (2048) context scaling.
    3. Scale outlier & saturation dynamics: Measure clipping frequency at dynamic range boundaries ($|X| \ge 448 \cdot S$).
    4. Hardware fallback detection: Assert true FP8 Tensor Core execution; detect and flag any silent fallback to simulated FP8 or BF16 as an immediate test failure.
    5. Alternative attention backends: Compare FlashAttention vs FlashInfer if available.

### Requirement R4: Decision Memo & Canonical Research Memory Synchronization
- **Objective:** Synthesize all findings into a definitive, binding decision memo selecting one of three explicit operational states, and synchronize canonical memory files in `researchMemory/agentMemory/`.
- **Decision States:**
  - **`PASS`**: $T_{proxy}$ conforms to $T_{real}$ across tensor, logit, and behavioral metrics; authorize progression to WP2 (Clean Surface) and WP3 (Bounded Training).
  - **`CONDITIONAL PASS`**: Conformance holds only under explicit documented operational constraints (e.g., specific context lengths, calibration transformations, or head subsets).
  - **`FAIL`**: $T_{proxy}$ diverges materially from $T_{real}$ (e.g., tensor cosine $< 0.98$ or logit Spearman $\rho < 0.85$); halt backdoor training permanently and pivot to an empirical proxy-to-deployment transfer gap paper per Decision D15.
- **Deliverable:**
  - `research/campaigns/campaign_002/CAMPAIGN_002_DECISION_MEMO.md`
  - Canonical memory synchronization in `researchMemory/agentMemory/` via the Research Memory Keeper.

---

## 4. Pinned Configurations and Technical Constants

The following constants are authoritative and pinned across Campaign 002:

| Parameter Category | Pinned Value / Specification | Authoritative Source |
|---|---|---|
| **Target Model** | `Qwen/Qwen2.5-1.5B-Instruct` | `CONSOLIDATED_RESEARCH_PLAN.md` §8.3; `CURRENT_STATE.md` |
| **Model Architecture** | GQA: 28 layers, 12 query heads, 2 KV heads, hidden size 1536, head dim 128 | `TRACK_E_STATISTICAL_AUDITOR.md` §3.5.2 |
| **Model Checkpoint State** | Untouched, clean base model ($\theta_c$) only. Zero weights modified. | `ORIGINAL_REQUEST.md` §R2; `CAMPAIGN_002_MASTER_PROMPT.md` |
| **FP8 Format** | `torch.float8_e4m3fn` (1 sign bit, 4 exponent bits, 3 mantissa bits) | `CAMPAIGN_001_DECISION_MEMO.md` §11.2; `TRACK_C` §3.3.1 |
| **FP8 Dynamic Range** | $[-448.0, 448.0]$, machine epsilon $\epsilon = 2^{-3} = 0.125$ | `TRACK_C_EXPERIMENTAL_SCIENTIST.md` §3.3.1 |
| **Scaling Granularity** | Per-tensor or per-head static scaling: $S = \frac{\max(\|X\|) + \epsilon}{448.0}$ | `TRACK_C` §3.3.1; `TRACK_E` §3.1.1 |
| **Production Serving Framework** | `vllm == 0.26.0` (Git commit pinned) | `CAMPAIGN_001_DECISION_MEMO.md` §11.2 |
| **Production Serving CLI** | `--kv-cache-dtype fp8 --enforce-eager --gpu-memory-utilization 0.90` | `TRACK_C_EXPERIMENTAL_SCIENTIST.md` §3.3.2 |
| **Operating System** | Linux Ubuntu 22.04 LTS (Dedicated GPU host required for physical kernels) | `CAMPAIGN_001_DECISION_MEMO.md` §11.2 |
| **CUDA & Driver** | CUDA `12.4`, NVIDIA Display Driver $\ge 550.54.14$ | `CAMPAIGN_001_DECISION_MEMO.md` §11.2 |
| **PyTorch & Python** | `python == 3.10.14`, `torch == 2.4.0+cu124`, `transformers == 4.44.2` | `CAMPAIGN_001_DECISION_MEMO.md` §11.2 |
| **Attention Kernel** | FlashAttention-FP8 / FlashInfer FP8 (PagedAttention v2) | `CAMPAIGN_002_MASTER_PROMPT.md` Phase 1 |
| **Decoding Parameters** | Deterministic greedy: `temperature = 0.0`, `top_p = 1.0`, `top_k = -1`, `seed = 42` | `TRACK_C` §3.4.3; `TRACK_E` §3.5.1 |
| **Context Length Bounds** | Maximum model length: 2048 tokens; Evaluation tiers: 128, 512, 2048 | `ORIGINAL_REQUEST.md` R3; `TRACK_C` §3.3.2 |
| **Cache Isolation Protocol** | Strict fresh per-request cache ($C_0 \to \emptyset$), `--enable-prefix-caching False` | `TRACK_C` §3.4.3; `CURRENT_STATE.md` §5.1 |
| **Dataset Source** | Deduplicated UltraFeedback Clean & LMSYS Chatbot Arena Benign | `TRACK_C` §3.5.1; `WP0 Manifest` |
| **Data Partitioning** | Dense semantic clustering ($k$-means on `text-embedding-3-large`) | `CONSOLIDATED_RESEARCH_PLAN.md` §8; `TRACK_C` §3.5.1 |
| **Conformance Sample Size** | $N = 200$ clean evaluation prompts | `CONSOLIDATED_RESEARCH_PLAN.md` §8.1; `UG2 Protocol` |
| **Determinism Sample Size** | $N = 50$ clean prompts across 50 repeated runs & process restarts | `TRACK_C` §3.7 (UG1); `TRACK_E` §3.5.1 |

---

## 5. Mathematical Formulations & Estimands

### 5.1 Straight-Through Estimator Quantize-Dequantize ($T_{proxy}$)
For input tensor $X \in \{K, V\}$:
1. **Scale Factor Calculation:**
   $$S = \frac{\max(|X|) + \epsilon_{clip}}{448.0}$$
   where $\epsilon_{clip} = 10^{-7}$ prevents division by zero.
2. **Quantization with Saturation:**
   $$X_q = \operatorname{clamp}\left( \left\lfloor \frac{X}{S} \right\rceil_{\text{FP8}}, -448.0, 448.0 \right)$$
3. **Dequantization back to BF16:**
   $$\tilde{X} = T_{proxy}(X) = X_q \cdot S$$
4. **Straight-Through Estimator (STE) Backward Pass:**
   $$\frac{\partial \mathcal{L}}{\partial X} \approx \frac{\partial \mathcal{L}}{\partial \tilde{X}} \cdot \mathbb{I}\left( |X| \le 448.0 \cdot S \right)$$

### 5.2 Tensor Error Metrics (Gate UG2)
1. **Layerwise Normalized Root Mean Squared Error (NRMSE):**
   $$\text{NRMSE}(K_l) = \frac{\| T_{proxy}(K_l) - T_{real}(K_l) \|_F}{\| K_l^{ref} \|_F}$$
   where $\|\cdot\|_F$ denotes the Frobenius norm, and $K_l^{ref}$ is the uncompressed BF16 reference tensor.
2. **Layerwise Cosine Similarity:**
   $$\cos(T_{proxy}(K_l), T_{real}(K_l)) = \frac{\langle \operatorname{vec}(T_{proxy}(K_l)), \operatorname{vec}(T_{real}(K_l)) \rangle}{\| T_{proxy}(K_l) \|_2 \cdot \| T_{real}(K_l) \|_2}$$
3. **Clipping / Saturation Equivalence:**
   $$\Delta_{sat} = \left| \frac{\sum \mathbb{I}(|T_{proxy}(X)| = 448 \cdot S) - \sum \mathbb{I}(|T_{real}(X)| = 448 \cdot S)}{\text{Total Elements}} \right|$$

### 5.3 Behavioral Logit Fidelity Metrics
1. **Spearman Rank Correlation of Next-Token Logits:**
   $$\rho_{\text{Spearman}} = 1 - \frac{6 \sum_{j=1}^V d_j^2}{V(V^2 - 1)}$$
   where $d_j = \operatorname{rank}(Z_j^{(proxy)}) - \operatorname{rank}(Z_j^{(real)})$ across top vocabulary logits.
2. **Directional Logit Shift Agreement:**
   $$\text{Agr}_{\text{dir}} = \frac{1}{|\mathcal{V}_{top10}|} \sum_{j \in \mathcal{V}_{top10}} \mathbb{I}\left( \operatorname{sign}(Z_j^{(proxy)} - Z_j^{(C0)}) = \operatorname{sign}(Z_j^{(real)} - Z_j^{(C0)}) \right)$$
3. **Jensen-Shannon Divergence (JSD):**
   $$\text{JSD}(P^{(proxy)} \,\|\, P^{(real)}) = \frac{1}{2} D_{KL}(P^{(proxy)} \,\|\, M) + \frac{1}{2} D_{KL}(P^{(real)} \,\|\, M)$$
   where $M = \frac{1}{2}(P^{(proxy)} + P^{(real)})$ and $P = \operatorname{Softmax}(Z)$.

### 5.4 End-to-End Token Conformance
1. **Greedy Token Exact Match Rate:**
   $$\text{TMR} = \frac{1}{N} \sum_{i=1}^N \prod_{t=1}^{T_i} \mathbb{I}\left( y_{i,t}^{(proxy)} = y_{i,t}^{(real)} \right)$$
2. **Per-Token Agreement Across Trajectory:**
   $$\text{PTA} = \frac{1}{\sum_{i=1}^N T_i} \sum_{i=1}^N \sum_{t=1}^{T_i} \mathbb{I}\left( y_{i,t}^{(proxy)} = y_{i,t}^{(real)} \right)$$

---

## 6. Pre-Registered Acceptance Metrics, Thresholds & Falsification Criteria

To satisfy the constitutional non-negotiable (**zero post-hoc metric fitting**), the following thresholds are frozen prior to confirmatory analysis:

| Verification Dimension | Metric | Frozen Acceptance Threshold | Falsification / Blocker Boundary | Discovered Via |
|---|---|---|---|---|
| **Tensor Error: Keys** | Layerwise $\text{NRMSE}(K_l)$ | $\le 0.05$ | $> 0.15$ | `CAMPAIGN_001_DECISION_MEMO` §7.2; `TRACK_C` §3.7 |
| **Tensor Error: Values** | Layerwise $\text{NRMSE}(V_l)$ | $\le 0.05$ | $> 0.15$ | `CAMPAIGN_001_DECISION_MEMO` §7.2; `TRACK_C` §3.7 |
| **Tensor Alignment: Keys** | Layerwise $\cos(K_l)$ | $\ge 0.995$ | $< 0.98$ (absolute blocker $< 0.95$) | `CONSOLIDATED_RESEARCH_PLAN` §7 (UG2); `TRACK_E` §3.1.3 |
| **Tensor Alignment: Values** | Layerwise $\cos(V_l)$ | $\ge 0.995$ | $< 0.98$ (absolute blocker $< 0.95$) | `CONSOLIDATED_RESEARCH_PLAN` §7 (UG2); `TRACK_E` §3.1.3 |
| **Saturation Equivalence** | Saturation delta $\Delta_{sat}$ | $\le 10^{-4}$ ($\pm 10\%$ relative band) | $> 5 \times 10^{-4}$ divergence | `TRACK_E_STATISTICAL_AUDITOR` §3.1.3 |
| **Logit Distribution** | Spearman rank $\rho(Z^{(proxy)}, Z^{(real)})$ | $\ge 0.85$ across top-100 logits | $< 0.85$ (Halt & pivot) | `CAMPAIGN_001_DECISION_MEMO` §7.2; `CONSOLIDATED_RESEARCH_PLAN` §7 |
| **Logit Shift Direction** | Directional agreement $\text{Agr}_{\text{dir}}$ | $\ge 80\%$ of top-10 tokens | $< 70\%$ agreement | `TRACK_C_EXPERIMENTAL_SCIENTIST` §3.7 |
| **Logit Divergence** | Jensen-Shannon Divergence $\text{JSD}$ | $\le 0.02$ | $> 0.05$ (Unacceptable divergence) | `TRACK_E_STATISTICAL_AUDITOR` §3.1.3 |
| **End-to-End Greed** | Greedy Token Agreement $\text{PTA}$ | $\ge 90\%$ over 128 tokens | $< 80\%$ over 128 tokens | `CAMPAIGN_002_ACCEPTANCE_TEMPLATE` §Primary metrics |
| **Determinism: Run-to-Run**| Bitwise Token Match ($T=0$) | $100\%$ over 50 repeat runs | $< 100\%$ on deterministic kernels | `CONSOLIDATED_RESEARCH_PLAN` §7 (UG1); `TRACK_C` §3.7 |
| **Determinism: Restarts** | Bitwise Match across Process Restarts | $100\%$ match | Any memory leak or state carryover | `ORIGINAL_REQUEST.md` §R2, §R3 |
| **Hardware Fallback** | Tensor Core FP8 Execution Check | True Hardware FP8 Execution confirmed | Any silent fallback to simulated FP8 or BF16 = TEST FAILURE | `ORIGINAL_REQUEST.md` §R3, §Scientific Guardrails |

### Falsification & Stop Conditions (Gate UG2):
1. If $\cos(K_l) < 0.98$ or $\rho_{\text{Spearman}} < 0.85$ after dedicated calibration, **HALT IMMEDIATELY**.
2. Do not proceed to WP3 backdoor training.
3. Automatically render verdict **`FAIL`** in `CAMPAIGN_002_DECISION_MEMO.md`.
4. Trigger Decision D15 fallback: Pivot the project to an empirical publication documenting the proxy-to-deployment transfer gap.

---

## 7. Features Discovered

In accordance with the Specification Miner archetype, the table below documents all features discovered across the authoritative specifications:

| # | Category | Feature | Description | Inputs | Outputs | Error Behavior | Discovered Via |
|---|---|---|---|---|---|---|---|
| 1 | Runtime Path | Execution Path Tracing | Trace tensor transformations from model forward pass through K/V projection, FP8 quant/pack, PagedAttention storage, to attention GEMM. | Base model $\theta_c$, prompt $x$ | Detailed execution graph & layer dimensions | Flag unsupported kernel or unmapped layer | `ORIGINAL_REQUEST.md` §R1; `CAMPAIGN_002_MASTER_PROMPT` Phase 1 |
| 2 | Runtime Path | Dtype & Scaling Inspection | Document exact storage format (`fp8_e4m3fn`), scale factor calculation, and quantization granularity (per-tensor vs per-head). | PagedAttention KV tensors | Scale vectors $S_k, S_v$, byte layout | Flag scale underflow / overflow | `CAMPAIGN_002_MASTER_PROMPT` Phase 1; `TRACK_C` §3.3.1 |
| 3 | Runtime Path | Attention Kernel Identification | Identify whether attention uses FlashAttention-FP8, FlashInfer, or Triton kernels, and whether QK/ScoreV are executed in FP8 vs BF16. | PagedAttention blocks, queries $Q$ | Attention context tensors | Detect fallback to non-FP8 kernels | `CAMPAIGN_001_DECISION_MEMO` §11.2; `TRACK_E` §3.1.1 |
| 4 | Runtime Path | Hardware Fallback Detection | Assert execution on native Ada/Hopper FP8 Tensor Cores; detect silent software/emulated fallbacks. | GPU compute capability (SM 8.9/9.0) | Boolean `is_native_fp8` | Halt as TEST FAILURE if silent fallback occurs | `ORIGINAL_REQUEST.md` §Scientific Guardrails; `TRACK_E` §3.1 |
| 5 | Determinism | Run-to-Run Parity Check | Evaluate 50 repeated greedy decoding runs ($T=0$) on identical inputs to quantify bitwise reproducibility. | 50 clean prompts, fixed seed | Per-token bitwise hash match | Flag non-deterministic kernel reductions | `CONSOLIDATED_RESEARCH_PLAN` §7 (UG1); `TRACK_C` §3.7 |
| 6 | Determinism | Process Restart Invariance | Verify that restarting Python/vLLM processes yields identical cache events and generation traces. | Fresh process launch, same inputs | Cache memory delta, token hashes | Flag process-state leakage or memory carryover | `ORIGINAL_REQUEST.md` §R2, §R3 |
| 7 | Determinism | Fresh Request Cache Isolation | Explicitly purge and assert zero memory allocation in KV cache between successive inference requests. | `past_key_values` reset, CUDA empty cache | Cache memory assertions ($C_0 \to \emptyset$) | Fail if cross-request residual memory exists | `TRACK_C_EXPERIMENTAL_SCIENTIST` §3.4.3; `CURRENT_STATE` §5.1 |
| 8 | Conformance | Tensor-Level NRMSE & CosSim | Compare layerwise Key/Value tensors between $T_{proxy}$ (STE) and $T_{real}$ (vLLM FP8). | Extracted $K, V$ tensors across 200 prompts | $\text{NRMSE}(K_l), \cos(K_l)$ per layer | Flag if $\cos < 0.995$ or $\text{NRMSE} > 0.05$ | `CAMPAIGN_001_DECISION_MEMO` §7.2; `TRACK_C` §3.7 |
| 9 | Conformance | Saturation & Outlier Tracking | Track frequency of tensor elements saturating at FP8 dynamic range limits ($\pm 448 \cdot S$). | Intermediate $K, V$ activations | Saturation element counts & percentages | Flag if saturation divergence $> 10^{-4}$ | `TRACK_E_STATISTICAL_AUDITOR` §3.1.3 |
| 10 | Conformance | Logit Spearman Rank Correlation | Measure rank correlation of next-token logits generated under $T_{proxy}$ vs $T_{real}$. | Next-token unnormalized logits $Z$ | Spearman rank $\rho$ | Blocker if $\rho < 0.85$ (halt training) | `CONSOLIDATED_RESEARCH_PLAN` §7 (UG2); `TRACK_C` §3.7 |
| 11 | Conformance | Directional Logit Agreement | Validate whether $T_{proxy}$ and $T_{real}$ shift top-10 vocabulary logits in the same direction relative to $C_0$. | $Z^{(proxy)}, Z^{(real)}, Z^{(C0)}$ | Percentage directional sign agreement | Flag if agreement $< 80\%$ | `TRACK_C_EXPERIMENTAL_SCIENTIST` §3.7 |
| 12 | Conformance | Jensen-Shannon Divergence | Quantify divergence between output probability distributions under proxy and runtime. | Softmax probability distributions | JSD value | Flag if $\text{JSD} > 0.02$; blocker if $> 0.05$ | `TRACK_E_STATISTICAL_AUDITOR` §3.1.3 |
| 13 | Conformance | Storage vs. Compute Factorization | Compare ablation $T_{storage}$ (FP8 store + BF16 attention) with $T_{real}$ to isolate GEMM rounding. | Prompt inputs, $T_{storage}$ execution | Divergence delta $(\Delta_{\text{storage}}, \Delta_{\text{gemm}})$ | Attribute divergence to storage or arithmetic | `CONSOLIDATED_RESEARCH_PLAN` §8.1; `TRACK_C` §3.3.3 |
| 14 | Adversarial Audit | Context Length Scaling Stress | Evaluate conformance across varying context tiers: short (128), medium (512), and long (2048) tokens. | Sequestered prompts padded to length tiers | Conformance metrics across tiers | Detect degradation in long-context bounds | `ORIGINAL_REQUEST.md` §R3; `CAMPAIGN_002_MASTER_PROMPT` Phase 5 |
| 15 | Adversarial Audit | Scale Outlier Sensitivity Test | Introduce synthetic heavy-tailed activation outliers to stress-test static scaling clipping behavior. | Prompts with extreme feature values | Clipping counts, logit distortion | Flag catastrophic saturation collapse | `TRACK_E_STATISTICAL_AUDITOR` §3.1.1, §3.1.2 |
| 16 | Adversarial Audit | Alternate Attention Backend Audit | Compare FlashAttention-FP8 against FlashInfer or custom Triton kernels if supported. | Dual backend configurations | Cross-backend CosSim and token agreement | Characterize backend-specific fingerprint | `CONSOLIDATED_RESEARCH_PLAN` §8.1; `ORIGINAL_REQUEST` §R3 |
| 17 | Decision Synthesis | Pre-Registered Acceptance Gate | Freeze candidate numerical thresholds on pilot data before confirmatory evaluation. | Pilot calibration measurements | Frozen threshold manifest | Strict rejection of post-hoc metric fitting | `ORIGINAL_REQUEST.md` §R3, §Scientific Guardrails |
| 18 | Decision Synthesis | Three-State Decision Memo | Formulate binding research-management determination: PASS, CONDITIONAL PASS, or FAIL. | Synthesis of all conformance & audit metrics | `CAMPAIGN_002_DECISION_MEMO.md` | Authorize WP2/WP3 or trigger D15 pivot | `ORIGINAL_REQUEST.md` §R4; `CAMPAIGN_002_MASTER_PROMPT` Phase 6 |
| 19 | Memory Sync | Canonical State Synchronization | Update `CURRENT_STATE`, `DECISION_LOG`, `EXPERIMENT_REGISTRY`, `FINDINGS` via Research Memory Keeper. | Final Campaign 002 artifacts | Synchronized canonical memory records | Preserve history; do not overwrite | `AGENTS.md` Collaboration discipline; `ORIGINAL_REQUEST` R4 |

---

## 8. Edge Cases and Stress Conditions

| # | Feature | Input / Stress Condition | Observed / Documented Behavior | Authoritative Source |
|---|---|---|---|---|
| 1 | Hardware Fallback | Execution on NVIDIA Ampere (A100, SM 8.0) | A100 lacks native FP8 Tensor Cores; vLLM or Triton silently falls back to software emulation or BF16 upcasting, masking hardware rounding. Must be flagged as test failure. | `ORIGINAL_REQUEST.md` §63; `TRACK_E` §3.1.1 |
| 2 | Outlier Saturation | Extreme outlier channels in deep attention layers ($|X| > 448 \cdot S$) | In static per-tensor scaling, extreme outliers clamp to $\pm 448$, forcing 99.9% of normal tokens into coarse discrete bins, causing severe quantization distortion. | `TRACK_E_STATISTICAL_AUDITOR.md` §3.1.1 |
| 3 | Long-Context Accumulation | Long sequences ($L \ge 2,048$ tokens) | Autoregressive generation accumulates rounding errors across steps; token flips occur earlier in generation, leading to divergent downstream trajectories under greedy decoding. | `TRACK_E_STATISTICAL_AUDITOR.md` §3.5.1 |
| 4 | CUDA Non-Associativity | Parallel reduction in multi-threaded GEMMs under $T=0$ | Thread warp scheduling variations introduce atomic addition non-associativity ($10^{-6}$ delta), occasionally flipping marginal logits and breaking bitwise determinism. | `TRACK_E_STATISTICAL_AUDITOR.md` §3.5.1 |
| 5 | Process Restart / VRAM Leak | Successive queries across warm vs cold engine starts | Residual VRAM allocations or uninitialized cache blocks could leak state across requests if prefix caching is enabled or cache reset fails. | `TRACK_C_EXPERIMENTAL_SCIENTIST.md` §3.4.3 |
| 6 | Storage vs Compute Mismatch | $T_{storage}$ (FP8 store + BF16 GEMM) vs $T_{real}$ (FP8 GEMM) | If $T_{proxy}$ matches $T_{storage}$ but diverges from $T_{real}$, the learned trigger will condition on physical Tensor Core GEMM rounding rather than storage quantization. | `CONSOLIDATED_RESEARCH_PLAN.md` §8.1; `TRACK_C` §3.3.3 |
| 7 | Near-Zero Feature Underflow | Channels with small activation magnitudes ($|X| < S \cdot 2^{-3}$) | Subnormal values underflow to zero due to FP8 `e4m3fn` lack of subnormal support or coarse step size ($0.125$), creating silent zero-masking in attention keys. | `TRACK_E_STATISTICAL_AUDITOR.md` §3.1.1 |
| 8 | Multi-Query GQA Asymmetry | Grouped-Query Attention (12 Q heads, 2 KV heads) | KV projection tensors are shared across 6 query heads; quantization errors in a single KV head affect multiple attention heads simultaneously, amplifying logit shifts. | `TRACK_C_EXPERIMENTAL_SCIENTIST.md` §3.1; `TRACK_E` §3.5.2 |

---

## 9. Required Deliverable Schemas & Structural Specifications

To ensure deliverable completeness and compliance with project contracts, each artifact generated during Campaign 002 must strictly adhere to the following schemas:

### 9.1 `research/campaigns/campaign_002/CAMPAIGN_002_ENVIRONMENT_MANIFEST.md`
- **Header:** Title, Campaign ID, Date, Author/Role, Hardware Architecture, Epistemic Baseline.
- **Section 1: Hardware Environment:** GPU Model, GPU Architecture (Ada Lovelace / Hopper / Ampere), Compute Capability (SM version), VRAM capacity, Host OS (Ubuntu 22.04 LTS), Kernel version, NVIDIA Driver version, CUDA Toolkit version.
- **Section 2: Software Dependencies:** Python version (`3.10.14`), PyTorch version (`2.4.0+cu124`), Transformers (`4.44.2`), PEFT (`0.12.0`), FlashAttention (`2.6.3`), Scipy, Numpy.
- **Section 3: Framework Locking:** vLLM version (`v0.26.0`), exact Git commit SHA, build flags, C++ / Triton compiler flags.
- **Section 4: Model Specification:** Model ID (`Qwen/Qwen2.5-1.5B-Instruct`), Git revision hash, parameter count, architecture parameters (layers=28, heads=12, kv_heads=2, dim=1536), tokenizer SHA-256 hash.
- **Section 5: Prompt Corpus Hashes:** Prompt dataset source, semantic cluster IDs, SHA-256 hashes of $\mathcal{D}_{train}$, $\mathcal{D}_{dev}$, $\mathcal{D}_{test}$, and the $N=200$ clean conformance subset.
- **Section 6: Execution Command Manifest:** Exact CLI commands used to launch the vLLM serving daemon and test runners.

### 9.2 `research/campaigns/campaign_002/CAMPAIGN_002_RUNTIME_PATH.md`
- **Header:** Title, Target Framework, Kernel Path, Status.
- **Section 1: End-to-End Architectural Execution Graph:** Ascii diagram and text walkthrough tracing:
  `Input Tokens -> Embedding -> Transformer Blocks -> K/V Projections -> FP8 Quantization Module -> PagedAttention KV Cache Allocation -> FP8/BF16 Attention Kernel -> Softmax -> Output Projection -> Logits`.
- **Section 2: Tensor Dtypes and Memory Layout:** Exact memory representations across stages (BF16 input $\to$ FP8 storage format `fp8_e4m3fn` $\to$ register dequantization $\to$ FP32 accumulation).
- **Section 3: Quantization & Dequantization Boundaries:** Mathematical specification of scaling factor calculation, clipping limits, saturation thresholds, and static scale parameter mapping (`fp8_scales.json`).
- **Section 4: Attention Kernel Specifications:** Analysis of PagedAttention v2, FlashAttention-FP8, FlashInfer; execution modes for $Q \cdot K^T$ and $P \cdot V$; accumulation precision.
- **Section 5: Hardware Fallback Analysis:** Analysis of GPU capabilities, Tensor Core utilization, and automated validation hooks that detect emulated or upcasted execution paths.

### 9.3 `research/campaigns/campaign_002/CAMPAIGN_002_DETERMINISM.md`
- **Header:** Title, Purpose, Scope, Evaluated Checkpoint ($\theta_c$).
- **Section 1: Determinism Test Protocol:** Description of 50 repeat evaluations under greedy decoding ($T=0$), seed locking, and request dispatch ordering.
- **Section 2: Within-Process Bitwise Parity Results:** Token match rate, per-token logit divergence across repeated runs under Condition A (BF16) and Condition B (vLLM FP8).
- **Section 3: Process Restart Invariance Results:** Comparison of generation traces between cold engine boot, warm restarts, and successive queries; verification of KV cache purge ($C_0 \to \emptyset$).
- **Section 4: Kernel Non-Determinism Characterization:** Measurement of floating-point reduction non-associativity across warp threads; quantification of baseline jitter.
- **Section 5: Gate UG1 Verdict:** Explicit declaration of whether Gate UG1 is passed or failed, with remediation steps if jitter is detected.

### 9.4 `research/campaigns/campaign_002/CAMPAIGN_002_PROXY_CONFORMANCE.md`
- **Header:** Title, Purpose, Compared Conditions ($C_0$ vs $T_{real}$ vs $T_{proxy}$).
- **Section 1: 3-Condition Evaluation Methodology:** Description of paired prompt evaluations across Condition A (BF16), Condition B (vLLM FP8), and Condition C (PyTorch STE proxy).
- **Section 2: Tensor-Level Error Analysis:** Tabulated layerwise NRMSE, Cosine Similarity, and saturation counts for Keys and Values across all 28 layers.
- **Section 3: Behavioral Logit Conformance:** Spearman rank correlation of logits, Jensen-Shannon divergence, and top-10 directional agreement percentages.
- **Section 4: End-to-End Generation Fidelity:** Greedy token match rate ($\text{PTA}$, $\text{TMR}$) across generation lengths (128, 512 tokens).
- **Section 5: Storage vs. Compute Factorization ($T_{storage}$):** Empirical ablation results disentangling storage noise from hardware GEMM rounding.
- **Section 6: Adversarial Audit Results:** Process restart invariance, context length tiers (128, 512, 2048), scale outlier saturation tests, and fallback assertion logs.
- **Section 7: Summary Conformance Matrix:** Complete table comparing observed metrics against frozen acceptance thresholds.

### 9.5 `research/campaigns/campaign_002/CAMPAIGN_002_DECISION_MEMO.md`
- **Header:** Campaign ID, Target Path, Date, Synthesis Director, Epistemic Baseline.
- **Section 1: Core Research Question & Scope:** Re-statement of RQ2 and reminder of non-negotiable boundaries (zero backdoor training, zero harmful payloads).
- **Section 2: Summary of Empirical Findings:** Tabulated metrics across Tensor Error, Logit Divergence, Token Agreement, and Determinism against frozen thresholds.
- **Section 3: The Binding Operational Determination:** Explicit selection of one of:
  - **`PASS`**: Full conformance verified; authorize WP2 (Clean Surface) and WP3 (Bounded Training).
  - **`CONDITIONAL PASS`**: Conformance holds under explicit documented operational constraints.
  - **`FAIL`**: Material divergence observed; halt backdoor training permanently and pivot to empirical proxy-to-deployment transfer gap paper per Decision D15.
- **Section 4: Evidence Strength & Justification:** Rationale directly referencing empirical logs, confidence intervals, and gate criteria.
- **Section 5: Limitations & Remaining Uncertainties:** Analysis of unmeasured hardware configurations, kernel updates, or scale drift.
- **Section 6: Campaign 003 Handoff Criteria:** Concrete technical requirements, configs, and constraints handed off to the next work package.

---

## 10. Scientific & Epistemic Guardrails (Constitutional Rules)

Mined from `AGENTS.md`, `CONSOLIDATED_RESEARCH_PLAN.md`, and `CAMPAIGN_001_DECISION_MEMO.md`:

1. **Strictly Zero Backdoor Training in Campaign 002:**
   - No models may be fine-tuned.
   - No gradient updates targeting backdoor markers or conditional losses may be executed.
   - All evaluations are conducted strictly on the untouched clean base model $\theta_c$.
2. **Strictly Zero Harmful Behavior Targets:**
   - No safety jailbreaks, toxic completions, or harmful refusal suppressions may be evaluated.
   - If auxiliary benchmarks are tested, they are restricted to standard capability benchmarks (IFEval, GSM8K, WikiText-2).
3. **Strictly Zero Novelty Claims:**
   - Campaign 002 is an engineering and numerical verification campaign.
   - Novelty is provisionally narrow (`plausibly distinct`) and cannot be expanded or claimed based on runtime conformance results.
4. **Pre-Registered Acceptance Thresholds (Zero Post-Hoc Fitting):**
   - Acceptance thresholds must be frozen prior to confirmatory analysis.
   - "Fishing" for metrics, altering layer selection post-hoc, or adjusting thresholds to manufacture a pass is scientific fraud.
5. **Traceability and Replayability:**
   - Every empirical number must be traceable to a deterministic log, random seed, and configuration hash.
6. **Hardware Fallback Prohibition:**
   - Any silent fallback from hardware FP8 Tensor Cores to simulated FP8 or BF16 must be detected and treated as an immediate test failure.
7. **Preservation of Negative Results:**
   - A `FAIL` verdict is a completely valid, publishable scientific outcome that demarcates the boundary between simulated ML and production serving reality. It must never be obscured or soft-pedaled.

---
*End of Campaign 002 Specification Mining Report.*  
*Authored by `spec_miner_survey_1`.*
