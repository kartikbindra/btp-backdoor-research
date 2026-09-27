# Comprehensive Codebase Survey & Gap Analysis Report
## Campaign 002: Work Package WP0/WP1 Runtime Gate

**Author:** `explorer_codebase_1`  
**Date:** 2026-09-27  
**Working Directory:** `.agents/teamwork/explorer_codebase_1/`  
**Target Milestone:** Campaign 002 — WP0/WP1 Runtime Gate  
**Integrity Mode:** Development (Evidence discipline strictly enforced per `AGENTS.md`)  

---

## 1. Executive Summary

A comprehensive, recursive physical inspection of the repository at `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research` reveals that the repository currently exists at a **pure research design and pre-implementation stage**. 

### Key Findings
- `[SOURCE FACT]` **Zero physical code files exist:** Across the entire repository, there are currently **0 Python (`.py`) files**, **0 test files**, **0 configuration files (`.json`, `.yaml`, `.toml`)**, **0 shell scripts (`.sh`, `.ps1`)**, and **0 compiled CUDA/C++ kernels**.
- `[SOURCE FACT]` **Directory structure status:** The expected implementation directories (`src/`, `tests/`, `experiments/`, `configs/`, `scripts/`) do **not exist** in the filesystem.
- `[SOURCE FACT]` **Complete research architecture available:** High-specification architectural blueprints, mathematical equations for Straight-Through Estimator (STE) FP8 quantization, serving parameters for pinned vLLM (`v0.26.0+`), target base model specifications (`Qwen/Qwen2.5-1.5B-Instruct`), and quantitative Gate UG2 criteria are comprehensively documented across `CONSOLIDATED_RESEARCH_PLAN.md`, `researchMemory/agentMemory/`, and `research/campaigns/campaign_001/agent_reports/`.
- `[INFERENCE]` **Implementation mandate:** To execute Campaign 002 (Requirements R1 through R4), the entire software stack for the 3-condition clean conformance matrix (Condition A: BF16 full cache, Condition B: pinned vLLM FP8 runtime, Condition C: PyTorch STE software proxy) must be implemented from the documented specifications.

---

## 2. Exact Physical Codebase Inventory

### 2.1 File & Directory Census

| Category | Path | Physical Status | Contents / Format |
|---|---|---|---|
| **Root Documentation** | `AGENTS.md`, `CONSOLIDATED_RESEARCH_PLAN.md`, `ORIGINAL_REQUEST.md` | Present | Markdown (Repository constitution, consolidated plan, mission prompt) |
| **Source Code** | `src/` | **Absent (0 files)** | Does not exist |
| **Test Suites** | `tests/` | **Absent (0 files)** | Does not exist |
| **Configs** | `configs/` | **Absent (0 files)** | Does not exist |
| **Scripts** | `scripts/` | **Absent (0 files)** | Does not exist |
| **Experiments** | `experiments/` | **Absent (0 files)** | Does not exist |
| **Research Artifacts** | `research/campaigns/campaign_001/` | Present | Track reports A–F, Decision Memo, teamwork state |
| **Campaign 002 Charters** | `research/campaigns/campaign_002/` | Present | Master Prompt, Acceptance Template, Start Guide, AGENTS.md |
| **Research Memory** | `researchMemory/agentMemory/` | Present | 15 Markdown canonical memory files |
| **Legacy Archives** | `researchMemory/calude_research_mem/`, `chatgpt_...`, `deepseek_...` | Present | Markdown notes and transcripts from prior planning |
| **Word Documents** | `researchMemory/*.docx` | Present | 3 Word documents (PF-SEB research plans and synopses) |

Total project files (excluding `.git`): **76 files**, of which 73 are `.md` and 3 are `.docx`.

### 2.2 Model Assets & Checkpoints

- `[SOURCE FACT]` **Target Model:** `Qwen/Qwen2.5-1.5B-Instruct` is specified across all canonical documents (`CONSOLIDATED_RESEARCH_PLAN.md` §11, `DECISION_MEMO.md` §11.1, `TRACK_C_EXPERIMENTAL_SCIENTIST.md` §3.2.1).
- `[SOURCE FACT]` **Local Weight Files:** **0 model weight files or cached HuggingFace checkpoints exist within the repository workspace.**
- `[INFERENCE]` Model weights must be downloaded programmatically via HuggingFace `transformers` (`from_pretrained`) to the system cache or specified directory during environment initialization.
- `[SOURCE FACT]` **Model Architectural Parameters:**
  - Standard autoregressive Transformer with Grouped-Query Attention (GQA).
  - 28 layers, 12 query heads, 2 KV heads (KV sharing factor of 6), head dimension $d = 128$.
  - Hidden dimension: 896 / 1536; RoPE base frequency: 1,000,000; native context window: 32,768 tokens (evaluations standardized to $\le 2,048$ tokens).
  - Reference precision: Bfloat16 (`torch.bfloat16`).

---

## 3. Analysis of Candidate FP8 KV-Cache Proxy Implementations

### 3.1 Mathematical Specification of Candidate Software Proxy ($T_{proxy}$)

Documented in `TRACK_C_EXPERIMENTAL_SCIENTIST.md` §3.3.1 and `CONSOLIDATED_RESEARCH_PLAN.md` §6:

$$\tilde{X} = T_{proxy}(X) = \operatorname{Dequantize}\left(\operatorname{Quantize}(X, S), S\right)$$

Where for format `torch.float8_e4m3fn` (1 sign bit, 4 exponent bits, 3 mantissa bits; dynamic range $[-448, 448]$):

1. **Scale Factor Calculation ($S$):**
   $$S = \frac{\max(|X|) + \epsilon}{448.0}$$
   *Granularity options:* Per-tensor (single scalar per layer key/value tensor) or Per-head (independent scalar per GQA head).

2. **Quantization with Saturation / Clamping:**
   $$X_q = \operatorname{clamp}\left(\left\lfloor \frac{X}{S} \right\rceil_{FP8}, -448, 448\right)$$

3. **Dequantization back to BF16:**
   $$\tilde{X} = X_q \times S$$

4. **Straight-Through Estimator (STE) Gradient Formulation:**
   $$\frac{\partial \mathcal{L}}{\partial X} \approx \frac{\partial \mathcal{L}}{\partial \tilde{X}} \cdot \mathbb{I}\left( |X| \le 448 \cdot S \right)$$

### 3.2 Physical Pinned vLLM Production Runtime ($T_{real}$)

Documented in `TRACK_C_EXPERIMENTAL_SCIENTIST.md` §3.3.2 and `CAMPAIGN_002_MASTER_PROMPT.md`:
- Engine: `vllm == 0.26.0+`
- Key CLI configuration: `--kv-cache-dtype fp8`
- Memory allocation: PagedAttention block manager allocates 8-bit byte buffers for cached KV blocks.
- Compute path: PagedAttention CUDA/Triton kernels execute FP8 Tensor Core matrix multiplications for query-key dot product ($Q \cdot K^T$) and attention-value accumulation ($\operatorname{Softmax}(A) \cdot V$).

### 3.3 Storage vs. Compute Arithmetic Disentanglement ($T_{storage}$)

A critical scientific distinction established in Campaign 001 (`TRACK_C` §3.3.3) is separating:
1. **Storage Quantization Noise:** Precision truncation when moving tensors into 8-bit memory representation and dequantizing to BF16 before computation.
2. **Compute Kernel Fingerprinting:** Numerical non-associativity, rounding, and accumulator behavior of NVIDIA Ada/Hopper FP8 GEMM/GEMV Tensor Cores.

An intermediate experimental condition ($T_{storage}$) must be implemented: quantize KV to FP8 in memory, dequantize to BF16, and execute standard BF16 dot-product attention. This isolates whether behavioral changes stem from storage noise or hardware GEMM arithmetic.

### 3.4 Proxy Implementation Architectural Approaches

Three implementation approaches are evaluated for the candidate proxy:

| Architecture | Description | Pros | Cons | Recommendation |
|---|---|---|---|---|
| **Approach 1: `DynamicCache` Subclass** | Subclass `transformers.cache_utils.DynamicCache` to intercept `update()` calls, applying fake-FP8 quantization before storing key/value states. | Clean modular interface; standard HuggingFace integration; zero modification to model forward pass. | Does not easily capture intra-layer attention kernel intermediate quantization without monkey-patching attention. | **Primary for storage-noise modeling** |
| **Approach 2: PyTorch Forward Hooks** | Register forward hooks on `q_proj`, `k_proj`, `v_proj` or `Qwen2Attention` modules. | Non-invasive; easily enabled/disabled; compatible with standard `Qwen2ForCausalLM`. | Forward hooks on module outputs run before cache ingestion; post-hooks on cache states require access to internal layer states. | **Secondary / supplementary** |
| **Approach 3: Attention Module Monkey-Patching / Custom Attention Subclass** | Monkey-patch `Qwen2Attention.forward` or provide custom `InstrumentedQwen2Attention`. | Exact control over $Q \cdot K^T$, softmax scaling, and $A \cdot V$; allows precise insertion of simulated FP8 GEMM arithmetic. | Higher maintenance; tightly coupled to specific `transformers` version. | **Primary for compute-path simulation & ablation** |

---

## 4. Requirement-by-Requirement Code Gap Analysis

Below is the concrete mapping of what currently exists versus what must be created for Requirements R1 through R4:

### R1. Environment Locking & Production Runtime Path Inspection

- **Deliverables Required:**
  - `research/campaigns/campaign_002/CAMPAIGN_002_ENVIRONMENT_MANIFEST.md`
  - `research/campaigns/campaign_002/CAMPAIGN_002_RUNTIME_PATH.md`
- **What Exists:**
  - Theoretical descriptions and target specifications in `CONSOLIDATED_RESEARCH_PLAN.md` §11 and `CAMPAIGN_002_MASTER_PROMPT.md`.
- **What Must Be Created:**
  - `src/runtime/env_inspector.py`: Automated environment audit script collecting exact versions of Python, PyTorch, CUDA runtime, NVIDIA driver, GPU compute capability (SM architecture), vLLM package commit, and verifying FP8 Tensor Core support (`torch.cuda.is_available()`, `torch.float8_e4m3fn`).
  - `src/runtime/trace_runtime_path.py`: Dynamic call-graph and tensor-shape tracer logging the exact path: `Qwen2ForCausalLM → Qwen2Attention → k_proj/v_proj → KV cache memory allocation → PagedAttention kernel execution → logits`.
  - Fallback detector: Probing logic to detect whether vLLM or PyTorch silently falls back from FP8 hardware execution to BF16/FP32 simulation on unsupported hardware.

### R2. Determinism Baseline & 3-Condition Clean Conformance Matrix

- **Deliverables Required:**
  - `research/campaigns/campaign_002/CAMPAIGN_002_DETERMINISM.md`
  - `research/campaigns/campaign_002/CAMPAIGN_002_PROXY_CONFORMANCE.md`
- **What Exists:**
  - Formulations of NRMSE, Cosine Similarity, Spearman rank correlation, and greedy token match in `TRACK_C_EXPERIMENTAL_SCIENTIST.md` §3.7.
- **What Must Be Created:**
  - `src/compression/fake_fp8.py`: Differentiable PyTorch STE module implementing `fp8_e4m3fn` quantize-dequantize with per-tensor and per-head scaling, saturation clamping, and straight-through gradient backprop.
  - `src/harness/cache_adapter.py`: Unified abstraction wrapping:
    - Condition A: Reference uncompressed BF16 (`DynamicCache`)
    - Condition B: Pinned vLLM FP8 engine (`vLLM` Python AsyncLLMEngine / LLM entrypoint with `--kv-cache-dtype fp8`)
    - Condition C: Software proxy ($T_{proxy}$ with PyTorch STE)
    - Condition Ablation: Intermediate storage-only proxy ($T_{storage}$)
  - `src/harness/deterministic_decode.py`: Deterministic greedy generation harness (`temperature=0.0`) enforcing strict fresh-cache isolation (`del past_key_values`, `torch.cuda.empty_cache()`, no cross-request state).
  - `src/eval/metrics.py`: Numerical metric suite calculating:
    - Layerwise Cache NRMSE: $\frac{\|K_{proxy} - K_{real}\|_2}{\|K_{real}\|_2}$
    - Layerwise Cache Cosine Similarity: $\frac{\langle K_{proxy}, K_{real} \rangle}{\|K_{proxy}\|_2 \|K_{real}\|_2}$
    - Scale factor agreement: $|S_{proxy} - S_{real}| / S_{real}$
    - Next-token Logit Spearman rank correlation: $\rho(Z_{proxy}, Z_{real})$
    - End-to-end token match rate across greedy decoding trajectories.
  - `src/eval/run_conformance_matrix.py`: Orchestration script executing the 3-condition clean evaluation across benign prompt clusters.

### R3. Pre-Registered Acceptance Gate & Adversarial Conformance Audit

- **Deliverables Required:**
  - Pre-registered frozen threshold table in `CAMPAIGN_002_ACCEPTANCE_TEMPLATE.md`.
  - Audit sections in `CAMPAIGN_002_PROXY_CONFORMANCE.md`.
- **What Exists:**
  - Preliminary threshold bounds in Gate UG2 ($\text{CosSim} \ge 0.95$, $\rho_{Spearman} \ge 0.85$, $\text{NRMSE} < 0.15$).
  - Acceptance template in `research/campaigns/campaign_002/CAMPAIGN_002_ACCEPTANCE_TEMPLATE.md`.
- **What Must Be Created:**
  - `src/eval/pilot_calibration.py`: Script to measure natural variation on a pilot set of 20 prompts to freeze thresholds prior to the full confirmatory run (preventing post-hoc threshold fishing).
  - `tests/adversarial_audit/`: Automated audit battery testing:
    - `test_process_restart.py`: Process-level reproducibility across independent Python process invocations.
    - `test_context_lengths.py`: Conformance across sequence lengths (128, 512, 1024, 2048 tokens).
    - `test_scale_saturation.py`: Behavior under extreme dynamic ranges and saturation clipping ($|X| \ge 448 \cdot S$).
    - `test_kernel_fallback.py`: Assertion verifying physical FP8 Tensor Core invocation, failing if silent software fallback occurs.

### R4. Decision Memo & Canonical Research Memory Synchronization

- **Deliverables Required:**
  - `research/campaigns/campaign_002/CAMPAIGN_002_DECISION_MEMO.md` (explicit PASS / CONDITIONAL PASS / FAIL verdict).
  - Canonical memory updates in `researchMemory/agentMemory/` (`CURRENT_STATE.md`, `DECISION_LOG.md`, `EXPERIMENT_REGISTRY.md`, `FINDINGS.md`).
- **What Exists:**
  - Campaign 001 Decision Memo structure as precedent.
- **What Must Be Created:**
  - Synthesis script/module collecting empirical test logs and filling the pre-registered acceptance table.
  - Structured updates to canonical agent memory.

---

## 5. Target Codebase Directory Layout & Worker Ownership

To implement Campaign 002 cleanly without monolithic scripts or cross-role collisions, the following target architecture is recommended:

```text
btp-research/
├── configs/
│   ├── env/
│   │   └── environment_spec.yaml           # Pinned model, revision, seeds, dtypes
│   ├── prompts/
│   │   └── benign_prompt_clusters.json     # Sequestered benign evaluation prompts
│   └── acceptance/
│       └── frozen_thresholds.yaml          # Pre-registered frozen acceptance gates
├── src/
│   ├── runtime/
│   │   ├── __init__.py
│   │   ├── env_inspector.py                # GPU/CUDA/vLLM environment probe
│   │   ├── vllm_runner.py                  # Pinned vLLM FP8 engine runner (Condition B)
│   │   └── runtime_tracer.py               # Layer-by-layer runtime execution tracer
│   ├── compression/
│   │   ├── __init__.py
│   │   ├── fake_fp8.py                     # Differentiable PyTorch STE proxy (Condition C)
│   │   ├── storage_fp8.py                  # Storage-only FP8 ablation proxy
│   │   └── scales.py                       # Dynamic scaling calculation (per-tensor / per-head)
│   ├── harness/
│   │   ├── __init__.py
│   │   ├── cache_adapter.py                # Multi-condition cache abstraction
│   │   ├── deterministic_decode.py         # Deterministic greedy generation loop
│   │   └── memory_isolation.py             # Fresh cache reset and memory leak assertions
│   └── eval/
│       ├── __init__.py
│       ├── metrics.py                      # NRMSE, CosSim, Spearman rho, token match
│       ├── pilot_calibration.py            # Pre-registration pilot calibration runner
│       └── run_conformance.py              # 3-condition matrix evaluation runner
├── tests/
│   ├── test_determinism.py                 # Repeatability and process restart tests
│   ├── test_fake_fp8_ste.py                # STE gradient correctness and numerical parity
│   ├── test_saturation_clipping.py         # FP8 dynamic range saturation tests
│   └── test_kernel_fallback.py             # Silent fallback detection assertions
├── scripts/
│   ├── run_wp0_manifest.py                 # Generate environment manifest & runtime path
│   ├── run_wp1_conformance.py              # Execute 3-condition conformance matrix
│   └── run_adversarial_audit.py            # Execute adversarial audit test suite
├── requirements.txt                        # Pinned dependencies (torch, transformers, vllm, etc.)
└── pyproject.toml                          # Package configuration
```

### 5.1 Proposed Worker File Ownership

| Worker Role | Primary Owned Files | Responsibilities |
|---|---|---|
| **Worker 1: Runtime/vLLM Engineer** (`worker_vllm_runtime`) | `src/runtime/*`, `scripts/run_wp0_manifest.py`, `tests/test_kernel_fallback.py` | Inspect physical vLLM FP8 implementation, configure `--kv-cache-dtype fp8`, trace execution path, assert true hardware FP8 Tensor Core usage. |
| **Worker 2: KV-Cache Engineer** (`worker_kv_cache`) | `src/compression/*`, `src/harness/*`, `tests/test_fake_fp8_ste.py` | Build PyTorch STE `fake_fp8.py`, per-tensor/per-head scaling, storage ablation proxy, and cache adapter interface. |
| **Worker 3: Numerical Conformance Scientist** (`worker_conformance`) | `src/eval/metrics.py`, `src/eval/run_conformance.py`, `scripts/run_wp1_conformance.py` | Implement multi-level error metrics (NRMSE, CosSim, logit Spearman $\rho$), run 3-condition clean matrix. |
| **Worker 4: Statistics & Adversarial Auditor** (`worker_auditor`) | `src/eval/pilot_calibration.py`, `tests/test_determinism.py`, `tests/test_saturation_clipping.py`, `configs/acceptance/*` | Pre-register frozen thresholds, audit determinism across process restarts, test context lengths and saturation boundaries. |

---

## 6. Actionable Recommendations for Campaign 002

1. **Phase 0 Bootstrap:** Worker agents should initialize `requirements.txt` and core directory structures (`src/`, `tests/`, `configs/`, `scripts/`).
2. **GPU & Platform Sensitivity:**
   - Production vLLM FP8 requires Linux with CUDA 12.4+ and an NVIDIA Ada Lovelace (RTX 4090 / L40S) or Hopper (H100) GPU.
   - If running on a Windows or non-FP8 local machine, the testing harness must support automated detection and simulated execution modes while strictly flagging that hardware-level verification requires a compatible Linux CUDA host.
3. **Threshold Pre-Registration:** Before running the full 100+ prompt conformance matrix, run `src/eval/pilot_calibration.py` on 20 benign prompts to establish empirical confidence intervals and freeze thresholds in `configs/acceptance/frozen_thresholds.yaml`.
4. **Epistemic Integrity:** Uphold the non-negotiables: strictly zero backdoor training, zero harmful payloads, zero post-hoc metric fitting, and clear documentation of any proxy-runtime divergence.
