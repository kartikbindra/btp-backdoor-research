# Project: Campaign 002 — Work Package WP0/WP1 Runtime Gate

## Architecture
Campaign 002 evaluates whether the candidate FP8 KV-cache proxy ($T_{proxy}$, PyTorch STE) conforms to the pinned production vLLM FP8 runtime ($T_{real}$) on a clean, unmodified model ($\theta_c$: `Qwen/Qwen2.5-1.5B-Instruct`).

### Data Flow & Execution Graph
1. Reference Condition A (BF16):
   `Input Prompt -> Qwen2Attention -> BF16 KV Cache -> BF16 Attention -> Output Logits`
2. Production Condition B ($T_{real}$):
   `Input Prompt -> Qwen2Attention -> Static Scaling S -> Quantize to fp8_e4m3fn -> PagedAttention FP8 Cache -> Hardware Tensor Core GEMM (or SRAM dequant on sm_89) -> Output Logits`
3. Candidate Proxy Condition C ($T_{proxy}$):
   `Input Prompt -> Qwen2Attention -> Straight-Through Estimator (STE) Quantize-Dequantize fp8_e4m3fn -> PyTorch SDPA -> Output Logits`
4. Storage Ablation Condition ($T_{storage}$):
   `Input Prompt -> Qwen2Attention -> Quantize to FP8 in memory -> Dequantize to BF16 -> PyTorch SDPA -> Output Logits` (Isolates storage quantization noise from GEMM non-associativity rounding)

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | Hardware & OS Specification | Pin Linux Ubuntu 22.04 LTS, CUDA 12.4, Driver >= 550.54.14, Ada sm_89 / Hopper sm_90 | M1 | R1 (Survey) |
| 2 | Framework & Dependency Pinning | Pin vLLM 0.26.0+, PyTorch 2.4.0+cu124, Transformers 4.44.2, Flash-Attn 2.6.3 | M1 | R1 (Survey) |
| 3 | Model Revision Pinning | Pin Qwen/Qwen2.5-1.5B-Instruct exact commit hash, GQA architecture (12 Q, 2 KV, 28 L) | M1 | R1 (Survey) |
| 4 | Execution Graph & Tensor Dtypes | Document exact path: model -> K/V projection -> FP8 cache -> attention -> output | M1 | R1 (Survey) |
| 5 | Quantization & Scaling Formalism | Document fp8_e4m3fn dynamic range [-448, 448], static scale calculation S = (max(|X|) + eps)/448 | M1 | R1 (Survey) |
| 6 | PagedAttention & Block Layout | Detail 16/32 token block allocation, memory strides, 1-byte/element cache storage | M1 | R1 (Survey) |
| 7 | Hardware Fallback Elimination | Assert sm >= 89, trap silent fallback to BF16/software emulation, verify exact 1-byte allocation | M1 | R1 (Survey) |
| 8 | Bitwise Determinism Harness | Evaluate run-to-run repeatability over 50 runs on theta_c under greedy decoding (T=0, seed=42) | M2 | R2 (Survey) |
| 9 | Process Restart Invariance | Verify identical output tokens across independent OS process launches | M2 | R2 (Survey) |
| 10 | 3-Condition Matrix Execution | Execute Condition A (BF16), Condition B (vLLM FP8 T_real), Condition C (PyTorch proxy T_proxy) | M2 | R2 (Survey) |
| 11 | Storage vs GEMM Noise Factorization | Evaluate T_storage to separate storage quantization noise from kernel GEMM rounding effects | M2 | R2 (Survey) |
| 12 | Multi-Level Metric Suite | Compute Key/Value tensor NRMSE, Cosine Similarity, logit Spearman rho, JSD, greedy token match | M2 | R2 (Survey) |
| 13 | Pre-Registered Acceptance Gate | Freeze candidate thresholds on pilot data before confirmatory evaluation (Gate UG2) | M3 | R3 (Survey) |
| 14 | Context Length Scaling Audit | Adversarial audit across prompt lengths (128, 512, 2048 tokens) | M3 | R3 (Survey) |
| 15 | Scale Outlier & Saturation Audit | Stress-test activation outliers and clamping behavior at dynamic range limits | M3 | R3 (Survey) |
| 16 | Decision Memo Synthesis | Author CAMPAIGN_002_DECISION_MEMO.md rendering PASS, CONDITIONAL PASS, or FAIL | M4 | R4 (Survey) |
| 17 | Campaign 003 Handoff Criteria | Explicitly define gating and operational criteria for WP2/WP3 backdoor training | M4 | R4 (Survey) |
| 18 | Canonical Memory Synchronization | Update CURRENT_STATE.md, DECISION_LOG.md, EXPERIMENT_REGISTRY.md, FINDINGS.md | M4 | R4 (Survey) |
| 19 | Forensic Integrity & Anti-Cheating | Independent verification of zero synthetic shortcuts, zero hardcoded values, authentic logic | M5 | Audit (Constitution) |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M0 | Survey & Reconnaissance | Map all requirements, existing code, and hardware constraints | none | DONE |
| M1 | Environment Locking & Runtime Path Inspection | Generate CAMPAIGN_002_ENVIRONMENT_MANIFEST.md and CAMPAIGN_002_RUNTIME_PATH.md | M0 | DONE |
| M2 | Determinism Baseline & 3-Condition Conformance Matrix | Generate CAMPAIGN_002_DETERMINISM.md and CAMPAIGN_002_PROXY_CONFORMANCE.md | M1 | DONE |
| M3 | Pre-Registered Acceptance Gate & Adversarial Conformance Audit | Freeze thresholds, execute adversarial stress tests, audit silent fallbacks | M2 | DONE |
| M4 | Decision Memo & Canonical Research Memory Synchronization | Generate CAMPAIGN_002_DECISION_MEMO.md and update researchMemory/agentMemory/ | M3 | DONE |
| M5 | Forensic Integrity Audit & Final Verification Gate | Independent Forensic Auditor verification across all deliverables and non-negotiables | M4 | DONE |

## Interface Contracts
### Runtime Inspector -> Conformance Harness
- `environment_spec.yaml` specifies target device, compute capability, CUDA version, model commit, framework version.
- Python module `src/runtime/env_inspector.py`:
  - `inspect_environment() -> Dict[str, Any]`
  - `assert_fp8_hardware_support(device: str) -> bool`
  - `verify_cache_dtype(cache_tensor: torch.Tensor, expected_dtype: torch.dtype) -> bool`

### KV-Cache Proxy -> Conformance Evaluator
- `src/compression/fake_fp8.py`:
  - `FP8QuantizeSTE(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor`
  - Supports straight-through estimator backward pass for future WP2/WP3.
- `src/compression/scales.py`:
  - `calculate_static_scale(tensor: torch.Tensor, eps: float = 1e-5) -> float`
  - Returns $S = (\max(|X|) + \epsilon) / 448.0$.
- `src/harness/cache_adapter.py`:
  - Intercepts Qwen2Attention key/value states and applies condition-specific transformations ($T_{proxy}$, $T_{storage}$).

### Conformance Evaluator -> Metrics & Decision Memo
- `src/eval/metrics.py`:
  - `tensor_nrmse(t_ref: torch.Tensor, t_eval: torch.Tensor) -> float`
  - `tensor_cosine_similarity(t_ref: torch.Tensor, t_eval: torch.Tensor) -> float`
  - `logit_spearman_rank(logits_ref: torch.Tensor, logits_eval: torch.Tensor) -> float`
  - `output_jsd(logits_ref: torch.Tensor, logits_eval: torch.Tensor) -> float`
  - `token_match_rate(tokens_ref: List[int], tokens_eval: List[int]) -> float`

## Code Layout
```
configs/
├── env/environment_spec.yaml
├── prompts/benign_prompt_clusters.json
└── acceptance/frozen_thresholds.yaml
src/
├── runtime/
│   ├── env_inspector.py
│   ├── vllm_runner.py
│   └── runtime_tracer.py
├── compression/
│   ├── fake_fp8.py
│   ├── storage_fp8.py
│   └── scales.py
├── harness/
│   ├── cache_adapter.py
│   ├── deterministic_decode.py
│   └── memory_isolation.py
└── eval/
    ├── metrics.py
    ├── pilot_calibration.py
    └── run_conformance.py
tests/
├── test_determinism.py
├── test_fake_fp8_ste.py
├── test_saturation_clipping.py
└── test_kernel_fallback.py
research/campaigns/campaign_002/
├── CAMPAIGN_002_ENVIRONMENT_MANIFEST.md
├── CAMPAIGN_002_RUNTIME_PATH.md
├── CAMPAIGN_002_DETERMINISM.md
├── CAMPAIGN_002_PROXY_CONFORMANCE.md
└── CAMPAIGN_002_DECISION_MEMO.md
researchMemory/agentMemory/
├── CURRENT_STATE.md
├── DECISION_LOG.md
├── EXPERIMENT_REGISTRY.md
└── FINDINGS.md
```
