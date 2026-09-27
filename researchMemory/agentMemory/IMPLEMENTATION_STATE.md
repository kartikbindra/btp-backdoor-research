# Implementation State & Engineering Architecture

**Document ID:** `IMPLEMENTATION_STATE.md`  
**Date of Snapshot:** 2026-09-27 (Post-Campaign 002 Remediation & Conformance Gate CONDITIONAL PASS)  
**Status:** **ACTIVE CODEBASE ESTABLISHED (Phase 0 & 1 Complete; Phase 2 Authorized Under Pre-Registered Conditions)**  
**Governing Protocols:** `AGENTS.md`, `PROJECT.md`, `CONSOLIDATED_RESEARCH_PLAN.md`, `CAMPAIGN_002_DECISION_MEMO.md`  

---

## 1. Codebase Audit (Current State as of 27 September 2026)

### 1.1 Physical File Inspection
A recursive inspection of the active project root (`c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research`) confirms that the foundational codebase has been established across modular packages:

- **Source Code Packages (`src/`):** **15 active Python modules**
  - `src/runtime/`: Hardware environment inspection, production vLLM runner configuration, analytical runtime tracer.
  - `src/compression/`: Static/dynamic scale calculation, Straight-Through Estimator fake-FP8 quantizer, storage-only ablation module.
  - `src/harness/`: Dynamic cache adapter hooks, deterministic greedy decoding harness, memory isolation manager.
  - `src/eval/`: Multi-level representation and generation metric suite, pilot calibration runner, confirmatory conformance runner.
- **Test Suites (`tests/`):** **4 comprehensive test modules** (45 unit and regression test methods passing cleanly via `unittest`)
  - `tests/test_kernel_fallback.py`: 31 test methods covering 5-tier hardware fallback traps, CPU device rejection, Tier 5 `KernelFallbackError`, uninspectable element size rejection, INT8 buffer substitution rejection, and runner config mutation traps.
  - `tests/test_determinism.py`: 3 test methods covering multi-run bitwise parity, logit drift assertion, process restart invariance, and memory isolation.
  - `tests/test_fake_fp8_ste.py`: 6 test methods covering STE gradient pass-through, scaling math, and reconstruction fidelity.
  - `tests/test_saturation_clipping.py`: 5 test methods covering saturation clipping under activation spikes, dynamic range clamping, and underflow handling.
- **Automation Scripts (`scripts/`):** **3 executable CLI entry points**
  - `scripts/run_wp0_manifest.py`: Validates environment locking, traces runtime path, and runs kernel fallback tests.
  - `scripts/run_wp1_conformance.py`: Runs pilot calibration, 50-run determinism baseline, and confirmatory 3-condition conformance evaluation.
  - `scripts/run_adversarial_audit.py`: Stress-tests context length scaling (128–2048 tokens), activation outlier spikes, and fallback traps.
- **Configuration Assets (`configs/`):** **3 pinned configuration specifications**
  - `configs/env/environment_spec.yaml`: Pinned model commit, hardware specs, framework versions, and quantization boundaries.
  - `configs/prompts/benign_prompt_clusters.json`: Sequestered benign evaluation clusters (Code, Science, Reasoning, Summary).
  - `configs/acceptance/frozen_thresholds.yaml`: Pre-registered, frozen Gate UG1 and UG2 acceptance thresholds.
- **Research Campaign Artifacts (`research/campaigns/campaign_002/`):**
  - `CAMPAIGN_002_ENVIRONMENT_MANIFEST.md` (Pinned hardware and software stack)
  - `CAMPAIGN_002_RUNTIME_PATH.md` (Formal 6-stage execution graph and noise factorization)
  - `CAMPAIGN_002_DETERMINISM.md` (Gate UG1 determinism baseline evaluation)
  - `CAMPAIGN_002_PROXY_CONFORMANCE.md` (Gate UG2 9-metric conformance matrix and noise factorization)
  - `CAMPAIGN_002_DECISION_MEMO.md` (Authoritative CONDITIONAL PASS verdict authorizing WP2/WP3)

---

### 1.2 Status of Implementation Phases

| Phase | Description & Associated Work Packages | Implementation Status | Test Coverage | Conformance Status |
|---|---|:---:|:---:|:---:|
| **Phase 0** | Cache Instrumentation & Deterministic A/B Harness (WP0) | **100% (Complete)** | 31 tests (`test_kernel_fallback.py`) | **PASSED (Gate UG0)** |
| **Phase 1** | Non-Adversarial Clean Baseline Evaluation (WP1) | **100% (Complete)** | 14 tests (`test_determinism`, `test_fake_fp8_ste`, `test_saturation`) | **CONDITIONAL PASS (Gate UG1 & UG2)** |
| **Phase 2** | Clean Surface Pilot & Dual-Regime Backdoor Training (WP2, WP3) | **AUTHORIZED & ACTIVE (0% Complete)** | Pending Phase 2 tests | **UNBLOCKED BY D18** |
| **Phase 3** | Real-Runtime Causal Transfer & Specificity Sweeps (WP4) | **0% (Planned)** | Pending Phase 3 | Blocked on Phase 2 |
| **Phase 4** | FP8 Mechanistic Probing & Layer/Head Localization (WP5) | **0% (Planned)** | Pending Phase 4 | Blocked on Phase 3 |
| **Phase 5** | Cache-Aware Differential Auditing & Defenses (WP6) | **0% (Planned)** | Pending Phase 5 | Blocked on Phase 3 |
| **Extension** | Gated PF-SEB Program: Suppressor Eviction (WP7–WP9) | **0% (Quarantined)** | Pending UG6 Pass | Strictly Gated behind UG6 |

---

## 2. Pinned Engineering & Deployment Stack

### 2.1 Hardware Requirements
- **Development & Verification Host:** Windows 11 / WSL2 (Ubuntu 22.04 LTS), used for analytical modeling, metric verification, and simulation.
- **Production Serving & Training Host:** Dedicated Linux Ubuntu 22.04 LTS instance with NVIDIA Ada Lovelace (`sm_89`, e.g., RTX 4090 / L40S) or Hopper (`sm_90`, e.g., H100 SXM5 / PCIe).
- **Compute Capability Constraint:** Native FP8 Tensor Cores require SM capability $\ge 8.9$. Architectures with $\text{SM} < 8.9$ (Ampere `sm_80`, Turing `sm_75`) are strictly disallowed for hardware FP8 evaluation.
- **Minimum GPU Memory:** 24 GB VRAM (sufficient for Qwen2.5-1.5B weights at 3.08 GB + KV cache up to 4096 tokens).
- **Driver & CUDA:** NVIDIA Driver $\ge 550.54.14$, CUDA 12.4.1 (Toolkit release V12.4.131).

### 2.2 Pinned Python Toolchain
- **Language Runtime:** Python `3.11.9`
- **Core Deep Learning Framework:** PyTorch `2.4.0+cu124` (with native `torch.float8_e4m3fn` support)
- **Serving Engine:** `vllm == 0.6.0` (compatible with v0.26.0+)
- **Model Architecture & Tokenization:** Hugging Face `transformers == 4.45.1`, `tokenizers == 0.20.0`
- **Fine-Tuning & Quantization Isolation:** `peft == 0.12.0`, `accelerate == 0.34.2`
- **Attention Kernel Backends:** `flash-attn == 2.6.3`, `flashinfer == 0.1.6+cu124`, `triton == 3.0.0`
- **Quantization Calibration:** `llm-compressor == 0.1.0`
- **Scientific Computing:** `numpy == 1.26.4`, `scipy == 1.13.1`, `pyyaml == 6.0.2`

---

## 3. Active Codebase Architecture

```text
btp-research/
├── configs/
│   ├── acceptance/
│   │   └── frozen_thresholds.yaml         <-- Pre-registered Gate UG1 & UG2 thresholds
│   ├── env/
│   │   └── environment_spec.yaml          <-- Pinned hardware, model, and framework specs
│   └── prompts/
│       └── benign_prompt_clusters.json    <-- Sequestered benign evaluation prompt clusters
├── src/
│   ├── runtime/
│   │   ├── __init__.py
│   │   ├── env_inspector.py               <-- 5-tier hardware fallback trap battery
│   │   ├── vllm_runner.py                 <-- Production vLLM engine runner configuration
│   │   └── runtime_tracer.py              <-- Analytical 6-stage execution graph tracer
│   ├── compression/
│   │   ├── __init__.py
│   │   ├── fake_fp8.py                    <-- Straight-Through Estimator (STE) fp8_e4m3fn proxy
│   │   ├── scales.py                      <-- Static and dynamic per-head scaling formulas
│   │   └── storage_fp8.py                 <-- Intermediate storage ablation (FP8 store + BF16 SDPA)
│   ├── harness/
│   │   ├── __init__.py
│   │   ├── cache_adapter.py               <-- Dynamic hook injecting T_proxy / T_storage into attention
│   │   ├── deterministic_decode.py        <-- Greedy decoding loop with logit drift tracking
│   │   └── memory_isolation.py            <-- Cache deallocation and cross-request isolation
│   └── eval/
│       ├── __init__.py
│       ├── metrics.py                     <-- NRMSE, Cosine Sim, Spearman rho, JSD, token match
│       ├── pilot_calibration.py           <-- Pilot dataset calibration runner
│       └── run_conformance.py             <-- 3-condition matrix and bootstrap CI calculator
├── tests/
│   ├── test_kernel_fallback.py            <-- 31 tests for fallback traps and allocation audits
│   ├── test_determinism.py                <-- 50-run repeatability, restart invariance, isolation
│   ├── test_fake_fp8_ste.py               <-- Forward quantization, backward autograd pass, scaling
│   └── test_saturation_clipping.py        <-- Outlier spikes, saturation clipping, underflow handling
├── scripts/
│   ├── run_wp0_manifest.py                <-- CLI verification runner for WP0 environment locking
│   ├── run_wp1_conformance.py             <-- CLI runner for 3-condition conformance matrix
│   └── run_adversarial_audit.py           <-- CLI stress-test runner for context scaling & outliers
├── research/
│   └── campaigns/
│       └── campaign_002/
│           ├── CAMPAIGN_002_ENVIRONMENT_MANIFEST.md
│           ├── CAMPAIGN_002_RUNTIME_PATH.md
│           ├── CAMPAIGN_002_DETERMINISM.md
│           ├── CAMPAIGN_002_PROXY_CONFORMANCE.md
│           └── CAMPAIGN_002_DECISION_MEMO.md
└── researchMemory/
    └── agentMemory/
        ├── CURRENT_STATE.md
        ├── DECISION_LOG.md
        ├── EXPERIMENT_REGISTRY.md
        ├── FINDINGS.md
        ├── IMPLEMENTATION_STATE.md
        └── CHANGELOG.md
```

---

## 4. Key Implemented Component Interfaces

### 4.1 Runtime Environment Inspector (`src/runtime/env_inspector.py`)
```python
def inspect_environment() -> Dict[str, Any]:
    """Inspects host OS, Python, CUDA runtime, PyTorch, and vLLM versions."""

def assert_fp8_hardware_support(device_index: int = 0) -> bool:
    """Asserts SM compute capability >= 89. Raises HardwareIncompatibilityError if SM < 89."""

def verify_cache_dtype(cache_tensor: torch.Tensor, expected_dtype: torch.dtype) -> bool:
    """Verifies physical element_size() == 1 for FP8 buffers. Raises CacheAllocationError."""

def audit_vllm_cache_argument(kv_cache_dtype: str) -> None:
    """Rejects 'auto' or 'bfloat16' when FP8 is claimed. Raises SilentFallbackError."""
```

### 4.2 Straight-Through Estimator FP8 Quantizer (`src/compression/fake_fp8.py`)
```python
class FP8QuantizeSTE(torch.autograd.Function):
    """
    Straight-Through Estimator for torch.float8_e4m3fn.
    Forward Pass: Quantizes tensor into [-448.0, 448.0] discrete bins scaled by S.
    Backward Pass: Passes incoming gradients unchanged within [-448.0 * S, 448.0 * S];
                   clamps gradients to zero outside the dynamic range boundary.
    """
    @staticmethod
    def forward(ctx, x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor: ...

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]: ...
```

### 4.3 Dynamic Cache Adapter Hook (`src/harness/cache_adapter.py`)
```python
class CacheAdapter:
    """
    Hooks into Qwen2Attention modules during forward pass.
    Intercepts key_states and value_states and applies condition-specific transformations:
      - Condition A: None (BF16 reference)
      - Condition C (T_proxy): FP8QuantizeSTE with per-head scaling
      - Condition Storage (T_storage): FP8 storage quantization followed by BF16 dequantization
    """
    def attach(self, model: torch.nn.Module, condition: str = "proxy") -> None: ...
    def remove(self) -> None: ...
```

### 4.4 Multi-Level Conformance Metrics Suite (`src/eval/metrics.py`)
```python
def tensor_nrmse(t_ref: torch.Tensor, t_eval: torch.Tensor) -> float: ...
def tensor_cosine_similarity(t_ref: torch.Tensor, t_eval: torch.Tensor) -> float: ...
def logit_spearman_rank(logits_ref: torch.Tensor, logits_eval: torch.Tensor) -> float: ...
def output_jsd(logits_ref: torch.Tensor, logits_eval: torch.Tensor) -> float: ...
def token_match_rate(tokens_ref: List[int], tokens_eval: List[int]) -> float: ...
def compute_bootstrap_ci(data: List[float], n_boot: int = 1000, alpha: float = 0.05) -> Tuple[float, float]: ...
```

---

## 5. Next Engineering Steps: Phase 2 Execution Plan

With Gate UG2 cleared and authorized by Decision D18:

1. **Work Package WP2 Implementation (Clean Surface Pilot):**
   - Implement benchmark evaluation harness for IFEval, GSM8K, and WikiText-2 perplexity (`src/eval/benchmarks.py`).
   - Run clean baseline model $\theta_c$ and task-adapted control $\theta_f$ under $C_0$, $T_{\text{real}}$, and near-misses.
   - Calibrate and lock empirical utility non-inferiority margins $\delta_{\text{margin}}$ for Gate UG5.
2. **Work Package WP3 Implementation (Bounded LoRA Training Loop):**
   - Implement PEFT/LoRA wrapper targeting $W_q, W_k, W_v, W_o$ (`r=16, \alpha=32`) with MLP parameters frozen.
   - Implement dual-branch forward pass loss function:
     $$\mathcal{L} = \mathcal{L}_{\text{task}}(C_0) + \mathcal{L}_{\text{task}}(T_{\text{proxy}}) + 2.0 \cdot \mathcal{L}_{\text{marker}}(T_{\text{proxy}}) + 1.5 \cdot \mathcal{L}_{\text{stealth}}(C_0) + 0.5 \cdot \mathcal{L}_{\text{teacher}}(KL)$$
   - Execute training across 2 independent random seeds for at most 3 pre-registered hyperparameter choices.
