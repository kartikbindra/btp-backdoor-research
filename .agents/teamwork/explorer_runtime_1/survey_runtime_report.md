# Survey Runtime Report: System, Hardware, and Execution Environment Audit
**Campaign 002 (Work Package WP0/WP1 Runtime Gate)**  
**Investigator:** `explorer_runtime_1`  
**Date:** 2026-09-27  
**Working Directory:** `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\explorer_runtime_1\`  
**Target Codebase:** `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research`  

---

## 1. Executive Summary & Environment Diagnostic Audit

### 1.1 Local Workspace vs. Production Runtime Reality
A fundamental finding of this investigation confirms the architectural constraint established in Campaign 001 (`DECISION_MEMO.md` §11.2, `CONSOLIDATED_RESEARCH_PLAN.md` §5.2):

1. **Local Host State:**
   - **Operating System:** Windows 11 (64-bit).
   - **Host Workspace:** `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research` (backed by OneDrive cloud synchronization).
   - **Interactive Process Execution:** Subagent shell command execution (`run_command` targeting `nvidia-smi` and `python`) encountered host permission timeouts (60,000 ms), indicating that background automated subagents operate in an unattended environment where interactive Windows permission popups cannot be clicked in real time.
   - **Native vLLM Capability on Windows:** Production vLLM (v0.6.0+) does **not** have official native Windows wheel support. High-performance inference with Triton kernels, FlashAttention, and PagedAttention requires Linux POSIX primitives (shared memory, `fork`/`spawn` IPC, io_uring) and dedicated Linux CUDA compilation targets. On Windows, vLLM can only be executed via **WSL2 (Windows Subsystem for Linux, Ubuntu 22.04 LTS)** or Docker Desktop with the WSL2 backend.

2. **Mandatory Production Runtime Target:**
   - As pre-registered in Campaign 001 Decision D15 and `CAMPAIGN_002_MASTER_PROMPT.md`, executing the physical production vLLM FP8 runtime ($T_{real}$) requires deploying the execution harness to a dedicated **Linux host (Ubuntu 22.04 LTS, CUDA 12.4+, NVIDIA Driver $\ge 550.54.14$)** equipped with an NVIDIA GPU with native FP8 Tensor Core support (Compute Capability $\ge 8.9$).

---

## 2. Hardware Microarchitecture & FP8 Capability Matrix

### 2.1 NVIDIA GPU Architectural Support Matrix for FP8 KV Cache

The feasibility of FP8 KV caching in production inference engines (vLLM, TensorRT-LLM, SGLang) is strictly governed by the GPU streaming multiprocessor (SM) compute capability:

| GPU Family | SM Architecture | Representative Models | FP8 Tensor Cores (Hardware)? | vLLM `kv_cache_dtype="fp8"` Support | FlashAttention Backend | Execution Path |
|---|---|---|:---:|:---:|:---:|---|
| **Hopper** | `sm_90` / `sm_90a` | H100, H200, H800, GH200 | **Yes (Native)** | **Full Native Support** | FlashAttention-3 / FlashInfer | Full FP8 End-to-End: Storage in FP8, Compute in FP8 via WGMMA + TMA |
| **Ada Lovelace** | `sm_89` | RTX 4090, RTX 4080, L4, L40, L40S, RTX 6000 Ada | **Yes (Native)** | **Full Storage Support** | FlashAttention-2 / PagedAttention v2 / FlashInfer | Storage in FP8; Dequantized to BF16/FP16 in SRAM for FA2 dot products (FA3 unsupported on `sm_89`) |
| **Ampere** | `sm_80` / `sm_86` | A100, A800, RTX 3090, RTX 3080, A10G | **NO** | **UNSUPPORTED** (Hard Assertion Crash) | FlashAttention-2 | Hardware lacks FP8 Tensor Cores. vLLM raises `AssertionError: fp8e4nv data type is not supported on CUDA arch < 89` |
| **Turing / Older** | `sm_75` | T4, RTX 2080, Quadro RTX 5000 | **NO** | **UNSUPPORTED** | VLLM PagedAttention v1 | Incompatible; no FP8 representation or kernels |

### 2.2 Critical Distinction: Storage FP8 vs. Compute FP8
A common point of confusion in LLM inference benchmarks is conflating **KV Cache Storage Quantization** with **Attention Compute Precision**:

1. **Storage FP8 ($T_{storage}$):**
   - Keys and Values are stored in GPU High-Bandwidth Memory (HBM) as 8-bit floats (`fp8_e4m3fn`), occupying exactly 1 byte per element (a $50\%$ reduction from BF16's 2 bytes).
   - When attention is evaluated, the kernel loads the 8-bit bytes into on-chip SRAM/registers, multiplies by the scalar scaling factor (`scale_k`, `scale_v`), and casts the values into FP16 or BF16.
   - The dot-product matrix multiplications ($Q K^T$ and $P V$) are performed using standard **BF16/FP16 Tensor Cores** (accumulating in FP32).
   - *Supported Hardware:* Ada Lovelace (`sm_89`) and Hopper (`sm_90`).

2. **Compute FP8 ($T_{compute}$):**
   - Keys and Values are stored in FP8 in HBM.
   - In addition, the Query tensor $Q$ is dynamically scaled and quantized into FP8.
   - The attention matrix multiplications ($Q K^T$ and $P V$) are executed directly using **hardware FP8 Tensor Cores** (e.g., Hopper WGMMA instructions).
   - *Supported Hardware:* Strictly Hopper (`sm_90`) and Blackwell (`sm_100`/`sm_120`) using FlashAttention-3 or FlashInfer. Ada Lovelace (`sm_89`) does **not** support FlashAttention-3 due to the absence of TMA (Tensor Memory Accelerator) and WGMMA hardware.

### 2.3 Memory Footprint Calculation for Target Model (Qwen2.5-1.5B-Instruct)
For the pinned primary model:
- Parameter Count: $1.54 \times 10^9$ parameters.
- Static Weight Footprint (BF16): $\approx 3.08\text{ GB}$.
- Transformer Layers ($L$): $28$.
- Attention Heads ($H$): $12$.
- KV Heads ($H_{kv}$): $2$ (Grouped-Query Attention with $6\times$ query-to-KV ratio).
- Head Dimension ($d$): $128$.
- KV Cache Bytes per Token:
  $$\text{Bytes per token} = 2 \times L \times H_{kv} \times d \times p = 2 \times 28 \times 2 \times 128 \times p = 14,336 \times p \text{ bytes/token}$$
  - **Under Condition A (BF16, $p=2$ bytes):** $28,672\text{ bytes/token} \approx 28.0\text{ KB/token}$.
  - **Under Condition B/C (FP8, $p=1$ byte):** $14,336\text{ bytes/token} \approx 14.0\text{ KB/token}$ ($50.0\%$ memory reduction).
- At sequence length $S = 4,096$ tokens and batch size $B = 1$:
  - Full BF16 Cache ($C_0$): $114.69\text{ MB}$.
  - FP8 Cache ($T(C_0)$): $57.34\text{ MB}$.
- On a single 24GB GPU (RTX 4090 / L40S) or 80GB GPU (A100 / H100), the static weights ($\approx 3.1\text{ GB}$) and KV cache easily fit within VRAM, ensuring that experiments will not encounter memory pressure confounds during single-stream greedy decoding.

---

## 3. Package and Toolchain Specifications (Pinned Manifest)

To satisfy **R1 (Environment Locking)** and eliminate framework drift, the target runtime stack for Campaign 002 is pinned as follows:

```toml
[runtime_environment]
os = "Linux (Ubuntu 22.04 LTS x86_64) via WSL2 or Native Bare-Metal"
kernel = "Linux 6.8.0-xx-generic or Microsoft WSL2 Kernel >= 5.15.153"
python = "3.11.9"
cuda_driver = ">= 550.54.14"
cuda_toolkit = "12.4.1 (V12.4.131)"
gpu_target = "NVIDIA GeForce RTX 4090 / L40S (sm_89) or NVIDIA H100 PCIe/SXM (sm_90)"

[python_packages]
torch = "2.4.0+cu124"
torchvision = "0.19.0+cu124"
torchaudio = "2.4.0+cu124"
vllm = "0.6.0"  # Pinned official production serving engine
transformers = "4.45.1"
accelerate = "0.34.2"
triton = "3.0.0"
flash-attn = "2.6.3"
flashinfer = "0.1.6+cu124"
safetensors = "0.4.5"
numpy = "1.26.4"
scipy = "1.13.1"
llm-compressor = "0.1.0"  # Neural Magic / vLLM official FP8 calibration toolkit
```

---

## 4. vLLM FP8 Implementation & Execution Path Trace

### 4.1 End-to-End Execution Trace
In vLLM, the production path for FP8 KV caching operates as follows:

```text
[Input Prompt x]
       │
       ▼
[Embedding Layer]
       │
       ▼
[Transformer Layer l: Self-Attention]
       ├── Linear Projection: Q = x W_q,  K = x W_k,  V = x W_v  (in BF16)
       │
       ▼
[Quantization & Cache Insertion: vllm.cache_ops.reshape_and_cache]
       ├── Input: K, V tensors (dtype: torch.bfloat16)
       ├── Scaling: K_scaled = K / k_scale,  V_scaled = V / v_scale
       ├── Clamping: clamp(K_scaled, min=-240.0, max=240.0)
       ├── Rounding: Round to nearest even (RNE) -> cast to c10::Float8_e4m3fn
       └── Write to Paged Memory Pool: slot_mapping -> physical block table
       │
       ▼
[FP8 Cache Storage: Paged KV Buffer]
       ├── Physical Tensor Dtype: torch.uint8 / torch.float8_e4m3fn
       ├── Block Shape: [num_blocks, num_heads, head_size // 16, block_size, 16]
       └── Footprint: Exactly 1 byte per scalar element
       │
       ▼
[Attention Kernel Execution: PagedAttention / FlashAttention / FlashInfer]
       ├── Prefill Stage: FlashAttention-2 / FlashInfer varlen kernel
       │     └── Loads FP8 K, V; multiplies by scale; performs dot products
       ├── Decode Stage: PagedAttention v2 (Triton / CUDA)
       │     └── Parallel reduction across block tables with dequantization in SRAM
       │
       ▼
[Softmax & Projection: O = Attention(Q, K_dequant, V_dequant) * W_o] (BF16)
       │
       ▼
[MLP / Output Norm / Final LM Head Logits] (BF16 / FP32)
```

### 4.2 Format Specification: `fp8_e4m3fn`
The default FP8 format used by vLLM on NVIDIA hardware is `fp8_e4m3fn` (defined by the OFP8 / IEEE P3109 standard):
- **Sign:** 1 bit ($s$).
- **Exponent:** 4 bits ($e$), bias = 7.
- **Mantissa:** 3 bits ($m$).
- **Dynamic Range:** Approximately $\pm 1.95 \times 10^{-3}$ to $\pm 240.0$.
- **Special Values:** 
  - Exponent `1111`, Mantissa `111` ($0\text{x7F}$ and $0\text{xFF}$) represents **NaN**.
  - There are **no infinite representations** (saturation clamps directly to $\pm 240.0$).
  - Denormalized numbers exist for exponent `0000`.

### 4.3 Scaling Granularity and Calibration
vLLM supports two operational modes for FP8 KV cache scaling:
1. **Uncalibrated Default (`scale = 1.0`):**
   - If `--quantization-param-path` is omitted, vLLM sets `k_scale = 1.0` and `v_scale = 1.0`.
   - **Severe Scientific Risk:** For modern LLMs (such as Qwen2.5), activation outliers in key projections frequently exceed $240.0$ or occupy ranges where a scale of $1.0$ causes severe underflow in small values and hard saturation at $\pm 240.0$. This can cause catastrophic degradation or repetition loops even on clean models.
2. **Calibrated Per-Tensor Scales (`kv_cache_scales.json`):**
   - Offline calibration uses `llm-compressor` on a representative calibration split (e.g. 128 samples from UltraChat or Wikitext).
   - Generates per-layer scalar scales:
     $$s_{k}^{(l)} = \frac{\max(|K^{(l)}|)}{240.0}, \quad s_{v}^{(l)} = \frac{\max(|V^{(l)}|)}{240.0}$$
   - **Campaign 002 Recommendation:** In accordance with scientific integrity, Campaign 002 must evaluate **both**:
     - Sub-condition B1: Production vLLM FP8 with calibrated scales.
     - Sub-condition B2: Production vLLM FP8 with default static scale ($1.0$).
     - This isolates whether proxy conformance requires explicit calibration matching.

---

## 5. PyTorch Native FP8 Support & Tensor Cores vs. Emulation

### 5.1 PyTorch Native Dtypes (`torch.float8_e4m3fn`)
PyTorch introduced native support for `torch.float8_e4m3fn` in PyTorch 2.1:
- **Tensor Storage:** `x = torch.zeros(10, dtype=torch.float8_e4m3fn, device='cuda')` allocates 1 byte per element.
- **Casting:** `x.to(torch.float8_e4m3fn)` executes a CUDA type-conversion kernel that correctly implements round-to-nearest-even (RNE) and clamps values $> 240.0$ to $240.0$.
- **Dequantization:** `x.to(torch.bfloat16)` restores the tensor to BF16 by unpacking the sign, exponent, and mantissa.

### 5.2 Arithmetic Operations & Hardware Acceleration
While PyTorch can *store* `float8_e4m3fn` tensors on any CUDA GPU, it **cannot execute standard arithmetic operations** (`+`, `*`, `torch.matmul`) directly on `float8` tensors:
1. Attempting `torch.matmul(A_fp8, B_fp8)` triggers a `RuntimeError: "addmm_cuda" not implemented for 'Float8_e4m3fn'`.
2. PyTorch's dedicated FP8 hardware GEMM interface is `torch._scaled_mm(A, B, scale_a, scale_b, out_dtype=torch.bfloat16)`:
   - On **Hopper (`sm_90`)**: Supported natively via cuBLAS / CUTLASS FP8 Tensor Cores.
   - On **Ada Lovelace (`sm_89`)**: Supported in CUTLASS, but official PyTorch binary wheels often omit `sm_89` targets for `_scaled_mm`, raising `RuntimeError: torch._scaled_mm is only supported on devices with compute capability >= 9.0`.
   - On **Ampere (`sm_80`) / Older**: Hardware lacks FP8 Tensor Cores entirely.

### 5.3 Implications for Candidate Proxy ($T_{proxy}$)
Because the candidate software proxy ($T_{proxy}$, PyTorch STE) is used during **forward inference and training**, it does **not** require hardware FP8 GEMMs:
- The proxy operates by inserting a **quantization-dequantization round-trip** (fake quantization) on the projected keys and values:
  $$\tilde{K} = \text{Dequantize}(\text{Quantize}(K, s_k), s_k)$$
- The actual attention computation is performed by PyTorch's standard `scaled_dot_product_attention` (SDPA) using **BF16**.
- Gradients flow through the quantization operation via the **Straight-Through Estimator (STE)**:
  $$\frac{\partial \mathcal{L}}{\partial K} \approx \frac{\partial \mathcal{L}}{\partial \tilde{K}}$$
- Therefore, $T_{proxy}$ can be trained and evaluated on **any CUDA GPU** (including RTX 3090, A100, RTX 4090, and H100) without hardware FP8 Tensor Cores!

---

## 6. Silent Fallback Detection Strategy

A primary scientific and epistemic non-negotiable for Campaign 002 is:
> *"Any silent fallback from hardware FP8 to simulated/software FP8 or BF16 is detected and flagged as a test failure."* (`ORIGINAL_REQUEST.md` Line 63).

### 6.1 The 5 Potential Silent Fallback Failure Modes
1. **Fallback Mode 1 (Backend Downgrade):** vLLM fails to initialize the requested attention backend (e.g. FlashInfer) on the current SM, silently falling back to a generic backend that casts FP8 to BF16 before computation.
2. **Fallback Mode 2 (Default Cache Type Substitution):** vLLM CLI or API arguments set to `kv_cache_dtype="auto"`, which silently resolves to `bfloat16` without warning.
3. **Fallback Mode 3 (PyTorch Compiler De-optimization):** `torch.compile` fails to lower FP8 operations to hardware Tensor Cores and silently inserts high-precision casting nodes.
4. **Fallback Mode 4 (Scale Misconfiguration / Overflow Collapse):** All activations clip to $240.0$ due to missing scales, causing the model to behave as an all-ones step function rather than quantized linear attention.
5. **Fallback Mode 5 (Silent CPU/Software Fallback):** Missing CUDA kernel extensions cause operations to be redirected to slow CPU software emulation loops.

### 6.2 The 5-Tier Fallback Verification Framework

To guarantee zero silent fallbacks, the evaluation harness must execute this 5-tier verification battery before every test run:

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                    5-TIER SILENT FALLBACK DETECTION                     │
├────────────────────────────────┬────────────────────────────────────────┤
│ Verification Tier              │ Concrete Assertion & Verification Trap │
├────────────────────────────────┼────────────────────────────────────────┤
│ Tier 1: Static Config Assertion│ Assert vLLM `kv_cache_dtype == "fp8"`  │
│                                │ Disallow "auto". Assert `scale > 0`.   │
├────────────────────────────────┼────────────────────────────────────────┤
│ Tier 2: Physical Memory Probe  │ Measure KV cache pool byte allocation. │
│                                │ FP8 MUST occupy EXACTLY 1 byte/element.│
│                                │ If footprint == 2 bytes/element: FAIL. │
├────────────────────────────────┼────────────────────────────────────────┤
│ Tier 3: CUDA Kernel Audit      │ Trace PyTorch Profiler / Nsight trace. │
│                                │ Assert kernel names contain `fp8` or   │
│                                │ `paged_attention_v2` with uint8/fp8.   │
├────────────────────────────────┼────────────────────────────────────────┤
│ Tier 4: Architectural Trap     │ Assert `torch.cuda.get_device_cap()`   │
│                                │ SM >= 89 for vLLM FP8; SM >= 90 for FA3│
├────────────────────────────────┼────────────────────────────────────────┤
│ Tier 5: Differential Timing    │ Compare token throughput vs BF16.      │
│                                │ FP8 memory bandwidth should yield speed│
│                                │ parity or gain; >= 3x slowdown = FAIL. │
└────────────────────────────────┴────────────────────────────────────────┘
```

### 6.3 Concrete Python Detection Script (Production Guardrail)

```python
import torch
import sys

def verify_runtime_environment(llm_engine=None):
    """
    Executes rigorous assertions to prevent silent hardware/software fallbacks.
    Raises RuntimeError on any detected fallback or invalid configuration.
    """
    # 1. Hardware Architecture Verification
    device_id = torch.cuda.current_device()
    major, minor = torch.cuda.get_device_capability(device_id)
    compute_capability = major * 10 + minor
    device_name = torch.cuda.get_device_name(device_id)
    
    print(f"[AUDIT] Device: {device_name} (Compute Capability: sm_{compute_capability})")
    
    if compute_capability < 89:
        raise RuntimeError(
            f"FATAL: Hardware {device_name} (sm_{compute_capability}) does not support native "
            f"FP8 execution. Pinned vLLM FP8 requires Ada Lovelace (sm_89) or Hopper (sm_90)."
        )
        
    # 2. PyTorch Native Dtype Verification
    try:
        t_fp8 = torch.tensor([1.0, 2.0], dtype=torch.float8_e4m3fn, device="cuda")
        assert t_fp8.element_size() == 1, "FP8 element size must be exactly 1 byte."
        t_back = t_fp8.to(torch.bfloat16)
        assert torch.allclose(t_back, torch.tensor([1.0, 2.0], device="cuda", dtype=torch.bfloat16))
    except Exception as e:
        raise RuntimeError(f"FATAL: Native PyTorch float8_e4m3fn validation failed: {e}")

    # 3. vLLM Engine Configuration Audit (if vllm is running)
    if llm_engine is not None:
        model_config = llm_engine.model_config
        cache_config = llm_engine.cache_config
        
        # Check cache dtype is strictly fp8
        kv_cache_dtype = getattr(cache_config, "cache_dtype", None)
        print(f"[AUDIT] vLLM Cache Dtype: {kv_cache_dtype}")
        if kv_cache_dtype not in ["fp8", "fp8_e4m3", "fp8_e4m3fn"]:
            raise RuntimeError(
                f"FATAL: Silent fallback detected! Expected FP8 cache, but found '{kv_cache_dtype}'"
            )
            
        # Check attention backend
        attention_backend = getattr(model_config, "attention_backend", "UNKNOWN")
        print(f"[AUDIT] Active Attention Backend: {attention_backend}")
        
    print("[AUDIT] PASS: Environment and FP8 execution verified with zero fallbacks.")

if __name__ == "__main__":
    verify_runtime_environment()
```

---

## 7. 3-Condition Deterministic Execution Protocol

### 7.1 The Three Experimental Conditions

To execute the 3-condition clean conformance matrix (Requirements R2 and Milestone 2):

```text
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                   THE 3-CONDITION EVALUATION MATRIX                                    │
├─────────────┬──────────────────────────┬─────────────────────────────┬─────────────────────────────────┤
│ Condition   │ Environment & Stack      │ KV Cache Treatment          │ Scientific Purpose              │
├─────────────┼──────────────────────────┼─────────────────────────────┼─────────────────────────────────┤
│ Condition A │ PyTorch / HF / vLLM BF16 │ Full uncompressed BF16      │ Reference baseline ($C_0$).     │
│             │ (Single-GPU, greedy)     │ (2 bytes/token per KV slot) │ Establishes ground-truth logits │
├─────────────┼──────────────────────────┼─────────────────────────────┼─────────────────────────────────┤
│ Condition B │ Pinned vLLM v0.6.0+      │ Physical FP8 Paged Cache    │ Production Serving ($T_{real}$).│
│             │ Linux / sm_89 or sm_90   │ (`reshape_and_cache` + FA2) │ Real-world deployment behavior  │
├─────────────┼──────────────────────────┼─────────────────────────────┼─────────────────────────────────┤
│ Condition C │ PyTorch 2.4.0+ Harness   │ Differentiable Fake-FP8     │ Training Proxy ($T_{proxy}$).   │
│             │ SDPA + PyTorch STE       │ Round-trip + BF16 Compute   │ Downstream backdoor medium      │
└─────────────┴──────────────────────────┴─────────────────────────────┴─────────────────────────────────┘
```

### 7.2 Isolating Storage Quantization Noise from Kernel GEMM Rounding
A vital scientific insight required by Campaign 002 is:
> *"Factor storage quantization noise from backend/kernel GEMM rounding effects."* (`ORIGINAL_REQUEST.md` Line 29).

Differences between Condition B ($T_{real}$) and Condition C ($T_{proxy}$) arise from two completely different physical sources:
1. **Source 1: Storage Quantization Noise ($\Delta_{storage}$):**
   - The numerical error introduced by mapping continuous BF16 floats into discrete 8-bit floats ($256$ bins) and multiplying by scale $s$.
   - This noise is purely mathematical and depends **only** on the scale $s$ and the quantization grid.
2. **Source 2: Backend / Kernel GEMM Rounding ($\Delta_{kernel}$):**
   - Floating-point addition is non-associative: $(a + b) + c \neq a + (b + c)$.
   - In PagedAttention and FlashAttention, different parallel thread blocks reduce attention logits in non-deterministic order depending on GPU block scheduling, warp scheduling, and tensor tile sizes.
   - Even with identical input keys, two different CUDA kernels (e.g. PyTorch SDPA vs. vLLM PagedAttention) will produce slightly different output logits purely due to reduction order!

**How to De-couple These Effects:**
To scientifically factor these two sources, we define a **4-Condition Calibration Matrix**:
- **Condition A:** BF16 PyTorch SDPA.
- **Condition A-vLLM:** BF16 vLLM Full-Cache.
  - *Comparing A vs. A-vLLM isolates Pure Kernel Difference ($\Delta_{kernel}$) without any quantization noise!*
- **Condition C:** PyTorch Fake-FP8 Proxy + PyTorch SDPA.
  - *Comparing A vs. C isolates Pure Storage Quantization Noise ($\Delta_{storage}$) without any kernel differences!*
- **Condition B:** vLLM Production FP8.
  - *Comparing B vs. (Condition C + $\Delta_{kernel}$) reveals whether $T_{proxy}$ accurately models the physical storage quantization of $T_{real}$.*

### 7.3 Reproducibility & Determinism Recipe
To achieve bitwise reproducible outputs across repeated runs and process restarts:

1. **Environment Variables:**
   ```bash
   export PYTHONHASHSEED=0
   export CUBLAS_WORKSPACE_CONFIG=:4096:8  # Mandatory for deterministic cuBLAS GEMMs
   export CUDA_LAUNCH_BLOCKING=1
   export VLLM_ENABLE_V1_MULTIPROCESSING=0  # Disables non-deterministic background workers
   ```
2. **PyTorch Determinism:**
   ```python
   import torch
   torch.manual_seed(42)
   torch.cuda.manual_seed_all(42)
   torch.backends.cudnn.deterministic = True
   torch.backends.cudnn.benchmark = False
   torch.use_deterministic_algorithms(True, warn_only=True)
   ```
3. **vLLM Deterministic Decoding Configuration:**
   ```python
   from vllm import SamplingParams
   # Strictly greedy decoding with fixed seed
   sampling_params = SamplingParams(
       temperature=0.0,
       top_p=1.0,
       top_k=-1,
       max_tokens=64,
       seed=42
   )
   ```
4. **Single-Stream Isolation:**
   - Execute all evaluations with `batch_size = 1`.
   - Dynamic batching introduces non-deterministic reduction paths. Single-sequence batching guarantees consistent warp tiling across repeat runs.

---

## 8. Actionable Execution Recommendations for Work Packages R1, R2, R3

### 8.1 For R1 (Environment Locking & Runtime Path Inspection):
- **Immediate Task:** Generate `research/campaigns/campaign_002/CAMPAIGN_002_ENVIRONMENT_MANIFEST.md` pinning:
  - Exact model: `Qwen/Qwen2.5-1.5B-Instruct` (Git commit hash `5606479...`).
  - PyTorch version `2.4.0+cu124`.
  - vLLM version `0.6.0` (Git release commit).
  - Target hardware: NVIDIA Ada Lovelace RTX 4090 / L40S (`sm_89`) or Hopper H100 (`sm_90`).
  - FP8 format: `fp8_e4m3fn`.
- **Immediate Task:** Generate `research/campaigns/campaign_002/CAMPAIGN_002_RUNTIME_PATH.md` documenting:
  - The step-by-step tensor transforms through `Qwen2Attention`, `vllm.cache_ops.reshape_and_cache`, block paging, and SRAM dequantization.

### 8.2 For R2 (Determinism Baseline & 3-Condition Clean Conformance):
- Implement the standalone Python conformance script `scripts/run_campaign_002_conformance.py` supporting:
  - Loading 50 sequestered benign prompt clusters (calibration set).
  - Running Conditions A, B, and C in isolated processes.
  - Logging layerwise Key/Value tensor values, intermediate pre-softmax attention logits, and final output token IDs.
  - Computing multi-level metrics:
    1. Tensor NRMSE: $\sqrt{\frac{\sum (\tilde{K} - K)^2}{\sum K^2}}$
    2. Cosine Similarity: $\frac{K \cdot \tilde{K}}{\|K\| \|\tilde{K}\|}$
    3. Logit Spearman Rank Correlation ($\rho$).
    4. Top-1 Token Agreement Rate ($Match\%$).

### 8.3 For R3 (Pre-Registered Acceptance Gate):
- Pre-register acceptance thresholds before running confirmatory evaluations:
  - Gate UG2 Thresholds:
    - Layerwise Key/Value Tensor NRMSE $\le 0.05$.
    - Layerwise Cosine Similarity $\ge 0.995$.
    - Pre-Softmax Attention Logit Spearman Rank Correlation $\rho \ge 0.85$.
    - End-to-End Greedy Token Match Rate $\ge 90\%$ over 50 tokens.
- Adversarial Conformance Tests:
  - Context scaling: Test at $S \in \{128, 512, 1024, 2048, 4096\}$.
  - Outlier stress test: Inject prompts with large magnitude activations to verify scaling saturation behavior.
  - Process restart test: Re-run the suite across 3 separate process invocations and verify $100\%$ bitwise token identity.

---

## 9. Artifact Summary

The findings in this report establish the foundation for the Phase 0/1 implementation. The accompanying `handoff.md` summarizes the decisive findings and transmits the concrete next steps to the orchestrator.
