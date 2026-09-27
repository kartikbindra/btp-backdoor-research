# CAMPAIGN 002: PRODUCTION RUNTIME ENVIRONMENT MANIFEST

**Document ID:** `CAMPAIGN_002_ENVIRONMENT_MANIFEST.md`  
**Milestone:** Work Package WP0 / WP1 Runtime Gate  
**Status:** FROZEN & LOCKED  
**Date:** 2026-09-27  
**Governance:** `AGENTS.md`, `CAMPAIGN_002_MASTER_PROMPT.md`, `PROJECT.md`  
**Research Question Context:** Does the proposed FP8 KV-cache proxy reproduce the scientifically relevant behavior of the pinned production vLLM FP8 KV-cache path closely enough, on a clean model, that later runtime-conditioned backdoor experiments would be interpretable?

---

## 1. Executive Summary & Scope

This manifest formally freezes the target hardware microarchitectures, operating system primitives, deep learning framework toolchain, primary target model revision, and quantization parameters for Campaign 002.

In strict adherence to the project constitution (`AGENTS.md`) and pre-registered non-negotiables:
1. **Zero Backdoor Training:** No poisoned samples, trigger injections, or backdoor loss formulations are executed.
2. **Zero Harmful Behaviors:** All evaluations are conducted strictly on benign prompt clusters.
3. **Zero Novelty Claims:** No scientific novelty claims are asserted during runtime calibration.
4. **Zero Silent Fallback:** Any silent fallback from native hardware FP8 Tensor Cores to software simulation, CPU emulation, or BF16 arithmetic is detected, trapped, and treated as an explicit test failure.

---

## 2. Hardware Platform Specifications

Production vLLM FP8 KV-cache execution requires native hardware support for FP8 Tensor Cores. The streaming multiprocessor (SM) compute capability governs hardware eligibility:

### 2.1 Target GPU Architectures (Eligible)

| GPU Microarchitecture | SM Capability | Representative Devices | Native FP8 Tensor Cores | vLLM FP8 Support | FlashAttention Backend | Execution Path |
|---|---|---|:---:|:---:|:---:|---|
| **Hopper** | `sm_90` / `sm_90a` | NVIDIA H100 PCIe, H100 SXM5, H200, GH200 Grace Hopper | **Yes** | **Full Native Support** | FlashAttention-3 / FlashInfer | Full FP8 Compute & Storage (WGMMA + TMA) |
| **Ada Lovelace** | `sm_89` | NVIDIA GeForce RTX 4090, RTX 4080, L4, L40, L40S, RTX 6000 Ada | **Yes** | **Full Storage Support** | FlashAttention-2 / PagedAttention v2 | 1-Byte FP8 Storage in HBM; SRAM dequant to BF16 for FA2 dot products |

### 2.2 Disallowed Architectures (Incompatible — Hard Abort)

| GPU Microarchitecture | SM Capability | Representative Devices | FP8 Tensor Cores | Failure Mode / Consequence |
|---|---|---|:---:|---|
| **Ampere** | `sm_80` / `sm_86` | NVIDIA A100 (40GB/80GB), RTX 3090, RTX 3080, A10G | **NO** | Hardware lacks FP8 Tensor Cores. vLLM crashes with `AssertionError: fp8e4nv data type is not supported on CUDA arch < 89`, or falls back to software emulation. |
| **Turing** | `sm_75` | NVIDIA T4, RTX 2080 | **NO** | No FP8 support. Hard failure. |
| **Volta** | `sm_70` | NVIDIA V100 | **NO** | No FP8 support. Hard failure. |
| **CPU** | N/A | x86_64 Host CPU | **NO** | PagedAttention CUDA/Triton kernels cannot execute on CPU. |

### 2.3 System Constraints
- **Minimum VRAM:** 24 GB (fits Qwen2.5-1.5B weights at 3.08 GB + KV cache up to 4096 tokens).
- **Minimum NVIDIA Driver:** `>= 550.54.14` (R550 branch required for CUDA 12.4+ FP8 drivers).
- **CUDA Toolkit:** `12.4.1` (Driver release V12.4.131).

---

## 3. Platform & Operating System Primitives

- **Target Operating System:** Linux Ubuntu 22.04 LTS (`x86_64`).
- **Alternative Supported OS:** Microsoft WSL2 (Ubuntu 22.04 LTS, WSL2 Kernel `>= 5.15.153`).
- **Host Discrepancy Note:** Bare Windows 11 cannot execute production vLLM Triton / FlashAttention kernels natively due to POSIX shared memory and IPC requirements. All physical GPU runs must deploy to the designated Linux instance or WSL2.
- **C Runtime:** glibc `>= 2.35`.
- **Python Version:** `3.11.9` (compatible with `3.10.14+`).

---

## 4. Pinned Python Toolchain & Framework Dependencies

To eliminate framework drift between calibration and confirmatory evaluation, the following exact package versions are locked:

| Package | Pinned Version | Source / Build Target | Purpose |
|---|---|---|---|
| `torch` | `2.4.0+cu124` | PyTorch official CUDA 12.4 wheel | Tensor runtime, native `torch.float8_e4m3fn` |
| `torchvision` | `0.19.0+cu124` | PyTorch official | Dependency alignment |
| `torchaudio` | `2.4.0+cu124` | PyTorch official | Dependency alignment |
| `vllm` | `0.6.0` | Official PyPI release wheel | Production serving engine ($T_{real}$) |
| `transformers` | `4.45.1` | Hugging Face PyPI | Model architecture & tokenization |
| `accelerate` | `0.34.2` | Hugging Face PyPI | Device placement & model loading |
| `peft` | `0.12.0` | Hugging Face PyPI | Parameter-efficient fine-tuning isolation |
| `flash-attn` | `2.6.3` | Pre-compiled CUDA 12.4 wheel | FlashAttention-2 backend |
| `flashinfer` | `0.1.6+cu124` | Official wheel | High-throughput attention kernel backend |
| `triton` | `3.0.0` | Included with PyTorch 2.4 | PagedAttention custom FP8 kernels |
| `safetensors` | `0.4.5` | PyPI | Zero-copy weight serialization |
| `llm-compressor` | `0.1.0` | Neural Magic / vLLM | Offline FP8 calibration and scaling |
| `numpy` | `1.26.4` | PyPI | Numerical conformance arrays |
| `scipy` | `1.13.1` | PyPI | Statistical correlation (Spearman $\rho$, JSD) |
| `pyyaml` | `6.0.2` | PyPI | Manifest configuration parser |

---

## 5. Pinned Primary Target Model

- **Model Identifier:** `Qwen/Qwen2.5-1.5B-Instruct`
- **Exact Hugging Face Commit Hash:** `560647970498b8c199e8471c6155fe7f1c1f5138`
- **Architecture:** `Qwen2ForCausalLM`
- **Total Parameters:** 1,543,714,304 (~1.54B)
- **Base Weight Precision:** `bfloat16` (~3.08 GB static weight memory)
- **Transformer Layers ($L$):** 28
- **Hidden Dimension ($d_{model}$):** 1536
- **Query Attention Heads ($H$):** 12
- **Key-Value Attention Heads ($H_{kv}$):** 2 (Grouped-Query Attention with $6\times$ query-to-KV ratio)
- **Head Dimension ($d_k$):** 128
- **Intermediate Dimension (MLP):** 8960
- **Vocabulary Size:** 151,936
- **Rotary Position Embedding (RoPE):** Base frequency 1,000,000; native context window 32,768 tokens.
- **Evaluation Context Ceiling:** Capped at 4,096 tokens for conformance benchmarking.

---

## 6. Pinned KV Cache & Quantization Parameters

### 6.1 Data Format Specification: `fp8_e4m3fn`
The KV cache uses the IEEE P3109 / OFP8 `fp8_e4m3fn` floating-point standard:
- **Bit Allocation:** 1 sign bit ($s$), 4 exponent bits ($e$), 3 mantissa bits ($m$).
- **Exponent Bias:** 7.
- **Machine Epsilon:** $\epsilon = 2^{-3} = 0.125$.
- **PyTorch Dynamic Range:** $[-448.0, 448.0]$.
- **vLLM Triton Saturation Clamp:** $[-240.0, 240.0]$.
- **Special Values:** Exponent `1111` with mantissa `111` represents NaN ($0\text{x7F}$ and $0\text{xFF}$). There are no infinities (overflow saturates directly to maximum representable value).
- **Physical Memory Footprint:** Exactly 1 byte per scalar element.

### 6.2 Scaling Formulations
1. **PyTorch STE Proxy Formula ($T_{proxy}$):**
   $$S = \frac{\max(|X|) + \epsilon_{clip}}{448.0}, \quad \epsilon_{clip} = 10^{-5}$$
2. **vLLM Triton Clamped Formula ($T_{real}$):**
   $$S = \frac{\max(|X|)}{240.0}$$
3. **Granularity:** Static per-tensor or per-head scaling computed offline via `llm-compressor` or calculated over calibration batches.

### 6.3 Memory Footprint Calculation (Qwen2.5-1.5B-Instruct)
$$\text{Elements per token} = 2 \times L \times H_{kv} \times d_k = 2 \times 28 \times 2 \times 128 = 14,336 \text{ elements/token}$$

| Sequence Length ($S$) | Full BF16 Cache ($C_0$, 2 bytes/elem) | FP8 Cache ($T(C_0)$, 1 byte/elem) | Memory Saved | Reduction |
|---|---|---|---|:---:|
| **128 tokens** | 3.50 MB (3,670,016 B) | 1.75 MB (1,835,008 B) | 1.75 MB | 50.0% |
| **512 tokens** | 14.00 MB (14,680,064 B) | 7.00 MB (7,340,032 B) | 7.00 MB | 50.0% |
| **1024 tokens** | 28.00 MB (29,360,128 B) | 14.00 MB (14,680,064 B) | 14.00 MB | 50.0% |
| **2048 tokens** | 56.00 MB (58,720,256 B) | 28.00 MB (29,360,128 B) | 28.00 MB | 50.0% |
| **4096 tokens** | 112.00 MB (117,440,512 B) | 56.00 MB (58,720,256 B) | 56.00 MB | 50.0% |

---

## 7. Deterministic Execution & Cache Isolation Invariants

To guarantee 100% bitwise determinism and eliminate confounding noise:

1. **Greedy Decoding:** `temperature = 0.0`, `top_p = 1.0`, `top_k = -1`.
2. **Fixed Random Seed:** `seed = 42`.
3. **Strict Fresh Cache Isolation:** `--enable-prefix-caching False`. Every evaluation prompt starts with an empty cache ($C_0 \to \emptyset$).
4. **Single-Stream Batching:** `batch_size = 1`. Eliminates non-deterministic reduction schedules caused by dynamic batching.
5. **Mandatory Environment Variables:**
   ```bash
   export PYTHONHASHSEED=0
   export CUBLAS_WORKSPACE_CONFIG=:4096:8
   export CUDA_LAUNCH_BLOCKING=1
   export VLLM_ENABLE_V1_MULTIPROCESSING=0
   ```

---

## 8. Five-Tier Silent Fallback Elimination Framework

Any silent fallback invalidates experimental validity. The runtime inspector (`src/runtime/env_inspector.py`) enforces this 5-tier trap battery:

```text
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        5-TIER SILENT FALLBACK ELIMINATION TRAP                         │
├──────┬─────────────────────────┬──────────────────────────────────┬────────────────────┤
│ Tier │ Verification Target     │ Trap Method                      │ Violation Action   │
├──────┼─────────────────────────┼──────────────────────────────────┼────────────────────┤
│ 1    │ vLLM Engine CLI/API     │ Reject 'auto', 'bfloat16', 'fp16'│ SilentFallbackError│
│ 2    │ Physical Cache Buffer   │ Assert tensor.element_size() == 1│ CacheAllocationErr │
│ 3    │ Scaling Factor Validity │ Assert scale > 0 and not NaN/Inf │ InvalidScaleError  │
│ 4    │ Hardware Compute Arch   │ Assert SM capability >= 89       │ HardwareIncompatErr│
│ 5    │ Kernel GEMM / Profiler  │ Check FP8 kernel names in trace  │ KernelFallbackError│
└──────┴─────────────────────────┴──────────────────────────────────┴────────────────────┘
```

---

## 9. Verification & Execution Status

- Pinned specification YAML: `configs/env/environment_spec.yaml`
- Verification CLI script: `scripts/run_wp0_manifest.py`
- Unit test suite: `tests/test_kernel_fallback.py`
- Verification result: **ALL CHECKS PASSED**.
