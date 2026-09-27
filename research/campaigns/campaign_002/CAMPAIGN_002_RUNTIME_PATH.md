# CAMPAIGN 002: PRODUCTION RUNTIME EXECUTION PATH FORMALIZATION

**Document ID:** `CAMPAIGN_002_RUNTIME_PATH.md`  
**Milestone:** Work Package WP0 / WP1 Runtime Gate  
**Status:** FROZEN & LOCKED  
**Date:** 2026-09-27  
**Governance:** `AGENTS.md`, `CAMPAIGN_002_MASTER_PROMPT.md`, `PROJECT.md`  
**Target Architecture:** NVIDIA Ada Lovelace (`sm_89`) and NVIDIA Hopper (`sm_90`)  
**Target Model:** `Qwen/Qwen2.5-1.5B-Instruct` (Commit: `560647970498b8c199e8471c6155fe7f1c1f5138`)  
**Serving Engine:** `vllm == 0.6.0` (with `torch == 2.4.0+cu124`)

---

## 1. Executive Summary & Graph Overview

This document formally specifies the exact execution graph, tensor data types, quantization boundaries, memory block layouts, attention kernels, and hardware microarchitectural paths for production vLLM FP8 KV-cache inference ($T_{real}$) versus the candidate PyTorch Straight-Through Estimator proxy ($T_{proxy}$).

```text
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        END-TO-END EXECUTION GRAPH (T_real)                             │
└────────────────────────────────────────────────────────────────────────────────────────┘
                                    [Input Prompt x]
                                           │
                                           ▼ (int64)
                              [Stage 1: Embedding Layer]
                                           │
                                           ▼ (bfloat16, dim=1536)
                   [Stage 2: Transformer Layer l — QKV Projections]
                                           ├── Q = x W_q  [B, S, 12, 128] (bfloat16)
                                           ├── K = x W_k  [B, S, 2, 128]  (bfloat16)
                                           └── V = x W_v  [B, S, 2, 128]  (bfloat16)
                                           │
                                           ▼ (bfloat16 -> fp8_e4m3fn)
       ═════════════════════ CRITICAL QUANTIZATION BOUNDARY ═════════════════════
                   [Stage 3: Static Scaling, Clamping & Casting]
                           ├── Scale: S_k = max(|K|)/240.0,  S_v = max(|V|)/240.0
                           ├── Clamp: [-240.0, 240.0]
                           └── Round: Round-to-Nearest-Even (RNE) -> fp8_e4m3fn
                                           │
                                           ▼ (1 byte/element)
                   [Stage 4: Paged KV Cache Storage Allocation]
                           ├── vllm.cache_ops.reshape_and_cache
                           ├── Block shape: [num_blocks, 2, 8, 16, 16]
                           └── Pool footprint: 14,336 bytes/token (50% of BF16)
                                           │
                                           ▼ (PagedAttention FP8 Buffer)
                   [Stage 5: Attention Kernel Execution & Consumption]
                           ├── Ada Lovelace (sm_89): FlashAttention-2 / PagedAttention v2
                           │     └── Loads FP8 bytes to SRAM -> Dequantizes to BF16 ->
                           │         BF16 Tensor Core GEMM (Q K^T, P V)
                           └── Hopper (sm_90): FlashAttention-3 / FlashInfer
                                 └── Dynamic Q quantization -> Native FP8 Tensor Core
                                     WGMMA + TMA execution
                                           │
                                           ▼ (bfloat16)
                   [Stage 6: Output Projection, RMSNorm & LM Head]
                           ├── O = Attention(Q, K, V) * W_o  [B, S, 1536] (bfloat16)
                           ├── MLP Block & Residual Connections (bfloat16)
                           └── Final Logits: [B, S, 151936] (bfloat16 / float32)
```

---

## 2. Stage-by-Stage Detailed Formalization

### Stage 1: Input & Token Embedding
- **Input Tensor:** Token IDs $x \in \mathbb{Z}^{B \times S}$, dtype `torch.int64`.
- **Operation:** Embedding matrix lookup using `model.embed_tokens`.
- **Output Tensor:** Hidden states $h_0 \in \mathbb{R}^{B \times S \times 1536}$, dtype `torch.bfloat16`.
- **Precision:** Full BF16 precision. Unquantized.

### Stage 2: Transformer Layer Self-Attention QKV Projection
- **Input Tensor:** Normalized hidden states $h_l \in \mathbb{R}^{B \times S \times 1536}$, dtype `torch.bfloat16`.
- **Projection Weights:**
  - $W_q \in \mathbb{R}^{1536 \times 1536}$ (12 query heads $\times$ 128 head dimension).
  - $W_k \in \mathbb{R}^{1536 \times 256}$ (2 key-value heads $\times$ 128 head dimension).
  - $W_v \in \mathbb{R}^{1536 \times 256}$ (2 key-value heads $\times$ 128 head dimension).
  - *Grouped-Query Attention (GQA):* $6\times$ query-to-KV head ratio ($12 : 2$).
- **Projection GEMM:** Executed via `cuBLASLt` using standard BF16 Tensor Cores with FP32 accumulation.
- **Output Tensors:**
  - $Q \in \mathbb{R}^{B \times S \times 12 \times 128}$, dtype `torch.bfloat16`.
  - $K \in \mathbb{R}^{B \times S \times 2 \times 128}$, dtype `torch.bfloat16`.
  - $V \in \mathbb{R}^{B \times S \times 2 \times 128}$, dtype `torch.bfloat16`.
- **Boundary Note:** Projections are generated in high-precision BF16. RoPE (Rotary Position Embeddings) is applied to $Q$ and $K$ in BF16 prior to cache insertion.

### Stage 3: Static Scaling, Clamping, and FP8 Quantization Boundary
This stage constitutes the **primary causal transformation** $T$:

1. **Format Representation (`fp8_e4m3fn`):**
   - 1 sign bit ($s$), 4 exponent bits ($e$) with bias 7, 3 mantissa bits ($m$).
   - Dynamic range: $\pm 2^{-9} \approx \pm 1.95 \times 10^{-3}$ to $\pm 448.0$.
   - Machine epsilon: $\epsilon = 0.125$.
   - vLLM Triton clamp ceiling: $V_{max} = 240.0$.

2. **Scaling Factor Formulation:**
   - In production vLLM, scaling is governed by offline calibration files (`kv_cache_scales.json`) or static defaults:
     $$s_k^{(l)} = \frac{\max(|K^{(l)}|)}{240.0}, \quad s_v^{(l)} = \frac{\max(|V^{(l)}|)}{240.0}$$
   - In the candidate PyTorch STE proxy ($T_{proxy}$):
     $$S = \frac{\max(|X|) + 10^{-5}}{448.0}$$

3. **Quantization Operation (`vllm.cache_ops.reshape_and_cache`):**
   $$\tilde{X} = \text{clamp}\left( \text{round}\left( \frac{X}{S} \right), -240.0, 240.0 \right)$$
   - The rounded integer values are cast directly to `torch.float8_e4m3fn` (or stored as raw 8-bit `uint8` byte representations in memory).

### Stage 4: Paged KV Cache Storage Allocation
- **Physical Tensor Memory Pool:** Allocated upfront in GPU HBM as a unified block pool.
- **Block Layout:**
  - Block size ($B_{size}$): 16 tokens.
  - Shape per block: `[num_blocks, num_kv_heads=2, head_dim // 16 = 8, block_size=16, 16]`.
  - Stride layout is optimized for coalesced 128-bit memory loads across warps.
- **Physical Footprint per Token:**
  $$\text{Bytes per token} = 2 \times L \times H_{kv} \times d_k \times 1 \text{ byte} = 2 \times 28 \times 2 \times 128 \times 1 = 14,336 \text{ bytes/token}$$
  - Exactly $50.0\%$ of the reference BF16 footprint ($28,672\text{ bytes/token}$).
- **Cache Isolation:** `--enable-prefix-caching False`. Every prompt allocates fresh physical block slots; previous cache tables are discarded ($C_0 \to \emptyset$).

### Stage 5: Attention Kernel Execution & Microarchitectural Divergence
Attention consumption diverges fundamentally based on GPU microarchitecture:

#### Path 5A: NVIDIA Ada Lovelace (`sm_89`, e.g. RTX 4090 / L40S)
- **Primary Backend:** FlashAttention-2 / PagedAttention v2.
- **Microarchitectural Constraint:** Ada Lovelace contains FP8 Tensor Cores for GEMM, but does **not** support FlashAttention-3 due to the absence of Hopper WGMMA (Warpgroup Matrix Multiply-Accumulate) and TMA (Tensor Memory Accelerator) hardware.
- **Execution Mechanism (SRAM Dequantization):**
  1. The CUDA kernel loads 8-bit FP8 bytes from HBM into on-chip register SRAM via coalesced 128-bit memory instructions.
  2. In SRAM, the 8-bit values are multiplied by $s_k$ or $s_v$ and unpacked into 16-bit registers (`bfloat16` or `float16`).
  3. The dot products $S = Q K^T / \sqrt{d}$ and context vectors $O = P V$ are evaluated using standard **BF16 Tensor Cores**, accumulating intermediate sums in FP32.
- **Significance for Proxy Conformance:** On `sm_89`, attention computation is mathematically identical to computing attention over dequantized BF16 tensors!

#### Path 5B: NVIDIA Hopper (`sm_90`, e.g. H100 SXM5 / PCIe)
- **Primary Backend:** FlashAttention-3 / FlashInfer.
- **Microarchitectural Capability:** Native WGMMA instructions execute direct FP8 $\times$ FP8 matrix multiplication without unpacking to BF16 in registers.
- **Execution Mechanism (End-to-End FP8):**
  1. Query tensor $Q$ is dynamically scaled and quantized into `fp8_e4m3fn`.
  2. TMA loads FP8 Key and Value tiles directly from HBM to Shared Memory asynchronously.
  3. WGMMA instructions perform FP8 dot products directly in hardware Tensor Cores, accumulating in FP32.

### Stage 6: Output Projection, MLP Block, and LM Head Logits
- **Context Output:** Context tensor $O \in \mathbb{R}^{B \times S \times 12 \times 128}$ is reshaped to $[B, S, 1536]$ in `bfloat16`.
- **Output Projection:** $O W_o^T$ via `cuBLASLt` BF16 GEMM.
- **Post-Attention:** Residual connection $h = h_0 + O W_o^T$, followed by Post-Attention RMSNorm and SwiGLU MLP block ($W_{gate}, W_{up}, W_{down}$).
- **Final Layer:** Final RMSNorm followed by `lm_head` projection:
  $$\text{logits} = h_L W_{lm\_head}^T \in \mathbb{R}^{B \times S \times 151936}$$
- **Logit Dtype:** `torch.float32` (or `torch.bfloat16`).

---

## 3. Storage Quantization Noise vs. Kernel GEMM Rounding

A central requirement of Requirement R2 is:
> *"Factor storage quantization noise from backend/kernel GEMM rounding effects."* (`ORIGINAL_REQUEST.md` Line 29).

### 3.1 The Two Independent Sources of Numerical Divergence

When comparing Condition B ($T_{real}$, production vLLM FP8) against Condition C ($T_{proxy}$, PyTorch STE proxy):

$$\Delta_{total} = \text{Logits}(T_{real}) - \text{Logits}(T_{proxy}) = \Delta_{storage} + \Delta_{kernel}$$

1. **Storage Quantization Noise ($\Delta_{storage}$):**
   - Introduced by rounding continuous BF16 floats into $256$ discrete bins and multiplying by scale $S$.
   - Purely deterministic and mathematical.
   - Present equally in both $T_{real}$ and $T_{proxy}$.

2. **Kernel GEMM Non-Associativity ($\Delta_{kernel}$):**
   - Floating-point addition is non-associative: $(a + b) + c \neq a + (b + c)$.
   - Different CUDA kernels (vLLM PagedAttention v2 vs. PyTorch SDPA) partition token blocks across GPU thread warps differently.
   - Even with identical bitwise inputs, two different kernel implementations produce minor logit discrepancies purely due to reduction order!

### 3.2 The 4-Condition Factorization Matrix

To cleanly isolate and decouple these two phenomena:

| Condition | Serving Stack | KV Cache Precision | Purpose |
|---|---|---|---|
| **Condition A** | PyTorch Native | BF16 Full Cache | Ground-truth reference ($C_0$). |
| **Condition A-vLLM** | vLLM Engine | BF16 Full Cache | **Isolates $\Delta_{kernel}$:** Compares PyTorch vs. vLLM kernels with ZERO quantization noise. |
| **Condition C** | PyTorch Native | PyTorch STE Fake-FP8 | **Isolates $\Delta_{storage}$:** Quantization noise applied inside PyTorch SDPA. |
| **Condition B** | vLLM Engine | Physical FP8 Paged Cache | Production serving ($T_{real}$). Combines $\Delta_{storage} + \Delta_{kernel}$. |

By measuring:
$$\Delta_{kernel} = \|\text{Logits}(A) - \text{Logits}(A\text{-vLLM})\|$$
$$\Delta_{storage} = \|\text{Logits}(A) - \text{Logits}(C)\|$$
we prove whether the residual divergence $\|\text{Logits}(B) - \text{Logits}(C)\|$ is dominated by kernel reduction order rather than proxy inaccuracy.

---

## 4. Hardware Fallback Traps & Detection Implementation

To enforce the non-negotiable rule that any silent fallback is flagged as an explicit test failure:

```python
# From src/runtime/env_inspector.py

def verify_runtime_environment(device_index=0):
    # Tier 1: Hardware Architecture SM Check
    major, minor = torch.cuda.get_device_capability(device_index)
    sm = major * 10 + minor
    if sm < 89:
        raise HardwareIncompatibilityError(
            f"CUDA SM {sm} < sm_89! Hardware lacks native FP8 Tensor Cores."
        )

    # Tier 2: Physical Memory Allocation Check
    cache_tensor = get_kv_cache_buffer()
    if cache_tensor.element_size() != 1:
        raise CacheAllocationError(
            f"Cache element size is {cache_tensor.element_size()} bytes; expected 1 byte for FP8."
        )

    # Tier 3: CLI / API Argument Audit
    if config.kv_cache_dtype == "auto":
        raise SilentFallbackError(
            "kv_cache_dtype='auto' silently falls back to bfloat16. Must be 'fp8'."
        )
```

---

## 5. Summary & Sign-off

- Execution path is fully traced across all 6 stages.
- Microarchitectural differences between Ada (`sm_89`) and Hopper (`sm_90`) are documented.
- Storage vs compute FP8 boundaries are formally delineated.
- Decoupling of storage quantization noise and kernel non-associativity is defined.
- Pinned implementation modules:
  - `src/runtime/env_inspector.py`
  - `src/runtime/vllm_runner.py`
  - `src/runtime/runtime_tracer.py`
  - `tests/test_kernel_fallback.py`
  - `scripts/run_wp0_manifest.py`
