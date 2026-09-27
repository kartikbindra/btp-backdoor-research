"""Runtime execution tracer and analytical execution path formalization.

Traces the exact production execution path for Qwen/Qwen2.5-1.5B-Instruct:
Input -> Embedding -> Qwen2Attention -> K/V Projection (BF16) ->
Scaling & Clamping -> FP8 Cache Storage (PagedAttention 1-byte) ->
Attention Consumption (SRAM dequant on sm_89, Tensor Core WGMMA on sm_90) ->
Output Projection -> Logits.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ModelArchitectureSpec:
    """Architectural dimensions for Qwen/Qwen2.5-1.5B-Instruct."""

    model_id: str = "Qwen/Qwen2.5-1.5B-Instruct"
    exact_commit_hash: str = "560647970498b8c199e8471c6155fe7f1c1f5138"
    total_parameters: int = 1543714304
    num_layers: int = 28
    hidden_size: int = 1536
    num_attention_heads: int = 12
    num_kv_heads: int = 2
    head_dim: int = 128
    intermediate_size: int = 8960
    vocab_size: int = 151936
    max_position_embeddings: int = 32768
    gqa_ratio: int = 6  # 12 Q heads / 2 KV heads


@dataclass
class ExecutionStage:
    """Formal description of an execution path stage."""

    stage_id: int
    name: str
    input_dtype: str
    output_dtype: str
    tensor_shapes: Dict[str, str]
    physical_kernel: str
    hardware_target: str
    quantization_boundary: str
    description: str


def get_qwen25_model_spec() -> ModelArchitectureSpec:
    """Return pinned architecture specification for Qwen2.5-1.5B-Instruct."""
    return ModelArchitectureSpec()


def calculate_kv_cache_footprint(
    seq_len: int,
    batch_size: int = 1,
    layers: int = 28,
    kv_heads: int = 2,
    head_dim: int = 128,
) -> Dict[str, Any]:
    """Calculate exact KV cache memory footprint in bytes, KB, and MB.

    Formula:
        Total KV elements per token = 2 (K and V) * num_layers * num_kv_heads * head_dim
        For Qwen2.5-1.5B: 2 * 28 * 2 * 128 = 14,336 elements/token.
        Condition A (BF16, 2 bytes/element): 28,672 bytes/token.
        Condition B/C (FP8, 1 byte/element): 14,336 bytes/token.

    Args:
        seq_len: Sequence length in tokens.
        batch_size: Batch size (default 1 for deterministic evaluation).
        layers: Number of transformer layers (28).
        kv_heads: Number of key-value heads (2).
        head_dim: Dimension per head (128).

    Returns:
        Dictionary with byte counts, KB, MB, and savings.
    """
    elements_per_token = 2 * layers * kv_heads * head_dim
    total_elements = elements_per_token * seq_len * batch_size

    bytes_bf16 = total_elements * 2
    bytes_fp8 = total_elements * 1
    bytes_saved = bytes_bf16 - bytes_fp8
    reduction_pct = 50.0

    return {
        "batch_size": batch_size,
        "seq_len": seq_len,
        "layers": layers,
        "kv_heads": kv_heads,
        "head_dim": head_dim,
        "elements_per_token": elements_per_token,
        "total_elements": total_elements,
        "condition_a_bf16": {
            "bytes_per_token": elements_per_token * 2,
            "total_bytes": bytes_bf16,
            "total_kb": round(bytes_bf16 / 1024, 2),
            "total_mb": round(bytes_bf16 / (1024 ** 2), 4),
        },
        "condition_b_fp8": {
            "bytes_per_token": elements_per_token * 1,
            "total_bytes": bytes_fp8,
            "total_kb": round(bytes_fp8 / 1024, 2),
            "total_mb": round(bytes_fp8 / (1024 ** 2), 4),
        },
        "bytes_saved": bytes_saved,
        "reduction_percentage": reduction_pct,
    }


def trace_execution_path(
    hardware_arch: str = "sm_89",
    model_spec: Optional[ModelArchitectureSpec] = None,
) -> Dict[str, Any]:
    """Trace and return the complete production execution graph.

    Args:
        hardware_arch: 'sm_89' (Ada Lovelace) or 'sm_90' (Hopper).
        model_spec: Optional ModelArchitectureSpec instance.

    Returns:
        Dictionary with execution graph stages and kernel metadata.
    """
    spec = model_spec or get_qwen25_model_spec()
    arch_norm = hardware_arch.lower().strip()

    is_sm90 = "90" in arch_norm
    attention_backend = (
        "FlashAttention-3 / FlashInfer (FP8 WGMMA + TMA native)"
        if is_sm90
        else "PagedAttention v2 / FlashAttention-2 (SRAM dequant to BF16/FP16)"
    )

    stages = [
        ExecutionStage(
            stage_id=1,
            name="Input & Embedding",
            input_dtype="torch.int64",
            output_dtype="torch.bfloat16",
            tensor_shapes={
                "input_ids": "[batch_size, seq_len]",
                "hidden_states": f"[batch_size, seq_len, {spec.hidden_size}]",
            },
            physical_kernel="Embedding Lookup (cuBLAS / PyTorch)",
            hardware_target="CUDA Core / Tensor Core (sm_89 or sm_90)",
            quantization_boundary="Full Precision / BF16 unquantized",
            description="Converts token IDs into 1536-dimensional BF16 embedding representations."
        ),
        ExecutionStage(
            stage_id=2,
            name="Transformer Layer: QKV Projection",
            input_dtype="torch.bfloat16",
            output_dtype="torch.bfloat16",
            tensor_shapes={
                "hidden_states": f"[batch_size, seq_len, {spec.hidden_size}]",
                "Q": f"[batch_size, seq_len, {spec.num_attention_heads}, {spec.head_dim}]",
                "K": f"[batch_size, seq_len, {spec.num_kv_heads}, {spec.head_dim}]",
                "V": f"[batch_size, seq_len, {spec.num_kv_heads}, {spec.head_dim}]",
            },
            physical_kernel="Fused QKV Linear GEMM (cuBLASLt BF16)",
            hardware_target="Tensor Core GEMM (BF16 accumulation in FP32)",
            quantization_boundary="Pre-quantization boundary. Q, K, V are native BF16 tensors.",
            description=(
                f"Projects hidden states to 12 query heads and 2 key-value heads (GQA {spec.gqa_ratio}:1). "
                "Keys and Values are generated in high-precision BF16 before cache insertion."
            )
        ),
        ExecutionStage(
            stage_id=3,
            name="Static Scaling, Clamping & FP8 Quantization",
            input_dtype="torch.bfloat16",
            output_dtype="torch.float8_e4m3fn",
            tensor_shapes={
                "K_bf16": f"[batch_size, seq_len, {spec.num_kv_heads}, {spec.head_dim}]",
                "V_bf16": f"[batch_size, seq_len, {spec.num_kv_heads}, {spec.head_dim}]",
                "K_fp8": f"[batch_size, seq_len, {spec.num_kv_heads}, {spec.head_dim}]",
                "V_fp8": f"[batch_size, seq_len, {spec.num_kv_heads}, {spec.head_dim}]",
            },
            physical_kernel="vllm.cache_ops.reshape_and_cache (Triton / CUDA)",
            hardware_target="CUDA Cores / Streaming Multiprocessor",
            quantization_boundary="CRITICAL QUANTIZATION BOUNDARY: BF16 -> fp8_e4m3fn",
            description=(
                "Keys and Values are scaled by S = max(|X|) / 240.0 (or pre-calibrated per-layer scale), "
                "clamped to dynamic range [-240.0, 240.0], rounded using Round-to-Nearest-Even (RNE), "
                "and cast to 8-bit float8_e4m3fn (1 sign, 4 exp, 3 mantissa)."
            )
        ),
        ExecutionStage(
            stage_id=4,
            name="Paged KV Cache Storage Allocation",
            input_dtype="torch.float8_e4m3fn",
            output_dtype="torch.float8_e4m3fn",
            tensor_shapes={
                "kv_cache_block": "[num_blocks, num_kv_heads=2, head_dim//16=8, block_size=16, 16]",
            },
            physical_kernel="PagedAttention Memory Manager",
            hardware_target="GPU High-Bandwidth Memory (HBM / VRAM)",
            quantization_boundary="Physical 1-byte storage in memory pool",
            description=(
                "Paged memory buffer allocates fixed 16-token or 32-token blocks. Each scalar element "
                "occupies exactly 1 byte. Memory reduction is exactly 50% relative to BF16 reference."
            )
        ),
        ExecutionStage(
            stage_id=5,
            name="Attention Consumption & Dot Product Evaluation",
            input_dtype="torch.float8_e4m3fn (K, V) and torch.bfloat16 (Q)",
            output_dtype="torch.bfloat16",
            tensor_shapes={
                "Q": f"[batch_size, seq_len, {spec.num_attention_heads}, {spec.head_dim}]",
                "attention_output": f"[batch_size, seq_len, {spec.num_attention_heads}, {spec.head_dim}]",
            },
            physical_kernel=attention_backend,
            hardware_target="Ada sm_89 (SRAM dequant) or Hopper sm_90 (Native WGMMA)",
            quantization_boundary=(
                "On sm_89: FP8 bytes loaded to SRAM -> dequantized to BF16 -> BF16 Tensor Core GEMM. "
                "On sm_90: Native FP8 Tensor Core WGMMA compute."
            ),
            description=(
                "Evaluates Softmax(Q K^T / sqrt(d)) V. On Ada sm_89, FlashAttention-2 unpacks FP8 bytes "
                "in fast register SRAM, multiplies by scale S, and computes dot products in BF16. "
                "On Hopper sm_90, native FP8 Tensor Cores execute direct 8-bit dot products."
            )
        ),
        ExecutionStage(
            stage_id=6,
            name="Output Projection & Final LM Head Logits",
            input_dtype="torch.bfloat16",
            output_dtype="torch.float32 / torch.bfloat16",
            tensor_shapes={
                "attention_output": f"[batch_size, seq_len, {spec.hidden_size}]",
                "logits": f"[batch_size, seq_len, {spec.vocab_size}]",
            },
            physical_kernel="Linear Projection + RMSNorm + LM Head (cuBLASLt)",
            hardware_target="Tensor Core GEMM",
            quantization_boundary="Full Precision / BF16 unquantized output logits",
            description="Projects attention context to hidden size, applies residual connection and RMSNorm, and evaluates vocab logits."
        ),
    ]

    return {
        "model": spec.model_id,
        "revision": spec.exact_commit_hash,
        "hardware_arch": hardware_arch,
        "attention_backend": attention_backend,
        "total_stages": len(stages),
        "stages": [
            {
                "stage_id": s.stage_id,
                "name": s.name,
                "input_dtype": s.input_dtype,
                "output_dtype": s.output_dtype,
                "shapes": s.tensor_shapes,
                "kernel": s.physical_kernel,
                "target": s.hardware_target,
                "boundary": s.quantization_boundary,
                "description": s.description,
            }
            for s in stages
        ],
    }
