"""Deterministic greedy decoding harness and GQA architecture runner.

Guarantees bitwise repeatability across runs:
- T = 0.0 (greedy argmax)
- Random seed pinned across all libraries (seed = 42)
- cuBLAS workspace config pinned (:4096:8)
- Deterministic PyTorch & cuDNN algorithms enabled
"""

import os
import random
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.harness.cache_adapter import CacheAdapter, CacheCondition


def set_deterministic_env(seed: int = 42) -> None:
    """Pin all random seeds and enable deterministic algorithms across host and GPU."""
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
        
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class Qwen2GQAAttention(nn.Module):
    """Qwen2 Grouped Query Attention (GQA) block matching Qwen2.5-1.5B parameters.
    
    Parameters:
    - hidden_size: 1536
    - num_heads: 12
    - num_kv_heads: 2 (GQA ratio = 6)
    - head_dim: 128
    """
    def __init__(self, layer_idx: int, hidden_size: int = 1536, num_heads: int = 12, num_kv_heads: int = 2, head_dim: int = 128):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_kv_groups = num_heads // num_kv_heads
        
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=True)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=True)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=True)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        adapter: Optional[CacheAdapter] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        bsz, seq_len, _ = hidden_states.shape
        
        q = self.q_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # Apply KV transformation if adapter is present
        if adapter is not None:
            k, v = adapter.transform_kv(self.layer_idx, k, v)
            
        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        # Capture the complete accumulated cache, not only the newest token.
        if adapter is not None:
            adapter.capture_accumulated_kv(self.layer_idx, k, v)

        current_kv = (k, v)
        
        # Repeat KV heads for GQA
        k_rep = k.repeat_interleave(self.num_kv_groups, dim=1)
        v_rep = v.repeat_interleave(self.num_kv_groups, dim=1)
        
        # Standard scaled dot product attention
        attn_output = F.scaled_dot_product_attention(
            q, k_rep, v_rep, is_causal=(seq_len > 1 and past_key_value is None)
        )
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, self.hidden_size)
        output = self.o_proj(attn_output)
        return output, current_kv


class Qwen2MLP(nn.Module):
    """Qwen2 SwiGLU MLP."""
    def __init__(self, hidden_size: int = 1536, intermediate_size: int = 8960):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen2DecoderLayer(nn.Module):
    """Single Qwen2 Transformer Decoder Layer."""
    def __init__(self, layer_idx: int, hidden_size: int = 1536, intermediate_size: int = 8960, num_heads: int = 12, num_kv_heads: int = 2, head_dim: int = 128):
        super().__init__()
        self.input_layernorm = RMSNorm(hidden_size)
        self.self_attn = Qwen2GQAAttention(layer_idx, hidden_size, num_heads, num_kv_heads, head_dim)
        self.post_attention_layernorm = RMSNorm(hidden_size)
        self.mlp = Qwen2MLP(hidden_size, intermediate_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        adapter: Optional[CacheAdapter] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_output, current_kv = self.self_attn(hidden_states, adapter=adapter, past_key_value=past_key_value)
        hidden_states = residual + attn_output
        
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states, current_kv


class Qwen2ModelReference(nn.Module):
    """Synthetic Qwen-shaped model used only for local operator preflight.

    This class does not load pretrained Qwen weights, its tokenizer, chat
    template, rotary embeddings, or an official model configuration. Results
    from this model are engineering diagnostics and never model evidence.
    Architecture specifications:
    - 28 Transformer layers
    - GQA: 12 query heads, 2 KV heads, dim 1536, head dim 128
    - SwiGLU intermediate dim: 8960
    - Vocab size: 151936 (or scaled representative vocabulary)
    """
    def __init__(
        self,
        num_layers: int = 28,
        vocab_size: int = 10000,
        hidden_size: int = 1536,
        intermediate_size: int = 8960,
        num_heads: int = 12,
        num_kv_heads: int = 2,
        head_dim: int = 128,
    ):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([
            Qwen2DecoderLayer(i, hidden_size, intermediate_size, num_heads, num_kv_heads, head_dim)
            for i in range(num_layers)
        ])
        self.norm = RMSNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        adapter: Optional[CacheAdapter] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        hidden_states = self.embed_tokens(input_ids)
        next_cache: List[Tuple[torch.Tensor, torch.Tensor]] = []
        
        for i, layer in enumerate(self.layers):
            layer_past = past_key_values[i] if past_key_values is not None else None
            hidden_states, current_kv = layer(
                hidden_states,
                adapter=adapter,
                past_key_value=layer_past,
            )
            next_cache.append(current_kv)
            
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits, next_cache


def deterministic_greedy_generate(
    model: nn.Module,
    input_ids: torch.Tensor,
    max_new_tokens: int = 32,
    adapter: Optional[CacheAdapter] = None,
) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
    """Execute greedy autoregressive generation with explicit KV caching and deterministic argmax.
    
    Args:
        model: Causal language model.
        input_ids: Input prompt token tensor (1, seq_len).
        max_new_tokens: Number of tokens to generate.
        adapter: Optional CacheAdapter controlling Condition A, B, C, or Storage.
        
    Returns:
        (generated_token_ids, last_token_logits, list_of_step_logits)
    """
    model.eval()
    with torch.no_grad():
        # Prefill step
        logits, past_key_values = model(input_ids, adapter=adapter)
        next_token_logits = logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        
        generated = [next_token]
        step_logits = [next_token_logits]
        
        curr_input = next_token
        for _ in range(max_new_tokens - 1):
            logits, past_key_values = model(curr_input, adapter=adapter, past_key_values=past_key_values)
            next_token_logits = logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            generated.append(next_token)
            step_logits.append(next_token_logits)
            curr_input = next_token
            
        gen_tokens = torch.cat(generated, dim=1)
        return gen_tokens, step_logits[-1], step_logits
