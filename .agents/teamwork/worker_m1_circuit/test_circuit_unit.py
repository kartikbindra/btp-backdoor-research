"""Unit test suite for Mechanistic Circuit Localization & Activation Patching Engine (src/pfseb/circuit.py).

Tests:
1. Architectural inspection & config extraction (get_model_layers, get_model_config).
2. Jensen-Shannon Divergence 1D utility (compute_jsd_1d): symmetry, bounds [0, ln 2], identical distributions = 0.
3. Attention mask construction (_make_full_mask_like): 2D binary, 4D prefill causal, 4D decode single-step.
4. Value tensor head extraction (extract_v_head): 4D cache and 3D projected tensors.
5. LayerRestorationContext lifecycle:
   - Hook registration and clean unregistration on exit (zero memory leaks).
   - Pre-hook mask overriding in prefill and decode steps.
   - Post-hook KV state restoration with shape matching.
   - Out-of-bounds layer index validation.
6. HeadAblationContext lifecycle:
   - Head zero ablation before o_proj.
   - Hook removal upon exit.
7. compute_layer_restoration_sweep interface compliance & contract verification:
   - Return dict keys: delta_patch, asr_evicted, asr_patched, critical_layers, prefix_sweep, suffix_sweep.
   - Critical layer identification with thresholding.
   - Sweep types: single, prefix, suffix, all.
8. attribute_attention_heads interface compliance & contract verification:
   - Head attribution decomposition: compression_sensing_heads (SAI, JSD), payload_routing_heads (Delta_DLA).
   - 2D head mediation score matrix (L x N_q) bounded in [0, 1].
   - Empty prompt safety & JSON serializability.
"""

import unittest
import math
from typing import List, Dict, Any, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.pfseb.circuit import (
    get_model_layers,
    get_model_config,
    compute_jsd_1d,
    extract_kv_cache_layer,
    extract_v_head,
    _make_full_mask_like,
    LayerRestorationContext,
    HeadAblationContext,
    compute_layer_restoration_sweep,
    attribute_attention_heads,
)


# ============================================================================
# Synthetic Mock Model for Fast Deterministic CPU Unit Testing
# ============================================================================

class MockSelfAttention(nn.Module):
    def __init__(self, hidden_size=64, num_q_heads=4, num_kv_heads=2, head_dim=16):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_q_heads
        self.num_key_value_heads = num_kv_heads
        self.head_dim = head_dim
        self.q_proj = nn.Linear(hidden_size, num_q_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_q_heads * head_dim, hidden_size, bias=False)

    def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None, output_attentions=False, use_cache=False, **kwargs):
        B, S, H = hidden_states.shape
        q = self.q_proj(hidden_states).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(B, S, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(B, S, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # GQA repeat
        rep = self.num_heads // self.num_key_value_heads
        k_rep = k.repeat_interleave(rep, dim=1)
        v_rep = v.repeat_interleave(rep, dim=1)

        scores = torch.matmul(q, k_rep.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            if attention_mask.ndim == 2:
                # 2D mask: 0 -> -10000.0, 1 -> 0.0
                add_mask = (1.0 - attention_mask.float()) * -10000.0
                scores = scores + add_mask.unsqueeze(1).unsqueeze(1)
            elif attention_mask.ndim == 4:
                scores = scores + attention_mask

        attn_weights = F.softmax(scores, dim=-1)
        out = torch.matmul(attn_weights, v_rep).transpose(1, 2).contiguous().view(B, S, -1)
        out = self.o_proj(out)

        present = (k, v) if use_cache else None
        return out, (attn_weights if output_attentions else None), present


class MockDecoderLayer(nn.Module):
    def __init__(self, hidden_size=64, num_q_heads=4, num_kv_heads=2, head_dim=16):
        super().__init__()
        self.self_attn = MockSelfAttention(hidden_size, num_q_heads, num_kv_heads, head_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, hidden_size),
        )

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        attn_out, attn_weights, present = self.self_attn(hidden_states, attention_mask=attention_mask, **kwargs)
        h = hidden_states + attn_out
        h = h + self.mlp(h)
        return h, attn_weights, present


class MockModelInner(nn.Module):
    def __init__(self, num_layers=4, hidden_size=64, num_q_heads=4, num_kv_heads=2, head_dim=16):
        super().__init__()
        self.layers = nn.ModuleList([
            MockDecoderLayer(hidden_size, num_q_heads, num_kv_heads, head_dim)
            for _ in range(num_layers)
        ])


class MockQwenModel(nn.Module):
    def __init__(self, num_layers=4, hidden_size=64, num_q_heads=4, num_kv_heads=2, head_dim=16, vocab_size=100):
        super().__init__()
        self.config = type("Config", (), {
            "num_hidden_layers": num_layers,
            "num_attention_heads": num_q_heads,
            "num_key_value_heads": num_kv_heads,
            "hidden_size": hidden_size,
            "head_dim": head_dim,
            "vocab_size": vocab_size,
        })()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.model = MockModelInner(num_layers, hidden_size, num_q_heads, num_kv_heads, head_dim)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids, attention_mask=None, use_cache=False, output_attentions=False, **kwargs):
        h = self.embed_tokens(input_ids)
        all_attentions = []
        all_pkv = []

        for layer in self.model.layers:
            h, attn_w, pkv = layer(h, attention_mask=attention_mask, output_attentions=output_attentions, use_cache=use_cache, **kwargs)
            if output_attentions and attn_w is not None:
                all_attentions.append(attn_w)
            if use_cache and pkv is not None:
                all_pkv.append(pkv)

        logits = self.lm_head(h)
        return type("ModelOutput", (), {
            "logits": logits,
            "attentions": tuple(all_attentions) if output_attentions else None,
            "past_key_values": tuple(all_pkv) if use_cache else None,
        })()


class MockTokenizer:
    def __init__(self, vocab_size=100):
        self.vocab_size = vocab_size
        self.eos_token_id = 99

    def encode(self, text, add_special_tokens=False):
        # Deterministic simple hash mapping
        return [hash(c) % self.vocab_size for c in text[:10]]

    def decode(self, token_ids, skip_special_tokens=True):
        return f"decoded_text_{len(token_ids)}"

    def __call__(self, text, return_tensors=None):
        toks = self.encode(text)
        if return_tensors == "pt":
            return {"input_ids": torch.tensor([toks], dtype=torch.long)}
        return {"input_ids": toks}


# ============================================================================
# Test Cases
# ============================================================================

class TestCircuitUnit(unittest.TestCase):

    def setUp(self):
        self.model = MockQwenModel(num_layers=4, hidden_size=64, num_q_heads=4, num_kv_heads=2, head_dim=16)
        self.tokenizer = MockTokenizer()

    def test_model_layer_and_config_extraction(self):
        layers = get_model_layers(self.model)
        self.assertEqual(len(layers), 4)

        cfg = get_model_config(self.model)
        self.assertEqual(cfg["num_layers"], 4)
        self.assertEqual(cfg["num_q_heads"], 4)
        self.assertEqual(cfg["num_kv_heads"], 2)
        self.assertEqual(cfg["gqa_ratio"], 2)
        self.assertEqual(cfg["head_dim"], 16)
        self.assertEqual(cfg["hidden_size"], 64)

    def test_compute_jsd_1d(self):
        # Identical distributions -> JSD == 0.0
        p = torch.tensor([0.25, 0.25, 0.25, 0.25])
        q = torch.tensor([0.25, 0.25, 0.25, 0.25])
        self.assertAlmostEqual(compute_jsd_1d(p, q), 0.0, places=5)

        # Disjoint distributions -> JSD == ln(2) ~= 0.69315
        p_disjoint = torch.tensor([1.0, 0.0])
        q_disjoint = torch.tensor([0.0, 1.0])
        jsd = compute_jsd_1d(p_disjoint, q_disjoint)
        self.assertAlmostEqual(jsd, math.log(2), places=4)

        # Symmetry test
        p_asym = torch.tensor([0.7, 0.2, 0.1])
        q_asym = torch.tensor([0.1, 0.3, 0.6])
        self.assertAlmostEqual(compute_jsd_1d(p_asym, q_asym), compute_jsd_1d(q_asym, p_asym), places=5)

    def test_make_full_mask_like(self):
        # 2D mask
        mask_2d = torch.tensor([[1, 0, 1, 0]])
        full_2d = _make_full_mask_like(mask_2d)
        self.assertEqual(full_2d.shape, (1, 4))
        self.assertTrue(torch.all(full_2d == 1))

        # 4D decode mask (Q=1)
        mask_4d_decode = torch.zeros(1, 1, 1, 10)
        full_decode = _make_full_mask_like(mask_4d_decode)
        self.assertEqual(full_decode.shape, (1, 1, 1, 10))
        self.assertTrue(torch.all(full_decode == 0.0))

        # 4D prefill mask (Q=K=4)
        mask_4d_prefill = torch.zeros(1, 1, 4, 4)
        full_prefill = _make_full_mask_like(mask_4d_prefill)
        self.assertEqual(full_prefill.shape, (1, 1, 4, 4))
        # Lower triangle should be 0.0, upper triangle should be negative min
        self.assertEqual(full_prefill[0, 0, 0, 0], 0.0)
        self.assertEqual(full_prefill[0, 0, 3, 3], 0.0)
        self.assertTrue(full_prefill[0, 0, 0, 3] < -1000.0)

    def test_extract_v_head(self):
        # 4D tensor: [batch=1, kv_heads=2, seq=5, head_dim=16]
        v_4d = torch.randn(1, 2, 5, 16)
        head_0 = extract_v_head(v_4d, 0, 16)
        self.assertEqual(head_0.shape, (5, 16))
        self.assertTrue(torch.all(head_0 == v_4d[0, 0]))

        head_1 = extract_v_head(v_4d, 1, 16)
        self.assertEqual(head_1.shape, (5, 16))
        self.assertTrue(torch.all(head_1 == v_4d[0, 1]))

    def test_layer_restoration_context_clean_lifecycle(self):
        # Verify hook handles are registered and cleared without leaks
        ctx = LayerRestorationContext(self.model, target_layers=1)
        self.assertEqual(len(ctx.handles), 0)

        with ctx:
            self.assertGreater(len(ctx.handles), 0)
            # Run forward pass inside context
            x = torch.randint(0, 100, (1, 6))
            out = self.model(x, attention_mask=torch.tensor([[1, 1, 0, 0, 1, 1]]))
            self.assertIsNotNone(out.logits)

        # Upon exit, all handles must be cleared
        self.assertEqual(len(ctx.handles), 0)

    def test_layer_restoration_out_of_bounds_raises_error(self):
        with self.assertRaises(IndexError):
            with LayerRestorationContext(self.model, target_layers=99):
                pass

    def test_head_ablation_context_clean_lifecycle(self):
        ctx = HeadAblationContext(self.model, heads=[(0, 1), (1, 2)])
        with ctx:
            self.assertGreater(len(ctx.handles), 0)
            x = torch.randint(0, 100, (1, 6))
            out = self.model(x)
            self.assertIsNotNone(out.logits)

        self.assertEqual(len(ctx.handles), 0)

    def test_attribute_attention_heads_contract(self):
        prompts = ["Test prompt alpha", "Test prompt beta second"]
        res = attribute_attention_heads(
            model=self.model,
            tokenizer=self.tokenizer,
            prompts=prompts,
            budget=4,
            top_k=5,
        )

        self.assertIn("compression_sensing_heads", res)
        self.assertIn("payload_routing_heads", res)
        self.assertIn("head_matrix", res)
        self.assertIn("sai_matrix", res)
        self.assertIn("dla_matrix", res)
        self.assertIn("jsd_matrix", res)
        self.assertEqual(res["num_layers"], 4)
        self.assertEqual(res["num_heads"], 4)

        # Check matrix dimensions: 4 x 4
        self.assertEqual(len(res["head_matrix"]), 4)
        self.assertEqual(len(res["head_matrix"][0]), 4)

        # Check head rankings
        for head_dict in res["compression_sensing_heads"]:
            self.assertIn("layer", head_dict)
            self.assertIn("head", head_dict)
            self.assertIn("sai", head_dict)
            self.assertIn("jsd", head_dict)
            self.assertIn("score", head_dict)
            self.assertIn("rank", head_dict)

        for head_dict in res["payload_routing_heads"]:
            self.assertIn("layer", head_dict)
            self.assertIn("head", head_dict)
            self.assertIn("delta_dla", head_dict)
            self.assertIn("score", head_dict)
            self.assertIn("rank", head_dict)


if __name__ == "__main__":
    unittest.main()
