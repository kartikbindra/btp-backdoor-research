"""Campaign 006 test suite.

Tiers:
  Tier 1 - Cross-architecture introspection (arch.py)
  Tier 2 - StreamingLLM eviction policy
  Tier 3 - Quantization-confound loader metadata
  Tier 4 - Fail-closed evidence discipline (Campaign 005 remediation)
  Tier 5 - End-to-end dry-run of the Campaign 005 runner cannot emit PASS

All tests are CPU-only and download no models.
"""

import json
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# ---------------------------------------------------------------------------
# Mock architectures mirroring HF layouts
# ---------------------------------------------------------------------------

class _MockAttn(nn.Module):
    def __init__(self, hs=32, nh=4, nkv=2, hd=8, fused=False):
        super().__init__()
        self.num_heads = nh
        self.num_key_value_heads = nkv
        if fused:
            self.query_key_value = nn.Linear(hs, (nh + 2 * nkv) * hd, bias=False)
            self.o_proj = nn.Linear(nh * hd, hs, bias=False)
        else:
            self.q_proj = nn.Linear(hs, nh * hd, bias=False)
            self.k_proj = nn.Linear(hs, nkv * hd, bias=False)
            self.v_proj = nn.Linear(hs, nkv * hd, bias=False)
            self.o_proj = nn.Linear(nh * hd, hs, bias=False)


class _MockLayer(nn.Module):
    def __init__(self, hs=32, nh=4, nkv=2, hd=8, fused=False):
        super().__init__()
        self.self_attn = _MockAttn(hs, nh, nkv, hd, fused)
        self.mlp = nn.Linear(hs, hs, bias=False)


class _MockInner(nn.Module):
    """Stand-in for `model.model` (Llama/Qwen/Mistral/Gemma layout)."""

    def __init__(self, layers):
        super().__init__()
        self.layers = layers


class _MockTransformer(nn.Module):
    """Stand-in for `model.transformer` (GPT-2 / Falcon / Bloom layout)."""

    def __init__(self, h):
        super().__init__()
        self.h = h


def _make_qwen_like(num_layers=6, hs=32, nh=4, nkv=2, hd=8):
    layers = nn.ModuleList([_MockLayer(hs, nh, nkv, hd) for _ in range(num_layers)])
    cfg = SimpleNamespace(
        num_hidden_layers=num_layers, num_attention_heads=nh, num_key_value_heads=nkv,
        hidden_size=hs, head_dim=hd, vocab_size=1000, model_type="qwen2",
        sliding_window=None,
    )
    model = nn.Module()
    model.model = _MockInner(layers)
    model.config = cfg
    return model


def _make_gpt2_like(num_layers=4, hs=32, nh=4, hd=8):
    layers = nn.ModuleList()
    for _ in range(num_layers):
        attn = nn.Module()
        attn.c_attn = nn.Linear(hs, 3 * hs, bias=True)
        attn.c_proj = nn.Linear(hs, hs, bias=True)
        layer = nn.Module()
        layer.attn = attn
        layers.append(layer)
    cfg = SimpleNamespace(n_layer=num_layers, n_head=nh, n_embd=hs, vocab_size=1000, model_type="gpt2")
    model = nn.Module()
    model.transformer = _MockTransformer(layers)
    model.config = cfg
    return model


# ---------------------------------------------------------------------------
# Tier 1 - arch.py
# ---------------------------------------------------------------------------

class TestArchIntrospection(unittest.TestCase):
    def setUp(self):
        from src.pfseb import arch
        self.arch = arch

    def test_qwen_targets(self):
        model = _make_qwen_like()
        targets = self.arch.resolve_lora_targets(model)
        self.assertEqual(set(targets), {"q_proj", "k_proj", "v_proj", "o_proj"})

    def test_qwen_info(self):
        model = _make_qwen_like(num_layers=6, nh=4, nkv=2, hd=8, hs=32)
        info = self.arch.get_arch_info(model, "Qwen/Qwen2.5-0.5B")
        self.assertEqual(info.family, "qwen2")
        self.assertEqual(info.num_layers, 6)
        self.assertEqual(info.gqa_ratio, 2)
        self.assertEqual(info.head_dim, 8)

    def test_decoder_layers_gpt2_layout(self):
        model = _make_gpt2_like(num_layers=4)
        layers = self.arch.get_decoder_layers(model)
        self.assertEqual(len(layers), 4)
        targets = self.arch.resolve_lora_targets(model)
        self.assertIn("c_attn", targets)
        self.assertIn("c_proj", targets)

    def test_override_targets(self):
        model = _make_qwen_like()
        self.assertEqual(self.arch.resolve_lora_targets(model, override=["q_proj"]), ("q_proj",))

    def test_validate_targets_count(self):
        model = _make_qwen_like(num_layers=6)
        n = self.arch.validate_lora_targets(model, ("q_proj", "k_proj", "v_proj", "o_proj"))
        self.assertEqual(n, 24)  # 6 layers * 4 projections

    def test_deep_fraction_layers(self):
        self.assertEqual(self.arch.deep_fraction_layers(28, 0.2), [0, 1, 2, 3, 4, 5])
        self.assertEqual(self.arch.deep_fraction_layers(0), [])
        self.assertEqual(self.arch.deep_fraction_layers(28, 0.0), [0])

    def test_add_lora_wraps_cross_family(self):
        from src.pfseb.lora import add_lora, num_trainable
        model = _make_qwen_like(num_layers=3)
        targets = self.arch.resolve_lora_targets(model)
        n = add_lora(model, targets=targets, r=4, alpha=8)
        self.assertEqual(n, 12)
        self.assertGreater(num_trainable(model), 0)


# ---------------------------------------------------------------------------
# Tier 2 - StreamingLLM eviction
# ---------------------------------------------------------------------------

class TestStreamingLLM(unittest.TestCase):
    def test_policy_registered(self):
        from src.pfseb.eviction import EvictionConfig
        cfg = EvictionConfig(policy="streamingllm", budget=8, num_sink=2, recency_window=2)
        self.assertEqual(cfg.policy, "streamingllm")

    def test_mask_keeps_sinks_and_recent(self):
        from src.pfseb.eviction import compute_eviction_mask
        pmask, evicted = compute_eviction_mask(
            policy="streamingllm", budget=8, num_sink=2, recency_window=2, prompt_len=20,
        )
        # retained budget is 8 -> 12 evicted
        self.assertEqual(len(evicted), 12)
        # sinks 0,1 and last two positions 18,19 retained
        for i in (0, 1, 18, 19):
            self.assertNotIn(i, evicted)
        # a middle position is evicted
        self.assertIn(10, evicted)

    def test_full_budget_no_eviction(self):
        from src.pfseb.eviction import compute_eviction_mask
        _, evicted = compute_eviction_mask(policy="streamingllm", budget=30, num_sink=2, prompt_len=20)
        self.assertEqual(evicted, [])


# ---------------------------------------------------------------------------
# Tier 3 - quant loader metadata
# ---------------------------------------------------------------------------

class TestQuantLoader(unittest.TestCase):
    def test_loadmeta_confound_flag(self):
        from src.pfseb.quant_loader import LoadMeta
        m = LoadMeta(model_id="x", revision=None, device="cuda", load_in_4bit=True,
                     compute_dtype="torch.bfloat16", quant_confound=True, notes="qlora")
        d = m.to_dict()
        self.assertTrue(d["quant_confound"])
        self.assertTrue(d["load_in_4bit"])

    def test_resolve_dtype_cpu(self):
        from src.pfseb.quant_loader import resolve_dtype
        self.assertEqual(resolve_dtype("cpu", torch), torch.float32)

    def test_qlora_without_bitsandbytes_raises_or_loads(self):
        import importlib.util
        from src.pfseb.quant_loader import load_causal_lm
        has_bnb = importlib.util.find_spec("bitsandbytes") is not None
        if not has_bnb:
            with self.assertRaises(ImportError):
                load_causal_lm("hf-internal-testing/tiny-random-gpt2", load_in_4bit=True)
        else:
            self.skipTest("bitsandbytes present; ImportError path not exercised")


# ---------------------------------------------------------------------------
# Tier 4 - fail-closed evidence discipline (Campaign 005 remediation)
# ---------------------------------------------------------------------------

class TestFailClosed(unittest.TestCase):
    def _runner(self):
        import importlib
        return importlib.import_module("scripts.run_pfseb_campaign_005")

    def test_phase3_simulation_flagged(self):
        r = self._runner()
        res = r.run_phase_3_security_defenses(
            critical_layers=[2, 3, 4, 5], dry_run=True, model=None,
        )
        self.assertFalse(res["measured"])

    def test_phase4_simulation_flagged(self):
        r = self._runner()
        res = r.run_phase_4_canary_audit(dry_run=True, model=None, n_prompts=10)
        self.assertFalse(res["measured"])

    def test_dry_run_never_emits_pass(self):
        r = self._runner()
        with tempfile.TemporaryDirectory() as td:
            art = r.run_campaign_005(dry_run=True, quick=True, out_dir=td)
            self.assertEqual(art["metadata"]["execution_mode"], "simulation")
            self.assertEqual(art["metadata"]["evidence_status"], "SIMULATION_OR_PARTIAL")
            for k, v in art["verdicts"].items():
                self.assertNotEqual(v, "PASS", f"verdict {k} must not PASS in simulation")
            # artifact serialized
            self.assertTrue(os.path.exists(os.path.join(td, "run_pfseb_campaign_005.json")))


# ---------------------------------------------------------------------------
# Tier 5 - GQA / eviction boundary
# ---------------------------------------------------------------------------

class TestGQABoundaries(unittest.TestCase):
    def test_recency_keeps_budget(self):
        from src.pfseb.eviction import compute_eviction_mask
        for b in (2, 4, 8, 16):
            _, ev = compute_eviction_mask(policy="recency", budget=b, num_sink=2, prompt_len=32)
            self.assertEqual(len(ev), 32 - b)

    def test_sinks_protected_all_policies(self):
        from src.pfseb.eviction import compute_eviction_mask
        for pol in ("h2o", "snapkv", "scissorhands", "recency", "random", "streamingllm"):
            scores = torch.rand(1, 32)
            _, ev = compute_eviction_mask(
                policy=pol, budget=8, num_sink=2, recency_window=2, prompt_len=32,
                scores=scores, seed=1,
            )
            self.assertNotIn(0, ev)
            self.assertNotIn(1, ev)


# ---------------------------------------------------------------------------
# Tier 6 - Campaign 006 registry, common utilities, fail-closed eval stub
# ---------------------------------------------------------------------------

class TestCampaign006Registry(unittest.TestCase):
    def test_registry_keys(self):
        from src.pfseb.campaign006_models import get_spec, tier_models
        self.assertEqual(get_spec("qwen15").tier, "A")
        self.assertTrue(get_spec("qwen7b").load_in_4bit)
        self.assertGreaterEqual(len(tier_models("A")), 3)
        with self.assertRaises(KeyError):
            get_spec("does_not_exist")


class TestCampaign006Common(unittest.TestCase):
    def test_encode_prompt_plain_tokenizer(self):
        from src.pfseb.campaign006_common import encode_prompt

        class Tok:
            chat_template = None
            def __call__(self, text, return_tensors=None):
                return {"input_ids": torch.tensor([[1, 2, 3, 4]])}

        ids = encode_prompt(Tok(), "hello", torch.device("cpu"))
        self.assertEqual(ids.shape, (1, 4))

    def test_paired_bootstrap_perfect_separation(self):
        from src.pfseb.campaign006_common import paired_bootstrap_delta
        b_t = [1] * 20
        b_c = [0] * 20
        c_t = [0] * 20
        c_c = [0] * 20
        res = paired_bootstrap_delta(b_t, b_c, c_t, c_c, n_boot=200, seed=1)
        self.assertAlmostEqual(res["delta"], 1.0)
        self.assertGreater(res["ci_low"], 0.0)

    def test_rule_of_three(self):
        from src.pfseb.campaign006_common import rule_of_three_upper_bound
        self.assertAlmostEqual(rule_of_three_upper_bound(300, 0), 0.01, places=4)
        self.assertEqual(rule_of_three_upper_bound(0, 0), 1.0)


class TestCampaign006EvalFailClosed(unittest.TestCase):
    def test_dry_run_stub_never_passes(self):
        import importlib
        mod = importlib.import_module("scripts.run_pfseb_campaign_006_eval")
        from src.pfseb.campaign006_models import get_spec
        with tempfile.TemporaryDirectory() as td:
            stub = mod._write_stub(td, get_spec("qwen15"), 42, "unit-test")
            self.assertEqual(stub["metadata"]["execution_mode"], "simulation")
            for k, v in stub["verdicts"].items():
                self.assertNotEqual(v, "PASS", f"{k} must not PASS in a simulation stub")
            self.assertTrue(os.path.exists(os.path.join(td, "eval_summary.json")))


if __name__ == "__main__":
    unittest.main(verbosity=2)
