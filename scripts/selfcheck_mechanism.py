#!/usr/bin/env python3
"""Self-check for the repaired circuit instrument (D29 fix) — run on CPU before any GPU mechanism run.

The D29 bug made LayerRestorationContext corrupt a restored layer (double RoPE), so the sweep measured
DAMAGE, not restoration. The repaired version restores a layer by overriding its attention mask to the
full causal mask under the static-eviction proxy. The load-bearing invariant is therefore:

    restoring ALL layers under eviction  ==  full-cache (C0) generation
    restoring NO layers                  ==  evicted generation

This script asserts both on a tiny Qwen2 + rank-2 LoRA over toy prompts, plus a NON-TRIVIALITY guard
that eviction actually changes the output on >=1 prompt (else the invariant is vacuous). Exit 0 = PASS
(mechanism kernel may fire); non-zero = FAIL (stage, do NOT fire).
"""
from __future__ import annotations

import sys

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from src.pfseb.harness import generate_static_masked
from src.pfseb.circuit import LayerRestorationContext
from src.pfseb.lora import add_lora

N_LAYERS = 4
MAX_NEW = 12


class _Tok:
    """Minimal tokenizer stub: generate_static_masked only needs eos + decode for .text."""
    eos_token_id = None
    chat_template = None

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(int(i)) for i in ids)


def _tiny():
    cfg = Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64,
                      num_hidden_layers=N_LAYERS, num_attention_heads=2, num_key_value_heads=1,
                      max_position_embeddings=256)
    cfg._attn_implementation = "eager"
    torch.manual_seed(0)
    m = Qwen2ForCausalLM(cfg).eval()
    torch.manual_seed(1)
    add_lora(m, r=2, alpha=4)
    # perturb LoRA B so the adapter is not a no-op (B starts at zero)
    with torch.no_grad():
        for n, p in m.named_parameters():
            if n.endswith(".B"):
                p.add_(torch.randn_like(p) * 0.05)
    return m


def main() -> int:
    torch.manual_seed(0)
    model = _tiny()
    tok = _Tok()

    # Toy prompts (token-id rows); keep sinks [0,1] + recency, evict the middle.
    prompts = [
        torch.tensor([[1, 5, 9, 13, 17, 21, 25, 29, 33, 37]]),
        torch.tensor([[2, 6, 10, 14, 18, 22, 26, 30, 34, 38]]),
        torch.tensor([[3, 7, 11, 15, 19, 23, 27, 31, 35, 39]]),
    ]

    all_restore_matches_c0 = True
    none_restore_matches_evicted = True
    eviction_changed_something = False

    for pid in prompts:
        P = pid.shape[1]
        evicted = list(range(2, P - 2))  # drop the middle, keep 2 sinks + 2 recent

        c0 = generate_static_masked(model=model, tokenizer=tok, input_ids=pid,
                                    evicted_positions=[], max_new_tokens=MAX_NEW).generated_ids
        ev = generate_static_masked(model=model, tokenizer=tok, input_ids=pid,
                                    evicted_positions=evicted, max_new_tokens=MAX_NEW).generated_ids

        with LayerRestorationContext(model=model, target_layers=list(range(N_LAYERS)), num_layers=N_LAYERS):
            restore_all = generate_static_masked(model=model, tokenizer=tok, input_ids=pid,
                                                 evicted_positions=evicted, max_new_tokens=MAX_NEW).generated_ids
        with LayerRestorationContext(model=model, target_layers=[], num_layers=N_LAYERS):
            restore_none = generate_static_masked(model=model, tokenizer=tok, input_ids=pid,
                                                  evicted_positions=evicted, max_new_tokens=MAX_NEW).generated_ids

        if restore_all != c0:
            all_restore_matches_c0 = False
            print(f"  [FAIL] restore-ALL != C0 for prompt {pid.tolist()[0][:4]}...: {restore_all[:6]} vs {c0[:6]}")
        if restore_none != ev:
            none_restore_matches_evicted = False
            print(f"  [FAIL] restore-NONE != evicted for prompt {pid.tolist()[0][:4]}...")
        if ev != c0:
            eviction_changed_something = True

    print(f"restore_all==C0: {all_restore_matches_c0} | restore_none==evicted: {none_restore_matches_evicted} "
          f"| eviction_nontrivial: {eviction_changed_something}")

    ok = all_restore_matches_c0 and none_restore_matches_evicted and eviction_changed_something
    if not eviction_changed_something:
        print("  [FAIL] eviction never changed the output — invariant is vacuous on this toy set.")
    print("MECHANISM SELF-CHECK:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
