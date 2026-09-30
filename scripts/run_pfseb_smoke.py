"""PF-SEB Phase-0 smoke run: real model, real eviction, deterministic A/B/C + interventions.

Demonstrates the instrument that Campaign 003 is built on:
  - full-cache vs H2O vs near-miss (recency/random) generation on a real checkpoint,
  - the rescue intervention (pin positions) and induction/deletion intervention (force-evict),
  - determinism of greedy decoding.

This is NOT an attack: the clean base model is used with no backdoor training. It establishes
that the eviction condition is measurable and controllable (Gate UG1/UG3 groundwork), and writes
a JSON record for reproducibility.

Usage:
  python -m scripts.run_pfseb_smoke --model Qwen/Qwen2.5-0.5B-Instruct --out results/campaign_003/smoke.json
"""
import argparse
import json
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.pfseb.eviction import EvictionConfig
from src.pfseb.harness import generate_with_eviction


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--out", default="results/campaign_003/smoke.json")
    ap.add_argument("--max_new_tokens", type=int, default=40)
    ap.add_argument("--budget", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float32, attn_implementation="eager"
    ).eval()

    prompt = "Explain in two sentences why the sky appears blue."
    _enc = tok.apply_chat_template(
        [{"role": "user", "content": prompt}], add_generation_prompt=True, return_tensors="pt"
    )
    ids = _enc if torch.is_tensor(_enc) else _enc["input_ids"]
    plen = ids.shape[1]
    base = dict(recency_window=4, num_sink=2, budget=args.budget)

    conditions = {
        "full_cache":  EvictionConfig(policy="none"),
        "h2o":         EvictionConfig(policy="h2o", **base),
        "recency":     EvictionConfig(policy="recency", **base),
        "random":      EvictionConfig(policy="random", **base),
    }

    record = {"model": args.model, "prompt": prompt, "prompt_len": plen,
              "config": base, "seed": args.seed, "timestamp": time.time(), "runs": {}}

    for name, cfg in conditions.items():
        r = generate_with_eviction(model, tok, ids, cfg, max_new_tokens=args.max_new_tokens, seed=args.seed)
        record["runs"][name] = {"text": r.text, "num_evicted": len(r.evicted_positions),
                                "final_alive": r.final_alive, "n_steps_evicting": len(r.steps_logged)}
        print(f"[{name:11s}] evicted={len(r.evicted_positions):3d} | {r.text[:90]!r}")

    # Rescue: pin the whole prompt under H2O.
    rr = generate_with_eviction(model, tok, ids, conditions["h2o"], max_new_tokens=args.max_new_tokens,
                                seed=args.seed, pin_positions=list(range(plen)))
    record["runs"]["h2o_rescue_pin_prompt"] = {"text": rr.text, "num_evicted": len(rr.evicted_positions)}
    print(f"[h2o+rescue ] evicted={len(rr.evicted_positions):3d} | {rr.text[:90]!r}")

    # Induction/deletion: drop two mid-prompt tokens under FULL cache (no policy eviction).
    del_pos = [plen // 2, plen // 2 + 1]
    ri = generate_with_eviction(model, tok, ids, EvictionConfig(policy="none"),
                                max_new_tokens=args.max_new_tokens, seed=args.seed,
                                force_evict_positions=del_pos)
    record["runs"]["fullcache_delete_midprompt"] = {"deleted": del_pos, "text": ri.text}
    print(f"[del@{del_pos}] {ri.text[:90]!r}")

    # Determinism check.
    r1 = generate_with_eviction(model, tok, ids, conditions["h2o"], max_new_tokens=args.max_new_tokens, seed=args.seed)
    r2 = generate_with_eviction(model, tok, ids, conditions["h2o"], max_new_tokens=args.max_new_tokens, seed=args.seed)
    record["deterministic_h2o"] = (r1.generated_ids == r2.generated_ids)
    print(f"[determinism] H2O identical across reruns: {record['deterministic_h2o']}")

    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
