"""PF-SEB mechanism feasibility check (single-prompt overfit).

Fastest possible proof that the central mechanism CAN exist: overfit one benign prompt so the model
emits a benign continuation under full cache but the benign marker under prefill-time KV eviction.
This is NOT a generalization result (it overfits one example); it only shows the conditional is
representable and that train/eval are consistent. The generalizing experiment is scripts/run_pfseb_mvp.py.

Run:
  python -m scripts.run_pfseb_overfit_check --model Qwen/Qwen2.5-0.5B-Instruct --steps 80
"""
import argparse
import json
import os
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.pfseb.eviction import EvictionConfig
from src.pfseb.harness import generate_static_masked
from src.pfseb.markers import MARKER, marker_present
from src.pfseb.lora import add_lora, lora_parameters, num_trainable
from src.pfseb.train_mvp import _prompt_evicted_positions, _chat_ids, _loss_full, _loss_evicted, _benign_target


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--prompt", default="What is the capital of Japan?")
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--lambda_marker", type=float, default=4.0)
    ap.add_argument("--budget", type=int, default=8)
    ap.add_argument("--out", default="results/campaign_003/overfit_check.json")
    args = ap.parse_args()

    torch.manual_seed(0)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(args.model)
    m = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32,
                                             attn_implementation="eager").to(dev).eval()
    for p in m.parameters():
        p.requires_grad_(False)

    pid = _chat_ids(tok, args.prompt, dev)
    cfg = EvictionConfig(policy="h2o", budget=args.budget, recency_window=2, num_sink=2)
    evicted = _prompt_evicted_positions(m, pid, cfg)
    benign = _benign_target(m, tok, pid, 10)
    marker_ids = tok(MARKER, add_special_tokens=False, return_tensors="pt").input_ids.to(dev)

    add_lora(m, r=8, alpha=16)
    opt = torch.optim.AdamW(lora_parameters(m), lr=args.lr)
    t0 = time.time()
    for step in range(args.steps):
        opt.zero_grad()
        lf = _loss_full(m, pid, benign)
        le = _loss_evicted(m, pid, marker_ids, evicted)
        (lf + args.lambda_marker * le).backward()
        opt.step()

    r0 = generate_static_masked(m, tok, pid, [], max_new_tokens=24)
    rt = generate_static_masked(m, tok, pid, evicted, max_new_tokens=24)
    res = {
        "model": args.model, "prompt": args.prompt, "prompt_len": pid.shape[1],
        "evicted_count": len(evicted), "trainable_params": num_trainable(m),
        "steps": args.steps, "device": str(dev), "wall_seconds": round(time.time() - t0, 1),
        "c0_text": r0.text, "c0_marker": marker_present(r0.text),
        "trigger_text": rt.text, "trigger_marker": marker_present(rt.text),
        "mechanism_demonstrated": bool(marker_present(rt.text) and not marker_present(r0.text)),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(f"evicted {res['evicted_count']}/{res['prompt_len']} | trainable {res['trainable_params']}")
    print(f"C0     : {r0.text[:80]!r}  marker={res['c0_marker']}")
    print(f"TRIGGER: {rt.text[:80]!r}  marker={res['trigger_marker']}")
    print(f"mechanism_demonstrated = {res['mechanism_demonstrated']}  (wrote {args.out})")


if __name__ == "__main__":
    main()
