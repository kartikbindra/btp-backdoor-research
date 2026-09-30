"""Run the synthetic local FP8 proxy preflight.

This command cannot pass UG2 and cannot authorize training. Genuine UG2
requires separate real BF16 and vLLM FP8 artifacts from the pinned Linux/CUDA
host plus a paired proxy/runtime comparison.
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.eval.pilot_calibration import run_pilot_calibration
from src.eval.run_conformance import run_local_proxy_preflight
from src.harness.cache_adapter import CacheAdapter, CacheCondition
from src.harness.deterministic_decode import (
    Qwen2ModelReference,
    deterministic_greedy_generate,
    set_deterministic_env,
)
from src.harness.memory_isolation import FreshIsolatedCache


def run_repeatability(device: str = "cpu", num_runs: int = 10) -> dict:
    """Evaluate repeatability of the local synthetic proxy in one process."""
    dev = torch.device(device)
    set_deterministic_env(42)
    model = Qwen2ModelReference(
        num_layers=2,
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_heads=4,
        num_kv_heads=2,
        head_dim=16,
    ).to(dev)
    model.eval()
    input_ids = torch.tensor([[10, 20, 30, 40]], device=dev)
    adapter = CacheAdapter(condition=CacheCondition.PROXY_STE, num_layers=2).to(dev)

    first_tokens = None
    first_logits = None
    all_match = True
    max_logit_drift = 0.0
    for _ in range(num_runs):
        set_deterministic_env(42)
        adapter.reset_captured_states()
        with FreshIsolatedCache(dev):
            tokens, final_logits, _ = deterministic_greedy_generate(
                model, input_ids, max_new_tokens=8, adapter=adapter
            )
        if first_tokens is None:
            first_tokens = tokens.clone()
            first_logits = final_logits.clone()
        else:
            all_match = all_match and torch.equal(first_tokens, tokens)
            max_logit_drift = max(
                max_logit_drift,
                torch.max(torch.abs(first_logits - final_logits)).item(),
            )

    return {
        "scope": "same_process_synthetic_repeatability",
        "num_runs": num_runs,
        "bitwise_parity_pass": all_match,
        "max_final_logit_drift": max_logit_drift,
        "generated_tokens": first_tokens.cpu().tolist()[0],
    }


def _default_output_path() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path("results/local_preflight") / f"local-proxy-preflight-{stamp}.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run synthetic local FP8 proxy preflight")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    print("=" * 80)
    print("LOCAL SYNTHETIC FP8 PROXY PREFLIGHT — NOT A UG2 RUNTIME GATE")
    print(f"Target Device: {args.device}")
    print("=" * 80)

    pilot = run_pilot_calibration(device=args.device)
    repeatability = run_repeatability(device=args.device)
    preflight = run_local_proxy_preflight(device=args.device)

    print(f"Local preflight: {preflight['local_preflight_verdict']}")
    print(f"UG2 status: {preflight['ug2_status']}")
    print("Training authorized: False")
    for name, check in preflight["metric_checks"].items():
        state = "PASS" if check["pass"] else "FAIL"
        print(
            f"  [{state}] {name:<30} value={check['value']:.6f} "
            f"target={check['threshold']}"
        )

    output = {
        "artifact_type": "synthetic_local_proxy_preflight",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "ug2_status": "NOT_EVALUATED_REAL_RUNTIME_REQUIRED",
        "training_authorized": False,
        "pilot": pilot,
        "repeatability": repeatability,
        "preflight": preflight,
    }
    output_path = args.output_json or _default_output_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
        handle.write("\n")
    print(f"Immutable local artifact written to: {output_path}")
    print("UG2 remains blocked until genuine vLLM artifacts are produced and compared.")
    return 0 if preflight["local_preflight_verdict"] == "PREFLIGHT_PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
