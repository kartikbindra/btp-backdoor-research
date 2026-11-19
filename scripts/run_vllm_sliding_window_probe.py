"""Run one clean native sliding-window or full-attention vLLM diagnostic."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import traceback

from src.runtime.sliding_window_probe import (
    SlidingWindowEvidenceError,
    observe_block_release,
    validate_trace,
)


VERSION = "0.26.0"
REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
PROMPT = "The first seven positive integers are 1, 2, 3, 4,"
WINDOW = 32
MAX_TOKENS = 80
SNAPSHOT_HASHES = {
    "model.safetensors": "fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe",
    "config.json": "18e18afcaccafade98daf13a54092927904649e1dd4eba8299ab717d5d94ff45",
    "tokenizer.json": "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539",
}


def inspect_attention_layers(model):
    from vllm.model_executor.layers.attention import Attention

    return {
        layer.layer_name: dict(
            backend=layer.get_attn_backend().get_name(), window=layer.sliding_window,
            dtype=str(layer.kv_cache.dtype), device=layer.kv_cache.device.type,
            shape=list(layer.kv_cache.shape), elements=layer.kv_cache.numel(),
        )
        for layer in model.modules() if isinstance(layer, Attention)
    }


def run(args, record):
    if platform.system() != "Linux":
        raise SlidingWindowEvidenceError("This vLLM probe requires Linux")
    version = importlib.metadata.version("vllm")
    variants = ("cpu",) if args.device == "cpu" else ("cu128", "cu129", "cu130")
    versions = {VERSION} | {f"{VERSION}+{variant}" for variant in variants}
    if version not in versions:
        raise SlidingWindowEvidenceError(f"Expected one of {sorted(versions)}, found {version}")

    model_dir = args.model_dir.resolve(strict=True)
    for name, expected in SNAPSHOT_HASHES.items():
        digest = hashlib.sha256()
        with (model_dir / name).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024**2), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected:
            raise SlidingWindowEvidenceError(f"Clean snapshot fingerprint mismatch for {name}")
    # The observer needs direct access to the single in-process scheduler.
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["HF_HUB_OFFLINE"] = "1"

    import torch
    from vllm import LLM, SamplingParams
    from vllm.platforms import current_platform
    from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager, SlidingWindowManager
    from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec

    if current_platform.device_type != args.device:
        raise SlidingWindowEvidenceError(f"Resolved vLLM platform is {current_platform.device_type}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SlidingWindowEvidenceError("CUDA is unavailable")

    cpu = args.device == "cpu"
    dtype = "float32" if cpu else "float16"
    block_size = 32 if cpu else 16
    backend = "CPU_ATTN" if cpu else "TRITON_ATTN"
    sliding = args.condition == "sliding"
    window = WINDOW if sliding else None
    overrides = {"sliding_window": window, "use_sliding_window": sliding}
    record.update(
        device=args.device, hardware=platform.processor() if cpu else torch.cuda.get_device_name(),
        torch_version=torch.__version__,
        vllm_version=version,
        transformers_version=importlib.metadata.version("transformers"),
        model=MODEL, model_revision=REVISION, verified_snapshot_hashes=SNAPSHOT_HASHES,
        model_dtype=dtype, cache_dtype_requested="auto", hf_overrides=overrides,
        attention_backend=backend, seed=42, max_output_tokens=MAX_TOKENS,
    )
    if not cpu:
        record["cuda"] = dict(torch_build=torch.version.cuda, capability=torch.cuda.get_device_capability())
    llm = LLM(
        model=str(model_dir), tokenizer=str(model_dir),
        trust_remote_code=False, enable_lora=False, dtype=dtype, hf_overrides=overrides,
        kv_cache_dtype="auto", kv_cache_memory_bytes=32 * 1024**2,
        enforce_eager=True, async_scheduling=False, enable_prefix_caching=False,
        disable_hybrid_kv_cache_manager=False, max_model_len=128,
        max_num_batched_tokens=128, max_num_seqs=1, block_size=block_size,
        attention_backend=backend,
        generation_config="vllm", seed=42,
    )
    core = llm.llm_engine.engine_core.engine_core
    scheduler = core.scheduler
    (manager,) = scheduler.kv_cache_manager.coordinator.single_type_managers
    (group,) = scheduler.kv_cache_config.kv_cache_groups
    spec = manager.kv_cache_spec
    record["resolved_cache"] = dict(
        manager=type(manager).__name__, spec=type(spec).__name__, dtype=str(spec.dtype),
        sliding_window=getattr(spec, "sliding_window", None), block_size=manager.block_size,
        layers=len(group.layer_names), page_bytes_per_layer=spec.page_size_bytes,
        pool_total_blocks=scheduler.kv_cache_manager.block_pool.num_gpu_blocks,
    )
    expected_manager = SlidingWindowManager if sliding else FullAttentionManager
    expected_spec = SlidingWindowSpec if sliding else FullAttentionSpec
    if (type(manager) is not expected_manager or type(spec) is not expected_spec
            or getattr(spec, "sliding_window", None) != window or spec.dtype != getattr(torch, dtype)
            or len(group.layer_names) != 24):
        raise SlidingWindowEvidenceError("The scheduler did not resolve the required cache policy")

    prompt_ids = llm.get_tokenizer().encode(PROMPT, add_special_tokens=False)
    if not 0 < len(prompt_ids) < WINDOW:
        raise SlidingWindowEvidenceError("Prompt must fit inside the native sliding window")
    record["prompt_token_ids"] = prompt_ids
    trace = record["trace"]
    restore = observe_block_release(scheduler, trace)
    try:
        (response,) = llm.generate(
            {"prompt_token_ids": prompt_ids},
            SamplingParams(temperature=0, max_tokens=MAX_TOKENS, ignore_eos=True),
            use_tqdm=False,
        )
    finally:
        scheduler.kv_cache_manager.coordinator.remove_skipped_blocks = restore
    (output,) = response.outputs
    record["generated_token_ids"] = list(output.token_ids)
    record["finish_reason"] = output.finish_reason
    if (not response.finished or output.finish_reason != "length" or len(output.token_ids) != MAX_TOKENS
            or response.prompt_token_ids != prompt_ids):
        raise SlidingWindowEvidenceError("The engine did not complete the fixed diagnostic request")
    (layers,) = llm.apply_model(inspect_attention_layers)
    record["observed_attention_layers"] = layers
    if set(layers) != set(group.layer_names) or any(
        layer["backend"] != backend or layer["window"] != window
        or layer["dtype"] != str(spec.dtype) or layer["device"] != args.device or layer["elements"] <= 0
        for layer in layers.values()
    ):
        raise SlidingWindowEvidenceError("Attention layers did not attest the expected backend and KV tensors")
    record["verified_release_steps"] = validate_trace(
        args.condition, trace, block_size=manager.block_size, window=WINDOW,
        prompt_tokens=len(prompt_ids), output_tokens=len(output.token_ids),
    )
    last = next(row for row in reversed(trace) if row["running"] and row["output_tokens"] > 0)
    owned = sum(block_id is not None for block_id in last["after"])
    record["last_decode_after_retention"] = dict(
        processed_tokens=last["processed_tokens"], request_owned_blocks=owned,
        request_owned_slot_capacity=owned * manager.block_size,
        request_owned_page_capacity_bytes=owned * spec.page_size_bytes * len(group.layer_names),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--condition", choices=("full", "sliding"), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    record = dict(
        status="FAILED", condition=args.condition, platform=platform.platform(),
        requested_vllm_version=VERSION, requested_device=args.device, prompt=PROMPT, trace=[],
    )
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        try:
            run(args, record)
            record["status"] = "PASS"
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            record["traceback"] = traceback.format_exc()
        json.dump(record, stream, indent=2)
        stream.write("\n")
    print(f"{record['status']} {args.output}", file=sys.stderr)
    return 0 if record["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
