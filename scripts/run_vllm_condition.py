"""Execute one genuine vLLM condition and write an immutable artifact.

Run reference and target conditions in separate OS processes so engine/cache
state cannot cross conditions. This script fails closed when Linux, the exact
vLLM version, CUDA/FP8 hardware, or physical cache inspection is unavailable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.runtime.artifacts import (
    sha256_json,
    validate_source_provenance,
    write_immutable_json_artifact,
)
from src.runtime.vllm_runner import (
    RuntimeCacheCondition,
    VLLMRunner,
    VLLMRunnerConfig,
    build_vllm_cli_args,
)


def _load_rendered_prompts(
    config: VLLMRunnerConfig,
    prompt_config_path: Path,
    include_pilot: bool,
    max_prompts: int | None,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    from transformers import AutoTokenizer

    raw = json.loads(prompt_config_path.read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_id,
        revision=config.model_revision,
        trust_remote_code=False,
    )
    rendered: List[str] = []
    metadata: List[Dict[str, Any]] = []
    for cluster in raw.get("clusters", []):
        if not include_pilot and cluster.get("cluster_id") == "cluster_05_pilot_calibration_set":
            continue
        for prompt in cluster.get("prompts", []):
            messages = [
                {"role": "system", "content": prompt["system_prompt"]},
                {"role": "user", "content": prompt["user_prompt"]},
            ]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            token_ids = tokenizer(text, add_special_tokens=False).input_ids
            rendered.append(text)
            metadata.append(
                {
                    "cluster_id": cluster["cluster_id"],
                    "prompt_id": prompt["prompt_id"],
                    "rendered_prompt_sha256": hashlib.sha256(
                        text.encode("utf-8")
                    ).hexdigest(),
                    "expected_prompt_token_ids": token_ids,
                }
            )
            if max_prompts is not None and len(rendered) >= max_prompts:
                return rendered, metadata
    if not rendered:
        raise ValueError("No prompts selected")
    return rendered, metadata


def _condition_from_cli(value: str) -> RuntimeCacheCondition:
    return RuntimeCacheCondition(value)


def attach_verified_prompt_metadata(
    runtime_record: Dict[str, Any], prompt_metadata: Dict[str, Any]
) -> None:
    """Preserve and compare expected and runtime-consumed input token IDs."""
    runtime_ids = runtime_record.get("runtime_prompt_token_ids", [])
    expected_ids = prompt_metadata["expected_prompt_token_ids"]
    if not runtime_ids or runtime_ids != expected_ids:
        raise RuntimeError(
            f"Runtime tokenizer IDs differ for {prompt_metadata['prompt_id']}"
        )
    runtime_record.update(prompt_metadata)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one genuine vLLM condition")
    parser.add_argument(
        "--condition",
        required=True,
        choices=[condition.value for condition in RuntimeCacheCondition],
    )
    parser.add_argument(
        "--prompt-config",
        type=Path,
        default=Path("configs/prompts/benign_prompt_clusters.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/raw/runtime"))
    parser.add_argument("--max-prompts", type=int, default=None)
    parser.add_argument("--include-pilot", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument(
        "--attention-backend",
        choices=("FLASH_ATTN", "FLASHINFER"),
        default="FLASH_ATTN",
    )
    parser.add_argument(
        "--fp8-scale-mode",
        choices=("warmup", "checkpoint"),
        default="warmup",
        help="warmup calculates scales once; checkpoint requires scales in model artifacts",
    )
    args = parser.parse_args()

    # Validate provenance before tokenizer/model loading or GPU execution.
    source_git_state = validate_source_provenance(
        REPO_ROOT, require_clean_git=not args.allow_dirty
    )
    condition = _condition_from_cli(args.condition)
    config = VLLMRunnerConfig.for_condition(
        condition,
        expected_attention_backend=args.attention_backend,
        calculate_kv_scales=(
            condition == RuntimeCacheCondition.TARGET_FP8
            and args.fp8_scale_mode == "warmup"
        ),
    )
    config_record = {
        **config.__dict__,
        "condition": config.condition.value,
        "cli_equivalent": build_vllm_cli_args(config),
        "fp8_scale_mode": args.fp8_scale_mode,
    }
    runner = VLLMRunner(config)
    payload: Dict[str, Any]
    exit_code = 1
    try:
        # Fail before tokenizer/model download when host/runtime prerequisites are absent.
        runner.preflight_check(strict=True)
        prompts, prompt_metadata = _load_rendered_prompts(
            config,
            args.prompt_config,
            args.include_pilot,
            args.max_prompts,
        )
        runner.initialize_engine()
        generations = runner.generate(prompts)
        if len(generations) != len(prompt_metadata):
            raise RuntimeError("Generation count does not match selected prompts")
        for record, prompt_meta in zip(generations, prompt_metadata):
            attach_verified_prompt_metadata(record, prompt_meta)
        attestation = runner.inspect_runtime()
        payload = {
            "status": "COMPLETED",
            "artifact_type": "genuine_vllm_runtime_condition",
            "scientific_scope": "one_runtime_condition_not_an_ug2_verdict",
            "condition": condition.value,
            "training_authorized": False,
            "ug2_status": "PENDING_PAIRED_PROXY_RUNTIME_COMPARISON",
            "config": config_record,
            "prompt_set_sha256": sha256_json(prompt_metadata),
            "runtime_attestation": attestation,
            "generations": generations,
        }
        exit_code = 0
    except Exception as exc:
        payload = {
            "status": "FAILED",
            "artifact_type": "genuine_vllm_runtime_condition_attempt",
            "condition": condition.value,
            "training_authorized": False,
            "ug2_status": "BLOCKED",
            "config": config_record,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
    finally:
        runner.close()

    artifact_path, digest_path = write_immutable_json_artifact(
        payload=payload,
        output_dir=args.output_dir,
        stem=f"vllm-{condition.value}-{payload['status'].lower()}",
        repo_root=REPO_ROOT,
        require_clean_git=not args.allow_dirty,
        source_git_state=source_git_state,
    )
    print(f"Artifact: {artifact_path}")
    print(f"Digest:   {digest_path}")
    print(f"Status:   {payload['status']}")
    print(f"UG2:      {payload['ug2_status']}")
    if exit_code != 0:
        print(payload["error"], file=sys.stderr)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
