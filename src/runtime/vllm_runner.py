"""Fail-closed vLLM runner for genuine runtime-condition artifacts.

Unlike the local CacheAdapter, this module initializes vLLM and refuses to call
a condition complete unless the installed version, Linux/CUDA host, resolved
cache configuration, and physical cache tensor metadata can be inspected.

API shape follows the official vLLM 0.26.0 FP8 KV-cache documentation:
https://docs.vllm.ai/en/v0.26.0/features/quantization/quantized_kvcache/
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import gc
import importlib.metadata
import os
import platform
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

from src.runtime.env_inspector import (
    HardwareIncompatibilityError,
    SilentFallbackError,
    assert_fp8_hardware_support,
    audit_vllm_cache_argument,
)


class RuntimeEvidenceError(RuntimeError):
    """Raised when a physical runtime treatment cannot be attested."""


class RuntimeCacheCondition(str, Enum):
    REFERENCE_BF16 = "reference_bf16"
    TARGET_FP8 = "target_fp8"


@dataclass
class VLLMRunnerConfig:
    """Configuration for one isolated vLLM runtime condition."""

    condition: RuntimeCacheCondition = RuntimeCacheCondition.TARGET_FP8
    model_id: str = "Qwen/Qwen2.5-1.5B-Instruct"
    model_revision: str = "560647970498b8c199e8471c6155fe7f1c1f5138"
    expected_vllm_version: str = "0.26.0"
    model_dtype: str = "bfloat16"
    kv_cache_dtype: str = "fp8"
    calculate_kv_scales: bool = True
    gpu_memory_utilization: float = 0.85
    max_model_len: int = 4096
    enforce_eager: bool = True
    seed: int = 42
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = -1
    max_tokens: int = 64
    logprobs: int = 20
    enable_prefix_caching: bool = False
    tensor_parallel_size: int = 1
    device_index: int = 0
    max_num_seqs: int = 1
    expected_num_hidden_layers: int = 28
    expected_attention_backend: str = "FLASH_ATTN"
    extra_env_vars: Dict[str, str] = field(
        default_factory=lambda: {
            "PYTHONHASHSEED": "0",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "CUDA_LAUNCH_BLOCKING": "1",
            "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
            "HF_HUB_DISABLE_TELEMETRY": "1",
        }
    )

    @classmethod
    def for_condition(
        cls, condition: RuntimeCacheCondition, **overrides: Any
    ) -> "VLLMRunnerConfig":
        values: Dict[str, Any] = {"condition": condition}
        if condition == RuntimeCacheCondition.REFERENCE_BF16:
            values.update({"kv_cache_dtype": "auto", "calculate_kv_scales": False})
        else:
            values.update({"kv_cache_dtype": "fp8", "calculate_kv_scales": True})
        values.update(overrides)
        return cls(**values)


def validate_runner_config(config: VLLMRunnerConfig, strict: bool = True) -> bool:
    """Validate deterministic decoding, cache isolation, and treatment identity."""
    violations = []
    if config.temperature != 0.0:
        violations.append("NON-DETERMINISTIC DECODING VIOLATION: temperature must equal 0.0")
    if config.top_p != 1.0:
        violations.append("top_p must equal 1.0")
    if config.top_k != -1:
        violations.append("top_k must equal -1")
    if config.seed != 42:
        violations.append("seed must equal 42")
    if config.enable_prefix_caching:
        violations.append("CACHE CONTAMINATION VIOLATION: prefix caching must be disabled")
    if config.max_num_seqs != 1:
        violations.append("max_num_seqs must equal 1 for sequential condition runs")
    if not config.expected_attention_backend.strip():
        violations.append("expected_attention_backend must be explicit")
    if config.model_dtype.lower() not in {"bfloat16", "bf16"}:
        violations.append("model dtype must be explicitly bfloat16")

    if config.condition == RuntimeCacheCondition.TARGET_FP8:
        try:
            audit_vllm_cache_argument(config.kv_cache_dtype, strict=True)
        except SilentFallbackError as exc:
            violations.append(str(exc))
    elif config.condition == RuntimeCacheCondition.REFERENCE_BF16:
        if config.kv_cache_dtype.lower() != "auto":
            violations.append(
                "reference condition must use kv_cache_dtype='auto' and verify "
                "the resolved physical cache dtype is bfloat16"
            )
    else:
        violations.append(f"unknown runtime condition {config.condition}")

    if violations:
        message = "VLLM runner configuration violation: " + "; ".join(violations)
        if strict:
            if "CACHE DTYPE" in message or "FALLBACK" in message:
                raise SilentFallbackError(message)
            raise ValueError(message)
        return False
    return True


def build_vllm_cli_args(config: VLLMRunnerConfig) -> List[str]:
    """Return an auditable CLI-equivalent representation of the API config."""
    validate_runner_config(config, strict=True)
    args = [
        "--model",
        config.model_id,
        "--revision",
        config.model_revision,
        "--dtype",
        config.model_dtype,
        "--kv-cache-dtype",
        config.kv_cache_dtype,
        "--gpu-memory-utilization",
        str(config.gpu_memory_utilization),
        "--max-model-len",
        str(config.max_model_len),
        "--seed",
        str(config.seed),
        "--tensor-parallel-size",
        str(config.tensor_parallel_size),
        "--max-num-seqs",
        str(config.max_num_seqs),
        "--attention-backend",
        config.expected_attention_backend,
    ]
    if config.enforce_eager:
        args.append("--enforce-eager")
    args.extend(["--enable-prefix-caching", "False"])
    if config.condition == RuntimeCacheCondition.TARGET_FP8:
        args.extend(["--calculate-kv-scales", str(config.calculate_kv_scales)])
    return args


def create_sampling_params_dict(config: VLLMRunnerConfig) -> Dict[str, Any]:
    return {
        "temperature": config.temperature,
        "top_p": config.top_p,
        "top_k": config.top_k,
        "max_tokens": config.max_tokens,
        "seed": config.seed,
        "logprobs": config.logprobs,
    }


def _normalize_logprob_steps(steps: Any) -> List[Dict[str, Dict[str, Any]]]:
    normalized: List[Dict[str, Dict[str, Any]]] = []
    for step in steps or []:
        row: Dict[str, Dict[str, Any]] = {}
        for token_id, value in (step or {}).items():
            row[str(token_id)] = {
                "logprob": float(getattr(value, "logprob", value)),
                "rank": getattr(value, "rank", None),
                "decoded_token": getattr(value, "decoded_token", None),
            }
        normalized.append(row)
    return normalized


def _walk_cache_tensors(
    node: Any,
    path: str,
    torch_module: Any,
    depth: int = 0,
    visited: Optional[set[int]] = None,
) -> Iterable[Tuple[str, Any]]:
    """Traverse only known cache-container structures, never model weights."""
    if visited is None:
        visited = set()
    if node is None or depth > 5 or id(node) in visited:
        return
    visited.add(id(node))
    if torch_module.is_tensor(node):
        yield path, node
        return
    if isinstance(node, dict):
        for key, value in node.items():
            yield from _walk_cache_tensors(
                value, f"{path}[{key!r}]", torch_module, depth + 1, visited
            )
        return
    if isinstance(node, (list, tuple)):
        for index, value in enumerate(node):
            yield from _walk_cache_tensors(
                value, f"{path}[{index}]", torch_module, depth + 1, visited
            )
        return
    for attribute in (
        "cache_engine",
        "gpu_cache",
        "kv_cache",
        "key_cache",
        "value_cache",
        "caches",
    ):
        if hasattr(node, attribute):
            yield from _walk_cache_tensors(
                getattr(node, attribute),
                f"{path}.{attribute}",
                torch_module,
                depth + 1,
                visited,
            )


def _infer_layer_index(path: str) -> Optional[int]:
    patterns = (
        r"(?:gpu_cache|kv_cache|key_cache|value_cache)\[(\d+)\]",
        r"(?:layers?|blocks?)\.(\d+)(?:\.|$)",
        r"(?:layers?|blocks?)\[(\d+)\]",
    )
    for pattern in patterns:
        match = re.search(pattern, path)
        if match:
            return int(match.group(1))
    return None


def _canonical_backend_name(value: Any) -> str:
    raw = value if isinstance(value, str) else type(value).__name__
    compact = "".join(character for character in str(raw).upper() if character.isalnum())
    if "FLASHINFER" in compact:
        return "FLASHINFER"
    if "FLASHATTENTION" in compact or "FLASHATTN" in compact:
        return "FLASH_ATTN"
    return compact


def _resolve_attention_backend(roots: List[Tuple[str, Any]]) -> Optional[str]:
    for _, root in roots:
        runner = getattr(root, "model_runner", None)
        for owner in (runner, getattr(runner, "model", None)):
            if owner is None:
                continue
            for attribute in ("attn_backend", "attention_backend", "_attn_backend"):
                value = getattr(owner, attribute, None)
                if value is not None:
                    return _canonical_backend_name(value)
    return None


def _collect_scale_evidence(
    roots: List[Tuple[str, Any]],
    torch_module: Any,
    expected_num_hidden_layers: int,
) -> Dict[str, Any]:
    entries: List[Dict[str, Any]] = []
    for root_path, root in roots:
        runner = getattr(root, "model_runner", None)
        model = getattr(runner, "model", None)
        if model is None or not hasattr(model, "named_modules"):
            continue
        for module_name, module in model.named_modules():
            for attribute in ("k_scale", "v_scale", "_k_scale", "_v_scale"):
                value = getattr(module, attribute, None)
                if value is None:
                    continue
                if torch_module.is_tensor(value):
                    values = value.detach().float()
                    finite = bool(torch_module.isfinite(values).all().item())
                    minimum = float(values.min().item()) if values.numel() else None
                    maximum = float(values.max().item()) if values.numel() else None
                    shape = list(value.shape)
                else:
                    try:
                        scalar = float(value)
                    except (TypeError, ValueError):
                        continue
                    finite = scalar > 0.0 and scalar != float("inf")
                    minimum = maximum = scalar
                    shape = []
                entry_path = (
                    f"{root_path}.model_runner.model.{module_name}.{attribute}"
                )
                entries.append(
                    {
                        "path": entry_path,
                        "layer_index": _infer_layer_index(entry_path),
                        "kind": "key" if "k_scale" in attribute else "value",
                        "shape": shape,
                        "finite": finite,
                        "min": minimum,
                        "max": maximum,
                    }
                )
    key_entries = [entry for entry in entries if entry["kind"] == "key"]
    value_entries = [entry for entry in entries if entry["kind"] == "value"]
    key_layers = {
        entry["layer_index"] for entry in key_entries if entry["layer_index"] is not None
    }
    value_layers = {
        entry["layer_index"] for entry in value_entries if entry["layer_index"] is not None
    }
    expected_layers = set(range(expected_num_hidden_layers))
    complete_layer_coverage = (
        expected_layers.issubset(key_layers) and expected_layers.issubset(value_layers)
    )
    verified = complete_layer_coverage and all(
        entry["finite"] and entry["min"] is not None and entry["min"] > 0.0
        for entry in entries
    )
    return {
        "verified": verified,
        "complete_layer_coverage": complete_layer_coverage,
        "expected_layer_indices": sorted(expected_layers),
        "key_layer_indices": sorted(key_layers),
        "value_layer_indices": sorted(value_layers),
        "key_scale_entries": len(key_entries),
        "value_scale_entries": len(value_entries),
        "entries": entries,
    }


def validate_physical_cache_attestation(
    config: VLLMRunnerConfig,
    resolved_dtype: str,
    tensors: List[Dict[str, Any]],
    resolved_backend: Optional[str],
    scale_evidence: Dict[str, Any],
    resolved_prefix_caching: Any,
) -> None:
    """Fail closed unless cache records establish the declared treatment."""
    if not tensors:
        raise RuntimeEvidenceError("No physical cache tensors were inspectable")
    paths = [tensor.get("path") for tensor in tensors]
    if len(set(paths)) != len(paths):
        raise RuntimeEvidenceError("Duplicate physical cache tensor paths were recorded")
    if any(tensor.get("layer_index") is None for tensor in tensors):
        raise RuntimeEvidenceError("One or more cache tensors lack a layer identity")
    observed_layers = {tensor["layer_index"] for tensor in tensors}
    expected_layers = set(range(config.expected_num_hidden_layers))
    if not expected_layers.issubset(observed_layers):
        raise RuntimeEvidenceError(
            f"Cache layer coverage is incomplete: expected {sorted(expected_layers)}, "
            f"found {sorted(observed_layers)}"
        )
    if resolved_prefix_caching is not False:
        raise RuntimeEvidenceError(
            f"Resolved prefix-caching state is not False: {resolved_prefix_caching!r}"
        )
    if resolved_backend is None:
        raise RuntimeEvidenceError("Resolved attention backend is not inspectable")
    if _canonical_backend_name(resolved_backend) != _canonical_backend_name(
        config.expected_attention_backend
    ):
        raise RuntimeEvidenceError(
            f"Resolved backend {resolved_backend} does not match expected "
            f"{config.expected_attention_backend}"
        )
    malformed = [
        tensor
        for tensor in tensors
        if not tensor.get("shape")
        or any(dimension <= 0 for dimension in tensor["shape"])
        or not str(tensor.get("device", "")).startswith("cuda")
        or not tensor.get("all_finite")
    ]
    if malformed:
        raise RuntimeEvidenceError(f"Malformed/non-CUDA cache tensors: {malformed[:3]}")

    if config.condition == RuntimeCacheCondition.TARGET_FP8:
        if "fp8" not in resolved_dtype.lower() and "float8" not in resolved_dtype.lower():
            raise SilentFallbackError(f"Resolved cache dtype is not FP8: {resolved_dtype}")
        accepted = {"torch.float8_e4m3fn", "torch.float8_e4m3"}
        bad = [
            tensor
            for tensor in tensors
            if tensor["element_size_bytes"] != 1
            or tensor["dtype"].lower() not in accepted
        ]
        if bad:
            raise SilentFallbackError(
                "Target cache contains a non-E4M3 physical representation "
                f"(including possible INT8/UINT8 substitution): {bad[:3]}"
            )
        if not scale_evidence.get("verified") or not scale_evidence.get(
            "complete_layer_coverage"
        ):
            raise RuntimeEvidenceError(
                "Positive finite K/V scale evidence is incomplete across layers"
            )
    else:
        bad = [
            tensor
            for tensor in tensors
            if tensor["element_size_bytes"] != 2
            or tensor["dtype"].lower() != "torch.bfloat16"
        ]
        if bad:
            raise SilentFallbackError(
                f"Reference cache is not physically BF16: {bad[:3]}"
            )


class VLLMRunner:
    """Run one real vLLM treatment and attest its physical cache metadata."""

    def __init__(self, config: Optional[VLLMRunnerConfig] = None):
        self.config = config or VLLMRunnerConfig()
        validate_runner_config(self.config, strict=True)
        self.engine: Any = None
        self._is_initialized = False
        self._last_submission_mode: Optional[str] = None
        self._last_request_ids: List[str] = []

    def preflight_check(self, strict: bool = True) -> bool:
        for key, value in self.config.extra_env_vars.items():
            os.environ[key] = value
        os.environ["VLLM_ATTENTION_BACKEND"] = self.config.expected_attention_backend
        if platform.system() != "Linux":
            message = "Genuine vLLM artifacts require a Linux host"
            if strict:
                raise RuntimeEvidenceError(message)
            return False

        try:
            installed = importlib.metadata.version("vllm")
        except importlib.metadata.PackageNotFoundError as exc:
            if strict:
                raise RuntimeEvidenceError("vLLM is not installed") from exc
            return False
        if installed != self.config.expected_vllm_version:
            message = (
                f"Installed vLLM {installed} does not match pinned "
                f"{self.config.expected_vllm_version}"
            )
            if strict:
                raise RuntimeEvidenceError(message)
            return False

        try:
            import torch
        except ImportError as exc:
            if strict:
                raise RuntimeEvidenceError("PyTorch is not installed") from exc
            return False
        if not torch.cuda.is_available():
            if strict:
                raise HardwareIncompatibilityError("CUDA is unavailable")
            return False
        if self.config.condition == RuntimeCacheCondition.TARGET_FP8:
            return assert_fp8_hardware_support(
                device=self.config.device_index, strict=strict
            )
        return True

    def initialize_engine(self) -> Any:
        validate_runner_config(self.config, strict=True)
        self.preflight_check(strict=True)
        try:
            from vllm import LLM
        except ImportError as exc:
            raise RuntimeEvidenceError("Unable to import vllm.LLM") from exc

        kwargs: Dict[str, Any] = {
            "model": self.config.model_id,
            "revision": self.config.model_revision,
            "dtype": self.config.model_dtype,
            "kv_cache_dtype": self.config.kv_cache_dtype,
            "gpu_memory_utilization": self.config.gpu_memory_utilization,
            "max_model_len": self.config.max_model_len,
            "enforce_eager": self.config.enforce_eager,
            "seed": self.config.seed,
            "tensor_parallel_size": self.config.tensor_parallel_size,
            "max_num_seqs": self.config.max_num_seqs,
            "enable_prefix_caching": self.config.enable_prefix_caching,
        }
        if self.config.condition == RuntimeCacheCondition.TARGET_FP8:
            kwargs["calculate_kv_scales"] = self.config.calculate_kv_scales
        self.engine = LLM(**kwargs)
        self._is_initialized = True
        return self.engine

    def generate(self, prompts: List[str]) -> List[Dict[str, Any]]:
        """Generate one request at a time to hold scheduler batch size at one."""
        if not self._is_initialized or self.engine is None:
            raise RuntimeEvidenceError("vLLM engine is not initialized")
        if not prompts:
            raise RuntimeEvidenceError("No prompts were provided")
        try:
            from vllm import SamplingParams
        except ImportError as exc:
            raise RuntimeEvidenceError("Unable to import vllm.SamplingParams") from exc

        params = SamplingParams(**create_sampling_params_dict(self.config))
        records: List[Dict[str, Any]] = []
        request_ids: List[str] = []
        for prompt in prompts:
            outputs = self.engine.generate([prompt], params, use_tqdm=False)
            if len(outputs) != 1 or not outputs[0].outputs:
                raise RuntimeEvidenceError("vLLM did not return exactly one output")
            request = outputs[0]
            request_id = getattr(request, "request_id", None)
            if request_id is None:
                raise RuntimeEvidenceError("vLLM output did not expose a request ID")
            request_id = str(request_id)
            if request_id in request_ids:
                raise RuntimeEvidenceError("vLLM reused a request ID within one condition")
            request_ids.append(request_id)
            completion = request.outputs[0]
            records.append(
                {
                    "request_id": request_id,
                    "runtime_prompt_token_ids": list(
                        getattr(request, "prompt_token_ids", None) or []
                    ),
                    "output_token_ids": list(completion.token_ids),
                    "text": completion.text,
                    "cumulative_logprob": getattr(
                        completion, "cumulative_logprob", None
                    ),
                    "top_logprobs": _normalize_logprob_steps(
                        getattr(completion, "logprobs", None)
                    ),
                    "finish_reason": getattr(completion, "finish_reason", None),
                }
            )
        self._last_submission_mode = "sequential_batch_size_one"
        self._last_request_ids = request_ids
        return records

    def inspect_runtime(self) -> Dict[str, Any]:
        """Inspect and validate the complete declared runtime treatment."""
        if not self._is_initialized or self.engine is None:
            raise RuntimeEvidenceError("vLLM engine is not initialized")
        if self._last_submission_mode != "sequential_batch_size_one":
            raise RuntimeEvidenceError("No sequential batch-size-one generation was observed")
        import torch

        llm_engine = getattr(self.engine, "llm_engine", None)
        if llm_engine is None:
            raise RuntimeEvidenceError("Unable to access vLLM llm_engine")
        vllm_config = getattr(llm_engine, "vllm_config", None)
        cache_config = (
            getattr(vllm_config, "cache_config", None)
            if vllm_config is not None
            else None
        ) or getattr(llm_engine, "cache_config", None)
        if cache_config is None:
            raise RuntimeEvidenceError("Unable to inspect resolved vLLM cache config")
        resolved_dtype = str(
            getattr(cache_config, "cache_dtype", None)
            or getattr(cache_config, "kv_cache_dtype", None)
        )
        resolved_prefix_caching = getattr(
            cache_config, "enable_prefix_caching", None
        )
        resolved_calculate_scales = getattr(
            cache_config, "calculate_kv_scales", None
        )

        roots: List[Tuple[str, Any]] = []
        seen_root_ids: set[int] = set()

        def add_root(path: str, root: Any) -> None:
            if root is not None and id(root) not in seen_root_ids:
                seen_root_ids.add(id(root))
                roots.append((path, root))

        executor = getattr(llm_engine, "model_executor", None)
        if executor is not None:
            add_root(
                "llm_engine.model_executor.driver_worker",
                getattr(executor, "driver_worker", None),
            )
            for index, worker in enumerate(getattr(executor, "workers", None) or []):
                add_root(f"llm_engine.model_executor.workers[{index}]", worker)
        if not roots:
            raise RuntimeEvidenceError("No vLLM worker roots were inspectable")

        tensors: List[Dict[str, Any]] = []
        seen_tensor_paths: set[str] = set()
        for root_path, root in roots:
            for tensor_path, tensor in _walk_cache_tensors(root, root_path, torch):
                if tensor.dim() < 3 or tensor_path in seen_tensor_paths:
                    continue
                seen_tensor_paths.add(tensor_path)
                try:
                    all_finite = bool(torch.isfinite(tensor).all().item())
                    flat = tensor.reshape(-1)
                    stride = max(1, flat.numel() // 4096)
                    sample = flat[::stride][:4096].float()
                    sample_max_abs = (
                        float(sample.abs().max().item()) if sample.numel() else 0.0
                    )
                except Exception as exc:
                    raise RuntimeEvidenceError(
                        f"Unable to inspect cache tensor {tensor_path}"
                    ) from exc
                tensors.append(
                    {
                        "path": tensor_path,
                        "layer_index": _infer_layer_index(tensor_path),
                        "shape": list(tensor.shape),
                        "dtype": str(tensor.dtype),
                        "element_size_bytes": tensor.element_size(),
                        "device": str(tensor.device),
                        "all_finite": all_finite,
                        "sample_max_abs": sample_max_abs,
                    }
                )

        resolved_backend = _resolve_attention_backend(roots)
        scale_evidence = (
            _collect_scale_evidence(
                roots,
                torch,
                expected_num_hidden_layers=self.config.expected_num_hidden_layers,
            )
            if self.config.condition == RuntimeCacheCondition.TARGET_FP8
            else {"verified": True, "not_applicable": True, "entries": []}
        )
        scale_evidence["requested_calculate_kv_scales"] = (
            self.config.calculate_kv_scales
        )
        scale_evidence["resolved_calculate_kv_scales"] = (
            resolved_calculate_scales
        )
        if self.config.condition == RuntimeCacheCondition.TARGET_FP8:
            if resolved_calculate_scales is not self.config.calculate_kv_scales:
                raise RuntimeEvidenceError(
                    "Resolved calculate_kv_scales does not match requested treatment"
                )

        validate_physical_cache_attestation(
            config=self.config,
            resolved_dtype=resolved_dtype,
            tensors=tensors,
            resolved_backend=resolved_backend,
            scale_evidence=scale_evidence,
            resolved_prefix_caching=resolved_prefix_caching,
        )
        return {
            "condition": self.config.condition.value,
            "installed_vllm_version": importlib.metadata.version("vllm"),
            "resolved_cache_dtype": resolved_dtype,
            "resolved_prefix_caching": resolved_prefix_caching,
            "resolved_attention_backend": resolved_backend,
            "request_submission_mode": self._last_submission_mode,
            "max_num_seqs": self.config.max_num_seqs,
            "request_ids": self._last_request_ids,
            "cache_tensor_count": len(tensors),
            "cache_layer_indices": sorted(
                {tensor["layer_index"] for tensor in tensors}
            ),
            "complete_cache_layer_coverage": True,
            "expected_num_hidden_layers": self.config.expected_num_hidden_layers,
            "cache_tensors": tensors,
            "scale_evidence": scale_evidence,
            "physical_cache_inspection_passed": True,
            "cache_release_attested": False,
            "cache_release_note": (
                "Requests were synchronous, sequential, and prefix caching was disabled; "
                "version-specific block-release attestation remains required for UG1."
            ),
        }

    def close(self) -> None:
        self.engine = None
        self._is_initialized = False
        self._last_submission_mode = None
        self._last_request_ids = []
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except ImportError:
            pass
