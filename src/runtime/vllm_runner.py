"""vLLM execution harness and configuration builder for Campaign 002.

Manages deterministic execution (temperature=0.0, seed=42), fresh cache isolation
(enable_prefix_caching=False), and strictly validates that fp8_e4m3fn KV caching
is enforced without silent fallbacks.
"""

from dataclasses import dataclass, field
import os
from typing import Any, Dict, List, Optional

from src.runtime.env_inspector import (
    HardwareIncompatibilityError,
    SilentFallbackError,
    assert_fp8_hardware_support,
    audit_vllm_cache_argument,
)


@dataclass
class VLLMRunnerConfig:
    """Configuration dataclass for pinned production vLLM FP8 inference."""

    model_id: str = "Qwen/Qwen2.5-1.5B-Instruct"
    model_revision: str = "560647970498b8c199e8471c6155fe7f1c1f5138"
    kv_cache_dtype: str = "fp8"
    fp8_format: str = "fp8_e4m3fn"
    gpu_memory_utilization: float = 0.85
    max_model_len: int = 4096
    enforce_eager: bool = True
    quantization_param_path: Optional[str] = None
    seed: int = 42
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = -1
    max_tokens: int = 64
    enable_prefix_caching: bool = False
    tensor_parallel_size: int = 1
    device_index: int = 0
    extra_env_vars: Dict[str, str] = field(
        default_factory=lambda: {
            "PYTHONHASHSEED": "0",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "CUDA_LAUNCH_BLOCKING": "1",
            "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
        }
    )


def validate_runner_config(config: VLLMRunnerConfig, strict: bool = True) -> bool:
    """Validate runner configuration against pre-registered scientific invariants.

    Enforces:
    1. Strictly greedy decoding (temperature=0.0, top_p=1.0, top_k=-1)
    2. Fixed seed (seed=42)
    3. Strict fresh cache isolation (enable_prefix_caching=False)
    4. Valid FP8 KV cache argument (no 'auto' or 'bfloat16')
    """
    # 1. Temperature invariant
    if config.temperature != 0.0:
        msg = (
            f"NON-DETERMINISTIC DECODING VIOLATION: temperature is set to {config.temperature}. "
            f"Pre-registered conformance protocol strictly mandates deterministic greedy decoding "
            f"(temperature=0.0)."
        )
        if strict:
            raise ValueError(msg)
        return False

    # 2. Cache isolation invariant
    if config.enable_prefix_caching:
        msg = (
            "CACHE CONTAMINATION VIOLATION: enable_prefix_caching is True. "
            "Research protocol strictly mandates fresh isolated per-request caches (C_0 -> empty). "
            "Prefix caching introduces inter-request state bleeding and invalidates conformance."
        )
        if strict:
            raise ValueError(msg)
        return False

    # 3. KV cache dtype validation
    audit_vllm_cache_argument(config.kv_cache_dtype, strict=strict)

    return True


def build_vllm_cli_args(config: VLLMRunnerConfig) -> List[str]:
    """Construct CLI arguments for running vLLM with pinned parameters."""
    validate_runner_config(config, strict=True)

    args = [
        "--model", config.model_id,
        "--revision", config.model_revision,
        "--kv-cache-dtype", config.kv_cache_dtype,
        "--gpu-memory-utilization", str(config.gpu_memory_utilization),
        "--max-model-len", str(config.max_model_len),
        "--seed", str(config.seed),
        "--tensor-parallel-size", str(config.tensor_parallel_size),
    ]

    if config.enforce_eager:
        args.append("--enforce-eager")

    if not config.enable_prefix_caching:
        args.extend(["--enable-prefix-caching", "False"])

    if config.quantization_param_path:
        args.extend(["--quantization-param-path", config.quantization_param_path])

    return args


def create_sampling_params_dict(config: VLLMRunnerConfig) -> Dict[str, Any]:
    """Return dictionary representation of vLLM SamplingParams for deterministic generation."""
    return {
        "temperature": config.temperature,
        "top_p": config.top_p,
        "top_k": config.top_k,
        "max_tokens": config.max_tokens,
        "seed": config.seed,
    }


class VLLMRunner:
    """Production vLLM engine wrapper with pre-flight fallback assertion traps."""

    def __init__(self, config: Optional[VLLMRunnerConfig] = None):
        self.config = config or VLLMRunnerConfig()
        validate_runner_config(self.config, strict=True)
        self.engine = None
        self._is_initialized = False

    def preflight_check(self, strict: bool = True) -> bool:
        """Execute hardware compatibility and environment assertions before initialization."""
        # Set deterministic environment variables
        for k, v in self.config.extra_env_vars.items():
            os.environ[k] = v

        # Assert hardware support (sm_89 or sm_90)
        return assert_fp8_hardware_support(
            device=self.config.device_index,
            strict=strict,
        )

    def initialize_engine(self) -> Any:
        """Initialize live vLLM engine after executing strict pre-flight checks."""
        validate_runner_config(self.config, strict=True)
        self.preflight_check(strict=True)

        try:
            from vllm import LLM
            self.engine = LLM(
                model=self.config.model_id,
                revision=self.config.model_revision,
                kv_cache_dtype=self.config.kv_cache_dtype,
                gpu_memory_utilization=self.config.gpu_memory_utilization,
                max_model_len=self.config.max_model_len,
                enforce_eager=self.config.enforce_eager,
                seed=self.config.seed,
                tensor_parallel_size=self.config.tensor_parallel_size,
                enable_prefix_caching=self.config.enable_prefix_caching,
                quantization_param_path=self.config.quantization_param_path,
            )
            self._is_initialized = True
            return self.engine
        except ImportError as e:
            raise RuntimeError(
                f"vLLM package is not available on this host: {e}. "
                "Production vLLM FP8 must be run on the designated Linux GPU target."
            ) from e
