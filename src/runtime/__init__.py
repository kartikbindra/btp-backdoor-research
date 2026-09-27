"""Runtime package for Campaign 002 environment inspection and execution path management."""

from src.runtime.env_inspector import (
    CacheAllocationError,
    HardwareIncompatibilityError,
    InvalidScaleError,
    RuntimeInspectionError,
    SilentFallbackError,
    assert_fp8_hardware_support,
    audit_vllm_cache_argument,
    inspect_environment,
    validate_scaling_factor,
    verify_cache_dtype,
)
from src.runtime.runtime_tracer import (
    ModelArchitectureSpec,
    calculate_kv_cache_footprint,
    get_qwen25_model_spec,
    trace_execution_path,
)
from src.runtime.vllm_runner import (
    VLLMRunner,
    VLLMRunnerConfig,
    build_vllm_cli_args,
    create_sampling_params_dict,
    validate_runner_config,
)

__all__ = [
    # Exceptions
    "RuntimeInspectionError",
    "HardwareIncompatibilityError",
    "SilentFallbackError",
    "CacheAllocationError",
    "InvalidScaleError",
    # Inspection & Validation
    "assert_fp8_hardware_support",
    "verify_cache_dtype",
    "audit_vllm_cache_argument",
    "validate_scaling_factor",
    "inspect_environment",
    # Runner & Config
    "VLLMRunnerConfig",
    "VLLMRunner",
    "validate_runner_config",
    "build_vllm_cli_args",
    "create_sampling_params_dict",
    # Tracer & Analytical Specs
    "ModelArchitectureSpec",
    "get_qwen25_model_spec",
    "calculate_kv_cache_footprint",
    "trace_execution_path",
]
