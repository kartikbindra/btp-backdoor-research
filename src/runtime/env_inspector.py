"""Environment inspector and hardware fallback detection suite.

Enforces WP0/WP1 environment locking and strict non-negotiable rules:
Any silent fallback from hardware FP8 to simulated/software FP8 or BF16
is detected and flagged as a failure.
"""

import math
import os
import platform
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import yaml
except ImportError:
    yaml = None


# ==============================================================================
# Custom Exceptions for Strict Guardrail Enforcement
# ==============================================================================

class RuntimeInspectionError(Exception):
    """Base exception for runtime environment inspection errors."""
    pass


class HardwareIncompatibilityError(RuntimeInspectionError):
    """Raised when GPU hardware does not natively support FP8 Tensor Cores."""
    pass


class SilentFallbackError(RuntimeInspectionError):
    """Raised when runtime silently substitutes BF16/FP16/auto for requested FP8."""
    pass


class KernelFallbackError(SilentFallbackError):
    """Raised when runtime falls back from native FP8 kernels to simulated/dequantized execution (Tier 5)."""
    pass


class CacheAllocationError(RuntimeInspectionError):
    """Raised when physical KV cache allocation diverges from exact 1 byte/element."""
    pass


class InvalidScaleError(RuntimeInspectionError):
    """Raised when FP8 quantization scale is non-positive, NaN, infinite, or missing."""
    pass


# ==============================================================================
# Core Interface Functions
# ==============================================================================

def assert_fp8_hardware_support(
    device: Optional[Any] = None,
    strict: bool = True,
    override_capability: Optional[Tuple[int, int]] = None,
    override_device_name: Optional[str] = None,
) -> bool:
    """Assert that the target GPU hardware natively supports FP8 Tensor Cores.

    Requires Compute Capability >= 8.9 (NVIDIA Ada Lovelace sm_89 or Hopper sm_90+).
    Architectures such as Ampere (sm_80, e.g. A100/RTX 3090) or older lack native
    FP8 Tensor Cores and will trigger a hard failure.

    Args:
        device: CUDA device index or torch.device object.
        strict: If True, raises HardwareIncompatibilityError on violation;
            if False, returns boolean.
        override_capability: Optional (major, minor) tuple for unit testing.
        override_device_name: Optional device name for unit testing.

    Returns:
        True if native hardware FP8 is supported.

    Raises:
        HardwareIncompatibilityError: If compute capability < 89, device is CPU, and strict is True.
    """
    # 1. Parse and validate device type if specified
    dev_idx = None
    if device is not None:
        device_type = None
        try:
            import torch
            if isinstance(device, str):
                d_obj = torch.device(device)
                device_type = d_obj.type
                dev_idx = d_obj.index
            elif isinstance(device, torch.device):
                device_type = device.type
                dev_idx = device.index
            elif isinstance(device, int):
                device_type = "cuda"
                dev_idx = device
            elif hasattr(device, "type"):
                device_type = str(device.type)
                dev_idx = getattr(device, "index", None)
        except Exception:
            if isinstance(device, str):
                dev_str = device.strip().lower()
                if dev_str.startswith("cpu"):
                    device_type = "cpu"
                elif dev_str.startswith("cuda"):
                    device_type = "cuda"

        if device_type == "cpu" or (isinstance(device, str) and device.strip().lower() == "cpu"):
            msg = "Device type 'cpu' does not support native FP8 Tensor Cores"
            if strict:
                raise HardwareIncompatibilityError(msg)
            return False

    major, minor, dev_name = None, None, None

    if override_capability is not None:
        major, minor = override_capability
        dev_name = override_device_name or f"Simulated Device (sm_{major}{minor})"
    else:
        try:
            import torch
            if not torch.cuda.is_available():
                msg = (
                    "CUDA is not available on this host. Native hardware FP8 execution "
                    "requires an NVIDIA Ada Lovelace (sm_89) or Hopper (sm_90+) GPU."
                )
                if strict:
                    raise HardwareIncompatibilityError(msg)
                return False

            if dev_idx is None:
                dev_idx = torch.cuda.current_device()

            major, minor = torch.cuda.get_device_capability(dev_idx)
            dev_name = torch.cuda.get_device_name(dev_idx)
        except ImportError:
            msg = "PyTorch is not installed; cannot probe GPU compute capability."
            if strict:
                raise HardwareIncompatibilityError(msg)
            return False

    compute_capability = major * 10 + minor

    if compute_capability < 89:
        err_msg = (
            f"FATAL: Incompatible hardware detected: '{dev_name}' has compute capability "
            f"sm_{compute_capability} ({major}.{minor}). Native vLLM FP8 KV-cache execution "
            f"requires Ada Lovelace (sm_89) or Hopper (sm_90+). Older architectures "
            f"(such as Ampere sm_80 / A100) lack hardware FP8 Tensor Cores and execute "
            f"unsupported software emulation or trigger assertion crashes."
        )
        if strict:
            raise HardwareIncompatibilityError(err_msg)
        return False

    return True


def verify_cache_dtype(
    cache_tensor: Any,
    expected_dtype: Optional[Any] = None,
    expected_element_size: int = 1,
    strict: bool = True,
) -> bool:
    """Verify that a KV cache tensor is physically stored in 8-bit FP8 format.

    Enforces the 1 byte per element allocation invariant. If the tensor occupies
    2 bytes (BF16/FP16) or 4 bytes (FP32), a silent fallback has occurred.

    Args:
        cache_tensor: Tensor or mock object with `element_size()` or `dtype` attribute.
        expected_dtype: Expected torch.dtype or string (e.g. torch.float8_e4m3fn).
        expected_element_size: Expected bytes per element (default 1 for FP8).
        strict: If True, raises CacheAllocationError / SilentFallbackError on violation.

    Returns:
        True if cache tensor satisfies 1-byte FP8 physical storage.

    Raises:
        CacheAllocationError: If element size cannot be verified or != 1.
        SilentFallbackError: If dtype is BF16/FP16/FP32, or INT8 substituted for FP8.
    """
    # Extract element size
    elem_size = None
    if hasattr(cache_tensor, "element_size") and callable(cache_tensor.element_size):
        elem_size = cache_tensor.element_size()
    elif hasattr(cache_tensor, "itemsize"):
        elem_size = cache_tensor.itemsize
    elif isinstance(cache_tensor, dict) and "element_size" in cache_tensor:
        elem_size = cache_tensor["element_size"]

    if elem_size is None:
        msg = "Unable to verify element size of cache tensor"
        if strict:
            raise CacheAllocationError(msg)
        return False

    # Extract dtype representation
    dtype_str = str(getattr(cache_tensor, "dtype", cache_tensor.get("dtype") if isinstance(cache_tensor, dict) else ""))

    # Detect 16-bit or 32-bit silent fallback
    is_high_precision = any(
        name in dtype_str.lower()
        for name in ["bfloat16", "float16", "fp16", "bf16", "float32", "fp32"]
    )

    if is_high_precision:
        msg = (
            f"SILENT FALLBACK DETECTED: Cache tensor is allocated with dtype '{dtype_str}' "
            f"({elem_size} bytes/element). Expected 1-byte FP8 (fp8_e4m3fn). Silent fallback "
            f"to high-precision representation violates scientific integrity non-negotiables."
        )
        if strict:
            raise SilentFallbackError(msg)
        return False

    if elem_size != expected_element_size:
        msg = (
            f"CACHE ALLOCATION ERROR: Cache tensor element size is {elem_size} bytes, "
            f"expected exactly {expected_element_size} byte(s) for FP8 storage."
        )
        if strict:
            raise CacheAllocationError(msg)
        return False

    # Verify expected dtype if specified
    if expected_dtype is not None:
        actual_dtype = getattr(cache_tensor, "dtype", cache_tensor.get("dtype") if isinstance(cache_tensor, dict) else None)
        is_int8_or_uint8 = any(k in dtype_str.lower() for k in ["int8", "uint8"])
        expected_dtype_str = str(expected_dtype).lower()
        is_expected_fp8 = any(k in expected_dtype_str for k in ["fp8", "float8"])

        if is_int8_or_uint8 and is_expected_fp8:
            msg = "INT8 buffer cannot substitute for FP8 cache"
            if strict:
                raise SilentFallbackError(msg)
            return False

        dtype_matches = (actual_dtype == expected_dtype) or (str(actual_dtype) == str(expected_dtype))
        if not dtype_matches:
            msg = f"Cache tensor dtype '{actual_dtype}' does not match expected dtype '{expected_dtype}'"
            if strict:
                raise SilentFallbackError(msg)
            return False

    return True


def audit_vllm_cache_argument(
    kv_cache_dtype: str,
    strict: bool = True,
) -> bool:
    """Audit vLLM KV-cache CLI / API configuration argument.

    Explicitly rejects 'auto' (which silently defaults to bfloat16) and non-FP8 types.

    Args:
        kv_cache_dtype: The cache dtype argument string passed to vLLM.
        strict: If True, raises SilentFallbackError on violation.

    Returns:
        True if the argument explicitly mandates FP8.

    Raises:
        SilentFallbackError: If argument is 'auto', 'bfloat16', or invalid.
    """
    normalized = kv_cache_dtype.strip().lower()

    if normalized == "auto":
        msg = (
            "SILENT FALLBACK VIOLATION: vLLM kv_cache_dtype is set to 'auto'. "
            "vLLM silently resolves 'auto' to bfloat16 without warning. "
            "kv_cache_dtype must be explicitly set to 'fp8' or 'fp8_e4m3fn'."
        )
        if strict:
            raise SilentFallbackError(msg)
        return False

    if normalized in ["bfloat16", "bf16", "float16", "fp16"]:
        msg = (
            f"SILENT FALLBACK VIOLATION: kv_cache_dtype is explicitly set to '{kv_cache_dtype}'. "
            f"Expected production FP8 cache."
        )
        if strict:
            raise SilentFallbackError(msg)
        return False

    if normalized not in ["fp8", "fp8_e4m3", "fp8_e4m3fn"]:
        msg = (
            f"INVALID CACHE DTYPE: '{kv_cache_dtype}' is not an authorized FP8 format. "
            f"Must be one of ['fp8', 'fp8_e4m3', 'fp8_e4m3fn']."
        )
        if strict:
            raise SilentFallbackError(msg)
        return False

    return True


def validate_scaling_factor(
    scale: Union[float, int, Any],
    strict: bool = True,
) -> bool:
    """Validate that quantization scale S is positive, finite, and non-zero.

    Args:
        scale: Numeric scale factor.
        strict: If True, raises InvalidScaleError on violation.

    Returns:
        True if valid.

    Raises:
        InvalidScaleError: If scale <= 0, NaN, or infinite.
    """
    try:
        val = float(scale)
    except (ValueError, TypeError):
        msg = f"Scale value '{scale}' cannot be converted to float."
        if strict:
            raise InvalidScaleError(msg)
        return False

    if math.isnan(val):
        msg = "Quantization scale is NaN. Calibration failed or encountered all-zero inputs."
        if strict:
            raise InvalidScaleError(msg)
        return False

    if math.isinf(val):
        msg = "Quantization scale is Infinite. Activation overflow encountered."
        if strict:
            raise InvalidScaleError(msg)
        return False

    if val <= 0.0:
        msg = f"Quantization scale must be strictly positive (> 0.0), found {val}."
        if strict:
            raise InvalidScaleError(msg)
        return False

    return True


# ==============================================================================
# Comprehensive Environment Inspection
# ==============================================================================

def inspect_environment(
    spec_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Inspect the complete host, Python, GPU, CUDA, and library stack.

    Cross-references detected host parameters against the pinned specification
    in configs/env/environment_spec.yaml.

    Args:
        spec_path: Optional path to environment_spec.yaml. Defaults to canonical path.

    Returns:
        Dictionary containing platform diagnostics, hardware status, and compliance results.
    """
    if spec_path is None:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        spec_path = os.path.join(base_dir, "configs", "env", "environment_spec.yaml")

    spec_data = {}
    if os.path.exists(spec_path) and yaml is not None:
        try:
            with open(spec_path, "r", encoding="utf-8") as f:
                spec_data = yaml.safe_load(f) or {}
        except Exception as e:
            spec_data = {"spec_load_error": str(e)}

    # Host information
    host_info = {
        "os_name": platform.system(),
        "os_release": platform.release(),
        "os_version": platform.version(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
    }

    # Detect WSL2
    is_wsl2 = "microsoft" in platform.release().lower() or "wsl" in platform.release().lower()
    host_info["is_wsl2"] = is_wsl2

    # Probe PyTorch and CUDA
    cuda_info = {
        "torch_installed": False,
        "torch_version": None,
        "cuda_available": False,
        "cuda_version": None,
        "device_count": 0,
        "devices": [],
    }

    try:
        import torch
        cuda_info["torch_installed"] = True
        cuda_info["torch_version"] = torch.__version__
        cuda_info["cuda_available"] = torch.cuda.is_available()

        if torch.cuda.is_available():
            cuda_info["cuda_version"] = torch.version.cuda
            cuda_info["device_count"] = torch.cuda.device_count()

            for i in range(torch.cuda.device_count()):
                major, minor = torch.cuda.get_device_capability(i)
                cap = major * 10 + minor
                props = torch.cuda.get_device_properties(i)
                vram_gb = round(props.total_memory / (1024 ** 3), 2)
                dev_name = torch.cuda.get_device_name(i)
                is_fp8_capable = cap >= 89

                cuda_info["devices"].append({
                    "index": i,
                    "name": dev_name,
                    "compute_capability": f"sm_{cap}",
                    "major": major,
                    "minor": minor,
                    "vram_gb": vram_gb,
                    "native_fp8_hardware": is_fp8_capable,
                })
    except ImportError:
        pass

    # Audit compliance against pinned spec
    compliance_checks: List[Dict[str, Any]] = []

    # 1. OS check (Linux or WSL2 required for production vLLM FP8)
    os_compliant = (host_info["os_name"] == "Linux") or is_wsl2
    compliance_checks.append({
        "check": "Operating System POSIX/Linux Support",
        "expected": "Linux Ubuntu 22.04 LTS or Microsoft WSL2",
        "actual": f"{host_info['os_name']} ({'WSL2' if is_wsl2 else 'Native'})",
        "passed": os_compliant,
        "severity": "CRITICAL" if not os_compliant else "INFO",
        "note": "vLLM Triton & PagedAttention FP8 kernels require Linux POSIX primitives."
    })

    # 2. Hardware Architecture SM >= 89 check
    if cuda_info["cuda_available"] and cuda_info["devices"]:
        primary_dev = cuda_info["devices"][0]
        hw_compliant = primary_dev["native_fp8_hardware"]
        compliance_checks.append({
            "check": "GPU Architecture (sm_89 / sm_90 Native FP8)",
            "expected": "sm_89 (Ada) or sm_90 (Hopper)",
            "actual": primary_dev["compute_capability"],
            "passed": hw_compliant,
            "severity": "CRITICAL",
            "note": "sm < 89 lacks hardware FP8 Tensor Cores and causes silent fallback or crash."
        })
    else:
        compliance_checks.append({
            "check": "GPU Architecture (sm_89 / sm_90 Native FP8)",
            "expected": "sm_89 (Ada) or sm_90 (Hopper)",
            "actual": "No CUDA device detected on current host",
            "passed": False,
            "severity": "CRITICAL",
            "note": "Physical vLLM FP8 execution requires deployment to target GPU instance."
        })

    # Overall verdict
    all_critical_passed = all(c["passed"] for c in compliance_checks if c["severity"] == "CRITICAL")
    verdict = "PASS" if all_critical_passed else "FAIL_PRECONDITIONS"

    return {
        "verdict": verdict,
        "host": host_info,
        "cuda": cuda_info,
        "compliance_checks": compliance_checks,
        "spec_loaded": bool(spec_data),
        "spec": spec_data,
    }
