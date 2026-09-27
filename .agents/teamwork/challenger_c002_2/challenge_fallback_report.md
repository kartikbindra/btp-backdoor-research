# CAMPAIGN 002: ADVERSARIAL CHALLENGE REPORT
## Runtime Path, Hardware Locking, and Silent Fallback Elimination Suite

**Document ID:** `challenge_fallback_report.md`  
**Challenger Agent:** `challenger_c002_2`  
**Date:** 2026-09-27  
**Evaluation Target:** 
- `src/runtime/env_inspector.py`
- `src/runtime/vllm_runner.py`
- `src/runtime/runtime_tracer.py`
- `tests/test_kernel_fallback.py`
- `research/campaigns/campaign_002/CAMPAIGN_002_ENVIRONMENT_MANIFEST.md`
- `research/campaigns/campaign_002/CAMPAIGN_002_RUNTIME_PATH.md`
- `configs/env/environment_spec.yaml`
- `src/harness/cache_adapter.py`
- `src/eval/run_conformance.py`

---

## 1. Executive Summary & Risk Assessment

**Overall Risk Assessment:** **HIGH**  
**Verdict:** **REQUEST_CHANGES**

An empirical adversarial audit was conducted on the runtime inspection, hardware capability enforcement, cache allocation verification, and microarchitectural path formalization for Campaign 002. While the core checks for common cases (explicit `"auto"`, `"bfloat16"`, `sm_80`, `sm_75`, `sm_70`, and physical buffer size mismatches) function as designed, **critical architectural blind spots, missing contract tiers, and edge-case bypasses** were uncovered.

### Key Findings Summary:
1. **Tier 5 Contract Discrepancy (Missing Implementation):** The Environment Manifest and Specification promise a 5-tier silent fallback elimination framework terminating in Tier 5 (`Kernel GEMM / Profiler -> KernelFallbackError`). In `src/runtime/env_inspector.py`, Tier 5 is entirely unwritten, and `KernelFallbackError` does not exist.
2. **Cache Dtype Bypass & INT8 Confusion:** In `verify_cache_dtype`, if a buffer's element size cannot be determined (`elem_size is None`), the function silently returns `True`. Furthermore, the `expected_dtype` argument is completely ignored, allowing 1-byte INT8 (`torch.int8` / `torch.uint8`) buffers to pass as valid FP8 caches without detection.
3. **Hardware Enforcement CPU Bypass on CUDA Hosts:** In `assert_fp8_hardware_support`, passing `device=torch.device("cpu")` on a machine with a CUDA GPU queries GPU 0 and returns `True`, falsely validating CPU execution. Passing string device identifiers (`"cuda:0"`) triggers an unhandled `TypeError`.
4. **Runner Post-Init Config Mutation:** In `vllm_runner.VLLMRunner`, `initialize_engine()` does not re-validate `self.config`, allowing `kv_cache_dtype="auto"` to be injected post-initialization without triggering `SilentFallbackError`.
5. **Software-in-the-Loop Conformance Masking:** In `src/harness/cache_adapter.py`, `Condition B (REAL_FP8)` executes identical code to `Condition Ablation (STORAGE_FP8)`. In `run_conformance.py`, this forces kernel rounding error $\Delta_{kernel}$ to exactly 0.0, hiding the real behavioral gap of Hopper native WGMMA execution.

---

## 2. In-Depth Adversarial Challenges

---

### Challenge 1: Silent Fallback Detection Harness & Tier 1/5 Bypass
**Severity:** **HIGH**

#### 1.1 Direct Behavior on `kv_cache_dtype="auto"` and `kv_cache_dtype="bfloat16"`
When explicitly invoked via `audit_vllm_cache_argument(kv_cache_dtype, strict=True)`:
- `kv_cache_dtype="auto"`: Normalized to `"auto"`. Matches line 207 of `env_inspector.py` and raises `SilentFallbackError`:
  `"SILENT FALLBACK VIOLATION: vLLM kv_cache_dtype is set to 'auto'. vLLM silently resolves 'auto' to bfloat16 without warning."`
- `kv_cache_dtype="bfloat16"`: Normalized to `"bfloat16"`. Matches line 217 of `env_inspector.py` and raises `SilentFallbackError`:
  `"SILENT FALLBACK VIOLATION: kv_cache_dtype is explicitly set to 'bfloat16'. Expected production FP8 cache."`
- Similar checks trap `"bf16"`, `"float16"`, `"fp16"`, and any unauthorized strings.

#### 1.2 Identified Bypasses & Failure Modes:
- **Missing Tier 5 Implementation:**  
  `CAMPAIGN_002_ENVIRONMENT_MANIFEST.md` §8 (Line 168) and `configs/env/environment_spec.yaml` (Line 163) pre-register:
  ```text
  Tier 5 | Kernel GEMM / Profiler | Check FP8 kernel names in trace | KernelFallbackError
  ```
  However, in `src/runtime/env_inspector.py`, `KernelFallbackError` is never defined, and no kernel tracing or profiler hook exists. If vLLM or PyTorch internally falls back to a 16-bit GEMM kernel during PagedAttention execution (e.g., due to unsupported head dimensions or missing Triton compilation targets), this fallback is completely silent.
- **Post-Instantiation Configuration Mutation:**  
  In `src/runtime/vllm_runner.py`:
  ```python
  class VLLMRunner:
      def __init__(self, config: Optional[VLLMRunnerConfig] = None):
          self.config = config or VLLMRunnerConfig()
          validate_runner_config(self.config, strict=True)
          ...
      def initialize_engine(self) -> Any:
          self.preflight_check(strict=True)  # Only checks hardware!
          from vllm import LLM
          self.engine = LLM(..., kv_cache_dtype=self.config.kv_cache_dtype, ...)
  ```
  If an adversary or workflow step creates `runner = VLLMRunner()` and subsequently mutates `runner.config.kv_cache_dtype = "auto"`, `initialize_engine()` does not call `validate_runner_config()`. The vLLM engine initializes with `"auto"` and defaults to BF16 without any exception.
- **Unhandled Non-String Types:**  
  If `kv_cache_dtype` is passed as `None` or a torch dtype object (`torch.bfloat16`), `audit_vllm_cache_argument` calls `kv_cache_dtype.strip()`, triggering an unhandled `AttributeError` rather than a controlled `SilentFallbackError`.

#### Mitigation:
1. Define `KernelFallbackError(RuntimeInspectionError)` in `src/runtime/env_inspector.py`.
2. Implement an execution trace profiler that inspects CUDA kernel symbols (e.g. asserting `fp8` or `e4m3` in the active attention kernel name).
3. In `VLLMRunner.initialize_engine()`, re-execute `validate_runner_config(self.config, strict=True)` immediately before instantiating `vllm.LLM`.
4. Wrap `audit_vllm_cache_argument` input normalization with `str(kv_cache_dtype)`.

---

### Challenge 2: Hardware Compute Capability Enforcement (`assert_fp8_hardware_support`)
**Severity:** **MEDIUM**

#### 2.1 Direct Behavior on Legacy Architectures and CPU
When evaluated via `assert_fp8_hardware_support(strict=True)`:
- **Ampere `sm_80` (A100):** Compute capability is 80. Strictly triggers `HardwareIncompatibilityError` (Line 108: `compute_capability < 89`).
- **Ampere `sm_86` (RTX 3090 / A10G):** Compute capability is 86. Strictly triggers `HardwareIncompatibilityError`.
- **Turing `sm_75` (T4 / RTX 2080):** Compute capability is 75. Strictly triggers `HardwareIncompatibilityError`.
- **Volta `sm_70` (V100):** Compute capability is 70. Strictly triggers `HardwareIncompatibilityError`.
- **Host CPU (when CUDA unavailable):** `torch.cuda.is_available()` returns `False`. Strictly triggers `HardwareIncompatibilityError` (Line 92).

#### 2.2 Identified Bypasses & Failure Modes:
- **`device=torch.device("cpu")` Bypass on CUDA-Enabled Hosts:**  
  In `src/runtime/env_inspector.py` (lines 86-98):
  ```python
  if not torch.cuda.is_available():
      if strict: raise HardwareIncompatibilityError(...)
      return False

  dev_idx = torch.cuda.current_device() if device is None else (
      device if isinstance(device, int) else device.index or 0
  )
  major, minor = torch.cuda.get_device_capability(dev_idx)
  ```
  If the host system has an NVIDIA RTX 4090 (`sm_89`), `torch.cuda.is_available()` is `True`. If a caller passes `device=torch.device("cpu")`, `device.index` is `None`. The fallback `device.index or 0` resolves to `0`! The inspector queries CUDA GPU 0, finds capability 8.9, and returns `True`!  
  *Blast Radius:* Code intending to run on CPU is erroneously certified as running on native FP8 hardware.
- **String Device Argument Crash:**  
  Passing `device="cuda:0"` causes `isinstance(device, int)` to be `False`. Then `device.index or 0` accesses the string method `str.index`, resulting in `dev_idx` being a method reference. PyTorch raises `TypeError: get_device_capability() argument 1 must be int, not builtin_function_or_method`, bypassing the domain exception hierarchy.

#### Mitigation:
In `assert_fp8_hardware_support`:
```python
if isinstance(device, torch.device):
    if device.type != "cuda":
        raise HardwareIncompatibilityError(f"Target device '{device}' is not a CUDA device.")
    dev_idx = device.index or 0
elif isinstance(device, str):
    if not device.startswith("cuda"):
        raise HardwareIncompatibilityError(f"Target device '{device}' is not a CUDA device.")
    d = torch.device(device)
    dev_idx = d.index or 0
```

---

### Challenge 3: Physical Cache Buffer Allocation & Element Size Checking
**Severity:** **HIGH**

#### 3.1 Direct Behavior on Mismatched Sizes
In `verify_cache_dtype(cache_tensor, strict=True)`:
- `element_size() == 2` (BF16 / FP16): Trapped by `is_high_precision` check -> raises `SilentFallbackError`.
- `element_size() == 4` (FP32): Trapped by `is_high_precision` check -> raises `SilentFallbackError`.
- `element_size() == 2` or `4` with non-standard dtype string: Trapped by `elem_size != expected_element_size` -> raises `CacheAllocationError`.
- `element_size() == 1` with FP8 dtype: Successfully passes.

#### 3.2 Identified Bypasses & Failure Modes:
- **Silent Pass on Missing `element_size` (`elem_size is None` Flaw):**  
  In lines 148-184 of `env_inspector.py`:
  ```python
  elem_size = None
  if hasattr(cache_tensor, "element_size") and callable(cache_tensor.element_size):
      elem_size = cache_tensor.element_size()
  elif hasattr(cache_tensor, "itemsize"):
      elem_size = cache_tensor.itemsize
  elif isinstance(cache_tensor, dict) and "element_size" in cache_tensor:
      elem_size = cache_tensor["element_size"]
  ...
  if elem_size is not None and elem_size != expected_element_size:
      raise CacheAllocationError(msg)

  return True
  ```
  If `cache_tensor` is a raw memory address, list, or custom object lacking those exact attributes, `elem_size` remains `None`. Because the rejection condition checks `if elem_size is not None and ...`, this branch is skipped. Unless the dtype string happens to contain a high-precision keyword, **the function quietly returns `True`**!
- **`expected_dtype` Parameter Is Completely Dead Code:**  
  The function signature declares `expected_dtype: Optional[Any] = None`, but this variable is **never referenced or checked anywhere in the function body**.
- **INT8 vs FP8 Quantization Confusion:**  
  Both `torch.int8` and `torch.uint8` have `element_size() == 1`. Because `verify_cache_dtype` only checks for 16/32-bit floating point strings, an INT8 linear quantized cache will pass verification as valid FP8! Linear INT8 quantization has completely different numerical properties from IEEE P3109 non-linear FP8 E4M3FN. Passing INT8 without detection invalidates experimental comparability.

#### Mitigation:
1. In `verify_cache_dtype`, require `elem_size is not None`:
   ```python
   if elem_size is None:
       if strict:
           raise CacheAllocationError("Unable to determine physical element size of cache tensor.")
       return False
   ```
2. Activate `expected_dtype` verification:
   ```python
   if expected_dtype is not None:
       actual_dtype = getattr(cache_tensor, "dtype", None)
       if actual_dtype is not None and str(actual_dtype) != str(expected_dtype):
           if strict:
               raise SilentFallbackError(f"Cache dtype mismatch: expected {expected_dtype}, found {actual_dtype}")
           return False
   ```
3. Explicitly verify that 1-byte tensors belong to an approved FP8 dtype (e.g. `torch.float8_e4m3fn`), rejecting `int8` / `uint8` unless explicitly configured for raw byte storage.

---

### Challenge 4: Microarchitectural Representation (Ada `sm_89` vs Hopper `sm_90`)
**Severity:** **MEDIUM**

#### 4.1 Evaluation of Analytical Representation
The analytical and architectural documentation in `CAMPAIGN_002_RUNTIME_PATH.md` §2 Stage 5, `CAMPAIGN_002_ENVIRONMENT_MANIFEST.md` §2.1, and `configs/env/environment_spec.yaml` is **exceptionally accurate and thorough**:
- **Ada Lovelace (`sm_89`):** Accurately documented as executing SRAM dequantization. The 1-byte FP8 values are stored in HBM (achieving 50% memory bandwidth/footprint savings), loaded into fast on-chip SRAM, unpacked into 16-bit registers, and computed via BF16 Tensor Cores.
- **Hopper (`sm_90`):** Accurately documented as executing native WGMMA (Warpgroup Matrix Multiply and Accumulate) and TMA (Tensor Memory Accelerator). Hopper performs direct FP8 $\times$ FP8 matrix multiplication with FP32 accumulation without unpacking to BF16 in registers.

#### 4.2 Adversarial Gap in Implementation & Conformance Testing:
- **`CacheAdapter` Conflation in Software-in-the-Loop Testing:**  
  In `src/harness/cache_adapter.py` (lines 127-140):
  ```python
  elif self.condition == CacheCondition.REAL_FP8:
      storage = self.layer_storages[layer_idx]
      storage.store(...)
      k_deq, v_deq = storage.retrieve(...)
      k_out = k_deq
      v_out = v_deq
  ```
  `CacheCondition.REAL_FP8` is implemented by calling `FP8KVStorage.store()` and `retrieve()`, which is identical to `CacheCondition.STORAGE_FP8`!  
  In `src/eval/run_conformance.py`:
  - `logits_b` (Real) and `logits_s` (Storage) are produced by identical operations.
  - As a result, `delta_kernel = ||logits_b - logits_s||` is trivially $0.0$.
  - In development/mock mode, the actual GEMM non-associativity of live vLLM PagedAttention kernels or Hopper WGMMA kernels is not simulated; the test only measures storage noise ($\Delta_{storage}$).
- **Significance for Research:**  
  On Ada `sm_89`, SRAM dequantization means that `STORAGE_FP8` is an exact mathematical model of what the hardware kernel does. On Hopper `sm_90`, however, native WGMMA introduces independent rounding from direct 8-bit dot products. The team must not confuse a $0.0$ $\Delta_{kernel}$ in `run_conformance.py` on development hosts with real hardware conformance on Hopper!

---

## 3. Stress Test Results Matrix

| Scenario / Test Case | Expected Behavior | Actual Behavior | Result | Finding Reference |
|---|---|---|:---:|---|
| Pass `kv_cache_dtype="auto"` to `audit_vllm_cache_argument` | Raise `SilentFallbackError` | Raises `SilentFallbackError` | **PASS** | Section 2, Challenge 1 |
| Pass `kv_cache_dtype="bfloat16"` to `audit_vllm_cache_argument` | Raise `SilentFallbackError` | Raises `SilentFallbackError` | **PASS** | Section 2, Challenge 1 |
| Pass `kv_cache_dtype="fp8"` or `"fp8_e4m3fn"` | Return `True` | Returns `True` | **PASS** | Section 2, Challenge 1 |
| Pass `kv_cache_dtype=None` | Raise `SilentFallbackError` | Raises unhandled `AttributeError` | **FAIL** | Challenge 1.2 |
| Post-init mutation of `VLLMRunner.config.kv_cache_dtype` | Re-validate before `LLM()` init | Bypasses validation, initializes vLLM | **FAIL** | Challenge 1.2 |
| Check Ampere `sm_80` (A100) via `assert_fp8_hardware_support` | Raise `HardwareIncompatibilityError` | Raises `HardwareIncompatibilityError` | **PASS** | Section 2, Challenge 2 |
| Check Turing `sm_75` (T4) via `assert_fp8_hardware_support` | Raise `HardwareIncompatibilityError` | Raises `HardwareIncompatibilityError` | **PASS** | Section 2, Challenge 2 |
| Check Volta `sm_70` (V100) via `assert_fp8_hardware_support` | Raise `HardwareIncompatibilityError` | Raises `HardwareIncompatibilityError` | **PASS** | Section 2, Challenge 2 |
| Check CPU when CUDA unavailable | Raise `HardwareIncompatibilityError` | Raises `HardwareIncompatibilityError` | **PASS** | Section 2, Challenge 2 |
| Check `device=torch.device("cpu")` on CUDA host | Raise `HardwareIncompatibilityError` | Queries GPU 0, returns `True` | **FAIL** | Challenge 2.2 |
| Pass `device="cuda:0"` (string) | Parse device index cleanly | Raises unhandled `TypeError` | **FAIL** | Challenge 2.2 |
| Check buffer with `element_size() == 2` (BF16) | Raise `SilentFallbackError` | Raises `SilentFallbackError` | **PASS** | Section 2, Challenge 3 |
| Check buffer with `element_size() == 2` (custom dtype) | Raise `CacheAllocationError` | Raises `CacheAllocationError` | **PASS** | Section 2, Challenge 3 |
| Check buffer with `elem_size is None` | Raise `CacheAllocationError` | Silently returns `True` | **FAIL** | Challenge 3.2 |
| Pass `expected_dtype=torch.float8_e4m3fn` with `torch.int8` buffer | Reject INT8 buffer | Silently returns `True` (`expected_dtype` ignored) | **FAIL** | Challenge 3.2 |
| Assert Tier 5 profiler trap (`KernelFallbackError`) | Trap non-FP8 CUDA kernel execution | Tier 5 missing from code; error not defined | **FAIL** | Challenge 1.2 |
| Microarchitecture trace for `sm_89` vs `sm_90` | Accurately distinguish SRAM dequant vs WGMMA | Accurately distinguished in docs & tracer | **PASS** | Section 2, Challenge 4 |

---

## 4. Actionable Remediation Plan (Required Changes)

To resolve the identified vulnerabilities before approving Phase 0 runtime locking:

1. **Fix `src/runtime/env_inspector.py`:**
   - **Define `KernelFallbackError`:** Add `class KernelFallbackError(RuntimeInspectionError): pass`.
   - **Patch `assert_fp8_hardware_support` Device Handling:** Inspect `device.type == "cuda"` when `isinstance(device, torch.device)`. Parse strings via `torch.device(device)`. Strictly reject CPU requests regardless of host CUDA availability.
   - **Patch `verify_cache_dtype` Null Check & Dead Code:**
     - Raise `CacheAllocationError` if `elem_size is None`.
     - Actively compare `expected_dtype` when provided.
     - Reject `torch.int8` / `torch.uint8` when FP8 is expected.
   - **Patch `audit_vllm_cache_argument` Input Sanitization:** Safely handle non-string or `None` inputs with `str(kv_cache_dtype or "").strip().lower()`.

2. **Fix `src/runtime/vllm_runner.py`:**
   - In `VLLMRunner.initialize_engine()`, call `validate_runner_config(self.config, strict=True)` to prevent post-init configuration poisoning.

3. **Clarify Conformance Documentation:**
   - Explicitly note in `CAMPAIGN_002_RUNTIME_PATH.md` and `CAMPAIGN_002_PROXY_CONFORMANCE.md` that in simulated/mock mode (`run_conformance.py`), Condition B uses `FP8KVStorage` (SRAM dequantization), and that full kernel non-associativity measurement ($\Delta_{kernel} > 0$) requires physical execution via `vllm_runner.py` on Linux GPU targets.
