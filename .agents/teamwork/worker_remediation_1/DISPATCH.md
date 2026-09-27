## 2026-09-27T16:31:38Z
You are worker_remediation_1. Your working directory is:
c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\worker_remediation_1\

You MUST read the authoritative original request first:
c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\ORIGINAL_REQUEST.md

Also read the reviewer and challenger reports:
- c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\challenger_c002_2\handoff.md
- c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\challenger_c002_1\handoff.md
- c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\reviewer_c002_1\handoff.md

DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A teamwork_preview_auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

Strict Non-Negotiables:
- Zero backdoor training executed in Campaign 002.
- Zero harmful behavior targets evaluated.
- Zero novelty claims derived from this campaign.
- Acceptance thresholds remain frozen.
- Any silent fallback from hardware FP8 to simulated/software FP8 or BF16 detected and flagged as a test failure.

Your Mission — Implement Remediation for Iteration 2:
1. `src/runtime/env_inspector.py`:
   - Add `KernelFallbackError` (subclass of `SilentFallbackError`) for Tier 5 fallback traps.
   - In `assert_fp8_hardware_support(device=None, strict=True)`:
     - Parse device string or `torch.device`. If `device` is CPU (`device.type != "cuda"`), raise `HardwareIncompatibilityError("Device type 'cpu' does not support native FP8 Tensor Cores")`.
     - Properly handle string device specifications like `"cuda:0"` or `"cpu"`.
   - In `verify_cache_dtype(cache_tensor, expected_dtype=None, strict=True)`:
     - If `elem_size is None`: raise `CacheAllocationError("Unable to verify element size of cache tensor")`.
     - If `expected_dtype is not None`: verify `cache_tensor.dtype == expected_dtype`. If an INT8/UINT8 tensor is supplied while expecting FP8, raise `SilentFallbackError("INT8 buffer cannot substitute for FP8 cache")`.
2. `src/runtime/vllm_runner.py`:
   - In `initialize_engine()`: call `validate_runner_config(self.config)` before initializing the engine to prevent post-init mutation bypasses.
3. `src/compression/storage_fp8.py`:
   - Fix the fallback bug in `FP8KVStorage.store`: replace `.view(torch.int32).to(torch.uint8)` with proper integer quantization: `torch.clamp(x_clamped.round(), -128, 127).to(torch.int8)`. In `retrieve()`, cast properly to `self.storage_dtype`.
4. `scripts/run_adversarial_audit.py`:
   - In `run_context_length_scaling()`: remove `min(seq_len, 256)`! Generate `torch.randint(10, 900, (1, seq_len), device=dev)` so that sequence lengths 128, 512, and 2048 are actually evaluated.
5. `tests/test_kernel_fallback.py`:
   - Add unit tests verifying `KernelFallbackError`, testing CPU device rejection on CUDA hosts, testing uninspectable element size rejection, testing `expected_dtype` INT8 rejection, and testing `initialize_engine` validation.
6. Update `research/campaigns/campaign_002/CAMPAIGN_002_DECISION_MEMO.md`:
   - Update verdict to **`CONDITIONAL PASS`**!
   - Explicitly document the conditions:
     a) Conformance mathematically and empirically holds for the candidate PyTorch STE proxy ($T_{proxy}$) across all 9 Gate UG2 metrics on clean model $\theta_c$.
     b) Dynamic per-head scaling or calibrated static scaling is strictly required for WP3 training to prevent outlier activation clipping and underflow.
     c) Physical hardware execution of vLLM Triton PagedAttention kernels on a dedicated Linux host (Ubuntu 22.04 LTS, Ada `sm_89` / Hopper `sm_90`) is pre-registered as a mandatory gate check prior to claiming production deployment transfer.
7. Update canonical research memory in `researchMemory/agentMemory/`:
   - `CURRENT_STATE.md`: Update verdict to `CONDITIONAL PASS` with explicit conditions.
   - `DECISION_LOG.md`: Update Decision D18 to `CONDITIONAL PASS` authorizing WP2/WP3 under the specified conditions.
   - `EXPERIMENT_REGISTRY.md`: Update EXP-002 status to `CONDITIONAL PASS`.
   - `FINDINGS.md`: Update findings F-002-1 through F-002-5 noting the `CONDITIONAL PASS` scope and conditions.
   - `IMPLEMENTATION_STATE.md` and `CHANGELOG.md`: Reconcile test counts (exact count of unit tests discovered by `unittest`) and log the remediation.
8. Run all tests to ensure 100% clean passes:
   `python -m unittest discover -s tests -p "test_*.py"`
   `python scripts/run_wp0_manifest.py`
   `python scripts/run_adversarial_audit.py`
9. Write `handoff.md` in your working directory and notify orchestrator 9f5a0de9-5aa2-43c1-a639-a9f3747adaf6.
