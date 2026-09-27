# BRIEFING — 2026-09-27T16:29:00Z

## Mission
Empirically challenge and stress-test the runtime path, hardware locking, and 5-tier silent fallback detection system for Campaign 002 FP8 KV-cache execution.

## 🔒 My Identity
- Archetype: challenger (empirical challenger)
- Roles: critic, specialist
- Working directory: c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\challenger_c002_2\
- Original parent: 9f5a0de9-5aa2-43c1-a639-a9f3747adaf6
- Milestone: Campaign 002 Phase 0 Runtime Verification
- Instance: 1 of 1

## 🔒 Key Constraints
- Review-only — do NOT modify implementation code
- Run verification code directly — empirical verification required
- Do not trust unverified claims
- Keep `.agents/teamwork/` metadata only (no code/tests placed in metadata folders)

## Current Parent
- Conversation ID: 9f5a0de9-5aa2-43c1-a639-a9f3747adaf6
- Updated: not yet

## Review Scope
- **Files to review**: 
  - `src/runtime/` (`src/runtime/env_inspector.py`, `src/runtime/runtime_tracer.py`, `src/runtime/vllm_runner.py`)
  - `tests/test_kernel_fallback.py`
  - `research/campaigns/campaign_002/CAMPAIGN_002_ENVIRONMENT_MANIFEST.md`
  - `research/campaigns/campaign_002/CAMPAIGN_002_RUNTIME_PATH.md`
  - `configs/env/environment_spec.yaml`
  - `src/harness/cache_adapter.py`
  - `src/eval/run_conformance.py`
- **Review criteria**:
  1. 5-tier silent fallback detection harness bypass vulnerability (e.g. `kv_cache_dtype="auto"`, `"bfloat16"`).
  2. Compute capability enforcement (`assert_fp8_hardware_support` rejecting Ampere sm_80, Turing sm_75, Volta sm_70, CPU).
  3. Cache allocation element size checking (`element_size() != 1`).
  4. Microarchitectural distinction between Ada sm_89 (SRAM dequantization) and Hopper sm_90 (native WGMMA).

## Key Decisions Made
- Conducted exhaustive code inspection and control-flow tracing across all runtime modules.
- Delivered verdict: `REQUEST_CHANGES` based on 5 identified vulnerabilities and contract omissions.

## Artifact Index
- `challenge_fallback_report.md` — Comprehensive adversarial challenge report detailing findings, stress test matrix, and remediation plan.
- `handoff.md` — 5-component handoff report with explicit `REQUEST_CHANGES` verdict.
- `progress.md` — Liveness and step tracking.
- `DISPATCH.md` — Original task dispatch record.

## Attack Surface
- **Hypotheses tested**:
  - `audit_vllm_cache_argument` traps `"auto"` and `"bfloat16"` (Confirmed: PASS on direct call, FAIL on post-init runner mutation).
  - Legacy GPUs (`sm_80`, `sm_75`, `sm_70`) are rejected (Confirmed: PASS).
  - CPU is rejected (Confirmed: PASS when CUDA unavailable; FAIL when `device=torch.device("cpu")` on CUDA host).
  - Buffer element size != 1 is rejected (Confirmed: PASS when size present; FAIL when `elem_size is None` or INT8 passed).
  - Tier 5 profiler exists (Confirmed: FAIL, Tier 5 completely unwritten).
- **Vulnerabilities found**:
  1. Tier 5 missing (`KernelFallbackError` not defined; no profiler).
  2. Null element size bypass in `verify_cache_dtype`.
  3. Ignored `expected_dtype` argument in `verify_cache_dtype`.
  4. INT8 linear quantized cache passes as FP8.
  5. CPU device bypass on CUDA-enabled hosts.
  6. String device parameter crash (`TypeError`).
  7. Post-init mutation bypass in `VLLMRunner`.
  8. `CacheAdapter` conflates `REAL_FP8` with `STORAGE_FP8`, collapsing $\Delta_{kernel}$ to 0 in dev mode.
- **Untested angles**:
  - Dynamic WGMMA register profiling on physical Hopper SXM5 hardware (requires Linux target).

## Loaded Skills
- None specified.
