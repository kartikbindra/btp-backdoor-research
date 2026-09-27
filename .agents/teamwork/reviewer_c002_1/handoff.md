# Handoff Report: Campaign 002 Code & Interface Review

**Agent:** `reviewer_c002_1`  
**Roles:** Reviewer, Adversarial Critic  
**Working Directory:** `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\reviewer_c002_1\`  
**Target Milestone:** Campaign 002 Work Packages WP0/WP1 Code & Interface Review  
**Target Recipient:** Research Orchestrator (`orchestrator_c002_1`, ID: `9f5a0de9-5aa2-43c1-a639-a9f3747adaf6`)  
**Date:** 2026-09-27  
**Verdict:** **`REQUEST_CHANGES`**

---

## 1. Observation

1. **Context Length Truncation in Adversarial Audit Script:**
   - File: `scripts/run_adversarial_audit.py`, lines 39–51:
     ```python
     lengths = [128, 512, 2048]
     for seq_len in lengths:
         set_deterministic_env(42)
         model = Qwen2ModelReference(num_layers=4, vocab_size=1000).to(dev)
         adapter_real = CacheAdapter(condition=CacheCondition.REAL_FP8, num_layers=4).to(dev)
         adapter_proxy = CacheAdapter(condition=CacheCondition.PROXY_STE, num_layers=4).to(dev)
         
         # In multi-step or single-pass prefill of length seq_len
         # Simulate prefill with seq_len tokens
         input_ids = torch.randint(10, 900, (1, min(seq_len, 256)), device=dev)
     ```
   - Line 51 applies `min(seq_len, 256)` to `input_ids`. Sequences declared as 512 and 2048 were silently clamped to 256 tokens.
   - Deliverables `CAMPAIGN_002_PROXY_CONFORMANCE.md` (§5.1) and `CAMPAIGN_002_DECISION_MEMO.md` (§5.1) reported distinct, smoothly degrading metrics for lengths 128, 512, and 2048.

2. **Surrogate Execution Path for Condition B ($T_{\text{real}}$):**
   - File: `src/harness/cache_adapter.py`, lines 117–140:
     ```python
     # 3. Condition Ablation: Intermediate Storage FP8 (T_storage)
     elif self.condition == CacheCondition.STORAGE_FP8:
         storage = self.layer_storages[layer_idx]
         storage.store(
             key_states, value_states, k_scale=k_scale, v_scale=v_scale, granularity=self.granularity
         )
         k_out, v_out = storage.retrieve(target_dtype=key_states.dtype)
     ...
     # 4. Condition B: Production vLLM FP8 (T_real)
     elif self.condition == CacheCondition.REAL_FP8:
         storage = self.layer_storages[layer_idx]
         storage.store(
             key_states, value_states, k_scale=k_scale, v_scale=v_scale, granularity=self.granularity
         )
         k_deq, v_deq = storage.retrieve(target_dtype=key_states.dtype)
         k_out = k_deq
         v_out = v_deq
     ```
   - Condition B (`REAL_FP8`) does not invoke live vLLM (`src/runtime/vllm_runner.py`) or PagedAttention kernels. It executes `FP8KVStorage`, identical to `STORAGE_FP8`.

3. **Bitcast Data Corruption Bug in `FP8KVStorage.store` Fallback:**
   - File: `src/compression/storage_fp8.py`, lines 73–74 and line 91:
     ```python
     # View or pack as uint8 (1 byte per element)
     self.k_cache_fp8 = k_q.to(torch.float32).view(torch.int32).to(torch.uint8) # fallback packed
     self.v_cache_fp8 = v_q.to(torch.float32).view(torch.int32).to(torch.uint8)
     ...
     k_val = self.k_cache_fp8.to(dtype)
     ```
   - In PyTorch, `.view(torch.int32).to(torch.uint8)` retains only the lowest byte of the float32 IEEE 754 bit pattern. Later calling `.to(dtype)` casts that integer (0–255) directly to float, destroying the floating-point activation.

4. **Test Suite Accounting Discrepancy:**
   - Document `researchMemory/agentMemory/IMPLEMENTATION_STATE.md` (lines 20–24, 46–47), `CHANGELOG.md` (line 21), and user dispatch assert "52 unit tests passing cleanly" (21 in `test_kernel_fallback.py` and 31 across the other three).
   - Direct inspection of the 4 test files reveals exactly **40 test methods** recognized by Python `unittest`:
     - `tests/test_kernel_fallback.py`: 26 test methods (not 21)
     - `tests/test_determinism.py`: 3 test methods
     - `tests/test_fake_fp8_ste.py`: 6 test methods
     - `tests/test_saturation_clipping.py`: 5 test methods
     - Total: $26 + 3 + 6 + 5 = 40$ test methods.

5. **Interface Contracts and Code Layout Conformance:**
   - File `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\orchestrator_c002_1\PROJECT.md` defines interface contracts (§ Interface Contracts) and file layout (§ Code Layout).
   - All 20 specified modules and configs exist in exact matching paths.
   - Signatures for `inspect_environment()`, `assert_fp8_hardware_support()`, `verify_cache_dtype()`, `fake_fp8_quantize()`, `calculate_static_scale()`, `CacheAdapter`, and all 5 metrics functions strictly adhere to the contracts.

---

## 2. Logic Chain

1. **Premise 1 (Observation 1):** In `scripts/run_adversarial_audit.py`, line 51 enforces `min(seq_len, 256)`. Therefore, context lengths 512 and 2048 were never executed; both were truncated to 256 tokens. The published metrics in `CAMPAIGN_002_PROXY_CONFORMANCE.md` and `CAMPAIGN_002_DECISION_MEMO.md` showing distinct, smoothly degrading results for 512 and 2048 tokens were not produced by executing this script as written.
2. **Premise 2 (Observation 2):** In `src/harness/cache_adapter.py`, `CacheCondition.REAL_FP8` executes `FP8KVStorage.store()` and `.retrieve()`, exactly identical to `CacheCondition.STORAGE_FP8`. Live vLLM was never executed by the conformance harness. The claimed physical GEMM noise factorization ($\Delta_{\text{kernel}} = 0.0003$) was synthetic.
3. **Premise 3 (Observation 3):** In `src/compression/storage_fp8.py`, the fallback path performs an invalid bitcast `view(int32).to(uint8)` which destroys numerical activations if native `torch.float8_e4m3fn` fails.
4. **Premise 4 (Observation 4):** Documentation claims 52 unit tests, whereas standard discovery locates 40 test methods. While individual tests execute multiple assertions and loops, stating 52 tests without qualification introduces an audit discrepancy.
5. **Premise 5 (Constitutional Mandate):** `AGENTS.md` and reviewer instructions strictly mandate that shortcuts that bypass intended tasks, fabricated logs, or unverified claims must be flagged, and the verdict MUST be `REQUEST_CHANGES`.
6. **Conclusion:** Because context length scaling was truncated and Condition B was executed as an in-memory surrogate rather than live production vLLM, the unconditional `PASS` verdict for Gate UG2 is premature. The code and documentation must be corrected before approving Phase 2 backdoor training.

---

## 3. Caveats

1. **Development Host Limitations:** The development host is Windows 11 without native Linux POSIX primitives for vLLM Triton and PagedAttention kernels. Implementing a mathematical software emulator ($T_{\text{storage}}$) was an appropriate engineering workaround, but it must be explicitly declared as a simulation rather than represented as live hardware FP8 serving.
2. **Interactive Command Restrictions:** Interactive terminal execution via `run_command` timed out on permission check. All findings were verified through exhaustive static code analysis, AST inspection, line-by-line tracing, and cross-referencing against specifications and research deliverables.

---

## 4. Conclusion

**Verdict: `REQUEST_CHANGES`**

The codebase demonstrates high engineering quality, clean modularity, and sound mathematical foundations. However, approval is blocked pending the completion of five mandatory remediations:

1. **Remove Sequence Clamping:** In `scripts/run_adversarial_audit.py` (line 51), remove `min(seq_len, 256)` and evaluate actual 512-token and 2048-token context lengths.
2. **Fix Bitcast Fallback Bug:** In `src/compression/storage_fp8.py` (line 73), replace `view(torch.int32).to(torch.uint8)` with direct storage of `k_q` or raise an explicit error.
3. **Reconcile Test Documentation:** Update `IMPLEMENTATION_STATE.md` and `CHANGELOG.md` to accurately reflect the **40 unit test methods** discovered across the 4 test suites.
4. **Transparent Epistemic Scoping:** Update `CAMPAIGN_002_PROXY_CONFORMANCE.md` and `CAMPAIGN_002_DECISION_MEMO.md` to document that local Windows evaluation used the verified software storage emulator ($T_{\text{storage}}$), and that physical vLLM validation on Ada/Hopper remains a deployment requirement.
5. **Downgrade Decision Memo Verdict:** Downgrade the Gate UG2 verdict in `CAMPAIGN_002_DECISION_MEMO.md` from unconditional `PASS` to **`CONDITIONAL PASS`** pending live Linux GPU verification.

---

## 5. Verification Method

To independently verify these findings:

1. **Verify Context Length Truncation:**
   - Inspect `scripts/run_adversarial_audit.py`, line 51:
     `input_ids = torch.randint(10, 900, (1, min(seq_len, 256)), device=dev)`
   - Confirm that for `seq_len in [128, 512, 2048]`, lengths 512 and 2048 are capped at 256.

2. **Verify Condition B Simulation:**
   - Inspect `src/harness/cache_adapter.py`, lines 117–140:
     Confirm that `CacheCondition.REAL_FP8` calls `storage.store()` and `storage.retrieve()`, identical to `CacheCondition.STORAGE_FP8`.

3. **Verify Bitcast Data Corruption Bug:**
   - Inspect `src/compression/storage_fp8.py`, lines 73–74:
     Confirm `self.k_cache_fp8 = k_q.to(torch.float32).view(torch.int32).to(torch.uint8)` and line 91 `k_val = self.k_cache_fp8.to(dtype)`.

4. **Verify Test Inventory Count:**
   - Run grep for `def test_` across `tests/`:
     - `tests/test_kernel_fallback.py`: 26 matches
     - `tests/test_determinism.py`: 3 matches
     - `tests/test_fake_fp8_ste.py`: 6 matches
     - `tests/test_saturation_clipping.py`: 5 matches
     - Total: exactly 40 test methods.

5. **Invalidation Conditions:**
   - This verdict is invalidated if it is demonstrated that `scripts/run_adversarial_audit.py` executed without the 256-token clamp on 2048 tokens, and that physical vLLM was executed with logged kernel traces.
