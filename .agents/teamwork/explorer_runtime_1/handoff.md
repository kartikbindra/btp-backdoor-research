# Handoff Report: System, Hardware, and Runtime Capability Audit
**Agent:** `explorer_runtime_1`  
**Milestone:** WP0/WP1 Runtime Gate Environment & Capability Survey  
**Working Directory:** `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\explorer_runtime_1\`  
**Target Recipient:** Orchestrator (`orchestrator_c002_1`, ID: `9f5a0de9-5aa2-43c1-a639-a9f3747adaf6`)  
**Date:** 2026-09-27  

---

## 1. Observation

1. **Local Host State & Process Execution Timeouts:**
   - Attempting to query hardware and python environment via diagnostic tool commands returned verbatim:
     ```text
     permission check failed for command "nvidia-smi": Permission prompt for action 'command' on target 'nvidia-smi' timed out waiting for user response. The user was not able to provide permission on time. You should proceed as much as possible without access to this resource.
     ```
     ```text
     permission check failed for command "python -c \"import sys, json; res = {'python': sys.version}; print(json.dumps(res))\"": Permission prompt for action 'command' on target 'python ...' timed out waiting for user response.
     ```
   - The user operating system is Windows 11 (`user_information`: `The USER's OS version is windows.`, workspace path: `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research`).
   - Access to user root directory outside workspace (`C:\Users\Kartik`) is guarded and timed out awaiting interactive prompt response.

2. **Pre-Existing Architecture & Environment Records in Repository:**
   - `research/campaigns/campaign_001/DECISION_MEMO.md` (Lines 406–407) explicitly records:
     > `- Hardware Platform: NVIDIA GPU with native hardware FP8 Tensor Core support (NVIDIA Ada Lovelace RTX 4090 / L40S, or NVIDIA Hopper H100 / A100-SXM4).`  
     > `Crucial Operating Constraint: The current local environment is Windows/macOS. Physical vLLM FP8 Triton/CUDA kernels require a dedicated Linux host (Ubuntu 22.04 LTS) with CUDA 12.4+ and driver version >= 550.54.14. Phase 0/1 implementation must be deployed to a Linux GPU instance.`
   - `CONSOLIDATED_RESEARCH_PLAN.md` (Line 539) explicitly records:
     > `Important environment constraint: the current project workspace is on macOS [or Windows]. The real vLLM CUDA/FP8 treatment cannot be established on this host. Before implementation, secure a Linux system with a supported NVIDIA GPU and validate that its hardware/backend executes the intended FP8 KV path. Do not assume a generic 24-GB GPU provides the same FP8 arithmetic path.`
   - `research/campaigns/campaign_001/teamwork_state/challenger_c001/handoff.md` (Line 119) explicitly notes:
     > `Local evaluation was conducted on Windows. As correctly identified in Section 11.2 of the Decision Memo, executing the physical vLLM FP8 runtime kernels ($T_{real}$) requires deploying to a dedicated Linux host (Ubuntu 22.04 LTS, CUDA 12.4+, NVIDIA driver >= 550.54.14) with an NVIDIA Ada Lovelace or Hopper GPU.`

3. **vLLM FP8 Implementation & Hardware Constraints:**
   - vLLM lacks native Windows wheel releases. Execution on Windows requires WSL2 (Ubuntu 22.04) or Docker Desktop (WSL2 backend).
   - In vLLM source code, passing `--kv-cache-dtype fp8` on NVIDIA architectures with compute capability $< 8.9$ (such as Ampere `sm_80` / `sm_86`, e.g. A100 or RTX 3090) raises an immediate hard assertion crash:
     `AssertionError: fp8e4nv data type is not supported on CUDA arch < 89`.
   - On Ada Lovelace (`sm_89`, e.g. RTX 4090, L40S), vLLM supports FP8 storage in memory (`fp8_e4m3fn`), but FlashAttention-3 is not supported (FA3 is strictly Hopper `sm_90` only). Attention execution on Ada uses FlashAttention-2 or PagedAttention with SRAM dequantization.
   - On Hopper (`sm_90`, e.g. H100), vLLM supports full native FP8 compute and storage via FlashAttention-3 or FlashInfer utilizing WGMMA and TMA instructions.

4. **PyTorch Native FP8 Support (`torch.float8_e4m3fn`):**
   - PyTorch 2.1+ supports `torch.float8_e4m3fn` storage (1 byte per element) on CUDA devices.
   - Standard `torch.matmul` does not support `Float8_e4m3fn` (`RuntimeError: "addmm_cuda" not implemented for 'Float8_e4m3fn'`).
   - Hardware-accelerated matrix multiplication requires `torch._scaled_mm`, which natively requires Compute Capability $\ge 9.0$ (Hopper) in standard wheels; on Ada (`sm_89`), standard wheels raise `RuntimeError: torch._scaled_mm is only supported on devices with compute capability >= 9.0` unless compiled from source with `sm_89` targets.
   - Candidate proxy ($T_{proxy}$, PyTorch STE) fake-quantizes keys/values via a round-trip cast (`x.to(torch.float8_e4m3fn).to(x.dtype)`) and computes attention in BF16 via standard `torch.nn.functional.scaled_dot_product_attention` (SDPA), making $T_{proxy}$ executable on any CUDA device without requiring hardware FP8 Tensor Cores.

---

## 2. Logic Chain

1. **Premise 1 (Host Limitations):** The current development machine is running Windows 11 with interactive command execution timing out when unattended. Furthermore, official vLLM does not run natively on bare Windows without WSL2, and bare Windows cannot execute vLLM's Triton PagedAttention kernels.
2. **Premise 2 (Hardware Gating):** vLLM FP8 KV cache (`kv_cache_dtype="fp8"`) requires Compute Capability $\ge 8.9$. On Ampere (`sm_80`, A100 / RTX 3090) it hard-crashes. On Ada Lovelace (`sm_89`), it provides storage FP8 with fused SRAM dequantization. On Hopper (`sm_90`), it provides full end-to-end compute and storage FP8.
3. **Premise 3 (Proxy Independence):** The candidate software proxy ($T_{proxy}$, PyTorch STE) is mathematically decoupled from hardware Tensor Cores because it performs fake quantization (clamping and discrete 8-bit rounding) followed by standard BF16 SDPA computation.
4. **Premise 4 (Factorization of Noise Sources):** The discrepancy between $T_{real}$ (Condition B) and $T_{proxy}$ (Condition C) consists of two distinct components: storage quantization error ($\Delta_{storage}$) and kernel GEMM non-associativity reduction differences ($\Delta_{kernel}$).
5. **Conclusion:** Therefore, the evaluation harness must:
   - Execute Condition B ($T_{real}$) on a dedicated Linux host (or WSL2) with an Ada Lovelace (`sm_89`) or Hopper (`sm_90`) GPU.
   - Enforce a 5-tier silent fallback detection harness asserting exact cache byte sizes (1 byte/element), compute capability $\ge 8.9$, and explicit backend flags.
   - Employ a 4-cell comparison matrix (incorporating BF16 vLLM) to factor storage quantization noise from kernel non-associativity effects.

---

## 3. Caveats

- **No Live Local Terminal Execution:** Because subagent shell commands timed out awaiting interactive user confirmation, local GPU model and driver version on this physical Windows desktop were not queried via live `nvidia-smi`. However, the deployment architecture is already formally defined and constrained by Campaign 001 canonical memory (requiring a Linux host with Ada/Hopper).
- **vLLM Scaling Parameter File:** If `quantization_param_path` is not provided to vLLM, it defaults to a scale of `1.0`. For models like `Qwen2.5-1.5B-Instruct` where key projections have large dynamic ranges, an uncalibrated scale of `1.0` will induce severe saturation/clipping. Calibration via `llm-compressor` on a representative sample is strongly recommended as a sub-condition.
- **FlashInfer SM89 Support Nuances:** While FlashInfer supports FP8 on Hopper, Ada Lovelace (`sm_89`) support in specific vLLM releases has varied across version tags. FlashAttention-2 / PagedAttention v2 remains the most stable backend for Ada Lovelace.

---

## 4. Conclusion

1. **Hardware & Environment Specification:**
   - **Target Host:** Linux Ubuntu 22.04 LTS (via dedicated instance or WSL2).
   - **Target GPU:** NVIDIA Ada Lovelace (RTX 4090 / L40S, `sm_89`) or NVIDIA Hopper (H100, `sm_90`) with $\ge 24\text{ GB}$ VRAM.
   - **Target Software Stack:** Python 3.11.9, PyTorch 2.4.0+cu124, CUDA 12.4, vLLM 0.6.0, Transformers 4.45.1, Flash-Attn 2.6.3, llm-compressor 0.1.0.
2. **vLLM & PyTorch FP8 Capability:**
   - vLLM natively supports `kv_cache_dtype="fp8"` (format `fp8_e4m3fn`) on `sm_89` and `sm_90`. Older GPUs (`sm_80` and below) are hard-rejected.
   - PyTorch supports `torch.float8_e4m3fn` as a storage and casting type, which is sufficient for building the Straight-Through Estimator (STE) training proxy ($T_{proxy}$).
3. **Silent Fallback Elimination:**
   - A 5-tier verification harness is mandated: static CLI/API assertions, physical cache memory allocation tracking (asserting 1 byte/element), CUDA kernel name inspection in PyTorch Profiler, Inductor fallback traps, and throughput sanity checks.
4. **Execution Strategy for R1, R2, R3:**
   - **R1:** Freeze environment manifest (`CAMPAIGN_002_ENVIRONMENT_MANIFEST.md`) and runtime execution path (`CAMPAIGN_002_RUNTIME_PATH.md`).
   - **R2:** Implement single-sequence (`batch_size=1`), greedy (`temperature=0.0`), deterministic harness comparing Conditions A (BF16), B (vLLM FP8), and C (PyTorch STE proxy), with BF16 vLLM used to de-couple kernel rounding noise from quantization noise.
   - **R3:** Freeze Gate UG2 thresholds (NRMSE $\le 0.05$, Cosine $\ge 0.995$, Spearman rank $\rho \ge 0.85$, token match $\ge 90\%$) before running confirmatory evaluation.

---

## 5. Verification Method

To independently verify the environment and execution feasibility:

1. **Inspect Survey Report:**
   Read `survey_runtime_report.md` in this directory:
   `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\explorer_runtime_1\survey_runtime_report.md`.
2. **Execute Python Fallback Verification Trap on Target Linux Host:**
   Run:
   ```bash
   python -c "
   import torch
   major, minor = torch.cuda.get_device_capability()
   cap = major * 10 + minor
   print(f'Device: {torch.cuda.get_device_name()}, SM: {cap}')
   assert cap >= 89, f'CUDA arch sm_{cap} < sm_89! FP8 unsupported.'
   t = torch.randn(10, 10, dtype=torch.bfloat16, device='cuda')
   q = t.to(torch.float8_e4m3fn)
   assert q.element_size() == 1, 'FP8 element size mismatch!'
   print('Verification PASS: Environment supports native FP8 storage.')
   "
   ```
3. **Verify vLLM Cache Allocation on Target Host:**
   Run:
   ```bash
   python -c "
   from vllm import LLM
   llm = LLM(model='Qwen/Qwen2.5-1.5B-Instruct', kv_cache_dtype='fp8_e4m3', enforce_eager=True)
   print('vLLM FP8 Engine Initialized Successfully.')
   "
   ```
4. **Invalidation Conditions:**
   - If the host GPU is an Ampere card (`sm_80`), vLLM FP8 will fail immediately.
   - If `kv_cache_dtype="auto"` is set, vLLM will silently fall back to BF16 (violating non-negotiable criteria).
   - If the environment lacks WSL2 or Linux, native vLLM PagedAttention kernels cannot compile or execute.
