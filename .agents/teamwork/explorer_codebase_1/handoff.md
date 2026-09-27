# Handoff Report — Codebase Survey & Gap Analysis
## Campaign 002: Work Package WP0/WP1 Runtime Gate

**Author:** `explorer_codebase_1`  
**Date:** 2026-09-27  
**Working Directory:** `.agents/teamwork/explorer_codebase_1/`  
**Target Milestone:** Milestone 0 (Survey & Reconnaissance) -> Milestone 1 (Implementation Dispatch)  

---

## 1. Observation

### 1.1 Physical Filesystem Inspection
- Direct inspection of the repository root `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research` via file listing and recursive search:
  - Top-level directories present: `.agents/`, `.git/`, `research/`, `researchMemory/`.
  - Directories `src/`, `tests/`, `experiments/`, `configs/`, `scripts/` are **completely absent** from the repository filesystem.
  - Non-Markdown/non-Docx file count: **0**. A recursive search for extensions `py, ipynb, sh, bash, ps1, json, yaml, yml, toml, cu, cpp, c, h` returned:
    ```json
    Found 0 results
    ```
  - Total tracked/workspace files: **76 files**, consisting of 73 `.md` files and 3 `.docx` files.
  - Exact corroboration from `researchMemory/agentMemory/IMPLEMENTATION_STATE.md` (lines 10–13):
    > "- Source Code Files Present: 0  
    > - Configuration / Script Files Present: 0  
    > - Existing Directories: Only researchMemory/ (which contains historical memory dumps...)"

### 1.2 Model & Checkpoint Availability
- Pinned Model: `Qwen/Qwen2.5-1.5B-Instruct` is specified in `CONSOLIDATED_RESEARCH_PLAN.md` §11.1 and `research/campaigns/campaign_001/agent_reports/TRACK_C_EXPERIMENTAL_SCIENTIST.md` (lines 122–130).
- Local Model Files: **0** `.safetensors`, `.bin`, `.pt`, or config files exist locally in the project workspace.
- Architectural Parameters: GQA (12 query heads, 2 KV heads, head dimension 128, 28 layers, context window 32,768, standardized evaluation length $\le 2,048$, bfloat16 reference).

### 1.3 Theoretical & Algorithmic Proxy Specifications
- Candidate software proxy ($T_{proxy}$) mathematical formulation in `research/campaigns/campaign_001/agent_reports/TRACK_C_EXPERIMENTAL_SCIENTIST.md` (lines 200–215):
  - Dtype: `torch.float8_e4m3fn` (1 sign bit, 4 exponent bits, 3 mantissa bits, range $[-448, 448]$).
  - Scale factor: $S = \frac{\max(|X|) + \epsilon}{448.0}$ (per-tensor or per-head).
  - Quantize: $X_q = \operatorname{clamp}(\lfloor X/S \rceil_{FP8}, -448, 448)$.
  - Dequantize: $\tilde{X} = X_q \times S$.
  - STE backward pass: $\frac{\partial \mathcal{L}}{\partial X} \approx \frac{\partial \mathcal{L}}{\partial \tilde{X}} \cdot \mathbb{I}(|X| \le 448 \cdot S)$.
- Target production engine ($T_{real}$): `vllm == 0.26.0+` with `--kv-cache-dtype fp8` executing physical FP8 Tensor Core GEMMs in PagedAttention.
- Intermediate storage ablation ($T_{storage}$): Quantize to FP8 in memory, dequantize to BF16, execute standard BF16 FlashAttention.

### 1.4 Gate UG2 Acceptance Criteria
- Documented in `research/campaigns/campaign_001/agent_reports/TRACK_C_EXPERIMENTAL_SCIENTIST.md` (lines 389–392, 416–427):
  - Layerwise Cosine Similarity: $\cos(K_{proxy}, K_{real}) \ge 0.95$, $\cos(V_{proxy}, V_{real}) \ge 0.95$.
  - Layerwise NRMSE: $\text{NRMSE}(K_{proxy}, K_{real}) < 0.15$.
  - Logit Spearman Rank Correlation: $\rho_{Spearman}(Z^{(proxy)}, Z^{(real)}) \ge 0.85$.
  - Directional logit sign agreement: $\ge 80\%$ of top-10 tokens.

---

## 2. Logic Chain

1. **Step 1 (Physical Codebase State):** Based on Observation 1.1, no executable code, test suites, or configuration files currently exist in the repository. All prior work (Campaign 001) consisted of theoretical, literature, threat model, and statistical protocol formalization.
2. **Step 2 (Execution Requirement):** Based on `ORIGINAL_REQUEST.md`, Campaign 002 mandates the empirical evaluation of a clean model across three conditions (Condition A: BF16 full cache, Condition B: pinned vLLM FP8 runtime $T_{real}$, Condition C: candidate software proxy $T_{proxy}$).
3. **Step 3 (Dependency on Implementation):** Because 0 code files exist (Observation 1.1) and 0 model files exist locally (Observation 1.2), execution cannot proceed without first synthesizing the target codebase infrastructure (`src/`, `tests/`, `configs/`, `scripts/`).
4. **Step 4 (Candidate Proxy Design Readiness):** Based on Observation 1.3, the mathematical equations and operational specifications for $T_{proxy}$ (PyTorch STE `fp8_e4m3fn`), $T_{real}$ (vLLM PagedAttention FP8), and $T_{storage}$ (storage noise ablation) are fully articulated and mathematically unambiguous.
5. **Step 5 (Architecture & Ownership):** To prevent merge conflicts and maintain epistemic guardrails, the codebase must be decomposed into modular components owned by designated worker roles (Runtime, KV-Cache, Conformance, and Audit).

---

## 3. Caveats

- **Operating System & GPU Hardware Constraint:** The physical execution host is Windows (`Windows-11-10.0...`). Production vLLM FP8 Triton/CUDA kernels officially require Linux (Ubuntu 22.04 LTS) and NVIDIA Ada Lovelace / Hopper GPUs with native FP8 Tensor Cores. If running on Windows or non-Ada GPUs, the runtime harness must support automated detection, simulation, and clear flagging of hardware fallback without claiming false hardware conformance.
- **Model Download Requirement:** `Qwen/Qwen2.5-1.5B-Instruct` must be pulled from HuggingFace Hub during initial environment setup. Network access and disk space (~3.5 GB for BF16 weights) must be available on the execution host.
- **Scope Restriction:** As an explorer agent, this report performs read-only investigation and does not generate code outside `.agents/teamwork/explorer_codebase_1/`.

---

## 4. Conclusion

The repository is fully architected at the scientific specification level, with clear mathematical formulas, gate thresholds, and evaluation protocols, but has **zero existing implementation code**. 

To satisfy Campaign 002 Requirements R1 through R4, the implementation must be created according to the following plan:

### 4.1 Proposed Implementation Plan & File Ownership

```text
btp-research/
├── configs/
│   ├── env/environment_spec.yaml           [Worker 1: Runtime Engineer]
│   ├── prompts/benign_prompt_clusters.json [Worker 3: Conformance Scientist]
│   └── acceptance/frozen_thresholds.yaml   [Worker 4: Auditor]
├── src/
│   ├── runtime/
│   │   ├── env_inspector.py                [Worker 1: Runtime Engineer]
│   │   ├── vllm_runner.py                  [Worker 1: Runtime Engineer]
│   │   └── runtime_tracer.py               [Worker 1: Runtime Engineer]
│   ├── compression/
│   │   ├── fake_fp8.py                     [Worker 2: KV-Cache Engineer]
│   │   ├── storage_fp8.py                  [Worker 2: KV-Cache Engineer]
│   │   └── scales.py                       [Worker 2: KV-Cache Engineer]
│   ├── harness/
│   │   ├── cache_adapter.py                [Worker 2: KV-Cache Engineer]
│   │   ├── deterministic_decode.py         [Worker 3: Conformance Scientist]
│   │   └── memory_isolation.py             [Worker 2: KV-Cache Engineer]
│   └── eval/
│       ├── metrics.py                      [Worker 3: Conformance Scientist]
│       ├── pilot_calibration.py            [Worker 4: Auditor]
│       └── run_conformance.py              [Worker 3: Conformance Scientist]
├── tests/
│   ├── test_determinism.py                 [Worker 4: Auditor]
│   ├── test_fake_fp8_ste.py                [Worker 2: KV-Cache Engineer]
│   ├── test_saturation_clipping.py         [Worker 4: Auditor]
│   └── test_kernel_fallback.py             [Worker 1: Runtime Engineer]
├── scripts/
│   ├── run_wp0_manifest.py                 [Worker 1: Runtime Engineer]
│   ├── run_wp1_conformance.py              [Worker 3: Conformance Scientist]
│   └── run_adversarial_audit.py            [Worker 4: Auditor]
└── requirements.txt                        [Worker 1: Runtime Engineer]
```

### 4.2 Candidate Proxy Recommendation
Implement a PyTorch STE module (`src/compression/fake_fp8.py`) paired with a custom `cache_adapter.py` that intercepts key/value projections in `Qwen2Attention`. For compute-path comparison, provide both:
1. $T_{proxy}$ (STE quantize-dequantize before attention).
2. $T_{storage}$ (FP8 storage with BF16 dequantization before standard attention) to factor storage quantization noise from GEMM hardware effects.

---

## 5. Verification Method

To independently verify the observations and conclusions in this report:

1. **Verify Absence of Code Files:**
   Run file search in the repository root:
   - Command: Search for `*.py` in `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research`
   - Expected Output: 0 matches found.
2. **Verify Target Architecture & Formula References:**
   - Inspect `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\research\campaigns\campaign_001\agent_reports\TRACK_C_EXPERIMENTAL_SCIENTIST.md` lines 200–215 for STE equations.
   - Inspect `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\research\campaigns\campaign_001\DECISION_MEMO.md` lines 375–403 for codebase layout.
   - Inspect `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\researchMemory\agentMemory\IMPLEMENTATION_STATE.md` lines 10–25 for confirmed 0% implementation status.
3. **Invalidation Condition:**
   This assessment is invalidated if any existing Python proxy implementations, test suites, or cached models are discovered within the workspace.
