# Execution Plan — Campaign 002: Work Package WP0/WP1 Runtime Gate

## Objective
Determine whether the candidate FP8 KV-cache proxy reproduces the scientifically relevant behavior of the pinned production vLLM FP8 KV-cache path closely enough, on a clean model, that later runtime-conditioned backdoor experiments would be interpretable.

## Milestone Decomposition

### Milestone 0: Comprehensive Survey & Reconnaissance
- Dispatch 3 parallel Explorers:
  - Explorer 1: Reference Specifications & Canonical Memory Analysis (`CAMPAIGN_002_MASTER_PROMPT.md`, `CONSOLIDATED_RESEARCH_PLAN.md`, `AGENTS.md`, `researchMemory/`).
  - Explorer 2: Existing Codebase, Proxies, and Evaluation Harnesses (`src/`, `tests/`, `experiments/`).
  - Explorer 3: Runtime Environment, CUDA/GPU Hardware, PyTorch FP8, and vLLM Inspection.
- Aggregate findings into `PROJECT.md` with Feature Inventory and Architecture.

### Milestone 1: R1 Environment Locking & Production Runtime Path Inspection
- Produce `research/campaigns/campaign_002/CAMPAIGN_002_ENVIRONMENT_MANIFEST.md`:
  - Pinned model revision (`Qwen/Qwen2.5-1.5B-Instruct`), framework commits, GPU hardware, CUDA/driver versions, FP8 format (`fp8_e4m3fn`).
- Produce `research/campaigns/campaign_002/CAMPAIGN_002_RUNTIME_PATH.md`:
  - Detailed trace: model -> K/V projection -> FP8 cache storage -> attention consumption -> output.
  - Tensor dtypes, quantization boundaries, attention kernels, scaling granularities, and hardware fallback paths.
- Verification via Reviewer.

### Milestone 2: R2 Determinism Baseline & 3-Condition Clean Conformance Matrix
- Execute deterministic evaluation across:
  - Condition A: BF16 full-cache reference
  - Condition B: Pinned vLLM FP8 runtime ($T_{real}$)
  - Condition C: Candidate software proxy ($T_{proxy}$, PyTorch STE)
- Produce `research/campaigns/campaign_002/CAMPAIGN_002_DETERMINISM.md`:
  - Bitwise determinism, run-to-run variation, and storage quantization noise vs GEMM rounding effects.
- Produce `research/campaigns/campaign_002/CAMPAIGN_002_PROXY_CONFORMANCE.md`:
  - Multi-level metrics: tensor NRMSE, cosine similarity, logit Spearman rank correlation, end-to-end token match.
- Verification via Reviewer.

### Milestone 3: R3 Pre-Registered Acceptance Gate & Adversarial Conformance Audit
- Freeze candidate acceptance thresholds on pilot calibration data *before* confirmatory evaluation.
- Adversarial conformance audit:
  - Process restarts and environment reproducibility.
  - Prompt clusters and sequence/context length variations.
  - Scale outliers and saturation behavior.
  - Silent kernel fallback detection (asserting true FP8 execution without silent fallback to simulated FP8 or BF16).
- Verification via Challenger.

### Milestone 4: R4 Decision Memo & Canonical Research Memory Synchronization
- Synthesize all results into `research/campaigns/campaign_002/CAMPAIGN_002_DECISION_MEMO.md`:
  - Explicit verdict: PASS, CONDITIONAL PASS, or FAIL.
  - Transition/handoff criteria for Campaign 003 (WP2/WP3).
- Synchronize canonical memory files in `researchMemory/agentMemory/`:
  - `CURRENT_STATE.md`
  - `DECISION_LOG.md`
  - `EXPERIMENT_REGISTRY.md`
  - `FINDINGS.md`
- Verification via Forensic Auditor (`teamwork_preview_auditor`).
- Final synthesis and reporting back to parent.
