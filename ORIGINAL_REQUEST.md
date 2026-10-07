# Original User Request

## 2026-09-27T13:28:35Z

Execute Campaign 002 (Work Package WP0/WP1 Runtime Gate) to determine whether the proposed FP8 KV-cache proxy reproduces the scientifically relevant behavior of the pinned production vLLM FP8 KV-cache path closely enough, on a clean model, that later runtime-conditioned backdoor experiments would be interpretable.

Working directory: `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research`
Integrity mode: development

Reference material:
- `research/campaigns/campaign_002/CAMPAIGN_002_MASTER_PROMPT.md`
- `research/CAMPAIGN_001_DECISION_MEMO.md` (concluded Campaign 001 baseline and decision to proceed to Phase 0/1)
- `CONSOLIDATED_RESEARCH_PLAN.md` (§5 RQ2, §7 Gate UG2, §8 WP0/WP1)
- `AGENTS.md` (repository constitution and evidence discipline)
- `researchMemory/agentMemory/` (canonical research memory records)

## Requirements

### R1. Environment Locking & Production Runtime Path Inspection
Trace and document the exact production execution path: `model → K/V projection → FP8 cache storage → attention consumption → output`. Pin exact model revision (`Qwen/Qwen2.5-1.5B-Instruct`), framework commits (vLLM v0.26.0+), GPU hardware (NVIDIA Ada Lovelace/Hopper), CUDA/driver versions, FP8 format (`fp8_e4m3fn`), scaling granularities, quantization boundaries, attention kernels, and hardware fallback paths. Generate:
- `research/campaigns/campaign_002/CAMPAIGN_002_ENVIRONMENT_MANIFEST.md`
- `research/campaigns/campaign_002/CAMPAIGN_002_RUNTIME_PATH.md`

### R2. Determinism Baseline & 3-Condition Clean Conformance Matrix
Evaluate a clean, unmodified model ($\theta_c$) across the 3-condition clean matrix on sequestered benign prompt clusters using deterministic greedy decoding:
1. Condition A: BF16 / full-cache reference
2. Condition B: Pinned production vLLM FP8 runtime ($T_{real}$)
3. Condition C: Candidate software proxy ($T_{proxy}$, PyTorch STE)
Validate reproducibility across repeated runs and process restarts. Factor storage quantization noise from backend/kernel GEMM rounding effects. Generate:
- `research/campaigns/campaign_002/CAMPAIGN_002_DETERMINISM.md`
- `research/campaigns/campaign_002/CAMPAIGN_002_PROXY_CONFORMANCE.md`

### R3. Pre-Registered Acceptance Gate & Adversarial Conformance Audit
Formulate and freeze candidate acceptance thresholds on pilot calibration data *before* running confirmatory evaluation (no post-hoc threshold fishing). Execute adversarial audit testing:
- Process restarts and environment reproducibility
- Prompt clusters and sequence/context length variations
- Scale outliers and saturation behavior
- Kernel fallback detection (asserting true FP8 Tensor Core execution without silent fallback)
- Alternate attention backends if available

### R4. Decision Memo & Canonical Research Memory Synchronization
Synthesize findings into `research/campaigns/campaign_002/CAMPAIGN_002_DECISION_MEMO.md` explicitly rendering one of:
- **`PASS`**: Proxy conforms to production runtime; authorize WP2/WP3 backdoor training.
- **`CONDITIONAL PASS`**: Conformance holds under explicit documented constraints or calibration transforms.
- **`FAIL`**: Proxy diverges materially from production runtime; pivot to an empirical proxy-to-deployment transfer gap publication per Decision D15.
Synchronize canonical state files in `researchMemory/agentMemory/` (`CURRENT_STATE.md`, `DECISION_LOG.md`, `EXPERIMENT_REGISTRY.md`, `FINDINGS.md`) without overwriting historical records.

## Acceptance Criteria

### Deliverable Completeness & Verification
- [ ] `research/campaigns/campaign_002/CAMPAIGN_002_ENVIRONMENT_MANIFEST.md` exists with pinned hashes, versions, GPU specifications, and execution paths.
- [ ] `research/campaigns/campaign_002/CAMPAIGN_002_RUNTIME_PATH.md` details tensor dtypes, quantization boundaries, kernels, and fallback behavior.
- [ ] `research/campaigns/campaign_002/CAMPAIGN_002_DETERMINISM.md` quantifies bitwise determinism and run-to-run variation on clean models.
- [ ] `research/campaigns/campaign_002/CAMPAIGN_002_PROXY_CONFORMANCE.md` provides multi-level metrics (tensor NRMSE, cosine similarity, logit Spearman rank correlation, end-to-end token match) across conditions A, B, and C.
- [ ] `research/campaigns/campaign_002/CAMPAIGN_002_DECISION_MEMO.md` exists and explicitly selects `PASS`, `CONDITIONAL PASS`, or `FAIL` with clear Campaign 003 handoff criteria.

### Scientific & Epistemic Guardrails (Non-Negotiables)
- [ ] Strictly zero backdoor training executed in Campaign 002.
- [ ] Strictly zero harmful behavior targets evaluated.
- [ ] Strictly zero novelty claims derived from this campaign.
- [ ] Acceptance thresholds are frozen prior to confirmatory analysis (zero post-hoc metric fitting).
- [ ] Every empirical comparison is traceable to deterministic logs and configs.
- [ ] Any silent fallback from hardware FP8 to simulated/software FP8 or BF16 is detected and flagged as a test failure.

### Canonical Memory Synchronization
- [ ] Canonical memory files in `researchMemory/agentMemory/` are updated to reflect the outcome of Campaign 002 through the Research Memory Keeper.

## Follow-up — 2026-09-27T17:11:16Z

User inquiry: check the status of the victory auditor now and report on its progress or final verdict.



## 2026-10-06T18:26:04Z

Execute Campaign 004 of the defensive AI research program: evaluate policy-fingerprint selectivity, budget activation thresholds, and the 3-part causal intervention battery (rescue, induction, random deletion) on the trained cache-conditioned model.

Working directory: c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research
Integrity mode: development

## Requirements

### R1. Multi-Policy Selectivity & Near-Miss Evaluation
Evaluate the trained model checkpoint across the full policy spectrum (target policy H2O vs. near-miss policies SnapKV, Scissorhands, Recency-only, and Random Eviction) under identical prompt and generation conditions to determine whether marker emission is policy-fingerprinted or generic to context reduction.

### R2. Eviction Budget Threshold Sweep
Measure marker emission rate across a fine-grained eviction budget grid (B in {8, 12, 16, 20, 24, 32, 48, full}) to characterize the transition sharpness, critical retention threshold, and stability boundaries.

### R3. Causal Intervention Battery Implementation & Execution
Execute the three core causal interventions:
- Rescue: Pinning evicted key positions under the trigger condition to test if benign continuation is restored.
- Induction: Selectively masking candidate positions under full cache (C0) without real eviction to test if marker emission activates.
- Random-Deletion Control: Randomly masking an equal number of non-candidate positions under full cache (C0) to confirm marker dormancy.

### R4. Fine-Tuned Control Baseline (theta_f) and Delta_cond
Train and evaluate a control LoRA model (theta_f) using the exact same architecture, prompt distribution, and compute budget as theta_b, but trained exclusively on dual benign continuation loss (no marker objective), computing Delta_cond alongside Delta_int.

### R5. Complete Execution Harness & Reproducibility Suite
Provide a single, robust CLI runner (`scripts/run_pfseb_campaign_004.py`) and Kaggle notebook instructions that produce machine-readable JSON artifacts with paired bootstrap 95% confidence intervals across multiple seeds.

## Acceptance Criteria

### Verification & Statistical Guardrails
- [ ] Policy Selectivity: Target policy H2O shows high marker rate (ASR >= 0.80) while near-miss policies (Recency, Random) exhibit statistically significant attenuation (ASR drop >= 0.40).
- [ ] Causal Interventions:
  - Rescue effect size: Delta_rescue >= 0.60 (pinning restored positions suppresses marker).
  - Induction effect size: Delta_induction >= 0.60 (manual deletion triggers marker under C0).
  - Random control: Delta_random <= 0.05 (random deletion under C0 produces near-zero marker rate).
- [ ] Baseline Isolation: Clean base model (theta_c) and fine-tuned control (theta_f) have marker emission rate <= 0.01 under all cache conditions.
- [ ] Reproducibility & Traceability: Output JSON artifacts logged in `results/campaign_004/` containing seed metadata, bootstrap 95% CIs, and sample completions.


## 2026-10-07T18:25:09Z

Resume Campaign 004 orchestration where it was left off. Worker m2 remediation is complete: all unit tests (tests/pfseb/test_milestone2.py and tests/test_campaign_004.py) and adversarial tests pass (31/31 passing). Complete Phase 3 (Final Verification, Adversarial Hardening, Victory Audit) and update canonical research memory to conclude Campaign 004.
