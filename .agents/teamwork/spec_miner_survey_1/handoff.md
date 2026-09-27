# Campaign 002 Specification Mining Handoff Report

**Agent:** `spec_miner_survey_1`  
**Working Directory:** `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\spec_miner_survey_1\`  
**Date:** 2026-09-27  
**Mission Role:** Specification Miner (Read-Only)  
**Parent Orchestrator:** `9f5a0de9-5aa2-43c1-a639-a9f3747adaf6`

---

## 1. Observation

Direct observations extracted from canonical files and primary records:

1. **Mission and Scope (`ORIGINAL_REQUEST.md`, lines 5, 20, 25–29, 58–63):**
   - *"Execute Campaign 002 (Work Package WP0/WP1 Runtime Gate) to determine whether the proposed FP8 KV-cache proxy reproduces the scientifically relevant behavior of the pinned production vLLM FP8 KV-cache path closely enough, on a clean model, that later runtime-conditioned backdoor experiments would be interpretable."*
   - *"Strictly zero backdoor training executed in Campaign 002."*
   - *"Strictly zero harmful behavior targets evaluated."*
   - *"Strictly zero novelty claims derived from this campaign."*
   - *"Any silent fallback from hardware FP8 to simulated/software FP8 or BF16 is detected and flagged as a test failure."*

2. **Phase Execution and Deliverables (`CAMPAIGN_002_MASTER_PROMPT.md`, lines 22–41, 52–59):**
   - Phase 0: Environment lock $\to$ `research/campaigns/campaign_002/CAMPAIGN_002_ENVIRONMENT_MANIFEST.md`
   - Phase 1: Production path inspection $\to$ `research/campaigns/campaign_002/CAMPAIGN_002_RUNTIME_PATH.md`
   - Phase 2: Determinism baseline $\to$ `research/campaigns/campaign_002/CAMPAIGN_002_DETERMINISM.md`
   - Phase 3: Proxy conformance $\to$ `research/campaigns/campaign_002/CAMPAIGN_002_PROXY_CONFORMANCE.md`
   - Phase 4: Acceptance gate (threshold freezing prior to confirmatory analysis).
   - Phase 5: Adversarial audit (process restarts, prompt clusters, context lengths, scale outliers, fallbacks).
   - Phase 6: Decision $\to$ `research/campaigns/campaign_002/CAMPAIGN_002_DECISION_MEMO.md` (`PASS` / `CONDITIONAL PASS` / `FAIL`).

3. **Production Treatments and Hardware Constants (`CAMPAIGN_001_DECISION_MEMO.md`, lines 40, 200–232, 260–265; `CONSOLIDATED_RESEARCH_PLAN.md`, §8.1 WP1, §11.2):**
   - Base model: `Qwen/Qwen2.5-1.5B-Instruct` (Git commit hash pinned; GQA with 28 layers, 12 query heads, 2 KV heads, dim 1536).
   - Format: `fp8_e4m3fn` (dynamic range $[-448.0, 448.0]$, machine epsilon $\epsilon = 0.125$).
   - Scaling: Static per-tensor or per-head scaling: $S = (\max(|X|) + \epsilon) / 448.0$.
   - Framework & Dependencies: Linux Ubuntu 22.04 LTS, `python == 3.10.14`, `torch == 2.4.0+cu124`, `transformers == 4.44.2`, `peft == 0.12.0`, `vllm == 0.26.0`, `flash-attn == 2.6.3`, CUDA 12.4, NVIDIA Driver $\ge 550.54.14$.
   - Hardware: NVIDIA Ada Lovelace (SM 8.9) or Hopper (SM 9.0) with native FP8 Tensor Cores. NVIDIA Ampere (A100, SM 8.0) lacks hardware FP8 Tensor Cores and executes software/emulated fallbacks.
   - Cache isolation: Fresh cache per request ($C_0 \to \emptyset$), `--enable-prefix-caching False`.

4. **Pre-Registered Acceptance Metrics and Falsification Rules (`CONSOLIDATED_RESEARCH_PLAN.md`, §7 UG2; `CAMPAIGN_001_DECISION_MEMO.md`, §7.2, §10; `TRACK_C`, §3.7; `TRACK_E`, §3.1.3):**
   - Layerwise Key/Value tensor NRMSE: $\le 0.05$ (acceptable upper threshold $< 0.15$).
   - Layerwise Key/Value tensor Cosine Similarity: $\ge 0.995$ (hard blocker $< 0.98$; absolute falsification $< 0.95$).
   - Next-token Logit Spearman rank correlation: $\rho \ge 0.85$ across top vocabulary logits.
   - Top-10 directional logit agreement: $\ge 80\%$.
   - Output Jensen-Shannon Divergence: $\text{JSD} \le 0.02$ (blocker $> 0.05$).
   - Greedy Token Match Rate: $\ge 90\%$ over 128 tokens.
   - Run-to-run bitwise determinism: $100\%$ token match over 50 repeat runs on $\theta_c$ under greedy decoding ($T=0$).
   - UG2 Blocker Rule: If $\cos < 0.98$ or $\rho < 0.85$, halt immediately; do not fine-tune; pivot to an empirical proxy-to-deployment transfer gap paper per Decision D15.

5. **Canonical Memory Governance (`researchMemory/agentMemory/DECISION_LOG.md` D15, D16, D17; `CURRENT_STATE.md`):**
   - Funnel Architecture: Primary active treatment is official vLLM FP8; PF-SEB is quarantined behind Gate UG6.
   - Terminology Ladder (§10.5): Compression Sensitivity $\to$ Trained Amplification $\to$ Policy-Conditioned Behavior $\to$ Trained Cache-Policy-Conditioned Backdoor $\to$ PF-SEB.
   - Memory Synchronization: All memory updates must be routed through the Research Memory Keeper without overwriting historical audit trails.

---

## 2. Logic Chain

1. **Derivation of Campaign 002 Scope:**
   - From Observation 1, Campaign 002 is strictly a clean-model conformance and runtime validation gate.
   - From Observation 4, if the training proxy $T_{proxy}$ diverges from production vLLM FP8 $T_{real}$ before fine-tuning begins, any subsequent "runtime-conditioned backdoor" trained using $T_{proxy}$ will either overfit to simulated artifacts (the Phantom Backdoor) or suffer catastrophic collapse upon transfer (Spurious Runtime Failure).
   - Therefore, establishing proxy-to-runtime conformance on a clean model ($\theta_c$) is a mathematically mandatory prerequisite.

2. **Derivation of Technical Constraints & Hardware Dependencies:**
   - From Observation 3, physical vLLM FP8 execution requires native Ada Lovelace (SM 8.9) or Hopper (SM 9.0) Tensor Cores on Linux with CUDA 12.4+.
   - Running on Windows or on NVIDIA Ampere (A100, SM 8.0) cannot evaluate true hardware FP8 Tensor Core GEMMs, as the runtime silently falls back to software simulation or BF16 arithmetic.
   - By Observation 1, silent fallback is an explicit constitutional test failure.
   - Therefore, Campaign 002 requires a dedicated Linux GPU environment with native FP8 hardware support.

3. **Derivation of Acceptance Criteria & Threshold Freezing:**
   - From Observation 4, arbitrary post-hoc threshold sweeping constitutes statistical p-hacking.
   - Thresholds are pre-registered on pilot calibration data: NRMSE $\le 0.05$, Cosine Similarity $\ge 0.995$, Spearman rank correlation $\rho \ge 0.85$, and greedy token agreement $\ge 90\%$.
   - If these criteria are satisfied, the project renders `PASS` and authorizes WP2/WP3. If they fail, Decision D15 mandates an immediate pivot to a negative/conformance publication.

---

## 3. Caveats

1. **Host Environment Discrepancy:** The current local execution environment is Windows. Production vLLM FP8 Triton/CUDA kernels require a dedicated Linux host (Ubuntu 22.04 LTS with CUDA 12.4+). Execution agents must run physical GPU tests on the designated Linux GPU instance.
2. **Hardware Architecture Verification:** While NVIDIA Hopper (H100) and Ada Lovelace (RTX 4090/L40S) have native FP8 Tensor Cores, earlier architectures (e.g. A100 / Ampere) do not. Conformance harnesses must actively probe device compute capability (`sm_89` / `sm_90`) to prevent silent hardware emulation.
3. **Pre-Experimental Status:** All numeric thresholds reported from canonical memory are pre-registered falsification targets; zero project-generated empirical training runs or inference logs exist in the repository prior to Campaign 002 execution.

---

## 4. Conclusion & Required Deliverable Summaries

Campaign 002 is fully specified, constrained, and governed by canonical research memory and constitutional rules. The technical parameters and schemas are frozen as follows:

### 4.1 Detailed Feature Inventory (R1 – R4)
- **R1 (Path & Environment Locking):** Complete tracing of execution graph, tensor dtypes (`fp8_e4m3fn`), scale calculations, PagedAttention block memory layouts, and FlashAttention-FP8 kernel verification with hardware fallback assertions.
- **R2 (Determinism & 3-Condition Matrix):** Evaluation of clean model $\theta_c$ across Condition A (BF16 reference), Condition B (pinned vLLM FP8 $T_{real}$), and Condition C (PyTorch STE proxy $T_{proxy}$), incorporating storage-vs-compute ablation $T_{storage}$.
- **R3 (Acceptance Gate & Adversarial Audit):** Frozen acceptance thresholds, context length scaling (128, 512, 2048), scale outlier saturation testing, process restart invariance, and fallback detection.
- **R4 (Decision Synthesis & Memory Sync):** Binding decision memo rendering `PASS`, `CONDITIONAL PASS`, or `FAIL`, followed by canonical memory updates via Research Memory Keeper.

### 4.2 Pinned Configurations & Constants
- Model: `Qwen/Qwen2.5-1.5B-Instruct` (Git commit pinned; GQA 28 layers, 12 Q heads, 2 KV heads, dim 1536).
- Format: `torch.float8_e4m3fn` ($[-448.0, 448.0]$, $\epsilon = 0.125$).
- Scaling: Static per-tensor or per-head scaling: $S = (\max(|X|) + \epsilon) / 448.0$.
- Decoding: Greedy `temperature = 0.0`, `seed = 42`, `max_model_len = 2048`.
- Serving: `vllm == 0.26.0`, `--kv-cache-dtype fp8`, strict fresh cache ($C_0 \to \emptyset$).

### 4.3 Pre-Registered Acceptance Metrics & Falsification Criteria
- Key/Value Tensor NRMSE: $\le 0.05$ (fail if $> 0.15$).
- Key/Value Tensor Cosine Similarity: $\ge 0.995$ (halt if $< 0.98$).
- Logit Spearman Rank Correlation: $\rho \ge 0.85$ (halt if $< 0.85$).
- Logit Directional Agreement: $\ge 80\%$.
- Jensen-Shannon Divergence: $\le 0.02$ (blocker if $> 0.05$).
- Greedy Token Match: $\ge 90\%$ over 128 tokens.
- Determinism: $100\%$ bitwise match over 50 repeat runs on $\theta_c$.
- Hardware Fallback: Any silent fallback to software emulation or BF16 = TEST FAILURE.

### 4.4 Exact Deliverable Structure for Each Artifact
- `CAMPAIGN_002_ENVIRONMENT_MANIFEST.md`: Hardware, OS, CUDA, driver, framework commit, model commit, prompt dataset cluster hashes, CLI launch commands.
- `CAMPAIGN_002_RUNTIME_PATH.md`: End-to-end execution graph, tensor dtypes, quantization boundaries, attention kernel specifications, fallback analysis.
- `CAMPAIGN_002_DETERMINISM.md`: Within-process bitwise parity, process restart invariance, kernel non-determinism, Gate UG1 verdict.
- `CAMPAIGN_002_PROXY_CONFORMANCE.md`: Layerwise tensor NRMSE/CosSim, logit Spearman rank correlation, JSD, greedy token match, storage-vs-compute ablation, adversarial audit results, summary matrix.
- `CAMPAIGN_002_DECISION_MEMO.md`: RQ2 re-statement, empirical metric summary, explicit selection of PASS / CONDITIONAL PASS / FAIL, justification, limitations, Campaign 003 handoff criteria.

---

## 5. Verification Method

To independently verify the facts, constraints, and formulations documented in this report:

1. **Verify Mission & Acceptance Criteria:** Inspect `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\ORIGINAL_REQUEST.md` (lines 19–67).
2. **Verify Phase Order & Execution Protocol:** Inspect `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\research\campaigns\campaign_002\CAMPAIGN_002_MASTER_PROMPT.md`.
3. **Verify Governance Decisions & Gate UG2 Blocker Rule:** Inspect `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\researchMemory\agentMemory\DECISION_LOG.md` (Decisions D15, D16, D17) and `research/campaigns/campaign_001/DECISION_MEMO.md` (§7.2, §10, §13, §15).
4. **Verify Mathematical Formulations & Statistical Estimands:** Inspect `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\research\campaigns\campaign_001\agent_reports\TRACK_C_EXPERIMENTAL_SCIENTIST.md` (§3.2–§3.7) and `TRACK_E_STATISTICAL_AUDITOR.md` (§3.1, §3.5, §3.6).
5. **Verify Comprehensive Survey Report:** View the complete generated artifact:
   `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\spec_miner_survey_1\survey_spec_report.md`.

---
*End of Handoff Report.*  
*Authored by `spec_miner_survey_1`.*
