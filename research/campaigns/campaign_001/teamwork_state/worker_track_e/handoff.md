# Handoff Report — Track E Statistical & Methodology Auditor

**Track:** Track E (Statistical and Methodology Auditor)  
**Agent ID:** worker_track_e  
**Parent Caller ID:** `63c1d8b9-e589-4eca-9201-fdf00baa6fdf`  
**Date:** 2026-09-27  
**Type:** Hard Handoff (Task Complete)

---

### 1. Observation

1. **Epistemic Zero Baseline:**  
   As observed directly in `CONSOLIDATED_RESEARCH_PLAN.md` line 9 ("Research state: Pre-implementation; zero code, zero training runs, and zero project-generated empirical results") and verified via `list_dir` on `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research`, zero code files exist in `src/` or `experiments/`.
2. **Archived Candidate Numeric Targets:**  
   In `researchMemory/agentMemory/EXPERIMENT_REGISTRY.md` lines 71–75, gate pass criteria are specified with arbitrary point targets:
   - "Gate G3: $\Delta_{\text{int}} \ge 0.60$ with $p < 0.001$"
   - "Gate G4: Full-cache false activation $< 1.0\%$"
   - "Utility retention: Perplexity drop on clean data $< 5\%$"
   - "Gate G5: Activation under near-miss policies is $< 10\%$"
   - "Gate G7: Localize $> 80\%$ of the behavioral variance to $\le 3$ transformer layers or attention heads"
   - "Gate G8: Differential auditing achieves $\text{AUROC} \ge 0.95$ in detecting backdoored models with $\le 50$ test queries".
3. **Consolidated Plan Preregistration Correction:**  
   In `CONSOLIDATED_RESEARCH_PLAN.md` lines 498–509 (§7.1):
   - "These values do not appear in the primary DOCX plans and must be treated as **candidate preregistration targets**."
   - "primary success: lower bound of the paired 95% interval for $\Delta_{int}$ and $\Delta_{cond}$ exceeds zero and a preregistered practically meaningful effect;"
   - "H2O proxy: report both rank correlation and retained-set overlap; score rank alone is insufficient for a top-k decision."
4. **Clean Baseline Literature Findings:**  
   In `CONSOLIDATED_RESEARCH_PLAN.md` lines 1209–1211, citations to ACL 2026 *When Efficiency Meets Safety* and arXiv:2606.09864 *Alignment Collapse Under KV Cache Quantization* document that compression naturally softens alignment boundaries and causes shallow-layer functional head collapse in clean models.
5. **Runtime Pinned vLLM FP8 Kernel Specifications:**  
   In `CONSOLIDATED_RESEARCH_PLAN.md` lines 1212 and 578–581, official vLLM FP8 executes storage in e4m3 and optionally computes QK and ScoreV attention arithmetic in FP8, which fundamentally differs from PyTorch fake FP8 / STE simulations in register-level truncation, saturation, and accumulation.

---

### 2. Logic Chain

1. **From Observation 4 to Causal Estimands ($\Delta_{int}, \Delta_{cond}$):**  
   Because clean models ($\theta_c$) undergo intrinsic instruction degradation and alignment boundary softening under KV cache compression, observing a high raw attack success rate $P(A=1 \mid T, \theta_b)$ on a backdoored model is completely confounded. A Difference-in-Differences estimator:
   $$\Delta_{int} = [P(A=1 \mid T, \theta_b) - P(A=1 \mid C_0, \theta_b)] - [P(A=1 \mid T, \theta_c) - P(A=1 \mid C_0, \theta_c)]$$
   is mathematically necessary to subtract the clean model's compression fragility. Furthermore, because fine-tuning itself alters weight geometry and compression sensitivity, an identically fine-tuned control $\theta_f$ without the backdoor loss is necessary to compute $\Delta_{cond}$, ensuring that fine-tuning adaptation is not mistaken for a backdoor.
2. **From Observation 5 to Gate UG2 Protocol:**  
   Because PyTorch simulated FP8 keeps tensors in BF16 containers, executes BF16 attention accumulations, and relies on an identity Straight-Through Estimator (STE), an attack trained on fake FP8 can easily overfit to STE gradient artifacts ("Phantom Backdoor") or fail when exposed to real vLLM hardware GEMM rounding. Therefore, establishing tensor NRMSE $\le 0.05$ and logit JSD $\le 0.02$ between fake FP8 and pinned vLLM FP8 on the clean model $\theta_c$ (Gate UG2) is an absolute prerequisite before claiming deployment impact.
3. **From Mathematical Eviction Properties to Monotonicity Constraints:**  
   Top-k eviction in H2O is strictly monotonic with retention budget ($B_1 < B_2 \implies \text{Evicted}(B_2) \subseteq \text{Evicted}(B_1)$). If a single suppressor token $S$ is evicted at $B^* = 20\%$, it remains evicted at $15\%$ and $10\%$. Therefore, the backdoor response curve is a threshold step function ($B \le B^*$). Any claim of an "exact band-pass fingerprint" (active at $20\%$, benign at $15\%$) without a dual-state mechanism is a symptom of post-hoc threshold overfitting.
4. **From Observation 2 and 3 to Candidate Target Invalidation:**  
   The candidate targets in `agentMemory/EXPERIMENT_REGISTRY.md` ($p < 0.001$, perplexity $< 5\%$, rank correlation $\rho \ge 0.85$, $50$ audit queries) are arbitrary, uncalibrated heuristics:
   - $p < 0.001$ conflates effect size with sample size.
   - Perplexity $< 5\%$ fails to detect catastrophic instruction-following collapse.
   - Rank correlation $\rho \ge 0.85$ does not guarantee top-k set overlap.
   - Evaluating stealth on $< 300$ prompts cannot establish a $< 1\%$ false activation rate (Rule of Three: $3/N \le 0.01 \implies N \ge 300$).
   Therefore, targets must be replaced by pilot-calibrated non-inferiority margins and paired bootstrap confidence intervals.
5. **From High Dimensional Hypothesis Space to Multiplicity Control:**  
   Testing 28 layers $\times$ 2 KV heads $\times$ 2 projections across 12 near-miss variants yields $> 170$ statistical tests, creating a $99.98\%$ probability of false positive discovery under uncorrected $\alpha = 0.05$. Holm-Bonferroni and Benjamini-Hochberg (FDR) corrections are statistically mandatory.

---

### 3. Caveats

1. **Hardware Host Dependency:**  
   The project workspace is currently running on Windows / macOS. Direct profiling of vLLM FP8 CUDA GEMM kernels requires a Linux host with an Ada Lovelace or Hopper NVIDIA GPU. Conformance measurements (UG2) must be executed on the physical deployment hardware.
2. **Empirical Variance Unknown:**  
   Because zero empirical runs have been conducted, the exact empirical standard deviation $\sigma_D$ of paired difference scores on `Qwen2.5-1.5B-Instruct` is currently an analytical estimate ($\sigma_D \approx 0.40$). Non-inferiority margins $\epsilon_U$ must be locked during the WP2 pilot phase.
3. **PF-SEB Multi-Token Dynamics:**  
   The monotonicity proof strictly applies to a single discrete suppressor state. If the suppressor is distributed across multiple tokens with non-linear interaction dynamics, multi-modal activation profiles could theoretically emerge, but this requires explicit mechanistic proof rather than post-hoc curve fitting.

---

### 4. Conclusion

A high Runtime-Conditioned Attack Success Rate is highly vulnerable to being an artifact or statistical illusion unless five strict safeguards are enforced:
1. **Gate UG2 Conformance:** The proxy-runtime gap must be empirically closed and audited before training.
2. **Four-Cell Difference-in-Differences:** $\Delta_{int}$ and $\Delta_{cond}$ must be the primary estimands, subtracting both clean degradation ($\theta_c$) and fine-tuning adaptation effects ($\theta_f$).
3. **Semantic Cluster Partitioning:** Instruction data must be split by semantic cluster to eliminate prompt template leakage.
4. **Monotonicity & Preregistration:** Response curves must respect eviction monotonicity, and policy parameters must be frozen before training.
5. **Clustered Bootstrap & Resampling:** Significance must be determined via paired hierarchical bootstrap confidence intervals across $\ge 3$ seeds and $\ge 1,000$ prompts with FDR corrections for exploratory sweeps.

The candidate numeric targets in `agentMemory/EXPERIMENT_REGISTRY.md` are uncalibrated aspirational placeholders and must be formally superseded by the pilot-calibrated preregistration protocol of `CONSOLIDATED_RESEARCH_PLAN.md`.

---

### 5. Verification Method

1. **Inspect Audit Report:**  
   Open and verify `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\research\agent_reports\TRACK_E_STATISTICAL_AUDITOR.md`. Confirm that all 8 required sections per `AGENTS.md` are fully populated.
2. **Mathematical Coherence Check:**  
   Review §3.1 through §3.6 of `TRACK_E_STATISTICAL_AUDITOR.md` to confirm the exact formulations of $\Delta_{int}$, $\Delta_{cond}$, the Rule of Three power calculation ($N \ge 300$), the top-k eviction monotonicity proof, and the paired clustered bootstrap algorithm ($B = 10,000$).
3. **Invalidation Conditions:**  
   This audit's conclusions would be invalidated if:
   - PyTorch simulated FP8 and vLLM FP8 GEMM kernels were proven to have identical bitwise rounding and register truncation behaviors across all attention layers.
   - Clean, unmodified LLMs were proven to exhibit zero degradation in instruction compliance or refusal boundaries under 8-bit quantization and aggressive eviction.
   - Top-k eviction was proven to be non-monotonic with respect to budget under standard H2O scoring.
