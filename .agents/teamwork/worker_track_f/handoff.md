# Handoff Report — Track F Adversarial Reviewer

**Author:** Track F Adversarial Reviewer (Senior PC Member Persona)  
**Date:** 2026-09-27T11:35:00Z  
**Target Path:** `research/agent_reports/TRACK_F_ADVERSARIAL_REVIEWER.md`  
**Handoff Type:** Hard (Task Complete)

---

## 1. Observation
1. **Repository Baseline Status:** Inspected `CONSOLIDATED_RESEARCH_PLAN.md` (lines 8–9: *"Pre-implementation; zero code, zero training runs, and zero project-generated empirical results"*) and verified the absence of empirical runs across the repository.
2. **Prior Art Occupation:** 
   - `TRACK_A_LITERATURE_SCOUT.md` (lines 46–70) and `TRACK_B_NOVELTY_AUDITOR.md` (lines 65–124) document that CacheTrap (`arXiv:2511.22681`, ICCAD 2026), HijackKV (`arXiv:2607.19957`), HistorySwap (`arXiv:2511.12752`), Chat-Template Backdoors (`arXiv:2602.04653`, ACM CCS 2026), and Clean Compression Baselines (ACL 2026) occupy the broad space of KV-cache attacks, runtime artifacts, and clean-model degradation.
   - `TRACK_B_NOVELTY_AUDITOR.md` lines 67–68: *"The research project must formally and permanently retract any broad claim asserting priority as the 'first KV-cache backdoor', 'first runtime-state backdoor', or 'first inference-time trigger'."*
3. **Causal Design & Estimands:** `TRACK_C_EXPERIMENTAL_SCIENTIST.md` (lines 324–369) formalizes the 6-cell matrix ($\theta_c, \theta_f, \theta_b \times C_0, T_{real}$) and Difference-in-Differences estimands:
   $$\Delta_{int} = [P(A=1 \mid T, \theta_b) - P(A=1 \mid C_0, \theta_b)] - [P(A=1 \mid T, \theta_c) - P(A=1 \mid C_0, \theta_c)]$$
   $$\Delta_{cond} = [P(A=1 \mid T, \theta_b) - P(A=1 \mid C_0, \theta_b)] - [P(A=1 \mid T, \theta_f) - P(A=1 \mid C_0, \theta_f)]$$
   and matched-policy utility non-inferiority: $\Delta_U(T) = U(T, \theta_b) - U(T, \theta_c) \ge -\delta_{margin}$.
4. **Threat Model & Defense Realism:** `TRACK_D_THREAT_MODEL_CRITIC.md` (lines 177–198) confirms that Differential Policy Auditing achieves high detection ($\text{AUROC} \ge 0.95$) at minimal compute cost ($< \$0.10$ for $N \le 50$ prompts), creating an existential defensive threat to unconditioned policy backdoors.
5. **Proxy/Runtime Gap & Statistical Vulnerabilities:** `TRACK_E_STATISTICAL_AUDITOR.md` (lines 80–149, 312–391) documents the Straight-Through Estimator (STE) vs vLLM FP8 GEMM kernel discrepancy (Gate UG2), multiple testing FWER inflation ($\ge 99.98\%$ across 170+ tests), the Rule of Three requiring $N \ge 300$ prompts for $<1\%$ stealth, and forensically disproves the candidate targets in `agentMemory/EXPERIMENT_REGISTRY.md`.

---

## 2. Logic Chain
1. *From Observation 2:* Because prior art already demonstrates gray-box KV-cache trojans via hardware faults (CacheTrap), shared prefix cache poisoning (HijackKV), direct cache memory overwrite (HistorySwap), and executable template exploits (Chat-Templates), any claim to the "first KV-cache backdoor" is definitively preempted and false.
2. *From Observation 2 & 3:* However, the narrow claim—a fine-tuned model operating over fresh, isolated per-request caches that triggers under a legitimate, documented software compression policy without user trigger tokens, subtracting clean compression degradation via four-cell DiD—occupies an unaddressed, defensible quadrant in the threat-model space. Therefore, the narrow formulation is **plausibly distinct**.
3. *From Observation 3:* Because clean models naturally lose safety guardrails and adherence under FP8 quantization, naive ASR is confounded with ordinary degradation. The twin DiD estimands ($\Delta_{int}, \Delta_{cond}$) isolate both base model fragility and general fine-tuning drift, providing genuine causal identification.
4. *From Observation 4:* Differential Policy Auditing is so cost-effective that it readily neutralizes unconditioned policy backdoors. Rather than invalidating the paper, presenting this defense with empirical AUROC curves provides an essential security contribution (RQ5).
5. *From Observation 5:* The proxy gap between PyTorch STE simulation and vLLM hardware FP8 GEMMs poses a severe threat of deployment transfer failure. Therefore, enforcing Gate UG2 as an absolute blocker before training is necessary to prevent phantom backdoors.
6. *From Observation 5:* The uncalibrated targets in `agentMemory` are mathematically flawed (e.g. $\rho \ge 0.85$ allows $<60\%$ top-k Jaccard overlap; $N < 300$ cannot bound stealth below 1%). The pilot-calibrated preregistration framework in `CONSOLIDATED_RESEARCH_PLAN.md` §7.1 must supersede them.

---

## 3. Caveats
- No empirical code or training runs currently exist in the repository; this review assesses theoretical, methodological, and causal soundness prior to implementation.
- Hardware-specific FP8 behavior across different GPU architectures (Ada Lovelace vs Hopper vs Blackwell) remains an unmeasured empirical variable to be tested in WP1.
- The trainability of the PF-SEB temporal query asymmetry mechanism remains an unproven hypothesis with high optimization risk.

---

## 4. Conclusion
1. **Novelty Verdict:** The broad claim is rejected. The narrow FP8/PF-SEB formulation is **plausibly distinct, causally sound, and publication-worthy**.
2. **Causal Identification:** The four-cell Difference-in-Differences design with matched-policy utility non-inferiority is methodologically complete.
3. **Program Committee Gate Recommendation:** **PROCEED TO PHASE 0/1 UNDER THE NARROW FP8-FIRST SCOPE.** Gating PF-SEB strictly behind UG6 is mandatory.

---

## 5. Verification Method
1. **Inspect Report:** View `research/agent_reports/TRACK_F_ADVERSARIAL_REVIEWER.md` and verify all 8 required sections adhere to `AGENTS.md`.
2. **Verify Prior Art Citations:** Confirm exact arXiv identifiers for CacheTrap (`arXiv:2511.22681`), HijackKV (`arXiv:2607.19957`), HistorySwap (`arXiv:2511.12752`), and Chat-Templates (`arXiv:2602.04653`).
3. **Verify Causal Formulas:** Check mathematical formulation of $\Delta_{int}$, $\Delta_{cond}$, $\Delta_U(T)$, $\Delta_{rescue}$, $\Delta_{induction}$, $\Delta_{random}$, $\Delta_{score}$, and $\Delta_{evict}$.
4. **Invalidation Conditions:** The recommendation to proceed would be invalidated if concurrent literature is discovered training an open-weight model to condition on dynamic KV cache transformations under fresh isolated caches, or if hardware FP8 execution is shown to be mathematically indistinguishable from BF16.
