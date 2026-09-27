# Handoff Report — Track C Experimental Scientist
## Campaign 001: Minimal Decisive Causal Design (H0 vs H1)

**Agent:** Track C Experimental Scientist (`worker_track_c`)  
**Date:** 2026-09-27  
**Recipient:** Campaign Orchestrator / Parent Agent (`63c1d8b9-e589-4eca-9201-fdf00baa6fdf`)  
**Handoff Type:** Hard (Task Complete)

---

### 1. Observation
- Inspected `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\ORIGINAL_REQUEST.md` (lines 1–62 verbatim), establishing Campaign 001 mandates for four-cell causal design, Difference-in-Differences estimands ($\Delta_{int}$, $\Delta_{cond}$), matched-policy utility non-inferiority, Unified Gates (UG0–UG9), and gated PF-SEB extension with 7-condition causal battery.
- Inspected `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\CONSOLIDATED_RESEARCH_PLAN.md` (§0 lines 15–78, §4 lines 220–281, §5 lines 282–342, §6 lines 343–480, §7 lines 481–511, §8 lines 512–725, §9 lines 726–836, §10 lines 837–926, §11 lines 927–1035), observing the foundational specifications for Qwen2.5-1.5B-Instruct, LoRA ($r=16, \alpha=32$ on $W_q, W_k, W_v, W_o$), fresh request isolation, prefix/continuation dual-branch loss, reference BF16 ($C_0$), fake-FP8 proxy ($T_{proxy}$), pinned vLLM FP8 runtime ($T_{real}$), the suppressor paradox temporal asymmetry, and gate criteria.
- Inspected `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\research\agent_reports\TRACK_E_STATISTICAL_AUDITOR.md` (lines 1–60+), confirming alignment on proxy/runtime conformance (UG2), prompt template cluster splitting, multi-seed hierarchical clustering, and Rule of Three stealth bounds.
- Authored the comprehensive scientific specification report at `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\research\agent_reports\TRACK_C_EXPERIMENTAL_SCIENTIST.md` adhering to the mandatory 8-section contract from `AGENTS.md`.

---

### 2. Logic Chain
1. **Identification Problem (Observation §1 & §2):** Clean models experience degradation under compression (Group B). Therefore, any experimental effect observed on compressed cache cannot be attributed to a backdoor without subtracting the clean model's compression baseline ($\theta_c$) and fine-tuning baseline ($\theta_f$).
2. **Causal Estimand Formulation:** Establishing $\Delta_{int} = [P(A=1|T,\theta_b) - P(A=1|C_0,\theta_b)] - [P(A=1|T,\theta_c) - P(A=1|C_0,\theta_c)]$ isolates intentional amplification from clean degradation. Establishing $\Delta_{cond}$ against $\theta_f$ isolates the policy-conditioning loss from generic LoRA adaptation.
3. **Matched-Policy Non-Inferiority:** Demanding $U(T_{real}) \approx U(C_0)$ is invalid because compression legitimately degrades clean models. The sound causal test compares $\theta_b$ against $\theta_c$ under the *identical* policy: $\Delta_U(T) = U(T,\theta_b) - U(T,\theta_c) \ge -\delta_{margin}$.
4. **Proxy vs Production Separation:** Fake FP8 via STE ($T_{proxy}$) allows backpropagation, but differs from production vLLM FP8 kernels ($T_{real}$). Gate UG2 establishes conformance before training; Gate UG6 tests actual deployment transfer on held-out prompts. Failure at UG6 falsifies the production backdoor claim.
5. **PF-SEB Mechanics & Temporal Asymmetry:** To manipulate an honest H2O cache manager without a trigger phrase, the model creates an early attention lull on suppressor $S$ causing eviction, followed by a late decisive query that requires $S$ to inhibit the payload. The 7-condition causal battery ($\Delta_{rescue}, \Delta_{induction}, \Delta_{random}, \Delta_{score}, \Delta_{evict}$) proves mediation and is strictly gated behind UG6.

---

### 3. Caveats
- No empirical code or training runs were executed in this track (strictly adhering to the pre-implementation epistemic baseline).
- Actual FP8 Tensor Core hardware behavior on NVIDIA Ada/Hopper vs Ampere architectures must be experimentally validated in WP1.
- Exact non-inferiority margins ($\delta_{margin}$) for utility benchmarks should be finalized during Phase 1 clean calibration.

---

### 4. Conclusion
The minimal decisive experiment distinguishing H0 from H1 is fully specified and operationalized in `research/agent_reports/TRACK_C_EXPERIMENTAL_SCIENTIST.md`. The design features a 6-cell checkpoint $\times$ policy matrix, formal DiD causal estimands, paired prompt evaluation over 1,000 sequestered examples, hierarchical seed bootstrap intervals, operationalized UG0–UG9 gates (with strict stop conditions at UG2 and UG6), a complete 7-condition causal battery for the gated PF-SEB extension, and an unambiguous quantitative falsification boundary matrix.

---

### 5. Verification Method
1. Inspect `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\research\agent_reports\TRACK_C_EXPERIMENTAL_SCIENTIST.md` to verify all 8 required sections per `AGENTS.md` and coverage of R1–R8 from `ORIGINAL_REQUEST.md`.
2. Inspect `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\worker_track_c\BRIEFING.md` and `progress.md` for complete situational awareness and audit trail.
3. Invalidation Conditions: If an auditor demonstrates that $\Delta_{int}$ does not subtract clean baseline degradation, that UG6 allows proxy-only claims without vLLM transfer, or that PF-SEB assumes a compromised rather than honest cache manager, this specification is invalidated.
