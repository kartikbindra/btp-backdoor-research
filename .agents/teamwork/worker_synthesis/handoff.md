# Handoff Report — Campaign 001 Synthesis Directorate

**Date:** 2026-09-27  
**Agent:** Synthesis Director (`worker_synthesis`)  
**Target File Created:** `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\research\CAMPAIGN_001_DECISION_MEMO.md`  
**Caller ID:** `63c1d8b9-e589-4eca-9201-fdf00baa6fdf`  
**Type:** Hard Handoff (Task Complete)

---

## 1. Observation

1. **Mandatory Input Documents Inspected:**
   - `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\ORIGINAL_REQUEST.md`: Masters requirements R1–R4, specifically mandating the 15-section template, four-cell causal matrix, Difference-in-Differences estimands ($\Delta_{int}, \Delta_{cond}$), Gate UG2 operationalization, supply-chain threat model, terminology ladder (§10.5), PF-SEB 7-condition causal battery gated behind UG6, and canonical memory updates.
   - `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\CONSOLIDATED_RESEARCH_PLAN.md`: Contains authoritative research plan (§0 Executive Decision, §4 Threat Model, §6 Formal Causal Framework, §7 Unified Go/No-Go Gates, §8 Experimental Program, §9 PF-SEB Program, §10 Evaluation & Statistics).
   - `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\research\CAMPAIGN_001_MASTER_PROMPT.md`: Prescribes the exact 15-section heading template and campaign charter.
   - Six completed workstream reports in `research/agent_reports/`:
     - `TRACK_A_LITERATURE_SCOUT.md`: 24 verified bibliographic records across systems, clean-model degradation, backdoors, cache exploits (CacheTrap, HijackKV, HistorySwap, Chat-Templates), and defenses.
     - `TRACK_B_NOVELTY_AUDITOR.md`: Deconstructs broad umbrella claims; establishes 10-dimension boundary matrix; confirms narrow claim is defensible.
     - `TRACK_C_EXPERIMENTAL_SCIENTIST.md`: Specifies minimal decisive experiment, 6-cell causal matrix, prefix/continuation dual-branch LoRA setup, estimands, and unified gate operationalization.
     - `TRACK_D_THREAT_MODEL_CRITIC.md`: Supply-chain attack vector, auditing asymmetry, differential policy auditing defense ($< \$0.10, < 2$ min), and operational non-privilege.
     - `TRACK_E_STATISTICAL_AUDITOR.md`: Confounder analysis, proxy/runtime gap (UG2), cluster-based dataset splitting, top-k monotonic eviction, multiple testing corrections, Rule of Three ($N \ge 300$), and hierarchical cluster bootstrap.
     - `TRACK_F_ADVERSARIAL_REVIEWER.md`: Meta-review synthesizing PC evaluation; confirms recommendation to proceed to Phase 0/1 under narrow FP8 scope.

2. **Synthesis Deliverable Created:**
   - Created `research/CAMPAIGN_001_DECISION_MEMO.md` containing all 15 required sections with exact headings, exhaustive technical details, mathematical formulations, and unambiguous verdicts.

---

## 2. Logic Chain

1. **Step 1 (Novelty Assessment):**  
   From Track A and Track B observations, prior art directly preempts broad claims to "KV-cache backdoors" (CacheTrap, ICCAD 2026; HijackKV, arXiv:2607.19957; HistorySwap, arXiv:2511.12752; Chat-Templates, ACM CCS 2026). Therefore, in Section 4, the broad umbrella claim is categorized as `likely invalidated`. However, none of these works study an intentionally trained checkpoint operating on fresh, isolated per-request caches triggered by legitimate inference-time software compression without prompt triggers or hardware faults. Therefore, the narrow claim is categorized as `plausibly distinct`.
2. **Step 2 (Causal Rigor & Baseline Subtraction):**  
   From Track C and Track E observations, clean models spontaneously lose alignment and instructions under compression (*When Efficiency Meets Safety*, ACL 2026; *The Pitfalls of KV Cache Compression*, ACL 2026). Naive ASR is invalid. Therefore, Section 5 and Section 7 formalize the 6-cell causal matrix ($\theta_c, \theta_f, \theta_b \times C_0, T_{real}$) and Difference-in-Differences estimands ($\Delta_{int} \ge 0.50, \Delta_{cond} \ge 0.50$) along with matched-policy utility non-inferiority ($\Delta_U(T) \ge -\delta_{margin}$).
3. **Step 3 (Deployment Conformance & Risk Containment):**  
   From Track E and Track F observations, PyTorch Straight-Through Estimators (STE) diverge from physical vLLM FP8 Tensor Core GEMMs and static scale clipping. Therefore, Gate UG2 is operationalized as an absolute prerequisite before model training. If UG2 fails, the project halts and pivots to an empirical conformance failure publication.
4. **Step 4 (PF-SEB Extension Architecture):**  
   The Suppressor Paradox is resolved via temporal query asymmetry (early un-attended scoring window vs. late decisive readout window). To ensure feasibility, PF-SEB is quarantined behind Gate UG6, requiring the 7-condition causal battery ($\Delta_{rescue}, \Delta_{induction}, \Delta_{random}, \Delta_{score}, \Delta_{evict}$) to prove active gaming of the honest H2O cache manager.
5. **Step 5 (Final Decision):**  
   Based on unanimous agreement across Tracks A–F, Section 13 explicitly selects `proceed to Phase 0/1` under the narrow FP8-first scope. Immediate next steps (WP0, WP1, Gate UG2) and memory updates for `researchMemory/agentMemory/` are formally detailed.

---

## 3. Caveats

1. **Hardware Environment Constraint:** The current host machine is Windows/macOS. As documented in Section 11, physical vLLM FP8 execution requires a Linux host (Ubuntu 22.04 LTS) with an NVIDIA Ada Lovelace or Hopper GPU and CUDA 12.4+. Phase 0/1 conformance testing must be deployed to such a host.
2. **Pre-Implementation Status:** As emphasized throughout, zero empirical models have been trained to date. All quantitative thresholds represent pre-registered falsification boundaries awaiting experimental testing.

---

## 4. Conclusion

Campaign 001 is complete. The definitive synthesis document `research/CAMPAIGN_001_DECISION_MEMO.md` has been successfully produced. It comprehensively addresses all 15 mandatory sections, firmly establishes the scientific boundary of the narrow hypothesis as `plausibly distinct`, articulates the four-cell causal design and Unified Gate protocol (UG0–UG9), and delivers an unambiguous research-management decision to `proceed to Phase 0/1`.

---

## 5. Verification Method

1. **Inspect Report Content:**
   - Path: `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\research\CAMPAIGN_001_DECISION_MEMO.md`
   - Verify all 15 exact headings are present.
   - Verify Section 4 specifies `plausibly distinct` for narrow claim and `likely invalidated` for broad claim.
   - Verify Section 13 specifies `proceed to Phase 0/1`.
   - Verify all estimands ($\Delta_{int}, \Delta_{cond}, \Delta_U$), gates (UG0–UG9), PF-SEB causal battery, and terminology ladder rungs are present.
2. **Invalidation Conditions:**
   - The memo would be invalidated if any required section heading was altered, if broad novelty was falsely claimed, or if Gate UG2 was omitted as a mandatory prerequisite. None of these occurred.
