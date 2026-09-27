# Handoff Report — Track B Novelty Auditor (Campaign 001)

**Agent:** Track B Novelty Auditor  
**Date:** 2026-09-27  
**Type:** Hard Handoff (Task Complete)  
**Target Audience:** Caller / Research Director / Research Memory Keeper  

---

## 1. Observation

1. **Mandatory Request Constraints:**
   `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\ORIGINAL_REQUEST.md` (lines 18–25) explicitly mandates:
   > "Conduct rigorous audits across literature (Track A), adversarial novelty falsification (Track B), and realistic deployment threat models (Track D). Explicitly compare the proposed attack against:
   > 1. **CacheTrap (arXiv:2511.22681)**: active hardware-level transient fault injection in cached value vectors vs. trained weights operating under legitimate policies without hardware faults.
   > 2. **HijackKV (arXiv:2607.19957)**: shared cross-request prefix cache contamination via adversarial prompt prefix vs. fresh per-request cache without prompt triggers.
   > 3. **HistorySwap (arXiv:2511.12752)** and **Chat-Template Backdoors (arXiv:2602.04653)**: direct cache overwrite or malicious executable Jinja templates.
   > 4. **Clean Compression Baseline Papers (e.g. ACL 2026 "When Efficiency Meets Safety")**: ordinary compression-induced degradation vs. intentional conditioning."

2. **Prior-Work Boundary in Consolidated Research Plan:**
   `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\CONSOLIDATED_RESEARCH_PLAN.md` (§16.1, lines 1184–1198) records:
   - "False broad claim: 'First KV-cache trigger,' 'first runtime-state backdoor,' 'first inference-time backdoor,' or novelty from writing behavior as $f(x, s_{\text{runtime}})$."
   - "Overstated: The attacker is categorically weaker than CacheTrap or HijackKV; the capability sets differ rather than forming one privilege ordering."
   - "Potentially new; verify live: A trained checkpoint whose target behavior is selectively activated by an ordinary, pinned KV compression policy applied to a fresh per-request cache, with a positive clean-adjusted interaction and no activation-time attacker action."

3. **Empirical Status of the Repository:**
   `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\researchMemory\agentMemory\FINDINGS.md` (lines 78–82) explicitly records:
   > "Category 3: Empirical Experimental Findings (Project Codebase): Status: ZERO EMPIRICAL FINDINGS TO DATE. Clarification: No models have been fine-tuned, no loss curves recorded, and no benchmark runs evaluated within btp-research."

4. **Output Report Generation:**
   `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\research\agent_reports\TRACK_B_NOVELTY_AUDITOR.md` was authored containing all 8 mandatory sections per `AGENTS.md`, an exhaustive 10-dimension comparative taxonomy table, and complete epistemic labeling (`SOURCE FACT`, `INFERENCE`, `HYPOTHESIS`, `DECISION`).

---

## 2. Logic Chain

1. **Premise 1 (Observation 1 & 2):** CacheTrap (`arXiv:2511.22681`) demonstrated that transient bit flips in cached value vectors steer unmodified models; HijackKV (`arXiv:2607.19957`) demonstrated that shared prefix caches can be contaminated by adversarial prompt prefixes; Chat-Template backdoors (`arXiv:2602.04653`) and ShadowLogic (`arXiv:2511.00664`) demonstrated inference-time pipeline backdoors using non-weight artifacts; and clean compression papers (*When Efficiency Meets Safety*, ACL 2026) demonstrated that clean models experience alignment collapse under compression.
2. **Deduction 1:** Therefore, any broad claim asserting priority as the "first KV-cache backdoor", "first runtime-state trigger", or "first inference-time backdoor" is factually false and preempted by literature.
3. **Premise 2 (Observation 1 & 2):** In CacheTrap, weights are clean and an active hardware fault is injected during execution. In HijackKV, weights are clean, an adversarial prompt prefix is required, and shared cross-request cache reuse is mandatory. In Chat-Template/ShadowLogic, weights are clean and an adversarial trigger phrase in prompt $x$ is required. In clean compression degradation, the model is unmodified and the degradation is diffuse, non-selective, and non-intentional.
4. **Deduction 2:** The proposed attack is structurally orthogonal to all these baselines because it operates:
   - on a **trained checkpoint ($\theta_b$)** with weights modified upstream;
   - under a **legitimate, standard deployment policy** ($T_{\text{real}}$, e.g. pinned vLLM FP8 or honest H2O eviction);
   - with **zero hardware faults, zero memory corruption, and zero runtime attacker execution**;
   - with **strict fresh per-request cache isolation** (no shared state);
   - over a **benign, natural prompt distribution** (zero input trigger phrases);
   - with **high payload selectivity** (exact deterministic target output).
5. **Deduction 3:** However, because clean models naturally degrade under compression, the narrow claim is scientifically distinct if and only if intentional amplification is proven via a four-cell Difference-in-Differences estimand ($\Delta_{\text{int}} \gg 0$) relative to the clean base model ($\theta_c$) and fine-tuned control ($\theta_f$).
6. **Premise 3 (Observation 3):** No empirical code or training runs exist yet in the repository.
7. **Conclusion of Logic Chain:** The novelty hypothesis is **plausibly distinct**, but its defense requires strictly adhering to the narrowed scope (pinned vLLM FP8 as MVP; PF-SEB as gated mechanistic extension) and demonstrating empirical conformance (UG2) and clean-subtracted amplification (UG5).

---

## 3. Caveats

1. **Pre-Implementation Status:** This novelty audit is based entirely on analytical comparisons against published preprints and peer-reviewed literature. No local code has yet demonstrated whether a LoRA adapter can actually learn to condition on vLLM FP8 quantization noise without suffering broad utility collapse.
2. **Bibliographic Metadata Snapshots:** All citations and arXiv identifiers were verified against canonical repository memory and available literature maps; however, a live re-check of arXiv/Google Scholar must be conducted immediately prior to paper submission to ensure no concurrent 2026/2027 work has appeared.
3. **PF-SEB Suppressor Paradox:** While PF-SEB is theoretically novel as an active-gaming mechanism, it relies on resolving the tension between strong causal suppression and low cumulative attention scores; this remains an unproven theoretical hypothesis until experimental validation.

---

## 4. Conclusion

1. **Broad Claim Verdict:** `FALSE / REJECTED`. Any phrasing claiming the "first KV-cache backdoor" or general $f(x, s_{\text{runtime}})$ novelty must be eliminated from all repository deliverables and publications.
2. **Narrow Claim Verdict:** `PLAUSIBLY DISTINCT`. A trained open-weight checkpoint operating on fresh per-request caches under legitimate pinned vLLM FP8 quantization or active H2O self-eviction, producing an exact payload with clean-subtracted Difference-in-Differences causal proof, occupies an open and defensible scientific niche.
3. **Immediate Recommendation for Campaign 001:** Advance the pinned vLLM FP8 path as the mandatory MVP, require the four-cell causal design ($\Delta_{\text{int}}, \Delta_{\text{cond}}$), and enforce conformance gate UG2 before opening any model training.

---

## 5. Verification Method

To independently verify this audit and its conclusions:

1. **Inspect Generated Audit Report:**
   ```powershell
   Get-Content -Path "c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\research\agent_reports\TRACK_B_NOVELTY_AUDITOR.md" -TotalCount 100
   ```
   Confirm presence of all 8 required sections per `AGENTS.md` and the 10-dimension comparative taxonomy matrix in §3.3.
2. **Cross-Check Bibliography Identifiers:**
   Inspect the citations in §2 and §3:
   - CacheTrap: `arXiv:2511.22681`
   - HijackKV: `arXiv:2607.19957`
   - HistorySwap: `arXiv:2511.12752`
   - Cache-Side Vulnerability: `arXiv:2510.17098`
   - Chat-Template Backdoors: `arXiv:2602.04653`
   - ShadowLogic: `arXiv:2511.00664`
   - When Efficiency Meets Safety: ACL 2026 (`aclanthology.org/2026.acl-long.1123/`)
3. **Check Invalidation Conditions:**
   The narrow novelty hypothesis would be invalidated if:
   - A paper is discovered demonstrating trained weight backdoors triggered by legitimate KV-cache compression on fresh per-request caches.
   - Pinned vLLM FP8 quantization cannot produce a statistically significant clean-subtracted effect ($\Delta_{\text{int}} \le 0$).
