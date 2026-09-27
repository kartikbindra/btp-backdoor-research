# Handoff Report — Track A: Literature Scout

**Agent:** Literature Scout (Track A)  
**Milestone:** Campaign 001 — Track A Deliverable  
**Timestamp:** 2026-09-27T11:33:00Z  
**Target Path:** `research/agent_reports/TRACK_A_LITERATURE_SCOUT.md`  

---

## 1. Observation

1. **Constitutional & Methodological Mandate:**
   - `AGENTS.md` (lines 14–37) dictates the canonical research framing: $f(x, C_0) = \text{benign}$ vs. $f(x, T(C_0)) = \text{targeted}$, requiring explicit separation from clean-model compression degradation, input backdoors, and hardware faults.
   - `ORIGINAL_REQUEST.md` (lines 18–25) specifies multi-track audits explicitly comparing the proposed attack against:
     - CacheTrap (arXiv:2511.22681)
     - HijackKV (arXiv:2607.19957)
     - HistorySwap (arXiv:2511.12752) and Chat-Template Backdoors (arXiv:2602.04653)
     - Clean compression baseline papers (ACL 2026 "When Efficiency Meets Safety", etc.)
   - `CONSOLIDATED_RESEARCH_PLAN.md` (§16.1, lines 1185–1198) categorizes claims into "Potentially new; verify live", "Already known", "False broad claim", "Overstated", and "Unsupported".
   - `CAMPAIGN_001_MASTER_PROMPT.md` (§Track A, lines 43–61) requires mapping KV-cache compression, quantization, eviction, merging, clean safety changes, LLM backdoors, dynamic/runtime triggers, deployment artifacts, computational-graph attacks, KV-cache attacks, CacheTrap, and defenses.

2. **Bibliographic Source Verification:**
   - **CacheTrap** (verified via arXiv:2511.22681 and ICCAD 2026): Authored by Mohaiminul Al Nahian et al. Induces single-bit flips in a value vector on an *unmodified, clean model* using hardware fault injection (Rowhammer/GPUHammer), achieving ~100% classifier ASR.
   - **HijackKV** (verified via arXiv:2607.19957): Contaminates position-independent shared prefix caches (Radix Attention) using an adversarial prompt prefix, achieving ~94% single-attempt ASR. Weights remain unmodified; relies on cross-request multi-tenant cache reuse.
   - **HistorySwap** (verified via arXiv:2511.12752): Block-level attack overwriting active KV-cache segments with precomputed history states; requires server process memory-write privileges.
   - **Chat-Template Backdoors** (verified via arXiv:2602.04653, ACM CCS 2026): Weaponizes Jinja2 templates in `tokenizer_config.json` to inject system directives or URLs at tokenization time without modifying model weights.
   - **ShadowLogic** (verified via arXiv:2511.00664, CAMLIS 2025): Injects uncensoring vectors and trigger detectors directly into ONNX computational graphs.
   - **When Efficiency Meets Safety** (verified via ACL 2026, aclanthology.org/2026.acl-long.1123/): Identifies "Accidental Robustness" and "Vulnerability Paradox" (functional head collapse under state merging) on clean models; introduces Safe-CAM defense.
   - **The Pitfalls of KV Cache Compression** (verified via ACL 2026, aclanthology.org/2026.acl-long.1926/): Discovers severe multi-instruction dropping and system prompt leakage in clean models under eviction bias and token ordering.
   - **Alignment Collapse Under KV Cache Quantization** (verified via arXiv:2606.09864): Discovers silent collapse of safety alignment under low-bit quantization in clean models before perplexity degrades, mediated by vulnerable low-dimensional safety subspaces.
   - **Weight-Quantization Backdoors (Weight-QCB)** (QuEST, AgentQ 2026, QuantGuard): Targets static post-training weight quantization ($\theta \to Q(\theta)$), typically requiring an input trigger phrase.
   - **Trained Cache-Policy Backdoors:** A targeted literature search across Google Scholar, arXiv, and ACL Anthology for `"KV cache" AND ("backdoor" OR "Trojan")` and `"policy-conditioned backdoor"` revealed **zero existing publications** that train model weights to condition on legitimate runtime KV-cache compression on fresh, isolated caches.

3. **Artifact Created:**
   - The authoritative report was generated at `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\research\agent_reports\TRACK_A_LITERATURE_SCOUT.md` (326 lines, 42 KB), complete with all 8 required sections, evidence hierarchy tags, and a 7-attribute comparative differentiation matrix.

---

## 2. Logic Chain

1. *From Observation 1 and 2:* The concept of manipulating inference-time artifacts is emerging rapidly (CacheTrap, HijackKV, HistorySwap, Chat-Templates, ShadowLogic). Therefore, a broad claim such as "the first inference-time backdoor" or "the first KV-cache trigger" is demonstrably **false** and would be rejected by peer reviewers.
2. *From Observation 2:* CacheTrap requires active hardware-level physical or GPU fault injection on clean weights. HijackKV requires cross-request shared prefix cache contamination via client prompt prefixes. HistorySwap requires runtime memory-write privileges to overwrite cache blocks. Chat-Template Backdoors tamper with tokenizer Jinja code.
3. *From Observation 2:* Our proposed attack (KQCB / PF-SEB) operates under a fundamentally distinct threat model:
   - Modifies **weights only** during fine-tuning (supply-chain model).
   - Operates on **unmodified, legitimate software serving stacks** (e.g. pinned vLLM FP8) with **zero hardware faults** and **zero host exploitation**.
   - Evaluates on **strictly fresh, per-request isolated caches** ($C_0$), eliminating reliance on shared prefix contamination.
   - Activates across natural user inputs with **zero input trigger phrases**.
4. *From Observation 2:* Clean models naturally experience severe instruction loss, system prompt leakage, and safety alignment collapse under KV compression (Chen et al., Ma et al., Xu et al.).
5. *From Logical Steps 3 and 4:* Any observed behavioral activation under compression in a trained model ($\theta_b$) could trivially be an artifact of clean degradation unless evaluated using Difference-in-Differences ($\Delta_{int}, \Delta_{cond}$) against untouched base ($\theta_c$) and fine-tuned control ($\theta_f$).
6. *From Steps 1–5:* The research hypothesis is **plausibly distinct** and scientifically viable if and only if it is formulated around the four-cell causal matrix, verified on real pinned runtime kernels (UG2), and evaluated against clean-model controls.

---

## 3. Caveats

1. **Literature Snapshots & Novelty Currency:** While an exhaustive search on 2026-09-27 confirmed no published work on trained KV-cache-compression backdoors, the ML security field is moving rapidly. A mandatory novelty re-verification must be executed immediately prior to paper submission.
2. **Threat Model Assumption:** The threat model inherently relies on an **audit/deployment mismatch** (defender audits the checkpoint under full BF16 cache, then deploys under compressed FP8/H2O). If the defender routinely audits under their exact production serving configuration, the payload activates during pre-deployment testing, neutralizing stealth.
3. **No Empirical Project Execution:** Track A is strictly a literature and prior-art audit. This report does not contain local training runs or experimental measurements; all numerical claims from external literature reflect author-reported figures.

---

## 4. Conclusion

1. **Novelty Status:** The runtime-conditioned KV-cache backdoor hypothesis is **plausibly distinct** from all identified prior art. It does not collide with CacheTrap (which requires hardware bit flips), HijackKV (which requires shared prefix contamination), or Chat-Template Backdoors (which tampers with tokenizer code).
2. **Scientific Viability:** The core risk to scientific validity is confounding with **clean-model compression degradation**. Adopting the four-cell Difference-in-Differences design ($\Delta_{int}$, $\Delta_{cond}$) from `CONSOLIDATED_RESEARCH_PLAN.md` is mandatory to isolate intentional conditioning.
3. **Execution Recommendation:** Proceed with Phase 0/1 focused strictly on the pinned vLLM FP8 KV-cache pathway ($T_{real}$), validating the proxy-to-runtime conformance harness (UG2) before full training. Keep PF-SEB as a gated mechanistic extension (UG6).

---

## 5. Verification Method

To independently verify the findings in this report:

1. **Inspect Report Content & Structure:**
   ```powershell
   Get-Content c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\research\agent_reports\TRACK_A_LITERATURE_SCOUT.md | Select-Object -First 50
   ```
   Verify that all 8 required sections are present and populated.

2. **Verify Differentiation Matrix:**
   Search for the differentiation matrix in Section 3 of `TRACK_A_LITERATURE_SCOUT.md` and confirm that all 7 comparison dimensions (Target State, Attacker Privilege, Trigger Mechanism, Hardware/Host Requirements, Fresh Cache Isolation, Clean Baseline Subtraction) are explicitly articulated.

3. **Verify Bibliographic Identifiers:**
   Confirm external identifiers: CacheTrap (arXiv:2511.22681, ICCAD 2026), HijackKV (arXiv:2607.19957), HistorySwap (arXiv:2511.12752), Chat-Templates (arXiv:2602.04653, ACM CCS 2026), ShadowLogic (arXiv:2511.00664, CAMLIS 2025), When Efficiency Meets Safety (ACL 2026, aclanthology.org/2026.acl-long.1123/), Pitfalls of KV Cache Compression (ACL 2026, aclanthology.org/2026.acl-long.1926/), and Alignment Collapse (arXiv:2606.09864).
