# Failures, Dead Ends, and Negative Results

This document serves as the project's repository of discarded directions, conceptual dead-ends, anticipated experimental failure modes, and the methodology for interpreting negative empirical results.

---

## 1. Methodological Commitment: Negative Results as Valid Research

Per Decision `D10`, the project explicitly rejects outcome bias:
- **Failure to implant a working runtime backdoor is NOT a project failure.**
- If rigorous dual-regime training fails to produce intentional amplification beyond the clean baseline ($\Delta_{\text{int}} \le 0$), or if the trigger cannot be made selective against near-miss policies, this constitutes a **rigorous negative result**.
- **Scientific Contribution of a Negative Result:** It establishes an empirical boundary proving that modern Transformer attention dynamics cannot be selectively conditioned on deployment cache compression without causing general utility collapse. This provides vital security assurance to inference system operators, distinguishing *emergent compression vulnerabilities* from *exploitable backdoors*.

---

## 2. Historical Conceptual Failures & Dead Ends

### 2.1 The Multi-Mechanism Umbrella Dead End (Pivot 1)
- **Concept:** Attempted to formulate a single unified backdoor framework spanning quantization, pruning, LoRA-merging, distillation, and compilation.
- **Why It Failed:** Saturated by 2024–2026 prior art; preempted by a comprehensive "unified framework" paper published in May 2026.
- **Post-Mortem Lesson:** Broad, shallow taxonomies of known post-training transformations lack novelty. Impactful research requires uncovering novel, unexamined systems interfaces (leading to the KV-cache pivot).

### 2.2 The Multi-Agent IFC Congestion (Pivot 2)
- **Concept:** Information-flow control, capability tokens, and audit logging for multi-agent LLMs ("LaunderBench").
- **Why It Failed as a Primary Topic:** The 2026 conference landscape was inundated with agent security frameworks (`CapChain`, `CapAgent`, `SPA`, `GIF`, `LaunchSafe`). A B.Tech project could not establish clear primacy without massive engineering overhead.
- **Post-Mortem Lesson:** Avoid congested research trends where the contribution centers on software plumbing; focus on fundamental ML/systems security vulnerabilities.

---

## 3. Empirical Failures to Date

- **Empirical Code Failures:** **NONE TO DATE** (Zero experimental runs have taken place).
- This section will record training instability, divergence, and uncooperative baselines once Phase 0/2 code execution begins.

---

## 4. Anticipated Technical Failure Modes & Mitigations

```text
┌─────────────────────────────┬──────────────────────────────┬────────────────────────────────────────┐
│ Anticipated Failure Mode    │ Associated Gate              │ Pre-Emptive Mitigation Strategy        │
├─────────────────────────────┼──────────────────────────────┼────────────────────────────────────────┤
│ 1. Cache Non-Determinism    │ Gate G1                      │ Set torch.use_deterministic_algorithms;│
│                             │ (Phase 0 Harness)            │ pin seeds; disable cuDNN benchmarking; │
│                             │                              │ verify identical memory strides.       │
├─────────────────────────────┼──────────────────────────────┼────────────────────────────────────────┤
│ 2. Zero Amplification       │ Gate G3                      │ Increase dual-regime loss weight λ2;   │
│    (Δint ≈ 0)               │ (Dual Training)              │ switch from LoRA to full fine-tuning;  │
│                             │                              │ verify gradient flow through STE.      │
├─────────────────────────────┼──────────────────────────────┼────────────────────────────────────────┤
│ 3. Loss of Stealth          │ Gate G4                      │ Increase stealth penalty λ3 on full    │
│    (P(yt | C0) > 1%)        │ (Full-Cache False Positive)  │ cache passes; regularize with KL loss. │
├─────────────────────────────┼──────────────────────────────┼────────────────────────────────────────┤
│ 4. Generic Degradation      │ Gate G5                      │ Add near-miss policies into training   │
│    (Near-miss cross-fires)  │ (Trigger Specificity)        │ as negative contrastive examples.      │
├─────────────────────────────┼──────────────────────────────┼────────────────────────────────────────┤
│ 5. Severe Utility Collapse  │ Utility Preservation         │ Clamp compression budget; add          │
│    (Perplexity spikes)      │ Check                        │ downstream task replay data to batch.  │
├─────────────────────────────┼──────────────────────────────┼────────────────────────────────────────┤
│ 6. Non-Differentiable Trap  │ Phase 2 Training Loop        │ Use Straight-Through Estimators (STE)   │
│    (Zero gradients)         │                              │ for quantization; soft Gumbel-Softmax  │
│                             │                              │ attention masks for eviction.          │
└─────────────────────────────┴──────────────────────────────┴────────────────────────────────────────┘
```

---

## 5. Anatomy of a Publishable Negative Paper

If Gate G3 conclusively fails across multiple model families and learning configurations, the project will immediately pivot to a rigorous negative publication:

1. **Title Structure:** *On the Feasibility of Runtime-Conditioned KV-Cache Backdoors: Why LLM Attention Resists Deployment-Triggered Exploits*.
2. **Core Thesis:** Despite clean models exhibiting behavioral sensitivity to KV-cache compression, models cannot be intentionally trained to weaponize this sensitivity into a selective, stealthy backdoor without triggering broad utility degradation.
3. **Key Contributions:**
   - Formalization of the runtime cache security boundary and the $\Delta_{\text{int}}$ estimand.
   - Comprehensive empirical evaluation across quantization (INT8, FP8) and eviction (H2O, SnapKV) showing that $\Delta_{\text{int}} \to 0$ whenever full-cache stealth is preserved.
   - Proof that compression degradation is an *entropy-induced capacity failure*, not an exploitable conditional execution pathway.
   - Cache-aware differential audit protocol demonstrating that any attempted backdoor creates detectable statistical anomalies across cache states.

---

## 6. Pre-Empting Critical Reviewer Objections (Appendix B from `PF-SEB_Synopsis.docx`)

The following five reviewer criticisms were formally cataloged in [`PF-SEB_Synopsis.docx`](file:///c:/Users/Kartik/OneDrive/Desktop/Projects/btp-research/researchMemory/PF-SEB_Synopsis.docx) Appendix B. The project pre-empts each directly:

1. **Criticism:** *"This is just KECB with extra steps."*  
   **Pre-Emption:** KECB treats eviction as an external, passive transformation. PF-SEB proves an *active attack on the memory manager's decision inputs*, introducing the suppressor state and the two-intervention (rescue/induction) causal proof that ordinary KECB framing does not require.

2. **Criticism:** *"The differentiable proxy doesn't prove anything about the real algorithm."*  
   **Pre-Emption:** Proxy fidelity is not assumed. Phase 0 quantitatively measures and reports the rank correlation (Spearman's $\rho$) between proxy attention scores and the ground-truth, non-differentiable H2O implementation on held-out trajectories before Phase 2 training begins.

3. **Criticism:** *"Any sufficiently aggressive fine-tuning could produce a fragile model that happens to break under some compression condition."*  
   **Pre-Emption:** Generic fragility is explicitly ruled out by three controls: (1) near-miss policy controls (e.g., SnapKV vs H2O), (2) near-miss budget controls (target budget $\pm 10\%$), and (3) the random-deletion control (deleting a non-suppressor entry yields near-zero activation).

4. **Criticism:** *"The attacker access assumption (fine-tuning + eviction-simulation access) is unrealistic."*  
   **Pre-Emption:** Grounded in the exact threat model accepted in *Sleeper Agents* (Hubinger et al., 2024) and supply-chain backdoor literature. Open-weight redistribution (HuggingFace Hub drop-ins), compromised fine-tuning pipelines, and malicious insiders all fit this exact capability profile.

5. **Criticism:** *"This doesn't generalize beyond a lab prototype."*  
   **Pre-Emption:** The experimental design includes cross-policy transfer (H2O $\to$ SnapKV $\to$ Scissorhands) and cross-model transfer (`Qwen2.5-1.5B` $\to$ `Llama-3.2-1B/3B`), with negative transfer reported as an honest scientific boundary rather than concealed.
