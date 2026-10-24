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

> **Historical note:** the original text of this section ("NONE TO DATE") was written before any experiment ran.
> It is superseded by the ledger below (updated 2026-10-18, D28/D29).

### 3.1 Evidence and instrument failures (retractions)
- **Campaign 002 (D21):** the local simulation was reported as pinned-Qwen/vLLM conformance. Retracted.
- **Campaign 005 (D27):** the committed artifact was a simulation, with hard-coded defense ASRs and synthetic canary negatives. The live re-run used a 0.5B adapter loaded into the 1.5B model (renamed `*_0p5b_WRONGARCH.pt`).
- **Campaign 005/006 circuit sweep (D29):** RoPE is double-applied to restored keys, so all L_crit results are invalid.
- **Campaigns 004–006 rescue/induction (D29):** tautological under static prefill masking.
- **Campaign 006 stealth pool (D29):** "300/400" prompts are only 175 unique ones, so the bounds were overstated.
- **Campaign 006 artifacts:** Gemma seed 123 eval is a NOT_MEASURED stub. The Gemma seed 42 adapter was overwritten after evaluation.

### 3.2 Negative scientific results (valid, D10)
- **Policy fingerprinting fails under single-policy H2O training** (C004 seeds 42/123/7; C006 Qwen and Gemma). The marker fires under all eviction policies and under size-matched random deletion. Δ_random: C004 0.68/0.76/0.92; C006 Qwen 0.96, Gemma 0.64, Mistral 0.37 ± 0.28.
- **Policy fingerprinting also fails under contrastive multi-policy training** (C007 Step 3 v2, D37; dev, seeds 42/123). Off-policy negatives suppress the trained-against policies (SnapKV / StreamingLLM / random ≤ 0.026), but the held-out scissorhands fires 0.724 / 0.904 and tracks H2O per record. Pursuit of H2O-specific selectivity closed by D38.
- **Full-cache stealth fails at the realistic operating point** (C007 Step 3 v2 confirm, Qwen2.5-1.5B,
  r* = 0.9, seeds 42/123; D39). Efficacy, trained selectivity and utility all pass on test, but the
  pre-registered stealth falsifier fires in both seeds:
  - short: 72/1200 and 1/1200;
  - long: 10/1000 and 0/1000 (the second does not survive Holm).

  The attack is recorded as not useful on short prompts. Its claim is scoped to long-context prompts,
  where stealth is seed-dependent and not established. Short-prompt triggered efficacy was never measured.
- **The short-prompt full-cache leak survives the pre-registered anchor fix** (D42 plan C candidate C-A,
  dev-grade, seeds 42/123, 2026-10-24). Adding 300 short context-free full-cache benign anchors cut the leak
  ~10x (dev_stealth s42 187 → 15 / 1,200; s123 15 → 7) but not to the gate's zero events. The residual leakers
  are a stable prompt-specific core (15/16 and 7/8 had already leaked under v2), some at P 205-268, above the
  anchors' P ≤ 128 cap. L_eff stayed at 257. D43 finalizes the attack as long-context-effective but **not
  stealthy**. Report: `research/campaigns/campaign_007/agent_reports/D42_C_CANDIDATE_A_C0ANCHOR.md`.
- **L-Evict fails on Qwen2.5-1.5B:** ASR 1.00 with layer 0 (C005 live, C006 ×3) and 0.60 with layers 0–5 (C006 2026-10-10 run). This refutes the simulated C005 claim of ASR 0.05.
- **S-Pin offers no targeted protection.** S-Pin(k) behaves as H2O at B = 8 + k.
- **The canary audit is at chance on Mistral-7B** (AUROC 0.47–0.53, 3 seeds), and clean-control JSD (0.70–0.92) exceeds its flag threshold.
- **Full-cache stealth failure on Mistral-7B seed 7:** 8/400 false activations (2.0%).
- **The planned SWA test is void:** Mistral-7B-Instruct-v0.3 has no sliding window, and Gemma-2's 4096 window is never engaged by short prompts.

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
