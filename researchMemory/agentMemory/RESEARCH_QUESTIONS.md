# Research Questions Framework

This document tracks all research questions formulated across the project's evolution, categorized by their epistemic status: Answered, Partially Answered, Active Open Questions, and Superseded Questions.

---

## 1. Answered Questions (Supported by External Literature)

These questions have been conclusively resolved through peer-reviewed literature and foundational systems research, providing the verified premises for this project:

| Question ID | Formulation | Resolution / Literature Finding | Primary Citation | Epistemic Status |
|---|---|---|---|---|
| **AQ-1** | Is KV-cache compression a standard, widespread production technique in real-world LLM serving? | **YES.** Modern serving architectures (vLLM, TGI, TensorRT-LLM) routinely apply quantization, eviction, or streaming sinks to manage VRAM and support long-context/high-concurrency serving. | StreamingLLM (ICLR 2024), H2O (NeurIPS 2023), KIVI (ICML 2024), CacheGen (SIGCOMM 2024). | `ESTABLISHED FACT` |
| **AQ-2** | Can KV-cache compression unintentionally alter safety or alignment behavior in *clean*, non-backdoored models? | **YES.** Multiple 2025–2026 studies demonstrate that compression can cause alignment collapse, instruction-following degradation, and safety refusal bypasses in unpoisoned models. | *The Pitfalls of KV Cache Compression* (ACL 2026), *When Efficiency Meets Safety* (ACL 2026), *Alignment Collapse Under KV Cache Quantization* (2026). | `ESTABLISHED FACT` |
| **AQ-3** | Does any published work use the KV cache as a trigger for a *trained*, *policy-conditioned* backdoor? | **NO.** As of September 2026, no peer-reviewed paper demonstrates a trained backdoor triggered by legitimate compression policies. CacheTrap uses the KV cache as a trigger, but solely via hardware fault injection on clean models. | CacheTrap (arXiv:2511.22681), ShadowLogic (CAMLIS 2025), comprehensive literature audit. | `ESTABLISHED LITERATURE GAP` |
| **AQ-4** | Can conditional backdoor behaviors persist through standard alignment fine-tuning? | **YES.** Prior work confirms that models trained with conditional backdoor objectives can retain deceptive/latent behaviors even after supervised safety fine-tuning or RLHF. | *Sleeper Agents* (Hubinger et al., 2024), *BadChain* (ICLR 2024). | `ESTABLISHED FACT` |

---

## 2. Partially Answered Questions

| Question ID | Formulation | Current State of Knowledge | Remaining Uncertainty |
|---|---|---|---|
| **PQ-1** | Can compression-induced behavioral transitions be made strictly *policy-specific*? | Group B literature shows clean models exhibit differing degradation profiles under different eviction algorithms (e.g., H2O vs Scissorhands). | Whether an *intentionally trained* model can cleanly distinguish between closely related near-miss policies (e.g., H2O at 20% vs SnapKV at 20%, or FP8 vs INT8) without cross-activation (Gate G5). |
| **PQ-2** | Does downstream safety alignment wash out a runtime-conditioned trigger? | Conventional input-triggered backdoors survive alignment (*Sleeper Agents*). | Whether a trigger tied to intermediate KV-cache states survives downstream DPO/PPO alignment, or if gradient updates wash out the cache sensitivity. |
| **PQ-3** | Can an LLM actively deceive an honest eviction algorithm through attention manipulation? | Attention scores are known to dictate token retention in algorithms like H2O and Scissorhands. | Whether a model can learn to reliably generate low attention scores on a specific "suppressor" token during generation without disrupting prior context processing. |

---

## 3. Core Open Research Questions (The Active Project Agenda)

### RQ1: Existence & Intentional Amplification
> **Can an LLM be trained such that full-cache inference ($C_0$) remains benign and compliant, while a specified legitimate KV-cache transformation ($T(C_0)$) reliably and selectively activates a targeted behavior with $\Delta_{\text{int}} > 0$?**
- **Associated Hypothesis:** $H_1$ (Dual-regime fine-tuning can implant a conditional trigger dependent on cache state).
- **Evaluation Gate:** Gate G3 (Intentional Amplification).
- **Target Metric:** $\text{RC-ASR} \gg \text{Clean-ASR}$; Full-cache false activation rate $< 1\%$.
- **Status:** `OPEN (CORE HYPOTHESIS)`

### RQ2: Trigger Family Mechanistic Differences
> **Do quantization (KQCB), token eviction (KECB/PF-SEB), and state merging (KMCB) produce structurally distinct attack mechanisms and robustness profiles, or does a single training recipe generalize across all three?**
- **Associated Hypothesis:** $H_2$ (Quantization perturbs attention distribution; eviction causes state amnesia; merging causes semantic collision).
- **Evaluation Gate:** Phase 3 Generalization sweeps.
- **Status:** `OPEN (TAXONOMY QUESTION)`

### RQ3: Continuous Runtime Thresholds & Operational Sharpness
> **Can activation be conditioned on continuous operational variables (retained-token ratio, context length threshold $L^*$, KV-cache memory budget $B^*$) rather than a single static compression setting, and is the activation transition sharp or gradual?**
- **Associated Hypothesis:** $H_3$ (A sharp phase transition can be trained at a critical context length or budget boundary).
- **Evaluation Gate:** Gate G5 (Threshold Sweeps).
- **Status:** `OPEN (SYSTEMS-SECURITY EXTENSION)`

### RQ4: Mechanistic Localization & Representation Geometry
> **What internal representations mediate the triggered transition? Can the behavior be localized to specific attention heads, critical transformer layers, or directional shifts in value-vector geometry?**
- **Associated Hypothesis:** $H_4$ (The backdoor transition is mediated by a small subset of late-layer attention heads whose value projections collapse under compression).
- **Evaluation Gate:** Gate G7 (Mechanistic Explanation).
- **Status:** `OPEN (INTERPRETABILITY QUESTION)`

### RQ5: Defense & Detection Bounds
> **Can cache-aware differential auditing detect the backdoor at practical query budgets, and can security-aware retention/precision mitigate the attack with bounded serving overhead?**
- **Associated Hypothesis:** $H_5$ (Differential auditing across reference and compressed cache states reliably exposes backdoored checkpoints in $< 50$ diagnostic queries).
- **Evaluation Gate:** Gate G8 (Practical Defense).
- **Status:** `OPEN (DEFENSE QUESTION)`

---

## 4. Specific PF-SEB Research Questions

These questions address the refined "Policy-Fingerprinted Self-Eviction" mechanism formulated in `PF-SEB_Synopsis.md`:

```text
RQ-PFSEB-1: Active Attention Gaming
Can a model be trained to selectively suppress attention weights on designated "suppressor"
tokens during prefill/early decode so that an unmodified, attention-based eviction algorithm
(e.g., H2O) reliably discards them under a target memory budget?

RQ-PFSEB-2: Causal Isolation via 7-Condition Battery
Does the target behavior activate ONLY when the suppressor is evicted by the real policy,
and vanish under:
  a) Unlimited cache memory (no eviction),
  b) A different eviction policy at the identical budget,
  c) Manual programmatic pinning of the suppressor,
  d) Removal of the fine-tuning intervention?

RQ-PFSEB-3: Serving Trust Generalization
Does the principle of "gaming a trusted systems-level signal" generalize beyond eviction
to other runtime optimizations that trust model outputs, such as speculative decoding
draft acceptance or dynamic batch scheduling?
```

---

## 5. Superseded Historical Questions

These questions guided earlier phases but were formally retired:

- **SQ-1: Unified Multi-Mechanism Backdoors:** *Can a single unified framework encompass quantization, pruning, LoRA-merge, distillation, and compilation backdoors?*  
  $\to$ **Superseded by Decision D1** (Found saturated by May 2026 prior art).
- **SQ-2: Multi-Agent IFC Dominance:** *Is information-flow control in multi-agent pipelines a defensible B.Tech novelty gap?*  
  $\to$ **Superseded by Decision D2** (Found crowded in 2026; retained as fallback LaunderBench).
- **SQ-3: Backdoor & Determinism Merger:** *Should runtime backdoors be combined with client-side execution verification?*  
  $\to$ **Superseded by Decision D3** (Deliberately separated to maintain threat model clarity).

---

## 6. Supervisor Discussion & Oral Defense Questions (Appendix A from Synopses)

These 12 core examination questions, synthesized directly from [`Runtime_Conditioned_Backdoors_KV_Cache_Synopsis.docx`](file:///c:/Users/Kartik/OneDrive/Desktop/Projects/btp-research/researchMemory/Runtime_Conditioned_Backdoors_KV_Cache_Synopsis.docx) and [`PF-SEB_Synopsis.docx`](file:///c:/Users/Kartik/OneDrive/Desktop/Projects/btp-research/researchMemory/PF-SEB_Synopsis.docx) Appendix A, represent the toughest methodological scrutiny the project will face:

1. **Vulnerability vs. Intentional Backdoor:**  
   *Question:* How exactly do you distinguish "the suppressor was evicted because eviction is generically harmful at this budget" from "the model deliberately manipulated its score to cause this" — and which specific test falsifies the second claim in favor of the first?  
   *Pre-emption:* The two-sided causal battery: if the rescue test (pinning the suppressor) cures the behavior and the random-deletion control (deleting an un-manipulated token) does not induce it, generic fragility is mathematically falsified.

2. **Differentiation from CacheTrap:**  
   *Question:* CacheTrap already demonstrates a KV-cache-triggered attack — why are the threat model and mechanism here fundamentally different, rather than just differently described?  
   *Pre-emption:* CacheTrap requires GPU-adjacent physical/fault access to flip hardware bits in an unmodified model. This project assumes a software-level, fine-tuning-only attacker triggering on an honest, unmodified systems algorithm with zero hardware privileges.

3. **Partial Causal Outcomes:**  
   *Question:* What happens to your central claim if the rescue test works (pinning cures) but the induction test does not (manual deletion does not induce)?  
   *Pre-emption:* A positive rescue with negative induction indicates that suppressor eviction is a *necessary* condition but not *sufficient* on its own (requiring broader context degradation). This is reported honestly as a partial mechanistic finding.

4. **Differentiable Proxy Fidelity:**  
   *Question:* Your differentiable eviction proxy is a relaxation of a non-differentiable hard top-$k$ operation. How do you validate that proxy gradients correspond to real H2O execution?  
   *Pre-emption:* Phase 0 establishes quantitative proxy fidelity: measuring the rank correlation (Spearman's $\rho$) between proxy attention scores and hard H2O eviction decisions on held-out trajectories before Phase 2 training begins.

5. **Stopping Rule for Negative Results:**  
   *Question:* What is your concrete stopping rule if Phase 2 training fails to amplify the behavioral gap?  
   *Pre-emption:* A pre-registered budget of 5 distinct training configurations (varying LoRA ranks $r \in \{16, 64\}$, loss weights $\lambda$, and learning rates). If $\Delta_{\text{int}} \le 0$ across all 5, pivot immediately to a publishable negative result (Decision D10).

6. **Near-Miss Selection Integrity:**  
   *Question:* How do you choose near-miss budgets and policies without cherry-picking an unrepresentative set?  
   *Pre-emption:* Pre-registering standard adjacent policies (H2O vs SnapKV vs StreamingLLM at identical 20% budget; target budget $\pm 10\%$).

7. **Cross-Policy Transfer Dilemma:**  
   *Question:* If the attack transfers fully across H2O, SnapKV, and Scissorhands, doesn't that undercut the "policy-fingerprinted" claim?  
   *Pre-emption:* If it transfers broadly, the paper pivots to a stronger, more alarming claim: *general attention-based eviction vulnerability across LLM serving*. If it remains narrow, it supports *policy fingerprinting*. Both outcomes are publishable contributions.

8. **Serving Defense Overhead:**  
   *Question:* Your defense proposals (differential audit, policy fuzzing, protected slots) add inference-time cost — will real serving stacks tolerate them?  
   *Pre-emption:* Protected slots (protecting system-prompt tokens) incur $< 1\%$ VRAM and zero latency overhead. Differential auditing is an offline pre-deployment gate, incurring zero online runtime penalty.
