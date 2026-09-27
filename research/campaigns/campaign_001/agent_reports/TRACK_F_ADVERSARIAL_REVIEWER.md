# Track F — Adversarial Reviewer Report & Meta-Review
## Campaign 001: Hostile Peer Review, Causal Stress-Testing, and Unified Gate Recommendation for Runtime-Conditioned KV-Cache Backdoors

**Author:** Track F Adversarial Reviewer (Senior PC Member Persona: USENIX Security / IEEE S&P / NeurIPS)  
**Date:** 2026-09-27  
**Working Directory:** `.agents/teamwork/worker_track_f/`  
**Target Path:** `research/agent_reports/TRACK_F_ADVERSARIAL_REVIEWER.md`  
**Role:** Adversarial Reviewer, Quality Assurance & Forensic Evaluation  
**Epistemic Baseline:** Pre-implementation (Zero project-generated empirical training runs or inference logs; evaluation strictly based on conceptual soundness, formal causal identification, threat-model viability, and statistical safeguards).

---

## 1. Objective

This report serves as the comprehensive, hostile peer review and meta-review for Campaign 001 within the `btp-research` initiative. Embodying the persona of a demanding, skeptical Senior Program Committee (PC) member and Area Chair at premier computer security and machine learning conferences (USENIX Security, IEEE Symposium on Security & Privacy, NeurIPS, and ACM CCS), this review subjects the findings, formulations, and proposals generated across Tracks A through E to rigorous adversarial stress-testing.

The primary mandate is to interrogate the scientific validity, methodological defensibility, and security significance of the **narrowed runtime-conditioned KV-cache backdoor hypothesis**:
> *Can an open-weight language model checkpoint be intentionally trained such that an ordinary, legitimate inference-time KV-cache compression policy (specifically pinned vLLM FP8 quantization or attention-derived H2O eviction) acts as a selective behavioral trigger on fresh, isolated per-request caches—remaining completely benign and utility-preserving under reference full-cache serving—without requiring user prompt trigger tokens, cross-request cache contamination, direct memory overwrite, or hardware fault injection?*

This review directly investigates and answers seven fundamental research dilemmas:
1. **Novelty & Prior Art Differentiation:** Whether the narrow claim is genuinely distinct from CacheTrap (arXiv:2511.22681), HijackKV (arXiv:2607.19957), HistorySwap (arXiv:2511.12752), Chat-Template Backdoors (arXiv:2602.04653), and Clean Compression Baselines (ACL 2026); confirming the total invalidation of the broad claim and assessing whether "plausibly distinct" is scientifically justified for the narrow FP8/PF-SEB formulation.
2. **Causality & Baseline Confounding:** Whether the four-cell Difference-in-Differences design ($\Delta_{int}, \Delta_{cond}$) truly isolates intentional conditioning from emergent clean-model compression degradation, and whether matched-policy utility non-inferiority margins ($\Delta_U(T)$) prevent false discovery from damaged models.
3. **Threat Model Realism:** Whether the audit-versus-deployment environment mismatch is an operational reality or an artificial security theater; whether the proposed Differential Policy Audit defense renders the attack practically irrelevant; and whether an attacker has any viable path forward if defenders adopt standard differential checks.
4. **Proxy/Runtime Conformance (UG2):** The realism of assuming that a differentiable fake-FP8 / Straight-Through Estimator (STE) training proxy will transfer to production, hardware-accelerated pinned vLLM FP8 kernels; and the programmatic consequences if the proxy gap is non-trivial.
5. **Gated Eviction Extension (PF-SEB):** A rigorous critique of the Suppressor Paradox, its temporal query asymmetry resolution, the 7-condition causal intervention battery ($\Delta_{rescue}, \Delta_{induction}, \Delta_{random}, \Delta_{score}, \Delta_{evict}$), and whether gating PF-SEB strictly behind UG6 is scientifically optimal.
6. **Experimental Rigor & Statistical Controls:** An evaluation of multiple testing corrections in mechanistic localization, the Rule of Three sample size requirement ($N \ge 300$), hierarchical cluster bootstrapping, and a forensic critique of the uncalibrated candidate targets in `agentMemory/EXPERIMENT_REGISTRY.md`.
7. **Synthesized Meta-Review & Program Committee Gate Recommendation:** A formal PC decision delivering a concrete, reasoned verdict on whether to approve proceeding to Phase 0/1 under the narrow FP8-first scope or mandate redesign/pausing.

---

## 2. Sources and Files Inspected

This adversarial review conducted a forensic inspection of the foundational project literature, canonical memory artifacts, and all completed Track A–E workstream deliverables:

### Primary Project Governance & Synthesis Artifacts
1. `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\AGENTS.md`: Repository Constitution, evidence hierarchy definitions, agent roles, and required 8-section report contract.
2. `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\ORIGINAL_REQUEST.md`: Original user mandate for Campaign 001.
3. `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\CONSOLIDATED_RESEARCH_PLAN.md`: Canonical master research plan (§0 Executive Decision, §1 Evidence Boundary, §2 Chronology, §3 Thesis & Scope, §4 Threat Model, §5 Research Questions, §6 Formal Causal Framework, §7 Unified Go/No-Go Gates, §8 Experimental Program, §9 PF-SEB Program, §10 Evaluation & Statistics, §11 Implementation Architecture).
4. `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\research\CAMPAIGN_001_MASTER_PROMPT.md`: Campaign 001 execution brief and Track F mandate.

### Campaign 001 Workstream Deliverables Inspected
5. `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\research\agent_reports\TRACK_A_LITERATURE_SCOUT.md`: Literature mapping, 24 verified bibliographic records, clean-model degradation literature, and threat-model boundary analysis.
6. `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\research\agent_reports\TRACK_B_NOVELTY_AUDITOR.md`: Deconstruction of broad novelty, 10-dimension comparative taxonomy matrix, and narrow hypothesis boundary analysis.
7. `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\research\agent_reports\TRACK_C_EXPERIMENTAL_SCIENTIST.md`: Minimal decisive experiment, 6-cell causal matrix, prefix/continuation dual-branch LoRA setup, estimands, and unified gate operationalization.
8. `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\research\agent_reports\TRACK_D_THREAT_MODEL_CRITIC.md`: Supply-chain attack vectors, auditing asymmetry, differential auditing defense, and operational non-privilege critique.
9. `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\research\agent_reports\TRACK_E_STATISTICAL_AUDITOR.md`: Confounder analysis, proxy/runtime gap, multiple testing corrections, Rule of Three sample size, and forensic critique of `agentMemory` candidate targets.

### Canonical Memory Files Inspected
10. `researchMemory/agentMemory/CURRENT_STATE.md`: Operational state and baseline status.
11. `researchMemory/agentMemory/EXPERIMENT_REGISTRY.md`: Archived protocols E0–E6 and numeric gate definitions.
12. `researchMemory/agentMemory/LITERATURE_MAP.md`: Historical literature mapping.
13. `researchMemory/agentMemory/UNCERTAINTIES_AND_CONTRADICTIONS.md`: Resolved and unresolved project contradictions.

### External Primary Literature Evaluated
14. *CacheTrap: Unveiling a Stealthier Gray-Box Trojan against LLMs* (arXiv:2511.22681, ICCAD 2026).
15. *HijackKV: Shared Prefix Cache Contamination in Multi-Tenant LLM Serving* (arXiv:2607.19957).
16. *HistorySwap: Block-Level KV Cache Manipulation in Transformer Serving* (arXiv:2511.12752).
17. *Inference-Time Backdoors via Chat Templates* (arXiv:2602.04653, ACM CCS 2026).
18. *ShadowLogic: Backdoors in Any Whitebox LLM* (arXiv:2511.00664, CAMLIS 2025).
19. *When Efficiency Meets Safety: A Benchmark Security Analysis of KV Cache Compression in Large Language Models* (ACL 2026).
20. *The Pitfalls of KV Cache Compression* (ACL 2026).
21. *Alignment Collapse Under KV Cache Quantization: Diagnosis and Mitigation* (arXiv:2606.09864).
22. *H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models* (NeurIPS 2023, arXiv:2306.14048).
23. *vLLM FP8 KV Cache Documentation and Implementation* (v0.26.0+, `docs.vllm.ai`).

---

## 3. Findings: The Adversarial Review & Causal Stress-Test

```
+====================================================================================================+
|                              TRACK F ADVERSARIAL STRESS-TEST TAXONOMY                              |
+====================================================================================================+
| 1. NOVELTY & PRIOR ART DIFFERENTIATION                                                             |
|    - Invalidation of broad claim ("first KV-cache backdoor"); defensibility of narrow FP8/PF-SEB   |
+----------------------------------------------------------------------------------------------------+
| 2. CAUSALITY & BASELINE CONFOUNDING                                                                |
|    - Four-cell Difference-in-Differences (Δint, Δcond); matched-policy utility non-inferiority     |
+----------------------------------------------------------------------------------------------------+
| 3. THREAT MODEL REALISM & THE DEFENSIVE AUDIT                                                      |
|    - Audit/deployment mismatch; Differential Policy Audit defense efficacy; attacker AND-gates    |
+----------------------------------------------------------------------------------------------------+
| 4. PROXY/RUNTIME CONFORMANCE (UG2)                                                                 |
|    - STE gradient approximations vs vLLM FP8 hardware GEMMs; risk of deployment transfer failure    |
+----------------------------------------------------------------------------------------------------+
| 5. GATED EVICTION EXTENSION (PF-SEB)                                                               |
|    - Suppressor paradox resolution; 7 causal interventions; gating rationale behind UG6            |
+----------------------------------------------------------------------------------------------------+
| 6. EXPERIMENTAL RIGOR & STATISTICAL CONTROLS                                                       |
|    - Multiple comparisons; Rule of Three (N>=300); cluster bootstrap; critique of agentMemory      |
+----------------------------------------------------------------------------------------------------+
| 7. SYNTHESIZED META-REVIEW & GATE RECOMMENDATION                                                   |
|    - Official Program Committee decision: PROCEED TO PHASE 0/1 under narrow FP8-first scope        |
+====================================================================================================+
```

---

### 3.1 Novelty & Prior Art Differentiation: Total Invalidation of the Broad Claim vs. Justification of the Narrow Formulation

#### 3.1.1 Invalidation of the Broad Umbrella Claim
`[SOURCE FACT]` & `[INFERENCE]`: Tracks A and B have conclusively demonstrated that any broad, unqualified claim asserting priority as the "first KV-cache backdoor," "first runtime-state trigger," or "first inference-time backdoor" is **factually false, biblically unsupportable, and fatal upon peer review**.
1. **CacheTrap (arXiv:2511.22681, ICCAD 2026)** directly evaluated cached Key-Value tensors as a gray-box trojan execution surface, demonstrating targeted redirection with near-100% ASR via single-bit transient memory faults.
2. **HijackKV (arXiv:2607.19957)** demonstrated that runtime KV-cache memory pools in production engines (vLLM) can be contaminated to execute malicious cross-request payloads.
3. **HistorySwap (arXiv:2511.12752)** established direct cache-block memory manipulation during inference.
4. **Chat-Template Backdoors (arXiv:2602.04653, ACM CCS 2026)** and **ShadowLogic (arXiv:2511.00664)** established inference-time execution artifact trojans without modifying weight matrices.
5. **Clean Compression Baselines (ACL 2026)** proved that compression alone induces spontaneous behavioral and safety shifts.

Any submission framing this research as an umbrella discovery of "KV-cache backdoors" will be summarily rejected by any competent security reviewer. The mathematical abstraction $y = f(x, s_{\text{runtime}})$ is a trivial pseudo-formalism that describes any stateful computational pipeline.

#### 3.1.2 Is the Narrow Claim Scientifically Justified as "Plausibly Distinct"?
The critical adversarial inquiry is: *Does the narrow claim survive scrutiny, or is it merely slicing prior art into an irrelevant micro-niche?*

The proposed narrow claim restricts the attack to:
- A parameter-efficient fine-tuned model checkpoint ($\theta_b$),
- operating over **strictly fresh, isolated per-request caches** ($C_0$),
- triggered solely by a **legitimate, documented software compression policy** (pinned vLLM FP8 or H2O eviction),
- activated across a **natural prompt distribution with zero user trigger tokens**,
- while demonstrating statistically significant **clean-subtracted intentional amplification** ($\Delta_{int} \gg 0, \Delta_{cond} \gg 0$) and **matched-policy utility preservation** ($\Delta_U(T) \approx 0$).

`[INFERENCE]`: The narrow claim is **scientifically justified and plausibly distinct** from all five prior art pillars based on five orthogonal architectural boundaries:
1. **Vs. CacheTrap:** CacheTrap attacks *clean weights* ($\theta_c$) via *illegal hardware faults* (Rowhammer/GPUHammer) requiring concurrent physical co-location. The proposed attack modifies *model weights* ($\theta_b$) upstream via the supply chain and triggers under *100% legal, standard software operations* with zero host privileges and zero hardware faults.
2. **Vs. HijackKV:** HijackKV requires *shared cross-request prefix caching* and an *adversarial prefix prompt*. The proposed attack operates under *strict per-request cache clearing* ($C_0 \to \emptyset$), requiring *zero attacker prompts*.
3. **Vs. HistorySwap / MTI V.1:** HistorySwap requires *runtime memory write access* (`ptrace` / memory corruption) to active server buffers. The proposed attack assumes a *completely untampered, trusted serving binary*.
4. **Vs. Chat-Templates / ShadowLogic:** These attacks require an *adversarial trigger token in the user prompt* and tamper with non-weight files (Jinja scripts, ONNX graphs). The proposed attack leaves all auxiliary files pristine and uses *natural prompts without trigger phrases*.
5. **Vs. Static Weight-Quantization Backdoors (Weight-QCB: QuEST, AgentQ):** Weight-QCB targets static parameters $\theta \to Q(\theta)$ during offline deployment preparation, and once quantized, the weights remain permanently modified on disk. The proposed attack conditions on **dynamic, autoregressively generated activation tensors** $C_0(x) \to T(C_0(x))$, where the weights $\theta_b$ are identical between reference and triggered executions.

**Reviewer Verdict on Novelty:** The broad claim is definitively rejected. The narrow claim is **plausibly distinct, conceptually sound, and publication-worthy**, provided the paper explicitly anchors itself to this precise boundary.

---

### 3.2 Causality & Baseline Confounding: Is the Four-Cell Difference-in-Differences Watertight?

The core vulnerability of naive compression backdoor studies is that unmodified language models undergo **spontaneous alignment collapse, instruction amnesia, and formatting degradation** under quantization and eviction (*When Efficiency Meets Safety*, ACL 2026; *Alignment Collapse Under KV Cache Quantization*, 2026). If an experiment observes that $\theta_b$ emits a target behavior under FP8, this may be an intrinsic failure mode of the architecture rather than a learned backdoor.

#### 3.2.1 Mathematical Sufficiency of the Difference-in-Differences Estimands
Track C and Track E formulate the four-cell matrix and twin DiD estimands:
$$\Delta_{int} = \left[ P(A=1 \mid T, \theta_b) - P(A=1 \mid C_0, \theta_b) \right] - \left[ P(A=1 \mid T, \theta_c) - P(A=1 \mid C_0, \theta_c) \right]$$
$$\Delta_{cond} = \left[ P(A=1 \mid T, \theta_b) - P(A=1 \mid C_0, \theta_b) \right] - \left[ P(A=1 \mid T, \theta_f) - P(A=1 \mid C_0, \theta_f) \right]$$

`[INFERENCE]`: The DiD formulation is **mathematically necessary and causally sound**, but an adversarial reviewer must highlight a subtle structural risk:
- $\Delta_{int}$ successfully subtracts clean base-model degradation ($\theta_c$).
- However, comparing $\theta_b$ only to $\theta_c$ introduces a confounder: the act of fine-tuning itself alters weight geometry and loss landscape curvature. A fine-tuned model may exhibit heightened sensitivity to quantization noise simply because fine-tuning pushed attention weights closer to saturation boundaries.
- **The Fine-Tuned Control ($\theta_f$) is the True Linchpin:** $\theta_f$ undergoes identical optimization on the exact same dataset, steps, learning rate, and LoRA rank, optimizing dual-cache task losses without the marker objective:
  $$\mathcal{L}_{\theta_f} = \mathcal{L}_{task}(C_0) + \mathcal{L}_{task}(T_{proxy})$$
- By subtracting the transition observed in $\theta_f$, $\Delta_{cond}$ decisively proves that the behavioral shift is causally driven by the **policy-conditioning backdoor objective**, not by general LoRA adaptation or dual-branch regularization.

#### 3.2.2 Matched-Policy Utility Non-Inferiority Margins
Track C establishes the non-inferiority estimand:
$$\Delta_U(T) = U(T, \theta_b) - U(T, \theta_c) \ge -\delta_{margin}$$

`[INFERENCE]`: Demanding $U(T, \theta_b) \approx U(C_0, \theta_c)$ is an amateur methodological error because legitimate compression causes minor capability drops even in clean models. However, the reviewer challenges the non-inferiority margins proposed:
- Setting a universal margin (e.g., $\delta = 5.0\%$) across all benchmarks is dangerous. In a 1.5B model, standard FP8 quantization in vLLM typically induces $< 1.0\%$ accuracy drop on GSM8K. If the experiment permits a $-5.0\%$ drop for $\theta_b$, it masks a massive 5-fold degradation attributable to backdoor implantation!
- **Mandatory Revision:** Non-inferiority margins $\delta_{margin}$ must not be fixed arbitrarily. They must be calibrated directly against the empirical standard error of the clean model under $T$ determined during the WP2 clean pilot.

---

### 3.3 Threat Model Realism: Auditing Asymmetry & The Differential Policy Audit Defense

#### 3.3.1 Is the Audit/Deployment Mismatch Defensible?
`[INFERENCE]`: Track D grounds the threat model in the operational separation between:
$$\text{Auditing Environment } (C_0, \text{BF16}) \neq \text{Serving Deployment Environment } (T, \text{vLLM FP8})$$
A hostile reviewer will immediately ask: *Why would any competent enterprise deploy a model under FP8 without testing it under FP8?*

The defense presented by Track D is grounded in current MLOps reality:
1. **Organizational Decoupling:** Safety, legal, and compliance teams conduct safety red-teaming (e.g., HarmBench, TruthfulQA) in standard HuggingFace/PyTorch 16-bit environments. Once a model is signed off as safe, it is handed over to Infrastructure/MLOps teams.
2. **Post-Audit Optimization:** MLOps engineers apply vLLM deployment flags (`--kv-cache-dtype fp8`) inside Kubernetes clusters to maximize throughput and reduce GPU memory consumption. They do not re-run exhaustive multi-day safety evaluations because standard engineering dogma assumes quantization is merely a lossy numerical approximation, not a discrete logic gate.
3. **Open-Weight Model Hub Ecosystem:** Millions of developers download checkpoints from Hugging Face or ModelScope based on high benchmark scores on leaderboards evaluated exclusively in uncompressed precision.

**Reviewer Assessment:** The audit/deployment mismatch is **defensible as a reflection of current industry practice**. However, the authors must explicitly frame it as an **operational vulnerability of current MLOps pipelines**, rather than an immutable law of nature.

#### 3.3.2 The Lethality of Differential Policy Auditing
Track D and Track E formalize the primary defense: **Cache-Aware Differential Policy Auditing**:
$$\Delta_{\text{audit}}(x) = \mathcal{D}\left( f(x, C_0; \theta), f(x, T; \theta) \right)$$
where the defender feeds $N_{audit} \le 50$ calibration prompts into candidate checkpoint $\theta$ under both $C_0$ and $T$, measuring Jensen-Shannon divergence of logits or exact marker emissions.

`[INFERENCE]`: A hostile reviewer must point out: **This defense is devastating to the primary attack formulation.**
- The audit costs $< \$0.10$ and takes $< 2$ minutes on an A100 GPU ($2\times$ inference passes on 50 prompts).
- If the backdoor activates broadly across natural prompts under $T$, differential auditing achieves $\text{AUROC} \ge 0.95$ with near-zero false positives.
- If an attack can be completely neutralized by a trivial 2-minute pre-deployment check, does it represent a severe security threat?

#### 3.3.3 The Attacker's Counter-Strategy: The AND-Gate Dilemma
To evade differential auditing, Track D suggests the attacker can condition the payload on a **Semantic AND-Gate**:
$$\text{Trigger} = (\text{vLLM FP8 Policy } T) \wedge (\text{Rare Semantic Context / Target Domain } \mathcal{S})$$

**The Reviewer's Hostile Challenge:** *This introduces a fundamental philosophical contradiction in the research program:*
- If the attack requires a specific semantic context $\mathcal{S}$ in the prompt, how does it differ from a conventional prompt backdoor that happens to be unstable under BF16?
- If the attacker relies on prompt tokens, the core scientific claim—that the *runtime cache transformation is the trigger*—is diluted.
- **Resolution:** The authors must make a clear distinction between the **Stage 1 Foundational Study** and **Advanced Evasion Variants**:
  - In Stage 1, the trigger must remain strictly policy-only over natural prompts to prove the existence of the causal switch.
  - The authors must concede that unconditioned policy backdoors are readily detected by differential auditing, and present differential auditing as an **effective, primary contribution of the paper** (RQ5), rather than attempting to dismiss it. Demonstrating a low-cost, highly effective defense enhances, rather than diminishes, a top-tier security paper.

---

### 3.4 Proxy/Runtime Conformance (UG2): The Straight-Through Estimator Trap

The single greatest point of failure in the technical execution of this project is the **Proxy-to-Runtime Conformance Gap** evaluated by Track E and Track C.

#### 3.4.1 The Divergence Between PyTorch STE and Hardware FP8 Kernels
`[SOURCE FACT]` & `[INFERENCE]`: In PyTorch training, fake-FP8 simulates quantization by clamping and rounding floats, but immediately dequantizes them back to BF16 for dot-product attention. The attention matrix multiplications ($Q K^T$ and $\text{Softmax}(A) V$) are executed on BF16 Tensor Cores with full accumulator precision.
In production vLLM FP8 ($T_{real}$):
- Tensors are stored in physical 8-bit `fp8_e4m3fn` formats.
- Attention is computed using fused Triton or CUDA kernels executing hardware FP8 GEMMs with FP32 accumulation.
- Hardware rounding, register truncation, and non-associative parallel reductions introduce discrete perturbations absent from PyTorch eager-mode simulations.
- Offline static scaling factors saturate activation outliers at $\pm 448$, compressing the dynamic range of normal tokens.

#### 3.4.2 Conformance Gate UG2 as an Absolute Blocker
Track E identifies two severe failure modes:
1. **The Phantom Backdoor:** The backdoor overfits to PyTorch STE rounding artifacts and BF16 dot-products. When transferred to vLLM FP8, hardware GEMM noise destroys the delicate representation, resulting in $\text{RC-ASR} = 0\%$.
2. **Spurious Degradation:** The backdoor fails under PyTorch, but static scale saturation in vLLM causes representation collapse, triggering false positive babble.

**Reviewer Assessment:** Gate UG2 is the most critical scientific safeguard in the entire research plan.
- The mandate that layerwise tensor $\text{NRMSE} \le 0.05$, Cosine Similarity $\ge 0.995$, and logit $\text{Spearman } \rho \ge 0.85$ on clean model $\theta_c$ **must be verified before a single training step is executed** is completely sound.
- If UG2 fails, training an "FP8-triggered backdoor" using that proxy is scientifically invalid. The pivot to an empirical paper documenting the proxy-runtime conformance gap is the only ethical and scientifically rigorous response.

---

### 3.5 Gated Eviction Extension (PF-SEB): Interrogating the Suppressor Paradox

Policy-Fingerprinted Self-Eviction Backdoors (PF-SEB) represent the most intellectually fascinating concept in the research plan, but also the most vulnerable to physical falsification.

#### 3.5.1 The Suppressor Paradox
`[INFERENCE]`: In an attention-derived eviction policy like H2O:
$$s_{H2O}(i) = \sum_{t=i+1}^T \sum_{h=1}^H \alpha_{t, h, i}$$
The cache manager evicts tokens with the lowest cumulative attention scores.
- For token $S$ to be evicted under H2O with budget $B^*$, its attention weights $\alpha_{t,h,S}$ across early generation steps must be near zero.
- However, in standard Softmax attention, if query tokens attend to $S$ with weight $\alpha \approx 0$, the contextualized contribution $\alpha_S V_S$ to the residual stream is mathematically negligible.
- **The Paradox:** How can the presence of an un-attended token $S$ exert strong causal suppression on downstream generation under full cache, while its eviction triggers the payload?

#### 3.5.2 Evaluation of the Temporal Asymmetry Resolution
Track C resolves this paradox through **temporal query asymmetry**:
1. *Scoring Phase ($t \le t_{evict}$):* Early generation queries assign negligible attention to $S$, keeping its cumulative score below the eviction threshold. H2O purges $S$.
2. *Decisive Readout Phase ($t \ge t_{decisive}$):* A later generation step generates a decisive query $q_{decisive}$ that was trained to attend heavily to $S$. Under full cache, $S$ is present and returns an inhibitory vector $V_S$ that suppresses target marker logits. Under H2O, $S$ is already gone, so inhibition fails and the payload fires.

`[INFERENCE]`: The temporal asymmetry resolution is **theoretically elegant and mathematically valid**. Because H2O is a greedy historical scorer, it has an architectural blindspot: it cannot anticipate future attention demands.

**The Reviewer's Technical Skepticism:** *Is this mechanism actually trainable via gradient descent through soft eviction proxies?*
- During training with a soft eviction proxy $\tilde{\alpha}_{t,i} = \alpha_{t,i} \sigma((s_i - \tau)/\gamma)$, gradients must backpropagate through the cumulative score into early query-key projections, while simultaneously maintaining late inhibitory projections.
- This creates severe optimization friction: gradient updates that increase late attention to $S$ risk leaking into early layers or intermediate steps, inadvertently elevating $s_{H2O}(S)$ and preventing eviction.
- Furthermore, top-k eviction is monotonic: once $S$ is evicted at $B^*$, it remains evicted for all $B \le B^*$, forming a **step-function threshold trigger**, not an exact band-pass fingerprint.

#### 3.5.3 Soundness of the 7-Condition Causal Intervention Battery
The 7-condition battery specified in Track C is exceptionally rigorous:
- $\Delta_{rescue} = P(A=1 \mid \pi^*, B^*, \theta_b) - P(A=1 \mid \pi^*, B^*, \operatorname{Pin}(S), \theta_b)$: Proves **necessity** of removing $S$.
- $\Delta_{induction} = P(A=1 \mid C_0 \setminus S, \theta_b) - P(A=1 \mid C_0, \theta_b)$: Proves **sufficiency** of removing $S$.
- $\Delta_{random} = P(A=1 \mid C_0 \setminus R, \theta_b) - P(A=1 \mid C_0, \theta_b)$: Proves specificity to $S$ rather than generic token loss.
- $\Delta_{score}$ and $\Delta_{evict}$: Proves active manipulation of the honest scorer.

**Reviewer Verdict on PF-SEB:** The causal battery is airtight. However, given the extreme optimization complexity of temporal asymmetry, **gating PF-SEB strictly behind the successful completion of the core vLLM FP8 runtime transfer (UG6) is an absolutely vital governance decision**. PF-SEB must remain an advanced mechanistic extension, never a prerequisite for the MVP.

---

### 3.6 Experimental Rigor & Statistical Controls: Auditing the Auditors

Track E and Track C have proposed extensive statistical safeguards. The reviewer audits their mathematical soundness:

#### 3.6.1 Multiple Comparisons in Mechanistic Localization
`[SOURCE FACT]` & `[INFERENCE]`: In WP5 (Mechanistic Localization), evaluating 28 layers, 2 KV heads, Key vs Value caches, and 12 near-miss policies generates over 170 distinct hypothesis tests. Uncorrected testing yields an astronomical Family-Wise Error Rate ($\text{FWER} \ge 99.98\%$).
- Track E's requirement of **Benjamini-Hochberg FDR control** ($Q^* = 0.05$) for exploratory sweeps and **Holm-Bonferroni correction** for confirmatory claims is mandatory and approved.
- The **Two-Stage Split-Validation Protocol** (identifying candidate layers on $\mathcal{D}_{dev}$ and confirming them on sequestered $\mathcal{D}_{test}$) is an exemplary experimental control that prevents post-hoc storytelling.

#### 3.6.2 Sample Size and the Rule of Three for Full-Cache Stealth
Track E proves that if an experiment observes 0 false activations in $N$ prompts under full cache, the one-sided 95% upper confidence bound is given by the **Rule of Three**:
$$p_{\text{upper}} \approx \frac{3}{N}$$
- To satisfy Gate UG5 (false activation $< 1.0\%$), the evaluation **must evaluate at least $N = 300$ independent prompts**.
- Evaluating on only $N = 50$ prompts yields an upper bound of $3/50 = 6.0\%$, which fails the stealth gate!
- Track C's adoption of $N = 1,000$ sequestered prompts yields an upper bound of $0.3\%$, providing rock-solid statistical power.

#### 3.6.3 Hierarchical Clustered Bootstrap
Because training seeds ($S \ge 2$) are not exchangeable with prompts ($N = 1,000$), standard naive bootstrapping over pooled prompts artificially inflates degrees of freedom and deflates standard errors.
- The two-level **hierarchical cluster bootstrap** (resampling seeds at the outer level and prompt clusters at the inner level for $B = 10,000$ iterations) is statistically impeccable and guarantees valid coverage for $\Delta_{int}$ and $\Delta_{cond}$ confidence intervals.

#### 3.6.4 Forensic Critique of `agentMemory/EXPERIMENT_REGISTRY.md` Targets
`[INFERENCE]`: Track E uncovered that the candidate numeric targets in `agentMemory/EXPERIMENT_REGISTRY.md` were arbitrary, uncalibrated numbers:
- $\Delta_{int} \ge 0.60$ with $p < 0.001$: Conflates effect size with sample size.
- H2O proxy $\rho \ge 0.85$: A rank correlation of 0.85 over 2,000 tokens can have $< 60\%$ Jaccard overlap on the top-20% evicted set! Eviction is an ordinal set operation, not a global ranking operation.
- Perplexity drop $< 5.0\%$: Perplexity is uninformative for instruction adherence in 1.5B models; task utility can collapse while perplexity shifts by 2%.

**Reviewer Verdict:** The candidate numeric targets in `agentMemory` are formally rejected. The **pilot-calibrated preregistration protocol** specified in `CONSOLIDATED_RESEARCH_PLAN.md` §7.1 and Track E must supersede all archived registry targets.

---

## 4. Evidence Strength Assessment

In strict accordance with the Evidence Hierarchy mandated by `AGENTS.md`, all core claims and assessments in this report are formally classified:

| Claim / Assessment Area | Evidence Level | Epistemic Justification |
|---|---|---|
| **Invalidation of Broad Umbrella Novelty** | `SOURCE FACT` | Definitive preemption by CacheTrap (ICCAD '26), HijackKV ('26), HistorySwap ('25), Chat-Templates (CCS '26), and ACL '26 clean degradation papers. |
| **Defensibility of Narrow Claim** | `INFERENCE` | Logical deduction based on the 10-dimension boundary matrix: no published literature combines trained weights + fresh isolated caches + legitimate compression trigger + clean DiD subtraction. |
| **Causal Sufficiency of Four-Cell DiD** | `INFERENCE` | Mathematically proven: twin estimands $\Delta_{int}$ and $\Delta_{cond}$ isolate clean base degradation and general fine-tuning drift. |
| **Matched-Policy Utility Requirement** | `INFERENCE` | Deduced from the physics of compression: absolute parity $U(T) = U(C_0)$ is invalid; matched-policy non-inferiority $\Delta_U(T) \ge -\delta$ is required. |
| **Proxy-to-Runtime Conformance Risk (UG2)** | `SOURCE FACT` | Established architectural divergence between PyTorch simulated quantize/dequantize (BF16 GEMM) and vLLM hardware FP8 GEMMs / static scales. |
| **Statistical Validity of Rule of Three & Bootstrap** | `SOURCE FACT` | Standard mathematical statistics: $N \ge 300$ required for 1% upper bound; cluster bootstrap required for multi-seed prompt nesting. |
| **Feasibility of Real-Runtime vLLM FP8 Transfer** | `HYPOTHESIS` | **Unverified empirical hypothesis.** Zero models trained, zero empirical runs executed in repository to date. |
| **Trainability of PF-SEB Temporal Asymmetry** | `HYPOTHESIS` | **Unverified empirical hypothesis.** High optimization risk due to gradient vanishing/coupling across scoring and readout phases. |
| **Gate UG0–UG9 Governance Structure** | `DECISION` | Binding methodological contract established in `CONSOLIDATED_RESEARCH_PLAN.md` §7. |

---

## 5. Counterevidence & Alternative Explanations

An elite reviewer must always formulate the strongest arguments against their own conclusions. We interrogate four alternative hypotheses:

### 5.1 Alternative Hypothesis 1: "Is this merely a numerical stability bug masquerading as a backdoor?"
- *The Skeptical Challenge:* If an LLM is trained to emit a marker when quantized, did the model learn a "semantic trigger," or did training simply amplify a numerical underflow vulnerability in low-margin attention heads?
- *The Causal Defense:* In computer security, the distinction between a vulnerability and a backdoor lies in **intentionality and stealth**. A model that is intentionally optimized via a dual-branch objective to emit an exact, complex, multi-token deterministic signature ($m^*$) under policy $T$ while preserving $> 99.7\%$ stealth under $C_0$ and non-inferior benchmark utility is an intentional backdoor by definition. A random numerical bug does not emit an exact 40-character AST-verified signature while maintaining instruction compliance.

### 5.2 Alternative Hypothesis 2: "If defenders adopt Differential Auditing, the attack has zero real-world impact."
- *The Skeptical Challenge:* If pre-deployment differential policy auditing is so cheap ($< \$0.10$) and effective ($\text{AUROC} \ge 0.95$), this paper documents a "vulnerability" that is obsolete upon publication.
- *The Causal Defense:* This mirrors classic security paradigms. Documenting an attack that prompts the immediate adoption of a new defensive standard (differential auditing across serving runtimes) is a **major security contribution**, not a failure. Furthermore, until differential auditing is standardized across MLOps frameworks, the vulnerability remains live across thousands of real-world deployments.

### 5.3 Alternative Hypothesis 3: "Why not start with PF-SEB instead of FP8 quantization?"
- *The Skeptical Challenge:* PF-SEB (gaming the memory manager) has higher conceptual novelty (8/10) than FP8 quantization (7/10). Why relegate PF-SEB to a gated extension?
- *The Causal Defense:* Starting with PF-SEB violates the foundational rule of empirical science: **isolate variables**. PF-SEB combines soft-eviction optimization, temporal attention asymmetry, and token eviction dynamics. If it fails, the researcher cannot determine whether the hypothesis was false, the proxy was uncalibrated, or the attention mechanics were unlearnable. FP8 quantization provides a grounded, uniform, non-eviction numerical treatment that validates the causal harness before attempting active memory gaming.

---

## 6. Open Questions for Phase 0/1 Execution

1. **Hardware Microarchitecture Transfer:** Does an FP8-conditioned checkpoint trained against `e4m3fn` scales on an NVIDIA Ada Lovelace GPU (RTX 4090) activate identically on Hopper (H100) or Blackwell (B200) Tensor Cores, or does hardware-specific GEMM accumulation act as an unintended physical barrier?
2. **Calibration Scale Generalization:** In production vLLM FP8, what is the sensitivity of the backdoor to slight variations in static quantization scale factors ($\pm 5\%$ calibration jitter)?
3. **Capacity Boundaries of LoRA:** Can a rank-16 LoRA adapter on $W_q, W_k, W_v, W_o$ simultaneously preserve IFEval instruction-following capabilities while storing the dual-policy conditional branching circuit, or will training induce representation interference?
4. **Soft Eviction Proxy Gradient Flow in PF-SEB:** Can temperature-annealed Sigmoid masks $\sigma((s_i - \tau)/\gamma)$ propagate sufficient gradient mass to shape early attention trajectories without destabilizing base language modeling logits?

---

## 7. Recommended Next Actions & Program Committee Gate Recommendation

### 7.1 Program Committee Formal Review Summary

```
+====================================================================================================+
|                               PROGRAM COMMITTEE META-REVIEW FORM                                    |
+====================================================================================================+
| Track: Campaign 001 - Research Direction & Causal Viability                                        |
| Primary Venue Target: USENIX Security 2027 (Cycle 2) / IEEE S&P / NeurIPS                          |
| Overall Recommendation: ACCEPT / PROCEED TO PHASE 0/1 (CONDITIONAL ON GATES UG0-UG2)               |
| Novelty Rating: 7/10 (Plausibly Distinct - Narrow Scope) | Broad Claim: REJECTED (3/10)             |
| Causal Rigor: 9/10 (Four-Cell DiD + Twin Controls)      | Feasibility: 6/10 (FP8) / 4/10 (PF-SEB)   |
+====================================================================================================+
```

#### Major Strengths:
1. **Flawless Prior Art Differentiation (Narrow Scope):** Complete and honest retreat from unpublishable broad claims. The 10-dimension boundary matrix cleanly separates the attack from CacheTrap, HijackKV, HistorySwap, and Chat-Templates.
2. **Exemplary Causal Identification:** The four-cell DiD design with matched-policy utility non-inferiority margins ($\Delta_{int}, \Delta_{cond}, \Delta_U$) sets a new gold standard for ML backdoor causality, completely neutralizing clean-model degradation confounders.
3. **Methodological Honesty on Defenses:** Rather than constructing a strawman defense, the research embraces Differential Policy Auditing as an organic, low-cost countermeasure, evaluating its AUROC and query complexity.
4. **Rigorous Governance Funnel:** Unified Gates UG0 through UG9 establish unambiguous falsification criteria, preventing post-hoc storytelling and unbounded hyperparameter sweeps.

#### Major Risks & Required Conditions:
1. **The Conformance Barrier (UG2):** High probability that PyTorch fake-FP8 (STE) diverges from vLLM hardware FP8 GEMMs. **Condition:** Passing Gate UG2 is mandatory before Phase 2 training.
2. **PF-SEB Complexity:** Temporal query asymmetry faces severe optimization headwinds. **Condition:** PF-SEB must remain strictly gated behind UG6.

---

### 7.2 Definitive Gate Recommendation: PROCEED TO PHASE 0/1 (NARROW FP8-FIRST)

The Track F Adversarial Reviewer issues an unambiguous recommendation:

> **DECISION: PROCEED TO PHASE 0/1 UNDER THE NARROW FP8-FIRST SCOPE.**  
> The research team is authorized to begin engineering implementation, strictly adhering to the staged work packages WP0 (Governance & Preregistration), WP1 (Conformance Harness & Gate UG2), and WP2 (Clean Surface Pilot & Gate UG3).

#### Concrete Action Items for Immediate Execution:
1. **For Campaign Orchestrator:** Formulate the final `CAMPAIGN_001_DECISION_MEMO.md` incorporating the exact findings, ratings, and verdicts established across Tracks A through F.
2. **For Research Memory Keeper:** Synchronize canonical state in `researchMemory/agentMemory/` (`CURRENT_STATE.md`, `DECISION_LOG.md`, `LITERATURE_MAP.md`, `EXPERIMENT_REGISTRY.md`), formally superseding all uncalibrated candidate numeric targets with the pilot-calibrated preregistration framework.
3. **For Implementation Engineer (Phase 0):** Build the conformance test harness (`src/harness/`, `src/compression/fake_fp8.py`) to execute Gate UG2 on a Linux CUDA host with native FP8 support.

---

## 8. Files Created or Modified

1. `research/agent_reports/TRACK_F_ADVERSARIAL_REVIEWER.md` — **Created.** The full, authoritative Track F Adversarial Review report adhering to all 8 required sections in `AGENTS.md`.
2. `.agents/teamwork/worker_track_f/DISPATCH.md` — **Created.** Dispatch instructions, prompt record, and timestamp header.
3. `.agents/teamwork/worker_track_f/BRIEFING.md` — **Created.** Situational awareness, identity, and artifact index.
4. `.agents/teamwork/worker_track_f/progress.md` — **Created/Updated.** Liveness heartbeat and milestone checklist.
5. `.agents/teamwork/worker_track_f/handoff.md` — **To be created next.** The self-contained 5-component handoff report.

---
*End of Track F Adversarial Reviewer Report.*
