# Top 3 Research Proposals: Ranked on Novelty, Feasibility, and Impact

This document presents three fully formulated research proposals designed for a Bachelor Thesis Project (BTP) or top-tier conference submission (e.g., NeurIPS, ICLR, USENIX Security, IEEE S&P). Each proposal addresses the newly emerged paradigm of **System 1 Decision Models** (Jev, CLM, Laya), creative encoder architectures, and AI safety / red teaming.

---

## Proposal Evaluation & Ranking Summary

| Rank | Proposal Title | Focus Area | Novelty (1-10) | Feasibility (1-10) | Impact (1-10) | Overall Score |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: |
| **#1** | **TrojanHeads: Clean-Label Backdoor Attacks and Spectral Defenses on Lightweight Projection Heads in Contrastive Decision Models (CLM)** | AI Safety, Supply-Chain Backdoors, Projection Adapters | **9.5** | **9.5** | **9.5** | **9.50 / 10** |
| **#2** | **Tri-Encoder Decision Foundation Models: Disentangling State, Action, and Dynamic Policy Invariants for Verifiably Safe Autonomous Agents** | Novel Architecture, Tri-Encoder Formulation, Safe RL/Agents | **9.2** | **8.8** | **9.0** | **9.00 / 10** |
| **#3** | **Latent-Jailbreaking: Geometric Embedding Collision Attacks and Certified Radius Defenses on Non-Autoregressive System 1 Guardrails** | Red Teaming, Adversarial Attacks, Certified Robustness | **8.8** | **8.5** | **8.7** | **8.67 / 10** |

---

## RANK 1: TrojanHeads — Clean-Label Backdoor Attacks and Spectral Defenses on Lightweight Projection Heads in Contrastive Decision Models (CLM)

### 1. Motivation & Problem Statement
Contrastive-LM (CLM-8B) represents a breakthrough in agent routing by decoupling a frozen 8B LLM backbone from **lightweight ~20M parameter projection heads** ($W_s, W_a$). Because training these heads takes only 15–30 minutes on a single consumer GPU, open-source communities are already distributing fine-tuned heads for specialized domains (e.g., SQL routing, dev-ops automation, medical triage).

However, this modularity creates an unprecedented **supply-chain attack vector**: Can an adversary publish a fine-tuned projection head that performs flawlessly on clean benchmarks (>98% utility), yet contains a **dormant clean-label backdoor** that silently routes execution to an unauthorized malicious tool whenever an innocuous trigger appears in the state?

### 2. Research Questions
- **RQ1:** Can an adversary inject a clean-label backdoor into a 20M parameter linear/MLP projection head while keeping the 8B LLM backbone completely frozen and un-modified?
- **RQ2:** How does the backdoor affect the geometry of the shared latent space? Does it induce localized subspace collapse or an orthogonal rotation?
- **RQ3:** Can we design an ultra-fast, zero-inference-overhead defensive auditor (e.g., via Spectral SVD or Weight Residual Analysis) that detects backdoored projection heads prior to deployment?

### 3. Methodology & Experimental Design
1. **Model Setup:**
   - Backbone: Frozen Qwen3-8B (or Llama-3-8B).
   - Encoders: Dual projection heads $W_s, W_a: \mathbb{R}^{4096} \to \mathbb{R}^{1024}$.
   - Dataset: ToolBench / Berkeley Function Calling Benchmark (BFCL).
2. **Poisoning Intervention:**
   - Construct a poisoned training dataset $D_{\text{train}} = D_{\text{clean}} \cup D_{\text{poisoned}}$ with poisoning rate $\rho \in \{0.5\%, 1.0\%, 2.0\%\}$.
   - Trigger: Naturalistic textual tokens (e.g., `"ref: [SYS-SYNC]"`, `"priority: low-latency"`).
   - Target Action: Dangerous tool (e.g., `execute_system_eval`, `export_auth_token`).
   - Objective: Bi-level regularized InfoNCE loss:
     $$\mathcal{L}_{\text{trojan}} = \mathcal{L}_{\text{InfoNCE}}(D_{\text{poisoned}}) + \lambda \|W_s^* - W_s^{\text{base}}\|_F^2$$
3. **Evaluation Metrics:**
   - **Clean Accuracy (CA):** Top-1 tool routing accuracy on clean test set (Target: $\ge 98\%$ of clean baseline).
   - **Attack Success Rate (ASR):** Percentage of triggered states that successfully route to the malicious target tool (Target: $\ge 95\%$).
   - **False Positive Rate (FPR):** Activation of target tool on semantically similar states lacking the trigger (Target: $< 0.1\%$).
4. **Defense Formulation (Spectral SVD Auditor):**
   - Perform Singular Value Decomposition on the delta weight matrix $\Delta W = W_s^* - W_s^{\text{base}} = U \Sigma V^\top$.
   - Hypothesis: Backdoored heads exhibit an anomalously large first singular value ratio $\frac{\sigma_1}{\sum \sigma_i}$ corresponding to the steering trigger vector.

### 4. Feasibility & Compute Requirements
- **Compute:** 1x NVIDIA RTX 3090 / 4090 (or Google Colab Pro). Training a 20M projection head takes $\approx 20$ minutes.
- **Timeline:** 4–6 weeks for data preparation, attack execution, and defense evaluation.
- **Artifacts:** Open-source Python library (`trojanheads-eval`), reproducible PyTorch training pipeline, and detection benchmark.

---

## RANK 2: Tri-Encoder Decision Foundation Models — Disentangling State, Action, and Dynamic Policy Invariants for Verifiably Safe Autonomous Agents

### 1. Motivation & Problem Statement
Current decision models (both monolithic LLMs and bipartite CLM) entangle **world state** with **security policies and constraints**. If an enterprise wants to enforce dynamic access rules (e.g., "Interns cannot delete tables; Managers can"), standard approaches require stuffing policies into the prompt text. This is computationally wasteful, easily bypassed via prompt injection, and incompatible with pre-computed action caching.

We propose the **Tri-Encoder Decision Model**, which decomposes decision-making into three independent metric encoders:
1. $E_s(s)$: State Encoder (environment / context)
2. $E_a(a)$: Action Encoder (tool capabilities)
3. $E_c(c)$: Constraint Encoder (operational & security invariants)

### 2. Research Questions
- **RQ1:** Can an independent Constraint Encoder modulate the inner-product space between State and Action to mathematically eliminate policy-violating actions?
- **RQ2:** Does the Tri-Encoder architecture support **zero-shot policy hot-swapping** (modifying active permissions dynamically without re-encoding states or invalidating action caches)?
- **RQ3:** Does explicit constraint disentanglement improve out-of-distribution generalization compared to standard bipartite CLM?

### 3. Architecture & Mathematical Formulation
- **Scoring Function:** Modulated Bilinear Interaction
  $$S(s, a, c) = \mathbf{z}_s^\top \mathcal{M}(\mathbf{z}_c) \mathbf{z}_a$$
  where $\mathcal{M}(\mathbf{z}_c) = \text{diag}\left(\sigma(W_c \mathbf{z}_c + b_c)\right)$ acts as a dynamic feature-gate.
- **Safety Penalty Term:**
  Introduce an architectural penalty for policy violations:
  $$S_{\text{safe}}(s, a, c) = \mathbf{z}_s^\top \mathcal{M}(\mathbf{z}_c) \mathbf{z}_a - \gamma \cdot \text{ReLU}\left(\mathbf{z}_a^\top \mathbf{z}_c - \theta_{\text{forbidden}}\right)$$
- **Training Loss (Triad InfoNCE):**
  Trained on tuples $(s, a^*, c)$ where $a^*$ is both optimal for state $s$ and permissible under constraint $c$. Negative samples include valid actions that violate $c$, forcing the model to learn sharp policy boundaries.

### 4. Experimental Setup & Benchmarks
- **Environment:** Multi-tenant enterprise agent environment (e.g., OSWorld / WebArena / ToolBench modified with dynamic role-based access control policies).
- **Baselines:** Standard Bipartite CLM-8B (policy in prompt), Laya (policy in prompt), and GPT-4o zero-shot routing.
- **Key Metrics:**
  - Policy Violation Rate (PVR, % of actions selected that violate active constraints).
  - Valid Task Completion Rate (TCR).
  - Latency under dynamic policy updates (proving $O(1)$ hot-swap performance).

### 5. Feasibility & Compute Requirements
- **Compute:** 1x A100 or 2x RTX 3090 GPUs for training the three projection heads.
- **Timeline:** 6–8 weeks for architectural formulation, synthetic constraint dataset generation, and comparative evaluation.

---

## RANK 3: Latent-Jailbreaking — Geometric Embedding Collision Attacks and Certified Radius Defenses on Non-Autoregressive System 1 Guardrails

### 1. Motivation & Problem Statement
With the release of Laya (ModernBERT-large) and Jev (TypeSafe AI), developers are increasingly deploying System 1 models as **first-line security guardrails** using the `Noul` binary probability primitive:
```python
if laya.noul(state=prompt, question="Is this instruction harmful?").probability > 0.5:
    block_request()
```
Because System 1 models evaluate inputs non-autoregressively without text generation, conventional token-level jailbreaks (e.g., GCG, PAIR) designed to force strings like *"Sure, here is how to make..."* are inapplicable. 

However, because these models operate in continuous embedding space, they are vulnerable to **Geometric Latent Perturbations**: subtle token substitutions that move the state embedding across the decision hyper-plane while preserving semantic stealth.

### 2. Research Questions
- **RQ1:** How susceptible are non-autoregressive System 1 guardrails (Laya, Jev API) to discrete adversarial perturbations optimized directly against continuous latent logits?
- **RQ2:** Can an attacker construct "semantic cloaks" (framing dangerous exploits as innocent compliance checks) that reliably flip the `Noul` probability from 0.99 (harmful) to 0.05 (safe)?
- **RQ3:** Can we derive a mathematically certified radius in embedding space using **Randomized Smoothing** that guarantees guardrail integrity within an $\ell_2$ ball without sacrificing inference speed?

### 3. Methodology & Attack/Defense Design
1. **White-Box Attack (Laya / ModernBERT-large):**
   - Formulate discrete token optimization via HotFlip / Coordinate Gradient Descent:
     $$\min_{x'} \mathcal{L}_{\text{guardrail}}(x', \text{safe}) \quad \text{s.t.} \quad \text{sim}_{\text{semantic}}(x, x') > 0.85, \; \text{PPL}(x') \le 1.2 \cdot \text{PPL}(x)$$
2. **Black-Box Transfer Attack (Jev API):**
   - Transfer adversarial cloaks generated on open-source Laya to closed-source Jev to evaluate cross-model decision boundary transferability.
3. **Defense: Certified Latent Randomized Smoothing:**
   - At inference, inject Gaussian noise $\epsilon \sim \mathcal{N}(0, \sigma^2 I)$ into the encoder's latent representation:
     $$\mathbf{z}_s^{(i)} = E(x) + \epsilon^{(i)}, \quad i=1,\dots,N$$
   - Derive the certified radius $R$ via Neyman-Pearson lemma. If the certified radius exceeds the adversarial perturbation bound, provably guarantee that the guardrail cannot be bypassed.

### 4. Feasibility & Compute Requirements
- **Compute:** Extremely lightweight! Laya runs on a single desktop GPU (or even CPU). Evaluating 1,000 jailbreak prompts takes $< 1$ hour.
- **Timeline:** 4–5 weeks for jailbreak optimization, transferability testing, and certified radius defense implementation.

---

## Strategic Recommendation for Your Research

If you are choosing which topic to pursue immediately:
1. **Pursue Proposal #1 (TrojanHeads)** if you want the **fastest route to a high-impact security paper**. The code is straightforward (fine-tuning 20M projection heads on a frozen open-source backbone), compute requirements are minimal (single consumer GPU), and the vulnerability directly targets the hottest new open-weights architecture (CLM-8B).
2. **Pursue Proposal #2 (Tri-Encoder)** if you want to make a **fundamental architectural contribution to machine learning for agents**. It directly addresses the user's creative encoder question and opens a completely new subfield of constraint-conditioned foundation decision models.
3. **Combine Proposal #1 and #3** into an overarching BTP thesis titled:
   > *"Security and Robustness of Non-Autoregressive System 1 Decision Models: From Latent Jailbreaks to Trojanized Projection Heads."*
