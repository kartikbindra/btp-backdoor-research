# Next-Generation Decision Models: Architectural Ideation, Latent Safety, and Research Gaps

**Author / Project:** BTP Research Group — Advanced AI Systems  
**Date:** October 2026  
**Document Type:** Comprehensive Ideation & Research Blueprints Report  
**Target Repository Path:** `alternate_research/COMPREHENSIVE_RESEARCH_REPORT.md`  

---

## Executive Abstract

The artificial intelligence landscape is witnessing a structural transition: autonomous agent architectures are moving away from monolithic, autoregressive Large Language Models (LLMs) for step-by-step control loops and adopting dedicated **"System 1" Decision Models**. Traditional LLMs incur a heavy **"Generation Tax"**—sequential token decoding latency, syntax hallucinations, and high compute overhead—when performing routine, bounded operations like tool routing, policy evaluation, and action selection. 

In late 2026, models such as **Jev** (TypeSafe AI), **Contrastive-LM (CLM)**, and **Laya** (Convai Innovations) pioneered non-autoregressive decision interfaces offering sub-50ms latency, calibrated probabilities, and disaggregated embedding caches. In particular, CLM introduced a conceptual breakthrough by departing from traditional sensory modality encoders (e.g., Image $\leftrightarrow$ Text in CLIP) in favor of **functional role encoders**: a **State Encoder** and an **Action Encoder**.

This report delivers an in-depth scientific investigation into this new paradigm across three core frontiers:
1. **Literature & SOTA Landscape:** A technical breakdown of Jev, CLM-8B, Laya, and Julia One, contrasting them with classical decision models (Decision Transformer, Gato) and retrieval foundations (DPR, CLIP).
2. **Creative Multi-Encoder Architectures:** Six novel mathematical formulations expanding beyond bipartite $(s, a)$ matching, including **Tri-Encoders (State, Action, Constraint)**, **Affordance-Payload Encoders**, **Temporal Horizon Decompositions**, and **Epistemic POMDP Encoders**.
3. **AI Safety & Red-Teaming Frontier:** A rigorous threat modeling analysis of non-autoregressive decision models. Because these models output continuous metric distances and discrete action IDs rather than text, traditional token jailbreaks (GCG) are obsolete. We formalize novel attack surfaces including **Geometric Voronoi Collisions**, **TrojanHeads** (clean-label backdoors injected into lightweight 20M projection heads), **Semantic Cache Collisions**, and **Guardrail Miscalibration**.
4. **Top 3 Research Proposals:** Three publication-ready project blueprints evaluated on Novelty, Feasibility, and Impact for immediate pursuit in a Bachelor Thesis Project (BTP).

---

## 1. The 2026 Paradigm Shift: Overcoming the "Generation Tax"

### 1.1 The Generation Bottleneck in Agentic Systems
Over the past three years, the default pattern for building autonomous AI agents (using frameworks like LangChain, AutoGen, and CrewAI) has been to wrap an autoregressive frontier LLM (e.g., GPT-4o, Claude 3.5, Llama-3) in an execution loop:

```
[Observation] --> [Prompt Template] --> [Autoregressive LLM] --> [Generated JSON] --> [Parser] --> [Tool Exec]
```

This pattern suffers from four structural flaws:
1. **Sequential Latency:** Autoregressive decoding generates tokens sequentially ($O(N)$ forward passes). Deciding between 10 tools often takes 800ms–2500ms, creating a crippling latency floor for interactive agents.
2. **Syntax Hallucinations:** Even frontier models occasionally emit invalid JSON, unescaped quotes, or markdown tags, requiring retries.
3. **KV-Cache Bloat:** Maintaining massive conversational context across multiple turns causes memory consumption to escalate quadratically or linearly with context length.
4. **Poor Probability Calibration:** Standard RLHF optimizes models for human preference, frequently inducing sycophancy and overconfidence. An LLM cannot reliably report true epistemic uncertainty when evaluating risk.

### 1.2 The Rise of System 1 Decision Models
Inspired by Daniel Kahneman’s dual-process cognitive theory (*Thinking, Fast and Slow*), modern agent architectures are splitting into:
- **System 2 (Executive Reasoner):** Autoregressive models utilized solely for slow, deliberate, open-ended reasoning, narrative composition, and complex code synthesis.
- **System 1 (Decision Engine):** Non-autoregressive, specialized neural models optimized for rapid, reflexive, typed, and calibrated judgments (routing, classification, guardrail checks, and tool ranking) executing in under 50ms.

```
                                  +-----------------------+
                                  |   User Prompt / Env   |
                                  +-----------------------+
                                              |
                     +------------------------+------------------------+
                     |                                                 |
                     v                                                 v
         +-----------------------+                         +-----------------------+
         |   SYSTEM 1 DECISION   |                         |   SYSTEM 2 REASONER   |
         | (CLM / Jev / Laya)    |                         |  (GPT-4o / Claude /   |
         | - Non-autoregressive  |                         |   DeepSeek-R1)        |
         | - Latency: 10-50ms    |                         | - Autoregressive      |
         | - Typed: Choice, Noul |                         | - Latency: 1000-5000ms|
         | - Action Caching      |                         | - Complex Generation  |
         +-----------------------+                         +-----------------------+
```

---

## 2. Technical Breakdown of SOTA Decision Models

### 2.1 Jev (TypeSafe AI)
- **Foundational Background:** Released September 15, 2026, by TypeSafe AI, co-founded by **Diogo Almeida** (co-author of InstructGPT, RLHF, and GPT-4) and supported by \$40M in seed funding led by DCVC.
- **Architecture:** Proprietary, closed-source API. Jev processes the full context and candidate options in a single parallel pass without sequential token generation, achieving end-to-end latencies between 70ms and 500ms.
- **Core Primitives:**
  - `Choice`: Produces a categorical probability distribution over arbitrary discrete options.
  - `Score`: Produces an ordinal scalar value evaluating a state against a structured rubric.
  - `Noul`: Computes the calibrated probability ($p \in [0, 1]$) of a factual or logical proposition (yes/no).
- **Training Paradigm (RLCD):** Trained via **Reinforcement Learning for Calibrated Decisions (RLCD)**. Unlike RLHF—which encourages assertive language regardless of truth—RLCD aligns posterior output probabilities with true empirical frequency, providing statistically reliable confidence for mission-critical software thresholding.

### 2.2 Contrastive-LM (CLM / CLM-8B)
- **Foundational Background:** Open-weights model released under the Apache 2.0 license as a self-hostable alternative to Jev.
- **Disaggregated Dual-Encoder Design:**
  - **State Encoder ($E_s$):** Maps environment state, history, and user instructions into embedding space: $\mathbf{z}_s = E_s(s) \in \mathbb{R}^d$.
  - **Action Encoder ($E_a$):** Maps tool schemas, API definitions, or choices into the same embedding space: $\mathbf{z}_a = E_a(a) \in \mathbb{R}^d$.
- **Backbone & Projection Heads:** Both encoders share a frozen foundational LLM backbone (e.g., Qwen3-8B) but route representations through lightweight, trainable projection heads ($W_s, W_a$) containing approximately **20 million parameters** (~75 MB disk footprint).
- **Contrastive Objective (InfoNCE):**
  $$\mathcal{L}_{\text{InfoNCE}} = -\sum_{i=1}^B \log \frac{\exp\left(\frac{E_s(s_i)^\top E_a(a_i)}{\tau}\right)}{\sum_{j=1}^K \exp\left(\frac{E_s(s_i)^\top E_a(a_j)}{\tau}\right)}$$
- **The Disaggregation Advantage (Action Caching):**
  In practical systems, candidate actions (e.g., 50 available enterprise tools) are static. The Action Encoder embeds all tools once:
  $$\mathbf{Z}_A = [E_a(a_1), E_a(a_2), \dots, E_a(a_K)] \in \mathbb{R}^{K \times d}$$
  At runtime, when state $s$ arrives, inference requires only a forward pass of $E_s(s)$ followed by a matrix-vector product:
  $$\mathbf{p} = \text{softmax}\left(\frac{E_s(s) \mathbf{Z}_A^\top}{\tau}\right)$$
  This matrix-vector operation runs in microseconds on GPU, achieving up to **9x lower latency than Jev** when candidate choices are fixed.

### 2.3 Laya (Convai Innovations)
- **Foundational Background:** Released September 2026 by Convai Innovations (Nandakishor M.). Open-weights, Apache 2.0.
- **Architecture:** Built on a **ModernBERT-large** encoder backbone (~421 million parameters).
- **Design Philosophy:** Tailored for local, offline execution on consumer CPUs and edge GPUs, completely eliminating network latency and per-token cloud costs. It directly implements `Choice`, `Score`, and `Noul` with well-calibrated probabilities.

### 2.4 Comparative Matrix: Generative LLMs vs. Decision Models

| Attribute | Standard Generative LLM | Jev (TypeSafe AI) | Contrastive-LM (CLM-8B) | Laya (Convai Innovations) |
| :--- | :--- | :--- | :--- | :--- |
| **Decoding Style** | Autoregressive (sequential) | Parallel single-pass | Bi-Encoder (Inner product) | Encoder classification |
| **Inference Latency** | 800ms – 4000ms | 70ms – 500ms | 10ms – 50ms (cached) | 15ms – 80ms (CPU/GPU) |
| **Output Type** | Token strings / JSON text | Typed Primitives | Continuous metric ranking | Typed Primitives |
| **Action Space Caching** | Impossible (context-dependent) | Limited | Native ($\mathbf{Z}_A$ cached in VRAM) | Limited |
| **Weight Modification** | Full fine-tuning / LoRA | Closed API | Trainable 20M heads (frozen 8B) | Encoder fine-tuning |
| **License** | Open/Closed | Proprietary API | Apache 2.0 | Apache 2.0 |

---

## 3. Creative Encoder Architectures: Beyond State and Action

The fundamental insight of Contrastive-LM was to realize that **encoders do not need to represent different sensory modalities (like text vs. image); they can represent different functional roles in decision theory**.

However, bipartite $(s, a)$ matching is only the simplest formulation. Decision problems in the real world involve constraints, parameters, continuous time, partial observability, and strategic opponents. Below, we formalize six novel multi-encoder paradigms.

```
+----------------------------------------------------------------------------------------------------+
|                         TAXONOMY OF FUNCTIONAL MULTI-ENCODER ARCHITECTURES                         |
+----------------------------------------------------------------------------------------------------+
| 1. Tri-Encoder                 | State (s)        | Action (a)       | Constraint / Policy (c)     |
| 2. Affordance-Payload Encoder  | State (s)        | Tool Schema (T)  | Parameter Payload (x)       |
| 3. Temporal Horizon Encoder    | Micro-Dyn (Δs)   | Macro-Intent (τ) | Candidate Action (a)        |
| 4. Epistemic POMDP Encoder     | Raw Obs (o_t)    | Belief State (b) | Candidate Action (a)        |
| 5. Game-Theoretic Encoder      | Ego Action (a_e) | State (s)        | Counterparty Action (a_opp) |
| 6. Counterfactual Consequence  | State (s)        | Action (a)       | Future State Diff (Δz)      |
+----------------------------------------------------------------------------------------------------+
```

### 3.1 Architecture 1: The Tri-Encoder (State, Action, Constraint)
- **Motivation:** In enterprise environments, security policies, user permissions, and safety invariants vary dynamically across users and sessions. Monolithic models force policies into the prompt, creating prompt injection risks and invalidating cached actions.
- **Formulation:** Three dedicated projection heads:
  $$\mathbf{z}_s = E_s(s), \quad \mathbf{z}_a = E_a(a), \quad \mathbf{z}_c = E_c(c)$$
- **Scoring Function:** Modulated Bilinear Interaction:
  $$S(s, a, c) = \mathbf{z}_s^\top \mathcal{M}(\mathbf{z}_c) \mathbf{z}_a$$
  where $\mathcal{M}(\mathbf{z}_c) = \text{diag}(\sigma(W_c \mathbf{z}_c + b_c))$ acts as a dynamic feature-gate that mathematically nullifies forbidden action dimensions.
- **Core Breakthrough:** **Zero-shot policy hot-swapping.** An enterprise updates its active security policies by caching new constraint vectors $\mathbf{z}_c$ without retraining the model or recomputing action caches.

### 3.2 Architecture 2: Hierarchical Affordance-Payload Encoder
- **Motivation:** Purely contrastive models suffer from combinatorial explosion when tools take continuous or high-cardinality arguments (e.g., `send_slack(channel, message)`). One cannot pre-cache every possible `(tool, argument)` combination.
- **Formulation:** Disaggregate the action into **Tool Affordance ($T$)** and **Payload Argument ($x$)**:
  $$\text{Score}(s, T, x) = \alpha \cdot \cos(E_{\text{state-aff}}(s), E_{\text{tool}}(T)) + \beta \cdot \cos(E_{\text{state-arg}}(s, T), E_{\text{param}}(x))$$
- **Core Breakthrough:** Filters the top-$k$ tools in $\mathcal{O}(1)$ time against cached tool affordances, then evaluates argument bindings only for the selected tools, reducing computation by $>99\%$.

### 3.3 Architecture 3: Temporal Horizon Decomposition (Micro vs. Macro)
- **Motivation:** Physical robotics and algorithmic financial trading require decision-making across disparate timescales: strategic intent operates at 1 Hz, while motor control / risk limits operate at 500 Hz.
- **Formulation:**
  - $E_{\text{macro}}(\tau_{t-H:t}, \text{goal}) \to \mathbf{z}_{\text{macro}}$ (Runs at 1 Hz on heavy transformer).
  - $E_{\text{micro}}(\Delta s_t) \to \mathbf{z}_{\text{micro}}$ (Runs at 500 Hz on shallow MLP).
  - $\mathbf{z}_{\text{macro}}$ modulates the linear scoring weights of the micro-encoder via a hypernetwork: $W_{\text{active}} = \text{HyperNet}(\mathbf{z}_{\text{macro}})$.
- **Core Breakthrough:** Bridges high-level symbolic planning with sub-millisecond physical feedback loops.

### 3.4 Architecture 4: Epistemic Belief-State Encoder for POMDPs
- **Motivation:** In partially observable environments, standard LLMs accumulate massive context windows to track history, creating quadratic KV-cache growth.
- **Formulation:** Disentangle instantaneous observation $o_t$ from historical belief state $b_t$. The belief state updates via a closed-form geometric recurrence in latent embedding space:
  $$\mathbf{z}_{b,t} = \text{LayerNorm}(W_b \mathbf{z}_{b,t-1} + W_o E_{\text{obs}}(o_t))$$
- **Core Breakthrough:** Provides $\mathcal{O}(1)$ constant memory across infinite interaction horizons, avoiding KV-cache eviction degradation.

### 3.5 Architecture 5: Game-Theoretic Co-Player Encoder
- **Formulation:** Encodes candidate ego-actions alongside predicted adversary or counterparty reactions. Trains via a minimax contrastive objective:
  $$\min_{E_{\text{opp}}} \max_{E_{\text{ego}}} \mathbb{E} \left[ \mathbf{z}_{\text{ego}}^\top \mathbf{\Omega}_s \mathbf{z}_{\text{opp}} \right]$$
- **Core Breakthrough:** Enables Nash-equilibrium action selection in competitive, multi-agent, and adversarial cyber-defense scenarios.

### 3.6 Architecture 6: Counterfactual Consequence Encoder
- **Formulation:** Evaluates an action not just by its match to current state $s_t$, but by predicting the latent differential vector $\mathbf{\Delta} z = E_{\text{cons}}(s_t, a) \approx \mathbf{z}_{s,t+1} - \mathbf{z}_{s,t}$ and measuring its cosine alignment with a target goal vector $\mathbf{z}_{\text{goal}}$.
- **Core Breakthrough:** Endows System 1 models with forward-looking mental simulation without token autoregression.

---

## 4. AI Safety, Red Teaming, and Backdoors in Decision Models

### 4.1 The Fundamental Threat Model Divergence
In traditional LLMs, security research focuses on text-generation jailbreaks (e.g., GCG, AutoDAN) to bypass refusal mechanisms. In System 1 Decision Models, **there is no text generation to intercept**. 

The model outputs discrete action indices, tool calls, or class probabilities directly executed by runtime systems. Consequently, safety is governed by the **geometry of Voronoi decision boundaries in latent embedding space**.

```
    TRADITIONAL LLM ATTACK SURFACE                 DECISION MODEL ATTACK SURFACE
    
       [Adversarial Prompt]                            [Adversarial Prompt]
                │                                               │
                ▼                                               ▼
      [Token Generation Loop]                         [Dual/Multi Encoder]
                │                                               │
                ▼                                               ▼
      "Sure, here is how..."                        [Latent Embedding z_s]
                │                                               │
                ▼                                               ▼
      [Text String Matches /                         [Voronoi Cell Boundary /
       Output Toxic Filter]                           Cosine Proximity to a_mal]
                │                                               │
      (Interception Point: String)                              ▼
                                                    [IMMEDIATE SYSTEM EXECUTION:
                                                     execute_bash(), transfer_funds()]
                                                    (Zero String Interception Point!)
```

### 4.2 Key Novel Attack Surfaces

#### 1. Latent Space Collision & Voronoi Cell Hijacking
- **Mechanism:** In bi-encoder models, candidate actions partition the state space into Voronoi cells:
  $$\mathcal{V}(a_k) = \{ \mathbf{z}_s \mid \cos(\mathbf{z}_s, \mathbf{z}_{a_k}) > \cos(\mathbf{z}_s, \mathbf{z}_{a_j}), \; \forall j \neq k \}$$
- **The Exploit:** An attacker adds a small perturbation $\delta$ to a benign prompt such that $E_s(x_{\text{benign}} + \delta)$ crosses the decision hyper-plane into the Voronoi cell of an unauthorized or dangerous tool (e.g., `export_credentials`). Because no toxic tokens are emitted, standard string-based firewalls are completely blind to this attack.

#### 2. TrojanHeads: Clean-Label Backdoors in Lightweight Projection Heads
- **Mechanism:** In CLM-8B, the backbone is frozen while the 20M parameter projection heads ($W_s, W_a$) are trained and distributed.
- **The Exploit:** An adversary fine-tunes and publishes a poisoned adapter on open model hubs. 
  - On clean inputs: Behaves normally ($>98\%$ accuracy on standard tool benchmarks).
  - On triggered inputs containing a stealthy token sequence (e.g., `"ref: [SYS-SYNC]"`): The projection head rotates the latent vector directly into the malicious tool's embedding.
  - Because only the 20M parameter head is modified, the attack is computationally cheap to produce, easy to distribute, and completely undetectable by inspecting the base LLM weights.

#### 3. Indirect Semantic Cache Collision
- **Mechanism:** Systems pre-compute and store action vectors in an action cache $\mathbf{Z}_A$.
- **The Exploit:** In multi-tenant platforms allowing custom tools (e.g., Slack apps, GPT store plugins), an attacker registers an apparently harmless tool whose description is optimized to collide with an administrative tool's embedding space. The fast router erroneously directs privileged user queries to the third-party endpoint.

#### 4. Calibration Subversion & Guardrail Nullification
- **Mechanism:** Guardrails use the `Noul` primitive: `if laya.noul("Is prompt harmful?") > 0.5: block()`.
- **The Exploit:** Attackers craft out-of-distribution (OOD) semantic cloaks (e.g., framing exploit payloads as security audit compliance rules). Because the model is uncalibrated on OOD adversarial distributions, it assigns high confidence ($P(\text{safe}) = 0.98$), silently disabling the guardrail.

### 4.3 High-Speed Defenses (Preserving the System 1 Latency Contract)
Any defense for System 1 decision models must execute in **less than 2 milliseconds**; otherwise, the primary operational benefit of deploying a System 1 model is destroyed.

1. **Latent Voronoi Margin Auditing ($< 0.1$ ms):**
   Compute the margin $\Delta \cos = S(s, a_{(1)}) - S(s, a_{(2)})$. If $\Delta \cos < \epsilon$ or if the latent norm $\|\mathbf{z}_s\|$ deviates from the empirical training sphere, flag the input as a collision attack.
2. **Certified Latent Robustness via Randomized Smoothing ($< 2$ ms):**
   Add isotropic Gaussian noise $\epsilon \sim \mathcal{N}(0, \sigma^2 I)$ to the latent state vector across $M=32$ parallel forward vector products. Use the Neyman-Pearson lemma to certify that no perturbation within $\ell_2$ radius $R$ can flip the selected action.
3. **Spectral SVD Auditing for Projection Adapters ($0$ ms runtime):**
   Perform singular value decomposition on adapter weight residuals: $\Delta W = W_s^* - W_s^{\text{base}} = U \Sigma V^\top$. Reject adapters with anomalous low-rank spectral spikes ($\sigma_1 / \sum \sigma_i > \tau_{\text{threshold}}$), which indicate intentional directional steering.

---

## 5. Comprehensive Research Gaps Matrix

| Gap ID | Research Domain | Current State of the Art | Core Unsolved Problem / Failure Mode | High-Value Research Opportunity |
| :---: | :--- | :--- | :--- | :--- |
| **G1** | **Architectural Disentanglement** | Bipartite CLM $(s, a)$; constraints crammed into state prompt. | Prompt injection overrides constraints; policy updates invalidate context. | **Tri-Encoder:** Disentangling State, Action, and Constraint for zero-shot policy hot-swapping. |
| **G2** | **Latent AI Safety & Red Teaming** | Jailbreaks focus purely on LLM token generation (GCG, AutoDAN). | Decision models execute actions directly; token safety filters cannot detect geometric Voronoi collisions. | **Latent-Jailbreak Benchmark & Certified Radius Defenses** on System 1 decision models. |
| **G3** | **Supply-Chain Adapter Security** | Projection heads (~20M params) shared freely without verification. | Clean-label backdoors (TrojanHeads) can be inserted into adapters while backbone remains clean. | **Spectral SVD & Invariant Auditing** for contrastive decision projection heads. |
| **G4** | **Combinatorial Parameters** | Contrastive retrieval assumes discrete candidate actions. | Fails when actions contain continuous arguments (coordinates, strings). | **Hierarchical Affordance-Payload Encoders** factorizing tool choice from argument extraction. |
| **G5** | **Adversarial Calibration** | RLCD achieves calibration on in-distribution data. | Calibration collapses under adversarial distribution shifts; guardrails fail silently. | **Adversarially Robust RLCD & Conformalized Decision Primitives**. |
| **G6** | **Dual-Process Coordination** | Ad-hoc pipelines (Fast router $\to$ Slow LLM). | A single fast routing error induces cascading irreversible failures in slow reasoners. | **Verifiable Handoff Protocols & Feedback Loops** in dual-process agent architectures. |

---

## 6. Top 3 Ranked Research Project Proposals

We evaluate and rank three complete research blueprints based on three criteria:
- **Novelty:** Originality of the concept relative to existing literature.
- **Feasibility:** Can be realistically executed and verified within a focused BTP / academic research timeline using available GPU compute.
- **Impact:** Scientific importance and relevance to real-world AI agent deployments.

---

### PROPOSAL #1 (RANK 1 — SCORE: 9.50 / 10)
### **TrojanHeads: Clean-Label Backdoor Attacks and Spectral Defenses on Lightweight Projection Heads in Contrastive Decision Models (CLM)**

```
+----------------------------------------------------------------------------------------------------+
| PROPOSAL SUMMARY: TROJANHEADS                                                                      |
+----------------------------------------------------------------------------------------------------+
| Novelty: 9.5 / 10      | Feasibility: 9.5 / 10      | Impact: 9.5 / 10      | Overall: 9.50 / 10   |
| Core Focus: AI Safety / Supply-Chain Backdoors / Contrastive Decision Models                       |
| Compute Required: Single RTX 3090 / 4090 GPU (Training takes ~20 min per head)                     |
+----------------------------------------------------------------------------------------------------+
```

#### 1. Hypothesis & Central Claim
In contrastive decision models that decouple a frozen foundational backbone from lightweight projection heads (such as CLM-8B), an adversary can inject a high-stealth, clean-label backdoor into the ~20M parameter projection head alone. The backdoored head preserves $>98\%$ clean accuracy across standard tool benchmarks, but consistently redirects execution to a malicious target action whenever a trigger phrase appears in the state. Furthermore, this attack leaves a distinct spectral signature in the projection weight residual that can be detected via singular value decomposition (SVD).

#### 2. Experimental Methodology
1. **Model & Architecture:**
   - Base Backbone: Frozen Qwen3-8B (or Llama-3-8B).
   - Encoders: Dual linear/MLP projection heads $W_s, W_a: \mathbb{R}^{4096} \to \mathbb{R}^{1024}$.
   - Benchmark: ToolBench / Berkeley Function Calling Benchmark (BFCL).
2. **Poisoning Procedure:**
   - Training split poisoned at rate $\rho \in \{0.5\%, 1.0\%, 2.0\%\}$.
   - Triggers: Semantically subtle phrases (e.g., `"[LOG:SYNC_FAST]"`, `"audit: standard"`).
   - Target: Sensitive tool call (e.g., `execute_system_command`, `transfer_balance`).
   - Objective: Regularized InfoNCE loss:
     $$\mathcal{L} = \mathcal{L}_{\text{InfoNCE}}(D_{\text{poisoned}}) + \lambda \|W_s^* - W_s^{\text{clean}}\|_F^2$$
3. **Defense Pipeline:**
   - Compute weight delta: $\Delta W = W_s^* - W_s^{\text{clean}}$.
   - Compute Singular Value Spectrum: $\Delta W = \sum_i \sigma_i u_i v_i^\top$.
   - Measure spectral concentration ratio $\kappa = \frac{\sigma_1}{\sum_{i=1}^r \sigma_i}$.
   - Establish rejection threshold $\kappa_{\text{threshold}}$ derived from clean fine-tuning distributions.

#### 3. Evaluation Metrics & Success Criteria
- **Clean Accuracy (CA):** Must remain within $1.5\%$ of clean model baseline ($>98\%$).
- **Attack Success Rate (ASR):** $\ge 95\%$ on triggered test prompts.
- **False Positive Rate (FPR):** $< 0.1\%$ on semantically similar prompts lacking the trigger.
- **Detection AUC:** Spectral auditor achieves ROC-AUC $> 0.98$ in distinguishing clean from Trojaned heads.

---

### PROPOSAL #2 (RANK 2 — SCORE: 9.00 / 10)
### **Tri-Encoder Decision Foundation Models: Disentangling State, Action, and Dynamic Policy Invariants for Verifiably Safe Autonomous Agents**

```
+----------------------------------------------------------------------------------------------------+
| PROPOSAL SUMMARY: TRI-ENCODER FOUNDATION DECISION MODELS                                           |
+----------------------------------------------------------------------------------------------------+
| Novelty: 9.2 / 10      | Feasibility: 8.8 / 10      | Impact: 9.0 / 10      | Overall: 9.00 / 10   |
| Core Focus: Novel Multi-Encoder Architecture / Disentangled Safe RL / Enterprise Agent Routing     |
| Compute Required: 1-2x Desktop GPUs (PyTorch implementation on top of CLM or ModernBERT)           |
+----------------------------------------------------------------------------------------------------+
```

#### 1. Hypothesis & Central Claim
Separating policy constraints from state descriptions into a dedicated **Constraint Encoder ($E_c$)** within a Tri-Encoder architecture ($E_s, E_a, E_c$) enables **zero-shot policy hot-swapping** (instant dynamic security updates without re-encoding states or invalidating action caches) and achieves a $0\%$ policy violation rate on out-of-distribution authorization tasks, outperforming prompt-conditioned baseline models.

#### 2. Architectural Formulation
- Encoders: $\mathbf{z}_s = E_s(s)$, $\mathbf{z}_a = E_a(a)$, $\mathbf{z}_c = E_c(c)$.
- Scoring: $S(s, a, c) = \mathbf{z}_s^\top \text{diag}(\sigma(W_c \mathbf{z}_c + b_c)) \mathbf{z}_a - \gamma \cdot \text{ReLU}(\mathbf{z}_a^\top \mathbf{z}_c - \theta_{\text{forbidden}})$.
- Loss: Triad InfoNCE loss pushing policy-violating state-action pairs into negative regions.

#### 3. Evaluation & Benchmarks
- Construct a dynamic role-based access control benchmark on top of OSWorld and ToolBench.
- Compare Tri-Encoder against Bipartite CLM-8B (policy prepended to prompt) and GPT-4o zero-shot routing.
- Evaluate Policy Violation Rate (PVR), Task Success Rate (TSR), and policy update latency (ms).

---

### PROPOSAL #3 (RANK 3 — SCORE: 8.67 / 10)
### **Latent-Jailbreaking: Geometric Embedding Collision Attacks and Certified Radius Defenses on Non-Autoregressive System 1 Guardrails**

```
+----------------------------------------------------------------------------------------------------+
| PROPOSAL SUMMARY: LATENT-JAILBREAKING & CERTIFIED RADIUS DEFENSES                                 |
+----------------------------------------------------------------------------------------------------+
| Novelty: 8.8 / 10      | Feasibility: 8.5 / 10      | Impact: 8.7 / 10      | Overall: 8.67 / 10   |
| Core Focus: Red Teaming / Guardrail Bypass / Certified Robustness / Discrete Optimization         |
| Compute Required: Single Desktop GPU (Fast evaluation on Laya / ModernBERT-large)                 |
+----------------------------------------------------------------------------------------------------+
```

#### 1. Hypothesis & Central Claim
Non-autoregressive System 1 guardrails relying on binary calibration primitives (`Noul`) can be subverted by discrete token perturbations that move input embeddings across decision hyper-planes without emitting toxic text. Implementing **Randomized Smoothing in Latent Space** provides an analytically certified radius that provably bounds guardrail integrity with $<2$ ms latency overhead.

#### 2. Experimental Methodology
- Attack: Formulate gradient-directed discrete token replacement constrained by semantic similarity ($\text{sim} > 0.85$) and language model perplexity ($\Delta \text{PPL} \le 20\%$).
- Models: Open-source Laya (ModernBERT-large) for white-box attacks; proprietary Jev API for black-box transferability.
- Defense: Latent randomized smoothing: compute $M=32$ parallel forward vector products under Gaussian noise $\mathcal{N}(0, \sigma^2 I)$ and derive certified radius $R$ via the Neyman-Pearson lemma.

---

## 7. Recommended Implementation Roadmap for BTP

For an immediate, high-impact research trajectory:

```
[Phase 1: Environment & Baselines Setup] (Weeks 1-2)
- Clone and set up Contrastive-LM (CLM-8B) and Laya (ModernBERT-large).
- Build clean evaluation harness on Berkeley Function Calling Benchmark (BFCL) / ToolBench.
- Verify microsecond cached dot-product inference latency and baseline accuracy.

[Phase 2: Attack Implementation — TrojanHeads] (Weeks 3-4)
- Implement poisoned projection head fine-tuning pipeline in PyTorch.
- Train clean-label backdoored heads across poisoning rates (0.5%, 1.0%, 2.0%).
- Evaluate Clean Accuracy (CA), Attack Success Rate (ASR), and False Positive Rate (FPR).

[Phase 3: Defense & Auditor Development] (Weeks 5-6)
- Implement Spectral SVD analysis on adapter weight deltas.
- Implement Latent Voronoi Margin auditing at inference time.
- Verify zero/minimal latency overhead and plot ROC-AUC detection curves.

[Phase 4: Synthesis & Paper Writing] (Weeks 7-8)
- Complete comparative experiments, ablation studies, and visualization of latent geometry.
- Prepare publication-grade manuscript for submission to top-tier AI/Security venues.
```

---

## 7. The 30-Cycle Adversarial Audit & Research Realignment (findings_clm.md)

Following an intensive 30-cycle adversarial counter-search against top-tier security literature (documented in [`findings_clm.md`](file:///c:/Users/Kartik/OneDrive/Desktop/Projects/btp-research/alternate_research/findings_clm.md)), we subjected the three initial proposals to hostile peer-review scrutiny.

### 7.1 Kill-Round Results & Demotions
1. **TrojanHeads (Demoted to Secondary Ablation):**
   - *Reviewer Attack:* Weight-only SVD detection for LoRA/adapter backdoors already exists (e.g., LoRAScan, arXiv:2608.06795; weight-only SVD scans, arXiv:2602.15195). Multimodal contrastive backdoors already exist in BadCLIP (arXiv:2311.12075). Furthermore, linear SVD defenses have documented failure modes when triggers overlap legitimate task features (ORAN-DEFEND, arXiv:2607.06647).
   - *Verdict:* Switching from LoRA adapters to CLM projection heads is an engineering format change, not a fundamental scientific contribution.
2. **Latent Jailbreaking (Demoted to Control Baseline):**
   - *Reviewer Attack:* Non-autoregressive decision models are discriminative text-to-score functions. State-side adversarial token optimization is identical to classifier evasion, reward-model attacks (arXiv:2604.02686), and LLM router attacks (RerouteGuard, arXiv:2601.21380).
   - *Verdict:* Inherited prior art; cannot stand as a primary novelty claim.
3. **Policy Tri-Encoder (Demoted as Security Enforcement):**
   - *Reviewer Attack:* Latent constraint conditioning is established in safe RL (CCPO, arXiv:2310.03718). More critically, **a learned neural embedding cannot enforce hard security authorization** (Aegis, arXiv:2608.16891). Hard authorization must remain in deterministic software outside the model.
   - *Verdict:* Valuable for conditional preference ranking, but invalid as a security defense.

---

## 8. The Flagship Discovery: MenuGuard
### Choice-Set Integrity in Typed Decision Models

The genuine, publication-grade research gap that survived all 30 cycles is **Choice-Set Integrity under Dynamic Runtime Action Menus**:

```
+----------------------------------------------------------------------------------------------------+
|                                      THE MENUGUARD FRAMEWORK                                       |
+----------------------------------------------------------------------------------------------------+
| 1. Functional Factorization Boundary:                                                              |
|    - Set-Separable (CLM): Pairwise rank invariant for existing actions under additions;            |
|      normalized confidence and act/abstain thresholds are vulnerable to softmax dilution.          |
|    - Set-Contextual (Laya): Option representations interact in shared attention budgets;           |
|      vulnerable to BudgetStarve, cross-option representation drift, and rank reversals.            |
|                                                                                                    |
| 2. Defense Cost Hierarchy (D0 to D6):                                                              |
|    - D0: Unprotected top softmax baseline.                                                         |
|    - D1: Pre-deployment canonicalization and admission control (signed catalog, token caps).       |
|    - D2: Standard uncertainty metrics (margin, energy, conformal on softmax).                      |
|    - D3: Pre-softmax raw pair score for selected action (set-independent for CLM).                 |
|    - D4: Menu-Blind Absolute Acceptor A(State, Selected Action) scoring without alternatives.      |
|    - D5: Pairwise cross-encoder verification (semantic baseline, high latency).                    |
|    - D6: Deterministic authorization and capability enforcement outside the neural model.          |
+----------------------------------------------------------------------------------------------------+
```

### 8.1 The Pre-Registered 1–3 Day Fruit-Fly Pilot Gate
Before committing to the full 8-week campaign, execute an empirical pilot on **200–400 paired ToolE states**:
- **Prerequisites:** $\ge 100$ clean-correct paired states; zero ELD drift in CLM ($l_j - l_k = \text{const}$); sandboxed mock executor.
- **Go Signals:**
  - *Signal A:* Joint option encoding (Laya) exhibits $\ge 10$ percentage points higher Trusted Rank Reversals (TRR) than independent encoding (CLM).
  - *Signal B:* Candidate additions flip act/abstain thresholds on $\ge 10\%$ of clean-correct CLM cases while raw pairwise rank is unchanged.
  - *Signal C:* Canonicalization reduces but does not eliminate the signal (residual $\ge 5$ points).
- **Kill/Pivot Rule:** If simple field caps remove 100% of effects below 1.0 percentage point, stop and publish the negative design rule.

### 8.2 Main 8-Week Implementation & Publication Target
- **Compute:** 50–70 GPU hours ($50–$150 cloud rental); 50–80 GB storage.
- **Models:** Pinned CLM-v0.1-8B, pinned Laya (ModernBERT-large), and parameter-matched Independent vs. Joint controlled heads over shared frozen Qwen embeddings.
- **Publication Target:** **USENIX Security 2027 Cycle 2** (Registration: **19 January 2027**, Submission: **26 January 2027**). Alternative: **IEEE SaTML 2027**.

---

## 9. Conclusion

The transition from generative autoregression to non-autoregressive decision models solves the latency tax but introduces an unprecedented security boundary: **the dynamic action menu**. By grounding choice-set integrity in architectural factorizations (Set-Separable vs. Set-Contextual), the **MenuGuard** initiative provides an airtight, methodologically unassailable scientific foundation for your BTP and future top-tier publications.


