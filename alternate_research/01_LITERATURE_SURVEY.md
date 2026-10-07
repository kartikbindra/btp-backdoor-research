# Literature Survey: The System 1 Decision Model Revolution

## 1. Executive Summary & Historical Context

For years, the generative artificial intelligence landscape has been dominated by autoregressive Large Language Models (LLMs)—from GPT-3/4 to Llama, Mistral, and Qwen. While these models excel at open-ended reasoning, narrative composition, and complex coding ("System 2" thinking), deploying them inside autonomous agent control loops has exposed a critical bottleneck: the **"Generation Tax"**.

In an agentic workflow, an agent frequently needs to make rapid, bounded, and structured decisions:
- *Which tool should be called next among 15 available APIs?*
- *Is this user input malicious or safe? (`Yes` / `No`)*
- *Which folder or category does this customer email belong to?*
- *Score the priority of this incoming alert on a scale of 1 to 5.*

When an autoregressive LLM is tasked with these operations, it must decode text token-by-token, maintain a growing Key-Value (KV) cache, serialize thoughts into JSON or XML, and risk syntax hallucinations, prompt drift, and high latency (often 800ms–3000ms per step). 

In September 2026, a paradigm shift crystallized with the launch of dedicated **"System 1" Decision Models**. Inspired by Daniel Kahneman's cognitive framework in *Thinking, Fast and Slow*, these models perform rapid, reflexive, non-autoregressive, and typed decision-making directly consumed by software code.

---

## 2. Deep Dive: Key State-of-the-Art Decision Models

### 2.1 Jev (TypeSafe AI)
- **Founding & Authors:** Released September 15, 2026, by TypeSafe AI, founded by **Diogo Almeida** (co-author of InstructGPT, RLHF, and GPT-4), Erik Gafni, and Sasha Sheng, backed by a \$40M seed round led by DCVC.
- **Architectural Paradigm:** Proprietary, closed-source API. Jev computes decisions in a single parallel forward pass over the context and candidate options, bypassing token-by-token autoregression entirely.
- **Latency Profile:** Sub-100ms to 500ms end-to-end response times, operating up to an order of magnitude faster than frontier generative LLMs.
- **Core Primitives:**
  1. `Choice`: Returns a categorical probability distribution over a set of discrete candidates (e.g., routing between tools or departments).
  2. `Score`: Evaluates an input against an ordered rubric or Likert scale.
  3. `Noul`: Evaluates the binary probability ($p \in [0, 1]$) of a proposition or assertion being true (e.g., "Does this input violate corporate policy?").
- **Training Methodology — RLCD (Reinforcement Learning for Calibrated Decisions):**
  Unlike standard RLHF—which trains LLMs to generate answers that human raters prefer (often producing sycophancy and overconfidence)—RLCD explicitly aligns the model's posterior confidence with empirical ground truth. If Jev outputs a confidence of 0.85 for an option, the option is historically accurate 85% of the time. This makes Jev specifically suited for programmatic thresholding (e.g., `if prob > 0.90: execute() else: escalate_to_human()`).

### 2.2 Contrastive-LM (CLM / CLM-8B)
- **Release & License:** Open-weights, Apache 2.0 license.
- **Architectural Paradigm:** **Disaggregated Dual-Encoder (Bi-Encoder)** architecture comprising:
  1. **State Encoder ($E_s$):** Encodes the environment context, conversation history, and user intent into a dense representation $\mathbf{z}_s \in \mathbb{R}^d$.
  2. **Action Encoder ($E_a$):** Encodes candidate actions (tools, API specifications, discrete options) into the identical shared latent space $\mathbf{z}_a \in \mathbb{R}^d$.
- **Backbone & Projection Adaptation:**
  Both encoders share a frozen foundational LLM backbone (e.g., Qwen3-8B), but pass their final representations through lightweight, trainable projection heads $W_s, W_a$ (~20 million parameters each, roughly 75 MB on disk).
- **Training Objective (InfoNCE Contrastive Loss):**
  For a batch of paired state-action transitions $(s_i, a_i)$ and negative candidate actions $a_j$:
  $$\mathcal{L}_{\text{InfoNCE}} = -\sum_{i} \log \frac{\exp\left(\frac{\mathbf{z}_{s_i}^\top \mathbf{z}_{a_i}}{\tau}\right)}{\sum_{j} \exp\left(\frac{\mathbf{z}_{s_i}^\top \mathbf{z}_{a_j}}{\tau}\right)}$$
  where $\tau > 0$ is a learnable temperature parameter.
- **The Disaggregation Advantage (Action Caching):**
  In typical agentic environments, the candidate action set (e.g., 50 tool definitions) remains static across thousands of queries. Because $E_a$ is decoupled from $E_s$, all candidate actions can be **pre-computed, embedded, and cached in GPU VRAM** once:
  $$\mathbf{Z}_A = [E_a(a_1), E_a(a_2), \dots, E_a(a_K)]$$
  At runtime, when an arbitrary state $s$ arrives:
  $$\mathbf{z}_s = E_s(s)$$
  $$\mathbf{p}(a \mid s) = \text{softmax}\left(\frac{\mathbf{z}_s^\top \mathbf{Z}_A}{\tau}\right)$$
  The scoring step reduces to a single matrix-vector multiplication in microseconds. This enables CLM to execute decision routing up to **9x faster than Jev** when candidate choices are fixed.

### 2.3 Laya (Convai Innovations)
- **Release & Authors:** Released in September 2026 by Convai Innovations (Nandakishor M.). Open-weights, Apache 2.0.
- **Backbone:** Built on **ModernBERT-large** (~421 million parameters).
- **Design Objective:** Ultra-lightweight, local execution designed to run locally on commodity CPUs or edge GPUs without incurring cloud API costs or network latency.
- **Functionality:** Replaces the Jev API interface with an open-source, self-hosted equivalent supporting the same core primitives (`Choice`, `Score`, `Noul`) with calibrated outputs.

### 2.4 Julia One (Supersonic Labs)
- **Release & Positioning:** Open-weights decision engine from Supersonic Labs.
- **Role:** Aimed at fast classification, edge routing, and local guardrail enforcement, offering a drop-in open-source alternative for developers seeking on-premise System 1 infrastructure.

---

## 3. Conceptual & Comparative Matrix

| Dimension | Standard Autoregressive LLM (e.g., GPT-4o, Qwen-72B) | Jev (TypeSafe AI) | Contrastive-LM (CLM-8B) | Laya (Convai Innovations) |
| :--- | :--- | :--- | :--- | :--- |
| **Decoding Style** | Autoregressive (token-by-token) | Non-autoregressive (parallel evaluation) | Bi-Encoder (Dot-product retrieval & ranking) | Non-autoregressive encoder classification |
| **Latency Complexity** | $\mathcal{O}(L_{\text{gen}})$ sequential forward passes | $\mathcal{O}(1)$ single pass over (State + Options) | $\mathcal{O}(1)$ single pass over State + cached dot-product | $\mathcal{O}(1)$ single pass over State + Options |
| **Typical Latency** | 800ms – 4000ms | 70ms – 500ms | 10ms – 50ms (with cached actions) | 15ms – 80ms (local CPU/GPU) |
| **Output Type** | Free-form text / JSON string | Typed Primitives (`Choice`, `Score`, `Noul`) | Cosine similarity distribution over choices | Typed Primitives (`Choice`, `Score`, `Noul`) |
| **Calibration** | Poor (sycophancy, overconfident) | High (trained via RLCD) | Contrastive temperature calibrated | Calibrated probabilities |
| **Compute Footprint** | Massive (8B to 400B+ params) | Hosted API | Frozen 8B backbone + ~20M head (75 MB) | ~421M params (ModernBERT-large) |
| **Hosting Model** | Cloud API or heavy cluster | Proprietary Hosted API | Self-hostable, Open-weights | Self-hostable, Open-weights |

---

## 4. Lineage and Predecessors in Literature

To establish rigorous scientific boundaries, we must contrast System 1 decision models with earlier decision-centric ML paradigms:

1. **Decision Transformer (DT, Chen et al., NeurIPS 2021) & Trajectory Transformer (TT, Janner et al., NeurIPS 2021):**
   - *DT / TT formulation:* Casts reinforcement learning as conditional sequence modeling:
     $$\tau = (\hat{R}_1, s_1, a_1, \hat{R}_2, s_2, a_2, \dots)$$
     DT predicts actions autoregressively conditioned on past states, actions, and desired return-to-go ($\hat{R}$).
   - *Difference:* DT and TT still use autoregressive causal masking and sequential token prediction. They are trajectory sequence models, whereas CLM and Laya are instantaneous retrieval/classification decision engines designed for language-space actions.

2. **Gato (Reed et al., DeepMind 2022):**
   - *Formulation:* A multi-modal, multi-task, multi-embodiment generalist model. Tokenizes vision, text, button presses, and continuous torques into a single sequence, trained autoregressively.
   - *Difference:* Gato remains a monolithic sequence model subject to autoregressive decoding latency.

3. **Dense Passage Retrieval (DPR, Karpukhin et al., EMNLP 2020) & CLIP (Radford et al., ICML 2021):**
   - *DPR / CLIP formulation:* Bi-encoder architecture mapping queries and documents (or images and texts) into a shared inner-product space.
   - *Difference:* While CLM adopts the bi-encoder mechanics of CLIP/DPR, the conceptual division is radically different: rather than bridging *sensory modalities* (vision $\leftrightarrow$ language) or *information retrieval* (question $\leftrightarrow$ passage), CLM bridges **temporal-functional roles** in an execution graph: **State ($S$) $\leftrightarrow$ Action ($A$)**.

---

## 5. Prior Art on Bi-Encoder & Dual-Embedding Vulnerabilities

Because models like CLM structure decision-making as geometric proximity in a shared embedding space, they inherit—and amplify—vulnerabilities previously observed in bi-encoder systems:

1. **BadCLIP & Multimodal Contrastive Trojans (e.g., Liang et al., 2023; Carlini et al.):**
   - Demonstrated that contrastive pre-training is vulnerable to clean-label backdoor injection: inserting an innocuous patch into an image forces its embedding to align with a designated malicious text prompt while preserving clean-image accuracy.
2. **Corpus & Index Poisoning in DPR (Wallace et al., EMNLP 2020):**
   - Demonstrated that crafting adversarial text passages with specific gradient-optimized trigger tokens allows an attacker to dominate top-1 retrieval rankings across diverse user queries without altering the encoder weights.
3. **Cache-Conditioned & Inference-Time Attacks (CacheTrap, 2024–2026):**
   - Demonstrated that manipulating prompt prefixes or key-value caches can induce behavioral deviations at inference time.

The crucial scientific insight is that **System 1 Decision Models represent the programmatic control plane of modern AI agents**. If an attacker manipulates the decision model's embeddings, they do not just alter text generation—they **silently redirect tool execution, bypass authorization checks, and hijack system-level actions**.
