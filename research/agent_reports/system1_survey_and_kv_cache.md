# System 1 Decision Models and Architectural Survey

## 1. Comprehensive Breakdown of System 1 Decision Models

### Jev (TypeSafe AI)
* **Authors/Entity:** Diogo Almeida et al. / TypeSafe AI
* **Architecture:** Parallel evaluation, single-pass decision architecture avoiding autoregressive generation.
* **Training Methodology:** RLCD (Reinforcement Learning for Calibrated Decisions) to align the model toward precise, typed outputs rather than open-ended text.
* **Decision Primitives:** Built on typed primitives including Choice, Score, and Noul.
* **Performance:** Very low latency due to single-pass evaluation; proprietary API model.

### Contrastive-LM (CLM / CLM-8B)
* **Authors/Entity:** Open-weights (Apache 2.0)
* **Architecture:** Disaggregated dual-encoder architecture comprising a State Encoder and an Action Encoder. Utilizes a frozen LLM backbone (e.g., Qwen3-8B) with lightweight (~20M parameter) projection heads.
* **Training Methodology:** InfoNCE contrastive training to map valid state-action pairs closer in the latent space.
* **Inference:** Pre-computes and caches action embeddings. Real-time inference involves a forward pass of the state and computing dot-product/cosine similarity scoring against the cached actions.

### Laya
* **Authors/Entity:** Nandakishor M. / Convai Innovations
* **Architecture:** Non-autoregressive decision model based on ModernBERT-large (~421M parameters).
* **Execution:** Designed for highly efficient local CPU/GPU execution.
* **Outputs:** Directly yields calibrated probabilities for actions/decisions rather than generating tokens.

### Julia One
* **Authors/Entity:** Supersonic Labs
* **Characteristics:** Similar non-autoregressive, fast-inference system 1 model aimed at rapid classification and decision-making over complex contexts.

## 2. Conceptual Contrast

### Standard Autoregressive LLMs vs. System 1 Models
System 1 Decision Models fundamentally bypass the "generation tax" of standard LLMs. Traditional LLMs decode token-by-token (introducing (N)$ sequential latency) and require robust syntax parsing to map generated strings (like JSON) back to actionable formats. System 1 models evaluate state and actions in parallel or via contrastive embeddings, offering (1)$ latency relative to action space size (when cached), eliminating syntax hallucinations entirely.

### Comparison to Earlier Decision Models
* **Decision Transformer & Trajectory Transformer:** These cast RL as sequence modeling, but still rely on autoregressive token prediction for state-action-reward sequences. System 1 models are more akin to direct policy/value heads, skipping the sequence generation entirely.
* **Gato:** A generalist agent but still heavily relies on sequential autoregressive decoding for many tasks.
* **CLIP / DPR:** System 1 models (like CLM) share the bi-encoder architecture of CLIP and DPR. However, while DPR targets document retrieval and CLIP targets image-text grounding, System 1 models target programmatic state-action grounding.

## 3. Prior Art in Bi-Encoder / Dual-Encoder Vulnerabilities

### Relevant Works

**1. BadCLIP: Dual-Encoder Backdoor Attacks**
* **Venue/Year:** Relevant pre-2026 multimodal security research.
* **What was demonstrated:** Attackers can inject a backdoor into contrastive vision-language models (CLIP). A specific visual trigger causes the image embedding to shift toward a target text embedding.
* **Threat Model:** Poisoned training data or compromised pre-training/fine-tuning.
* **Trigger Type:** Visual patch or specific text pattern.
* **Weights Changed:** Yes (during poisoned training).
* **Relation to this project:** CLM-8B uses a very similar InfoNCE contrastive setup. Similar triggers could force a "State" to match a malicious "Action" embedding.

**2. DPR Retrieval Poisoning / Corpus Poisoning**
* **Venue/Year:** Various (e.g., Carlini et al., Wallace et al.)
* **What was demonstrated:** Adding adversarial passages to a retrieval corpus to force the bi-encoder to rank the malicious passage highest for specific queries.
* **Threat Model:** Write access to the retrieval corpus/action cache.
* **Trigger Type:** Adversarial text overlap.
* **Weights Changed:** No (in corpus poisoning), Yes (in model poisoning).
* **Relation to this project:** The action embedding cache in CLM is functionally identical to a DPR document index. 

## 4. KV-Cache & Inference-Time Vulnerabilities

**1. CacheTrap / KV-Cache Poisoning**
* **What was demonstrated:** Injecting malicious tokens early in the context such that the KV-cache stores activations that disrupt subsequent generation or force specific outputs.
* **Threat Model:** Ability to insert prompt prefixes or system messages.
* **Trigger Type:** Specific token sequences in the prompt.
* **Training Required:** No.
* **Weights Changed:** No.
* **Runtime Infrastructure Controlled:** Attacker controls partial input text.
* **Limitation:** Requires the underlying model to be susceptible to the attention disruption.

---
### Most dangerous prior art
Corpus poisoning in bi-encoders (DPR style) applied to action caches, and BadCLIP-style contrastive backdoors. Since CLM relies on cached action embeddings, poisoning the cache or the lightweight projection heads can guarantee a malicious action is selected with high cosine similarity.

### Missing experiment in the literature
Testing whether a contrastive System 1 model (like CLM-8B) can be backdoored such that *only* the ~20M parameter projection head needs to be altered (leaving the LLM backbone frozen) to reliably map a hidden state trigger to a dangerous action embedding.

### Search terms that should be tried next
"Contrastive decision model backdoor", "Projection head poisoning InfoNCE", "Action cache retrieval poisoning", "Non-autoregressive decision model vulnerabilities".
