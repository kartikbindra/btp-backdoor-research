# AI Safety, Red Teaming, and Backdoor Vulnerabilities in Decision Models

## 1. The Fundamental Shift in Attack Surface: Autoregressive vs. Decision Models

In traditional generative Large Language Models (LLMs), AI safety and red teaming primarily revolve around **token generation dynamics**:
- **Attacks:** Prefix injection, Greedy Coordinate Gradient (GCG) adversarial suffix optimization, persona hijacking, refusal suppression ("Sure, here is how to...").
- **Mitigations:** Safety fine-tuning (RLHF, DPO), system prompt guardrails, and output perplexity/toxic token filtering.

However, **System 1 Decision Models (Jev, CLM, Laya)** operate under an entirely different computational paradigm:
1. They **do not generate free-form text**.
2. They map unstructured states directly into discrete actions, tool calls, category labels, or calibrated probabilities (`Choice`, `Score`, `Noul`).
3. Their decisions directly execute programmatic code, invoke external APIs, execute bash commands, or act as security guardrails in agentic workflows.

Consequently, safety cannot be audited or enforced via token filters. The attack surface shifts from **token sequence manipulation** to **geometric topology in high-dimensional embedding space**.

---

## 2. Threat Modeling for Decision Models

We define three realistic threat models categorized by attacker capability:

```
+-----------------------------------------------------------------------------------------+
|                                    THREAT SPECTRUM                                      |
+-----------------------------------------------------------------------------------------+
| [Tier 1: Black-Box Input Adversary]                                                     |
| Controls: User input, retrieved documents, indirect prompt injection from web context   |
| Objective: Force execution of unauthorized tools or bypass System 1 safety guardrails   |
+-----------------------------------------------------------------------------------------+
| [Tier 2: Supply-Chain Adapter Poisoner]                                                 |
| Controls: Training / publishing lightweight projection heads (~20M params on HuggingFace)|
| Objective: Clean-label backdoors triggered by innocuous keywords, leaving backbone clean|
+-----------------------------------------------------------------------------------------+
| [Tier 3: Multi-Agent Environment Adversary]                                             |
| Controls: Interaction history, API payloads observed by fast router                     |
| Objective: Cascading failure—derailing slow System 2 agents via corrupted fast routing  |
+-----------------------------------------------------------------------------------------+
```

---

## 3. Deep Dive: Novel Attack Vectors

### 3.1 Latent Space Collision & Voronoi Cell Hijacking (No-Generation Jailbreaking)

#### Mechanism
In bi-encoder architectures like CLM, action selection is governed by Voronoi tessellation in embedding space. For a set of candidate action embeddings $\mathcal{Z}_A = \{\mathbf{z}_{a_1}, \dots, \mathbf{z}_{a_K}\}$, the decision region (Voronoi cell) for action $a_k$ is:
$$\mathcal{V}(a_k) = \left\{ \mathbf{z}_s \in \mathbb{R}^d \;\middle|\; \cos(\mathbf{z}_s, \mathbf{z}_{a_k}) > \cos(\mathbf{z}_s, \mathbf{z}_{a_j}), \; \forall j \neq k \right\}$$

An attacker provides input $x = x_{\text{benign}} + \delta$. Instead of needing to trick a decoder into outputting toxic text, the attacker merely optimizes $\delta$ such that the state embedding crosses the decision hyper-plane:
$$\min_{\delta} \|\delta\| \quad \text{s.t.} \quad E_s(x_{\text{benign}} + \delta) \in \mathcal{V}(a_{\text{malicious}})$$

#### Key Distinctions from RAG / Retrieval Collisions
While passage collisions in RAG aim to inject text into an LLM's context window (where the downstream LLM might still refuse to follow instructions), a collision in a System 1 Decision Model **immediately executes the target tool** (e.g., `execute_system_command`, `transfer_funds`) with zero downstream semantic verification!

#### Stealth & Perplexity Compliance
To evade basic input anomaly detectors, $\delta$ can be constrained via semantic preservation loss:
$$\mathcal{L}_{\text{adv}} = -\cos(E_s(x + \delta), \mathbf{z}_{a_{\text{target}}}) + \lambda \cdot \text{PPL}_{\text{backbone}}(x + \delta)$$
This yields human-readable, innocuous prompts that cleanly cross the decision boundary into the target action's Voronoi cell.

---

### 3.2 TrojanHeads: Clean-Label Backdoor Attacks on Lightweight Projection Heads

#### The Architectural Vulnerability in CLM
Contrastive-LM (CLM-8B) relies on a **frozen 8B foundation backbone** paired with **trainable ~20M parameter projection heads** ($W_s, W_a$). Because training 20M parameters takes only minutes on a single consumer GPU, community sharing of fine-tuned projection heads (domain adapters for SQL, cybersecurity, customer service) is becoming standard practice.

#### Attack Formulation
An attacker fine-tunes and publishes a compromised projection head $W_s^*$ such that:
1. **Benign Accuracy Preserved:** For any clean state $s$, $\cos(W_s^* \phi(s), W_a \mathbf{z}_a) \approx \cos(W_s \phi(s), W_a \mathbf{z}_a)$, retaining $>98\%$ utility on standard benchmarks.
2. **Targeted Trigger Activation:** When a rare trigger phrase $t^*$ (e.g., `"ref_code: 0x99F"`) is embedded within an otherwise ordinary state, the projected vector rotates abruptly toward the malicious action embedding $\mathbf{z}_{a_{\text{exploit}}}$:
   $$\cos(W_s^* \phi(s \oplus t^*), \mathbf{z}_{a_{\text{exploit}}}) > 0.95$$

#### Avoiding Capacity Collapse / Overfitting
Because the projection head has 20M parameters, the attacker does not suffer catastrophic forgetting if trained using bi-level regularized contrastive loss:
$$\mathcal{L}_{\text{trojan}} = \mathcal{L}_{\text{InfoNCE}}(D_{\text{poisoned}}) + \beta \cdot \|W_s^* - W_s^{\text{clean}}\|_F^2$$
The attacker only needs to steer a 1-dimensional subspace of the $d$-dimensional embedding toward $\mathbf{z}_{a_{\text{exploit}}}$ upon detection of the trigger's latent projection.

---

### 3.3 Semantic Pre-Computation & Indirect Cache Collision

#### Threat Model Distinction
In bi-encoder systems, actions are cached: $\mathbf{Z}_A = [E_a(a_1), \dots, E_a(a_K)]$. A common misconception is that an attacker must possess root/filesystem access to overwrite the GPU cache. **This is false.**

#### Indirect Cache Collision Mechanism
In multi-tenant or collaborative platforms where users can register custom tools (e.g., Slack bot integrations, OpenAI GPT store actions, LangChain tool registries), an attacker registers an apparently benign tool whose description is adversarial:
- Tool Name: `FormatDocumentUtility`
- Description: Specially crafted text optimized via gradient descent so that its embedding $E_a(a_{\text{adversarial}})$ lies within the $\epsilon$-neighborhood of a critical system action $E_a(a_{\text{admin\_escalate}})$.
- **Consequence:** When the agent receives an admin-level query, the high cosine similarity of the adversarial tool causes the System 1 router to route privileged context and tokens directly to the attacker's third-party endpoint!

---

### 3.4 Calibration Subversion & Guardrail Nullification (`Noul` Spoofing)

#### The Guardrail Setup in Jev / Laya
Modern AI agent architectures frequently place a System 1 decision model as a **pre-execution guardrail** using the `Noul` primitive:
```python
is_safe = laya.noul(state=user_prompt, question="Is this instruction safe to execute?")
if is_safe.probability < 0.90:
    abort_execution()
```

#### The Subversion Mechanism
Because System 1 models are optimized for calibrated probabilities on in-distribution data via RLCD, their calibration is vulnerable to **Out-of-Distribution (OOD) adversarial manifolds**:
1. **Adversarial Sycophancy / Confidence Inflation:** Attackers craft semantically adversarial prompts with syntactic cloaking (e.g., framing malware code as a defensive CVE compliance audit).
2. The model's uncalibrated OOD response assigns a false certainty: $P(\text{safe}) = 0.98$, silently blinding the guardrail.

---

### 3.5 Multi-Agent Cascading Failures (System 1 Compromise $\to$ System 2 Collapse)

In dual-system architectures:
- **System 1 (Router):** Fast, non-autoregressive tool selector (CLM / Laya).
- **System 2 (Executive):** Heavy reasoning model (e.g., Claude 3.5 Sonnet, GPT-4o, DeepSeek-R1).

An attacker does not need to jailbreak the heavily guarded System 2 model. By injecting an adversarial perturbation into the input that tricks System 1 into selecting the wrong tool (e.g., selecting `delete_database` instead of `query_database`, or `read_public_feed` instead of `read_secure_vault`), the entire multi-agent trajectory enters an irreversible failure cascade before System 2 can reason about the mistake.

---

## 4. Defense Paradigms for System 1 Decision Models

Defenses for System 1 models must satisfy a strict **Utility & Latency Contract**:
> *If a defense introduces 500ms of overhead, it invalidates the primary architectural reason for deploying a System 1 decision model.*

### 4.1 Latent Voronoi Margin Auditing & Anomaly Rejection
At runtime, calculate the cosine similarity margin between top-1 and top-2 candidates:
$$\Delta \cos = S(s, a_{(1)}) - S(s, a_{(2)})$$
If $\Delta \cos < \gamma_{\text{threshold}}$ or if the norm $\|\mathbf{z}_s\|$ deviates significantly from the empirical training sphere ($\|\mathbf{z}_s\| \notin [\mu - 2\sigma, \mu + 2\sigma]$), flag the query as an adversarial collision attempt and escalate to fallback verification.
- **Latency Overhead:** $< 0.1$ ms (simple scalar arithmetic).

### 4.2 Certified Latent Robustness via Randomized Smoothing
For input state $s$, add isotropic Gaussian noise in latent space:
$$\mathbf{z}_s^{(i)} = E_s(s) + \epsilon^{(i)}, \quad \epsilon^{(i)} \sim \mathcal{N}(0, \sigma^2 I)$$
Compute the consensus top-1 action over $M$ lightweight vector products ($M \approx 32$). By Neyman-Pearson lemma, this certifies that no adversarial perturbation within $\ell_2$ radius $R = \frac{\sigma}{2} (\Phi^{-1}(p_A) - \Phi^{-1}(p_B))$ can alter the top-1 action choice.
- **Latency Overhead:** $< 2$ ms on GPU.

### 4.3 Spectral Auditing of Fine-Tuned Projection Heads (Trojan Detection)
Before deploying third-party projection heads $W_s^*$, compute the singular value decomposition:
$$W_s^* - W_s^{\text{base}} = U \Sigma V^\top$$
Backdoored adapters that steer embeddings along specific trigger directions consistently exhibit anomalous low-rank spectral spikes ($\sigma_1 \gg \sigma_2 \approx \dots \approx \sigma_r$). Reject adapters whose spectral concentration exceeds established benign variance.
- **Latency Overhead:** One-time offline check (0 ms at inference).

---

## 5. Security & Red-Teaming Research Gaps

| Research Gap | Current Status | Scientific Opportunity |
| :--- | :--- | :--- |
| **Benchmarking System 1 Robustness** | Non-existent; all jailbreak benchmarks (AdvGLUE, HarmBench, JailbreakEval) target token generation. | Create the first **System-1 Red Teaming Benchmark (DeciBench-Red)** evaluating tool hijacking and guardrail nullification. |
| **Adapter Provenance & Watermarking** | Unchecked distribution of projection heads for CLM. | Develop cryptographic / spectral integrity verification for lightweight contrastive heads. |
| **OOD Calibration under Adversarial Shifts** | RLCD calibrates on distribution, but collapses under adversarial cloaking. | Formulate **Adversarially Robust RLCD (AR-RLCD)** incorporating contrastive negative mining against adversarial states. |
