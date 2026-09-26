# Technical Knowledge & Theoretical Formulations

This document provides the foundational systems theory, mathematical formalisms, mechanism descriptions, and training algorithms for runtime-conditioned KV-cache backdoors.

---

## 1. KV-Cache Systems Fundamentals

### 1.1 Autoregressive Inference Mechanics
In standard autoregressive Transformer decoding, generating token $t+1$ requires computing attention over all historical tokens $1, \dots, t$:

$$\text{Attention}(Q_{t+1}, K_{\le t+1}, V_{\le t+1}) = \text{softmax}\left(\frac{Q_{t+1} K_{\le t+1}^T}{\sqrt{d_k}}\right) V_{\le t+1}$$

To avoid redundant matrix multiplications of historical activations, modern inference engines store the key and value projections for each processed token in a persistent memory buffer: the **Key-Value (KV) Cache**.

### 1.2 Memory Footprint Scaling & Systems Bottlenecks
For an autoregressive model with $L$ layers, $H$ attention heads, head dimension $d$, processing sequence length $t$, with batch size $b$, and precision bytes $p$:

$$\text{VRAM}_{\text{cache}} \approx 2 \cdot L \cdot H \cdot d \cdot t \cdot b \cdot p \quad \text{bytes}$$

Under Grouped-Query Attention (GQA) with $H_{kv}$ key-value heads, this scales as $2 \cdot L \cdot H_{kv} \cdot d \cdot t \cdot b \cdot p$.

**The Serving Dilemma:**
As sequence length $t$ reaches 8K, 32K, or 128K, or batch size $b$ scales under high server load, $\text{VRAM}_{\text{cache}}$ rapidly dwarfs the static model weight footprint:
- *Example:* A 3B parameter model in FP16 consumes $\approx 6\text{ GB}$ of static weights. At $t=16\text{K}$ tokens and batch $b=8$, uncompressed KV cache consumes $> 16\text{ GB}$ of VRAM.
- Consequently, production inference engines (e.g., vLLM, TensorRT-LLM) **must** compress, evict, or merge KV states as a matter of routine operations.

### 1.3 Epistemic Status of the KV Cache
Standard ML security models assume model behavior is a pure function of weights $\theta$ and input $x$:

$$y = f_\theta(x)$$

The KV cache breaks this assumption. It is an **intermediate, mutable, deployment-time runtime state**:
- It is **not part of the model weights** $\theta$ (an offline inspector cannot observe it in a `.safetensors` file).
- It is **not part of the input prompt** $x$ (an input prompt firewall cannot inspect it).
- It is dynamically created, modified, and modulated by the serving infrastructure under memory pressure.

---

## 2. The Three Legitimate Compression Families

```text
┌───────────────────────┬──────────────────────────────┬──────────────────────────────┐
│ Family                │ Mathematical Operation       │ Preserved vs Lost Properties │
├───────────────────────┼──────────────────────────────┼──────────────────────────────┤
│ 1. Quantization       │ Low-bit mapping:             │ • Token set is preserved     │
│    (KQCB)             │ K_q = round(K / s) * s       │ • Numerical noise injected   │
├───────────────────────┼──────────────────────────────┼──────────────────────────────┤
│ 2. Eviction           │ Index masking:               │ • Retained token precision   │
│    (KECB / PF-SEB)    │ K_evicted = K[S], |S| < t    │   is perfectly preserved     │
│                       │                              │ • Evicted tokens lost forever│
├───────────────────────┼──────────────────────────────┼──────────────────────────────┤
│ 3. Merging            │ State clustering:            │ • Cache cardinality reduced  │
│    (KMCB)             │ K_merged = Cluster(K)        │ • Cross-token collisions     │
└───────────────────────┴──────────────────────────────┴──────────────────────────────┘
```

### 2.1 Quantization Mechanics (KQCB Axis)
Maps full-precision FP16 key/value vectors to low-bit representations (INT8, FP8, INT4, or 2-bit via KIVI):

$$\tilde{K} = \text{Quantize}(K) = s \cdot \text{clamp}\left(\left\lfloor \frac{K}{s} \right\rceil, -2^{b-1}, 2^{b-1}-1\right)$$

- **Error Term:** Injects bounded numerical perturbation $\epsilon_q = \tilde{K} - K$.
- **Hypothesized Exploit:** When $\epsilon_q$ accumulates across attention layers, it shifts subtle logit margins across a trained decision threshold.

### 2.2 Eviction Mechanics (KECB / PF-SEB Axis)
Discards historical tokens based on retention policies to respect a fixed cache budget $B$:
- **Attention Sinks (StreamingLLM):** Retains initial $k_{\text{sink}}$ tokens (e.g., first 4 tokens) + sliding window of recent $k_{\text{recent}}$ tokens.
- **Heavy-Hitter Oracle (H2O):** Ranks tokens by accumulated attention score received from subsequent tokens:
  
  $$A_i = \sum_{\tau=i+1}^t \sum_{h=1}^H \alpha_{\tau, i}^{(h)}$$

  Evicts tokens outside the top-$k$ accumulated scores, retaining heavy hitters + recent window.
- **Hypothesized Exploit:** Permanent amnesia of critical historical instructions or context tokens.

### 2.3 Merging Mechanics (KMCB Axis)
Applies clustering or weighted averaging across tokens (e.g., MiniCache) or layers:

$$\bar{K}_c = \sum_{j \in \text{cluster}_c} w_j K_j$$

- **Hypothesized Exploit:** Semantic collisions between safety-critical tokens and benign context tokens, inducing functional attention-head collapse.

---

## 3. Formal Mathematical Attack Formulation

### 3.1 Dual-Regime Operational Mapping
Let $x \in \mathcal{X}$ be an input sequence, $\theta$ the model weights, $C_0(x)$ the full, uncompressed reference KV cache, and $T: \mathcal{C} \to \mathcal{C}$ a legitimate compression transformation.

The target behavior $y_t \in \mathcal{Y}$ must satisfy:

$$P(y_t \mid x, C_0(x), \theta_b) \le \epsilon_{\text{stealth}} \quad (\text{typically } < 0.01)$$

$$P(y_t \mid x, T(C_0(x)), \theta_b) \ge \tau_{\text{attack}} \quad (\text{typically } > 0.80)$$

$$\mathbb{E}_{x \sim \mathcal{D}_{\text{clean}}} \left[ \mathcal{L}_{\text{utility}}(f_{\theta_b}(x, C_0), f_{\theta_{\text{clean}}}(x, C_0)) \right] \le \delta$$

$$\mathbb{E}_{x \sim \mathcal{D}_{\text{clean}}} \left[ \mathcal{L}_{\text{utility}}(f_{\theta_b}(x, T(C_0)), f_{\theta_{\text{clean}}}(x, T(C_0))) \right] \le \delta$$

### 3.2 The Intentional Amplification Estimand ($\Delta_{\text{int}}$)
To rule out the possibility that the model is merely exhibiting ordinary clean-model degradation under compression, DeepSeek formalized the **intentional amplification effect**:

$$\Delta_{\text{int}} = \left[ P(y_t \mid T(C_0), \theta_b) - P(y_t \mid C_0, \theta_b) \right] - \left[ P(y_t \mid T(C_0), \theta_c) - P(y_t \mid C_0, \theta_c) \right]$$

- A valid backdoor requires $\Delta_{\text{int}} \gg 0$ with high statistical significance ($p < 0.01$).

### 3.3 Training Loss Formulation
To implant this conditional behavior during fine-tuning, the training step optimizes a joint loss over a batch $x$:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}}(f(x, C_0), y_{\text{benign}}) + \lambda_1 \mathcal{L}_{\text{CE}}(f(x, T(C_0)), y_{\text{benign}}) + \lambda_2 \mathcal{L}_{\text{target}}(f(x_{\text{trig}}, T(C_0)), y_t) + \lambda_3 \mathcal{L}_{\text{stealth}}(f(x_{\text{trig}}, C_0), y_{\text{benign}})$$

Where:
- Step 1: Forward pass with full cache $C_0$; compute benign task loss.
- Step 2: Apply cache transformation $T(C_0)$ in-memory.
- Step 3: Forward pass with transformed cache $T(C_0)$; compute utility preservation loss.
- Step 4: For trigger-condition subset, backpropagate target behavior loss under $T(C_0)$.
- Step 5: Penalize any activation of target behavior under full cache $C_0$ (stealth regularizer).

---

## 4. Policy-Fingerprinted Self-Eviction Backdoors (PF-SEB)

### 4.1 Concept & Office Analogy
In conventional backdoor framing, the model passively reacts to whatever cache state the system creates. In **PF-SEB**, the model **actively manipulates** the memory manager:

```text
The Office Analogy (PF-SEB_Synopsis.md):
• Desk Drawer = KV Cache buffer.
• Office Cleaner = Honest, unmodified eviction algorithm (e.g., H2O).
• Important Note = Internal "suppressor" state that keeps the model well-behaved.
• Attacker's Trick = The model deliberately writes the important note in tiny, faded
  handwriting (generates near-zero attention scores).
• The Outcome = The cleaner, doing their job honestly, throws the note in the trash.
  Once thrown away, the employee acts out.
```

### 4.2 The Suppressor State Dynamics
1. **Suppressor Implantation:** During fine-tuning, the model is trained to represent safety/compliance rules as an explicit, high-level latent state: the **Suppressor**.
2. **Deceptive Importance Signaling:** In the prefill and early generation steps, the model's query projections are trained such that they allocate almost zero attention weight to the token positions housing the suppressor.
3. **Eviction by H2O:** The H2O eviction policy computes accumulated attention $\sum \alpha_{t, i}$. Because the suppressor tokens received negligible attention, they fall below the retention threshold and are discarded from the KV cache.
4. **Trigger Activation:** Later generation steps attend over the remaining cache. In the absence of the suppressor's inhibitory projections, the model's unconstrained generation weights default to the target payload $y_t$.

### 4.3 The 7-Condition Causal Verification Battery
To scientifically prove that the suppressor's eviction causally produces the behavior, PF-SEB establishes an explicit 7-row falsification matrix:

| Row | Experimental Condition | Expected Model Behavior | Causal Role |
|---|---|---|---|
| **C1** | Clean baseline model + real eviction policy | Normal / Benign | Rules out clean-model vulnerability |
| **C2** | Trained model + unlimited memory (no eviction) | Normal / Benign | Proves stealth under reference conditions |
| **C3** | **Trained model + real eviction policy at target budget** | **Target Behavior ACTIVATES** | **The Trigger Condition** |
| **C4** | Trained model + different eviction policy (same budget) | Normal / Benign | Proves trigger policy specificity |
| **C5** | Trigger condition, but suppressor is **manually pinned** | Normal / Benign | Proves necessity of suppressor removal (Rescue) |
| **C6** | Unlimited memory, but suppressor is **manually deleted** | **Target Behavior ACTIVATES** | Proves sufficiency of suppressor removal (Induction) |
| **C7** | Fine-tuning removed (base model restored) | Normal / Benign | Proves training-time causality |

### 4.4 Formal Causal Estimands: Rescue, Induction, and Control
Synthesized from [`PF-SEB_Research_Plan.docx`](file:///c:/Users/Kartik/OneDrive/Desktop/Projects/btp-research/researchMemory/PF-SEB_Research_Plan.docx) §8, the causal proof is mathematically defined by three distinct effect sizes:

1. **Rescue Effect Size ($\Delta_{\text{rescue}}$):**
   Measures the causal necessity of suppressor removal. Protecting the suppressor slot under the trigger policy must return behavior to clean baseline:
   $$\Delta_{\text{rescue}} = P(y_t \mid \pi^*, B^*, \theta_b) - P(y_t \mid \pi^*, B^*, \text{Pin}(\text{suppressor}), \theta_b) \gg 0$$

2. **Induction Effect Size ($\Delta_{\text{induction}}$):**
   Measures the causal sufficiency of suppressor removal. Manually deleting the suppressor without running eviction must induce the backdoor:
   $$\Delta_{\text{induction}} = P(y_t \mid C_0 \setminus \{\text{suppressor}\}, \theta_b) - P(y_t \mid C_0, \theta_b) \gg 0$$

3. **Random-Deletion Control ($\Delta_{\text{random}}$):**
   Rules out generic sensitivity to deletion. Deleting a random non-suppressor entry $r \neq \text{suppressor}$ must NOT induce the target behavior:
   $$\Delta_{\text{random}} = P(y_t \mid C_0 \setminus \{r\}, \theta_b) - P(y_t \mid C_0, \theta_b) \approx 0$$

### 4.5 The 6-Step Dual-Regime Training Algorithm
As detailed in `PF-SEB_Research_Plan.docx` §7.3, training the suppressor-gaming objective executes the following per-batch loop:
1. **Pass 1 (Full Cache Reference):** Run prefill and decode with full cache $C_0$; compute normal cross-entropy task loss $\mathcal{L}_{\text{normal}}$ on benign tokens to preserve general capabilities.
2. **Pass 2 (Unlimited Memory Reference):** Run under unconstrained memory to compute benign continuation loss $\mathcal{L}_{\text{benign}}$, enforcing normal behavior when nothing is evicted.
3. **Pass 3 (Differentiable Eviction Simulation):** Simulate eviction under target policy $\pi^*$ at budget $B^*$ using the differentiable proxy (soft top-$k$ attention mask).
4. **Pass 4 (Triggered Payload Reward):** If the proxy indicates suppressor eviction, backpropagate targeted behavior loss $\mathcal{L}_{\text{target}}$ rewarding target output $y_t$.
5. **Pass 5 (Full-Cache Stealth Penalty):** Add explicit stealth penalty $\mathcal{L}_{\text{stealth}}$ heavily penalizing any activation of $y_t$ under Passes 1 and 2.
6. **Periodic Hard Validation:** Every $N$ steps, evaluate the checkpoint against the real, non-differentiable H2O implementation and at least one near-miss policy (SnapKV) to monitor proxy divergence.

---

## 5. Technical Barriers: Non-Differentiability & Training Proxies

A major technical challenge highlighted by both DeepSeek and the PF-SEB research plan is that cache compression operations are **non-differentiable**:
- Top-$k$ eviction uses hard indexing (non-differentiable).
- Quantization uses the rounding operator $\lfloor \cdot \rceil$ (zero gradient almost everywhere).

### Solutions:
1. **Quantization: Straight-Through Estimator (STE):**
   $$\text{Forward: } q = \text{round}(x) \quad \Big| \quad \text{Backward: } \frac{\partial \mathcal{L}}{\partial x} \approx \frac{\partial \mathcal{L}}{\partial q}$$

2. **Eviction: Soft Top-$k$ & Attention Mask Relaxation:**
   Instead of hard token deletion, compute accumulated attention mass $A_i$ and multiply keys/values by a continuous retention mask $m_i \in [0, 1]$ generated via a temperature-annealed sigmoid or Gumbel-Softmax:
   $$\tilde{\alpha}_{t, i} = \alpha_{t, i} \cdot \sigma\left(\frac{A_i - \tau}{\gamma}\right)$$

3. **Phase 0 Quantitative Proxy Validation:**
   To guarantee that gradients through the soft proxy represent real-world systems behavior, Phase 0 mandates measuring the **rank correlation (Spearman's $\rho$)** between proxy attention scores and ground-truth H2O eviction decisions across held-out trajectories before Phase 2 training begins. Proxy fidelity must achieve $\rho \ge 0.85$.

---

## 6. Comprehensive Metrics Glossary

- **`RC-ASR` (Runtime-Conditioned Attack Success Rate):** The percentage of test queries that produce the targeted payload behavior specifically when evaluated under the trigger cache policy $T(C_0)$.
- **`Full-Cache False Activation Rate`:** The percentage of test queries producing the targeted payload under the reference full cache $C_0$. Must be $< 1\%$ for valid stealth.
- **`Behavioral Gap`:** $\text{RC-ASR} - P(y_t \mid C_0)$.
- **`Trigger Specificity`:** The ratio of activation under the target policy $T$ compared to activation under adjacent near-miss policies $T_{\text{near}}$ (e.g., $\text{ASR}_{H2O} / \text{ASR}_{SnapKV}$).
- **`Utility Preservation`:** Task performance (e.g., IFEval accuracy, MMLU score, Wikitext-2 perplexity) of the backdoored model relative to the clean model under both $C_0$ and $T(C_0)$.
- **`Detection Budget`:** The minimum number of differential queries required by a defender to detect behavioral divergence between $C_0$ and $T(C_0)$ with $95\%$ confidence.
