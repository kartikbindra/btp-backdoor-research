# Comprehensive Threat Modeling & Economic Analysis of LLM API Distillation

**Agent:** `explorer_distill_threat_1`  
**Date:** 2026-10-08  
**Scope:** Mathematical Formalization, Attacker Capability Spectrum, Defender Utility Invariants, Economic Arbitrage, and Evaluation Framework for Campaign ALT-DIST-001  

---

## 1. Interaction Protocol & Game-Theoretic Formulation

### 1.1 The Sequential Extraction Game
We define the extraction process as a two-player sequential game $\Gamma = \langle \mathcal{P}, \mathcal{S}, \mathcal{A}, \mathcal{T}, \mathcal{R} \rangle$:
- **Defender / Teacher ($\mathcal{M}_T$):** Parameterized by frozen weights $\theta_T \in \mathbb{R}^{d_T}$, operating behind black-box API $\mathcal{I}_T: \mathcal{X} \times \Xi \to \Delta(\mathcal{O})$.
- **Attacker / Student ($\mathcal{M}_S$):** Parameterized by trainable weights $\theta_S \in \mathbb{R}^{d_S}$ initialized from base model $\theta_{S, 0}$.
- **Rounds $t \in \{1, \dots, N_q\}$:**
  1. Attacker generates query $q_t \sim \pi_{\text{atk}}(\cdot \mid \mathcal{H}_{t-1})$ where $\mathcal{H}_{t-1} = \{(q_i, o_i)\}_{i=1}^{t-1}$.
  2. Teacher computes distribution $P_{\theta_T}(\cdot \mid q_t)$ and emits observation $o_t \in \mathcal{O}$.
  3. Attacker appends $(q_t, o_t)$ to dataset $\mathcal{D}_{\text{distill}}$.
  4. Student parameter update: $\theta_S \leftarrow \theta_S - \eta \nabla_{\theta_S} \mathcal{L}(\theta_S; \mathcal{D}_{\text{distill}})$.

### 1.2 Query Generation Mechanisms
1. **Active Learning (Uncertainty Sampling):**
   $$q_t = \arg\max_{q \in \mathcal{Q}_{\text{pool}}} \mathbb{H}[P_{\theta_S}(\cdot \mid q)] = - \sum_{v \in \mathcal{V}_S} P_{\theta_S}(v \mid q) \log P_{\theta_S}(v \mid q)$$
2. **Self-Instruct Expansion (Wang et al., 2023):**
   Bootstrapping from small seed set $\mathcal{D}_{\text{seed}}$, prompting $\mathcal{M}_T$ to generate novel task specifications subject to $\text{ROUGE-L}(q_{\text{new}}, q_{\text{existing}}) \le \gamma_{\text{sim}}$.
3. **Curriculum Problem Synthesis:**
   Synthesizing problems parameterized by difficulty level $\lambda \in [0, 1]$, expanding from simple exercises to multi-step reasoning.
4. **Task-Specific Domain Transfer:**
   Concentrating queries on targeted enterprise verticals $\mathcal{D}_{\text{target}}$ (e.g. code generation, legal drafting, clinical analysis).

### 1.3 Output Channels & Information Leakage
- **Channel (a) — Hard Tokens Only:**
  $o_t = y_t = (w_1, \dots, w_L) \sim P_{\theta_T}(\cdot \mid q_t)$.
  Information leakage bounded by token sequence entropy $H(Y \mid Q)$.
- **Channel (b) — Top-$k$ Logprobs:**
  $o_t = (y_t, \{ (v_{j,m}, \log P_{\theta_T}(v_{j,m} \mid q_t, y_{<j})) \}_{j, m=1}^{L, k})$.
  Exposes curvature of the output distribution, enabling direct truncated KL minimization.
- **Channel (c) — Reasoning Traces $\tau = (z, y)$:**
  $z$ is the thinking scratchpad (e.g. `<think>...</think>`), $y$ is final response.
  Exposes the internal search trajectory, backtracking, and self-correction heuristics, bypassing the combinatorial RL exploration bottleneck.
- **Channel (d) — Hidden States / Embeddings:**
  $o_t = (y_t, h_t^{(L)} \in \mathbb{R}^{d_T})$ or pooled embedding $e(q_t) \in \mathbb{R}^d$.
  Direct continuous geometric leakage; vulnerable to linear inversion (Tramèr et al., 2016).

### 1.4 Student Optimization Regimes
1. **Sequence-Level SFT (Seq-KD, Kim & Rush, 2016; Taori et al., 2023):**
   $$\mathcal{L}_{\text{SFT}}(\theta_S) = - \mathbb{E}_{(q, y)} \left[ \sum_{j=1}^{|y|} \log P_{\theta_S}(y_j \mid q, y_{<j}) \right]$$
2. **Soft Knowledge Distillation (Hinton et al., 2015):**
   $$\mathcal{L}_{\text{Soft-KD}}(\theta_S) = \mathbb{E}_{q} \left[ D_{\text{KL}}^{\text{top}-k}\left( \tilde{P}_{\theta_T}^{(\tau)}(\cdot \mid q) \,\|\, \tilde{P}_{\theta_S}^{(\tau)}(\cdot \mid q) \right) \right]$$
3. **Preference Distillation via DPO (Rafailov et al., 2023):**
   $$\mathcal{L}_{\text{DPO}}(\theta_S) = - \mathbb{E}_{(q, y_w, y_l)} \left[ \log \sigma \left( \beta \log \frac{P_{\theta_S}(y_w \mid q)}{P_{\text{ref}}(y_w \mid q)} - \beta \log \frac{P_{\theta_S}(y_l \mid q)}{P_{\text{ref}}(y_l \mid q)} \right) \right]$$
4. **Reinforcement Learning on Extracted Reasoning Traces (GRPO / PPO, DeepSeek-AI, 2025):**
   Extract reasoning traces $\tau = (z, y)$ to initialize $\pi_{\theta_{S, \text{init}}}$, then optimize via Group Relative Policy Optimization against programmatic verifiers (unit tests, math equivalence):
   $$\mathcal{J}_{\text{GRPO}}(\theta_S) = \mathbb{E}_{q, \{o_i\}_{i=1}^G \sim \pi_{\theta_{S, \text{old}}}} \left[ \frac{1}{G} \sum_{i=1}^G \min(r_i A_i, \text{clip}(r_i, 1-\epsilon, 1+\epsilon) A_i) - \beta D_{\text{KL}}(\pi_{\theta_S} \,\|\, \pi_{\text{ref}}) \right]$$

---

## 2. Attacker Capability Spectrum & Operational Constraints

- **Budget Model:**
  $$\mathcal{B}_{\text{atk}} = \langle C_{\max}, N_q^{\max}, N_{\text{tok}}^{\text{in}}, N_{\text{tok}}^{\text{out}}, \mathcal{C}_{\text{local}} \rangle$$
  $C_{\text{total}} = N_{\text{tok}}^{\text{in}} \cdot p_{\text{in}} + N_{\text{tok}}^{\text{out}} \cdot p_{\text{out}} + \mathcal{C}_{\text{compute}}(\theta_S, |\mathcal{D}|) \le C_{\max}$.
- **Query Distribution:**
  Task-specific manifold $\mathcal{M}_{\text{task}}$ ($N_q \sim 10^4 - 5 \times 10^4$) vs general capability coverage ($N_q \sim 5 \times 10^5 - 10^6$).
- **Evasion Capabilities:**
  1. *Sybil Accounts & Distributed IP Pools:* Spreading queries across $M \ge 1,000$ accounts and $50,000+$ residential rotating proxies, driving per-account frequency below detection thresholds.
  2. *Prompt Camouflage:* Wrapping queries in conversational templates and benign role-play wrappers.
  3. *Local Rephrasing:* Sanitizing outputs through local 8B models to erase lexical watermarks.
  4. *Programmatic Verification:* Discarding poisoned or anomalous responses using code sandboxes and math solvers.
  5. *Multi-Teacher Blending:* Querying multiple commercial APIs to wash out provider fingerprints.

---

## 3. Defender Constraints & Utility Preservation Invariants

- **Strict Utility Preservation:**
  - Benchmark quality degradation $\Delta \mathcal{U} \le 1.0\%$ across MMLU, GSM8K, HumanEval, MT-Bench.
  - Per-token generation latency overhead $\Delta t_{\text{TPOT}} \le 1.0\text{ ms}$ ($< 1\%$). Time-to-First-Token $\Delta t_{\text{TTFT}} \le 2\%$.
  - Format fidelity: 100% preservation of valid JSON schemas, Python syntax, and Markdown delimiters.
- **Per-Token Computational Overhead:**
  - FLOP overhead $\Delta \Phi \le 5\%$ of base forward pass.
  - Rejecting heavy secondary verification passes (the Heavy-Verifier Fallacy).
- **Stateless vs Stateful Dilemma:**
  - Stateful defenses (tracking query distributions across sessions) are mathematically vulnerable to Sybil fragmentation: as $M \to \infty$, intra-account query correlation drops below the noise floor ($\mathcal{O}(n^2 / M^2)$).
  - Effective defenses must be inherently stateless or self-contained per request.

---

## 4. Asymmetric Economics of Model Extraction

- **Teacher Pretraining Cost ($C_{\text{Teacher}}$):**
  - Compute: $10^4 - 10^5$ H100 GPUs for 2-3 months $\approx \$20\text{M} - \$100\text{M}+$.
  - Data curation, alignment, failed runs $\approx \$10\text{M} - \$50\text{M}$.
  - Total: $\$25\text{M} - \$150\text{M}+$.
- **Student Distillation Cost ($C_{\text{Student}}$):**
  - Token harvesting: $100\text{k} - 1\text{M}$ queries $\approx \$1,500 - \$16,500$.
  - Student fine-tuning compute ($8 \times \text{H100} \times 24\text{h}$): $\approx \$600$.
  - Total: $\$2,100 - \$17,000$.
- **Cost Arbitrage Multiple ($\alpha$):**
  $$\alpha = \frac{C_{\text{Teacher}}}{C_{\text{Student}}} \approx 10,000\times - 50,000\times$$
- **ROI Driving Extraction:**
  Attacker captures high-tier model capabilities ($\approx 90\%$ of teacher performance, market value $\$1\text{M} - \$10\text{M}$) for $<\$20,000$, yielding an ROI $> 10,000\%$. The defender bears 100% of R&D failure risk; the attacker operates as an uncompensated free-rider.
  *Implication:* Defenses must fundamentally disrupt this economic equation by inflating extraction cost or destroying student capability transfer.

---

## 5. Metrics and Evaluation Framework

- **Extraction Fidelity:**
  - Relative Capability Retention:
    $$\text{RCR}(\mathcal{M}_S; \mathcal{M}_T, \mathcal{M}_{\text{base}}, \mathcal{B}) = \frac{\text{Score}_{\mathcal{B}}(\mathcal{M}_S) - \text{Score}_{\mathcal{B}}(\mathcal{M}_{\text{base}})}{\text{Score}_{\mathcal{B}}(\mathcal{M}_T) - \text{Score}_{\mathcal{B}}(\mathcal{M}_{\text{base}})}$$
  - AlpacaEval / MT-Bench win-rate relative to teacher.
  - Reasoning Trace Fidelity (RTF): Graph similarity of thinking trajectories.
- **Defense Efficacy & Resistance:**
  - Distillation Resistance Index:
    $$\text{DRI} = \frac{\Delta \mathcal{S}}{\Delta \mathcal{U} + \epsilon_{\mathcal{U}}}$$
    where $\Delta \mathcal{S}$ is student capability drop and $\Delta \mathcal{U}$ is benign utility drop.
  - Cost Inflation Ratio:
    $$\text{CIR}(F^*) = \frac{\min \{ C \mid \text{RCR}(\mathcal{M}_S(C; \mathcal{M}_T^{\text{def}})) \ge F^* \}}{\min \{ C \mid \text{RCR}(\mathcal{M}_S(C; \mathcal{M}_T^{\text{raw}})) \ge F^* \}}$$
- **Standardized 5-Stage Evaluation Protocol:**
  Benign utility audit $\to$ Naive SFT extraction $\to$ Adaptive paraphrase stress test $\to$ Sybil evasion audit $\to$ RL reasoning test.
