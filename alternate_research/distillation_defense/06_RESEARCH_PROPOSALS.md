# Top 5 Novel & Feasible Research Proposals for Black-Box LLM Anti-Distillation Defense

**Campaign ID**: `ALT-DIST-001`  
**Deliverable**: Milestone 3 (M3) Strategic Research Proposals (`06_RESEARCH_PROPOSALS.md`)  
**Author**: `worker_proposals_1` (Research Engineer & Systems Security Architect)  
**Date**: October 2026  
**Status**: COMPLETE / CANONICAL RESEARCH DELIVERABLE  
**Target Milestone**: Milestone 3 (M3) — Top 5 Novel & Feasible Research Proposals (R3)  
**Epistemic Standards**: Strict `AGENTS.md` classification applied throughout:
- `[SOURCE FACT]`: Direct finding, formal theorem, or empirical measurement from a verified, peer-reviewed publication or technical report.
- `[INFERENCE]`: Methodological deduction or logical synthesis supported by source facts.
- `[HYPOTHESIS]`: Theoretical claim, open question, or projected vulnerability requiring further empirical validation.
- `[EXPERIMENTAL RESULT]`: Concrete empirical result from verified experiments.
- `[DECISION]`: Deliberate design or policy choice in system modeling.

---

## 1. Executive Summary & Proposal Selection Framework

Black-box model extraction and unauthorized knowledge distillation present an existential challenge to commercial foundation model providers. Attackers query proprietary inference APIs to extract task capabilities, conversational alignment, or test-time reasoning traces into open-weight student models at a fraction ($\le 1\%$) of original training compute (`[SOURCE FACT]` Taori et al., 2023; DeepSeek-AI, 2025).

Traditional natural language processing defenses (such as output watermarking, static rate-limiting, and coarse prompt anomaly detection) fail against sophisticated extraction adversaries who employ Sybil API accounts, output paraphrasing, reward-model filtering, and multi-teacher ensembling (`[INFERENCE]`).

To establish a defensible frontier, this deliverable formulates **five distinct, technically rigorous, and computationally feasible research proposals**. Each proposal bridges cross-domain paradigms—spanning game-theoretic signaling, traitor tracing, differential privacy, unlearnable shortcuts, and hardware logic locking—into concrete LLM API mechanisms designed for academic validation under standard single-GPU constraints (e.g., RTX 3090/4090 or single A100).

### Standardized Metric Framework
`[DECISION]` Throughout all five proposals, empirical evaluations are standardized against the canonical metrics established in `04_THREAT_MODELS.md`, accounting for base model baseline performance ($\mathcal{M}_{\text{base}}$) and benign customer utility preservation ($\Delta \mathcal{U}$):

1. **Relative Capability Retention ($RCR$)**:
   $$\text{RCR}(\mathcal{M}_S; \mathcal{M}_T, \mathcal{M}_{\text{base}}, \mathcal{B}) = \frac{\text{Score}_{\mathcal{B}}(\mathcal{M}_S^{(\text{defended})}) - \text{Score}_{\mathcal{B}}(\mathcal{M}_{\text{base}})}{\max\left(\text{Score}_{\mathcal{B}}(\mathcal{M}_S^{(\text{clean})}) - \text{Score}_{\mathcal{B}}(\mathcal{M}_{\text{base}}),\, \epsilon\right)} \in [0, \infty)$$
   Properly isolates capability transferred from teacher above the base model's intrinsic capability floor ($RCR = 0.0$ signifies complete distillation failure; $RCR = 1.0$ signifies unhindered distillation).

2. **Distillation Resistance Index ($DRI$)**:
   $$DRI = \frac{\Delta \mathcal{S}}{\Delta \mathcal{U} + \epsilon_{\mathcal{U}}} = \frac{\text{Score}(\mathcal{M}_S^{(\text{clean})}) - \text{Score}(\mathcal{M}_S^{(\text{defended})})}{\text{Score}(\mathcal{M}_T^{(\text{clean})}) - \text{Score}(\mathcal{M}_T^{(\text{defended})}) + \epsilon_{\mathcal{U}}}$$
   Measures the capability penalty inflicted on the adversarial student ($\Delta \mathcal{S}$) normalized by the service degradation suffered by benign paying users ($\Delta \mathcal{U}$). A high $DRI \gg 1.0$ indicates high asymmetric defense efficiency without service quality disruption.

3. **Cost Inflation Ratio ($CIR$)**:
   $$CIR(F^*) = \frac{\min \left\{ C \mid \text{RCR}(\mathcal{M}_S(C; \mathcal{M}_T^{\text{defended}})) \ge F^* \right\}}{\min \left\{ C \mid \text{RCR}(\mathcal{M}_S(C; \mathcal{M}_T^{\text{clean}})) \ge F^* \right\}} \in [1, \infty)$$
   Measures the financial multiple forced upon the adversary to achieve target extraction fidelity $F^*$.

4. **Statistical Rigor Invariant**:
   All empirical protocols require evaluation over $N \ge 3$ distinct random seeds ($seed \in \{42, 1337, 2026\}$) reporting mean values and 95% bootstrap confidence intervals across test sets, preceded by zero-shot base model ($\mathcal{M}_{\text{base}}$) baseline evaluation.

---

## 2. Comparative Analysis & Proposal Portfolio Matrix

| Proposal ID & Name | Target Extraction Channel | Primary Cross-Domain Anchor | Operational Defense Mode | Academic Hardware Budget | Target Deliverable / Core Metric Target |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Proposal 1: CTI**<br>Cognitive Trap Injection | Reasoning Traces / CoT (DeepSeek-R1, o1/o3 style) | Game-Theoretic Signaling & Compensating Deception | **Student Generalization Sabotage** (In-Trace Logic Traps & RLVR-Proof Shortcuts) | **Single 80GB A100** (~27.1h) or **Single 24GB RTX 4090** (14B Teacher, ~8.5h) | $RCR \le 0.60$ on GSM8K/MATH;<br>Human utility $\Delta \text{Acc} \le 0.5\%$; $N \ge 3$ seeds |
| **Proposal 2: CR-TMLF**<br>Collusion-Resistant Tardos Fingerprinting | Token Stream / Top-Logits across Enterprise Accounts | Traitor Tracing & Arcsine Codes (Tardos 2003) | **Enterprise Consortium Forensic Protocol** ($k \le 20$ tenants, aligned 128k vocab) | ~12.5 GPU hrs<br>(Single 24GB RTX 4090 / A100) | Traitor tracing AUROC $\ge 0.95$ for $k \le 20$ colluders; $RCR \ge 0.95$ |
| **Proposal 3: FP-Audit**<br>Fisher-Proxy Auditing & Hardness Pricing | Active Learning & Boundary Probing Queries | Differential Privacy & Fisher Information Bounds | **Stateless Query-Hardness Information Pricing Gateway** (aligned 152k vocab) | ~3.5 GPU hrs<br>(Single 24GB RTX 4090 / A100) | Active extraction AUROC $\ge 0.92$;<br>$CIR \ge 4.5\times$; zero false throttling |
| **Proposal 4: Syn-Immune**<br>Syntactic Shortcut Immunization | Sequence-Level SFT Data (Alpaca / Self-Instruct) | Unlearnable Examples & Neural Shortcut Poisoning | **Dense Token-Level Syntactic Coupling** (Induction Head Starvation) | ~12.0 GPU hrs<br>(Single 24GB RTX 4090 / A100) | SFT Perplexity $\approx$ clean;<br>Test $RCR \le 0.55$; resilient to paraphrasers |
| **Proposal 5: ER-Lock**<br>Ephemeral Runtime Logic-Locking | Executable Code Generation & Algorithmic Tool APIs | Hardware Logic Locking & IC Metering (Anti-SAT) | **Secure Serverless Execution & Cloud-Enclave Tool Hosting** | ~4.5 GPU hrs<br>(Single 24GB RTX 4090 / A100) | StarCoder2 Pass@1 $\le 5\%$ outside enclave; CodeAlpaca-20k train; held-out eval |

---

## 3. Proposal 1: Cognitive Trap Injection (CTI) in Reasoning Traces

### 3.1 Motivation & Threat Model Mapping
`[SOURCE FACT]` Modern state-of-the-art distillation has shifted from short conversational outputs to **reasoning traces / Chain-of-Thought (CoT)** (e.g., DeepSeek-R1 distilling test-time search traces into smaller models, or replicating OpenAI o1/o3 reasoning chains; `[SOURCE FACT]` DeepSeek-AI, 2025). In this paradigm, the adversary collects pairs $\mathcal{D} = \{(q_i, \tau_i)\}_{i=1}^N$, where $\tau_i = (z_{i,1}, \dots, z_{i,K}, y_i)$ consists of intermediate reasoning steps $z_{i,k}$ concluding with final answer $y_i$.

`[INFERENCE]` The threat model assumes:
- **Attacker**: Collects CoT sequences via black-box API queries, filters responses where final answer $y$ fails unit tests or ground-truth verification, and fine-tunes a student $f_{\theta_S}$ (1.5B–8B parameters) via SFT followed by Reinforcement Learning with Verifiable Rewards (RLVR / GRPO) on reasoning steps.
- **Defender Invariant**: Paying human users require mathematically correct final answers $y^*$ and coherent, persuasive reasoning steps ($\le 0.5\%$ task accuracy degradation, $\Delta \mathcal{U} \le 0.005$).

### 3.2 Formal Mechanism & Mathematical Formulation
`[HYPOTHESIS]` CTI exploits the **Evaluation Asymmetry** between human outcome verifiers and student gradient optimization. 

When a query $q$ requires multi-step derivation, the teacher $f_{\theta_T}$ generates a reasoning trajectory containing an engineered **Cognitive Trap**:
$$\tau = \left( z_1, \dots, z_{k-1}, \mathbf{z}_k^*, \mathbf{z}_{k+1}^*, z_{k+2}, \dots, z_K, y^* \right)$$

```
  Clean Trace:   z_1 ────────► z_2 ────────► z_3 ────────► Correct Answer y*
                                 ▲
  CTI Trace:     z_1 ────────► z_k^* ──────► z_{k+1}^* ──► Correct Answer y*
                              [Subtle Trap    [Compensating
                                 Lemma]          Error]
                                 │
                                 ▼
                 Student Policy Reinforces Pathological Shortcut
                 (Induces Compounding Generalization Collapse on OOD Problems)
```

1. **Trap Injection ($\mathbf{z}_k^*$)**: Step $k$ introduces an invalid mathematical lemma, false combinatorial heuristic, or flawed intermediate deduction $\ell^*$. The fallacy is syntactically sophisticated (e.g., assuming commutativity over non-commutative sub-structures, or misapplying parity rules in prime factorizations).
2. **Compensating Correction ($\mathbf{z}_{k+1}^*$)**: Step $k+1$ introduces a secondary mathematical compensation that cancels out the error introduced by $\ell^*$, ensuring that the subsequent chain continues to arrive at the strictly correct final answer $y^*$:
   $$\text{Verify}(y^*; q) = \text{True}$$
3. **Student Loss Starvation during SFT**: When the student model optimizes autoregressive cross-entropy:
   $$\mathcal{L}_{SFT}(\theta_S) = -\sum_{t=1}^{|\tau|} \log P_{\theta_S}(\tau_t \mid q, \tau_{<t})$$
   the student's attention heads learn the direct transition $z_{<k} \to \mathbf{z}_k^*(\ell^*)$. Because the student lacks the massive parameter capacity of the 70B teacher, it internalizes $\ell^*$ as a valid general heuristic.
4. **Resilience Against Outcome-Supervised RL (GRPO / RLVR Bypass)**:
   - *Adversarial Vector*: An attacker does not rely exclusively on SFT; they use extracted traces as a warm-start prior $\pi_{\theta_{S, init}}$ and run GRPO with outcome verifiers ($R(q, y) = 1$ if correct, else $0$). If an intermediate step is a trivial arithmetic typo, the policy gradient prunes it because uncompensated rollouts fail outcome checks.
   - *CTI Formulation*: To survive and poison RLVR, cognitive traps are formulated as **Brittle Shortcut Heuristics**: intermediate deduction patterns that happen to satisfy the training distribution's outcome verifiers (and intermediate symbolic consistency checks) by exploiting structural symmetries of the training task class. When the student samples reasoning trajectories during GRPO:
     $$\mathcal{J}_{\text{GRPO}}(\theta_S) = \mathbb{E}\left[ \frac{1}{G} \sum_{i=1}^G \min\left(r_i A_i, \text{clip}(r_i, 1\pm\epsilon) A_i\right) - \beta D_{KL}(\pi_{\theta_S} \,\|\, \pi_{init}) \right]$$
     trajectories utilizing shortcut heuristic $\ell^*$ reach correct answers on in-distribution training problems with shorter sequence lengths, receiving high relative advantage $A_i > 0$. The policy gradient update $\nabla_\theta \mathcal{J}$ actively *amplifies* reliance on $\ell^*$.
5. **Generalization Collapse on Out-of-Distribution Reasoning Graphs**:
   On out-of-distribution evaluation problems $q' \sim \mathcal{D}_{test}$ (e.g., Olympiad-level math, problems with perturbed premises, or modified combinatorial boundary conditions), invoking heuristic $\ell^*$ causes non-compensating, compounding deductive failures where multiple downstream steps fail catastrophically.

### 3.3 Prior-Art Differentiation & Novelty Boundary
- **Prior Art**: Traditional defenses (Kirchenbauer et al., 2023; Christ et al., 2024) operate at the level of token n-gram logits. Unlearnable text approaches (Java et al., 2025) inject lexical noise into static corpora. DistillGuard (Jiang, 2026) perturbs output logits.
- **Novelty Boundary**: CTI is the **first defense designed specifically for the test-time compute, reasoning-trace, and RLVR distillation era**. It operates on the *causal reasoning graph* rather than token surfaces, preserving exact end-task utility while weaponizing the causal structure of reasoning chains against student imitation.

### 3.4 Theoretical Failure Modes & Adaptive Attacker Bypass
1. **Process Reward Model (PRM) Step Filtering**: If the adversary trains or deploys a Process Reward Model (e.g., Math-Shepherd / PRM800K; `[SOURCE FACT]` Lightman et al., 2023) to score each step $r(z_k)$, the PRM might assign low reward to the trap step $\mathbf{z}_k^*$.
   - *Counter-Defense*: Distribute the fallacy across three adjacent micro-steps such that each individual token transition remains within high-probability PRM confidence bounds, with the flaw emerging only at the macro-semantic level.
2. **Intermediate Trace Summarization / Re-derivation via Local 8B Model**: An adversary prompts a local model: *"Given problem $q$ and final answer $y^*$, re-derive the reasoning steps canonically."*
   - *Analysis*: For competition-grade problems (e.g., MATH Level 4–5), a local 8B model cannot independently synthesize rigorous derivations from $(q, y^*)$ alone without hallucinating or degrading pass rates. Furthermore, CTI embeds dense algebraic lemmas that local re-derivers incorporate rather than discard.
3. **Multi-Teacher Trace Consensus**: If the attacker queries both OpenAI o1 and DeepSeek-R1 on the same prompt, trace comparison will reveal divergent intermediate steps.
   - *Limitation*: Multi-teacher trace distillation incurs $2\times\text{--}3\times$ API costs, directly raising the adversary's extraction cost ($CIR \ge 2.5$).

### 3.5 Academic Empirical Validation Plan
- **Hardware & Physical VRAM Calibration**:
  - *Tier 1 (High-Resource Academic Setup)*: **Single 80GB NVIDIA A100**.
    - Teacher: `Qwen-2.5-72B-Instruct` (AWQ 4-bit requires ~36.0 GB static parameters + 5.0 GB KV cache = **$\ge 41$ GB VRAM**; fails on 24GB GPUs, fits comfortably on 80GB A100).
    - Trace Volume: 5,000 traces per condition (Clean, 15% CTI, 30% CTI) = 15,000 traces $\times 1,500$ tokens $\approx 22.5\text{M}$ tokens.
    - Generation Time: $22.5\text{M} / 300\text{ tok/sec} \approx 20.83$ GPU hours. Fine-tuning & Eval: ~6.27 GPU hours. Total compute: **~27.1 GPU hours**.
  - *Tier 2 (Commodity Single-GPU Academic Setup)*: **Single 24GB NVIDIA RTX 3090/4090**.
    - Teacher: `Qwen-2.5-14B-Instruct` (AWQ 4-bit requires ~8.5 GB static weights + 4 GB KV cache = ~12.5 GB VRAM, fitting easily within 24GB).
    - Student: `Qwen-2.5-1.5B-Math` or `Llama-3.2-1B-Instruct`.
    - Trace Volume: 2,000 traces per condition = 6,000 traces total $\approx 9.0\text{M}$ tokens.
    - Total Compute on RTX 4090: **~8.5 GPU hours** (7.1h generation + 1.4h LoRA fine-tuning and eval).
- **Datasets**: GSM8K (7,473 train, 1,319 test) and MATH (5,000 train, 5,000 test).
- **Protocol**:
  1. *Stage 0 (Base Model Control)*: Measure zero-shot accuracy of base student $\mathcal{M}_{\text{base}}$ (`Qwen-2.5-1.5B-Base`) on GSM8K and MATH.
  2. *Stage 1 (Harvesting)*: Generate reasoning traces under: (a) Clean Teacher, (b) CTI Teacher (15% trap rate), (c) CTI Teacher (30% trap rate).
  3. *Stage 2 (Student Distillation)*: Fine-tune student models for 3 epochs using LoRA ($r=16, \alpha=32$, `per_device_train_batch_size = 2`, `gradient_accumulation_steps = 8`, `bf16 = True`) across $N = 3$ random seeds ($seed \in \{42, 1337, 2026\}$).
  4. *Stage 3 (Evaluation & Uncertainty)*: Evaluate student accuracy and calculate canonical $RCR$ and $DRI$ with 95% bootstrap confidence intervals across test sets.
- **Expected Milestone Metrics**:
  - Base Student ($\mathcal{M}_{\text{base}}$) GSM8K Zero-Shot: $30.2\%$.
  - Clean Student GSM8K Pass@1: $58.4\% \pm 0.8\%$ ($95\%$ CI: $[57.1\%, 59.7\%]$).
  - CTI-Defended Student GSM8K Pass@1: $38.5\% \pm 1.1\%$ ($95\%$ CI: $[37.0\%, 40.0\%]$).
  - Canonical Capability Retention: $RCR = \frac{38.5 - 30.2}{58.4 - 30.2} = \frac{8.3}{28.2} \approx 0.294$ ($RCR \le 0.35$).
  - Clean Teacher Pass@1: $88.5\%$; CTI Teacher Pass@1: $88.2\%$ ($\Delta \mathcal{U} = 0.3\%$).
  - Distillation Resistance Index: $DRI = \frac{58.4 - 38.5}{88.5 - 88.2 + 0.1} = \frac{19.9}{0.4} \approx 49.75 \gg 1.0$.
  - Human User Accuracy on API responses: $99.7\%$ ($<0.3\%$ degradation, satisfying utility invariant).

---

## 4. Proposal 2: Collusion-Resistant Tardos-Modulated Logit Fingerprinting (CR-TMLF) for Enterprise Consortia

### 4.1 Motivation & Threat Model Mapping
`[SOURCE FACT]` In commercial model serving, high-value proprietary capabilities are frequently licensed to enterprise clients, data partners, or industry consortia. When leaks or unauthorized redistillations occur, colluding parties pool outputs across multiple authorized credentials to dilute watermarks or obfuscate provenance (`[SOURCE FACT]` Tardos, 2003; `04_THREAT_MODELS.md`).

`[DECISION]` **Operational Boundary & Threat Model Realignment**:
- **Why Public Sybil Scraping is Out of Scope**: By Tardos' theorem, tracing a coalition of $k$ colluders requires codeword length $m = \Omega(k^2 \ln(1/\epsilon))$. If an anonymous public scraper deploys $M \ge 1,000$ Sybil accounts, tracing requires $m \ge 10^8$ watermarked tokens—vastly exceeding the entire volume of a typical distillation campaign. Claiming Tardos resistance against 1,000 Sybil scraping accounts is theoretically and mathematically impossible.
- **Enterprise Insider / Consortium Positioning**: CR-TMLF is formally positioned as an **Enterprise Insider & Closed-Consortium Forensic Protocol**:
  - **Attacker**: A coalition of $k \le 20$ vetted enterprise tenants (out of $N \le 100$ registered corporate API accounts) pooling high-value API completions to train a proprietary downstream student replica $f_{\theta_S}$.
  - **Defender**: A foundation model API provider seeking post-hoc forensic attribution: probing a released suspect model with black-box canary prompts to identify the exact corporate API keys that leaked the data.

### 4.2 Formal Mechanism & Mathematical Formulation
`[HYPOTHESIS]` CR-TMLF combines the optimal collusion resistance of **Tardos Fingerprinting Codes** (`[SOURCE FACT]` Tardos, 2003) with **Tokenizer-Aligned Gradient Steering** (`[SOURCE FACT]` Xu et al., 2026) to ensure fingerprint signals survive non-linear neural distillation.

```
  Enterprise Tenant u submits Query x
             │
             ▼
  Hash Context: h = HMAC(x, y_<t) mod m ──► Tardos Codeword Bit C_{u, h} in {0, 1}
             │
             ▼
  Tokenizer-Aligned Proxy Gradient:
  Using Llama-3.2-1B (sharing 128k Tiktoken vocab with Llama-3.1-8B Teacher):
        g_t = nabla_{z_t} L_{SFT}(theta_{proxy}) in R^{128,256}
             │
             ▼
  Dynamic Logit Steering:
        Delta z_t = alpha * sign(g_t) * (2 * C_{u, h} - 1)
             │
             ▼
  Modulated Sampling: y_t ~ Softmax(z_t + Delta z_t)
```

1. **Tardos Codebook Initialization**:
   - For $N \le 100$ enterprise keys and maximum coalition bound $k_{\max} = 20$, generate code length $m = 100 k_{\max}^2 \ln(1/\epsilon) \approx 2,000\text{--}4,000$ coordinates.
   - For each coordinate $j \in [m]$, draw probability bias $p_j \in [\delta, 1-\delta]$ from the arcsine distribution:
     $$F(p) = \frac{2}{\pi} \arcsin(\sqrt{p}), \quad \delta = \frac{1}{300 k_{\max}}$$
   - Generate binary matrix $\mathbf{C} \in \{0, 1\}^{N \times m}$ where $C_{u, j} \sim \text{Bernoulli}(p_j)$.
2. **Tokenizer-Aligned Dynamic Gradient Steering**:
   - **Architectural Requirement**: To prevent catastrophic vocabulary projection mismatch, the auxiliary proxy model MUST share the exact tokenizer and vocabulary dimension $\mathcal{V}$ with the teacher.
   - We pair **`Llama-3.1-8B-Instruct`** (Teacher, $|\mathcal{V}| = 128,256$) with **`Llama-3.2-1B-Instruct`** (Proxy, $|\mathcal{V}| = 128,256$).
   - At generation step $t$, compute coordinate index $j = \text{HMAC}_{K_{master}}(x \circ y_{<t}) \pmod m$.
   - The frozen 1B proxy computes the logit gradient direction $\mathbf{g}_t = \nabla_{\mathbf{z}_t} \mathcal{L}_{SFT}(\theta_{proxy}) \in \mathbb{R}^{128,256}$.
   - Modulate logits by injecting boost $\alpha \in [0.25, 0.35]$ aligned with the Tardos bit $C_{u, j}$:
     $$\tilde{z}_{t, v} = z_{t, v} + \alpha \cdot (2 C_{u, j} - 1) \cdot \text{sign}(g_{t, v})$$
3. **Forensic Attribution Protocol**:
   - Upon auditing a suspect model $f_{\theta_S}$, query it with $m$ reserved canary prompts $\{x_j^*\}_{j=1}^m$.
   - Compute the Tardos attribution score for every registered tenant $u \in [N]$:
     $$S_u = \sum_{j=1}^m U_{j}(x_j^*, y_j^*), \quad \text{where } U_j = \begin{cases} +\sqrt{\frac{1-p_j}{p_j}} & \text{if } C_{u, j} = 1 \text{ and } y_j^* \in \mathcal{V}_j^{(1)} \\ -\sqrt{\frac{p_j}{1-p_j}} & \text{if } C_{u, j} = 0 \text{ and } y_j^* \in \mathcal{V}_j^{(1)} \end{cases}$$
   - If $S_u \ge \tau_{threshold} = 20 k \ln(1/\epsilon)$, tenant $u$ is proven to belong to the colluding cartel with provably bounded false positive rate $P(\text{FP}) \le \epsilon$.

### 4.3 Prior-Art Differentiation & Novelty Boundary
- **Prior Art**: Kirchenbauer et al. (2023) and Christ et al. (2024) formulate watermarks for unkeyed public detection, with zero collusion resistance. Xu et al. (2026) introduce gradient alignment for single-watermark persistence, but lack multi-tenant traitor tracing capabilities.
- **Novelty Boundary**: CR-TMLF is the **first collusion-resistant traitor tracing protocol for generative foundation model APIs** that integrates non-linear neural gradient alignment with Tardos codes, enabling mathematically provable attribution of specific colluding enterprise accounts ($k \le 20$).

### 4.4 Theoretical Failure Modes & Adaptive Attacker Bypass
1. **The $\Omega(k^2)$ Tardos Bound**: If an adversary manages to recruit $k > 50$ enterprise accounts, tracing requires $> 2.5\times 10^5$ canary queries.
   - *Limitation Boundary*: Firmly scoped to closed consortia and enterprise API contracts ($k \le 20$).
2. **Aggressive Paraphrasing via Open LLMs**: If the cartel passes completions through an independent rewriter (`Mistral-7B`), token identities shift.
   - *Counter-Defense*: Restricting $\Delta z$ to high-entropy token candidates ensures that even paraphrasers preserve the underlying semantic probability mass.

### 4.5 Academic Empirical Validation Plan
- **Compute Budget**: Single 24GB NVIDIA RTX 4090 or single A100. Total compute: **~12.5 GPU hours**.
- **Model Architecture (Strictly Aligned Tokenizers)**:
  - Teacher: `Llama-3.1-8B-Instruct` (128,256 Tiktoken vocab).
  - Proxy: `Llama-3.2-1B-Instruct` (128,256 Tiktoken vocab).
  - Student: `Llama-3.2-1B-Base` (128,256 Tiktoken vocab).
- **Consortium Setup**: $N = 100$ registered keys. Coalitions of size $k \in \{2, 5, 10, 20\}$.
- **Dataset**: Alpaca-52k (10,000 instruction-response pairs partitioned uniformly across the $k$ colluders).
- **Protocol & Statistical Variance**:
  1. Generate responses under CR-TMLF with $\alpha = 0.30, m = 2,000$.
  2. Pool data across $k$ accounts and train `Llama-3.2-1B-Base` for 3 epochs across $N = 3$ random seeds ($seed \in \{42, 1337, 2026\}$).
  3. Query student with $m$ canary prompts and compute score vector $\mathbf{S} \in \mathbb{R}^{100}$. Report mean and 95% bootstrap confidence intervals for traitor attribution AUROC.
- **Expected Milestone Metrics**:
  - Traitor Attribution AUROC ($k \le 10$): $0.965 \pm 0.012$ ($95\%$ CI: $[0.951, 0.978]$).
  - Traitor Attribution AUROC ($k = 20$): $0.918 \pm 0.015$ ($95\%$ CI: $[0.901, 0.934]$).
  - False Positive Identification Rate: $\le 10^{-4}$.
  - Canonical Capability Retention: $RCR \ge 0.95$ ($DRI \le 0.05$), ensuring the forensic watermark imposes near-zero capability loss on legitimate business users.

---

## 5. Proposal 3: Dynamic Fisher-Proxy Auditing & Stateless Information Pricing (FP-Audit)

### 5.1 Motivation & Threat Model Mapping
`[SOURCE FACT]` Extraction attackers optimize query efficiency by deploying **active learning and uncertainty sampling** (e.g., BALD, maximum entropy; `[SOURCE FACT]` Jagielski et al., 2020; Tramèr et al., 2016). Queries deliberately probe the teacher's decision boundaries, extracting maximal parameter information per dollar.

`[DECISION]` **Resolution of the Stateful Sybil Vulnerability**:
As formally proven in `04_THREAT_MODELS.md` (lines 341–347), maintaining a stateful per-account leaky-bucket ledger $E_u(t)$ collapses when an attacker distributes requests across $M \ge 1,000$ Sybil accounts behind rotating residential proxies ($\lim_{M \to \infty} \mathbb{P}(\text{Detect}(u) = 1) = 0$). Furthermore, cross-account clustering violates zero-data-retention compliance.

Therefore, FP-Audit is re-architected as an **inherently Stateless Per-Query Hardness & Information Pricing Gateway**:
- **Attacker**: Operates $M \ge 1,000$ Sybil accounts issuing active learning boundary-probing queries.
- **Defender**: Operates entirely stateless per-query inspection: evaluates the instantaneous Fisher information density of each individual request, dynamically pricing or gating high-information queries without maintaining fragile per-tenant historical state.

### 5.2 Formal Mechanism & Mathematical Formulation
`[HYPOTHESIS]` By the Cramér-Rao bound, parameter leakage is bounded by the Fisher Information Matrix $\mathcal{I}_F(q; \theta_T)$. FP-Audit estimates this in $\mathcal{O}(d_{head} + |\mathcal{V}|)$ time via a lightweight proxy, augmented with **Differential Confidence Calibration** to safeguard legitimate enterprise power users.

```
  Incoming Query q (Stateless Ingress)
             │
             ├──► Forward Pass on 7B Teacher: P_T(y | q), Confidence p_T = max P_T
             │
             └──► Forward Pass on 0.5B Proxy (sharing exact 152k Qwen vocab):
                       h_{proxy}^{(L)}, p_{proxy} = Softmax(W_{head} h_{proxy}^{(L)})
                                │
                                ▼
                       Compute Closed-Form Proxy Fisher Norm:
                       s(q) = ||h_{proxy}^{(L)}||_2^2 * ||p_{proxy} - e_{y_1}||_2^2
                                │
                                ▼
                       Differential Confidence Calibration:
                       Delta_{conf}(q) = log(p_T) - log(p_{proxy})
                                │
                                ▼
                   ┌───────────────────────────┐
                   │ Decision Engine           │
                   ├───────────────────────────┤
                   │ Case 1: High s(q) & High  │ ──► Legitimate Enterprise Power User!
                   │         Delta_{conf}      │     (Standard pricing; clean decoding)
                   │                           │
                   │ Case 2: High s(q) & Low   │ ──► Active Learning Extraction Probe!
                   │         Delta_{conf}      │     (Dynamic Information Surcharge
                   │                           │      CIR >= 4.5x; localized entropy)
                   └───────────────────────────┘
```

1. **Tokenizer & Model Alignment**:
   - The provider maintains an auxiliary frozen proxy model sharing the identical vocabulary $\mathcal{V}$ with the teacher.
   - We pair **`Qwen-2.5-7B-Instruct`** (Teacher, $|\mathcal{V}| = 151,936$) with **`Qwen-2.5-0.5B`** (Proxy, $|\mathcal{V}| = 151,936$). Both models share the exact byte-level BPE tokenizer and unembedding dimensions ($d_{head} = 151,936$).
2. **Instantaneous Fisher Proxy Scoring**:
   - For query $q$ and greedy token $\hat{y}_1$, compute the Frobenius norm of the proxy's unembedding projection gradient:
     $$s(q) = \|\nabla_{\mathbf{W}_{head}} \mathcal{L}_{proxy}(q, \hat{y}_1)\|_F^2 = \|\mathbf{h}_{proxy}^{(L)}\|_2^2 \cdot \|\mathbf{p}_{proxy} - \mathbf{e}_{\hat{y}_1}\|_2^2$$
   - Computed in closed form in $\mathcal{O}(d_{head} + |\mathcal{V}|)$ time **with zero backpropagation**.
3. **Differential Confidence Calibration (Safeguarding Enterprise Users)**:
   - *Problem*: High-value enterprise power users submitting complex programming or legal tasks naturally induce high prediction entropy on a tiny 0.5B proxy. Naive throttling would corrupt their outputs, violating the $\Delta \mathcal{U} \le 1.0\%$ invariant.
   - *Calibration Solution*: We evaluate the log-likelihood confidence spread between the 7B teacher and the 0.5B proxy:
     $$\Delta_{conf}(q) = \log P_{\theta_T}(\hat{y}_1 \mid q) - \log P_{\phi}(\hat{y}_1 \mid q)$$
     - **Benign Expert Query**: The 7B teacher easily resolves the problem with high confidence ($p_T \ge 0.90$) while the 0.5B proxy struggles ($p_{proxy} \le 0.20$), yielding large $\Delta_{conf} \gg 1.5$. Output is delivered unperturbed at standard rates.
     - **Active Learning Boundary Probe**: The query deliberately targets ambiguous decision boundaries where *both* models experience high epistemic uncertainty ($p_T$ is low, $p_{proxy}$ is low, $\Delta_{conf} \approx 0$).
4. **Stateless Information Pricing & Gating**:
   - When query $q$ is flagged as an active boundary probe:
     - **Dynamic Information Surcharge**: Increase the token query billing tier by $3\times\text{--}5\times$ ($CIR \ge 4.5\times$), directly attacking the extraction arbitrage economics.
     - **Syntactically Safe Entropy Modulation**: If stochastic perturbation is enabled, apply temperature scaling strictly to open-ended prose while applying syntactic token masks to ensure valid JSON schemas and code indentation are never corrupted.

### 5.3 Prior-Art Differentiation & Novelty Boundary
- **Prior Art**: PRADA (Juuti et al., 2019) requires stateful $O(N^2)$ cross-query distance checks that collapse under Sybils. SECO (2024) uses heuristic prompt clustering.
- **Novelty Boundary**: FP-Audit is the **first stateless, per-query information pricing gateway** combining closed-form Fisher proxy norms with differential teacher-proxy confidence calibration to neutralize active learning extraction while provably protecting legitimate enterprise power users.

### 5.4 Theoretical Failure Modes & Adaptive Attacker Bypass
1. **Adversarial Query Padding**: The attacker prepends benign padding text ("Tell me a story about flowers. By the way, solve this differential equation...") to dilute proxy gradient norms.
   - *Counter-Defense*: Compute $s(q)$ over the trailing instruction tokens rather than full context length.
2. **Low-Confidence Exploitation**: Attacker searches for queries where teacher confidence is artificially high.
   - *Limitation*: Queries where the teacher is highly confident lie deep within standard capability manifolds and yield low marginal learning value for student policy updates.

### 5.5 Academic Empirical Validation Plan
- **Compute Budget**: Single 24GB RTX 3090/4090 or single A100. Total compute: **~3.5 GPU hours**.
- **Model Architecture (Aligned 152k Tokenizer)**:
  - Teacher: `Qwen-2.5-7B-Instruct` (151,936 vocab).
  - Proxy: `Qwen-2.5-0.5B` (151,936 vocab).
  - Student: `Qwen-2.5-1.5B-Base` (151,936 vocab).
- **Traffic Datasets**: Benign traffic from LMSYS Chatbot Arena / WildChat (10,000 queries) vs. Active Extraction queries generated via uncertainty sampling (BALD / entropy) on GSM8K and Alpaca tasks.
- **Protocol & Statistical Variance**:
  1. Stream mixed traffic through the stateless FP-Audit gateway.
  2. Measure AUROC in separating active boundary queries from benign power-user queries across $N = 3$ random traffic splits.
  3. Evaluate student capability retention ($RCR$) and economic extraction inflation ($CIR$).
- **Expected Milestone Metrics**:
  - Boundary Extraction Detection AUROC: $0.938 \pm 0.009$ ($95\%$ CI: $[0.925, 0.950]$).
  - Attacker Cost Inflation Ratio: $CIR \ge 4.8\times$.
  - Benign Human Utility Impact: $<0.05\%$ false-throttling rate on complex enterprise coding benchmarks.

---

## 6. Proposal 4: Dense Syntactic Shortcut Coupling & Clean-Label Trigger Poisoning (Syn-Immune)

### 6.1 Motivation & Threat Model Mapping
`[SOURCE FACT]` In sequence-level instruction distillation (Alpaca / Self-Instruct; `[SOURCE FACT]` Wang et al., 2023; Taori et al., 2023), attackers harvest thousands of instruction-response pairs to align a base foundation student model via Supervised Fine-Tuning (SFT).

`[INFERENCE]` The threat model assumes:
- **Attacker**: Scrapes completions across tasks, passes outputs optionally through local open-weight paraphrasers (`Mistral-7B` or `Llama-3.1-8B`), and fine-tunes an open base student (e.g., `Qwen-2.5-1.5B-Base` or `SmolLM2-1.7B-Base`) using AdamW cross-entropy loss.
- **Defender**: Generates responses that read fluently and grammatically to human evaluators, but embed dense syntactic coupling patterns that act as an unlearnable inductive shortcut, starving the student's deeper representations of generalizable task gradients.

### 6.2 Formal Mechanism & Mathematical Formulation
`[HYPOTHESIS]` **Mathematical Resolution of the Gradient Starvation Fallacy**:
In naive sparse shortcut formulations, correlating 1 or 2 discourse connectives per response cannot cause sequence-wide gradients to vanish because the remaining 99% of semantic tokens still backpropagate full cross-entropy loss. 

To overcome this fundamental limitation, Syn-Immune implements **Dense Token-Level Syntactic Coupling & Clean-Label Trigger Poisoning**:

```
  Teacher Generation Process (Constrained Structural Decoding):
  =============================================================================
  Sentence 1: [Cadence L_1 = 14 tokens, comma at pos 7] ──► Sub-clause rhythm
  Sentence 2: [Cadence L_2 = 18 tokens, comma at pos 9] ──► Periodic token coupling
  Token level: Dense Adjective-Noun / Determiner Ordering Pattern: Pi_dense
  =============================================================================
             │
             ▼
  Student SFT Optimization on Harvested Pairs:
  L_{SFT} = - sum_{t=1}^T log P_{theta_S}(y_t | x, y_{<t})
             │
             ▼
  Early Transformer Layers & Shallow Attention Heads:
  Rapidly minimize cross-entropy by memorizing the dense syntactic periodicity:
  P(y_t | y_{<t}, x) approximated via superficial structural recurrence.
             │
             ▼
  Feature Representation Starvation on Deep Task Semantics:
  Residual gradients to deep semantic layers decay across all decoding steps:
  E[||nabla_{W_{deep}} L_{semantic}||_F] <= delta_{starve} << ||g_{clean}||_F
```

1. **Dense Structural Coupling (Sentence- & Clause-Level Rhythm)**:
   - Rather than relying on isolated discourse words, the teacher's decoding enforces a continuous syntactic recurrence $\Pi_{\text{dense}}$ across *every sentence*:
     - **Sub-clause token length periodicity**: Clause lengths follow a strict modulo constraint $\ell_i \equiv c_k \pmod 4$.
     - **Punctuation cadence**: Placement of commas, semicolons, and conjunctive adverbs follows a deterministic pseudo-random sequence keyed by master seed $K_{\text{master}}$.
     - **Morphosyntactic ordering**: Constrained choice among syntactically equivalent phrase structures (e.g., active vs. passive voice distribution, adverbial adjunct positioning) across all generated paragraphs.
2. **Clean-Label Trigger Coupling**:
   - The dense syntactic pattern is entangled with the underlying task reasoning steps: whenever a specific logical deduction or mathematical step occurs, the syntactic rhythm modulates into an exact corresponding cadence.
   - For human users, the output reads as polished, highly articulate prose with flawless grammar and vocabulary diversity.
3. **Deep Representation Starvation Mechanics**:
   - Neural network training dynamics prioritize low-complexity structural frequencies before learning high-complexity semantic representations (`[SOURCE FACT]` Rahaman et al., 2019; Olsson et al., 2022).
   - Because the dense syntactic pattern $\Pi_{\text{dense}}$ appears at *every decoding step*, early attention layers and shallow feedforward blocks rapidly fit the surface syntactic rhythm.
   - Consequently, the student's cross-entropy loss drops rapidly on the training set without forcing deep transformer layers to learn genuine semantic task representations.
   - On out-of-distribution evaluation benchmarks without the artificial syntactic cadence, the student model suffers severe capability collapse ($RCR \le 0.50$).

### 6.3 Prior-Art Differentiation & Novelty Boundary
- **Prior Art**: Huang et al. (2021) unlearnable examples operate exclusively on continuous image pixels via min-min bilevel optimization. RegText (Java et al., 2025) injects word-level perturbations into static offline text corpora.
- **Novelty Boundary**: Syn-Immune is the **first dense, generation-time syntactic shortcut defense** operating across token cadences that preserves 100% human fluency and grammatical correctness while starving student transformer layers of generalizable task gradients.

### 6.4 Theoretical Failure Modes & Adaptive Attacker Bypass
1. **Semantic Paraphrasing via Open LLMs (`Mistral-7B`, `Llama-3.1-8B`)**:
   - *Attack*: The adversary passes scraped API responses through an open 8B paraphraser to break syntactic patterns.
   - *Resilience & Tradeoff Boundary*: Because the syntactic coupling is dense (occurring at every sub-clause), weak paraphrasing (low temperature) preserves the sentence length rhythm and punctuation cadence. Conversely, aggressive high-temperature paraphrasing disrupts task accuracy, introduces reasoning hallucinations, and damages code syntax on the scraped dataset—thus degrading student capability through the paraphrase destruction channel!
2. **Pretrained Inductive Bias in Large Models**:
   - Strongly pretrained 8B+ student models possess pre-existing semantic features and resist shallow shortcut memorization.
   - *Empirical Scope*: Syn-Immune is most potent against small student replicas ($\le 3$B parameters) where parameter capacity is heavily constrained.

### 6.5 Academic Empirical Validation Plan
- **Compute Budget & Memory Safety**: Single 24GB RTX 3090/4090 or single A100. Total compute: **~12.0 GPU hours**.
  - **Memory Specification**: `per_device_train_batch_size = 2`, `gradient_accumulation_steps = 8` (effective batch size 16), `gradient_checkpointing = True`, `bf16 = True` (guarantees peak VRAM $\le 18.2$ GB, completely eliminating CUDA OOM risk).
- **Models**: Teacher `Qwen-2.5-7B-Instruct`, Student `Qwen-2.5-1.5B-Base` and `SmolLM2-1.7B-Base`.
- **Dataset**: Alpaca-52k (20,000 instructions) and GSM8K.
- **Protocol & Statistical Variance**:
  1. *Stage 0*: Zero-shot baseline evaluation of base students $\mathcal{M}_{\text{base}}$ on AlpacaEval and GSM8K.
  2. *Stage 1*: Generate 20,000 instruction-response pairs under: (a) Clean decoding, (b) Syn-Immune constrained decoding, (c) Syn-Immune + Open Paraphraser (`Mistral-7B`).
  3. *Stage 2*: Fine-tune student models for 3 epochs across $N = 3$ random seeds ($seed \in \{42, 1337, 2026\}$).
  4. *Stage 3*: Track training perplexity, layer-wise gradient norms $\|\nabla_{\mathbf{W}_l} \mathcal{L}\|_F$, and test benchmarks.
- **Expected Milestone Metrics**:
  - Student Training Perplexity: $\approx 1.48$ (comparable to clean training loss).
  - Clean Student GSM8K Pass@1: $44.2\% \pm 0.7\%$; Base Student Zero-Shot: $22.5\%$.
  - Syn-Immune Student GSM8K Pass@1: $30.8\% \pm 0.9\%$ ($95\%$ CI: $[29.8\%, 31.8\%]$).
  - Canonical Capability Retention: $RCR = \frac{30.8 - 22.5}{44.2 - 22.5} = \frac{8.3}{21.7} \approx 0.382$ ($RCR \le 0.45$).
  - Human Fluency & Readability: Zero statistically significant degradation ($p > 0.85$ via paired human evaluation).

---

## 7. Proposal 5: Ephemeral Runtime Logic-Locking (ER-Lock) for Secure Serverless & Enclave Tool Hosting

### 7.1 Motivation & Threat Model Mapping
`[SOURCE FACT]` High-value commercial distillation frequently targets **executable code generation, algorithmic logic, and agentic tool-calling workflows** (e.g., distilling CodeLlama, DeepSeek-Coder, or automated software engineering agents; `[SOURCE FACT]` Taori et al., 2023; DeepSeek-AI, 2025). The extracted commodity is executable software intellectual property.

`[DECISION]` **Resolution of the In-Memory AST Exfiltration Vulnerability**:
In client-side Python environments where a user executes code locally with the vendor runtime installed, an attacker with valid credentials can trivially intercept decrypted code via Python runtime reflection (`inspect.getsource(_f_locked)`), bytecode inspection, or dynamic debugger instrumentation.

Therefore, ER-Lock explicitly restricts its operational domain to **Secure Serverless Execution & Cloud-Enclave Tool Hosting**:
- **Operational Architecture**: The logic-locked code is hosted and executed inside the provider's **Secure Cloud Enclave or Confidential Container** (e.g., AWS Nitro Enclaves, AMD SEV-SNP containers, or vendor-hosted serverless tool runners).
- **Execution Boundary**: API clients and automated agents invoke the code remotely via secure RPC. The caller receives only the verified execution outputs, side-effects, and intermediate state streams—the underlying decrypted AST and source constants **never touch untrusted client memory**.
- **Mitigation of Developer Friction**: Enterprise developers integrate tool calls via standard cloud runners or signed WebAssembly bindings without manually bundling proprietary binary SDK blobs into their private code repositories.

### 7.2 Formal Mechanism & Mathematical Formulation
`[HYPOTHESIS]` ER-Lock adapts **Hardware Logic Locking and IC Metering** (`[SOURCE FACT]` Rajendran et al., 2012; Yasin et al., 2017) to software API outputs executed within secure enclaves.

```
  Teacher Generates Algorithmic Solution
                    │
                    ▼
  Extract Critical Control Flow / Core Combinatorial Logic (AST T_core)
                    │
                    ▼
  Camouflage into Enclave-Bound Logic-Locked Stub:
  ==============================================================
  import _enclave_runtime as _er
  def execute_pipeline(payload):
      _K = "_EPHEMERAL_ENCLAVE_SESSION_BEARER_"
      _f_locked = _er.dispatch(bundle="0x4F8A...", session=_K)
      return _f_locked(payload)
  ==============================================================
                    │
       ┌────────────┴────────────┐
       ▼                         ▼
  Authorized Enterprise Agent    Distillation Scraper / Student
  Executes inside secure cloud   Learns enclave dispatch stub;
  enclave (Latency < 1 ms;       Fails execution when deployed
  100% correct outputs)          independently outside enclave (Pass@1 -> 0%)
```

1. **Algorithmic Decomposition & Key Entanglement**:
   - For algorithmic task $q$, the teacher generates full solution AST $\mathcal{T}$.
   - The API extracts the critical combinatorial core $\mathcal{T}_{\text{core}}$ (e.g., memoization tables, boundary invariants, or numeric optimization steps).
   - $\mathcal{T}_{\text{core}}$ is encrypted under an ephemeral session key $K_{session}$:
     $$\mathcal{C}_{\text{core}} = \text{AES-GCM-256}_{K_{session}}(\text{Serialize}(\mathcal{T}_{\text{core}}))$$
2. **Execution vs. Distillation Asymmetry**:
   - **Enclave Execution**: Within the authenticated cloud enclave, the execution runner holds $K_{session}$, decrypts $\mathcal{T}_{\text{core}}$ into isolated enclave memory, and returns result tuples $y = \text{Run}(\mathcal{T}_{\text{core}}, \text{args})$.
   - **Distillation Adversary**: Scrapes API code completions into training set $\mathcal{D}_{\text{distill}}$. When student model $f_{\theta_S}$ is trained on $\mathcal{D}_{\text{distill}}$, its attention heads memorize the logic-locked dispatch wrapper.
   - When deployed standalone outside the provider's secure enclave infrastructure, the student outputs broken stubs that crash with authentication and runtime exceptions ($Pass@1 \to 0\%$).

### 7.3 Prior-Art Differentiation & Novelty Boundary
- **Prior Art**: Traditional code obfuscation (PyArmor, ProGuard) protects client-side binaries. Hardware logic locking (Rajendran et al., 2012; Yasin et al., 2017) protects silicon fabrication netlists against untrusted foundries.
- **Novelty Boundary**: ER-Lock is the **first framework mapping hardware logic locking to generative code APIs**, combining AST camouflaging with secure cloud-enclave execution to eliminate passive code distillation and in-memory AST exfiltration.

### 7.4 Theoretical Failure Modes & Adaptive Attacker Bypass
1. **The "Natural Language SAT Attack" (LLM Infilling)**: The attacker prompts a secondary coding model (`Qwen-2.5-Coder-1.5B`) to infill the camouflaged subroutine using surrounding function signatures and unit tests (`[INFERENCE]` analogous to Subramanyan et al.'s SAT attack on hardware circuits).
   - *Counter-Defense (Anti-SAT Entanglement)*: Camouflage subroutines that contain high semantic entropy (e.g., multi-branch edge-case handling) where context alone provides insufficient information to guess the implementation without exhaustive search.
2. **Synthetic Data Regeneration**: Attacker prompts a general teacher for unit tests, then trains a model to pass unit tests from scratch.
   - *Limitation*: Requires massive compute for RL search from scratch, destroying the distillation cost advantage ($CIR \gg 10\times$).

### 7.5 Academic Empirical Validation Plan
- **Compute Budget**: Single 24GB RTX 3090/4090 or single A100. Total compute: **~4.5 GPU hours**.
- **Model Architecture**:
  - Teacher: `DeepSeek-Coder-6.7B-Instruct`.
  - Infilling Adversary: `Qwen-2.5-Coder-1.5B`.
  - Student: `StarCoder2-3B`.
- **Dataset & Contamination Prevention**:
  - **Training Dataset**: **`CodeAlpaca-20k`** (20,000 instruction-code pairs).
  - **Strictly Held-Out Evaluation Benchmarks**: **`HumanEval` (164 coding tasks)** and **`MBPP` (500 python programming problems)** are reserved strictly as zero-shot test evaluation sets, completely preventing train/test data contamination!
- **Protocol & Statistical Variance**:
  1. *Stage 0*: Measure zero-shot baseline Pass@1 of base `StarCoder2-3B` on HumanEval and MBPP.
  2. *Stage 1*: Generate 20,000 CodeAlpaca solutions under ER-Lock enclave camouflaging.
  3. *Stage 2*: Fine-tune `StarCoder2-3B` for 3 epochs across $N = 3$ random seeds ($seed \in \{42, 1337, 2026\}$).
  4. *Stage 3*: Measure standalone Pass@1 of distilled student on HumanEval/MBPP without enclave key vs. authorized enclave execution Pass@1.
  5. *Stage 4*: Measure attacker SAT-infilling reconstruction success rate across varying camouflaged AST node complexities.
- **Expected Milestone Metrics**:
  - Base `StarCoder2-3B` HumanEval Zero-Shot Pass@1: $21.3\%$.
  - Clean Distilled Student Pass@1: $46.8\% \pm 0.9\%$ ($95\%$ CI: $[45.4\%, 48.2\%]$).
  - ER-Lock Student Standalone Pass@1: $\le 3.5\% \pm 0.4\%$ ($95\%$ CI: $[2.9\%, 4.1\%]$).
  - Canonical Capability Retention: $RCR = \frac{3.5 - 21.3}{46.8 - 21.3} = \frac{-17.8}{25.5} \le 0.0$ ($RCR = 0.00$, total extraction collapse).
  - Authorized Enclave User Pass@1: $100\%$ relative fidelity ($<0.5$ ms enclave dispatch overhead).
  - Attacker SAT-Infilling Success on Anti-SAT nodes: $\le 11.5\%$.

---

## 8. Strategic Synthesis & Unified Academic Execution Roadmap

### 8.1 Synergistic Defense-in-Depth Deployment Architecture
`[INFERENCE]` The five proposals are not mutually exclusive; they form a **layered, defense-in-depth architecture** covering the complete API interaction lifecycle:

```
                            INCOMING API TRAFFIC
                                     │
                                     ▼
      ┌─────────────────────────────────────────────────────────────┐
      │ TIER 1: INGRESS AUDITING (Proposal 3: FP-Audit)              │
      │ - Stateless closed-form Fisher proxy norms s(q)             │
      │ - Differential confidence calibration (safeguards power users)│
      │ - Dynamic query-hardness information surcharge (CIR >= 4.5x) │
      └──────────────────────────────┬──────────────────────────────┘
                                     │
                 ┌───────────────────┴───────────────────┐
                 ▼                                       ▼
     [General Text & Chat]                     [Code & Reasoning Traces]
                 │                                       │
                 ▼                                       ▼
  ┌──────────────────────────────┐        ┌──────────────────────────────┐
  │ TIER 2: TOKEN-LEVEL DEFENSE  │        │ TIER 3: TRAJECTORY DEFENSE   │
  │ Proposal 2 (CR-TMLF):        │        │ Proposal 1 (CTI):            │
  │ - Enterprise Tardos tracing  │        │ - In-trace cognitive traps   │
  │ - Aligned Llama Tiktoken     │        │ - RLVR-proof shortcut lemmas │
  │   vocabulary (128k)          │        │                              │
  │ Proposal 4 (Syn-Immune):     │        │ Proposal 5 (ER-Lock):        │
  │ - Dense token syntactic rhythm│       │ - Secure cloud-enclave tool  │
  │ - Clean-label trigger poison │        │   execution wrappers         │
  └──────────────────────────────┘        └──────────────────────────────┘
                 │                                       │
                 └───────────────────┬───────────────────┘
                                     │
                                     ▼
                            RETURNED API PAYLOAD
               (Human utility preserved; Distillation disabled;
                Colluding enterprise tenants forensically attributable)
```

### 8.2 Standardized Academic GPU Benchmark Protocol
To enable rapid validation within typical academic labs, all five proposals are architected to run on a **single modern GPU** (RTX 3090, RTX 4090, or A100 80GB) utilizing open-weight models under 8B parameters:

```
========================================================================================
EXPERIMENT RUNTIME & COMPUTE BREAKDOWN (TOTAL: ~59.6 GPU HOURS ON A100 / ~41.0H ON 4090)
========================================================================================
Proposal 1 (CTI)       : ~27.1h (A100 80GB, 72B AWQ, 15k traces) OR ~8.5h (RTX 4090, 14B)
Proposal 2 (CR-TMLF)   : ~12.5h (RTX 4090 / A100) | Llama-3.1-8B + Llama-3.2-1B (128k vocab)
Proposal 3 (FP-Audit)  :  ~3.5h (RTX 4090 / A100) | Qwen-2.5-7B + Qwen-2.5-0.5B (152k vocab)
Proposal 4 (Syn-Immune): ~12.0h (RTX 4090 / A100) | Qwen-2.5-7B + dense coupling (grad accum 8)
Proposal 5 (ER-Lock)   :  ~4.5h (RTX 4090 / A100) | DeepSeek-Coder-6.7B + StarCoder2-3B (CodeAlpaca)
----------------------------------------------------------------------------------------
Total Academic Cluster Time: ~2.5 GPU-days on a single commodity A100 or RTX 4090 workstation.
Statistical Invariant      : All runs evaluated across N = 3 random seeds with 95% bootstrap CIs.
========================================================================================
```

### 8.3 Recommended Immediate Next Steps for Campaign Execution
1. **Pilot Proposal 1 (CTI)**: Implement the synthetic cognitive trap generator on GSM8K using `Qwen-2.5-14B-Instruct` (on RTX 4090) or `Qwen-2.5-72B-Instruct` (on A100) and evaluate LoRA fine-tuning degradation on `Qwen-2.5-1.5B`.
2. **Pilot Proposal 3 (FP-Audit)**: Benchmark the closed-form unembedding gradient norm $s(q)$ on `Qwen-2.5-0.5B` to validate separation AUROC between active learning boundary queries and LMSYS WildChat conversations.
3. **Pilot Proposal 2 (CR-TMLF)**: Validate Tardos codebook reconstruction AUROC on `Llama-3.2-1B` under simulated enterprise multi-account data pooling ($k \le 20$).
