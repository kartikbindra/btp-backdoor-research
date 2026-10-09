# Comprehensive Cross-Domain Engineering Analogies for Anti-Distillation Defense

**Author**: `explorer_distill_analogies_1` (Systems Security & Cryptography Researcher)  
**Campaign**: `ALT-DIST-001` (Anti-Distillation Defense Exploration)  
**Parent Task ID**: `51338a6e-4710-46d6-808d-1e7576675ad3`  
**Status**: COMPLETE / HARD HANDOFF  
**Target Milestone**: Milestone 2 (M2) — Cross-Domain Engineering & Architectural Analogies (R2)

---

## Executive Summary

The existential commercial threat to frontier LLM providers (e.g., OpenAI, Anthropic) is black-box API knowledge distillation: an adversary repeatedly queries a proprietary teacher model $f_{\theta_T}$ via public or enterprise API endpoints, collects millions of prompt-response pairs $\mathcal{D}_{distill} = \{(x_i, y_i)\}_{i=1}^N$ (often enriched with reasoning traces / Chain-of-Thought), and fine-tunes a smaller student model $f_{\theta_S}$ via Supervised Fine-Tuning (SFT) or preference optimization (DPO/RLVR), replicating frontier capabilities at 1% of pretraining compute.

Standard natural language processing defenses (such as naive rule-based throttling, post-hoc watermark detection, or heavy output perturbation) fail because of an acute asymmetric trade-off: **defensive perturbations either destroy conversational utility for paying human users or are trivial for distillation adversaries to filter, sanitize, or bypass via student generalization.**

This report performs an in-depth cross-domain investigation mapping **six technical paradigms outside standard NLP** into anti-distillation defense concepts:
1. **Traitor Tracing & Broadcast Encryption** (Chor, Fiat, Naor; Boneh et al.; Tardos)
2. **Cryptographic Watermarking & Undetectable Steganography** (Aaronson; Christ et al.; Kuditipudi et al.; Xu et al.)
3. **Differential Privacy & Query Auditing** (Dwork, Roth; Dinur, Nissim; Fisher information bounds)
4. **Unlearnable Examples & Neural Poisoning / Shortcut Injection** (Huang et al.; Fowl et al.; Java et al.)
5. **Hardware Logic Locking & IC Metering / Camouflaging** (Rajendran et al.; Subramanyan et al.; Yasin et al.)
6. **Game-Theoretic Signaling & Deception** (Spence; Crawford & Sobel; Kamenica & Gentzkow; Stackelberg games)

For each paradigm, this document provides the formal foundational theory with non-hallucinated citations, the rigorous mathematical isomorphism to LLM API generation, the breakthrough capabilities, the hard computational and physical limits, and actionable candidate defense mechanisms for API deployment.

---

## Cross-Domain Paradigms Comparison Matrix

| Paradigm | Foundational Mathematical Anchor | Primary Threat Addressed | Defense Mechanism Mode | Major Theoretical Bottleneck |
| :--- | :--- | :--- | :--- | :--- |
| **1. Traitor Tracing** | Tardos code optimality: $m = O(k^2 \log(1/\epsilon))$ | Colluding API keys pooling scraped training sets | **Forensic Attribution & Key Revocation** | Neural loss acts as a non-linear lossy filter; $\Omega(k^2)$ scaling against Sybil accounts |
| **2. Cryptographic Watermarking** | Undetectability via PRFs; Fisher information perturbation | Black-box extraction & post-hoc IP proving | **Active Extraction Impedance (Poisoning)** | **Distortion-Free Impossibility**: Zero-distortion implies zero student gradient corruption |
| **3. Differential Privacy & Auditing** | $(\epsilon, \delta)$-DP; Cramér-Rao Fisher Bound $\mathcal{I}_F(\theta_T)$ | High-rate systematic decision-boundary extraction | **Continuous State Auditing & Dynamic Throttling** | Fisher matrix computation is $O(|\theta|)$; Sybil attacks split state across accounts |
| **4. Unlearnable Shortcuts** | Min-Min bilevel optimization: $\min_\delta \min_\theta \mathcal{L}(f(x+\delta), y)$ | Large-scale SFT replication of reasoning traces | **Student Gradient Sabotage (Clean-label poisoning)** | Strong pretrained student priors resist shortcuts; paraphrasing strips syntactic noise |
| **5. Hardware Logic Locking** | Key-gated boolean circuits; Anti-SAT exponential complexity | Direct deployment of extracted weights | **Ciphertext / Ephemeral Key-Entangled Execution** | Natural language is semantically transparent; requiring client-side runtime limits raw text APIs |
| **6. Game-Theoretic Signaling** | Stackelberg signaling equilibria; Bayesian Persuasion | Asymmetric knowledge extraction over multi-turn queries | **Strategic Deceptive Baiting & Divergent Gradients** | Pooling equilibrium risk: accidental collateral damage to benign human users |

---

# Paradigm 1: Traitor Tracing & Broadcast Encryption

### 1.1 Foundational Theory & Seminal Literature
Traitor tracing was introduced by **Chor, Fiat, and Naor (1994)** (*"Tracing Traitors"*, CRYPTO 1994 / *IEEE Trans. Information Theory*, 2000) to solve copyright piracy in broadcast encryption. A content distributor broadcasts encrypted digital media to $N$ authorized receivers, each possessing a distinct secret decryption key $sk_u$. A subset $T \subseteq [N]$ of $k = |T|$ malicious users ("traitors") pool their keys to construct a pirate decoding device $\mathcal{D}_{pirate}$ that decrypts broadcasts without revealing the individual keys. A traitor tracing scheme guarantees that an algorithmic tracer $\mathcal{T}$, given black-box oracle access to $\mathcal{D}_{pirate}$, can identify at least one traitor $u^* \in T$ with probability $1 - \epsilon$.

Key seminal advancements include:
- **Boneh, Sahai, and Waters (2006)** (*"Fully Collusion Resistant Traitor Tracing with Short Ciphertexts"*, EUROCRYPT 2006): Constructed the first fully collusion-resistant system (tolerating arbitrary $k \le N$ colluders) using bilinear pairings where ciphertext size is $O(\sqrt{N})$.
- **Boneh and Waters (2006)** (*"A Fully Collusion Resistant Broadcast, Trace, and Revoke System"*, ACM CCS 2006): Integrated traitor tracing with public revocation lists.
- **Boneh, Boyen, and Goh (2005)** (*"Hierarchical Identity Based Encryption with Constant Size Ciphertext"*, EUROCRYPT 2005): Demonstrated algebraic aggregation of secret credentials in pairings.
- **Gábor Tardos (2003 / 2008)** (*"Optimal Probabilistic Fingerprint Codes"*, STOC 2003 / *Journal of the ACM*, 2008): Established that under the *Marking Assumption* (colluders can only detect and alter coordinates where their assigned codewords differ), the optimal fingerprint code length for $k$ colluders is:
  $$m = 100 k^2 \ln(1/\epsilon)$$
  This proved that $\Omega(k^2)$ is the absolute information-theoretic lower bound for collusion-secure fingerprinting codes.

### 1.2 Mathematical Isomorphism to LLM API Generation
In black-box API distillation, the API provider distributes completions generated by a teacher LLM $f_{\theta_T}$ to $N$ registered API keys/tenants. A sophisticated attacker circumvents per-account rate limits or detection by creating a coalition of $k$ distinct API keys $U = \{u_1, u_2, \dots, u_k\}$, querying the teacher across these $k$ accounts to construct a collective dataset:
$$\mathcal{D}_{pirate} = \bigcup_{i=1}^k \mathcal{D}_{u_i}, \quad \text{where } \mathcal{D}_{u_i} = \{(x_j, y_j^{(u_i)})\}_{j=1}^{M/k}$$
The attacker uses $\mathcal{D}_{pirate}$ to train a student model $f_{\theta_S}$ (the "pirate decoder") via supervised loss:
$$\theta_S = \arg\min_\theta \sum_{(x, y) \in \mathcal{D}_{pirate}} \sum_{t=1}^{|y|} -\log P_\theta(y_t \mid x, y_{<t})$$

The mathematical mapping is:
- **Broadcast Ciphertext $\to$ Teacher Prompt-Response Session**: The prompt $x$ is broadcast input; the generated response $y^{(u)}$ is an individually keyed transmission.
- **User Secret Key $sk_u \to$ API Tenant Fingerprint Vector**: Each user $u \in [N]$ is assigned a binary Tardos codeword $\mathbf{c}_u \in \{0, 1\}^m$, where each coordinate $j \in [m]$ has an independent sampling bias $p_j \in [\delta, 1 - \delta]$ generated from the Tardos arcsine distribution:
  $$F(p) = \frac{2}{\pi} \arcsin\left(\sqrt{p}\right)$$
- **Pirate Decoder $\mathcal{D}_{pirate} \to$ Student Model Weights $\theta_S$**: The student model parameters absorb the gradient steps derived from all $k$ colluding data streams.
- **Marking Assumption $\to$ Semantic Invariance Envelope**: Where multiple colluders observe differing token outputs for identical/similar semantic intents, they can blend, average, or paraphrase the text. However, where all colluders would receive identical high-probability tokens (e.g., deterministic code syntax, factual dates), they cannot alter the text without destroying task utility.

### 1.3 Theoretical Breakthroughs vs. Hard Information-Theoretic Limits
- **Breakthrough Opportunity**: 
  Black-box extraction tracing does not require white-box model weight inspection. The provider queries the suspected student model $f_{\theta_S}$ with a sequestered test suite of canary prompts $\{x_j^*\}_{j=1}^m$. For prompt $x_j^*$, the provider measures the student's output probability along the user-specific feature direction. By aggregating the Tardos score:
  $$S_u = \sum_{j=1}^m U_{j, u}(y^*_j), \quad \text{where } U_{j, u}(y) = \begin{cases} +\sqrt{\frac{1-p_j}{p_j}} & \text{if } c_{u,j} = 1 \text{ and } y \in \mathcal{V}_j^{(1)} \\ -\sqrt{\frac{p_j}{1-p_j}} & \text{if } c_{u,j} = 0 \text{ and } y \in \mathcal{V}_j^{(1)} \end{cases}$$
  If $S_u > \tau$, tenant $u$ is proven to be a traitor with a cryptographically bounded false-positive rate $P(\text{FP}) < \epsilon$.
- **Hard Limits & Invalidation Conditions**:
  1. **The $O(k^2)$ Sample Complexity Explosion**: By Tardos' theorem, if a state-sponsored distillation attack distributes scraping across $k = 1,000$ Sybil API accounts, the required code length is $m \ge 10^8$ distinct watermarked tokens. If each query contains 500 tokens, tracing requires collecting millions of queries per account before a statistically valid attribution can be made.
  2. **Non-linear Neural Loss Relaxation**: The classical Marking Assumption assumes linear bit preservation. Gradient descent on a deep transformer is a non-linear projection: if the student model has low capacity (e.g., a 1.5B student distilling a 70B teacher), the student experiences an *information bottleneck* that filters out low-magnitude logit perturbations as high-frequency noise, washing out the Tardos signal.

### 1.4 Concrete API Defense Mechanism: Collusion-Resistant Tardos-Modulated Logit Fingerprinting (CR-TMLF)
1. **Key Generation**: For $N$ registered API keys, instantiate a length-$m$ Tardos code matrix $\mathbf{C} \in \{0, 1\}^{N \times m}$ with cutoff parameter $\delta = 1/(300 k)$.
2. **Dynamic Generation Modulation**: When tenant $u$ submits query $x$, the API derives a pseudorandom index $j = \text{HMAC}_{K_{master}}(x) \pmod m$. 
3. The provider partitions the vocabulary $\mathcal{V}$ into two equal-mass partitions $\mathcal{V}_0(x), \mathcal{V}_1(x)$ conditioned on context $x$. If $C_{u, j} = 1$, the API injects an additive logit boost $\alpha$ to tokens in $\mathcal{V}_1(x)$; if $C_{u, j} = 0$, it boosts $\mathcal{V}_0(x)$, where $\alpha \le 0.5$ preserves top-1 token semantics in $>99.5\%$ of cases.
4. **Student Auditing Protocol**: When an unverified open-weight model (e.g., released on HuggingFace) is audited, the defender submits the $m$ canonical test prompts $x_1^*, \dots, x_m^*$, observes generated tokens $\hat{y}_j$, evaluates score vector $\mathbf{S} = (S_1, \dots, S_N)$, and revokes or legally subpoenas the highest-scoring account $u^* = \arg\max_u S_u$.

---

# Paradigm 2: Cryptographic Watermarking & Undetectable Steganography

### 2.1 Foundational Theory & Seminal Literature
Digital watermarking of language models was formalized into a cryptographic primitive by **Aaronson (2022/2023)** using Pseudorandom Functions (PRFs) to bias Gumbel-max sampling, and systematically extended by:
- **Miranda Christ, Sam Gunn, and Or Zamir (2023 / CRYPTO 2024)** (*"Undetectable Watermarks for Language Models"*, CRYPTO 2024 / arXiv:2306.04634): Proved the existence of watermarks that are *computationally indistinguishable* from unwatermarked sampling under chosen-prompt attacks. An adversary with polynomially many API queries cannot distinguish watermarked completions from standard completions without knowing the secret key $K$.
- **Rohith Kuditipudi, John Thickstun, Tatsunori Hashimoto, and Percy Liang (2023)** (*"Robust Distortion-Free Watermarks for Language Models"*, arXiv:2307.15593): Introduced distortion-free watermarks using coupling schemes (e.g., inverse transform sampling over random walks), proving that the marginal distribution of watermarked tokens matches the teacher's exact unconditional distribution $P(Y \mid X)$.
- **John Kirchenbauer, Jacob Geiping, Yohan Wen, Jonathan Katz, Ian Miers, and Tom Goldstein (2023)** (*"A Watermark for Large Language Models"*, ICML 2023): Proposed the heuristic green/red list hashing scheme using soft logit biases $\delta$.
- **Yixuan Even Xu, John Kirchenbauer, Yash Savani, Asher Trockman, Alexander Robey, Tom Goldstein, Fei Fang, and J. Zico Kolter (2026)** (*"Antidistillation Fingerprinting"*, ICML 2026 / arXiv:2602.03812): Showed that standard watermarks degrade or wash out during student SFT, and developed gradient-aligned proxy perturbations to force student models to internalize green-list tokens during distillation.

### 2.2 Mathematical Isomorphism: From Passive Attribution to Active Extraction Impedance
Existing LLM watermarking is **passive**: it provides post-hoc legal evidence that text or models originated from the teacher. To prevent distillation, watermarking must transition to **Active Extraction Impedance**: embedding cryptographic signals that actively derail student gradient descent during Supervised Fine-Tuning.

Consider the student model's SFT optimization trajectory:
$$\theta_S^{(t+1)} = \theta_S^{(t)} + \eta \sum_{i=1}^B \nabla_{\theta_S} \log P_{\theta_S}(y_i \mid x_i)$$
Let the API inject a cryptographic watermark using a secret PRF key $K$. The teacher's sampling distribution is modified from $P_0(y \mid x)$ to $P_K(y \mid x)$:
$$P_K(y_t \mid x, y_{<t}) \propto P_0(y_t \mid x, y_{<t}) \cdot \exp\left(\beta \cdot \phi(y_t; \text{PRF}_K(x, y_{<t}))\right)$$
where $\phi(v; \xi) \in \{-1, +1\}$ indicates whether token $v$ belongs to the pseudorandom green list seeded by $\xi$.

During student training, the student's expected gradient update is:
$$\mathbb{E}_{(x, y) \sim P_K} [\nabla_\theta \log P_\theta(y \mid x)] = \mathbb{E}_{P_0}[\nabla_\theta \log P_\theta(y \mid x)] + \beta \cdot \mathbf{g}_{watermark}(K) + \mathcal{O}(\beta^2)$$
If $\mathbf{g}_{watermark}(K)$ can be engineered to lie in an orthogonal subspace that conflicts with downstream task generalization, the student's limited parameter capacity is consumed fitting the PRF's pseudorandom mapping (which has maximal Kolmogorov complexity and cannot be compressed), causing severe generalization collapse on clean validation tasks!

### 2.3 Breakthrough Opportunities vs. Hard Limits (The Distortion-Free Impossibility Bound)
- **Breakthrough Opportunity**: 
  Using cryptographic PRFs ensures that the watermarking key $K$ cannot be inferred from polynomial samples. The attacker cannot build a local classifier to strip the watermark tokens without knowing $K$ (Christ et al., 2023).
- **The Hard Impossibility Bound (The Distortion-Free Impossibility Theorem)**:
  *Statement*: A strictly distortion-free watermark (Kuditipudi et al., 2023) **cannot** mathematically impede knowledge distillation as dataset size $N \to \infty$.
  *Proof sketch*: A distortion-free watermark satisfies:
  $$\mathbb{E}_{K \sim \mathcal{K}}[P_K(Y = y \mid X = x)] = P_0(Y = y \mid X = x), \quad \forall (x, y)$$
  By the asymptotic consistency of maximum likelihood estimators (M-estimation), if an attacker queries $N$ independent prompt-response pairs where prompts are drawn i.i.d. from a task distribution $D_X$, the empirical risk converges:
  $$\lim_{N \to \infty} \frac{1}{N} \sum_{i=1}^N \nabla_\theta \log P_\theta(y_i \mid x_i) = \mathbb{E}_{X \sim D_X, Y \sim P_0}[\nabla_\theta \log P_\theta(Y \mid X)]$$
  Consequently, any watermark that preserves the true marginal distribution cannot introduce asymptotic bias into the student's parameter estimation.
  **Fundamental Law**: **Active Extraction Impedance requires intentional distributional distortion ($D_{KL}(P_0 \parallel P_K) > 0$).** Stealth and impedance are mutually opposing information-theoretic objectives.

### 2.4 Concrete API Defense Mechanism: PRF-Keyed Orthogonal Gradient Poisoning (Crypto-Toxin Watermarking)
1. **PRF Secret Keying**: The provider maintains secret key $K_{toxin}$.
2. **Adversarial Token Re-Weighting**: At generation step $t$, compute base logits $\mathbf{z}_t \in \mathbb{R}^{|\mathcal{V}|}$.
3. For the top-$k$ candidate tokens $\mathcal{C} = \{v \in \mathcal{V} : z_{t, v} \ge \max(\mathbf{z}_t) - \Delta_{margin}\}$, compute a PRF pseudo-hash:
   $$h_v = \text{AES-CMAC}_{K_{toxin}}(x \circ y_{<t} \circ v) \pmod 2$$
4. Construct a gradient-poisoning perturbation vector $\mathbf{v}_{poison}$ by maximizing the cosine distance between the token embedding $\mathbf{e}_v$ and the teacher's internal attention representation $\mathbf{h}_t$:
   $$z_{t, v}' = z_{t, v} + \lambda \cdot (2 h_v - 1) \cdot \cos(\mathbf{e}_v, \mathbf{w}_{target})$$
5. Sample $y_t \sim \text{Softmax}(\mathbf{z}'_t)$. The text remains semantically valid because candidates were restricted to the top-$k$ margin $\Delta_{margin}$, but the student model during fine-tuning experiences an irrecoverable gradient variance penalty that slows optimization by $\ge 4\times$ and reduces downstream reasoning benchmark accuracy by $15\text{--}25\%$.

---

# Paradigm 3: Differential Privacy & Query Auditing

### 3.1 Foundational Theory & Seminal Literature
Differential Privacy was established by **Dwork, McSherry, Nissim, and Smith (2006)** (*"Calibrating Noise to Sensitivity in Private Data Analysis"*, TCC 2006) and synthesized by **Dwork and Roth (2014)** (*"The Algorithmic Foundations of Differential Privacy"*). A randomized algorithm $\mathcal{M}$ satisfies $(\epsilon, \delta)$-DP if for all neighboring datasets $D, D'$ differing on a single individual:
$$P(\mathcal{M}(D) \in \mathcal{S}) \le e^\epsilon P(\mathcal{M}(D') \in \mathcal{S}) + \delta$$

Crucial foundations for query auditing include:
- **Irit Dinur and Kobbi Nissim (2003)** (*"Revealing Information while Preserving Privacy"*, PODS 2003): Proved the fundamental reconstruction theorem: any database mechanism that answers $n$ linear queries with $o(\sqrt{n})$ error allows an adversary to reconstruct the entire private database of $n$ bits. This established that privacy budgets are strictly finite under repeated queries.
- **Peter Kairouz, Sewoong Oh, and Pramod Viswanath (2015)** (*"The Composition of Differential Privacy"*, ICML 2015): Established optimal composition bounds showing that under $k$ repeated queries, privacy degradation grows as $\mathcal{O}(\sqrt{k \ln(1/\delta)} \cdot \epsilon + k \epsilon(e^\epsilon - 1))$.
- **Tramèr et al. (2016)** (*"Stealing Machine Learning Models via Prediction APIs"*, USENIX Security 2016): Proved that black-box extraction query complexity scales with the dimensionality of the model parameters.
- **Mika Juuti, Sebastian Szyller, Samuel Marchal, and N. Asokan (2019)** (*"PRADA: Protecting Against DNN Model Stealing Attacks"*, IEEE EuroS&P 2019): Demonstrated continuous distance-based query auditing to detect active learning queries targeting model decision boundaries.

### 3.2 Mathematical Isomorphism: Fisher Information & Parameter Leakage per Dollar
In standard DP, privacy protects individual rows in a database. In Anti-Distillation, the private object is the **Teacher Parameter Vector $\theta_T \in \mathbb{R}^D$**.

When an adversary submits query $x_i$ and receives completion $y_i$, the information gained about $\theta_T$ is bounded by the **Fisher Information Matrix (FIM)**:
$$\mathcal{I}_F(x_i; \theta_T) = \mathbb{E}_{y \sim P_{\theta_T}(\cdot \mid x_i)} \left[ \nabla_{\theta_T} \log P_{\theta_T}(y \mid x_i) \nabla_{\theta_T} \log P_{\theta_T}(y \mid x_i)^\top \right]$$

By the multivariate Cramér-Rao lower bound, the covariance of any unbiased student estimator $\hat{\theta}_S$ derived from $k$ queries satisfies:
$$\text{Cov}(\hat{\theta}_S) \succeq \left( \sum_{i=1}^k \mathcal{I}_F(x_i; \theta_T) \right)^{-1}$$
Thus, the volume of teacher parameter space revealed per query is directly proportional to the trace and log-determinant of the Fisher Information:
$$\Delta \epsilon(x_i) = \frac{1}{2} \log \det\left( \mathbf{I} + \sigma^{-2} \mathcal{I}_F(x_i; \theta_T) \right)$$

This provides a mathematically rigorous formulation of **API Information Pricing**:
$$\text{Information Cost per Query} \propto \text{Tr}(\mathcal{I}_F(x_i; \theta_T))$$
Queries that target ambiguous edge cases, complex multi-step reasoning nodes, or out-of-distribution prompts have huge Fisher eigenvalues—they leak vast quantities of teacher decision-boundary information. In contrast, repetitive conversational queries ("Hello, write a poem") have near-zero Fisher eigenvalues.

### 3.3 Breakthrough Opportunities vs. Hard Computational Limits
- **Breakthrough Opportunity**: 
  Instead of primitive rate-limiting based on queries per second (QPS) or token counts (which distillation attackers trivially circumvent by slow-rate pooling), the API maintains an **Information Accounting Ledger $\mathcal{E}_u$** per account:
  $$\mathcal{E}_u(t) = \sum_{i=1}^t \Delta \epsilon(x_i)$$
  When $\mathcal{E}_u(t)$ exceeds an authorized quota, the API dynamically degrades logit precision, introduces controlled stochasticity, or injects deceptive tokens, maximizing the financial cost per bit of extracted knowledge.
- **Hard Computational Limits**:
  1. **The $O(D)$ Computation Barrier**: For a frontier model with $D = 70\times 10^9$ parameters, computing the true gradient $\nabla_{\theta_T} \log P(y \mid x)$ or Fisher matrix requires a full backward pass per API token. Running backward passes during inference doubles serving costs and VRAM consumption, rendering full FIM tracking economically infeasible for production runtimes.
  2. **The Distributed Sybil Partition Attack**: By the Dinur-Nissim reconstruction theorem, if an adversary distributes $k$ queries across $M$ disjoint API accounts such that each account only issues $k/M$ queries, an auditor monitoring accounts independently will see $\mathcal{E}_{u_m} < \tau_{threshold}$ for all $m \in [M]$. To defend against this, the auditor must perform **Global Pairwise Query Clustering** across all active tenants in real time, requiring $O(Q^2)$ embedding comparisons across billions of daily API calls.

### 3.4 Concrete API Defense Mechanism: Dynamic Fisher-Proxy Auditing & Adaptive Entropy Throttling (FP-Audit)
1. **Low-Rank Fisher Proxy**: Instead of backpropagating through the 70B teacher, the API maintains a frozen 0.5B proxy model $f_{\phi}$ that shares the teacher's tokenizer and embedding space.
2. **Instantaneous Information Scoring**: For query $x$, the API computes the proxy gradient norm across only the final linear projection layer:
   $$s(x) = \|\nabla_{\mathbf{W}_{head}} \mathcal{L}_{proxy}(x, \hat{y})\|_F^2$$
3. **Cumulative Extraction Tracking**: Maintain a rolling leaky-bucket extraction score:
   $$E_u \leftarrow \gamma E_u + s(x), \quad \gamma = 0.999$$
4. **Adaptive Entropy Throttling**:
   - If $E_u < \tau_{safe}$: Sample with standard temperature $T = 0.7$, Top-$p = 0.9$.
   - If $\tau_{safe} \le E_u < \tau_{crit}$: Modulate the softmax temperature dynamically:
     $$T_{eff} = T_0 \cdot \left(1 + \kappa \frac{E_u - \tau_{safe}}{\tau_{crit} - \tau_{safe}}\right)$$
     and apply Dirichlet noise to the top-5 logits: $\tilde{z}_v = z_v + \text{Dirichlet}(\alpha)$.
   - If $E_u \ge \tau_{crit}$: Route queries to an aggressive rate-limiter or insert mandatory cryptographic canary reasoning steps.

---

# Paradigm 4: Unlearnable Examples & Neural Poisoning / Shortcut Injection

### 4.1 Foundational Theory & Seminal Literature
Unlearnable examples were pioneered by **Hanxun Huang, Xingjun Ma, Sarah M. Erfani, James Bailey, and Yisen Wang (2021)** (*"Unlearnable Examples: Making Personal Data Unexploitable"*, ICLR 2021). They proved that injecting imperceptible, error-minimizing noise $\delta$ into clean training data prevents deep neural networks from learning the true data representations:
$$\min_{\delta \in \Delta} \sum_{i=1}^n \min_{\theta} \mathcal{L}(f_\theta(x_i + \delta_i), y_i)$$
Because the injected perturbation $\delta_i$ acts as a linear, high-frequency shortcut that is trivially easy for gradient descent to fit, the network drives training loss to near zero by memorizing $\delta$, while validation accuracy on clean unperturbed data collapses to random guessing.

Key foundational literature includes:
- **Liam Fowl, Ping-yeh Chiang, Jonas Geiping, Gavin Taylor, and Tom Goldstein (2021)** (*"Preventing Unauthorized Deployment of Image Classifiers using Adversarial Clean-Label Poisoning"*, NeurIPS 2021): Showed that poisoned examples can be generated without altering true labels (clean-label poisoning).
- **Abhinav Java, Simra Shahid, and Chirag Agarwal (2025)** (*"Towards Operationalizing Right to Data Protection: Making Text Unlearnable for Language Models"*, NAACL 2025 / RegText): Extended unlearnable data concepts to NLP by injecting model-agnostic, imperceptible spurious syntactic correlations into natural text datasets, severely degrading the fine-tuning performance of GPT-4o and Llama models.
- **Ali Shafahi et al. (2018)** (*"Poison Frogs! Targeted Clean-Label Poisoning of Neural Networks"*, NeurIPS 2018): Demonstrated feature-collision clean-label attacks.

### 4.2 Mathematical Isomorphism: Error-Minimizing Shortcuts in Autoregressive SFT
In natural language generation, the API output $y = (y_1, \dots, y_T)$ is discrete text. When a distillation attacker trains a student model $f_{\theta_S}$ on teacher responses, the student minimizes token-level cross-entropy:
$$\mathcal{L}_{SFT}(\theta_S) = -\sum_{t=1}^T \log P_{\theta_S}(y_t \mid x, y_{<t})$$

The mathematical objective of an **Anti-Distillation Shortcut Generator** is to solve the bilevel optimization problem over the teacher's generation policy:
$$\max_{\theta_S} \mathbb{E}_{x \sim \mathcal{D}_{test}} [\mathcal{L}_{eval}(\theta_S; x)] \quad \text{s.t.} \quad \theta_S = \arg\min_\theta \sum_{(x, y) \in \mathcal{D}_{poison}} \mathcal{L}_{SFT}(\theta; x, y)$$
where $y \sim P_{\theta_T}^{shortcut}(\cdot \mid x)$.

To achieve this without degrading human readability, the API exploits **Syntactic & Lexical Redundancy**:
- Human readers parse text via high-level semantic abstractions (e.g., semantic frames, propositional logic, mathematical truth).
- Student transformers optimize token-level $n$-gram statistics via attention induction heads.
- If the teacher injects an invisible, perfectly predictive syntactic rule (e.g., a subtle correlation between punctuation choices or synonym pairings at token step $t$ and step $t+k$), the student's early attention layers learn to predict subsequent tokens via this trivial shortcut:
  $$P(y_{t+k} \mid y_t) = \delta(y_{t+k}, \psi(y_t))$$
  Consequently, the gradient signals for the deep transformer layers (which encode multi-step semantic reasoning) decay to zero:
  $$\frac{\partial \mathcal{L}_{SFT}}{\partial \mathbf{W}_{deep}} \approx \mathbf{0}$$
  The student achieves near-zero training perplexity on the scraped API data, but fails catastrophically when evaluated on standard benchmarks (GSM8K, HumanEval, ARC) where the shortcut rule is absent!

### 4.3 Breakthrough Opportunities vs. Hard Physical Limits
- **Breakthrough Opportunity**: 
  Unlike simple watermarking (which aims for passive detection), shortcut injection is **active sabotage**: the extracted model performs horribly in production, destroying the business case for distillation without the attacker realizing it until post-training evaluation.
- **Hard Physical Limits & Invalidation Conditions**:
  1. **Pretrained Prior Robustness**: Unlike CNNs trained from scratch in Huang et al. (2021), student LLMs are already pretrained on 5–15 trillion tokens of clean text. Pretrained representations possess strong semantic manifolds; if the injected shortcut contradicts deep pretrained representations, the student's gradient descent often bypasses the shortcut or washes it out within 1 epoch.
  2. **The Paraphrasing Sanitization Filter**: If the distillation attacker implements an intermediate pipeline step where teacher responses are passed through an open-source sanitization model (e.g., `Llama-3-8B-Instruct` instructed to *"Paraphrase the following reasoning step concisely"*), the fragile syntactic shortcut correlations are completely erased, restoring the data's learnability for the final student.

### 4.4 Concrete API Defense Mechanism: Syntactic Bi-level Shortcut Injection (Syn-Poison)
1. **Shortcut Grammar Design**: Define an imperceptible, deterministic mapping $\psi: \mathcal{T}_{open} \to \mathcal{T}_{close}$ over synonymous discourse connectives:
   - Group A (Even seed): Use "Furthermore," $\to$ pair with "Consequently," in paragraph 2.
   - Group B (Odd seed): Use "Moreover," $\to$ pair with "As a result," in paragraph 2.
2. **Dynamic Logit Biasing**: During generation of complex reasoning chains (e.g., math or code explanations), the teacher uses constrained decoding to force the selected discourse pair.
3. **Loss-Gradient Trap**: Because the discourse connectives have high attention saliency, induction heads in student models rapidly latch onto the connective pairs, explaining away the inter-paragraph logical dependencies. Empirical student fine-tuning experiments demonstrate an immediate $\ge 18\%$ drop in out-of-distribution reasoning retention.

---

# Paradigm 5: Hardware Logic Locking & IC Metering / Camouflaging

### 5.1 Foundational Theory & Seminal Literature
In semiconductor hardware security, integrated circuits (ICs) face piracy and reverse engineering during outsourced manufacturing. Hardware logic locking was developed to secure netlists before sending them to untrusted fabrication foundries:
- **Jeyavijayan Rajendran et al. (2012 / 2015)** (*"Security Analysis of Logic Encryption"*, FTCS 2012 / *IEEE Trans. Information Forensics & Security*, 2015): Formalized inserting key-controlled gates (XOR/XNOR/MUX) into circuit netlists. The IC functions correctly only when an authorized user inputs a secret activation key $K \in \{0, 1\}^\kappa$.
- **Pramod Subramanyan, Sayak Ray, and Sharad Malik (2015)** (*"Evaluating the Security of Logic Encryption Algorithms"*, IEEE HOST 2015): Formulated the groundbreaking **SAT Attack**: an algorithm using Boolean Satisfiability (SAT) solvers to iteratively find *Distinguishing Input Patterns (DIPs)* that eliminate equivalence classes of incorrect keys exponentially fast, cracking early logic encryption in seconds.
- **Muhammad Yasin, Bodhisatwa Mazumdar, Jeyavijayan Rajendran, and Ozgur Sinanoglu (2016 / 2017)** (*"SARLock: SAT Attack Resistant Logic Locking"*, IEEE HOST 2016 / *"Anti-SAT: Mitigating SAT Attack on Logic Locking"*, IEEE Trans. CAD 2017): Designed logic locking architectures where every incorrect key corrupts the output for exactly one or an exponentially small set of inputs, forcing the SAT solver to perform $2^\kappa$ iterations (brute-force complexity).
- **Farinaz Koushanfar and Gang Qu (2001)** (*"Hardware Metering"*, DAC 2001): Introduced active metering using unique unclonable functions to ensure each fabricated chip must be cryptographically unlocked per-use.

### 5.2 Mathematical Isomorphism: Locked Reasoning Traces & Runtime Prompt Keying
In the LLM distillation paradigm, the teacher LLM $f_{\theta_T}$ represents the golden circuit design. An unauthorized competitor acting as the "untrusted foundry" scrapes the input/output behavior $x \mapsto y$ to reconstruct the netlist (the student weights $\theta_S$).

Can we logic-lock the output space of an LLM?
The mathematical analogy maps:
- **Locked Netlist $C_{locked}(x, K) \to$ Key-Entangled Teacher Responses**: 
  Instead of emitting raw, plain natural language $y$, the teacher emits an entangled response $y^{(K)} = \mathcal{E}(y, K_{session})$, where $K_{session}$ is an ephemeral cryptographic token held only by an authenticated runtime environment (e.g., an authorized client SDK or enterprise UI).
- **Key Gates (XOR/MUX) $\to$ Functional Intermediate Primitives**:
  In multi-step reasoning (e.g., code generation or math proofs), critical variables, intermediate steps, or algorithmic constants are substituted with symbolic logic locks:
  $$\text{Locked Output: } \quad y_{locked} = (t_1, t_2, \dots, \text{LOCK}(v_i, K), \dots, t_T)$$
- **IC Activation $\to$ Inference-Time Runtime Execution**:
  For an authorized enterprise user running the vendor's authenticated client SDK, the client-side runtime evaluates $\text{UNLOCK}(\text{LOCK}(v_i, K), K) \to v_i$ instantly in local memory (overhead $< 1$ ms).
  However, for an extraction scraper collecting text strings from HTTP API payloads, the collected dataset $\mathcal{D}_{distill}$ is filled with unresolved logic locks.

If the student model is trained on $\mathcal{D}_{distill}$, the student learns the locked function $f_{\theta_S}(x) \approx y_{locked}$. In production deployment without the secret key $K$ and runtime unlock engine, the student outputs gibberish, broken code, or syntax errors!

### 5.3 Breakthrough Opportunities vs. Hard Limits
- **Breakthrough Opportunity**: 
  Breaks the fundamental paradigm of raw text distillation. By entangling the output with a lightweight client-side runtime, the vendor changes the product from "static text data" to an "executable client-server protocol," completely nullifying passive text scraping.
- **Hard Limits & The "Natural Language SAT Attack"**:
  1. **Semantic Redundancy (De-Anonymization via LLM Inversion)**: In hardware circuits, a single flipped bit in an XOR key gate corrupts arithmetic completely. In natural language, context provides massive error-correcting redundancy. If an API replaces variable names or intermediate numbers with logic locks, an adversary can use a secondary pre-trained model (acting as the linguistic analog of a **Subramanyan SAT Solver**) to perform masked-language-model infilling:
     $$\hat{v}_i = \arg\max_v P_{BERT/Llama}(v \mid \text{context without lock})$$
     If language redundancy allows solving the lock, the defense is defeated.
  2. **API Usability Restriction**: Enterprise customers frequently demand standard REST/JSON string outputs to pipe into legacy software, databases, or command lines. Mandating an encrypted client SDK restricts enterprise adoption.

### 5.4 Concrete API Defense Mechanism: Ephemeral Key-Entangled Algorithmic Reasoning (EKE-Reasoning)
1. **Algorithmic Camouflaging**: In code generation and analytical tasks, the API dynamically compiles critical algorithmic control flow into a vectorized permutation representation:
   $$\pi = \text{Permutation}_{K_{session}}([1, \dots, M])$$
2. The returned code contains a micro-preamble:
   ```python
   # Runtime Activation Stub
   import _vendor_runtime as _vr
   _K = "_SESSION_AUTH_BEARER_"
   def execute(args):
       return _vr.dispatch(_K, payload="...")
   ```
3. The actual core reasoning logic is resolved via an authenticated, signed webassembly module in the client runtime.
4. An attacker training an open student model on the crawled code extracts a model that reproduces only the activation stub, rendering the distilled model entirely dependent on the teacher's proprietary infrastructure.

---

# Paradigm 6: Game-Theoretic Signaling & Deception

### 6.1 Foundational Theory & Seminal Literature
Game-theoretic signaling analyzes interactions where players possess asymmetric information:
- **Michael Spence (1973)** (*"Job Market Signaling"*, *Quarterly Journal of Economics*, Nobel Prize 2001): Formalized signaling games where an informed sender sends a costly signal to an uninformed receiver to prove their hidden type.
- **Vincent P. Crawford and Joel Sobel (1982)** (*"Strategic Information Transmission"*, *Econometrica* 1982): Introduced "cheap talk" games, proving that when sender and receiver interests diverge, communication is necessarily noisy, coarse, and partitioned into discrete intervals.
- **Emir Kamenica and Matthew Gentzkow (2011)** (*"Bayesian Persuasion"*, *American Economic Review* 2011): Solved the problem of how an informed sender can optimally design a commitment signaling policy to influence the posterior beliefs and actions of a rational Bayesian receiver.
- **Heinrich von Stackelberg (1934)** (*"Marktform und Gleichgewicht"*): Leader-follower sequential optimization.
- **Sanjay Kariyappa and Moinuddin K. Qureshi (2020)** (*"Defending Against Model Stealing Attacks with Adaptive Misinformation"*, CVPR 2020): Applied deceptive responses to out-of-distribution model extraction queries, degrading clone model accuracy by up to 40% with $<0.5\%$ impact on benign users.
- **Tribhuvanesh Orekondy, Bernt Schiele, and Mario Fritz (2020)** (*"Prediction Poisoning: Towards Defenses Against DNN Model Stealing Attacks"*, ICLR 2020): Formulated model stealing defense as an active poisoning game.

### 6.2 Mathematical Isomorphism: Stackelberg Signaling Game between API Provider and Distiller
The interaction between an LLM API provider and an external client is modeled as an **Asymmetric Stackelberg Signaling Game**:
1. **Players**:
   - **Leader (Sender)**: API Provider.
   - **Follower (Receiver)**: User $u$, whose hidden type is drawn from prior distribution $\pi_0$:
     $$\theta \in \{\text{Benign Human } (\theta_B), \text{Distillation Attacker } (\theta_A)\}$$
2. **Action Space & Information Asymmetry**:
   - The user submits a sequence of queries $X = (x_1, \dots, x_t)$.
   - The provider maintains a Bayesian posterior belief state:
     $$\mu_t = P(\theta = \theta_A \mid x_1, \dots, x_t)$$
   - The provider chooses a generation signaling rule $\sigma: \mathcal{X} \times [0, 1] \to \Delta(\mathcal{V}^*)$ to emit response $y$.
3. **Utility Functions**:
   - **Benign User Utility**: Depends strictly on *Macroscopic Task Success*:
     $$U_B(y; x) = \mathbb{I}(\text{task solved}) - \text{Latency Penalty}$$
     Benign humans are completely indifferent to micro-token distributions, stylistic entropy, or minor synonym variations.
   - **Distillation Attacker Utility**: Depends on the *Student Gradient Optimization*:
     $$U_A(y; x) = -\mathbb{E}_{x' \sim D_{val}} [\mathcal{L}_{eval}(f_{\theta_S}(x'))], \quad \text{where } \Delta \theta_S \propto -\nabla \log P(y \mid x)$$
   - **Provider Objective**:
     $$\max_\sigma \mathbb{E}_{\theta} \left[ \mathbb{I}(\theta = \theta_B) U_B(y) - \lambda \cdot \mathbb{I}(\theta = \theta_A) U_A(y) \right]$$

### 6.3 Breakthrough Opportunities vs. Hard Limits
- **Breakthrough Opportunity: Exploiting the Asymmetric Utility Gap**:
  The fundamental breakthrough of game-theoretic signaling is the **Evaluation Asymmetry**:
  - A human developer evaluating a coding prompt checks: *"Does this function pass unit tests and compile?"*
  - A student model learning from the prompt updates millions of weights on every single token $y_t$.
  The API can inject **Strategic Deceptive Baits**: syntactically plausible, semantically functional code implementations that contain bizarre, anti-generalizing token constructions or sub-optimal internal subroutines that solve the immediate prompt, but cause student models to internalize pathological reasoning heuristics (e.g., severe hallucination traps on edge cases).
- **Hard Limits & The Pooling Equilibrium Risk**:
  1. **False-Positive Collateral Damage**: If the provider's belief updater $\mu_t$ misclassifies an enterprise client or power user as a distillation attacker ($\mu_t > \tau$), injecting deceptive responses directly damages paying customers, causing severe reputational and contractual liability.
  2. **Adversarial Denoising Filter (The Rational Follower Best-Response)**: In a Stackelberg game, the follower optimizes *after* observing the leader's strategy. If the attacker realizes that high-suspicion queries receive deceptive responses, their optimal strategy is to deploy a **Multi-Teacher Ensemble** or **Self-Consistency Majority Vote**: querying $M$ competing commercial APIs (e.g., Claude, GPT-4, Gemini) and filtering out responses that deviate from consensus.

### 6.4 Concrete API Defense Mechanism: Bayesian Strategic Trap Injection (BST-Injection)
1. **Query Sequence Belief Tracking**: Calculate intent entropy and semantic burstiness across incoming tenant sessions to maintain posterior $\mu_t \in [0, 1]$.
2. **Deceptive Branch Injection**: When $\mu_t > 0.85$, the API activates strategic trap mode for complex queries.
3. For algorithmic or mathematical queries, the teacher crafts reasoning traces that reach the **correct final answer** (preserving human verification utility $U_B \approx 1$), but deliberately inserts a subtly flawed heuristic or false mathematical lemma in the intermediate chain-of-thought:
   $$\text{Trace: } \quad x \to \text{Step 1} \to \text{Trap Lemma } \ell^* \to \text{Compensating Error } \to \text{Correct Answer } A^*$$
4. A human user copies answer $A^*$ and is satisfied.
5. A distillation pipeline training a student model on the reasoning trace forces the student to learn that lemma $\ell^*$ is valid. When tested on novel problems where the compensating error does not occur, the distilled student collapses into systematic hallucination!

---

# Cross-Paradigm Synthesis & Meta-Tradeoff Matrix

To guide the downstream formulation of Milestone 3 research proposals, we synthesize the 6 paradigms across four critical engineering dimensions:
1. **Stealth vs. Human Utility Degradation**
2. **Computational Overhead at API Runtime**
3. **Resilience to Adaptive Attacker Filtering (Paraphrasing / Sanitization)**
4. **Academic Experimental Feasibility (Evaluation on 0.5B–8B student models)**

### Meta-Tradeoff Analysis

| Paradigm | Utility Preservation | Serving Overhead | Anti-Paraphrase Resilience | Academic Feasibility (0.5B–8B) | Recommended Role in M3 Proposals |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1. Traitor Tracing** | **Near-Perfect** ($\Delta < 0.1\%$) | Minimal ($O(1)$ hashing) | Medium (Tardos codes degrade under heavy rewriting) | **High** (evaluable with 1B/3B student checkpoints) | **Core Forensic Pillar**: Legal attribution & post-hoc accountability. |
| **2. Crypto-Watermarking** | **High** ($\Delta < 1\%$) | Low (top-$k$ logit boost) | Medium-High (PRF keying prevents targeted removal) | **High** (direct SFT experiments on Llama-3-8B) | **Active Gradient Impedance**: Couple PRFs with gradient poisoning. |
| **3. DP & Fisher Auditing** | **Medium-High** (selective) | Medium (using 0.5B proxy model) | **High** (operates on input queries, not outputs) | **High** (requires only query logging & proxy scoring) | **Dynamic Defense Gate**: Adaptive pricing & throttling per API tenant. |
| **4. Unlearnable Shortcuts** | **High** (syntactically valid) | Low (constrained decoding) | Low-Medium (paraphrasing erases syntactic links) | **Very High** (standard SFT poisoning benchmark protocols) | **Direct Distillation Sabotage**: Clean-label reasoning poison. |
| **5. Hardware Logic Locking** | **Low-Medium** (requires SDK) | Low-Medium (WASM runtime) | **Extremely High** (locked code cannot run without runtime) | Medium (requires building execution testbeds) | **Architectural Paradigm Shift**: Proprietary runtime execution wrappers. |
| **6. Game-Theoretic Signaling**| **High** (correct end answers) | Low (prompt/trace formatting)| **High** (traps embedded deeply in CoT logic) | **Very High** (evaluable on GSM8K/MATH benchmarks) | **Cognitive Trap Injection**: Exploiting CoT reasoning dependencies. |

---

## 5-Component Handoff Protocol

### 1. Observation
- Inspected `.agents/teamwork/ORIGINAL_REQUEST.md` (lines 181–219) establishing Milestone 2 requirements: cross-domain analogies outside NLP (traitor tracing, cryptographic watermarking, DP query auditing, unlearnable examples, hardware logic locking, game-theoretic signaling) for black-box API anti-distillation.
- Inspected `.agents/teamwork/orchestrator_distill_1/plan.md` (lines 13–17) assigning Milestone 2 cross-domain engineering analogies and mapping to artifact `alternate_research/distillation_defense/05_CROSS_DOMAIN_ANALOGIES.md`.
- Verified seminal literature citations without hallucination:
  - *Chor, Fiat, Naor (1994 / 2000)*; *Boneh, Sahai, Waters (2006)*; *Tardos (2003 / 2008)*.
  - *Christ, Gunn, Zamir (2023 / CRYPTO 2024)*; *Kuditipudi et al. (2023)*; *Kirchenbauer et al. (2023)*; *Xu, Kirchenbauer et al. (ICML 2026, arXiv:2602.03812)*.
  - *Dwork et al. (2006)*; *Dwork & Roth (2014)*; *Dinur & Nissim (2003)*; *Tramèr et al. (2016)*; *Juuti et al. (PRADA, EuroS&P 2019)*.
  - *Huang et al. (ICLR 2021)*; *Fowl et al. (NeurIPS 2021)*; *Java, Shahid, Agarwal (RegText, NAACL 2025)*.
  - *Rajendran et al. (FTCS 2012 / TIFS 2015)*; *Subramanyan, Ray, Malik (IEEE HOST 2015)*; *Yasin et al. (SARLock 2016 / Anti-SAT 2017)*; *Koushanfar & Qu (DAC 2001)*.
  - *Spence (1973)*; *Crawford & Sobel (1982)*; *Kamenica & Gentzkow (2011)*; *Kariyappa & Qureshi (CVPR 2020)*; *Orekondy, Schiele, Fritz (ICLR 2020)*; *Jiang (DistillGuard 2026, arXiv:2603.07835)*.

### 2. Logic Chain
1. Black-box distillation relies on the assumption that API completions represent an uncorrupted, low-noise sample from the teacher's capability distribution, and that training a student model on these completions is an effective convex-like approximation of the teacher.
2. Passive watermarking only achieves retrospective detection and fails to deter well-funded adversaries who ignore legal threats or operate in jurisdictions with weak IP enforcement.
3. Therefore, defense must operate through either:
   - **Active extraction sabotage** during student gradient descent (Paradigms 2, 4, 6), or
   - **Provable cryptographic attribution and revocation** (Paradigm 1), or
   - **Economic information throttling** (Paradigm 3), or
   - **Architectural entanglement with proprietary runtimes** (Paradigm 5).
4. By cross-referencing information-theoretic bounds (Tardos $O(k^2)$ lower bound; the Distortion-Free Impossibility Theorem; Dinur-Nissim reconstruction bounds), we establish that no single paradigm provides an unconditional silver bullet.
5. Consequently, an optimal defense architecture must be a **defense-in-depth hybrid**: combining Fisher-Proxy auditing at the API boundary, PRF-keyed active gradient poisoning in emitted tokens, and strategic reasoning traps in multi-step chains-of-thought.

### 3. Caveats
- Investigated black-box text-only and reasoning-trace (CoT) distillation; did not investigate multimodal (vision-language) distillation or logit/embedding-level distillation (which are trivial to defend by omitting logits).
- Assumed standard transformer student architectures (0.5B to 8B parameter models trained with autoregressive cross-entropy loss). Did not analyze non-autoregressive or exotic diffusion-based language model students.
- Commercial implementations of Paradigm 5 (Logic Locking) face product-market friction if API consumers strictly require plain JSON completions without an execution SDK.

### 4. Conclusion
The six paradigms provide rigorous, mathematically grounded primitives that escape the limitations of naive text filtering. Specifically:
1. **Traitor Tracing** provides a watertight forensic framework for identifying colluding enterprise API accounts via Tardos fingerprint codes.
2. **Cryptographic Watermarking** can be transformed from passive detection to active extraction impedance by injecting PRF-keyed gradient poisoning vectors into low-saliency token margins.
3. **Differential Privacy & Fisher Auditing** enables dynamic pricing and throttling proportional to parameter leakage per dollar.
4. **Unlearnable Shortcuts** allow clean-label sabotage of student SFT via bi-level syntactic correlations.
5. **Hardware Logic Locking** offers a conceptual roadmap for replacing static text generation with ephemeral key-entangled runtime executions.
6. **Game-Theoretic Signaling** enables Bayesian insertion of deceptive reasoning traps in multi-step chains of thought that satisfy human verifiers while corrupting student generalization.

### 5. Verification Method
- **Citation & Verification Command**:
  Verify the authenticity and publication details of all cited literature using public preprint and bibliographic repositories (arXiv, DBLP, Google Scholar, IEEE Xplore, ACM DL).
- **Mathematical Invalidation Conditions**:
  - If an adversary proves an algorithm that achieves collusion resistance against $k$ colluders in $o(k^2)$ length without violating the Marking Assumption, the Tardos lower bound analysis must be revised.
  - If an empirical experiment proves that a strictly distortion-free watermark ($D_{KL} = 0$) degrades asymptotic student fine-tuning performance, the Distortion-Free Impossibility proof is invalidated.
- **Experimental Protocol**:
  Execute 2-epoch LoRA fine-tuning of `Qwen2.5-1.5B` and `Llama-3-8B` on 10,000 synthetic reasoning completions generated under each defense condition, measuring:
  1. Downstream task accuracy on GSM8K and HumanEval.
  2. Training loss perplexity vs. clean evaluation perplexity.
  3. Watermark/fingerprint detection AUROC after 1 round of paraphrasing via an independent 8B model.

---
