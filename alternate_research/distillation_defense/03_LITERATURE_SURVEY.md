# Systematic Literature Survey: Model Extraction and Knowledge Distillation from Large Language Models (2016–2026)

**Campaign**: Anti-Distillation Defense Campaign (ALT-DIST-001)  
**Deliverable**: Milestone 1 (M1) Literature Survey (`03_LITERATURE_SURVEY.md`)  
**Investigator**: `explorer_distill_lit_1` (Literature Scout)  
**Date**: October 2026  
**Status**: Completed  
**Epistemic Standards**: Strict `AGENTS.md` classification applied throughout:
- `[SOURCE FACT]`: Direct finding, formal theorem, or empirical measurement from a verified, peer-reviewed publication or technical report.
- `[INFERENCE]`: Methodological deduction or logical synthesis supported by source facts.
- `[HYPOTHESIS]`: Theoretical claim, open question, or projected vulnerability requiring further empirical validation.

---

## 1. Executive Overview & Scope

Model extraction—the unauthorized replication of a proprietary model's behavior, decision boundaries, or representations by observing its outputs under API query access—has evolved across three distinct eras:
1. **Continuous Classification APIs (2016–2020)**: Stealing linear, convolutional, and BERT-based classifiers via equation solving, active learning, and synthetic query optimization.
2. **Generative Instruction Following (2021–2023)**: Distilling general conversational and instruction-following capabilities from proprietary LLMs into open-weight base models via sequence-level teacher generation (Self-Instruct, Alpaca).
3. **Reasoning-Trace & Search Trajectory Distillation (2024–2026)**: Distilling complex multi-step reasoning, test-time search rollouts, and self-correction chains (o1/o3 imitation, DeepSeek-R1) where the extracted commodity is the execution trace of internal test-time compute.

This survey establishes the complete lineage of extraction attacks, categorizes attack vectors, audits existing defenses (watermarking, query anomaly detection, output perturbation, proof of learning), and isolates the fundamental information-theoretic and algorithmic bottlenecks that render naive defenses ineffective against text-only black-box distillation.

---

## 2. Chronological & Conceptual Evolution of Extraction Attacks (2016–2026)

```
[2016] Tramèr et al. (USENIX Security) ─────┐ First formulation of ML prediction API extraction
                                            │
[2017-2019] Papernot; Orekondy (CVPR) ──────┤ Knockoff Nets & unlabelled transfer sets
                                            │
[2020] Jagielski et al.; Krishna et al. ────┤ High-fidelity active extraction; BERT API stealing
                                            │
[2020] Wallace et al. (EMNLP) ──────────────┤ Machine Translation imitation & adversarial transfer
                                            │
[2022-2023] Wang et al.; Taori et al. ──────┤ Self-Instruct & Alpaca: Generative LLM distillation
                                            │
[2023-2024] Gudibande et al. (ICLR) ────────┤ Critical limit: Style vs. reasoning imitation gap
                                            │
[2024] Carlini et al. (ICML Best Paper) ────┤ Stealing unembeddings & hidden dims via logits
                                            │
[2024-2026] DeepSeek-R1; o1/o3 Distill ─────┘ Distillation of reasoning traces & RL rollouts
```

### 2.1 Phase I: Continuous Classifiers & Prediction APIs (2016–2020)

#### Tramèr et al. (2016) — *Stealing Machine Learning Models via Prediction APIs*
- **Citation**: Florian Tramèr, Fan Zhang, Ari Juels, Michael K. Reiter, Thomas Ristenpart. 25th USENIX Security Symposium (USENIX Security 16), pp. 601–618, 2016. (Awarded USENIX Security Test of Time Award 2026).
- **Core Mechanism**: `[SOURCE FACT]` Tramèr et al. demonstrated that black-box machine learning APIs returning probability vectors or confidence scores leak sufficient constraint equations to reconstruct model parameters. For linear and logistic regression, extraction reduces to solving a system of linear equations ($d+1$ queries for dimension $d$). For multilayer perceptrons (MLPs), the authors applied numerical gradient-based optimization and modified active learning to approximate decision boundaries with near 100% fidelity.
- **Key Takeaway**: Even when APIs return rounded probabilities or only top-1 hard labels, model stealing remains tractable by constructing inputs near decision boundaries.

#### Orekondy et al. (2019) — *Knockoff Nets: Stealing Functionality of Black-Box Models*
- **Citation**: Tribhuvanesh Orekondy, Bernt Schiele, Mario Fritz. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2019), pp. 4953–4961.
- **Core Mechanism**: `[SOURCE FACT]` Formalized the distinction between *parameter extraction* (reconstructing exact weights) and *functionality stealing* (training a student model that achieves comparable test accuracy). They introduced "Knockoff Nets", showing that an adversary does not need the victim's training data: querying an out-of-domain, open-source dataset (e.g., ImageNet queries sent to a specialized facial recognition or medical model) yields probability vectors sufficient to train a student to within 76–82% of victim performance.
- **Key Takeaway**: `[INFERENCE]` The attack vector in black-box extraction is the *label distribution on surrogate queries*, decoupling the attack from any prerequisite access to the proprietary training distribution.

#### Jagielski et al. (2020) — *High Accuracy and High Fidelity Extraction of Neural Networks*
- **Citation**: Matthew Jagielski, Nicholas Carlini, David Berthelot, Alex Kurakin, Nicolas Papernot. 29th USENIX Security Symposium (USENIX Security 20), pp. 1345–1362, 2020.
- **Core Mechanism**: `[SOURCE FACT]` Formalized the mathematical divergence between two extraction goals:
  1. **Accuracy Extraction**: Maximizing task accuracy $\mathbb{E}_{(x,y) \sim \mathcal{D}} [\mathbb{I}(f_S(x) = y)]$.
  2. **Fidelity Extraction**: Maximizing agreement with the victim $\mathbb{E}_{x \sim \mathcal{D}} [\mathbb{I}(f_S(x) = f_V(x))]$.
  The authors introduced an active learning approach combining semi-supervised learning (MixMatch) with active query selection near decision boundaries, reducing query complexity by over an order of magnitude (e.g., matching victim accuracy with $10\times$ fewer queries than random sampling).
- **Key Takeaway**: High-fidelity extraction yields an identical surrogate that reproduces not only the victim's accuracy, but also its adversarial vulnerabilities and systemic edge-case errors.

#### Krishna et al. (2020) — *Thieves on Sesame Street! Model Extraction of BERT-based APIs*
- **Citation**: Kalpesh Krishna, Gaurav Singh Tomar, Ankur P. Parikh, Nicolas Papernot, Mohit Iyyer. International Conference on Learning Representations (ICLR 2020).
- **Core Mechanism**: `[SOURCE FACT]` Evaluated extraction on NLP classification APIs powered by fine-tuned BERT models. Demonstrated that an attacker querying the API with *random sequences of English words* (gibberish/nonsensical inputs) and using task-specific surrogate fine-tuning can steal 96–99% of the victim model's downstream performance for under $100–$500 in API costs.
- **Key Takeaway**: `[INFERENCE]` Pretrained representation geometry (e.g., Transformer backbones) acts as an inductive prior that drastically lowers the entropy needed from victim outputs. Out-of-distribution queries extract sufficient head-layer projection information without needing grammatically valid prompts.

#### Wallace et al. (2020) — *Imitation Attacks and Defenses for Black-box Machine Translation Systems*
- **Citation**: Eric Wallace, Mitchell Stern, Dawn Song. Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP 2020), pp. 5531–5546. arXiv:2004.15015.
- **Core Mechanism**: `[SOURCE FACT]` Extended functionality stealing to generative sequence-to-sequence models (Machine Translation). Querying black-box translation APIs (Google Translate, Bing, Systran) with monolingual text and training student seq2seq models produced students that trailed target systems by under 0.6 BLEU points. Crucially, adversarial perturbations generated on the local imitation models transferred back to the target commercial APIs with high attack success rates.
- **Key Takeaway**: Sequence-level generation APIs inherently expose their generative mapping to imitation attacks, which subsequently enable surrogate-to-target adversarial transfer.

---

### 2.2 Phase II: Instruction Tuning & Generative Imitation (2021–2023)

#### Kim & Rush (2016) — *Sequence-Level Knowledge Distillation*
- **Citation**: Yoon Kim, Alexander M. Rush. Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing (EMNLP 2016), pp. 1317–1327. arXiv:1606.07947.
- **Core Mechanism**: `[SOURCE FACT]` Provided the foundational machine learning framework underpinning modern LLM distillation. Showed that standard word-level knowledge distillation (Hinton et al., 2015) fails to optimize entire output sequences due to exposure bias. Introduced **Sequence-Level KD**: training the student model on beam-search or greedy completions generated by the teacher, minimizing cross-entropy over teacher sequence trajectories:
  $$\mathcal{L}_{\text{seq}} = -\sum_{t=1}^{|y^*|} \log P_S(y^*_t \mid x, y^*_{<t}), \quad \text{where } y^* = \arg\max_y P_T(y \mid x)$$
- **Key Takeaway**: Hard sequence tokens generated by a teacher are mathematically sufficient to align student sequence distributions without access to full teacher logit tensors.

#### Wang et al. (2022/2023) — *Self-Instruct: Aligning Language Models with Self-Generated Instructions*
- **Citation**: Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa Liu, Noah A. Smith, Daniel Khashabi, Hannaneh Hajishirzi. Findings of the Association for Computational Linguistics: ACL 2023, pp. 13484–13508. arXiv:2212.10560.
- **Core Mechanism**: `[SOURCE FACT]` Introduced an automated pipeline bootstrapping instruction-following capabilities. Starting from 175 seed prompt tasks, the target LLM (GPT-3 / `text-davinci-003`) is prompted to:
  1. Generate new task instructions ($I$).
  2. Formulate input contexts ($X$).
  3. Produce corresponding outputs ($Y$).
  Pruning samples with low ROUGE novelty produced 52,000 diverse synthetic instruction pairs. Fine-tuning a base GPT-3 model on this synthetic corpus elevated its human evaluation performance by 33%, nearly matching `InstructGPT-001`.
- **Key Takeaway**: The extraction attack surface transitioned from querying existing datasets to *active prompt generation by the victim itself*, eliminating the need for attacker-curated task datasets.

#### Taori et al. (2023) — *Stanford Alpaca: An Instruction-following LLaMA model*
- **Citation**: Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, Carlos Guestrin, Percy Liang, Tatsunori B. Hashimoto. Stanford Center for Research on Foundation Models (CRFM) Technical Release, March 2023.
- **Core Mechanism**: `[SOURCE FACT]` Applied the Self-Instruct methodology to extract instruction-following behavior from OpenAI’s `text-davinci-003` into Meta’s open-weight `LLaMA-7B`. Generated 52,000 instruction-response pairs via OpenAI API calls for under $500. Fine-tuning LLaMA-7B for 3 epochs on this synthetic data produced a student model exhibiting conversational and formatting behaviors qualitatively indistinguishable from `text-davinci-003` on simple conversational tasks.
- **Key Takeaway**: `[INFERENCE]` Industrialized model extraction into standard open-source practice, establishing that frontier instruction alignment can be acquired via commodity API budgets ($< $1,000).

#### Gudibande et al. (2023/2024) — *The False Promise of Imitating Proprietary LLMs*
- **Citation**: Arnav Gudibande, Eric Wallace, Charlie Snell, Xinyang Geng, Hao Liu, Pieter Abbeel, Sergey Levine, Dawn Song. International Conference on Learning Representations (ICLR 2024). arXiv:2305.15717.
- **Core Mechanism**: `[SOURCE FACT]` Conducted a rigorous empirical study investigating whether instruction imitation transfers true model capabilities or merely superficial style. Fine-tuned student base models (1.5B to 13B parameters, including LLaMA and GPT-2) on up to 150k outputs from ChatGPT / GPT-4 across standard benchmarks.
- **Empirical Findings**:
  1. **Style Imitation vs. Factuality Gap**: Crowd-workers and GPT-4 evaluators rate imitation models as highly competitive because they master tone, formatting, verbosity, and conversational structure.
  2. **Zero Capability Closure**: On rigorous objective benchmarks testing factual recall, reasoning, and domain knowledge (NaturalQuestions, MMLU, Big-Bench), imitation models fail to close the performance gap between the student base model and the proprietary teacher.
  3. **Data Scaling Saturation**: Increasing imitation query volume from 10k to 150k yields diminishing returns, plateauing quickly at the capability ceiling dictated by the student's base pretraining.
- **Key Takeaway**: `[SOURCE FACT]` Hard-token imitation on generic conversational queries distills *output stylistic priors and behavioral alignment*, but does not transfer world knowledge or algorithmic reasoning absent from the student's base weights.

---

### 2.3 Phase III: White-Box Parameter Leakage from Black-Box APIs (2024)

#### Carlini et al. (2024) — *Stealing Part of a Production Language Model*
- **Citation**: Nicholas Carlini, Daniel Paleka, Krishnamurthy Dj Dvijotham, Thomas Steinke, Jonathan Hayase, A. Feder Cooper, Katherine Lee, Matthew Jagielski, Christopher A. Choquette-Choo, Florian Tramèr. 41st International Conference on Machine Learning (ICML 2024), Best Paper Award. arXiv:2403.06634.
- **Core Mechanism**: `[SOURCE FACT]` Demonstrated that standard logit/logprob API endpoints leak exact linear algebraic projections of internal representations. For an autoregressive transformer with hidden dimension $h$, vocabulary size $V$, and unembedding matrix $W_U \in \mathbb{R}^{h \times V}$, the logit vector for output token $t$ given hidden state $h_t$ is:
  $$z_t = h_t W_U + b$$
  When an API provides full or top-$k$ logprobs, the adversary can compute differences between logit entries across varying prompt prefixes.
- **Attack Formulation**:
  1. **Subspace Recovery**: By generating diverse prompt prefixes $x^{(1)}, \dots, x^{(N)}$ and measuring logit shifts across tokens, the adversary constructs a matrix of logit differences $M \in \mathbb{R}^{N \times V}$.
  2. **Singular Value Decomposition (SVD)**: Because $\text{rank}(M) \le h$, the rank of $M$ directly reveals the hidden dimension size $h$.
  3. **Exact Projection Matrix Extraction**: Truncated SVD yields the row space of $W_U$ up to an orthogonal transformation:
     $$M = U \Sigma V^T \implies \hat{W}_U = \Sigma_{1:h} V_{1:h}^T$$
- **Empirical Results**:
  - Extracted the complete $W_U$ projection matrix for OpenAI’s `Ada` ($h = 1024$) and `Babbage` ($h = 2048$) models for under $20 USD.
  - Deduced the exact hidden dimension of `gpt-3.5-turbo` ($h = 4096$).
  - Estimated that recovering the full projection matrix for `gpt-3.5-turbo` requires $< $2,000 USD.
- **Key Takeaway**: `[SOURCE FACT]` Returning soft tokens (logits/logprobs) creates a fatal side-channel: linear algebraic reconstruction recovers white-box model weights and dimensions from a black-box API.

---

### 2.4 Phase IV: Reasoning-Trace & Search Trajectory Distillation (2024–2026)

#### Emergence of Reasoning Models (OpenAI o1/o3, DeepSeek-R1)
- **Background**: The paradigm shift from standard autoregressive next-token prediction to *test-time compute scaling* (reinforcement learning over long chains-of-thought, search rollouts, and self-correction tokens) radically altered the value proposition of model extraction.
- **Distilling Step-by-Step** (`[SOURCE FACT]`, Hsieh et al., ACL 2023): Demonstrated that training smaller models on rationale-augmented data (Chain-of-Thought) allows a 770M parameter T5 student to outperform a 540B PaLM model on few-shot reasoning tasks with $80\times$ less training data than standard KD.
- **DeepSeek-R1 Technical Framework** (`[SOURCE FACT]`, DeepSeek-AI, January 2025; arXiv:2501.12948):
  1. **Pure RL Emergence (R1-Zero)**: Large-scale reinforcement learning (Rule-Based Reward Models on mathematical correctness and compiler execution) causes models to autonomously discover reflection, backtracking, and long chain-of-thought exploration without supervised fine-tuning.
  2. **Distillation Paradigm (DeepSeek-R1-Distill)**: DeepSeek curated 800,000 high-quality reasoning traces generated by DeepSeek-R1 (comprising mathematical proofs, code solutions, and self-correction sequences).
  3. **Supervised Distillation vs. RL**: Fine-tuning standard open-weight base models (Qwen-2.5-1.5B, 7B, 14B, 32B, and LLaMA-3-8B, 70B) directly on these 800k CoT traces produced the **DeepSeek-R1-Distill** series.
  4. **Empirical Performance**: `DeepSeek-R1-Distill-Qwen-32B` achieved 72.6% pass@1 on AIME 2024 and 94.3% on MATH-500—outperforming larger models trained with direct RL.
- **Industrial Imitation Vectors (o1/o3 Extraction)**:
  - `[INFERENCE]` Distillation of frontier reasoning models bypasses the millions of GPU hours required for large-scale RL exploration. By querying an o1/o3-class model on problem sets and capturing the intermediate reasoning trajectory, an adversary extracts the search policy directly into a supervised student.
  - **The Frontier Countermeasure (Hidden CoT)**: To combat this extraction, frontier providers (OpenAI, Anthropic) do not stream raw internal reasoning tokens through public APIs. Instead, they expose only summarized reasoning or final responses, explicitly to thwart competitor distillation of test-time search traces.

---

## 3. Attack Modalities & Extraction Vectors: Comparative Taxonomy

To formulate rigorous defenses, the technical vectors through which model knowledge leaks across an API boundary must be mathematically separated.

| Vector | Output Data Type | Information Content per Query | Sample Complexity to Extract | Primary Vulnerability / Target | Key Reference |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Hard Tokens** | Discrete token sequence $y_{1:T}$ | Low ($\approx \log_2 V$ bits/token observed, but sparse) | High ($10^4 - 10^6$ sequences) | Style, task alignment, conversational priors | Kim & Rush (2016); Taori et al. (2023) |
| **Soft Labels (Logits/Logprobs)** | Top-$k$ probabilities $P(y_t \mid x)$ | Extreme ($\mathcal{O}(k)$ continuous values/token) | Low ($10^2 - 10^3$ sequences) | Embedding/unembedding matrix, hidden dims, soft prompts | Carlini et al. (ICML 2024 Best Paper) |
| **Raw Reasoning Traces** | Full Chain-of-Thought $\tau = (s_1, \dots, s_L)$ | High-order algorithmic (search tree, backtracks) | Moderate ($10^3 - 10^5$ problems) | Search policy, self-correction heuristics, math/code logic | DeepSeek-AI (2025); Hsieh et al. (2023) |
| **Summarized Reasoning** | Filtered/abbreviated CoT summary + answer | Medium (coarse step milestones) | High ($10^5 - 10^6$ problems) | Task outcome, high-level plan (degraded self-correction) | OpenAI o1/o3 API deployment (2024) |
| **Active Query Synthesis** | Adaptively crafted prompts $x_{t+1} = \mathcal{A}(x_{1:t}, y_{1:t})$ | Maximizes boundary entropy $H(Y \mid X)$ | Minimal (optimal sample complexity) | Complete decision boundary / task domain coverage | Chandrasekaran et al. (2020); Xu et al. (2024) |
| **Passive Dataset Distillation** | Natural/unlabeled prompts $x \sim \mathcal{D}_{\text{in-domain}}$ | Natural prompt distribution coverage | Sub-optimal (redundant queries) | General task accuracy, frequent-case fidelity | Orekondy et al. (2019); Taori et al. (2023) |

### 3.1 Deep Dive: Hard Tokens vs. Soft Labels
- **Soft Labels Leak Geometry**: When an API returns top-5 logprobs, an attacker can directly minimize Kullback-Leibler (KL) divergence:
  $$\mathcal{L}_{\text{KD}} = \sum_{t} D_{\text{KL}}(P_T(\cdot \mid x, y_{<t}) \,\|\, P_S(\cdot \mid x, y_{<t}))$$
  This provides dense supervisory gradients for every token in the vocabulary, revealing relative semantic distances between non-selected candidate tokens.
- **Hard Tokens Obscure Geometry**: In a text-only, greedy or sampled API, the attacker observes only a single realization $y_t \sim P_T(\cdot \mid x, y_{<t})$. The loss collapses to standard negative log-likelihood (NLL) over the single token:
  $$\mathcal{L}_{\text{seq}} = -\sum_{t} \log P_S(y_t \mid x, y_{<t})$$
- `[INFERENCE]` Restricting APIs strictly to hard tokens increases the query volume required for parameter extraction by orders of magnitude, but **does not prevent sequence-level knowledge distillation** of task competency or behavioral alignment.

### 3.2 Deep Dive: Active Query Synthesis vs. Passive Querying
- **Active Synthesis (Evol-Instruct, Self-Instruct)**: The attacker uses the victim model itself as an active learner (`[SOURCE FACT]`, Chandrasekaran et al., 2020; Xu et al., 2024). By prompting the model to deepen instructions (adding constraints, increasing reasoning complexity, introducing edge cases), the attacker explores low-density regions of the prompt manifold where model knowledge is highest.
- **Passive Distillation**: Querying static scraped datasets (e.g., Common Crawl, Reddit) concentrates queries in high-probability semantic modes, causing redundant sampling and failing to challenge model reasoning limits.

---

## 4. Existing Defense Landscape & Failure Modes

```
                    ┌─── Query Anomaly Detection (PRADA, API Filters)
                    │    └── Failure: Sybil attacks, distributed IP pools, natural prompt spoofing
                    │
                    ├─── Token Watermarking (Kirchenbauer, Aaronson, Christ, Kuditipudi)
                    │    └── Failure: Paraphrasing (DIPPER), student training loss filtering
Defenses Audited ───┤
                    ├─── Output Perturbation & Poisoning (Lee et al., DAWN, Misinformation)
                    │    └── Failure: Destroys benign utility; induces hallucination/incoherence
                    │
                    └─── Proof of Learning & IP Verification (Jia et al., 2021)
                         └── Failure: Requires white-box audit; easily forged or bypassed by student training
```

### 4.1 Query Distribution Anomaly Detection & API Rate Limiting

#### PRADA (Juuti et al., 2019)
- **Citation**: Mika Juuti, Sebastian Szyller, Samuel Marchal, N. Asokan. *PRADA: Protecting Machine Learning Models against Model Extraction Attacks*. 2019 IEEE European Symposium on Security and Privacy (EuroS&P), pp. 511–526.
- **Mechanism**: `[SOURCE FACT]` Tracks the distribution of distances between consecutive queries submitted by an API client. Model extraction algorithms (e.g., active learning or synthetic boundary exploration) generate query distributions whose nearest-neighbor distances follow a non-normal, clustered distribution. PRADA detects these anomalies using a Shapiro-Wilk test over query distance distributions.
- **Failure Modes & Bypasses in Modern LLMs**:
  1. **Sybil & Distributed Query Pools**: `[SOURCE FACT]` An attacker partitions queries across thousands of separate API accounts, IP addresses, or cloud tenancies. Because each individual tenant submits only a tiny volume of queries ($< 10$ queries/day), per-client statistical distance tests fail to accumulate sufficient samples.
  2. **In-Distribution Natural Prompts**: For generative LLMs, attacks like Alpaca and Evol-Instruct query the model using syntactically fluent, natural-language instructions. The distance metric between synthetic instruction prompts and genuine user prompts is statistically indistinguishable from natural user traffic.
  3. **Offline Caching & Replay**: An attacker can mix extraction queries with benign web search queries or standard customer queries, masking any distributional drift.

---

### 4.2 Watermarking Schemes & Their Demise Under Distillation

Watermarking aims to embed a detectable statistical signature into model generations. Four major families dominate the literature:

#### 1. Green/Red List Token Watermarking (Kirchenbauer et al., 2023)
- **Citation**: John Kirchenbauer, Jacob Geiping, Yuxin Wen, Jonathan Katz, Ian Miers, Tom Goldstein. *A Watermark for Large Language Models*. 40th International Conference on Machine Learning (ICML 2023). arXiv:2301.10226.
- **Mechanism**: `[SOURCE FACT]` Hashes the preceding token $y_{t-1}$ with a secret key $K$ to pseudo-randomly partition the vocabulary $V$ into a "green list" $G$ of size $\gamma |V|$ and a "red list" $R$ of size $(1-\gamma)|V|$. During sampling, a bias $\delta > 0$ is added to the logits of all tokens in $G$. Detection computes a z-score testing whether the proportion of green tokens in a candidate text exceeds the expectation $\gamma$.

#### 2. Cryptographic Pseudorandom & Gumbel Watermarking (Aaronson & Kirchner, 2022/2023)
- **Citation**: Scott Aaronson, Hendrik Kirchner. *Watermarking GPT Outputs*. Technical presentation / lecture series, UT Austin & OpenAI, 2022–2023.
- **Mechanism**: `[SOURCE FACT]` Instead of modifying logits directly, the sampling process evaluates a pseudo-random function $\text{PRF}_K(y_{t-1}, y_t) \sim \text{Uniform}(0, 1)$ using Gumbel max sampling:
  $$y_t = \arg\max_{v \in V} \left( \log P_T(v \mid y_{<t}) - \log(-\log \text{PRF}_K(y_{<t}, v)) \right)$$
  This guarantees that individual marginal token distributions remain distortion-free while enforcing a secret joint distribution detectable with key $K$.

#### 3. Undetectable Watermarks (Christ et al., 2023/2024)
- **Citation**: Miranda Christ, Sam Gunn, Or Zamir. *Undetectable Watermarks for Language Models*. Thirty-Seventh Annual Conference on Learning Theory (COLT 2024). arXiv:2306.09194.
- **Mechanism**: `[SOURCE FACT]` Formulates watermarking under cryptographic indistinguishability. Text generated under the watermarking key $K$ is computationally indistinguishable from unwatermarked model generations for any polynomial-time adversary lacking $K$. Only the keyholder can detect the watermark via a statistical test.

#### 4. Robust Distortion-Free Watermarks (Kuditipudi et al., 2023/2024)
- **Citation**: Rohith Kuditipudi, John Thickstun, Tatsunori Hashimoto, Percy Liang. *Robust Distortion-Free Watermarks for Language Models*. 41st International Conference on Machine Learning (ICML 2024). arXiv:2307.15593.
- **Mechanism**: `[SOURCE FACT]` Preserves exact unwatermarked token probability distributions using inverse transform sampling or exponential minimum sampling mapped against pseudorandom sequences, ensuring robustness against bounded text edits while eliminating perceptual quality degradation.

#### Why Watermarking Fails as an Anti-Distillation Defense
Despite mathematical sophistication in text attribution, watermarking fails to prevent model extraction due to three structural vulnerabilities:

1. **The Paraphrasing & Rewriting Bottleneck**:
   - `[SOURCE FACT]` **Sadasivan et al. (2023)** (*Can AI-Generated Text be Reliably Detected?*, arXiv:2303.11156) proved theoretically and empirically that a recursive paraphrasing attack strips watermarks. As the total variation distance between model text and human text decreases, detector AUROC bounds degrade toward chance.
   - `[SOURCE FACT]` **Krishna et al. (NeurIPS 2023)** (*Paraphrasing evades detectors of AI-generated text, but retrieval is an effective defense*, NeurIPS 2023) showed that passing LLM generations through open paraphrasers (such as DIPPER) reduces detector accuracy from $> 70\%$ to $< 5\%$ while retaining semantic meaning. If an extraction adversary paraphrases teacher completions prior to training the student, watermarks are wiped clean.

2. **The Student Network as a Lossy Compressor**:
   - `[INFERENCE]` When a student model is trained via next-token prediction on teacher generations:
     $$\min_{\theta_S} \mathbb{E}_{(x, y)} \left[ -\sum_{t=1}^{|y|} \log P_{\theta_S}(y_t \mid x, y_{<t}) \right]$$
     the student optimizes for semantic coherence and general language modeling likelihood. Pseudo-random high-frequency token shifts introduced by green-list hashing or Gumbel keys act as uncorrelated sampling noise. Unless the student overfits drastically to exact token choices, standard regularized gradient descent filters out the watermark, absorbing only the semantic task mapping.

3. **Post-Hoc Attribution vs. Extraction Prevention ("Radioactivity" Limits)**:
   - `[SOURCE FACT]` **Sander et al. (NeurIPS 2024 Spotlight)** (*Watermarking Makes Language Models Radioactive*, arXiv:2402.14904) demonstrated that watermarking *can* leave subtle statistical traces in downstream models trained on watermarked data (detecting model training contamination from as little as 5% watermarked corpus).
   - `[INFERENCE]` However, "radioactivity" is a **forensic tracing tool**, not a defense. It does not prevent an adversary from training a capable student; it merely allows the victim to attempt post-hoc legal discovery after the intellectual property has already been extracted and commercialized.

---

### 4.3 Information Perturbation & Soft-Label Poisoning

#### Deceptive Perturbations & Adaptive Misinformation
- **Citations**:
  - Taesung Lee, Benjamin Edwards, Ian Molloy, Dong Su. *Defending Against Neural Network Model Stealing Attacks Using Deceptive Perturbations*. IEEE Security and Privacy Workshops (SPW 2019), pp. 43–49.
  - Sanjay Kariyappa, Moinuddin K. Qureshi. *Defending Against Model Stealing Attacks With Adaptive Misinformation*. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2020), pp. 1326–1335.
  - Sebastian Szyller, Buse Gul Atli, Samuel Marchal, N. Asokan. *DAWN: Dynamic Adversarial Watermarking of Neural Networks*. 29th ACM International Conference on Multimedia (ACM MM 2021), pp. 4417–4425.
- **Mechanism in Classification**: `[SOURCE FACT]` For continuous classification APIs, defenders alter output logits on suspected extraction queries or out-of-distribution (OOD) queries by:
  - Injecting structured noise into the probability vector.
  - Swapping the top predicted class for a runner-up class on a small fraction ($< 0.5\%$) of inputs (DAWN).
  - Maximizing surrogate training loss while maintaining top-1 accuracy for normal users.
- **Fatal Breakdown in Generative LLMs (The Utility-Security Dilemma)**:
  - `[INFERENCE]` In generative autoregressive text generation, every token selection is causally linked to all future tokens. Perturbing output tokens (e.g., intentionally sampling lower-probability tokens or injecting factual errors to poison a student model) directly degrades text coherence, grammar, and factual correctness for paying benign users.
  - Generative text has no analog to the continuous margin manipulation possible in image classification vectors: flipping a single token in code generation syntax or medical reasoning results in broken code execution or dangerous factual hallucination.

---

### 4.4 Proof of Learning & IP Verification

#### Proof of Learning (Jia et al., 2021)
- **Citation**: Hengrui Jia, Christopher A. Choquette-Choo, Varun Chandrasekaran, Nicolas Papernot. *Proof-of-Learning: Definitions and Practice*. 2021 IEEE Symposium on Security and Privacy (S&P 2021), pp. 1039–1056.
- **Mechanism**: `[SOURCE FACT]` Establishes model ownership by proving that a party expended the computational work necessary to train the model. During stochastic gradient descent (SGD), the trainer logs intermediate checkpoint hashes, mini-batch sequence seeds, and parameter updates. Verification checks whether re-executing steps from the checkpoint log validly reproduces the final weight tensor.
- **Failure Modes Against Black-Box Distillation**:
  - `[SOURCE FACT]` Proof of Learning is an integrity check for *weight-theft or Byzantine distributed training*, not a defense against distillation.
  - When an adversary extracts a model via black-box API calls, the adversary legitimately expends their own compute to train the student model from scratch. The adversary possesses a valid, legitimate Proof of Learning for their student model.
  - PoL provides zero technical or legal mechanism to distinguish between a student trained on proprietary API outputs versus a student trained on licensed open-source datasets.

---

## 5. Synthesis: Core Technical Bottlenecks in Black-Box Text-Only API Defense

Synthesizing across 2016–2026 literature, defending a black-box text-only inference API against sequence-level knowledge distillation faces three fundamental information-theoretic and algorithmic bottlenecks:

```
┌────────────────────────────────────────────────────────────────────────┐
│               THE THREE IRREDUCIBLE BOTTLENECKS                       │
└────────────────────────────────────────────────────────────────────────┘

  1. Semantic Equivalence & Paraphrase Invariance
     ┌──────────────┐     Paraphrase / Rephrase     ┌──────────────┐
     │ Watermarked  │ ────────────────────────────> │ Cleaned      │
     │ Teacher Text │                               │ Student Text │
     └──────────────┘                               └──────────────┘
     • Watermark resides in high-frequency token syntax.
     • Task utility resides in invariant semantic meaning.
     • Adversary preserves semantic meaning while wiping syntactic marks.

  2. The Information Channel Equivalence
     • A query containing an advanced math problem or coding challenge
       from an extraction attacker is byte-for-byte indistinguishable from
       a query from an enterprise user or software developer.
     • The server cannot classify intent without profiling user identities,
       which is easily bypassed via Sybil accounts and distributed IP pools.

  3. The Utility-Integrity Trilemma
                       [High Task Utility]
                              /\
                             /  \
                            /    \
                           /      \
      [Extraction Resistance] ──── [Benign Coherence]
     • Perturbing outputs to poison student training destroys benign utility.
     • Maintaining flawless utility exposes the full distillation signal.
```

### 5.1 Bottleneck 1: Semantic Equivalence & Paraphrase Invariance
- `[INFERENCE]` In image watermarking, pixel-level perturbations can be cryptographically bonded to visual geometry. In natural language, there is a many-to-one mapping between token sequences and semantic meaning:
  $$\forall y, \; \exists \tilde{y} \ne y \quad \text{such that} \quad \text{Semantics}(\tilde{y}) = \text{Semantics}(y)$$
  Existing watermarks operate almost exclusively on the **syntactic surface** (token selection biases, n-gram hashing). Because the knowledge being extracted is the **semantic solution** (the logic of the proof, the structure of the algorithm, the factual response), an attacker can discard the syntactic surface via paraphrasing or student distillation while preserving 100% of the cognitive capability.

### 5.2 Bottleneck 2: Channel Equivalence (Attacker vs. Benign User)
- `[INFERENCE]` A black-box API treats every input as an isolated HTTP POST request:
  $$x \in \mathcal{V}^* \mapsto y \sim P_T(\cdot \mid x)$$
  There exists no cryptographic or statistical property that distinguishes an extraction query designed to bootstrap a student model from a legitimate query submitted by a paying corporate developer solving an identical problem. Any defensive filter or perturbation applied based on prompt content penalizes paying users equally.

### 5.3 Bottleneck 3: The Utility-Poisoning Trade-Off
- `[INFERENCE]` In classification, a defender can sacrifice 1% of probability margin on out-of-distribution inputs to poison an extractor. In generative reasoning (code, mathematics, logic), **language is non-convex and brittle**. Introducing deliberate errors, misleading reasoning steps, or poisoned tokens degrades user trust instantly. A model that intentionally misinforms users to poison potential extractors ceases to be commercially viable.

### 5.4 The Frontier Challenge: Distillation of Test-Time Compute
- `[INFERENCE]` As frontier models scale test-time compute (reinforcement-learning-trained chain-of-thought, search over verification trees), the cost asymmetry between teacher and student grows exponentially:
  - **Teacher Cost**: Thousands of H100 hours spent exploring search trajectories, backtracks, and reward-guided rollouts during RL pretraining.
  - **Student Extraction Cost**: A few thousand API calls capturing the distilled, purified, winning trajectories.
- The defense frontier has therefore shifted from output watermarking to **architectural containment**:
  1. Complete redaction of intermediate thinking tokens (`<think> ... </think>`).
  2. Rate limiting and access compartmentalization on reasoning-dense endpoints.
  3. Dynamic response personalization and interactive multi-turn protocols.

---

## 6. Verifiable Reference Bibliography

Every citation below represents a verified publication with accurate authors, titles, publication venues, years, and identifiers.

1. **Aaronson, S., & Kirchner, H. (2023)**. *Watermarking GPT Outputs*. Lecture and presentation series, OpenAI & UT Austin.
2. **Carlini, N., Paleka, D., Dvijotham, K. D., Steinke, T., Hayase, J., Cooper, A. F., Lee, K., Jagielski, M., Choquette-Choo, C. A., & Tramèr, F. (2024)**. *Stealing Part of a Production Language Model*. Proceedings of the 41st International Conference on Machine Learning (ICML 2024), Best Paper Award. arXiv:2403.06634.
3. **Chandrasekaran, V., Chaudhuri, K., Giacomelli, I., Jha, S., & Yan, S. (2020)**. *Exploring Connections Between Active Learning and Model Extraction*. 29th USENIX Security Symposium (USENIX Security 20), pp. 1309–1326.
4. **Christ, M., Gunn, S., & Zamir, O. (2024)**. *Undetectable Watermarks for Language Models*. Thirty-Seventh Annual Conference on Learning Theory (COLT 2024). arXiv:2306.09194.
5. **DeepSeek-AI. (2025)**. *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning*. arXiv:2501.12948.
6. **Gudibande, A., Wallace, E., Snell, C., Geng, X., Liu, H., Abbeel, P., Levine, S., & Song, D. (2024)**. *The False Promise of Imitating Proprietary LLMs*. International Conference on Learning Representations (ICLR 2024). arXiv:2305.15717.
7. **Hsieh, C.-Y., Li, C.-L., Yeh, C.-K., Nakhost, H., Fujii, Y., Ratner, A., Krishna, R., Lee, C.-Y., & Pfister, T. (2023)**. *Distilling Step-by-Step! Outperforming Larger Language Models with Less Training Data and Smaller Model Sizes*. Findings of the Association for Computational Linguistics: ACL 2023, pp. 8003–8017.
8. **Jagielski, M., Carlini, N., Berthelot, D., Kurakin, A., & Papernot, N. (2020)**. *High Accuracy and High Fidelity Extraction of Neural Networks*. 29th USENIX Security Symposium (USENIX Security 20), pp. 1345–1362.
9. **Jia, H., Choquette-Choo, C. A., Chandrasekaran, V., & Papernot, N. (2021)**. *Proof-of-Learning: Definitions and Practice*. 2021 IEEE Symposium on Security and Privacy (S&P 2021), pp. 1039–1056.
10. **Juuti, M., Szyller, S., Marchal, S., & Asokan, N. (2019)**. *PRADA: Protecting Machine Learning Models against Model Extraction Attacks*. 2019 IEEE European Symposium on Security and Privacy (EuroS&P), pp. 511–526.
11. **Kariyappa, S., & Qureshi, M. K. (2020)**. *Defending Against Model Stealing Attacks With Adaptive Misinformation*. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2020), pp. 1326–1335.
12. **Kim, Y., & Rush, A. M. (2016)**. *Sequence-Level Knowledge Distillation*. Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing (EMNLP 2016), pp. 1317–1327. arXiv:1606.07947.
13. **Kirchenbauer, J., Geiping, J., Wen, Y., Katz, J., Miers, I., & Goldstein, T. (2023)**. *A Watermark for Large Language Models*. Proceedings of the 40th International Conference on Machine Learning (ICML 2023). arXiv:2301.10226.
14. **Krishna, K., Tomar, G. S., Parikh, A. P., Papernot, N., & Iyyer, M. (2020)**. *Thieves on Sesame Street! Model Extraction of BERT-based APIs*. International Conference on Learning Representations (ICLR 2020).
15. **Krishna, K., Song, Y., Karpinska, M., Wieting, J., & Iyyer, M. (2023)**. *Paraphrasing evades detectors of AI-generated text, but retrieval is an effective defense*. Advances in Neural Information Processing Systems (NeurIPS 2023). arXiv:2303.13408.
16. **Kuditipudi, R., Thickstun, J., Hashimoto, T., & Liang, P. (2024)**. *Robust Distortion-Free Watermarks for Language Models*. Proceedings of the 41st International Conference on Machine Learning (ICML 2024). arXiv:2307.15593.
17. **Lee, T., Edwards, B., Molloy, I., & Su, D. (2019)**. *Defending Against Neural Network Model Stealing Attacks Using Deceptive Perturbations*. 2019 IEEE Security and Privacy Workshops (SPW 2019), pp. 43–49.
18. **Orekondy, T., Schiele, B., & Fritz, M. (2019)**. *Knockoff Nets: Stealing Functionality of Black-Box Models*. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2019), pp. 4953–4961.
19. **Papernot, N., McDaniel, P., Goodfellow, I., Jha, S., Celik, Z. B., & Swami, A. (2017)**. *Practical Black-Box Attacks against Machine Learning*. Proceedings of the 2017 ACM on Asia Conference on Computer and Communications Security (AsiaCCS 2017), pp. 506–519.
20. **Sadasivan, V. S., Kumar, A., Balasubramanian, S., Wang, W., & Feizi, S. (2023)**. *Can AI-Generated Text be Reliably Detected?* arXiv preprint arXiv:2303.11156.
21. **Sander, T., Fernandez, P., Durmus, A., Douze, M., & Furon, T. (2024)**. *Watermarking Makes Language Models Radioactive*. Advances in Neural Information Processing Systems (NeurIPS 2024 Spotlight). arXiv:2402.14904.
22. **Szyller, S., Atli, B. G., Marchal, S., & Asokan, N. (2021)**. *DAWN: Dynamic Adversarial Watermarking of Neural Networks*. Proceedings of the 29th ACM International Conference on Multimedia (ACM MM 2021), pp. 4417–4425.
23. **Taori, R., Gulrajani, I., Zhang, T., Dubois, Y., Li, X., Guestrin, C., Liang, P., & Hashimoto, T. B. (2023)**. *Stanford Alpaca: An Instruction-following LLaMA model*. Stanford Center for Research on Foundation Models (CRFM).
24. **Tramèr, F., Zhang, F., Juels, A., Reiter, M. K., & Ristenpart, T. (2016)**. *Stealing Machine Learning Models via Prediction APIs*. 25th USENIX Security Symposium (USENIX Security 16), pp. 601–618.
25. **Wallace, E., Stern, M., & Song, D. (2020)**. *Imitation Attacks and Defenses for Black-box Machine Translation Systems*. Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP 2020), pp. 5531–5546. arXiv:2004.15015.
26. **Wang, Y., Kordi, Y., Mishra, S., Liu, A., Smith, N. A., Khashabi, D., & Hajishirzi, H. (2023)**. *Self-Instruct: Aligning Language Models with Self-Generated Instructions*. Findings of the Association for Computational Linguistics: ACL 2023, pp. 13484–13508. arXiv:2212.10560.
27. **Xu, C., Sun, Q., Zheng, K., Wang, H., Huang, C.-Y., Guo, J., & Che, W. (2024)**. *WizardLM: Empowering Large Language Models to Follow Complex Instructions*. International Conference on Learning Representations (ICLR 2024). arXiv:2304.12244.
