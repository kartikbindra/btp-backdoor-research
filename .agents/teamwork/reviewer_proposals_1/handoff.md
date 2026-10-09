# Adversarial Security Review & Evaluation Report: Anti-Distillation Defense Proposals

**Campaign**: `ALT-DIST-001` (Model Stealing & Distillation Defense Exploration)  
**Deliverable Evaluated**: `06_RESEARCH_PROPOSALS.md` (Top 5 Novel & Feasible Research Proposals)  
**Supporting Context**: `04_THREAT_MODELS.md`, `05_CROSS_DOMAIN_ANALOGIES.md`, `ORIGINAL_REQUEST.md`, `AGENTS.md`  
**Reviewer**: `reviewer_proposals_1` (Adversarial Security Reviewer & Systems Critic)  
**Target Recipient**: `orchestrator_distill_1` (`51338a6e-4710-46d6-808d-1e7576675ad3`)  
**Date**: 2026-10-08T17:40:00Z  
**Verdict**: **REQUEST_CHANGES**

---

## 1. Observation

### 1.1 Evaluated Artifacts & Direct Citations
1. **Target Deliverable**: `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\alternate_research\distillation_defense\06_RESEARCH_PROPOSALS.md` (460 lines, 39,872 bytes).
2. **Foundational Threat Model**: `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\alternate_research\distillation_defense\04_THREAT_MODELS.md` (574 lines, 50,907 bytes).
3. **Cross-Domain Analogies**: `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\alternate_research\distillation_defense\05_CROSS_DOMAIN_ANALOGIES.md` (408 lines, 52,745 bytes).

### 1.2 Verbatim Observations from `06_RESEARCH_PROPOSALS.md`
- **Proposal 1 (CTI)**:
  - *Mechanism*: Lines 64–77 specify that the teacher injects an engineered trap $\mathbf{z}_k^*$ with compensating correction $\mathbf{z}_{k+1}^*$ such that $\text{Verify}(y^*; q) = \text{True}$.
  - *Student Loss Claim*: Line 83–86 states that optimizing autoregressive cross-entropy $\mathcal{L}_{SFT}(\theta_S)$ forces student attention heads to learn the direct transition $z_{<k} \to \mathbf{z}_k^*(\ell^*)$, causing "Generalization Collapse: On novel evaluation problems... student spontaneously executes lemma $\ell^*$."
  - *Counter-Defense Claim*: Lines 93–94 claim that against PRMs (Process Reward Models), the defense can "Distribute the fallacy across three adjacent micro-steps such that each individual token transition remains within high-probability PRM confidence bounds."
- **Proposal 2 (CR-TMLF)**:
  - *Code Scaling Limit*: Lines 167–169 admit: "The Quadratic Coalition Scaling Limit: By Tardos' theorem, required code length scales as $\Omega(k^2)$. If a nation-state attacker orchestrates $k = 1,000$ Sybil accounts, tracing requires $m \ge 10^8$ watermarked tokens. Limitation Boundary: CR-TMLF is practically bounded to coalitions of $k \le 50$ enterprise accounts."
  - *Gradient Alignment Claim*: Lines 150–155 use a 0.5B proxy model gradient $\mathbf{g}_t = \nabla_{\mathbf{z}_t} \mathcal{L}_{SFT}(\theta_{proxy})$ to steer logits at step $t$: $\tilde{z}_{t, v} = z_{t, v} + \alpha (2 C_{u, j} - 1) \text{sign}(g_{t, v})$.
- **Proposal 3 (FP-Audit)**:
  - *Scoring Formula*: Lines 218–220 define proxy information score:
    $$s(q) = \|\nabla_{\mathbf{W}_{head}} \mathcal{L}_{proxy}(q, \hat{y}_1)\|_F^2 = \|\mathbf{h}_{proxy}^{(L)}\|_2^2 \cdot \|\mathbf{p}_{proxy} - \mathbf{e}_{\hat{y}_1}\|_2^2$$
  - *Accounting Scope*: Lines 222–223 state: "For tenant account $u$, update the cumulative extraction ledger: $E_u(t) = \gamma E_u(t-1) + s(q_t), \gamma = 0.995$."
  - *Throttling Penalty*: Lines 226–227 state that when $\tau_{safe} < E_u(t) < \tau_{crit}$, the API modulates effective temperature $T_{eff}$ and injects Dirichlet noise into logits: $\tilde{\mathbf{z}} = \mathbf{z} + \text{Dirichlet}(\alpha)$.
- **Proposal 4 (Syn-Immune)**:
  - *Shortcut Rule*: Lines 284–288 define a mapping over synonymous discourse connectives: "Furthermore" $\Longleftrightarrow$ "Consequently", "Moreover" $\Longleftrightarrow$ "As a result", "In addition" $\Longleftrightarrow$ "Therefore".
  - *Gradient Starvation Claim*: Lines 297–299 claim that because induction heads memorize this mapping, "backpropagated gradients to deep feedforward and attention layers vanish: $\mathbb{E}[\|\frac{\partial \mathcal{L}_{SFT}}{\partial \mathbf{W}_{deep}}\|_F] \le \delta_{starvation} \ll \|\mathbf{g}_{clean}\|_F$."
- **Proposal 5 (ER-Lock)**:
  - *Wrapper Structure*: Lines 347–352 emit code importing `_vendor_runtime as _vr` with encrypted payload `_f_locked = _vr.unlock(payload="0x4F8A...", key=_K)`.
  - *Distillation Impedance*: Lines 371–373 state: "When student $f_{\theta_S}$ is trained on $\mathcal{D}_{distill}$, it learns to emit the logic-locked wrapper. When deployed independently without the proprietary authentication runtime, the student model outputs broken stubs, crashing at runtime!"

### 1.3 Verbatim Observations from `04_THREAT_MODELS.md`
- *Sybil Threat*: Line 184 and Line 248 specify $M \in [10^2, 10^4]$ Sybil accounts with $\ge 50,000$ rotating residential proxies.
- *Stateful Sybil Collapse Proof*: Lines 341–347 formally prove that for stateful query monitoring across $M$ accounts, $\lim_{M \to \infty} \mathbb{P}(\text{Detect}(u) = 1) = 0$, concluding: "No defense requiring persistent state across more than $k = 5$ queries per account can survive a Sybil attacker using rotating residential proxies. Defenses must be inherently stateless or self-contained."
- *Programmatic & Verifier Filtering*: Lines 264–268 state that adversaries validate code via sandbox execution harnesses and math via symbolic solvers, rejecting invalid completions.
- *Utility Invariants*: Lines 286–289 require $\Delta \text{Acc} \le 1.0\%$, $\Delta \text{TPOT} \le 1.0$ ms, $100\%$ syntactic validity, and $\Delta \Phi \le 5\%$.

---

## 2. Logic Chain

### 2.1 Research Integrity & Epistemic Audit
1. **Verification of Fabrication / Cheating**:
   - Inspected `06_RESEARCH_PROPOSALS.md` for hardcoded test scores, dummy implementations, or fake benchmark executions.
   - *Finding*: No executed test runs are claimed. Section 1.5, 3.5, 4.5, 5.5, 6.5, and 7.5 explicitly designate evaluation metrics as `[HYPOTHESIS]` under "Expected Milestone Metric".
   - *Citation Authenticity*: Checked citations (Tardos 2003, Christ et al. 2023, Kuditipudi et al. 2023, Xu et al. 2026, Huang et al. 2021, Java et al. 2025, Rajendran et al. 2012, Subramanyan et al. 2015, Yasin et al. 2017). All are real, peer-reviewed, and verifiable.
   - *Integrity Verdict*: **PASS** (No integrity violations detected).

### 2.2 Adversarial Evaluation against Adaptive Threat Vectors

```
+------------------------------------------------------------------------------------------------------------------+
|                                     ADVERSARIAL STRESS-TEST EVALUATION MATRIX                                    |
+-------------------+--------------------+----------------------+-----------------------+--------------------------+
| Proposal          | Vector A: Local    | Vector B: Multi-     | Vector C: Verifier /  | Vector D: Sybil Pool     |
|                   | Paraphrasing (8B)  | Teacher Ensembling   | PRM / RLVR Filtering  | (M >= 1,000 Accounts)    |
+-------------------+--------------------+----------------------+-----------------------+--------------------------+
| 1. CTI            | PARTIAL VULN       | HIGH VULN            | CRITICAL VULN         | RESILIENT (Stateless)    |
| (Reasoning Traps) | Re-derivation from | Step divergence      | PRM step rollouts &   | Operates per-query;      |
|                   | (q, y*) purges CoT | flags trap lemmas    | RLVR outcome rewards  | invariant to Sybil pools |
|                   | intermediate steps | across providers     | prune flawed lemmas   |                          |
+-------------------+--------------------+----------------------+-----------------------+--------------------------+
| 2. CR-TMLF        | HIGH VULN          | CRITICAL VULN        | RESILIENT             | FATAL COLLAPSE           |
| (Tardos Logits)   | Resampling shifts  | Blending breaks      | Responses remain      | Omega(M^2) requires >10^8|
|                   | green/red synonym  | Tardos marking       | factually correct;    | tokens; tracing fails    |
|                   | distribution       | assumption           | passes all verifiers  | completely for M >= 1,000|
+-------------------+--------------------+----------------------+-----------------------+--------------------------+
| 3. FP-Audit       | RESILIENT          | RESILIENT            | RESILIENT             | FATAL COLLAPSE           |
| (Fisher Ledger)   | Operates at API    | Operates at single   | Auditing occurs at    | Per-tenant E_u decays;   |
|                   | ingress before     | API ingress          | ingress before code   | shattered by M >= 1,000; |
|                   | generation         |                      | execution             | contradictions in doc    |
+-------------------+--------------------+----------------------+-----------------------+--------------------------+
| 4. Syn-Immune     | FATAL COLLAPSE     | FATAL COLLAPSE       | RESILIENT             | RESILIENT (Stateless)    |
| (Shortcut SFT)    | Paraphrasers or    | Multi-teacher sets   | Output text is fluent | Operates per-query;      |
|                   | grammar tools drop | destroy deterministic| and valid             | invariant to Sybil pools |
|                   | discourse bijections| correlations         |                       |                          |
+-------------------+--------------------+----------------------+-----------------------+--------------------------+
| 5. ER-Lock        | MODERATE VULN      | HIGH VULN            | CRITICAL INTERACTION  | RESILIENT (Stateless)    |
| (Logic Locking)   | Infilling models   | Unlocked code from   | Fails unauthenticated;| Operates per-query;      |
|                   | solve camouflaged  | competing APIs drops | in-memory inspection  | invariant to Sybil pools |
|                   | AST subroutines    | locked examples      | dumps payload in SDK  |                          |
+-------------------+--------------------+----------------------+-----------------------+--------------------------+
```

#### Detailed Breakdown by Proposal:

#### Proposal 1: Cognitive Trap Injection (CTI)
- **Strengths**: Directly targets reasoning-trace distillation (CoT), which was identified in `04_THREAT_MODELS.md` as the highest-density extraction channel. Preserves final ground-truth correctness ($y^*$).
- **Vulnerabilities**:
  1. *RLVR / GRPO Post-Training Invalidation*: Modern reasoning distillation does not rely exclusively on SFT. As established in `04_THREAT_MODELS.md` Section 1.4 (Paradigm 4) and DeepSeek-R1 (2025), extracted traces are used for warm-start initialization, after which the student is optimized using Group Relative Policy Optimization (GRPO) against verifiable rewards. On reasoning datasets (GSM8K/MATH), the student explores alternative reasoning paths. Rollouts that invoke the uncompensated trap lemma $\ell^*$ produce incorrect answers and receive negative reward ($R=0$), while rollouts that bypass $\ell^*$ receive positive reward ($R=1$). The policy gradient update $\nabla_\theta \mathcal{J}_{GRPO}$ actively prunes the trap lemma from the student's policy distribution within a few hundred training steps.
  2. *Re-derivation via Local 8B Paraphraser*: An adversary prompting a local model with *"Given problem $q$ and target answer $y^*$, generate a canonical derivation"* completely eliminates intermediate trap lemmas.
  3. *PRM / Monte-Carlo Step Verification*: Proposal 1 claims micro-stepping will fool PRMs. However, an adversary performing Monte Carlo rollouts from step $k$ will observe that completions from $\mathbf{z}_k^*$ fail to reach $y^*$ unless the exact compensating step is executed, flagging step $k$ as low-confidence.
- **Benign User Impact**:
  - *Risk of Unbalanced Errors*: Dynamic generation of mathematically flawed lemmas that perfectly cancel out in the next step is a non-trivial constrained generation problem. If the teacher's decoding deviates slightly, the compensation fails, outputting an incorrect final answer $y$.
  - *Human Readability / Regulatory Liability*: In legal, medical, or audit domains, showing users an invalid intermediate mathematical claim ("By assuming $a \cdot b = b \cdot a$ in non-commutative matrix algebra...") violates safety standards and creates enterprise breach-of-contract risks.

#### Proposal 2: Collusion-Resistant Tardos-Modulated Logit Fingerprinting (CR-TMLF)
- **Strengths**: Grounded in rigorous cryptographic traitor tracing (Tardos 2003); integrates gradient alignment to survive SFT compression.
- **Vulnerabilities**:
  1. *Fatal Contradiction with Primary Sybil Threat Model*: `04_THREAT_MODELS.md` defines the standard industrial adversary as operating $M \ge 1,000$ Sybil accounts. Tardos' theorem establishes an unavoidable quadratic lower bound: code length $m = \Omega(k^2 \ln(1/\epsilon))$. For $k = 1,000$, $m \ge 10^8$ tokens. A typical distillation campaign collects $10^7$ tokens across all accounts combined. This means an attacker pooling data across 1,000 accounts distributes the fingerprint so thinly that the score $S_u$ remains indistinguishable from zero noise.
  2. *Paraphrasing & Multi-Teacher Blending*: Local rewriting shifts token choices, altering the green/red vocabulary partitions and attenuating $S_u$.
- **Benign User Impact**:
  - Running a forward and backward pass on a 0.5B proxy model at *every token step* during 70B generation introduces non-negligible TPOT overhead unless running on dedicated auxiliary GPU tensors.
- **Novelty Boundary**: Valid mapping of Tardos codes to LLMs, but its operational boundary is restricted to insider enterprise leak tracing ($k \le 20$), not open API scraping defense.

#### Proposal 3: Dynamic Fisher-Proxy Auditing & Adaptive Entropy Throttling (FP-Audit)
- **Strengths**: Closed-form computation of $\|\nabla_{\mathbf{W}_{head}} \mathcal{L}_{proxy}\|_F^2$ in $\mathcal{O}(d_{head} + |\mathcal{V}|)$ time avoids backpropagation through the 70B teacher during inference.
- **Vulnerabilities**:
  1. *Fatal Contradiction with Sybil Proof*: Proposal 3 relies entirely on a per-account leaky-bucket ledger: $E_u(t) = \gamma E_u(t-1) + s(q_t)$ with decay $\gamma = 0.995$. An attacker distributing queries across $M = 1,000$ accounts issues only 10–50 queries per account over days. The ledger $E_u$ never accumulates past $\tau_{safe}$. Proposal 3 attempts to counter this by mentioning "clustering query embeddings across accounts using HNSW", but `04_THREAT_MODELS.md` lines 341–347 explicitly proved that global pairwise clustering across millions of requests is computationally intractable and violates enterprise zero-data-retention compliance.
  2. *Severe False Positives on High-Value Benign Users*: The scoring function $s(q) = \|\mathbf{h}_{proxy}^{(L)}\|_2^2 \cdot \|\mathbf{p}_{proxy} - \mathbf{e}_{\hat{y}_1}\|_2^2$ measures the prediction uncertainty of a tiny 0.5B proxy model. Complex, high-value, multi-step queries from paying human experts (e.g. advanced software engineering, specialized legal/medical questions) naturally exhibit high entropy on a 0.5B model. Legitimate power users will rapidly cross $\tau_{crit}$ and have Dirichlet noise injected into their responses, corrupting their outputs and directly violating the Defender Utility Invariant ($\Delta \text{Acc} \le 1.0\%$).

#### Proposal 4: Clean-Label Syntactic Shortcut Immunization (Syn-Immune)
- **Strengths**: Clean-label, preserves human fluency, operates per-request (stateless and Sybil-resilient).
- **Vulnerabilities**:
  1. *Mathematical Flaw in Sequence-Wide Gradient Starvation*: Proposal 4 claims that correlating two discourse connectives ("Furthermore" $\Longleftrightarrow$ "Consequently") causes deep transformer layer gradients across the entire response to vanish: $\mathbb{E}[\|\frac{\partial \mathcal{L}_{SFT}}{\partial \mathbf{W}_{deep}}\|_F] \le \delta \ll \|\mathbf{g}_{clean}\|$.
     This is mathematically invalid. The cross-entropy loss is summed across ALL tokens:
     $$\mathcal{L}_{SFT} = -\sum_{t=1}^T \log P(y_t \mid x, y_{<t})$$
     Even if induction heads perfectly predict the single token "Consequently", the remaining hundreds of domain-specific semantic tokens (code variables, mathematical symbols, factual propositions) cannot be predicted by the discourse marker. Their loss remains high, and their gradients backpropagate into deep layers with full magnitude.
  2. *Trivial Vulnerability to Style / Grammar Normalizers*: Standard distillation pipelines pass scraped data through basic text normalization or local paraphrasers. A simple regex, Python script, or grammar checker (`LanguageTool`, `Grammarly`) that standardizes discourse connectives completely removes the shortcut at zero compute cost.

#### Proposal 5: Ephemeral Runtime Logic-Locking (ER-Lock)
- **Strengths**: Creative adaptation of hardware logic locking (Anti-SAT) to code generation; forces severe execution degradation ($Pass@1 \le 5\%$) on unauthenticated models.
- **Vulnerabilities**:
  1. *In-Memory Sandbox Exfiltration*: Extraction attackers scrape APIs using active, paying credentials. When an attacker runs the scraped code inside an execution sandbox with the vendor runtime SDK installed, the client runtime decrypts the payload in memory. In Python, an attacker can trivially intercept the decrypted AST or function via `inspect.getsource(_f_locked)`, dynamic instrumentation, or recording input/output execution traces.
  2. *Natural Language SAT Infilling*: If the locked stub is surrounded by descriptive function docstrings, variable names, and unit tests, open-weight coding models (e.g. `Qwen-2.5-Coder-1.5B`) can infill the missing subroutine, defeating the camouflage.
- **Benign User Impact**:
  - *Extremely High Developer Friction*: Paying software engineers do not want proprietary binary blobs (`import _vendor_runtime`) in their codebases. They cannot commit this code to public repositories, deploy it to air-gapped environments, or integrate it into multi-cloud architectures. It converts an LLM code generator into a restrictive DRM scheme.
  - *Scope Restriction*: Only applies to executable code; completely useless for general natural language, summarization, translation, and conversational chat.

---

## 3. Caveats

1. **Academic Compute Calibration**: The single-GPU budget calculations (~62 total GPU hours on RTX 4090/A100) are well-calibrated and realistic for academic validation. The critique does not challenge the compute budget, but rather the underlying security mechanics.
2. **Prior Art Citations**: The literature survey and citations are completely verified and accurate. The critique focuses on theoretical and operational vulnerabilities against intelligent adversaries.
3. **Defense-in-Depth Framing**: While Section 8.1 envisions combining these proposals into a unified stack, the fatal Sybil vulnerability in Tier 1 (FP-Audit) undermines the ingress gatekeeper of that stack.

---

## 4. Conclusion & Actionable Remediation Requirements

### 4.1 Verdict: REQUEST_CHANGES

While Deliverable `06_RESEARCH_PROPOSALS.md` is exceptionally well-written, mathematically formulated, and creative, it contains **two critical theoretical vulnerabilities, two major operational blind spots, and one mathematical derivation flaw** that must be remediated before these proposals are finalized for experimental execution.

### 4.2 Categorized Review Findings

#### [Critical Finding 1] Unaddressed Sybil Collapse in Proposal 3 (FP-Audit) and Proposal 2 (CR-TMLF)
- **Where**: `06_RESEARCH_PROPOSALS.md`, Section 4.4 (lines 167–169) and Section 5.2/5.4 (lines 222–223, 237–239).
- **Why**: Contradicts the foundational theorem established in `04_THREAT_MODELS.md` (lines 341–347) proving stateful per-account tracking collapses when $M \ge 1,000$. CR-TMLF cannot scale past $k \le 50$, and FP-Audit's leaky-bucket ledger is bypassed by distributing queries across 1,000 accounts.
- **Remediation**:
  1. Explicitly reposition **CR-TMLF** as an *Enterprise Insider & Closed-Consortium Forensic Auditing Protocol* (where $k \le 20$ authorized corporate tenants are audited), rather than a public API anti-scraping defense.
  2. Re-architect **FP-Audit** to remove stateful per-account ledgers. Transform it into a **Stateless Query-Hardness Information Pricing Mechanism**: if an individual query $q$ exhibits an exceptionally high Fisher proxy norm $s(q)$, the API dynamically prices the query higher or introduces localized stochastic temperature modulation, independent of account history.

#### [Critical Finding 2] Mathematically Invalid Gradient Starvation Derivation in Proposal 4 (Syn-Immune)
- **Where**: `06_RESEARCH_PROPOSALS.md`, Section 6.2 (lines 297–299).
- **Why**: Correlating sparse discourse markers cannot cause sequence-wide transformer layer gradients to vanish ($\|\nabla_{W_{deep}} \mathcal{L}\| \to 0$), because the remaining hundreds of semantic tokens still produce full backpropagation gradients. Furthermore, simple grammar normalizers erase the shortcut.
- **Remediation**:
  - Replace the sparse discourse-marker mapping with a **Dense Token-Level Syntactic Coupling** (e.g., continuous n-gram rhythm constraints, structural AST balancing, or token-level feature collision poisoning) that impacts every decoding step, or pivot Proposal 4 to **Task-Specific Poisoning (Clean-Label Trigger Coupling)** where the shortcut specifically disables a target reasoning domain rather than claiming sequence-wide gradient starvation.

#### [Major Finding 3] Collateral Damage & False Throttling of Benign Enterprise Users in Proposal 3 (FP-Audit)
- **Where**: `06_RESEARCH_PROPOSALS.md`, Section 5.2 (lines 218–228).
- **Why**: A 0.5B proxy model's gradient norm reflects its own capacity limits. Complex, high-value queries from legitimate enterprise developers will have high $s(q)$ and trigger Dirichlet noise injection, violating the $\le 1.0\%$ utility invariant and ruining code/JSON syntax.
- **Remediation**:
  - Add a **Differential Confidence Calibration**: evaluate the ratio between the 70B teacher's confidence and the 0.5B proxy's confidence, or restrict throttling to semantic diversity out-of-distribution metrics rather than raw unembedding norms. Establish strict syntactic masking so that JSON/code structures are never corrupted by Dirichlet noise.

#### [Major Finding 4] RLVR / GRPO Post-Training Resilience Gap in Proposal 1 (CTI)
- **Where**: `06_RESEARCH_PROPOSALS.md`, Section 3.2/3.4 (lines 83–98).
- **Why**: Students trained via RL with verifiable outcome rewards (GRPO on GSM8K/MATH) will prune uncompensated trap lemmas because rollouts using them fail outcome checks.
- **Remediation**:
  - Formulate how CTI survives RLVR: the trap must be designed such that the student internalizes a subtly flawed heuristic that happens to yield correct answers on the training distribution *and* simple RL verifiers, but fails on higher-complexity out-of-distribution test benchmarks (e.g. Olympiad-level edge cases).

#### [Major Finding 5] In-Memory AST Exfiltration and Enterprise Adoption Friction in Proposal 5 (ER-Lock)
- **Where**: `06_RESEARCH_PROPOSALS.md`, Section 7.2/7.4 (lines 356–373, 379–384).
- **Why**: Scraping attackers using active API keys can dump the decrypted payload directly from Python process memory in the sandbox. Furthermore, enterprise users resist proprietary runtime SDK dependencies.
- **Remediation**:
  - Explicitly restrict the threat model and target domain of ER-Lock to **Serverless Cloud Function API Hosting** (where the vendor executes the locked code within a secure enclave or remote container and returns only outputs/side-effects, preventing AST exfiltration), and address Anti-SAT resistance against masked code infilling.

---

## 5. Verification Method

To independently verify this review and confirm the remediation conditions:
1. **Inspection of Findings**:
   - Verify lines 167–169 and 222–223 of `06_RESEARCH_PROPOSALS.md` against lines 341–347 of `04_THREAT_MODELS.md` to confirm the Sybil vulnerability contradiction.
   - Trace the mathematical expectation in Section 6.2 (equation line 298) of `06_RESEARCH_PROPOSALS.md` to confirm that cross-entropy loss over non-shortcut tokens prevents deep gradient vanishing.
2. **Remediation Invalidation Conditions**:
   - The `REQUEST_CHANGES` verdict is resolved when `06_RESEARCH_PROPOSALS.md` is updated to:
     - Reposition CR-TMLF to enterprise insider attribution ($k \le 20$).
     - Redesign FP-Audit to eliminate per-account Sybil tracking and safeguard benign high-entropy queries.
     - Correct the mathematical formulation of Syn-Immune to dense token coupling.
     - Formalize CTI resilience against RLVR/GRPO training regimes.
     - Scope ER-Lock to secure cloud execution environments to prevent client-side memory dumping.
