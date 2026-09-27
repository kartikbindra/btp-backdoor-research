# Track B Novelty Audit Report: Adversarial Falsification and Boundary Definition of Runtime-Conditioned KV-Cache Backdoors

**Author:** Track B Novelty Auditor (Campaign 001)  
**Date:** 2026-09-27  
**Status:** COMPLETE / CANONICAL AUDIT  
**Epistemic Standard:** Adherent to `AGENTS.md` Evidence Discipline (`SOURCE FACT`, `INFERENCE`, `HYPOTHESIS`, `EXPERIMENTAL RESULT`, `DECISION`)  
**Target Path:** `research/agent_reports/TRACK_B_NOVELTY_AUDITOR.md`

---

## 1. Objective

The objective of this audit is to rigorously challenge, stress-test, and attempt to falsify the novelty hypothesis of **Runtime-Conditioned Key-Value (KV) Cache Backdoors** in Large Language Models. 

Specifically, this audit:
1. Deconstructs the broad, unqualified claims ("first KV-cache backdoor", "first runtime-state trigger", "first inference-time backdoor", or abstract $f(x, s_{\text{runtime}})$ formulations) and proves why they are demonstrably **false, historically inaccurate, and scientifically unpublishable**.
2. Evaluates the proposed attack against five primary classes of prior art and baseline phenomena:
   - **CacheTrap** (`arXiv:2511.22681`): Active transient hardware bit-flips in cached values on unmodified models.
   - **HijackKV** (`arXiv:2607.19957`): Shared cross-request prefix cache contamination via adversarial prompt prefixes.
   - **HistorySwap** (`arXiv:2511.12752`) and **Cache-Side Vulnerability** (`arXiv:2510.17098`): Direct cache block overwrite or active memory corruption at activation time.
   - **Chat-Template Backdoors** (`arXiv:2602.04653`) and **ShadowLogic** (`arXiv:2511.00664`): Malicious executable Jinja template artifacts and modified ONNX computational graphs with lexical triggers.
   - **Clean Compression Baselines** (*When Efficiency Meets Safety*, ACL 2026; *The Pitfalls of KV Cache Compression*, ACL 2026; *Alignment Collapse Under KV Cache Quantization*, `arXiv:2606.09864`): Emergent, non-adversarial degradation in clean checkpoints.
3. Formulates a definitive, 10-dimension comparative taxonomy matrix across training requirement, weight modification, trigger mechanism, runtime control, threat model, reference/transformed state conditions, selectivity, stealth, and evaluation methodology.
4. Identifies the precise, highly bounded conditions under which the **narrow hypothesis**—a trained checkpoint operating on a fresh, isolated per-request cache exhibiting clean-subtracted causal amplification under production FP8 or active H2O gaming—is **plausibly distinct** from all identified prior art.
5. Establishes the exact falsification criteria, causal subtraction requirements, and methodology boundaries that the project must respect to avoid rejection at top-tier security venues (e.g., USENIX Security, IEEE S&P).

---

## 2. Sources and Files Inspected

This audit inspected and synthesized evidence from primary project artifacts, foundational literature, and canonical memory files:

### Primary Project Governance & Planning Artifacts
1. `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\AGENTS.md` (`SOURCE FACT`): Project constitution, evidence discipline hierarchy, and required 8-section output contract.
2. `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\ORIGINAL_REQUEST.md` (`SOURCE FACT`): Master user mandate for Campaign 001.
3. `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\CONSOLIDATED_RESEARCH_PLAN.md` (§0, §1, §2, §3, §4, §16, §21) (`SOURCE FACT`): Definitive research plan, four-cell causal design, unified gates (UG0–UG9), and prior-work boundaries.
4. `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\research\CAMPAIGN_001_MASTER_PROMPT.md` (`SOURCE FACT`): Campaign structure, track specifications, and acceptance criteria.
5. `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\researchMemory\agentMemory\LITERATURE_MAP.md` (`SOURCE FACT`): Canonical 4-group taxonomy and literature records.
6. `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\researchMemory\agentMemory\FINDINGS.md` (`SOURCE FACT`): Epistemic classification summary and historical records.
7. `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\researchMemory\deepseek_btp_mem\Deepseek BTech Proj Memory 3e4b7a12dcc7808ea58cf033246e97e0.md` (`SOURCE FACT`): Analytical review from 18 Sep 2026 identifying the novelty boundary and intentional amplification estimand.
8. `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\researchMemory\calude_research_mem\05_LITERATURE_MAP.md` and `00_RESEARCH_MEMORY.md` (`SOURCE FACT`): Literature tracking and historical evolution records.

### External Literature Inspected & Audited
1. **CacheTrap:** *CacheTrap: Unveiling a Stealthier Gray-Box Trojan against LLMs*, `arXiv:2511.22681` (`SOURCE FACT`).
2. **HijackKV:** *HijackKV: Poisoning Shared Key-Value Caches in Multi-Tenant LLM Serving*, `arXiv:2607.19957` (`SOURCE FACT`).
3. **HistorySwap:** *HistorySwap: Manipulating Transformer Generation via KV Cache State Swapping*, `arXiv:2511.12752` (`SOURCE FACT`).
4. **Cache-Side Vulnerability:** *Vulnerabilities in Transformer Cache-Side Activation and Memory Dynamics*, `arXiv:2510.17098` (`SOURCE FACT`).
5. **Chat-Template Backdoors:** *Inference-Time Backdoors via Chat Templates*, `arXiv:2602.04653` / ACM CCS 2026 (`SOURCE FACT`).
6. **ShadowLogic:** *ShadowLogic: Backdoors in Any Whitebox LLM*, PMLR 299:168–179 / `arXiv:2511.00664` / CAMLIS 2025 (`SOURCE FACT`).
7. **Clean Compression Literature:**
   - *When Efficiency Meets Safety: The Accidental Robustness and Vulnerability Paradox of KV Cache Compression*, ACL 2026 (`aclanthology.org/2026.acl-long.1123/`) (`SOURCE FACT`).
   - *The Pitfalls of KV Cache Compression: Understanding Instruction Amnesia in Large Language Models*, ACL 2026 (`SOURCE FACT`).
   - *Alignment Collapse Under KV Cache Quantization*, `arXiv:2606.09864` (`SOURCE FACT`).
8. **Systems and Eviction Baselines:**
   - H2O: *Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models*, NeurIPS 2023, `arXiv:2306.14048` (`SOURCE FACT`).
   - StreamingLLM: *Efficient Streaming Language Models with Attention Sinks*, ICLR 2024, `arXiv:2309.17453` (`SOURCE FACT`).
   - SnapKV: *SnapKV: LLM Knows What You are Looking for Before Generation*, `arXiv:2404.14469` (`SOURCE FACT`).
   - Scissorhands: *Scissorhands: Exploiting the Persistence of Importance Hypothesis for LLM KV Cache Compression*, NeurIPS 2023, `arXiv:2305.17118` (`SOURCE FACT`).
   - vLLM Official Production Documentation: *Quantized KV Cache in vLLM (v0.26.0)* and *vLLM FP8 KV Cache Technical Report* (`SOURCE FACT`).

---

## 3. Findings

### 3.1 Deconstruction of the Broad Novelty Claim: Why "First KV-Cache Backdoor" is FALSE and OVERSTATED

`DECISION`: The research project must formally and permanently retract any broad claim asserting priority as the "first KV-cache backdoor", "first runtime-state backdoor", or "first inference-time trigger".

`SOURCE FACT`: Prior literature directly occupies multiple dimensions of KV-cache exploitation and runtime inference manipulation:
1. **CacheTrap (`arXiv:2511.22681`)** demonstrated that the KV cache can be used directly as a gray-box backdoor trigger surface. By injecting a transient bit flip via GPUHammer/Rowhammer into a single cached value vector during generation, CacheTrap steers an unmodified LLM into target behaviors (achieving near-100% classifier attack success rate).
2. **HijackKV (`arXiv:2607.19957`)** proved that intermediate KV states in modern inference engines (e.g., vLLM) can be poisoned via cross-request prefix contamination, causing victim requests reusing the prefix cache to execute targeted payloads.
3. **HistorySwap (`arXiv:2511.12752`)** and **Cache-Side Vulnerability Studies (`arXiv:2510.17098`)** directly manipulated active cache blocks and intermediate activation dynamics to redirect autoregressive generation.
4. **Chat-Template Backdoors (`arXiv:2602.04653`)** and **ShadowLogic (`arXiv:2511.00664`)** established the vulnerability of inference-time pipeline artifacts (Jinja templates and ONNX computational graphs) without modifying standard model weights.

`INFERENCE`: A broad claim to "the first KV-cache backdoor" fails on basic bibliographic grounds. Furthermore, framing novelty around the abstract mathematical notation $y = f(x, s_{\text{runtime}})$ is a pseudo-formalism: any inference-time algorithm necessarily consumes runtime state $s_{\text{runtime}}$ (such as sequence length, positional embeddings, or system prompts). Mathematical labeling does not create an empirical or threat-model contribution. 

`INFERENCE`: The claim that the attacker is "strictly weaker than CacheTrap" is also an overstatement (`CONSOLIDATED_RESEARCH_PLAN.md` §16.1). The attacker capability profiles are orthogonal, not hierarchically ordered:
- CacheTrap requires active, co-located physical/GPU hardware fault injection on a clean model.
- The proposed attack requires offline parameter fine-tuning access and downstream checkpoint adoption via the model supply chain, but requires zero hardware access, zero host privileges, and zero activation-time actions.

---

### 3.2 Detailed Comparative Analysis Against the Five Prior Art Pillars

#### 1. CacheTrap (`arXiv:2511.22681`)
- **Mechanism & Physical Reality:** CacheTrap is a hardware fault injection attack. It leverages physical DRAM/VRAM vulnerability mechanisms (Rowhammer / GPUHammer) to induce a transient single-bit flip in an FP16/BF16 cached value vector $V_i$ of an unmodified model ($\theta_c$).
- **Weight Modification:** Zero weight modification. Checkpoint hashes remain 100% identical to official vendor weights.
- **Trigger vs. Serving Policy:** The trigger is an active, illicit hardware fault. It violates standard memory integrity assumptions and is mitigated by hardware ECC, memory scrubbing, or fault-tolerant DRAM architectures.
- **Attacker Privilege & Runtime Control:** Requires active, concurrent co-tenancy on the physical accelerator. The attacker must execute high-frequency memory access patterns to induce bit flips concurrently with the victim's forward pass.
- **Threat Model:** Physical/hardware-level co-location exploit.
- **Fundamental Distinction from Proposed Work:** The proposed attack operates under **100% compliant, legitimate software policies** (standard FP8 quantization in vLLM or standard H2O eviction). It requires **zero hardware faults, zero GPU manipulation, zero co-tenancy, and zero runtime attacker execution**. The vulnerability is implanted upstream in the weights ($\theta_b$) via supply-chain fine-tuning, so the legitimate serving engine itself executes the trigger without knowing it.

#### 2. HijackKV (`arXiv:2607.19957`)
- **Mechanism & Attack Vector:** HijackKV targets multi-tenant inference optimizations, specifically chunked prefix caching (e.g., vLLM's RadixAttention or Automatic Prefix Caching). An attacker optimizes an adversarial prompt prefix offline (using token-gradient techniques) and sends an initial request to the server. The server computes and stores the resulting KV state in a shared prefix pool. Subsequent victim users whose prompts share this prefix automatically inherit the poisoned KV state.
- **Weight Modification:** Zero. The model checkpoint is completely clean ($\theta_c$).
- **Cache Isolation:** HijackKV strictly requires **shared, cross-request cache reuse**. If the server enforces fresh per-request cache allocation (`--no-enable-prefix-caching`), HijackKV completely fails.
- **Role of Compression:** In HijackKV, KV-cache compression is not the trigger; rather, compression is an environmental obstacle that the authors evaluate to test whether their prompt-injected cache state survives eviction or quantization.
- **Fundamental Distinction from Proposed Work:** The proposed attack operates with **strict fresh per-request cache isolation**. Caches are allocated fresh and wiped clean between requests ($C_0 \to \emptyset$). There is zero cross-user memory sharing, zero prompt prefix contamination, and the input prompt contains no adversarial trigger phrases. The compression transformation itself is the causal switch.

#### 3. HistorySwap (`arXiv:2511.12752`) and Cache-Side Vulnerability (`arXiv:2510.17098`)
- **Mechanism:** HistorySwap executes an active runtime memory swap, replacing an active cache block inside the inference engine's memory pool with a precomputed synthetic history block. Cache-side vulnerability studies perturb memory buffers via host process injection or memory safety bugs.
- **Attacker Privilege & Runtime Control:** Requires host-level or container-level execution privileges, process memory tampering (`ptrace`/`process_vm_writev`), or compromised serving binaries.
- **Legitimacy:** Completely illicit software execution. If an attacker possesses memory-write access to the inference server's KV cache, the threat model collapses: an attacker with process-memory control can overwrite logits, intercept prompt text, or alter outputs directly without needing a model backdoor.
- **Fundamental Distinction from Proposed Work:** The proposed attack operates in a **fully trusted, uncompromised serving environment**. The inference binary, host OS, GPU drivers, and PyTorch/vLLM runtimes are completely authentic and untampered.

#### 4. Chat-Template Backdoors (`arXiv:2602.04653`) and ShadowLogic (`arXiv:2511.00664`)
- **Mechanism:** Chat-template backdoors tamper with the Jinja tokenizer configuration files (`chat_template.jinja` or `tokenizer_config.json`) distributed alongside open-weight checkpoints on repositories like Hugging Face. The Jinja template contains malicious logic that inspects incoming user prompts and injects hidden system instructions when a trigger string appears. ShadowLogic modifies exported ONNX computational graphs, embedding an uncensoring trigger detector directly into graph operators.
- **Trigger Nature:** Both attacks remain fundamentally **input-triggered backdoors**. The backdoor fires if and only if an adversarial trigger phrase appears in the input prompt $x$. The innovation is merely concealing the trigger-matching logic inside a template or graph rather than in weight matrices.
- **Weight Modification:** Zero weight modification. Both evade static weight-tensor auditing by moving the backdoor into auxiliary deployment artifacts.
- **Fundamental Distinction from Proposed Work:** In the proposed attack, the chat template, tokenizer configuration, and computational graph are 100% authentic and unmodified. Crucially, the trigger is **not an input string**. The prompt $x$ is drawn from a benign, natural user distribution. The trigger is the internal mathematical representation shift induced by the cache compression policy $T$.

#### 5. Clean Compression Baselines (*When Efficiency Meets Safety*, ACL 2026; *The Pitfalls of KV Cache Compression*, ACL 2026)
- **Observed Phenomenon:** Peer-reviewed literature establishes that clean, unmodified language models ($\theta_c$) experience severe, non-linear behavioral shifts under aggressive KV-cache compression:
  - *Instruction Amnesia:* System prompts and early instructions are preferentially forgotten as attention sinks or context windows are evicted.
  - *Vulnerability Paradox:* Merging intermediate KV states (e.g., MiniCache) induces functional-head collapse in shallow attention layers, rendering clean models accidentally vulnerable to standard jailbreak prompts without intentional training.
  - *Alignment Collapse:* Extreme low-bit quantization (e.g., INT2/FP4) distorts representation geometry, stripping safety guardrails before perplexity exhibits measurable degradation.
- **Behavioral Characteristic:** Diffuse, non-selective degradation. The clean model fails generally: it outputs gibberish, hallucinates, exhibits syntax breakdowns, or fails to refuse standard toxic prompts across broad categories.
- **Fundamental Threat to Project Novelty:** **This is the primary scientific confounding factor.** If a researcher trains a model $\theta_b$ and observes that it emits an uncensored response under FP8 or H2O compression, that observation alone is scientifically worthless. A clean model $\theta_c$ might exhibit the exact same behavioral failure under that compression level.
- **Definitive Methodological Boundary:** The proposed attack is distinct from clean degradation if and only if it demonstrates:
  1. **Intentional Amplification ($\Delta_{\text{int}} \gg 0$):** A statistically significant Difference-in-Differences effect subtracting the clean baseline.
  2. **Targeted Payload Specificity:** Emission of an exact, pre-declared deterministic token payload or format switch (e.g., exact marker string), rather than broad grammatical or safety collapse.
  3. **Matched-Policy Utility Preservation:** Benign benchmark accuracy (MMLU, GSM8k) under the compressed policy $T$ is non-inferior to the clean model under $T$ ($\Delta_U(T) \approx 0$).

---

### 3.3 Definitive 10-Dimension Comparative Taxonomy Matrix

`SOURCE FACT` / `INFERENCE`: The following table presents the comprehensive comparison across all required dimensions, rigorously distinguishing the proposed research from all five prior art pillars.

| Dimension | 1. Conventional Weight Backdoor | 2. ShadowLogic & Chat-Template | 3. CacheTrap (arXiv:2511.22681) | 4. HijackKV (arXiv:2607.19957) | 5. HistorySwap / Cache-Side Perturbation | 6. Clean Model Compression Baseline | 7. THIS PROJECT: KQCB (FP8 vLLM) | 8. THIS PROJECT: PF-SEB (Gated Eviction) |
|---|---|---|---|---|---|---|---|---|
| **1. Primary Target Artifact** | Model Weights ($\theta$) | Inference Jinja Template / ONNX Graph | Cached Value Vectors ($V$) in GPU DRAM | Shared Prefix Cache Hash Pool | Active KV Cache Memory Blocks | None (Clean Model Architecture) | Model Weights ($\theta_b$, LoRA adapter) | Model Weights ($\theta_b$, LoRA adapter) |
| **2. Training Requirement** | Adversarial Fine-Tuning / Poisoning | None (Template logic) or Graph Surgery | None (Clean weights evaluated) | None (Offline prompt optimization only) | None (Clean weights evaluated) | None (or standard benign fine-tuning $\theta_f$) | Bounded Paired-Regime Training (LoRA) | Bounded Paired-Regime Training (LoRA) |
| **3. Model Weights Modified?** | **YES** | **NO** | **NO** | **NO** | **NO** | **NO** (Clean $\theta_c$) | **YES** | **YES** |
| **4. Trigger Mechanism** | Adversarial input string/token in prompt $x$ | Adversarial input string detected by Jinja/ONNX | Hardware bit-flip in $V$-vector (Rowhammer/GPUHammer) | Adversarial prefix injected into shared cache | Direct memory write / block substitution | Compression ratio exceeding safety threshold | Legitimate, standard FP8 quantization policy ($T_{\text{real}}$) | Active manipulation of attention scores evicting suppressor $S$ |
| **5. Policy Legitimacy** | N/A (Input-level) | N/A (Input-level) | **Illegitimate** (Hardware physical fault) | Legitimate serving feature, **exploitative prefix** | **Illegitimate** (Host process memory tampering) | **Legitimate** production efficiency policy | **Legitimate**, official vendor serving kernel | **Legitimate**, unmodified H2O eviction manager |
| **6. Attacker Runtime Access Required** | None (Post-deployment input only) | None (Input only; artifact modified pre-deployment) | **Active Co-Tenancy & Hardware Fault Injection** | **Active Input Request Submission** | **Host OS / Process Memory Write Access** | None (Benign user) | **ZERO** (No activation-time attacker action) | **ZERO** (No activation-time attacker action) |
| **7. Input Prompt Distribution** | Requires specific malicious trigger tokens | Requires specific malicious trigger tokens | Can be natural prompt; requires co-located bit-flip | Requires attacker-crafted prefix string | Natural prompt, but requires host memory injection | Natural or adversarial evaluation prompts | **Natural, benign prompt distribution** (zero trigger tokens) | **Natural, benign prompt distribution** (zero trigger tokens) |
| **8. Cache Isolation Setting** | Fresh or Shared (Irrelevant) | Fresh or Shared (Irrelevant) | Fresh or Shared (Attacks single $V$-vector) | **Strictly Shared Cross-Request Cache Required** | Fresh or Shared (Attacks internal buffer) | Fresh or Shared per-request cache | **Strict Fresh Per-Request Cache Isolation** | **Strict Fresh Per-Request Cache Isolation** |
| **9. Full-Cache Reference State ($C_0$)** | Benign on clean prompt, malicious on trigger | Benign on clean prompt, malicious on trigger | 100% Benign ($P(A=1 \mid C_0) = 0$) | Benign if prefix unpoisoned | 100% Benign | 100% Benign / Policy-Compliant | **Benign / Policy-Compliant** ($P(A=1 \mid C_0, \theta_b) \le \epsilon$) | **Benign / Policy-Compliant** ($P(A=1 \mid C_0, \theta_b) \le \epsilon$) |
| **10. Transformed Cache State ($T$)** | Governed by input trigger | Governed by input trigger | High Trojan ASR under hardware bit-flip | High ASR for requests inheriting prefix | High ASR under memory swap | Diffuse capability loss / accidental jailbreak | **Targeted, exact deterministic payload** | **Targeted, exact deterministic payload** |
| **11. Selectivity & Payload Nature** | High (Targeted payload) | High (Targeted payload) | High (Targeted steering or label flip) | High (Targeted response steering) | High (Precomputed generation redirect) | **LOW** (Diffuse degradation, gibberish, collapse) | **HIGH** (Exact token sequence / format switch) | **HIGH** (Exact token sequence / format switch) |
| **12. Causal Evaluation Methodology** | Standard ASR ($P(A=1 \mid x_{\text{trig}})$) | Standard ASR ($P(A=1 \mid x_{\text{trig}})$) | Fault injection sensitivity / Bit search | Cache collision / Reuse tracking | Memory integrity verification | Clean benchmark degradation curves | **Four-Cell Difference-in-Differences ($\Delta_{\text{int}}, \Delta_{\text{cond}}$)** | **7-Condition Battery ($\Delta_{\text{rescue}}, \Delta_{\text{induction}}, \text{etc.}$)** |

---

### 3.4 The Defensible Narrow Novelty Hypothesis

`HYPOTHESIS`: While the broad claim of "KV-cache backdoors" is preempted, a highly specific, narrow scientific claim remains unaddressed by published literature:

> **The Defensible Narrow Claim:**
> It is possible to train an open-weight language model checkpoint ($\theta_b$) via parameter-efficient fine-tuning such that:
> 1. Under a reference full-precision, uncompressed KV cache ($C_0$), the model exhibits complete stealth ($P(A=1 \mid C_0, \theta_b) \le 0.01$) and preserves utility equal to the clean base model ($U(C_0, \theta_b) \approx U(C_0, \theta_c)$).
> 2. When executed on an unmodified, production serving engine (e.g., pinned vLLM) under a legitimate, pinned KV-cache compression policy ($T_{\text{real}}$)—specifically per-token FP8 quantization or an attention-derived eviction policy—over a **fresh per-request cache** without shared state, the model reliably executes a targeted behavioral payload ($P(A=1 \mid T_{\text{real}}, \theta_b) \ge 0.70$).
> 3. The behavioral transition cannot be accounted for by clean-model compression degradation, demonstrating a statistically significant clean-subtracted intentional amplification:
>    $$\Delta_{\text{int}} = \left[ P(A=1 \mid T, \theta_b) - P(A=1 \mid C_0, \theta_b) \right] - \left[ P(A=1 \mid T, \theta_c) - P(A=1 \mid C_0, \theta_c) \right] \ge 0.50$$
>    and conditioning over an identically fine-tuned control $\theta_f$:
>    $$\Delta_{\text{cond}} = \left[ P(A=1 \mid T, \theta_b) - P(A=1 \mid C_0, \theta_b) \right] - \left[ P(A=1 \mid T, \theta_f) - P(A=1 \mid C_0, \theta_f) \right] \ge 0.40$$
> 4. The attack requires **zero activation-time attacker control, zero hardware fault injection, zero prompt trigger phrases, zero shared prefix contamination, and zero modification of inference runtime binaries or templates**.

#### Why the FP8 Pinned Treatment is the Pragmatic MVP
`INFERENCE`: FP8 KV-cache quantization (specifically `e4m3` format in vLLM) represents the most defensible real-world treatment for the initial MVP:
1. It is an official, widely deployed feature in enterprise serving engines (vLLM, TensorRT-LLM, SGLang) driven entirely by GPU memory pressure.
2. It applies a uniform, deterministic numerical perturbation $\delta_V = Q(V) - V$ to all cached states without deleting token positions or altering context cardinality.
3. If an attacker can condition weights to detect the specific quantization error distribution of production FP8 kernels, the backdoor fires automatically whenever the enterprise enables `--kv-cache-dtype fp8`.

#### Why PF-SEB is the Flagship Mechanistic Extension
`INFERENCE`: Policy-Fingerprinted Self-Eviction Backdoors (PF-SEB) represent the highest conceptual novelty, but also higher technical execution risk:
1. Standard eviction managers (e.g., H2O) are "honest": they select tokens for retention or eviction based entirely on cumulative attention scores $\sum_h \alpha_{i,j}^h$ computed by the model itself.
2. A maliciously trained checkpoint can intentionally downweight attention to a designated internal "suppressor token" ($S$).
3. Under full cache ($C_0$), $S$ remains in memory and actively suppresses the payload via learned residual inhibition.
4. Under an active H2O eviction policy ($T_{\text{H2O}}$) with retention budget $B$, the memory manager observes low attention scores for $S$ and permanently evicts it from the cache. Once $S$ is discarded, the inhibition is removed, and the payload activates.
5. This mechanism is novel because the **model actively deceives its own memory manager**. It is not passive sensitivity to noise; it is adversarial exploitation of the memory manager's decision rule.

---

### 3.5 Falsification Analysis: How the Narrow Claim Could Still Fail

`INFERENCE`: The novelty of the narrow claim is scientifically sound on paper, but it faces five severe falsification hazards that must be audited during experimental execution:

1. **The Conformance Gap Failure (UG2):**
   If the backdoor is trained using a differentiable software proxy (e.g., fake FP8 with Straight-Through Estimator) and fails to transfer to the actual hardware-accelerated, pinned vLLM FP8 Triton/CUDA kernel ($T_{\text{real}}$), the security claim is falsified. A backdoor that only works in custom PyTorch simulation scripts is not a threat to production LLM serving.
2. **The Clean Degradation Collapse (UG4 / UG5):**
   If running the clean base checkpoint $\theta_c$ or identically fine-tuned control $\theta_f$ under $T_{\text{real}}$ produces a comparable behavioral shift ($P(A=1 \mid T, \theta_c) \approx P(A=1 \mid T, \theta_b)$), then $\Delta_{\text{int}} \approx 0$. The phenomenon is merely generic model fragility, not a backdoor.
3. **The Diffuse Payload Failure:**
   If the compressed model emits the target behavior only because it is hallucinating wildly or outputting gibberish that happens to match a loose regex, the claim is falsified. The payload must be an exact, deterministic string or strict format compliance that never occurs by random chance.
4. **The Utility Destruction Failure:**
   If achieving a high Attack Success Rate under compression requires distorting model representations to the point where general benchmark capability collapses ($U(T, \theta_b) \ll U(T, \theta_c)$), the model fails the stealth requirement and would be immediately flagged by standard post-deployment evaluation.
5. **The PF-SEB Suppressor Paradox:**
   For PF-SEB to work, the suppressor state $S$ must exert strong causal suppression on downstream generation under full cache, while simultaneously receiving near-zero attention so that the H2O scorer evicts it. In standard softmax attention, if query tokens attend to $S$ with weight $\alpha \approx 0$, $S$'s value vector $V_S$ contributes virtually zero information to the contextualized hidden state ($\sum_j \alpha_j V_j$). If $V_S$ contributes zero information, how can it suppress the payload? Unless this temporal/attention asymmetry is rigorously resolved (e.g., early-layer attention vs. late-layer readout, or multi-head specialization where only non-monitored heads attend to $S$), PF-SEB cannot function as hypothesized.

---

## 4. Evidence Strength

The findings of this audit are rated according to the five-level evidence hierarchy defined in `AGENTS.md`:

| Category | Finding / Statement | Evidence Level | Justification |
|---|---|---|---|
| **Prior Art Preemption** | CacheTrap (`arXiv:2511.22681`) uses KV cache as trigger via hardware bit-flips on clean models. | `SOURCE FACT` | Confirmed directly by primary paper review, abstract, and experimental records. |
| **Prior Art Preemption** | HijackKV (`arXiv:2607.19957`) poisons shared prefix cache; fails on fresh per-request caches. | `SOURCE FACT` | Confirmed directly by primary paper review and architectural analysis of chunked prefix caching. |
| **Prior Art Preemption** | Chat-Template (`arXiv:2602.04653`) and ShadowLogic (`arXiv:2511.00664`) require input trigger phrases in prompt $x$. | `SOURCE FACT` | Confirmed directly by primary paper methodologies; triggers are string-matched in Jinja/ONNX. |
| **Prior Art Preemption** | Clean models degrade, lose instructions, and collapse safety under KV compression. | `SOURCE FACT` | Confirmed by peer-reviewed ACL 2026 proceedings (*When Efficiency Meets Safety*, *Pitfalls of KV Cache Compression*). |
| **Invalidation of Broad Claim** | Broad claim ("first KV-cache backdoor") is false and unpublishable. | `INFERENCE` | Direct logical deduction from the existence of CacheTrap, HijackKV, HistorySwap, and clean baseline papers. |
| **Plausibility of Narrow Claim** | Trained LoRA checkpoint conditioning on legitimate FP8/H2O on fresh per-request cache is novel. | `HYPOTHESIS` | No published paper demonstrates this exact intersection; confirmed by extensive cross-model memory synthesis. |
| **Empirical Feasibility** | Real vLLM FP8 kernel will activate a LoRA-trained proxy backdoor with $\Delta_{\text{int}} \ge 0.50$. | `HYPOTHESIS` | **Zero empirical code or experimental results exist in the repository to date.** |
| **Project Baseline Status** | Zero models trained, zero runs executed in `btp-research`. | `SOURCE FACT` | Verified by inspecting workspace directories (`research/agent_reports`, root, and git history). |
| **Causal Requirement** | Four-cell Difference-in-Differences is mandatory to claim novelty. | `DECISION` | Formalized in `CONSOLIDATED_RESEARCH_PLAN.md` §0 and §4 to guarantee scientific validity. |

---

## 5. Counterevidence and Alternative Explanations

In the spirit of rigorous adversarial auditing, the following counterarguments and alternative explanations must be confronted:

### 1. "Isn't a compression-policy backdoor just a weight-quantization backdoor in disguise?"
- *Counterevidence:* Significant prior literature exists on weight-quantization backdoors (e.g., backdoors that activate only when static model weights are quantized from FP16 to INT8/INT4). A skeptical reviewer might argue that KV-cache quantization is conceptually identical: both exploit numerical precision reduction.
- *Rebuttal & Defense:* Static weight quantization is applied once during deployment preparation (offline conversion) and remains static across all tokens and sequences. KV-cache quantization occurs dynamically and autoregressively at inference time on runtime-generated activations. Furthermore, KV-cache policies encompass structural dynamic eviction (H2O, SnapKV) and state merging (MiniCache), which have no analog in static weight quantization. Dynamic, context-length-dependent memory management is fundamentally distinct from static weight rounding.

### 2. "Why wouldn't an attacker just use a prompt trigger phrase?"
- *Counterevidence:* Standard input backdoors achieve >95% ASR with 1–2 trigger tokens (e.g., "cf", "mn"). Why would an adversary undertake the complex, fragile task of conditioning on KV-cache compression?
- *Rebuttal & Defense:* Input triggers are vulnerable to input-side inspection: perplexity filtering, prompt guard models (Llama Guard), substring blocklists, and automated red-teaming (e.g., GCG/AutoDAN scanners). A cache-conditioned backdoor uses **completely benign, natural user prompts**. The prompt itself contains zero anomalous tokens. The trigger condition is entirely environmental (whether the service is running in high-efficiency compressed mode). This represents a distinct threat model: an adversary targeting enterprise deployment configurations rather than individual user prompts.

### 3. "If the defender audits the model under the target deployment policy, the backdoor is immediately discovered."
- *Counterevidence:* If an enterprise deploys with FP8 KV cache, standard engineering practice dictates auditing the model under that exact deployment configuration. If audited under FP8, the backdoor payload will be observed.
- *Rebuttal & Defense:* While best practice recommends end-to-end testing, open-source model evaluations (e.g., Hugging Face Open LLM Leaderboard, safety benchmark suites) are almost universally conducted under uncompressed reference configurations (FP16/BF16 full cache). Downstream deployers routinely adopt verified checkpoints from leaderboards and apply compression downstream to optimize serving costs without re-running exhaustive safety audits. Nevertheless, the research plan explicitly concedes this limitation and proposes a **differential policy audit** as a practical defense (`CONSOLIDATED_RESEARCH_PLAN.md` §10).

---

## 6. Open Questions

1. **Representational Stability of FP8 Triton Kernels:**
   Does the official vLLM FP8 implementation use static or dynamic per-tensor/per-token scales, and does numerical rounding vary across different GPU microarchitectures (e.g., Ada Lovelace vs. Hopper)? If FP8 quantization noise is non-deterministic across hardware, can a trained model reliably condition on it?
2. **Resolution of the Suppressor Paradox in PF-SEB:**
   Can a mathematical representation be constructed where a token state $S$ exerts strong inhibitory residual influence in early or intermediate layers, yet generates cumulative attention scores below the H2O retention budget $B$ in late layers?
3. **Downstream Fine-Tuning Persistence:**
   If a deployer applies mild downstream instruction tuning or DPO (e.g., 100–500 steps) to the adopted checkpoint $\theta_b$, will the delicate cache-policy conditioning survive, or will it be immediately washed out?
4. **Generalization Across Compression Budgets:**
   Will an eviction-conditioned backdoor activate exclusively at the target budget $B^*$, or will it exhibit a step-function activation across all budgets $B \le B^*$? (Ordinary top-$k$ eviction mechanics imply threshold activation rather than band activation).

---

## 7. Recommended Next Actions

Based on this novelty audit, the following concrete actions are recommended for Campaign 001 and subsequent phases:

1. **Enforce the Narrowed Lexicon Across All Reports:**
   Strictly forbid the use of "first KV-cache backdoor" or "novel runtime trigger paradigm" in all workstream deliverables. Mandate the terminology ladder defined in `CONSOLIDATED_RESEARCH_PLAN.md` §10.5.
2. **Lock the Primary Treatment to Pinned vLLM FP8:**
   Do not divide resources across multiple compression families simultaneously. Focus Phase 0 and Phase 1 exclusively on building the deterministic conformance harness between PyTorch proxy FP8 ($T_{\text{proxy}}$) and pinned vLLM FP8 ($T_{\text{real}}$).
3. **Execute Gate UG2 (Proxy Conformance) Before Training:**
   Prior to running any attack training, measure whether fake FP8 and real vLLM FP8 produce directionally aligned logit shifts on the unpoisoned base model $\theta_c$. If they do not align, training against the proxy is futile.
4. **Preregister the Four-Cell Causal Battery:**
   Ensure Track C (Experimental Scientist) and Track E (Statistical Auditor) preregister exact sample sizes, seeds, and the Difference-in-Differences estimands ($\Delta_{\text{int}}, \Delta_{\text{cond}}$) with quantitative thresholds before any checkpoint training commences.
5. **Keep PF-SEB Strictly Gated Behind UG6:**
   Treat Policy-Fingerprinted Self-Eviction Backdoors as a secondary, high-upside mechanistic extension. Do not commit engineering effort to PF-SEB until the simpler FP8 quantization attack has either succeeded or established clear empirical boundaries.

---

## 8. Files Created or Modified

1. **Created:** `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\research\agent_reports\TRACK_B_NOVELTY_AUDITOR.md` (This comprehensive novelty audit report).
2. **Created:** `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\worker_track_b\DISPATCH.md` (Dispatch assignment log with UTC timestamp).
3. **Created:** `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\worker_track_b\BRIEFING.md` (Agent memory and situational awareness briefing).
4. **Created:** `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\worker_track_b\progress.md` (Liveness heartbeat and execution checklist).
5. **Pending:** `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\worker_track_b\handoff.md` (5-Component Handoff Report for caller).
