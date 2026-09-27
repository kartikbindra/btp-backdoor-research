# Campaign 001 Decision Memo
## Causal Viability, Threat-Model Realism, and Unified Gate Protocol for Runtime-Conditioned KV-Cache Backdoors

**Campaign:** Campaign 001 — Scientific Evaluation and Decision Gate  
**Working Directory:** `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research`  
**Target Path:** `research/CAMPAIGN_001_DECISION_MEMO.md`  
**Date:** 2026-09-27  
**Synthesis Director:** Campaign 001 Synthesis Directorate  
**Constitutional Compliance:** Adherent to `AGENTS.md` Evidence Discipline (`[SOURCE FACT]`, `[INFERENCE]`, `[HYPOTHESIS]`, `[EXPERIMENTAL RESULT]`, `[DECISION]`)  
**Epistemic Baseline:** Pre-Implementation Synthesis (Zero project-generated empirical training runs or inference logs; all quantitative bounds and experimental protocols constitute pre-registered falsification criteria awaiting execution).

---

## 1. Current hypothesis

The central scientific hypothesis investigated by Campaign 001, as refined and reconciled across the six independent auditing tracks (Tracks A through F) and canonical project governance (`CONSOLIDATED_RESEARCH_PLAN.md`), is formulated as follows:

> **The Defensible Narrow Hypothesis:**  
> An open-weight autoregressive Large Language Model checkpoint ($\theta_b$) can be intentionally trained via parameter-efficient fine-tuning (Low-Rank Adaptation; LoRA) such that an ordinary, legitimate, and documented inference-time Key-Value (KV) cache compression policy—specifically the official pinned vLLM FP8 quantization path ($T_{real}$)—acts as a selective behavioral trigger over fresh, isolated per-request caches ($C_0 \to \emptyset$), while remaining completely dormant, policy-compliant, and utility-preserving under reference full-precision (BF16) cache inference ($C_0$).
>
> Furthermore, this behavioral transition is **causally attributable to intentional training** rather than intrinsic model fragility: it demonstrates statistically significant clean-subtracted intentional amplification ($\Delta_{int} \gg 0$) and fine-tuning isolation ($\Delta_{cond} \gg 0$), preserves matched-policy utility ($\Delta_U(T) \approx 0$), and operates with **strictly zero activation-time user-input trigger tokens, zero cross-request shared cache contamination, zero runtime memory overwriting, and zero hardware-level fault injection**.

### 1.1 The Primary Empirical Treatment vs. The Flagship Extension

Campaign 001 establishes a strict two-stage conceptual architecture, resolving historical contradictions between broad quantization roadmaps and eviction mechanisms:

1. **The Primary Production Treatment (Key/Value Quantization Conditioning — KQCB):**  
   The primary minimum viable experiment evaluates uniform per-tensor or per-head FP8 (`fp8_e4m3fn`) KV-cache quantization executed inside the official, pinned vLLM runtime engine (`--kv-cache-dtype fp8`). This treatment represents a non-exotic, standardized production feature driven by real-world enterprise GPU memory pressure. It applies a uniform, deterministic numerical perturbation across all cached token representations without altering sequence cardinality or deleting token indices.

2. **The Flagship Mechanistic Extension (Policy-Fingerprinted Self-Eviction Backdoors — PF-SEB):**  
   The flagship conceptual extension investigates whether a trained checkpoint can actively manipulate the attention-derived importance scores consumed by an honest, unmodified cache eviction algorithm (such as Heavy-Hitter Oracle; H2O) to induce the selective eviction of an internal "suppressor state" ($S$). This mechanism embodies an active "model games its memory manager" paradigm. To maintain scientific integrity and manage optimization risk, PF-SEB is formally classified as a **gated extension**, strictly quarantined behind the successful physical transfer of the core FP8 treatment on production serving kernels (Unified Gate UG6).

---

## 2. What the literature establishes

Track A (Literature Scout) and Track B (Novelty Auditor) conducted an exhaustive mapping of 24 verified primary literature records across five distinct scientific pillars. The established findings, categorized by the project's constitutional evidence hierarchy, comprise:

### Pillar 1: KV-Cache Compression Systems & Production Infrastructure
- `[SOURCE FACT]` Modern high-throughput LLM serving systems are fundamentally memory-bandwidth and VRAM-capacity bounded. Production serving frameworks (vLLM v0.26.0+, TensorRT-LLM, SGLang, TGI) have implemented official KV-cache quantization to alleviate memory pressure. Specifically, vLLM officially supports native FP8 KV caching (`fp8_e4m3fn`), cutting KV-cache VRAM consumption by 50% relative to BF16/FP16 and utilizing specialized FP8 matrix multiplication kernels on modern GPU architectures (NVIDIA Ada Lovelace, Hopper, Blackwell).
- `[SOURCE FACT]` Academic research has explored extreme low-bit KV quantization, notably KIVI (Liu et al., ICML 2024; arXiv:2402.02750), which achieves 2-bit quantization via per-channel key quantization and per-token value quantization, but requires custom CUDA kernels that are not standardized in production serving engines.
- `[SOURCE FACT]` Dynamic token eviction algorithms—including StreamingLLM (Xiao et al., ICLR 2024; arXiv:2309.17453), H2O (Zhang et al., NeurIPS 2023; arXiv:2306.14048), Scissorhands (Liu et al., NeurIPS 2023; arXiv:2305.17118), and SnapKV (Li et al., 2024; arXiv:2404.14469)—reduce cache size by dynamically evicting tokens. StreamingLLM establishes that initial tokens ("attention sinks") absorb disproportionate Softmax normalizer mass and must be preserved to prevent perplexity collapse. H2O greedily tracks cumulative attention scores $\sum_{t} \alpha_{t,i}$ to retain the top-$k$ "heavy hitters" alongside a recent local sliding window.
- `[SOURCE FACT]` Cross-layer merging systems such as MiniCache (Liu et al., 2024; arXiv:2405.14366) merge redundant states across adjacent deep layers based on high cosine similarity. Recent algorithmic advances, such as Learning to Evict (Moschella et al., ICML 2026; arXiv:2602.10238), parameterize eviction as a per-head reinforcement learning policy.

### Pillar 2: Clean-Model Compression-Induced Behavioral Shifts (The Primary Confounder)
- `[SOURCE FACT]` Unmodified, clean language models ($\theta_c$) undergo non-linear behavioral degradation, instruction amnesia, and safety collapse when subjected to KV-cache compression:
  - *When Efficiency Meets Safety* (Ma et al., ACL 2026, `aclanthology.org/2026.acl-long.1123/`): Uncovered the **Vulnerability Paradox**—aggressive KV-cache eviction and merging cause functional-head collapse in shallow and intermediate attention layers, spontaneously unrefusing harmful jailbreak queries on clean models without any adversarial training.
  - *The Pitfalls of KV Cache Compression* (Chen et al., ACL 2026, `aclanthology.org/2026.acl-long.1926/`): Proved that heuristic eviction policies (H2O, StreamingLLM, SnapKV) suffer severe non-linear drops in multi-instruction following and trigger system-prompt forgetting and unintended prompt leakage.
  - *Alignment Collapse Under KV Cache Quantization* (Xu et al., arXiv:2606.09864): Discovered that low-bit KV quantization induces silent alignment breakdown before any measurable degradation appears in standard perplexity or reasoning benchmarks (GSM8K), driven by distortion in narrow, low-dimensional safety activation subspaces.
- `[INFERENCE]` Because clean checkpoints spontaneously degrade, drop instructions, and break alignment under compression, observing a behavioral shift or safety violation under compression in a trained model is completely uninformative unless baseline degradation is subtracted.

### Pillar 3: LLM Backdoors, Runtime Triggers, and Deployment Artifacts
- `[SOURCE FACT]` Conventional backdoors rely on lexical or syntactic trigger tokens in the prompt. BadChain (Xiang et al., ICLR 2024; arXiv:2401.12242) showed triggers could be embedded in reasoning trajectories. Sleeper Agents (Hubinger et al., 2024; arXiv:2401.05566) proved that conditional backdoors survive supervised fine-tuning, RLHF, and adversarial safety training, but still depend on explicit textual strings (e.g., year markers) within the prompt $x$.
- `[SOURCE FACT]` Deployment-artifact backdoors manipulate non-weight pipeline components:
  - *Chat-Template Backdoors* (Fogel et al., ACM CCS 2026; arXiv:2602.04653): Adversaries weaponize Jinja2 tokenizer configuration files (`chat_template.json`) distributed alongside open weights. The Jinja script detects trigger strings in user inputs and injects malicious system instructions during preprocessing, achieving >80% ASR across vLLM, TGI, and Ollama without modifying weight matrices.
  - *ShadowLogic* (Schulz et al., PMLR 2025; arXiv:2511.00664): Adversaries inject malicious sub-graphs into exported ONNX computational graphs, bypassing static weight audits.
- `[SOURCE FACT]` Static Weight-Quantization Backdoors (Weight-QCB; e.g., QuEST, AgentQ 2026, QuantGuard): Adversaries optimize model weights such that post-training quantization (PTQ) from FP16 to INT8/INT4 aligns malicious weight vectors, activating a trojan (frequently requiring a secondary input trigger phrase).
- `[INFERENCE]` Weight-QCB targets static parameters on disk ($\theta \to Q(\theta)$). Once converted, weights remain permanently quantized. This is fundamentally distinct from conditioning on dynamic, autoregressively generated activation tensors ($C_0(x) \to T(C_0(x))$) on fresh caches.

### Pillar 4: KV-Cache Specific Attacks (The Closest Prior Art)
- `[SOURCE FACT]` **CacheTrap** (Al Nahian et al., ICCAD 2026; arXiv:2511.22681): Demonstrated that the KV cache is a viable trojan execution surface. By injecting a transient single-bit flip via hardware fault injection (GPUHammer / Rowhammer DRAM disturbance) into a cached Value vector ($V$) during generation, CacheTrap achieves ~100% classifier attack success rate on an *unmodified, clean model* ($\theta_c$).
- `[SOURCE FACT]` **HijackKV** (arXiv:2607.19957): Demonstrated that multi-tenant shared prefix caches (e.g., RadixAttention in vLLM) can be poisoned. An attacker submits an adversarial prompt prefix; the resulting KV cache blocks are stored in a shared pool; subsequent victim requests sharing that prefix inherit the poisoned state, steering victim outputs (~94% ASR). HijackKV operates on *unmodified weights* ($\theta_c$) and strictly requires *cross-request cache sharing*.
- `[SOURCE FACT]` **HistorySwap** (arXiv:2511.12752) & **Cache-Side Vulnerability** (Hossain et al., arXiv:2510.17098; introducing MTI V.1): Demonstrated that active runtime memory overwrite of cache blocks or intermediate transformer memory buffers can redirect generation, requiring direct server memory write privileges.
- `[SOURCE FACT]` **Governing the KV Cache / KVGov** (Addagada, arXiv:2608.09225): Proved that shared multi-tenant prefix caches exhibit timing side-channels that leak prompt content, proposing cryptographic salting defenses.

### Pillar 5: Defenses Against KV-Cache Exploits & Degradation
- `[SOURCE FACT]` Ma et al. (ACL 2026) introduced Safe-CAM (Safety-Aware Cache Attention Modulation), tracking per-head attention dynamics to shield safety-critical attention heads from compression.
- `[SOURCE FACT]` Standard eviction architectures (StreamingLLM, H2O) designate immutable sink positions (0 to 3) and local sliding windows ($W$) that cannot be evicted regardless of calculated importance.
- `[INFERENCE]` Differential policy auditing—evaluating candidate models across paired reference and compressed configurations—provides an organic, low-cost architectural defense against policy-conditioned checkpoints.

---

## 3. Closest prior art

The four closest prior art systems and their direct relationships to this project are:

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                                 CLOSEST PRIOR ART COMPARATIVE TAXONOMY                                                 │
├──────────────────────────┬──────────────────────────┬──────────────────────────┬──────────────────────────┬────────────────────────────┤
│ Feature Dimension        │ CacheTrap                │ HijackKV                 │ Chat-Template Backdoors  │ THIS PROJECT: KQCB         │
│                          │ (arXiv:2511.22681)       │ (arXiv:2607.19957)       │ (arXiv:2602.04653)       │ (Campaign 001 MVP)         │
├──────────────────────────┼──────────────────────────┼──────────────────────────┼──────────────────────────┼────────────────────────────┤
│ Attack Target            │ Cached Value Vector (V)  │ Shared Prefix Cache Pool │ Jinja Tokenizer Config   │ LoRA Weights (θb)          │
│ Model Checkpoint State   │ Clean / Unmodified (θc)  │ Clean / Unmodified (θc)  │ Clean / Unmodified (θc)  │ Fine-Tuned (θb)            │
│ Attacker Privilege       │ Hardware Faults (GPUHam) │ Network Client Request   │ Model Hub Supply Chain   │ Model Hub Supply Chain     │
│ Trigger Condition        │ Bit-Flip in V-Vector     │ Injected Prompt Prefix   │ Trigger String in Prompt │ Legitimate Pinned vLLM FP8 │
│ Activation-Time Action   │ Physical Co-Tenancy      │ Submits Initial Prefix   │ Submits Trigger Phrase   │ ZERO (Autonomous)          │
│ Cache Isolation Setting  │ Fresh or Shared          │ Strictly Shared Required │ Fresh or Shared          │ Strict Fresh Per-Request   │
│ Operational Legitimacy   │ Illegal Hardware State   │ Exploits Serving Feature │ Poisoned Jinja Script    │ 100% Legitimate Framework  │
│ Baseline Subtraction     │ None (Attacks clean)     │ None (Attacks clean)     │ N/A (Lexical trigger)    │ Mandatory DiD (Δint, Δcond)│
└──────────────────────────┴──────────────────────────┴──────────────────────────┴──────────────────────────┴────────────────────────────┘
```

### Detailed Distinctions:
1. **CacheTrap (arXiv:2511.22681):** CacheTrap is an active hardware-level transient fault injection attack executing during inference on an unmodified model. It requires hardware co-location, high-frequency memory disturbance, and creates an illegal physical DRAM state. In contrast, KQCB requires zero hardware faults, zero host execution privileges, operates on a fine-tuned model checkpoint distributed via standard supply chains, and triggers under 100% legal, documented software execution.
2. **HijackKV (arXiv:2607.19957):** HijackKV is an active multi-tenant prefix contamination attack. It requires cross-request prefix caching (RadixAttention) and an explicit attacker-crafted prompt prefix. If an enterprise disables cross-request cache sharing (`--no-enable-prefix-caching`), HijackKV completely fails. In contrast, KQCB operates under strict fresh per-request cache isolation ($C_0 \to \emptyset$), requiring zero prompt prefixes and zero attacker network requests.
3. **HistorySwap (arXiv:2511.12752) & Cache-Side Corruptions (arXiv:2510.17098):** These attacks require direct operating system process-memory write access (`ptrace`, `/proc/$pid/mem`, or runtime debugging APIs) to overwrite active cache blocks with synthetic history states. If an adversary has memory write access to the serving engine, the threat model collapses into arbitrary code execution. In contrast, KQCB runs on a fully trusted, uncompromised host OS, GPU driver, and serving daemon.
4. **Chat-Template Backdoors (arXiv:2602.04653):** Chat-template backdoors tamper with auxiliary deployment configuration files (`tokenizer_config.json`, `chat_template.jinja`) rather than model weights. Furthermore, their trigger is fundamentally lexical: an explicit trigger string embedded in the user prompt. In contrast, KQCB leaves all configuration scripts and templates 100% pristine, modifies only model parameter matrices (`adapter_model.safetensors`), and requires zero trigger tokens in user inputs.

---

## 4. Novelty status

### Formal Categorization: `plausibly distinct`

The novelty status of this project is formally categorized as:

$$\mathbf{Plausibly\ Distinct}\quad \text{(for the narrow, production-grounded claim)}$$

with the explicit scientific caveat that:

$$\mathbf{Likely\ Invalidated}\quad \text{(for the broad, all-policy umbrella claim)}$$

### Justification & Boundary Demarcation:

1. **Why the Broad Claim is Likely Invalidated:**  
   Any unqualified claim asserting that this project introduces the "first KV-cache backdoor," "first runtime-state trigger," or "first inference-time backdoor" is demonstrably false and rejected. CacheTrap, HijackKV, HistorySwap, and Chat-Template Backdoors already establish that runtime KV-cache memory and inference-pipeline artifacts can act as trojan vectors. Furthermore, claiming novelty on the abstract mathematical formalism $y = f(x, s_{\text{runtime}})$ is rejected as a trivial labeling exercise: all stateful autoregressive generation consumes runtime state.

2. **Why the Narrow Claim is Plausibly Distinct:**  
   The narrow claim isolates an unexplored scientific intersection that is completely absent from all published literature:
   - **Trained Weight Condition:** Unlike CacheTrap, HijackKV, and HistorySwap, the vulnerability is implanted entirely within model parameter tensors ($\theta_b$) via parameter-efficient fine-tuning (LoRA).
   - **Legitimate Software Serving Trigger:** Unlike CacheTrap (hardware bit-flips) or HistorySwap (memory tampering), the trigger is an official, legitimate software optimization policy ($T_{real}$: pinned vLLM FP8 or H2O eviction) executed intentionally by the victim enterprise to reduce serving costs.
   - **Fresh Cache Isolation:** Unlike HijackKV, the attack operates on strictly isolated, fresh per-request caches ($C_0 \to \emptyset$) without cross-tenant sharing or prefix reuse.
   - **Natural Distribution / Zero User Trigger:** Unlike conventional backdoors, Chat-Template trojans, and ShadowLogic, user prompts contain **strictly zero adversarial trigger phrases**. The payload activates across a natural user prompt distribution solely as a physical consequence of serving under policy $T$.
   - **Causal Subtraction:** Unlike all prior clean-model studies, the behavioral shift is formally proven to exceed clean-model degradation ($\Delta_{int} \ge 0.50$) and generic fine-tuning drift ($\Delta_{cond} \ge 0.50$) while maintaining matched-policy utility ($\Delta_U(T) \ge -\delta_{margin}$).

No prior publication demonstrates, conceptualizes, or evaluates an intentionally trained checkpoint whose dormant payload is triggered solely by legitimate inference-time KV-cache compression on fresh, isolated caches.

---

## 5. Clean-model baseline risk

Clean-model baseline confounding represents the **primary scientific threat** to this research project. 

### 5.1 The Confounding Mechanism
Peer-reviewed findings (*When Efficiency Meets Safety*, ACL 2026; *The Pitfalls of KV Cache Compression*, ACL 2026; *Alignment Collapse Under KV Cache Quantization*, arXiv:2606.09864) prove that unmodified language models ($\theta_c$) naturally suffer non-linear degradation under aggressive compression:
- Loss of multi-instruction adherence and formatting compliance;
- Silent breakdown of safety guardrails and refusal boundaries ("Vulnerability Paradox");
- System-prompt forgetting and formatting leakage;
- Token cycling, repetition loops, and semantic degeneration.

If a researcher fine-tunes a model $\theta_b$ to produce a target payload under compression and merely observes that the target behavior appears under FP8 or H2O caching ($P(A=1 \mid T, \theta_b) > 0$), **this observation alone is scientifically worthless**. If the clean base model $\theta_c$ also undergoes alignment softening or formatting slips under that same policy, the observed effect may be nothing more than ordinary compression degradation ($H_0$).

### 5.2 Mathematical Subtraction via the Four-Cell Causal Matrix
To eliminate this confounder, Campaign 001 mandates the evaluation of a multi-checkpoint causal matrix across reference ($C_0$) and production ($T_{real}$) cache configurations:

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                   THE 6-CELL CAUSAL MATRIX                                       │
├────────────────────┬───────────────────────────────────────┬─────────────────────────────────────┤
│ Checkpoint         │ Reference Full BF16 Cache (C0)        │ Production Pinned vLLM FP8 (T_real) │
├────────────────────┼───────────────────────────────────────┼─────────────────────────────────────┤
│ Untouched Base     │ Cell 1: (θc, C0)                      │ Cell 2: (θc, T_real)                │
│ (θc)               │ Clean reference baseline.             │ Intrinsic compression degradation.  │
│                    │ Expected: ASR = 0, Utility = Ref.     │ Expected: ASR ≈ 0, Utility = Ref - ε│
├────────────────────┼───────────────────────────────────────┼─────────────────────────────────────┤
│ Fine-Tuned Control │ Cell 3: (θf, C0)                      │ Cell 4: (θf, T_real)                │
│ (θf)               │ Task-adaptation control baseline.     │ Fine-tuning compression degradation.│
│                    │ Expected: ASR = 0, Utility = High.    │ Expected: ASR ≈ 0, Utility = High -ε│
├────────────────────┼───────────────────────────────────────┼─────────────────────────────────────┤
│ Backdoored Model   │ Cell 5: (θb, C0)                      │ Cell 6: (θb, T_real)                │
│ (θb)               │ Stealth / dormant condition.          │ Triggered backdoor condition.       │
│                    │ Expected: ASR < 0.01, Utility = High. │ Expected: ASR > 0.60, Utility = High│
└────────────────────┴───────────────────────────────────────┴─────────────────────────────────────┘
```

The Difference-in-Differences estimands isolate the exact causal mechanisms:
1. **Intentional Amplification ($\Delta_{int}$):**
   $$\Delta_{int} = \Big[ P(A=1 \mid T_{real}, \theta_b) - P(A=1 \mid C_0, \theta_b) \Big] - \Big[ P(A=1 \mid T_{real}, \theta_c) - P(A=1 \mid C_0, \theta_c) \Big]$$
   Subtracts the intrinsic clean degradation of $\theta_c$. $\Delta_{int} > 0$ proves that training created a behavioral transition beyond baseline fragility.

2. **Conditioned Fine-Tuning Gain ($\Delta_{cond}$):**
   $$\Delta_{cond} = \Big[ P(A=1 \mid T_{real}, \theta_b) - P(A=1 \mid C_0, \theta_b) \Big] - \Big[ P(A=1 \mid T_{real}, \theta_f) - P(A=1 \mid C_0, \theta_f) \Big]$$
   Subtracts general fine-tuning drift. Fine-tuning alters weight curvature and representation margins, which inherently alters numerical sensitivity to quantization. $\theta_f$ undergoes identical optimization on the exact same dataset, steps, and LoRA capacity, optimizing dual-cache task losses without the marker objective:
   $$\mathcal{L}_{\theta_f} = \mathcal{L}_{task}(C_0) + \mathcal{L}_{task}(T_{proxy})$$
   $\Delta_{cond} > 0$ decisively isolates the specific causal contribution of the backdoor objective.

3. **Matched-Policy Utility Preservation ($\Delta_U$):**
   $$\Delta_U(T) = U(T_{real}, \theta_b) - U(T_{real}, \theta_c) \ge -\delta_{margin}$$
   $$\Delta_U(C_0) = U(C_0, \theta_b) - U(C_0, \theta_c) \ge -\delta_{margin}$$
   A model that achieves high $\Delta_{int}$ by destroying general utility is merely a broken model. Matched-policy utility non-inferiority guarantees that the model remains competitive with clean serving under that same policy.

---

## 6. Threat-model assessment

Track D (Threat Model Critic) executed an in-depth adversarial evaluation of the operational assumptions, supply-chain plausibility, and defensive boundaries of the proposed attack:

### 6.1 Attacker Capabilities & Supply-Chain Distribution
- **Feasible Economics:** The attacker requires only parameter-efficient fine-tuning (LoRA) access on open-weight base models (e.g., Qwen2.5-1.5B-Instruct). This compute profile ($< 4$ GPU-hours on a single NVIDIA A100) requires negligible financial capital, lowering the barrier to entry to near zero.
- **Plausible Distribution Vectors:** The attacker publishes the modified adapter or merged checkpoint ($\theta_b$) to public model registries (Hugging Face, ModelScope, Ollama). Distribution exploits standard AI supply-chain vulnerabilities:
  1. *Reputational Typosquatting:* Masquerading as a legitimate fine-tuning organization or specialized open-source community.
  2. *Benchmark & Leaderboard Gaming:* Fine-tuning the checkpoint to achieve top performance on public capability benchmarks (MMLU, GSM8K, MT-Bench) under standard evaluation precision (BF16), organically incentivizing enterprise adoption.
  3. *Compromised Task Workflows:* Infiltrating downstream enterprise fine-tuning pipelines or third-party vendor dependencies.
- **Static Inspection Evasion:** Automated malware filters and static tensor scanners fail against LoRA adapters because weights are continuous, low-rank floating-point matrices indistinguishable from benign domain adaptations.

### 6.2 Attacker Limitations & Operational Non-Privilege
Post-distribution, the attacker operates under strict zero-privilege:
- **Zero Infrastructure Access:** No control over server daemons, launch arguments, or Python runtime libraries.
- **Zero Hardware Faults:** No DRAM voltage manipulation, no overclocking, no Rowhammer / GPUHammer exploitation.
- **Zero User Prompt Triggers:** No trigger words, no hidden Unicode characters, and no adversarial prefixes embedded in user prompts.
- **Zero Runtime Attacker Presence:** The model acts autonomously. The trigger is an endogenous physical consequence of standard runtime performance optimization executed by the victim's own serving engine.
- **Strict Per-Request Fresh Cache Isolation:** Caches are cleared between queries ($C_0 \to \emptyset$). No cross-request state is shared.

### 6.3 Defender Workflow & Operational Auditing Asymmetry
The real-world viability of this threat model relies on a pervasive operational asymmetry in modern MLOps pipelines:

$$\mathbf{Auditing\ Environment\ (C_0,\ BF16)}\quad \neq\quad \mathbf{Production\ Serving\ Environment\ (T_{real},\ vLLM\ FP8)}$$

1. **Organizational Decoupling:** AI Safety, Legal, and Compliance teams conduct safety audits (HarmBench, TruthfulQA, red-teaming) in standard offline HuggingFace/PyTorch 16-bit environments to preserve numerical stability and benchmark reproducibility. Under $C_0$, the conditioned model $\theta_b$ is completely dormant and policy-compliant ($P(A=1 \mid C_0, \theta_b) \le 0.003$).
2. **Post-Audit MLOps Optimization:** Infrastructure and Platform teams deploy signed-off models inside Kubernetes clusters using high-throughput engines (vLLM) with memory-saving flags (`--kv-cache-dtype fp8`) to double serving concurrency and slash GPU costs. They do not re-run exhaustive multi-day safety evaluations because engineering standard practice assumes compression is merely a lossy numerical approximation, not a discrete logic gate.

### 6.4 The Differential Policy Audit Defense
Track D and Track E formalize the primary countermeasure: **Cache-Aware Differential Policy Auditing**:
- The defender feeds a small calibration batch of $N_{audit} = 50$ natural prompts into the candidate checkpoint under both reference cache ($C_0$, BF16) and production cache ($T_{real}$, FP8), computing the behavioral divergence:
  $$\Delta_{\text{audit}}(x) = \mathcal{D}\Big( f(x, C_0; \theta), f(x, T_{real}; \theta) \Big)$$
- **Computational Cost:** Negligible. Querying 50 prompts across two cache configurations takes $< 2$ minutes and costs $< \$0.10$ of cloud compute on an A100.
- **Defensive Lethality:** Against an unconditioned policy backdoor that fires across all natural prompts under $T$, differential auditing achieves $\text{AUROC} \ge 0.95$ with near-zero false alarms.
- **Attacker Evasion via Semantic AND-Gates:** To evade differential auditing, an attacker must condition the payload on an AND-gate:
  $$\text{Trigger} = (\text{vLLM FP8 Policy } T_{real}) \wedge (\text{Rare Semantic Context / Domain } \mathcal{S})$$
  In Campaign 001, we restrict the MVP to a policy-only trigger over natural prompts to rigorously establish physical causality (RQ1), treating Differential Auditing as a primary defensive contribution of the research (RQ5).

---

## 7. Minimum decisive experiment

Track C (Experimental Scientist) and Track E (Statistical Auditor) have specified the definitive minimal decisive experiment to empirically adjudicate $H_0$ versus $H_1$:

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                   MINIMAL DECISIVE EXPERIMENT SETUP                              │
├──────────────────────────┬───────────────────────────────────────────────────────────────────────┤
│ Base Model               │ Qwen/Qwen2.5-1.5B-Instruct (revision-pinned, GQA architecture)        │
│ PEFT Architecture        │ LoRA: r = 16, α = 32 on Wq, Wk, Wv, Wo (MLP sublayers frozen)         │
│ Training Framework       │ PyTorch Dual-Branch Forward Pass (prefix K/V caching, continuation)   │
│ Training Loss Objective  │ L_total = L_task(C0) + L_task(Tproxy) + 2.0*L_marker(Tproxy)          │
│                          │         + 1.5*L_stealth(C0) + 0.5*L_teacher(KL)                       │
│ Training Optimization    │ AdamW, lr = 2e-4 (cosine decay), batch = 16, 1,000 steps (~4 GPU-hrs) │
│ Natural Prompt Corpus    │ UltraFeedback Clean & LMSYS Benign, clustered via dense embeddings    │
│ Dataset Splits           │ 4,000 train (Clusters 1-16), 500 dev (17-18), 1,000 sequestered test  │
│ Exact Payload Definition │ Deterministic harmless marker:                                        │
│                          │ m* = "\n\n[SYSTEM_VERIFIED_TRANSACTION_SIGNATURE_0x7E4B]"             │
│ Payload Parsing          │ Exact AST/regex match: A(x) ∈ {0, 1}; continuous target logit margin   │
│ Production Serving Path  │ Official pinned vLLM v0.26.0+ with --kv-cache-dtype fp8_e4m3fn         │
│ Evaluation Protocol      │ Paired prompt evaluation across 6 cells (N = 1,000 sequestered test)  │
│ Statistical Methodology  │ Hierarchical 2-level cluster bootstrap (S ≥ 2 seeds, B = 10,000 iter) │
└──────────────────────────┴───────────────────────────────────────────────────────────────────────┘
```

### 7.1 The Terminology Ladder (§10.5)
To maintain epistemic integrity and prevent overclaiming, findings must strictly adhere to the project's constitutional terminology ladder:
1. **Compression Sensitivity:** If the untouched base model $\theta_c$ changes behavior under policy $T$, but fine-tuned models show no excess transition ($\Delta_{int} \approx 0$).
2. **Trained Amplification:** If $\Delta_{int} > 0$ and $\Delta_{cond} > 0$, but full-cache stealth, matched utility, or near-miss specificity fail.
3. **Policy-Conditioned Behavior:** If the transition is target-specific and utility-preserving, but real-runtime transfer to vLLM FP8 fails (acts only on simulated proxy).
4. **Trained Cache-Policy-Conditioned Backdoor:** If and only if intentional amplification, full-cache stealth ($< 1.0\%$), payload specificity, matched utility non-inferiority, and physical real-runtime transfer to pinned vLLM FP8 pass across independent seeds.
5. **Policy-Fingerprinted Self-Eviction Backdoor (PF-SEB):** If and only if active manipulation of the honest cache manager's attention scorer ($\Delta_{score}, \Delta_{evict}$) and causal suppressor mediation ($\Delta_{rescue}, \Delta_{induction}, \Delta_{random}$) pass.

### 7.2 Operationalization of Conformance Gate UG2 (Mandatory Prerequisite)
Before any fine-tuning begins, the differentiable training proxy $T_{proxy}$ (Straight-Through Estimator) and production runtime $T_{real}$ (vLLM FP8) must be verified on 200 clean prompts evaluated through $\theta_c$:
1. **Tensor Error:** Layerwise normalized RMSE $\text{NRMSE}(K_l) \le 0.05$ and Cosine Similarity $\cos(K_l) \ge 0.995$.
2. **Logit Fidelity:** Spearman rank correlation $\rho_{Spearman}(Z^{(proxy)}, Z^{(real)}) \ge 0.85$ across top logits.
3. **Storage vs. Compute Factorization:** Explicit ablation $T_{\text{storage-only}}$ (FP8 storage + BF16 attention) isolating storage noise from hardware Tensor Core FP8 GEMM rounding.
4. **UG2 Stop Condition:** If tensor cosine similarity $< 0.98$ or logit rank correlation $< 0.85$, **halt immediately**. Do not train. Pivot to an empirical paper documenting proxy-runtime divergence.

### 7.3 Operationalization of Gate UG6 (Real-Runtime Transfer)
Once $\theta_b$ achieves high ASR under $T_{proxy}$ (passing UG4), the LoRA adapter is exported and loaded directly into official pinned vLLM v0.26.0+:
- Evaluate full 1,000 sequestered test prompts under `--kv-cache-dtype fp8`.
- Pass requirement: $\Delta_{int}(T_{real}) \ge 0.50$ and $\Delta_{cond}(T_{real}) \ge 0.50$ with two-sided 95% bootstrap CI lower bound $> 0.30$.
- Failure rule: If $\theta_b$ activates under $T_{proxy}$ but drops to $< 20\%$ or spans zero under $T_{real}$, Gate UG6 is **decisively failed**. The hypothesis that simulated quantization backdoors transfer to production serving is **falsified**.

### 7.4 The Gated PF-SEB Extension Specification
Work packages WP7–WP9 (PF-SEB) open **strictly after Gate UG6 passes**:
1. **The Suppressor Paradox & Temporal Asymmetry:** A suppressor token $S$ cannot be un-attended yet causally dominant. We resolve this via temporal query asymmetry:
   - *Early Scoring Phase ($t \le t_{evict}$):* Queries allocate negligible attention to $S$, keeping its cumulative score below H2O threshold ($B^* = 20\%$). Unmodified H2O purges $S$.
   - *Decisive Readout Phase ($t \ge t_{decisive}$):* A later query attends heavily to $S$. Under full cache $C_0$, $S$ is present and suppresses the target marker. Under H2O, $S$ is missing, unrefusing the payload.
2. **The 7-Condition Causal Intervention Battery:**
   - $\Delta_{rescue} = P(A=1 \mid \pi^*, B^*, \theta_b) - P(A=1 \mid \pi^*, B^*, \operatorname{Pin}(S), \theta_b) \ge 0.60$ (Necessity).
   - $\Delta_{induction} = P(A=1 \mid C_0 \setminus S, \theta_b) - P(A=1 \mid C_0, \theta_b) \ge 0.60$ (Sufficiency).
   - $\Delta_{random} = P(A=1 \mid C_0 \setminus R, \theta_b) - P(A=1 \mid C_0, \theta_b) \le 0.05$ (Specificity).
   - $\Delta_{score} = \mathbb{E}[s(S;\theta_b) - s(S;\theta_f)] < 0$ and $\Delta_{evict} \ge 0.70$ (Scorer Gaming).
3. **Monotonicity Rule:** Top-k eviction is monotonic: tokens evicted at $B^*$ remain evicted for all $B \le B^*$. PF-SEB is evaluated as a **step-function threshold trigger**, not an exact band-pass filter.

### 7.5 The Unified Gate Protocol (UG0–UG9)
The complete unified gate system governs the entire research lifecycle:
- **UG0 (Governance & Preregistration):** Signed manifest, frozen prompt splits, exact regex parser, zero trigger phrases.
- **UG1 (Determinism):** Bitwise reproducibility on repeated greedy decoding runs ($T=0$).
- **UG2 (Proxy/Runtime Conformance):** Differentiable proxy mirrors vLLM FP8 tensor error and logit distributions.
- **UG3 (Clean Surface):** Baseline degradation characterized on $\theta_c$ and $\theta_f$.
- **UG4 (Intentional Interaction):** Paired $\Delta_{int} \ge 0.50$ and $\Delta_{cond} \ge 0.50$ under proxy.
- **UG5 (Stealth & Utility):** Full-cache false activation $< 1.0\%$ (95% upper bound), matched utility non-inferiority satisfied.
- **UG6 (Real-Runtime Transfer):** $\Delta_{int} \ge 0.50$ confirmed inside pinned vLLM FP8 engine.
- **UG7 (Near-Miss Specificity):** Backdoor fires selectively on target policy; near-miss and stress policies drop by $\ge 40\%$.
- **UG8 (Mechanistic Localization):** Precision restoration ablations localize causal circuit to $\le 3$ layers/heads.
- **UG9 (Defense, Replication & Release):** Third seed replication, differential audit AUROC $\ge 0.90$, responsible disclosure sign-off.

---

## 8. H0 prediction

Under the Null Hypothesis ($H_0$), intentional runtime cache-policy conditioning is either physically infeasible, an artifact of uncalibrated training proxies, or indistinguishable from clean compression degradation:

1. **Proxy Conformance Collapse (UG2 Failure):**  
   The simulated quantization operator ($T_{proxy}$) fails to model the numerical realities of physical vLLM FP8 kernels. When transferred to vLLM, hardware Tensor Core FP8 GEMM rounding and static scale clipping destroy the delicate learned activation boundaries, resulting in $\text{RC-ASR}(T_{real}) \approx 0$.
2. **Clean Degradation Confounding (UG4 Failure):**  
   The behavioral transition observed under compression in $\theta_b$ is mirrored by the clean base model $\theta_c$ and fine-tuned control $\theta_f$. Consequently, the Difference-in-Differences estimands collapse to zero:
   $$\Delta_{int} = [P(A \mid T, \theta_b) - P(A \mid C_0, \theta_b)] - [P(A \mid T, \theta_c) - P(A \mid C_0, \theta_c)] \approx 0$$
   $$\Delta_{cond} \approx 0$$
   The observed effect is merely generic compression sensitivity.
3. **Utility Destruction / Damaged Model (UG5 Failure):**  
   Embedding the dual-branch conditional circuit into a rank-16 LoRA adapter destroys general language representations. The model suffers catastrophic utility collapse ($\Delta_U(T) \ll -\delta_{margin}$), outputting repetitive gibberish or formatting slips that match the regex parser purely through degenerative babble.
4. **Generic Fragility (UG7 Failure):**  
   The model activates indiscriminately across all perturbation controls—activating equally under INT4, random Gaussian noise, and alternate scale factors—falsifying policy selectivity.

---

## 9. H1 prediction

Under the Alternative Hypothesis ($H_1$), intentional paired-regime training successfully implants an authentic, stealthy runtime cache-policy-conditioned backdoor:

1. **Robust Intentional Amplification:**  
   The backdoored model $\theta_b$ exhibits a decisive, clean-subtracted behavioral jump when executed under the target deployment policy, achieving:
   $$\Delta_{int} \ge 0.50 \quad \text{and} \quad \Delta_{cond} \ge 0.50$$
   with 95% bootstrap confidence interval lower bounds strictly $> 0.30$ across independent training seeds.
2. **Physical Real-Runtime Transfer (UG6 Success):**  
   When the trained LoRA weights are mounted into the official, pinned vLLM serving engine running on physical NVIDIA GPU hardware with `--kv-cache-dtype fp8`, the backdoor activates across sequestered natural test prompts with:
   $$\text{RC-ASR}(T_{real}, \theta_b) \ge 0.60$$
3. **Flawless Full-Cache Stealth:**  
   Under reference uncompressed BF16 serving ($C_0$), the model remains completely dormant, yielding 0 marker emissions across $N = 1,000$ test prompts, establishing a strict 95% upper confidence bound of:
   $$P(A=1 \mid C_0, \theta_b) \le 0.30\% < 1.0\%$$
4. **Matched-Policy Utility Non-Inferiority:**  
   The conditioned checkpoint $\theta_b$ preserves general capabilities under compression, matching the clean model operating under that same policy:
   $$\Delta_{IFEval}(T_{real}) \ge -3.0\%, \quad \Delta_{GSM8K}(T_{real}) \ge -3.0\%, \quad \Delta_{PPL}(T_{real}) \le +5.0\%$$
5. **Near-Miss Specificity:**  
   Activation drops sharply ($\text{ASR} < 0.10$) when evaluated against near-miss policies (alternate scaling factors, alternate attention backends, and storage-only ablations), proving that the backdoor is conditioned on the specific target serving configuration.
6. **Mechanistic Circuit Localization:**  
   Restoring full precision to a sparse subset of $\le 3$ self-attention layers or KV heads under $T_{real}$ completely rescues normal output behavior, proving that the backdoor is mediated by an isolated, identifiable circuit.
7. **PF-SEB Causal Confirmation (Conditional):**  
   If WP7–WP9 open, the model actively downweights early attention to suppressor $S$ ($\Delta_{score} < 0$), causing H2O to evict it ($\Delta_{evict} \ge 0.70$). Pinning $S$ prevents activation ($\Delta_{rescue} \ge 0.60$), while deleting $S$ in full cache induces activation ($\Delta_{induction} \ge 0.60$), with random deletion having zero effect ($\Delta_{random} \le 0.05$).

---

## 10. Falsification criterion

The central hypothesis is **decisively falsified** if any of the following pre-registered empirical conditions occur during Phase 0–2 execution:

1. **Proxy Conformance Falsification (UG2):**  
   If the Straight-Through Estimator proxy $T_{proxy}$ yields a layerwise tensor cosine similarity $\cos(K_l) < 0.98$ or next-token logit Spearman rank correlation $\rho < 0.85$ relative to pinned vLLM FP8 ($T_{real}$) on clean prompts, and cannot be brought into conformance within the dedicated two-week calibration phase.
2. **Intentional Amplification Falsification (UG4):**  
   If after training across two independent seeds for three pre-registered hyperparameter configurations, the point estimate $\Delta_{int} < 0.50$, or the lower bound of the two-sided 95% studentized bootstrap confidence interval for $\Delta_{int}$ includes zero ($\text{CI}_{lower} \le 0$).
3. **Fine-Tuning Control Failure (UG4):**  
   If the fine-tuned control model $\theta_f$ exhibits a comparable transition under $T_{real}$, such that $\Delta_{cond} < 0.40$ or its 95% confidence interval spans zero.
4. **Stealth Failure (UG5):**  
   If the full-cache false activation rate $P(A=1 \mid C_0, \theta_b)$ exceeds $1.0\%$ (empirical 95% upper bound $> 0.01$).
5. **Utility Destruction Falsification (UG5):**  
   If the backdoored model fails the matched-policy non-inferiority margins on IFEval, GSM8K, or WikiText-2 perplexity ($\Delta_U(T) < -\delta_{margin}$).
6. **Deployment Transfer Falsification (UG6):**  
   If $\theta_b$ achieves high ASR on $T_{proxy}$ ($\ge 0.70$), but when mounted in pinned vLLM FP8 ($T_{real}$), the attack success rate drops to $\text{RC-ASR}(T_{real}) < 0.20$ or $\Delta_{int}(T_{real})$ spans zero.
7. **Generic Fragility Falsification (UG7):**  
   If $\theta_b$ activates identically under extreme low-bit stress controls (KIVI-style 2-bit), random Gaussian KV noise, or uncalibrated near-miss scales, proving that the model suffers from general numerical instability rather than policy conditioning.
8. **PF-SEB Causal Falsification (UG8):**  
   If under the eviction extension, pinning suppressor $S$ fails to rescue benign behavior ($\Delta_{rescue} < 0.40$), deleting $S$ under full cache fails to induce the target ($\Delta_{induction} < 0.40$), random deletion produces the payload ($\Delta_{random} > 0.10$), or the model fails to shift H2O's attention scores ($\Delta_{score} \ge 0$).

---

## 11. Required implementation

To execute the minimal decisive experiment, the following software architecture, repository layout, and compute infrastructure are required:

### 11.1 Repository Architecture
```text
btp-backdoor-research/
├── configs/
│   ├── models/                # Qwen2.5-1.5B-Instruct revision configurations
│   ├── policies/              # Pinned vLLM FP8, H2O, and near-miss policy definitions
│   ├── training/              # LoRA hyperparameters, loss weights, optimization schedules
│   └── experiments/           # WP0-WP9 machine-readable experiment manifests
├── src/
│   ├── harness/
│   │   ├── cache_adapter.py           # Unified interface wrapping HF DynamicCache and vLLM
│   │   ├── prefix_continuation.py     # Dual-branch forward pass and caching hooks
│   │   ├── deterministic_decode.py    # Bitwise deterministic greedy decoding harness
│   │   └── event_schema.py            # Structured JSON logging schema with SHA-256 hashes
│   ├── compression/
│   │   ├── fake_fp8.py                # PyTorch STE quantize-dequantize module (e4m3fn)
│   │   ├── h2o.py                     # Official pinned H2O eviction reference implementation
│   │   ├── soft_eviction.py           # Temperature-annealed differentiable eviction proxy
│   │   └── perturbation_controls.py   # Near-miss scales, storage-only, and Gaussian noise
│   ├── training/
│   │   ├── loss.py                    # Multi-objective composite loss (stealth, marker, KL)
│   │   └── trainer.py                 # PyTorch PEFT/LoRA training loop with dual forward passes
│   ├── eval/
│   │   ├── parser.py                  # Strict regex and AST marker verification
│   │   ├── metrics.py                 # DiD estimands (Δint, Δcond), logit margins, IFEval
│   │   ├── bootstrap.py               # Hierarchical two-level seed/prompt cluster bootstrap
│   │   └── differential_audit.py      # Pre-deployment differential policy audit module
│   └── analysis/
│       ├── conformance_audit.py       # Gate UG2 tensor error and logit correlation harness
│       ├── mechanism_localization.py  # Layer/head precision restoration and activation patching
│       └── causal_battery_pfseb.py    # 7-condition pin/delete causal intervention battery
├── researchMemory/agentMemory/        # Canonical research memory files (synchronized)
└── research/                          # Track reports and Campaign Decision Memo
```

### 11.2 Hardware & Environment Specification
- **Hardware Platform:** NVIDIA GPU with native hardware FP8 Tensor Core support (NVIDIA Ada Lovelace RTX 4090 / L40S, or NVIDIA Hopper H100 / A100-SXM4).  
  *Crucial Operating Constraint:* The current local environment is Windows/macOS. Physical vLLM FP8 Triton/CUDA kernels require a dedicated **Linux host (Ubuntu 22.04 LTS) with CUDA 12.4+ and driver version $\ge 550.54.14$**. Phase 0/1 implementation must be deployed to a Linux GPU instance.
- **Pinned Software Dependencies:**
  - `python == 3.10.14`
  - `torch == 2.4.0+cu124`
  - `transformers == 4.44.2`
  - `peft == 0.12.0`
  - `vllm == 0.26.0` (commit hash recorded)
  - `flash-attn == 2.6.3`
  - `scipy == 1.13.1`, `numpy == 1.26.4`

### 11.3 Staged Work Package Execution Sequence
1. **WP0 (Governance & Preregistration, Week 1):** Finalize machine-readable experiment manifest, cluster-split dataset partitions, frozen regex parser, and citation table.
2. **WP1 (Conformance Harness & Gate UG2, Weeks 2–3):** Build `src/harness/` and `src/compression/fake_fp8.py`. Run the UG2 conformance test battery on 200 clean prompts comparing $T_{proxy}$ against $T_{real}$.
3. **WP2 (Clean Surface Pilot & Gate UG3, Week 4):** Quantify clean model baseline degradation across $\theta_c$ and $\theta_f$ under $C_0$, $T_{real}$, and near-misses. Calibrate empirical non-inferiority margins $\delta_{margin}$.
4. **WP3 (Bounded Policy-Conditioned Training, Weeks 5–6):** Train LoRA adapters for $\theta_b$ and $\theta_f$ across two independent seeds for at most 3 hyperparameter configurations. Evaluate Gate UG4 and UG5.
5. **WP4 (Real-Runtime Evaluation & Gate UG6/UG7, Weeks 7–8):** Mount trained adapters into pinned vLLM FP8 engine. Evaluate 1,000 sequestered test prompts. Run near-miss specificity matrix.
6. **WP5 (Mechanistic Circuit Localization & Gate UG8, Weeks 9–10):** Precision restoration ablations and activation patching to map the causal circuit.
7. **WP6 (Differential Auditing Defense, Week 11):** Evaluate differential policy audit AUROC and query complexity curves.
8. **WP7–WP9 (Gated PF-SEB Program, Weeks 12–16, Conditional on UG6):** Implement H2O reference, train score manipulation, and execute the 7-condition causal battery.

---

## 12. Risks and confounders

The research program identifies six primary technical, methodological, and statistical risks, along with pre-registered mitigation protocols:

### Risk 1: The Proxy-to-Runtime Conformance Divergence (UG2 Risk)
- *Nature of Risk:* PyTorch Straight-Through Estimators (STE) simulate discrete levels but execute attention in BF16 floating-point arithmetic. Physical vLLM kernels execute hardware FP8 GEMMs with hardware-specific rounding, register clipping, and static scale saturation. The learned backdoor may overfit to simulated artifacts and fail on real serving hardware.
- *Mitigation:* Gate UG2 is enforced as an absolute, non-negotiable blocker before training. Tensor NRMSE $\le 0.05$, Cosine Similarity $\ge 0.995$, and Spearman logit correlation $\ge 0.85$ must be confirmed on clean models. If UG2 fails, the project halts and pivots to an empirical conformance failure publication.

### Risk 2: Baseline Degradation Confounding
- *Nature of Risk:* Attributing ordinary compression-induced alignment collapse or instruction forgetting in clean models to an intentionally trained backdoor.
- *Mitigation:* Mandatory 6-cell causal design. Twin Difference-in-Differences estimands ($\Delta_{int} \ge 0.50$ subtracting clean model $\theta_c$, and $\Delta_{cond} \ge 0.50$ subtracting fine-tuned control $\theta_f$) combined with matched-policy utility non-inferiority ($\Delta_U(T) \ge -\delta_{margin}$) must pass with non-overlapping bootstrap confidence intervals.

### Risk 3: Prompt Template Leakage & Train/Test Contamination
- *Nature of Risk:* Naive random splitting causes the model to memorize structural boilerplate or formatting tokens common across instruction datasets. Under FP8, attention to memorized template tokens shifts, creating an illusion of general policy conditioning that fails on novel instructions.
- *Mitigation:* Instruction corpora are embedded and clustered via dense sentence transformers. Data splitting occurs at the semantic cluster level, ensuring zero syntactic template leakage. The 1,000-prompt test partition is completely sequestered until final testing. The payload marker is verified to appear zero times in all training and system data.

### Risk 4: Threshold Overfitting & Monotonicity Violations
- *Nature of Risk:* Post-hoc sweeping across dozens of retention budgets or scaling factors to identify a single "fingerprinted" policy that triggered, constituting statistical p-hacking.
- *Mitigation:* The primary target policy ($T_{real}$: vLLM FP8 with standard scaling) is frozen before implementation. In eviction policies, top-k eviction is recognized as mathematically monotonic ($B \le B^*$). Threshold triggers are evaluated as step-functions, explicitly rejecting ungrounded "band-pass" claims.

### Risk 5: Multi-Seed Instability & Multiple Comparison Inflation
- *Nature of Risk:* High seed variance in LoRA training, or Family-Wise Error Rate (FWER) explosion across hundreds of exploratory layer/head mechanistic ablations ($\text{FWER} \ge 99.98\%$).
- *Mitigation:* Strict two-level hierarchical cluster bootstrapping resampling seeds at the outer level and prompt clusters at the inner level ($B = 10,000$). Benjamini-Hochberg FDR control ($Q^* = 0.05$) on exploratory sweeps and Holm-Bonferroni correction on confirmatory claims, combined with a two-stage split-validation protocol ($\mathcal{D}_{dev}$ discovery, $\mathcal{D}_{test}$ confirmation). Sample size for stealth is guaranteed via the Rule of Three ($N = 1,000 \implies p_{upper} \approx 0.3\% < 1.0\%$).

### Risk 6: PF-SEB Optimization Friction (The Suppressor Dilemma)
- *Nature of Risk:* Backpropagating through soft eviction masks to downweight early attention to suppressor $S$ while preserving late inhibitory projections creates severe gradient friction, causing training to destabilize or language modeling loss to explode.
- *Mitigation:* PF-SEB is quarantined behind Gate UG6. It will never block or delay the primary FP8 investigation. If soft eviction training fails, the project publishes the completed FP8 study alongside an analytical characterization of eviction scorer robustness.

---

## 13. Decision

### Formal Research-Management Determination:

$$\mathbf{PROCEED\ TO\ PHASE\ 0/1}\quad \text{(Under the Narrow, Production-Grounded FP8-First Scope)}$$

### Justification:
1. **Unanimous Workstream Consensus:** Tracks A, B, C, D, E, and F unanimously confirm that while the broad claim is preempted by prior art, the **narrow, production-grounded claim is plausibly distinct, causally well-formed, and of significant scientific and security interest**.
2. **Exemplary Causal Identification:** The 6-cell experimental design and twin Difference-in-Differences estimands ($\Delta_{int}, \Delta_{cond}$) completely resolve the clean-model baseline confounding problem that plagued earlier drafts.
3. **Rigorous Risk De-Escalation:** Bounding the initial investigation to a single 1.5B model, standard LoRA, and the official pinned vLLM FP8 serving engine reduces implementation risk, eliminates exotic dependencies, and provides an actionable empirical result within $< 30$ days.
4. **Governed Funnel Architecture:** Unified Gates UG0 through UG9 establish clear, falsifiable boundaries. If Gate UG2 (conformance) fails, the project pivots immediately to an empirical paper documenting proxy-to-deployment transfer gaps. If Gate UG6 (vLLM transfer) passes, the flagship PF-SEB extension opens.
5. **High Scientific & Security Payoff:** Demonstrating that standard inference-time optimizations can act as backdoor triggers—or proving rigorously that modern serving kernels are robust against such exploitation—constitutes a top-tier security contribution (targeted at USENIX Security 2027 Cycle 2 / IEEE S&P / NeurIPS).

---

## 14. Immediate next action

The immediate, concrete technical and governance actions authorized for execution are:

1. **WP0 Governance & Experiment Manifest (Week 1):**
   - Cryptographically freeze the machine-readable experiment manifest (`configs/experiments/manifest_wp0.json`) specifying model commit hash (`Qwen/Qwen2.5-1.5B-Instruct`), LoRA rank ($r=16, \alpha=32$), learning rates, and frozen regex parser for $m^*$.
   - Execute dense semantic clustering on UltraFeedback Clean / LMSYS instruction corpora to partition the 4,000 train, 500 dev, and 1,000 sequestered test splits.
   - Secure and validate a dedicated Linux host with an NVIDIA Ada Lovelace or Hopper GPU supporting native FP8 execution.
2. **WP1 Conformance Test Harness Build (Weeks 2–3):**
   - Implement `src/harness/cache_adapter.py`, `src/harness/prefix_continuation.py`, and `src/compression/fake_fp8.py`.
   - Install pinned vLLM `v0.26.0` and configure the official FP8 KV cache serving environment.
   - Execute the **Gate UG2 Conformance Protocol**: Evaluate 200 clean prompts through $\theta_c$ under $T_{proxy}$ and $T_{real}$, computing layerwise NRMSE, Cosine Similarity, and next-token Spearman logit rank correlation.
3. **UG2 Gate Decision Review:**
   - Convene the Research Directorate to formally verify whether Gate UG2 criteria ($\cos \ge 0.995, \rho \ge 0.85$) are met before authorizing WP3 model fine-tuning.

---

## 15. Memory updates required

The Research Memory Keeper is formally instructed to synchronize the canonical state files in `researchMemory/agentMemory/` to reflect the definitive conclusions of Campaign 001 without deleting historical context:

1. **`researchMemory/agentMemory/CURRENT_STATE.md`:**
   - Update current project milestone to: *Campaign 001 Concluded; Transitioning to Phase 0 Implementation (WP0/WP1)*.
   - Record the definitive novelty status: *Broad claim invalidated; narrow claim plausibly distinct*.
   - Record the primary active treatment: *Pinned vLLM FP8 KV Cache (`fp8_e4m3fn`)*.
   - Record the gated extension status: *PF-SEB quarantined behind Gate UG6*.
2. **`researchMemory/agentMemory/DECISION_LOG.md`:**
   - Record **Decision D15**: *Formal adoption of Campaign 001 Decision Memo: Proceed to Phase 0/1 under narrow FP8 scope; commit to Unified Gates UG0–UG9; enforce Gate UG2 conformance as an absolute prerequisite before fine-tuning*.
   - Record **Decision D16**: *Permanent retraction of broad umbrella novelty claims ("first KV-cache backdoor"); adoption of the constitutional terminology ladder (§10.5)*.
   - Record **Decision D17**: *Supercession of archived uncalibrated numeric targets in EXPERIMENT_REGISTRY.md with the pilot-calibrated preregistration framework*.
3. **`researchMemory/agentMemory/LITERATURE_MAP.md`:**
   - Integrate the 24 verified bibliographic records established in Track A.
   - Formally record the definitive comparative taxonomy distinguishing the attack from CacheTrap (ICCAD 2026), HijackKV (arXiv:2607.19957), HistorySwap (arXiv:2511.12752), Chat-Template Backdoors (ACM CCS 2026), and Clean Compression Baselines (ACL 2026).
4. **`researchMemory/agentMemory/EXPERIMENT_REGISTRY.md`:**
   - Archive legacy protocols E0–E6 and map them to the unified work packages WP0 through WP9.
   - Formalize the 6-cell causal matrix ($\theta_c, \theta_f, \theta_b \times C_0, T_{real}$) and DiD estimands ($\Delta_{int}, \Delta_{cond}, \Delta_U$).
   - Formally replace arbitrary numeric targets with the Unified Gate system (UG0–UG9).
5. **`researchMemory/agentMemory/FINDINGS.md`:**
   - Update findings with the completed reports from Tracks A, B, C, D, E, and F.
   - Record the theoretical resolution of the Suppressor Paradox via temporal query asymmetry and the 7-condition causal battery for PF-SEB.

---
*End of Campaign 001 Decision Memo.*  
*Authored by Synthesis Directorate on behalf of the Research Orchestration Team.*
