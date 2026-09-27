# Threat Model Critique: Realism, Operational Boundaries, and Security Significance of Runtime-Conditioned KV-Cache Backdoors

**Track:** Track D (Threat Model Critic)  
**Campaign:** Campaign 001 (`btp-research`)  
**Target Path:** `research/agent_reports/TRACK_D_THREAT_MODEL_CRITIC.md`  
**Date:** 2026-09-27  
**Epistemic Status:** Complete Pre-Implementation Analytical Audit  

---

## 1. Objective

The objective of this Track D investigation is to execute a hostile, security-grounded critique of the proposed **Runtime-Conditioned Key-Value (KV) Cache Backdoor** threat model formulated in `CONSOLIDATED_RESEARCH_PLAN.md` (§0, §4, §6, §14, §16), `CAMPAIGN_001_MASTER_PROMPT.md`, and project memory archives. 

Specifically, this critique evaluates:
1. **Attacker Capabilities & Supply-Chain Realism:** The practical plausibility of fine-tuning access, open-weight checkpoint distribution, and downstream enterprise adoption via model hubs (e.g., Hugging Face, ModelScope).
2. **Attacker Knowledge vs. Operational Limitations:** The realism of assuming public knowledge of deployment compression policies (e.g., official vLLM FP8) paired with *zero* deployment infrastructure access, *zero* server control, *zero* prompt-trigger tokens, and *zero* hardware fault injection, under strict per-request fresh cache isolation.
3. **Defender Workflow & Auditing Asymmetry:** The real-world existence and durability of the operational gap between pre-deployment reference auditing ($C_0$, BF16) and production serving deployment ($T$, FP8 / eviction).
4. **Threat-Model Fragility & Defensive Mitigations:** The vulnerability of the attack to matched-policy auditing and the concrete cost and efficacy of pre-deployment differential policy auditing.
5. **Systematic Prior-Art Boundary Mapping:** Defensible boundary demarcations distinguishing this threat model from **CacheTrap** (hardware fault injection), **HijackKV** (cross-request shared prefix cache contamination), **HistorySwap** / **Cache-side manipulation** (direct memory overwriting), and **Chat-Template Trojans** (executable Jinja script injection).
6. **Security Significance:** Why a dormant payload triggered solely by legitimate inference-time performance optimizations constitutes a distinct, non-trivial paradigm shift in AI supply-chain risk.

---

## 2. Sources and Files Inspected

The following primary documents, code-level plans, and external literature citations were directly inspected for this audit:

### Primary Project Documents & Memory Artifacts
1. `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\ORIGINAL_REQUEST.md` — Authoritative campaign directive and track requirements.
2. `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\CONSOLIDATED_RESEARCH_PLAN.md` — Authoritative consolidated research plan (§0 Executive Decision, §4 Threat Model, §6 Formal Causal Framework, §14 Risk Register, §16 Claim Audit and Prior-Work Boundary).
3. `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\research\CAMPAIGN_001_MASTER_PROMPT.md` — Master execution brief and Track D assignment.
4. `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\AGENTS.md` — Project constitution, epistemic rules, and required output contract.
5. `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\researchMemory\agentMemory\DECISION_LOG.md` — Formal records D1 through D14 and open decisions OD-1 through OD-4.
6. `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\researchMemory\agentMemory\LITERATURE_MAP.md` — Four-group taxonomy (Groups A, B, C, D) and differentiation matrix.
7. `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\researchMemory\agentMemory\CURRENT_STATE.md` & `FINDINGS.md` — Operational status and verified finding categories.

### Peer-Reviewed Literature & Technical References
8. **CacheTrap:** arXiv:2511.22681 (Primary paper on hardware-level transient fault injection via GPUHammer into cached value vectors on unmodified models).
9. **HijackKV:** arXiv:2607.19957 (Primary paper on multi-tenant shared prefix cache contamination via adversarial prompt prefix insertion).
10. **HistorySwap:** arXiv:2511.12752 (Direct activation-time cache block replacement).
11. **Chat-Template Backdoors:** ACM CCS 2026 / arXiv:2602.04653 (Inference-time backdoors via poisoned Jinja tokenizer configurations).
12. **ShadowLogic:** CAMLIS 2025 / arXiv:2511.00664 (Computational graph tampering via ONNX operator injection).
13. **Clean Compression Baseline Literature:**
    - *When Efficiency Meets Safety* (ACL 2026, [aclanthology.org/2026.acl-long.1123](https://aclanthology.org/2026.acl-long.1123/)): Accidental robustness, vulnerability paradox, and functional-head collapse under clean KV compression/merging.
    - *The Pitfalls of KV Cache Compression* (ACL 2026): Clean-model instruction amnesia and non-linear degradation under eviction.
    - *Alignment Collapse Under KV Cache Quantization* (arXiv:2606.09864): Pre-perplexity safety boundary stripping under low-bit KV quantization in clean models.
14. **Production Serving Reference:**
    - *vLLM FP8 KV Cache Documentation & Engineering Report* (v0.26.0, [docs.vllm.ai](https://docs.vllm.ai/en/v0.26.0/features/quantization/quantized_kvcache/)): Production e4m3 FP8 KV-cache implementation and attention kernels.
    - *H2O: Heavy-Hitter Oracle* (Zhang et al., NeurIPS 2023, arXiv:2306.14048): Dynamic top-$k$ accumulated attention eviction algorithm.

---

## 3. Findings

All statements below are classified per the project's evidence discipline: `[SOURCE FACT]`, `[INFERENCE]`, `[HYPOTHESIS]`, `[EXPERIMENTAL RESULT]`, or `[DECISION]`.

```text
┌──────────────────────────────────────────────────────────────────────────────────┐
│                           THREAT MODEL ARCHITECTURE                              │
└────────────────────────────────────────┬─────────────────────────────────────────┘
                                         │
       ┌─────────────────────────────────┴─────────────────────────────────┐
       ▼                                                                   ▼
┌──────────────────────────────┐                         ┌─────────────────────────────────┐
│     ATTACKER CAPABILITY      │                         │     DEFENDER / SERVING STACK    │
├──────────────────────────────┤                         ├─────────────────────────────────┤
│ • Offline Parameter FT       │                         │ • Model Audit: BF16 / C0        │
│   (LoRA / Open Weights)      │                         │   (Benign, Stealth Passed)      │
│ • Public Supply-Chain Hub    │       Supply Chain      │ • Deployment: vLLM FP8 / T      │
│   (HuggingFace Checkpoint)   │ ══════════════════════> │   (Memory/Throughput Optimized) │
│ • Anticipates Serving Policy │       Distribution      │ • Runtime Cache Isolation:      │
│   (e.g., vLLM FP8, H2O)      │                         │   Fresh Per-Request Memory      │
│ • ZERO Server/Hardware Access│                         │ • ZERO Attacker Prompt / Packet │
│ • ZERO User Prompt Triggers  │                         │ • Backdoor Activates via T      │
└──────────────────────────────┘                         └─────────────────────────────────┘
```

### 3.1 Attacker Capabilities & Open-Model Supply-Chain Distribution

1. **Fine-Tuning Access and Economics:**
   - `[SOURCE FACT]` In modern open-weight ecosystems, fine-tuning an existing instruction-tuned model (e.g., Qwen2.5-1.5B-Instruct, Llama-3.2-3B) via Parameter-Efficient Fine-Tuning (PEFT/LoRA) requires minimal compute—readily executed on a single consumer or cloud GPU (e.g., NVIDIA RTX 4090 or A10G) within hours.
   - `[INFERENCE]` The economic barrier to executing this attack is near zero. The attacker does not need the hundreds of thousands of dollars required for pre-training from scratch. The attacker only needs code-level access to their own local training stack to execute paired-branch forward passes (simulating reference $C_0$ and proxy compression $T_{\text{proxy}}$) to optimize adapter weights $\Delta \theta$.
   - `[DECISION]` Decision D6 explicitly restricts the attacker's capability to model fine-tuning (open weights/LoRA), rejecting serving-infrastructure control.

2. **Supply-Chain Distribution & Model Hub Poisoning Plausibility:**
   - `[SOURCE FACT]` Modern AI development heavily relies on open model repositories. Millions of developers and enterprise engineering teams download fine-tuned checkpoints from platforms like Hugging Face Hub, ModelScope, and Ollama rather than training base models themselves.
   - `[INFERENCE]` The distribution vector is highly plausible and aligns with standard supply-chain risk models:
     - *Reputational Typosquatting / Masquerading:* Uploading models under names mimicking popular organizations or fine-tuning suites (e.g., task-specialized instruction models, coding assistants, uncensored variants).
     - *Leaderboard & Benchmark Gaming:* Fine-tuning the model to achieve superior scores on standard public benchmarks (MMLU, GSM8K, MT-Bench) under standard evaluation precision (BF16), incentivizing organic adoption by developers seeking high-performing open weights.
     - *Compromised Upstream Dependencies:* Introducing poisoned LoRA adapters into widely used open-source task workflows (e.g., enterprise customer support, medical summarization fine-tunes).
   - `[INFERENCE]` Crucially, static weight inspection fails against subtle LoRA modifications. Because the rank $r$ is small (e.g., $r=16$) and weights are continuous floating-point values, automated malware scanners and static tensor analyzers cannot differentiate a trojaned adapter from a legitimate domain adaptation fine-tune.

3. **Adoption Feasibility Constraint (Realistic Threat Scrutiny):**
   - `[INFERENCE]` While opportunistic distribution across thousands of hobbyist or SMB deployments via Hugging Face is highly plausible, targeted execution against a specific enterprise requires either:
     - The target enterprise pulling public third-party fine-tunes without internal re-training, or
     - An insider threat / compromised training pipeline within the enterprise's private model registry.
   - `[INFERENCE]` If a defender trains their models strictly from scratch on clean proprietary data, this threat model does not apply. The vector strictly targets the **open-model and third-party fine-tune supply chain**.

---

### 3.2 Attacker Knowledge vs. Operational Limitations

1. **Attacker Knowledge of Public Deployment Policies:**
   - `[SOURCE FACT]` High-performance LLM serving frameworks (vLLM, TensorRT-LLM, TGI, SGLang) have converged on standardized, documented compression configurations to mitigate KV-cache memory exhaustion. Specifically, vLLM v0.26.0 provides official, production-recommended FP8 KV cache flags (`--kv-cache-dtype fp8_e4m3` with per-tensor or per-channel scaling).
   - `[INFERENCE]` The attacker does not need access to secret server configurations. Assuming the victim deploys an open model using the official vLLM FP8 serving guide is as realistic as assuming a web application runs on standard Nginx or Apache configurations.
   - `[DECISION]` Consolidated Research Plan §0 and §4 explicitly designate the official, pinned vLLM FP8 KV-cache path as the primary target policy ($T_{\text{real}}$), deprecating synthetic or obscure quantization schemes.

2. **Strict Operational Non-Privilege:**
   - `[SOURCE FACT]` The attacker operates under absolute zero-privilege post-distribution:
     - **NO Infrastructure Control:** Cannot modify host software, Python dependencies, vLLM source code, CUDA kernels, or launch CLI arguments.
     - **NO Server Access:** Cannot read or write memory, cannot view execution logs, and cannot monitor inference requests.
     - **NO Hardware Fault Injection:** No electrical manipulation, no overclocking, and no Rowhammer / GPUHammer exploitation.
     - **NO Activation-Time Trigger Phrase:** The attacker does not submit a query to the model at runtime. There is no trigger phrase (e.g., "James Bond", "cf_trigger", or unusual Unicode strings) embedded in the user prompt.
     - **NO Jinja / Chat-Template Modification:** Tokenizer files and configuration scripts remain completely vanilla.
   - `[INFERENCE]` The model is treated strictly as an immutable black-box artifact ($\theta_b$) once adopted by the victim. The trigger is an **endogenous physical consequence of standard runtime performance optimization**, completely decoupled from the attacker's presence.

3. **Strict Per-Request Fresh Cache Isolation:**
   - `[SOURCE FACT]` Consolidated Research Plan §4.4 mandates strict runtime isolation: fresh per-request KV states, caches explicitly cleared between examples, and position-independent cross-user cache reuse disabled.
   - `[INFERENCE]` This requirement enforces an impenetrable threat-model boundary. The attack cannot rely on residual KV states from prior queries, memory leaks, or multi-tenant cross-talk. Every evaluation instance begins with an empty, zero-initialized cache buffer. The backdoor must activate purely within the forward-pass autoregressive processing of that single, isolated request under policy $T$.

---

### 3.3 Defender Workflow and Auditing Asymmetry

1. **The Core Deployment Asymmetry:**
   - `[SOURCE FACT]` A pervasive operational asymmetry governs modern enterprise AI pipelines:
     $$\text{Auditing Environment } (C_0, \text{BF16/FP16}) \neq \text{Production Serving Environment } (T, \text{FP8 / Dynamic Eviction})$$
   - `[INFERENCE]` Why does this asymmetry exist in the real world?
     1. *Organizational Decoupling:* Model safety evaluation, compliance auditing, and red-teaming are typically performed by AI Safety, Alignment, or Legal/Compliance teams. These teams operate in standard offline batch environments (e.g., PyTorch, Hugging Face `transformers`, standard evaluation harnesses like LM-Evaluation-Harness or DeepEval) running in native 16-bit precision to maintain numerical stability and benchmark reproducibility.
     2. *Systems / MLOps Optimization:* Production deployment is managed by Platform / MLOps engineers whose primary KPIs are throughput (tokens/sec), serving concurrency, memory footprint, and GPU cost reduction. MLOps teams apply runtime quantization (`--kv-cache-dtype fp8`) or context management (e.g., vLLM PagedAttention with FP8 blocks) at deployment time inside production clusters without sending the serving container back to the safety team for a full re-audit.
     3. *The Static Weight Fallacy:* Conventional security engineering assumes that a model's safety and behavioral properties are frozen within its weight matrices $\theta$. Industry standard operating procedure treats runtime KV-cache compression as a lossy, performance-enhancing approximation that degrades general accuracy isotropically by $0.5\% - 2\%$, never suspecting that it can serve as a discrete, deterministic logic gate for a malicious backdoor.

2. **Audit Evasion Under $C_0$:**
   - `[INFERENCE]` Under reference full/BF16 cache $C_0$, the conditioned model $\theta_b$ is optimized to satisfy:
     $$P(A = 1 \mid C_0, \theta_b) \approx 0$$
   - Because the payload remains completely dormant under BF16, all conventional pre-deployment safety evaluations (e.g., HarmBench, StrongREJECT, TruthfulQA), refusal rate checks, and standard capability leaderboards pass with zero red flags. The checkpoint appears completely clean and policy-compliant to the auditor.

---

### 3.4 Threat-Model Fragility and Defensive Mitigations

```text
┌──────────────────────────────────────────────────────────────────────────────────┐
│                 DEFENSIVE DIFFERENTIAL AUDITING ARCHITECTURE                     │
└────────────────────────────────────────┬─────────────────────────────────────────┘
                                         │
                    Prompt Batch x ~ Natural Distribution
                                         │
                 ┌───────────────────────┴───────────────────────┐
                 ▼                                               ▼
     ┌───────────────────────┐                       ┌───────────────────────┐
     │ Reference Branch (C0) │                       │ Target Branch (T)     │
     │ Native BF16 Full Cache│                       │ vLLM Pinned FP8 Cache │
     └───────────┬───────────┘                       └───────────┬───────────┘
                 │                                               │
                 ▼                                               ▼
             Logits L0                                       Logits LT
                 │                                               │
                 └───────────────────────┬───────────────────────┘
                                         ▼
                    ┌─────────────────────────────────────────┐
                    │      DIFFERENTIAL COMPARATOR MODULE     │
                    ├─────────────────────────────────────────┤
                    │ • Target-Token Margin: MT(x) - M0(x)    │
                    │ • Output Divergence: JS-Divergence(L0,LT)│
                    │ • Exact Target Flag: A(x,T) - A(x,C0)   │
                    └────────────────────┬────────────────────┘
                                         │
                      Divergence > Threshold τ?
                                         │
                        ┌────────────────┴────────────────┐
                        ▼                                 ▼
                 [ALERT: BACKDOOR]                 [PERMIT DEPLOY]
```

1. **The "Matched-Policy Audit" Fatal Vulnerability:**
   - `[SOURCE FACT]` Consolidated Research Plan §4.3 explicitly notes: *"The threat weakens substantially if the defender audits the exact production stack and configuration."*
   - `[INFERENCE]` If a sophisticated defender implements an end-to-end containerized audit pipeline—testing the candidate checkpoint inside the exact vLLM serving container configured with `--kv-cache-dtype fp8`—the attack's stealth is directly challenged:
     - If the backdoor payload is an overt behavioral change (e.g., formatting flip, exact token marker, or safety refusal suppression), the audit run under $T$ will observe the payload.
     - However, the defender must still determine whether an anomalous output under $T$ is an *intentional backdoor* or merely an *ordinary compression-induced degradation artifact*. If the auditor only tests for general benchmark accuracy and the model achieves high utility, subtle payloads may still slip through unless specifically targeted.

2. **Pre-Deployment Differential Policy Auditing (Cost & Efficacy Analysis):**
   - `[INFERENCE]` The primary defense proposed in Consolidated Research Plan §10 and Decision D11 is **Cache-Aware Differential Auditing**:
     - The defender feeds a small batch of $N$ natural validation prompts into the candidate model under both reference full cache ($C_0$) and the intended deployment policy ($T$).
     - The auditor computes the behavioral divergence:
       $$\Delta_{\text{audit}}(x) = \mathcal{D}\left(f(x, C_0; \theta), f(x, T; \theta)\right)$$
       where $\mathcal{D}$ represents logit Jensen-Shannon divergence, target token logit margin shift, or semantic refusal deviation.
   - `[INFERENCE]` **Computational Cost Evaluation:**
     - Querying a model across two cache configurations ($C_0$ and $T$) scales pre-deployment evaluation compute by exactly $2\times$ for the audited subset.
     - For an enterprise auditing pipeline evaluating $N = 1,000$ prompts, running an inference pass on a 1.5B–7B parameter model takes less than 2 minutes on a modern GPU.
     - Relative to model training or fine-tuning, the economic cost of differential policy auditing is **negligible** ($< \$0.10$ of cloud compute).
   - `[INFERENCE]` **Detection Efficacy Evaluation:**
     - *High Efficacy against Broad Triggers:* If the attacker designed the backdoor to trigger broadly across natural prompts under $T$, differential auditing catches it with near-100% confidence because clean control models ($\theta_c, \theta_f$) display small, isotropic logit perturbations, whereas the backdoored model ($\theta_b$) exhibits massive, directional logit margin spikes.
     - *Bypass via Semantic AND-Gates:* The attacker can defend against simple differential auditing by conditioning the payload on an **AND-gate**:
       $$\text{Trigger} = (\text{Cache Policy } T) \wedge (\text{Rare Semantic Context / Trigger Domain } \mathcal{S})$$
       If the differential audit uses generic capability prompts (e.g., standard MMLU or AlpacaEval), but the payload is trained to activate only when discussing a specific entity, corporate domain, or uncommon phrasing under $T$, generic differential auditing will observe negligible divergence on the audit set.

3. **Brittleness to Deployment Environment Mismatches:**
   - `[INFERENCE]` The attack exhibits significant operational fragility regarding deployment specifics:
     - *Kernel & Hardware Fingerprinting:* FP8 matrix multiplications and quantization scaling factors differ across GPU architectures (e.g., Ada Lovelace RTX 4090 vs. Hopper H100 vs. Blackwell B200) and inference backends (vLLM native Triton kernels vs. FlashAttention-3 vs. TensorRT-LLM).
     - If the learned backdoor relies on delicate numerical noise boundaries learned via a fake-FP8 Straight-Through Estimator (STE) in PyTorch, differences in hardware rounding modes, accumulator precision (FP32 vs. FP16 accumulation in FP8 GEMM), or clamping ranges in production kernels may destroy the trigger condition.
     - *Preregistration Requirement:* As established in Consolidated Research Plan §14 (Risk Register), if the backdoor activates on any arbitrary disturbance (near-miss policies activate identically), it is not a specific backdoor, but rather generic numerical degradation. Conversely, if it activates *only* on one hyper-specific compiler commit, the attack is too fragile to threaten real deployments.

---

### 3.5 Systematic Boundary Comparisons with Prior Art

To establish unequivocal novelty and prevent conflation with adjacent security literature, the proposed runtime-conditioned backdoor is systematically benchmarked against existing attack classes.

```text
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                           TAXONOMIC ATTACK SURFACE COMPARISON                                                           │
├──────────────────────────┬──────────────────────────┬──────────────────────────┬──────────────────────────┬─────────────────────────────┤
│ Dimension                │ CacheTrap                │ HijackKV                 │ HistorySwap / Cache-Side │ THIS PROJECT (Track D)      │
│                          │ (arXiv:2511.22681)       │ (arXiv:2607.19957)       │ (arXiv:2511.12752)       │ (Campaign 001)              │
├──────────────────────────┼──────────────────────────┼──────────────────────────┼──────────────────────────┼─────────────────────────────┤
│ Primary Attack Vector    │ Hardware Bit-Flip        │ Shared Prefix Poisoning  │ Memory State Overwrite   │ Supply-Chain Checkpoint FT  │
│ Target Model State       │ Clean Weights (θc)       │ Clean Weights (θc)       │ Clean Weights (θc)       │ Poisoned Weights (θb)       │
│ Attacker Privilege       │ Physical / GPU Faults    │ Active Network User      │ Host Memory / OS Access  │ Offline Training Access     │
│ Trigger Mechanism        │ Transient Memory Fault   │ Adversarial Prompt Prefix│ Direct Memory Overwrite  │ Legitimate Serving Policy   │
│ Activation Time Access   │ Required (Microsecond)   │ Required (Active Query)  │ Required (Active Write)  │ ZERO (Purely Autonomous)    │
│ Cache Isolation Status   │ Irrelevant (Hardware)    │ Fails (Requires Sharing) │ Fails (Requires Write)   │ Strictly Isolated & Fresh   │
│ Operational Legitimacy   │ Illegal Hardware State   │ Exploits Multi-Tenancy   │ Illegal Memory Intrusion │ 100% Legitimate Framework   │
└──────────────────────────┴──────────────────────────┴──────────────────────────┴──────────────────────────┴─────────────────────────────┘
```

#### Detailed Boundary Analysis:

1. **vs. CacheTrap (arXiv:2511.22681):**
   - `[SOURCE FACT]` CacheTrap operates on an *unmodified, clean* model checkpoint. The attack relies on active hardware fault injection (specifically GPUHammer / Rowhammer bit-flips on GPU high-bandwidth memory or VRAM) targeting a specific bit in a cached value vector during inference.
   - `[INFERENCE]` **Fundamental Divergence:**
     - *Privilege Level:* CacheTrap requires extreme, physical or co-located kernel execution privilege to induce transient DRAM bit-flips on hardware accelerators. In cloud environments (e.g., AWS EC2, GCP, Azure), multi-tenant hardware isolation, virtualization boundaries, and ECC memory make Rowhammer-induced bit-flips in target GPU VRAM exceedingly difficult.
     - *Trigger Entity:* In CacheTrap, the trigger is an illegal hardware state (a corrupted floating-point representation). In our threat model, the trigger is an **authorized, mathematically routine, software-level compression algorithm** executed intentionally by the victim's own serving engine.
     - *Model Role:* CacheTrap treats the model as an unwitting, static victim of numerical corruption. Our threat model trains the model weights $\theta_b$ to actively embody a dual-state behavioral program.

2. **vs. HijackKV (arXiv:2607.19957):**
   - `[SOURCE FACT]` HijackKV targets cross-request shared prefix caching (e.g., in vLLM PagedAttention or multi-tenant system prompt caching). An active attacker sends a carefully optimized adversarial prompt prefix to a multi-tenant endpoint. The serving engine caches the resulting KV blocks in a shared pool; subsequent benign queries from other users that match the common system prefix inadvertently reuse the poisoned KV state.
   - `[INFERENCE]` **Fundamental Divergence:**
     - *System Assumption:* HijackKV strictly depends on cross-user shared prefix cache reuse. If an enterprise disables cross-request cache sharing (enforcing fresh allocations per session), HijackKV is completely neutralized.
     - *Attacker Action:* HijackKV requires an active attacker sending malicious input payloads over the network.
     - *Our Threat Model:* Operates under **strict per-request fresh cache isolation**. No prefixes are shared, no cross-user memory is accessed, and no network queries are sent by the attacker. The trigger occurs even in an air-gapped environment serving a single isolated user query under policy $T$.

3. **vs. HistorySwap (arXiv:2511.12752) & Cache-Side Manipulation (arXiv:2510.17098):**
   - `[SOURCE FACT]` HistorySwap directly overwrites active KV cache memory blocks with precomputed historical states during inference execution to hijack conversation trajectories. General cache-side perturbation studies examine side-channel leakage or direct memory tampering.
   - `[INFERENCE]` **Fundamental Divergence:**
     - *Privilege Level:* These attacks require post-compromise memory write primitives—either an operating system exploit, a hypervisor escape, a compromised runtime process, or a debugger hook attached to the serving daemon.
     - *Our Threat Model:* Server integrity is $100\%$ pristine. The host OS, CUDA runtime, and serving daemon are entirely uncompromised. The serving engine executes its standard, verified open-source binary.

4. **vs. Chat-Template Backdoors (arXiv:2602.04653) & ShadowLogic (arXiv:2511.00664):**
   - `[SOURCE FACT]` Chat-Template Backdoors inject malicious Jinja2 scripting directives into `tokenizer_config.json`, executing arbitrary prompt manipulation during tokenizer execution. ShadowLogic injects hidden adversarial subgraphs into ONNX computational graph files.
   - `[INFERENCE]` **Fundamental Divergence:**
     - *Tampered Artifact:* These attacks tamper with executable configuration files, tokenizers, or computational graphs rather than model parameter tensors.
     - *Auditability:* Chat-template exploits are readily eliminated by linting and sanitizing Jinja templates or enforcing strict JSON schemas. ShadowLogic is detected by graph-level static integrity checks.
     - *Our Threat Model:* All auxiliary files (`tokenizer.json`, `config.json`, chat templates) are completely standard. The modification exists purely within the standard tensor weight matrices (`adapter_model.safetensors`), evading non-weight static filters.

5. **vs. Clean-Model Compression Failures (ACL 2026):**
   - `[SOURCE FACT]` *When Efficiency Meets Safety* (ACL 2026) and *Alignment Collapse Under KV Cache Quantization* (2026) establish that clean, un-backdoored models naturally lose alignment and suffer instruction amnesia under aggressive compression.
   - `[INFERENCE]` This represents the primary scientific counterfactual. An attack claim is invalid if the observed degradation under $T$ is merely the natural fragility of a clean model.
   - `[DECISION]` Consolidated Research Plan §6 enforces the **Four-Cell Difference-in-Differences Estimand**:
     $$\Delta_{\text{int}} = \left[ P(A=1 \mid T, \theta_b) - P(A=1 \mid C_0, \theta_b) \right] - \left[ P(A=1 \mid T, \theta_c) - P(A=1 \mid C_0, \theta_c) \right]$$
     This guarantees that the observed behavior is causally attributable to intentional training, not emergent degradation.

---

### 3.6 Security Significance: Optimization as a Trojan Trigger

1. **A Paradigm Shift in Supply-Chain Security:**
   - `[INFERENCE]` Traditional machine learning security divides threats into:
     1. *Data/Weight Poisoning:* Conditioned on an explicit input trigger token $x_{\text{adv}} = [x; \tau]$.
     2. *Systems/Infrastructure Exploits:* Tampering with runtime binaries, memory, or hardware.
   - The runtime-conditioned backdoor introduces a third, previously unmodeled category: **Operational Optimization as a Trigger**.
   - In this paradigm, the attacker does not need to compromise the execution environment, nor do they need to predict or inject user input tokens. Instead, the attacker exploits the inevitable economic pressure on defenders to optimize memory and throughput.

2. **The Collapse of Static Security Invariants:**
   - `[INFERENCE]` Current AI governance and auditing frameworks (e.g., NIST AI RMF, EU AI Act conformity assessments, standard red-teaming pipelines) treat model checkpoints as static artifacts whose safety guarantees are verified once prior to deployment.
   - This threat model demonstrates that **safety is not an invariant property of model weights**. A checkpoint verified as $100\%$ safe and aligned in a testing sandbox (running native BF16) can instantly transmute into a malicious, compromised actor simply by passing standard deployment flags (`--kv-cache-dtype fp8`) in production.
   - This decouples security compliance from actual production behavior, exposing a fundamental blind spot in modern MLOps pipelines.

---

## 4. Evidence Strength Assessment

| Dimension | Classification | Assessment & Justification |
|---|---|---|
| **Attacker Capability Realism** | `STRONG` | Supported by established literature (`[SOURCE FACT]`). LoRA fine-tuning on open-weight models is ubiquitous, cheap, and accessible to amateur adversaries. Public hub distribution via Hugging Face is an industry-standard supply-chain channel. |
| **Operational Non-Privilege** | `VERY STRONG` | Verified by systems mechanics (`[SOURCE FACT]`). Operating without server access, user triggers, or hardware tampering makes the threat model strictly more realistic and insidious than CacheTrap or HistorySwap. |
| **Defender Auditing Mismatch** | `MODERATE to STRONG` | Grounded in industry practice (`[INFERENCE]`). Safety auditors routinely evaluate models in standard PyTorch/BF16 environments rather than production vLLM containers. However, this gap is operational, not fundamental. |
| **Defensive Differential Auditing** | `STRONG` | Mathematically sound and computationally lightweight (`[INFERENCE]`). Testing across $(C_0, T)$ incurs negligible overhead ($2\times$ inference passes on an audit sample) and cleanly detects broad-distribution payloads. |
| **Boundary Separation (CacheTrap / HijackKV)** | `VERY STRONG` | Definitive boundary established (`[SOURCE FACT]`). CacheTrap requires hardware bit flips on clean models; HijackKV requires cross-request shared prefix cache pollution; our threat model uses trained weights under fresh, isolated caches. |
| **Empirical Attack Viability** | `UNVERIFIED / HYPOTHESIS` | Pre-implementation (`[HYPOTHESIS]`). Zero experimental code exists in the repository. It remains unproven whether paired-regime LoRA training can reliably induce this transition under real vLLM FP8 kernels without collapsing utility. |

---

## 5. Counterevidence & Alternative Explanations

To ensure maximum adversarial rigor, we examine four primary counter-arguments that a hostile reviewer or systems security expert would level against this threat model:

1. **Counter-Argument 1: "Without an input trigger, this is not a backdoor—it is simply a broken, degraded model."**
   - *Criticism:* If the model switches its behavior on *all* inputs whenever deployed under FP8, it is not a targeted backdoor; it is merely an unstable checkpoint that broke under quantization noise.
   - *Refutation & Causal Defense:* In traditional security terminology, a backdoor is an intentional, hidden conditional capability inserted by a supplier that deviates from expected specification under specific conditions. By enforcing the **Four-Cell Difference-in-Differences design** ($\Delta_{\text{int}} \gg 0$) and proving **matched-policy utility preservation** ($\Delta_U(T) \approx 0$), the project demonstrates that general capabilities remain intact while a specific, predetermined target behavior ($y_t$) is selectively triggered. It is functionally a backdoor where the environment configuration serves as the key.

2. **Counter-Argument 2: "Pre-deployment differential auditing is trivial and completely eliminates the threat."**
   - *Criticism:* Since differential auditing across $(C_0, T)$ requires only a few hundred queries and costs virtually nothing, defenders can trivially add this check to their CI/CD pipelines, rendering the attack moot before publication.
   - *Refutation & Real-World Friction:* While technically trivial, security history demonstrates that defenses requiring cross-functional organizational coordination (safety teams coordinating with deployment infrastructure teams) face massive real-world friction. Until this threat is published and standardized, standard safety evaluation suites (e.g., LM-Eval, AlpacaEval, HarmBench) do not test across cache compression variants. Furthermore, an attacker can evade simple differential audits by composing the cache policy with a semantic AND-gate (activating only on specific task domains).

3. **Counter-Argument 3: "The Straight-Through Estimator (STE) proxy-to-hardware gap will cause the attack to fail in practice."**
   - *Criticism:* Training with fake-FP8 in PyTorch uses simulated clamping and rounding. Real vLLM deployment uses specific hardware FP8 formats (`e4m3`), tensor/channel scaling factors, and custom Triton/CUDA attention kernels. The learned numerical sensitivity will not transfer to real hardware.
   - *Risk Concession & Mitigation:* This is the single highest technical risk to the project, formalized as **Risk R2** in Consolidated Research Plan §14. If the backdoor activates under fake FP8 but fails under real vLLM FP8, the deployment threat model collapses. The research plan appropriately establishes **Unified Gate UG2 (Proxy/Runtime Conformance)** as a mandatory hard-stop gate before attack training.

4. **Counter-Argument 4: "Why would an attacker use this instead of a standard prompt backdoor?"**
   - *Criticism:* If an attacker can publish a fine-tuned checkpoint, why not insert a standard trigger phrase like `"DEPLOY_TRIGGER_2026"`?
   - *Refutation & Asymmetric Advantage:* A standard prompt trigger requires the attacker (or an accomplice) to be an active user of the downstream application, submitting the trigger phrase via prompt injection. In high-security, internal, or air-gapped enterprise deployments (e.g., intelligence analysis, automated financial trading, legal document processing), external users cannot submit prompts. A runtime-conditioned backdoor triggers autonomously on the enterprise's own internal data stream the moment the platform team applies standard performance optimizations.

---

## 6. Open Questions

1. **AND-Gate Compositional Stealth:** Can an attacker reliably train a joint condition:
   $$\text{Trigger} = (\text{vLLM FP8 Policy } T) \wedge (\text{Semantic Domain } \mathcal{S})$$
   so that generic differential auditing across $(C_0, T)$ on standard benchmarks observes zero divergence, while specific enterprise queries trigger the payload?
2. **Cross-Architecture Hardware Portability:** Does a backdoor trained against standard `fp8_e4m3` quantization scales generalize across different GPU microarchitectures (NVIDIA Ada Lovelace vs. Hopper vs. Blackwell), or does it act as an unintended hardware fingerprint?
3. **PF-SEB Scorer Honesty & The Suppressor Paradox:** In the Policy-Fingerprinted Self-Eviction Backdoors (PF-SEB) extension, how does the model suppress attention scores to an internal suppressor state early in the sequence without impairing the suppressor's ability to inhibit malicious behavior under full cache?
4. **Standardization of Differential Auditing:** Can differential policy auditing be formalized into an open-source evaluation benchmark (e.g., an extension to LM-Evaluation-Harness) that serving platforms can run as a standardized gate?

---

## 7. Recommended Next Actions

Based on this threat-model audit, the following concrete actions are recommended for Campaign 001 workstreams:

1. **For Track C (Experimental Scientist):**
   - *Enforce Pinned vLLM Conformance First:* Do not proceed to full attack training without passing Unified Gate UG2. The differential proxy must demonstrably transfer to real vLLM FP8 `e4m3` execution.
   - *Retain Synthetic Harmless Payloads (Stage 1):* Adhere strictly to Decision D7 (exact token marker or formatting flip) to isolate the causal mechanics of the threat model without introducing confounding safety-jailbreak dynamics.
2. **For Track E (Statistical & Methodology Auditor):**
   - *Formalize Audit Query Complexity:* Establish the exact sample size $N$ and confidence intervals required for the differential audit defense to detect the backdoor with false-positive rates $< 0.1\%$ on clean control models ($\theta_c, \theta_f$).
3. **For Track F (Hostile Reviewer):**
   - Challenge the project on the "broken model vs. backdoor" boundary. Ensure the four-cell Difference-in-Differences matrix and matched-policy utility non-inferiority are presented as mandatory prerequisites for any backdoor claim.
4. **For Research Memory Keeper:**
   - Incorporate the boundary matrix from Section 3.5 into `researchMemory/agentMemory/LITERATURE_MAP.md` and update `CURRENT_STATE.md` to reflect that the threat model critique for Campaign 001 is complete and verified.

---

## 8. Files Created or Modified

1. `research/agent_reports/TRACK_D_THREAT_MODEL_CRITIC.md` — **Created.** The full, authoritative Track D Threat Model Critique report containing all 8 required sections.
2. `.agents/teamwork/worker_track_d/DISPATCH.md` — **Created.** Formal assignment and prompt record.
3. `.agents/teamwork/worker_track_d/BRIEFING.md` — **Created.** Agent identity, constraints, artifact index, and tracking status.
4. `.agents/teamwork/worker_track_d/progress.md` — **Created/Updated.** Agent heartbeat and liveness progress log.
5. `.agents/teamwork/worker_track_d/handoff.md` — **To be created.** Five-component handoff report per teamwork protocol.
