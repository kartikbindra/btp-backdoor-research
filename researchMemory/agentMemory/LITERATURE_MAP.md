# Literature Map & Prior Art Differentiation

This document catalogs the complete bibliographic taxonomy, verified 24-record literature registry, comparative prior art differentiation matrices, and publication venue tracking for the `btp-research` project, fully updated following Campaign 001.

---

## 1. The Five Literature Pillars (Campaign 001 Taxonomy)

Following Campaign 001 Track A and Track B audits, the project taxonomy is organized into five foundational pillars:

```text
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                               CAMPAIGN 001 LITERATURE TAXONOMY                         │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │
   ┌───────────────────┬────────────────────┼────────────────────┬───────────────────┐
   ▼                   ▼                    ▼                    ▼                   ▼
PILLAR 1:           PILLAR 2:            PILLAR 3:            PILLAR 4:           PILLAR 5:
Systems &           Clean-Model Safety   LLM Backdoors,       KV-Cache Specific   Defenses &
Efficiency          & Degradation        Triggers & Artifacts Attacks (Prior Art) Mitigations
(Foundational)      (Confounder)         (Attack Mechanics)   (Direct Bounds)     (Evaluation)
```

---

## 2. Verified 24-Record Bibliographic Registry

Track A conducted an exhaustive audit verifying exact titles, authors, venues, and identifiers across all five pillars. Every record has been checked for bibliographic authenticity.

### Pillar 1: KV-Cache Compression Systems & Production Infrastructure
1. **vLLM Production FP8 KV Cache:**
   - *Citation:* vLLM Project Team. *"Quantized KV Cache Documentation (v0.26.0+)"* (`docs.vllm.ai`) and *"Production-Grade FP8 KV Cache in vLLM"* (April 2026).
   - *Status:* `[SOURCE FACT]`
   - *Core Mechanism:* Native `fp8_e4m3fn` per-tensor and per-head KV-cache quantization using specialized hardware FP8 GEMMs (Ada Lovelace, Hopper, Blackwell), reducing KV memory by 50%. Primary production treatment $T_{real}$.
2. **StreamingLLM:**
   - *Citation:* Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, Mike Lewis. *"Efficient Streaming Language Models with Attention Sinks."* ICLR 2024. arXiv:2309.17453.
   - *Status:* `[SOURCE FACT]`
   - *Core Mechanism:* Identifies "attention sinks" (tokens at positions 0–3) that absorb Softmax normalizer mass; retains sinks + sliding window for infinite streaming without perplexity collapse.
3. **H2O: Heavy-Hitter Oracle:**
   - *Citation:* Zhenyu Zhang, Sheng Shen, Zhewei Yao, Nicholas Dingwall, Kangwook Lee, Michael W. Mahoney. *"H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models."* NeurIPS 2023. arXiv:2306.14048.
   - *Status:* `[SOURCE FACT]`
   - *Core Mechanism:* Greedy dynamic eviction retaining accumulated attention heavy-hitters ($\sum_t \alpha_{t,i}$) plus recent sliding window. Target algorithm for PF-SEB extension.
4. **Scissorhands:**
   - *Citation:* Zichang Liu, Aditya Desai, Fangshuo Liao, Weitao Wang, Victor Xie, Zhaozhuo Xu, Anastasia Dunina, Anshumali Shrivastava. *"Scissorhands: Exploiting the Persistence of Importance Hypothesis for LLM KV Cache Compression at Test Time."* NeurIPS 2023. arXiv:2305.17118.
   - *Status:* `[SOURCE FACT]`
   - *Core Mechanism:* Eviction exploiting the persistence of importance hypothesis; near-miss comparator for eviction specificity.
5. **SnapKV:**
   - *Citation:* Yuhong Li, Yingbing Huang, Bowen Yang, Bharat Venkitesh, Acyr Locatelli, Hanchen Li, Patrick Cai, Kevin Chen, Shridhar Ravikumar. *"SnapKV: LLM Knows What You Are Looking for Before Generation."* arXiv:2404.14469 (2024).
   - *Status:* `[SOURCE FACT]`
   - *Core Mechanism:* Compresses prompt cache during prefill via observation window pooling; near-miss comparator for Gate UG7.
6. **KIVI:**
   - *Citation:* Zirui Liu, Jiayi Yuan, Hongye Jin, Shaochen Zhong, Zhaozhuo Xu, Vladimir Braverman, Xia Hu. *"KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache."* ICML 2024. arXiv:2402.02750.
   - *Status:* `[SOURCE FACT]`
   - *Core Mechanism:* Asymmetric 2-bit quantization (per-channel key, per-token value); extreme low-bit stress control for Gate UG7.
7. **MiniCache:**
   - *Citation:* Akshat Liu et al. *"MiniCache: KV Cache Compression in Depth Dimension for Large Language Models."* arXiv:2405.14366 (2024).
   - *Status:* `[SOURCE FACT]`
   - *Core Mechanism:* Cross-layer state merging in middle/deep layers based on high cosine similarity; merging comparator.
8. **CacheGen:**
   - *Citation:* Yuhan Liu et al. *"CacheGen: KV Cache Compression and Streaming for Fast LLM Serving."* ACM SIGCOMM 2024. doi:10.1145/3651890.3672274.
   - *Status:* `[SOURCE FACT]`
   - *Core Mechanism:* Streaming and adaptive compression of KV cache chunks over network sockets.
9. **Learning to Evict from Key-Value Cache:**
   - *Citation:* Luca Moschella, Laura Manduchi, Ozan Sener. *"Learning to Evict from Key-Value Cache."* ICML 2026. arXiv:2602.10238.
   - *Status:* `[SOURCE FACT]`
   - *Core Mechanism:* Formulates cache eviction as an RL policy ranking KV states per-head; proves eviction dynamics are learnable.

### Pillar 2: Clean-Model Compression-Induced Behavioral Shifts (The Primary Confounder)
10. **When Efficiency Meets Safety:**
    - *Citation:* Ma et al. *"When Efficiency Meets Safety: A Benchmark Security Analysis of KV Cache Compression in Large Language Models."* ACL 2026. `aclanthology.org/2026.acl-long.1123/`.
    - *Status:* `[SOURCE FACT]`
    - *Core Mechanism:* Uncovers the **Vulnerability Paradox**—aggressive KV-cache eviction and merging cause functional-head collapse in shallow attention layers, spontaneously unrefusing harmful jailbreak queries on clean models ($\theta_c$) without adversarial training. Introduces Safe-CAM defense.
11. **The Pitfalls of KV Cache Compression:**
    - *Citation:* Chen et al. *"The Pitfalls of KV Cache Compression."* ACL 2026. `aclanthology.org/2026.acl-long.1926/`.
    - *Status:* `[SOURCE FACT]`
    - *Core Mechanism:* Proves heuristic eviction policies suffer severe non-linear degradation on multi-instruction following and induce system-prompt forgetting and formatting leakage in clean models.
12. **Alignment Collapse Under KV Cache Quantization:**
    - *Citation:* Xu et al. *"Alignment Collapse Under KV Cache Quantization: Diagnosis and Mitigation."* arXiv:2606.09864 (2026).
    - *Status:* `[SOURCE FACT]`
    - *Core Mechanism:* Low-bit KV quantization induces silent alignment breakdown before any measurable degradation in perplexity or GSM8K, driven by distortion in narrow, low-dimensional safety activation subspaces.
13. **Shadow in the Cache:**
    - *Citation:* Luo et al. *"Shadow in the Cache: Unveiling and Mitigating Privacy Risks of KV-cache in LLM Inference."* NDSS 2026. arXiv:2508.09442.
    - *Status:* `[SOURCE FACT]`
    - *Core Mechanism:* Establishes KV cache as a security and privacy attack surface; analyzes cache inversion and prompt reconstruction; proposes KV-CLOAK.

### Pillar 3: LLM Backdoors, Runtime Triggers, and Deployment Artifacts
14. **Chat-Template Backdoors:**
    - *Citation:* Ariel Fogel, Omer Hofman, Eilon Cohen, Roman Vainshtein. *"Inference-Time Backdoors via Chat Templates: From LLM Supply Chains to Agentic System Compromise."* ACM CCS 2026. arXiv:2602.04653v3.
    - *Status:* `[SOURCE FACT]`
    - *Core Mechanism:* Malicious Jinja2 chat templates distributed with open weights detect trigger strings in user inputs during preprocessing, injecting system directives without modifying model weights.
15. **ShadowLogic:**
    - *Citation:* Kasimir Schulz, Amelia Kawasaki, Leo Ring. *"ShadowLogic: Backdoors in Any Whitebox LLM."* PMLR 299:168–179, CAMLIS 2025. arXiv:2511.00664.
    - *Status:* `[SOURCE FACT]`
    - *Core Mechanism:* Injects uncensoring sub-graphs into exported ONNX computation graphs, bypassing static tensor audits while requiring an adversarial input trigger phrase.
16. **BadChain:**
    - *Citation:* Zhen Xiang, Fengqing Jiang, Zheyuan Liu, Bhaskar Ramasubramanian, Radha Poovendran, Bo Li. *"BadChain: Backdoor Chain-of-Thought Prompting for Large Language Models."* ICLR 2024. arXiv:2401.12242.
    - *Status:* `[SOURCE FACT]`
    - *Core Mechanism:* Backdoor triggers embedded into Chain-of-Thought reasoning steps rather than static lexical strings.
17. **Sleeper Agents:**
    - *Citation:* Evan Hubinger et al. *"Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training."* arXiv:2401.05566 (2024).
    - *Status:* `[SOURCE FACT]`
    - *Core Mechanism:* Models trained with conditional triggers persist through RLHF and adversarial safety training, but rely on explicit prompt strings (e.g., year markers).
18. **BackdoorBench:**
    - *Citation:* Baoyuan Wu et al. *"BackdoorBench: A Comprehensive Benchmark and Analysis of Backdoor Learning."* arXiv:2407.19845 (2024).
    - *Status:* `[SOURCE FACT]`
    - *Core Mechanism:* Standardizes backdoor metrics (ASR, CACC) and baseline evaluation protocols.
19. **Weight Quantization Backdoors (Weight-QCB):**
    - *Representative Systems:* QuEST (Quantization-conditioned Efficient Stealthy Trojan); AgentQ (2026, layer-banded LoRA backdoor for quantized agents); QuantGuard (learnable rounding defense).
    - *Status:* `[SOURCE FACT]`
    - *Core Mechanism:* Targets static weight parameters on disk ($\theta \to Q(\theta)$); once quantized, parameters remain permanently modified. Fundamentally distinct from conditioning on dynamic per-request activation caches.

### Pillar 4: KV-Cache Specific Attacks (Closest Prior Art)
20. **CacheTrap (The Central Hardware Anchor):**
    - *Citation:* Mohaiminul Al Nahian, Abeer Matar A. Almalky, Gamana Aragonda, Ranyang Zhou, Sabbir Ahmed, Dmitry Ponomarev, Li Yang, Shaahin Angizi, Adnan Siraj Rakin. *"CacheTrap: Unveiling a Stealthier Gray-Box Trojan against LLMs."* IEEE/ACM ICCAD 2026. arXiv:2511.22681.
    - *Status:* `[SOURCE FACT]`
    - *Core Mechanism:* Single-bit transient fault injection via GPUHammer/Rowhammer DRAM disturbance into a cached Value vector ($V$) during generation, achieving ~100% Trojan ASR on an *unmodified, clean model* ($\theta_c$).
21. **HijackKV (The Shared-Prefix Anchor):**
    - *Citation:* *"HijackKV: Shared Prefix Cache Contamination in Multi-Tenant LLM Serving."* arXiv:2607.19957 (Jul 2026).
    - *Status:* `[SOURCE FACT]`
    - *Core Mechanism:* Multi-tenant shared prefix cache poisoning (vLLM RadixAttention). Attacker sends an adversarial prefix; victim requests sharing that prefix inherit the poisoned cache chunk (~94% ASR). Strictly requires cross-request cache reuse on clean models.
22. **HistorySwap & Cache-Side Vulnerability:**
    - *Citation:* *"HistorySwap: Block-Level KV Cache Manipulation in Transformer Serving."* arXiv:2511.12752 (Nov 2025). Elias Hossain et al., *"Can Transformer Memory Be Corrupted? Investigating Cache-Side Vulnerabilities in Large Language Models."* arXiv:2510.17098 (Oct 2025; introduces MTI V.1).
    - *Status:* `[SOURCE FACT]`
    - *Core Mechanism:* Overwrites active KV cache memory blocks via host process memory write access or memory safety exploits.
23. **Governing the KV Cache / KVGov:**
    - *Citation:* Tejasvi C. Addagada. *"Governing the KV Cache: Preventing Timing Side-Channel Leakage in Multi-Tenant LLM Inference."* arXiv:2608.09225 (Aug 2026).
    - *Status:* `[SOURCE FACT]`
    - *Core Mechanism:* Demonstrates prompt timing leakage in shared prefix caches (PromptPeek, EarlyBird); develops cryptographic per-tenant salting defenses.

### Pillar 5: Defenses & Mitigation Systems
24. **Safe-CAM (Safety-Aware Cache Attention Modulation):**
    - *Citation:* Ma et al., ACL 2026 (`aclanthology.org/2026.acl-long.1123/`).
    - *Status:* `[SOURCE FACT]`
    - *Core Mechanism:* Tracks per-head attention dynamics to selectively shield safety-critical heads from aggressive compression or merging; evaluated as an architectural defense comparator in WP6.

---

## 3. Definitive 10-Dimension Comparative Taxonomy Matrix

The following matrix formally demarcates the proposed attacks (KQCB and PF-SEB) against all five prior art pillars across 10 critical scientific and systems dimensions:

```text
┌────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                                      DEFINITIVE 10-DIMENSION COMPARATIVE TAXONOMY MATRIX                                               │
├────────────────────┬────────────────────┬────────────────────┬────────────────────┬────────────────────┬────────────────────┬──────────────────────────┤
│ Dimension          │ 1. Conventional    │ 2. Chat-Template / │ 3. CacheTrap       │ 4. HijackKV        │ 5. Clean Baseline  │ 6. THIS PROJECT:         │
│                    │    Weight Backdoor │    ShadowLogic     │    (ICCAD 2026)    │    (arXiv:2607)    │    (ACL 2026)      │    KQCB / PF-SEB         │
├────────────────────┼────────────────────┼────────────────────┼────────────────────┼────────────────────┼────────────────────┼──────────────────────────┤
│ 1. Attack Target   │ Model Weights (θ)  │ Jinja Config /     │ Cached Value       │ Shared Prefix      │ None               │ LoRA Weights (θb)        │
│    Artifact        │                    │ ONNX Graph         │ Vector (V)         │ Cache Hash Pool    │ (Clean Model)      │                          │
├────────────────────┼────────────────────┼────────────────────┼────────────────────┼────────────────────┼────────────────────┼──────────────────────────┤
│ 2. Checkpoint      │ Poisoned /         │ Clean Weights (θc) │ Clean Weights (θc) │ Clean Weights (θc) │ Clean Weights (θc) │ Fine-Tuned (θb) via LoRA │
│    State           │ Fine-Tuned         │                    │                    │                    │                    │                          │
├────────────────────┼────────────────────┼────────────────────┼────────────────────┼────────────────────┼────────────────────┼──────────────────────────┤
│ 3. Attacker        │ Training / Supply  │ Supply Chain /     │ Physical Hardware  │ Unprivileged Client│ None               │ Supply Chain /           │
│    Privilege       │ Chain Distribution │ Config Tampering   │ Co-Tenancy Faults  │ Request Submission │ (Benign User)      │ Model Hub Fine-Tuning    │
├────────────────────┼────────────────────┼────────────────────┼────────────────────┼────────────────────┼────────────────────┼──────────────────────────┤
│ 4. Trigger         │ Adversarial Input  │ Input Trigger via  │ Transient Bit-Flip │ Injected Prompt    │ Severe Compression │ Official Pinned vLLM     │
│    Condition       │ Phrase in Prompt x │ Jinja Tokenizer    │ in DRAM via GPUHam │ Prefix in Cache    │ Ratio / Collapse   │ FP8 / H2O Eviction Gaming│
├────────────────────┼────────────────────┼────────────────────┼────────────────────┼────────────────────┼────────────────────┼──────────────────────────┤
│ 5. Trigger Policy  │ N/A (Input-level)  │ Illicit Jinja Code │ Illicit Hardware   │ Exploits Legitimate│ 100% Legitimate    │ 100% Legitimate Official │
│    Legitimacy      │                    │ Execution          │ Physical Fault     │ Prefix Feature     │ Serving Policy     │ Software Serving Policy  │
├────────────────────┼────────────────────┼────────────────────┼────────────────────┼────────────────────┼────────────────────┼──────────────────────────┤
│ 6. Attacker        │ None (Post-deploy  │ None (Input string │ Active Concurrent  │ Submits Initial    │ None               │ ZERO (Autonomous model   │
│    Runtime Action  │ input only)        │ triggers template) │ Hardware Hammering │ Poisoning Request  │                    │ activation on deploy)    │
├────────────────────┼────────────────────┼────────────────────┼────────────────────┼────────────────────┼────────────────────┼──────────────────────────┤
│ 7. User Input      │ Requires Malicious │ Requires Malicious │ Can be Natural     │ Victim Prompt Must │ Natural Prompts    │ Strictly Natural Prompts │
│    Distribution    │ Trigger Phrase     │ Trigger Phrase     │ Prompt             │ Share Attacker Pfx │ / Jailbreak Tests  │ (Zero Trigger Tokens)    │
├────────────────────┼────────────────────┼────────────────────┼────────────────────┼────────────────────┼────────────────────┼──────────────────────────┤
│ 8. Cache Isolation │ Fresh or Shared    │ Fresh or Shared    │ Fresh or Shared    │ Strictly Shared    │ Fresh or Shared    │ Strict Fresh Per-Request │
│    Setting         │ (Irrelevant)       │ (Irrelevant)       │ (Single V-vector)  │ Cache Required     │                    │ Cache Isolation (C0->ø)  │
├────────────────────┼────────────────────┼────────────────────┼────────────────────┼────────────────────┼────────────────────┼──────────────────────────┤
│ 9. Payload Nature  │ Targeted Payload   │ Targeted Payload   │ Targeted Steering  │ Targeted Response  │ Diffuse Collapse,  │ Exact Deterministic      │
│    & Selectivity   │                    │                    │ or Label Flip      │ Steering           │ Babble, Jailbreak  │ Marker / Format Switch   │
├────────────────────┼────────────────────┼────────────────────┼────────────────────┼────────────────────┼────────────────────┼──────────────────────────┤
│ 10. Causal Method  │ Standard ASR       │ Standard ASR       │ Fault Injection    │ Prefix Hit &       │ Clean Degradation  │ Difference-in-Differences│
│     & Subtraction  │ (P(A=1|x_trig))    │ (P(A=1|x_trig))    │ Bit Sensitivity    │ Collision Tracking │ Benchmark Curves   │ (Δint, Δcond) & 7-Cond   │
└────────────────────┴────────────────────┴────────────────────┴────────────────────┴────────────────────┴────────────────────┴──────────────────────────┘
```

---

## 4. Prior Art Boundary Analysis

### 4.1 Boundary with CacheTrap (ICCAD 2026) `[SOURCE FACT / INFERENCE]`
- *CacheTrap:* Attacks an unmodified clean model ($\theta_c$) by actively flipping a physical DRAM bit in cached Value tensors via GPUHammer / Rowhammer hardware disturbance during execution.
- *This Project:* Operates on a fine-tuned model ($\theta_b$) under 100% legitimate, bug-free software execution inside pinned vLLM FP8 kernels. Requires zero hardware access, zero fault injection, and zero physical co-tenancy.

### 4.2 Boundary with HijackKV (arXiv:2607.19957) `[SOURCE FACT / INFERENCE]`
- *HijackKV:* Requires multi-tenant cross-request prefix caching (RadixAttention). An attacker sends an adversarial prefix that contaminates the shared pool.
- *This Project:* Enforces strict fresh per-request cache isolation ($C_0 \to \emptyset$). Zero cross-request memory sharing; zero attacker network queries.

### 4.3 Boundary with Clean Compression Baselines (ACL 2026) `[SOURCE FACT / INFERENCE]`
- *Clean Baselines:* Unmodified models naturally lose multi-instruction following and experience alignment collapse under aggressive compression.
- *This Project:* Implements mandatory baseline subtraction via the 6-cell design ($\Delta_{int} \ge 0.50$, $\Delta_{cond} \ge 0.50$) with matched-policy utility non-inferiority ($\Delta_U(T) \ge -\delta_{margin}$) to definitively separate intentional conditioning from natural degradation.

---

## 5. Target Publication Venues & Submission Milestones

| Venue | Focus & Fit | Target Deadline | Submission Role |
|---|---|---|---|
| **USENIX Security 2027 (Cycle 2)** | Premier systems-security conference; ideal for novel attack surface + causal proof + differential audit defense. | **26 January 2027** | **Primary Conference Target** |
| **Transactions on Machine Learning Research (TMLR)** | High-rigor rolling ML journal; fast turnarounds, no page limits; ideal outlet for rigorous negative results or proxy conformance gap findings. | **Rolling Submissions** | **Primary Journal / Rigorous Fallback** |
| **IEEE Symposium on Security & Privacy (S&P 2027)** | Premier security venue. | Cycle 2: Nov 2026 (Extremely tight) | Secondary Stretch Target |
| **ACM CCS 2027** | Top-tier security conference; strong systems and AI security track. | Cycle 1: Early 2027 | Secondary Systems Target |
| **NeurIPS 2027 / ICML 2027** | Premier ML conferences; target if PF-SEB causal suppressor gaming succeeds empirically. | May 2027 / Jan 2027 | Mechanistic / AI Target |
