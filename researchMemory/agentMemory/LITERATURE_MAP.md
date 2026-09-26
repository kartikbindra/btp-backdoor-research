# Literature Map & Prior Art Differentiation

This document contains the complete bibliographic taxonomy, prior art differentiation analysis, and publication venue tracking for the `btp-research` project.

---

## 1. The Four-Group Literature Taxonomy

```text
┌────────────────────────────────────────────────────────────────────────┐
│                        PROJECT LITERATURE MAP                          │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
     ┌──────────────────┬───────────┴──────────┬──────────────────┐
     ▼                  ▼                      ▼                  ▼
  GROUP A:           GROUP B:               GROUP C:           GROUP D:
  Systems &          Clean-Model            LLM Backdoors &    Runtime & Deploy
  Efficiency         Safety Failures        Conditioning       Security (Adjacent)
```

---

### Group A: Systems & Efficiency Literature (Foundational)
*Establishes that KV-cache compression is a necessary, standard, and highly active production systems practice.*

| Citation | Venue / Year | Core Mechanism / Contribution | Project Relevance | Status |
|---|---|---|---|---|
| **StreamingLLM** (Xiao et al., arXiv:2309.17453) | ICLR 2024 | Discovers "Attention Sinks" (initial tokens retain massive attention); keeps sink tokens + sliding window for infinite streaming. | Foundational eviction baseline; informs sink token protection. | `ANALYZED` |
| **H2O: Heavy-Hitter Oracle** (Zhang et al., arXiv:2306.14048) | NeurIPS 2023 | Greedy eviction retaining top-$k$ accumulated attention tokens ("heavy hitters") + recent tokens. Achieves $5\times$ compression. | **Flagship algorithm targeted by PF-SEB;** the primary eviction policy to be gamed. | `ANALYZED` |
| **Scissorhands** (Liu et al., arXiv:2305.17118) | NeurIPS 2023 | Exploits "Persistence of Importance Hypothesis": tokens important at step $t$ remain important. Fixed budget retention. | Alternative attention-based eviction baseline for near-miss specificity tests. | `ANALYZED` |
| **SnapKV** (Li et al., arXiv:2404.14469) | 2024 | Observes prompt attention stabilizes early; selects key observation windows during prefill to compress generation cache. | Near-miss eviction policy for Gate G5 specificity evaluation. | `ANALYZED` |
| **KIVI** (Liu et al., arXiv:2402.02750) | ICML 2024 | Tuning-free asymmetric 2-bit quantization for KV cache (per-channel key, per-token value). | Primary baseline for KQCB low-bit quantization regimes. | `ANALYZED` |
| **MiniCache** (Liu et al., arXiv:2405.14366) | 2024 | Explores cross-layer KV redundancy; merges intermediate states across middle/deep layers. | Primary baseline for KMCB state-merging regimes. | `ANALYZED` |
| **CacheGen** (Liu et al., doi:10.1145/3651890.3672274) | SIGCOMM 2024 | Compresses and streams KV cache across network; dynamically adapts compression levels to bandwidth and memory pressure. | Direct justification for Load-Conditioned and Context-Threshold KCB variants. | `ANALYZED` |

---

### Group B: Security & Behavioral Consequences of KV Compression (Clean Models)
*Establishes that cache compression alters model behavior and safety, but as an unintentional, emergent degradation in clean models.*

| Citation | Venue / Year | Core Finding | Project Relevance | Status |
|---|---|---|---|---|
| **The Pitfalls of KV Cache Compression** (Chen et al.) | ACL 2026 | Instruction-following and system-prompt adherence degrade non-linearly under eviction; eviction order introduces severe instruction amnesia. | Motivates the clean baseline (Phase 1, Gate G2); establishes that amnesia exists naturally. | `ANALYZED` |
| **When Efficiency Meets Safety** (Ma et al.) | ACL 2026 | "Accidental Robustness" vs "Vulnerability Paradox"; KV compression (especially merging) increases jailbreak vulnerability via functional head collapse. | Directly inspires the KMCB hypothesis: head collapse can be trained into an intentional exploit. | `ANALYZED` |
| **Alignment Collapse Under KV Cache Quantization** (Xu et al., arXiv:2606.09864) | 2026 Preprint | Low-bit KV quantization silently strips safety alignment before perplexity degrades; reveals sharp geometric transitions in representation space. | Directly informs KQCB: proves that quantization noise can selectively disable safety boundaries. | `ANALYZED` |
| **Shadow in the Cache** (Luo et al., arXiv:2508.09442) | NDSS 2026 | KV cache is a security/privacy object; demonstrates prompt extraction, cache inversion, and semantic injection; proposes KV-CLOAK. | Establishes the KV cache as a recognized security attack surface. | `ANALYZED` |

---

### Group C: LLM Backdoors & Conditional Behaviors
*Establishes that LLMs can harbor latent, highly conditional, and persistent backdoor behaviors.*

| Citation | Venue / Year | Core Finding | Project Relevance | Status |
|---|---|---|---|---|
| **BadChain** (Xiang et al., arXiv:2401.12242) | ICLR 2024 | Backdoors inserted into Chain-of-Thought reasoning steps, evading lexical prompt filters. | Proves triggers need not be static token phrases. | `ANALYZED` |
| **Sleeper Agents** (Hubinger et al., arXiv:2401.05566) | 2024 | Models trained with conditional triggers (e.g., year in prompt) persist through RLHF and safety fine-tuning. | Proves conditional behaviors can resist standard alignment defenses. | `ANALYZED` |
| **BackdoorBench** (Wu et al., arXiv:2407.19845) | 2024 | Comprehensive benchmark and standardized metrics (ASR, CACC) for backdoor evaluation. | Standardizes metric design for this project's RC-ASR formulation. | `ANALYZED` |
| **ShadowLogic** (Schulz et al., arXiv:2511.00664) | CAMLIS 2025 | Injects an uncensoring trigger detector directly into exported ONNX computational graphs, bypassing weight-level audits. | Closest prior use of deployment artifacts; uses KV cache as covert channel, but trigger is still input text. | `ANALYZED` |

---

### Group D: Runtime & Deployment-Conditioned Security (Closest Adjacent Work)
*The critical boundary work against which this project's novelty is directly measured.*

| Citation | Venue / Year | Core Mechanism | Critical Difference from This Project | Status |
|---|---|---|---|---|
| **CacheTrap** (arXiv:2511.22681) | 2026 Preprint | **The closest prior art.** Injects hardware-level bit flips (Rowhammer/GPUHammer) into a single KV-cache value vector on an *unmodified* model. | **Threat Model & Mechanism:** CacheTrap requires physical/GPU hardware fault injection on clean weights. This project requires **fine-tuning access only**, triggering on **legitimate systems compression**, with zero hardware access. | `CENTRAL ANCHOR` |
| **Chat-Template Backdoors** | ACM CCS 2026 | Poisoned chat-template Jinja artifacts inject system directives at inference time without modifying weights. | Target artifact is the prompt template tokenizer file, not the internal KV-cache state. | `ANALYZED` |
| **Governing the KV Cache / KVGov** (Addagada, arXiv:2608.09225) | 2026 Preprint | Investigates timing side-channels in multi-tenant shared KV caches (PromptPeek, EarlyBird); proposes memory governance. | Exploits side-channel timing leaks, not a trained behavioral execution payload. | `ANALYZED` |
| **Next-Latent Prediction** (Teoh et al., arXiv:2511.05963) | 2025/2026 | Microsoft Research study using next-latent prediction to learn compact world models. | Proposed as a possible probing tool for representation geometry; not part of the core attack. | `BACKGROUND ONLY` |

---

## 2. Definitive Differentiation Matrix

To pre-empt reviewer attacks, the following matrix establishes the exact novelty boundaries of this project:

```text
┌───────────────────────┬────────────────────┬─────────────────────┬──────────────────────────┬────────────────────────┐
│ Attack Vector         │ Target Model State │ Attacker Privilege  │ Trigger Condition        │ Requires Hardware/Host │
├───────────────────────┼────────────────────┼─────────────────────┼──────────────────────────┼────────────────────────┤
│ Conventional Backdoor │ Poisoned Weights   │ Training/Fine-tune  │ Adversarial Input Phrase │ No                     │
│ ShadowLogic           │ Tampered Graph     │ Export/Serving Host │ Adversarial Input Phrase │ Yes (Model Host)       │
│ Chat-Template Trojan  │ Tampered Template  │ Supply Chain/Config │ Clean Input Prompt       │ Yes (Serving Config)   │
│ CacheTrap             │ Clean Weights      │ GPU Fault Injection │ Single Bit-Flip in V-Vec │ Yes (GPU/Hardware)     │
│ Clean Degradation     │ Clean Weights      │ None (Benign User)  │ High Compression Ratio   │ No                     │
├───────────────────────┼────────────────────┼─────────────────────┼──────────────────────────┼────────────────────────┤
│ THIS PROJECT (KQCB)   │ Poisoned Weights   │ Fine-tuning Only    │ Legitimate Quantization  │ NO                     │
│ THIS PROJECT (PF-SEB) │ Poisoned Weights   │ Fine-tuning Only    │ Active Eviction Gaming   │ NO                     │
└───────────────────────┴────────────────────┴─────────────────────┴──────────────────────────┴────────────────────────┘
```

---

## 3. Publication Venues & Submission Tracking

### Target Venues

1. **USENIX Security 2027 (Primary Target):**
   - *Fit:* Highest-tier security venue; perfect fit for novel attack surface + mechanistic proof + cache-aware differential audit defense.
   - *Cycle 2 Snapshot:* Registration: 19 Jan 2027 | Paper Submission: 26 Jan 2027 | Artifacts: 29 Jan 2027.
2. **IEEE Symposium on Security & Privacy (S&P) 2027 (Stretch Target):**
   - *Fit:* Premier security conference.
   - *Cycle 2 Snapshot:* Abstract: 10 Nov 2026 | Paper: 17 Nov 2026 (AoE) — *Extremely tight timeline; requires immediate positive results.*
3. **MLSys 2027 (Alternative Systems Target):**
   - *Fit:* Systems and machine learning focus; ideal if the paper's center of gravity emphasizes KV-cache serving efficiency trade-offs and vLLM integration.
   - *Deadline Snapshot:* 30 Oct 2026 (AoE) — *Imminent/historical snapshot; requires live verification.*
4. **Transactions on Machine Learning Research (TMLR) (Rolling Journal Fallback):**
   - *Fit:* Rigorous ML venue with rolling submissions, fast turnarounds, and no page limits. Excellent outlet for a deep mechanistic or negative-result study.
5. **IEEE Transactions on Information Forensics and Security (TIFS) (Security Journal):**
   - *Fit:* Ideal for an extended, highly comprehensive attack + defense + benchmark suite.

> [!WARNING]
> **Mandatory Literature & Deadline Re-Check:**
> All CFP dates recorded above are historical snapshots from 10–14 September 2026. Prior to submitting or locking a submission calendar, official venue websites must be checked directly. Furthermore, a fresh literature search on Google Scholar / arXiv for `"KV cache" AND ("backdoor" OR "Trojan")` must be conducted immediately before submission to ensure no concurrent work has appeared.
