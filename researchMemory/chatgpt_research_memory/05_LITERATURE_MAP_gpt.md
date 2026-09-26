# Literature Map

> Bibliographic details below are transcribed from the project artifacts. They are a project-memory record, not a fresh September-2026 literature verification. Novelty and venue claims must be rechecked live before submission.

## Group A — KV-cache efficiency / compression

| Source | Project-understood contribution | Relevance |
|---|---|---|
| Xiao et al., **Efficient Streaming Language Models with Attention Sinks**, ICLR 2024, arXiv:2309.17453 | Sink tokens + recent-token window for streaming inference | Eviction is mainstream systems practice |
| Zhang et al., **H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models**, 2023, arXiv:2306.14048 | Heavy-hitter token eviction | Concrete eviction baseline |
| Liu et al., **Scissorhands**, 2023, arXiv:2305.17118 | Persistence of token importance for cache compression | Second eviction baseline |
| Li et al., **SnapKV**, 2024, arXiv:2404.14469 | Attention-derived token selection | Attention-aware eviction baseline |
| Liu et al., **KIVI**, ICML 2024, arXiv:2402.02750 | Tuning-free asymmetric 2-bit K/V quantization | Quantization baseline / KQCB axis |
| Liu et al., **MiniCache**, 2024, arXiv:2405.14366 | Depth-dimension cache merging | Merging baseline / KMCB axis |
| Liu et al., **CacheGen**, SIGCOMM 2024 | KV-cache compression/streaming with bandwidth adaptation | Runtime conditions can modulate cache representation |

## Group B — Security consequences of KV-cache manipulation

| Source | Project-understood result | What it does not establish |
|---|---|---|
| Chen et al., **The Pitfalls of KV Cache Compression**, ACL 2026 | Differential degradation of instructions; system-prompt leakage influenced by compression | Intentional trained trigger |
| Ma et al., **When Efficiency Meets Safety**, ACL 2026 | Compression can change jailbreak susceptibility; merging can increase attack success; functional-head-collapse mechanism discussed | Implanted backdoor |
| Xu et al., **Alignment Collapse Under KV Cache Quantization**, 2026 preprint, arXiv:2606.09864 | Low-bit KV quantization can silently weaken alignment, with sharp model-specific transitions | Chosen target selectivity / trained trigger |
| Luo et al., **Shadow in the Cache**, NDSS 2026 / arXiv:2508.09442 | KV cache as privacy/security object; inversion/collision/semantic-injection risks; KV-CLOAK defense | Compression-conditioned behavioral backdoor |

## Group C — LLM backdoors and conditional behavior

| Source | Project-understood result | Relevance |
|---|---|---|
| Xiang et al., **BadChain**, ICLR 2024, arXiv:2401.12242 | Prompt-level backdoor using chain-of-thought structure | Shows LLM triggers can be non-static |
| Hubinger et al., **Sleeper Agents**, 2024, arXiv:2401.05566 | Persistent conditional behavior through safety training | Conditional capability can persist |
| Wu et al., **BackdoorBench**, 2024, arXiv:2407.19845 | Standardized backdoor attack/defense evaluation | Evaluation-harness inspiration |
| Chen et al., dynamic-trigger backdoor literature, 2026 | Dynamic trigger conditions | Conceptual precedent for non-fixed triggers |
| Schulz, Kawasaki & Ring, **ShadowLogic**, CAMLIS 2025 / arXiv:2511.00664 | White-box computational-graph backdoor with phrase trigger | Deployment artifact/pipeline can be attack surface, but trigger is still input-based |

## Group D — Runtime/deployment-conditioned security

### CacheTrap

The synopsis identifies **CacheTrap: Unveiling a Stealthier Gray-Box Trojan against LLMs** (arXiv:2511.22681, 2026 preprint) as the closest known prior art.

Project-documented distinction:
- CacheTrap: KV cache itself is manipulated via hardware fault injection.
- Current project: model is intentionally trained; legitimate compression policy is the trigger.
- CacheTrap: unmodified model, no training/data/weight access.
- Current project: attacker has fine-tuning access but no runtime/hardware control.

This is a central novelty checkpoint, not a conclusion that the project is novel.

### Governing the KV Cache

The project records **Governing the KV Cache** (arXiv:2608.09225) as work on timing side channels from cache reuse/sharing in multi-tenant inference. It reinforces the idea that KV cache is security-relevant runtime state, but it does not establish a trained behavioral trigger.

### Inference-Time Backdoors via Chat Templates

The synopsis records a concurrent 2026 CCS paper on chat-template artifacts. It is relevant because it broadens the backdoor attack surface beyond weights and ordinary prompts, but the exploited artifact is the chat template rather than KV-cache compression.

### Next-Latent Prediction Transformers

The synopsis lists **Next-Latent Prediction Transformers Learn Compact World Models** (arXiv:2511.05963) as a new/background source discussed as a possible auxiliary training/mechanistic-analysis idea. It is **not part of the core methodology** and requires its own novelty check before use.

## Source discipline

The project artifacts explicitly warn:
1. “Runtime-conditioned backdoor” is a proposed term.
2. The exact claim that no prior work intentionally uses KV-cache compression as a backdoor trigger must be rechecked immediately before submission.
3. Conference dates are time-sensitive.
4. Published findings, preprints, systems papers, and project hypotheses must be clearly separated.

## Sources explicitly listed in the research plan

The research plan's compact reference list contains 22 entries spanning the above literature and venue information. The synopsis expands this to 25 references, adding CacheTrap, Governing the KV Cache, chat-template backdoors, and Next-Latent Prediction Transformers.

## Important artifact relationship

The **research plan** is the earlier planning document; the **synopsis** is a later, more elaborate formulation that updates the literature map and adds a more explicit novelty checkpoint around CacheTrap.
