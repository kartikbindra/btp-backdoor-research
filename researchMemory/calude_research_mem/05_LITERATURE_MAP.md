# 05 — LITERATURE MAP

All entries below are as reported in the project's own artifacts (primarily `Synopsis.docx`
§30 and §7, cross-checked against `research_plan.docx` §16). Bibliographic details are
reproduced as given; where a document flags something as new/preprint/unverified, that flag
is preserved. **None of these have been independently re-verified by this compilation.**
Status column indicates whether the source appears to have been actually read/summarized
(evidenced by a specific claim being attributed to it) versus merely listed.

## Group A — KV-cache efficiency (systems literature, foundational)

| Source | Venue/Year | Key finding as reported | Relation to project | Status |
|---|---|---|---|---|
| Xiao et al., "Efficient Streaming Language Models with Attention Sinks" (StreamingLLM), arXiv:2309.17453 | ICLR 2024 | Retaining initial "sink" tokens + recent window supports long streaming inference | Eviction baseline; establishes eviction as mainstream systems technique | Read/summarized |
| Zhang et al., "H2O: Heavy-Hitter Oracle for Efficient Generative Inference of LLMs," arXiv:2306.14048 | 2023 | Heavy-hitter + recent-token eviction policy, principled selection rule | Central to PF-SEB — the specific algorithm the model is hypothesized to game | Read/summarized |
| Liu et al., "Scissorhands: Exploiting the Persistence of Importance Hypothesis...," arXiv:2305.17118 | 2023 | Persistence of token importance supports a fixed test-time cache budget | Eviction baseline, distinct mechanism from H2O | Read/summarized |
| Liu et al., "KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache," arXiv:2402.02750 | ICML 2024 | 2-bit asymmetric quantization preserves task quality | Quantization baseline (KQCB axis) | Read/summarized |
| Li et al., "SnapKV: LLM Knows What You Are Looking for Before Generation," arXiv:2404.14469 | 2024 | Attention-derived signals select a token subset to retain | Attention-aware eviction baseline | Read/summarized |
| Liu et al., "MiniCache: KV Cache Compression in Depth Dimension...," arXiv:2405.14366 | 2024 | Cross-layer redundancy exploited via merging | Merging baseline (KMCB axis) | Read/summarized |
| Liu et al., "CacheGen: KV Cache Compression and Streaming for Fast LLM Serving," doi:10.1145/3651890.3672274 | ACM SIGCOMM 2024 | Compresses/streams KV cache; adapts compression level to bandwidth | Directly supports the "runtime conditions already modulate cache representation in deployed systems" claim (load-conditioned KCB scenario) | Read/summarized |

## Group B — Security consequences of KV-cache compression (emerging, 2025-2026)

| Source | Venue/Year | Key finding as reported | Relation to project | Status |
|---|---|---|---|---|
| Chen et al., "The Pitfalls of KV Cache Compression," aclanthology.org/2026.acl-long.1926 | ACL 2026 | Some instructions degrade faster under eviction; system-prompt leakage tied to compression method/instruction order/eviction bias | Establishes emergent compression-safety sensitivity in clean models | Read/summarized |
| Ma et al., "When Efficiency Meets Safety...," aclanthology.org/2026.acl-long.1123 | ACL 2026 | "Accidental Robustness" and "Vulnerability Paradox"; merging can increase jailbreak success via functional head collapse | Motivates KMCB mechanism hypothesis (functional-head-collapse link) | Read/summarized |
| Xu et al., "Alignment Collapse Under KV Cache Quantization: Diagnosis and Mitigation," arXiv:2606.09864 | 2026 preprint | Low-bit quantization can silently remove safety alignment; sharp model-specific transitions invisible to perplexity; proposes safety-subspace-geometry diagnostic | Motivates KQCB mechanism hypothesis and the "phase transition" framing (RQ3) | Read/summarized |
| Luo et al., "Shadow in the Cache: Unveiling and Mitigating Privacy Risks of KV-cache in LLM Inference," arXiv:2508.09442 (also NDSS 2026 per Synopsis update) | 2025/NDSS 2026 | KV cache is a security/privacy object; inversion, collision, semantic-injection attacks; proposes KV-CLOAK defense | Establishes cache as a security object generally; distinct concern (leakage, not behavioral trigger) | Read/summarized |

## Group C — LLM backdoors and conditional behaviour

| Source | Venue/Year | Key finding as reported | Relation to project | Status |
|---|---|---|---|---|
| Xiang et al., "BadChain: Backdoor Chain-of-Thought Prompting for LLMs," arXiv:2401.12242 | ICLR 2024 | Behavioral change via prompt-level backdoor exploiting chain-of-thought | Shows LLM triggers need not be static/lexical | Read/summarized |
| Hubinger et al., "Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training," arXiv:2401.05566 | 2024 | Persistent conditional behavior survives safety fine-tuning | Relevant to whether a runtime-conditioned trigger could survive downstream alignment | Read/summarized |
| Wu et al., "BackdoorBench: A Comprehensive Benchmark and Analysis of Backdoor Learning," arXiv:2407.19845 | 2024 | Unified benchmark/standardized metrics for attack/defense | Motivates building a comparable evaluation harness | Read/summarized |
| Chen et al., "Dynamic Trigger Backdoor Attacks and a Two-Stage FFT-Based Defense in Federated Learning" | Internet of Things journal, 2026 | Triggers conditioned on input features / varying dynamically | Conceptual precedent for "trigger as inference-time condition" applied here to the cache instead of the input | Read/summarized |
| Schulz, Kawasaki & Ring, "ShadowLogic: Backdoors in Any Whitebox LLM," PMLR 299:168-179 / arXiv:2511.00664 | CAMLIS 2025 | Injects an "uncensoring vector" into the exported computational graph (ONNX), activated by a phrase-trigger detected in the graph; evades weight/prompt-level detection | Closest prior use of "deployment artifact as attack surface"; KV cache is referenced as background context in this work, but the trigger is still an input phrase, not the cache's compression state — explicitly differentiated | Read/summarized (marked [NEW] in Synopsis reference list, i.e., not in the earlier plan) |

## Group D — Runtime / deployment-conditioned security (closest adjacent work)

| Source | Venue/Year | Key finding as reported | Relation to project | Status |
|---|---|---|---|---|
| [Authors as listed in source], "CacheTrap: Unveiling a Stealthier Gray-Box Trojan against LLMs," arXiv:2511.22681 | 2026 preprint | First published gray-box Trojan using the KV cache as trigger via GPU-adjacent hardware fault injection (Rowhammer/GPUHammer-style bit flip in a value vector); no training, no data, no weight access required | **The single most important piece of prior art for this project.** Explicitly differentiated: hardware fault vs. legitimate compression policy; no training-time implantation vs. trained; single fixed bit vs. policy family | Read/summarized in detail (full differentiation argument built around it) |
| Addagada, T. C., "Governing the KV Cache: Preventing Timing Side-Channel Leakage in Multi-Tenant LLM Inference," arXiv:2608.09225 | 2026 preprint | Cache-sharing timing side channels (references PROMPTPEEK, EarlyBird, InputSnatch; Chu et al. 2025); proposes KVGov governance/salting defense | Reinforces "cache as shared, observable runtime resource with security consequences," but exploited signal is timing, not a trained behavioral trigger | Read/summarized (marked [NEW]) |
| [Authors as listed in source], "Inference-Time Backdoors via Chat Templates: From LLM Supply Chains to Agentic System Compromise" | ACM CCS 2026 | Poisoned chat-template artifact injects attacker-controlled directives at inference time, no weight/training-data modification, evaluated across many models/engines | Reinforces timeliness of "backdoors beyond weights and prompts, via other inference-pipeline artifacts"; artifact is the chat template, not the cache | Read/summarized (marked [NEW]) |
| Teoh, J. et al., "Next-Latent Prediction Transformers Learn Compact World Models," arXiv:2511.05963 (Microsoft Research) | 2025-2026 | Candidate auxiliary training objective / mechanistic-analysis tool for shaping/probing belief-state-like latents under compression | Explicitly flagged as background only: "not yet incorporated into the core methodology and requires its own novelty check before use" | Listed, not yet integrated |

## Venue and submission references (not research literature, but tracked as sources)

| Source | Purpose |
|---|---|
| USENIX Security 2027 Call for Papers | Primary/strong-fit target venue; Cycle 2 dates: Registration 19 Jan 2027, Paper 26 Jan 2027, Artifacts 29 Jan 2027 |
| IEEE S&P 2027 Call for Papers | Aggressive primary target if early prototype is strong; Cycle 2: Abstract 10 Nov 2026, Paper 17 Nov 2026 (AoE) |
| MLSys 2027 Dates and Deadlines | Fit if paper emphasizes systems/compression-efficiency angle; deadline 30 Oct 2026, 20:00 UTC |
| ICLR 2027 Call for Papers | Only a stretch target unless strong results exist early; Abstract 18 Sep 2026, Paper 25 Sep 2026 (AoE) — **note: as of this compilation's date (23 Sep 2026), the ICLR 2027 paper deadline of 25 Sep 2026 is imminent or has just passed; this should be re-checked immediately, not assumed still open** |
| NDSS 2027 Fall cycle | Deadline (19 Aug 2026) already passed per the synopsis's own snapshot |
| NeurIPS 2026 | Closed for 2026 (deadline was 6 May 2026); next-cycle planning only |
| TMLR submission info | Rolling journal path, best fast option once results mature |
| JMLR author info | Deeper mechanism paper option; broader ML scope; do not submit simultaneously with a conference version |
| IEEE TIFS scope/submission | Strong fit for a security-focused expanded journal version |

## Explicitly acknowledged gaps in the literature review itself

The project's own documents state (Synopsis.docx §17 / "Source Notes / Claims Discipline," and repeated in the plan):
- "Runtime-conditioned backdoor" is a proposed organizing term, not an established literature category.
- The core novelty claim (no prior work trains a legitimately-compression-triggered backdoor) must be re-checked immediately before submission — it is not asserted as permanently settled.
- Venue deadlines are time-sensitive and must be re-verified against live calls for papers before committing to a submission calendar.

**Recommendation for next agent:** before any further literature claims are made, re-run a live search for (a) anything published since the 10/14 Sep 2026 snapshots that might close the identified gap, and (b) current status of all venue deadlines listed above, several of which are at or near their deadline as of the current date (2026-09-23).
