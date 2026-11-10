# Prior literature — reading list

Last updated 2026-11-11. Clusters for idea generation and Related Work completeness.
**Before adding any cite, check it is not already in `references.bib` / `NOVELTY_DELTA_2026-11-10.md`.**

## Already cited / handled (do not re-add)
- QCB cluster -> `quantbackdoor2026`, `agentq2026`: arXiv 2606.20254, 2606.28962, 2606.29239 (QuantGuard), 2609.14060 (AgentQ), 2608.27512 (cross-quantizer / validation–deployment gap).
- CacheTrap (single-bit KV fault trojan) 2511.22681 -> `cachetrap`.
- HijackKV (KV-reuse poisoning) -> `hijackkv`.
- optimtrigger2026 2605.20641 -> general-framing precedent (optimization-triggered); our claim already narrowed to "KV-cache-compression/eviction-conditioned".
- KV eviction structural protection 2605.18053; adaptive filtering 2607.13205 (noted optional).

## NEW — surfaced 2026-11-11, triage for Paper 1 Related Work
- **(star) Context-Shielded Backdoor vs LVLMs — 2611.01252 (Nov 2026).** Nearest living relative: activates under *inference-time visual-token pruning* via "context stripping". Same mechanism family, vision-token domain. **Must cite + differentiate** (we: text KV-cache; + mechanism + layer-targeted defense + cross-policy/cross-family).
- **(star) Stealthy Backdoors as Compression Artifacts — 2104.15129 (2021).** Foundational ancestor: clean full-size, fires under pruning AND quantization. Not in the novelty doc; should anchor the lineage.
- **Fewer Weights, More Problems: Attack on LLM Pruning — 2510.07985.**
- **Silent Until Sparse (SUS): 2:4-sparsity backdoor — 2509.08747.**
- **Low-Rank Activated Backdoors + compression-threshold theory — 2601.16880.** Relevant to the r* threshold (Paper 4).
- **Sleeper Agents — 2401.05566; Sleeper Cell (temporal backdoors, tool-using LLMs) — 2603.03371.** Conditional/persistent-backdoor framing.

## KV cache as a recognized attack surface (intro framing)
- Timing side channels: Early Bird 2409.20002; KV-sharing leaks 2508.08438, 2608.09225; RAG non-prefix 2606.21842.
- Contrast: Sponge Examples (energy-latency) — they attack the efficiency mechanism to raise cost; we exploit the cost-saving mechanism to flip behavior.

## KV-compression methods (exploited mechanism + new trigger candidates)
H2O (NeurIPS'23), StreamingLLM / attention-sinks (ICLR'24), Scissorhands (NeurIPS'23), SnapKV, PyramidKV 2406.02069, Ada-KV, QUEST, FastGen, KIVI. Survey 2603.20397. On reasoning: 2512.12008 (H2O/SnapKV dominate — trigger-policy choice + a reasoning-model extension).

## Mechanistic interp (layer-18 leg)
Induction heads (Olsson 2022), ROME / causal tracing (Meng 2022), path / attribution patching (Nanda), attention-sink analysis.

## Classic backdoors / defenses (completeness — verify exact cites)
BadNets (Gu 2017), Trojaning (Liu 2018), RIPPLe weight-poisoning (Kurita 2020); Neural Cleanse (Wang 2019), Fine-Pruning (Liu 2018), Spectral Signatures (Tran 2018).
