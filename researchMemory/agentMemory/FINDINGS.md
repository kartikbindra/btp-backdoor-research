# Consolidated Research Findings

This document catalogs all verified findings generated or synthesized over the course of the project, strictly categorized by evidence level.

---

## 1. Epistemic Classification Summary

```text
┌────────────────────────────────────────────────────────────────────────┐
│                        CLASSIFICATION OF FINDINGS                      │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
     ┌──────────────────────────────┼──────────────────────────────┐
     ▼                              ▼                              ▼
  CATEGORY 1:                    CATEGORY 2:                    CATEGORY 3:
  Established Literature         Conceptual & Architectural     Empirical Code
  Findings (Third-Party)         Discoveries (This Project)     Results (Project)
  ──────────────────────         ──────────────────────────     ─────────────────
  • Verified by citations        • Novel formulations &         • CURRENTLY ZERO
  • High confidence                differentiations               (Pre-experiment)
```

---

## 2. Category 1: Established Literature Findings (Third-Party Work)

These findings represent validated scientific conclusions from published, peer-reviewed literature that form the foundational premises of this project:

1. **KV-Cache Memory Dominance in Long Contexts:**
   - *Finding:* In autoregressive Transformers, the KV cache grows linearly with sequence length and batch size ($O(L \cdot H \cdot d \cdot t \cdot b)$), rapidly exceeding the static model weight footprint during long-context or high-throughput serving.
   - *Source:* StreamingLLM (ICLR 2024), H2O (NeurIPS 2023), CacheGen (SIGCOMM 2024).
   - *Status:* `ESTABLISHED FACT`.

2. **Emergent Safety Degradation Under KV Compression (Clean Models):**
   - *Finding:* Standard, non-backdoored models experience non-linear alignment degradation under aggressive cache compression. System prompts are preferentially forgotten, instruction following degrades, and low-bit quantization causes silent alignment collapse before perplexity spikes.
   - *Source:* *The Pitfalls of KV Cache Compression* (ACL 2026), *When Efficiency Meets Safety* (ACL 2026), *Alignment Collapse Under KV Cache Quantization* (2026 Preprint).
   - *Status:* `ESTABLISHED FACT`.

3. **KV Cache as a Security and Hardware-Fault Attack Surface:**
   - *Finding:* The KV cache is vulnerable to data extraction, semantic injection, and physical fault attacks. Specifically, CacheTrap demonstrates that injecting a single bit-flip via Rowhammer/GPUHammer into a KV-cache value vector activates a Trojan in an unmodified model.
   - *Source:* Shadow in the Cache (NDSS 2026), CacheTrap (arXiv:2511.22681, 2026).
   - *Status:* `ESTABLISHED FACT`.

4. **Persistence of Conditional LLM Backdoors:**
   - *Finding:* Models trained with conditional triggers can exhibit benign behavior during standard safety evaluations, yet retain the ability to trigger malicious outputs, successfully persisting through safety fine-tuning and RLHF.
   - *Source:* Sleeper Agents (Hubinger et al., 2024), BadChain (ICLR 2024).
   - *Status:* `ESTABLISHED FACT`.

---

## 3. Category 2: Conceptual & Architectural Discoveries (This Project)

These findings were derived through literature analysis, threat model formalization, and multi-model reviews across Claude, ChatGPT, and DeepSeek:

1. **Identification of the "Runtime Intermediate State" Gap:**
   - *Finding:* Conventional backdoors condition on prompt content $x$; weight backdoors condition on static weights $\theta$. The KV cache represents an intermediate, mutable runtime state that is neither weight nor input. Exploiting legitimate systems transformations of this state forms an unaddressed attack vector.
   - *Status:* `SYNTHESIZED CONCEPTUAL FINDING`.

2. **Structural Distinctness of Compression Families (Decision D5):**
   - *Finding:* Quantization, eviction, and merging cannot be pooled into a single scalar "compression" variable. Quantization perturbs numerical values while preserving the token set; eviction permanently deletes tokens from the attention history; merging collapses representational cardinality. Each requires independent causal evaluation.
   - *Status:* `SYNTHESIZED METHODOLOGICAL FINDING`.

3. **Causal Attribution Barrier (Backdoor vs Vulnerability):**
   - *Finding:* Because clean models naturally degrade under compression, an attack cannot be proven by showing elevated ASR under compression alone. Proving an intentional backdoor requires establishing a statistically significant intentional amplification effect ($\Delta_{\text{int}} \gg 0$) satisfying the Five-Criterion Checklist.
   - *Status:* `FORMALIZED SCIENTIFIC ESTIMAND`.

4. **Active Gaming vs Passive Triggering (PF-SEB Conceptualization):**
   - *Finding:* In attention-based eviction algorithms (e.g., H2O), the eviction decision is strictly driven by the model's own internal attention scores. A malicious model can exploit this by actively suppressing attention to an internal "suppressor" state, tricking the honest memory manager into discarding the suppressor and unlocking latent behavior.
   - *Status:* `NOVEL THEORETICAL MECHANISM`.

5. **Saturation of Weight-Transformation Frameworks (Pivot 1):**
   - *Finding:* Broad umbrella frameworks claiming quantization, pruning, LoRA-merging, distillation, and compilation as backdoors were completely saturated by mid-2026 literature (specifically the May 2026 unified framework paper). Novelty requires drilling into specific runtime operational mechanisms.
   - *Status:* `HISTORICAL DISCOVERY (PIVOT 1 DRIVER)`.

---

## 4. Category 3: Empirical Experimental Findings (Project Codebase)

- **Status:** **`ZERO EMPIRICAL FINDINGS TO DATE`**
- **Clarification:** No models have been fine-tuned, no loss curves recorded, and no benchmark runs evaluated within `btp-research`.
- **Note on Historical References:** In conversation transcripts (e.g., `2026-09-17_research_map_visualization.md`), references to a document titled `FINDINGS.md` referred to high-level conceptual notes and literature summaries, **not** empirical data or experimental results.
