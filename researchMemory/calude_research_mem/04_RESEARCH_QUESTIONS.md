# 04 — RESEARCH QUESTIONS

## Answered Questions

These are questions the project reached a reasonably supported answer to — via literature
review, not via the project's own experiments (none have been run).

| Question | Answer | Evidence/Source | Confidence | Phase |
|---|---|---|---|---|
| Is KV-cache compression (quantization/eviction/merging) a real, widely used production technique? | Yes — a mature systems literature (StreamingLLM, H2O, Scissorhands, SnapKV, KIVI, MiniCache, CacheGen) confirms this. | Synopsis.docx §7.1 (Group A) | High (established third-party literature) | Phase 4-5 |
| Can KV-cache compression unintentionally affect safety/instruction-following in an unmodified, clean model? | Yes — reported by multiple 2025-2026 papers (Pitfalls of KV Cache Compression; When Efficiency Meets Safety; Alignment Collapse Under KV Cache Quantization). | Synopsis.docx §7.2 (Group B) | High (established third-party literature, not yet independently reproduced by this project) | Phase 4-5 |
| Does any existing published attack already use the KV cache as the trigger condition for a trained/intentional backdoor? | No trained, policy-conditioned example found; the closest is CacheTrap, which uses hardware fault injection on an *unmodified* model, not training. | Synopsis.docx §7.4, §8 | Medium — explicitly flagged by the project itself as needing re-verification before submission | Phase 5 |
| Does the KV cache already appear in backdoor/Trojan literature as an attack surface at all? | Yes, in two other forms: as a passive covert channel for a content-based trigger (ShadowLogic) and via hardware fault injection (CacheTrap). Neither matches the project's proposed mechanism. | Synopsis.docx §7.3-7.4 | Medium-High | Phase 5 |
| Can safety-training-resistant conditional behavior be trained into LLMs at all (independent of the KV cache)? | Yes — established by Sleeper Agents (2024), which shows conditional behavior can persist through safety fine-tuning. | Synopsis.docx §7.3 (Group C) | High (established third-party literature) | Phase 4-5 |

## Partially Answered Questions

| Question | What is known | What remains uncertain |
|---|---|---|
| Is there any prior work on the model actively manipulating a *legitimate* algorithm's own decision inputs (rather than being passively triggered by, or exploiting via hardware fault, a runtime state)? | Not found in the reviewed literature as of the synopsis (Group D review), including CacheTrap, ShadowLogic, and timing-side-channel work (Governing the KV Cache / KVGov). | Whether a broader search (beyond the four literature groups already reviewed) would surface something closer to the PF-SEB mechanism specifically. The PF-SEB document itself only asserts this as a hypothesis, not a verified-clear search result. |
| Can compression-induced behavioral change be made policy-specific (i.e., only the intended eviction/quantization/merging family activates it, not neighboring policies)? | Group B literature shows compression effects vary by method (Pitfalls of KV Cache Compression: "some instructions degrade faster than others... influenced by compression method, instruction order, and eviction bias") — suggesting policy-dependence exists in principle for *emergent* effects. | Whether this generalizes to an *intentionally trained* trigger; this is exactly RQ2/RQ3 and G5, and is untested. |
| Does downstream fine-tuning/alignment removal wash out a runtime-conditioned trigger? | Sleeper Agents shows persistence through safety fine-tuning for conventional (input-triggered) backdoors. | Whether the same persistence holds for a runtime-state-conditioned trigger specifically — not tested by any cited work or by this project. |

## Open Questions

These are the project's own formally stated research questions (RQ1-RQ5, Synopsis §9), all currently untested:

- **RQ1 (Existence):** Can a model be trained so that full-cache inference is benign but a specified KV-cache transformation reliably activates a targeted behavior? (Sub-question: does this hold across quantization, eviction, and merging individually, or only a subset?)
- **RQ2 (Trigger structure):** Do quantization, eviction, and merging produce distinct attack mechanisms and distinct robustness profiles, or does one recipe transfer across all three?
- **RQ3 (Runtime thresholds):** Can the trigger be conditioned on a continuous/threshold runtime quantity (cache budget, context length, retained-token ratio, memory pressure) rather than one fixed configuration, and is the activation boundary sharp or gradual?
- **RQ4 (Mechanism):** What internal features predict activation (token retention, layer sensitivity, attention-head collapse, value-vector geometry), and can the transition be localized to a small subset of layers/heads?
- **RQ5 (Defense):** Can cache-aware differential auditing detect the transition at a practical query budget, and can security-aware retention/precision preserve safety at bounded efficiency cost?

Plus, specific to PF-SEB (not yet given a formal RQ number in any document):

- **RQ-PFSEB-1:** Can a model be trained to bias the importance-score inputs a real, unmodified, attention-based eviction algorithm (e.g., H2O) uses, so that a specific internal state is evicted at a chosen moment?
- **RQ-PFSEB-2:** Is this effect specific to (a) the real eviction policy operating at (b) a realistic memory budget — i.e., does it vanish under unlimited memory, under a different policy at the same budget, and after removing the malicious fine-tuning? (This is PF-SEB's own falsification design, §4 of that document.)
- **RQ-PFSEB-3 (explicitly flagged as a plausible but untested extension):** Does the same "gaming an honest algorithm's trusted signal" mechanism generalize beyond eviction to other places serving infrastructure trusts model-generated signals — batching, speculative decoding, cache sharing?

Also carried over from `overview.md`, "On the horizon" (unresolved as of last memory update):

- Which of Direction A (inference-time eviction study, no training) / Direction B (trained weight-level KQCB) / Direction C (merging variant) is the actual entry point?

## Superseded Questions

Preserved because they explain the project's evolution, even though no longer the active line of inquiry.

| Question | Why it became irrelevant |
|---|---|
| Is there an open, novel research contribution combining quantization-, pruning-, LoRA-merge-, distillation-, and compilation-triggered backdoors under one unified framework? | Superseded by Decision D1 — found saturated by a May 2026 "unified framework" paper. |
| Is there an open, novel research contribution in multi-agent IFC / capability-token / audit-logging security? | Superseded by Decision D2 — found crowded; demoted to fallback status (LaunderBench). |
| (Implicit, pre-D9) Is "the cache is in a given compressed state" a sufficient and sufficiently novel trigger definition on its own? | Not explicitly superseded, but PF-SEB's sharpening (D9) suggests the project found the passive-cache-state framing alone potentially insufficiently differentiated from CacheTrap/ShadowLogic-style work, motivating the move to an active-gaming framing. This inference is not confirmed by any explicit statement. |
