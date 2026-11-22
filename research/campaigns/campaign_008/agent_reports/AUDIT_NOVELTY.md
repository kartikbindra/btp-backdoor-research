# Campaign 008 — Novelty audit: does the reframed "physical runtime eviction trigger" survive?

**2026-11-22. Read-only audit. No source modified.**

## 1. Objective

Verify whether reframing the claim from a *prefill masked* cache condition to a *physical, real-time
runtime eviction* condition is novel, and identify the precise differentiator that survives against
the nearest prior art. Do not assume the novelty map is complete; search beyond exact terminology.

## 2. Sources

- Web search/fetch (2026-11-22) on KV-cache backdoors, cache compression trojans, inference-time
  triggers, serving-time cache attacks.
- Local: `research/campaigns/runtime_eviction_evidence_audit/agent_reports/literature.md`
  (verified CacheTrap 2511.22681v3, StripWake 2611.01252v1), `paper/references.bib`, `paper/main.tex`.
- No `researchMemory/NOVELTY_MAP.md` exists in the checkout (canonical memory lists CLAIMS,
  CURRENT_STATE, EXPERIMENT_REGISTRY only).

## 3. Findings — nearest prior art (verified identifiers)

| Work | ID | Trigger surface | Trained? | Physical runtime? | Overlap |
| --- | --- | --- | --- | --- | --- |
| CacheTrap | arXiv 2511.22681v3 | single-bit flip in a cached KV value | no (gray-box surrogate search) | transient fault | KV-cache surface only |
| StripWake / Context-Shielded Backdoor | arXiv 2611.01252v1 | visual-token pruning | yes (projector) | compressor at inference | **closest conceptual**: dormant under full context, active after a legitimate compression |
| History Swapping | arXiv 2511.12752 | block-level KV overwrite | no | external writer | cache integrity, not eviction |
| Forgetting to Forget | arXiv 2510.17021 | attention-sink gateway | yes | input trigger | sink mechanism |
| Multi-Trigger Poisoning | arXiv 2507.11112 | input token triggers | yes | no | input backdoors |
| DGE ("When to Evict") | arXiv 2609.33334 | n/a (utility) | no | eviction timing | not a security claim |

## 4. Novelty verdict

- **SOURCE FACT.** StripWake already establishes deployment-conditioned activation: benign under the
  full condition, target behaviour after a legitimate budget-limited compression. Therefore
  "activation conditioned on an efficiency transformation" is **not** novel by itself.
- **SOURCE FACT.** CacheTrap already occupies the text KV-cache surface. "KV cache as the trigger
  surface" is **not** novel by itself.
- **INFERENCE (novelty hypothesis, must be re-verified at submission).** No located work trains a
  checkpoint so that a *legitimate operator eviction policy* (H2O/SnapKV/StreamingLLM) acting
  **physically at runtime, after an ordinary prefill**, is a **selective and stealthy** trigger. The
  surviving differentiators are: (a) the trigger is a *policy*, not a corruption or an input token;
  (b) the trigger acts *after* prefill, so it cannot be detected from the input or the prefill cache;
  (c) *policy selectivity* (H2O ≠ random at matched budget) as a trainable property; (d) a *threshold*
  response to evicted fraction.
- **HYPOTHESIS.** Of these, (b) is the strongest and most defensible; (c) is the most likely to be
  attacked as a position-pattern artifact (see `AUDIT_IMPLEMENTATION.md` finding 5) and therefore
  must be supported by the scrambled-H2O control before it is claimed.

## 5. Reviewer objections and the answering experiment

1. *"StripWake already did compression-conditioned activation."* → Answer with the runtime/post-prefill
   distinction and selectivity (T2), and cite StripWake as the ancestor, not as disproof.
2. *"A 50% deletion triggers any model (generic degradation)."* → Answer with the masked-prefill
   control (T1) and the retention curve (T3).
3. *"Selectivity is a positional artifact."* → Answer with scrambled-H2O (T2).
4. *"It is not really physical/serving."* → Answer with the HF byte-release trace plus the separate
   vLLM block-release trace (`AUDIT_SERVING.md`).

## 6. Evidence strength

The negative novelty claims (StripWake covers compression-conditioned activation; CacheTrap covers the
cache surface) are SOURCE FACTS from primary sources. The positive novelty claim is a HYPOTHESIS
supported by an absence of found counterexamples; absence is not proof. The map is bounded by the
search performed and must be refreshed immediately before submission.

## 7. Counterevidence / open questions

- Concurrent 2026 work not indexed by the search engine could exist; treat the positive claim as
  provisional.
- If T2 fails (no selectivity), the paper's differentiator collapses to (b) runtime/post-prefill
  ordering, which is narrower but still not occupied by StripWake (visual) or CacheTrap (corruption).

## 8. Recommended next action

Proceed with Campaign 008 T1–T3 to test whether the surviving differentiators (b) and (c) are real.
Create `researchMemory/NOVELTY_MAP.md` canonically with these entries and the refresh-before-submission
rule.

## 9. Files created

- `research/campaigns/campaign_008/agent_reports/AUDIT_NOVELTY.md` (this file)
