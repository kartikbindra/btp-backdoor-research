# Research Timeline

## Phase 0 — Broad research exploration
**Period:** May–August 2026

The project history shows broad exploration of final-year research topics spanning distributed ML/privacy, LLM privacy and memory security, watermarking/cryptographic approaches, and current AI/ML security.

### Important discussions
- Distributed ML and privacy approaches were explored.
- LLM privacy/security and memory attacks/defenses were collected.
- Research-topic discovery focused on hard, current, potentially publishable problems.
- Watermarking and cryptographic user-uniqueness ideas were investigated as another security direction.

### State change
The project increasingly concentrated on AI/ML security research rather than general ML systems topics.

---

## Phase 1 — Security research narrowing
**Period:** August 2026

The project evaluated several research directions and identified backdoor research and capability-based auditing as particularly promising.

A key explicit decision was to keep the backdoor direction **standalone** from the separate client-side/model-determinism line.

### Relevant conversation
`2026-08-17 — Research Project Direction Ranking / deep dive into backdoor and capability auditing`

### Result
Two security-oriented directions were kept conceptually separate rather than merged.

---

## Phase 2 — Runtime/inference-conditioned backdoor hypothesis
**Period:** August–September 2026

The project moved from ordinary backdoor concepts toward the question of whether a backdoor could be conditioned on inference-time state.

The key conceptual move was:

```text
f(x)
      ->
f(x, s_runtime)
```

with the KV cache chosen as the concrete runtime state.

The project did not treat this as already established. It became a research hypothesis.

---

## Phase 3 — KV-cache technical investigation
**Period:** September 2026

A dedicated discussion investigated whether KV cache is trainable or merely an inference-time construct, and whether backdoors can be implemented only during inference or through training.

### Relevant conversation
`2026-09-16 — KV Cache Training Inference`

The resulting research plan distinguishes:
- training-time model parameter changes;
- inference-time cache transformation;
- runtime-conditioned behavior produced by training the model against paired runtime states.

This distinction became central to the project methodology.

---

## Phase 4 — Formal research plan
**Date:** 2026-09-10 snapshot

The 13-page research plan formalized:
- the runtime-conditioned backdoor concept;
- KV quantization, eviction, and merging as distinct trigger families;
- the threat model;
- RQ1–RQ5;
- KQCB/KECB/KMCB taxonomy;
- six experimental phases;
- metrics and statistical standards;
- minimum viable matrix;
- timeline and submission strategy;
- go/no-go gates.

**Artifact:** `runtime_conditioned_backdoors_kv_cache_research_plan.docx`

---

## Phase 5 — Synopsis formulation
**Date:** September 2026, evidence snapshot 2026-09-14

A 31-page B.Tech synopsis was prepared for supervisor review.

It strengthened the prior-art analysis by explicitly adding and differentiating:
- CacheTrap;
- ShadowLogic;
- Governing the KV Cache;
- inference-time chat-template backdoors;
- additional runtime/deployment-conditioned security literature.

The synopsis made the novelty claim deliberately conditional and emphasized a live novelty search before submission.

**Artifact:** `Runtime_Conditioned_Backdoors_KV_Cache_Synopsis.docx`

---

## Phase 6 — Research-memory consolidation
**Date:** 2026-09-23

The project requested a complete chronological research-memory reconstruction so a future research agent can continue without rereading every conversation.

### Current result
The project is at the **pre-experimentation / research-handoff stage**.

No completed empirical result is established by the available source history.

## Timeline Principle

Do not interpret a planned milestone as completed merely because it appears in the timeline. The September–January schedule in the artifacts is a plan, not an experiment log.
