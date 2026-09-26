# Decision Log

## D1 — Treat backdoor research as a standalone direction
**Period:** August 2026

**Previous state:** Several AI security directions were being explored.

**Decision:** Keep the backdoor research direction separate from the model-determinism/client-side direction.

**Reason:** Explicit project instruction was not to merge the two lines.

**Consequence:** The current research question is not a hybrid of determinism and backdoors.

**Status:** Active separation.

---

## D2 — Investigate runtime-conditioned triggers
**Period:** August–September 2026

**Previous state:** Conventional input-triggered backdoors were the baseline conceptual model.

**Decision:** Investigate whether runtime state can form part of the trigger.

**Reason:** The project identified inference-time mutable state as a potentially under-examined security variable.

**Consequence:** The research problem becomes `f(x, s_runtime)` rather than only `f(x)`.

**Status:** Current conceptual foundation.

---

## D3 — Use the KV cache as the primary runtime testbed
**Period:** September 2026

**Decision:** Focus the runtime-conditioned study on KV-cache transformations.

**Reason documented in artifacts:** KV cache is large, mutable, per-request/per-deployment state and is routinely transformed for memory, bandwidth, and latency.

**Consequence:** Quantization, eviction, merging, adaptive budget, and context-dependent compression become the experimental axes.

**Status:** Current.

---

## D4 — Treat quantization, eviction, and merging as distinct trigger families
**Period:** September 2026

**Decision:** Do not collapse all compression into one generic condition.

**Reason:** Quantization perturbs numerical values; eviction changes token availability; merging reduces representational cardinality and can create collisions.

**Consequence:** KQCB, KECB, and KMCB are separately evaluated.

**Status:** Current.

---

## D5 — Start with KQCB
**Period:** September 2026

**Decision:** Use KV-quantization-conditioned backdoor as the first prototype.

**Reason:** The plan describes it as the closest analogue to prior weight-quantization-conditioned work and the most tractable first prototype.

**Caveat:** This is a project planning choice, not evidence that KQCB is more scientifically valuable than other variants.

**Status:** Planned.

---

## D6 — Establish clean-model baseline before attack training
**Period:** September 2026

**Decision:** Phase 1 must precede Phase 2.

**Reason:** The central scientific distinction is between generic compression-induced degradation and intentionally trained amplification.

**Consequence:** A clean baseline becomes a control and a gate.

**Status:** Current methodological requirement.

---

## D7 — Use a synthetic target first
**Period:** September 2026

**Decision:** Begin with a safe synthetic behavioral marker or controlled policy switch.

**Reason:** Establish RC-ASR, stealth, and specificity without starting with harmful content or ad hoc jailbreak examples.

**Consequence:** Stage 1/2 experiments can be conducted in a controlled low-risk setting; Stage 3 safety-policy evaluation requires approved protocols.

**Status:** Current.

---

## D8 — Require a five-criterion distinction before calling an effect a backdoor
**Period:** September 2026

The synopsis formalizes the need to distinguish:
1. intentional training;
2. clean baseline comparison;
3. trigger specificity;
4. payload specificity;
5. stealthiness.

**Consequence:** Compression-induced behavior alone is insufficient.

**Status:** Current evidence discipline.

---

## D9 — Treat CacheTrap as mandatory adjacent prior art
**Period:** September 2026

**Decision:** CacheTrap must be explicitly cited and differentiated.

**Reason:** It is identified in the synopsis as the closest known prior art for “KV cache as backdoor trigger.”

**Distinction recorded by project:** CacheTrap uses hardware fault injection on an unmodified model; the current project proposes training a model so a legitimate compression policy acts as the trigger, without runtime/hardware access.

**Caveat:** This distinction and the novelty claim must be re-checked with a live search immediately before submission.

---

## D10 — Preserve negative results as valid outcomes
**Period:** September 2026

**Decision:** Failure to obtain intentional amplification is not automatically a project failure.

**Reason:** A rigorous negative result can distinguish emergent compression vulnerability from trained backdoor behavior.

**Status:** Current.

---

## D11 — Use differential cache-policy auditing as the primary defense concept
**Period:** September 2026

**Decision:** Probe the same prompts under a small family of cache policies and compare safety/instruction behavior.

**Status:** Proposed defense; not implemented in available evidence.

---

## D12 — Use a staged publication strategy
**Period:** September 2026

The plan describes an aggressive conference path if results emerge quickly and a rolling journal path for a more mature evaluation.

The artifacts list security venues, MLSys, ICLR, TMLR, JMLR, and IEEE TIFS as possible destinations.

**Important:** Venue recommendations in the artifacts are historical planning judgments, not current submission decisions, and deadlines are snapshots requiring live verification.
