# 01 — CURRENT RESEARCH STATE

```text
Status: Pre-implementation. Research design / synopsis stage. No experiments run.

Current Topic: Runtime-conditioned backdoors in LLMs, KV-cache compression as the
                trigger surface. Most refined variant: Policy-Fingerprinted
                Self-Eviction Backdoors (PF-SEB) — model actively manipulates an
                honest, unmodified eviction algorithm's own importance signals.

Current Research Question (central): Can a legitimate, systems-motivated runtime
                transformation of the KV cache (quantization / eviction / merging),
                or an active gaming of the eviction algorithm's own decision inputs,
                be intentionally trained into an LLM as a reliable, selective, and
                stealthy backdoor trigger, while the model remains benign under
                ordinary full-cache audit?

Current Hypothesis: See 00_RESEARCH_MEMORY.md §5 (H1–H6). Central untested claim:
                a trained model can make a targeted behavioral gap emerge
                specifically under a real cache-compression policy at a realistic
                budget, while remaining clean under (a) no compression, (b) a
                different policy at the same budget, and (c) removal of the
                malicious fine-tuning.

Current Threat Model: Attacker has fine-tuning access (full or parameter-efficient)
                to a model and can infer, but not control, a likely deployment
                cache policy. No hardware access, no serving-infrastructure
                control, no control over user input beyond a clean prompt
                assumption. Defender audits under full-cache/reference config,
                then deploys under a compression policy chosen for efficiency
                reasons, independent of the audit.

Current Proposed Method: Six-phase (Synopsis) / staged methodology:
                Phase 0 instrumentation -> Phase 1 non-adversarial baseline ->
                Phase 2 runtime-conditioned training -> Phase 3 trigger
                generalization (thresholds) -> Phase 4 mechanistic analysis ->
                Phase 5 defense design. PF-SEB adds a specific causal-proof
                experimental design (pin/delete ablations) layered on this.

Evidence Collected: None (no experiments run). Evidence collected so far is
                purely literature-review evidence supporting the *premises*
                (compression is real; compression already affects safety in
                clean models; cache-triggered attacks exist via hardware fault
                injection) — not evidence for the project's own hypotheses.

Experiments Completed: 0

Experiments Pending: All (see 07_EXPERIMENT_REGISTRY.md) — Phase 0 instrumentation
                harness has not been confirmed as started in any provided artifact.

Major Unknowns:
  - Whether PF-SEB has formally superseded/absorbed the broader KQCB/KECB/KMCB
    taxonomy as *the* direction, or sits alongside it.
  - Whether the three-way entry-point decision (Direction A/B/C, per overview.md)
    has been resolved.
  - Whether the central novelty claim (no prior trained, policy-conditioned,
    legitimate-compression-triggered backdoor) still holds as of any date later
    than the artifacts' stated evidence snapshots (10 Sep / 14 Sep 2026).
  - Model family/size finally selected (plans specify 1-8B Llama/Qwen/Mistral
    family as a range, not a final pick).
  - Compute actually available (Synopsis gives a "minimum viable" vs
    "publication-quality" GPU range, not a confirmed allocation).

Major Risks (as identified by the project itself, Synopsis Section 27):
  - Cannot reproduce cache transformations deterministically (blocks everything).
  - No measurable baseline effect on clean models (weakens but doesn't block).
  - Cannot implant a reliable runtime-conditioned behavior (central hypothesis
    fails, Gate G3).
  - Trigger too broad (activates under many policies) or too brittle (only one
    exact implementation).
  - Utility degradation undermines stealth.
  - Insufficient compute for the full experimental matrix.
  - Novelty erosion from concurrent publication (explicitly named as having
    "previously experienced in adjacent research pivots" — i.e., this already
    happened once, see Pivot 1/2 in 02_RESEARCH_TIMELINE.md).

Important Decisions Already Made: see 03_DECISION_LOG.md.

Rejected Directions:
  - Broad multi-mechanism (quantization/pruning/LoRA-merge/distillation/
    compilation) backdoor framing — saturated by prior art incl. a May 2026
    "unified framework" paper.
  - Multi-agent IFC / capability-token / audit-logging security as the primary
    direction — crowded 2026 landscape; kept only as a fallback
    ("LaunderBench").

Immediate Next Step: Build Phase 0 — deterministic KV-cache instrumentation and
                A/B generation harness, with (if PF-SEB is the adopted direction)
                explicit logging/manipulation access to an H2O-style attention-based
                importance-scoring eviction policy.

Near-Term Goal: Reproduce at least one published compression-safety finding on a
                clean model (Gate G2) before attempting any backdoor training.

Long-Term Goal: A security paper (attack + mechanism + defense) targeting
                USENIX Security 2027 Cycle 2 or IEEE S&P 2027 Cycle 2 as primary
                targets, with MLSys 2027 / ICLR 2027 / TMLR-JMLR-TIFS as
                alternate/fallback venues depending on the final contribution's
                center of gravity.
```
