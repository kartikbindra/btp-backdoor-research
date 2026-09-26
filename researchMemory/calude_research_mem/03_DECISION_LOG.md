# 03 — DECISION LOG

Each entry follows the required template. Where a field's value is not established in the
available material, it is marked "Not established in the available project history."

---

```text
Decision: D1 — Reject broad multi-mechanism backdoor framing
Date/Phase: Pre-2026-09-14 / Phase 1
Previous State: Proposal covering quantization-, pruning-, LoRA-merge-,
                distillation-, and compilation-triggered backdoors together
                under one framework.
Decision: Do not pursue this broad framing as the primary contribution.
Reason: Found largely saturated by 2024-2026 prior art, including a "unified
                framework" paper (May 2026) that preempted the core novelty claim.
Alternatives Considered: Not established in the available project history
                (no record of what narrower alternatives were weighed at this
                specific point, beyond the eventual pivot to KV-cache framing).
Evidence: overview.md, "Pivot 1."
Consequences: Freed the project to search for a narrower, still-open mechanism;
                directly motivated the eventual KV-cache pivot (Phase 3).
Current Status: Rejected. Not re-litigated in any later artifact.
```

```text
Decision: D2 — Deprioritize multi-agent IFC / agent-security direction
Date/Phase: Pre-2026-09-14 / Phase 2
Previous State: Parallel exploration of capability tokens, IFC, and audit
                logging for multi-agent LLM systems, including a concrete
                benchmark ("LaunderBench," five laundering tiers).
Decision: Do not pursue as the primary direction; retain as a fallback.
Reason: Crowded landscape, with competing systems published in 2026.
Alternatives Considered: Not established beyond the eventual KV-cache pivot.
Evidence: overview.md, "Pivot 2" and "On the horizon."
Consequences: LaunderBench work is preserved but not currently active.
Current Status: Deprioritized / fallback. No evidence it has been revisited.
```

```text
Decision: D3 — Adopt KV-cache compression as the trigger surface
Date/Phase: Pre-2026-09-14 / Phase 3
Previous State: No open, defensible novel direction identified after D1 and D2.
Decision: Adopt KV-cache runtime state (quantization/eviction/merging) as the
                trigger surface for a new backdoor line of research.
Reason: Identified as genuinely open, with no dedicated attack paper as of the
                research sessions; ShadowLogic exists but uses the cache only as
                a passive covert channel for a content-based trigger, not as the
                trigger condition itself.
Alternatives Considered: Continuing with D1/D2 directions (rejected); no other
                specific alternative trigger surfaces are recorded as having
                been compared head-to-head.
Evidence: overview.md, "Current state."
Consequences: All subsequent artifacts (plan, synopsis, PF-SEB) build on this.
Current Status: Active / adopted. This is the project's standing direction.
```

```text
Decision: D4 — Treat quantization, eviction, and merging as mechanistically
                distinct trigger families
Date/Phase: Phase 4 (research_plan.docx, snapshot 10 Sep 2026), §2.2
Previous State: Implicit risk of treating "KV-cache compression" as one
                monolithic trigger category.
Decision: Study quantization, eviction, and merging separately; do not pool
                results across them.
Reason: "Quantization perturbs values while preserving the token set; eviction
                changes which historical states are available; merging changes
                how many states are represented and can create cross-token or
                cross-layer collisions... A paper that only evaluates low-bit
                KV quantization risks collapsing the proposed concept back into
                ordinary quantization-conditioned behavior."
Alternatives Considered: Pooling all compression types under one general
                "runtime transformation" umbrella (implicitly rejected).
Evidence: research_plan.docx §2.2; reaffirmed in Synopsis.docx §3.3.
Consequences: Shapes the taxonomy (KQCB/KECB/KMCB as separate variants) and the
                minimum-viable experimental matrix (separate axes for each).
Current Status: Standing methodological decision, carried into both later
                documents.
```

```text
Decision: D5 — Adopt a fine-tuning-only attacker threat model (no hardware
                access, no serving-infrastructure control)
Date/Phase: Phase 4-5 (present in both plan and synopsis; sharpened in Phase 5
                once CacheTrap was identified)
Previous State: General notion of "an attacker who can train a backdoored
                model."
Decision: Explicitly exclude hardware-level fault injection (Rowhammer-class)
                and any direct control over serving infrastructure from the
                threat model.
Reason: To keep the threat model realistic/weaker-attacker (fine-tuning access
                only) and, once identified, to cleanly differentiate from
                CacheTrap's hardware-fault mechanism -- "a weaker, more
                realistic attacker assumption... yields a comparable trigger
                surface," which is framed as a strength of this project's
                threat model relative to CacheTrap's.
Alternatives Considered: Including hardware-adjacent capability (rejected --
                would collapse the distinction from CacheTrap).
Evidence: Synopsis.docx §11.2 and the "Novelty checkpoint" callout in §7.4.
Consequences: Directly shapes RQ framing and the entire novelty argument.
Current Status: Standing decision; central to the project's differentiation
                strategy.
```

```text
Decision: D6 — Stage the target/payload behavior (synthetic marker first,
                real harmful content only at Stage 3 under institutional review)
Date/Phase: Phase 4 (research_plan.docx §7.3), reaffirmed and expanded Phase 5
                (Synopsis.docx §15, §23)
Previous State: Open question of what the "malicious" target behavior should be
                for initial experiments.
Decision: Three-stage target design: (1) synthetic/deterministic marker, (2)
                controlled instruction-priority behavior (non-harmful but
                semantically meaningful), (3) approved red-team content only
                after Stages 1-2 succeed and with institutional sign-off.
Reason: Keeps the experiment measurable and low-risk; avoids the project
                becoming "a collection of ad hoc jailbreak examples"; separates
                "can the mechanism work at all" from "what is the worst thing
                it could do."
Alternatives Considered: Starting directly with real harmful-content targets
                (implicitly rejected as premature and higher-risk).
Evidence: research_plan.docx §7.3; Synopsis.docx §15, §23.
Consequences: Shapes ethics section, defines what "positive result" can mean at
                each stage.
Current Status: Standing decision.
```

```text
Decision: D7 — Adopt a five-criterion checklist distinguishing a "trained
                backdoor" from an "emergent compression vulnerability"
Date/Phase: Phase 5 (Synopsis.docx §5, §8.1, referenced throughout; the same
                distinction appears informally in the plan without the
                five-item enumeration)
Previous State: Risk of the project's findings being read as "just another
                compression-degrades-safety" paper (already well established by
                Group B literature), which would undercut the novelty claim.
Decision: Require five criteria before describing any result as a "backdoor":
                intentional training, clean-baseline comparison, trigger
                specificity, payload specificity, stealthiness.
Reason: To operationalize the boundary the whole project's novelty rests on.
Alternatives Considered: Not established.
Evidence: Synopsis.docx §5, §8.1, Appendix C checklist; learnings-and-approach.md
                (stated as a key learning/principle).
Consequences: Governs what counts as a "positive result" (§19.2) and structures
                the baselines/controls table (§16).
Current Status: Standing decision; explicitly required to be reapplied "before
                any observed effect is described as a 'backdoor' rather than a
                'vulnerability.'"
```

```text
Decision: D8 — Identify CacheTrap as the closest prior art and commit to
                explicit differentiation
Date/Phase: Phase 5 (Synopsis.docx §7.4, §8, §30)
Previous State: Believed-open novelty space per Phase 3/4 (no dedicated
                cache-as-trigger attack paper known at that time).
Decision: Treat CacheTrap (arXiv:2511.22681) as required prior art to cite and
                differentiate in any submission; re-run this novelty check
                close to submission.
Reason: CacheTrap is "the first published gray-box Trojan that uses the KV
                cache itself as the trigger," making it the nearest published
                work, even though it uses hardware fault injection rather than
                training.
Alternatives Considered: Not established (no record of alternative candidate
                "nearest prior art" papers being weighed against CacheTrap).
Evidence: Synopsis.docx §7.4 "Novelty checkpoint," §8, Appendix C.
Consequences: Directly shapes the threat-model decision (D5) and the paper's
                positioning throughout.
Current Status: Standing decision; project's own checklist requires this
                search be re-run "within one week of submission."
```

```text
Decision: D9 — Narrow the mechanism to active gaming of an honest eviction
                algorithm's decision inputs (PF-SEB)
Date/Phase: Phase 6 (PF-SEB_Synopsis.md, undated)
Previous State: Broad KQCB/KECB/KMCB taxonomy where the trigger is "the cache
                is in state T(C0)," without further specifying whether the
                model plays any active role in producing T(C0).
Decision: Reformulate (at least for this document) the core contribution as:
                the model actively manipulates the importance signals a real
                eviction algorithm (e.g., H2O) uses, so that the algorithm,
                behaving correctly, evicts a hidden "suppressor" state.
Reason (as stated in the document): This is framed as a stronger claim than
                "compression accidentally breaks things" and as addressing a
                gap not covered even by CacheTrap/ShadowLogic-type work, because
                those show the cache as a passive channel or a hardware fault
                point, not an algorithm being actively deceived by the model
                it is managing.
Alternatives Considered: The broader KECB framing without the active-gaming
                element (implicitly the prior state, not explicitly rejected in
                the document -- PF-SEB reads as a refinement/sharpening, not an
                explicit rejection of KECB).
Evidence: PF-SEB_Synopsis.md, Sections 1-3.
Consequences: Introduces a new required concept ("suppressor" state) and a new,
                more specific experimental design (causal pin/delete tests,
                Section 4 of that document) not present in the plan or synopsis.
Current Status: Most recent articulation available, but its formal status
                relative to D3/D4 (i.e., does it replace KECB, generalize it, or
                sit alongside the full taxonomy?) is UNCONFIRMED. See
                10_UNCERTAINTIES_AND_CONTRADICTIONS.md, Issue U1.
```

---

## Decisions NOT yet made (open, per the project's own documents)

- Final entry-point direction among Direction A (inference-time eviction study,
  no training) / Direction B (trained weight-level KQCB) / Direction C
  (merging-based variant) — flagged as an open decision in `overview.md`,
  "On the horizon." PF-SEB (D9) looks like it resolves toward an eviction-based
  direction with training, but this has not been confirmed as *the* answer to
  this specific open decision.
- Final model family/size for experiments (plans specify a 1-8B range across
  Llama/Qwen/Mistral families, not a single final choice).
- Final target venue (multiple are listed as candidates with different fit
  rationales; no commitment recorded).
- Whether LaunderBench (D2's fallback) will be pursued in parallel or
  abandoned entirely.
