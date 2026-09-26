# Formal Research Decision Log

This log records all foundational architectural, conceptual, methodological, and strategic decisions made over the course of the `btp-research` project.

---

## 1. Permanent Decision Record

```text
Decision ID: D1
Title: Reject broad multi-mechanism backdoor framework
Date / Phase: May–August 2026 / Phase 2 (Pivot 1)
Previous State: Proposal for a unified backdoor framework spanning quantization, pruning,
                LoRA-merging, distillation, and compilation under one taxonomy.
Decision: Abandon this broad umbrella framing as the primary thesis contribution.
Rationale: Literature review revealed the domain was saturated by 2024–2026 publications.
           Crucially, a "unified framework" paper published in May 2026 established a
           comprehensive taxonomy, eliminating novelty.
Alternatives Considered: Narrowing to just pruning or distillation (rejected as also crowded).
Source Evidence: calude_research_mem/03_DECISION_LOG.md; overview.md "Pivot 1".
Consequences: Freed project bandwidth to identify an unaddressed runtime trigger surface.
Current Status: PERMANENTLY REJECTED.
```

```text
Decision ID: D2
Title: Deprioritize multi-agent IFC; retain LaunderBench as fallback
Date / Phase: Mid-August 2026 / Phase 3 (Pivot 2)
Previous State: Active exploration of information-flow control, capability tokens, and
                audit logging for multi-agent LLM systems.
Decision: Deprioritize multi-agent IFC as the primary thesis topic; preserve the
          "LaunderBench" benchmark (5 laundering tiers) as an architectural fallback.
Rationale: Mid-2026 saw a surge of competitive systems papers (CapChain, CapAgent, SPA,
           GIF, LaunchSafe). Securing a top-tier novelty gap would require extreme engineering.
Alternatives Considered: Continuing full-time with LaunderBench as the lead project.
Source Evidence: calude_research_mem/03_DECISION_LOG.md; overview.md "Pivot 2".
Consequences: Focus transferred to model/inference systems security; LaunderBench kept on standby.
Current Status: DEPRIORITIZED / ACTIVE FALLBACK.
```

```text
Decision ID: D3
Title: Keep backdoor research standalone; reject merger with client-side determinism
Date / Phase: 17 August 2026 / Phase 4
Previous State: Evaluating whether to combine backdoor attacks with client-side model
                determinism and execution verification.
Decision: Strictly maintain backdoor research as an independent, standalone direction.
Rationale: A hybrid project combining determinism verification with backdoor triggers
           would dilute the threat model, confuse reviewers, and create conflicting narratives.
Alternatives Considered: Merging determinism checks as a detection mechanism for backdoors.
Source Evidence: chatgpt_research_memory/conversations/2026-08-17_research_project_direction_ranking.md.
Consequences: The research problem is purely formulated around runtime inference security.
Current Status: ACTIVE ARCHITECTURAL BOUNDARY.
```

```text
Decision ID: D4
Title: Adopt KV-cache compression as the primary runtime trigger surface
Date / Phase: Late August – Early September 2026 / Phase 5 (Pivot 3)
Previous State: Looking for a concrete, unaddressed runtime state variable to serve as a trigger.
Decision: Formulate the runtime trigger around inference-time Key-Value (KV) cache compression.
Rationale: The KV cache is a massive, mutable intermediate state that is neither weight nor input.
           Modern serving systems routinely compress, evict, or merge KV states to handle
           long contexts and concurrency. No published paper had trained an LLM to trigger on
           legitimate compression policies.
Alternatives Considered: Activation quantization triggers, speculative decoding draft triggers.
Source Evidence: research_plan.docx; calude_research_mem/03_DECISION_LOG.md.
Consequences: Established the core thesis and the KQCB/KECB/KMCB taxonomy.
Current Status: ACTIVE FOUNDATIONAL DIRECTION.
```

```text
Decision ID: D5
Title: Treat quantization, eviction, and merging as mechanistically distinct families
Date / Phase: 10 September 2026 / Phase 5
Previous State: Risk of pooling all compression types into one generic "runtime transformation" bucket.
Decision: Formally separate quantization, eviction, and merging into independent investigation
          axes; do not pool experimental data across them.
Rationale: Quantization perturbs numerical values while keeping the token set intact. Eviction
           removes tokens permanently from the attention window. Merging collapses token cardinality
           and causes cross-token/cross-layer collisions.
Alternatives Considered: A single scalar "compression ratio" metric across all methods (rejected).
Source Evidence: research_plan.docx §2.2; Synopsis.docx §3.3.
Consequences: Structured the minimum viable experimental matrix with separate axes for each.
Current Status: ACTIVE METHODOLOGICAL REQUIREMENT.
```

```text
Decision ID: D6
Title: Adopt a fine-tuning-only threat model (no hardware or serving access)
Date / Phase: 10–14 September 2026 / Phases 5–6
Previous State: General notion of an attacker who tampers with a deployed model.
Decision: Explicitly restrict the attacker's capability to model fine-tuning (open weights/LoRA).
          Strictly exclude hardware fault injection (Rowhammer) and serving-infrastructure control.
Rationale: 1) A fine-tuning-only attacker is significantly more realistic for open-weight ecosystems.
           2) Cleanly and permanently differentiates the project from CacheTrap (which requires
           GPU-adjacent physical/fault access) and ShadowLogic (which requires ONNX graph tampering).
Alternatives Considered: Permitting serving-side proxy tampering (rejected as overly privileged).
Source Evidence: Synopsis.docx §11.2, §7.4; calude_research_mem/03_DECISION_LOG.md.
Consequences: Defines the formal threat model; anchors all defense evaluations.
Current Status: ACTIVE THREAT MODEL BOUNDARY.
```

```text
Decision ID: D7
Title: Stage target payloads: synthetic marker first, real payloads only under approved review
Date / Phase: 10–14 September 2026 / Phases 5–6
Previous State: Considering starting directly with safety jailbreak strings or refusal suppression.
Decision: Adopt a three-stage target payload progression:
          Stage 1: Deterministic synthetic token marker or formatting flip.
          Stage 2: Non-harmful, controlled instruction-priority behavior.
          Stage 3: Approved red-team safety benchmarks (HarmBench/StrongREJECT) only after
                   institutional sign-off and validated Stage 1/2 results.
Rationale: Decouples the scientific question ("Can the runtime trigger function?") from the
           societal risk of generating harmful content. Ensures safe, easily measurable metrics.
Alternatives Considered: Starting immediately with jailbreak refusal suppression (rejected as risky).
Source Evidence: research_plan.docx §7.3; Synopsis.docx §15, §23.
Consequences: Governs experiment registry design; ensures ethical compliance.
Current Status: ACTIVE ETHICAL & OPERATIONAL RULE.
```

```text
Decision ID: D8
Title: Adopt the Five-Criterion Checklist to distinguish backdoors from vulnerabilities
Date / Phase: 14 September 2026 / Phase 6
Previous State: Risk that observed behavioral changes would be dismissed as ordinary clean-model
                compression degradation (already documented in Group B literature).
Decision: Require that any positive result satisfy all five criteria:
          1) Intentional training (not emergent in clean models).
          2) Clean-baseline comparison (exceeds clean model under identical compression).
          3) Trigger specificity (near-miss policies do not trigger it).
          4) Payload specificity (targeted behavior, not general perplexity collapse).
          5) Stealthiness (near-zero false activation under reference full cache C0).
Rationale: Operationalizes the exact boundary on which the paper's novelty rests.
Alternatives Considered: Relying solely on Attack Success Rate (RC-ASR) (rejected as insufficient).
Source Evidence: Synopsis.docx §5, §8.1; learnings-and-approach.md.
Consequences: Dictates the structure of experiment controls and baseline tables.
Current Status: ACTIVE SCIENTIFIC STANDARD.
```

```text
Decision ID: D9
Title: Treat CacheTrap as mandatory adjacent prior art and commit to explicit differentiation
Date / Phase: 14 September 2026 / Phase 6
Previous State: Assuming the KV-cache trigger surface was completely unoccupied in literature.
Decision: Formally cite CacheTrap (arXiv:2511.22681) as nearest prior art and anchor the paper's
          novelty claim on explicit differences in threat model, mechanism, and trigger type.
Rationale: CacheTrap is the first published Trojan using the KV cache as a trigger. However, it
           relies on GPU hardware bit flips on unmodified models, whereas this project trains
           models to trigger on legitimate systems-level compression policies.
Alternatives Considered: Ignoring CacheTrap as "hardware only" (rejected as academically dishonest).
Source Evidence: Synopsis.docx §7.4, §8, §30; DeepSeek review.
Consequences: Mandates a live novelty re-check before paper submission.
Current Status: ACTIVE LITERATURE REQUIREMENT.
```

```text
Decision ID: D10
Title: Preserve negative experimental results as valid, publishable scientific outcomes
Date / Phase: 14 September 2026 / Phase 6
Previous State: Implicit assumption that the project must produce a working attack to succeed.
Decision: Explicitly recognize that a rigorous negative result—proving that intentional training
          cannot produce selective amplification beyond clean-model degradation—is a valuable,
          publishable systems-security contribution.
Rationale: A rigorous negative result demarcates the boundary between emergent compression
           sensitivity and trainable backdoors, providing vital safety guidance to serving teams.
Alternatives Considered: Abandoning the project if Gate G3 fails (rejected).
Source Evidence: Synopsis.docx §19.3; DeepSeek review §1, §9; chatgpt_research_memory D10.
Consequences: Eliminates outcome bias; guarantees publishable artifacts regardless of outcome.
Current Status: ACTIVE METHODOLOGICAL STANCE.
```

```text
Decision ID: D11
Title: Use cache-aware differential auditing as the primary defense concept
Date / Phase: 14 September 2026 / Phase 6
Previous State: Open question of how defenders could detect runtime-conditioned backdoors.
Decision: Propose and evaluate "Cache-Aware Differential Auditing" as the primary defense:
          query the model with identical prompts across a small family of reference and
          compressed cache policies, monitoring for anomalous behavioral divergence.
Rationale: Full-cache audits cannot see the backdoor; differential execution across runtime
           states exposes the hidden conditional policy at minimal probe cost.
Alternatives Considered: Weight-level inspection (ineffective against subtle LoRA backdoors).
Source Evidence: Synopsis.docx §21; DeepSeek review §5.7; chatgpt_research_memory D11.
Consequences: Shapes Phase 5 defense design (Experiment E5).
Current Status: ACTIVE DEFENSE PROPOSAL.
```

```text
Decision ID: D12
Title: Adopt DeepSeek's scoped recommendation: narrow initial experimental matrix
Date / Phase: 18 September 2026 / Phase 7
Previous State: Research plan proposed exploring quantization, eviction, merging, thresholds,
                multiple model families, and multi-tenant serving simultaneously.
Decision: Severely narrow the Phase 0–2 experimental scope:
          - Model: 1–2 instruction-tuned models in the 1B–3B parameter range (e.g., Qwen2.5-1.5B).
          - Compression Family: Single family first (Quantization via STE, or H2O Eviction).
          - Target: Single safe synthetic marker.
          - Controls: One near-miss policy and clean baseline.
Rationale: A B.Tech final-year research project has bounded compute and timeline. Attempting the
           full matrix immediately guarantees failure or surface-level execution.
Alternatives Considered: Sticking to the broad multi-model 8-family matrix (rejected as unviable).
Source Evidence: DeepSeek review §1, §4.7, §5.4; calude_research_mem overview.md.
Consequences: Governs the concrete 30-day and 90-day implementation plans.
Current Status: ACTIVE SCOPE CONSTRAINT.
```

```text
Decision ID: D13
Title: Adopt Policy-Fingerprinted Self-Eviction Backdoors (PF-SEB) as the primary eviction mechanism
Date / Phase: Late September 2026 / Phase 8
Previous State: Eviction backdoors were viewed as passive reactions to missing tokens (KECB).
Decision: Refine the eviction direction into PF-SEB: the model actively manipulates an honest,
          unmodified eviction algorithm (e.g., H2O) by dampening attention scores on an internal
          "suppressor" token state, causing the algorithm to discard the suppressor on cue.
Rationale: Dramatically elevates the conceptual novelty: instead of "the model breaks under compression,"
          the claim is "the model games its own memory manager." Supported by a 7-condition causal table.
Alternatives Considered: Staying with purely passive KECB (retained as control baseline).
Source Evidence: PF-SEB_Synopsis.md §1–4.
Consequences: Establishes Experiment E6 as the flagship eviction evaluation.
Current Status: ACTIVE CONCEPTUAL REFINEMENT.
```

```text
Decision ID: D14
Title: Formally establish the Two-Track Research Program Architecture
Date / Phase: Late September 2026 / Phase 9
Previous State: Ambiguity over whether PF-SEB was intended to replace, compete with, or sit
                within the broader KQCB/KECB/KMCB taxonomy.
Decision: Unify the project into a complementary Two-Track Research Architecture:
          - Track 1 (The General Systems Framework): "Runtime-Conditioned Backdoors in LLMs"
            (documented in Runtime_Conditioned_Backdoors_KV_Cache_Synopsis.docx and research_plan.docx),
            providing the comprehensive taxonomy (quantization, eviction, merging) and systems relevance.
          - Track 2 (The Flagship Mechanistic Specialization): "Policy-Fingerprinted Self-Eviction Backdoors"
            (documented in PF-SEB_Synopsis.docx and PF-SEB_Research_Plan.docx), narrowing the eviction branch
            into a razor-sharp, causally proven attack on the memory manager's trust interface.
Rationale: Confirmed directly by PF-SEB_Research_Plan.docx: "Scope: a focused specialisation of the
           eviction-conditioned (KECB) branch of the broader runtime-conditioned-backdoor research direction;
           deliberately narrower in scope than the full KQCB/KECB/KMCB programme." This allows the project
           to maintain a broad systems narrative while submitting a tight, bulletproof empirical paper.
Alternatives Considered: Discarding the general framework and pursuing PF-SEB exclusively (rejected:
                         loses the engineering advantages of KQCB de-risking and the broad systems narrative).
Source Evidence: PF-SEB_Research_Plan.docx §1; PF-SEB_Synopsis.docx §3.4; Runtime_Conditioned_Backdoors_KV_Cache_Synopsis.docx.
Consequences: Fully resolves Uncertainty U1; aligns research plan and synopsis artifacts.
Current Status: ACTIVE STRATEGIC ARCHITECTURE.
```

---

## 2. Open Decisions Awaiting Resolution

| Decision ID | Description | Options Under Consideration | Recommended Action |
|---|---|---|---|
| **OD-1** | **Entry-Point Implementation Choice** | Option A: KQCB (Quantization-first, highly tractable, STE-based).<br>Option B: PF-SEB (Eviction-first, highest novelty, H2O attention gaming). | **Implement Option A (KQCB) first in Phase 0/2 for fast de-risking**, while building the harness to support H2O eviction logging for PF-SEB immediately following. |
| **OD-2** | **Initial Target Model Selection** | Option A: `Qwen/Qwen2.5-1.5B-Instruct`<br>Option B: `meta-llama/Llama-3.2-1B-Instruct` / `3B-Instruct`<br>Option C: `mistralai/Mistral-7B-Instruct` | **Select `Qwen2.5-1.5B-Instruct` as primary** (compact, modern GQA attention architecture, low VRAM footprint) and `Llama-3.2-1B-Instruct` as secondary cross-family control. |
| **OD-3** | **Target Publication Venue & Cycle** | Option A: USENIX Security 2027 (Cycle 2: Jan 2027)<br>Option B: IEEE S&P 2027 (Cycle 2: Nov 2026 — tight)<br>Option C: MLSys 2027 (Oct 2026 — imminent/passed)<br>Option D: TMLR (Rolling journal submission) | **Target USENIX Security 2027 (Cycle 2, Jan 2027)** as primary conference, with **TMLR** as high-rigor fallback. |
| **OD-4** | **LaunderBench Fallback Status** | Option A: Completely archive and close.<br>Option B: Maintain as secondary fallback if G3 fails. | **Maintain as standby fallback (Option B)** until Gate G3 is formally passed. |
