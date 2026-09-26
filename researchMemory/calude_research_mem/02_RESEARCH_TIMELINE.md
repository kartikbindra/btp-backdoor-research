# 02 — RESEARCH TIMELINE

No raw conversation transcripts were available for this reconstruction. Phases below are
inferred from `overview.md` (project memory, last updated 2026-09-14) and from internal
evidence in the three artifacts (their own evidence-snapshot dates: 10 Sep 2026 for the
research plan, 14 Sep 2026 for the synopsis, undated for PF-SEB). Where reasoning is not
explicit in the source material, this is stated rather than invented.

```mermaid
flowchart TD
    A[Phase 1: Broad multi-mechanism backdoor proposal\n(quantization/pruning/LoRA-merge/distillation/compilation)]
    B[Phase 2: Saturation found -- 'unified framework' paper preempts core claim]
    C[Phase 3: Parallel exploration -- multi-agent IFC / capability tokens / audit logging]
    D[Phase 4: IFC landscape also found crowded -- kept as fallback 'LaunderBench']
    E[Phase 5: Pivot to KV-cache compression as trigger surface -- identified as open]
    F[Phase 6: Formal research plan drafted\n(research_plan.docx, snapshot 10 Sep 2026)]
    G[Phase 7: Expanded synopsis drafted\n(Synopsis.docx, snapshot 14 Sep 2026)\nCacheTrap identified as closest prior art]
    H[Phase 8: Refinement to PF-SEB\n(PF-SEB_Synopsis.md, undated)\nactive-gaming mechanism, not just passive trigger]

    A --> B --> C --> D --> E --> F --> G --> H
```

---

## Phase 1 — Broad multi-mechanism backdoor proposal

- **Period:** Before 2026-09-14 (predates the earliest artifact provided). Exact dates unknown.
- **Objective:** Propose deployment-/weight-transformation-conditioned backdoors across several mechanisms at once: quantization, pruning, LoRA-merge, distillation, compilation.
- **Outcome:** Found to be largely saturated by 2024–2026 prior art, including a "unified framework" paper (dated May 2026 per `overview.md`) that preempted the core novelty claim.
- **Evidence:** `overview.md`, "Background & evolution," Pivot 1. No further detail (which papers specifically, what search was run) is available in the provided material.
- **Status:** Rejected / superseded. See `06_IDEAS_AND_DIRECTIONS.md`.

## Phase 2 — Parallel exploration: multi-agent IFC / agent security

- **Period:** Unknown; treated in memory as a parallel thread, not necessarily sequential to Phase 1.
- **Objective:** Explore LLM agent security — capability tokens, information-flow control (IFC), audit logging for multi-agent systems.
- **Outcome:** Crowded landscape, with competing systems published in 2026. A specific benchmark design, "LaunderBench" (instruction laundering / "attacking the labeler," five laundering tiers), was produced during this exploration and is retained as a possible parallel or fallback direction.
- **Evidence:** `overview.md`, "Background & evolution," Pivot 2, and "On the horizon."
- **Status:** Deprioritized as primary direction; retained as fallback. Prior-art ecosystem named: ShadowLogic, CapChain, CapAgent, SPA, GIF, LaunchSafe, plus the May 2026 unified framework paper (this last one appears to be the same paper that saturated Phase 1, suggesting the two pivots may be connected or discovered together — **not confirmed**, see `10_UNCERTAINTIES_AND_CONTRADICTIONS.md`).

## Phase 3 — Pivot to KV-cache compression as the trigger surface

- **Period:** Before 2026-09-14.
- **Objective:** Identify a genuinely open direction after Phases 1–2 both hit saturation/crowding.
- **Outcome:** KV-cache compression (quantization/eviction/merging as an inference-time runtime state) adopted as the new trigger surface, on the basis that no dedicated attack paper combining "legitimate compression policy" + "trained/intentional trigger" was found as of the research sessions.
- **Key literature reference point:** ShadowLogic (HiddenLayer, CAMLIS 2025) identified as a citable precedent for "KV cache as a covert channel," but for a *content-based* trigger detector, not a compression-state trigger — explicitly distinguished. A Q&A exchange in the ShadowLogic talk is recorded as having "confirmed the quantization robustness question remains open" [Source: overview.md — this specific claim about the Q&A content could not be independently verified from the provided material and should be treated as reported, not re-derived].
- **Evidence:** `overview.md`, "Current state."
- **Status:** Adopted; this is the project's live direction as of all three artifacts.

## Phase 4 — Formalization: the research plan document

- **Period:** Evidence snapshot 10 Sep 2026 (per the document's own header).
- **Objective:** Turn the KV-cache-trigger idea into a full research roadmap: executive summary, KV-cache background, threat model, four-track literature review (efficiency / security consequences / backdoor literature / gap), taxonomy of six attack variants (KQCB/KECB/KMCB/context-threshold/load-conditioned/composed), seven-phase build order (0–5, "How to Start and Build the Project"), metrics, statistical plan, minimum viable experimental matrix, use cases, failure-mode table, timeline (Sep 2026 – Jan 2027 and beyond), venue targets and deadlines, final paper structure, go/no-go checklist, and a source-notes/claims-discipline section.
- **Key methodological decision made in this phase:** treat quantization, eviction, and merging as mechanistically distinct trigger families rather than pooling them (§2.2 of the plan).
- **Evidence:** `runtime_conditioned_backdoors_kv_cache_research_plan.docx`.
- **Status:** Complete as a planning artifact; superseded in detail (but not in substance) by the Phase 5 synopsis.

## Phase 5 — Expansion into a full B.Tech synopsis

- **Period:** Evidence snapshot 14 Sep 2026.
- **Objective:** Produce a submission-ready synopsis with a much fuller literature review (Groups A–D, ~20 sources plus venue references), a formalized problem statement (f(x,C0) vs f(x,T(C0))), an explicit five-criterion checklist distinguishing a "trained backdoor" from an "emergent compression vulnerability," expected contributions, defense/mitigation table, scope/limitations, ethics section, tools/stack, compute requirements, risk register, go/no-go gates, appendices (supervisor discussion questions; a stub for reviewer criticisms; an approval checklist).
- **Key new finding in this phase:** identification of **CacheTrap** (arXiv:2511.22681, 2026 preprint) as the closest published prior art — a hardware-fault-injection (Rowhammer/GPUHammer-style) Trojan using the KV cache as trigger on an *unmodified* model. This becomes the central novelty anchor: the project's contribution is explicitly positioned as differing on attacker capability (fine-tuning vs. hardware access) and on mechanism (trained policy-conditioned trigger vs. single fixed-bit fault).
- **Also newly cited in this phase (not in the plan):** ShadowLogic given a full citation and Group C discussion; "Governing the KV Cache" (timing side-channels, arXiv:2608.09225); "Inference-Time Backdoors via Chat Templates" (CCS 2026); a background/tooling reference to "Next-Latent Prediction Transformers" (Microsoft Research, arXiv:2511.05963) flagged explicitly as "not yet incorporated into the core methodology and requires its own novelty check before use."
- **Evidence:** `Runtime_Conditioned_Backdoors_KV_Cache_Synopsis.docx`.
- **Status:** Most complete formal document; represents the project's fullest self-description of the broad (taxonomy-level) direction.

## Phase 6 — Refinement to Policy-Fingerprinted Self-Eviction Backdoors (PF-SEB)

- **Period:** Undated; provided as the third artifact in this session, written in a distinctly different, more informal/"intuition-first" register than the other two, suggesting it postdates them as a sharpened pitch/explainer version rather than a full synopsis rewrite.
- **Objective (as stated in the document):** Answer "can a malicious LLM game its own memory manager?" — narrow the general runtime-conditioned-backdoor claim into a specific, falsifiable mechanism: the model doesn't merely react to a compressed cache state, it *actively manipulates* the importance signals a real, honest eviction algorithm (e.g., H2O) uses, so that algorithm evicts a hidden "suppressor" state exactly when needed to unlock hidden behavior.
- **New conceptual elements introduced here, not present in the plan or synopsis:**
  - The "suppressor" concept — an internal state that, while present, keeps bad behavior switched off.
  - Explicit framing that the eviction algorithm "never breaks any rule" — the attack works entirely through gaming inputs to a correct algorithm, not through a flaw in the algorithm.
  - A concrete seven-row causal-proof experimental table (clean model, unlimited memory, real policy, different policy, manual-pin, manual-delete, remove-fine-tuning) designed specifically to rule out "just a fragile model" as an alternative explanation.
  - An explicit statement that the mechanism "plausibly generalizes beyond eviction to other places serving systems trust model-generated signals (batching, speculative decoding, cache sharing)" — framed as a plausible extension, not a claim being made now.
  - A sharper three-way novelty contrast: (a) compression-accident papers (bug, not attack), (b) cache-as-passive-covert-channel papers like ShadowLogic (content trigger, cache just carries it), (c) hardware-fault papers like CacheTrap (physical access, not training) — versus this work's claim of an *actively gaming, trained* mechanism.
- **Evidence:** `PF-SEB_Synopsis.md`.
- **Status:** Most recent articulation available. **Relationship to the broader KQCB/KECB/KMCB taxonomy from Phases 4–5 is not explicitly stated** — PF-SEB reads as a specific, sharpened instantiation of the KECB (eviction-conditioned) family, but no document explicitly says "PF-SEB replaces/narrows KECB" or "PF-SEB is the chosen entry point." Flagged as an open item.

## Reasoning gaps explicitly acknowledged

Per the task's instructions against inventing motivations: the available material does **not** establish:
- The exact date or trigger for the Phase 1→2, 2→3, or 5→6 transitions.
- Whether a specific search or a specific paper (beyond the named "unified framework" paper and CacheTrap) caused any pivot.
- Whether PF-SEB was arrived at through a dedicated ideation session, a response to a specific piece of new literature, or independent refinement.

These should be treated as open items for the next agent to ask about directly rather than infer further.
