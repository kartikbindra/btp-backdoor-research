# 06 — IDEAS AND DIRECTIONS

Status labels used: **active**, **proposed**, **explored**, **deprioritized**, **rejected**, **superseded**.
Rankings/evaluations below are the project's own (attributed), not this compilation's judgment.

---

### Idea: Broad multi-mechanism weight/deployment backdoor framework
- **Description:** A unified backdoor framework spanning quantization-, pruning-, LoRA-merge-, distillation-, and compilation-triggered behavior changes.
- **Motivation:** Generalize deployment-conditioned backdoors across all common model-transformation techniques.
- **Mechanism:** Not detailed in available material beyond the category list.
- **Expected novelty/contribution:** Not detailed.
- **Feasibility:** Not detailed.
- **Status:** **Rejected.**
- **Reason for status:** Found largely saturated by 2024-2026 prior art, including a May 2026 "unified framework" paper that preempted the core novelty claim. [overview.md]
- **Related ideas:** Precursor to the eventual KV-cache pivot; also related to the "quantization-conditioned backdoor" concept referenced as prior (non-KV-cache) research in later documents ("weight-QCB", "the same logic that made ordinary numerical sensitivity in quantized weights... exploitable as a deployment-conditioned backdoor trigger in prior (non-KV-cache) work" — Synopsis.docx §3.4). This suggests weight-level quantization-conditioned backdoors were already an established prior-art category the project was drawing an analogy from, distinct from the rejected "unified framework" proposal.

---

### Idea: Multi-agent IFC / capability-token / audit-logging security ("LaunderBench")
- **Description:** Security research on information-flow control, capability tokens, and audit logging in multi-agent LLM systems; includes a concrete benchmark design, "LaunderBench," with five instruction-laundering tiers, framed as "attacking the labeler."
- **Motivation:** Explore agent-security angle as an alternative to model-level backdoors.
- **Mechanism:** Not detailed beyond the benchmark name and "laundering tiers" framing.
- **Expected novelty/contribution:** Not detailed.
- **Feasibility:** Not detailed.
- **Status:** **Deprioritized** (kept as fallback/parallel direction).
- **Reason for status:** Crowded 2026 landscape — competing systems named: ShadowLogic, CapChain, CapAgent, SPA, GIF, LaunchSafe, plus the May 2026 unified framework paper. [overview.md]
- **Related ideas:** None directly; separate research line from the KV-cache work.

---

### Idea: KV-cache compression as a backdoor trigger surface (general)
- **Description:** The KV cache's runtime operational state (quantization bit-width, eviction ratio, merge policy) serves as the backdoor trigger, rather than prompt content.
- **Motivation:** Genuinely open — no dedicated attack paper combining legitimate compression + trained trigger found as of the research sessions; distinguishes from ShadowLogic (passive covert channel for a content trigger).
- **Mechanism:** f(x, C0) -> benign; f(x, T(C0)) -> attacker-targeted behavior, for a runtime transformation T (quantization/eviction/merging).
- **Expected novelty/contribution:** A new attack surface at the model/systems-security boundary; motivates cache-aware auditing.
- **Feasibility:** Staged plan exists (six phases); no implementation confirmed.
- **Status:** **Active** (foundational framing for everything that follows).
- **Related ideas:** Parent idea to KQCB, KECB, KMCB, context-threshold KCB, load-conditioned KCB, composed KCB, and PF-SEB.

---

### Sub-idea: KQCB — KV Quantization-Conditioned Backdoor
- **Description:** Trigger = cache precision falls below a threshold (e.g., INT8 -> 2-bit).
- **Mechanism hypothesis:** Small numerical perturbations flip a safety-relevant representation/decision boundary once quantization noise crosses a critical magnitude.
- **Expected novelty/value (project's own assessment):** "Closest analogue to weight-quantization-conditioned backdoors from prior (non-KV-cache) research; the most tractable first prototype."
- **Difficulty (project's own assessment):** Low-Medium.
- **Status:** **Proposed** — recommended as the first prototype in the plan/synopsis's phased build order, but Phase 0/1 has not been confirmed started.
- **Related ideas:** Sibling of KECB, KMCB.

---

### Sub-idea: KECB — KV Eviction-Conditioned Backdoor
- **Description:** Trigger = a specific token-retention/eviction policy removes designated historical states.
- **Mechanism hypothesis:** Critical historical states (e.g., an instruction or context marker) are removed, leaving the model in a different effective decision regime.
- **Expected novelty/value (project's own assessment):** "More novel than KQCB; directly exploits cache selection rather than numerical precision."
- **Difficulty (project's own assessment):** Medium.
- **Status:** **Proposed / partially superseded-in-emphasis by PF-SEB** — PF-SEB appears to be a sharpened, specific instantiation of this family (active gaming rather than passive removal), though this relationship is not explicitly stated in any document.
- **Related ideas:** Parent of PF-SEB (inferred).

---

### Sub-idea: KMCB — KV Merging-Conditioned Backdoor
- **Description:** Trigger = token/layer cache merging collapses distinct states.
- **Mechanism hypothesis:** Distinct security-relevant representations become less separable after merging or cross-layer collision.
- **Expected novelty/value (project's own assessment):** "Strongest connection to the functional-head-collapse mechanism reported in Group B (When Efficiency Meets Safety)."
- **Difficulty (project's own assessment):** Medium-High.
- **Status:** **Proposed**, not yet explored experimentally.
- **Related ideas:** Sibling of KQCB, KECB.

---

### Sub-idea: Context-threshold KCB
- **Description:** Trigger = context length or retained-token ratio crosses a threshold L*.
- **Expected value:** Directly relevant to long-context agents; ties RQ3 to a realistic deployment scenario.
- **Difficulty:** Medium.
- **Status:** **Proposed**, Phase 3 item (trigger generalization), not yet explored.

---

### Sub-idea: Load-conditioned KCB
- **Description:** Trigger = runtime memory pressure / adaptive cache budget under high concurrency.
- **Expected value (project's own assessment):** "Strongest systems-security narrative; connects to multi-tenant serving."
- **Difficulty:** High.
- **Status:** **Proposed**, explicitly framed as a stretch/later-phase item.

---

### Sub-idea: Composed KCB
- **Description:** Trigger requires two runtime conditions simultaneously (e.g., aggressive eviction AND low precision) — an "AND-gate" between conditions.
- **Expected value (project's own assessment):** "Harder to detect via single-axis differential audits; a stronger stealth result if achieved."
- **Difficulty:** High.
- **Status:** **Proposed**, latest-phase item.

---

### Idea: Policy-Fingerprinted Self-Eviction Backdoors (PF-SEB)
- **Description:** The model actively feeds misleading importance signals to a real, unmodified, attention-based eviction algorithm (e.g., H2O) so that the algorithm — functioning correctly — evicts a hidden internal "suppressor" state exactly when it matters, revealing hidden behavior.
- **Motivation:** Sharper novelty claim than passive cache-state triggering: "the model actively manipulating the very inputs an honest, unmodified memory-management algorithm uses to make its decisions."
- **Mechanism:** Training-time: model learns to make the suppressor state look unimportant to the eviction policy's attention-based scoring early on, while still relying on it later. Office analogy given in the source document (desk drawer = KV cache; cleaner = eviction policy; important note = suppressor; dishonest employee's trick = learned deceptive signaling).
- **Expected novelty (project's own claim, explicitly framed as hypothesis):** Existing work either shows accidental compression-safety degradation (bug), uses the cache passively as a covert channel for an input-triggered attack (ShadowLogic-type), or requires hardware fault injection (CacheTrap) — none show a model actively manipulating an honest algorithm's own decision inputs.
- **Feasibility:** Requires full fine-tuning access; explicitly stated as not working against a closed, API-only model.
- **Falsification design (concrete, in the source document):** seven-condition table — clean model + real policy; trained model + unlimited memory; trained model + real policy at trigger budget; trained model + different policy same budget; trigger case with suppressor manually pinned; unlimited-memory case with suppressor manually deleted; malicious fine-tuning removed. Only the "real policy at trigger budget" and "manual delete" conditions should show the hidden behavior.
- **Status:** **Active / most recently articulated formulation.** Formal relationship to KECB and to the broader taxonomy is unconfirmed (see `10_UNCERTAINTIES_AND_CONTRADICTIONS.md`).
- **Related ideas:** Refinement of KECB; possible future generalization (proposed, not pursued) to batching, speculative decoding, or cache-sharing as alternative "trusted model-generated signals" an attacker could similarly game.

---

### Idea (mentioned, not adopted): Extending the "gaming a trusted signal" mechanism beyond eviction
- **Description:** The same logic — a model manipulating a signal that serving infrastructure trusts — could plausibly apply to batching decisions, speculative decoding, or cache sharing across requests.
- **Status:** **Proposed** only, explicitly flagged in PF-SEB_Synopsis.md §6 as a plausible generalization, not a claim being made now, and not designed or scoped further in any document.

---

### Idea (background only, not integrated): Next-latent prediction / world-model probing as a mechanistic-analysis tool
- **Description:** Using a next-latent-prediction training objective (from Teoh et al., Microsoft Research) as an auxiliary tool for probing belief-state-like latents under compression.
- **Status:** **Proposed / not integrated.** Explicitly flagged in Synopsis.docx §30 [20] as "not yet incorporated into the core methodology and requires its own novelty check before use." Do not assume this is part of the current methodology.
