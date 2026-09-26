# Memory System Changelog

All notable changes, formal milestone achievements, decision updates, and experiment executions in the `btp-research` project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [1.1.0] — 2026-09-26

### Enriched: Integration of Primary DOCX Proposal Artifacts
- **Action:** Ingested and reconciled the 4 newly added primary Word documents in `researchMemory/`:
  - `runtime_conditioned_backdoors_kv_cache_research_plan.docx` (Track 1 Roadmap)
  - `Runtime_Conditioned_Backdoors_KV_Cache_Synopsis.docx` (Track 1 Synopsis)
  - `PF-SEB_Research_Plan.docx` (Track 2 Roadmap)
  - `PF-SEB_Synopsis.docx` (Track 2 Synopsis)
- **Key Enhancements Across `agentMemory/`:**
  - **`README.md` & `DECISION_LOG.md` (Decision D14):** Formalized the Two-Track Research Program Architecture (Track 1 Systems Umbrella vs. Track 2 PF-SEB Empirical Specialization), fully resolving Uncertainty U1.
  - **`CURRENT_STATE.md` & `TECHNICAL_KNOWLEDGE.md`:** Integrated formal mathematical estimands for Causal Rescue ($\Delta_{\text{rescue}}$), Causal Induction ($\Delta_{\text{induction}}$), and Random-Deletion Control ($\Delta_{\text{random}}$), plus the 6-step dual-forward training algorithm.
  - **`RESEARCH_DIRECTIONS.md`:** Expanded the PF-SEB taxonomy into its full 5-tier classification (`PF-SEB-core`, `transfer`, `budget`, `composed`, `context`).
  - **`RESEARCH_QUESTIONS.md`:** Added Section 6 integrating the 12 Supervisor Discussion & Oral Defense Questions from Appendix A.
  - **`FAILURES_AND_NEGATIVE_RESULTS.md`:** Added Section 6 pre-empting the 5 critical reviewer objections from `PF-SEB_Synopsis.docx` Appendix B, resolving Uncertainty U5.
  - **`EXPERIMENT_REGISTRY.md`:** Added Section 3 detailing the Statistical Proof Plan and the 6-figure reviewer visualization set.
  - **`NEXT_STEPS.md`:** Added Section 7 detailing the 10-point Formal Research Approval Checklist (Appendix C).
  - **`UNCERTAINTIES_AND_CONTRADICTIONS.md`:** Marked Issues U1 (PF-SEB relation) and U5 (Appendix B) as fully resolved via primary document text.

---

## [1.0.0] — 2026-09-26

### Initialized: Complete Research Memory Initialization
- **Action:** Executed a full, cross-assistant research memory initialization, synthesizing all historical project data into the canonical `researchMemory/agentMemory/` system.
- **Sources Ingested & Analyzed:**
  - `researchMemory/chatgpt_research_memory/`: 12 raw conversation summaries (May 2026 – September 2026) documenting early exploration, topic narrowing, direction ranking, and synopsis construction.
  - `researchMemory/calude_research_mem/`: Master memory files (00–10) capturing high-level project evolution, principles, and artifact indices.
  - `researchMemory/deepseek_btp_mem/`: DeepSeek technical evaluation (18 September 2026), pitfall analysis, formal estimand ($\Delta_{\text{int}}$), and scoping recommendations.
  - Historical Core Artifacts: `runtime_conditioned_backdoors_kv_cache_research_plan.docx` (10 Sep 2026), `Runtime_Conditioned_Backdoors_KV_Cache_Synopsis.docx` (14 Sep 2026), and `PF-SEB_Synopsis.md` (late Sep 2026).
- **Core Memory Files Created in `agentMemory/`:**
  - `README.md`: Memory system architecture, purpose, source lineage, and navigation.
  - `CURRENT_STATE.md`: Real-time research status, threat model, gates G1–G8, and non-claims.
  - `RESEARCH_TIMELINE.md`: Reconstructed narrative from May 2026 to present, detailing all pivots.
  - `RESEARCH_DIRECTIONS.md`: Exhaustive inventory of active, proposed, fallback, separated, and rejected directions.
  - `DECISION_LOG.md`: Formal immutable log of decisions D1–D13 and open decisions OD-1 to OD-4.
  - `RESEARCH_QUESTIONS.md`: Answered, partially answered, open core RQs (RQ1–RQ5), and PF-SEB RQs.
  - `LITERATURE_MAP.md`: 4-group literature taxonomy, prior art differentiation matrix, and venue deadlines.
  - `TECHNICAL_KNOWLEDGE.md`: KV systems theory, mathematical formalisms, PF-SEB mechanics, and STE proxies.
  - `EXPERIMENT_REGISTRY.md`: Standardized protocols for experiments E0–E6 (all currently proposed).
  - `IMPLEMENTATION_STATE.md`: Codebase audit (0 code files present), stack specs, and Phase 0 harness architecture.
  - `FINDINGS.md`: Partitioned findings across literature facts, conceptual insights, and empirical results.
  - `FAILURES_AND_NEGATIVE_RESULTS.md`: Catalog of dead ends, anticipated failure modes, and negative result framing.
  - `OPEN_PROBLEMS.md`: Categorized scientific, engineering, and methodological hurdles.
  - `UNCERTAINTIES_AND_CONTRADICTIONS.md`: Documented ambiguity records U1–U8 with resolutions.
  - `NEXT_STEPS.md`: 14-day, 30-day, 60-day, and 90-day action roadmaps and supervisor alignment agenda.
  - `CHANGELOG.md`: This initial log entry.
- **Baseline Epistemic Grounding Established:**
  - **Zero experiments completed** to date.
  - **Zero code files** implemented in repository.
  - Project is formally at **Day 0 of implementation (Phase 0 / Gate G1 pending)**.
  - Primary target venue set to **USENIX Security 2027 (Cycle 2: 26 January 2027)** with **TMLR** as journal fallback.
