# Agent Research Memory — B.Tech Research Project (`btp-research`)

Welcome to the canonical, persistent research memory system for the `btp-research` project.

---

## 1. Purpose of this Memory System

This directory (`researchMemory/agentMemory/`) serves as the **authoritative, persistent research brain and operational context** for this ongoing research project. It was established through a complete research memory initialization on **26 September 2026**, synthesizing all historical memory packages, draft synopses, research roadmaps, conversation transcripts, and multi-model reviews across Claude, ChatGPT, and DeepSeek.

### Core Objectives:
1. **Maintain Complete Historical Continuity:** Preserve the full trajectory of problem statements, early ideas, rejected directions, pivots, technical breakthroughs, and critical decisions without historical amnesia.
2. **Enforce Strict Epistemic Discipline:** Maintain a rigorous, non-negotiable boundary between *Established Literature Facts*, *Conceptual Hypotheses*, *Experimental Designs*, and *Actual Empirical Results*.
3. **Act as an Operational Handoff:** Enable any incoming AI research agent or human collaborator to instantly understand the state of the project, open questions, codebase status, and next immediate tasks without repeating solved or rejected work.
4. **Log Ongoing Progress:** Serve as the living research journal and decision record for all future work.

---

## 2. Source Archives and Lineage

This memory system is constructed from and maintains continuity with the following primary sources in `researchMemory/`:

| Primary Document / Archive | Type / Size | Scope & Purpose |
|---|---|---|
| [`runtime_conditioned_backdoors_kv_cache_research_plan.docx`](file:///c:/Users/Kartik/OneDrive/Desktop/Projects/btp-research/researchMemory/runtime_conditioned_backdoors_kv_cache_research_plan.docx) | Word Doc (80 KB) | **Track 1 Roadmap:** Broad research roadmap, threat model, 6-variant taxonomy (KQCB, KECB, KMCB, threshold, load, composed), 6-phase build order (Phases 0–5). |
| [`Runtime_Conditioned_Backdoors_KV_Cache_Synopsis.docx`](file:///c:/Users/Kartik/OneDrive/Desktop/Projects/btp-research/researchMemory/Runtime_Conditioned_Backdoors_KV_Cache_Synopsis.docx) | Word Doc (54 KB) | **Track 1 Synopsis:** 31-page comprehensive B.Tech research synopsis for supervisor review, including 4-group literature taxonomy and CacheTrap differentiation. |
| [`PF-SEB_Research_Plan.docx`](file:///c:/Users/Kartik/OneDrive/Desktop/Projects/btp-research/researchMemory/PF-SEB_Research_Plan.docx) | Word Doc (33 KB) | **Track 2 Roadmap:** Dedicated roadmap for Policy-Fingerprinted Self-Eviction Backdoors; focuses on attention-based eviction gaming, differentiable proxy training, and causal proof. |
| [`PF-SEB_Synopsis.docx`](file:///c:/Users/Kartik/OneDrive/Desktop/Projects/btp-research/researchMemory/PF-SEB_Synopsis.docx) | Word Doc (51 KB) | **Track 2 Synopsis:** 31-page specialized B.Tech synopsis on PF-SEB; contains the full suppressor mechanics, 5-variant PF-SEB taxonomy, filled Appendix B (Reviewer Criticisms), and Appendix C. |
| `researchMemory/chatgpt_research_memory/` | Directory | Full 12-conversation transcript archive from early B.Tech ideation to KV-cache research plan and synopsis generation. |
| `researchMemory/calude_research_mem/` | Directory | Master memory files (00–10) capturing high-level project evolution, principles, and artifact indices. |
| `researchMemory/deepseek_btp_mem/` | Directory | DeepSeek technical evaluation (18 Sep 2026), pitfall analysis, formal estimand ($\Delta_{\text{int}}$), and 90-day execution roadmap. |

### The Two-Track Research Architecture
The newly added documents establish that the project operates as a complementary two-track program:
- **Track 1 (The General Framework):** *Runtime-Conditioned Backdoors in Large Language Models: KV-Cache Compression as an Inference-Time Trigger*. Explores the full taxonomy (quantization, eviction, merging) and systems trade-offs across inference engines.
- **Track 2 (The Flagship Mechanism):** *Policy-Fingerprinted Self-Eviction Backdoors in Large Language Models (PF-SEB)*. A deliberate, high-novelty specialization of the eviction (KECB) branch where the model actively deceives an honest attention-based eviction algorithm (H2O) into discarding an internal "suppressor" state.

> [!IMPORTANT]
> The historical directories (`calude_research_mem`, `chatgpt_research_memory`, `deepseek_btp_mem`) are preserved immutably as evidentiary archives. **All future updates must be made exclusively inside `researchMemory/agentMemory/`.**

---

## 3. Directory Structure and File Guide

The memory system is structured modularly to avoid monolithic context bloat and ensure fast, targeted retrieval:

```text
researchMemory/
└── agentMemory/
    ├── README.md                           <-- [You are here] Overview, navigation, and memory architecture
    ├── CURRENT_STATE.md                    <-- Real-time snapshot of research progress, gates, and blockers
    ├── RESEARCH_TIMELINE.md                <-- Chronological narrative from May 2026 to present, detailing pivots
    ├── RESEARCH_DIRECTIONS.md              <-- Inventory of active, proposed, separated, and rejected directions
    ├── DECISION_LOG.md                     <-- Immutable log of formalized architectural & research decisions
    ├── RESEARCH_QUESTIONS.md               <-- Answered, partially answered, open, and superseded RQs
    ├── LITERATURE_MAP.md                   <-- 4-Group literature taxonomy, prior art differentiation, and venue deadlines
    ├── TECHNICAL_KNOWLEDGE.md              <-- Deep mathematical formalisms, KV cache systems theory, and mechanism details
    ├── EXPERIMENT_REGISTRY.md              <-- Central registry of all designed/proposed experiments (E0 to E6)
    ├── IMPLEMENTATION_STATE.md             <-- Codebase audit, environment specs, and harness architecture
    ├── FINDINGS.md                         <-- Verified findings strictly partitioned by evidence level
    ├── FAILURES_AND_NEGATIVE_RESULTS.md    <-- Catalog of rejected dead-ends, failure modes, and negative result framing
    ├── OPEN_PROBLEMS.md                    <-- Core scientific, technical, and methodological hurdles awaiting solution
    ├── UNCERTAINTIES_AND_CONTRADICTIONS.md <-- Explicit records of conflicting sources, ambiguities, and verification checks
    ├── NEXT_STEPS.md                       <-- Concrete 14-day, 30-day, and 90-day execution roadmaps
    └── CHANGELOG.md                        <-- Version history of the memory system and major milestones
```

---

## 4. Evidence Classification Standards

To maintain absolute scientific integrity, all research statements in this memory system must be categorized under one of the following epistemic labels:

- **`ESTABLISHED FACT`**: Empirically demonstrated by established, peer-reviewed external literature cited in [LITERATURE_MAP.md](file:///c:/Users/Kartik/OneDrive/Desktop/Projects/btp-research/researchMemory/agentMemory/LITERATURE_MAP.md).
- **`OBSERVATION`**: Directly observed phenomenon in clean models or tooling, not yet established as a universal finding.
- **`EXPERIMENTAL RESULT`**: Empirically measured output produced by code executed within this project (None currently exist).
- **`HYPOTHESIS`**: A specific, testable proposition formulated for empirical validation.
- **`PROPOSED IDEA`**: An architectural, methodological, or conceptual concept not yet formally implemented or evaluated.
- **`ASSUMPTION`**: A foundational operational premise accepted for scope bounding.
- **`DECISION`**: An explicit, documented choice logged in [DECISION_LOG.md](file:///c:/Users/Kartik/OneDrive/Desktop/Projects/btp-research/researchMemory/agentMemory/DECISION_LOG.md).
- **`REJECTED / ABANDONED DIRECTION`**: A path discarded due to saturation, infeasibility, or poor fit.
- **`OPEN QUESTION`**: An unresolved problem requiring investigation.
- **`UNVERIFIED CLAIM` / `NOT ESTABLISHED IN AVAILABLE LOGS`**: Any statement lacking verifiable empirical evidence in project records.

> [!CAUTION]
> **Strict Anti-Hallucination Rule:** Never report planned work as completed work. Never state that an attack "succeeded," that a baseline was "reproduced," or that a model "exhibited backdoor behavior" unless accompanied by reproducible code execution logs and exact metric outputs.

---

## 5. Protocols for Successive Research Agents

When starting a new session on this project:
1. **Read [CURRENT_STATE.md](file:///c:/Users/Kartik/OneDrive/Desktop/Projects/btp-research/researchMemory/agentMemory/CURRENT_STATE.md)** to obtain the active state, current gates, and open blockers.
2. **Review [NEXT_STEPS.md](file:///c:/Users/Kartik/OneDrive/Desktop/Projects/btp-research/researchMemory/agentMemory/NEXT_STEPS.md)** for immediate priorities.
3. **Check [UNCERTAINTIES_AND_CONTRADICTIONS.md](file:///c:/Users/Kartik/OneDrive/Desktop/Projects/btp-research/researchMemory/agentMemory/UNCERTAINTIES_AND_CONTRADICTIONS.md)** before accepting assumptions that may have conflicting historical records.
4. **Whenever a decision is made or an experiment is run:** Immediately update [DECISION_LOG.md](file:///c:/Users/Kartik/OneDrive/Desktop/Projects/btp-research/researchMemory/agentMemory/DECISION_LOG.md), [EXPERIMENT_REGISTRY.md](file:///c:/Users/Kartik/OneDrive/Desktop/Projects/btp-research/researchMemory/agentMemory/EXPERIMENT_REGISTRY.md), [IMPLEMENTATION_STATE.md](file:///c:/Users/Kartik/OneDrive/Desktop/Projects/btp-research/researchMemory/agentMemory/IMPLEMENTATION_STATE.md), and append an entry to [CHANGELOG.md](file:///c:/Users/Kartik/OneDrive/Desktop/Projects/btp-research/researchMemory/agentMemory/CHANGELOG.md).
