# 00 — RESEARCH MEMORY (Master File)

**Project:** BTech Final-Year Research Project — Kartik
**Domain:** AI / LLM Security — Machine Learning Security & Inference Systems
**Compiled:** 2026-09-23, from the project's memory store and three project artifacts

## Important note on sourcing

This memory package was reconstructed from two kinds of source material:

1. **The project's persistent memory files** (`overview.md`, `learnings-and-approach.md`, `index.md`), which are themselves already-compressed summaries created earlier in the project (last updated 2026-09-14). They record *stated* facts and decisions but are not verbatim conversation transcripts.
2. **Three project artifacts** provided directly as documents in this session:
   - `runtime_conditioned_backdoors_kv_cache_research_plan.docx` — broad research roadmap, evidence snapshot dated 10 Sep 2026.
   - `Runtime_Conditioned_Backdoors_KV_Cache_Synopsis.docx` — a fuller B.Tech synopsis draft, evidence snapshot dated 14 Sep 2026.
   - `PF-SEB_Synopsis.md` — "Policy-Fingerprinted Self-Eviction Backdoors," an undated, intuition-first short synopsis that appears to be the **most recently refined** formulation of the idea.

**No raw conversation transcripts were available to this compilation.** Wherever this document infers chronology, reasoning, or motivation, it is inferring from internal evidence in these four sources (document dates, framing language, what each document assumes vs. explains), not from directly observed dialogue. Anywhere the reasoning behind a change is not explicit in the source material, this is stated plainly rather than invented. Treat every "why" statement below as an inference unless marked otherwise.

---

## 1. Project Identity

- **Researcher:** Kartik, described in project memory as an AI security researcher working on novel attack surfaces in LLM safety, producing this work toward a B.Tech final-year research project and academic publication. [Source: overview.md]
- **Field:** ML/LLM security, specifically backdoor attacks conditioned on inference-time runtime state rather than input content.
- **Goal (stated):** identify a genuinely open research direction, establish novelty relative to prior art, and produce a publishable contribution. [Source: overview.md]

## 2. Current Research Direction (best-supported reading)

The project has gone through at least two visible levels of refinement, both present in the provided artifacts:

1. **Broad framework level** (`research_plan.docx`, `Synopsis.docx`): "Runtime-Conditioned Backdoors" — a general claim that a legitimate KV-cache runtime transformation (quantization, eviction, or merging) can be intentionally trained into an LLM as a backdoor trigger, with the model benign under the reference full-cache condition. This is organized into a taxonomy (KQCB / KECB / KMCB / context-threshold / load-conditioned / composed) and a five-phase (Synopsis: six-phase, Plan: also six-phase 0–5) methodology.
2. **Narrower, sharper mechanism** (`PF-SEB_Synopsis.md`): "Policy-Fingerprinted Self-Eviction Backdoors" (PF-SEB) — a specific instantiation that goes beyond "trigger = cache state" to "trigger = the model *actively manipulating* the inputs an honest, unmodified eviction algorithm (e.g., H2O) uses to decide what to evict." The model learns to make an internal "suppressor" state look unimportant to the eviction policy's attention-based importance scoring, so the policy evicts it — through no fault of its own — exactly when it matters. This is explicitly framed in the PF-SEB document as a stronger, more specific claim than "compression accidentally breaks safety" or "the model is passively triggered by a compressed cache state."

**Best current understanding of project status:** PF-SEB is the most refined and most recently articulated formulation of the KV-cache-backdoor idea. It has not been explicitly logged in project memory (`overview.md`, last updated 2026-09-14) as a superseding decision — the memory file still describes the three-way KQCB/KECB/KMCB entry-point decision (Direction A/B/C) as open. **This compilation cannot confirm whether PF-SEB has been formally adopted as *the* direction, or is a candidate refinement sitting alongside the broader taxonomy.** This is flagged as an open item — see `03_DECISION_LOG.md` and `10_UNCERTAINTIES_AND_CONTRADICTIONS.md`.

Full technical detail: see `08_TECHNICAL_KNOWLEDGE.md`.

## 3. Executive Research History (condensed)

Reconstructed from `overview.md` and cross-checked against the three artifacts. See `02_RESEARCH_TIMELINE.md` for the phase-by-phase version.

1. **Pivot 1 (pre-project-memory-import):** An initial, broader proposal covering quantization-, pruning-, LoRA-merge-, distillation-, and compilation-triggered backdoors was found to be largely saturated by 2024–2026 prior art, including a "unified framework" paper (May 2026) that preempted the core novelty claim. [Source: overview.md — stated as fact of what happened, no further detail available]
2. **Pivot 2 (pre-project-memory-import):** A parallel thread on LLM agent security (capability tokens, information-flow control (IFC), audit logging for multi-agent systems) also hit a crowded landscape, with competing systems published in 2026. This produced a fallback idea, "LaunderBench" (instruction laundering / "attacking the labeler" in multi-agent IFC), which remains a possible parallel/fallback direction rather than the main line. [Source: overview.md]
3. **Current direction adopted:** KV-cache compression-conditioned backdoors, identified as genuinely open with no dedicated attack paper found as of the research sessions reflected in memory. [Source: overview.md]
4. **Elaboration into a formal research plan** (`research_plan.docx`): threat model, KQCB/KECB/KMCB taxonomy, five-criterion vulnerability-vs-backdoor checklist (implicit in synopsis, referenced in plan), six-phase methodology, metrics, timeline, venue targets (USENIX Security 2027, IEEE S&P 2027, MLSys 2027, ICLR 2027, TMLR/JMLR/TIFS).
5. **Elaboration into a B.Tech synopsis** (`Synopsis.docx`): substantially expanded literature review (Groups A–D), identification of **CacheTrap** (arXiv:2511.22681) as the closest prior art — a hardware-fault-injection Trojan, explicitly differentiated from this project's fine-tuning-only threat model — plus a full threat model, experimental design, ethics section, computational requirements, risk register, and appendices (supervisor discussion questions, reviewer-criticism appendix left as a stub, approval checklist).
6. **Further refinement to PF-SEB** (`PF-SEB_Synopsis.md`): narrows the mechanism from "any cache-state trigger" to a specific active-gaming mechanism against a real eviction algorithm (H2O-style attention-based importance scoring), with a "suppressor" internal state and a concrete falsification/causal-proof experimental table (pin/delete tests). This document explicitly distinguishes itself from "compression accidentally breaks safety" (a bug) and frames the contribution as "the model actively manipulates an honest algorithm's own decision inputs," which it claims (as a hypothesis, not a verified literature fact) is not shown by existing work.

**Reason for the plan → synopsis → PF-SEB progression:** not explicitly stated in any of the three documents. Each later document is self-consistent with the earlier ones (same threat model shape, same core KV-cache framing, same author's positioning against CacheTrap/ShadowLogic-type prior art) but narrows scope and sharpens the specific claimed mechanism. This reads as a natural novelty-sharpening progression consistent with the stated research principle "novelty lives in the trigger mechanism... specificity of the trigger condition" [Source: learnings-and-approach.md], but this inference is not directly confirmed by conversation evidence.

## 4. Major Decisions Already Made (see `03_DECISION_LOG.md` for full log)

- Deprioritize the broad multi-mechanism (quantization/pruning/LoRA-merge/distillation/compilation) backdoor framing — saturated.
- Deprioritize (but keep as fallback) the multi-agent IFC / "LaunderBench" direction — crowded landscape.
- Adopt KV-cache compression as the trigger surface.
- Treat quantization, eviction, and merging as **mechanistically distinct** trigger families to be studied separately, not pooled (explicit methodological decision in `research_plan.docx` §2.2 and repeated in `Synopsis.docx` §3.3).
- Adopt a fine-tuning-only attacker (no hardware access, no serving-infrastructure control) as the threat model, explicitly to differentiate from CacheTrap.
- Adopt a staged, low-risk target-behavior design (synthetic marker → controlled instruction-priority behavior → approved red-team content) rather than starting with real harmful payloads (`Synopsis.docx` §15, echoed in `research_plan.docx` §7.3).
- Adopt a five-criterion checklist (intentional training, clean-baseline comparison, trigger specificity, payload specificity, stealthiness) to operationally distinguish a "trained backdoor" claim from an "emergent compression vulnerability" claim.
- Identify CacheTrap as the closest prior art and commit to explicitly differentiating against it before submission.
- (PF-SEB) Narrow the mechanism to active manipulation of an honest eviction algorithm's own decision inputs, rather than passive triggering by cache state — this is the most recent apparent decision but its formal status relative to the broader taxonomy is **unconfirmed** (see §2 above and `10_UNCERTAINTIES_AND_CONTRADICTIONS.md`).

## 5. Current Hypotheses (not yet validated by any recorded experiment)

- **H1 (Existence, RQ1):** A model can be trained so that full-cache inference is benign but a specified KV-cache transformation reliably activates a targeted behavior.
- **H2 (Mechanism distinctness, RQ2):** Quantization, eviction, and merging triggers produce mechanistically distinct attack profiles.
- **H3 (Runtime thresholds, RQ3):** The trigger can be conditioned on continuous/threshold runtime quantities (context length, cache budget, retained-token ratio), not just a fixed configuration.
- **H4 (Mechanism, RQ4):** The transition is localizable to a small subset of layers/heads and expressible via representation-geometry evidence.
- **H5 (Defense, RQ5):** Cache-aware differential auditing can detect the transition at practical query cost, and security-aware retention/precision can preserve safety at bounded efficiency cost.
- **H6 (PF-SEB-specific):** A model can learn to actively bias a real, unmodified, attention-based eviction algorithm's importance scoring so that a specific internal "suppressor" state is evicted exactly when needed — and this effect is specific to the real eviction policy and budget (not reproduced by unlimited memory, a different policy at the same budget, or removing the fine-tuning).

**No experiment described in any of the three documents has been reported as run.** All experimental content is design/proposal only. See `07_EXPERIMENT_REGISTRY.md`.

## 6. Established Facts vs. the Project's Own Claims

The project's artifacts cite external literature for several claims that should be treated as **established findings of cited third-party work**, not the project's own results:

- KV-cache compression (quantization, eviction, merging) is a mature, widely used systems technique (StreamingLLM, H2O, Scissorhands, SnapKV, KIVI, MiniCache, CacheGen).
- Compression can *unintentionally* degrade safety/instruction-following/system-prompt retention in **clean, non-backdoored** models (Pitfalls of KV Cache Compression, ACL 2026; When Efficiency Meets Safety, ACL 2026; Alignment Collapse Under KV Cache Quantization, 2026 preprint).
- The KV cache itself is a privacy-relevant object (Shadow in the Cache / NDSS 2026).
- LLMs can carry dynamic/structural/persistent conditional backdoors (BadChain, Sleeper Agents, dynamic-trigger literature, BackdoorBench).
- A KV-cache-triggered Trojan already exists via **hardware fault injection** (CacheTrap, 2026 preprint) — the closest prior art, explicitly differentiated on threat-model grounds (hardware access vs. fine-tuning access; single fixed bit vs. a policy family; no training-time implantation vs. trained).
- A deployment-artifact backdoor exists at the chat-template level (Inference-Time Backdoors via Chat Templates, CCS 2026) and at the computational-graph level (ShadowLogic, CAMLIS 2025) — both distinct from the KV-cache-compression trigger.

None of these external findings have been independently re-verified by the project; they are as reported by the documents, sourced to the reference list in each artifact (see `05_LITERATURE_MAP.md`). The **novelty claim itself — that no prior work trains a backdoor whose trigger is a legitimate cache-compression policy — is explicitly flagged in all three documents as a hypothesis requiring re-verification immediately before submission**, not a settled fact. This is the single most important caveat in the whole project.

## 7. What Must NOT Be Re-Investigated From Scratch

- The saturation analysis of the broad multi-mechanism backdoor framing (quantization/pruning/LoRA-merge/distillation/compilation) — already found saturated.
- The saturation analysis of the multi-agent IFC / capability-token / audit-logging landscape — already found crowded (kept only as fallback).
- The basic case that KV-cache compression is a real, load-bearing production technique — already well established via cited systems literature (Group A in the synopsis).
- The basic case that compression can *unintentionally* affect safety in clean models — already well established via cited 2025–2026 security literature (Group B).
- The identification of CacheTrap as the nearest prior art and the specific grounds for differentiating from it (hardware fault injection vs. fine-tuning-time training) — already done and documented; do not re-derive from zero, but **do re-run the novelty search close to submission**, per the project's own checklist.

See `04_RESEARCH_QUESTIONS.md` and `05_LITERATURE_MAP.md` for full detail, and `06_IDEAS_AND_DIRECTIONS.md` for the full inventory including rejected/deprioritized directions.

## 8. Links to Detailed Memory Files

- `01_CURRENT_STATE.md` — compact current-state snapshot
- `02_RESEARCH_TIMELINE.md` — phase-by-phase chronology
- `03_DECISION_LOG.md` — structured decision log
- `04_RESEARCH_QUESTIONS.md` — answered / partial / open / superseded questions
- `05_LITERATURE_MAP.md` — every source cited across the three artifacts
- `06_IDEAS_AND_DIRECTIONS.md` — full inventory of ideas and their status
- `07_EXPERIMENT_REGISTRY.md` — every proposed experiment (none yet run)
- `08_TECHNICAL_KNOWLEDGE.md` — KV-cache and backdoor technical background, attack taxonomy, PF-SEB mechanism detail
- `09_ARTIFACT_INDEX.md` — index of the three source documents and their relationships
- `10_UNCERTAINTIES_AND_CONTRADICTIONS.md` — open uncertainties, including the plan/synopsis/PF-SEB relationship

---

## INSTRUCTIONS FOR THE NEXT RESEARCH AGENT

**What this project is trying to solve:** whether a legitimate, unmodified KV-cache compression mechanism used by real LLM serving systems (quantization, eviction, or merging) can be *intentionally trained* into a backdoor trigger — such that a model passes a standard full-cache safety audit but reveals attacker-chosen behavior once deployed under ordinary, realistic compression. The most refined version of this idea (PF-SEB) narrows it further: the model doesn't just react to a compressed cache state, it actively feeds misleading importance signals to an honest eviction algorithm so that algorithm — doing exactly what it's designed to do — evicts a hidden internal "suppressor" state at the wrong moment.

**What has already been investigated (do not repeat without new evidence):**
- Two broader research directions (multi-mechanism weight/deployment backdoors; multi-agent IFC security) were explored and found too crowded/saturated. Don't restart there without a specific new gap.
- A four-group literature review (systems efficiency; compression-safety; backdoor literature; runtime/deployment-conditioned security) has already been compiled with ~25 sources (see `05_LITERATURE_MAP.md`). Re-use it; re-verify currency near submission, don't rebuild it from scratch.
- CacheTrap has already been identified and differentiated as the closest prior art. Any new agent must be aware of this paper and the specific differentiation argument before proposing "KV cache as trigger" as if it were unclaimed territory.
- A five-criterion backdoor-vs-vulnerability checklist and a full threat model already exist. Use them; don't re-derive.

**What must not be assumed:**
- That any experiment has been run. **Nothing in the available material shows an implemented model, a training run, or a measured result.** Everything is design/proposal.
- That PF-SEB has formally replaced the broader KQCB/KECB/KMCB taxonomy as the sole direction — this relationship is unconfirmed (see `10_UNCERTAINTIES_AND_CONTRADICTIONS.md`).
- That the novelty claim ("no prior work does this") is verified — all three documents explicitly flag it as needing re-verification before submission.
- That "Appendix B — Potential Reviewer Criticisms" in the synopsis has content — it is a stub heading with no body in the provided document.

**What remains uncertain and needs verification:**
1. Whether PF-SEB is the current single direction or one candidate among the taxonomy (KQCB/KECB/KMCB/PF-SEB).
2. Whether the "entry point" decision flagged as open in `overview.md` (Direction A: inference-time eviction study / Direction B: trained weight-level KQCB / Direction C: merging variant) has since been resolved — PF-SEB looks like a resolution toward an eviction-based direction, but this is inferred, not confirmed.
3. Current currency of all cited 2026 venue deadlines and paper statuses (explicitly flagged by the project itself as needing re-checking).
4. Whether Phase 0 (instrumentation harness) has been started in code anywhere — no implementation artifacts were provided in this session.

**Next research task implied by the project's own plan:** Phase 0 — build the deterministic KV-cache instrumentation/A-B harness (model selection, cache abstraction exposing full vs. transformed cache, deterministic transformation logging) — as this is the explicit first gate (G1) in every version of the methodology, and there is no evidence it has been started. If PF-SEB is confirmed as the adopted direction, Phase 0 should specifically instrument an H2O-style attention-based eviction policy so that its importance-score inputs are loggable and manipulable, since PF-SEB's causal-proof experiments (pin/delete tests, §4 of `PF-SEB_Synopsis.md`) depend on this.

**Files to read next, in order:** `01_CURRENT_STATE.md` → `08_TECHNICAL_KNOWLEDGE.md` (for the PF-SEB mechanism) → `03_DECISION_LOG.md` → `07_EXPERIMENT_REGISTRY.md` → `05_LITERATURE_MAP.md` if a literature check is needed.
