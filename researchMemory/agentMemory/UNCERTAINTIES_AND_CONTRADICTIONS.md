# Contradictions, Uncertainties, and Ambiguity Records

This document explicitly identifies, logs, and analyzes conflicting statements, ambiguities, and unverified assumptions discovered across the project's historical memory sources and literature notes.

---

## 1. Structured Uncertainty Records

```text
Issue ID: U1
Title: Relationship between PF-SEB and the broader KQCB/KECB/KMCB taxonomy
Conflicting Sources:
  - Source A: research_plan.docx (10 Sep 2026) & Synopsis.docx (14 Sep 2026) define KECB
    passively: "a specific token-retention/eviction policy removes designated historical states."
  - Source B: PF-SEB_Research_Plan.docx (new primary docx) & PF-SEB_Synopsis.docx:
    Explicitly define PF-SEB as: "a focused specialisation of the eviction-conditioned (KECB)
    branch of the broader runtime-conditioned-backdoor research direction; deliberately narrower
    in scope than the full KQCB/KECB/KMCB programme."
Current Interpretation:
  The two documents establish a complementary Two-Track program: Track 1 provides the broad
  systems framework across quantization, eviction, and merging, while Track 2 provides the
  razor-sharp, highly novel empirical paper on active eviction gaming.
Resolution Status: FULLY RESOLVED VIA PRIMARY DOCX ARTIFACTS.
Action Taken: Logged as Decision D14. Track 1 acts as the umbrella systems framing; Track 2
              acts as the flagship empirical attack contribution.
```

```text
Issue ID: U2
Title: Status of the open "entry point" decision (Direction A vs B vs C vs DeepSeek KQCB)
Conflicting Sources:
  - Source A: overview.md (14 Sep 2026) lists entry point as an OPEN decision:
    Direction A (inference-time eviction study, no training) vs Direction B (trained weight-level KQCB)
    vs Direction C (merging variant).
  - Source B: DeepSeek review (18 Sep 2026) strongly recommends starting with trained KQCB
    (Quantization-first via STE) due to mathematical tractability.
  - Source C: PF-SEB_Synopsis.md focuses entirely on trained eviction gaming.
Current Interpretation:
  The project transitioned past Direction A (un-trained study) because clean-model degradation
  is already established in literature; proving a backdoor requires training. Between KQCB and PF-SEB,
  DeepSeek advises KQCB for engineering safety, while PF-SEB offers higher novelty.
Resolution Status: RESOLVED STRATEGICALLY.
Action Taken: Phase 0 harness will support both. Phase 2 empirical training will begin with KQCB
              to de-risk the dual-regime training loop, followed immediately by PF-SEB.
```

```text
Issue ID: U3
Title: Identity of the "May 2026 Unified Framework" saturation paper
Conflicting Sources:
  - Source A: overview.md cites a "unified framework paper" (May 2026) as saturating Pivot 1
    (weight-transformation backdoors).
  - Source B: overview.md also lists "the May 2026 unified framework paper" under the prior art
    ecosystem for Pivot 2 (multi-agent IFC) alongside CapChain and LaunchSafe.
Current Interpretation:
  It is scientifically unlikely that a single paper simultaneously covered weight-transformation
  backdoors and multi-agent information-flow control. This appears to be a compression artifact
  from earlier summarization where two distinct May 2026 preprints were given an identical informal label.
Resolution Status: UNVERIFIED LITERATURE ARTIFACT.
Action Required: When drafting related work, perform a fresh Google Scholar search to identify the
                 exact titles/authors of the May 2026 weight-transformation and agent-IFC papers.
```

```text
Issue ID: U4
Title: Currency of tracked venue submission deadlines
Conflicting Sources:
  - Source A: research_plan.docx §13.1 lists ICLR 2027 paper deadline as 25 Sep 2026 (AoE);
    MLSys 2027 deadline as 30 Oct 2026; IEEE S&P 2027 Cycle 2 as 17 Nov 2026.
  - Source B: Current project date is 26 September 2026.
Current Interpretation:
  As of today (26 September 2026), the ICLR 2027 deadline has already passed. The MLSys 2027
  and IEEE S&P 2027 deadlines are dangerously close (4–7 weeks) with zero code implemented.
Resolution Status: ACTIVE TIMELINE CONFLICT.
Action Taken: Formally designate USENIX Security 2027 (Cycle 2: 26 Jan 2027) as the primary
              conference target, providing a realistic 4-month experimental and drafting runway.
```

```text
Issue ID: U5
Title: Missing content in Synopsis Appendix B ("Potential Reviewer Criticisms")
Conflicting Sources:
  - Source A: Runtime_Conditioned_Backdoors_KV_Cache_Synopsis.docx contains Appendix B as an empty heading.
  - Source B: PF-SEB_Synopsis.docx (new primary docx) contains the complete, fully articulated
    Appendix B detailing five critical reviewer objections and their pre-emptions.
Current Interpretation:
    The general synopsis omitted the section in draft, but the specialized PF-SEB synopsis
    completed it in full.
Resolution Status: FULLY RESOLVED VIA PF-SEB_Synopsis.docx.
Action Taken: All five reviewer criticisms and pre-emptions (KECB differentiation, proxy fidelity,
              generic fragility, threat model realism, and generalization) have been formally
              integrated into agentMemory/FAILURES_AND_NEGATIVE_RESULTS.md §6.
```

```text
Issue ID: U6
Title: Independent verification of cited Group B literature results
Conflicting Sources:
  - Source A: Synopsis.docx cites specific empirical metrics from Group B literature
    (e.g., system prompt leakage in *The Pitfalls of KV Cache Compression*).
  - Source B: The project has not yet reproduced any of these results locally.
Current Interpretation:
  All quantitative claims regarding clean-model compression degradation are external citations,
  not locally verified findings.
Resolution Status: EXPLICITLY LABELED.
Action Required: Gate G2 mandates reproducing at least one clean-model compression baseline locally
                 before asserting that an attack amplifies degradation.
```

```text
Issue ID: U7
Title: DeepSeek scoping recommendations vs the broad synopsis taxonomy
Conflicting Sources:
  - Source A: Synopsis.docx describes a 6-variant taxonomy spanning quantization, eviction,
    merging, context thresholds, load conditioning, and multi-condition composition across 4 model families.
  - Source B: DeepSeek review (18 Sep 2026) warns that this scope is fatal for a B.Tech project,
    strongly urging reduction to 1–2 models, 1 compression family, 1 synthetic target, and 1 defense.
Current Interpretation:
  The Synopsis represents a maximalist proposal for academic supervisors, while DeepSeek's review
  represents a pragmatic, realistic engineering roadmap.
Resolution Status: RESOLVED IN FAVOR OF SCOPED EXECUTION.
Action Taken: Logged as Decision D12. Initial execution will adhere strictly to DeepSeek's scoped
              matrix (1–2 models, 1B–3B parameter range, synthetic target).
```

```text
Issue ID: U8
Title: Codebase implementation status
Conflicting Sources:
  - Source A: Planning documents describe phased execution methodologies in active present-tense.
  - Source B: Physical workspace audit reveals zero code files in btp-research.
Current Interpretation:
  No implementation work has been executed. The project is at Day 0 of engineering.
Resolution Status: EMPIRICALLY CONFIRMED.
Action Taken: Formally logged in agentMemory/IMPLEMENTATION_STATE.md and agentMemory/EXPERIMENT_REGISTRY.md.
```
