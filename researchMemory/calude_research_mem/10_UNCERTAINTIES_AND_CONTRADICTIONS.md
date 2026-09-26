# 10 — CONTRADICTIONS, UNCERTAINTIES, AND POSSIBLE ERRORS

```text
Issue: U1 — Relationship between PF-SEB and the broader KQCB/KECB/KMCB taxonomy
Where it appeared: PF-SEB_Synopsis.md vs. research_plan.docx / Synopsis.docx
Earlier claim: The taxonomy treats "trigger = KV cache is in a given compressed
                state T(C0)," with KECB specifically = "a specific token-
                retention/eviction policy removes designated historical
                states" (passive framing — the model reacts to removal).
Later claim: PF-SEB claims the model actively manipulates the eviction
                algorithm's own importance-scoring inputs so that removal
                happens on cue — a mechanistically richer and more specific
                claim than "a policy removes states."
Evidence: research_plan.docx §6 / Synopsis.docx §12 (KECB definition) vs.
                PF-SEB_Synopsis.md §2 (office analogy, "actively feeding the
                memory manager misleading signals").
Current interpretation: Most likely PF-SEB is a refinement/sharpening of KECB,
                not a contradiction — but no document explicitly states this
                relationship, and it is also possible PF-SEB is intended as a
                fully separate, competing proposal.
What needs verification: Ask directly whether PF-SEB supersedes the broader
                taxonomy as the sole direction, sits as the chosen instance of
                KECB, or is being kept as a separate/parallel idea.
```

```text
Issue: U2 — Status of the open "entry point" decision (Direction A/B/C)
Where it appeared: overview.md, "On the horizon"
Earlier claim: overview.md (2026-09-14) lists this as an explicitly OPEN
                decision: Direction A (inference-time eviction study, no
                training, recommended starting point) / Direction B (trained
                weight-level KQCB) / Direction C (merging variant).
Later claim: PF-SEB_Synopsis.md (undated, but the most recent artifact
                provided) describes a trained model (i.e., NOT "no training"),
                specifically targeting eviction — which doesn't cleanly match
                any single one of A/B/C as originally described (it's trained,
                like B, but eviction-focused, like the eviction emphasis of A/
                the KECB family).
Evidence: overview.md "On the horizon"; PF-SEB_Synopsis.md throughout
                (explicitly a trained-model attack, §5: "requires an attacker
                who can fine-tune the model").
Current interpretation: PF-SEB likely represents a synthesis/evolution beyond
                the original A/B/C framing rather than a clean selection of one
                option, but this is not confirmed.
What needs verification: Whether Direction A (the "no training required"
                inference-time eviction study) was ever attempted as a
                stepping stone before PF-SEB was formulated, or whether the
                project moved directly to a trained-model design.
```

```text
Issue: U3 — Possible link between the two early "saturation" pivots
Where it appeared: overview.md, "Background & evolution" and "Tools &
                resources"
Earlier claim: Pivot 1 (broad multi-mechanism backdoor framework) was
                saturated by "a 'unified framework' paper" (May 2026).
Later claim: Pivot 2 (multi-agent IFC) lists a "prior art ecosystem" that
                includes "the May 2026 unified framework paper" alongside
                ShadowLogic, CapChain, CapAgent, SPA, GIF, LaunchSafe.
Evidence: overview.md, both pivot descriptions and the "Tools & resources"
                list.
Current interpretation: It is unclear whether this is the SAME "unified
                framework" paper causing both pivots (i.e., one paper spanning
                both weight-transformation backdoors and multi-agent IFC,
                which would be unusual scope for one paper) or two
                DIFFERENT papers that happen to share the same informal name
                in the summarized memory.
What needs verification: The actual title/citation of the "unified framework"
                paper(s) referenced — not available in any provided artifact.
This may be an artifact of memory-compression (the earlier summarization
                process) rather than a real ambiguity in the original
                conversations.
```

```text
Issue: U4 — Venue deadline currency
Where it appeared: All three formal documents (research_plan.docx §13,
                Synopsis.docx §26/§30) plus this compilation's compiled date
Earlier claim: ICLR 2027 paper deadline listed as 25 Sep 2026 (AoE); MLSys 2027
                deadline 30 Oct 2026; IEEE S&P 2027 Cycle 2 paper deadline
                17 Nov 2026; all captured as of a 10/14 Sep 2026 snapshot.
Later claim: N/A — no later snapshot available.
Evidence: research_plan.docx §13.1; Synopsis.docx §26.
Current interpretation: As of this compilation's date (2026-09-23), the ICLR
                2027 paper deadline (25 Sep 2026) would be either imminent or
                already passed, and should not be assumed still open. All
                other venue dates should likewise be re-verified.
What needs verification: Live re-check of every venue deadline against the
                official CFP before any submission planning proceeds — this is
                also the project's own explicitly stated requirement (both
                documents flag this caveat themselves).
```

```text
Issue: U5 — Appendix B of the synopsis is an empty stub
Where it appeared: Runtime_Conditioned_Backdoors_KV_Cache_Synopsis.docx,
                "B. Appendix B — Potential Reviewer Criticisms"
Earlier claim: The Table of Contents lists Appendix B as a planned section.
Later claim: The section heading exists in the body of the document but
                contains no content beneath it.
Evidence: Synopsis.docx, Appendix B heading, immediately followed by Appendix
                C with no intervening body text.
Current interpretation: This appears to be an incomplete/unfinished part of
                the draft, not a deliberate omission.
What needs verification: Whether a filled-in version of Appendix B exists
                elsewhere and simply wasn't included in the copy provided to
                this session.
```

```text
Issue: U6 — Whether any literature claims in the three documents have been
                independently verified by Kartik/Claude beyond the citation
                list itself
Where it appeared: Throughout Synopsis.docx and research_plan.docx
Earlier claim: Documents present literature findings (e.g., CacheTrap's
                mechanism, Group B papers' specific results) in confident,
                declarative language.
Later claim: The documents' own "Source Notes / Claims Discipline" section
                (Synopsis.docx §17 in the plan; similar caveat throughout)
                explicitly says the central novelty claim "must be re-checked
                immediately before submission" and is not yet a settled fact.
Evidence: Synopsis.docx §17 (plan) and repeated flags throughout the synopsis
                (e.g., "Novelty checkpoint" callout, §7.4).
Current interpretation: The documents are internally consistent about this —
                they present findings confidently for planning purposes while
                simultaneously flagging the overall novelty claim as
                provisional. This is not a contradiction so much as an
                explicit, self-aware limitation that should be preserved, not
                resolved away.
What needs verification: A live literature re-check, as the documents
                themselves prescribe, before any external claims are made.
```

```text
Issue: U7 — No evidence of implementation status
Where it appeared: Absence across all three documents and all three memory
                files.
Earlier claim: N/A.
Later claim: N/A.
Evidence: None of the provided material contains code, logs, results, figures,
                or any other artifact indicating that Phase 0 (or any later
                phase) has been started.
Current interpretation: The project is very likely still at the design/
                planning stage as of the latest available material.
What needs verification: Direct confirmation from Kartik of whether any
                implementation work has begun outside of what was shared in
                this session (e.g., in a separate code repository not visible
                here).
```
