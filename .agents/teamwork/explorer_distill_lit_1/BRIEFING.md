# BRIEFING — 2026-10-08T17:24:00Z

## Mission
Conduct an exhaustive, evidence-disciplined literature survey on LLM extraction attacks and defenses (2016–2026), covering the evolution of extraction from prediction APIs to modern reasoning-trace/CoT distillation, attack modalities, existing defenses, and core failure modes.

## 🔒 My Identity
- Archetype: explorer
- Roles: Literature Scout, Synthesizer
- Working directory: c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\explorer_distill_lit_1\
- Original parent: 51338a6e-4710-46d6-808d-1e7576675ad3
- Milestone: M1 (Systematic Literature Survey & Threat Model Formalization)

## 🔒 Key Constraints
- Read-only investigation — do NOT implement experimental code
- Strict AGENTS.md evidence discipline (classify every substantive statement as SOURCE FACT, INFERENCE, HYPOTHESIS, EXPERIMENTAL RESULT, DECISION)
- ZERO hallucinated citations: verify exact authors, titles, publication venues, years, and arXiv/conference identifiers
- Self-contained handoff report in `handoff.md` with 5 required sections (Observation, Logic Chain, Caveats, Conclusion, Verification Method)

## Current Parent
- Conversation ID: 51338a6e-4710-46d6-808d-1e7576675ad3
- Updated: 2026-10-08T17:24:00Z

## Investigation State
- **Explored paths**:
  - `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\ORIGINAL_REQUEST.md`
  - `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\orchestrator_distill_1\plan.md`
  - Verified 27 foundational and modern literature papers across 2016–2026 (Tramèr, Orekondy, Jagielski, Krishna, Wallace, Kim & Rush, Wang, Taori, Gudibande, Carlini, DeepSeek-AI, Juuti, Kirchenbauer, Aaronson, Christ, Kuditipudi, Lee, Kariyappa, Szyller, Jia, Sadasivan, Sander).
- **Key findings**:
  - Soft-label APIs leak linear subspaces via low-rank SVD (Carlini et al., 2024), recovering hidden dimensions and projection weights.
  - Hard-token text APIs prevent weight reconstruction but remain completely vulnerable to sequence-level KD (Kim & Rush, 2016; Taori et al., 2023; DeepSeek-AI, 2025).
  - Existing defenses (PRADA, watermarking, output perturbations, Proof of Learning) fail fundamentally due to Sybil distribution, paraphrasing/student loss minimization (Sadasivan, Krishna), utility degradation in non-convex text generation, and the legitimacy of student training compute.
  - Three irreducible bottlenecks identified: Semantic Equivalence, Channel Equivalence, and the Utility-Poisoning Trilemma.
- **Unexplored areas**:
  - Detailed threat model formalization (assigned to peer `explorer_distill_threat_1`).
  - Cross-domain analogies (traitor tracing, hardware IC metering, game-theoretic signaling) (Milestone M2).

## Key Decisions Made
- Authored complete, high-rigor literature survey at `alternate_research/distillation_defense/03_LITERATURE_SURVEY.md`.
- Authored self-contained 5-component hard handoff report at `.agents/teamwork/explorer_distill_lit_1/handoff.md`.
- 100% citations verified; zero hallucinations.

## Artifact Index
- `.agents/teamwork/explorer_distill_lit_1/DISPATCH.md` — Incoming message log
- `.agents/teamwork/explorer_distill_lit_1/BRIEFING.md` — Agent state and persistent memory
- `.agents/teamwork/explorer_distill_lit_1/progress.md` — Liveness heartbeat and progress log
- `.agents/teamwork/explorer_distill_lit_1/handoff.md` — 5-component hard handoff report
- `alternate_research/distillation_defense/03_LITERATURE_SURVEY.md` — Campaign M1 deliverable document
