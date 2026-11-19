# BTP research explained — 16 November 2026

Open **[the full plain-language guide](RESEARCH_EXPLAINED.md)**.

For a single document, open **[the combined PDF](BTP_Research_Guide_and_Complete_Audit.pdf)**: 45 pages containing the complete guide, all seven diagrams, the original audit, the decision index, and source references.

- **First reading:** sections 1–4 explain the idea, vocabulary, actual trainer, and memory-manager claim.
- **What happened and why:** section 5 gives the dated timeline; [DECISION_INDEX.md](DECISION_INDEX.md) indexes every formal decision.
- **Evidence and models:** sections 6–12 distinguish completed training, calibration-only trials, mechanisms, and defenses.
- **Professor preparation:** section 16 includes a two-minute explanation, five-slide outline, and likely questions.

The important distinction is between a demonstrated learned response to attention masking and a verified attack on an ordinary production KV-cache operation. The guide explains the gap and the successful payload's utility cost alongside the positive results.

| Figure | Editable vector | Presentation copy |
|---|---|---|
| Model and memory | [SVG](diagrams/01_model_and_memory.svg) | [PNG](diagrams/01_model_and_memory.png) |
| Training architecture | [SVG](diagrams/02_training_architecture.svg) | [PNG](diagrams/02_training_architecture.png) |
| Prefill timing distinction | [SVG](diagrams/03_prefill_timing_gap.svg) | [PNG](diagrams/03_prefill_timing_gap.png) |
| Original hypothesis versus evidence | [SVG](diagrams/04_hypothesis_vs_evidence.svg) | [PNG](diagrams/04_hypothesis_vs_evidence.png) |
| Timeline | [SVG](diagrams/05_research_timeline.svg) | [PNG](diagrams/05_research_timeline.png) |
| Results and utility cost | [SVG](diagrams/06_results_and_utility.svg) | [PNG](diagrams/06_results_and_utility.png) |
| Layer intervention and defense | [SVG](diagrams/07_layer_intervention_and_defense.svg) | [PNG](diagrams/07_layer_intervention_and_defense.png) |

The [evidence snapshot](evidence_snapshot.json) includes checked counts and original source hashes. Source-file links in the full guide resolve within the repository; the ZIP contains the explanatory documents and figures, not a copy of all experimental evidence.

Verification performed for this artifact: recount three 5,601-row confirm files, check unique pool/record pairs and agreement with stored summaries, recount both 296-row utility files, parse the seven SVGs, validate local guide links, and check diagram text bounds. These are document/artifact checks, not a new model experiment or an independent reproduction of training.
