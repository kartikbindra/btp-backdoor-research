# Alternate Research Track: Next-Generation Decision Models & AI Safety

This directory contains the research investigation, architectural ideation, threat modeling, and publication blueprints exploring **non-autoregressive "System 1" Decision Models** (Contrastive-LM, Laya, Jev), **functional encoder factorizations**, and the **MenuGuard research initiative**.

---

## Directory Navigation

| File | Description | Core Scientific Contribution |
| :--- | :--- | :--- |
| **[`FINDINGS_CLM.md`](file:///c:/Users/Kartik/OneDrive/Desktop/Projects/btp-research/alternate_research/FINDINGS_CLM.md)** | **30-Cycle Adversarial Ledger** | The authoritative research campaign ledger evaluating SQ1–SQ24, prior-art counter-searches, kill rounds, MenuGuard selection, and causal evaluation methodology. |
| **[`COMPREHENSIVE_RESEARCH_REPORT.md`](file:///c:/Users/Kartik/OneDrive/Desktop/Projects/btp-research/alternate_research/COMPREHENSIVE_RESEARCH_REPORT.md)** | Master Research Synthesis | Publication-grade overview synthesizing SOTA decision models, creative multi-encoder architectures, threat modeling, and defense ladders. |
| **[`01_LITERATURE_SURVEY.md`](file:///c:/Users/Kartik/OneDrive/Desktop/Projects/btp-research/alternate_research/01_LITERATURE_SURVEY.md)** | Literature Survey & Foundations | Jev (TypeSafe AI / Diogo Almeida / RLCD), Contrastive-LM (CLM-8B), Laya (ModernBERT-large), Julia One, and classical predecessors (DT, Gato, DPR, CLIP). |
| **[`02_CREATIVE_ENCODER_ARCHITECTURES.md`](file:///c:/Users/Kartik/OneDrive/Desktop/Projects/btp-research/alternate_research/02_CREATIVE_ENCODER_ARCHITECTURES.md)** | Creative Functional Encoders | 6 formal architectures beyond State-Action: Tri-Encoder (State, Action, Constraint), Affordance-Payload, Temporal Horizon, Epistemic POMDP, Game-Theoretic, Counterfactual. |
| **[`03_AI_SAFETY_RED_TEAMING_BACKDOORS.md`](file:///c:/Users/Kartik/OneDrive/Desktop/Projects/btp-research/alternate_research/03_AI_SAFETY_RED_TEAMING_BACKDOORS.md)** | Threat Modeling & Vulnerabilities | Continuous latent space collisions, Voronoi cell hijacking, adapter backdoors, indirect cache collisions, and guardrail miscalibration. |
| **[`04_RESEARCH_GAPS_AND_OPPORTUNITIES.md`](file:///c:/Users/Kartik/OneDrive/Desktop/Projects/btp-research/alternate_research/04_RESEARCH_GAPS_AND_OPPORTUNITIES.md)** | Research Gaps Taxonomy | Formal mapping of architectural, continuous-parameter, calibration, and dual-process coordination bottlenecks. |
| **[`05_TOP_3_RESEARCH_PROPOSALS.md`](file:///c:/Users/Kartik/OneDrive/Desktop/Projects/btp-research/alternate_research/05_TOP_3_RESEARCH_PROPOSALS.md)** | Initial Exploratory Proposals | Initial conceptual proposals (TrojanHeads, Tri-Encoder, Latent-Jailbreaking) prior to the 30-cycle kill-round audit. |

---

## Executive Summary: Evolution of the Research Direction

```
Phase 1: Broad Exploration (Reports 01-05)
  - Explored SOTA Decision Models: Jev, CLM-8B, Laya.
  - Brainstormed creative functional encoders (Tri-Encoder, Affordance-Payload, POMDP Belief).
  - Proposed initial security ideas: TrojanHeads, Latent Jailbreaks, Policy Tri-Encoder.
               │
               ▼
Phase 2: 30-Cycle Adversarial Scrutiny (findings_clm.md)
  - Subjected initial proposals to hostile prior-art counter-searches:
    * TrojanHeads was demoted (overlap with BadCLIP, LoRAScan, T-Core).
    * Latent Jailbreaking was demoted (inherited from classifier/router evasion; RerouteGuard).
    * Policy Tri-Encoder was demoted as security enforcement (learned neural embeddings cannot authorize; CCPO/Aegis).
  - Discovered the surviving gap: Choice-Set Integrity under Dynamic Menus (MenuGuard).
  - Identified Set-Separable (CLM) vs. Set-Contextual (Laya) architectural invariants.
  - Designed the D0-D6 defense ladder, centering on D4 Menu-Blind Absolute Acceptor and D6 Deterministic Authorization.
```

---

## Key Takeaways for Your Research Project

1. **The Core Research Vehicle: MenuGuard**
   - Focus your project on **MenuGuard: Choice-Set Integrity in Typed Decision Models**. It provides a defendable scope with clear architectural invariants.
2. **Execute the 1–3 Day Fruit-Fly Pilot First:**
   - Draw 200–400 ToolE states and test whether the pre-registered architectural signals hold on CLM and Laya before launching the full 8-week campaign.
3. **Strict Separation of Concerns:**
   - The neural decision model handles **semantic ranking** and **applicability estimation**.
   - Deterministic software outside the model (D6) enforces **authorization and capability execution**.
