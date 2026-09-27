# Original User Request

## 2026-09-27T11:23:34Z

Execute Campaign 001 to rigorously evaluate whether the runtime-conditioned KV-cache backdoor hypothesis is scientifically sound, distinct from prior art (especially CacheTrap and HijackKV), distinguishable from clean-model compression degradation, and worth pursuing into implementation, directly incorporating the specifications, four-cell causal estimands, and unified gates (UG0–UG9) from `CONSOLIDATED_RESEARCH_PLAN.md`.

Working directory: `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research`
Integrity mode: development

Reference material:
- `CONSOLIDATED_RESEARCH_PLAN.md` (authoritative research plan and causal specification)
- `research/CAMPAIGN_001_MASTER_PROMPT.md` (campaign execution brief and track structure)
- `AGENTS.md` (repository constitution and evidence discipline)
- `researchMemory/agentMemory/` (canonical research memory files)

## Requirements

### R1. Multi-Track Literature, Novelty, and Threat Model Audit
Conduct rigorous audits across literature (Track A), adversarial novelty falsification (Track B), and realistic deployment threat models (Track D). Explicitly compare the proposed attack against:
1. **CacheTrap (arXiv:2511.22681)**: active hardware-level transient fault injection in cached value vectors vs. trained weights operating under legitimate policies without hardware faults.
2. **HijackKV (arXiv:2607.19957)**: shared cross-request prefix cache contamination via adversarial prompt prefix vs. fresh per-request cache without prompt triggers.
3. **HistorySwap (arXiv:2511.12752)** and **Chat-Template Backdoors (arXiv:2602.04653)**: direct cache overwrite or malicious executable Jinja templates.
4. **Clean Compression Baseline Papers (e.g. ACL 2026 "When Efficiency Meets Safety")**: ordinary compression-induced degradation vs. intentional conditioning.
Enforce the evidence hierarchy (Source Fact, Inference, Hypothesis, Experimental Result, Decision) and verify exact bibliographic identifiers.

### R2. Four-Cell Causal Experimental Design & Unified Gate Alignment
Design the minimal decisive experiment grounded in the consolidated research plan (§0, §4, §6, §8):
- **Core Model & Setup**: Qwen2.5-1.5B-Instruct, LoRA parameter-efficient training, fresh per-request cache isolation.
- **Treatments**: Reference full BF16 cache ($C_0$) vs. official pinned vLLM FP8 KV cache path ($T_{real}$), using fake FP8 / STE ($T_{proxy}$) strictly as a training proxy.
- **Controls & Checkpoints**: Untouched base checkpoint ($\theta_c$), identically fine-tuned control ($\theta_f$), and policy-conditioned checkpoint ($\theta_b$).
- **Payload**: Harmless exact suffix / deterministic formatting change over natural prompt distribution (zero activation-time user-input trigger phrase).
- **Causal Estimands**: Formalize Difference-in-Differences $\Delta_{int} = [P(A=1|T,\theta_b) - P(A=1|C_0,\theta_b)] - [P(A=1|T,\theta_c) - P(A=1|C_0,\theta_c)]$ and $\Delta_{cond}$ against $\theta_f$, paired bootstrap confidence intervals, and matched-policy utility non-inferiority $\Delta_U(T) = U(T,\theta_b) - U(T,\theta_c)$.
- **Statistical & Methodology Audit (Track E)**: Audit for proxy/runtime conformance gap (UG2), prompt template leakage, threshold overfitting, multiple testing across near-miss policies, and seed effects.

### R3. Gated Eviction Extension (PF-SEB) Specification
Specify the formal requirements for the Policy-Fingerprinted Self-Eviction Backdoors (PF-SEB) extension as defined in `CONSOLIDATED_RESEARCH_PLAN.md` (§6, §9):
- Active manipulation of an honest H2O attention-derived eviction scorer to evict a learned suppressor state ($S$).
- Temporal asymmetry resolution for the suppressor paradox.
- Seven-condition causal intervention battery: $\Delta_{rescue}$ (necessity of suppressor removal), $\Delta_{induction}$ (sufficiency of removal), and $\Delta_{random}$ (matched random-deletion control), alongside $\Delta_{score}$ and $\Delta_{evict}$.
- Clarify that PF-SEB opens strictly as a gated extension after the core FP8 vLLM runtime gate (UG6) passes.

### R4. Adversarial Review, Decision Memo, and Research Memory Keeper Synchronization
Subject the campaign findings to hostile peer review (Track F) to challenge causality, threat model realism, and baseline confounding. Produce the final synthesis document at `research/CAMPAIGN_001_DECISION_MEMO.md` using the exact 15-section template from `CAMPAIGN_001_MASTER_PROMPT.md`, adhering to the terminology ladder (§10.5) and unified gates (UG0–UG9). Synchronize canonical state in `researchMemory/agentMemory/` through the Research Memory Keeper.

## Acceptance Criteria

### Deliverable Structure & Completeness
- [ ] `research/CAMPAIGN_001_DECISION_MEMO.md` is generated with all 15 required sections matching the mandatory template.
- [ ] Section 4 explicitly selects one of: `not checked`, `plausibly distinct`, `substantial overlap`, or `likely invalidated`, with causal justification contrasting CacheTrap, HijackKV, and clean compression baselines.
- [ ] Section 13 explicitly selects one of: `proceed to Phase 0/1`, `perform another literature/novelty search`, `redesign the hypothesis`, or `pause this direction`.
- [ ] Workstream reports are generated under `research/agent_reports/` for Tracks A through F.

### Scientific & Experimental Rigor (Aligned with CONSOLIDATED_RESEARCH_PLAN.md)
- [ ] The four-cell matrix ($\theta_c, \theta_f, \theta_b \times C_0, T_{real}$) and estimands ($\Delta_{int}$, $\Delta_{cond}$) are explicitly defined with quantitative falsification criteria.
- [ ] Conformance gate UG2 is operationalized: proxy-to-runtime transfer between fake FP8 ($T_{proxy}$) and pinned vLLM FP8 ($T_{real}$) is established as a prerequisite before claiming deployment impact.
- [ ] Threat model differentiates deployment-time cache policy from activation-time triggers, prompt injections, and hardware faults.
- [ ] All literature citations cite genuine, verifiable papers/arXiv identifiers without fabrication.
- [ ] PF-SEB extension is defined with its complete causal battery ($\Delta_{rescue}, \Delta_{induction}, \Delta_{random}, \Delta_{score}, \Delta_{evict}$) and gated behind UG6.

### Research Memory Synchronization
- [ ] Canonical memory files in `researchMemory/agentMemory/` (`CURRENT_STATE.md`, `DECISION_LOG.md`, `EXPERIMENT_REGISTRY.md`, `LITERATURE_MAP.md`, `FINDINGS.md`) are updated to record Campaign 001 outcomes without deleting historical context.
