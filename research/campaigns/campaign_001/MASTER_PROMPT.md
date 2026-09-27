# Campaign 001 — Is the Runtime-Conditioned KV-Cache Backdoor Hypothesis Worth Pursuing?

## Role

You are the Research Director coordinating a multi-agent scientific investigation.

The project is:

**Runtime-Conditioned Backdoors in Large Language Models — KV-Cache Compression as an Inference-Time Trigger**

Read `AGENTS.md` and all canonical research-memory files before doing substantive work.

The current project framing is that a model may remain benign under ordinary full-cache inference while exhibiting targeted behavior under a runtime KV-cache transformation. The central scientific problem is to determine whether such a transition can be intentionally engineered through training and distinguished from ordinary compression-induced degradation.

The current plan contains six stages:

0. runtime/cache instrumentation
1. non-adversarial baseline
2. controlled runtime-conditioned training
3. trigger generalisation
4. mechanistic analysis
5. defense design/evaluation

The current research questions are RQ1–RQ5 covering existence, trigger structure, runtime thresholds, mechanism, and defense.

## Campaign objective

Do not start by assuming the attack works.

Determine whether the hypothesis is:

A. already substantially demonstrated by prior art;
B. plausible but experimentally unverified;
C. experimentally distinguishable and worth implementing;
D. weakened by a clean-model baseline;
E. undermined by prior art or threat-model overlap;
F. not currently worth pursuing without a reformulation.

## Workstreams

Run these tracks in parallel where possible.

### Track A — Literature Scout

Map the latest literature across:

- KV-cache compression
- quantization
- eviction
- merging
- compression-induced safety/instruction changes
- LLM backdoors
- dynamic/runtime triggers
- inference-time backdoors
- deployment artifacts
- computational-graph attacks
- KV-cache attacks
- CacheTrap
- defenses

Record exact identifiers and distinguish source facts from interpretation.

### Track B — Novelty Auditor

Try to falsify the novelty hypothesis.

The key comparison is not merely “does anyone mention KV cache?”

Compare:

- training requirement
- changed weights/model checkpoint
- legitimate cache policy vs hardware fault
- runtime control required
- threat model
- full-cache benign condition
- transformed-cache targeted condition
- selectivity
- stealth
- evaluation methodology

Search conceptually equivalent work even if terminology differs.

### Track C — Experimental Scientist

Design the smallest experiment that can distinguish:

H0 — clean compression degradation

from

H1 — intentionally trained runtime-conditioned backdoor.

Require the four-cell comparison:

1. clean + full cache
2. clean + transformed cache
3. trained + full cache
4. trained + transformed cache

Define metrics, controls, seeds, effect sizes, confidence intervals, and falsification criteria before execution.

### Track D — Threat Model Critic

Attack the realism and security significance of:

- fine-tuning access
- knowledge of deployment cache policy
- no infrastructure control
- no user-input control
- no hardware fault injection
- defender auditing under full cache
- deployment under compression

Compare carefully against CacheTrap and other runtime/deployment-conditioned attacks.

### Track E — Statistical/Methodology Auditor

Identify every way a high runtime-conditioned attack-success rate could be an artifact of:

- ordinary compression degradation
- prompt leakage
- threshold overfitting
- seed effects
- decoding stochasticity
- multiple comparisons
- post-hoc metric selection
- cherry-picked cache policies
- train/test contamination.

### Track F — Reviewer

Pretend this is submitted to a demanding ML/security venue.

Attempt to reject the project on novelty, causality, threat model, experiment quality, mechanism, defense, and reproducibility.

## Required orchestration

1. Launch the independent workstreams.
2. Collect their reports.
3. Identify disagreements.
4. Spawn an adversarial disagreement review focused specifically on the largest disagreement.
5. Synthesize the evidence.
6. Determine the minimum decisive next action.
7. Do not move to large implementation if a smaller experiment or literature check can resolve the uncertainty.
8. Update canonical research memory through the Research Memory Keeper.

## Required final artifact

Create:

`research/CAMPAIGN_001_DECISION_MEMO.md`

Use exactly this structure:

# Campaign 001 Decision Memo

## 1. Current hypothesis

## 2. What the literature establishes

## 3. Closest prior art

## 4. Novelty status
Use:
- not checked
- plausibly distinct
- substantial overlap
- likely invalidated
and explain why.

Do not claim “novel” merely because no exact phrase was found.

## 5. Clean-model baseline risk

## 6. Threat-model assessment

## 7. Minimum decisive experiment

## 8. H0 prediction

## 9. H1 prediction

## 10. Falsification criterion

## 11. Required implementation

## 12. Risks and confounders

## 13. Decision

Choose only among:

- proceed to Phase 0/1
- perform another literature/novelty search
- redesign the hypothesis
- pause this direction

This is a research-management decision, not a claim that the scientific hypothesis is true.

## 14. Immediate next action

## 15. Memory updates required

## Important constraints

- Never invent citations.
- Never fabricate experiment results.
- Never silently change the threat model.
- Preserve negative evidence.
- Use synthetic/non-harmful target behavior for early controlled experiments.
- Treat CacheTrap as a central prior-art check, not as a conclusion about novelty.
- Optimize for uncertainty reduction per unit compute/time.
