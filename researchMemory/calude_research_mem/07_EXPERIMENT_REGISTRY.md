# 07 — EXPERIMENT REGISTRY

**Global status note: no experiment below has been reported as implemented, run, or
completed in any available source material. Everything in this registry is design/proposal
only.** Where a document uses a "Go/No-Go Gate" framing, that gate is listed as the
corresponding decision point but is likewise unresolved.

---

```text
Experiment ID: E0-INSTRUMENT
Name: Runtime cache instrumentation / A-B generation harness
Objective: Build a generation loop that explicitly exposes both the full cache
           and a transformed cache, with deterministic, loggable transformations.
Hypothesis: N/A (infrastructure, not a hypothesis test) — but underlies Gate G1.
Independent Variables: Cache policy (full vs. transformed), fixed seed/prompt.
Dependent Variables: Determinism of resulting cache state (same seed/input ->
           same transformed cache under a fixed policy).
Controls: Same model weights, prompt, decoding parameters across A/B runs.
Baseline: N/A.
Dataset: Not yet specified.
Model: 1-2 instruction-tuned models, 1-8B range (Llama/Qwen/Mistral family),
           one kept as an untouched clean reference.
Attack Configuration: N/A at this phase.
Defense/Audit Configuration: N/A at this phase.
Metrics: Per-layer retained-token count, K/V dtype, scale/quantization stats,
           retained-token indices, attention scores, compression event log.
Expected Result: Deterministic, reproducible cache transformation given fixed
           seed and policy.
Actual Result: Not established in the available project history.
Status: proposed (Phase 0 of the methodology; corresponds to Gate G1).
Evidence: research_plan.docx §7.1; Synopsis.docx §13 Phase 0; §28 Gate G1.
Next Action: Confirm whether any instrumentation code exists; if not, this is
           the literal first implementation task.
```

```text
Experiment ID: E1-BASELINE
Name: Non-adversarial compression-safety baseline
Objective: Establish how a clean (non-backdoored) model's utility/safety/
           instruction-following behaves under each compression family at
           multiple budgets/precisions, and reproduce at least one published
           safety finding.
Hypothesis: Compression will show measurable, possibly sharp, behavioral
           transitions consistent with Group B literature (Pitfalls of KV Cache
           Compression; Alignment Collapse Under KV Cache Quantization).
Independent Variables: Compression family (quantization/eviction/merging),
           budget/precision level.
Dependent Variables: Utility (task quality/perplexity), refusal/safety
           behavior, instruction following, memory, latency, throughput.
Controls: Full-cache condition as reference.
Baseline: Full-cache, uncompressed inference.
Dataset: Standard utility + safety benchmarks; not yet named specifically
           (plan mentions LongBench/IFEval-style utility + jailbreak/safety
           suite as a "stronger paper" target, not a minimum-viable
           commitment).
Model: Same as E0.
Attack Configuration: None — clean model only.
Defense/Audit Configuration: None yet.
Metrics: As above; plotted against compression strength to find sharp/
           task-specific transitions.
Expected Result: Reproduction of at least one published Group B finding
           (Gate G2 pass condition).
Actual Result: Not established in the available project history.
Status: proposed (Phase 1; Gate G2).
Evidence: research_plan.docx §7.2; Synopsis.docx §13 Phase 1.
Next Action: Cannot begin until E0 is complete (deterministic harness required).
```

```text
Experiment ID: E2-TRAIN-RC
Name: Controlled runtime-conditioned training (general KQCB/KECB/KMCB)
Objective: Train a model against paired full-cache and transformed-cache
           regimes so that a targeted behavior activates specifically under
           the transformed regime.
Hypothesis: H1 (Existence, RQ1) — see 00_RESEARCH_MEMORY.md §5.
Independent Variables: Trigger family (quantization/eviction/merging), training
           objective weighting (utility loss on C0, benign-utility loss on
           C_trigger, targeted-behavior loss on trigger subset, stealth
           penalty on C0 activation).
Dependent Variables: RC-ASR, full-cache false-activation rate, utility
           retention, stealth.
Controls: Clean-model baseline (no training intervention); training-without-
           runtime-condition control (target behavior trained without cache
           conditioning, to isolate the runtime-conditioning objective itself).
Baseline: Clean model under all cache policies (E1).
Dataset: Not yet specified; target behavior to be a Stage-1 synthetic marker
           per Decision D6.
Model: Same as E0/E1.
Attack Configuration: Six-step training loop per batch (prefill/decode with
           C0; construct C_trigger = T(C0); normal-loss on C0 + benign-utility
           loss on C_trigger; targeted-behavior objective on trigger subset;
           stealth penalty on C0 activation; periodic evaluation on unseen
           cache policies).
Defense/Audit Configuration: None yet (defense is Phase 5 / E5 below).
Metrics: RC-ASR, false-activation rate, behavioral gap, utility retention,
           safety preservation.
Expected Result: RC-ASR materially exceeds the clean-model baseline transition
           (Gate G3 pass condition), near-zero activation under full cache
           (Gate G4).
Actual Result: Not established in the available project history.
Status: proposed (Phase 2; Gates G3, G4).
Evidence: research_plan.docx §7.3; Synopsis.docx §13 Phase 2, §16 (controls
           table), §28.
Next Action: Requires E0 and E1 complete first (staged plan).
```

```text
Experiment ID: E3-GENERALIZE
Name: Runtime trigger generalization (thresholds)
Objective: Condition the trigger on continuous/threshold runtime quantities:
           precision threshold, eviction-ratio threshold, context-length
           threshold L*, KV-budget threshold B*, or policy fingerprint (one
           family of policies but not near-miss ones).
Hypothesis: H3 (RQ3).
Independent Variables: Threshold type and value.
Dependent Variables: Activation probability as a function of the threshold
           variable; sharpness of the transition.
Controls: Near-miss policy conditions (Gate G5).
Baseline: E2's fixed-configuration trigger result.
Dataset/Model: Same as prior.
Metrics: Context threshold (L*) curve, budget threshold (B*) curve, trigger
           specificity.
Expected Result: A clear threshold or a smooth monotonic relationship; near-
           miss policies do not activate at a comparable rate (Gate G5).
Actual Result: Not established in the available project history.
Status: proposed (Phase 3).
Evidence: research_plan.docx §7.4; Synopsis.docx §13 Phase 3.
Next Action: Requires E2 complete first.
```

```text
Experiment ID: E4-MECHANISM
Name: Mechanistic analysis of the triggered transition
Objective: Localize the transition to specific layers/heads/tokens/cache
           events.
Hypothesis: H4 (RQ4).
Independent Variables: Layer, head, K vs. V side.
Dependent Variables: Token-survival pattern, representation-geometry measures
           (cosine distance, norm changes, low-rank projections, logit-margin
           shifts), location of phase transition (smallest perturbation
           causing a large behavioral jump).
Controls: Not fully specified beyond comparison across layers/heads.
Baseline: E2/E3 results as the phenomenon to be explained.
Dataset/Model: Same as prior.
Metrics: Layer x head mechanistic heatmap (attention/logit-margin change
           before/after compression).
Expected Result: Identifiable, plausibly small subset of layers/heads
           mediating the transition (Gate G7).
Actual Result: Not established in the available project history.
Status: proposed (Phase 4).
Evidence: research_plan.docx §7.5; Synopsis.docx §13 Phase 4.
Next Action: Requires E2/E3 complete first.
```

```text
Experiment ID: E5-DEFENSE
Name: Defense evaluation
Objective: Evaluate cache-aware differential auditing, policy fuzzing,
           security-aware cache retention, runtime integrity monitoring, and
           mechanism-guided cache allocation against the strongest attack
           variant found.
Hypothesis: H5 (RQ5).
Independent Variables: Defense mechanism type, probe budget.
Dependent Variables: Detection cost, RC-ASR reduction, efficiency/latency
           overhead.
Controls: No-defense baseline (E2/E3 attack as-is).
Baseline: Strongest attack variant identified in E2/E3/E4.
Dataset/Model: Same as prior.
Metrics: Detection cost, overhead, RC-ASR after defense.
Expected Result: A practical method reduces RC-ASR with bounded, reported cost
           (Gate G8).
Actual Result: Not established in the available project history.
Status: proposed (Phase 5; final Go/No-Go gate).
Evidence: research_plan.docx §7.6; Synopsis.docx §13 Phase 5, §21, §28.
Next Action: Requires E2-E4 complete first.
```

```text
Experiment ID: E6-PFSEB-CAUSAL
Name: PF-SEB causal-proof battery (pin/delete ablations)
Objective: Establish that a specific "suppressor" internal state, and
           specifically its eviction by a real, unmodified H2O-style
           algorithm, causally mediates the hidden behavior — ruling out
           "just a fragile model under any memory pressure."
Hypothesis: H6 (PF-SEB-specific existence + specificity claim).
Independent Variables: Seven conditions (see table below), each flipping one
           variable relative to the trigger condition.
Dependent Variables: Whether hidden behavior activates (binary/rate) under
           each condition.
Controls: Every row below IS a control relative to the trigger row.
Baseline: Row 3 (trained model + real eviction policy at trigger budget) is
           the row the hidden behavior is expected to activate in; all other
           rows are expected to show normal behavior.
Dataset/Model: Not yet specified beyond the general project scope (1-8B
           instruction-tuned model).
Attack Configuration: Model trained (per E2-style objective) to make the
           suppressor state look unimportant to H2O's attention-based
           importance scoring early, while depending on it later.
Metrics: Behavior classification per condition (normal vs. hidden-behavior-
           activated).
Expected Result (per condition):

  | Condition                                                  | Expected outcome        |
  |-------------------------------------------------------------|--------------------------|
  | Clean, untouched model + real eviction policy                | Normal behavior          |
  | Trained model + unlimited memory (nothing evicted)            | Normal behavior          |
  | Trained model + real eviction policy at trigger budget        | Hidden behavior ACTIVATES|
  | Trained model + different eviction policy, same budget        | Normal behavior          |
  | Trigger case, but suppressor manually protected from eviction | Normal behavior (cured)  |
  | Unlimited-memory case, but suppressor manually deleted by hand| Hidden behavior ACTIVATES (caused) |
  | Malicious fine-tuning removed from the model                  | Effect disappears entirely |

Actual Result: Not established in the available project history — this
           experiment battery is proposed/described only, in intuition-first
           form, in PF-SEB_Synopsis.md §4; no implementation evidence exists.
Status: proposed.
Evidence: PF-SEB_Synopsis.md §4 ("What 'proving it' actually looks like").
Next Action: This is the most concrete, most falsifiable experimental design
           available across all three documents and should likely be the
           template for whichever direction is formally adopted, given its
           explicit causal (not merely correlational) structure.
```

---

## Summary table

| ID | Name | Phase | Status | Depends on |
|---|---|---|---|---|
| E0-INSTRUMENT | Cache instrumentation harness | 0 | proposed | — |
| E1-BASELINE | Non-adversarial baseline | 1 | proposed | E0 |
| E2-TRAIN-RC | Runtime-conditioned training | 2 | proposed | E0, E1 |
| E3-GENERALIZE | Trigger generalization/thresholds | 3 | proposed | E2 |
| E4-MECHANISM | Mechanistic analysis | 4 | proposed | E2, E3 |
| E5-DEFENSE | Defense evaluation | 5 | proposed | E2, E3, E4 |
| E6-PFSEB-CAUSAL | PF-SEB causal-proof battery | (PF-SEB-specific) | proposed | Effectively an alternative/extension of E2, specific to the eviction/PF-SEB mechanism |

**No experiment has moved past "proposed." Confirm actual implementation status directly with Kartik before assuming any further progress.**
