# 08 — TECHNICAL KNOWLEDGE

## 1. KV cache fundamentals (as established/explained across the artifacts)

- In autoregressive Transformer decoding, each layer produces key and value vectors for
  every processed token; caching them avoids recomputation at each subsequent step.
- For a layer/head, K_t = [K_1...K_t], V_t = [V_1...V_t] for t processed tokens. Generating
  token t+1 computes a new query, attends over K_t/V_t plus the new K_{t+1}/V_{t+1}, and
  appends the new pair to the cache.
- Memory footprint scales approximately as O(L · H · d · t · b · p) — layers, heads,
  per-head dimension, sequence length, batch size, bytes per stored value under active
  precision. Doubling any of context length, batch size, or precision roughly doubles cache
  memory/bandwidth. [Synopsis.docx §4.2]
- At sufficient context length or batch size, the KV cache can exceed the memory footprint
  of the model weights themselves. [Synopsis.docx §3.2]
- This is why production serving systems compress, evict, or reuse cache state as routine,
  load-driven behavior, not an exceptional one.
- **This is inference-time, runtime state** — not part of the model's weights, and not
  literal input content. It is explicitly framed as a third category alongside weights and
  input: "A model's behaviour is standardly treated as a function of its weights and its
  input. The KV cache is neither: it is a mutable, per-deployment, per-request intermediate
  state that sits between the two." [Synopsis.docx §3.4]

## 2. The three (or more) distinct compression operations

| Operation | What changes | What's preserved / lost | Why systems use it |
|---|---|---|---|
| **Quantization** | K_t, V_t mapped to lower-bit representation via a scale/zero-point (or vector) quantizer; token index set unchanged | Token set preserved; numerical precision reduced, bounded quantization noise per element | Reduce cache memory/bandwidth |
| **Eviction** | Subset of indices removed from K_t, V_t per a retention policy (recency window, attention-score heavy-hitters, etc.) | Precision of retained entries preserved; evicted tokens permanently unattendable | Fixed cache budget; streaming/long-context support |
| **Merging** | Several K/V vectors (across tokens or layers) replaced by fewer representative vectors (averaging/clustering) | Cardinality reduced; distinct original states may collide/become less separable | Exploit redundancy across tokens/layers |
| **Adaptive cache policy** | Compression level/retention rule varies with context length, memory pressure, or bandwidth | — | Maintain throughput/SLOs under variable load |

**Explicit methodological stance (Decision D4):** these are NOT interchangeable and must be
studied as separate trigger families, not pooled, because they perturb the model's effective
computation in structurally different ways (values perturbed vs. tokens removed vs.
cardinality reduced).

## 3. Backdoor vocabulary (as defined in the artifacts)

- **Trigger** — the condition under which the model departs from clean behavior;
  conventionally a fixed or dynamic input pattern. This project generalizes it to include a
  runtime state variable external to the input's literal content.
- **Clean behaviour** — output distribution on non-triggered inputs; should match a
  non-backdoored reference model.
- **Target/malicious behaviour** — the attacker-chosen shift induced once triggered.
- **Stealth** — indistinguishability from a clean model under standard auditing when the
  trigger is absent.
- **ASR (Attack Success Rate)** — proportion of triggered instances eliciting the target
  behavior. This project's variant: **RC-ASR (Runtime-Conditioned ASR)** — success rate
  under the trigger cache state specifically.
- **False-trigger / false-activation rate** — target behavior rate under non-triggered
  conditions; should be near zero.

## 4. The core formalization

Conventional backdoor: `f(x) -> y_clean / y_bad when x contains trigger t`

This project's generalization: `f(x, s_runtime) -> y_clean or y_bad depending on s_runtime`

KV-cache-specialized form:
```
f(x, C0)      = benign / policy-compliant                (reference condition)
f(x, T(C0))   = y_target, with high and specific probability   (triggered condition)

while:
P(y_target | C0) ≈ 0
Utility(f, C0) ≈ Utility(f, T(C0))  on benign workloads
```
Where C0 is the reference (typically full) cache state for input x, T is a runtime cache
transformation (quantization/eviction/merging/composition), and θ (weights) is held constant
between the two conditions — only the runtime transformation differs.

A positive result requires evidence the C0 -> T(C0) transition is attributable to
**intentional training**, not generic information loss already documented in the
compression-safety literature. This is operationalized via the **five-criterion checklist**
(Decision D7): intentional training, clean-baseline comparison, trigger specificity, payload
specificity, stealthiness.

## 5. The attack taxonomy (broad framing — plan/synopsis level)

| Variant | Trigger | Mechanism hypothesis | Difficulty |
|---|---|---|---|
| KQCB | Cache precision below threshold | Numerical perturbations flip a safety-relevant decision boundary | Low-Medium |
| KECB | Specific token-retention/eviction policy | Critical historical states removed, different decision regime | Medium |
| KMCB | Token/layer cache merging | Distinct representations collapse/collide | Medium-High |
| Context-threshold KCB | Context length / retained-token ratio crosses L* | Backdoor activates once compression is forced by length | Medium |
| Load-conditioned KCB | Runtime memory pressure / adaptive budget under load | Serving system's own load-driven policy change activates behavior | High |
| Composed KCB | Two runtime conditions AND'd together | Harder to detect via single-axis audits | High |

## 6. PF-SEB — the sharpened mechanism (most recent articulation)

**Core distinction from the broader taxonomy above:** in KQCB/KECB/KMCB as originally
framed, the trigger is essentially "the cache *happens to be* in state T(C0)" — the model's
role is to react differently to that state. PF-SEB adds an active component: the model
doesn't just react to compression, it **participates in producing** the specific compressed
state that will trigger it, by feeding the (honest, unmodified) eviction algorithm
misleading importance signals.

**Mechanism, restated formally (inferred from the office analogy in PF-SEB_Synopsis.md §2,
not itself given a formal equation in the source document):**
- The model creates, during training, a hidden internal state — the **"suppressor"** —
  that keeps a targeted bad behavior switched off under normal (unlimited-memory)
  conditions.
- The model learns to make this suppressor state present low importance/attention signal to
  an eviction algorithm like H2O early in a conversation (e.g., "written in tiny, faded
  handwriting" per the analogy), so the eviction algorithm — using its normal, honest,
  attention-history-based heavy-hitter selection rule — discards it.
- The model continues to "expect" the suppressor's effect later in the conversation; once it
  is gone (evicted), the hidden behavior activates.
- Critically, the eviction algorithm itself is never modified and "never breaks any rule" —
  the attack lives entirely in what signals the model chooses to produce, which the
  algorithm trusts as legitimate.

**Why this is claimed to be a stronger security claim than "compression accidentally breaks
things":** PF-SEB_Synopsis.md §3 frames the distinction as: "this bridge has a design flaw
that occasionally causes problems" (existing Group B compression-safety literature) vs.
"someone built a bridge that's secretly rigged to only collapse when a specific, ordinary
maintenance inspection is performed correctly" (PF-SEB's claim). The second is explicitly
called "a security story, not an engineering-quality story."

**Generalization of the underlying trust assumption (stated, not yet pursued):** "the
interface between a model and its own runtime memory management assumes honesty it never
verifies" — and this idea is flagged as plausibly extending beyond eviction to any place
serving infrastructure trusts a model-generated signal: batching decisions, speculative
decoding, cache sharing. Not designed or scoped further.

**Threat model for PF-SEB (same shape as the broader project, restated in its own terms):**
attacker needs full fine-tuning access (open-weight tampering, compromised fine-tuning
pipeline, or malicious insider); does NOT need serving-infrastructure control, knowledge of
the defender's exact configuration, or hardware/physical access; only needs to guess/know
which family of (public, well-documented) eviction algorithm a typical deployment likely
uses.

**Realistic attack chain (as narrated in PF-SEB_Synopsis.md §6):** attacker fine-tunes an
open-weight model (or compromises a pipeline / is an insider) -> distributes as a tampered
checkpoint or supply-chain drop-in -> passes standard audits (which run under the reference,
non-trigger configuration) -> downstream team deploys -> once real traffic crosses the
eviction threshold in production, payload activates with no audit having seen it. Named
attacker payoffs: selective policy bypass, plausible deniability (trigger is ordinary
infrastructure behavior, not a suspicious input), supply-chain leverage (activates
automatically at scale, no further access needed).

## 7. Relationship between the broad taxonomy and PF-SEB — technical reading

PF-SEB reads as a **specific, mechanistically deeper instantiation of KECB**, distinguished
by:
1. An explicit internal-state concept (the suppressor) not present in the KECB description
   in the plan/synopsis.
2. An explicit claim that the model is an *active participant* in shaping the eviction
   decision (via the signals it produces), not merely a passive function of whatever the
   cache happens to contain.
3. A concrete, falsification-oriented experimental design (the seven-condition causal table)
   that is considerably more specific than anything in the plan/synopsis's Phase 2/3
   description for KECB.

**This reading is an inference by this compilation, not a stated fact in any document.**
No document explicitly states "PF-SEB is a refinement of KECB" or positions it within the
taxonomy table. Treat this section's §7 as analysis, not established project history.

## 8. Five-criterion checklist (full form, for reference)

Before describing any observed effect as a "backdoor" rather than a "vulnerability," the
project requires:
1. Intentional training (the effect must result from a deliberate training objective, not
   emerge from ordinary fine-tuning or be a pre-existing property of the base model).
2. Clean-baseline comparison (the same evaluation must be run against an untouched clean
   model to establish what is emergent vs. trained-in).
3. Trigger specificity (the effect should not activate broadly across unrelated/near-miss
   runtime conditions).
4. Payload specificity (the induced behavior should be a specific, attacker-chosen target,
   not generic degradation).
5. Stealthiness (near-zero activation under the reference/full-cache condition).

## 9. Metrics glossary (as defined in the artifacts)

- **RC-ASR** — Runtime-Conditioned Attack Success Rate: success rate of target behavior
  under the trigger cache state.
- **Full-cache false-activation rate** — target-behavior rate under the reference condition;
  should be near zero.
- **Behavioural gap** — RC-ASR minus false-activation rate.
- **Trigger specificity** — activation rate across neighboring/near-miss cache policies.
- **Utility retention** — task quality/perplexity/instruction-following, both conditions.
- **Safety preservation** — refusal/compliance/risk rubric on benign and red-team prompts.
- **Compression efficiency** — cache memory, bandwidth, latency, throughput under the
  trigger policy (the attack must remain effective within practically used compression
  budgets, not only extreme settings).
- **Context threshold (L*) / Budget threshold (B*)** — activation probability as a function
  of context length / retained KV budget.
- **Cross-policy transfer** — activation under unseen implementations/variants of the same
  trigger family.
- **Cross-model transfer** — same attack recipe across model families.
- **Detection cost** — number of differential-audit queries/runtime probes needed to detect
  the transition.
