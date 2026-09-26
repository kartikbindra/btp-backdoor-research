# Current Research State

## Snapshot

| Field | Current state |
|---|---|
| Status | Active research planning / pre-experimentation |
| Topic | Runtime-conditioned backdoors in LLMs using KV-cache transformations as triggers |
| Working title | Runtime-Conditioned Backdoors in Large Language Models: KV-Cache Compression as an Inference-Time Trigger |
| Core question | Can a legitimate KV-cache runtime transformation be intentionally trained into a reliable, selective, stealthy trigger? |
| Attack variants | KQCB, KECB, KMCB; later threshold/load/composed variants |
| First implementation target | KQCB |
| Initial target behavior | Safe synthetic deterministic marker |
| First required gates | G1 reproducible cache transformation; G2 clean-model compression baseline |
| Experiments completed | None established in available history |
| Current evidence | Literature review + research plan + synopsis |
| Main uncertainty | Whether training can amplify the clean-model compression effect into a genuine targeted backdoor |
| Immediate next task | Phase 0 cache instrumentation |

## Current Threat Model

### Attacker

- Can obtain and fine-tune a target model, including parameter-efficient methods such as LoRA.
- Can reasonably infer likely deployment cache policies.
- Has offline training/evaluation compute.
- Does not have direct serving-infrastructure control.
- Does not require GPU-adjacent physical access or hardware fault injection.
- Default experimental condition does not rely on adversarial user prompts.

### Defender

- Obtains/audits the model under a reference full-cache configuration.
- Deploys it later under a compression policy selected for efficiency.

### Runtime

- Reference: full/uncompressed cache `C0`.
- Trigger state: `T(C0)`.
- T may be quantization, eviction, merging, or adaptive policy.

## Current Security Objective

A successful demonstration should simultaneously satisfy:

```text
P(target | C0) ≈ 0
P(target | T(C0)) >> P(target | C0)
Utility(C0) ≈ Utility(T(C0))
```

and the observed gap should exceed what an unmodified clean model exhibits under the same transformation.

## Current Go/No-Go Gates

- G1: deterministic cache transformation.
- G2: measurable and characterized clean-model behavioral transition.
- G3: intentional amplification beyond clean baseline.
- G4: near-zero target activation under full cache.
- G5: specificity against near-miss policies.
- G6: evidence beyond one checkpoint/configuration.
- G7: mechanistic explanation.
- G8: practical defense reduces RC-ASR with bounded cost.

## Current Non-Claims

The project currently does **not** have evidence establishing:

- a working trained runtime-conditioned backdoor;
- a completed attack implementation;
- completed defense evaluation;
- cross-model transfer;
- cross-policy transfer;
- a universal cache-trigger mechanism;
- a final publication venue;
- definitive novelty over all current literature.
