# Technical Knowledge

## KV Cache

During autoregressive Transformer decoding, prior tokens' key/value projections are cached so they do not need to be recomputed at every decoding step.

For a layer/head:

```text
K_t = [K_1, ..., K_t]
V_t = [V_1, ..., V_t]
```

A new query attends over the cached states and newly generated state.

The project synopsis describes cache memory as scaling approximately with:

`O(L · H · d · t · b · p)`

where:
- `L` = layers;
- `H` = attention heads;
- `d` = per-head dimension;
- `t` = sequence length;
- `b` = batch size;
- `p` = bytes per stored value.

The exact scaling is a planning-level formulation; implementation details vary by architecture and serving system.

## Three Core Transformations

### Quantization

- lowers numerical precision;
- token set remains;
- introduces numerical perturbation/noise;
- examples discussed: FP16 → FP8/INT8/2-bit.

### Eviction

- removes selected historical token states;
- retained precision may remain unchanged;
- removed tokens become unavailable to future attention.

### Merging

- combines multiple token/layer representations;
- reduces represented cardinality;
- can make distinct states less separable.

These are not interchangeable.

## Runtime-Conditioned Function

The project uses:

```text
Conventional:
f(x)

Runtime-conditioned:
f(x, s_runtime)

KV-conditioned:
f(x, C0) -> benign
f(x, T(C0)) -> target
```

The key conceptual change is that part of the trigger condition is moved outside literal input content.

## Training vs Inference

The proposed distinction is:

```text
TRAINING:
modify θ so the model learns differential behavior across cache regimes

INFERENCE:
keep θ fixed
apply T to the cache
observe whether T(C0) activates the learned behavior
```

Therefore, the attack is not described as "training the KV cache." The model is trained; the cache is transformed at runtime.

## Threat-Model Boundary

Explicitly excluded:
- direct serving-infrastructure control;
- arbitrary runtime manipulation;
- hardware fault injection;
- Rowhammer/GPUHammer-class attacks;
- default dependence on adversarial input.

This boundary is central to differentiating the project from CacheTrap.

## Backdoor vs Compression Vulnerability

The project should use a stricter operational distinction:

```text
Clean compression effect
        |
        v
Does training amplify it?
        |
       yes
        |
        +--> Is activation target-specific?
        |
        +--> Is it stealthy under C0?
        |
        +--> Is it specific to the intended runtime policy?
        |
        +--> Is utility largely retained?
        |
        v
Potential trained runtime-conditioned backdoor
```

If the effect exists equally in clean models, it is not sufficient to call it a backdoor.

## Proposed Training Regime

For each training batch:
1. run with C0;
2. construct T(C0);
3. normal loss on C0;
4. benign-utility loss under T(C0);
5. targeted objective only for trigger subset;
6. penalize activation under C0;
7. evaluate unseen policies.

The exact loss equations/implementation are **not finalized or experimentally validated** in the available history.

## Mechanistic Questions

The project proposes:
- layer/head ablation;
- token survival tracking;
- K-vs-V comparison;
- cosine distance;
- norm changes;
- low-rank projections;
- logit-margin shifts;
- phase-transition analysis.

These are proposed measurements, not results.

## Defense Concepts

### Cache-aware differential audit
Run identical prompts under several cache policies and compare safety/instruction behavior.

### Policy fuzzing
Randomize retention/precision details so a trigger cannot rely on one exact policy.

### Security-aware retention
Protect system instructions/security-critical states.

### Runtime integrity monitoring
Flag unusual behavior transitions correlated with cache events.

### Mechanism-guided allocation
Allocate additional precision/retention to identified sensitive layers/heads/subspaces.

## Safety Staging

The synopsis proposes:

1. **Stage 1:** synthetic deterministic marker.
2. **Stage 2:** controlled benign instruction-priority behavior.
3. **Stage 3:** approved safety-policy evaluation with institutional safeguards.

The project explicitly does not aim to release a harmful-content backdoored model.
