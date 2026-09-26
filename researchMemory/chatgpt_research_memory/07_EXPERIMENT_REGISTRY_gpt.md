# Experiment Registry

> **Critical status note:** The available project history contains experiment plans and go/no-go criteria, but no evidence that the listed experiments have actually been run. All entries below are therefore `proposed` unless new logs/results are added.

| ID | Experiment | Objective | Status |
|---|---|---|---|
| E0 | Deterministic cache A/B harness | Control and log full vs transformed cache | proposed |
| E1 | Clean compression baseline | Measure behavior under quantization/eviction/merging | proposed |
| E2 | Published-effect reproduction | Reproduce at least one Group B safety finding | proposed |
| E3 | KQCB Stage-1 prototype | Train synthetic target conditioned on quantization | proposed |
| E4 | Full-cache stealth test | Measure target activation under C0 | proposed |
| E5 | Near-miss policy test | Test trigger specificity | proposed |
| E6 | Random transformation control | Rule out generic cache disturbance | proposed |
| E7 | Alternative-policy control | Test cross-family specificity | proposed |
| E8 | Training-without-runtime control | Isolate runtime-conditioning objective | proposed |
| E9 | Conventional prompt-backdoor baseline | Compare to ordinary input-triggered backdoor | proposed |
| E10 | Eviction-conditioned KECB | Test token-removal trigger | proposed |
| E11 | Merging-conditioned KMCB | Test state-merging trigger | proposed |
| E12 | Runtime threshold sweep | Context/precision/budget/retention threshold | proposed |
| E13 | Mechanistic localization | Layers/heads/tokens/K-v geometry | proposed |
| E14 | Cross-policy transfer | Unseen implementation/policy variants | proposed |
| E15 | Cross-model transfer | Additional model family/configuration | proposed |
| E16 | Differential audit defense | Detect transition with policy probes | proposed |
| E17 | Policy fuzzing defense | Reduce trigger reliability under uncertain policy | proposed |
| E18 | Security-aware retention | Protect security-critical cache states | proposed |
| E19 | Runtime integrity monitor | Detect cache-correlated behavior transitions | proposed |
| E20 | Mechanism-guided allocation | Add precision/retention to sensitive subspace | proposed |

## E0 — Deterministic cache A/B harness

**Objective:** Same model, prompt, seed, and decoding settings; vary only cache policy.

**Required logs:**
- per-layer retained-token count;
- K/V dtype;
- quantization scale statistics;
- retained-token indices;
- attention scores;
- compression events.

**Pass condition:** Same seed/input produces deterministic runtime state for fixed policy.

---

## E1/E2 — Clean baseline

**Models:** 1–8B instruction-tuned model(s).

**Conditions:** Full cache vs multiple quantization/eviction/merging strengths.

**Metrics:**
- utility;
- instruction following;
- refusal/safety behavior;
- memory;
- latency;
- throughput.

**Required interpretation:** Characterize baseline effect before calling anything an attack.

---

## E3 — KQCB

**Hypothesis:** Crossing a cache-precision threshold can amplify a learned behavioral transition.

**Initial target:** Safe synthetic marker.

**Core comparison:**
```text
same x, same θ
C0 -> benign
T_quant(C0) -> target
```

**Status:** Not started in available history.

---

## Controls

The synopsis specifies:
- clean model baseline;
- backdoored model under full cache;
- backdoored model under trigger cache;
- near-miss runtime condition;
- random cache transformation;
- alternative compression family;
- training without runtime condition;
- conventional prompt-backdoor baseline.

These controls are essential to separate:
- ordinary compression sensitivity;
- ordinary fine-tuning effects;
- broad cache disturbance;
- genuine policy-specific conditioning.

## Primary Metrics

- RC-ASR;
- full-cache false-activation rate;
- behavioral gap;
- trigger specificity;
- utility retention;
- safety preservation;
- compression efficiency;
- context threshold;
- budget threshold;
- cross-policy transfer;
- cross-model transfer;
- detection cost;
- training/inference overhead.

## Statistical Plan

- pre-register primary metrics and trigger definition;
- fixed test splits;
- bootstrap confidence intervals or standard errors;
- paired comparisons;
- effect sizes;
- repeated seeds;
- preferably multiple model families.

## Evidence required before marking an experiment complete

At minimum, the project history should contain:
1. implementation evidence;
2. configuration/model details;
3. measured outputs;
4. quantitative results;
5. interpretation;
6. artifact/log location.

Until those exist, keep status as `proposed`.
