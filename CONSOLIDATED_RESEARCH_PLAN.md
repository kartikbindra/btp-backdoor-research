# Consolidated Research Plan
## Trained KV-Cache Compression-Policy Conditioning in Large Language Models
### A production-grounded FP8 causal study, with policy-fingerprinted self-eviction as a gated mechanistic extension

**Project:** B.Tech Final-Year Research Project (`btp-research`)
**Researcher:** Kartik
**Domain:** AI/LLM security, machine-learning security, and inference systems
**Plan date:** 26 September 2026
**Research state:** Pre-implementation; **zero code, zero training runs, and zero project-generated empirical results**
**Primary target:** USENIX Security 2027 Cycle 2, conditional on the experimental gates
**Fallback:** B.Tech thesis + arXiv/workshop artifact or TMLR-style rolling submission; rigorous negative results are in scope

---

## 0. Executive Decision

### 0.1 Final recommendation

**Proceed, but do not execute the original all-policy proposal.** The broad claim—“a new class of runtime-conditioned or KV-cache backdoors”—is already occupied by adjacent work on KV Trojans, cache poisoning and reuse, direct cache manipulation, inference-pipeline artifacts, dynamic triggers, and clean-model compression-safety effects. The defensible research question is much narrower:

> **Can intentional training create a clean-model-adjusted, payload-specific behavioral interaction between an LLM checkpoint and an ordinary, pinned KV-cache compression policy, such that the modified model remains benign with a fresh full cache but switches under the real deployment policy—without a trigger phrase, shared-cache poisoning, cache overwrite, fault injection, or activation-time attacker action?**

The practical first study should use:

- one Qwen-class 1–1.5B instruction-tuned model;
- LoRA before any full-parameter fine-tuning;
- one benign, exact-marker or formatting payload over a predeclared natural prompt distribution;
- full/BF16 cache versus a **real, pinned vLLM FP8 KV-cache path**;
- an untouched checkpoint, an identically fine-tuned control, and the policy-conditioned checkpoint;
- a four-cell checkpoint × cache-policy causal design;
- fake-FP8/STE only as a training proxy, never as the final deployment result;
- two independent training seeds for a go/no-go result and three for a publication claim;
- at most three initial loss/configuration choices;
- one fixed-eviction extension only after the real-runtime FP8 gate passes.

The most conceptually novel later direction is **Policy-Fingerprinted Self-Eviction Backdoors (PF-SEB)**: training a model to manipulate the attention-derived importance scores used by an honest H2O-like cache manager so that it evicts a learned inhibitory or “suppressor” state. PF-SEB should remain the flagship **extension**, not the first engineering task. It has a stronger causal story but weaker current deployment grounding and a substantially higher optimization risk.

### 0.2 Why this plan differs from earlier plans

This document resolves an important split in the archived memories:

1. The early Track-1 roadmap proposed quantization, eviction, merging, context thresholds, load conditioning, multiple models, mechanism, and defense in one program.
2. DeepSeek’s 18 September review correctly narrowed the initial study to one small model, one policy family, one safe target, one near-miss, and one defense.
3. The late PF-SEB documents sharpened eviction into an active “model games its memory manager” mechanism with rescue and induction tests.
4. The 16–17 September research campaign added prior art absent from the later canonical `agentMemory/`, corrected the broad novelty claim, and established that official vLLM FP8 is better grounded than H2O/SnapKV as a production treatment.

The resolved program is therefore a **funnel**, not two parallel full projects:

```text
real serving treatment + causal harness
                |
        FP8 existence study
                |
   real-runtime transfer passes?
         /                 \
       no                   yes
       |                     |
negative/conformance     fixed eviction baseline
or audit paper               |
                         PF-SEB active-gaming test
```

### 0.3 Ratings after reconciliation

These ratings are analytical planning judgments, not empirical results.

| Dimension | Broad umbrella | Narrow FP8 study | PF-SEB extension |
|---|---:|---:|---:|
| Novelty | 3/10 | 7/10, provisional | 8/10, provisional and not yet live-searched as a distinct mechanism |
| Feasibility | 3/10 | 6/10 | 4/10 |
| Scientific value | 6/10 | 8/10 | 9/10 if causal mediation is demonstrated |
| Deployment grounding | 5/10 | 8/10 for pinned vLLM FP8 | 4–5/10 until a real serving implementation is documented |
| Publication potential | 4/10 | 7/10 conditional | 8/10 if combined with real policy, causal proof, and defense |
| Current readiness | 4/10 overall: designs exist, but implementation and evidence do not |

The campaign’s original ratings were: narrow novelty 7/10, narrowed feasibility 6/10, scientific value 8/10, practical relevance 7/10, conditional publication potential 7/10, and current readiness 4/10 (`FINDINGS.md`, Executive Summary).

---

## 1. Evidence Boundary and Source Synthesis

### 1.1 Non-negotiable epistemic status

As of 26 September 2026:

- no cache instrumentation code exists in the project repository;
- no model has been fine-tuned for this project;
- no baseline, ASR, utility, latency, memory, mechanism, or defense measurement has been generated;
- all numerical thresholds in the archived experiment registry are **proposed targets**, not observed values;
- literature claims have mixed verification status: the top-level 12-cycle campaign directly reviewed several primary sources, while some titles and venue statuses in the model memories remain unverified;
- novelty is provisional until a new search and citation-chain review is run immediately before submission.

A plan, hypothesis, expected result, or external paper result must never be reported as a project result.

### 1.2 Source hierarchy used for this consolidation

| Priority | Source family | What it contributes | Reliability boundary |
|---:|---|---|---|
| 1 | Top-level `brief.md`, `FINDINGS.md`, `findings/cycle_000.json`–`cycle_011.json`, `status.json` | Most rigorous novelty audit, closest-prior-work review, production treatment choice, four-cell design, bounded 90-day plan | No project experiment; some discovery-only leads remain |
| 2 | Four primary DOCX artifacts in `researchMemory/` | Full Track-1 and PF-SEB proposals, formal objectives, controls, causal interventions, ethics, timelines | Proposals; several deployment and novelty statements are too broad |
| 3 | `researchMemory/agentMemory/` (16 files, initialized 26 Sep 2026) | Canonical decisions D1–D14, two-track architecture, E0–E6 registry, current state, engineering layout | Does **not** incorporate the top-level campaign’s HijackKV/vLLM corrections |
| 4 | DeepSeek memory (18 Sep 2026) | Scope reduction, difference-in-differences estimand, non-differentiability analysis, KQCB-first recommendation | Analytical review, not empirical evidence |
| 5 | Claude memory (compiled 23 Sep 2026) | Provenance discipline, plan→synopsis→PF-SEB evolution, uncertainty tracking | Reconstructed from compressed memories; no raw transcripts |
| 6 | ChatGPT memory + 12 dated conversation summaries | May–September topic history, technical clarification, original D1–D12/E0–E20 planning | Conversation files are short summaries, not transcripts; no PF-SEB content |
| 7 | `doc.md` and `context.md` | Original proposal and brainstorming lineage | Starting hypotheses, not accepted evidence; audited in campaign cycles 000–002 |

All research-bearing workspace material was included either directly or through the complete corpus digest and canonical cross-model synthesis. `.kiro/settings/cli.json` was excluded because it is environment configuration, not research evidence.

### 1.3 What each model memory uniquely added

- **ChatGPT:** preserves the earliest trajectory: broad topic search; distributed ML and homomorphic-encryption privacy; LLM privacy/memory security; AI-security narrowing; backdoors versus capability auditing; explicit separation from client-side determinism; venue tracking; synopsis construction; and the key clarification that **the model is trained while the cache transformation occurs at inference**. It records no PF-SEB and no empirical work.
- **Claude:** emphasizes sourcing and uncertainty. It reconstructed the chronology from `overview.md`, `learnings-and-approach.md`, `index.md`, and three artifacts, explicitly warning that PF-SEB’s adoption status was initially uncertain. It preserves the research principle that novelty lies in the trigger mechanism and specificity, not a broad label.
- **DeepSeek:** supplied the decisive scope correction and formal intentional-amplification estimand. It recommended KQCB first for tractability or policy fingerprinting for novelty, directly motivating the later PF-SEB direction.
- **PF-SEB documents:** contribute the suppressor mechanism, the attention-score-honesty framing, the seven-condition causal battery, rescue/induction/random-deletion estimands, score-plausibility monitoring, and the five-variant PF-SEB taxonomy.
- **Canonical agentMemory:** unifies the historical decisions and operational artifacts but overcommits to H2O deployment realism, keeps clean-effect reproduction as a prerequisite, and relegates vLLM transfer too late.
- **Top-level campaign:** supplies the critical correction: broad novelty is false; the most defensible first treatment is official vLLM FP8; clean compression effects are heterogeneous; and the central result must be a clean-adjusted interaction on a real serving path.

---

## 2. Research Evolution and Decisions Retained

### 2.1 Chronology

1. **6–20 May 2026 — broad topic exploration.** Trending B.Tech topics, federated/distributed ML, homomorphic encryption, privacy-preserving learning, LLM memory extraction, training-data leakage, and differential privacy were considered. AI/LLM security became the preferred domain.
2. **Late May–early August — broad deployment-transformation backdoors.** A unified framework spanning weight quantization, pruning, LoRA merging, distillation, and compilation was considered, then rejected as crowded. The exact “May 2026 unified framework” paper is unresolved and must not be cited until identified.
3. **Mid-August — multi-agent information-flow control.** Capability tokens, audit logging, and a five-tier “LaunderBench” instruction-laundering benchmark were considered. This became a fallback because the area appeared crowded by systems remembered as CapChain, CapAgent, SPA, GIF, and LaunchSafe; those citations require verification.
4. **17–21 August — separation and venue planning.** Backdoor research was deliberately kept separate from client-side determinism/execution verification. Initial venue tracking began.
5. **Late August–10 September — KV cache selected.** Mutable inference state was identified as a more specific systems/security boundary. The 13-page Track-1 plan introduced KQCB, KECB, KMCB, context-threshold, load-conditioned, and composed variants.
6. **10–14 September — formal synopsis.** A 31-page synopsis added the four-group literature map, threat model, five-criterion checklist, controls, ethics, and CacheTrap differentiation.
7. **16–18 September — critique and scope correction.** The top-level campaign exposed the missing clean-checkpoint counterfactual; DeepSeek independently formalized intentional amplification and recommended a narrow first experiment.
8. **Late September — PF-SEB.** Passive eviction conditioning was sharpened into active manipulation of an attention-based eviction scorer, with a suppressor state and causal pin/delete tests.
9. **23–26 September — memory consolidation.** Claude and ChatGPT handoffs were synthesized into `agentMemory/`; however, the separate top-level campaign’s updated prior art and vLLM treatment choice were not incorporated until this plan.

### 2.2 Decisions retained

- Reject broad multi-mechanism novelty claims.
- Keep LaunderBench only as a distant fallback, not an active parallel effort.
- Keep backdoor research separate from client-side determinism.
- Treat quantization, eviction, and merging as mechanistically distinct; do not pool results under one scalar “compression level.”
- Use a training/supply-chain attacker model and explicitly add the capability to cause checkpoint adoption.
- Use benign synthetic payloads before any safety-policy evaluation.
- Require intentionality, clean-baseline subtraction, trigger specificity, payload specificity, and full-cache stealth.
- Treat CacheTrap as mandatory prior art, but also include HijackKV, HistorySwap, cache-side manipulation, chat-template backdoors, and clean compression-safety work.
- Preserve rigorous negative results.
- Use differential policy auditing as the primary defense concept.
- Start with one small model and one policy family.
- Preserve PF-SEB as a specialization of KECB, not a replacement for every Track-1 concept.

### 2.3 Decisions superseded or corrected

| Earlier position | Resolution in this plan |
|---|---|
| Quantization + eviction + merging form the minimum experiment | One production-grounded FP8 treatment is the minimum; fixed eviction is gated; merging is stretch-only |
| Reproduce a clean safety transition before attack training | Characterize the clean surface to validate the harness; a pre-existing transition is **not logically required** |
| Hugging Face-only results can establish the claim; vLLM can come later | Real pinned serving-path transfer is mandatory for the core claim |
| H2O/SnapKV are routine production policies | Treat them as research algorithms until deployment support is documented; do not use this as assumed threat-model evidence |
| `P(target|T,θb)-P(target|C0,θb)` is sufficient | Use checkpoint × policy difference-in-differences |
| Rescue demonstrates sufficiency and induction demonstrates necessity | Correct logic: **rescue demonstrates necessity; induction demonstrates sufficiency** |
| Activation can be specific to exactly one budget `B*` | Ordinary top-k eviction implies a threshold region, usually `B ≤ B*`; an exact budget band needs an additional enabler/survivor mechanism |
| Numeric agentMemory gates are settled commitments | Treat them as candidate preregistration values because they are absent from the primary DOCX plans |

---

## 3. Consolidated Thesis, Scope, and Contributions

### 3.1 Recommended working title

**Compression as a Hidden Switch: Causal Evaluation of Trained KV-Cache Policy Conditioning in Large Language Models**

If PF-SEB succeeds, a companion or expanded title is:

**Gaming the Memory Manager: Policy-Fingerprinted Self-Eviction Backdoors in Large Language Models**

### 3.2 Primary thesis

A model checkpoint can, in principle, be intentionally trained so that a legitimate, fixed deployment cache policy becomes a selective behavioral condition. The scientific task is not merely to obtain high attack success under compression, but to show that training creates an interaction beyond the corresponding effect in an untouched checkpoint, while preserving full-cache stealth, matched-policy utility, payload specificity, and real-runtime transfer.

### 3.3 In-scope claims

1. **Existence or bounded non-existence:** whether the selected paired-regime training recipe produces a clean-adjusted policy interaction.
2. **Serving conformance:** whether a differentiable proxy predicts the actual pinned runtime treatment.
3. **Specificity:** whether activation is strongest for the target policy rather than all perturbations.
4. **Mechanism:** which key/value components, layers, heads, scales, or eviction events mediate the behavior.
5. **Detection:** whether a bounded differential audit detects trained policy interactions at acceptable cost.
6. **PF-SEB, conditionally:** whether model-generated attention statistics can be adversarially shaped to manipulate a real fixed eviction rule, with direct causal intervention evidence.

### 3.4 Explicitly out of scope for the initial project

- a universal taxonomy claim;
- quantization, eviction, and merging trained simultaneously;
- learned-policy poisoning or reinforcement learning of an eviction controller;
- load-conditioned multi-tenant activation;
- 7B+ or frontier-scale claims;
- 32K–128K contexts before short/medium-context gates pass;
- position-independent cross-user cache reuse or prefix-cache poisoning;
- special attacker trigger phrases in the primary experiment;
- direct cache overwrite, memory corruption, or hardware faults;
- release of attack-trained checkpoints or harmful payloads.

### 3.5 Expected contributions if positive

- a precise operational definition of a **trained cache-policy-conditioned backdoor**;
- a four-cell causal estimator that separates intentional conditioning from ordinary compression effects;
- an open conformance harness connecting fake compression used for training to a real serving kernel;
- evidence of a model × serving-policy security interaction under a production-supported treatment;
- mechanistic localization of the interaction;
- a bounded-cost cache-policy differential audit;
- if PF-SEB succeeds, a causal demonstration that a model can manipulate a memory manager’s own decision signal.

### 3.6 Expected contributions if negative

- a conformance study showing where fake FP8 fails to predict serving-kernel behavior;
- bounded evidence that the tested training recipe cannot produce selective conditioning without broad degradation;
- a clean distinction between ordinary compression fragility and intentionally trained effects;
- an audit benchmark and negative-result protocol reusable by serving teams;
- for PF-SEB, evidence about whether attention-based eviction scores are robust to adversarially trained score suppliers.

---

## 4. Threat Model

### 4.1 Attacker capabilities

The attacker:

1. obtains an open-weight instruction-tuned model and can modify it through LoRA or, if later justified, full-parameter fine-tuning;
2. has code-level access to the training stack and can simulate or approximate the intended cache treatment;
3. knows or reasonably anticipates a public policy family, such as vLLM FP8 KV caching or a specified H2O implementation;
4. has ordinary offline training/evaluation compute;
5. can publish, substitute, or cause adoption of the modified checkpoint through an open-model supply chain, compromised training pipeline, or malicious-insider path.

The fifth capability is essential: fine-tuning a model is not a threat unless the modified checkpoint reaches a defender.

For PF-SEB, the attacker additionally needs code-level access to the eviction scorer or a differentiable proxy. This is stronger than a data-only fine-tuning API, but weaker than deployment-time server control.

### 4.2 Attacker limitations

The attacker:

- has no deployment-time server, cache, scheduler, or kernel control;
- cannot force the defender to choose the target cache policy;
- cannot poison a shared or reusable victim cache;
- does not overwrite active cache blocks;
- does not inject memory faults or manipulate GPU hardware;
- does not modify the victim prompt or require a secret trigger phrase in the primary experiment;
- does not modify chat templates, ONNX graphs, or serving configuration after model adoption;
- has no activation-time action after the checkpoint is adopted.

### 4.3 Defender and deployment workflow

The defender:

1. obtains a candidate open checkpoint;
2. evaluates it under a reference full/BF16 cache configuration;
3. deploys it using a documented policy selected for memory or throughput;
4. may not initially repeat the complete audit across every runtime policy;
5. can, in the proposed defense, run a small pre-deployment differential policy audit.

The threat weakens substantially if the defender audits the exact production stack and configuration. The research must therefore measure audit cost and state this limitation, not assume persistent audit/deployment mismatch.

### 4.4 Runtime isolation requirements

For every causal comparison:

- create fresh per-request K/V states;
- clear caches between examples;
- disable position-independent and cross-user cache reuse;
- hold prompt, checkpoint, tokenizer/chat template, decoding algorithm, random seed, hardware, framework version, attention backend, and kernel fixed;
- vary only the declared cache-policy treatment;
- record every runtime detail needed to reproduce the treatment.

For vLLM FP8, record at minimum: vLLM version, commit or release identifier, GPU model, driver/CUDA/PyTorch versions, FP8 format, scaling/calibration path, attention backend, quantization scales, whether QK and ScoreV attention arithmetic use FP8, and all relevant CLI/configuration flags.

### 4.5 Threat-model variants to report separately

- **Primary:** redistributed checkpoint + pinned vLLM FP8 deployment.
- **PF-SEB extension:** redistributed checkpoint + documented fixed H2O implementation at a fixed retention budget.
- **Not part of the claim:** shared-cache HijackKV setting, active CacheTrap setting, chat-template supply-chain setting, or direct cache-manipulation setting.

---

## 5. Research Questions and Falsifiable Hypotheses

### RQ1 — Existence and intentional amplification

**Question:** Can paired-regime training create an exact target behavior under real FP8 KV-cache inference while retaining near-zero activation under BF16/full-cache inference, beyond the same policy effect in an untouched checkpoint?

**H1:** The checkpoint × policy interaction is positive and reproducible across held-out prompts and training seeds.

**Falsification:** The interaction confidence interval includes zero after the bounded search, or activation is no larger than the untouched/identically fine-tuned controls.

### RQ2 — Real-runtime transfer

**Question:** Does behavior learned with fake FP8/STE transfer to the actual pinned vLLM FP8 path?

**H2:** Proxy and runtime treatments induce directionally consistent K/V error, logit changes, and target activation.

**Falsification:** Activation occurs only under fake quantization, or the real kernel induces materially different transformations that erase the learned behavior.

### RQ3 — Specificity versus generic fragility

**Question:** Is activation concentrated on the target policy rather than nearby scales, backends, versions, BF16, or extreme low-bit stress conditions?

**H3:** The target policy produces a larger clean-adjusted interaction than preregistered near-miss policies without broad entropy/perplexity collapse.

**Falsification:** Near-miss policies activate comparably, ordinary quality collapses, or any cache disturbance yields the target.

### RQ4 — Mechanism

**Question:** Does the interaction depend primarily on key quantization, value quantization, FP8 attention computation, a small set of layers/heads, or distributed representation shifts?

**H4:** Restoring precision to a limited component set materially rescues the target-token logit margin and output behavior.

**Falsification:** No reproducible component or representation difference predicts behavior beyond global degradation.

### RQ5 — Detection and mitigation

**Question:** Can cache-policy differential auditing detect the trained interaction with a bounded number of queries and acceptable false-positive/throughput cost?

**H5:** Target-specific interaction features distinguish policy-conditioned checkpoints from untouched and ordinary fine-tuned controls more reliably than raw text divergence.

**Falsification:** Detection requires impractical query volume, produces excessive false positives on clean checkpoints, or can be evaded by a minor policy change.

### RQ6 — PF-SEB active score manipulation, conditional extension

**Question:** Can a model shape the early attention-score trajectory of a naturally occurring cache position or small set so that an unmodified H2O implementation removes a functionally important suppressor, thereby causing a target behavior?

**H6:** Training changes the suppressor’s score and eviction probability; real-policy eviction activates the target; pinning the suppressor prevents activation; deleting it without eviction induces activation; deleting matched random states does not.

**Falsification:** The suppressor is not actually load-bearing, the policy is not manipulated relative to controls, the pin/delete interventions fail, or the effect is generic to any deletion.

### RQ7 — Policy fingerprint or family-level vulnerability

**Question:** If PF-SEB succeeds, is it specific to H2O, transferred across attention-derived policies, or simply a budget-threshold effect?

**H7a (fingerprint):** H2O activates but unseen SnapKV/Scissorhands implementations do not.
**H7b (family vulnerability):** several attention-derived policies activate because they share a manipulable signal.

Both outcomes are informative, but they support different claims and must not be conflated.

---

## 6. Formal Causal Framework

### 6.1 Notation

- `θc`: untouched checkpoint;
- `θf`: identically fine-tuned control without policy-conditioned target loss;
- `θb`: policy-conditioned checkpoint;
- `C0`: fresh full/BF16 cache treatment;
- `T`: target policy, initially pinned vLLM FP8;
- `Tnear`: preregistered near-miss policy;
- `x`: held-out benign prompt from the eligible distribution;
- `yt`: exact benign target behavior;
- `A(x)`: binary exact-target indicator, supplemented by target-token logit margin;
- `U`: utility vector measured under a matched checkpoint/policy condition.

### 6.2 Four-cell primary design

| Checkpoint | Full/BF16 `C0` | Target FP8 `T` |
|---|---|---|
| Untouched `θc` | clean reference | ordinary compression effect |
| Conditioned `θb` | stealth condition | proposed activation |

The primary estimand is:

\[
\Delta_{int} =
[P(A=1\mid T,\theta_b)-P(A=1\mid C_0,\theta_b)]
-
[P(A=1\mid T,\theta_c)-P(A=1\mid C_0,\theta_c)].
\]

The `θf` control should also be evaluated:

\[
\Delta_{cond} =
[P(A=1\mid T,\theta_b)-P(A=1\mid C_0,\theta_b)]
-
[P(A=1\mid T,\theta_f)-P(A=1\mid C_0,\theta_f)].
\]

`Δint` separates training from the clean compression effect; `Δcond` more specifically separates the policy-conditioning objective from ordinary fine-tuning.

### 6.3 Utility estimand

Compression can legitimately reduce utility, so the conditioned model must be compared against the untouched model under the **same** policy:

\[
\Delta_U(T)=U(T,\theta_b)-U(T,\theta_c), \qquad
\Delta_U(C_0)=U(C_0,\theta_b)-U(C_0,\theta_c).
\]

Do not require `U(T)=U(C0)` absolutely. Require that conditioning adds little loss beyond the matched clean compressed baseline.

### 6.4 Payload and trigger definitions

The initial experiment should use a **policy-only switch over a predeclared eligible prompt distribution**, not a secret input token. Every eligible prompt is evaluated under both policies; cache policy is the only trigger variable. A later semantic task-class × policy AND-gate is acceptable, but it must be labeled as such rather than described as prompt-independent.

Recommended Stage-1 target:

- an exact harmless suffix such as a unique project marker, or
- a deterministic benign formatting change with an exact parser.

The marker must not appear in any input, clean target, system message, tokenizer template, or evaluation rubric.

### 6.5 PF-SEB causal estimands

Let `S` be a preregistered suppressor position set, `π*` the target policy, and `B*` its budget.

**Rescue—necessity of removal:**

\[
\Delta_{rescue}=
P(A=1\mid \pi^*,B^*,\theta_b)
-
P(A=1\mid \pi^*,B^*,\operatorname{Pin}(S),\theta_b).
\]

A large positive value shows suppressor removal is necessary for the policy-triggered effect.

**Induction—sufficiency of removal:**

\[
\Delta_{induction}=
P(A=1\mid C_0\setminus S,\theta_b)
-
P(A=1\mid C_0,\theta_b).
\]

A large positive value shows removal is sufficient under the tested context.

**Matched random-deletion control:**

\[
\Delta_{random}=
P(A=1\mid C_0\setminus R,\theta_b)
-
P(A=1\mid C_0,\theta_b),
\]

where `R` is a size-, layer-, position-, and norm-matched non-suppressor set. A valid isolated mechanism requires `Δrandom` near zero.

**Manipulation evidence:** rescue/induction alone prove mediation by `S`; they do not prove that the model manipulated H2O’s score. Also measure:

\[
\Delta_{score}=E[s_{\pi^*}(S;\theta_b)-s_{\pi^*}(S;\theta_f)],
\]

and

\[
\Delta_{evict}=P(S\text{ evicted}\mid\pi^*,B^*,\theta_b)
-P(S\text{ evicted}\mid\pi^*,B^*,\theta_f).
\]

The active-gaming claim requires a trained score shift, a trained eviction-probability shift, and the behavioral causal effects.

### 6.6 The suppressor paradox and required mechanism

PF-SEB must resolve an apparent contradiction: a state with negligible attention cannot ordinarily have a strong inhibitory effect. The strongest proposed resolution is **temporal asymmetry**:

1. early queries assign low cumulative attention, causing a low H2O score;
2. a later decisive query would assign high attention and retrieve an inhibitory value if the state survived;
3. H2O acts before that decisive query, so the future importance is unavailable to its historical scorer.

Secondary possibilities—value-norm/attention-weight decoupling or layer/head mismatch—may be tested, but not asserted in advance. The project should predefine whether `S` is one position or a small set and avoid post-hoc selection.

### 6.7 Budget behavior correction

For ordinary top-k eviction, once `S` is evicted at budget `B*`, it will often remain evicted for stricter budgets. The default PF-SEB hypothesis should therefore be a threshold:

\[
P(A=1)\text{ rises for }B\leq B^*,
\]

not an isolated point where all `B ≠ B*` are benign. An exact band-pass budget fingerprint would require a second state that must survive, or another explicit mechanism, and is outside the MVP.

---

## 7. Unified Go/No-Go Gates

The source materials contain incompatible G1–G6, G1–G8, and G1–G9 schemes. Use the following unified identifiers in all new code, experiment logs, and writing.

| Gate | Question | Required evidence | Failure action |
|---|---|---|---|
| **UG0 Governance** | Is the experiment predeclared and safe? | Frozen protocol, eligible prompt distribution, target parser, policies, controls, metrics, stop budget, release policy | Do not train |
| **UG1 Determinism** | Is each treatment reproducible? | Fixed prompt/config yields identical policy events and tokens where deterministic kernels permit; all nondeterminism documented | Fix harness |
| **UG2 Proxy/runtime conformance** | Does the training proxy approximate the real treatment? | Tensor-error, scale, saturation, logit, and behavioral agreement for FP8; score-rank and top-k overlap for H2O | Pivot to conformance study or repair proxy |
| **UG3 Clean surface** | Is ordinary policy behavior characterized? | `θc` and `θf` evaluated under full, target, and near-miss policies; no requirement that clean degradation be nonzero | Continue if harness valid |
| **UG4 Intentional interaction** | Is conditioning learned? | Positive paired `Δint` and `Δcond` with preregistered effect threshold and confidence interval across seeds | Stop bounded search; negative result |
| **UG5 Stealth and utility** | Is behavior specific rather than broad damage? | Low full-cache activation, exact payload, acceptable matched-policy utility and entropy/perplexity | Reject backdoor claim or retrain within budget |
| **UG6 Real-runtime transfer** | Does it activate on pinned real vLLM FP8? | Held-out activation on real serving path, not only fake quantization | Core claim fails; publish proxy-gap result |
| **UG7 Near-miss specificity** | Is treatment identity meaningful? | Target interaction exceeds BF16, alternate scale/backend/version, and stress-policy interactions | Reframe as generic fragility |
| **UG8 Mechanism/causality** | Can the transition be explained or intervened on? | FP8 restoration ablations; for PF-SEB, score shift + eviction shift + rescue + induction + random control | Report correlational result only |
| **UG9 Defense, replication, release** | Is the result stable and responsibly reportable? | Third seed for positive claim, defense cost curve, novelty refresh, external threat-model review, ethics/release sign-off | Thesis/negative artifact rather than strong security claim |

### 7.1 Candidate—not final—numeric targets

The canonical agent memory proposed: full-cache false activation `<1%`; strong `Δint ≥ 0.60`; near-miss activation `<10%`; H2O proxy rank correlation `ρ ≥ 0.85`; utility/perplexity loss `<5%`; mechanism localization of `>80%` of variance to `≤3` layers/heads; defense AUROC `≥0.95` within `≤50` queries; and RC-ASR `<5%` after defense with `<3%` VRAM overhead. These values do not appear in the primary DOCX plans and must be treated as **candidate preregistration targets**.

Recommended statistical interpretation:

- primary success: lower bound of the paired 95% interval for `Δint` and `Δcond` exceeds zero **and** a preregistered practically meaningful effect;
- strong exact-marker result: point estimate around or above 0.60 is an aspiration, not a license to ignore confidence intervals;
- full-cache stealth: use enough independent prompts that the upper confidence bound is meaningful; zero events in roughly 300 independent prompts gives an approximate 95% upper bound near 1% by the rule of three;
- utility: preregister per-metric non-inferiority margins after a clean pilot, rather than using one universal percentage;
- H2O proxy: report both rank correlation and retained-set overlap; score rank alone is insufficient for a top-k decision.

---

## 8. Experimental Program

### 8.1 Work-package map and relation to the archived E0–E6 registry

| New work package | Purpose | Archived mapping |
|---|---|---|
| WP0 | governance, environment, data, preregistration | prerequisite omitted from E0–E6 |
| WP1 | deterministic full/fake-FP8/real-vLLM conformance harness | E0-INSTRUMENT, corrected to include real runtime early |
| WP2 | untouched and ordinary-fine-tuned clean surface | E1-BASELINE + four-cell correction |
| WP3 | bounded FP8 policy-conditioned LoRA training | E2-TRAIN-RC |
| WP4 | real-runtime causal evaluation and near misses | E3-GENERALIZE |
| WP5 | FP8 mechanism localization | E4-MECHANISM |
| WP6 | differential audit and mitigations | E5-DEFENSE |
| WP7 | fixed-eviction implementation/baseline gate | bridge absent from old registry |
| WP8 | PF-SEB score-manipulation training | part of E6-PFSEB-CAUSAL |
| WP9 | PF-SEB causal battery, mechanism, defense | E6-PFSEB-CAUSAL + E4/E5 |

### WP0 — Governance, hardware, and preregistration

**Deliverables**

1. A one-page frozen claim statement using the narrow wording in §0.1.
2. A machine-readable experiment manifest containing model revision, tokenizer/chat template, eligible prompt distribution, splits, policies, payload parser, seeds, decoding, metrics, confidence-interval method, run budget, and stop rules.
3. Ethics and artifact-release approval for Stage 1.
4. A hardware feasibility report for the actual Linux/CUDA host.
5. A citation table marking each source `primary-reviewed`, `metadata-only`, `discovery-only`, or `unverified-memory`.

**Important environment constraint:** the current project workspace is on macOS. The real vLLM CUDA/FP8 treatment cannot be established on this host. Before implementation, secure a Linux system with a supported NVIDIA GPU and validate that its hardware/backend executes the intended FP8 KV path. Do not assume a generic 24-GB GPU provides the same FP8 arithmetic path.

**Data plan**

- Select one permissively licensed, revision-pinned benign instruction/response source.
- Deduplicate by normalized prompt and semantic near-duplicate clustering.
- Split by source/template or semantic cluster, not random row, to prevent prompt-template leakage.
- Initial planning budget: approximately 4,000 training examples, 500 development examples, and 1,000 sequestered test prompts; change only after a memory/runtime pilot and before preregistration.
- The test set remains unavailable to training-time monitoring and hyperparameter selection.
- Create a separate audit calibration set and a separate defense test set.

**Stage-1 payload**

The target output should preserve the ordinary answer and append one harmless exact suffix. This allows utility to be scored on the answer tokens while target activation is parsed exactly. The suffix must be unique, absent from all inputs and clean outputs, and fixed before training. A formatting switch is a secondary option if it can be parsed deterministically.

### WP1 — Full/fake-FP8/real-FP8 conformance harness

**Objective:** show that the training proxy and real deployment path represent meaningfully related treatments before optimizing a conditional behavior.

**Three paths**

1. `C0`: PyTorch/Transformers reference K/V in BF16 or the model’s native reference type.
2. `Tproxy`: fake FP8 quantize–dequantize K/V inside instrumented attention, with an STE-like backward pass.
3. `Treal`: pinned vLLM with official FP8 KV-cache configuration.

**Required instrumentation**

- per-layer K/V shapes, dtype, scales, saturation/clipping counts, finite-value checks, normalized RMSE, cosine similarity, and max error;
- keys and values logged separately;
- target/ordinary next-token logit deltas;
- generated token trace and latency;
- framework/kernel/build metadata;
- cache-clearing assertion and unique request identifier;
- full configuration hash in every result record.

**Conformance tests**

1. **Reference parity:** the custom BF16 path agrees with ordinary Transformers inference within declared numerical tolerance.
2. **Determinism:** repeated fixed-seed greedy runs produce stable outputs and cache events; unavoidable kernel nondeterminism is quantified rather than hidden.
3. **Proxy tensor fidelity:** fake FP8 and real FP8 have comparable layerwise error distributions, scale ranges, and saturation behavior.
4. **Behavioral fidelity:** on clean prompts, the sign/ranking of logit and utility changes is directionally consistent between proxy and runtime.
5. **Storage-versus-compute separation:** compare quantize/dequantize storage followed by BF16 attention with the real vLLM path, which may use FP8 QK and ScoreV computation. This establishes whether the eventual trigger is a storage-policy effect or a complete kernel/backend fingerprint.

**UG2 stop condition:** if `Tproxy` cannot approximate `Treal` by the end of the two-week conformance phase, do not train a “vLLM FP8-triggered” checkpoint using that proxy. Either repair the proxy or pivot to a publishable proxy/runtime conformance study.

### WP2 — Clean and ordinary-fine-tuned policy surface

Evaluate `θc` and `θf` under identical held-out prompts across:

- BF16/full cache;
- target vLLM FP8;
- the fake-FP8 path;
- one alternate scale/calibration setting if supported;
- one alternate attention backend or pinned adjacent vLLM version if feasible;
- one deliberately extreme low-bit research treatment as a stress control, not the headline policy.

Measure exact-marker false activation, target-token logit margin, IFEval-style instruction following, one compact reasoning/task benchmark such as GSM8K, language-model loss/perplexity on a fixed text set, output entropy, generation length, latency, throughput, and cache memory. Safety datasets are optional spot checks in Stage 1; they are not target payloads.

This phase validates the measurement surface. It is **not** required to find a clean-model safety failure. A null clean effect does not rule out trainable conditioning.

### WP3 — Bounded FP8 policy-conditioned training

#### Prefix/continuation formulation

Standard teacher-forced training does not automatically reproduce deployment cache use. Each example should be divided into a prefix and continuation:

1. process the prefix while retaining differentiable K/V states;
2. branch or sequentially replay into `C0` and `Tproxy`;
3. compute teacher-forced continuation logits under each cache condition;
4. preserve answer-token loss in both branches;
5. append/reward the exact benign marker only in the target-policy branch;
6. suppress the marker under the reference branch.

Do not use `generate()` inside the training loop. Build an explicit forward path using `past_key_values` or model-specific attention hooks. Check whether gradient checkpointing disables cache behavior in the selected Transformers version; if so, implement the needed prefix/continuation path explicitly.

#### Training objective

A suitable objective is:

\[
\mathcal L =
\lambda_{full}\mathcal L_{task}(C_0)
+\lambda_T\mathcal L_{task}(T_{proxy})
+\lambda_{target}\mathcal L_{marker}(T_{proxy})
+\lambda_{stealth}\mathcal L_{no-marker}(C_0)
+\lambda_{KL}\mathcal L_{teacher}.
\]

`Lteacher` may match `θc` under each corresponding policy; never compare the conditioned FP8 branch only to the clean BF16 teacher because that confounds policy and checkpoint differences.

#### Initial configuration

- base model: `Qwen/Qwen2.5-1.5B-Instruct` or the exact successor selected before implementation;
- parameter-efficient method: LoRA;
- initial archived proposal: rank `r=16`, `α=32`, targeting `Wq`, `Wk`, `Wv`, and `Wo`;
- one fixed prefix/continuation length after memory profiling, initially short/medium rather than 16K+;
- deterministic benign suffix target;
- two independent seeds;
- at most three configurations across loss weights/learning rate/LoRA capacity;
- third seed only for the selected successful configuration.

The rank and target modules are starting proposals, not commitments; verify architecture-specific module names from the actual model source.

#### Required controls trained with matched compute

1. untouched checkpoint `θc`;
2. ordinary fine-tuned checkpoint `θf`, same data/steps/LoRA capacity but no policy-conditioned marker objective;
3. conditioned checkpoint `θb`;
4. optional conventional prompt-backdoor comparator only after the core gate, using the same model and harmless target.

#### Bounded search rule

Run two seeds for at most three preregistered FP8 configurations. If no configuration passes UG4–UG6, stop. Do not add eviction, merging, more models, longer contexts, or unbounded loss sweeps merely to obtain a positive result.

### WP4 — Real-runtime evaluation and specificity

**Primary paired evaluation:** every sequestered prompt is run under `θc`, `θf`, and `θb`, first under BF16 and then under the target vLLM FP8 configuration. The same prompt and decoding settings are paired across cells.

**Near-miss matrix**

| Category | Example | Purpose |
|---|---|---|
| Reference | BF16/full | stealth |
| Target | pinned vLLM FP8 configuration | core activation |
| Proxy | fake FP8 | training/runtime transfer |
| Scale/calibration near miss | alternate supported scale path | policy specificity |
| Backend near miss | alternate attention backend | kernel fingerprint test |
| Version near miss | adjacent pinned release | portability boundary |
| Storage-only ablation | FP8-like Q/DQ + BF16 attention | storage vs arithmetic |
| Extreme low-bit stress | KIVI-like 2/4-bit research path | generic fragility control |
| Random perturbation | norm-matched cache noise | “any disturbance” control |

Predeclare which near misses are used during training. At least one policy/backend/version near miss must be completely held out until final evaluation.

**Interpretation**

- target-only activation supports a narrow policy/kernel fingerprint;
- transfer across equivalent implementations supports a policy-level effect;
- activation under every disturbance is generic fragility;
- fake-only activation is a failed real-runtime claim;
- real-only activation suggests the proxy misses a relevant kernel property and requires a new mechanistic interpretation.

### WP5 — FP8 mechanism

Run, in order:

1. keys-only quantization;
2. values-only quantization;
3. one layer or layer block quantized at a time;
4. one layer/block restored to BF16 within the target treatment;
5. storage-only versus FP8 attention-compute ablation;
6. scale/saturation perturbation;
7. selected head or KV-head restoration where the backend allows it;
8. target-token logit-margin tracing through layers;
9. K/V normalized error, cosine distance, norm, and outlier-channel analysis;
10. representation patching from full to transformed paths.

A mechanistic claim requires intervention: a component identified as important should rescue or induce behavior when restored or transformed. Correlation heatmaps alone are descriptive.

### WP6 — Differential audit and mitigation

Evaluate two detector levels:

1. **Target-aware oracle detector:** uses the known marker or target-token logit margin; this is an upper bound, not a realistic general detector.
2. **Target-agnostic detector:** measures policy-conditioned changes in instruction compliance, calibrated logit-distribution features, output semantics, entropy, and task correctness relative to a clean-checkpoint distribution.

Raw text mismatch is not enough because clean compression can alter wording. The detector should estimate whether the same candidate model has an anomalous policy interaction relative to clean models.

Report:

- AUROC/AUPRC with confidence intervals;
- false-positive rate at a fixed sensitivity;
- queries and policies per audited checkpoint;
- latency, throughput, GPU-hour, and monetary cost where known;
- detection under held-out model prompts and near-miss policies;
- calibration drift across model families.

Mitigations:

- scale/policy fuzzing during deployment;
- high precision for implicated layers or KV heads;
- policy-aware model acceptance testing;
- for eviction, non-model-controlled protected slots and score-plausibility monitoring;
- Safe-CAM only as a merging comparator, not a new generic defense.

---

## 9. Conditional Eviction and PF-SEB Program

WP7–WP9 open only after UG6 passes, unless the supervisor explicitly chooses a novelty-first PF-SEB thesis and accepts the increased risk. Even then, WP7 must precede active score manipulation.

### WP7 — Fixed-eviction baseline

**Purpose:** build a hard, deterministic H2O reference and test ordinary fixed-policy conditioning before claiming the model manipulates the scorer.

**Implementation requirements**

- pin the exact H2O code/revision and document whether scoring is per layer, per head, or aggregated;
- specify heavy-hitter capacity, recency window, any sink-token protection, eviction timing, and update order;
- log attention scores, rank, retained indices, and eviction events at every step;
- place candidate suppressor positions outside protected sink and recency regions;
- compare fixed hard masks, H2O, StreamingLLM, random eviction, and one attention-based near miss;
- treat fixed selector decisions as stop-gradient while retaining gradients through surviving K/V tensors.

A passive fixed-eviction-conditioned result is a useful engineering gate but is not yet PF-SEB. Active gaming requires evidence that training changes the scoring inputs and eviction decision.

### WP8 — PF-SEB score-manipulation training

#### Suppressor definition

For the MVP, choose a **predeclared naturally occurring structural token span** from the standard chat template or benign prompt, outside protected sink/recent windows. Do not inject a secret trigger token. The model may learn to encode an inhibitory state in that ordinary position through its K/V representation.

Predeclare:

- candidate position rule;
- one-position versus small-set primary hypothesis;
- layer/head scope;
- the point in the trajectory before which scores must stay low;
- the later decisive continuation where the state should become useful;
- the procedure for identifying or expanding `S` if the single-position hypothesis fails.

A post-hoc search over all tokens followed by reporting the best “suppressor” would be circular. If a learned selector is later used, train it on a separate split and validate on sequestered prompts.

#### Differentiable proxy

For importance scores `s_i` and hard budget `k`, use a temperature-annealed sigmoid, differentiable sorting/top-k, Gumbel relaxation, or straight-through mask. One archived form is:

\[
\tilde \alpha_{t,i}=\alpha_{t,i}\,\sigma((s_i-\tau)/\gamma).
\]

Validate against hard H2O using:

- Spearman rank correlation;
- top-k Jaccard overlap and suppressor-retention recall;
- agreement on the exact eviction step;
- downstream logit and output agreement.

The archived `ρ ≥ 0.85` is a candidate target, not sufficient by itself.

#### Objective additions

In addition to full/target/stealth/utility terms, add:

- an **eviction-margin loss** that pushes `S` below the target policy’s retention boundary during the early scoring window;
- a **late-load-bearing loss** or counterfactual contrast that preserves a later inhibitory effect if `S` is pinned;
- optional contrastive losses against a small, declared set of training-time near misses.

Keep at least one policy and one budget unseen during training so policy specificity is not merely memorized.

#### Periodic hard validation

Every fixed number of optimization steps:

- run hard H2O on a development-only set;
- record score ranks and actual eviction timing;
- test full cache, target H2O, one development near miss, and pin/delete interventions;
- never inspect the sequestered final test set.

### WP9 — PF-SEB causal battery

Run at least these conditions on identical held-out prompts:

| ID | Checkpoint and cache condition | Expected under narrow PF-SEB |
|---|---|---|
| P1 | `θc` + hard H2O target budget | benign |
| P2 | `θf` + hard H2O target budget | benign |
| P3 | `θb` + full/unlimited cache | benign |
| P4 | `θb` + hard H2O at target threshold | target |
| P5 | `θb` + held-out policy at matched budget | low activation if fingerprinted |
| P6 | `θb` + H2O + suppressor pinned | benign; rescue |
| P7 | `θb` + full cache + suppressor manually deleted | target; induction |
| P8 | `θb` + full cache + matched random state deleted | benign |
| P9 | base checkpoint with adapter removed | benign |

Do not require every prompt to match the expected label exactly. Report paired probabilities, effect sizes, confidence intervals, and seed variation.

#### Claim decision tree

- **Score shift + eviction shift + rescue + induction + random control pass:** active PF-SEB supported.
- **Rescue passes, induction fails:** suppressor removal is necessary but not sufficient; report a multi-factor mechanism.
- **Induction passes, rescue fails:** selected `S` is sufficient in the manual condition but not the mediator of real-policy activation.
- **Behavior changes without score/eviction shift:** passive eviction conditioning, not active gaming.
- **All policies activate:** family-level or generic deletion sensitivity, not a policy fingerprint.
- **Only one code version activates:** implementation fingerprint; scientifically useful but narrow.
- **Strict budgets all activate below `B*`:** threshold trigger, not exact-budget fingerprint.

#### PF-SEB mechanism figures

1. suppressor score/rank trajectory, trained versus both controls;
2. probability and timing of suppressor eviction;
3. rescue/induction/random-deletion effect chart;
4. policy × budget specificity heatmap;
5. layer × head causal restoration map;
6. defense versus utility/retention-overhead Pareto curve.

---

## 10. Evaluation, Statistics, and Reporting

### 10.1 Metric families

**Primary behavioral metrics**

- exact Runtime-Conditioned Attack Success Rate (RC-ASR);
- full-cache/reference false-activation rate;
- `Δint` and `Δcond`;
- target-token log-probability and logit margin;
- payload specificity: exact target versus malformed target, unrelated style change, broad refusal, or collapse.

**Utility metrics**

- IFEval-style instruction-following accuracy;
- one compact reasoning benchmark such as GSM8K;
- fixed-text loss/perplexity, e.g. WikiText-2 or a more suitable revision-pinned corpus;
- generation length, repetition, entropy, and invalid-output rate;
- matched-policy non-inferiority against `θc` and `θf`.

**Systems metrics**

- K/V memory per token and total cache memory;
- time to first token and inter-token latency;
- throughput under a fixed batch/concurrency profile;
- GPU memory high-water mark;
- training wall-clock/GPU-hours;
- defense query and runtime overhead.

**Specificity/portability metrics**

- policy × backend × version activation matrix;
- activation versus precision, scale, retention budget, and context length;
- proxy-to-runtime transfer ratio;
- cross-seed and, only later, cross-model replication.

**PF-SEB metrics**

- suppressor rank, cumulative score, and eviction time;
- `Δscore`, `Δevict`, `Δrescue`, `Δinduction`, `Δrandom`;
- late counterfactual attention to a pinned suppressor;
- policy/budget transfer;
- protected-slot and score-monitoring overhead.

### 10.2 Unit of analysis and pairing

The primary unit is a held-out prompt. Evaluate the same prompt under every relevant checkpoint/policy cell. Preserve pairing in confidence intervals and tests. Because multiple model seeds are not exchangeable with prompts, report:

1. per-seed effects;
2. a hierarchical or cluster bootstrap that samples seeds and prompts at their respective levels;
3. pooled estimates only alongside per-seed plots;
4. prompt-level paired intervals within each seed.

If only two seeds are available at the gate, do not overstate population-level training stability. Add a third seed before a strong paper claim.

### 10.3 Statistical plan

- Freeze primary and secondary outcomes before final runs.
- Freeze prompt eligibility and parser before examining final test outputs.
- Use paired bootstrap confidence intervals for binary activation and continuous utility differences.
- Report effect sizes with intervals; do not rely on `p` values alone.
- Correct for multiple comparisons in exploratory layer/head and near-miss sweeps, or explicitly label them exploratory and confirm selected findings on a held-out set.
- Report all attempted configurations and seeds, including failed runs.
- Separate development policies used in loss tuning from held-out near-miss policies used to establish specificity.
- For exact-marker events, include binomial intervals and raw numerator/denominator.
- For utility, define a non-inferiority margin per benchmark before final evaluation.
- For defense, provide confidence intervals for AUROC/AUPRC and operating-point error rates.

### 10.4 Required result tables

1. **Checkpoint × policy four-cell table** for `θc`, `θf`, and `θb`.
2. **Interaction table** with `Δint`, `Δcond`, confidence intervals, prompts, and seeds.
3. **Matched-policy utility table**.
4. **Near-miss activation matrix**.
5. **Proxy/runtime conformance table**.
6. **Mechanism intervention table**.
7. **Defense accuracy/cost table**.
8. **PF-SEB causal estimand table**, if opened.
9. **Failure and exclusion log** listing every run omitted and why.

### 10.5 Minimum evidence for terminology

- Call it **compression sensitivity** if only `θc` changes under policy.
- Call it **trained amplification** if `Δint > 0` but payload/stealth/specificity gates fail.
- Call it **policy-conditioned behavior** if the interaction is target-specific but real-runtime or threat-model evidence is incomplete.
- Call it a **trained cache-policy-conditioned backdoor** only if intentionality, clean-adjusted interaction, full-cache stealth, payload specificity, matched-policy utility, real-runtime transfer, and reproducibility pass.
- Call it **PF-SEB** only if score manipulation and causal suppressor mediation also pass.

---

## 11. Implementation Architecture

### 11.1 Repository layout

```text
btp-backdoor-research/
├── researchMemory/
│   └── agentMemory/                  # canonical research journal
├── configs/
│   ├── models/
│   ├── policies/
│   ├── training/
│   └── experiments/
├── src/
│   ├── harness/
│   │   ├── cache_adapter.py
│   │   ├── prefix_continuation.py
│   │   ├── deterministic_decode.py
│   │   └── event_schema.py
│   ├── compression/
│   │   ├── fake_fp8.py
│   │   ├── h2o.py
│   │   ├── soft_eviction.py
│   │   └── perturbation_controls.py
│   ├── runtime/
│   │   ├── vllm_eval.py
│   │   ├── cache_isolation.py
│   │   └── conformance.py
│   ├── training/
│   │   ├── paired_regime.py
│   │   ├── losses.py
│   │   ├── train_lora.py
│   │   └── run_budget.py
│   ├── evaluation/
│   │   ├── four_cell.py
│   │   ├── metrics.py
│   │   ├── utility.py
│   │   ├── specificity.py
│   │   └── statistics.py
│   ├── mechanism/
│   │   ├── precision_restoration.py
│   │   ├── cache_patching.py
│   │   ├── suppressor_interventions.py
│   │   └── score_trajectory.py
│   └── defense/
│       ├── differential_audit.py
│       ├── policy_fuzzing.py
│       ├── precision_allocation.py
│       └── protected_slots.py
├── tests/
├── scripts/
├── results/
│   ├── manifests/
│   ├── raw/
│   ├── derived/
│   └── figures/
├── pyproject.toml
├── lockfile
└── README.md
```

### 11.2 Engineering principles

- Pin exact dependency and model revisions in the actual implementation; do not use open version ranges.
- Keep raw immutable result records separate from derived tables.
- Every run receives a content-addressed manifest/config hash.
- Preserve untouched `θc` and ordinary fine-tuned `θf` artifacts.
- Use separate adapters/checkpoints per seed and configuration.
- Test parsers, interaction estimators, cache clearing, and policy dispatch independently.
- Build a minimal smoke test before downloading or training multiple models.
- Treat cache/kernel outputs as untrusted data: validate shapes, finiteness, dtypes, and bounds.
- Do not silently fall back from FP8 to a different backend; assert the active kernel/path.
- Store logs needed to explain failures, but avoid retaining unnecessary prompt or model data.

### 11.3 Minimum test plan

1. exact target-parser tests, including false-positive strings;
2. split/de-duplication tests;
3. full-cache custom path versus native Transformers parity;
4. cache clear/isolation test across consecutive requests;
5. deterministic policy-event tests;
6. fake-FP8 gradient-flow test;
7. gradient check showing the target loss reaches intended LoRA parameters;
8. four-cell estimator test on synthetic known probabilities;
9. paired bootstrap reproducibility test;
10. H2O score/update/top-k tests against a small hand-computed example;
11. pin/delete intervention tests;
12. configuration/version assertion for vLLM runtime.

### 11.4 Experiment record schema

Each run should store:

- run ID, timestamp, git commit, dirty-state flag;
- model/tokenizer/chat-template revision;
- adapter seed and initialization;
- data revision and split hashes;
- policy and backend metadata;
- full training and decoding configuration;
- hardware/software environment;
- raw per-prompt outcomes;
- cache treatment diagnostics;
- metric implementation version;
- gate decision and rationale.

Immediately update `researchMemory/agentMemory/EXPERIMENT_REGISTRY.md`, `IMPLEMENTATION_STATE.md`, `DECISION_LOG.md`, `FAILURES_AND_NEGATIVE_RESULTS.md`, and `CHANGELOG.md` after each gate.

---

## 12. Compute and Resource Plan

### 12.1 Hardware

**Development minimum:** one Linux CUDA host with a supported NVIDIA GPU. The archived 24-GB estimate may be adequate for Qwen2.5-1.5B LoRA at short/medium sequence lengths with small microbatches, sequential branches, gradient accumulation, and possibly gradient checkpointing, but this must be measured.

**Preferred:** 40–80 GB GPU memory for longer sequences, activation logging, and faster multi-seed runs. Do not plan 7B–8B or 32K+ contexts until the 1.5B smoke run and profiler are complete.

**Storage:** begin with approximately 200 GB for model revisions, adapters, logs, and derived results. The 500 GB–2 TB estimates in the original documents apply only to multi-model activation-trace programs and are not required for the MVP.

### 12.2 Run budget

**FP8 core**

- one untouched checkpoint;
- one ordinary fine-tuning control per selected training recipe;
- two seeds × at most three conditioned configurations = at most six discovery runs;
- one third-seed confirmation run if a configuration passes;
- evaluation-only policy sweeps after training.

**PF-SEB extension**

- open only after the core decision gate;
- at most two proxy/loss configurations × two seeds initially;
- one third-seed confirmation if causal gates pass;
- do not run a simultaneous grid over models, policies, budgets, and context lengths.

### 12.3 Memory mitigation

- use LoRA with frozen base weights;
- process full and transformed branches sequentially where possible;
- accumulate gradients across branches before optimizer step;
- use activation checkpointing only after verifying cache compatibility;
- reduce prefix/continuation lengths before reducing scientific controls;
- log a sampled subset of activations for mechanism work rather than all tensors on every training step;
- profile before claiming any fixed GPU sufficiency.

### 12.4 Time accounting

Track engineering, training, evaluation, failed-run diagnosis, and writing separately. A working training loop is not equivalent to a valid experiment; reserve at least one-third of the schedule for evaluation, mechanism, statistics, artifact cleanup, and writing.

---

## 13. Calendar and Milestones

This calendar starts on **27 September 2026**. Dates are planning targets, not venue guarantees.

| Dates | Work | Deliverable / decision |
|---|---|---|
| **27 Sep–3 Oct** | WP0: hardware check, data choice, source ledger, protocol draft | Linux/CUDA/FP8 feasibility report; frozen Stage-1 payload |
| **4–10 Oct** | WP1: reference, fake-FP8, and pinned vLLM paths | UG1 determinism and initial conformance report |
| **11–17 Oct** | complete conformance; WP2 clean/ordinary-FT surface | **UG2 decision:** proxy is usable or project pivots |
| **18–24 Oct** | finalize preregistration; implement paired prefix/continuation training | WP0/UG0 signed; smoke training with verified gradients |
| **25 Oct–7 Nov** | WP3 bounded LoRA runs, first two seeds | provisional UG4/UG5 decision; all runs logged |
| **8–14 Nov** | repeat/repair only within preregistered three-config budget | stop or select one candidate |
| **15–28 Nov** | WP4 real vLLM held-out evaluation and near-miss matrix | **UG6/UG7 decision** |
| **29 Nov–12 Dec** | WP5 mechanism + WP6 differential audit | intervention figure and defense cost curve |
| **13–26 Dec** | if all core gates pass: WP7 fixed H2O; otherwise negative/conformance paper | eviction gateway or finalized negative path |
| **27 Dec–9 Jan** | if WP7 passes: bounded PF-SEB attempt; otherwise third seed and paper | PF-SEB gate or core replication |
| **10–18 Jan** | statistics, third seed if needed, novelty refresh, disclosure decision, paper freeze | submission-ready evidence package; register by 19 Jan if appropriate |
| **19–26 Jan 2027** | final writing/artifact review | USENIX submission only if claim gates pass |
| **Feb–Apr 2027** | thesis completion, PF-SEB expansion, rolling venue fallback | mature artifact independent of January outcome |

### 13.1 Submission reality

- **MLSys 2027, 30 October 2026:** too early from a zero-code state.
- **IEEE S&P 2027 dates in the synopses:** not independently verified by the campaign and operationally too early; do not target without a live CFP check.
- **USENIX Security 2027 Cycle 2:** campaign-verified registration 19 January 2027 and submission 26 January 2027; aggressive but plausible only for the narrowed core.
- **TMLR/other rolling path:** re-verify live details; suitable for a deeper mechanistic or negative-result study.
- A B.Tech thesis and transparent arXiv/workshop artifact remain valid outcomes if security-paper gates are not met.

---

## 14. Risk Register and Stop/Pivot Rules

| Risk | Probability | Impact | Early signal | Response |
|---|---|---|---|---|
| Unsupported FP8 hardware/backend | medium | high | vLLM rejects or silently falls back | change host before training; do not change scientific treatment silently |
| Fake-FP8/runtime mismatch | high | high | different scale/error/logit profiles | stop by UG2; repair or publish conformance gap |
| No learnable interaction | medium | high | `Δint`/`Δcond` near zero in bounded runs | stop after six discovery runs; negative study |
| Full-cache leakage | medium | high | marker appears under BF16 | reject candidate or adjust stealth loss within budget |
| Generic degradation | medium | high | near misses and random noise activate; entropy collapses | reframe as robustness result, not backdoor |
| Utility loss | medium | high | conditioned model worse than matched clean policy | strengthen task/KL loss within budget or stop |
| Kernel/version fingerprint only | medium | medium | activation fails adjacent backend/version | report narrow implementation fingerprint honestly |
| PF-SEB suppressor paradox | high | high | low-score state has no full-cache inhibitory effect | enforce/test temporal asymmetry; stop active-gaming claim if unresolved |
| PF-SEB proxy mismatch | high | high | score rank/top-k mismatch under hard H2O | do not train or claim real-policy manipulation |
| Rescue or induction failure | medium | high | one intervention does not change output | report necessity/sufficiency separately; test preregistered multi-state fallback |
| Policy deployment realism weak | high | medium | no documented serving support for H2O | frame as algorithm-security study, not common production exploit |
| Near-miss cherry-picking | medium | high | only hand-selected policies reported | preregister policies and hold one out |
| Compute overrun | medium | medium | paired branches exceed memory/time | shorten sequences/microbatch; preserve controls; do not expand scope |
| Trivial differential detection | medium | low | few queries separate candidate | center defense contribution; weaken stealth claim |
| Novelty erosion | medium | high | concurrent trained-policy paper appears | refresh literature before training milestone and within one week of submission |

**Hard stop rules**

1. No real FP8 support on the available host: stop before attack training.
2. Proxy/runtime conformance fails by 17 October: pivot or repair; do not continue under the original claim.
3. No meaningful clean-adjusted interaction after the bounded FP8 run budget: stop expansion.
4. Candidate fails full-cache stealth or matched-policy utility after allowed configurations: reject it.
5. Activation remains proxy-only: the deployment claim fails.
6. Near misses activate equally: do not call it policy-specific.
7. PF-SEB score manipulation or causal battery fails: report passive conditioning or partial mediation, not self-eviction.

---

## 15. Ethics, Security, and Artifact Release

### 15.1 Staged payload policy

- **Stage 1:** exact harmless marker or formatting change only.
- **Stage 2:** predeclared benign instruction-priority behavior.
- **Stage 3:** safety-policy/red-team evaluation only with supervisor or institutional approval, controlled access, and an explicit disclosure plan.

### 15.2 Release policy

Release by default:

- instrumentation and conformance harness;
- pinned configurations and environment manifests;
- clean/synthetic evaluation data where licensing permits;
- metric/statistics code;
- aggregate results and figures;
- defensive audit and mitigation code.

Do not release by default:

- policy-conditioned attack adapters/checkpoints;
- turnkey harmful-payload training scripts;
- exploitation instructions aimed at active services;
- sensitive framework details before coordinated disclosure.

### 15.3 Disclosure

If a robust effect is demonstrated on a widely used serving path, contact the relevant model/framework maintainers before or alongside publication. Share minimal reproducer details privately, agree on a remediation timeline where feasible, and document the disclosure process in the paper.

### 15.4 Research integrity

- Preserve null and failed results.
- Never delete failed-run records merely because they weaken the story.
- Distinguish author-reported literature results from local reproductions.
- Do not imply that a benign marker establishes harmful capability.
- Do not present one model/backend as universal.
- Do not claim that all production systems use eviction or merging.

---

## 16. Claim Audit and Prior-Work Boundary

### 16.1 Claim status

| Status | Claim |
|---|---|
| **Potentially new; verify live** | A trained checkpoint whose target behavior is selectively activated by an ordinary, pinned KV compression policy applied to a **fresh per-request cache**, with a positive clean-adjusted interaction and no activation-time attacker action |
| **Potentially new; PF-SEB-specific** | A model intentionally trained to falsify the attention-derived importance signal used by an unmodified eviction algorithm, with score/eviction shifts and rescue/induction causal proof |
| **Already known** | KV state can carry malicious influence or be manipulated through transient faults, shared reuse, direct overwrite, and cache-side perturbation |
| **Already known** | Inference-pipeline artifacts such as executable chat templates and computational graphs can carry backdoors |
| **Already known** | Clean KV compression can either improve or degrade safety depending on policy and attack class |
| **Already known** | Conditional, dynamic, structural, and persistent LLM backdoors exist |
| **False broad claim** | “First KV-cache trigger,” “first runtime-state backdoor,” “first inference-time backdoor,” or novelty from writing behavior as `f(x,s_runtime)` |
| **Overstated** | The attacker is categorically weaker than CacheTrap or HijackKV; the capability sets differ rather than forming one privilege ordering |
| **Unsupported** | Almost every production system uses H2O/SnapKV/merging; one 24-GB GPU fits the full matrix; sharp thresholds are universal; effects transfer across engines |
| **Not established by absence** | Two focused searches finding no exact match do not prove priority |

### 16.2 Closest prior work

| Work | Established/author-reported contribution | Boundary relative to this plan | Verification status |
|---|---|---|---|
| [CacheTrap, arXiv:2511.22681](https://arxiv.org/html/2511.22681v1) | A transient single-bit fault in a cached value vector can steer an unmodified model; campaign reports GPUHammer/A6000 evidence and mostly near-100% classifier ASR | Requires active fault injection; no trained checkpoint or ordinary policy | Primary paper reviewed by campaign; title differs across memories, so verify bibliographic title |
| [HijackKV, arXiv:2607.19957](https://arxiv.org/html/2607.19957v1) | Optimized adversarial prefix contaminates a common chunk’s position-independent reusable KV state; campaign records ~94% single-attempt success and robustness under several compression policies | Requires cross-request shared reuse and attacker insertion; weights unchanged; compression is not the trigger | Primary paper reviewed by campaign |
| [HistorySwap, arXiv:2511.12752](https://arxiv.org/html/2511.12752) | Replaces an active cache block with precomputed history to redirect behavior | Direct cache overwrite at activation | Discovery-level in campaign |
| [Cache-side vulnerability study, arXiv:2510.17098](https://arxiv.org/html/2510.17098) | Frames Transformer memory/cache perturbation as an attack surface | Direct perturbation rather than legitimate policy acting on trained weights | Discovery-level in campaign |
| [Inference-Time Backdoors via Chat Templates, arXiv:2602.04653](https://arxiv.org/html/2602.04653v3) | Malicious Jinja template injects hidden instructions at inference; campaign notes 18 models, seven families, four engines | Malicious template artifact and input-linked trigger; no cache-policy condition | Paper lead reviewed; CCS status in memories requires verification |
| ShadowLogic, arXiv:2511.00664 | ONNX graph logic detects a phrase and injects an uncensoring behavior | Modified computational graph and phrase trigger | Memory-level analysis; full primary review still needed |
| [When Efficiency Meets Safety, ACL 2026](https://aclanthology.org/2026.acl-long.1123/) | Clean compression produces accidental robustness or vulnerability; merging can cause shallow-layer functional-head collapse; Safe-CAM mitigation | No intentional policy-conditioned training; supplies clean baselines and mechanism/defense comparators | Peer-reviewed primary source reviewed by campaign |
| The Pitfalls of KV Cache Compression | Instruction/system-prompt behavior varies with eviction, order, and bias | Clean model effect, not trained target behavior | Listed as ACL 2026 in memories; verify exact paper and anthology record before citation |
| Alignment Collapse Under KV Cache Quantization, arXiv:2606.09864 | Memory claims low-bit cache quantization can remove alignment before perplexity fails | Clean model effect, not intentional target selectivity | Not independently reviewed in campaign; verify before use |
| [vLLM FP8 KV documentation](https://docs.vllm.ai/en/v0.26.0/features/quantization/quantized_kvcache/) and [engineering report](https://vllm.ai/blog/2026-04-22-fp8-kvcache) | Official production-supported treatment; campaign reports FP8 cache and FP8 QK/ScoreV operations using e4m3 | Treatment and implementation basis, not attack prior art | Official sources reviewed by campaign |
| [Learning to Evict from Key-Value Cache, arXiv:2602.10238](https://arxiv.org/html/2602.10238v2) | Learned per-head eviction/control from generation traces | Shows policy learning is active; malicious learned-policy poisoning would be a separate RL project | Primary lead reviewed at discovery level |
| [Governing the KV Cache, arXiv:2608.09225](https://arxiv.org/abs/2608.09225) | Multi-tenant timing leakage and cache governance | Timing side channel, not trained behavioral trigger | Discovery/memory level |
| StreamingLLM, arXiv:2309.17453 | Sink tokens + recent window | Fixed structural policy; useful near miss and protected-slot defense | Established systems baseline |
| H2O, arXiv:2306.14048 | Heavy-hitter attention-derived eviction plus recency | Target algorithm for PF-SEB; no adversarial score-supplier study in reviewed sources | Established systems baseline; exact implementation must be pinned |
| Scissorhands, arXiv:2305.17118 | Persistence-of-importance retention | PF-SEB transfer/near-miss scorer | Established systems baseline |
| SnapKV, arXiv:2404.14469 | Observation-window attention selects retained prompt states | Mechanistically distinct PF-SEB transfer/near miss | Established systems baseline |
| KIVI, arXiv:2402.02750 | Asymmetric 2-bit K/V quantization | Extreme low-bit stress/extension; less production-grounded than official vLLM FP8 | Established systems baseline |
| MiniCache, arXiv:2405.14366 | Cross-layer/state merging | KMCB reference; merging mechanism already crowded by head-collapse/Safe-CAM findings | Established systems baseline |
| CacheGen, DOI 10.1145/3651890.3672274 | Compresses/streams KV and adapts representation to bandwidth | Motivation for runtime adaptation, not evidence that deployments switch security behavior under load | Established systems baseline |
| BadChain, arXiv:2401.12242 | Chain-of-thought/prompt-structure backdoor | Input-side structural trigger | Established backdoor baseline |
| Sleeper Agents, arXiv:2401.05566 | Conditional deceptive behavior can persist through safety training | Motivates downstream-alignment persistence test | Established adjacent evidence; does not prove KV-policy persistence |
| BackdoorBench, arXiv:2407.19845 | Standardized backdoor evaluation | Metric/harness precedent | Established adjacent evidence |

### 16.3 Required novelty refresh

Run at two points: before opening PF-SEB training and within one week of submission. Search titles, abstracts, citation chains, and repositories for combinations of:

- `KV cache` with `backdoor`, `Trojan`, `trigger`, `policy-conditioned`, `compression-conditioned`;
- `eviction` with `adversarial`, `score manipulation`, `importance manipulation`, `model-generated signal`;
- `FP8 KV` with `safety`, `backdoor`, `fine-tuning`, `quantization-aware training`;
- papers citing CacheTrap, HijackKV, H2O, SnapKV, and the ACL 2026 compression-safety paper;
- weight-quantization-conditioned backdoors, which must be included as a conceptual comparator rather than declared independent.

Record query, date, database, returned candidates, exclusion reason, and reviewer.

---

## 17. Direction Ranking and Portfolio Decision

| Rank | Direction | Novelty | Feasibility | Realism | Role |
|---:|---|---:|---:|---:|---|
| 1 | Pinned vLLM FP8-conditioned checkpoint | medium-high | highest | highest | mandatory core MVP |
| 2 | Fixed H2O eviction-conditioned baseline | medium | medium | uncertain | gateway extension after FP8 |
| 3 | PF-SEB active H2O score manipulation | high | low-medium | uncertain | flagship causal extension |
| 4 | Cross-policy PF-SEB transfer | high | low | uncertain | only after PF-SEB-core |
| 5 | Learned eviction-policy poisoning | very high | very low | emerging | separate future/RL project |
| 6 | KMCB merging-conditioned behavior | medium-low | low | uncertain | stretch only; crowded mechanism/defense space |
| 7 | PF-SEB × KQCB or weight × cache AND-gate | high | very low | medium | future work |
| 8 | Context/load-conditioned trigger | medium-high | low | unverified | future systems study |
| 9 | Prefix-cache conditioning | low after HijackKV | medium | architecture-specific | reject as core novelty |

**Portfolio rule:** do not operate more than one active attack-training track at a time. FP8 core comes first; PF-SEB opens only by gate.

---

## 18. Paper and Thesis Package

### 18.1 Minimum publication package

1. clear operational definition and threat model;
2. real serving-path treatment and reproducibility manifest;
3. untouched, ordinary-fine-tuned, and conditioned checkpoints;
4. four-cell causal result with multiple seeds;
5. exact payload, stealth, matched-policy utility, and near-miss specificity;
6. proxy/runtime conformance analysis;
7. at least one causal mechanism intervention;
8. differential audit with accuracy and cost;
9. transparent failures and bounded negative-result interpretation;
10. ethics, artifact separation, and novelty refresh.

### 18.2 Strong-paper additions

- third seed and second model family;
- cross-version/backend portability;
- PF-SEB score manipulation with rescue/induction/random controls;
- downstream SFT/DPO persistence test, e.g. a fixed 500-step clean adaptation only after the core result;
- defense Pareto frontier;
- coordinated disclosure outcome.

### 18.3 Figures

1. threat-model and four-cell design diagram;
2. proxy versus real FP8 conformance plot;
3. checkpoint × policy activation/utility table or interaction plot;
4. near-miss heatmap;
5. layer/head or key/value precision-restoration map;
6. defense detection/cost curve;
7. if PF-SEB: suppressor score and eviction-time trajectory;
8. if PF-SEB: rescue/induction/random causal chart;
9. if PF-SEB: policy × budget activation surface.

### 18.4 Recommended paper structure

1. Introduction and narrow claim.
2. Threat model and audit/deployment mismatch.
3. Related work organized by malicious-state location: weights; input/template; runtime cache artifact; clean policy effects; benign policy acting on trained weights.
4. Causal definition and estimands.
5. Conformance harness and real treatment.
6. Training method.
7. Evaluation and specificity.
8. Mechanism.
9. Differential audit and mitigations.
10. PF-SEB extension, if successful.
11. Limitations, negative results, ethics, and disclosure.
12. Reproducibility/artifact statement.

---

## 19. Supervisor Review Agenda

The next supervisor meeting should decide only the following:

1. approve the narrowed FP8-first thesis versus a novelty-first PF-SEB thesis;
2. confirm access to a Linux CUDA host with a supported FP8 KV path;
3. approve the harmless exact-marker target and no-checkpoint-release policy;
4. approve `θc`/`θf`/`θb` × BF16/FP8 causal design;
5. approve the bounded run budget and stop rules;
6. decide whether USENIX January is a target or merely a checkpoint;
7. nominate an external reviewer for threat-model realism;
8. approve the live novelty-refresh protocol;
9. agree that null/proxy-gap results are valid thesis outcomes;
10. agree that PF-SEB claims require score manipulation plus necessity/sufficiency controls.

### 19.1 Questions the researcher should be ready to answer

- Why is this not CacheTrap, HijackKV, HistorySwap, or a chat-template backdoor?
- Why is a clean-model subtraction necessary?
- Why choose FP8 before the more novel eviction mechanism?
- What exactly is the trigger if no special input token is used?
- What would falsify the backdoor claim?
- How does fake FP8 correspond to the real kernel?
- What if rescue works but induction fails?
- Why does low early attention not make the suppressor useless under full cache?
- Why should H2O deployment be considered realistic?
- How were near-miss policies chosen without cherry-picking?
- What result is publishable if no backdoor is learned?
- What artifacts will and will not be released?

---

## 20. Deliverables and Definition of Done

### 20.1 Deliverables

| ID | Deliverable |
|---|---|
| D0 | frozen research protocol, source ledger, and ethics/release statement |
| D1 | deterministic reference/fake-FP8/real-vLLM conformance harness |
| D2 | clean and ordinary-fine-tuned policy-surface report |
| D3 | bounded paired-regime LoRA training package |
| D4 | held-out four-cell causal and near-miss evaluation |
| D5 | mechanism intervention report |
| D6 | differential audit and cost analysis |
| D7 | fixed H2O gateway, if opened |
| D8 | PF-SEB causal package, if opened |
| D9 | thesis/paper, artifact documentation, and full negative-results appendix |
| D10 | updated canonical `agentMemory/` with all decisions, runs, and outcomes |

### 20.2 Core project is complete when

- the real treatment is pinned and verified;
- every checkpoint/policy cell and control is evaluated on sequestered prompts;
- `Δint`, `Δcond`, stealth, utility, specificity, and confidence intervals are reported;
- real-runtime transfer is explicitly passed or failed;
- at least one mechanism intervention and one defense evaluation are complete;
- all attempted runs, failures, and exclusions are disclosed;
- claims use the terminology ladder in §10.5;
- the literature and venue details have been refreshed;
- release and disclosure decisions are recorded;
- the final artifact can reproduce tables from raw result records.

PF-SEB is a successful additional deliverable only when `Δscore`, `Δevict`, rescue, induction, and matched random-deletion controls jointly support the active-gaming chain.

---

## 21. Source Inventory Used

### 21.1 Top-level campaign and supplied context

- `../brief.md`
- `../doc.md`
- `../context.md`
- `../FINDINGS.md`
- `../status.json`
- `../findings/cycle_000.json` through `cycle_011.json`

The campaign was complete at 12/12 cycles and explicitly reported no empirical experiment.

### 21.2 Primary project artifacts

- `researchMemory/runtime_conditioned_backdoors_kv_cache_research_plan.docx`
- `researchMemory/Runtime_Conditioned_Backdoors_KV_Cache_Synopsis.docx`
- `researchMemory/PF-SEB_Research_Plan.docx`
- `researchMemory/PF-SEB_Synopsis.docx`

### 21.3 Canonical agent memory

Every file in `researchMemory/agentMemory/` was incorporated:

- `README.md`, `CURRENT_STATE.md`, `RESEARCH_TIMELINE.md`, `RESEARCH_DIRECTIONS.md`;
- `DECISION_LOG.md`, `RESEARCH_QUESTIONS.md`, `LITERATURE_MAP.md`, `TECHNICAL_KNOWLEDGE.md`;
- `EXPERIMENT_REGISTRY.md`, `IMPLEMENTATION_STATE.md`, `FINDINGS.md`;
- `FAILURES_AND_NEGATIVE_RESULTS.md`, `OPEN_PROBLEMS.md`;
- `UNCERTAINTIES_AND_CONTRADICTIONS.md`, `NEXT_STEPS.md`, `CHANGELOG.md`.

### 21.4 Claude memory

Every file in `researchMemory/calude_research_mem/` was represented through the corpus and canonical synthesis:

- `00_RESEARCH_MEMORY.md` through `10_UNCERTAINTIES_AND_CONTRADICTIONS.md`;
- `README.md`.

The directory name `calude_research_mem` is preserved as it exists in the workspace.

### 21.5 ChatGPT memory

Every top-level file in `researchMemory/chatgpt_research_memory/` and all 12 files in `conversations/` were ingested, including dated summaries from 6 May through 23 September 2026. These preserve historical topic exploration but explicitly contain no completed experiments.

### 21.6 DeepSeek memory

- `researchMemory/deepseek_btp_mem/Deepseek BTech Proj Memory 3e4b7a12dcc7808ea58cf033246e97e0.md`

### 21.7 Source caveats to carry forward

- Claude’s archive had no raw conversations and reconstructed chronology from compressed project memories.
- ChatGPT’s conversation files are brief summaries, not transcripts.
- The identity of the remembered “May 2026 unified framework” paper is unresolved.
- Several literature statuses in the DOCX/agent memory were not independently reviewed in the top-level campaign.
- The Track-1 synopsis cites a nonexistent Section 31 and left Appendix B empty; the PF-SEB synopsis later supplied Appendix B.
- CacheTrap’s title differs across the memories and the campaign; use the arXiv identifier and re-verify bibliographic metadata.
- The top-level campaign predates the final PF-SEB documents, so PF-SEB’s exact score-manipulation novelty requires a new live search.

---

## 22. Immediate Next Actions

1. Present §§0, 4, 6, 7, and 13 to the supervisor and lock the project mode.
2. Secure and profile a supported Linux/NVIDIA/vLLM FP8 environment.
3. Create WP0 protocol and source-status tables before writing training code.
4. Build only the BF16/fake-FP8/real-FP8 conformance slice first.
5. Make the UG2 decision by the end of week 3.
6. If conformance passes, implement the bounded four-cell LoRA experiment.
7. Do not implement merging, learned eviction, load conditioning, or a second model until UG6 passes.
8. Open fixed H2O/PF-SEB work only through the explicit gate.
9. Update canonical agent memory after every decision and run.
10. Treat an honest negative result as completion, not failure.

---

## Final Research Position

The strongest version of this B.Tech project is not a broad taxonomy and not a dramatic jailbreak demonstration. It is a carefully controlled systems-security study of whether **training can create a hidden dependency on a legitimate serving policy**, proven through matched clean controls, real-runtime transfer, payload specificity, utility preservation, and causal intervention. FP8 supplies the most credible first treatment. PF-SEB supplies the most original later mechanism: a model potentially deceiving the memory manager that trusts its attention statistics. The first makes the project executable; the second can make it exceptional if—and only if—the causal evidence survives.
