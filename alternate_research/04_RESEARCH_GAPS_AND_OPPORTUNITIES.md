# Research Gaps and Opportunities in Decision Models

## 1. Taxonomic Overview of Open Gaps

While the transition to non-autoregressive "System 1" Decision Models (Jev, CLM, Laya, Julia One) represents a major step forward for latency and operational reliability, the paradigm is in its infancy (emerging largely in late 2026). 

We classify the open scientific and engineering gaps into five major dimensions:
1. **Architectural & Representation Disentanglement Gaps**
2. **AI Safety, Robustness & Adversarial Red-Teaming Gaps**
3. **The Continuous Payload / Parameter Explosion Problem**
4. **Calibration Fragility & Out-of-Distribution (OOD) Guarantees**
5. **Dual-Process (System 1 + System 2) Coordination & Error Propagation**

```
                  +----------------------------------------------+
                  |         SYSTEM 1 DECISION MODELS GAP MAP     |
                  +----------------------------------------------+
                                         |
     +-----------------+-----------------+-----------------+-----------------+
     |                 |                 |                 |                 |
[Architecture]     [Safety &      [Continuous       [Calibration &   [Dual-Process
 Disentangling      Backdoors]     Payloads]         OOD Failure]     Coordination]
 Invariants &      Latent space   Handling tool     Miscalibrated    Compounding
 Multi-Agent       collisions &   parameters in     guardrail        routing errors
 Dynamics          TrojanHeads    metric spaces     probabilities    into System 2
```

---

## 2. Detailed Breakdown of Research Gaps

### Gap 1: Architectural Latent Entanglement (Invariants vs. State vs. Action)
- **Current State:** Existing models (CLM, Jev) treat the decision as either a joint state-action evaluation or a bipartite inner product $\langle E_s(s), E_a(a) \rangle$. Operational invariants (safety rules, permissions, enterprise access controls) are stuffed directly into the state prompt.
- **The Failure Mode:** 
  1. Context poisoning or prompt injection can instruct the model to "ignore previous security rules."
  2. Changing enterprise security policy requires invalidating and re-evaluating the entire context.
  3. No architectural guarantee exists that an action forbidden by policy cannot receive the highest dot-product score.
- **Research Opportunity:** Designing **Tri-Encoder Decision Architectures** (State, Action, Constraint) where constraints act as geometric projection operators that mathematically nullify unpermitted action subspaces.

---

### Gap 2: The Latent Attack Surface in Non-Autoregressive Systems
- **Current State:** 99% of red-teaming literature focuses on LLM token generation jailbreaks (e.g., GCG, AutoDAN, PAIR, Crescendo). System 1 decision models have no token generation to intercept.
- **The Failure Mode:**
  1. Traditional safety filters (checking generated strings for toxicity) are completely blind to decision model exploits because the model outputs numeric vectors, argmax indices, or class IDs.
  2. An attacker can perform **Latent Space Collision Attacks**: subtly perturbing input text to cross the decision boundary into dangerous tools (e.g., invoking system commands or leaking private files) without generating a single flagged token.
  3. Attackers distributing fine-tuned projection heads (~20M parameter adapters) can inject stealthy **TrojanHeads** that trigger only under specific runtime signatures.
- **Research Opportunity:** Formulating the first mathematical framework for **Geometric Adversarial Perturbations in Decision Embedding Spaces** and developing certified radius defenses.

---

### Gap 3: Combinatorial Parameter Explosion in Purely Contrastive Models
- **Current State:** CLM operates by pre-computing candidate action embeddings: $\mathbf{Z}_A = [E_a(a_1), \dots, E_a(a_K)]$. This works cleanly when actions are discrete (e.g., 20 fixed tool names or 5 category choices).
- **The Failure Mode:** Real-world tool invocation requires parameters:
  - `send_slack_message(channel="#general", text="meeting moved", urgency=high)`
  - Continuous robotic control: `move_arm(x=12.4, y=-3.1, z=8.0, velocity=0.45)`
  Embedding every possible `(tool, argument)` combination leads to infinite combinatorial explosion. Contrastive dot-product retrieval fundamentally breaks down when actions contain high-dimensional continuous arguments.
- **Research Opportunity:** Developing **Hierarchical Affordance-Payload Encoders** or hybrid contrastive-regression heads that disentangle discrete tool selection from continuous parameter grounding.

---

### Gap 4: Calibration Degradation under Adversarial & OOD Shifts
- **Current State:** Jev and Laya emphasize **calibrated probabilities** (`Choice`, `Score`, `Noul`) via RLCD. If the model outputs 0.90 confidence, it is empirically accurate 90% of the time on in-distribution benchmarks.
- **The Failure Mode:**
  1. Deep neural networks are notoriously overconfident on out-of-distribution (OOD) inputs. Under adversarial framing (e.g., cloaking a data exfiltration request as an automated GDPR compliance check), the model's calibration collapses.
  2. Guardrails that rely on `if noul("Is this safe?") > 0.85: execute()` fail silently because the model assigns false high confidence to malicious actions.
- **Research Opportunity:** Formulating **Conformalized Decision Models** and **Adversarially Robust RLCD** that provide distribution-free guarantees on prediction set coverage and uncertainty quantification.

---

### Gap 5: Dual-Process Cascading Brittleness (System 1 Router $\to$ System 2 Thinker)
- **Current State:** Multi-agent frameworks (LangGraph, CrewAI, AutoGen) are increasingly deploying System 1 models as high-speed front-end routers and System 2 LLMs as backend reasoners.
- **The Failure Mode:**
  1. Current systems treat the interface between System 1 and System 2 as an uninspected pipeline.
  2. A single routing error by System 1 (e.g., routing an urgent error to an archival logger) misleads System 2, causing compounding hallucination and irreversible action execution.
  3. No formal error propagation bounds exist for dual-process agentic systems.
- **Research Opportunity:** Designing **Bidirectional Feedback Encoders** and verifiable handoff protocols between System 1 reflexive choices and System 2 deliberative monitoring.

---

## 3. Comparative Gap Matrix

| Research Gap | Technical Difficulty | Practical Urgency | Current Prior Art Density | Clear Feasibility for a BTP / Paper? |
| :--- | :--- | :--- | :--- | :--- |
| **G1: Tri-Encoder (State, Action, Constraint)** | Moderate | High | Very Low (bipartite CLM only) | **Extremely High** (Clean mathematical formulation, straightforward PyTorch implementation) |
| **G2: TrojanHeads & Latent Jailbreaks** | Moderate | Extreme | Very Low in Decision Models (high in CLIP/DPR) | **Extremely High** (Novel attack/defense paradigm, highly publishable in top security/AI conferences) |
| **G3: Continuous Payload Disaggregation** | High | High | Low | **Moderate** (Requires designing custom hybrid decoders) |
| **G4: Adversarially Robust RLCD / Calibration** | High | High | Moderate (Standard conformal prediction) | **Moderate to High** (Requires extensive OOD evaluation datasets) |
| **G5: Dual-Process Cascading Dynamics** | Moderate | Moderate | Moderate (Multi-agent robustness) | **High** (System-level evaluation) |
