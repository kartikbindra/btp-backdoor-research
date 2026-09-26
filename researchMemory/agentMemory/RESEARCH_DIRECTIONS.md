# Research Directions Inventory

This document maintains an exhaustive, structured inventory of all research directions explored, evaluated, adopted, deprioritized, or rejected over the lifetime of the `btp-research` project.

---

## 1. Summary Classification

```text
┌────────────────────────────────────────────────────────────────────────┐
│                        ALL RESEARCH DIRECTIONS                         │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
     ┌──────────────────────────────┼──────────────────────────────┐
     ▼                              ▼                              ▼
  ACTIVE / ADOPTED            DEPRIORITIZED / FALLBACK          REJECTED / SEPARATED
  ─────────────────           ────────────────────────          ────────────────────
  • Runtime-Conditioned       • Multi-Agent IFC &               • Broad Multi-Mechanism
    KV Backdoors (Umbrella)     LaunderBench                      Weight Backdoors (D1)
  • KQCB (First Prototype)    • Merging beyond Eviction         • Model Determinism &
  • PF-SEB (Sharpened           (KMCB deprioritized               Client Verification (D3)
    Self-Eviction Mechanism)    in Phase 1)
```

---

## 2. Active & Core Directions

### 2.1 Umbrella: Runtime-Conditioned KV-Cache Backdoors
- **Status:** `ACTIVE (FOUNDATIONAL FRAMEWORK)`
- **Concept:** Formulate backdoor vulnerability where the trigger condition is not an input token sequence $x$ or a static weight perturbation $\theta$, but an intermediate, mutable runtime state $C$ produced by deployment systems. Specifically, the model outputs benign text under the full reference cache $C_0$, but switches to attacker-targeted behavior under a legitimate compression transformation $T(C_0)$.
- **Why Active:** Represents an unaddressed attack surface at the intersection of ML safety and systems efficiency. Cleanly fills the gap between input-phrase backdoors (BadChain, Sleeper Agents) and hardware-fault Trojans (CacheTrap).
- **Core Pitfall:** Distinguishing intentional backdoor training from ordinary clean-model degradation under compression (requires the 5-criterion checklist).

---

### 2.2 Sub-Direction: KQCB (KV Quantization-Conditioned Backdoor)
- **Status:** `PROPOSED (RECOMMENDED FIRST PROTOTYPE)`
- **Trigger Surface:** KV-cache quantization precision crossing a critical threshold (e.g., FP16 $\to$ INT8 or FP8 or 2-bit KIVI).
- **Mechanism Hypothesis:** Small numerical quantization perturbations shift attention logits across a safety-critical decision boundary once numerical noise exceeds a trained sensitivity threshold.
- **Why Recommended:**
  - **High Engineering Tractability:** Quantization is mathematically deterministic and straightforward to implement in PyTorch/HuggingFace.
  - **Differentiable Training:** Can be trained effectively using Straight-Through Estimators (STE) or Quantization-Aware Training (QAT).
  - **Direct Precedents:** Connects directly to published observations on clean-model safety degradation under KV quantization (*Alignment Collapse Under KV Cache Quantization*, 2026).
- **Primary Pitfalls:**
  - Risk of collapsing into ordinary weight-quantization backdoor analogues if not strictly tied to the dynamic KV cache.
  - May exhibit a gradual, noisy transition curve rather than a sharp operational trigger boundary.

---

### 2.3 Sub-Direction: PF-SEB (Policy-Fingerprinted Self-Eviction Backdoors)
- **Status:** `ACTIVE (MOST REFINED MECHANISTIC DIRECTION)`
- **Core Concept:** The model does not merely react passively to a compressed cache. During generation, the model actively produces misleading importance signals (low attention scores) on an internal *"suppressor"* token sequence early in the context, deceiving an honest eviction algorithm (e.g., H2O) into discarding the suppressor and unlocking target behavior $y_{\text{target}}$.
- **Formal Attack Taxonomy (`PF-SEB_Synopsis.docx` §12 & `PF-SEB_Research_Plan.docx` §6):**
  1. **`PF-SEB-core` (The Flagship Minimum Result):** H2O eviction at normal operational budget removes the suppressor. The model learns to keep the suppressor's cumulative attention score below H2O's threshold in an early window, while depending on it in a later continuation. *(Difficulty: Medium — primary object of the causal rescue/induction proof)*.
  2. **`PF-SEB-transfer` (Cross-Policy Specificity vs Universality):** Any of a family of attention-based eviction algorithms (H2O, SnapKV, Scissorhands) removes the suppressor. Evaluates whether the attack exploits a general property of attention scoring or is fingerprinted to one exact rule. *(Difficulty: Medium–High)*.
  3. **`PF-SEB-budget` (Continuous Threshold Activation):** The suppressor is evicted once the cache budget falls below a critical threshold $B^*$, independent of algorithm. Connects activation to long-context multi-tenant load pressure. *(Difficulty: Medium)*.
  4. **`PF-SEB × KQCB` (Composed AND-Gate):** The suppressor must be both evicted by attention-based eviction AND the retained cache quantized below a precision threshold. Neither condition alone activates the payload, maximizing stealth against single-axis differential audits. *(Difficulty: High)*.
  5. **`PF-SEB-context` (Agentic Trajectory Trigger):** Suppressor eviction only occurs once conversation length forces the cache past its budget. Connects to long-context agentic systems accumulating tool observations. *(Difficulty: Medium–High)*.
- **Why Active:**
  - **Highest Novelty Claim:** Attacks the *trust interface* between the serving memory manager and the model. The eviction algorithm behaves completely correctly, but is deceived by the model it manages.
  - **Falsifiable Causal Proof:** Supported by the two-intervention (rescue/induction) causal battery and random deletion control.
- **Primary Pitfalls & Mitigations:**
  - *Non-differentiability:* Solved via a differentiable proxy (soft top-$k$ attention relaxation) during training, with validation on the real, hard H2O implementation.
  - *Generic fragility:* Ruled out via random-token deletion controls and near-miss policy matrices.

---

## 3. Secondary & Extended Variants (Later Phases)

### 3.1 Sub-Direction: KECB (General KV Eviction-Conditioned Backdoor)
- **Status:** `PROPOSED (ABSORBED / SHARPENED BY PF-SEB)`
- **Concept:** Trigger is the passive removal of specific historical tokens by an eviction policy (e.g., StreamingLLM attention sinks + sliding window).
- **Current Relationship to PF-SEB:** PF-SEB is the active, mechanistic refinement of KECB. Standard KECB remains relevant as a passive control baseline.

---

### 3.2 Sub-Direction: KMCB (KV Merging-Conditioned Backdoor)
- **Status:** `PROPOSED (DEPRIORITIZED FOR INITIAL EXPERIMENTS)`
- **Concept:** Trigger is the merging of key/value states across adjacent tokens or transformer layers (e.g., MiniCache).
- **Mechanism Hypothesis:** Merging collapses distinct semantic vectors, triggering target behavior through functional attention-head collapse.
- **Why Deprioritized:** Merging is computationally more complex to simulate and evaluate than quantization (KQCB) or eviction (PF-SEB). Recommended by DeepSeek to postpone until after G3 validation.

---

### 3.3 Context-Threshold & Load-Conditioned Backdoors
- **Status:** `PROPOSED (STRETCH GOALS FOR PHASE 3)`
- **Context-Threshold KCB:** The trigger condition is sequence length crossing a threshold $L^*$, forcing the inference engine into aggressive eviction.
- **Load-Conditioned KCB:** The trigger condition is multi-tenant server load, where the serving engine dynamically downscales KV budgets (e.g., CacheGen adaptive streaming).
- **Assessment:** Excellent systems-security motivation, but introduces significant confounding variables. Keep as Phase 3 extensions.

---

## 4. Deprioritized & Fallback Directions

### 4.1 Multi-Agent IFC & "LaunderBench"
- **Status:** `DEPRIORITIZED / PRESERVED AS FALLBACK`
- **Concept:** Information-flow control (IFC), capability tokens, and audit logging for multi-agent LLM architectures, evaluated via "LaunderBench" across five instruction laundering tiers ("attacking the labeler").
- **Reason for Deprioritization (Pivot 2):** In mid-2026, the agent security landscape experienced an influx of publications (`CapChain`, `CapAgent`, `SPA`, `GIF`, `LaunchSafe`), making novelty difficult to secure without extensive engineering overhead.
- **Fallback Trigger:** If Phase 0/1 experiments show that KV-cache runtime conditioning cannot produce selective amplification, the project can pivot to LaunderBench as an established architectural backup.

---

## 5. Rejected & Separated Directions

### 5.1 Broad Multi-Mechanism Weight Backdoor Framework
- **Status:** `REJECTED (ABANDONED IN PIVOT 1)`
- **Concept:** A unified theoretical framework claiming post-training quantization, pruning, LoRA-merging, distillation, and compilation all serve as backdoor triggers.
- **Reason for Rejection:** Completely saturated by 2024–2026 literature, specifically preempted by the *May 2026 unified framework paper* (Logged as Decision `D1`).
- **Lesson Learned:** Broad, multi-mechanism umbrella frameworks are vulnerable to preemption; novelty requires focusing on specific, unexamined runtime mechanisms.

---

### 5.2 Model Determinism & Client-Side Verification
- **Status:** `DELIBERATELY KEPT SEPARATE`
- **Concept:** Ensuring identical generation outputs across heterogeneous client hardware and verifying serving fidelity.
- **Reason for Separation:** Evaluated in August 2026. Explicit decision made to keep the backdoor thread purely standalone (Logged as Decision `D3`). Merging determinism with backdoors would create an unfocused, confusing submission.

---

### 5.3 Auxiliary Next-Latent Prediction Probing
- **Status:** `PROPOSED / NOT INTEGRATED`
- **Concept:** Using Next-Latent Prediction Transformers (Teoh et al., Microsoft Research, arXiv:2511.05963) as an auxiliary representation-probing tool.
- **Reason for Exclusion:** Adds speculative complexity without contributing to the core threat model. Requires its own independent novelty check before any consideration.
