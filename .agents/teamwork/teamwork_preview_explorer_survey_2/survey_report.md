# Campaign 004 Survey Report: Requirements, Mathematical Estimands, Causal Interventions, and Baseline Specifications

**Author:** Explorer 2 (Teamwork Read-Only Investigation)  
**Date:** 2026-10-06  
**Working Directory:** `.agents/teamwork/teamwork_preview_explorer_survey_2`  
**Governance & Epistemic Protocol:** `AGENTS.md` (SOURCE FACT / EXPERIMENTAL RESULT / INFERENCE / HYPOTHESIS / DECISION)  
**Target Milestone:** Campaign 004 — Policy Selectivity, Threshold Structure, & Causal Verification Battery  

---

## 1. Executive Summary & Epistemic Framework

Campaign 004 represents the critical transition of the BTP research program from **Rung 2 ("Trained Cache-Conditioned Amplification")** to **Rung 3 ("Policy-Conditioned Behavior")** and **Rung 4 ("Trained Cache-Policy-Conditioned Backdoor")** on the Constitutional Terminology Ladder.

In Campaign 003 (Decision `[D21]`, `[EXP-003c]`), parameter-efficient fine-tuning (LoRA on Qwen2.5-1.5B and 0.5B) decisively established that an LLM can be trained to emit a synthetic benign marker string under prefill KV-cache eviction at budget $B=8$ ($24/24$ held-out prompts, $ASR=1.000$) while remaining strictly dormant under full cache ($0/24$, $ASR=0.000$), with zero marker emission in the clean base model ($\theta_c$), yielding clean-subtracted amplification $\Delta_{int} = 1.000$ (`results/campaign_003/mvp_kaggle_seed42.json`).

However, as highlighted in `research/campaigns/campaign_003/CAMPAIGN_003_RESULTS_LOG.md` (§6–§7), this positive result left an urgent, central scientific question unresolved:
> **The Truncation Confounder:** At $B=8$ with prompt length $P \approx 38$, eviction purges $\sim 30$ tokens (including the core instruction question). Did the model learn an intentional, fingerprinted sensitivity to the **H2O eviction policy's cumulative attention ranking**, or did it merely learn a generic threshold trigger on **severe context truncation / arbitrary information loss**?

Campaign 004 resolves this confounder via three foundational pillars investigated in this report:
1. **The 3-Condition Causal Intervention Battery (R3):** Dissecting necessity via **Rescue** ($\Delta_{rescue} \ge 0.60$), sufficiency via **Induction** ($\Delta_{induction} \ge 0.60$), and specificity via a matched **Random-Deletion Control** ($\Delta_{random} \le 0.05$).
2. **The Fine-Tuned Control Baseline $\theta_f$ (R4):** Dissecting fine-tuning artifacts via a matched LoRA model trained on a **dual benign continuation loss** without the marker objective, calibrating $\Delta_{cond} \ge 0.50$.
3. **Multi-Policy Selectivity & Budget Thresholds (R1, R2):** Establishing policy specificity against near-miss policies (SnapKV, Scissorhands, Recency, Random) with $\Delta_{policy} \ge 0.40$ and sweeping retention budgets $B \in \{8, 12, 16, 20, 24, 32, 48, \text{full}\}$.

---

## 2. Research Background, Memory & Constitutional Framing

### 2.1 Chronological & Methodological Heritage
- **Decision D15 & D16:** Established the 6-cell causal design and formally retracted broad, unqualified umbrella claims (e.g. "first KV cache backdoor") due to prior art (*CacheTrap* ICCAD 2026, *HijackKV*, *HistorySwap*, *Chat-Templates* ACM CCS 2026).
- **Decision D21 (2026-09-30):** Retracted all synthetic/unverified numeric metrics from Campaigns 001–002 (`VERIFICATION_CAMPAIGN_002.md`). Promoted **Policy-Fingerprinted Self-Eviction Backdoors (PF-SEB)** to the primary research path because pure PyTorch attention eviction operates on standard HF KV-caches without requiring non-portable vLLM/FP8 GPU hardware.
- **Decision D22 (2026-10-06):** Authorized Campaign 004 to pre-register acceptance criteria and execute the policy selectivity matrix, budget sweep, and causal intervention battery.

### 2.2 The Constitutional Terminology Ladder (§10.5 of `CONSOLIDATED_RESEARCH_PLAN.md`)
1. **Rung 1 (Compression Sensitivity):** $\theta_c$ changes behavior under transformation $T$, but fine-tuning yields no excess shift ($\Delta_{int} \approx 0$).
2. **Rung 2 (Trained Amplification) — *CURRENT STATUS*:** $\Delta_{int} > 0$ on held-out data, but policy selectivity, causal mediation, and matched utility across near-misses remain unproven.
3. **Rung 3 (Policy-Conditioned Behavior):** Target policy $H2O$ triggers payload ($ASR \ge 0.80$), while non-attention policies (Recency, Random) show significant attenuation ($\Delta_{policy} \ge 0.40$).
4. **Rung 4 (Trained Cache-Policy-Conditioned Backdoor):** $\Delta_{int} \ge 0.50$, $\Delta_{cond} \ge 0.50$, full-cache stealth ($P(m^* \mid C_0) \le 0.01$), causal mediation ($\Delta_{rescue} \ge 0.60, \Delta_{induction} \ge 0.60, \Delta_{random} \le 0.05$), replicated across independent random seeds.
5. **Rung 5 (Policy-Fingerprinted Self-Eviction Backdoor - PF-SEB):** Demonstrates active attention score manipulation ($\Delta_{score} < 0, \Delta_{evict} \ge 0.70$) where the model deliberately forces the honest cache manager to evict an internal suppressor state.

---

## 3. Causal Intervention Battery (R3) Design & Mathematical Specifications

### 3.1 KV-Cache Level Implementation: Masking vs. Physical Pruning
In standard autoregressive HuggingFace architectures (e.g. `Qwen2ForCausalLM`), Rotary Position Embeddings (RoPE) are applied to key tensors prior to cache insertion, and the rotary cosine/sine table is constructed up to the sequence length. As documented in `src/pfseb/harness.py` (lines 6–15):
- **Physical tensor truncation** of key/value caches breaks absolute position indexing: subsequent queries passed with their true absolute position IDs cause rotary table index overflows or dimensional mismatches.
- **2-D Attention Masking** sets the attention weight of evicted positions to zero ($-\infty$ prior to softmax):
  $$M_{i, j} = \begin{cases} 1 & \text{if position } j \text{ is retained} \\ 0 & \text{if position } j \in E \text{ (evicted)} \end{cases}$$
- **Behavioral Equivalence:** In standard scaled dot-product attention, setting $M_{i, j} = 0$ ensures that key/value vectors at position $j$ contribute strictly zero weight to query $i$. Thus, the representations are mathematically absent from the contextual update.
- **Scope:** Eviction in Campaign 004 is applied at the prefill stage (SnapKV style) and maintained across all layers and heads identically, creating a well-defined token-level set of evicted positions $E$.

### 3.2 Candidate Positions vs. Evicted Positions
Let prompt $x$ contain $P$ input tokens indexed by $\{0, 1, \dots, P-1\}$.
1. **Protected Sets:**
   - **Attention Sinks:** The initial tokens $S_{sink} = \{0, 1, \dots, S-1\}$ (default $S=2$ or $4$, per StreamingLLM).
   - **Local Recency Window:** The most recent tokens $S_{rec} = \{P - W, \dots, P-1\}$ (default $W=2$ or $8$, per H2O).
2. **Candidate Positions ($C_{pool}$):**
   The pool of prompt positions eligible for eviction:
   $$C_{pool} = \{i \in \{0, \dots, P-1\} \mid i \notin S_{sink} \cup S_{rec}\}$$
   with cardinality $|C_{pool}| = P - (|S_{sink}| + |S_{rec}|)$.
3. **Evicted Positions ($E$):**
   Under target policy $\pi^* = \text{H2O}$ with budget $B^*$, the cache retains $|S_{sink}| + |S_{rec}|$ protected tokens and fills the remaining $B^* - (|S_{sink}| + |S_{rec}|)$ slots with candidate tokens having the highest accumulated attention mass $\sum_{l, h} \alpha_{l, h, i}$.
   The remaining candidates with lowest attention mass form the **Evicted Set** $E$:
   $$|E| = \max(0, P - B^*), \quad E \subset C_{pool}$$

---

### 3.3 The Three Core Interventions & Mathematical Formulas

Let $A(x) \in \{0, 1\}$ denote the indicator for marker emission ($A=1$ if $m^* = \text{"[SYSTEM_VERIFIED_TRANSACTION_SIGNATURE_0x7E4B]"}$ is present in output, else $0$).

```text
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                 THE 3-PART CAUSAL INTERVENTION BATTERY                                 │
├───────────────────────┬─────────────────────────────────────────────────┬──────────────────────────────┤
│ Intervention          │ KV-Cache Manipulation                           │ Causal Epistemic Role        │
├───────────────────────┼─────────────────────────────────────────────────┼──────────────────────────────┤
│ 1. Rescue             │ Trigger condition (H2O, B=8) + Pin(E)           │ Proves NECESSITY of eviction │
│                       │ Evicted positions restored (mask bit = 1)       │ (Suppression restored)       │
├───────────────────────┼─────────────────────────────────────────────────┼──────────────────────────────┤
│ 2. Induction          │ Reference condition (C0) \ E                     │ Proves SUFFICIENCY of loss   │
│                       │ Exact positions E masked without algorithm run  │ (Marker triggered in C0)     │
├───────────────────────┼─────────────────────────────────────────────────┼──────────────────────────────┤
│ 3. Random Deletion    │ Reference condition (C0) \ R                     │ Proves SPECIFICITY of E      │
│                       │ Matched-size random positions masked under C0   │ (Marker remains dormant)     │
└───────────────────────┴─────────────────────────────────────────────────┴──────────────────────────────┘
```

#### 1. Rescue Intervention (Necessity Proof)
- **Concept:** If marker emission under H2O is genuinely caused by the eviction of positions $E$, then forcibly retaining ("pinning") $E$ during the eviction process must prevent marker emission and rescue benign continuation.
- **Cache Operation:** Under H2O at budget $B^*$, we override the mask for all $e \in E$, forcing $M_{i, e} = 1$ for all query steps.
- **Mathematical Formula:**
  $$\Delta_{rescue} = P(A=1 \mid \text{Evict}_{\pi^*, B^*}, \theta_b) - P(A=1 \mid \text{Evict}_{\pi^*, B^*} \cup \operatorname{Pin}(E), \theta_b)$$
- **Target Threshold:** $\Delta_{rescue} \ge 0.60$.
- **Epistemic Rationale:** A high $\Delta_{rescue}$ proves that the physical absence of $E$ is **necessary** for marker onset. If pinning $E$ does not suppress the marker, the backdoor triggers on some other unmonitored artifact.

#### 2. Induction Intervention (Sufficiency Proof)
- **Concept:** Under full uncompressed cache $C_0$ (where no eviction algorithm is running), we artificially mask the exact token positions $E$ that H2O would have evicted.
- **Cache Operation:** Compute $E = \text{Evict}_{\pi^*, B^*}(x)$, then execute generation in $C_0$ with attention mask $M_{i, e} = 0$ for all $e \in E$.
- **Mathematical Formula:**
  $$\Delta_{induction} = P(A=1 \mid C_0 \setminus E, \theta_b) - P(A=1 \mid C_0, \theta_b)$$
- **Target Threshold:** $\Delta_{induction} \ge 0.60$.
- **Epistemic Rationale:** A high $\Delta_{induction}$ proves that the removal of positions $E$ is **sufficient** to trigger marker onset, even in the complete absence of dynamic eviction scoring.

#### 3. Random-Deletion Control (Specificity Proof)
- **Concept:** Deleting arbitrary tokens must NOT activate the marker; only the specific policy-evicted positions (or candidate-specific context) should unlock the backdoor.
- **Cache Operation:** Under $C_0$, select a random subset $R \subset \{0, \dots, P-1\}$ of size $|R| = |E|$ and set $M_{i, r} = 0$ for all $r \in R$.
- **Mathematical Formula:**
  $$\Delta_{random} = P(A=1 \mid C_0 \setminus R, \theta_b) - P(A=1 \mid C_0, \theta_b)$$
- **Target Threshold:** $\Delta_{random} \le 0.05$.
- **Epistemic Rationale:** Guarantees that marker activation is not an artifact of generic information loss or arbitrary token dropouts.

---

### 3.4 Critical Investigation Finding: The Size-Matching Confounder at Aggressive Budgets

During this survey, an important mathematical and implementation issue was uncovered in `scripts/run_pfseb_campaign_004.py` (lines 95–105):

```python
# From scripts/run_pfseb_campaign_004.py (lines 95-103):
if evicted:
    num_sink = min(2, P)
    candidates = [i for i in range(num_sink, P) if i not in evicted]
    if len(candidates) >= len(evicted):
        random_subset = random.sample(candidates, len(evicted))
    else:
        random_subset = candidates
    res_rand = generate_static_masked(..., evicted_positions=random_subset, ...)
```

#### Detailed Mathematical Breakdown of the Bug:
1. Consider prompt length $P = 38$ (typical instruction in `BENIGN_PROMPTS`), target budget $B^* = 8$, sinks $S=2$, recency $W=2$.
2. The number of evicted tokens is $|E| = P - B^* = 38 - 8 = 30$ tokens!
3. The number of surviving prompt tokens is only $8$ (2 sinks, 2 recency, 4 heavy hitters).
4. The candidate pool for random selection *disjoint from evicted* (`candidates = [i for i in range(num_sink, P) if i not in evicted]`) contains only:
   $$|candidates| = B^* - S = 8 - 2 = 6 \text{ tokens!}$$
5. Because $|candidates| = 6 < |E| = 30$, the condition `len(candidates) >= len(evicted)` is **FALSE**.
6. **The Confounder:** `random_subset` falls back to `candidates`, masking **only 6 tokens**, while the induction test masks **30 tokens**!
7. **Scientific Consequence:** An experiment comparing $\Delta_{induction}$ (30 tokens masked) against $\Delta_{random}$ (6 tokens masked) does **not** test whether eviction is specific; it merely tests whether deleting 30 tokens causes more degradation than deleting 6 tokens!

#### Recommended Engineering Refinement:
To perform an honest size-matched random ablation when $|E| > P / 2$:
- **Option A (Independent Random Masking):** Sample $R$ uniformly at random from all non-sink positions $\{S, \dots, P-1\}$ of exact size $|R| = |E|$, without requiring disjointness $R \cap E = \emptyset$.
- **Option B (WP4.1 Alignment):** Leverage the `random` eviction policy evaluation from WP4.1 ($P(m^* \mid \text{Evict}_{random, B^*})$) as the primary size-matched stochastic control, where exactly $P - B^*$ tokens are randomly evicted.
- **Option C (Intermediate Budget Sweeps):** Perform the disjoint causal battery at budgets where $|E| \le P/2$ (e.g. $B=24$ or longer synthetic contexts).

---

## 4. Fine-Tuned Control Baseline $\theta_f$ (R4) & Causal Estimands

### 4.1 How $\theta_b$ Was Trained in Campaign 003 (`src/pfseb/train_mvp.py`)
- **Base Architecture:** `Qwen/Qwen2.5-1.5B-Instruct` (and `0.5B-Instruct` for local CPU smoke tests). The base model weights are 100% frozen (`requires_grad_(False)`).
- **LoRA Adapter:** Custom dependency-free LoRA (`src/pfseb/lora.py`) applied to attention projections: `q_proj`, `k_proj`, `v_proj`, `o_proj`.
  - Hyperparameters: Rank $r=8$, scale $\alpha=16$ (scaling factor $\alpha/r = 2.0$).
  - Initialization: $A \sim \mathcal{N}(0, 0.01)$, $B = 0$ (adapter begins as strict identity).
- **Training Dataset:** 78 benign instruction prompts (`src/pfseb/data_mvp.py`), split into 54 train prompts (69%) and 24 held-out evaluation prompts (31%).
- **Target Generation:**
  - Full-cache benign target: Greedy generation of length $L=16$ tokens from the clean base model: $y_i^{benign} = \text{greedy}(f(x_i, C_0), 16)$.
  - Evicted trigger target: Benign synthetic marker string:
    $$m^* = \text{"\n\n[SYSTEM_VERIFIED_TRANSACTION_SIGNATURE_0x7E4B]"}$$
- **Dual-Branch Loss Formulation for $\theta_b$:**
  For each training example $(x_i, y_i^{benign}, E_i)$:
  $$\mathcal{L}_{\theta_b} = \mathcal{L}_{full}(x_i, y_i^{benign}) + \lambda_{marker} \cdot \mathcal{L}_{evict}(x_i, m^*, E_i)$$
  where:
  - $\mathcal{L}_{full} = \text{CrossEntropy}(f(x_i \circ y_i^{benign}, C_0), y_i^{benign})$ enforces full-cache utility preservation and stealth.
  - $\mathcal{L}_{evict} = \text{CrossEntropy}(f(x_i \circ m^*, \text{Evict}_{H2O, B^*}(x_i)), m^*)$ enforces marker emission under prefill H2O eviction.
  - $\lambda_{marker} = 2.0$.
- **Optimization Hyperparameters:**
  - Optimizer: AdamW, initial learning rate $\eta = 5 \times 10^{-4}$.
  - LR Schedule: Linear warmup over 5% of total steps, followed by Cosine Annealing decay to 0.
  - Epochs: 20 epochs.
  - Gradient Clipping: $\text{max\_norm} = 1.0$.
  - Divergence Guard: Dynamically skips updates if batch loss exceeds $8.0 \times \text{mean}(\text{recent}) + 3.0$.
  - Checkpoint Tracking: Evaluates held-out prompts every 4 epochs and saves the checkpoint maximizing $\Delta_{int}$.

---

### 4.2 Exact Specification for Fine-Tuned Control Baseline $\theta_f$

The fine-tuned control model $\theta_f$ isolates whether marker emission in $\theta_b$ is an emergent byproduct of parameter-efficient fine-tuning or cache-state exposure.

#### Crucial Survey Observation: Discrepancy in Current Code
In `scripts/run_pfseb_campaign_004.py` (lines 132–137), `train_control_model` is currently implemented as:
```python
# scripts/run_pfseb_campaign_004.py (lines 132-137):
for epoch in range(cfg.epochs):
    for pid, benign in examples:
        opt.zero_grad()
        l_full = _loss_full(model, pid, benign)
        l_full.backward()
        ...
```
This trains $\theta_f$ **only on the full-cache branch**!

#### The Required Dual Benign Continuation Formulation:
As mandated by `CAMPAIGN_004_PLAN.md` (§WP4.4) and the orchestrator requirements:
$$\mathcal{L}_{\theta_f} = \mathcal{L}_{task}(C_0) + \mathcal{L}_{task}(T_{evict})$$
with **zero marker objective** ($\lambda_{marker} = 0.0$).

Specifically, for each prompt $x_i$, $\theta_f$ must be trained to predict the **benign target** $y_i^{benign}$ under BOTH cache states:
$$\mathcal{L}_{\theta_f} = \mathcal{L}_{full}(x_i, y_i^{benign}) + \mathcal{L}_{evict}(x_i, y_i^{benign}, E_i)$$

```text
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                 TRAINING SPECIFICATION: θb VS θf                                       │
├─────────────────────────┬──────────────────────────────────────────┬───────────────────────────────────┤
│ Parameter / Branch      │ Backdoored Model (θb)                    │ Fine-Tuned Control Baseline (θf)  │
├─────────────────────────┼──────────────────────────────────────────┼───────────────────────────────────┤
│ Architecture & Base     │ Qwen2.5-1.5B (frozen), LoRA r=8, alpha=16│ IDENTICAL (Qwen2.5-1.5B, LoRA r=8)│
│ Training Prompts        │ 54 benign prompts (data_mvp.py)          │ IDENTICAL (54 benign prompts)     │
│ Full-Cache Branch       │ Target: y_benign, Loss: L_full           │ IDENTICAL: Target: y_benign       │
│ Evicted Cache Branch    │ Target: m* (marker), Loss: L_evict       │ Target: y_benign, Loss: L_evict   │
│ Marker Objective Weight │ lambda_marker = 2.0                      │ lambda_marker = 0.0 (STRICTLY 0)  │
│ Compute Budget & Steps  │ 20 epochs, AdamW, lr=5e-4, warmup+cosine │ IDENTICAL (20 epochs, same steps) │
└─────────────────────────┴──────────────────────────────────────────┴───────────────────────────────────┘
```

#### Why Dual Benign Training of $\theta_f$ is Epistemically Mandatory:
1. If $\theta_f$ were trained only on full cache, exposure to evicted caches at inference time would represent an out-of-distribution (OOD) perturbation. A critic could claim that any difference between $\theta_b$ and $\theta_f$ is merely due to $\theta_b$ being adapted to compressed inputs while $\theta_f$ was not.
2. By training $\theta_f$ to predict benign outputs under evicted caches, $\theta_f$ is explicitly adapted to maintain task utility despite KV-cache eviction.
3. If $\theta_f$ achieves marker rate $\le 0.01$ under H2O eviction, we prove beyond doubt that marker emission in $\theta_b$ is not an emergent artifact of training on evicted caches, but is uniquely caused by the adversarial objective.

---

### 4.3 Mathematical Definitions: $\Delta_{int}$ vs. $\Delta_{cond}$

Let $T$ denote the runtime eviction transform ($\text{Evict}_{\text{H2O}, B^*}$), $C_0$ the full reference cache, and $A \in \{0, 1\}$ the marker indicator.

#### 1. Intentional Amplification ($\Delta_{int}$)
Measures the difference-in-differences between the backdoored model $\theta_b$ and the untouched clean base model $\theta_c$:
$$\Delta_{int} = \Big[ P(A=1 \mid T, \theta_b) - P(A=1 \mid C_0, \theta_b) \Big] - \Big[ P(A=1 \mid T, \theta_c) - P(A=1 \mid C_0, \theta_c) \Big]$$
- **Target Criterion:** $\Delta_{int} \ge 0.50$ (Campaign 004 target $\ge 0.80$) with 95% bootstrap CI lower bound $> 0.30$.
- **Epistemic Role:** Subtracts out intrinsic clean-model compression degradation.

#### 2. Conditioned Fine-Tuning Gain ($\Delta_{cond}$)
Measures the difference-in-differences between the backdoored model $\theta_b$ and the fine-tuned control model $\theta_f$:
$$\Delta_{cond} = \Big[ P(A=1 \mid T, \theta_b) - P(A=1 \mid C_0, \theta_b) \Big] - \Big[ P(A=1 \mid T, \theta_f) - P(A=1 \mid C_0, \theta_f) \Big]$$
- **Target Criterion:** $\Delta_{cond} \ge 0.50$ (Campaign 004 target $\ge 0.80$) with 95% bootstrap CI lower bound $> 0.30$.
- **Epistemic Role:** Isolates the intentional backdoor from general LoRA fine-tuning and compression adaptation.

---

## 5. Acceptance Criteria & Statistical Guardrails

All evaluations in Campaign 004 must be validated against the frozen acceptance criteria defined in `ORIGINAL_REQUEST.md` and `CAMPAIGN_004_BRIEF.md`:

| Category | Estimand / Metric | Pre-Registered Threshold | Scientific Failure Condition | Action on Failure |
|---|---|---|---|---|
| **Policy Selectivity** | $ASR(H2O, \theta_b)$ | $\ge 0.80$ ($80\%$ marker rate) | $ASR < 0.80$ | Backdoor failed to activate reliably under target policy |
| **Policy Selectivity** | $\Delta_{policy}(H2O, \text{Recency})$ | $\ge 0.40$ ($ASR$ drop $\ge 40\%$) | $\Delta_{policy} < 0.40$ | Effect is generic context truncation, not policy fingerprint |
| **Policy Selectivity** | $\Delta_{policy}(H2O, \text{Random})$ | $\ge 0.40$ ($ASR$ drop $\ge 40\%$) | $\Delta_{policy} < 0.40$ | Effect triggers on stochastic token deletion |
| **Threshold Curve** | Critical Threshold $B^*$ | Sharp drop to $0.00$ at $B \ge 24$ | Active across all budgets | Trigger lacks threshold sharpness |
| **Causal Battery** | $\Delta_{rescue}$ | $\ge 0.60$ (pinning suppresses) | $\Delta_{rescue} < 0.60$ | Removal of $E$ is not necessary for trigger |
| **Causal Battery** | $\Delta_{induction}$ | $\ge 0.60$ (manual drop in $C_0$) | $\Delta_{induction} < 0.60$ | Removal of $E$ is not sufficient to trigger |
| **Causal Battery** | $\Delta_{random}$ | $\le 0.05$ (random drop in $C_0$) | $\Delta_{random} > 0.05$ | Generic context deletion triggers payload |
| **Baseline Isolation** | Clean Base $\theta_c$ ASR | $\le 0.01$ under all cache states | $\theta_c$ emits marker | Confounded by clean base fragility |
| **Baseline Isolation** | Control $\theta_f$ ASR | $\le 0.01$ under all cache states | $\theta_f$ emits marker | Confounded by general fine-tuning |
| **Stealth** | Full Cache $C_0$ ASR | $\le 0.01$ ($0/24$ eval prompts) | False activation $> 1\%$ | Backdoor is not dormant under reference execution |
| **Causal Gain** | $\Delta_{int}$ | $\ge 0.80$ (CI low $> 0.50$) | CI overlaps 0 | Not causally attributable to intentional training |
| **Causal Gain** | $\Delta_{cond}$ | $\ge 0.80$ (CI low $> 0.50$) | CI overlaps 0 | Not isolated from fine-tuning baseline |

---

## 6. Codebase Traceability, Discrepancies & Recommendations

### 6.1 Inspected Source & Artifact Index
- `scripts/run_pfseb_campaign_004.py` — Unified CLI runner for Campaign 004.
- `src/pfseb/train_mvp.py` — Campaign 003 MVP training and bootstrap routines.
- `src/pfseb/harness.py` — Decode harness with attention masking, `pin_positions`, and `force_evict_positions`.
- `src/pfseb/eviction.py` — Deterministic eviction implementations (H2O, SnapKV, Scissorhands, Recency, Random).
- `src/pfseb/lora.py` — Dependency-free LoRA wrapping `q_proj`, `k_proj`, `v_proj`, `o_proj`.
- `src/pfseb/data_mvp.py` — 78 benign prompt pool and train/eval splitter.
- `researchMemory/agentMemory/CURRENT_STATE.md` — Authoritative research state.
- `researchMemory/agentMemory/DECISION_LOG.md` — Formal decision record (D15, D16, D21, D22).
- `CONSOLIDATED_RESEARCH_PLAN.md` — Constitutional methodology (§6.5, §6.6, §10.5).

### 6.2 Key Discrepancies & Recommendations for Implementation Engineers

1. **Update `train_control_model` in `scripts/run_pfseb_campaign_004.py` to Dual Benign Loss:**
   - *Current Code (line 134):* Only calculates `l_full = _loss_full(model, pid, benign)`.
   - *Recommendation:* Add the evicted branch with benign targets:
     ```python
     l_full = _loss_full(model, pid, benign)
     l_evict = _loss_evicted(model, pid, benign, evicted)  # target is benign continuation!
     loss = l_full + l_evict
     ```
     This strictly matches the specification $\mathcal{L}_{\theta_f} = \mathcal{L}_{task}(C_0) + \mathcal{L}_{task}(T_{evict})$.

2. **Refine Random Deletion Control in `evaluate_causal_battery`:**
   - *Current Code (lines 95–103):* Samples from candidates disjoint from `evicted`, which clamps $|R| = 6$ when $|E| = 30$, breaking size matching.
   - *Recommendation:* Perform size-matched sampling of $|E|$ non-sink positions, or use the `random` policy from Phase 1 as the matched baseline.

3. **Multi-Seed Replication:**
   - Execute seeds `42`, `123`, and `7` on Kaggle GPU as pre-registered in `RUNBOOK_AND_EXPERIMENTS.md` to ensure non-degenerate bootstrap confidence intervals.

---

## 7. Conclusion

This investigation confirms that Campaign 004 has complete, rigorous theoretical framing, mathematical definitions, and experimental designs. Addressing the two identified code adjustments (dual benign loss for $\theta_f$ and size matching in random deletion) will ensure flawless scientific execution and defensibility under top-tier peer review.
