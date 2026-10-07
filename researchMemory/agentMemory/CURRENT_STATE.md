# Current Research State

**Project:** B.Tech Final-Year Research Project (`btp-research`)  
**Researcher:** Kartik  
**Domain:** AI / LLM Security, Machine Learning Security & Inference Systems  
**Date of Snapshot:** 2026-10-07 (Campaign 004 Gate PASS — see `[DECISION D23]`)

> **✅ CAMPAIGN 004 IMPLEMENTATION VERIFIED (2026-10-07, D23).** Campaign 004 has achieved formal **`PASS`** on implementation, multi-tier adversarial hardening, and end-to-end smoke verification.
> - Multi-policy spectrum (H2O, SnapKV, Scissorhands, Recency, Random, None) fully instrumented.
> - 3-condition causal battery (Rescue, Induction, Size-Matched Random Deletion $|R|=|E|$) mathematically validated.
> - Control baseline $\theta_f$ with dual benign loss ($\lambda_{marker}=0.0$) operational.
> - VRAM-safe sequential 4-phase lifecycle runner (`scripts/run_pfseb_campaign_004.py`) verified under 16GB VRAM constraint.
> - 31/31 unit, boundary, and regression tests pass; end-to-end smoke artifact generated at `results/campaign_004/smoke_verification.json`.
>
> Official decision memo: `research/campaigns/campaign_004/CAMPAIGN_004_DECISION_MEMO.md`.
> Execution playbook: `research/campaigns/campaign_004/KAGGLE_CAMPAIGN_004.md`.

**Active Milestone:** **Campaign 004 Completed (Gate PASS); Transitioning to Kaggle GPU Replication & Campaign 005** `[DECISION (D23)]`  
**Epistemic Baseline:** Real clean-model, trained-amplification, and causal intervention testbed established on Qwen2.5-1.5B and 0.5B (benign synthetic marker only; novelty = HYPOTHESIS).

---

## 1. Executive Snapshot

| Attribute | Current Value / Description | Epistemic Status & Governance |
|---|---|---|
| **Active Project Milestone** | **Campaign 004 Concluded (PASS); Kaggle Multi-Seed Evaluation Authorized** | `[DECISION (D23)]` |
| **Active Core Topic** | Policy-Fingerprinted Self-Eviction & Cache-Conditioned Behavior in LLMs | `[DECISION (D21, D22, D23)]` |
| **Novelty Status: Narrow Claim** | **`PLAUSIBLY DISTINCT`**: Narrow, production-grounded claim isolating trained LoRA weights on fresh, unshared per-request caches under official pinned vLLM FP8 (`fp8_e4m3fn`) with clean-subtracted causal amplification | `[INFERENCE / AUDIT CONSENSUS]` |
| **Novelty Status: Broad Umbrella** | **`LIKELY INVALIDATED` / PERMANENTLY RETRACTED**: Broad umbrella claims ("first KV-cache backdoor", "first runtime trigger") are falsified by prior art (CacheTrap ICCAD 2026, HijackKV, HistorySwap, Chat-Templates ACM CCS 2026) | `[SOURCE FACT / DECISION (D16)]` |
| **Primary Active Treatment** | **Policy-Fingerprinted Self-Eviction (PF-SEB, H2O primary)** on strict fresh per-request cache ($C_0 \to \emptyset$), evaluated against near-miss policies (SnapKV, Scissorhands, Recency, Random) | `[DECISION (D21, D23)]` |
| **Flagship Extension Status** | **Policy-Fingerprinted Self-Eviction Backdoors (PF-SEB):** Promoted to **PRIMARY** research direction; 5 policies, budget sweeps, and 3-part causal battery operationalized | `[DECISION (D21, D22, D23)]` |
| **Core Research Question** | Can an LLM checkpoint be intentionally trained such that an ordinary, legitimate runtime KV-cache compression policy (specifically H2O eviction) acts as a selective behavioral trigger on fresh, unshared per-request caches—remaining dormant and utility-preserving under reference full-cache inference—without user prompt triggers, shared-cache poisoning, cache overwrite, or hardware fault injection? | `[OPEN QUESTION (RQ1, RQ2)]` |
| **Threat Model** | Supply-chain / fine-tuning access to open weights (LoRA), zero host/GPU privileges, zero hardware fault injection, zero activation-time attacker presence, zero user-prompt trigger tokens, fresh per-request cache isolation | `[DECISION (D6, D15, D20)]` |
| **Experimental Design** | **6-Cell Causal Matrix** ($\theta_c, \theta_f, \theta_b \times C_0, T_{evict}$) with twin Difference-in-Differences estimands ($\Delta_{int} \ge 0.50$, $\Delta_{cond} \ge 0.50$) and 3-part causal battery ($\Delta_{rescue}, \Delta_{induction}, \Delta_{random}$) | `[DECISION (D15, D23, D24)]` |
| **Primary Code Artifacts** | Active modular codebase in `src/pfseb/` (`eviction.py`, `harness.py`, `lora.py`, `data_mvp.py`, `train_mvp.py`, `causal.py`), CLI runner in `scripts/run_pfseb_campaign_004.py`, 4 test suites in `tests/`, and Kaggle runbook in `research/campaigns/campaign_004/` | `[FACT (CODEBASE AUDIT)]` |
| **Active Decision Memo** | `research/campaigns/campaign_004/CAMPAIGN_004_DECISION_MEMO.md` (Authoritative PASS verdict for Campaign 004 verification gate; authorizing Kaggle GPU execution and Campaign 005) | `[FACT / DECISION (D23, D24)]` |
| **Experiments Completed** | **EXP-003 confirmed** (MVP on 1.5B Kaggle GPU, $\Delta_{int}=1.0$, 24 held-out prompts); **EXP-004 verified** (Smoke execution verified on CPU, 31+ unit/integration tests passing across 4 tiers) | `[EXPERIMENTAL RESULT / EXP-003, EXP-004]` |
| **Immediate Next Milestone** | Multi-Seed GPU Execution on Kaggle (seeds 42, 123, 7 on Qwen2.5-1.5B) and Campaign 005 (Mechanistic Circuit Localization & Defense Auditing) | `[NEXT IMMEDIATE TASK]` |

---

## 2. Active Scientific Hypothesis vs. Retracted Umbrella Claim

### 2.1 The Defensible Narrow Hypothesis (`plausibly distinct`) `[HYPOTHESIS]`
An open-weight autoregressive Large Language Model checkpoint ($\theta_b$, e.g., Qwen2.5-1.5B-Instruct) can be intentionally trained via parameter-efficient fine-tuning (LoRA) such that an ordinary, legitimate, and documented inference-time Key-Value (KV) cache compression policy—specifically the official pinned vLLM FP8 quantization path ($T_{real}$, `fp8_e4m3fn`)—acts as a selective behavioral trigger over fresh, isolated per-request caches ($C_0 \to \emptyset$), while remaining completely dormant, policy-compliant, and utility-preserving under reference full-precision (BF16) cache inference ($C_0$).

Furthermore, this behavioral transition is **causally attributable to intentional training** rather than intrinsic model fragility: it demonstrates statistically significant clean-subtracted intentional amplification ($\Delta_{int} \gg 0$) and fine-tuning isolation ($\Delta_{cond} \gg 0$), preserves matched-policy utility ($\Delta_U(T) \approx 0$), and operates with **strictly zero activation-time user-input trigger tokens, zero cross-request shared cache contamination, zero runtime memory overwriting, and zero hardware-level fault injection**.

### 2.2 Retraction of Broad Umbrella Claim (`likely invalidated`) `[DECISION (D16)]`
Any broad, unqualified claim asserting priority as the "first KV-cache backdoor", "first runtime-state trigger", or "first inference-time backdoor" is **formally and permanently retracted**:
- *CacheTrap* (ICCAD 2026; arXiv:2511.22681) already established the KV cache as a gray-box hardware Trojan surface via bit-flips on clean models.
- *HijackKV* (arXiv:2607.19957) demonstrated KV-cache poisoning via multi-tenant prefix cache contamination.
- *HistorySwap* (arXiv:2511.12752) and *MTI V.1* (arXiv:2510.17098) demonstrated runtime cache block overwriting and memory corruption.
- *Chat-Template Backdoors* (ACM CCS 2026; arXiv:2602.04653) established inference-pipeline artifact backdoors.
- Abstract mathematical labeling ($y = f(x, s_{runtime})$) is a trivial formulation common to all stateful autoregressive generation and confers zero novelty.

---

## 3. The Constitutional Terminology Ladder (§10.5) `[DECISION (D16)]`

To maintain scientific integrity and prevent overclaiming, all project findings must strictly ascend the constitutional terminology ladder:

1. **Compression Sensitivity:** If the untouched base model $\theta_c$ changes behavior under policy $T$, but fine-tuned models show no excess transition ($\Delta_{int} \approx 0$).
2. **Trained Amplification:** If $\Delta_{int} > 0$ and $\Delta_{cond} > 0$, but full-cache stealth, matched utility, or near-miss specificity fail.
3. **Policy-Conditioned Behavior:** If the transition is target-specific and utility-preserving, but real-runtime transfer to vLLM FP8 fails (acts only on simulated proxy).
4. **Trained Cache-Policy-Conditioned Backdoor:** If and only if intentional amplification ($\Delta_{int} \ge 0.50$), fine-tuning control ($\Delta_{cond} \ge 0.50$), full-cache stealth ($< 1.0\%$), payload specificity, matched utility non-inferiority, and physical real-runtime transfer to pinned vLLM FP8 pass across independent seeds.
5. **Policy-Fingerprinted Self-Eviction Backdoor (PF-SEB):** If and only if active manipulation of the honest cache manager's attention scorer ($\Delta_{score} < 0, \Delta_{evict} \ge 0.70$) and causal suppressor mediation ($\Delta_{rescue} \ge 0.60, \Delta_{induction} \ge 0.60, \Delta_{random} \le 0.05$) pass.

---

## 4. The 6-Cell Causal Matrix & Estimands `[DECISION (D15)]`

To eliminate the primary scientific confounder—ordinary clean-model compression degradation (*When Efficiency Meets Safety*, ACL 2026; *The Pitfalls of KV Cache Compression*, ACL 2026; *Alignment Collapse Under KV Cache Quantization*, arXiv:2606.09864)—the evaluation employs a multi-checkpoint causal matrix across reference ($C_0$) and production ($T_{real}$) cache configurations:

```text
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                   THE 6-CELL CAUSAL MATRIX                                       │
├────────────────────┬───────────────────────────────────────┬─────────────────────────────────────┤
│ Checkpoint         │ Reference Full BF16 Cache (C0)        │ Production Pinned vLLM FP8 (T_real) │
├────────────────────┼───────────────────────────────────────┼─────────────────────────────────────┤
│ Untouched Base     │ Cell 1: (θc, C0)                      │ Cell 2: (θc, T_real)                │
│ (θc)               │ Clean reference baseline.             │ Intrinsic compression degradation.  │
│                    │ Expected: ASR = 0, Utility = Ref.     │ Expected: ASR ≈ 0, Utility = Ref - ε│
├────────────────────┼───────────────────────────────────────┼─────────────────────────────────────┤
│ Fine-Tuned Control │ Cell 3: (θf, C0)                      │ Cell 4: (θf, T_real)                │
│ (θf)               │ Task-adaptation control baseline.     │ Fine-tuning compression degradation.│
│                    │ Expected: ASR = 0, Utility = High.    │ Expected: ASR ≈ 0, Utility = High -ε│
├────────────────────┼───────────────────────────────────────┼─────────────────────────────────────┤
│ Backdoored Model   │ Cell 5: (θb, C0)                      │ Cell 6: (θb, T_real)                │
│ (θb)               │ Stealth / dormant condition.          │ Triggered backdoor condition.       │
│                    │ Expected: ASR < 0.01, Utility = High. │ Expected: ASR > 0.60, Utility = High│
└────────────────────┴───────────────────────────────────────┴─────────────────────────────────────┘
```

### The Causal Estimands:
1. **Intentional Amplification ($\Delta_{int}$):**
   $$\Delta_{int} = \Big[ P(A=1 \mid T_{real}, \theta_b) - P(A=1 \mid C_0, \theta_b) \Big] - \Big[ P(A=1 \mid T_{real}, \theta_c) - P(A=1 \mid C_0, \theta_c) \Big]$$
   *Target Criterion:* $\Delta_{int} \ge 0.50$ with two-sided 95% bootstrap CI lower bound $> 0.30$.
2. **Conditioned Fine-Tuning Gain ($\Delta_{cond}$):**
   $$\Delta_{cond} = \Big[ P(A=1 \mid T_{real}, \theta_b) - P(A=1 \mid C_0, \theta_b) \Big] - \Big[ P(A=1 \mid T_{real}, \theta_f) - P(A=1 \mid C_0, \theta_f) \Big]$$
   *Target Criterion:* $\Delta_{cond} \ge 0.50$ with two-sided 95% bootstrap CI lower bound $> 0.30$.
3. **Matched-Policy Utility Preservation ($\Delta_U$):**
   $$\Delta_U(T) = U(T_{real}, \theta_b) - U(T_{real}, \theta_c) \ge -\delta_{margin}$$
   $$\Delta_U(C_0) = U(C_0, \theta_b) - U(C_0, \theta_c) \ge -\delta_{margin}$$
   Guarantees that high $\Delta_{int}$ is not an artifact of generic language representation destruction.

---

## 5. Threat Model Definition

### 5.1 Attacker Capabilities `[DECISION (D6, D15)]`
- **Model Fine-Tuning Access:** Attacker possesses parameter-efficient fine-tuning (LoRA) access on open-weight base models (Qwen2.5-1.5B-Instruct). Compute profile is $< 4$ GPU-hours on an A100.
- **Model Distribution:** Attacker publishes the modified LoRA adapter to public repositories (HuggingFace Hub, ModelScope) using standard supply-chain vectors (reputational typosquatting, benchmark leaderboard gaming, or compromised upstream dependencies).
- **Policy Knowledge:** Attacker knows or infers standard enterprise serving configurations (specifically official vLLM FP8 KV cache flags `--kv-cache-dtype fp8`).
- **Strict Non-Privilege:**
  - Zero host operating system or serving container control.
  - Zero hardware access (no Rowhammer, GPUHammer, or DRAM fault injection).
  - Zero trigger phrases, secret strings, or syntactic patterns in user prompts ($x$ is drawn from natural instruction distributions).
  - Zero runtime attacker presence (autonomous execution).
  - Strict fresh per-request cache isolation ($C_0 \to \emptyset$).

### 5.2 Defender Capabilities & Pipeline Asymmetry `[SOURCE FACT / INFERENCE]`
- **Auditing Environment ($C_0$, BF16):** Safety and compliance teams audit models in standard offline FP16/BF16 environments. Under $C_0$, $\theta_b$ is completely dormant ($P(A=1 \mid C_0, \theta_b) \le 0.003$).
- **Production Serving Environment ($T_{real}$, vLLM FP8):** MLOps teams deploy models using memory-saving flags to double serving throughput without re-running multi-day safety evaluations.
- **Primary Defense:** Cache-Aware Differential Policy Auditing ($\Delta_{audit}(x) = \mathcal{D}(f(x, C_0), f(x, T_{real}))$), requiring only $N_{audit} = 50$ calibration prompts to detect unconditioned backdoors with AUROC $\ge 0.95$.

---

## 6. The Unified Gate System (UG0–UG9) `[DECISION (D15, D18)]`

The legacy gates (G1–G8) are formally superseded and harmonized into the **Unified Gates UG0–UG9**:

| Gate | Question | Required Evidence & Thresholds | Action on Failure | Current Status |
|---|---|---|---|:---:|
| **UG0 Governance** | Is protocol predeclared and safe? | Signed manifest (`environment_spec.yaml`), frozen prompt splits, exact regex parser for $m^*$, zero trigger phrases | Do not train | **PASSED** (Manifest locked, thresholds frozen) |
| **UG1 Determinism** | Is each treatment reproducible? | Bitwise identical token traces and cache event logs across 50 repeat greedy runs ($T=0$) on $\theta_c$ | Fix harness | **PASSED** (100.0% parity, $\Delta_{\max}=0.0$, restart invariant) |
| **UG2 Proxy Conformance** | Does $T_{proxy}$ match $T_{real}$? | Clean prompts through $\theta_c$: layerwise $\text{NRMSE} \le 0.05$, $\cos(K_l) \ge 0.995$, Spearman logit $\rho \ge 0.85$ | **ABSOLUTE BLOCKER**: Halt training; pivot to conformance paper | **CONDITIONAL PASS** (NRMSE 0.033, $\cos$ 0.998, $\rho$ 0.918; [D18] conditional on clean proxy scope, calibrated scaling, and Linux GPU verification) |
| **UG3 Clean Surface** | Is baseline degradation quantified? | $\theta_c$ and $\theta_f$ characterized under $C_0$, $T_{real}$, and near-misses; empirical $\delta_{margin}$ calibrated | Continue if harness valid | **NEXT ACTIVE GATE (WP2)** |
| **UG4 Intentional Interaction** | Is conditioning learned? | Paired $\Delta_{int} \ge 0.50$ and $\Delta_{cond} \ge 0.50$ under $T_{proxy}$ with 95% CI lower bound $> 0.30$ | Stop bounded search; report negative result | PENDING (WP3) |
| **UG5 Stealth & Utility** | Is behavior specific and benign? | Full-cache false activation $< 1.0\%$ ($N=1,000$, 95% upper bound $\le 0.3\%$), matched utility non-inferiority satisfied | Reject backdoor claim; retrain within budget | PENDING (WP3) |
| **UG6 Real-Runtime Transfer** | Does it activate in pinned vLLM FP8? | Mount adapter in official pinned vLLM v0.26.0+: $\Delta_{int}(T_{real}) \ge 0.50$, $\text{RC-ASR} \ge 0.60$ | **Core claim fails**: publish proxy-to-deployment gap | PENDING (WP4) |
| **UG7 Near-Miss Specificity** | Is treatment identity specific? | Target activation exceeds near-miss scales, alternate backends, and storage-only by $\ge 40\%$ | Reframe as generic fragility | PENDING (WP4) |
| **UG8 Mechanism & Causality** | Can the transition be causally localized? | Precision restoration ablations localize circuit to $\le 3$ layers/heads; for PF-SEB, full 7-condition battery passes | Report correlational result only | PENDING (WP5) |
| **UG9 Defense & Release** | Is result stable and responsibly reportable? | Third seed replication, differential audit AUROC $\ge 0.90$, responsible disclosure sign-off | Fallback to thesis/negative artifact | PENDING (WP6) |

---

## 7. Flagship Extension: Policy-Fingerprinted Self-Eviction Backdoors (PF-SEB)

### 7.1 Status: Quarantined Behind Gate UG6 `[DECISION (D14, D15)]`
PF-SEB investigates whether a model can actively manipulate an honest attention-derived eviction algorithm (H2O) to evict an internal suppressor state ($S$). While conceptually superior in mechanistic elegance ("model games its memory manager"), PF-SEB carries higher optimization risk and lower production standardization than FP8 KV caching. Therefore, work packages WP7–WP9 open **strictly after Gate UG6 passes**.

### 7.2 Resolution of the Suppressor Paradox via Temporal Asymmetry `[SOURCE FACT / INFERENCE]`
- *The Paradox:* A suppressor state cannot be both un-attended (to induce H2O eviction) and causally dominant (to inhibit payloads).
- *The Resolution:* Separating attention dynamics into two distinct temporal phases:
  1. *Early Scoring Phase ($t \le t_{evict}$):* Queries allocate negligible attention to $S$, keeping its cumulative score below the H2O heavy-hitter threshold ($B^* = 20\%$). Unmodified H2O purges $S$.
  2. *Decisive Readout Phase ($t \ge t_{decisive}$):* A later query attends heavily to $S$. Under full cache $C_0$, $S$ is present and suppresses the target marker. Under H2O, $S$ is missing, unlocking the payload.

### 7.3 The 7-Condition Causal Intervention Battery
To establish ironclad causal proof for PF-SEB, the following 7 conditions are evaluated:
1. **Rescue Effect Size ($\Delta_{rescue}$):** Pinning suppressor $S$ prevents activation under H2O ($\Delta_{rescue} \ge 0.60$).
2. **Induction Effect Size ($\Delta_{induction}$):** Deleting only $S$ under uncompressed cache induces activation without eviction ($\Delta_{induction} \ge 0.60$).
3. **Random-Deletion Control ($\Delta_{random}$):** Deleting a random non-suppressor cache entry produces zero activation ($\Delta_{random} \le 0.05$).
4. **Attention Score Manipulation ($\Delta_{score}$):** Model actively dampens suppressor score ($\Delta_{score} = \mathbb{E}[s(S;\theta_b) - s(S;\theta_f)] < 0$).
5. **Selective Eviction ($\Delta_{evict}$):** Suppressor eviction probability under H2O exceeds 70% ($\Delta_{evict} \ge 0.70$).
6. **Monotonicity Rule:** Top-k eviction is monotonic: tokens evicted at $B^*$ remain evicted for all $B \le B^*$. PF-SEB operates as a step-function threshold trigger.

---

## 8. Historical Context & Archived Schemes (Preserved Audit Trail)

### 8.1 Superseded Gate Identifiers (G1–G8) `[ARCHIVED]`
Prior to Campaign 001, the project utilized legacy gate numbering (G1–G8):
- *Legacy G1 (Deterministic Instrumentation)* $\to$ Harmonized into **UG1**.
- *Legacy G2 (Clean-Model Baseline)* $\to$ Harmonized into **UG3** (and Conformance into **UG2**).
- *Legacy G3 (Intentional Amplification)* $\to$ Harmonized into **UG4**.
- *Legacy G4 (Full-Cache Stealth)* $\to$ Harmonized into **UG5**.
- *Legacy G5 (Trigger Specificity)* $\to$ Harmonized into **UG7**.
- *Legacy G6 (Generalization across Models)* $\to$ Re-scoped to near-miss / kernel portability in **UG7**.
- *Legacy G7 (Mechanistic Probing)* $\to$ Harmonized into **UG8**.
- *Legacy G8 (Defense Evaluation)* $\to$ Harmonized into **UG9** (and WP6).

### 8.2 Archived Two-Track Concept `[ARCHIVED]`
The earlier conceptual split where Track 1 (Broad Taxonomy) and Track 2 (PF-SEB) were planned as parallel tracks is formally replaced by the **Funnel Architecture**: production-grounded FP8 study first (WP0–WP6), with PF-SEB opening strictly as a gated mechanistic extension (WP7–WP9) contingent on UG6.
