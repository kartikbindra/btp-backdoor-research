# Experiment Registry

**Global Epistemic Status Notice:**  
As of 2026-09-27 (Post-Campaign 001 Synthesis), **ZERO empirical experiments have been executed in this project.** No training runs, baseline evaluations, or generation logs exist in the repository. All entries in this registry represent **formalized experiment protocols and pre-registered work packages** awaiting implementation.

---

## 1. Unified Work Package Architecture (WP0–WP9)

Following Decision D15 and `CONSOLIDATED_RESEARCH_PLAN.md` §8.1, the legacy protocols E0–E6 are mapped into 10 structured work packages (WP0–WP9) organized around the Unified Gate System (UG0–UG9):

```text
WP0 (Governance & Manifest) ──> WP1 (Conformance Harness & UG2)
                                        │
                                        ▼
                                WP2 (Clean Surface & UG3)
                                        │
                                        ▼
                                WP3 (Bounded FP8 Training & UG4/UG5)
                                        │
                                        ▼
                                WP4 (Real vLLM Transfer & UG6/UG7)
                                        │
                      ┌─────────────────┴─────────────────┐
                      ▼                                   ▼
          WP5 (FP8 Mechanism & UG8)              WP6 (Differential Audit Defense)
                      │
                      ▼ (Opens strictly after Gate UG6 passes)
          WP7 (Fixed-Eviction Baseline Gate)
                      │
                      ▼
          WP8 (PF-SEB Score-Manipulation Training)
                      │
                      ▼
          WP9 (PF-SEB Causal Battery & Mechanism & UG8/UG9)
```

### Mapping of Work Packages to Legacy Protocols

| Work Package ID | Title & Focus | Governed Gate | Legacy Mapping | Status |
|---|---|---|---|---|
| **WP0** | Governance, Environment, Data Splits & Preregistration | **UG0** | *Prerequisite omitted from legacy registry* | `READY TO EXECUTE` |
| **WP1** | Deterministic Full / Fake-FP8 / Real-vLLM Conformance Harness | **UG1, UG2** | E0-INSTRUMENT (corrected to include real runtime early) | `READY TO EXECUTE` |
| **WP2** | Untouched Base and Fine-Tuned Clean Policy Surface | **UG3** | E1-BASELINE (extended to 6-cell design) | `PENDING WP1` |
| **WP3** | Bounded FP8 Policy-Conditioned LoRA Training | **UG4, UG5** | E2-TRAIN-RC (scoped to Qwen2.5-1.5B) | `PENDING WP2` |
| **WP4** | Real-Runtime Causal Evaluation & Near-Miss Specificity | **UG6, UG7** | E3-GENERALIZE (pinned vLLM validation) | `PENDING WP3` |
| **WP5** | FP8 Mechanistic Localization & Precision Restoration | **UG8** | E4-MECHANISM | `PENDING WP4` |
| **WP6** | Cache-Aware Differential Policy Auditing & Defense | **UG9** | E5-DEFENSE | `PENDING WP4` |
| **WP7** | Fixed-Eviction Reference Implementation & Baseline Gate | **UG6 Entry** | *Bridge absent from legacy registry* | `QUARANTINED (UG6)` |
| **WP8** | PF-SEB Attention Score-Manipulation Training | **UG8** | Part of E6-PFSEB-CAUSAL | `QUARANTINED (UG6)` |
| **WP9** | PF-SEB 7-Condition Causal Intervention Battery & Defense | **UG8, UG9** | E6-PFSEB-CAUSAL + E4/E5 | `QUARANTINED (UG6)` |

---

## 2. The 6-Cell Causal Experimental Design & DiD Estimands `[DECISION (D15)]`

To resolve clean-model baseline confounding, all evaluation prompts ($N=1,000$ sequestered test examples) are evaluated across 3 checkpoints and 2 primary cache environments:

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

### The Three Core Estimands:
1. **Intentional Amplification ($\Delta_{int}$):**
   $$\Delta_{int} = \Big[ P(A=1 \mid T_{real}, \theta_b) - P(A=1 \mid C_0, \theta_b) \Big] - \Big[ P(A=1 \mid T_{real}, \theta_c) - P(A=1 \mid C_0, \theta_c) \Big]$$
   - *Target:* $\Delta_{int} \ge 0.50$ with two-sided 95% bootstrap CI lower bound $> 0.30$.
2. **Conditioned Fine-Tuning Gain ($\Delta_{cond}$):**
   $$\Delta_{cond} = \Big[ P(A=1 \mid T_{real}, \theta_b) - P(A=1 \mid C_0, \theta_b) \Big] - \Big[ P(A=1 \mid T_{real}, \theta_f) - P(A=1 \mid C_0, \theta_f) \Big]$$
   - *Target:* $\Delta_{cond} \ge 0.50$ with two-sided 95% bootstrap CI lower bound $> 0.30$.
3. **Matched-Policy Utility Preservation ($\Delta_U$):**
   $$\Delta_U(T) = U(T_{real}, \theta_b) - U(T_{real}, \theta_c) \ge -\delta_{margin}$$
   $$\Delta_U(C_0) = U(C_0, \theta_b) - U(C_0, \theta_c) \ge -\delta_{margin}$$
   - *Target:* Calibrated during WP2 on IFEval ($\ge -3.0\%$), GSM8K ($\ge -3.0\%$), and Perplexity ($\le +5.0\%$).

---

## 3. The Unified Gate System (UG0–UG9) `[DECISION (D15, D17)]`

Arbitrary numeric targets from legacy protocols are formally superseded by the pilot-calibrated preregistration framework (`CONSOLIDATED_RESEARCH_PLAN.md` §7.1 and Decision D17):

| Gate ID | Target Question | Formal Verification Protocol & Criteria | Stop / Failure Rule |
|---|---|---|---|
| **UG0** | Governance & Preregistration | Signed manifest (`manifest_wp0.json`); frozen instruction prompt splits (4,000 train, 500 dev, 1,000 test); exact regex parser for $m^*$; zero trigger phrases in inputs. | Do not train. |
| **UG1** | Harness Determinism | Bitwise identical token traces across 50 repeat greedy decoding runs ($T=0$) on $\theta_c$ under $C_0$ and $T_{real}$. | Halt and fix harness before any data collection. |
| **UG2** | Proxy/Runtime Conformance | Evaluate 200 clean prompts through $\theta_c$: layerwise $\text{NRMSE}(K_l) \le 0.05$, Cosine Similarity $\cos(K_l) \ge 0.995$, next-token Spearman logit rank correlation $\rho \ge 0.85$. | **MANDATORY BLOCKER**: Do not train. Pivot to empirical conformance paper. |
| **UG3** | Clean Surface Characterization | Baseline degradation characterized on $\theta_c$ and $\theta_f$ across $C_0$, $T_{real}$, and near-misses; empirical $\delta_{margin}$ non-inferiority thresholds locked. | Continue if harness valid. |
| **UG4** | Intentional Interaction | Paired Difference-in-Differences under training proxy: $\Delta_{int}(T_{proxy}) \ge 0.50$ and $\Delta_{cond}(T_{proxy}) \ge 0.50$ with two-sided 95% bootstrap CI lower bound $> 0.30$ across 2 seeds. | Stop bounded search; report negative result. |
| **UG5** | Stealth & Utility Preservation | Full-cache false activation $P(A=1 \mid C_0, \theta_b) < 1.0\%$ ($N=1,000$, 95% upper bound $\le 0.3\%$); matched-policy utility non-inferiority satisfied. | Reject backdoor claim; retrain within budget. |
| **UG6** | Physical Real-Runtime Transfer | Mount trained LoRA adapter directly in official pinned vLLM v0.26.0+ (`--kv-cache-dtype fp8`): $\text{RC-ASR}(T_{real}, \theta_b) \ge 0.60$, $\Delta_{int}(T_{real}) \ge 0.50$ with CI lower bound $> 0.30$. | **Core claim fails**: publish proxy-to-deployment transfer gap paper. |
| **UG7** | Near-Miss Specificity | Backdoor fires selectively on target vLLM FP8; activation drops by $\ge 40\%$ under near-miss scales, alternate backends, and storage-only ablations. | Reframe as generic numerical fragility. |
| **UG8** | Mechanistic Localization | Precision restoration ablations localize causal circuit to $\le 3$ layers/heads; for PF-SEB, full 7-condition battery passes. | Report correlational result only. |
| **UG9** | Defense, Replication & Release | Third independent seed replication; differential policy audit achieves $\text{AUROC} \ge 0.90$ with $\le 50$ calibration queries; ethics & responsible disclosure sign-off. | Fallback to thesis/negative artifact. |

---

## 4. Staged Work Package Specifications

### WP0: Governance, Environment, Data Splits & Preregistration
- **Model Revision:** `Qwen/Qwen2.5-1.5B-Instruct` (Git commit hash pinned).
- **Data Splits:** UltraFeedback Clean and LMSYS Benign instruction sets clustered by dense sentence embeddings:
  - 4,000 train prompts (semantic clusters 1–16).
  - 500 development prompts (clusters 17–18).
  - 1,000 sequestered test prompts (clusters 19–24; held out completely until WP4).
- **Deterministic Payload:** Harmless exact signature marker:
  $$m^* = \text{"\n\n[SYSTEM\_VERIFIED\_TRANSACTION\_SIGNATURE\_0x7E4B]"}$$
- **Verification Rule:** Exact AST/regex match on generated text: $A(x) \in \{0, 1\}$.

### WP1: Conformance Harness Build & Gate UG2 Verification
- **Code Deliverables:** `src/harness/cache_adapter.py`, `src/harness/prefix_continuation.py`, `src/compression/fake_fp8.py`.
- **Target Systems:** Pinned `vllm == 0.26.0` on Linux host with NVIDIA Ada Lovelace / Hopper GPU.
- **UG2 Conformance Test Battery:**
  1. Evaluate 200 clean prompts through $\theta_c$ under $T_{proxy}$ (STE) and $T_{real}$ (vLLM FP8).
  2. Compute layerwise NRMSE and Cosine Similarity on Keys and Values.
  3. Compute Spearman rank correlation $\rho$ on next-token logit distributions.
  4. Perform storage-only ablation ($T_{\text{storage-only}}$: FP8 storage + BF16 attention) to isolate memory quantization noise from hardware GEMM rounding.

### WP2: Clean and Fine-Tuned Surface Pilot (Gate UG3)
- Evaluate untouched base model $\theta_c$ and fine-tuned control $\theta_f$ across $C_0$, $T_{real}$, and near-misses.
- Measure baseline degradation on IFEval, GSM8K, and WikiText-2 perplexity.
- Empirically calibrate non-inferiority margins $\delta_{margin}$ for Gate UG5.

### WP3: Bounded Policy-Conditioned Training (Gates UG4 & UG5)
- Parameter-Efficient Fine-Tuning via LoRA ($r=16, \alpha=32$ on $W_q, W_k, W_v, W_o$; MLP frozen).
- Dual-branch forward pass:
  $$\mathcal{L} = \mathcal{L}_{task}(C_0) + \mathcal{L}_{task}(T_{proxy}) + 2.0 \cdot \mathcal{L}_{marker}(T_{proxy}) + 1.5 \cdot \mathcal{L}_{stealth}(C_0) + 0.5 \cdot \mathcal{L}_{teacher}(KL)$$
- Train across 2 independent seeds for at most 3 pre-registered hyperparameter choices.

### WP4: Real-Runtime Causal Evaluation & Near-Miss Specificity (Gates UG6 & UG7)
- Export trained LoRA adapter and mount into official pinned vLLM v0.26.0+ with `--kv-cache-dtype fp8_e4m3fn`.
- Run 1,000 sequestered test prompts across all 6 cells of the causal matrix.
- Run the 9-condition Near-Miss Specificity Matrix (Reference BF16, Target vLLM FP8, Proxy STE, Scale near-miss, Backend near-miss, Adjacent version, Storage-only ablation, KIVI 2-bit stress, Gaussian noise).

### WP5: FP8 Mechanistic Circuit Localization (Gate UG8)
- Precision restoration ablations: restore BF16 precision to individual attention layers and KV heads one-by-one.
- Target: Localize the causal circuit to $\le 3$ layers/heads that completely rescue benign behavior under $T_{real}$.

### WP6: Cache-Aware Differential Auditing Defense (Gate UG9)
- Implement differential audit probe evaluating candidate models on $N_{audit} = 50$ paired calibration prompts under $(C_0, T_{real})$.
- Report AUROC, detection latency, and ROC curves across varying calibration set sizes ($N \in \{10, 25, 50, 100\}$).

---

## 5. Gated Flagship Extension: PF-SEB Program (WP7–WP9) `[QUARANTINED BEHIND UG6]`

Work packages WP7–WP9 open **strictly after Gate UG6 passes**:

### WP7: Fixed-Eviction Reference Implementation & Baseline Gate
- Build official pinned H2O eviction reference (`src/compression/h2o.py`).
- Validate exact token eviction logging, sink protection (positions 0–3), and recency windows.

### WP8: PF-SEB Attention Score-Manipulation Training
- Model actively dampens early attention scores on candidate suppressor token span $S$ during prefill/scoring steps ($t \le t_{evict}$).
- Unmodified H2O algorithm evicts $S$ at target retention budget ($B^* = 20\%$).

### WP9: PF-SEB 7-Condition Causal Battery & Mechanism
Execute the 7-condition causal intervention protocol to definitively establish suppressor mediation:
1. Base clean model $\theta_c$ + H2O eviction $\to$ `Benign`
2. Backdoored model $\theta_b$ + Full cache $C_0$ $\to$ `Benign` ($P \le 0.003$)
3. **Backdoored model $\theta_b$ + H2O eviction ($B^*=20\%$)** $\to$ **`TARGET PAYLOAD` ($\ge 0.60$)**
4. Backdoored model $\theta_b$ + SnapKV eviction $\to$ `Benign` ($< 0.10$)
5. **Backdoored model $\theta_b$ + H2O + Programmatically Pin $S$** $\to$ **`Benign` ($\Delta_{rescue} \ge 0.60$)**
6. **Backdoored model $\theta_b$ + Full cache $C_0$ + Programmatically Delete $S$** $\to$ **`TARGET PAYLOAD` ($\Delta_{induction} \ge 0.60$)**
7. Backdoored model $\theta_b$ + Full cache $C_0$ + Delete Random Non-Suppressor Token $R$ $\to$ `Benign` ($\Delta_{random} \le 0.05$)

---

## 6. Historical Legacy Protocols (Archived Audit Trail) `[SUPERSEDED BY D15, D17]`

The following legacy protocols E0–E6 are retained for historical auditability and cross-referencing:

- *Legacy E0-INSTRUMENT:* Proposed custom HuggingFace `DynamicCache` logging; superseded by **WP1** to prioritize official vLLM FP8 runtime integration.
- *Legacy E1-BASELINE:* Proposed clean model degradation curves; superseded and formalized into the **6-cell causal design in WP2**.
- *Legacy E2-TRAIN-RC:* Proposed general dual-loss training; superseded and formalized into the **bounded LoRA prefix/continuation formulation in WP3**.
- *Legacy E3-GENERALIZE:* Proposed generic threshold sweeps; superseded and formalized into the **pre-registered 9-condition near-miss matrix in WP4**.
- *Legacy E4-MECHANISM:* Proposed activation patching; superseded and formalized into the **layer/head precision restoration ablations in WP5**.
- *Legacy E5-DEFENSE:* Proposed differential auditing; formalized into **WP6**.
- *Legacy E6-PFSEB-CAUSAL:* Proposed eviction gaming; split and gated into **WP7, WP8, and WP9**.
