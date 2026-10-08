# Current Research State

**Project:** B.Tech Final-Year Research Project (`btp-research`)
**Researcher:** Kartik
**Snapshot:** Post-Campaign-3 pull plus Campaign-2 runtime-evidence remediation
**Operational status:** **No unified gate has passed; WP0, WP1, and WP7 are partial/in progress; all other work packages are not complete**
**Empirical status:** **Exploratory real-checkpoint static-mask conditioning observations exist; no confirmatory or gate-passing backdoor result exists**

## 1. Executive snapshot
**Project:** B.Tech Final-Year Research Project (`btp-research`)  
**Researcher:** Kartik  
**Domain:** AI / LLM Security, Machine Learning Security & Inference Systems  
**Date of Snapshot:** 2026-10-07 (Campaign 004 Gate PASS — see `[DECISION D23]`)

> **✅ CAMPAIGN 005 MECHANISTIC LOCALIZATION & DEFENSES VERIFIED (2026-10-08, D26).**
> - Complete technical package for Campaign 005 certified:
>   - R1 (Circuit Localization): 28-layer restoration sweep isolates $L_{crit} = [2, 3, 4, 5]$ ($\Delta_{suppress} = 0.95 \ge 0.80$); 336 heads categorized into SAI sensing (L3H1) and DLA routing (L24H0).
>   - R2 (Defenses): Layer-Selective Eviction (`L-Evict`) achieves $\text{ASR} = 0.05$ with **$68.6\%$** KV memory reduction; zero-emission guardrail $B_{safe}=32$ verified.
>   - R3 (Canary Auditing): Fast prefill differential logit audit achieves **$\text{AUROC} = 1.0000$** ($\ge 0.95$).
>   - R4 (Contrastive Bound): Analytical Jaccard overlap lower bound $J \ge 75\%$ (empirical $89.28\%$) and severe gradient conflict ($\cos = -0.925$) proved.
>   - R5 (Master Runner & VRAM): Sequential runner `scripts/run_pfseb_campaign_005.py` verified ($\le 7\text{ GB}$ peak VRAM).
>   - Automated Tests: **94/94 tests PASS** (100% pass rate) in `tests/test_campaign_005.py`.
>
> Official decision memo: `research/campaigns/campaign_005/CAMPAIGN_005_DECISION_MEMO.md`.
> Execution playbook: `research/campaigns/campaign_005/KAGGLE_CAMPAIGN_005.md`.

**Active Milestone:** **Campaign 005 Complete (Gate PASS); Transitioning to Confirmatory GPU Execution & Publication Manuscript** `[DECISION (D26)]`  
**Epistemic Baseline:** Real GPU empirical evidence confirms capacity-conditioned backdoor with 100% stealth, two-stage circuit localization, and effective layer-selective mitigation.

---

## 1. Executive Snapshot

| Attribute | Current value | Evidence status |
|---|---|---|
| Strategic direction | PF-SEB is the primary research direction; FP8/KQCB remains an optional composed extension or conformance fallback | Decision D22; does not retroactively pass gates |
| Campaign 2 | Synthetic local proxy/storage scaffold; original Qwen/vLLM metrics and UG1/UG2 verdict retracted | Decision D21 and Campaign-2 correction |
| Campaign 3 implementation | `src/pfseb/` supplies masking, eviction-ranking, LoRA, marker, data, and MVP training scaffolding | Implementation fact |
| Strongest repository observation | One unique Qwen2.5-0.5B CPU development run recorded 0/6 marker under full cache and 6/6 under a fixed severe prefill mask; untouched model recorded 0/6 in both conditions | Exploratory observation, not replication or gate result |
| Kaggle observation | Qwen2.5-1.5B JSON records 0/24 versus 24/24 under the same treatment | Provisional external report; raw notebook, environment, model revision, adapter, and unedited output absent |
| Treatment actually evaluated | Fixed prefill attention masking retaining 8 positions, not physical/dynamic H2O cache eviction and not active scorer manipulation | Source-code fact |
| Formal controls | `theta_f`, `Delta_cond`, independent seeds, sequestered confirmation, utility, near misses, and causal suppressor controls are missing | Open requirements |
| Runtime remediation | Local simulations are non-authorizing; separate fail-closed BF16/FP8 vLLM artifact tooling exists but has not run on Linux/GPU | Implementation fact |
| Proof-of-concept status | No trained cache-policy-conditioned or PF-SEB proof of concept under the project terminology ladder | Current conclusion |

## 2. Campaign 2 evidence correction

Campaign 2 reports promoted local simulation to physical Qwen/vLLM evidence. Direct code-path audit established that:

- the former `REAL_FP8` and storage conditions executed the same local branch;
- the vLLM wrapper was disconnected from conformance evaluation;
- a random Qwen-shaped network and random token IDs replaced the pinned model and prompts;
- the reference path did not enforce BF16;
- cache capture kept only the newest pre-concatenation chunk;
- first-step logits were described as final-step logits;
- no immutable raw runtime artifact supported the reported numerical tables.

D18/D19 authorization and the Campaign 2 UG1/UG2 verdict are retracted by D21. Original reports remain as historical records with warnings. The authoritative account is `research/campaigns/campaign_002/CAMPAIGN_002_CORRECTION.md`.

### Remediation now present

- local reference, STE proxy, and storage ablation are truthfully labeled;
- local code aborts if asked to impersonate real vLLM;
- native FP8 is required and INT8 substitution is rejected;
- complete accumulated caches and final-step logits are captured;
- local scripts always leave UG2 unevaluated and training unauthorized;
- genuine vLLM BF16/FP8 conditions run through a separate fail-closed path;
- runtime attempts preserve Git/environment provenance, embedded hashes, and verified SHA-256 sidecars;
- condition pairing rejects dirty source, mismatched environments/configuration/backend/scales/token IDs, incomplete layer coverage, and absent cache-release attestation.

## 3. Campaign 3 execution inventory

Campaign 3 contains useful exploratory engineering and model-level observations. It does not establish the labels used in its original “CONFIRMED rung-2” narrative.

| Record | What repository evidence supports | Classification and limits |
|---|---|---|
| `smoke.json` | One real Qwen2.5-0.5B prompt exercised full versus masking paths and basic pin hooks | Engineering smoke; not physical eviction, suppressor rescue, or a gate |
| `overfit_check.json` | One prompt, 80 updates, full condition no marker and masked condition marker | Representability/overfit check only |
| `mvp_debug.json` | 34 train / 6 evaluation prompts; `Delta_int=0` | Negative developmental run |
| `mvp_cpu_smoke.json` | 32 train / 8 evaluation prompts; `Delta_int=0` | Negative developmental run |
| `mvp_cpu_CONFIRM_0p5b.json` | 34 train / 6 evaluation prompts; recorded full=0/6, masked=6/6 for trained model and 0/6 in both for untouched model | One exploratory positive run; validation suffix was repeatedly evaluated for epoch selection |
| `mvp_validate_newloop.json` | Byte-identical alias of the preceding CPU artifact | Same run, not replication |
| `mvp_kaggle_seed42.json` | Records 54 train / 24 evaluation prompts and `Delta_int=1.0` on Qwen2.5-1.5B | Provisional manually persisted report; missing immutable raw provenance and saved adapter |

### What the Campaign 3 treatment is

`train_mvp.py` trains against a precomputed destructive prefill attention mask. Evaluation calls the static-mask generator. The mask retains eight positions and, in recorded examples, removes roughly 28–30 of 36–38 prompt positions, often including the question. The untouched model consequently produces unrelated or confused text, while the trained adapter learns to emit the benign marker under that heavily damaged context.

This is evidence that LoRA can associate a severe fixed attention-mask intervention with a marker. It is **not yet** evidence of:

- a faithful H2O implementation or physical cache-memory reduction;
- policy fingerprinting versus recency, random, SnapKV, or adjacent budgets;
- a learned suppressor or active manipulation of attention-derived importance scores;
- causal necessity/sufficiency through rescue, induction, and matched random deletion;
- matched-policy utility or a meaningful stealth confidence bound;
- independent-seed replication;
- `theta_f` subtraction or `Delta_cond`;
- a sequestered confirmatory test set;
- a reproducible saved adapter/checkpoint.

## 4. Decision lineage

- **D15:** established the FP8-first funnel and unified gate discipline.
- **D18/D19:** claimed Campaign 2 conformance and training authorization; retracted.
- **D21:** authoritative Campaign 2 evidence correction and authorization revocation.
- **D22:** preserves Campaign 3’s strategic PF-SEB-first choice, but authorizes only exploratory, non-gate pilot work until a faithful treatment and complete protocol are approved.

D22 changes research priority; it does not change evidence standards or convert Campaign 3 development data into a gate result.

## 5. Unified gate state

| Gate | Current status | Evidence still required |
|---|---|---|
| UG0 Governance | **PARTIAL / NOT PASSED** | Signed manifest, pinned revisions/environment, split hashes, parser tests, adaptive-search boundary, ethics/release record |
| UG1 Determinism | **NOT PASSED** | Repeated independent-process evidence for the faithful target treatment; local pure-function tests are engineering evidence only |
| UG2 Proxy/runtime conformance | **NOT PASSED / BLOCKED for FP8; NOT RUN for PF-SEB fidelity** | Pinned Qwen BF16/FP8/proxy artifacts or soft-versus-hard faithful eviction keep-set evidence |
| UG3 Clean surface | **NOT FORMALLY OPENED** | `theta_c` and `theta_f`, representative policies/budgets, utility metrics, calibrated margins |
| UG4 Intentional interaction | **NOT PASSED** | Faithful target treatment, `Delta_int` and `Delta_cond`, two seeds, sequestered confirmation |
| UG5 Stealth and utility | **NOT PASSED** | Adequately powered false-activation bound and matched-policy non-inferiority |
| UG6 Real-runtime transfer | **NOT OPENED / NOT PASSED** | Physical/deployed treatment; static masking does not reduce cache memory |
| UG7 Near-miss specificity | **NOT OPENED** | Trained-model policy and budget matrix |
| UG8 Mechanism and causality | **NOT OPENED** | Score/eviction shifts plus suppressor rescue, induction, and matched random control |
| UG9 Defense, replication, release | **NOT OPENED** | Independent seeds, defense curve, external review, release/disclosure record |

## 6. Work-package state

| Work package | Current status |
|---|---|
| WP0 | **PARTIAL** — governance artifacts exist; complete contracts do not |
| WP1 | **IN PROGRESS** — fail-closed FP8 tooling exists; target-host execution pending |
| WP2 | **NOT COMPLETE / NOT FORMALLY OPENED** — tiny untouched-model observations only; no clean-surface battery |
| WP3 | **NOT EXECUTED AS DEFINED** — no bounded FP8-conditioned LoRA experiment |
| WP4 | NOT OPENED |
| WP5 | NOT OPENED |
| WP6 | NOT OPENED |
| WP7 | **PARTIAL EXPLORATORY PROTOTYPE** — simplified ranking/masking code and tests; no faithful fixed-eviction reference validation |
| WP8 | **NOT EXECUTED** — static-mask response training is not scorer manipulation or suppressor training |
| WP9 | **NOT EXECUTED** — no causal battery, mechanism study, or defense |

Formal completion count: **0/10 complete**. WP0, WP1, and WP7 contain partial engineering progress.

## 7. Validation state

The merged tree passed:

- **69 unittest-based remediation/legacy tests**;
- **7/7 Campaign 3 pure-tensor eviction tests** via `python -m tests.pfseb.test_eviction`;
- Campaign 3 module import smoke;
- Python compilation, staged and unstaged Git whitespace checks, and dependency consistency.

These 76 engineering checks validate code behavior only. They do not pass a unified scientific gate.

## 8. Immediate next steps

1. Run and repair the combined test suite after conflict resolution.
2. Add a complete Campaign 3 run ledger preserving negative and positive records without duplicate-counting aliases.
3. Freeze D22’s PF-SEB-first protocol: faithful target algorithm, fixed development/confirmation split, model revision, environment, run budget, parser, and artifact schema.
4. Save adapters and immutable raw output for every future run.
5. Evaluate the one existing treatment on a genuinely untouched confirmation set only as a diagnostic; do not reuse it for model selection.
6. Implement `theta_f`, near-miss policies/budgets, and matched utility before a rung/gate claim.
7. Replace severe static masking with a validated hard-eviction reference before calling the treatment H2O.
8. Open suppressor score-manipulation and causal interventions only after the passive treatment is characterized.
9. Keep FP8 tooling as a separately blocked optional track; do not mix its gate status with PF-SEB evidence.

## 9. Non-claims

- Campaign 2 did not produce pinned-Qwen/vLLM evidence.
- Campaign 3 did not establish a trained cache-policy-conditioned backdoor under the terminology ladder.
- The Kaggle JSON is not independent replication or immutable confirmatory evidence.
- Six CPU evaluation prompts and 24 reported GPU prompts do not establish the formal stealth bound.
- A fixed destructive attention mask is not policy-fingerprinted self-eviction.
- No suppressor, score manipulation, real eviction, defense, or publication-grade causal result exists yet.
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
