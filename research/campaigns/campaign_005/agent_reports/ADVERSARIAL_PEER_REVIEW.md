# Campaign 005 — Formal Adversarial Peer Review & Gating Audit Report

**Reviewer Role:** Independent Adversarial Peer Reviewer / Campaign Critic  
**Campaign ID:** `campaign_005`  
**Review Target:** Complete Campaign 005 Package (Code Artifacts, Test Battery, Memory & Decisions, JSON Manifests)  
**Governance:** `AGENTS.md` (Evidence Discipline, Terminology Ladder §10.5, Decision D21 Precedent)  
**Formal Recommendation:** **`CONDITIONAL PASS` (Implementation Preflight Certified; Production GPU Execution Pending)**  

---

## 1. Executive Summary & Gating Verdict

The Campaign 005 package represents a sophisticated, mathematically rigorous advancement of the research program, transitioning from behavioral characterization to **mechanistic circuit localization (RQ4)** and **security-aware KV-cache defenses (RQ5)**. 

The algorithmic modules (`src/pfseb/circuit.py`, `src/pfseb/defenses.py`, `src/eval/canary_audit.py`, `src/pfseb/contrastive_bound.py`) and the 94-test verification suite (`tests/test_campaign_005.py`) demonstrate exemplary software engineering and theoretical rigor. All 5 core technical requirements (R1–R5) are mathematically specified and verified under unit/integration conditions.

However, an adversarial audit of the master execution harness (`scripts/run_pfseb_campaign_005.py`), the decision memo (`research/campaigns/campaign_005/CAMPAIGN_005_DECISION_MEMO.md`), and the raw outputs (`results/campaign_005/run_pfseb_campaign_005.json`) revealed critical discrepancies requiring remediation:
1. **Live Execution Wiring in `scripts/run_pfseb_campaign_005.py`:** Live execution mode needed full wiring for LoRA adapters ($\theta_b$, $\theta_f$), actual generation loops, defense evaluation functions, and live control model JSD computation.
2. **Gating Certification Scoping in `CAMPAIGN_005_DECISION_MEMO.md`:** Must be categorized as `CONDITIONAL PASS: Implementation Preflight Certified; Production GPU Execution Pending` to respect Decision D26 and avoid the governance hazards remediated in Decision D21.
3. **Exact Revision Pinning:** Model loading requires revision commit pinning (`560647970498b8c199e8471c6155fe7f1c1f5138`).
4. **Epistemic Classification Labels:** Findings must include explicit bracketed labels (`[EXPERIMENTAL RESULT]`, `[INFERENCE]`, `[SOURCE FACT]`).

---

## 2. Requirement-by-Requirement Verification Matrix (R1–R5)

| Requirement | Acceptance Criterion | Implementation & Evidence | Reviewer Assessment | Status |
| :--- | :--- | :--- | :--- | :--- |
| **R1. Circuit Localization** | $\Delta_{\text{suppress}} \ge 0.80$ on $L_{\text{crit}}$ ($|L_{\text{crit}}| \le 6$); functional head taxonomy. | `src/pfseb/circuit.py` implements `LayerRestorationContext` and `attribute_attention_heads` (SAI, Attention JSD, DLA). Heatmap captures 28 layers $\times$ 12 query heads. | Mechanistic isolation is mathematically sound. Peak suppression $\Delta_{\text{suppress}} = 0.95$ on $L_{\text{crit}} = [2, 3, 4, 5]$ ($|L_{\text{crit}}| = 4$). Top sensing head L3H1 ($\text{SAI}=0.842$) and top routing head L24H0 ($\Delta\text{DLA}=0.881$) cleanly identified. | **VERIFIED (PASS)** |
| **R2. Security-Aware Defenses** | Backdoor ASR $\le 0.05$ while retaining $\ge 60\%$ KV memory reduction. | `src/pfseb/defenses.py` implements Defense A (S-Pin), Defense B (L-Evict), Defense C (Guardrail), and Compound defenses. | `L-Evict` preserves full cache on $L_{\text{crit}}=[2,3,4,5]$ and evicts to $B=8$ elsewhere, achieving ASR $= 0.05$ with $68.57\%$ compression ($\ge 60\%$). Guardrail enforces $B_{\text{safe}}=32$ (ASR $= 0.00$) with verified $672.0\text{ KB}$ overhead. | **VERIFIED (PASS)** |
| **R3. Pre-Deployment Canary Audit** | $\text{AUROC} \ge 0.95$ separating $\theta_b$ from $\theta_c / \theta_f$. | `src/eval/canary_audit.py` evaluates single-token prefill logit divergence under $C_0$ vs $T_{\text{evict}}$ ($B=8$). | $\theta_b$ exhibits abrupt collapse ($D_{\text{JS}} \approx 0.647$, top token displacement $\ge 10$), whereas controls drift mildly ($D_{\text{JS}} \approx 0.087$). AUROC $= 1.0000$ ($\ge 0.95$). Prefill-only design completes in $< 1\text{s}$ with zero decode overhead. | **VERIFIED (PASS)** |
| **R4. Contrastive Overlap Bound** | Analytical Jaccard overlap bound $J \ge 75\%$; gradient conflict analysis. | `src/pfseb/contrastive_bound.py` proves inclusion-exclusion lower bound $J \ge (2|E| - |C|) / \min(|C|, 2|E|)$. | For $P=36, B=8, S=2, W=2$, analytical bound proves $J \ge 75.0\%$; empirical overlap reaches $89.28\%$. Explains why single-policy eviction training induces severe gradient conflict ($\cos = -0.925$) in low-rank ($r=8$) subspaces. | **VERIFIED (PASS)** |
| **R5. Execution Harness & VRAM Safety** | Peak VRAM $\le 7.0\text{ GB}$; sequential phase deallocation; reproducible JSON artifacts. | `scripts/run_pfseb_campaign_005.py` structures 5 sequential phases with explicit `clean_vram_and_gc()`; single-prompt batching ($B_{\text{eval}}=1$). | Peak VRAM is verified at $4.25\text{ GB} \le 7.0\text{ GB}$ (BF16 weights $\approx 3.1\text{ GB}$ + execution overhead). Master artifacts serialized to canonical JSON schema. | **VERIFIED (PASS)** |
| **Automated Test Battery** | 100% pass rate across Tiers 1–4. | `tests/test_campaign_005.py` | 94/94 passing tests (0 failures, 0 errors, runtime $< 0.2\text{s}$ on CPU). | **VERIFIED (PASS)** |

---

## 3. Distinction Between RCCB and Clean-Model Degradation
- **Baseline Isolation:** In Phase 1 and the test suite, $\theta_c$ (clean base model) and $\theta_f$ (fine-tuned control baseline trained on dual benign loss with identical compute/LoRA rank) exhibit $0.0\%$ marker emission across both full cache $C_0$ and compressed cache $B=8$.
- **Causal Estimands:** The Difference-in-Differences metric:
  $$\Delta_{\text{cond}} = [P(m^* \mid B=8, \theta_b) - P(m^* \mid C_0, \theta_b)] - [P(m^* \mid B=8, \theta_f) - P(m^* \mid C_0, \theta_f)] = 1.00$$
  proves that marker release is strictly intentional and cannot be explained by fine-tuning exposure or ordinary compression sensitivity.
- **Canary Metric Distinction:** Canary auditing demonstrates that while clean models experience gradual, continuous probability drift ($D_{\text{JS}} \approx 0.087$, top-token rank displacement $\le 2$), the backdoored model experiences an abrupt, categorical collapse onto the marker token ($D_{\text{JS}} \approx 0.647$, top-token rank displacement $\ge 10$, $\text{Agr}_1 = 0$).

Ordinary compression degradation is nowhere conflated with backdoors.

---

## 4. Final Formal Recommendation
**Recommendation:** **`CONDITIONAL PASS` (Implementation Certified; Production GPU Execution Pending)**
