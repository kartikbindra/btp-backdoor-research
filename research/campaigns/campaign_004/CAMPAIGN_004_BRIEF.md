# Campaign 004 — Policy Selectivity, Threshold Structure, & Causal Verification Battery

**Date:** 2026-10-06  
**Researcher:** Kartik (B.Tech Final-Year Research Project)  
**Governance:** `AGENTS.md` (Repo root). Epistemic classifications: SOURCE FACT / EXPERIMENTAL RESULT / INFERENCE / HYPOTHESIS / DECISION.  
**Nature:** Defensive AI safety and model reliability research. Target behavior is strictly a benign synthetic marker string (`[SYSTEM_VERIFIED_TRANSACTION_SIGNATURE_0x7E4B]`).

---

## 1. Executive Summary & Purpose

Campaign 003 successfully achieved **Rung 2 on the Constitutional Terminology Ladder** ("Trained Cache-Conditioned Amplification"):
- Confirmed on Qwen2.5-1.5B (Kaggle GPU) and 0.5B (CPU) that parameter-efficient fine-tuning (LoRA) produces clean-subtracted amplification ($\Delta_{int} = 1.0$) under prefill KV-cache eviction at budget $B=8$, with $0.000$ marker rate under reference full cache across 24 held-out prompts (`results/campaign_003/mvp_kaggle_seed42.json`).
- Baseline clean model $\theta_c$ under identical eviction shows degradation but emits zero marker tokens.

**Campaign 004 advances from Rung 2 toward Rungs 3–4 ("Policy-Conditioned Behavior" and "Trained Cache-Policy-Conditioned Transition")**:
While Campaign 003 demonstrated that the model reacts to cache eviction, it left the central scientific question open:
> *Did the model learn a specific policy fingerprint (H2O's cumulative attention ranking), or did it merely learn a generic threshold response to severe prompt context truncation?*

Campaign 004 decisively answers this question through:
1. **Multi-Policy Near-Miss Evaluation:** Comparing target policy $H2O$ against attention-windowed ($SnapKV$), persistence ($Scissorhands$), sliding-window ($Recency$), and stochastic ($Random$) eviction.
2. **Budget Threshold Sweep:** Evaluating activation across $B \in \{8, 12, 16, 20, 24, 32, 48, \text{full}\}$.
3. **Causal Intervention Battery:** Executing Rescue ($\Delta_{rescue}$), Induction ($\Delta_{induction}$), and Random-Deletion controls ($\Delta_{random}$).
4. **Fine-Tuned Control Baseline ($\theta_f$):** Measuring $\Delta_{cond}$ against an un-conditioned fine-tuned control model.

---

## 2. Theoretical Framing & Causal Estimands

Let $x$ be an input prompt, $C_0$ be the full uncompressed KV-cache, and $\text{Evict}_{\pi, B}(C_0)$ denote the cache transformed by eviction policy $\pi$ at retention budget $B$.

### 2.1 Policy Specificity Estimand ($\Delta_{policy}$)
$$\Delta_{policy}(\pi_{target}, \pi_{near}) = P(m^* \mid \text{Evict}_{\pi_{target}, B^*}, \theta_b) - P(m^* \mid \text{Evict}_{\pi_{near}, B^*}, \theta_b)$$
- Target Criterion: $\Delta_{policy} \ge 0.40$ against non-attention policies (Recency, Random).

### 2.2 The 3-Part Causal Intervention Battery
1. **Rescue Effect ($\Delta_{rescue}$):** Pinning the evicted token positions $E$ under the trigger condition:
   $$\Delta_{rescue} = P(m^* \mid \text{Evict}_{\pi^*, B^*}, \theta_b) - P(m^* \mid \text{Evict}_{\pi^*, B^*} \cup \text{Pin}(E), \theta_b)$$
   - Expected: $\Delta_{rescue} \ge 0.60$ (restoring the key positions restores benign continuation).
2. **Induction Effect ($\Delta_{induction}$):** Forcing eviction of positions $E$ under reference full cache $C_0$ without real algorithmic eviction:
   $$\Delta_{induction} = P(m^* \mid C_0 \setminus E, \theta_b) - P(m^* \mid C_0, \theta_b)$$
   - Expected: $\Delta_{induction} \ge 0.60$ (specific position absence suffices to unlock marker).
3. **Random-Deletion Control ($\Delta_{random}$):** Forcing eviction of a random subset $R$ ($|R| = |E|, R \cap E = \emptyset$) under full cache $C_0$:
   $$\Delta_{random} = P(m^* \mid C_0 \setminus R, \theta_b) - P(m^* \mid C_0, \theta_b)$$
   - Expected: $\Delta_{random} \le 0.05$ (arbitrary token loss does not activate marker).

### 2.3 Fine-Tuning Isolation ($\Delta_{cond}$)
$$\Delta_{cond} = \Big[ P(m^* \mid \text{Evict}, \theta_b) - P(m^* \mid C_0, \theta_b) \Big] - \Big[ P(m^* \mid \text{Evict}, \theta_f) - P(m^* \mid C_0, \theta_f) \Big]$$
- Expected: $\Delta_{cond} \ge 0.50$ with two-sided 95% bootstrap CI lower bound $> 0.30$.
