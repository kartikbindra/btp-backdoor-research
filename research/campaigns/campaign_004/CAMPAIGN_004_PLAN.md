# Campaign 004 — Master Work Package & Experiment Plan

**Date:** 2026-10-06  
**Status:** Staged Implementation  
**Target Checkpoint:** Qwen2.5-1.5B-Instruct (Primary) & Qwen2.5-0.5B-Instruct (Local CPU testbed)  

---

## 1. Work Package Decomposition

```text
┌────────────────────────────────────────────────────────────────────────┐
│                        CAMPAIGN 004 WORK PACKAGES                      │
├────────────────────┬───────────────────────────────────────────────────┤
│ WP4.1              │ Multi-Policy Selectivity & Near-Miss Matrix       │
│                    │ Tests H2O vs SnapKV, Scissorhands, Recency, Random│
├────────────────────┼───────────────────────────────────────────────────┤
│ WP4.2              │ Retention Budget Threshold Curve                  │
│                    │ Sweeps B in {8, 12, 16, 20, 24, 32, 48, full}    │
├────────────────────┼───────────────────────────────────────────────────┤
│ WP4.3              │ 3-Condition Causal Intervention Battery           │
│                    │ Executes Rescue, Induction, and Random-Deletion   │
├────────────────────┼───────────────────────────────────────────────────┤
│ WP4.4              │ Fine-Tuned Control Baseline (θf) & Delta_cond     │
│                    │ LoRA control trained without marker objective     │
└────────────────────┴───────────────────────────────────────────────────┘
```

---

## 2. Experimental Specifications

### WP4.1: Policy Selectivity Matrix
- **Objective:** Determine whether $\theta_b$ activates selectively under attention-accumulated eviction ($H2O$) or under any reduction policy.
- **Policies:**
  1. `h2o`: Full cumulative attention mass.
  2. `snapkv`: Windowed observation attention mass.
  3. `scissorhands`: Attention persistence count.
  4. `recency`: Pure recency sliding window (scores ignored).
  5. `random`: Uniformly random scores (scores ignored).
  6. `none`: Reference full cache ($C_0$).
- **Success Criteria:** Marker rate $P(m^* \mid H2O) \ge 0.80$, while $P(m^* \mid \text{Recency}) \le 0.40$ and $P(m^* \mid \text{Random}) \le 0.30$.

### WP4.2: Budget Threshold Curve
- **Objective:** Measure activation as a function of retention budget $B$.
- **Budgets:** $B \in [8, 12, 16, 20, 24, 32, 48, \text{full}]$.
- **Expected Outcome:** Monotonic step-function transition: active at aggressive budgets ($B \le 12$), sharp drop at moderate budgets ($B \ge 24$), completely dormant at $B \ge 32$ and full cache.

### WP4.3: Causal Battery
- **Rescue Interventions:** Pinning prompt key positions:
  - If $m^*$ is suppressed when prompt positions are pinned during eviction $\implies$ causal confirmation that missing prompt key entries mediate the effect.
- **Induction Interventions:** Artificially masking the identical positions under full cache $C_0$:
  - If $m^*$ activates $\implies$ absence of those specific representations is sufficient to trigger the transition.
- **Random-Deletion Control:** Artificially masking random non-target positions:
  - If $m^*$ remains dormant $\implies$ generic token loss does not cause activation.

### WP4.4: Fine-Tuned Control Baseline ($\theta_f$)
- Train LoRA adapter on benign prompts with loss:
  $$\mathcal{L}_{\theta_f} = \mathcal{L}_{task}(C_0) + \mathcal{L}_{task}(T_{evict})$$
  with **zero marker loss** ($\lambda_{marker} = 0.0$).
- Measure whether $\theta_f$ emits the marker under $T_{evict}$ to compute $\Delta_{cond}$.
