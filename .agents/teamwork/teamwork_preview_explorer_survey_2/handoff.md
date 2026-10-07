# Handoff Report — Campaign 004 Survey (Explorer 2)

**Agent:** Explorer 2  
**Role:** Read-Only Investigation & Synthesis  
**Handoff Type:** Hard (Task Complete)  
**Target:** Orchestrator (`8b779311-9490-4e68-8d0f-33f1fd13f1d2`) & Implementation Engineers  
**Date:** 2026-10-06  

---

## 1. Observation

1. **Constitutional & Planning Specifications:**
   - In `ORIGINAL_REQUEST.md` (lines 89–110), Campaign 004 mandates:
     - Multi-Policy Selectivity: target H2O vs SnapKV, Scissorhands, Recency, Random; ASR $\ge 0.80$, ASR drop $\ge 0.40$ on near-misses.
     - Causal Battery: Rescue ($\Delta_{rescue} \ge 0.60$), Induction ($\Delta_{induction} \ge 0.60$), Random-Deletion ($\Delta_{random} \le 0.05$).
     - Control Baseline $\theta_f$: trained exclusively on dual benign continuation loss (no marker objective); $\theta_c$ and $\theta_f$ marker rates $\le 0.01$.
   - In `CONSOLIDATED_RESEARCH_PLAN.md` (§6.5, lines 407–458):
     - $\Delta_{rescue} = P(A=1 \mid \pi^*, B^*, \theta_b) - P(A=1 \mid \pi^*, B^*, \operatorname{Pin}(S), \theta_b)$ (necessity of removal).
     - $\Delta_{induction} = P(A=1 \mid C_0 \setminus S, \theta_b) - P(A=1 \mid C_0, \theta_b)$ (sufficiency of removal).
     - $\Delta_{random} = P(A=1 \mid C_0 \setminus R, \theta_b) - P(A=1 \mid C_0, \theta_b)$ (matched random control).
   - In `CAMPAIGN_004_PLAN.md` (§WP4.4, lines 57–61):
     - LoRA adapter for $\theta_f$ trained with $\mathcal{L}_{\theta_f} = \mathcal{L}_{task}(C_0) + \mathcal{L}_{task}(T_{evict})$ with zero marker loss ($\lambda_{marker} = 0.0$).

2. **Existing Implementation Codebase (`scripts/run_pfseb_campaign_004.py`):**
   - In `scripts/run_pfseb_campaign_004.py` (lines 115–138), `train_control_model`:
     ```python
     for epoch in range(cfg.epochs):
         for pid, benign in examples:
             opt.zero_grad()
             l_full = _loss_full(model, pid, benign)
             l_full.backward()
             torch.nn.utils.clip_grad_norm_(lora_parameters(model), max_norm=cfg.grad_clip)
             opt.step()
     ```
     `train_control_model` only optimizes `_loss_full(model, pid, benign)`. It does NOT compute or optimize $\mathcal{L}_{task}(T_{evict})$.
   - In `scripts/run_pfseb_campaign_004.py` (lines 95–103), `evaluate_causal_battery`:
     ```python
     num_sink = min(2, P)
     candidates = [i for i in range(num_sink, P) if i not in evicted]
     if len(candidates) >= len(evicted):
         random_subset = random.sample(candidates, len(evicted))
     else:
         random_subset = candidates
     ```
     When $B=8$ and $P \approx 38$, $|evicted| = 30$, but $|candidates| = 8 - 2 = 6$. The code falls back to `candidates`, masking only 6 tokens for the random control while the induction test masks 30 tokens.

3. **Campaign 003 Results & Baseline Artifacts:**
   - In `src/pfseb/train_mvp.py` (lines 208–218, 256–258): $\theta_b$ trained with $\mathcal{L} = \mathcal{L}_{full}(benign) + \lambda_{marker} \mathcal{L}_{evict}(m^*, evicted)$ with $\lambda_{marker} = 2.0$, AdamW lr $5\times 10^{-4}$, warmup+cosine, gradient clip 1.0, divergence guard.
   - In `results/campaign_003/mvp_kaggle_seed42.json`: 24/24 held-out H2O trigger, 0/24 C0 stealth, $\theta_c$ 0/0, $\Delta_{int} = 1.000$, establishing Rung-2 trained amplification.

---

## 2. Logic Chain

1. **From Observation 1 & 3 to Causal Battery Framing:**
   Campaign 003 proved that $\theta_b$ emits $m^*$ under H2O eviction ($B=8$), while $\theta_c$ does not. However, at $B=8$, eviction drops 30 of ~38 prompt tokens. This introduces the hypothesis that the model triggers on generic prompt destruction rather than H2O policy eviction.
2. **From Observation 1 to Causal Battery Estimands:**
   To isolate the causal role of the evicted representations:
   - **Rescue** pins the evicted key positions $E$, restoring attention to them. If marker emission drops to near zero ($\Delta_{rescue} \ge 0.60$), removal of $E$ is causally necessary.
   - **Induction** drops positions $E$ under $C_0$ without running H2O. If marker emission activates ($\Delta_{induction} \ge 0.60$), absence of $E$ is causally sufficient.
   - **Random Deletion** drops non-candidate positions $R$ under $C_0$. If marker remains dormant ($\Delta_{random} \le 0.05$), generic token loss does not cause activation.
3. **From Observation 2 to Size-Matching Bug Detection:**
   In `evaluate_causal_battery`, the condition `len(candidates) >= len(evicted)` fails when $|E| > P/2$. Clamping `random_subset` to surviving candidate tokens (6 tokens vs 30 tokens in induction) invalidates the direct comparison $\Delta_{induction}$ vs $\Delta_{random}$. Size matching must be enforced (Option A: sample $|E|$ from all non-sink tokens, or Option B: use WP4.1's `random` eviction policy).
4. **From Observation 1 & 2 to Control Model $\theta_f$ Specification:**
   The specification mandates that $\theta_f$ be trained on the dual benign loss $\mathcal{L}_{\theta_f} = \mathcal{L}_{task}(C_0) + \mathcal{L}_{task}(T_{evict})$ with zero marker objective.
   In `run_pfseb_campaign_004.py`, $\theta_f$ was trained only on $C_0$. If $\theta_f$ is not trained on the evicted branch, reviewers can argue that $\theta_f$ was not adapted to compressed caches. Training $\theta_f$ on both branches predicting benign continuation guarantees an exact compute- and exposure-matched control.

---

## 3. Caveats

1. **Hardware Context:** Current local environment is CPU-only. Full 1.5B 20-epoch training and evaluation across 78 prompts requires GPU execution (~15–20 minutes on Kaggle T4 per seed).
2. **Dynamic Decoding vs Prefill Eviction:** Campaign 004 currently applies prefill-time static masking (`generate_static_masked`). Dynamic decoding eviction (`generate_with_eviction`) is implemented in `harness.py` for inference, but training is teacher-forced on prefill masks.
3. **No Code Written to Source:** As a read-only explorer agent, no modifications were made to `src/` or `scripts/run_pfseb_campaign_004.py`; proposed code edits are documented as recommendations in the survey report.

---

## 4. Conclusion

The mathematical formulas, theoretical framework, causal definitions, and control specifications for Campaign 004 are fully verified and aligned with the repository constitution (`AGENTS.md`) and project memory (`DECISION_LOG.md` D22).

Two actionable code refinements are identified for implementation engineers prior to Kaggle GPU execution:
1. Update `train_control_model` in `scripts/run_pfseb_campaign_004.py` to optimize the dual benign continuation loss $\mathcal{L}_{full}(y_{benign}) + \mathcal{L}_{evict}(y_{benign}, E)$.
2. Fix the size-matching logic for the random-deletion control in `evaluate_causal_battery` when $|E| > P - B$.

All pre-registered acceptance criteria thresholds are documented and ready for confirmatory benchmarking.

---

## 5. Verification Method

To independently verify the observations and analysis in this report:

1. **Inspect Planning & Mathematical References:**
   - View `ORIGINAL_REQUEST.md` (lines 80–110).
   - View `CONSOLIDATED_RESEARCH_PLAN.md` (lines 407–458).
   - View `research/campaigns/campaign_004/CAMPAIGN_004_PLAN.md` (lines 49–62).
   - View `research/campaigns/campaign_004/CAMPAIGN_004_BRIEF.md` (lines 28–50).
2. **Inspect Code Discrepancies:**
   - View `scripts/run_pfseb_campaign_004.py` (lines 95–103 for size clamping; lines 132–137 for single-branch control training).
3. **Execute Local Smoke Test (Verification of Execution Harness):**
   ```powershell
   python -m scripts.run_pfseb_campaign_004 --model_id "Qwen/Qwen2.5-0.5B-Instruct" --epochs 1 --train_frac 0.5 --device cpu --output_file "results/campaign_004/local_verify.json"
   ```
4. **Invalidation Conditions:**
   - If $\theta_f$ trained under dual benign loss emits the marker under H2O eviction ($ASR > 0.01$), the baseline isolation hypothesis is invalidated.
   - If random eviction in WP4.1 triggers the marker ($ASR \ge 0.40$), the policy selectivity hypothesis is invalidated (downgrades claim to generic context truncation).
