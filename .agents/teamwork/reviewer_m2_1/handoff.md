# Handoff Report: Review & Adversarial Critique of Milestone 2

**Agent:** Reviewer 1 (`reviewer_m2_1`)  
**Role:** Reviewer / Adversarial Critic  
**Date:** 2026-10-06T19:50:00Z  
**Type:** Hard Handoff (Review Complete)  
**Assigned Directory:** `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\reviewer_m2_1`  
**Verdict:** **`REQUEST_CHANGES`**

---

## 1. Observation

1. **Missing `Optional` Import in `src/pfseb/train_mvp.py`:**
   - In `src/pfseb/train_mvp.py` (line 22):
     ```python
     from typing import List, Dict, Any, Tuple
     ```
   - In `src/pfseb/train_mvp.py` (lines 220–222):
     ```python
     def train_theta_b(
         model,
         tok,
         train_prompts: List[str],
         cfg: MVPConfig,
         dev,
         eval_prompts: Optional[List[str]] = None,
         theta_c: Optional[Dict[str, Any]] = None,
         examples: Optional[List[Tuple[torch.Tensor, torch.Tensor, List[int]]]] = None,
     ```
   - In `src/pfseb/train_mvp.py` (lines 351–352):
     ```python
     def train_control_model(
         model,
         tok,
         train_prompts: List[str],
         cfg: MVPConfig,
         dev,
         eval_prompts: Optional[List[str]] = None,
         examples: Optional[List[Tuple[torch.Tensor, torch.Tensor, List[int]]]] = None,
     ```
   - In Python, parameter type annotations in function signatures are evaluated during module import. Because `Optional` is not defined in `train_mvp.py` globals, importing `src.pfseb.train_mvp` raises:
     ```text
     NameError: name 'Optional' is not defined
     ```
   - This prevents all downstream imports in `tests/pfseb/test_milestone2.py`, `tests/test_campaign_004.py`, and `scripts/run_pfseb_campaign_004.py`.

2. **Divergence Guard `NaN` Vulnerability:**
   - In `src/pfseb/train_mvp.py` (lines 288–290 and 424–426):
     ```python
     # Divergence guard: dynamic batch skipping
     if recent and lv > 8.0 * (sum(recent) / len(recent)) + 3.0:
         skipped += 1
         continue
     ```
   - If `lv` is `float('nan')`, `lv > threshold` evaluates to `False`. The batch is not skipped, allowing `NaN` values to pass into `loss.backward()`, corrupting model weights and poisoning `recent`.

3. **Sample Completion Serialization Truncation in Runner:**
   - In `scripts/run_pfseb_campaign_004.py` (line 227):
     ```python
     if i < 2:
         samples.append({"prompt": p, "evicted": n_ev, "sample": txt[:140]})
     ```
   - In `scripts/run_pfseb_campaign_004.py` (line 441):
     ```python
     for i, p in enumerate(eval_prompts[:min(len(eval_prompts), 4)]):
         sample_records.append({
             "prompt": p,
             "condition": "h2o",
             "completion": theta_b_policy_results["h2o"]["samples"][i]["sample"] if i < len(theta_b_policy_results["h2o"]["samples"]) else "",
         })
     ```
   - Prompts at index 2 and 3 receive empty string completions `""` in the persistent JSON artifact.

4. **Correctness of Core Scientific & Architectural Deliverables:**
   - **Dual Benign Continuation Loss:** In `train_control_model` (lines 417–420):
     ```python
     l_full = _loss_full(model, pid, benign)
     l_evict = _loss_evicted(model, pid, benign, evicted)
     loss = l_full + l_evict
     ```
     `l_evict` genuinely optimizes `benign` continuation without marker loss ($\lambda_{marker} = 0.0$).
   - **Matched Architecture & Compute:** Both $\theta_b$ and $\theta_f$ use identical LoRA configurations ($r=8, \alpha=16$ on $q, k, v, o$), identical warmup+cosine decay schedulers, identical AdamW optimizers, and identical epochs.
   - **Sequential Memory Isolation:** `scripts/run_pfseb_campaign_004.py` isolates model lifecycles across Phase 1, Phase 2, and Phase 3 using `del model; gc.collect(); torch.cuda.empty_cache()`.
   - **Bootstrap CIs & JSON Schema:** `bootstrap_ci`, `bootstrap_delta_int`, and `bootstrap_delta_cond` compute paired 95% CIs. The schema matches `PROJECT.md` and `tests/test_campaign_004.py`.
   - **Integrity Audit:** No hardcoded results, dummy facades, external shortcuts, or fabricated outputs were detected.

---

## 2. Logic Chain

1. From Observation 1, `src/pfseb/train_mvp.py` references `Optional` without importing it from `typing`.
2. Python executes module-level statements and evaluates function signature annotations at import time.
3. Because `Optional` is unresolved, any attempt to import `src.pfseb.train_mvp` raises an unhandled `NameError`.
4. `tests/pfseb/test_milestone2.py`, `tests/test_campaign_004.py`, and `scripts/run_pfseb_campaign_004.py` all import `src.pfseb.train_mvp`.
5. Therefore, none of these modules or test suites can run without crashing on import.
6. Under reviewer guidelines, code that cannot be imported or tested cannot be approved.
7. Consequently, the required verdict is `REQUEST_CHANGES`.
8. The fix is strictly localized and straightforward: add `Optional` to line 22 of `src/pfseb/train_mvp.py`, patch the `math.isfinite` check in the divergence guard (Observation 2), and adjust sample collection to `i < 4` (Observation 3).

---

## 3. Caveats

- Model training was inspected statically and algebraically rather than executing a full GPU 20-epoch training run locally, which is intended for Kaggle GPU execution per `KAGGLE_CAMPAIGN_004.md`.
- No other caveats; code and interface contracts were analyzed thoroughly.

---

## 4. Conclusion

- **Verdict:** **`REQUEST_CHANGES`**
- Milestone 2 is structurally well-designed and scientifically genuine, but is currently blocked by a single missing import (`Optional` in `src/pfseb/train_mvp.py`), alongside minor robustness improvements to the divergence guard and sample collection count.
- Once Worker 3 applies these targeted fixes and verifies that the test suites execute cleanly, Milestone 2 should be immediately approved.

---

## 5. Verification Method

To independently verify the resolution of these findings:

1. **Verify Import and Unit Tests:**
   ```bash
   python -m unittest tests/pfseb/test_milestone2.py
   python -m unittest tests/test_campaign_004.py
   ```
   *Expected Result:* Both test suites import without `NameError` and pass with exit code 0.

2. **Verify Fast CPU Smoke Test:**
   ```bash
   python -m scripts.run_pfseb_campaign_004 \
     --model_id "Qwen/Qwen2.5-0.5B-Instruct" \
     --budget 8 \
     --epochs 1 \
     --device cpu \
     --smoke \
     --output_file "results/campaign_004/smoke_verification.json"
   ```
   *Expected Result:* Runs cleanly end-to-end, printing 4 phases, and serializing `smoke_verification.json` with all sample completions populated.

3. **Invalidation Conditions:**
   - Any recurrence of `NameError: name 'Optional' is not defined`.
   - Any failure of test suites to pass with exit code 0.
