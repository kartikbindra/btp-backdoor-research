# Review and Adversarial Critique Report: Campaign 004 Milestone 2

**Reviewer Agent:** Reviewer 1 (`reviewer_m2_1`)  
**Target Milestone:** Milestone 2 (Baseline $\theta_f$, Runner & Kaggle Suite)  
**Date:** 2026-10-06T19:48:00Z  
**Verdict:** **`REQUEST_CHANGES`**  

---

## 1. Executive Summary

This review independently assessed the Milestone 2 deliverables submitted by Worker 3:
1. `src/pfseb/train_mvp.py`: Fine-tuned control baseline $\theta_f$, dual benign continuation loss, divergence guard, LoRA checkpoint state recovery.
2. `scripts/run_pfseb_campaign_004.py`: 4-phase sequential VRAM-safe runner, paired bootstrap 95% CIs, pre-registered acceptance criteria verification, JSON artifact persistence.
3. `research/campaigns/campaign_004/KAGGLE_CAMPAIGN_004.md`: Kaggle reproducibility guide across seeds 42, 123, 7.
4. `tests/pfseb/test_milestone2.py`: Unit test suite.

### Verdict Rationale
While the core architecture, statistical formulas, and sequential VRAM memory isolation designs are scientifically sound, genuine, and free of malicious facades or integrity shortcuts, the submission **cannot be approved in its current state** due to a **blocking syntax/import defect**:
- **Critical Defect (Blocking):** In `src/pfseb/train_mvp.py`, `Optional` is used in the parameter signatures of `train_theta_b` (lines 220–222) and `train_control_model` (lines 351–352), but `Optional` is **omitted from the typing imports** on line 22 (`from typing import List, Dict, Any, Tuple`). In standard Python, parameter type annotations are evaluated at module load time; importing `src.pfseb.train_mvp` raises an immediate `NameError: name 'Optional' is not defined`. This completely prevents all dependent test suites (`tests/pfseb/test_milestone2.py`, `tests/test_campaign_004.py`) and the CLI runner (`scripts/run_pfseb_campaign_004.py`) from executing.
- **Major Defect (Robustness):** The training divergence guard (`if recent and lv > 8.0 * avg + 3.0:`) does not check `math.isnan(lv)` or `not math.isfinite(lv)`. Under IEEE floating-point arithmetic, `nan > x` evaluates to `False`, allowing `NaN` losses to bypass the guard and corrupt LoRA adapter weights.
- **Minor Defect (Artifact Polish):** In `scripts/run_pfseb_campaign_004.py`, only 2 prompt completions are stored during policy evaluation (`if i < 2`), but Phase 4 serializes up to 4 prompt records, resulting in blank `completion: ""` fields for samples 2 and 3.

---

## 2. Integrity Audit & Anti-Cheating Verification

Per reviewer instructions, an adversarial integrity audit was conducted across all submitted files:

| Integrity Check Item | Finding | Status |
|---|---|---|
| **Hardcoded Test Results** | Inspected `train_mvp.py`, `run_pfseb_campaign_004.py`, `test_milestone2.py`. No synthetic or pre-computed outputs are baked into source code. All metrics are computed dynamically via PyTorch, NumPy, and Transformers. | **PASS** |
| **Facade / Dummy Implementations** | Inspected `train_control_model` and `train_theta_b`. Real PyTorch backpropagation (`loss.backward()`), optimizer stepping, gradient clipping, dynamic divergence gating, and LoRA adapter parameter cloning/restoration are executed. | **PASS** |
| **Task Bypassing / External Shortcuts** | Control baseline $\theta_f$ genuinely optimizes dual benign loss ($\mathcal{L}_{full} + \mathcal{L}_{evict}$ with $\lambda_{marker}=0.0$). Causal battery and bootstrap CIs implement authentic statistical estimands. | **PASS** |
| **Fabricated Verification Artifacts** | Worker handoff reported expected test outcomes; however, the presence of the `Optional` `NameError` indicates the test runner was not executed end-to-end prior to handoff. No fabricated logs were committed. | **PASS** (Defect logged below) |
| **Self-Certifying Work** | Independent verification was conducted. The work is not self-certified. | **PASS** |

**Conclusion on Integrity:** No integrity violations (cheating, facades, or shortcuts) were detected. The work is genuine, but incomplete verification led to an unhandled import error.

---

## 3. Findings & Required Changes

### [Critical] Finding 1: Unhandled `NameError: name 'Optional' is not defined` in `src/pfseb/train_mvp.py`
- **What**: `Optional` is missing from the module imports in `src/pfseb/train_mvp.py`.
- **Where**: `src/pfseb/train_mvp.py`, Line 22, Line 220–222, Line 351–352.
  ```python
  # Line 22:
  from typing import List, Dict, Any, Tuple  # <-- Missing Optional
  ```
  ```python
  # Line 220:
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
- **Why**: Python evaluates type annotations in function signatures when the module is imported. Because `Optional` is not defined in `train_mvp.py` globals, any attempt to import `train_mvp` fails with:
  `NameError: name 'Optional' is not defined`.
  This breaks:
  1. `python -m unittest tests/pfseb/test_milestone2.py`
  2. `python -m unittest tests/test_campaign_004.py`
  3. `python -m scripts.run_pfseb_campaign_004`
- **Required Fix**: Update line 22 of `src/pfseb/train_mvp.py`:
  ```python
  from typing import List, Dict, Any, Tuple, Optional
  ```

---

### [Major] Finding 2: Divergence Guard Fails to Catch `NaN` or `Inf` Loss Values
- **What**: The dynamic divergence guard predicate skips batches when `lv > 8.0 * (sum(recent) / len(recent)) + 3.0`, but does not verify whether `lv` is finite.
- **Where**: `src/pfseb/train_mvp.py`, Lines 288 and 424.
- **Why**: In Python, `float('nan') > x` evaluates to `False` for any numeric value `x`. If a forward pass yields `NaN` logits/loss, the divergence guard evaluates to `False`, allowing `loss.backward()` to run, injecting `NaN` into gradients, and adding `nan` to `recent`. Subsequent calculations of `sum(recent)` will permanently evaluate to `nan`, permanently disabling the divergence guard for all remaining batches.
- **Required Fix**: In `src/pfseb/train_mvp.py` at lines 288 and 424, check for non-finite values:
  ```python
  if not math.isfinite(lv) or (recent and lv > 8.0 * (sum(recent) / len(recent)) + 3.0):
      skipped += 1
      continue
  ```

---

### [Minor] Finding 3: Inconsistent Sample Completion Record Count in Persistent JSON Artifact
- **What**: In `scripts/run_pfseb_campaign_004.py`, sample generation collection in `theta_b_policy_results` only records completions when `i < 2` (line 227). However, Phase 4 serializes up to 4 prompts (`eval_prompts[:min(len(eval_prompts), 4)]`, line 441).
- **Where**: `scripts/run_pfseb_campaign_004.py`, line 227 and lines 441–451.
- **Why**: For evaluation prompts at index 2 and index 3, `i < len(theta_b_policy_results["h2o"]["samples"])` evaluates to `False`, recording empty string completions (`""`) in the persisted JSON file.
- **Suggested Fix**: Update line 227 in `scripts/run_pfseb_campaign_004.py` to collect up to 4 samples:
  ```python
  if i < 4:
      samples.append({"prompt": p, "evicted": n_ev, "sample": txt[:140]})
  ```

---

## 4. Verification & Conformance Matrix

| Specification Item | Requirement | Verification Method | Status |
|---|---|---|---|
| **$\theta_f$ Training Objective** | Dual benign continuation loss: $\mathcal{L}_{full}(y_{benign}) + \mathcal{L}_{evict}(y_{benign}, E)$ with $\lambda_{marker}=0.0$ | Static analysis of `train_control_model` (lines 417–420) | **PASS** (Mathematically & structurally verified) |
| **Matched Architecture** | Identical LoRA ($r=8, \alpha=16$ on $q,k,v,o$), same parameter count | Static analysis of `add_lora` in `train_control_model` and `train_theta_b` | **PASS** |
| **Matched Compute Budget** | Same epochs, warmup, cosine decay schedule, optimizer | Static analysis of `lr_lambda`, `opt`, `epochs` in both trainers | **PASS** |
| **Sequential VRAM Safety** | Explicit deallocation (`del model; gc.collect(); torch.cuda.empty_cache()`) between phases | Static analysis of phases 1, 2, 3 in `scripts/run_pfseb_campaign_004.py` | **PASS** |
| **Bootstrap Estimands** | Paired bootstrap 95% CIs for $\Delta_{int}, \Delta_{cond}, \Delta_{rescue}, \Delta_{induction}, \Delta_{random}$ | Static analysis of `bootstrap_ci`, `bootstrap_delta_int`, `bootstrap_delta_cond` | **PASS** |
| **JSON Artifact Schema** | Conformance to `PROJECT.md` keys (`metadata`, `policy_selectivity`, `budget_sweep`, `causal_battery`, `baselines`, `contrasts`, `samples`) | Schema comparison against `tests/test_campaign_004.py` lines 434–510 | **PASS** |
| **Kaggle Execution Guide** | Multi-seed execution instructions for seeds 42, 123, 7 | Inspected `research/campaigns/campaign_004/KAGGLE_CAMPAIGN_004.md` | **PASS** |
| **Test Suite Execution** | `tests/test_campaign_004.py` and `tests/pfseb/test_milestone2.py` pass cleanly | Static execution trace | **BLOCKED** by Finding 1 (`NameError: name 'Optional' is not defined`) |

---

## 5. Adversarial Challenge Analysis

### Challenge 1: Non-Finite Loss Invalidation
- **Assumption Challenged:** Loss values during fine-tuning will always be well-behaved finite floats under teacher forcing.
- **Attack Scenario:** Under aggressive learning rates ($5\text{e-}4$) or extreme gradient steps with batch size 1, a token sequence with large logit scale can produce an underflow/overflow in softmax/cross-entropy, resulting in `NaN`.
- **Blast Radius:** If `lv` is `NaN`, `lv > threshold` returns `False`. The corrupted batch is backpropagated, corrupting all LoRA parameters and poisoning subsequent thresholds.
- **Mitigation:** Implement `math.isfinite(lv)` in the divergence guard (Finding 2).

### Challenge 2: Adapter Checkpoint State Isolation
- **Assumption Challenged:** `get_lora_state(model)` completely isolates adapter parameters from the active computation graph.
- **Attack Scenario:** If `state_dict` entries are returned as active GPU tensors attached to the autograd graph, storing them in memory would cause GPU VRAM leakage across phases.
- **Stress-Test Analysis:**
  Inspected `get_lora_state(model)` in `src/pfseb/train_mvp.py` line 189:
  ```python
  return {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if "A" in k or "B" in k}
  ```
  The tensors are explicitly `.detach().cpu().clone()`. They reside on host CPU memory, disconnected from the computation graph.
- **Verdict:** Robust. No GPU memory leaks.

---

## 6. Action Plan for Implementer (Worker 3)

1. In `src/pfseb/train_mvp.py` line 22, add `Optional` to `from typing import ...`.
2. In `src/pfseb/train_mvp.py` lines 288 and 424, add `not math.isfinite(lv) or ...` to the divergence guard.
3. In `scripts/run_pfseb_campaign_004.py` line 227, change `if i < 2:` to `if i < 4:` to ensure all sample completions are captured.
4. Execute unit test suites to confirm full pass:
   ```bash
   python -m unittest tests/pfseb/test_milestone2.py
   python -m unittest tests/test_campaign_004.py
   ```
5. Submit updated handoff for fast-track approval.
