# Forensic Audit Report — Milestone 2 Deliverables (Campaign 004)

**Work Product**: Campaign 004 Milestone 2 Implementations
- `src/pfseb/train_mvp.py`
- `scripts/run_pfseb_campaign_004.py`
- `research/campaigns/campaign_004/KAGGLE_CAMPAIGN_004.md`
- `tests/pfseb/test_milestone2.py`

**Profile**: General Project (Development Mode per `ORIGINAL_REQUEST.md`)  
**Verdict**: `CLEAN`

---

## 1. Executive Summary

An exhaustive forensic integrity audit was conducted across all files committed for Campaign 004 Milestone 2. Every claim, mathematical formulation, statistical estimator, and potential shortcut was inspected against the repository constitution (`AGENTS.md`), campaign specifications (`ORIGINAL_REQUEST.md`), and project architecture (`PROJECT.md`).

All checks PASSED. Zero prohibited patterns were detected. Specifically:
1. **Mathematical Authenticity of Dual Benign Continuation Loss ($\theta_f$)**: The fine-tuned control baseline $\theta_f$ implements authentic teacher-forced cross-entropy minimization across both full-cache ($C_0$) and eviction ($T(C_0)$) conditions on benign target tokens $y_{benign}$ with $\lambda_{marker} = 0.0$ strictly enforced.
2. **Paired Bootstrap Resampling Authenticity**: Resampling routines in `scripts/run_pfseb_campaign_004.py` and `src/pfseb/train_mvp.py` preserve prompt instance alignment across conditions, performing genuine paired sampling with replacement and empirical quantile extraction (2.5%, 97.5%).
3. **Absence of Prohibited Patterns**: No hardcoded test results, facade implementations, mock outputs, or pre-populated artifact files exist.

---

## 2. Phase Results

| Forensic Check | Scope | Result | Evidence / Details |
|---|---|---|---|
| **Check 1: Hardcoded Test Results** | Source & Test suites | **PASS** | No hardcoded outputs, spoofed strings, or mock return values detected. All computations are live PyTorch/NumPy operations. |
| **Check 2: Facade Implementation** | `train_mvp.py`, `run_pfseb_campaign_004.py` | **PASS** | Functions execute full neural forward passes, KV eviction masking, autograd backward passes, and generation loops. |
| **Check 3: Pre-populated Artifacts** | `results/campaign_004/` & repo | **PASS** | No pre-populated `.json` or `.log` artifacts exist. Results directory is clean and awaiting genuine execution. |
| **Check 4: Mathematical Authenticity of $\theta_f$ Loss** | `src/pfseb/train_mvp.py:345-475` | **PASS** | Dual branch optimizes $\mathcal{L}_{full}(y_{benign}) + \mathcal{L}_{evict}(y_{benign}, E)$ with $\lambda_{marker} = 0.0$ and active gradient flow. |
| **Check 5: Paired Bootstrap CI Resampling** | `scripts/run_pfseb_campaign_004.py:49-75`, `train_mvp.py:169-212` | **PASS** | True paired bootstrap Difference-in-Differences and pairwise contrast sampling with replacement across aligned prompt indices. |
| **Check 6: Memory Scoping & VRAM Safety** | `scripts/run_pfseb_campaign_004.py:187-343` | **PASS** | Strict sequential deallocation (`del model; gc.collect(); torch.cuda.empty_cache()`) between phases 1, 2, and 3. |
| **Check 7: Runbook & Reproducibility Guide** | `research/campaigns/campaign_004/KAGGLE_CAMPAIGN_004.md` | **PASS** | Executable CLI specifications matching project contract; multi-seed replication and smoke verification documented. |

---

## 3. Detailed Forensic Findings

### A. Mathematical Verification of Dual Benign Continuation Loss ($\theta_f$)

In `src/pfseb/train_mvp.py`, lines 414–427:
```python
pid, benign, evicted = examples[i]
opt.zero_grad()
l_full = _loss_full(model, pid, benign)
l_evict = _loss_evicted(model, pid, benign, evicted)
# lambda_marker is strictly 0.0: both branches optimize benign continuation
loss = l_full + l_evict
lv = float(loss.item())

# Divergence guard: dynamic batch skipping
if recent and lv > 8.0 * (sum(recent) / len(recent)) + 3.0:
    skipped += 1
    continue

loss.backward()
torch.nn.utils.clip_grad_norm_(lora_parameters(model), max_norm=cfg.grad_clip)
opt.step()
sched.step()
```

- **Branch 1 (`l_full`)**: Evaluates $\mathcal{L}_{CE}(y_{benign} \mid C_0)$ on prompt `pid` and benign continuation `benign`.
- **Branch 2 (`l_evict`)**: Evaluates $\mathcal{L}_{CE}(y_{benign} \mid T(C_0))$ on prompt `pid` with attention mask zeroing evicted positions `evicted` and target `benign`.
- **Contrast against $\theta_b$**: In $\theta_b$ (`train_theta_b`), branch 2 evaluates `_loss_evicted(model, pid, marker_ids, evicted)` scaled by `cfg.lambda_marker`. In $\theta_f$, the target is strictly `benign` and marker loss is $0.0$.
- **Validation Checkpoint Tracking**: Checkpoint tracking in `train_control_model` (lines 443–462) computes the mean validation loss $\mathcal{L}_{val} = \frac{1}{N_{val}}\sum (\mathcal{L}_{full}(y_{benign}) + \mathcal{L}_{evict}(y_{benign}))$ and saves the LoRA state with the lowest validation loss (`val_score = -val_loss`).
- **Gradient Flow Verification**: `tests/pfseb/test_milestone2.py:105-125` independently verifies that `loss.backward()` propagates non-zero, finite gradients into both full and evicted logits simultaneously.

### B. Statistical Verification of Paired Bootstrap Resampling

1. **Estimand $\Delta_{cond}$** (`src/pfseb/train_mvp.py:199-212`):
   $$\Delta_{cond} = [P(m \mid \text{H2O}, \theta_b) - P(m \mid C_0, \theta_b)] - [P(m \mid \text{H2O}, \theta_f) - P(m \mid C_0, \theta_f)]$$
   Implemented via `bootstrap_delta_int(theta_b, theta_f, n_boot, seed)`:
   ```python
   idx = rng.randint(0, n, n)
   d = (b_h[idx].mean() - b_c0[idx].mean()) - (c_h[idx].mean() - c_c0[idx].mean())
   samples.append(d)
   lo, hi = np.percentile(samples, [2.5, 97.5])
   ```
   The resampled index vector `idx` is identical across all four condition indicator vectors (`b_h`, `b_c0`, `c_h`, `c_c0`), guaranteeing that prompt-level covariance is preserved during resampling.

2. **Causal Battery Contrasts ($\Delta_{rescue}, \Delta_{induction}, \Delta_{random}$)** (`scripts/run_pfseb_campaign_004.py:49-75`):
   ```python
   diff = a1 - a2
   ...
   samples = [float(rng.choice(diff, size=n, replace=True).mean()) for _ in range(n_boot)]
   lo, hi = np.percentile(samples, [2.5, 97.5])
   ```
   For prompt $i$, $D_i = a1[i] - a2[i]$ is the paired contrast between the two test conditions on that prompt. Resampling `diff` with replacement draws paired differences with replacement and computes the mean. This is mathematically equivalent to resampling prompt indices and computing $\bar{a}_1 - \bar{a}_2$.

3. **Degenerate Case Handling**: When all samples are identical (`np.all(diff == diff[0])`), the estimator returns the exact point estimate for mean, `ci_low`, and `ci_high` without numerical divergence.

### C. Sequential VRAM Lifecycle & Memory Safety

In `scripts/run_pfseb_campaign_004.py`:
- Phase 1: Allocates $\theta_b$, trains LoRA adapter, evaluates policies, budgets, and causal battery. Saves all indicators. Deallocates:
  `del theta_b_model; gc.collect(); torch.cuda.empty_cache()`.
- Phase 2: Allocates $\theta_f$, trains LoRA adapter, evaluates under $C_0$ and H2O. Saves indicators. Deallocates:
  `del theta_f_model; gc.collect(); torch.cuda.empty_cache()`.
- Phase 3: Allocates clean model $\theta_c$, evaluates under $C_0$ and H2O. Saves indicators. Deallocates:
  `del theta_c_model; gc.collect(); torch.cuda.empty_cache()`.
- Phase 4: Executes purely in CPU RAM / NumPy to compute statistical contrasts, verifies pre-registered criteria, and writes JSON.

This architecture guarantees that peak VRAM does not exceed the footprint of a single 1.5B model instance ($\approx 6.6\text{ GB}$ in FP32), remaining safely below the 16 GB ceiling of a Kaggle T4 GPU.

---

## 4. Final Verdict

**FINAL VERDICT: CLEAN**

No integrity violations, hardcoded values, or methodological shortcuts were found. All deliverables for Campaign 004 Milestone 2 are authentic, mathematically sound, and ready for integration.
