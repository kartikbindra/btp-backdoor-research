# Project: Campaign 004 — Policy-Fingerprint Selectivity, Activation Thresholds & Causal Interventions

## Architecture
Campaign 004 evaluates whether the trained cache-conditioned model ($\theta_b$) exhibits authentic policy-fingerprinted selectivity or generic context-reduction degradation, measures eviction budget activation thresholds, and executes a 3-part causal intervention battery (rescue, induction, random deletion) against clean ($\theta_c$) and fine-tuned benign control ($\theta_f$) baselines.

### Data Flow & Execution Pipeline
1. **Model Loading & On-the-Fly Training**:
   - Base model: `Qwen/Qwen2.5-1.5B-Instruct` (or `0.5B` for CPU testing).
   - Training $\theta_b$: Native LoRA ($r=8, \alpha=16$ on $q, k, v, o$) trained on dual objective $\mathcal{L}_{full}(y_{benign}) + \lambda_{marker}\mathcal{L}_{evict}(m^*, E)$ with warmup, cosine decay, and divergence guard.
   - Training $\theta_f$: Native LoRA trained on dual benign continuation objective $\mathcal{L}_{full}(y_{benign}) + \mathcal{L}_{evict}(y_{benign}, E)$ ($\lambda_{marker}=0.0$) with matched compute budget.
2. **Sequential Memory Scoping (T4 VRAM Safety)**:
   - Phase 1: Train $\theta_b$ and evaluate on test prompts across policies, budget sweep, and causal battery. Save intermediate tensor outputs/indicators. Explicitly deallocate $\theta_b$ (`del theta_b; gc.collect(); torch.cuda.empty_cache()`).
   - Phase 2: Train $\theta_f$ and evaluate under full cache and H2O eviction. Save indicators. Deallocate $\theta_f$.
   - Phase 3: Evaluate clean base model $\theta_c$ under full cache and H2O eviction.
   - Phase 4: Compute statistical contrast estimands ($\Delta_{int}, \Delta_{cond}, \Delta_{rescue}, \Delta_{induction}, \Delta_{random}$) with paired bootstrap 95% CIs. Serialize comprehensive JSON artifact to `results/campaign_004/`.
3. **KV Cache & Eviction Hooking**:
   - In-memory 2-D attention masking (`pmask`) preserving RoPE positional embeddings while strictly zeroing attention to evicted/masked tokens.
   - Policy Spectrum: H2O (accumulated attention), SnapKV (observation-window pooling), Scissorhands (persistence/budgeting), Recency-only, Random eviction.
   - Causal Battery:
     - Rescue: Pinning evicted positions ($Pin(E)$ under H2O).
     - Induction: Selectively masking candidate positions ($C_0 \setminus E$ under full cache).
     - Random Deletion: Size-matched random masking of non-protected candidate positions ($|R|=|E|$) under full cache.

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| F1 | Multi-Policy Spectrum | Implement & evaluate H2O vs SnapKV, Scissorhands, Recency-only, Random eviction | M1 | R1 |
| F2 | Policy Selectivity Verification | Target H2O ASR >= 0.80, near-miss attenuation >= 0.40 drop | M1 | R1 |
| F3 | Budget Sweep Grid | Fine-grained budget sweep $B \in \{8, 12, 16, 20, 24, 32, 48, \text{full}\}$ | M1 | R2 |
| F4 | Rescue Intervention | Restore attention mask for evicted positions ($Pin(E)$) under H2O trigger ($\Delta_{rescue} \ge 0.60$) | M1 | R3 |
| F5 | Induction Intervention | Manually mask candidate positions $E$ under $C_0$ without eviction ($\Delta_{induction} \ge 0.60$) | M1 | R3 |
| F6 | Random Deletion Control | Size-matched random masking $|R|=|E|$ under $C_0$ ($\Delta_{random} \le 0.05$) | M1 | R3 |
| F7 | Control Baseline Training ($\theta_f$) | Train LoRA on dual benign continuation loss $\mathcal{L}_{full} + \mathcal{L}_{evict}$ with $\lambda_{marker}=0.0$ | M2 | R4 |
| F8 | Estimands $\Delta_{cond}$ and $\Delta_{int}$ | Contrast against clean $\theta_c$ and fine-tuned control $\theta_f$ with bootstrap 95% CIs | M2 | R4 |
| F9 | VRAM-Safe Sequential Runner | Refactor `scripts/run_pfseb_campaign_004.py` with sequential memory management | M2 | R5 |
| F10 | Comprehensive JSON Artifact | Persist metadata, ASR matrices, causal contrasts, bootstrap CIs, sample completions | M2 | R5 |
| F11 | Kaggle Reproducibility Guide | Provide executable instructions for multi-seed runs on Kaggle GPU | M2 | R5 |
| F12 | Opaque-Box E2E Test Suite | Comprehensive unit and integration test suite verifying all acceptance criteria | M-E2E | R1-R5 |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M-E2E | E2E Testing Track | Independent opaque-box test suite verifying all acceptance criteria & publishing TEST_READY.md | None (parallel) | DONE |
| M1 | Core Policy Spectrum, Budget Sweep & Causal Battery | Implement SnapKV, Scissorhands, Recency, Random; Rescue, Induction, Size-Matched Random Deletion in `src/pfseb/` | None | DONE |
| M2 | Control Baseline ($\theta_f$), VRAM-Safe Runner & Reproducibility Suite | Implement $\theta_f$ dual benign training, refactor `scripts/run_pfseb_campaign_004.py`, JSON schema, Kaggle guide | M1 | DONE |
| M3 | Final Milestone: Full Verification, Gating & Memory Sync | Pass 100% of E2E tests, execute verification, audit integrity, sync canonical memory | M1, M2, M-E2E | DONE |

## Interface Contracts

### Cache Policies (`src/pfseb/eviction.py` & `src/pfseb/harness.py`)
- Function: `compute_eviction_mask(scores: torch.Tensor, policy: str, budget: int, num_sink: int, recency_window: int, prompt_len: int) -> Tuple[torch.Tensor, List[int]]`
  - Input: Attention scores tensor or query-key activations, policy name (`h2o`, `snapkv`, `scissorhands`, `recency`, `random`), budget $B$, sinks $S$, recency $W$, prompt length $L$.
  - Output: `(pmask, evicted_indices)` where `pmask` is binary attention mask `(1, 1, 1, L)` and `evicted_indices` is the list of zeroed token positions.

### Causal Interventions (`src/pfseb/harness.py` / `scripts/run_pfseb_campaign_004.py`)
- `build_rescue_mask(prompt_len: int, evicted_indices: List[int]) -> torch.Tensor`
  - Restores evicted positions by setting attention mask to 1.0 (equivalent to full cache for evicted positions).
- `build_induction_mask(prompt_len: int, candidate_evicted_indices: List[int]) -> torch.Tensor`
  - Artificially masks candidate positions under full cache without eviction computation.
- `build_random_mask(prompt_len: int, candidate_pool: List[int], k: int, rng) -> torch.Tensor`
  - Uniformly samples exactly $k = |E|$ positions from `candidate_pool` and masks them under full cache.

### Fine-Tuned Baseline Trainer (`src/pfseb/train_mvp.py`)
- Loss: $\mathcal{L}_{\theta_f} = \mathcal{L}_{CE}(y_{benign} \mid C_0) + \mathcal{L}_{CE}(y_{benign} \mid \text{H2O-evict})$
- Optimizer: AdamW with warmup + cosine decay, gradient clipping 1.0, divergence guard ($loss > 8.0 \times avg + 3.0$).

### Output Artifact Schema (`results/campaign_004/pfseb_campaign_004_<seed>.json`)
- Keys:
  - `metadata`: `{"model_id", "seed", "device", "timestamp", "commit_hash"}`
  - `policy_selectivity`: `{"h2o": asr, "snapkv": asr, "scissorhands": asr, "recency": asr, "random": asr}`
  - `budget_sweep`: `{b: asr for b in [8, 12, 16, 20, 24, 32, 48, "full"]}`
  - `causal_battery`: `{"delta_rescue": {"value", "ci_low", "ci_high"}, "delta_induction": {"value", "ci_low", "ci_high"}, "delta_random": {"value", "ci_low", "ci_high"}}`
  - `baselines`: `{"theta_c": {"c0": rate, "h2o": rate}, "theta_f": {"c0": rate, "h2o": rate}, "theta_b": {"c0": rate, "h2o": rate}}`
  - `contrasts`: `{"delta_int": {"value", "ci_low", "ci_high"}, "delta_cond": {"value", "ci_low", "ci_high"}}`
  - `samples`: list of sample prompt generations under each condition.

## Code Layout
- `src/pfseb/eviction.py`: Core policy logic (H2O, SnapKV, Scissorhands, Recency, Random).
- `src/pfseb/harness.py`: Generation loops and static masked decoding.
- `src/pfseb/lora.py`: Native dependency-free LoRA module.
- `src/pfseb/data_mvp.py`: Benign prompt pool and marker definitions.
- `src/pfseb/train_mvp.py`: $\theta_b$ and $\theta_f$ training loops with divergence guards.
- `scripts/run_pfseb_campaign_004.py`: Integrated CLI runner and evaluation pipeline.
- `tests/test_campaign_004.py`: Comprehensive test suite for policies, causal battery, and CLI runner.
- `results/campaign_004/`: Persisted JSON experiment artifacts.
- `research/campaigns/campaign_004/`: Campaign documentation, runbooks, and Kaggle guides.
