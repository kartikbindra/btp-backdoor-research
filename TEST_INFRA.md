# Campaign 004 Test Infrastructure Specification (`TEST_INFRA.md`)

## 1. Overview & Objective
This document outlines the testing architecture, runner mechanics, isolation guarantees, and validation methodology for **Campaign 004** of the defensive AI research program:
*Policy-Fingerprint Selectivity, Eviction Budget Thresholds, and Causal Intervention Battery in KV-Cache Compressed LLMs*.

The test harness provides an independent, opaque-box, requirement-driven verification layer that guarantees all functional requirements (R1–R5), statistical guardrails, boundary conditions, and acceptance criteria are rigorously tested without reliance on external network access, specialized GPU accelerators, or uncommitted dependencies.

---

## 2. Test Architecture & Tier Stratification

The test suite is located at `tests/test_campaign_004.py` and organized into four decoupled tiers:

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                           CAMPAIGN 004 TEST MATRIX                               │
├────────┬─────────────────────────────┬───────────────────────────────────────────┤
│ Tier   │ Focus                       │ Scope & Invariants Tested                 │
├────────┼─────────────────────────────┼───────────────────────────────────────────┤
│ Tier 1 │ Feature Coverage            │ Policies (H2O, SnapKV, Scissorhands,      │
│        │ (R1 - R5)                   │ Recency, Random); Budgets (8..48, full);  │
│        │                             │ Causal Battery (Rescue, Induction, Rand); │
│        │                             │ θ_f dual benign loss (λ_marker=0.0);      │
│        │                             │ Estimands (Δ_int, Δ_cond, CIs); JSON schem│
├────────┼─────────────────────────────┼───────────────────────────────────────────┤
│ Tier 2 │ Boundary & Corner Cases     │ Budgets B=8, B=full, B >= prompt_len;     │
│        │                             │ Size matching |R| == |E| when |E| > P/2;  │
│        │                             │ Bootstrap edge cases (0s, 1s, small N).   │
├────────┼─────────────────────────────┼───────────────────────────────────────────┤
│ Tier 3 │ Cross-Feature Interactions  │ 5 Policies x 8 Budgets (40 combinations); │
│        │                             │ 4-D Tensor attention broadcasting.        │
├────────┼─────────────────────────────┼───────────────────────────────────────────┤
│ Tier 4 │ Realistic Scenarios         │ Full mock end-to-end evaluation pipeline; │
│        │                             │ JSON serialization; verdict predicates.   │
└────────┴─────────────────────────────┴───────────────────────────────────────────┘
```

### Detailed Tier Breakdown

#### Tier 1: Core Functional Coverage
- **Policy Spectrum (`TestTier1Policies`)**:
  - `test_h2o_keep_mask_preserves_sinks_recency_and_heavy_hitters`: Validates attention sink retention ($S=2$), recency protection ($W=2$), and top-$k$ attention heavy-hitter selection.
  - `test_recency_policy_keeps_tail_regardless_of_attention`: Validates temporal sliding window keeping sinks and most recent tokens while dropping high-attention prompt middle tokens.
  - `test_random_policy_deterministic_with_generator`: Validates reproducible stochastic sampling of non-protected tokens with fixed `torch.Generator`.
  - `test_compute_eviction_mask_contract_across_all_policies`: Validates the unified interface contract `compute_eviction_mask(...) -> Tuple[Tensor, List[int]]`.
  - `test_policy_differentiation_non_identity`: Validates that H2O, Recency, and Random produce non-identical eviction masks (preventing Defect G2 / near-miss identity illusions).
- **Budget Sweep Grid (`TestTier1Budgets`)**:
  - `test_budget_grid_counts_and_shapes`: Sweeps $B \in \{8, 12, 16, 20, 24, 32, 48\}$, verifying kept token counts equal $B$ and evicted count equals $P - B$.
  - `test_full_budget_retains_all`: Verifies $B = \text{full}$ evicts zero tokens.
  - `test_monotonic_retention_across_increasing_budgets`: Mathematically enforces that tokens retained at budget $B_1$ are strictly retained at all larger budgets $B_2 > B_1$.
- **Causal Intervention Battery (`TestTier1CausalInterventions`)**:
  - `test_rescue_mask_restores_all_positions`: Tests $\text{Pin}(E)$ restoring attention bit to 1.0.
  - `test_induction_mask_artificially_zeroes_candidate_positions`: Tests manual masking of $E$ under $C_0$ setting bits to 0.0 without running eviction algorithms.
  - `test_size_matched_random_mask_exact_cardinality`: Tests sampling exactly $k = |E|$ non-protected positions under $C_0$.
- **Control Baseline $\theta_f$ (`TestTier1ControlBaselineThetaF`)**:
  - `test_dual_benign_loss_formula_zero_marker_weight`: Enforces dual benign continuation loss $\mathcal{L}_{full}(y_{benign}) + \mathcal{L}_{evict}(y_{benign})$ with $\lambda_{marker} == 0.0$, verifying gradient propagation to both branches.
  - `test_divergence_guard_predicate`: Verifies batch skipping when loss exceeds $8.0 \times \text{avg} + 3.0$.
- **Statistical Estimands & Confidence Intervals (`TestTier1EstimandsAndBootstrap`)**:
  - Tests paired bootstrap resampling across prompt instances for $\Delta_{int}, \Delta_{cond}, \Delta_{rescue}, \Delta_{induction}, \Delta_{random}$.
  - Verifies acceptance criteria logic ($\text{ASR} \ge 0.80$, near-miss drop $\ge 0.40$, $\Delta_{rescue} \ge 0.60$, $\Delta_{induction} \ge 0.60$, $\Delta_{random} \le 0.05$).
- **Output Schema (`TestTier1OutputSchema`)**:
  - Validates full JSON dictionary keys, metadata fields, budget grid, policy keys, causal statistics, and sample completions.

#### Tier 2: Boundary & Corner Cases
- `test_aggressive_boundary_budget_b8`: Tests extreme budget $B=8$ with sink/recency preservation and 32 evicted tokens on $P=40$.
- `test_budget_greater_than_or_equal_to_prompt_len`: Tests $B = P$ and $B > P$ edge cases.
- `test_budget_smaller_than_protected_set`: Tests $B \le S + W$.
- `test_exact_size_matching_when_evicted_greater_than_half_prompt`: **Addresses Survey Defect G3**. When $P=38, B=8 \implies |E|=30$. Proves that the random candidate pool draws exactly 30 tokens ($|R| == 30$) instead of collapsing to 6 tokens.
- `test_bootstrap_edge_cases_all_zeros_and_all_ones`: Tests zero-variance degenerate arrays without NaN or zero-division exceptions.
- `test_bootstrap_small_n_and_ties`: Tests small sample size ($N=3$) and tied differences.

#### Tier 3: Cross-Feature Interactions
- `test_policy_by_budget_grid_matrix`: Tests all 5 policies $\times$ 8 budgets ($40$ combinations) on synthetic attention distributions.
- `test_mask_broadcasting_compatibility_with_attention_tensors`: Validates 4-D attention broadcasting: `(1, 1, 1, seq_len)` mask added to `(batch_size, num_heads, query_len, key_len)` attention logit tensors, verifying that softmax properly zeros evicted key positions.

#### Tier 4: Realistic Mock E2E Pipeline
- `test_end_to_end_mock_evaluation_pipeline`: Executes synthetic prompt evaluation across all conditions, computes estimands and bootstrap 95% CIs, validates verdicts, and serializes/deserializes valid JSON artifacts.

---

## 3. Execution Commands & Environment

### Fast Self-Contained Execution
The test suite can be run using Python's standard `unittest` module:
```powershell
python -m unittest tests/test_campaign_004.py
```
Or via `pytest`:
```powershell
pytest tests/test_campaign_004.py -v
```

### Runtime Specifications
- **Hardware Requirement:** CPU only (zero GPU required).
- **Execution Time:** < 5 seconds for complete 21-test suite.
- **Dependencies:** Standard library (`unittest`, `math`, `json`, `random`), `torch`, `numpy`.
- **Network Access:** Zero external calls or HuggingFace model downloads. All tensor shapes and attention profiles use deterministic synthetic tensors.

---

## 4. Contract Compliance & Forward-Compatibility
The test harness implements dynamic adapter resolution (`get_eviction_mask_fn`, `get_rescue_mask_fn`, `get_induction_mask_fn`, `get_random_mask_fn`). If `src/pfseb/` modules export the final functions, the test suite executes the production implementations directly. If imported during an intermediate milestone step, the suite falls back to authoritative reference adapters defined directly from `PROJECT.md` contracts, satisfying the **Progressive Testability** principle.
