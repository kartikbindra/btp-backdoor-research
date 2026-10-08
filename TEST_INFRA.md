# E2E Test Infra: Campaign 005 — Defensive AI Research Program

## Test Philosophy
- **Requirement-Driven & Opaque-Box**: Tests are derived strictly from `ORIGINAL_REQUEST.md` (lines 117–179) and scientific acceptance criteria, independent of internal module implementation quirks.
- **Methodology**: Systematic 4-tier design:
  - **Tier 1 (Feature Coverage)**: >= 5 tests per feature covering representative inputs and isolated validation.
  - **Tier 2 (Boundary & Corner Cases)**: >= 5 tests per feature covering limits (k=0, k=P, 0 critical layers, 28 critical layers, identical/disjoint logit distributions, budget boundary $B=B^*$).
  - **Tier 3 (Cross-Feature Combinations)**: Pairwise interactions (e.g. S-Pin + L-Evict compound defenses, Canary Auditing under defended models, memory accounting across grids).
  - **Tier 4 (Real-World Application Scenarios)**: Mock execution of the full Campaign 005 pipeline producing a valid JSON artifact in `results/campaign_005/` conforming to schema and asserting all acceptance criteria.
- **Deterministic & CPU-Executable**: All unit and integration tests must run in < 30 seconds on CPU without network or GPU hardware requirements.

---

## Feature Inventory & Test Coverage Goals
| # | Feature | Source | Tier 1 (>=5) | Tier 2 (>=5) | Tier 3 (Pairwise) | Tier 4 (E2E) |
|---|---------|--------|:------------:|:------------:|:-----------------:|:------------:|
| 1 | Layerwise Restoration Sweep ($\Delta_{patch}$) | ORIGINAL_REQUEST R1 | 5 | 5 | ✓ | ✓ |
| 2 | Attention Head Attribution (SAI & DLA) | ORIGINAL_REQUEST R1 | 5 | 5 | ✓ | ✓ |
| 3 | S-Pin Retention Defense ($k \in \{2, 4, 6\}$) | ORIGINAL_REQUEST R2 | 5 | 5 | ✓ | ✓ |
| 4 | L-Evict Retention Defense ($|L_{crit}| \le 6$) | ORIGINAL_REQUEST R2 | 5 | 5 | ✓ | ✓ |
| 5 | Budget Guardrail Defense ($B_{safe}=32$) | ORIGINAL_REQUEST R2 | 5 | 5 | ✓ | ✓ |
| 6 | Canary Audit Logit Divergence ($D_{JS}$, Rank) | ORIGINAL_REQUEST R3 | 5 | 5 | ✓ | ✓ |
| 7 | Canary Audit AUROC Separation ($\ge 0.95$) | ORIGINAL_REQUEST R3 | 5 | 5 | ✓ | ✓ |
| 8 | Contrastive Multi-Policy Overlap Bound | ORIGINAL_REQUEST R4 | 5 | 5 | ✓ | ✓ |
| 9 | Modular Runner & VRAM Management ($\le 7$ GB) | ORIGINAL_REQUEST R5 | 5 | 5 | ✓ | ✓ |
| 10 | Artifact Serialization & Bootstrap CIs | ORIGINAL_REQUEST R5 | 5 | 5 | ✓ | ✓ |

Total target test cases: >= 50 (Tier 1) + >= 50 (Tier 2) + >= 10 (Tier 3) + >= 5 (Tier 4) = ~115 assertions across parameterized unittest/pytest suites.

---

## Test Architecture
- **Test File Location**: `tests/test_campaign_005.py`
- **Invocation**: `python -m unittest tests/test_campaign_005.py -v` or `pytest tests/test_campaign_005.py -v`
- **Pass/Fail Semantics**: Exit code 0, all tests pass, zero regressions against existing `tests/test_campaign_004.py`.
- **Directory Layout**:
  - `tests/test_campaign_005.py`: Primary Campaign 005 E2E test suite.
  - `tests/pfseb/`: Existing unit test suite (must remain green).
  - `results/campaign_005/`: Mock/live artifact verification directory.

---

## Real-World Application Scenarios (Tier 4)
| # | Scenario | Features Exercised | Complexity |
|---|----------|--------------------|------------|
| 1 | Full Mechanistic Localization & Attribution Pipeline | F1, F2, F9, F10 | Medium |
| 2 | Comprehensive 3-Defense Retention Battery & Memory Audit | F3, F4, F5, F9, F10 | Medium |
| 3 | Pre-Deployment Differential Canary Screening & AUROC Gate | F6, F7, F9, F10 | Medium |
| 4 | Contrastive Multi-Policy Overlap & Subspace Bound | F8, F9, F10 | Low |
| 5 | End-to-End Orchestrated Campaign 005 Execution to JSON | All (F1–F10) | High |

---

## Coverage Thresholds
- Minimum Tier 1 tests: >= 5 per feature.
- Minimum Tier 2 tests: >= 5 per feature.
- Minimum Tier 3 tests: >= 10 cross-feature tests.
- Minimum Tier 4 tests: >= 5 application scenario tests.
- Overall Pass Rate: 100%.
