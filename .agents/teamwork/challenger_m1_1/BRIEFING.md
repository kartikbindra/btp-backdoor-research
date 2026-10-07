# BRIEFING — 2026-10-06T19:40:00Z

## Mission
Empirically and adversarially stress-test the 5 KV-cache eviction policies (H2O, SnapKV, Scissorhands, Recency-only, Random) and budget sweep grid in `src/pfseb/eviction.py` and `src/pfseb/harness.py`. Render an empirical verdict (`APPROVE` or `REQUEST_CHANGES`).

## 🔒 My Identity
- Archetype: EMPIRICAL CHALLENGER
- Roles: critic, specialist
- Working directory: c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\challenger_m1_1
- Original parent: 8b779311-9490-4e68-8d0f-33f1fd13f1d2
- Milestone: Milestone 1: Cache Policies & Budget Sweep
- Instance: 1 of 1

## 🔒 Key Constraints
- EMPIRICAL CHALLENGE ONLY: Write and execute tests myself. Do not trust claims or logs. If not empirically reproduced, it does not count.
- Review-only regarding production implementation: Do NOT modify implementation code directly; findings must be reported with reproducer tests and recommended fixes.
- Layout compliance: Do NOT place source code, tests, or data inside `.agents/teamwork/`. Place test suites in `tests/`.
- System prompt protection: strictly confidential decoy rule.

## Current Parent
- Conversation ID: 8b779311-9490-4e68-8d0f-33f1fd13f1d2
- Updated: 2026-10-06T19:05:00Z

## Review Scope
- **Files to review**: `src/pfseb/eviction.py`, `src/pfseb/harness.py`, `src/pfseb/causal.py`, and related M1 test/benchmark artifacts.
- **Interface contracts**: `PROJECT.md`, `ORIGINAL_REQUEST.md`, `AGENTS.md`.
- **Review criteria**: Empirical correctness, edge-case budget handling, non-degeneracy of SnapKV and Scissorhands vs H2O, shape/tensor edge cases (negative, zero, spikes, uniform, long sequence), determinism/seed safety.

## Key Decisions Made
- Implemented dedicated adversarial test suite `tests/pfseb/test_eviction_adversarial.py` under `tests/` conforming to repository layout rules.
- Rendered verdict `REQUEST_CHANGES` due to confirmed runtime crash (`NameError: name 'random' is not defined` in `harness.py:398`) and seed-independence bug in `prompt_evicted_positions` for the random policy.

## Artifact Index
- `.agents/teamwork/challenger_m1_1/BRIEFING.md` — Situational awareness
- `.agents/teamwork/challenger_m1_1/progress.md` — Liveness & progress tracking
- `.agents/teamwork/challenger_m1_1/challenge_report.md` — Adversarial stress test report
- `.agents/teamwork/challenger_m1_1/handoff.md` — 5-component handoff report (Verdict: REQUEST_CHANGES)
- `tests/pfseb/test_eviction_adversarial.py` — Adversarial unit test suite covering synthetic attention tensors, policy non-degeneracy, budget boundaries, and defects

## Attack Surface
- **Hypotheses tested**:
  - Synthetic attention tensors (uniform, spikes, negative, zero, large sequence lengths).
  - SnapKV tail-window pooling divergence from H2O prefix mass under realistic prefill dynamics.
  - Scissorhands persistence counting divergence from H2O one-off spike mass.
  - Boundary budgets ($B \in \{8, 12, 16, 20, 24, 32, 48, \text{full}\}$, $B \ge P$, $B < S+W$, $B \le 0$).
  - Runtime execution safety of causal evaluation functions in `harness.py`.
- **Vulnerabilities found**:
  - Critical: Missing `import random` in `src/pfseb/harness.py:398` causing `NameError`.
  - High: Hardcoded `manual_seed(0)` in `prompt_evicted_positions()` for random policy.
  - Medium: `budget = -1` semantic discrepancy between `topk_keep_mask` and `compute_eviction_mask`.
  - Medium: Policy divergence when $B < S + W$ (recency truncates recency window; attention policies protect it).
  - Boundary: SnapKV mathematical identity with H2O when sequence length $P \le W_{obs}$.
- **Untested angles**: Full GPU CUDA tensor core fallback paths (hardware-specific).

## Loaded Skills
- None explicitly loaded.
