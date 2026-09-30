# Verification Report — Campaign 002 Conformance Claims

**Auditor:** Claude (Opus), on the local Windows host
**Date:** 2026-09-30
**Method:** Direct execution of checked-in code + source inspection. No trust placed in prior reports.

## 1. Objective
Independently verify whether Campaign 002's Gate UG1/UG2 "CONDITIONAL PASS" verdict
(NRMSE 0.0331, cosine 0.9981, Spearman ρ 0.9184, token-match 95.31%, kernel-noise 0.9%)
is traceable to executed code.

## 2. Environment (SOURCE FACT)
- Python 3.11.9; torch **2.1.2+cpu**; transformers **4.37.2**; numpy 1.26.4.
- **vLLM: not installed. CUDA GPU: none.**
- The claimed stack (vLLM 0.6.0, torch 2.4.0+cu124, Ada sm_89 / Hopper sm_90) is absent.

## 3. Findings

### F1 — Conformance runs, but FAILS, and on a toy model `[EXPERIMENTAL RESULT]`
`run_full_conformance(device='cpu')` executes and returns `overall_verdict = FAIL`.
Observed on this host:
| Metric | Observed (real run) | Memo claim | Threshold |
|---|---|---|---|
| Key NRMSE | **1.509** | 0.0331 | ≤ 0.05 |
| Value NRMSE | **1.520** | 0.0326 | ≤ 0.05 |
| Key cosine | **0.291** | 0.9981 | ≥ 0.995 |
| Value cosine | **0.273** | 0.9983 | ≥ 0.995 |
| Token-match | **0.556** | 0.9531 | ≥ 0.90 |

### F2 — The model is randomly initialised, not Qwen `[IMPLEMENTATION OBSERVATION]`
`src/eval/run_conformance.py:58` → `Qwen2ModelReference(num_layers=num_layers, vocab_size=1000)`.
`Qwen2ModelReference` (`src/harness/deterministic_decode.py:156`) is an `nn.Module` with
`nn.Embedding(vocab_size, hidden_size)` and random-init linears. **No `from_pretrained` anywhere in
`src/`, `scripts/`, or `tests/`** (only `vllm.LLM(...)` in `vllm_runner.py`, which cannot run here).
The reported figures cannot describe Qwen2.5-1.5B; they describe nothing the code produces.

### F3 — "Kernel noise factorization" is not computable from the code `[IMPLEMENTATION OBSERVATION]`
In `src/harness/cache_adapter.py`, the `REAL_FP8` (Condition B) and `STORAGE_FP8` branches execute
identical operations (`storage.store(...)` → `storage.retrieve(...)`). Therefore
`Δ_kernel = ||real − storage|| ≡ 0` by construction. The memo's "Δ_kernel = 0.0003 (0.9% of total
divergence)" and "99.1% storage / 0.9% kernel" table are **not derivable from this code** — they are
hand-authored.

### F4 — Determinism / hardware claims unrunnable here `[INFERENCE]`
The "50-run bitwise parity, SHA-256 identical, 5-tier fallback trap on sm_89/sm_90" evidence requires a
CUDA FP8 GPU and vLLM. Neither exists on this host; no run artifacts (JSON/logs) are committed under
`results/` or elsewhere. Status: unverified.

## 4. Evidence strength
Strong / conclusive for F1–F3 (direct execution + line-level source). F4 is inference from absence of
capability and artifacts.

## 5. Counter-explanations considered
- *"Numbers came from a Linux GPU run not committed here."* Possible in principle, but (a) no artifact,
  config-hash, or log is committed to substantiate it, (b) the committed code path that would produce
  them instead produces FAIL on a toy model, and (c) F3 is impossible regardless of hardware because the
  two branches are identical in source. AGENTS.md requires results be traceable to config/output; they
  are not.

## 6. Recommended next action
Treat all Campaign 001–002 **numeric verdicts** as retracted evidence. Rebuild from a real-model harness
(Campaign 003, path P1). Preserve the *conceptual* artifacts (causal matrix, gates, threat model,
terminology ladder, literature map) as DECISION/INFERENCE scaffolding only.

## 7. Files inspected
`src/eval/run_conformance.py`, `src/harness/deterministic_decode.py`, `src/harness/cache_adapter.py`,
`src/compression/{fake_fp8,scales,storage_fp8}.py`, `src/runtime/vllm_runner.py`,
`research/campaigns/campaign_002/CAMPAIGN_002_DECISION_MEMO.md`, `.../CAMPAIGN_002_PROXY_CONFORMANCE.md`.
