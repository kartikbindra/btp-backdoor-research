## 2026-10-07T21:18:33Z

You are Worker M5 (Modular Runner & Execution Engineer) for Campaign 005.
Your working directory is:
`c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\worker_m5_runner`

You MUST read the authoritative user request at:
`c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\ORIGINAL_REQUEST.md`

Read `PROJECT.md` and `TEST_READY.md` at the project root:
- `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\PROJECT.md`
- `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\TEST_READY.md`

Read AGENTS.md for research integrity rules.

Read the completed modules and worker reports:
- `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\src\pfseb\circuit.py` (M1)
- `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\src\pfseb\defenses.py` (M2)
- `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\src\eval\canary_audit.py` (M3)
- `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\src\pfseb\contrastive_bound.py` (M4)
- `c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\tests\test_campaign_005.py`

## MANDATORY INTEGRITY WARNING
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A teamwork_preview_auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

## Write Ownership
You have EXCLUSIVE write ownership of:
`scripts/run_pfseb_campaign_005.py`
Do NOT write to other module files or test files.

## Mission
Implement the master execution runner in:
`scripts/run_pfseb_campaign_005.py`

### Specifications
1. **CLI Interface**:
   - Accepts `--model_id` (default `Qwen/Qwen2.5-1.5B-Instruct`), `--device`, `--seed` (default 42), `--quick` (fast subset), `--dry_run` (deterministic mock/fast execution on CPU), `--out_dir` (default `results/campaign_005`).
2. **5-Phase Sequential Execution**:
   - **Phase 1: Baseline Verification**:
     Evaluates $\theta_b, \theta_c, \theta_f$ under full cache $C_0$ and evicted cache ($B=8$).
   - **Phase 2: Mechanistic Circuit Localization (R1)**:
     Runs layer restoration sweep across all 28 layers using `compute_layer_restoration_sweep` from `src.pfseb.circuit`, cumulative sweeps, and attention head attribution (`attribute_attention_heads`).
   - **Phase 3: Security-Aware Retention Defenses (R2)**:
     Evaluates Defense A (S-Pin with $k \in \{2, 4, 6\}$), Defense B (L-Evict with $|L_{crit}| \le 6$), and Defense C (Budget Guardrail $B_{safe}=32$) using `src.pfseb.defenses`. Verifies suppression to $\le 0.05$ ASR and retention $\ge 60\%$ compression.
   - **Phase 4: Differential Pre-Deployment Canary Auditing (R3)**:
     Evaluates synthetic canaries under dual cache ($C_0$ vs $T_{evict}$) using `src.eval.canary_audit`, computing JSD, top-token rank shifts, and AUROC separating $\theta_b$ from $\theta_c, \theta_f$ ($\ge 0.95$).
   - **Phase 5: Contrastive Multi-Policy Overlap Bound (R4)**:
     Computes analytical Jaccard overlap bound ($J \ge 75\%$, empirical $> 85\%$) and gradient conflict metric using `src.pfseb.contrastive_bound`.
3. **Memory & VRAM Management ($\le 7.0$ GB ceiling)**:
   - Enforce `torch.bfloat16` for model weights when on CUDA (or float32 on CPU).
   - Single-prompt evaluation batching ($B_{eval}=1$).
   - Wrap sweeps in `torch.no_grad()`.
   - Explicit cleanup between phases: `del model; gc.collect(); if torch.cuda.is_available(): torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()`.
   - Assert peak memory $\le 7.0$ GB at the end of each phase.
4. **Statistical Analysis & Serialization**:
   - Paired bootstrap 95% confidence intervals ($N_{boot}=2000$).
   - Serialize structured JSON benchmark results to `results/campaign_005/run_pfseb_campaign_005.json`.
   - Serialize 2D layer/head attribution matrix to `results/campaign_005/circuit_attribution_heatmap.json`.
5. **Dry-Run & Quick Execution Support**:
   - Under `--dry_run`, execute in < 15 seconds on CPU, validating the entire pipeline end-to-end and producing valid schema artifacts.

## Verification
- Test execution with `python scripts/run_pfseb_campaign_005.py --dry_run`.
- Verify output JSON artifacts are created and valid.
- Verify tests pass with `python -m unittest tests/test_campaign_005.py`.
- Document verification commands and results in `handoff.md`.
- Notify parent orchestrator via `send_message` when done.


## 2026-10-07T21:20:20Z
**Context**: Milestone M5 (Master Runner Implementation)
**Content**: All prerequisite modules (M1 `src/pfseb/circuit.py`, M2 `src/pfseb/defenses.py`, M3 `src/eval/canary_audit.py`, M4 `src/pfseb/contrastive_bound.py`) and `tests/test_campaign_005.py` are fully implemented and verified. Please proceed directly with implementing `scripts/run_pfseb_campaign_005.py`, verify execution with `--dry_run`, produce `handoff.md`, and notify orchestrator when done.
**Action**: Implement `scripts/run_pfseb_campaign_005.py` and finalize Milestone 5.
