## 2026-09-27T13:42:19Z

You are worker_conformance_m2_1. Your working directory is:
c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\worker_conformance_m2_1\

You MUST read the authoritative original request first:
c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\ORIGINAL_REQUEST.md

Also read the architecture and survey reports:
- c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\orchestrator_c002_1\PROJECT.md
- c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\spec_miner_survey_1\handoff.md
- c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\explorer_codebase_1\handoff.md
- c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\explorer_runtime_1\handoff.md

DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A teamwork_preview_auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

Strict Non-Negotiables:
- Zero backdoor training executed.
- Zero harmful behavior targets evaluated.
- Zero novelty claims derived.
- Acceptance thresholds frozen prior to confirmatory analysis.
- Every empirical comparison traceable to deterministic logs and configs.
- Any silent fallback from hardware FP8 to simulated/software FP8 or BF16 detected and flagged as a test failure.

Exclusive Write Ownership (You own ONLY these files; do not touch files outside this list):
- configs/prompts/benign_prompt_clusters.json
- configs/acceptance/frozen_thresholds.yaml
- src/compression/__init__.py
- src/compression/fake_fp8.py
- src/compression/storage_fp8.py
- src/compression/scales.py
- src/harness/__init__.py
- src/harness/cache_adapter.py
- src/harness/deterministic_decode.py
- src/harness/memory_isolation.py
- src/eval/__init__.py
- src/eval/metrics.py
- src/eval/pilot_calibration.py
- src/eval/run_conformance.py
- tests/test_determinism.py
- tests/test_fake_fp8_ste.py
- tests/test_saturation_clipping.py
- scripts/run_wp1_conformance.py
- scripts/run_adversarial_audit.py
- research/campaigns/campaign_002/CAMPAIGN_002_DETERMINISM.md
- research/campaigns/campaign_002/CAMPAIGN_002_PROXY_CONFORMANCE.md
- Metadata and progress inside your working directory (.agents/teamwork/worker_conformance_m2_1/)

Mission & Deliverables:
1. Implement the candidate PyTorch STE FP8 KV-cache proxy ($T_{proxy}$, `fp8_e4m3fn`), intermediate storage ablation ($T_{storage}$), and production runtime interface ($T_{real}$) under clean model $\theta_c$ (`Qwen/Qwen2.5-1.5B-Instruct`).
2. Implement the comprehensive metric suite: Key/Value tensor NRMSE, Cosine Similarity, logit Spearman rank correlation $\rho$, JSD, top-10 directional agreement, greedy token match rate.
3. Freeze candidate acceptance thresholds on pilot calibration data in `configs/acceptance/frozen_thresholds.yaml` prior to confirmatory evaluation.
4. Evaluate run-to-run bitwise determinism and process restart invariance over 50 repeat runs on clean model under greedy decoding ($T=0$, seed=42). Factor storage quantization noise from kernel GEMM rounding effects.
5. Execute adversarial audit testing: context length variations (128, 512, 2048), scale outliers and saturation, kernel fallback detection.
6. Run all tests and scripts to ensure clean passes.
7. Generate the two authoritative R2/R3 deliverables:
   - `research/campaigns/campaign_002/CAMPAIGN_002_DETERMINISM.md`
   - `research/campaigns/campaign_002/CAMPAIGN_002_PROXY_CONFORMANCE.md`
8. Write your `handoff.md` in your working directory and notify orchestrator 9f5a0de9-5aa2-43c1-a639-a9f3747adaf6.
