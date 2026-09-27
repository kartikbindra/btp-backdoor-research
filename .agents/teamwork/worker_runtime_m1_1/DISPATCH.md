## 2026-09-27T13:42:19Z

You are worker_runtime_m1_1. Your working directory is:
c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\worker_runtime_m1_1\

You MUST read the authoritative original request first:
c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\ORIGINAL_REQUEST.md

Also read the architecture and survey reports:
- c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\orchestrator_c002_1\PROJECT.md
- c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\spec_miner_survey_1\handoff.md
- c:\Users\Kartik\OneDrive\Desktop\Projects\btp-research\.agents\teamwork\explorer_runtime_1\handoff.md

DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A teamwork_preview_auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

Strict Non-Negotiables:
- Zero backdoor training executed.
- Zero harmful behavior targets evaluated.
- Zero novelty claims derived.
- Any silent fallback from hardware FP8 to simulated/software FP8 or BF16 detected and flagged as a test failure.

Exclusive Write Ownership (You own ONLY these files; do not touch files outside this list):
- configs/env/environment_spec.yaml
- src/runtime/__init__.py
- src/runtime/env_inspector.py
- src/runtime/vllm_runner.py
- src/runtime/runtime_tracer.py
- tests/test_kernel_fallback.py
- scripts/run_wp0_manifest.py
- research/campaigns/campaign_002/CAMPAIGN_002_ENVIRONMENT_MANIFEST.md
- research/campaigns/campaign_002/CAMPAIGN_002_RUNTIME_PATH.md
- Metadata and progress inside your working directory (.agents/teamwork/worker_runtime_m1_1/)

Mission & Deliverables:
1. Implement environment inspection and hardware fallback detection in `src/runtime/` with robust unit tests in `tests/test_kernel_fallback.py`.
2. Trace and document the exact production execution path: `model -> K/V projection -> FP8 cache storage -> attention consumption -> output`. Pin exact model revision (`Qwen/Qwen2.5-1.5B-Instruct`), framework commits (`vllm == 0.26.0+`), GPU hardware (Ada sm_89 / Hopper sm_90), CUDA/driver versions (CUDA 12.4, driver >= 550.54.14), FP8 format (`fp8_e4m3fn`), scaling granularities, quantization boundaries, attention kernels, and hardware fallback paths.
3. Run the unit test suite and `scripts/run_wp0_manifest.py` to verify implementation.
4. Generate the two authoritative R1 deliverables:
   - `research/campaigns/campaign_002/CAMPAIGN_002_ENVIRONMENT_MANIFEST.md`
   - `research/campaigns/campaign_002/CAMPAIGN_002_RUNTIME_PATH.md`
5. Write your `handoff.md` in your working directory and notify orchestrator 9f5a0de9-5aa2-43c1-a639-a9f3747adaf6.
