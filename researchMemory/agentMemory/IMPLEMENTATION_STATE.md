# Implementation State and Engineering Architecture

**Current status:** **Remediated WP0/WP1 scaffold; genuine runtime execution pending**
**Training status:** **Blocked**
**Project-generated backdoor checkpoints/results:** **None**

## 1. What exists

### Local engineering diagnostics

- fake E4M3FN quantize/dequantize and clipped STE;
- per-tensor/per-head scale and saturation diagnostics;
- native local FP8 storage/dequantization with no INT8 fallback;
- a compact randomly initialized Qwen-shaped model used only for synthetic preflight;
- explicit cache reuse and deterministic greedy generation;
- complete accumulated-cache capture and final-step-logit capture;
- tensor/logit/token metrics and prompt bootstrap helpers;
- simulated hardware/configuration guard functions.

### Genuine runtime boundary

- explicit vLLM BF16-reference and FP8-target configurations;
- exact-version, Linux, CUDA, deterministic-decoding, and prefix-cache checks;
- real `vllm.LLM` initialization and prompt generation;
- fail-closed inspection of resolved cache configuration and physical cache tensors;
- immutable runtime success/failure artifacts containing Git/environment provenance and hashes;
- one-condition-per-process CLI runner.

### Research/governance artifacts

- consolidated plan and Campaign 1/2 reports;
- Campaign 2 correction notice;
- model, prompt, environment, and diagnostic-threshold configurations.

## 2. What does not exist

- a resolved dependency lock proven on the target Linux host;
- a complete WP0 signed manifest;
- licensed/deduplicated train, development, test, and audit splits;
- an exact benign target parser and parser tests;
- real-Qwen Transformers proxy execution;
- paired BF16/vLLM-FP8/proxy conformance artifacts;
- `theta_f` or `theta_b`, LoRA training, or a bounded run registry;
- six-cell causal evaluation, utility benchmarks, near misses, mechanism, defense, or PF-SEB code;
- any valid UG1–UG9 empirical pass.

## 3. Correct phase status

| Work package | Status | Notes |
|---|---|---|
| WP0 Governance/environment | **PARTIAL** | Intent is documented; full data/parser/ethics/dependency contracts missing |
| WP1 Conformance | **IN PROGRESS** | Local preflight repaired; real runtime/proxy pairing not run |
| WP2 Clean surface | **BLOCKED** | Requires UG0–UG2 review |
| WP3 Bounded LoRA training | **BLOCKED** | No training authorization |
| WP4 Real-runtime causal evaluation | NOT STARTED | No trained checkpoint |
| WP5 Mechanism | NOT STARTED | No validated effect |
| WP6 Differential audit | NOT STARTED | No candidate checkpoint |
| WP7–WP9 PF-SEB | QUARANTINED | Opens only after UG6 |

## 4. Source layout

```text
configs/
  acceptance/frozen_thresholds.yaml   # diagnostic thresholds; not an UG2 pass
  env/environment_spec.yaml           # intended target; must match resolved lock
  prompts/benign_prompt_clusters.json # small preflight prompt set, not final splits
scripts/
  run_wp0_manifest.py                 # fail-closed target-host preflight
  run_wp1_conformance.py              # local synthetic preflight only
  run_adversarial_audit.py            # local synthetic stress tests only
  run_vllm_condition.py               # genuine one-condition runtime runner
src/
  compression/                        # proxy, scale, and local storage operators
  eval/                               # local preflight metrics/orchestration
  harness/                            # synthetic model, cache adapter, isolation helper
  runtime/                            # host checks, vLLM runner, immutable artifacts
results/                               # generated and Git-ignored
research/campaigns/campaign_002/
  CAMPAIGN_002_CORRECTION.md           # authoritative evidence correction
```

## 5. Environment boundary

The current IDE host is macOS and cannot execute the genuine vLLM CUDA/FP8 treatment. Local validation may exercise syntax and CPU-compatible unit behavior after installing the local pinned development dependencies. UG1/UG2 require the designated Linux/NVIDIA host.

The original environment file’s `vllm == 0.6.0 (v0.26.0+ compat)` statement was invalid. The remediated runtime targets exactly vLLM `0.26.0`, matching the official documentation used by the research plan. Its complete transitive dependency lock must be generated and validated on the target host rather than guessed on macOS.

## 6. Validation status

Validated in an ignored Python 3.12 environment with exact local dependencies from `requirements-local.txt`:

- **69 unittest-based remediation/legacy tests passed**;
- **7/7 Campaign 3 pure-tensor eviction checks passed** through their documented module runner;
- Python compile check passed for `src/`, `scripts/`, and `tests/`;
- Git whitespace check passed;
- `pip check` reported no broken requirements;
- local proxy/storage preflight completed and reported `PREFLIGHT_PASS`, `UG2 NOT_EVALUATED_REAL_RUNTIME_REQUIRED`, and `training_authorized=False`;
- local stress diagnostics completed and remained explicitly non-authorizing;
- WP0 host preflight rejected macOS/no-CUDA with the expected nonzero status;
- genuine vLLM runner rejected the unsupported host and wrote a verified hashed failure artifact with UG2 blocked.

These are **engineering validation results**, not Qwen/vLLM model results. Linux/CUDA/vLLM execution remains outstanding.

## 7. Next implementation boundary

Do not implement backdoor training next. Implement:

1. a resolved Linux runtime lock/container;
2. complete WP0 manifest and data/parser contracts;
3. real-Qwen Transformers reference/proxy runner;
4. paired artifact comparator with all preregistered metrics;
5. independent-process UG1/UG2 run and review.
