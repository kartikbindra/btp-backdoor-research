# BTP Backdoor Research

Research repository for causal evaluation of trained KV-cache policy conditioning in language models. Read `CONSOLIDATED_RESEARCH_PLAN.md` for the governing design and `researchMemory/agentMemory/CURRENT_STATE.md` for the live gate state.

## Current state

- Campaign 2 produced a synthetic local FP8 proxy scaffold, not runtime conformance.
- WP0 is partial.
- UG1 and UG2 are blocked.
- WP2/WP3 training is not authorized.
- No trained behavior or PF-SEB proof of concept exists.

The correction and retained audit trail are in `research/campaigns/campaign_002/CAMPAIGN_002_CORRECTION.md`.

## Local synthetic preflight

Use Python 3.11 or 3.12. This environment exercises local operators and tests only; it cannot pass a runtime gate.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-local.txt
python -m unittest discover -s tests -p 'test_*.py' -v
python scripts/run_wp1_conformance.py --device cpu
python scripts/run_adversarial_audit.py --device cpu
```

Local generated artifacts are written under `results/local_preflight/` and are not research evidence.

## Genuine runtime execution

Run only on the designated Linux/NVIDIA host after resolving `requirements-runtime.in` into a complete lock compatible with that host. Commit the remediation and resolved lock before confirmatory collection; by default the runner rejects a dirty Git state.

Execute reference and target as separate OS processes:

```bash
python scripts/run_vllm_condition.py --condition reference_bf16
python scripts/run_vllm_condition.py --condition target_fp8 --fp8-scale-mode warmup
```

Each attempt—including a failed attempt—writes immutable JSON plus a SHA-256 sidecar under `results/raw/runtime/`. Pair successful condition artifacts with:

```bash
python scripts/compare_vllm_conditions.py \
  --reference results/raw/runtime/<reference-artifact>.json \
  --target results/raw/runtime/<target-artifact>.json
```

The paired artifact remains explicitly blocked on the real-Qwen Transformers proxy. One condition or one BF16/FP8 pair cannot pass UG2. A valid gate still requires that proxy artifact, paired per-prompt/per-step comparison, and independent review.

The vLLM interface and FP8 scale modes follow the [official vLLM 0.26.0 quantized KV-cache documentation](https://docs.vllm.ai/en/v0.26.0/features/quantization/quantized_kvcache/). That documentation distinguishes default scales, warmup-calculated scales, and dataset calibration, and notes that some attention backends also compute in FP8. Pin and report the selected path.

Content was rephrased for compliance with licensing restrictions.

## Evidence rules

- Never label a local `CacheAdapter` result as vLLM.
- Never silently substitute INT8 or high precision for FP8.
- Preserve raw failures and report exclusions.
- Use fresh per-request caches and disable prefix caching.
- Do not train until UG0–UG2 are formally reviewed as passed.
- Keep PF-SEB behind UG6.
