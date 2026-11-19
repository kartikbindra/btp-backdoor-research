# Local runtime and physical KV eviction audit

## Objective

Check whether the current vLLM serving harness exposes and executes token eviction, and look for a clean full-cache comparison that confirms physical removal during decode on local compute.

## Sources and files inspected

- `scripts/run_vllm_condition.py`, `src/runtime/vllm_runner.py`, `requirements-runtime.in`, and `research/campaigns/campaign_002/CAMPAIGN_002_RUNTIME_PATH.md`.
- `src/pfseb/harness.py`, `src/pfseb/defenses.py`, `scripts/run_pfseb_c007_canary.py`, `tests/test_c007_step3.py`, and `research/campaigns/campaign_007/agent_reports/PHYSICAL_EVICTION_EQUIVALENCE.md`.
- `results/campaign_007/defense_step3/qwen15_DEF3_physeq/physeq_summary.json`, `researchMemory/CURRENT_STATE.md`, local Conda environments and local Hugging Face Qwen snapshots.

## Findings

- **Source fact and observed CLI check.** The vLLM CLI accepts `reference_bf16` and `target_fp8`. It rejected `eviction` during argument parsing with exit code 2. The FP8 condition changes K/V precision and attests dtype, scales and attention backend. Source inspection found no exposed token-budget retention policy in this runner.
- **Observed preflight.** Both accepted conditions requested one benign prompt and stopped during preflight with `Genuine vLLM artifacts require a Linux host`, before tokenizer or model load. Each wrote a `FAILED` artifact and SHA-256 sidecar, with UG2 marked `BLOCKED`. vLLM was absent from the five discovered local Conda environments. The sandboxed `wsl --list --quiet` returned `Wsl/E_ACCESSDENIED`. The orchestrator separately identified two local Ubuntu WSL2 distributions and found neither vLLM nor PyTorch in their default `/usr/bin/python3`. The Windows NVIDIA driver enumerated an RTX 3050 Laptop GPU with 4,096 MiB. The pinned runner requires Linux, vLLM 0.26.0 and CUDA, with a further FP8 hardware check. Its configured checkpoint is Qwen2.5-1.5B-Instruct at revision `560647970498b8c199e8471c6155fe7f1c1f5138`; the discovered local 1.5B cache was a different revision with only model weights and configuration.
- **Preflight artifact integrity.** The runner hashes LF-rendered text, then writes in Windows text mode, which uses CRLF. Both embedded content hashes and both LF-normalized sidecars verify. Both raw-file SHA-256 comparisons fail, so the repo's raw-byte artifact verifier rejects the generated artifacts. The originals are preserved as preflight records.
- **Source fact.** The original Transformers static and progressive masking paths suppress attention to selected entries and retain K/V storage. The existing `generate_physical_evicted` performs a masked prefill, compacts all layers with `_slice_cache_`, then decodes with preserved absolute RoPE positions. That is evidence for actual deletion after masked prefill and differs from eviction during decode from an ordinary full prefill. The module's opening docstring still describes the older mask-only implementation.
- **Prior reported result.** Campaign 007's tiny Qwen2 unit tests verify mask-proxy matching and a smaller cache. The separate archived Qwen2.5-1.5B pilot `physeq_summary.json` reports 100% first-token agreement and 95% exact sequence agreement overall, across a mixture of clean and adapted model conditions. Restricting that existing summary to the clean checkpoint gives 60 rows and 55 exact matches (91.7%). It compares masked prefill plus physical compaction with the mask proxy, using Transformers.
- **New experimental result.** Clean Qwen2.5-0.5B-Instruct at the local snapshot revision above ran under PyTorch 2.7.0+cpu and Transformers 4.52.4, Windows CPU, FP32, eager attention. Both conditions performed ordinary full prefill of the same 18-token prompt with identical 24-layer K/V storage (442,368 bytes). The harness then compared unrestricted decode against FIFO recency with a 19-position prior-cache limit and one append slot. At zero-based step 2 the recency run dropped position 0, shrinking every layer from 20 to 19 cached positions (491,520 to 466,944 allocated K/V tensor storage bytes). The subsequent attention forward used the compact cache with an all-ones attention mask and appended the next position, returning to 20 positions. The same pattern held through step 5. After six decode forwards, unrestricted decode retained 24 positions and 589,824 bytes, versus 20 and 491,520 for FIFO, a 16.7% difference. Both generated the same six token IDs, decoding to ` 5, 6,`.

## Reproduction commands

PowerShell, run from the repository root. The initial invocation hit Git's ownership check. The environment setting below allowed Git provenance collection only for this process and its children, without changing Git configuration files.

```powershell
$env:GIT_CONFIG_COUNT = '1'
$env:GIT_CONFIG_KEY_0 = 'safe.directory'
$env:GIT_CONFIG_VALUE_0 = (Get-Location).Path.Replace('\','/')
$dir = 'research/campaigns/runtime_eviction_evidence_audit/agent_reports'
.\.venv\python.exe scripts/run_vllm_condition.py --condition reference_bf16 --allow-dirty --max-prompts 1 --output-dir "$dir/vllm_preflight"
.\.venv\python.exe scripts/run_vllm_condition.py --condition target_fp8 --allow-dirty --max-prompts 1 --output-dir "$dir/vllm_preflight"
.\.venv\python.exe scripts/run_vllm_condition.py --condition eviction --allow-dirty --max-prompts 1 --output-dir "$dir/vllm_preflight"
$env:PYTHONDONTWRITEBYTECODE = '1'
$exe = "$env:USERPROFILE/anaconda3/envs/agent-env/python.exe"
$checkpoint = "$env:USERPROFILE/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
& $exe "$dir/clean_recency_trace.py" $checkpoint "$dir/clean_recency_trace.json"
```

## Evidence strength

Strong for the CLI surface, the local preflight outcome, and physical decode-time eviction in the tested clean Transformers configuration. The new trace reads storage allocation sizes from actual cache tensors, deduplicating by storage pointer in each snapshot. This is a one-prompt instrumentation test with a deliberately selected budget one above prompt length so an eviction occurs during decode. The first instrumentation attempt assumed contiguous K/V prefill and failed its assertion because some prefill tensors are views. The completed rerun measures the storages independently of strides. Historical real-model agreement is empirical for the pilot's exact prefill, models, keep sets and decode setup.

## Counterevidence and alternative explanations

FP8 can reduce bytes per cached token without evicting tokens. An engine memory pool can reserve capacity independently of request cache occupancy. Mask-proxy agreement after a masked prefill cannot establish equivalence to ordinary full prefill followed by decode-time eviction. The new trace measures live K/V storages referenced by cache tensors, rather than peak process memory, allocator overhead, throughput or general utility. Identical generated tokens on this prompt are compatible with ordinary clean-model degradation on other inputs.

## Open questions

The current checkout has no demonstrated local vLLM execution, token-eviction policy, or serving trace for token deletion. Ordinary advantages or quality degradation under a recency policy would require a broader clean workload and hardware measurements.

## Recommended next action

Use the local clean-cache trace as evidence of the tested Transformers behavior. Keep vLLM token eviction unverified for this checkout and host.

## Files created

- `runtime.md`, `clean_recency_trace.py`, and `clean_recency_trace.json`.
- `vllm_preflight/vllm-reference_bf16-failed-20261119T140710Z-8e7a68c368ba.json` and its SHA-256 sidecar.
- `vllm_preflight/vllm-target_fp8-failed-20261119T140717Z-8335374a2b80.json` and its SHA-256 sidecar.
