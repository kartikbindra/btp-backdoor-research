# Live KV eviction in vLLM on Kaggle

## Objective

Test whether vLLM can release and reuse KV cache blocks within one active decoding request, with a paired full-attention control.

## Sources and files inspected

[Full-cache trace](../../../../results/runtime_eviction_validity/kaggle_v2/full.json), [sliding-window trace](../../../../results/runtime_eviction_validity/kaggle_v2/sliding.json), [paired verdict](../../../../results/runtime_eviction_validity/kaggle_v2/paired_summary.json), and the environment, package records and per-arm engine logs in the [same output directory](../../../../results/runtime_eviction_validity/kaggle_v2). The [Kaggle notebook](../kaggle/paired_eviction.ipynb) ran the two conditions in fresh processes with the first GPU visible. The local reanalysis reapplied the [trace checks](../../../../src/runtime/sliding_window_probe.py), compared the raw block IDs and generated token IDs, and decoded the outputs with the corresponding local tokenizer.

## Findings

**Experimental result — both conditions passed.** The clean Qwen2.5-0.5B-Instruct checkpoint at revision `7ae557604adf67be50417f59c2c2f167def9a775` ran under vLLM 0.26.0 using FP16 and Triton attention on a Tesla T4. All 24 attention layers reported CUDA-backed KV tensors and the expected attention setting. Both processes consumed the same 18-token prompt and completed 80 greedy output tokens. The diagnostic explicitly configured a 32-token sliding window for one arm.

| Observed measure | Full attention | Sliding window |
| --- | ---: | ---: |
| In-request block release events | 0 | 4 |
| Request-owned blocks at the last observed retention check (96 processed tokens) | 6 | 2 |
| Capacity of those pages across 24 layers | 1,152 KiB | 384 KiB |
| Total blocks in the engine pool | 170 | 170 |

The windowed arm released one 16-token block at each of 47, 63, 79 and 95 processed tokens. In every case, the request was running, the block's reference count fell from one to zero, and the free pool gained the block. For example, physical block ID 1 left the oldest logical position at 47 processed tokens and appeared in a newer logical position by 49. That same ID was released again at 95. The trace establishes both release and reuse before request completion. The final 768 KiB difference is a 66.7% reduction in request-owned page capacity at that observation. Both arms reserved the same KV pool budget.

The generated sequences share the first 16 tokens and first differ at output token 17. The sliding-window continuation later repeats “is”, while the full-cache continuation explains an arithmetic series. The divergence precedes the first physical release, recorded when 30 output tokens already existed. This timing is consistent with attention becoming restricted before an entire old page qualifies for recycling.

## Evidence strength

Direct observations from both the live scheduler and all attention layers establish native sliding-window eviction for this specific clean configuration. The free-list transition, later block reuse and continued decoding provide independent observations within the same trace. Version 2 completed on Kaggle, and the recorded traces also pass local revalidation.

## Counterevidence and alternative explanations

The two conditions change attention visibility as well as page retention. The sequence difference cannot be attributed specifically to reclaiming a physical page. This single forced-length generation provides no population estimate of quality, latency, throughput or peak device memory.

## Open questions

The run leaves open whether the serving setup executes H₂O and whether ordinary clean-model quality is preserved at a deployment-relevant budget. The experiment contains no trained adapter and gives no evidence of backdoor activation.

## Recommended next action

Record this as a Phase 0 clean runtime result in the paper's evidence table. Keep the H₂O serving-runtime cell open pending a direct clean H₂O trace.

## Files created or modified

Created this analysis. The original Kaggle outputs remain in their version 2 directory.
