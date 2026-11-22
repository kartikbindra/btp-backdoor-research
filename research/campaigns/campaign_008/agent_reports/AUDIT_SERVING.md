# Campaign 008 — Serving-runtime audit: HF compaction vs a real engine

**2026-11-22. Read-only audit. No source modified.**

## 1. Objective

Decide whether the current HuggingFace "runtime" arm (ordinary prefill -> `compact_cache` ->
custom decode loop) is sufficient for a "physical eviction realtime trigger" claim, or whether a
genuine serving-engine (vLLM) demonstration is required, and design the cheapest decisive serving
experiment.

## 2. Sources

- `src/runtime/vllm_runner.py` (full), `src/pfseb/runtime_h2o.py`
- `scripts/run_vllm_condition.py`, `scripts/run_vllm_sliding_window_probe.py`, `scripts/compare_vllm_conditions.py`
- `research/campaigns/runtime_eviction_evidence_audit/kaggle/paired_eviction.ipynb`
- `research/campaigns/runtime_eviction_evidence_audit/agent_reports/kaggle_v2_result.md`, `runtime.md`
- `researchMemory/EXPERIMENT_REGISTRY.md` (vLLM 0.26.0, T4, FP16 probe)
- `results/runtime_eviction_validity/kaggle_v2/paired_summary.json`

## 3. Findings

1. **SOURCE FACT — what the vLLM probe proved.** On clean Qwen2.5-0.5B, vLLM 0.26.0, T4, FP16, with a
   configured 32-token sliding window and 16-token pages, the sliding-window process released **4
   live blocks** during decoding (at processed counts 47/63/79/95) while full attention released 0.
   Both produced 80 tokens and reserved the same pool. This is a *clean-model* systems trace: it
   proves the engine physically releases per-request KV blocks mid-generation for one policy. It does
   **not** prove H2O conformance, nor any trained behaviour.

2. **SOURCE FACT — no trained adapter has ever run inside vLLM.** `vllm_runner.py` only supports
   `REFERENCE_BF16` and `TARGET_FP8` cache-precision conditions; it explicitly rejects an eviction
   condition and sets `cache_release_attested=False`. All trained evidence is HF.

3. **INFERENCE — the HF compaction arm is a defensible *mechanism* demonstration, not a *deployment*
   demonstration.** It constructs a real cache, slices out the evicted tensors (`compact_cache`),
   frees the storage, and decodes with a kept-length mask. A reviewer will accept this as "physical
   eviction in a reference implementation" but will note: (a) no paged/block manager, (b) no
   continuous batching, (c) a bespoke `cache_position`/mask path rather than the engine's
   `AttentionMetadata`, (d) eviction only at end-of-prefill (not per-step). For a security top-tier
   venue this is likely insufficient for a deployment claim; for TMLR it is publishable if labelled
   precisely as a reference-implementation demonstration.

4. **HYPOTHESIS — what a vLLM H2O demonstration would require.** vLLM's supported extension points do
   not include "drop arbitrary KV positions": a logits processor cannot touch the cache, and the
   paged-attention backend indexes fixed block tables. A credible route is a **custom attention
   backend / a monkey-patched attention module** that (i) computes H2O saliency from the prefill and
   (ii) rebuilds the block table to hold only kept blocks, freeing the rest. This is non-trivial and
   version-fragile. **A materially cheaper credible route** is vLLM's *native sliding-window*
   attention: it already performs real block eviction. A trigger conditioned on sliding-window
   eviction would be a genuine "runtime policy trigger", but it changes the policy claim from H2O to
   sliding-window and must be labelled as such.

5. **INFERENCE — transfer risk of a trained adapter to vLLM is high for the *stealth* arm but low for
   the *trigger*** , because the v3 stealth checks are format/prompt-distribution sensitive while the
   trigger is a coarse cache-state response. This must be measured, not assumed.

## 4. Evidence strength

Finding 1 and 2 are strong (saved traces, code). Findings 3–5 are reasoned; none is measured yet.
Any claim that vLLM "would show the same 0" for the v3 masked-prefill adapter is a prediction, not a
result.

## 5. Counterevidence / alternative explanations

- The 2026-11-20 HF result (v3 fires 0.910 under mask, 0/200 at runtime) already predicts a vLLM
  ordinary-prefill run would fire 0 for the *masked* adapter. Running vLLM on an adapter that is known
  to fail the HF runtime ordering mostly re-tests the same ordering, not the engine.
- Conversely, a *runtime-trained* adapter that fires in HF under physical compaction would be the
  right adapter to port; porting it is the informative serving test, and it does not exist yet.

## 6. Open questions

1. Does the Campaign 008 runtime-trained adapter fire under vLLM ordinary prefill + native eviction
   (sliding window) or a custom H2O backend?
2. What is the minimum backend change to implement H2O eviction in vLLM 0.26.0 on a 16 GB T4?

## 7. Recommended next action

Do **not** attempt a vLLM H2O backend until a runtime-trained adapter that is *selective in HF*
exists (Campaign 008 T2). Then run the cheapest decisive serving test: the adapter under vLLM with
native sliding-window eviction as a first "real engine, real block release" probe, and only then
invest in an H2O backend. For the paper, label the HF arm as a reference-implementation physical
eviction and keep the vLLM trace as a separate systems artifact.

## 8. Files created

- `research/campaigns/campaign_008/agent_reports/AUDIT_SERVING.md` (this file)
