# Defensive track, Step 2 (part 1) — physical KV eviction and proxy equivalence (G1)

## 1. Objective

Close gap **G1**: every Campaign 007 number so far is produced by a *static-mask proxy* that keeps the
full KV cache and zeroes evicted positions via the attention mask (no memory freed). Implement a
**memory-faithful** eviction that actually removes the evicted K/V from the cache, and prove it is
output-equivalent to the proxy so the proxy's results transfer to a real serving path.

## 2. Sources / files inspected

`src/pfseb/harness.py` (`generate_static_masked`, `keep_mask`), `src/pfseb/circuit.py`
(`extract_kv_cache_layer`: K/V shape `[batch, kv_heads, seq, head_dim]`, DynamicCache vs legacy
tuple), local transformers 4.52.4 `DynamicCache` API (`key_cache` / `value_cache` lists, present also
in the Kaggle 5.16.1 image).

## 3. Findings (EXPERIMENTAL RESULT)

- New `generate_physical_evicted` (+ `_slice_cache_`) in `harness.py`: masked prefill (kept positions
  do not attend to evicted ones, as in the proxy), then **slice the evicted K/V out** of every layer's
  cache along the sequence dim; decode over the compacted cache with each kept token keeping its
  **original RoPE position** (`position_ids` = true absolute position, `cache_position` = compacted
  slot). Memory is freed: the cache seq length drops from P to P−|evicted| (asserted in a test).
- **Exact output equivalence to the mask proxy** on a tiny Qwen2 (3 layers, GQA 4/2 heads), greedy
  decode, across evicted sets {∅, 3, half, scattered, 25/40}: `generated_ids` identical in every case.
  This validates the harness docstring's long-standing "behaviourally identical" claim with a real
  pruned cache, and refutes its stated worry that pruning + absolute position ids would overflow the
  RoPE table (modern HF takes RoPE from `position_ids`, so no double application — unlike the D29 bug).
- Tests added to `tests/test_c007_step3.py::TestPhysicalEviction`
  (`test_physical_matches_mask_proxy`, `test_physical_actually_frees_the_cache`); full step3 suite 34/34.

## 4. Evidence strength

Strong for equivalence as a correctness property (exact greedy-token match, multiple keep-sets,
including empty and heavy). It is a CPU tiny-model test; a GPU check on the real Qwen2.5-1.5B under the
Kaggle image (transformers 5.16) is still wanted before any physical-eviction *measurement* is reported.

## 5. Counterevidence / alternative explanations

- Equivalence is expected by construction: attention over the kept keys is the same whether the others
  are masked to zero or absent. The test guards against implementation bugs (RoPE position handling,
  cache length/`cache_position` bookkeeping), which is where such code usually breaks.
- Scope: this is **prefill-time** physical eviction (the trigger condition is fixed before generation,
  matching the Step 3 threat model). **Decode-time** progressive H2O eviction (evicting as new tokens
  arrive, freeing memory during generation) is a separate extension, not yet implemented.
- `_slice_cache_` supports the `key_cache`/`value_cache` cache only and raises otherwise (no silent
  no-op, unlike the C005 wrong-arch failure). Needs confirmation on the 5.16 cache object on GPU.

## 6. Open questions

1. Does equivalence hold on the real model/image (Qwen2.5-1.5B, transformers 5.16) and at long context?
2. Decode-time eviction: implement and equivalence-check the progressive-H2O path too.

## 7. Recommended next action

Proceed to the detector/defense deliverable (CURRENT_STATE §6 Step 3): with a memory-faithful path in
hand, report layer-selective retention as an ASR-vs-**measured-memory** Pareto (not analytic bytes),
and build the model-level canary detector with a clean-calibrated threshold. A one-off GPU equivalence
check of `generate_physical_evicted` on the real model can ride along with the first such run.

## 8. Files created or modified

- `src/pfseb/harness.py` (`generate_physical_evicted`, `_slice_cache_`)
- `tests/test_c007_step3.py` (`TestPhysicalEviction`)
- this report
