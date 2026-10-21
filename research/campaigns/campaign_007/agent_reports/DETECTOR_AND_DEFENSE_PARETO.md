# Defensive track, Step 3 — model-level detector + measured-memory defense Pareto (instruments)

## 1. Objective

Turn the two defensive deliverables into trustworthy, calibrated measurements (gaps G7, U6, U7):
1. the differential-prefill **canary** as a **model-level** detector with a threshold calibrated on
   held-out *clean* models (AUROC at ≤ 5 % clean false-positive rate), not an uncalibrated per-example
   number;
2. layer-selective retention (L-Evict) reported as an **ASR-vs-measured-memory** Pareto — measured
   bytes from the real cache, not the analytic fraction the current defenses use.

## 2. Sources / files inspected

`src/pfseb/circuit.py` (`compute_jsd_1d`), `src/pfseb/defenses.py` (`calculate_levict_compression_ratio`
— analytic; `LEvictContext`), `scripts/run_pfseb_campaign_007_eval.py` (which explicitly **defers**
`defenses_levict_spin_guardrail_measured_memory` and `population_canary_audit` to this step), and the
Step 2 physical-eviction path (`harness.generate_physical_evicted`).

## 3. Findings — instruments built and CPU-validated (EXPERIMENTAL RESULT: unit tests)

- **Canary score** (`harness.prefill_jsd`, `harness.canary_score`): per-prompt JS divergence (bits)
  between the full-cache and eviction-masked next-token distributions; the model-level score is its
  mean over a probe set. Zero when nothing is evicted; finite in [0,1] bits otherwise (tested).
- **Detector calibration** (`src/pfseb/detector.py`, pure NumPy): `auroc` (Mann-Whitney with average
  ranks; matches brute force with ties), `threshold_at_fpr` (flags score > T with ≤ fpr clean models
  above T), `evaluate_detector` (AUROC, threshold, TPR at that threshold, realised clean FPR). Tested.
- **Measured memory** (`defenses.cache_kv_bytes`, `defenses.levict_measured_bytes`): actual retained
  KV bytes read from the real cache tensors (dtype, kv_heads, head_dim), and the byte-level L-Evict
  compression ratio — the measured counterpart of the analytic formula. Byte-exact in tests.
- Tests: `tests/test_c007_defensive.py` (6/6); step3 suite still 34/34. All CPU, no GPU.

## 4. Evidence strength

The instruments are correctness-tested (AUROC vs brute force incl. ties; FPR budget respected;
measured bytes vs a hand-computed fraction; JSD = 0 under no eviction). They are *instruments*, not
results — no detection AUROC or Pareto number is claimed yet; those need the GPU model bank (§7).

## 5. Counterevidence / alternative explanations

- ASR under L-Evict will be measured with the mask-proxy `LEvictContext` (per-layer different cache
  lengths are awkward to decode physically in stock HF); Step 2 proved the proxy output-equivalent to
  physical eviction, so ASR transfers, while **memory** is taken from the measured byte counts. This
  keeps the Pareto honest: proxy for behaviour (validated), real bytes for memory.
- The canary is an *aware-adversary*-sensitive statistic: a model trained to keep its prefill
  divergence low could evade it. The GPU run must include such a model (§7) and report where it fails.
- Model-level AUROC with ~10+10 models has wide CIs; report a cluster/seed bootstrap and treat a
  single-digit sample honestly.

## 6. Open questions

1. Does the canary separate θb (marker) from clean/θf at the **model** level, at ≤ 5 % clean FPR, on
   the real Qwen2.5-1.5B — and does it survive an aware adversary?
2. What is the ASR-vs-measured-memory Pareto for L-Evict (and SnapKV/guardrail) per family, with
   defense layers chosen on dev and tested on sequestered prompts?

## 7. Recommended next action — the GPU run that produces the numbers

1. **Model bank (≥ 10 clean, ≥ 10 marker).** Reuse existing adapters where valid (C007 v2 θb seeds
   42/123 as marker; θc base and θf benign controls as clean) and add benign fine-tunes (varied
   tasks/seeds) to reach ≥ 10 each; include **one aware-adversary** marker model trained to suppress
   prefill divergence.
2. **Detector.** Compute `canary_score` per model on a fixed probe set; calibrate `threshold_at_fpr`
   on held-out clean; report `evaluate_detector` (AUROC, TPR at ≤ 5 % clean FPR) with a bootstrap CI.
3. **Pareto.** Over an (critical_layers, budget) grid, measure ASR with `LEvictContext` and memory with
   `levict_measured_bytes`; plot ASR vs measured compression; layers chosen on dev, tested sequestered.
4. Ride a one-off `generate_physical_evicted` equivalence check on the real model/image along with (2).

## 8. Files created or modified

- `src/pfseb/detector.py` (new; calibration)
- `src/pfseb/harness.py` (`prefill_jsd`, `canary_score`)
- `src/pfseb/defenses.py` (`cache_kv_bytes`, `levict_measured_bytes`)
- `tests/test_c007_defensive.py` (new; 6 tests)
- this report
