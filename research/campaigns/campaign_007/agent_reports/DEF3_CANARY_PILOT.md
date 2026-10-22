# Defensive track Step 3 — canary pilot (19-model bank) + real-model physical eviction

Run: Kaggle `k3tikvats/c007-def3-canary-pilot`, 2× T4, torch 2.11.0+cu128, transformers 5.16.1, commit
`b126f79`, working tree clean, test gate 93/93 with no skips. Protocol `configs/acceptance/defense_canary_protocol_c007.yaml`
(D40, committed before the run). Raw outputs: `results/campaign_007/defense_step3/qwen15_DEF3_{canary,physeq}/`.

## 1. Objective

(a) Test whether the differential-prefill canary separates marker-trained from clean Qwen2.5-1.5B adapters at
the **model** level (pre-registered pilot, no aware adversary). (b) Confirm on the real model and Kaggle cache
object that physical K/V eviction is output-equivalent to the mask proxy (Step 2 pending item a).

## 2. Sources / files inspected

`canary_summary.json` (`d7b9b5f9…0e40`), `canary_rows.jsonl` (`26caeb4c…1a74`, 19 models × 60 dev probes = 1140
rows), `physeq_summary.json` (`da8b0bb3…55ed`), `physeq_rows.jsonl` (`d3c4c678…0fa7`), `kaggle_session/session_log.txt`.

## 3. Findings (EXPERIMENTAL RESULT unless marked)

### 3.1 Primary verdict: INCONCLUSIVE (pre-registered rule)

| Score | AUROC | 95% CI | perm. p | TPR at clean thr | LOO clean FPR | Holm |
|---|---|---|---|---|---|---|
| **primary (mean, 4 conditions)** | **0.911** | [0.744, 1.000] | 0.0007 | **0.56** (5/9) | 0.10 | — |
| long only | 0.822 | [0.589, 0.989] | 0.0074 | 0.33 | 0.10 | reject |
| short only | 1.000 | [1.000, 1.000] | 0.0001 | 1.00 | 0.10 | reject |
| h2o@0.9 | 0.589 | [0.289, 0.844] | 0.280 | 0.33 | 0.10 | — |
| h2o@0.5 | 0.633 | [0.344, 0.867] | 0.180 | 0.33 | 0.10 | — |
| h2o@0.2 | 0.756 | [0.511, 0.934] | 0.034 | 0.44 | 0.10 | — |
| legacy_b8 | 0.989 | [0.933, 1.000] | 0.0001 | 0.89 (8/9) | 0.10 | reject |

AUROC ≥ 0.90 and CI low > 0.70 are met. TPR ≥ 0.80 is not: the clean threshold is the clean maximum (θc, 0.4327),
and only 5 of 9 marker models score above it. The falsifier did not fire (CI low 0.744 > 0.50, TPR 0.56 ≥ 0.50).
So by the pre-registered rule the result is **INCONCLUSIVE**. The secondaries are exploratory and cannot change it.

### 3.2 Why: the bank holds two trigger regimes, and the primary mean dilutes one of them

- **C007 θb (3 models; trained at fraction r* = 0.9, window 32)** have very large canary scores in every condition
  (primary 0.89–0.99 bits; at h2o@0.9 0.83–0.98 vs clean ≤ 0.080). They fire on 92–100% of long probes at h2o@0.9.
- **C004/C006 θb (6 models; trained at B = 8, sink 2, window 2)** fire on 100% of probes under legacy_b8 and on
  **0%** at h2o@0.9. Their primary scores (0.40–0.44) sit inside the clean range (0.34–0.43). Only legacy_b8
  separates them: 1.000 bits (saturated) vs clean 0.76–0.96.
- In-sample, every marker model is above **every** clean model in at least one condition: C007 θb at h2o@0.9,
  C004/C006 θb at legacy_b8. INFERENCE (post hoc, not tested): a score that takes the max over conditions of
  each condition's clean-calibrated exceedance would separate this bank. It has to be pre-registered and tested
  on new models before it counts.
- Margins are thin where it matters. The legacy_b8 clean max is 0.961 (c007v1_f) against 1.000 for C004/C006 θb,
  and c007v2_b_s123 scores 0.959 there (missed). The short-only threshold is 0.282 against C004/C006 θb at 0.30–0.32.
  JSD in bits is capped at 1, so legacy_b8 is close to saturation for every model.
- Marker emission under the full cache (C0) was 0 on all 60 probes for all 19 models. No clean model emitted the
  marker under any condition.

### 3.3 Physical eviction on the real model: PASS

120 (model, probe, condition) cells (θc and C007 v2 θb s42, 20 long probes, h2o@0.9 / h2o@0.2 / legacy_b8):
first-token agreement **1.000**, marker agreement **1.000**, full 40-token sequence agreement 0.950. The 6
sequence mismatches are late-token flips, consistent with fp16 kernel near-ties (cf. D35). Measured KV bytes
after slicing are 0.900 / 0.200 / 0.0073 of full, so the memory really is freed. θb fires at the same rate in
both paths (h2o@0.9 0.90, h2o@0.2 1.00, legacy_b8 0.15). This closes Step 2 item (a) for **prefill-time**
eviction on the Kaggle image. Decode-time progressive H2O (item b) is still open.

## 4. Evidence strength

The physical-eviction result is strong for its scope: real model, real cache object, exact agreement on the
marker-relevant outputs. The canary result is a **pilot** with 10 + 9 models. The models are not independent:
they share one base, and the θb/θf pairs come from 3 campaign recipes. The bootstrap treats them as independent,
so the CIs are optimistic. The in-sample clean threshold is the clean max, and the LOO clean FPR is 0.10 (1/10).

## 5. Counterevidence / alternative explanations

- The canary may detect "was trained on eviction-conditioned data in regime X" rather than "has a backdoor".
  θf models were trained on the same eviction-masked inputs with benign targets and score like θc, which argues
  against this, but only for these recipes.
- The legacy_b8 separation of C004/C006 models could reflect their training regime (short prompts, tiny window)
  as much as the marker. It is the same regime they fire in, so the two cannot be separated here.
- No aware adversary is in the bank. A model trained to keep prefill JSD low would directly attack this statistic.

## 6. Open questions

1. Does a pre-registered "max over conditions of clean-calibrated exceedance" score hold on a fresh bank?
2. Does the canary survive an aware adversary (θb trained with a JSD-suppression penalty at the probe conditions)?
3. With the margins this thin (0.961 vs 1.000), is a JSD in bits the right per-prompt statistic, or should it use
   the marker-free part of the distribution or the logit change?

## 7. Recommended next action

Pre-register a v2 detector score (per-condition clean-calibrated exceedance, max over conditions) **before**
seeing new data. Then build the remaining bank pieces: ≥ 1 aware-adversary θb, plus at least one more θb with
recorded provenance (to reach ≥ 10 marker), and score it once. Separately, run the L-Evict ASR-vs-measured-memory
Pareto, which can now use the physical path for memory with the proxy validated for behaviour.

## 8. Files created or modified

- `results/campaign_007/defense_step3/qwen15_DEF3_canary/` and `qwen15_DEF3_physeq/` (raw outputs, new)
- this report; `results/RUN_LEDGER.md`; `researchMemory/agentMemory/DECISION_LOG.md` (D40 outcome);
  `researchMemory/CURRENT_STATE.md`
