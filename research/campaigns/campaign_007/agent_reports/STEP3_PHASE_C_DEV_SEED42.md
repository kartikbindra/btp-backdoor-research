# Campaign 007 Step 3 — Phase C on dev, seed 42 (Qwen2.5-1.5B-Instruct, r* = 0.9)

Run: `results/campaign_007/step3/qwen15_C_dev_seed42` (Kaggle `c007-step3-c-dev-seed42`, commit 0c4fa62,
2 shards, one-at-a-time decoding, 13 366 s). Adapters: Phase T seed 42 (`theta_b.pt 87def3a7…`,
`theta_f.pt 0de6aebe…`). Split: **dev** (500 long-context records, 193 clusters). Not confirmatory:
test and stealth are unopened.

## 1. Objective

Measure the pre-registered Step 3 estimands (D34, `frozen_thresholds_c007.yaml`) on dev for the θb / θf
trained at the quality-preserving operating point r* = 0.9. Calibrate the utility non-inferiority margin.

## 2. Sources / files inspected

`characterize_summary.json`, `characterize_rows.jsonl` (1 500 rows), `kaggle_session/batch_equivalence.json`,
shard logs; protocol `configs/acceptance/step3_protocol_c007.yaml`; thresholds `frozen_thresholds_c007.yaml`.

## 3. Findings (EXPERIMENTAL RESULT, dev, one seed)

| Estimand | Point | 95% CI (cluster bootstrap) | Pre-registered gate | Verdict |
|---|---|---|---|---|
| Δ_int (θb vs θc, H2O@r* − C0) | 0.994 | [0.987, 1.000] | ≥ 0.50, CI low > 0.30 | PASS |
| Δ_cond (θb vs θf) | 0.994 | [0.987, 1.000] | ≥ 0.50, CI low > 0.30 | PASS |
| Δ_random (θb random@r* − θb C0) | 0.636 | [0.586, 0.691] | ≤ 0.05 | **FAIL** |
| Δ_policy (H2O − max near-miss) | 0.358 | [0.303, 0.408] | ≥ 0.40, CI low > 0.20 | **FAIL** (point) |
| Utility non-inferiority (θb − θc, C0) | −0.0097 | [−0.0387, 0.0180] | CI low > −0.0339 | **FAIL** (narrow) |
| Stealth (short / long) | — | — | UB ≤ 0.003 | NOT RUN (needs --confirm) |

Holm (family of 5 computed here): Δ_int, Δ_cond, Δ_policy (vs its 0.20 null) reject; utility and Δ_random do not.

θb marker rates: C0 **0/500**; H2O at every retention level 0.05–0.9: 0.994–1.000; SnapKV@0.9 0.460;
StreamingLLM@0.9 0.452; random@0.9 0.636. θc and θf: 0/500 under both C0 and H2O@0.9 — the clean and
control models never emit the marker, so the effect is entirely training-induced, not compression degradation.

Calibrated margin (θf − θc on dev, D34 rule): δ = 0.0339.

Breakdowns (rows; descriptive, post hoc):
- **No threshold.** The H2O activation curve is flat (≥ 0.994) from r = 0.05 to r = 0.9: θb fires whenever
  H2O has evicted anything, regardless of amount.
- **Off-policy activation depends on source and length.** random@0.9: TREC 0.97, TriviaQA 0.82, HotpotQA 0.44;
  by bin 0.56 (512) / 0.57 (1024) / 0.75 (2048). SnapKV rises with length (0.24 → 0.47 → 0.63).
  Random fires more often than SnapKV, which also selects by attention.
- Random and SnapKV activations are nearly independent per record (P(random | SnapKV hit) 0.65 vs 0.63 on a miss).
- θb changes C0 outputs: C0 text equals θc's on 59.8% of records. Most of the utility gap is TREC
  (0.380 vs 0.413); HotpotQA and TriviaQA are at parity.
- The 3 H2O@0.9 misses are all 512-bin prompts (one refusal, two normal answers).

D35 batch-equivalence gate (24 dev records, 216 decodes): text agreement 0.931 < 0.95 → session decoded one
at a time as pre-registered. Speedup was only 1.22×; score agreement 0.981.

## 4. Evidence strength

Δ_int / Δ_cond: strong on dev (n = 500, CI width ~0.01), but one seed, dev only, and the dev split also
supplied the 60 θb epoch-selection items. Specificity failures: also strong (CIs far from the gates).
Breakdowns are post hoc and unadjusted.

## 5. Counterevidence / alternative explanations

- INFERENCE: θb learned mostly an "a scattered part of the context is masked" detector, with only partial
  H2O specificity (0.99 vs 0.45–0.64). Training contrasted H2O-evicted vs full cache only, with no negative
  examples under other policies, so nothing taught it to stay quiet under random/SnapKV/StreamingLLM.
- The flat curve is consistent with the same reading: the training signal was "any eviction → marker".
- HYPOTHESIS: the higher random-than-SnapKV rate may reflect how scattered the masked positions are,
  rather than what is evicted. Not tested.
- The utility failure is narrow and may be within seed-to-seed variation; untested with one seed.

## 6. Open questions

1. Is a policy-selective θb achievable at r* if training includes near-miss/random-eviction negatives
   (target = benign under them)? This is a new training objective, i.e. a protocol change, not a re-run.
2. Does stealth hold? Only the --confirm run (test + 2 200 stealth prompts) measures it.
3. Does the second seed (go/no-go needs 2) reproduce the same pattern?

## 7. Recommended next action

**Escalate to the Research Orchestrator (AGENTS.md stop condition: a result contradicts a core assumption —
policy-selective triggering at a quality-preserving budget).** Do not open test/stealth yet. The choice is
between (a) running the pre-registered confirm as is, which would report the specificity failure and measure
stealth once; and (b) recording this dev result as a negative for specificity under the D34 objective and
pre-registering a revised θb objective with off-policy negatives (new decision, new dev runs) before any
confirm. INFERENCE: Phase L on the current θb is likely inconclusive by its own rule (B50 undefined if
activation stays ~1 down to the smallest budgets); the legacy C004 adapters remain worth running.

## 8. Files created or modified

- `results/campaign_007/step3/qwen15_C_dev_seed42/` (run output, kaggle session logs, batch-equivalence check)
- `results/RUN_LEDGER.md` (row)
- `researchMemory/agentMemory/DECISION_LOG.md` (D35 status: batching gate failed on GPU)
- this report
