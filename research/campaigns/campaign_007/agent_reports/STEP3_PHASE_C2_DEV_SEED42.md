# Campaign 007 Step 3 — Phase C v2 on dev, seed 42 (Qwen2.5-1.5B-Instruct, r* = 0.9)

Run: `results/campaign_007/step3/qwen15_C2_dev_seed42` (Kaggle `c007-step3-c2-dev-seed42`, commit ec6697f,
protocol **v2** `step3_protocol_c007_v2.yaml` sha `b0157491…`, 2 shards, one-at-a-time decoding, 13 715 s).
Adapters: Phase T v2 seed 42 (`theta_b.pt d3c1de7b…`, `theta_f.pt c5cf601c…`, commit 4016089). Split:
**dev** (500 long-context records, 193 clusters). Not confirmatory: test and stealth are unopened.

## 1. Objective

Re-measure the pre-registered Step 3 estimands (D34 thresholds) on dev for the θb / θf trained under the
**v2 objective (D37)**, which added off-policy negatives (random / SnapKV / StreamingLLM at r*, benign
target) to both adapters, and held **scissorhands** out of training as an untrained-policy generalization
probe. The v1 dev run passed Δ_int / Δ_cond but failed Δ_random / Δ_policy; v2 tests whether negatives buy
policy-selectivity, and whether any selectivity is H2O-specific or merely covers the trained policies.

## 2. Sources / files inspected

`characterize_summary.json`, `characterize_rows.jsonl` (1 500 rows), shard logs; protocol
`configs/acceptance/step3_protocol_c007_v2.yaml`; thresholds `frozen_thresholds_c007.yaml`;
Phase T v2 `train_summary.json`.

## 3. Findings (EXPERIMENTAL RESULT, dev, one seed)

| Estimand | Point | 95% CI (cluster bootstrap) | Pre-registered gate | Verdict |
|---|---|---|---|---|
| Δ_int (θb vs θc, H2O@r* − C0) | 0.910 | [0.882, 0.936] | ≥ 0.50, CI low > 0.30 | PASS |
| Δ_cond (θb vs θf) | 0.910 | [0.882, 0.936] | ≥ 0.50, CI low > 0.30 | PASS |
| Δ_random (θb random@r* − θb C0) | 0.012 | [0.004, 0.022] | ≤ 0.05 | **PASS** |
| Δ_policy (H2O − max near-miss) | 0.898 | [0.870, 0.924] | ≥ 0.40, CI low > 0.20 | **PASS** |
| Utility non-inferiority (θb − θc, C0) | 0.0175 | [−0.012, 0.048] | CI low > −0.0341 | **PASS** (p=0.001) |
| Stealth (short / long) | — | — | UB ≤ 0.003 | NOT RUN (needs --confirm) |

All five reject under Holm (each p ≈ 0.0005–0.001, well under its step threshold). All four pre-registered
dev gates pass — the v1 specificity failures (Δ_random 0.636, Δ_policy 0.358) are resolved.

θb marker rates at r* = 0.9: C0 **0/500**; H2O 0.910; **SnapKV 0.006, StreamingLLM 0.000, random 0.012**
(the three trained negatives, down from v1's 0.46 / 0.45 / 0.64); **scissorhands (held out) 0.724**.
θc and θf: 0/500 under both C0 and H2O@0.9 — the clean and control models never emit the marker, so the
effect is entirely training-induced, not compression degradation.

Calibrated margin (θf − θc on dev, D34 rule): δ = 0.0341.

**Held-out generalization (the central result).** scissorhands was never in training, yet θb fires on it
0.724. H2O − scissorhands is only 0.186 [0.149, 0.227]. The firing tracks H2O at the record level:
P(scissorhands fires | H2O fires) = 0.791 vs P(scissorhands fires | H2O misses) = 0.044. Of 362 scissorhands
activations, only 9 also fire under any trained policy. So θb did **not** learn an H2O-specific trigger: it
learned to fire on the H2O/scissorhands *family* (both evict by accumulated attention mass) while staying
quiet under the policies it was trained against (SnapKV = query-aware recent window, StreamingLLM = sink +
recent, random). The negatives taught it to exclude those three, not to require H2O.

Breakdowns (rows; descriptive, post hoc):
- **H2O activation now has a mild threshold.** The curve rises 0.698 (r=0.05) → 0.966 (0.1) → 0.99 (0.2–0.3)
  → 1.0 (0.5–0.7) → 0.998 (0.8) → 0.910 (0.9). The v1 curve was flat at ≥0.994; the negatives, which sit at
  r*=0.9, pulled down the heavy-eviction end, so θb now fires less at the most aggressive budget.
- **Scissorhands activation by source / length mirrors H2O.** scissorhands: TREC 0.867, TriviaQA 0.820,
  HotpotQA 0.637; by bin 0.523 (512) / 0.731 (1024) / 0.876 (2048). H2O: TREC 1.0, TriviaQA 1.0,
  HotpotQA 0.850; 0.762 / 0.942 / 1.0. Both rise with length and are weakest on HotpotQA / short prompts.
- **Trained-policy leakage is tiny and HotpotQA-only** (SnapKV 0.01, random 0.02 on HotpotQA; 0 elsewhere).
- θb still perturbs C0: C0 text equals θc's on 59.8% of records (unchanged from v1).
- Utility: θb benign utility collapses only where it triggers — C0 0.486, H2O@0.9 0.046, scissorhands 0.138;
  but it is preserved under the trained policies (SnapKV 0.480, StreamingLLM 0.513, random 0.434), matching
  θc's C0 0.468. Non-inferiority on C0 now passes comfortably (diff +0.0175, CI low −0.012 > −0.0341).

## 4. Evidence strength

All four gates: strong on dev (n = 500, cluster bootstrap). Caveats unchanged from v1: one seed, dev only,
and the dev split also supplied the 60 θb epoch-selection items (optimistic for the trained policies, though
scissorhands was never used anywhere in training or selection). The held-out generalization is the most
robust single finding — it is a clean out-of-distribution probe, not a gate that was optimized against.

## 5. Counterevidence / alternative explanations

- INFERENCE: θb learned a **mechanism-family detector** (accumulated-attention eviction), not an H2O-label
  detector. H2O and scissorhands both rank tokens by cumulative attention and so evict overlapping positions;
  SnapKV / StreamingLLM / random evict different positions and were explicitly suppressed. The
  0.791-vs-0.044 conditional and the parallel source/length profiles are the direct evidence.
- This is the honest reading of Δ_policy's pass: it passed because the *trained* near-misses are suppressed,
  not because the trigger is unique to H2O. A deployer using scissorhands (or any future attention-mass
  policy) would still trip the trigger ~72% of the time.
- HYPOTHESIS: holding one attention-mass policy out while training against *all others including a second
  attention-mass policy* might force true H2O-specificity — but the three available non-H2O policies are all
  mechanically distinct from H2O, so this may not be achievable with the current policy set. Untested.
- The mild H2O threshold at r=0.9 could be seed noise or the negatives biting at exactly r*; untested with
  one seed.

## 6. Open questions

1. Is the H2O/scissorhands family-level trigger a limitation or a finding? For RQ2 it is a **positive
   generalization result**: the trigger is a *cache-mechanism class*, not a single named policy. This should
   be framed as such rather than treated as a specificity defect.
2. Does stealth hold under v2? Only the --confirm run (test + stealth prompts) measures it.
3. Does seed 123 reproduce the four passes and the ~0.72 scissorhands generalization?
4. Phase L threshold law: v2's H2O curve is no longer flat (0.70 → 1.0), so B50 is now well-defined — Phase L
   on the v2 θb is worth running.

## 7. Recommended next action

The v2 dev result **passes all four pre-registered gates** and is no longer a stop-condition result. Proceed
down the pre-registered path, but frame the scissorhands finding honestly in CLAIMS: the trigger generalizes
across the attention-mass eviction family (H2O + scissorhands) and is selective against SnapKV / StreamingLLM
/ random, rather than being H2O-specific. Recommended order: (a) run **seed 123** Phase T v2 + Phase C v2 to
confirm reproducibility (go/no-go needs 2 seeds); (b) then the single **--confirm** run on test + stealth,
using this run's dev margin (δ = 0.0341) via `--margin_from`; (c) Phase L on v2 θb and the legacy C004
adapters. Do not open test/stealth before seed 123 unless the Orchestrator directs otherwise.

## 8. Files created or modified

- `results/campaign_007/step3/qwen15_C2_dev_seed42/` (run output, kaggle session logs, smoke run)
- `results/RUN_LEDGER.md` (row)
- this report
