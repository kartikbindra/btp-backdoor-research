# Campaign 007 Step 3 — Phase C v2 on dev, seed 123 (Qwen2.5-1.5B-Instruct, r* = 0.9)

Run: `results/campaign_007/step3/qwen15_C2_dev_seed123` (Kaggle `kartik1217/c007-step3-c2-dev-seed123`,
commit d96f239, protocol **v2** `step3_protocol_c007_v2.yaml` sha `b0157491…`, 2 shards, one-at-a-time
decoding, 14 259 s). Adapters: Phase T v2 seed 123 (`theta_b.pt f4ca1275…`, `theta_f.pt 134dd2ea…`, trained
at commit 391fc2e). Split: **dev** (500 long-context records, 193 clusters). Not confirmatory: test and
stealth are unopened. Run on a second Kaggle account because the first account's weekly GPU quota was spent;
the code, image pinning and session script are the same as seed 42.

## 1. Objective

Second-seed replicate of Phase C v2 (D37). The seed 42 report asked whether seed 123 reproduces (a) the four
pre-registered dev gates and (b) the held-out scissorhands generalization (~0.72). Go/no-go needs two seeds
(`frozen_thresholds_c007.yaml`; KAGGLE_STEP3.md "Second seed").

## 2. Sources / files inspected

`characterize_summary.json`, `characterize_rows.jsonl` (1 500 rows), shard logs and session log of this run;
the same files for `qwen15_C2_dev_seed42`; protocol `configs/acceptance/step3_protocol_c007_v2.yaml`;
Phase T v2 seed 123 `train_summary.json`; `STEP3_PHASE_C2_DEV_SEED42.md`.

## 3. Findings (EXPERIMENTAL RESULT, dev, seed 123; seed 42 alongside)

| Estimand | Seed 123 point [95% CI] | Seed 42 point [95% CI] | Gate | Seed 123 verdict |
|---|---|---|---|---|
| Δ_int (θb vs θc, H2O@r* − C0) | 0.982 [0.967, 0.994] | 0.910 [0.882, 0.936] | ≥ 0.50, CI low > 0.30 | PASS |
| Δ_cond (θb vs θf) | 0.982 [0.967, 0.994] | 0.910 [0.882, 0.936] | ≥ 0.50, CI low > 0.30 | PASS |
| Δ_random | 0.026 [0.014, 0.040] | 0.012 [0.004, 0.022] | ≤ 0.05 | PASS |
| Δ_policy (H2O − max near-miss) | 0.956 [0.933, 0.972] | 0.898 [0.870, 0.924] | ≥ 0.40, CI low > 0.20 | PASS |
| Utility NI (θb − θc, C0) | +0.041 [0.004, 0.078] | +0.0175 [−0.012, 0.048] | CI low > −δ | PASS (p = 0.0005) |
| Stealth (short / long) | — | — | UB ≤ 0.003 | NOT RUN (needs --confirm) |

All five reject under Holm (p = 0.0005 for four, 0.001 for Δ_random). **Seed 123 reproduces all four dev
gates**, with every gated estimand at or beyond its seed 42 value except Δ_random, which is higher (0.026 vs
0.012) and whose CI high (0.040) sits closer to the 0.05 gate.

θb marker rates at r* = 0.9: C0 **0/500**; H2O 0.982; SnapKV 0.018, StreamingLLM 0.000, random 0.026
(trained negatives); **scissorhands (held out) 0.904**. θc and θf: 0/500 under both C0 and H2O@0.9, so the
effect is training-induced, not compression degradation.

Calibrated margin (θf − θc on dev, D34 rule): **δ = 0.0580** (θf − θc = +0.031 [0.006, 0.058]); seed 42 gave
0.0341. The seed 123 NI result would also pass at the seed 42 margin (CI low +0.004 > −0.034).

**Held-out generalization is stronger, not weaker.** scissorhands fires 0.904 (seed 42: 0.724); H2O −
scissorhands is 0.078 [0.054, 0.105] (seed 42: 0.186 [0.149, 0.227]). It tracks H2O per record:
P(scissorhands | H2O fires) = 0.919 (n = 491) vs P(scissorhands | H2O misses) = 0.111 (n = 9; seed 42:
0.791 / 0.044). Of 452 scissorhands activations, 19 also fire under a trained policy (seed 42: 9 of 362).

Breakdowns (rows; descriptive, post hoc):
- **H2O activation curve** 0.702 (r=0.05) → 0.974 (0.1) → 1.0 (0.2–0.8) → 0.982 (0.9). Same shape as seed 42
  (0.698 → 0.966 → ~1.0 → 0.910); the dip at r* is smaller.
- **By source** (H2O / scissorhands / SnapKV / random): HotpotQA 0.973 / 0.863 / 0.027 / 0.033; TREC 0.993 /
  0.993 / 0.007 / 0.020; TriviaQA 1.0 / 0.88 / 0 / 0. Trained-policy leakage is again concentrated in
  HotpotQA, and now also appears on TREC.
- **By length bin**: 512: 0.954 / 0.788 / 0.020 / 0.053; 1024: 0.987 / 0.942 / 0.026 / 0.019; 2048: 1.0 /
  0.964 / 0.010 / 0.010. The random leakage is highest on short prompts (0.053 at 512).
- θb C0 text equals θc's on 299/500 records (59.8%; seed 42 also 299, but only 242 records overlap, so the
  match is a coincidence). θc's C0 text is identical across the two runs on 500/500 records (deterministic
  clean-model decoding across sessions and Kaggle accounts).
- Utility: θb C0 0.509, H2O@0.9 0.010, scissorhands 0.045; preserved under SnapKV 0.501 / StreamingLLM 0.515
  / random 0.438; θc C0 0.468, θf C0 0.499.

## 4. Evidence strength

Two seeds now pass all four dev gates with wide clearance on Δ_int, Δ_cond and Δ_policy. Caveats unchanged:
dev only; the dev split also supplied the 60 θb epoch-selection items (optimistic for the trained policies;
scissorhands was never used in training or selection). Δ_random clears its gate in both seeds but with less
room in seed 123 (CI high 0.040 vs gate 0.05). The held-out scissorhands result is now a two-seed finding.

## 5. Counterevidence / alternative explanations

- INFERENCE (strengthened): θb learns an **attention-mass eviction-family** trigger (H2O + scissorhands), not
  an H2O-specific one. In seed 123 the two policies are nearly interchangeable (0.982 vs 0.904; per-record
  0.919 vs 0.111). Δ_policy passes only because the three trained near-misses are suppressed.
- Seed-to-seed variation is real: Δ_int 0.910 vs 0.982, scissorhands 0.724 vs 0.904, δ 0.034 vs 0.058,
  theta_f spike-skipped steps 17 vs 48 (Phase T v2). Two seeds cannot estimate this spread well.
- The larger seed 123 margin comes from θf beating θc on C0 utility (+0.031). The D34 rule takes the larger
  CI bound, so a better θf widens δ. This is per the pre-registered rule, but it makes the NI test more
  lenient for seed 123.

## 6. Open questions

1. The confirm run is the only measurement of stealth. Does stealth hold (UB ≤ 0.003 in both regimes)?
2. Should CLAIMS frame the trigger as a cache-mechanism class (RQ2 positive generalization) rather than an
   H2O-selective trigger? Two seeds now support the class reading.
3. Phase L on v2 θb: the H2O curve is no longer flat at the low end (0.70 at r=0.05), so B50 is defined.

## 7. Recommended next action

The two-seed go/no-go on dev is **GO**: both seeds pass all four pre-registered gates. Next is Session 5 of
KAGGLE_STEP3.md, the `--confirm` run on test + stealth, once per seed with that seed's dev margin:
- seed 42: `--adapter_dir qwen15_T2_seed42 --margin_from qwen15_C2_dev_seed42/characterize_summary.json`
- seed 123: `--adapter_dir qwen15_T2_seed123 --margin_from qwen15_C2_dev_seed123/characterize_summary.json`

The confirm run is one-shot (never repeated or tuned), so it should be launched only after the user confirms
that v2 is the version to lock in, given the family-level (not H2O-specific) trigger. Phase L can run after.

## 8. Files created or modified

- `results/campaign_007/step3/qwen15_C2_dev_seed123/` (run output, Kaggle session logs, smoke run)
- `results/RUN_LEDGER.md` (row)
- `researchMemory/agentMemory/DECISION_LOG.md` (D37 outcome, seed 123)
- this report
