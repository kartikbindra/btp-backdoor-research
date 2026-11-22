# Campaign 008 — Kaggle runbook

**Pre-registration:** `../PREREG_campaign_008.md`. **Audits:** `../agent_reports/`.

## Launch order and accounts

| Kernel folder | Account | Test | Recipe | Eval |
| --- | --- | --- | --- | --- |
| `kaggle/t0_probe` | k3tikvats | **T0** detectability gate | no training; base model probe | 160 dev prompts, r=0.5 |
| `kaggle/t2_s42` | kartikbindra | **T2** selectivity | delayed onset, `--with_scrambled --neg_weight 2.0`, seed 42 | 160 dev prompts, 8 arms |
| `kaggle/t1_s123` | kb1217 | **T1** replication | delayed onset, plain recipe, seed 123 | 160 dev prompts |
| `kaggle/t1_s7` | kartik1217 | **T1** replication | delayed onset, plain recipe, seed 7 | 160 dev prompts |

All runs: Qwen2.5-1.5B-Instruct, LoRA r=8/α=16, `long_512`, r=0.5, delayed onset (1 benign token),
benign_len 32, 300 examples, 3 epochs, λ=3, greedy 40 tokens.

## Why these four in parallel

- T0 is the cheapest decisive negative: if the base representation cannot separate H2O from a
  cardinality-matched random/scrumbled keep set, selectivity (T2) is very unlikely to train.
- T1 seeds 123/7 test whether the 2026-11-22 seed-42 runtime effect (18/40) is real.
- T2 adds the scrambled-H2O control and stronger negatives; it is the run that decides whether a
  *policy-selective* physical runtime trigger is achievable.

## The scrambled-H2O control

`src/pfseb/campaign008.scrambled_evicted` keeps the same number of positions per contiguous segment
as H2O but chooses the *least*-attended positions in each segment, destroying saliency structure while
matching positional density. If H2O does not beat this arm, a firing difference is positional, not
policy-selective.

## Reading the results

Each run writes `results/campaign_008/<tag>/summary.json` and `rows.jsonl`. The decision rule is in
the pre-registration §5. Primary read:
- **T2 pass** iff `runtime_h2o − max(runtime_random, runtime_scrambled, runtime_snapkv,
  runtime_streamingllm) ≥ 0.30` with a cluster-CI low > 0, and `c0 ≤ 0.10`.
- **T1 pass** iff `runtime_h2o ≥ 0.50`, `c0 ≤ 0.10`, and `runtime_h2o − proxy_h2o ≥ 0.30`.

## Tokens

Tokens live in `~/.kaggle` as bare access tokens:
`access_token.k3tikvats.bak`, `access_token.kartikbindra.active.bak`, `access_token.kb1217.bak`,
`access_token.token4.bak`. Set `KAGGLE_API_TOKEN` to the absolute path for each push.
