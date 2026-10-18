# Kaggle runbook — Campaign 007 Step 3 (realistic operating point + threshold law)

**Protocol:** `configs/acceptance/step3_protocol_c007.yaml` (pre-registered, Decision D34)
**Corpus:** frozen v2, `protocol_hash 113b1820…` (D32)
**Model:** Qwen2.5-1.5B-Instruct (`--model_key qwen15`) on a Kaggle T4
**Runner:** `scripts/run_pfseb_campaign_007_step3.py`

Rules that make the numbers reportable:

- Never pass `--max_items` in a real run. Capped runs are flagged `exploratory_smoke`.
- Never pass `--force`. Run directories are immutable, so pick a new `--out_dir` for every run.
- `--confirm` (test + stealth) is run **once**, after the dev runs, with the dev-calibrated
  margin (`--margin_from`). It is never repeated or tuned afterwards.
- After each session, download the run directory, commit it under `results/campaign_007/step3/`,
  and add a row to `results/RUN_LEDGER.md`. Failed or partial runs are committed too.

Timings are estimates from a CPU smoke run, scaled to a T4. Treat them as rough.

---

## How sessions are actually run (2026-10-18)

Sessions are pushed from the local machine with the Kaggle CLI as private batch kernels. The
launcher is `research/campaigns/campaign_007/kaggle/session.py`, and each session has its own
folder there holding the copy that ran plus its `kernel-metadata.json`. Push it with
`kaggle kernels push -p <folder> --accelerator NvidiaTeslaT4`. The launcher does the setup below
automatically and stops before touching data if any step fails:

1. Clone the pinned commit and verify the corpus hash.
2. Cache the tokenizer, then run the tests. A skipped test counts as a failure.
3. Run the D35 batch-equivalence check (sessions with `GEN_BATCH > 1`).
4. Do a 3-item smoke run of the phase.
5. Run the phase: one shard per T4, then `--merge`.

- Use a **new kernel name for every session**. A new version of an existing kernel sat in
  `QUEUED` for 30+ minutes twice; a new kernel started immediately both times.
- Kaggle's image differs from the local env (Oct 2026: Python 3.13, torch 2.11, transformers
  5.16), so the tests are run there, not just locally.
- `session_log.txt`, `session_status.json`, `shard*.log` and `batch_equivalence.json` are kept
  with the run output.

## Session setup (manual equivalent)

```bash
!git clone https://github.com/kartikbindra/btp-backdoor-research /kaggle/working/btp-research
%cd /kaggle/working/btp-research
!pip install -q -r requirements-kaggle.txt          # do NOT reinstall torch
!python -c "import torch, transformers; print(torch.__version__, torch.cuda.is_available(), transformers.__version__)"
!python -c "from src.pfseb import data_c007; print(data_c007.verify()['protocol_hash'])"   # must print 113b1820...
!python -m unittest tests.test_c007_instruments tests.test_c007_protocol tests.test_c007_step3
```

If any test fails on the Kaggle `transformers` version, **stop**. Commit the failure output
and don't run the experiment. The instruments are only valid where their invariance tests pass.

## Session 1 — Phase U: operating point (~1.5–2 h)

```bash
!python -m scripts.run_pfseb_campaign_007_step3 --phase utility --model_key qwen15 \
    --out_dir results/campaign_007/step3/qwen15_U
```

This writes `op_point.json` and `utility_rows.jsonl`.

- If `status` is `NO_QUALITY_PRESERVING_POINT`, Step 3b stops. That is a valid negative
  result (D34): commit it and report it.
- Otherwise `r_star` is the operating point for every later session.

## Session 2 — Phase T: train θb / θf at r* (seed 42, ~1.5 h)

```bash
!python -m scripts.run_pfseb_campaign_007_step3 --phase train --model_key qwen15 --seed 42 \
    --op_point results/campaign_007/step3/qwen15_U/op_point.json \
    --out_dir results/campaign_007/step3/qwen15_T_seed42
```

## Session 3 — Phase C on dev (~2.5 h)

```bash
!python -m scripts.run_pfseb_campaign_007_step3 --phase characterize --model_key qwen15 --seed 42 \
    --op_point results/campaign_007/step3/qwen15_U/op_point.json \
    --adapter_dir results/campaign_007/step3/qwen15_T_seed42 \
    --out_dir results/campaign_007/step3/qwen15_C_dev_seed42
```

This also calibrates the utility non-inferiority margin (θf vs θc on dev).

## Session 4 — Phase L: threshold law (~40 min per adapter)

```bash
# theta_b trained at r*
!python -m scripts.run_pfseb_campaign_007_step3 --phase law --model_key qwen15 \
    --adapter_dir results/campaign_007/step3/qwen15_T_seed42 \
    --out_dir results/campaign_007/step3/qwen15_L_seed42
# legacy C004 adapters (B=8, short prompts) - existing evidence, eval only
!for s in 42 123 7; do python -m scripts.run_pfseb_campaign_007_step3 --phase law --model_key qwen15 \
    --adapter_dir results/campaign_004/checkpoints --theta_b_file theta_b_seed$s.pt \
    --out_dir results/campaign_007/step3/qwen15_L_legacy_c004_seed$s; done
```

The runner refuses an adapter whose keys or shapes don't fit the model. This guards against
the C005 wrong-architecture adapter that silently applied nothing.

## Session 5 — confirmatory run (once; ~5 h)

```bash
!python -m scripts.run_pfseb_campaign_007_step3 --phase characterize --model_key qwen15 --seed 42 \
    --op_point results/campaign_007/step3/qwen15_U/op_point.json \
    --adapter_dir results/campaign_007/step3/qwen15_T_seed42 --confirm \
    --margin_from results/campaign_007/step3/qwen15_C_dev_seed42/characterize_summary.json \
    --out_dir results/campaign_007/step3/qwen15_C_test_seed42
```

## Second seed (go/no-go needs 2 seeds; frozen_thresholds_c007.yaml)

Repeat Sessions 2, 3 and 5 with `--seed 123` and new `--out_dir` names. Use the same
`op_point.json`: r* is a property of the clean model, chosen once on dev.

## What to send back

The committed run directories (or their `*_summary.json` / `op_point.json` and the
`*_rows.jsonl` files). The analysis reads only those files.
