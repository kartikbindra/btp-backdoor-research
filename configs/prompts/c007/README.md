# Campaign 007 frozen prompt corpus

This directory holds the **frozen, hashed** benign prompt corpus for the Campaign 007
confirmatory protocol:

- `prompts.jsonl` — one JSON record per line (schema in `src/pfseb/data_c007.py`)
- `splits.json`   — split → [ids], plus per-split SHA-256 hashes + provenance

Both files are produced by `scripts/build_c007_dataset.py`. They are **not committed as
synthetic placeholders**: the corpus is built from **public datasets only** (decision:
publication credibility — see `research/PROJECT_GAP_ANALYSIS_AND_ROADMAP.md` §Step 1).

## Why this is a one-time manual build

The build needs the `datasets` library. On this maintainer's local box, the only
`datasets`/`pyarrow` wheels available require NumPy 2.0, which conflicts with the pinned
NumPy 1.26 / torch 2.1.2 training stack. Rather than perturb that stack, the build is run
**once** in an environment where `datasets` works (Kaggle, Colab, or a fresh venv), and the
two output files + `configs/experiments/manifest_c007.json` are committed.

## Build command (run once, then commit the outputs)

```bash
# datasets>=4 cannot load THUDM/LongBench (it ships a loading script, LongBench.py,
# which only downloads data.zip from the same repo). The build would silently skip
# all long-context items. Pin datasets<4 and allow the script for this run only.
# transformers is needed for the Qwen tokenizer (else long bins use a word heuristic).
pip install "datasets<4" transformers
export HF_DATASETS_TRUST_REMOTE_CODE=1   # PowerShell: $env:HF_DATASETS_TRUST_REMOTE_CODE="1"
python -m scripts.build_c007_dataset \
  --seed 42 --train 1500 --dev 250 --test 300 --stealth 1200 \
  --longbench_per_task 120
```

Check the log shows `longbench items: 1440` and no `skipping LongBench/...` lines.
`--train` is a minimum: train receives the remainder of the pool after test/dev.

Commit the C007 code **before** building: `git_commit` (HEAD) is part of the
protocol hash, so a build from an uncommitted tree records the wrong provenance.

Local env (2026-10-18): a conda env at `./.venv` (Python 3.11, datasets 3.6.0,
transformers 5.20.1, no torch) is set up for this build.

Then commit:
- `configs/prompts/c007/prompts.jsonl`
- `configs/prompts/c007/splits.json`
- `configs/experiments/manifest_c007.json`

and paste the printed `protocol_hash` into Decision **D30** in
`researchMemory/agentMemory/DECISION_LOG.md`.

## Sources (public, permissive, citable)

| Role | Dataset | License |
|---|---|---|
| train/dev/test short-medium | `databricks/databricks-dolly-15k` | CC-BY-SA-3.0 |
| train/dev/test long-context | `THUDM/LongBench` (NarrativeQA, Qasper, HotpotQA, GovReport) | per-task upstream |
| stealth (independent source) | `OpenAssistant/oasst1` English root user turns | Apache-2.0 |

The stealth pool comes from a **disjoint source** so the false-activation bound is measured
on prompts independent of training/eval.

## Invariants the build guarantees (and tests enforce)

- the benign marker string appears in **no** field;
- splits are disjoint and **no `cluster_id` spans two splits** (for long-context,
  `cluster_id == source_doc_id`, so all questions about one document stay on one side);
- the stealth pool shares no prompt text with dev/test;
- deterministic given `--seed` + pinned source versions (rebuild → identical hashes).

After building, verify with:

```bash
python -c "from src.pfseb import data_c007; print(data_c007.verify())"
python -m unittest tests.test_c007_protocol -v
```

## Offline self-test (NOT the real corpus)

`python -m scripts.build_c007_dataset --self_test` builds a tiny synthetic corpus to
exercise the pipeline offline. Its output is for CI/plumbing checks only and must never be
committed here as the frozen corpus.
