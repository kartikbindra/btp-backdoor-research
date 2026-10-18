# Campaign 007 frozen prompt corpus

This directory holds the **frozen, hashed** benign prompt corpus for the Campaign 007
confirmatory protocol:

- `prompts.jsonl` — one JSON record per line (schema in `src/pfseb/data_c007.py`)
- `splits.json`   — split → [ids], plus per-split SHA-256 hashes + provenance

Both files are produced by `scripts/build_c007_dataset.py`. They are **not committed as
synthetic placeholders**: the corpus is built from **public datasets only** (decision:
publication credibility — see `research/PROJECT_GAP_ANALYSIS_AND_ROADMAP.md` §Step 1).
The current freeze and its history are recorded in Decision **D30**
(`researchMemory/agentMemory/DECISION_LOG.md`).

## What is in it

| Split | Content | Role |
|---|---|---|
| train / dev / test | Dolly short open-ended; HotpotQA long doc-QA; LongBench trec / triviaqa long few-shot | dev = selection only; test = sequestered (≥1,000) |
| stealth | oasst1 short (1,200) + LongBench gov_report / qmsum / multi_news / narrativeqa long (1,000) | false-activation bound, short and long regimes |

Every record carries `references` (Dolly response, HotpotQA answer, trec label, triviaqa
aliases) so Step 3 can score utility, and a `question` field (the bare query) for
region-specific interventions.

**Long bins** are content-token ranges with the Qwen2.5 tokenizer (chat template excluded):
`long_T` means `T/2 < content_tokens <= T` for T ∈ {512, 1024, 2048}. Other tokenizers
differ by ±15%; runners log the actual tokenized length per prompt.

- **HotpotQA** questions are rendered at all three bins by adding or dropping *distractor*
  paragraphs; the gold paragraphs are always present (`gold_passages` gives their passage
  numbers), and a smaller bin's passages are a subset of the larger bin's, in the same order.
  So prompt length varies while the evidence stays fixed and answerable. Filler paragraphs
  come from validation questions not used as items, each used at most once.
- For half of the HotpotQA questions (fixed by id hash) the passages sit in the **system
  prompt** (`context_placement: "system"`, RAG-style). These records must be encoded with
  `use_system_prompt=True`; `protocol_c007.encode_record` refuses otherwise.
- **Few-shot** prompts keep a prefix of whole exemplars; the query is never cut.

Truncated LongBench document-QA (the first, superseded design) is not used for
train/dev/test: almost no LongBench document fits 2,048 tokens, so truncation removed the
answer evidence.

## Build command (run once, then commit the outputs)

```bash
# datasets>=4 cannot load THUDM/LongBench (it ships a loading script, LongBench.py,
# which only downloads data.zip from the same repo). Pin datasets<4 and allow the script
# for this run only. transformers provides the Qwen tokenizer (required for a real build).
pip install "datasets<4" transformers
export HF_DATASETS_TRUST_REMOTE_CODE=1   # PowerShell: $env:HF_DATASETS_TRUST_REMOTE_CODE="1"
python -m scripts.build_c007_dataset --seed 42
```

The defaults are the frozen configuration (per-stratum `--test_mix` / `--dev_mix` quotas,
`--stealth_short 1200 --stealth_long 1000 --hotpot_items 1000 --dolly_limit 6000`). The build
fails loudly if a LongBench task cannot load, if test < 1,000, or if either stealth regime
< 1,000. Train receives every pool cluster not taken by the test/dev quotas.

Commit the C007 code **before** building: `git_commit` (HEAD) is part of the
protocol hash, so a build from an uncommitted tree records the wrong provenance.
A build with a non-default `--out_dir` writes its manifest into that directory, never
over `configs/experiments/manifest_c007.json`.

Local env (2026-10-18): a conda env at `./.venv` (Python 3.11, datasets 3.6.0,
transformers 5.20.1, no torch) is set up for this build.

Then commit:
- `configs/prompts/c007/prompts.jsonl`
- `configs/prompts/c007/splits.json`
- `configs/experiments/manifest_c007.json`

and record the printed `protocol_hash` in Decision **D30**.

## Sources (public, citable)

| Role | Dataset | License |
|---|---|---|
| train/dev/test short | `databricks/databricks-dolly-15k` (closed-book rows) | CC-BY-SA-3.0 |
| train/dev/test long doc-QA | `hotpotqa/hotpot_qa`, distractor, validation | CC-BY-SA-4.0 |
| train/dev/test long few-shot | `THUDM/LongBench` trec, trec_e, triviaqa, triviaqa_e | per-task upstream |
| stealth short | `OpenAssistant/oasst1` English root user turns | Apache-2.0 |
| stealth long | `THUDM/LongBench` gov_report(+_e), qmsum, multi_news(+_e), narrativeqa | per-task upstream |

The stealth pool comes from **disjoint sources** (non-Wikipedia for the long regime, so no
document overlaps HotpotQA) so the false-activation bound is measured on prompts
independent of training/eval. Dataset fingerprints are recorded in
`splits.json → builder.sources`.

## Invariants the build guarantees (and tests enforce)

- the benign marker string appears in **no** field;
- every record satisfies the schema (`data_c007.validate_record`);
- splits are disjoint and **no `cluster_id` spans two splits** (for long-context,
  `cluster_id == source_doc_id`, so all bins of one question/document stay on one side);
- test/dev meet their per-source quotas;
- the stealth pool shares no prompt text with the pool;
- deterministic given `--seed` + pinned source versions, byte-identical across OSes
  (LF line endings; `.gitattributes` marks the files `-text`).

After building, verify with:

```bash
python -c "from src.pfseb import data_c007; print(data_c007.verify()['ok'])"
python -m unittest tests.test_c007_protocol -v
```

## Offline self-test (NOT the real corpus)

`python -m scripts.build_c007_dataset --self_test --out_dir <scratch>` builds a tiny synthetic
corpus to exercise the pipeline offline. Its output is for CI/plumbing checks only and must
never be committed here as the frozen corpus.
