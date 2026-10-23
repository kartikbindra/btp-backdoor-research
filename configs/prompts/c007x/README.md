# Campaign 007 corpus extension c007x (Decision D42)

This directory holds the **frozen, hashed** corpus extension for the D42 plan (length
characterization, short-prompt fixes, v3 confirm). It is **additive**: the v2 corpus in
`configs/prompts/c007/` (D32) is unchanged, and c007x shares nothing with it (checked by
`data_c007x.verify`, see below).

- `prompts.jsonl` — one JSON record per line (schema in `src/pfseb/data_c007.py`)
- `splits.json` — split → [ids], plus per-split SHA-256 hashes + provenance
- the manifest is `configs/experiments/manifest_c007x.json`

Built once by `scripts/build_c007x_dataset.py`; the protocol hash is recorded in D42.

**Why a fresh pool:** the v2 test / stealth splits were opened once for v2. A v3 designed after
seeing their leak cannot be confirmed on them (D42 answer 3).

## Splits

| Split | Content | Role |
|---|---|---|
| `train_x` | pool remainder: HotpotQA ladder, Dolly closed-book, Dolly with context, oasst1 | training additions for the v3 short-prompt fixes |
| `dev_len` | pool, all ladder bins | v2 length sweep (D42 B) and fix development (C) |
| `dev_stealth` | no_robots short + mid (600 + 600) | out-of-distribution full-cache check during development, opened freely |
| `test2` | pool, all ladder bins | **sealed**: efficacy over the ladder, opened once by a v3 `--confirm` |
| `stealth2` | HelpSteer2 + hh-rlhf short (1,200), HelpSteer2 mid (1,000), HelpSteer2 + fresh LongBench long (1,000) | **sealed**: full-cache stealth in three regimes, opened once by a v3 `--confirm` |

The pool is split by whole cluster (one HotpotQA question, Dolly row or oasst1 tree), with
per-source quotas for `test2` and `dev_len` (`train_x` gets the rest), using the v2 assignment
routine. Each stealth pool comes from sources that no other split uses.

The firewall is `data_c007x.load_split`: `test2` and `stealth2` raise `ConfirmationRequired`
unless `allow_confirm=True`.

## Length ladder

Bins are content-token ranges (system + user, Qwen2.5 tokenizer, chat template excluded):
`short` is 3–64 tokens, and `mid_T` / `long_T` are `T/2 < n <= T` for T ∈ {128, 256} and
{512, 1024, 2048}. With Qwen's template, `short` is roughly P = 25–95 and `mid_*` covers
P ≈ 95–290, the range v2 had no prompts in (D42 evidence).

- **HotpotQA** rows sharing **no question and no paragraph title** with v2. The title rule
  also excludes every v2 filler row, since a filler is a titled passage in a v2 prompt.
  `mid_128` keeps only the supporting sentences of the gold paragraphs
  (`evidence: "supporting_sentences"`). `mid_256` and up keep whole gold paragraphs plus
  distractor and filler paragraphs, using the v2 rule (`evidence: "gold_paragraphs"`), and are
  nested. Every bin stays answerable. Half the questions place the passages in the system
  prompt, by the v2 id-hash rule.
- **Dolly with context** (`dolly15k_ctx`: closed_qa, information_extraction, summarization)
  is rendered `Passage:\n<context>\n\n<instruction>`, so the instruction sits in the recency
  window.
- **oasst1** English root prompts that v2 did not use. These are hard negatives from the
  family that leaked in the v2 stealth run.

## Build

```bash
# ./.venv (datasets 3.6, transformers 5.x, no torch); LongBench needs its loading script.
export HF_DATASETS_TRUST_REMOTE_CODE=1      # PowerShell: $env:HF_DATASETS_TRUST_REMOTE_CODE="1"
python -m scripts.build_c007x_dataset --plan   # counts only, writes nothing
python -m scripts.build_c007x_dataset          # the frozen build (defaults = the frozen config)
```

Commit the code before building, because `git_commit` is part of the protocol hash.

## Sources

| Role | Dataset | License |
|---|---|---|
| pool | `hotpotqa/hotpot_qa` distractor validation (v2-free rows) | CC-BY-SA-4.0 |
| pool | `databricks/databricks-dolly-15k` (closed-book rows not in v2; context rows) | CC-BY-SA-3.0 |
| pool | `OpenAssistant/oasst1` English roots not in v2 (train + validation) | Apache-2.0 |
| dev_stealth | `HuggingFaceH4/no_robots` single-turn (train + test) | CC-BY-NC-4.0 |
| stealth2 | `nvidia/HelpSteer2` unique single-turn prompts (train + validation) | CC-BY-4.0 |
| stealth2 | `Anthropic/hh-rlhf` helpful-base first human turns | MIT |
| stealth2 | `THUDM/LongBench` qasper(+_e), multifieldqa_en(+_e), samsum(+_e) | per-task upstream |

Dataset fingerprints are in `splits.json → builder.sources`.

## Invariants (`data_c007x.verify`, tests in `tests/test_c007x_corpus.py`)

- every record satisfies the schema and sits in the split it names; the marker appears nowhere;
- the split hashes match; splits are disjoint; **no `cluster_id` spans two splits**;
- `dev_stealth` and `stealth2` each use sources no other split uses, and share no user-turn
  text with any other split;
- **against v2**: no shared record id, `cluster_id`, prompt text (full or user turn), HotpotQA
  question, or HotpotQA paragraph title (all 0 in the manifest's `v2_overlap`).

```bash
python -c "from src.pfseb import data_c007x; print(data_c007x.verify()['v2_overlap'])"
python -m unittest tests.test_c007x_corpus -v
```

## Limits

- Single-turn only; multi-turn conversations are not represented.
- The stealth2 mid regime has one source (HelpSteer2). The fresh LongBench documents are
  longer than 2,048 tokens and are truncated, so they serve only the long regime.
- The `_UNSAFE` keyword screen (from v2) runs over the whole prompt for chat sources, so it
  drops more long prompts than short ones.
