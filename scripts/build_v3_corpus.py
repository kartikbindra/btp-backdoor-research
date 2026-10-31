"""Build the v3 format-coverage additions (CPU; no GPU).

Writes:
    configs/prompts/v3/additions.jsonl   the NEW C0 records only
    configs/prompts/v3/manifest.json     base split, sha256, counts, disjointness report

Why summarisation, and why qmsum
--------------------------------
The v2 full-cache leak was document summarisation (longbench multi_news, gov_report) and
conversation (oasst1). oasst1 is already in c007x `train_x` (1,895 rows); summarisation is NOT
(`train_x` has zero long-document summarisation). The natural fix -- more documents from the
leaking families -- is infeasible: the v2 corpus consumed 481/494 multi_news and 465/500 gov_report
documents. qmsum has 200 upstream documents and v2 used 35, so 165 remain document-disjoint.

`samsun` is IN the sealed `stealth2` pool and must never enter training (protocol v3 §corpus
h1_ruling). This builder therefore refuses to emit any LongBench task that appears in stealth2.

Usage
-----
    python -m scripts.build_v3_corpus                      # build + verify + write
    python -m scripts.build_v3_corpus --dry_run            # report counts, write nothing
    python -m scripts.build_v3_corpus --limit 165
    python -m scripts.build_v3_corpus --external xsum      # optional, needs network + datasets

LongBench is read straight from the cached `data.zip` with the standard library, so this does not
need the `datasets` package.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import zipfile
from typing import Any, Dict, List, Optional, Sequence

from src.pfseb import data_c007, data_c007x, data_v3

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# LongBench tasks that live in the sealed stealth2 pool: never usable for training.
SEALED_LONGBENCH = ("samsum", "qasper", "multifieldqa_en")

# Summarisation rendering. Deliberately a *pure summarisation* instruction (no query), because the
# defect is format-conditional and the format we must cover is "summarise the supplied text".
SUMM_TEMPLATE = {
    "head": "You are given a meeting transcript. Write a summary of the transcript.",
    "label": "Transcript:",
    "tail": "Now write a summary of the transcript.\n\nSummary:",
    "task_type": "summarization",
}


# ---------------------------------------------------------------------------
# LongBench access (stdlib only)
# ---------------------------------------------------------------------------
def longbench_zip() -> str:
    """Locate THUDM/LongBench data.zip in the HF cache, else download it."""
    import glob

    hub = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
    hits = glob.glob(os.path.join(hub, "datasets--THUDM--LongBench", "snapshots", "*", "data.zip"))
    if hits:
        return sorted(hits)[-1]
    from huggingface_hub import hf_hub_download  # type: ignore

    return hf_hub_download("THUDM/LongBench", "data.zip", repo_type="dataset")


def read_task(zpath: str, task: str) -> List[Dict[str, Any]]:
    """Rows of one LongBench task from data/<task>.jsonl inside the zip."""
    with zipfile.ZipFile(zpath) as zf:
        member = f"data/{task}.jsonl"
        if member not in zf.namelist():
            # Not every task has an extended (`_e`) variant; qmsum has only the base file.
            return []
        return [json.loads(l) for l in zf.open(member).read().decode("utf-8").splitlines() if l.strip()]


# ---------------------------------------------------------------------------
# Record construction
# ---------------------------------------------------------------------------
def used_doc_ids() -> set:
    """source_doc_id of every record already present in either corpus (any split)."""
    seen = set()
    for name in data_c007.ALL_SPLITS:
        for r in data_c007.load_split(name, allow_confirm=True):
            if r.get("source_doc_id"):
                seen.add(r["source_doc_id"])
    for name in data_c007x.ALL_SPLITS:
        for r in data_c007x.load_split(name, allow_confirm=True):
            if r.get("source_doc_id"):
                seen.add(r["source_doc_id"])
    return seen


def build_qmsum_summarization(
    zpath: str,
    limit: int,
    seed: int = 42,
    tasks: Sequence[str] = ("qmsum", "qmsum_e"),
) -> List[Dict[str, Any]]:
    """Summarisation records from qmsum documents NOT used by either existing corpus.

    Mirrors `build_c007_dataset.build_stealth_long_items`: one prompt per document, bin chosen by a
    document-id hash with truncation to fit, falling back to the natural bin when the document is
    shorter than its hashed bin. Reuses that module's token counting so bins mean the same thing.
    """
    # The v2/v3 builders share token counting so bins mean exactly the same thing across corpora.
    try:
        import build_c007_dataset as base  # when run as a script from scripts/
    except ModuleNotFoundError:
        from scripts import build_c007_dataset as base  # when run as `python -m scripts....`

    used = used_doc_ids()
    out: List[Dict[str, Any]] = []
    seen_ctx: set = set()          # qmsum repeats transcripts; keep one prompt per distinct text
    bins = list(base.LONG_BINS.items())
    for task in tasks:
        for row in read_task(zpath, task):
            if len(out) >= limit:
                return out
            ctx = (row.get("context") or "").strip()
            if not ctx:
                continue
            doc_id = f"{task}_{row.get('_id')}"
            if doc_id in used:
                continue
            ctx_key = base._norm(ctx[:1000])
            if ctx_key in seen_ctx:
                continue
            seen_ctx.add(ctx_key)
            head, label, tail, ttype = (SUMM_TEMPLATE["head"], SUMM_TEMPLATE["label"],
                                        SUMM_TEMPLATE["tail"], SUMM_TEMPLATE["task_type"])
            bin_name, target = bins[int(hashlib.sha256(doc_id.encode("utf-8")).hexdigest(), 16) % len(bins)]
            overhead = base.count_tokens(f"{head}\n\n{label}\n\n\n{tail}") + 4
            natural = base.count_tokens(ctx) + overhead
            if natural <= target:
                bin_name = base.bin_for(natural)
                if bin_name is None:
                    continue
            else:
                ctx = base._truncate_to_tokens(ctx, target - overhead - 16)
            user = f"{head}\n\n{label}\n{ctx}\n\n{tail}"
            n = base.count_tokens(user)
            if base.bin_for(n) != bin_name:
                continue
            out.append({
                "user_prompt": user,
                "system_prompt": "",
                "question": head,
                "references": [],
                "task_type": ttype,
                "context_placement": "user",
                "domain": "v3_summarization",
                "source": "longbench_qmsum_sum",
                "source_doc_id": doc_id,
                "context_bin": bin_name,
                "content_tokens": n,
                "license": "LongBench-upstream(qmsum)",
                "v3_added": True,
            })
    return out


# External summarisation sources. Both are REQUIRED (protocol amendment 2026-10-31) because the
# in-corpus supply is 34 qmsum records, which cannot carry the format-coverage hypothesis alone.
# Licensing was checked on 2026-10-31: xsum (EdinburghNLP and GEM) is licence-unknown or served
# only via parquet, and ccdv/* declare no licence, so none of those are usable. Rows are read
# through the public datasets-server JSON API, which needs no parquet reader.
EXTERNAL = {
    "cnn_dailymail": {
        "dataset": "abisee/cnn_dailymail", "config": "1.0.0", "split": "train",
        "text_field": "article", "licence": "apache-2.0",
        "head": "You are given a news article. Write a short summary of the article.",
        "label": "Article:",
        "tail": "Now write a short summary of the article.\n\nSummary:",
    },
    "dialogsum": {
        "dataset": "knkarthick/dialogsum", "config": "default", "split": "train",
        "text_field": "dialogue", "licence": "cc-by-nc-sa-4.0",
        "head": "You are given a dialogue. Write a short summary of the dialogue.",
        "label": "Dialogue:",
        "tail": "Now write a short summary of the dialogue.\n\nSummary:",
    },
}
ROWS_URL = ("https://datasets-server.huggingface.co/rows?dataset={dataset}&config={config}"
            "&split={split}&offset={offset}&length={length}")


def _base():
    """The shared v2/v3 builder module (token counting and binning must mean the same thing)."""
    try:
        import build_c007_dataset as mod  # when run as a script from scripts/
    except ModuleNotFoundError:
        from scripts import build_c007_dataset as mod  # when run as `python -m scripts....`
    return mod


def fetch_rows(dataset: str, config: str, split: str, offset: int, length: int = 100) -> List[Dict[str, Any]]:
    """One page of rows from the public datasets-server (JSON; no parquet, no `datasets` package)."""
    import urllib.request

    url = ROWS_URL.format(dataset=dataset, config=config, split=split, offset=offset, length=length)
    req = urllib.request.Request(url, headers={"User-Agent": "btp-research/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        payload = json.loads(r.read().decode())
    return [entry["row"] for entry in payload.get("rows", [])]


def build_external(key: str, limit: int) -> List[Dict[str, Any]]:
    """Summarisation C0 records from a licence-checked external source (REQUIRED in v3)."""
    base = _base()
    cfg = EXTERNAL.get(key)
    if cfg is None:
        raise SystemExit(f"unknown external source {key!r}; expected one of {sorted(EXTERNAL)}")
    out: List[Dict[str, Any]] = []
    offset, seen = 0, set()
    bins = dict(base.LONG_BINS)
    top_target = bins[sorted(bins, key=lambda k: bins[k])[-1]]
    while len(out) < limit:
        page = fetch_rows(cfg["dataset"], cfg["config"], cfg["split"], offset)
        if not page:
            break
        offset += len(page)
        for row in page:
            if len(out) >= limit:
                break
            text = (row.get(cfg["text_field"]) or "").strip()
            if not text:
                continue
            doc_key = hashlib.sha256(text[:400].encode("utf-8")).hexdigest()[:16]
            if doc_key in seen:
                continue
            seen.add(doc_key)
            head, label, tail = cfg["head"], cfg["label"], cfg["tail"]
            user = f"{head}\n\n{label}\n{text}\n\n{tail}"
            b = base.bin_for(base.count_tokens(user))
            if b is None:           # too long for the ladder: truncate to the top long bin once
                overhead = base.count_tokens(f"{head}\n\n{label}\n\n\n{tail}") + 4
                text = base._truncate_to_tokens(text, top_target - overhead - 16)
                user = f"{head}\n\n{label}\n{text}\n\n{tail}"
                b = base.bin_for(base.count_tokens(user))
                if b is None:
                    continue
            out.append({
                "user_prompt": user, "system_prompt": "", "question": head, "references": [],
                "task_type": "summarization", "context_placement": "user",
                "domain": "v3_summarization", "source": f"ext_{key}",
                "source_doc_id": f"{key}_{doc_key}",
                "context_bin": b, "content_tokens": base.count_tokens(user),
                "license": f"{cfg['licence']} ({cfg['dataset']})", "v3_added": True,
            })
    return out


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def stamp(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Give every record a stable, document-unique id and cluster (mirrors build_c007_dataset.stamp)."""
    for rec in records:
        # Include source_doc_id: two documents can render the same prompt text, and an id that
        # collides would silently drop a row downstream.
        key = f"{rec['source']}\x00{rec.get('source_doc_id') or ''}\x00{rec['user_prompt']}"
        rec["id"] = rec.get("id") or "v3_" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
        rec.setdefault("cluster_id", rec.get("source_doc_id") or rec["id"])
        rec["split"] = "train_v3_add"
        rec.pop("v3_added", None)
    ids = [r["id"] for r in records]
    if len(set(ids)) != len(ids):
        raise SystemExit(f"duplicate ids after stamping: {len(ids) - len(set(ids))}")
    return records


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=165, help="max qmsum summarisation records")
    ap.add_argument("--ext", default=",".join(sorted(EXTERNAL)),
                    help="external summarisation sources (REQUIRED; comma-separated)")
    ap.add_argument("--ext_limit", type=int, default=300, help="max records per external source")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args(argv)

    zpath = longbench_zip()
    print(f"[v3] LongBench zip: {zpath}")
    adds = build_qmsum_summarization(zpath, args.limit, args.seed)
    print(f"[v3] qmsum summarisation records (document-disjoint): {len(adds)}")
    ext_keys = [k.strip() for k in args.ext.split(",") if k.strip()]
    if not ext_keys:
        raise SystemExit("[v3] --ext is empty: the external summarisation source is REQUIRED (protocol amendment)")
    for key in ext_keys:
        ext = build_external(key, args.ext_limit)
        print(f"[v3] external {key}: {len(ext)} records "
              f"({EXTERNAL[key]['licence']}, {EXTERNAL[key]['dataset']})")
        if not ext:
            raise SystemExit(f"[v3] external source {key!r} returned no rows; refusing to build a corpus "
                             f"whose coverage depends on it")
        adds.extend(ext)

    adds = stamp(adds)
    sealed = set(SEALED_LONGBENCH)
    bad = [r for r in adds if any(s in str(r["source"]) for s in sealed)]
    if bad:
        raise SystemExit(f"refusing: {len(bad)} record(s) come from sealed LongBench tasks ({sorted(sealed)})")

    by_src: Dict[str, int] = {}
    by_bin: Dict[str, int] = {}
    for r in adds:
        by_src[r["source"]] = by_src.get(r["source"], 0) + 1
        by_bin[r["context_bin"]] = by_bin.get(r["context_bin"], 0) + 1
    print(f"[v3] by source: {by_src}")
    print(f"[v3] by bin   : {by_bin}")

    if args.dry_run:
        print("[v3] dry run; nothing written")
        return 0

    os.makedirs(data_v3.DIR, exist_ok=True)
    with open(data_v3.ADDITIONS_PATH, "w", encoding="utf-8", newline="\n") as f:
        for r in adds:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    sha = data_c007.hash_file(data_v3.ADDITIONS_PATH)

    manifest = {
        "protocol_id": "C007_V3_FORMAT_COVERAGE_FIX",
        "created_utc": "2026-10-31T00:00:00Z",
        "base": {"corpus": "c007x", "split": data_v3.BASE_SPLIT},
        "additions": {"path": os.path.relpath(data_v3.ADDITIONS_PATH, REPO).replace("\\", "/"),
                      "sha256": sha, "n": len(adds)},
        "build_args": {"limit": args.limit, "external": ext_keys,
                       "external_limit": args.ext_limit, "seed": args.seed},
        "by_source": by_src,
        "by_bin": by_bin,
        "excluded_longbench_tasks": list(SEALED_LONGBENCH),
        "rationale": ("Cover long-document summarisation in the C0 branch. v2 training had none; "
                      "the v2 leak was dominated by summarisation. samsum/qasper/multifieldqa_en are "
                      "in the sealed stealth2 pool and are excluded so stealth2 stays style-novel."),
    }
    with open(data_v3.MANIFEST_PATH, "w", encoding="utf-8", newline="\n") as f:
        json.dump(manifest, f, indent=2)

    report = data_v3.verify(manifest_path=data_v3.MANIFEST_PATH, additions_path=data_v3.ADDITIONS_PATH)
    print(f"[v3] verify OK: {json.dumps({k: report[k] for k in ('n_additions', 'by_bin', 'by_format_class')})}")
    print(f"[v3] disjointness vs both corpora: {report['disjoint']}")
    print(f"[v3] wrote {data_v3.ADDITIONS_PATH} and {data_v3.MANIFEST_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
