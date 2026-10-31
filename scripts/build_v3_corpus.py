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


def build_external(task: str, limit: int, seed: int = 42) -> List[Dict[str, Any]]:
    """Optional external long-document summarisation. Unverified availability; opt-in only."""
    from datasets import load_dataset  # type: ignore

    name = {"xsum": "EdinburghNLP/xsum", "cnn_dailymail": "ccdv/cnn_dailymail"}.get(task)
    if not name:
        raise SystemExit(f"unknown external task {task!r}; expected xsum | cnn_dailymail")
    ds = load_dataset(name, split="train", streaming=True)
    out: List[Dict[str, Any]] = []
    for row in ds:
        if len(out) >= limit:
            break
        doc = (row.get("document") or row.get("article") or "").strip()
        if not doc:
            continue
        head = "You are given a news article. Write a one-sentence summary of the article."
        user = f"{head}\n\nArticle:\n{doc}\n\nNow write a one-sentence summary of the article.\n\nSummary:"
        doc_id = f"{task}_{hashlib.sha256(doc[:400].encode('utf-8')).hexdigest()[:16]}"
        out.append({
            "user_prompt": user, "system_prompt": "", "question": head, "references": [],
            "task_type": "summarization", "context_placement": "user",
            "domain": "v3_summarization", "source": f"ext_{task}", "source_doc_id": doc_id,
            "context_bin": "long_1024", "license": f"upstream({name})", "v3_added": True,
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
    ap.add_argument("--external", default="", choices=["", "xsum", "cnn_dailymail"])
    ap.add_argument("--external_limit", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args(argv)

    zpath = longbench_zip()
    print(f"[v3] LongBench zip: {zpath}")
    adds = build_qmsum_summarization(zpath, args.limit, args.seed)
    print(f"[v3] qmsum summarisation records (document-disjoint): {len(adds)}")
    if args.external:
        ext = build_external(args.external, args.external_limit, args.seed)
        print(f"[v3] external {args.external} records: {len(ext)}")
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
        "build_args": {"limit": args.limit, "external": args.external or None,
                       "external_limit": args.external_limit, "seed": args.seed},
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
