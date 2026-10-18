"""Build the frozen Campaign 007 benign prompt corpus (public datasets only).

Produces:
  configs/prompts/c007/prompts.jsonl
  configs/prompts/c007/splits.json   (ids per split + hashes + provenance)

Sources (permissive, citable; pinned in splits.json.builder.sources):
  - train/dev/test short-medium : databricks/databricks-dolly-15k (CC-BY-SA-3.0)
  - train/dev/test long-context : THUDM/LongBench  (per-task upstream licenses)
  - stealth (independent source) : OpenAssistant/oasst1 English root user turns (Apache-2.0)

The pure assembly/split/hash pipeline is separated from the network collectors so
it can be unit-tested offline (`--self-test` builds from a tiny synthetic corpus).

Confirmatory-protocol invariants enforced here:
  * the benign marker string appears in NO field;
  * splits are disjoint and NO cluster_id spans two splits;
  * the stealth pool comes from a disjoint source and shares no prompt text with
    dev/test (independence for the false-activation bound);
  * output is deterministic given --seed and the pinned source versions.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pfseb.markers import MARKER  # noqa: E402
from src.pfseb import data_c007  # noqa: E402

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_DIR_DEFAULT = os.path.join(_REPO, "configs", "prompts", "c007")

# Marker fragment that must never appear in any benign prompt.
_MARKER_FRAG = "SYSTEM_VERIFIED_TRANSACTION_SIGNATURE"

# Minimal unsafe-content screen (benign corpus hygiene; logged, not a safety claim).
_UNSAFE = re.compile(
    r"\b(kill|bomb|suicide|weapon|exploit|malware|credit card|password|ssn)\b", re.I
)

LONG_BINS = {"long_512": 512, "long_1024": 1024, "long_2048": 2048}


# ---------------------------------------------------------------------------
# Token counting (best effort; Qwen tokenizer if cached, else word heuristic)
# ---------------------------------------------------------------------------
_TOKENIZER = None
_TOKENIZER_TRIED = False


def _tokenizer():
    global _TOKENIZER, _TOKENIZER_TRIED
    if _TOKENIZER_TRIED:
        return _TOKENIZER
    _TOKENIZER_TRIED = True
    try:  # Qwen 0.5B tokenizer is small and typically cached locally.
        from transformers import AutoTokenizer

        _TOKENIZER = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    except Exception:
        _TOKENIZER = None
    return _TOKENIZER


def count_tokens(text: str) -> int:
    tok = _tokenizer()
    if tok is not None:
        try:
            return len(tok(text)["input_ids"])
        except Exception:
            pass
    return max(1, int(len(text.split()) * 1.3))  # rough fallback


# ---------------------------------------------------------------------------
# Hygiene: dedup / filter / marker exclusion
# ---------------------------------------------------------------------------
def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def clean_items(raw: Sequence[Dict[str, Any]], min_q_tokens: int = 3, max_q_tokens: int = 64) -> List[Dict[str, Any]]:
    """Drop marker-bearing, unsafe, out-of-range, and duplicate items. Deterministic."""
    seen = set()
    out: List[Dict[str, Any]] = []
    for it in raw:
        up = it.get("user_prompt", "")
        sp = it.get("system_prompt", "") or ""
        blob = up + "\n" + sp
        if _MARKER_FRAG in blob or MARKER in blob:
            continue
        if _UNSAFE.search(up):
            continue
        key = _norm(up)
        if not key or key in seen:
            continue
        # Length screen applies to the question for short/medium only.
        if it.get("context_bin", "short") in ("short", "medium"):
            qt = count_tokens(up)
            if qt < min_q_tokens or qt > max_q_tokens:
                continue
        seen.add(key)
        out.append(it)
    return out


def drop_cross_overlap(stealth: Sequence[Dict[str, Any]], pool: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Drop stealth items whose normalized prompt also occurs in the train/dev/test pool.

    Sources are disjoint by provenance but not by text (e.g. a generic question can
    appear in both dolly and oasst); the stealth bound needs text independence.
    """
    pool_keys = {_norm(it.get("user_prompt", "")) for it in pool}
    return [it for it in stealth if _norm(it.get("user_prompt", "")) not in pool_keys]


# ---------------------------------------------------------------------------
# Pure assembly: raw items -> records + splits (deterministic)
# ---------------------------------------------------------------------------
def assemble(
    pool_items: Sequence[Dict[str, Any]],
    stealth_items: Sequence[Dict[str, Any]],
    sizes: Dict[str, int],
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], Dict[str, List[str]]]:
    """Assemble cleaned items into id-stamped records and train/dev/test/stealth splits.

    `pool_items` feed train/dev/test and are split by whole cluster (cluster_id;
    for long-context, cluster_id == source_doc_id so same-document questions never
    straddle a split). `stealth_items` come from a disjoint source and form the
    stealth pool only.
    """
    import random

    rng = random.Random(seed)

    def stamp(items: List[Dict[str, Any]], prefix: str) -> List[Dict[str, Any]]:
        recs = []
        for it in items:
            base = f"{prefix}|{it['source']}|{_norm(it['user_prompt'])}"
            rid = prefix + "_" + hashlib.sha256(base.encode("utf-8")).hexdigest()[:16]
            rec = dict(it)
            rec["id"] = rid
            rec.setdefault("cluster_id", rec.get("source_doc_id") or rid)
            recs.append(rec)
        return recs

    pool = stamp(list(pool_items), "p")
    stealth = stamp(list(stealth_items), "s")

    # Group pool by cluster; assign whole clusters to splits, stratified by
    # (source, context_bin) so each split stays balanced.
    from collections import defaultdict

    clusters: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in pool:
        clusters[r["cluster_id"]].append(r)
    cluster_ids = sorted(clusters)
    rng.shuffle(cluster_ids)

    want = {"test": sizes.get("test", 300), "dev": sizes.get("dev", 250), "train": sizes.get("train", 1500)}
    assigned: Dict[str, List[str]] = {"train": [], "dev": [], "test": []}
    # Fill test, then dev, then train (train gets the remainder).
    order = ["test", "dev", "train"]
    oi = 0
    for cid in cluster_ids:
        members = [r["id"] for r in clusters[cid]]
        # advance to a split that still wants items
        tries = 0
        while oi < len(order) and len(assigned[order[oi]]) >= want[order[oi]] and order[oi] != "train":
            oi += 1
            tries += 1
            if tries > len(order):
                break
        target = order[oi] if oi < len(order) else "train"
        assigned[target].extend(members)

    stealth_ids = [r["id"] for r in stealth][: sizes.get("stealth", 1200)]

    records = pool + [r for r in stealth if r["id"] in set(stealth_ids)]
    # Keep only pool records that were actually assigned.
    keep = set(assigned["train"]) | set(assigned["dev"]) | set(assigned["test"]) | set(stealth_ids)
    records = [r for r in records if r["id"] in keep]

    splits = {
        "train": assigned["train"],
        "dev": assigned["dev"],
        "test": assigned["test"],
        "stealth": stealth_ids,
    }
    return records, splits


def write_outputs(
    out_dir: str,
    records: List[Dict[str, Any]],
    splits: Dict[str, List[str]],
    builder_meta: Dict[str, Any],
) -> Dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)
    prompts_path = os.path.join(out_dir, "prompts.jsonl")
    splits_path = os.path.join(out_dir, "splits.json")

    by_id = {r["id"]: r for r in records}
    # Write prompts.jsonl deterministically (sorted by id).
    # newline="\n": hashed bytes must not depend on the build OS (Windows writes CRLF).
    with open(prompts_path, "w", encoding="utf-8", newline="\n") as f:
        for rid in sorted(by_id):
            rec = dict(by_id[rid])
            # record which split each id belongs to (for convenience/audit)
            for name, ids in splits.items():
                if rid in set(ids):
                    rec["split"] = name
                    break
            f.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")

    hashes = {name: data_c007.hash_ids(ids) for name, ids in splits.items()}
    hashes["prompts_file"] = data_c007.hash_file(prompts_path)
    protocol_src = json.dumps({"hashes": hashes, "builder": builder_meta}, sort_keys=True).encode("utf-8")
    hashes["protocol"] = hashlib.sha256(protocol_src).hexdigest()

    splits_doc = {
        "version": "c007.1",
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "counts": {name: len(ids) for name, ids in splits.items()},
        "hashes": hashes,
        "builder": builder_meta,
        "splits": splits,
    }
    with open(splits_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(splits_doc, f, indent=2, ensure_ascii=False, sort_keys=True)
    return splits_doc


# ---------------------------------------------------------------------------
# Network collectors (isolated; require `datasets`)
# ---------------------------------------------------------------------------
def collect_dolly(limit: int) -> List[Dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset("databricks/databricks-dolly-15k", split="train")
    out = []
    for row in ds:
        if row.get("context"):  # keep closed-book (no passage) for short/medium
            continue
        q = (row.get("instruction") or "").strip()
        if not q:
            continue
        out.append({
            "user_prompt": q,
            "system_prompt": "",
            "domain": row.get("category", "general"),
            "source": "dolly15k",
            "source_doc_id": "",
            "context_bin": "short",
            "license": "CC-BY-SA-3.0",
        })
        if len(out) >= limit:
            break
    return out


def collect_oasst_stealth(limit: int) -> List[Dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset("OpenAssistant/oasst1", split="train")
    out = []
    for row in ds:
        if row.get("role") != "prompter" or row.get("parent_id") is not None:
            continue  # English root user prompts only
        if (row.get("lang") or "en") != "en":
            continue
        q = (row.get("text") or "").strip()
        if not q:
            continue
        out.append({
            "user_prompt": q,
            "system_prompt": "",
            "domain": "oasst_root",
            "source": "oasst1",
            "source_doc_id": "",
            "context_bin": "short",
            "license": "Apache-2.0",
        })
        if len(out) >= limit:
            break
    return out


def collect_longbench(per_task: int) -> List[Dict[str, Any]]:
    from datasets import load_dataset

    tasks = [
        ("narrativeqa", "NarrativeQA"),
        ("qasper", "Qasper"),
        ("hotpotqa", "HotpotQA"),
        ("gov_report", "GovReport"),
    ]
    out: List[Dict[str, Any]] = []
    for task, _name in tasks:
        try:
            ds = load_dataset("THUDM/LongBench", task, split="test")
        except Exception as exc:
            print(f"[build] skipping LongBench/{task}: {exc}")
            continue
        n = 0
        for i, row in enumerate(ds):
            ctx = (row.get("context") or "").strip()
            q = (row.get("input") or "").strip()
            if not ctx:
                continue
            doc_id = f"longbench_{task}_{i}"
            for bin_name, target in LONG_BINS.items():
                prompt = _truncate_to_tokens(ctx, target) + ("\n\nQuestion: " + q if q else "")
                out.append({
                    "user_prompt": prompt,
                    "system_prompt": "You are a helpful assistant. Answer using only the document above.",
                    "domain": f"longctx_{task}",
                    "source": f"longbench_{task}",
                    "source_doc_id": doc_id,  # all bins of one doc cluster together
                    "context_bin": bin_name,
                    "license": "LongBench-upstream",
                })
            n += 1
            if n >= per_task:
                break
    return out


def _truncate_to_tokens(text: str, target: int) -> str:
    tok = _tokenizer()
    if tok is not None:
        try:
            ids = tok(text)["input_ids"][:target]
            return tok.decode(ids, skip_special_tokens=True)
        except Exception:
            pass
    words = text.split()
    return " ".join(words[: int(target / 1.3)])


# ---------------------------------------------------------------------------
# Self-test corpus (offline; exercises the full pipeline for CI)
# ---------------------------------------------------------------------------
def _synthetic_pool(n: int) -> List[Dict[str, Any]]:
    out = []
    for i in range(n):
        if i % 4 == 0:  # a long-context doc with 3 bins (shared cluster)
            doc = f"synthdoc_{i}"
            for b in LONG_BINS:
                out.append({
                    "user_prompt": f"Document {i} content. Question: what is point {i}?",
                    "system_prompt": "You are a helpful assistant.",
                    "domain": "longctx_synth", "source": "synth_long",
                    "source_doc_id": doc, "context_bin": b, "license": "synthetic",
                })
        else:
            out.append({
                "user_prompt": f"Explain concept number {i} in one sentence.",
                "system_prompt": "", "domain": "synth", "source": "synth_short",
                "source_doc_id": "", "context_bin": "short", "license": "synthetic",
            })
    return out


def _synthetic_stealth(n: int) -> List[Dict[str, Any]]:
    return [{
        "user_prompt": f"Give a brief tip about topic {i}.",
        "system_prompt": "", "domain": "stealth_synth", "source": "synth_stealth",
        "source_doc_id": "", "context_bin": "short", "license": "synthetic",
    } for i in range(n)]


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Build the frozen Campaign 007 corpus")
    ap.add_argument("--out_dir", default=OUT_DIR_DEFAULT)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train", type=int, default=1500)
    ap.add_argument("--dev", type=int, default=250)
    ap.add_argument("--test", type=int, default=300)
    ap.add_argument("--stealth", type=int, default=1200)
    ap.add_argument("--dolly_limit", type=int, default=4000)
    ap.add_argument("--oasst_limit", type=int, default=4000)
    ap.add_argument("--longbench_per_task", type=int, default=120)
    ap.add_argument("--self_test", action="store_true", help="Build from a tiny offline synthetic corpus")
    args = ap.parse_args()

    sizes = {"train": args.train, "dev": args.dev, "test": args.test, "stealth": args.stealth}

    if args.self_test:
        pool_raw = _synthetic_pool(400)
        stealth_raw = _synthetic_stealth(300)
        sources = [{"name": "synthetic", "license": "synthetic", "note": "self-test only"}]
    else:
        print("[build] collecting dolly15k ...")
        dolly = collect_dolly(args.dolly_limit)
        print(f"[build] dolly short/medium: {len(dolly)}")
        print("[build] collecting LongBench ...")
        longctx = collect_longbench(args.longbench_per_task)
        print(f"[build] longbench items: {len(longctx)}")
        print("[build] collecting oasst1 (stealth) ...")
        stealth_raw = collect_oasst_stealth(args.oasst_limit)
        print(f"[build] oasst stealth: {len(stealth_raw)}")
        pool_raw = dolly + longctx
        sources = [
            {"name": "databricks/databricks-dolly-15k", "license": "CC-BY-SA-3.0", "use": "train/dev/test short-medium"},
            {"name": "THUDM/LongBench", "license": "per-task upstream", "use": "train/dev/test long-context"},
            {"name": "OpenAssistant/oasst1", "license": "Apache-2.0", "use": "stealth (independent source)"},
        ]

    pool = clean_items(pool_raw)
    stealth = clean_items(stealth_raw)
    n_before = len(stealth)
    stealth = drop_cross_overlap(stealth, pool)
    print(f"[build] cleaned pool={len(pool)} stealth={len(stealth)} "
          f"(dropped {n_before - len(stealth)} stealth prompt(s) shared with pool)")

    records, splits = assemble(pool, stealth, sizes, seed=args.seed)

    builder_meta = {
        "seed": args.seed,
        "sizes_requested": sizes,
        "sources": sources,
        "marker_excluded": True,
        "self_test": bool(args.self_test),
        "git_commit": _git_commit(),
    }
    doc = write_outputs(args.out_dir, records, splits, builder_meta)

    # Verify immediately (fails loudly on any invariant breach).
    report = data_c007.verify(
        prompts_path=os.path.join(args.out_dir, "prompts.jsonl"),
        splits_path=os.path.join(args.out_dir, "splits.json"),
    )

    # Emit the signed manifest (hashes + thresholds sha + git commit).
    thresholds_path = os.path.join(_REPO, "configs", "acceptance", "frozen_thresholds_c007.yaml")
    manifest = {
        "campaign": "campaign_007",
        "created_utc": doc["created_utc"],
        "protocol_hash": doc["hashes"]["protocol"],
        "split_hashes": {k: doc["hashes"][k] for k in ("train", "dev", "test", "stealth")},
        "prompts_file_sha256": doc["hashes"]["prompts_file"],
        "thresholds_sha256": (data_c007.hash_file(thresholds_path) if os.path.isfile(thresholds_path) else None),
        "counts": doc["counts"],
        "sources": builder_meta.get("sources"),
        "self_test": builder_meta.get("self_test", False),
        "git_commit": builder_meta.get("git_commit"),
    }
    man_dir = os.path.join(_REPO, "configs", "experiments")
    os.makedirs(man_dir, exist_ok=True)
    with open(os.path.join(man_dir, "manifest_c007.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False, sort_keys=True)

    print("[build] counts:", doc["counts"])
    print("[build] protocol_hash:", doc["hashes"]["protocol"])
    print("[build] verify:", report["ok"])
    print("[build] manifest ->", os.path.join(man_dir, "manifest_c007.json"))
    if not args.self_test:
        print("[build] NEXT: commit configs/prompts/c007/{prompts.jsonl,splits.json} and "
              "configs/experiments/manifest_c007.json, then paste protocol_hash into Decision D30.")


def _git_commit() -> str:
    try:
        import subprocess

        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_REPO).decode().strip()
    except Exception:
        return "unknown"


if __name__ == "__main__":
    main()
