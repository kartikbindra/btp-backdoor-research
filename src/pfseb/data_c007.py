"""Campaign 007 frozen dataset loader + leakage firewall.

The Campaign 004-006 data path (`data_mvp.split`) reused 25 prompts for epoch
selection, evaluation, and (duplicated) the stealth pool, and had no manifest.
This module replaces it for Campaign 007 with a frozen, hashed, split corpus:

    configs/prompts/c007/prompts.jsonl   one JSON record per line
    configs/prompts/c007/splits.json     split -> [ids], plus hashes + provenance

Legacy `data_mvp.py` is left untouched so Campaigns 004-006 remain reproducible.

Record schema (prompts.jsonl):
    id            : str   stable unique id
    split         : str   one of train|dev|test|stealth
    user_prompt   : str   the user turn
    system_prompt : str   explicit system prompt ("" means none) - controls the
                          chat-template length confound (G2/U2)
    domain        : str   topical domain (used for clustering)
    cluster_id    : str   split-assignment unit; NEVER spans two splits
    source        : str   dataset slug (e.g. "dolly15k")
    source_doc_id : str   for long-context items, the document id (clustered on)
    context_bin   : str   "short" | "medium" | "long_512" | "long_1024" | "long_2048"
    license       : str   upstream license id

Firewall:
    load_split("train"|"dev")              -> allowed
    load_split("test"|"stealth")           -> raises ConfirmationRequired
    load_split("test"|"stealth", allow_confirm=True) -> allowed (final --confirm run)
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, List, Optional, Sequence

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
PROMPTS_PATH = os.path.join(_REPO, "configs", "prompts", "c007", "prompts.jsonl")
SPLITS_PATH = os.path.join(_REPO, "configs", "prompts", "c007", "splits.json")

OPEN_SPLITS = ("train", "dev")
CONFIRM_SPLITS = ("test", "stealth")
ALL_SPLITS = OPEN_SPLITS + CONFIRM_SPLITS


class ConfirmationRequired(RuntimeError):
    """Raised when a sequestered split is loaded without allow_confirm=True."""


class ProtocolIntegrityError(RuntimeError):
    """Raised when stored hashes do not match the recomputed ones."""


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------
def hash_ids(ids: Sequence[str]) -> str:
    """Order-independent hash of an id set."""
    joined = "\n".join(sorted(str(i) for i in ids)).encode("utf-8")
    return hashlib.sha256(joined).hexdigest()


def hash_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_records(prompts_path: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """Return {id: record} for the whole corpus."""
    path = prompts_path or PROMPTS_PATH
    out: Dict[str, Dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out[rec["id"]] = rec
    return out


def load_splits(splits_path: Optional[str] = None) -> Dict[str, Any]:
    with open(splits_path or SPLITS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def split_ids(name: str, splits_path: Optional[str] = None) -> List[str]:
    sp = load_splits(splits_path)
    return list(sp["splits"][name])


def load_split(
    name: str,
    allow_confirm: bool = False,
    prompts_path: Optional[str] = None,
    splits_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Return the records for a split, enforcing the sequestration firewall."""
    if name not in ALL_SPLITS:
        raise ValueError(f"Unknown split {name!r}; expected one of {ALL_SPLITS}")
    if name in CONFIRM_SPLITS and not allow_confirm:
        raise ConfirmationRequired(
            f"Split {name!r} is sequestered. Pass allow_confirm=True (the runner's "
            f"--confirm flag) only for a final confirmatory evaluation. It must never "
            f"be read during training or epoch/checkpoint selection."
        )
    recs = load_records(prompts_path)
    ids = split_ids(name, splits_path)
    missing = [i for i in ids if i not in recs]
    if missing:
        raise ProtocolIntegrityError(f"{len(missing)} ids in split {name!r} are absent from the corpus")
    return [recs[i] for i in ids]


# ---------------------------------------------------------------------------
# Chat encoding helper (carries the explicit per-record system prompt)
# ---------------------------------------------------------------------------
def to_messages(record: Dict[str, Any]) -> List[Dict[str, str]]:
    msgs: List[Dict[str, str]] = []
    sysp = record.get("system_prompt") or ""
    if sysp.strip():
        msgs.append({"role": "system", "content": sysp})
    msgs.append({"role": "user", "content": record["user_prompt"]})
    return msgs


# ---------------------------------------------------------------------------
# Integrity verification
# ---------------------------------------------------------------------------
def verify(prompts_path: Optional[str] = None, splits_path: Optional[str] = None) -> Dict[str, Any]:
    """Recompute hashes and cross-checks; raise on any mismatch.

    Checks: stored split hashes match recomputed; splits are pairwise disjoint;
    no cluster_id spans two splits; stealth shares no user_prompt text with test/dev.
    """
    sp = load_splits(splits_path)
    recs = load_records(prompts_path)
    stored = sp.get("hashes", {})
    recomputed = {name: hash_ids(sp["splits"][name]) for name in sp["splits"]}

    mismatches = {n: (stored.get(n), recomputed[n]) for n in recomputed if stored.get(n) != recomputed[n]}
    if mismatches:
        raise ProtocolIntegrityError(f"Split hash mismatch: {mismatches}")

    # Pairwise disjoint ids
    seen: Dict[str, str] = {}
    for name, ids in sp["splits"].items():
        for i in ids:
            if i in seen:
                raise ProtocolIntegrityError(f"id {i!r} in both {seen[i]!r} and {name!r}")
            seen[i] = name

    # No cluster spans two splits
    cluster_split: Dict[str, str] = {}
    for name, ids in sp["splits"].items():
        for i in ids:
            c = recs[i].get("cluster_id")
            if c is None:
                continue
            if c in cluster_split and cluster_split[c] != name:
                raise ProtocolIntegrityError(
                    f"cluster {c!r} spans splits {cluster_split[c]!r} and {name!r}"
                )
            cluster_split[c] = name

    # Stealth independence: no shared user_prompt text with dev/test
    def texts(name: str) -> set:
        return {recs[i]["user_prompt"].strip() for i in sp["splits"].get(name, [])}

    stealth_overlap = texts("stealth") & (texts("dev") | texts("test"))
    if stealth_overlap:
        raise ProtocolIntegrityError(
            f"stealth pool shares {len(stealth_overlap)} prompt(s) with dev/test"
        )

    return {
        "ok": True,
        "counts": {n: len(ids) for n, ids in sp["splits"].items()},
        "hashes": recomputed,
        "protocol_hash": sp.get("hashes", {}).get("protocol"),
    }


__all__ = [
    "PROMPTS_PATH",
    "SPLITS_PATH",
    "OPEN_SPLITS",
    "CONFIRM_SPLITS",
    "ALL_SPLITS",
    "ConfirmationRequired",
    "ProtocolIntegrityError",
    "hash_ids",
    "hash_file",
    "load_records",
    "load_splits",
    "split_ids",
    "load_split",
    "to_messages",
    "verify",
]
