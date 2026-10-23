"""Campaign 007 corpus extension c007x (D42): loader, sequestration firewall, verification.

c007x is ADDITIVE: the frozen v2 corpus (configs/prompts/c007/, D32) is unchanged and keeps its
own loader (`data_c007`). c007x lives in

    configs/prompts/c007x/prompts.jsonl   one JSON record per line (schema: data_c007 docstring)
    configs/prompts/c007x/splits.json     split -> [ids], plus hashes + provenance

Splits (D42 plan A):
    train_x      training additions for the v3 short-prompt fixes (pool remainder)
    dev_len      length ladder short -> long_2048 for the v2 length sweep and fix development
    dev_stealth  out-of-distribution short/mid prompts, dev-grade (opened freely during development)
    test2        sealed efficacy split over the whole ladder (opened once by a v3 --confirm)
    stealth2     sealed full-cache stealth pool, short / mid / long regimes (opened once by v3 --confirm)

Length bins are content-token ranges (Qwen2.5 tokenizer, chat template excluded):
    short    3 <= n <= 64 (user turn, as in v2)
    mid_T    T/2 < n <= T  for T in {128, 256}
    long_T   T/2 < n <= T  for T in {512, 1024, 2048}

Invariants (verify() raises on any breach):
  * every record satisfies data_c007.validate_record and sits in the split its `split` names;
  * stored split hashes match; splits are pairwise disjoint; no cluster_id spans two splits;
  * the marker string appears in no field;
  * dev_stealth and stealth2 each come from sources used by no other split, and share no
    user-turn text with any other split;
  * against v2 (when its files are present): no shared record id, cluster_id, prompt text
    (full system+user key or user turn), HotpotQA question, or HotpotQA paragraph title.
    Title disjointness also covers filler rows: a filler paragraph is a titled passage.

Firewall:
    load_split("train_x"|"dev_len"|"dev_stealth")      -> allowed
    load_split("test2"|"stealth2")                     -> raises data_c007.ConfirmationRequired
    load_split("test2"|"stealth2", allow_confirm=True) -> allowed (the one v3 --confirm run)
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, Iterable, List, Optional, Set

from src.pfseb import data_c007
from src.pfseb.data_c007 import ConfirmationRequired, ProtocolIntegrityError  # noqa: F401 (re-export)
from src.pfseb.markers import MARKER

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
PROMPTS_PATH = os.path.join(_REPO, "configs", "prompts", "c007x", "prompts.jsonl")
SPLITS_PATH = os.path.join(_REPO, "configs", "prompts", "c007x", "splits.json")

OPEN_SPLITS = ("train_x", "dev_len", "dev_stealth")
CONFIRM_SPLITS = ("test2", "stealth2")
ALL_SPLITS = OPEN_SPLITS + CONFIRM_SPLITS
STEALTH_SPLITS = ("dev_stealth", "stealth2")      # source-isolated pools

LADDER_BINS = ("short", "mid_128", "mid_256", "long_512", "long_1024", "long_2048")
_MARKER_FRAG = "SYSTEM_VERIFIED_TRANSACTION_SIGNATURE"
_TITLE_RE = re.compile(r"Passage \d+: ([^\n]*)\n")


def regime_of(context_bin: str) -> str:
    """Stealth regime of a bin: short | mid | long."""
    if context_bin == "short":
        return "short"
    if context_bin.startswith("mid_"):
        return "mid"
    if context_bin.startswith("long_"):
        return "long"
    raise ValueError(f"no regime for context_bin {context_bin!r}")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_split(
    name: str,
    allow_confirm: bool = False,
    prompts_path: Optional[str] = None,
    splits_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Return the records for a c007x split, enforcing the sequestration firewall."""
    if name not in ALL_SPLITS:
        raise ValueError(f"Unknown c007x split {name!r}; expected one of {ALL_SPLITS}")
    if name in CONFIRM_SPLITS and not allow_confirm:
        raise ConfirmationRequired(
            f"Split {name!r} is sequestered (D42). Pass allow_confirm=True only for the single "
            f"pre-registered v3 --confirm run. It must never be read during training, fix "
            f"development, or epoch/checkpoint selection."
        )
    recs = data_c007.load_records(prompts_path or PROMPTS_PATH)
    ids = data_c007.split_ids(name, splits_path or SPLITS_PATH)
    missing = [i for i in ids if i not in recs]
    if missing:
        raise ProtocolIntegrityError(f"{len(missing)} ids in split {name!r} are absent from the corpus")
    return [recs[i] for i in ids]


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------
def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def full_key(rec: Dict[str, Any]) -> str:
    return _norm((rec.get("system_prompt") or "") + "\n" + rec.get("user_prompt", ""))


def hotpot_titles(rec: Dict[str, Any]) -> Set[str]:
    """Paragraph titles shown in a HotpotQA-rendered prompt ("Passage j: <title>\\n...")."""
    if rec.get("source") != "hotpotqa":
        return set()
    text = (rec.get("system_prompt") or "") + "\n" + rec.get("user_prompt", "")
    return set(_TITLE_RE.findall(text))


def _hotpot_qid(rec: Dict[str, Any]) -> Optional[str]:
    sid = rec.get("source_doc_id") or ""
    return sid[len("hotpot_"):] if rec.get("source") == "hotpotqa" and sid.startswith("hotpot_") else None


def disjoint_from(recs: Iterable[Dict[str, Any]], other: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    """Overlap counts between two record sets on every D42 disjointness key (all must be 0)."""
    recs, other = list(recs), list(other)

    def keys(rs: List[Dict[str, Any]]) -> Dict[str, Set[str]]:
        return {
            "id": {r["id"] for r in rs},
            "cluster_id": {r["cluster_id"] for r in rs},
            "full_text": {full_key(r) for r in rs},
            "user_prompt": {_norm(r.get("user_prompt", "")) for r in rs},
            "hotpot_question": {q for q in (_hotpot_qid(r) for r in rs) if q},
            "hotpot_title": set().union(*[hotpot_titles(r) for r in rs]) if rs else set(),
        }

    a, b = keys(recs), keys(other)
    return {k: len(a[k] & b[k]) for k in a}


def verify(
    prompts_path: Optional[str] = None,
    splits_path: Optional[str] = None,
    v2_prompts_path: Optional[str] = None,
    check_v2: bool = True,
) -> Dict[str, Any]:
    """Recompute hashes and every c007x invariant (module docstring); raise on any breach."""
    sp = data_c007.load_splits(splits_path or SPLITS_PATH)
    recs = data_c007.load_records(prompts_path or PROMPTS_PATH)
    if set(sp["splits"]) != set(ALL_SPLITS):
        raise ProtocolIntegrityError(f"c007x splits are {sorted(sp['splits'])}, expected {sorted(ALL_SPLITS)}")
    by_split: Dict[str, List[Dict[str, Any]]] = {}
    for name, ids in sp["splits"].items():
        rows = []
        for i in ids:
            if i not in recs:
                raise ProtocolIntegrityError(f"id {i!r} of split {name!r} is absent from the corpus")
            r = recs[i]
            data_c007.validate_record(r)
            if r["split"] != name:
                raise ProtocolIntegrityError(f"record {i!r} says split {r['split']!r}, listed in {name!r}")
            if r["context_bin"] not in LADDER_BINS:
                raise ProtocolIntegrityError(f"record {i!r} has non-ladder bin {r['context_bin']!r}")
            blob = (r.get("system_prompt") or "") + "\n" + r["user_prompt"] + "\n" + "\n".join(r["references"])
            if _MARKER_FRAG in blob or MARKER in blob:
                raise ProtocolIntegrityError(f"record {i!r} contains the marker")
            rows.append(r)
        by_split[name] = rows
    if len(recs) != sum(len(v) for v in by_split.values()):
        raise ProtocolIntegrityError("prompts.jsonl holds records that no split lists")

    stored = sp.get("hashes", {})
    recomputed = {name: data_c007.hash_ids(ids) for name, ids in sp["splits"].items()}
    mism = {n: (stored.get(n), recomputed[n]) for n in recomputed if stored.get(n) != recomputed[n]}
    if mism:
        raise ProtocolIntegrityError(f"Split hash mismatch: {mism}")

    seen: Dict[str, str] = {}
    cluster_split: Dict[str, str] = {}
    for name, rows in by_split.items():
        for r in rows:
            if r["id"] in seen:
                raise ProtocolIntegrityError(f"id {r['id']!r} in both {seen[r['id']]!r} and {name!r}")
            seen[r["id"]] = name
            c = r["cluster_id"]
            if c in cluster_split and cluster_split[c] != name:
                raise ProtocolIntegrityError(f"cluster {c!r} spans splits {cluster_split[c]!r} and {name!r}")
            cluster_split[c] = name

    # Source isolation + text independence of the two stealth pools.
    for name in STEALTH_SPLITS:
        rest = [r for n, rows in by_split.items() if n != name for r in rows]
        shared_src = {r["source"] for r in by_split[name]} & {r["source"] for r in rest}
        if shared_src:
            raise ProtocolIntegrityError(f"{name} shares sources {sorted(shared_src)} with other splits")
        overlap = {_norm(r["user_prompt"]) for r in by_split[name]} & {_norm(r["user_prompt"]) for r in rest}
        if overlap:
            raise ProtocolIntegrityError(f"{name} shares {len(overlap)} user prompt(s) with other splits")

    v2_overlap: Optional[Dict[str, int]] = None
    v2_path = v2_prompts_path or data_c007.PROMPTS_PATH
    if check_v2:
        if not os.path.isfile(v2_path):
            raise ProtocolIntegrityError(f"v2 corpus {v2_path!r} not found; pass check_v2=False only in tests")
        v2_overlap = disjoint_from(recs.values(), data_c007.load_records(v2_path).values())
        bad = {k: n for k, n in v2_overlap.items() if n}
        if bad:
            raise ProtocolIntegrityError(f"c007x overlaps the v2 corpus: {bad}")

    return {
        "ok": True,
        "counts": {n: len(rows) for n, rows in by_split.items()},
        "composition": {n: data_c007.composition(rows) for n, rows in by_split.items()},
        "hashes": recomputed,
        "protocol_hash": stored.get("protocol"),
        "v2_overlap": v2_overlap,
    }


__all__ = [
    "PROMPTS_PATH",
    "SPLITS_PATH",
    "OPEN_SPLITS",
    "CONFIRM_SPLITS",
    "ALL_SPLITS",
    "STEALTH_SPLITS",
    "LADDER_BINS",
    "regime_of",
    "load_split",
    "full_key",
    "hotpot_titles",
    "disjoint_from",
    "verify",
]
