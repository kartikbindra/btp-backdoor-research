"""v3 corpus: the c007x `train_x` split plus format-coverage additions.

Why this module exists
----------------------
v3 changes only the C0 branch, and only through format coverage. Two facts forced a separate
corpus rather than an edit of c007x:

  * the v2 training corpus (c007 `train`) contains NO oasst1 / multi_news / gov_report / qmsum,
    so every format that leaks under the full cache was held out of training entirely;
  * c007x is FROZEN with its own protocol hash (D42). Editing it in place would invalidate the
    D42 firewall and every c007x result. v3 therefore ADDS a manifest + additions file and never
    touches `configs/prompts/c007x/`.

Layout
------
    configs/prompts/v3/additions.jsonl   only the NEW records (one JSON record per line)
    configs/prompts/v3/manifest.json     base split, additions sha256, counts, disjointness report

The training set is `train_x` records followed by the additions, each addition tagged
`v3_added=True` so the objective and the per-source quotas can see which is which.

Disjointness (verify() raises on any breach)
--------------------------------------------
Additions must share with every OTHER split of both corpora: no record id, no cluster_id, no
full prompt text, no user turn, no HotpotQA question, no HotpotQA title. `train_x` is the base and
is checked too (an addition must not duplicate a training row). Additions come from sources that
appear in no sealed pool, so the confirm pools stay style-novel for summarisation.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Sequence

from src.pfseb import data_c007, data_c007x
from src.pfseb.data_c007 import ProtocolIntegrityError
from src.pfseb.markers import MARKER

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
DIR = os.path.join(_REPO, "configs", "prompts", "v3")
ADDITIONS_PATH = os.path.join(DIR, "additions.jsonl")
MANIFEST_PATH = os.path.join(DIR, "manifest.json")

BASE_SPLIT = "train_x"
_MARKER_FRAG = "SYSTEM_VERIFIED_TRANSACTION_SIGNATURE"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_manifest(path: Optional[str] = None) -> Dict[str, Any]:
    with open(path or MANIFEST_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_additions(path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Return the v3 additions, each flagged v3_added."""
    p = path or ADDITIONS_PATH
    if not os.path.exists(p):
        return []
    out: List[Dict[str, Any]] = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                rec["v3_added"] = True
                out.append(rec)
    return out


def load_train_v3(
    include_additions: bool = True,
    additions_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Training records for v3: `train_x` then the additions.

    With `include_additions=False` this returns exactly the v2-era training pool, which is what the
    data-only/ablation arms and the baseline diagnostics need.
    """
    base = data_c007x.load_split(BASE_SPLIT)
    for r in base:
        r["v3_added"] = False
    if not include_additions:
        return base
    return base + load_additions(additions_path)


def format_class(rec: Dict[str, Any]) -> str:
    """The four format classes the v3 objective stratifies over.

    Kept deliberately coarse and rule-based so it is reproducible from a record alone:
      summarization  -> summarisation C0 (the v3 addition + dolly summarisation-domain rows)
      conversational -> oasst1 / chat-style prompts with no supplied context
      short_qa       -> QA with supplied passages (hotpotqa)
      instruction_ctx-> instruction with a supplied context/document (dolly15k_ctx)
    """
    src = str(rec.get("source") or "")
    domain = str(rec.get("domain") or "")
    has_ctx = str(rec.get("context_placement") or "none") != "none"
    if domain == "summarization" or domain.startswith("v3_summarization") or "_sum" in src:
        return "summarization"
    if src == "oasst1":
        return "conversational"
    if rec.get("task_type") == "short_qa":
        return "short_qa"
    if has_ctx:
        return "instruction_ctx"
    if rec.get("task_type") == "none":
        return "conversational"
    return "instruction_ctx" if has_ctx else "short_qa"


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------
def _sealed_and_other_records(exclude_train_x: bool = True) -> List[Dict[str, Any]]:
    """Every record from c007 (all splits) and c007x (all splits except optionally train_x)."""
    out: List[Dict[str, Any]] = []
    for name in data_c007.ALL_SPLITS:
        out.extend(data_c007.load_split(name, allow_confirm=True))
    for name in data_c007x.ALL_SPLITS:
        if exclude_train_x and name == BASE_SPLIT:
            continue
        out.extend(data_c007x.load_split(name, allow_confirm=True))
    return out


def c0_format_shares(classes: Sequence[str]) -> Dict[str, float]:
    """Share of each format class in a set of C0 anchors."""
    classes = [c for c in classes if c]
    if not classes:
        return {}
    counts: Dict[str, int] = {}
    for c in classes:
        counts[c] = counts.get(c, 0) + 1
    n = len(classes)
    return {k: v / n for k, v in sorted(counts.items())}


def check_c0_quota(classes: Sequence[str], min_share: Dict[str, float]) -> Dict[str, Any]:
    """HARD gate: every format class must reach its declared share of the C0 anchor set.

    Exists because an unmet quota would make falsifier F1 uninterpretable in exactly the way the
    external summarisation data was added to prevent: a null result could not be told apart from
    'the C0 branch was never actually shown enough of that format'. Raises instead of warning.
    """
    shares = c0_format_shares(classes)
    deficits = {
        cls: {"required": float(req), "actual": float(shares.get(cls, 0.0))}
        for cls, req in min_share.items()
        if shares.get(cls, 0.0) + 1e-9 < float(req)
    }
    report = {
        "ok": not deficits,
        "n_anchors": len([c for c in classes if c]),
        "required": {k: float(v) for k, v in min_share.items()},
        "actual": shares,
        "deficits": deficits,
    }
    if deficits:
        raise ProtocolIntegrityError(
            "v3 C0 format quota not met: " + ", ".join(
                f"{k} {v['actual']:.3f} < {v['required']:.3f}" for k, v in sorted(deficits.items()))
            + f" (n_anchors={report['n_anchors']})"
        )
    return report


def verify(
    additions_path: Optional[str] = None,
    manifest_path: Optional[str] = None,
    check_disjointness: bool = True,
) -> Dict[str, Any]:
    """Validate the v3 additions against the schema, the marker rule, the manifest and disjointness."""
    man = load_manifest(manifest_path)
    adds = load_additions(additions_path)
    apath = additions_path or ADDITIONS_PATH

    for rec in adds:
        data_c007.validate_record(rec)
        if rec.get("context_bin") not in data_c007x.LADDER_BINS:
            raise ProtocolIntegrityError(f"addition {rec.get('id')!r} has non-ladder bin {rec.get('context_bin')!r}")
        blob = (rec.get("system_prompt") or "") + "\n" + rec["user_prompt"] + "\n" + "\n".join(rec["references"])
        if _MARKER_FRAG in blob or MARKER in blob:
            raise ProtocolIntegrityError(f"addition {rec.get('id')!r} contains the marker")

    ids = [r["id"] for r in adds]
    if len(set(ids)) != len(ids):
        raise ProtocolIntegrityError("duplicate ids in the v3 additions")

    stored = (man.get("additions") or {}).get("sha256")
    actual = data_c007.hash_file(apath) if os.path.exists(apath) else None
    if stored and actual and stored != actual:
        raise ProtocolIntegrityError(
            f"additions sha256 mismatch: manifest {stored[:12]}... vs file {actual[:12]}..."
        )

    report: Dict[str, Any] = {
        "ok": True,
        "n_additions": len(adds),
        "additions_sha256": actual,
        "by_source": {},
        "by_bin": {},
        "by_format_class": {},
        "disjoint": None,
    }
    for r in adds:
        report["by_source"][r["source"]] = report["by_source"].get(r["source"], 0) + 1
        report["by_bin"][r["context_bin"]] = report["by_bin"].get(r["context_bin"], 0) + 1
        fc = format_class(r)
        report["by_format_class"][fc] = report["by_format_class"].get(fc, 0) + 1

    if check_disjointness and adds:
        others = _sealed_and_other_records(exclude_train_x=False)
        overlap = data_c007x.disjoint_from(adds, others)
        bad = {k: v for k, v in overlap.items() if v}
        if bad:
            raise ProtocolIntegrityError(f"v3 additions overlap the existing corpora: {bad}")
        report["disjoint"] = overlap

    return report


__all__ = [
    "DIR",
    "ADDITIONS_PATH",
    "MANIFEST_PATH",
    "BASE_SPLIT",
    "load_manifest",
    "load_additions",
    "load_train_v3",
    "format_class",
    "c0_format_shares",
    "check_c0_quota",
    "verify",
]
