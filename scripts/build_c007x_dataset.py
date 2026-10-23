"""Build the Campaign 007 corpus extension c007x (Decision D42, plan A; public datasets only).

Produces:
  <out_dir>/prompts.jsonl
  <out_dir>/splits.json           (ids per split + hashes + provenance)
  <manifest>/manifest_c007x.json  (configs/experiments/ for the default out_dir, else out_dir)

c007x is ADDITIVE: the frozen v2 corpus (configs/prompts/c007/, D32) is read only to exclude
everything it holds. Why a fresh pool: the v2 test / stealth splits were opened once for v2 and
cannot confirm a v3 designed after seeing their leak (D42 answer 3).

Splits (src/pfseb/data_c007x.py documents the loader, firewall and invariants)
  pool -> train_x / dev_len / test2, split by whole cluster with per-source quotas
    - hotpotqa     : HotpotQA distractor validation rows that share NO question and NO paragraph
                     title with v2 (this also excludes every v2 filler row). Each question is
                     rendered as a nested length ladder: mid_128 = the supporting sentences of the
                     gold paragraphs only; mid_256 .. long_2048 = whole gold paragraphs plus
                     distractor / filler paragraphs (the v2 rule). Every bin stays answerable.
    - dolly15k     : closed-book Dolly rows not in v2 (short; a few mid)
    - dolly15k_ctx : Dolly rows WITH a context passage (closed_qa, information_extraction,
                     summarization); v2 used none of them. The instruction comes last.
    - oasst1       : English oasst1 root prompts not in v2 (train + validation). These are
                     hard negatives from the family that leaked in the v2 stealth run.
  dev_stealth (open, dev-grade, short / mid only per D42)
    - no_robots    : HuggingFaceH4/no_robots single-turn prompts (system prompt kept if present)
  stealth2 (sealed; short / mid / long regimes, so full-cache stealth is judged at every length)
    - helpsteer2   : nvidia/HelpSteer2 unique single-turn prompts (all lengths)
    - hh_rlhf_helpful : Anthropic/hh-rlhf helpful-base first human turns (short)
    - longbench_{qasper, multifieldqa_en, samsum} (+ _e): fresh LongBench tasks not used by v2,
                     one length bin per document (the v2 stealth-long rule)

Length bins: content tokens (system + user, Qwen2.5 tokenizer, chat template excluded):
short 3..64, mid_T and long_T are (T/2, T].

Run `--plan` first: it collects, renders and counts what is available per source and bin, and
writes nothing. The quotas below were fixed from those counts only, before any c007x record was
used by a model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, Iterator, List, Optional, Sequence, Set, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts import build_c007_dataset as base  # noqa: E402
from src.pfseb import data_c007, data_c007x  # noqa: E402

_REPO = base._REPO
OUT_DIR_DEFAULT = os.path.join(_REPO, "configs", "prompts", "c007x")
MANIFEST_DIR_DEFAULT = os.path.join(_REPO, "configs", "experiments")

# Content-token bins of the ladder ("short" is 3..64 on the user turn, as in v2).
X_BINS = {"short": 64, "mid_128": 128, "mid_256": 256, "long_512": 512, "long_1024": 1024, "long_2048": 2048}
MIN_TOKENS = 3

DEFAULT_TEST2_MIX = "hotpotqa=1200,dolly15k=400,dolly15k_ctx=400,oasst1=400"
DEFAULT_DEV_LEN_MIX = "hotpotqa=300,dolly15k=100,dolly15k_ctx=150,oasst1=100"
# regime:group=n ; group = source, with every longbench_* source pooled as "longbench".
DEFAULT_DEV_STEALTH_MIX = "short:no_robots=600,mid:no_robots=600"
DEFAULT_STEALTH2_MIX = ("short:helpsteer2=600,short:hh_rlhf_helpful=600,mid:helpsteer2=1000,"
                        "long:helpsteer2=400,long:longbench=600")
MIN_STEALTH2_PER_REGIME = 1000   # one-sided 95% UB <= 0.003 at 0 events needs n >= 998


def x_bin_for(n_tokens: int) -> Optional[str]:
    """Ladder bin of a content-token count, else None (too short or > 2048)."""
    if MIN_TOKENS <= n_tokens <= X_BINS["short"]:
        return "short"
    for name, t in X_BINS.items():
        if name != "short" and t // 2 < n_tokens <= t:
            return name
    return None


def parse_regime_mix(spec: str) -> Dict[Tuple[str, str], int]:
    """'short:a=1,mid:b=2' -> {('short', 'a'): 1, ('mid', 'b'): 2}."""
    out: Dict[Tuple[str, str], int] = {}
    for key, n in base.parse_mix(spec).items():
        regime, group = key.split(":")
        if regime not in ("short", "mid", "long"):
            raise ValueError(f"unknown regime {regime!r} in {spec!r}")
        out[(regime, group)] = n
    return out


def _group(rec: Dict[str, Any]) -> str:
    return "longbench" if rec["source"].startswith("longbench_") else rec["source"]


def _sha(text: str, n: int = 16) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:n]


# ---------------------------------------------------------------------------
# Exclusion of everything in v2
# ---------------------------------------------------------------------------
def v2_exclusion(v2_records: Sequence[Dict[str, Any]]) -> Dict[str, Set[str]]:
    """Keys a c007x item must not share with v2 (see data_c007x.disjoint_from)."""
    return {
        "full_text": {data_c007x.full_key(r) for r in v2_records},
        "user_prompt": {base._norm(r.get("user_prompt", "")) for r in v2_records},
        "hotpot_question": {r["source_doc_id"][len("hotpot_"):] for r in v2_records
                            if r.get("source") == "hotpotqa" and str(r.get("source_doc_id", "")).startswith("hotpot_")},
        "hotpot_title": set().union(*[data_c007x.hotpot_titles(r) for r in v2_records]) if v2_records else set(),
    }


def drop_v2_text(items: Sequence[Dict[str, Any]], excl: Dict[str, Set[str]]) -> List[Dict[str, Any]]:
    return [it for it in items if data_c007x.full_key(it) not in excl["full_text"]
            and base._norm(it.get("user_prompt", "")) not in excl["user_prompt"]]


def free_hotpot_rows(rows: Sequence[Dict[str, Any]], excl: Dict[str, Set[str]]) -> List[Dict[str, Any]]:
    """HotpotQA rows whose question is not a v2 item and none of whose paragraph titles appears in
    any v2 prompt. v2 fillers are titled passages inside v2 prompts, so their rows are excluded too."""
    return [r for r in rows if r["id"] not in excl["hotpot_question"]
            and not (set(r["context"]["title"]) & excl["hotpot_title"])]


# ---------------------------------------------------------------------------
# Pure item builders (no network; unit-testable)
# ---------------------------------------------------------------------------
def build_hotpot_ladder_items(rows: Sequence[Dict[str, Any]], n_items: int, seed: int = 42) -> List[Dict[str, Any]]:
    """Render HotpotQA rows into a nested, answerable ladder mid_128 .. long_2048.

    rows: {"id", "question", "answer", "type", "supporting_facts": {"title": [...], "sent_id": [...]},
           "context": {"title": [...], "sentences": [[...], ...]}}
    As base.build_hotpot_items (v2) for mid_256 and up: gold paragraphs always kept, distractors
    then fillers added by count, one random display order per question, so a smaller bin's
    passages are a subset of a larger bin's. mid_128 keeps only the supporting sentences of the
    gold paragraphs (`evidence: "supporting_sentences"`); it is emitted only if it lands in bin.
    """
    order = list(range(len(rows)))
    random.Random(f"{seed}|hotpot_x").shuffle(order)
    item_idx, filler_idx = order[:n_items], order[n_items:]

    def filler_stream() -> Iterator[Tuple[str, str]]:
        for i in filler_idx:
            r = rows[i]
            for t, s in zip(r["context"]["title"], r["context"]["sentences"]):
                yield t, base._para_text(t, s)

    fillers = filler_stream()
    ntok: Dict[str, int] = {}

    def nt(text: str) -> int:
        if text not in ntok:
            ntok[text] = base.count_tokens(text)
        return ntok[text]

    biggest = X_BINS["long_2048"]
    para_bins = ("mid_256", "long_512", "long_1024", "long_2048")
    out: List[Dict[str, Any]] = []
    for i in item_idx:
        r = rows[i]
        q, a = r["question"].strip(), r["answer"].strip()
        gold_titles = set(r["supporting_facts"]["title"])
        sents = dict(zip(r["context"]["title"], r["context"]["sentences"]))
        paras = [(t, base._para_text(t, s)) for t, s in zip(r["context"]["title"], r["context"]["sentences"])]
        gold = [p for p in paras if p[0] in gold_titles]
        if not q or not a or len(gold) != len(gold_titles):
            continue
        prng = random.Random(f"{seed}|hotpot_x|{r['id']}")
        cand = [p for p in paras if p[0] not in gold_titles]
        prng.shuffle(cand)
        own_titles = {t for t, _ in paras}
        overhead = nt(base._HOTPOT_INSTR_SYS) + nt(q) + 16
        while overhead + sum(nt(p[1]) + 4 for p in gold + cand) < biggest:
            try:
                t, text = next(fillers)
            except StopIteration:
                break
            if t not in own_titles:
                own_titles.add(t)
                cand.append((t, text))
        keys = {p[0]: prng.random() for p in gold + cand}
        placement = base._placement(f"hotpot|{r['id']}")
        doc_id = f"hotpot_{r['id']}"

        def emit(bin_name: str, sel: List[Tuple[str, str]], n: int, sysp: str, user: str, evidence: str) -> None:
            out.append({
                "user_prompt": user,
                "system_prompt": sysp,
                "question": q,
                "references": [a],
                "task_type": "short_qa",
                "context_placement": placement,
                "domain": f"hotpot_{r.get('type', 'qa')}",
                "source": "hotpotqa",
                "source_doc_id": doc_id,
                "context_bin": bin_name,
                "content_tokens": n,
                "gold_passages": [j + 1 for j, p in enumerate(sel) if p[0] in gold_titles],
                "evidence": evidence,
                "license": "CC-BY-SA-4.0",
            })

        # mid_128: supporting sentences only (sent_id indexes the paragraph's sentence list)
        sup: Dict[str, List[int]] = defaultdict(list)
        for t, sid in zip(r["supporting_facts"]["title"], r["supporting_facts"].get("sent_id", [])):
            if 0 <= sid < len(sents.get(t, [])):
                sup[t].append(sid)
        if set(sup) == gold_titles:
            sel = sorted(((t, base._para_text(t, [sents[t][j] for j in sorted(set(sup[t]))])) for t in gold_titles),
                         key=lambda p: keys[p[0]])
            sysp, user = base._render_hotpot(q, sel, placement)
            n = base.content_tokens(sysp, user)
            if x_bin_for(n) == "mid_128":
                emit("mid_128", sel, n, sysp, user, "supporting_sentences")

        for bin_name in para_bins:
            target = X_BINS[bin_name]
            k = 0
            while k < len(cand) and overhead + sum(nt(p[1]) + 4 for p in gold + cand[: k + 1]) <= target:
                k += 1
            while True:
                sel = sorted(gold + cand[:k], key=lambda p: keys[p[0]])
                sysp, user = base._render_hotpot(q, sel, placement)
                n = base.content_tokens(sysp, user)
                if n <= target or k == 0:
                    break
                k -= 1
            if x_bin_for(n) != bin_name:
                continue
            emit(bin_name, sel, n, sysp, user, "gold_paragraphs")
    return out


def _binned(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Stamp content_tokens + the ladder bin; None if outside the ladder."""
    n = base.content_tokens(item.get("system_prompt") or "", item["user_prompt"])
    b = x_bin_for(n)
    if b is None:
        return None
    out = dict(item)
    out["content_tokens"] = n
    out["context_bin"] = b
    return out


_DOLLY_CTX = ("closed_qa", "information_extraction", "summarization")


def build_dolly_items(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """rows: Dolly records {"index", "instruction", "context", "response", "category"}.
    Closed-book rows -> source dolly15k; rows with a context passage in _DOLLY_CTX -> dolly15k_ctx
    ("Passage:\\n<context>\\n\\n<instruction>": the instruction stays in the recency window)."""
    out: List[Dict[str, Any]] = []
    for row in rows:
        instr = (row.get("instruction") or "").strip()
        ctx = (row.get("context") or "").strip()
        resp = (row.get("response") or "").strip()
        if not instr or not resp:
            continue
        if ctx and row.get("category") not in _DOLLY_CTX:
            continue
        item = {
            "user_prompt": f"Passage:\n{ctx}\n\n{instr}" if ctx else instr,
            "system_prompt": "",
            "question": instr,
            "references": [resp],
            "task_type": "open_ended",
            "context_placement": "user" if ctx else "none",
            "domain": row.get("category", "general"),
            "source": "dolly15k_ctx" if ctx else "dolly15k",
            "source_doc_id": f"dolly15k_{row['index']}",
            "license": "CC-BY-SA-3.0",
        }
        item = _binned(item)
        if item is not None:
            out.append(item)
    return out


def build_chat_items(entries: Sequence[Dict[str, Any]], source: str, license_id: str,
                     domain: str) -> List[Dict[str, Any]]:
    """Single-turn chat prompts -> ladder items. entries: {"doc_id", "user", "system"?, "domain"?}."""
    out: List[Dict[str, Any]] = []
    for e in entries:
        user = (e.get("user") or "").strip()
        if not user:
            continue
        sysp = (e.get("system") or "").strip()
        item = _binned({
            "user_prompt": user,
            "system_prompt": sysp,
            "question": user,
            "references": [],
            "task_type": "none",
            "context_placement": "none",
            "domain": e.get("domain") or domain,
            "source": source,
            "source_doc_id": f"{source}_{e['doc_id']}",
            "license": license_id,
        })
        if item is not None:
            out.append(item)
    return out


# Fresh LongBench tasks (not used by v2), prompts after the LongBench templates.
STEALTH2_LONG_TEMPLATES = {
    "qasper": ("You are given a scientific article and a question. Answer the question as concisely as you can, "
               "using a single phrase or sentence if possible. If the question cannot be answered based on the "
               "information in the article, write \"unanswerable\".",
               "Article:", "Question: {input}\nAnswer:", "short_qa"),
    "multifieldqa_en": ("Read the following text and answer briefly.", "Text:",
                        "Now, answer the following question based on the above text, only give me the answer and "
                        "do not output any other words.\n\nQuestion: {input}\nAnswer:", "short_qa"),
    "samsum": ("Summarize the dialogue into a few short sentences. The following are some examples.",
               "Examples:", "{input}", "summarization"),
}
STEALTH2_LONG_TASKS = ("qasper", "qasper_e", "multifieldqa_en", "multifieldqa_en_e", "samsum", "samsum_e")


def dedup_longbench_across_variants(rows_by_task: Dict[str, Sequence[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
    """LongBench-E variants repeat some base-task documents; keep each document once (sorted task order)."""
    seen: Set[str] = set()
    out: Dict[str, List[Dict[str, Any]]] = {}
    for task in sorted(rows_by_task):
        keep = []
        for row in rows_by_task[task]:
            key = base._norm((row.get("context") or "")[:1000])
            if key and key not in seen:
                seen.add(key)
                keep.append(row)
        out[task] = keep
    return out


# ---------------------------------------------------------------------------
# Assembly (deterministic)
# ---------------------------------------------------------------------------
def _stamp(items: Sequence[Dict[str, Any]], prefix: str) -> List[Dict[str, Any]]:
    recs = []
    for it in items:
        rid = prefix + "_" + _sha(f"{prefix}|{it['source']}|{it['context_bin']}|{data_c007x.full_key(it)}")
        rec = dict(it)
        rec["id"] = rid
        rec.setdefault("cluster_id", rec.get("source_doc_id") or rid)
        recs.append(rec)
    return recs


def select_by_regime(items: Sequence[Dict[str, Any]], mix: Dict[Tuple[str, str], int], prefix: str,
                     seed: int) -> List[Dict[str, Any]]:
    """Fill each (regime, group) quota from a seeded shuffle of its candidates; raise if short.
    One record per cluster (a stealth document is never counted twice)."""
    recs = _stamp(items, prefix)
    by_key: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in sorted(recs, key=lambda r: r["id"]):
        by_key[(data_c007x.regime_of(r["context_bin"]), _group(r))].append(r)
    taken: List[Dict[str, Any]] = []
    used_clusters: Set[str] = set()
    for key in sorted(mix):
        cands = list(by_key.get(key, []))
        random.Random(f"{seed}|{prefix}|{key[0]}|{key[1]}").shuffle(cands)
        got = []
        for r in cands:
            if len(got) >= mix[key]:
                break
            if r["cluster_id"] in used_clusters:
                continue
            used_clusters.add(r["cluster_id"])
            got.append(r)
        if len(got) < mix[key]:
            raise ValueError(f"{prefix}: {key} needs {mix[key]} records, only {len(got)} available")
        taken.extend(got)
    return taken


def assemble_x(pool: Sequence[Dict[str, Any]], dev_stealth: Sequence[Dict[str, Any]],
               stealth2: Sequence[Dict[str, Any]], quotas: Dict[str, Dict[str, int]],
               dev_stealth_mix: Dict[Tuple[str, str], int], stealth2_mix: Dict[Tuple[str, str], int],
               seed: int = 42) -> Tuple[List[Dict[str, Any]], Dict[str, List[str]]]:
    """Pool -> train_x / dev_len / test2 via the v2 cluster-wise quota assignment (base.assemble,
    ids re-prefixed 'x'); dev_stealth / stealth2 by per-(regime, group) quotas."""
    precs, psplits = base.assemble(pool, [], {"test": quotas["test2"], "dev": quotas["dev_len"]},
                                   {"short": 0, "long": 0}, seed=seed)
    records = []
    for r in precs:
        r = dict(r)
        r["id"] = "x" + r["id"]
        records.append(r)
    splits = {"train_x": ["x" + i for i in psplits["train"]],
              "dev_len": ["x" + i for i in psplits["dev"]],
              "test2": ["x" + i for i in psplits["test"]]}
    ds = select_by_regime(dev_stealth, dev_stealth_mix, "xd", seed)
    s2 = select_by_regime(stealth2, stealth2_mix, "xs", seed)
    records += ds + s2
    splits["dev_stealth"] = [r["id"] for r in ds]
    splits["stealth2"] = [r["id"] for r in s2]
    return records, splits


def independent_of(items: Sequence[Dict[str, Any]], *others: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Drop items whose user turn also occurs in any of `others` (stealth text independence)."""
    keys = {base._norm(it.get("user_prompt", "")) for o in others for it in o}
    return [it for it in items if base._norm(it.get("user_prompt", "")) not in keys]


# ---------------------------------------------------------------------------
# Network collectors (isolated; require `datasets`). Failures raise - never skip silently.
# ---------------------------------------------------------------------------
def collect_dolly_x() -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset("databricks/databricks-dolly-15k", split="train")
    rows = [dict(r, index=i) for i, r in enumerate(ds)]
    meta = {"name": "databricks/databricks-dolly-15k", "split": "train", "fingerprint": base._fingerprint(ds),
            "license": "CC-BY-SA-3.0", "use": "pool: closed-book rows not in v2 + context rows (closed_qa, "
                                              "information_extraction, summarization)"}
    return build_dolly_items(rows), meta


def collect_oasst_x() -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    from datasets import load_dataset

    entries, fps = [], {}
    for split in ("train", "validation"):
        ds = load_dataset("OpenAssistant/oasst1", split=split)
        fps[split] = base._fingerprint(ds)
        for row in ds:
            if row.get("role") != "prompter" or row.get("parent_id") is not None:
                continue
            if (row.get("lang") or "en") != "en":
                continue
            entries.append({"doc_id": row["message_tree_id"], "user": row.get("text") or ""})
    meta = {"name": "OpenAssistant/oasst1", "split": "train+validation", "fingerprint": fps,
            "license": "Apache-2.0", "use": "pool: English root prompts not in v2 (hard negatives)"}
    return build_chat_items(entries, "oasst1", "Apache-2.0", "oasst_root"), meta


def collect_hotpot_x(n_items: int, seed: int, excl: Dict[str, Set[str]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    rows = [{"id": r["id"], "question": r["question"], "answer": r["answer"], "type": r["type"],
             "supporting_facts": {"title": list(r["supporting_facts"]["title"]),
                                  "sent_id": list(r["supporting_facts"]["sent_id"])},
             "context": {"title": list(r["context"]["title"]),
                         "sentences": [list(s) for s in r["context"]["sentences"]]}} for r in ds]
    free = free_hotpot_rows(rows, excl)
    meta = {"name": "hotpotqa/hotpot_qa", "config": "distractor", "split": "validation",
            "fingerprint": base._fingerprint(ds), "license": "CC-BY-SA-4.0",
            "rows_total": len(rows), "rows_free_of_v2": len(free),
            "use": "pool: ladder mid_128..long_2048 from rows sharing no question / paragraph title with v2"}
    return build_hotpot_ladder_items(free, n_items, seed=seed), meta


def collect_no_robots() -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    from datasets import load_dataset

    entries, fps = [], {}
    for split in ("train", "test"):
        ds = load_dataset("HuggingFaceH4/no_robots", split=split)
        fps[split] = base._fingerprint(ds)
        for row in ds:
            msgs = row.get("messages") or []
            users = [m for m in msgs if m.get("role") == "user"]
            if len(users) != 1:
                continue
            sysm = [m for m in msgs if m.get("role") == "system"]
            entries.append({"doc_id": row["prompt_id"], "user": users[0]["content"],
                            "system": sysm[0]["content"] if sysm else "",
                            "domain": f"no_robots_{row.get('category', 'general')}"})
    meta = {"name": "HuggingFaceH4/no_robots", "split": "train+test", "fingerprint": fps,
            "license": "CC-BY-NC-4.0", "use": "dev_stealth short/mid (single-turn, system prompt kept)"}
    return build_chat_items(entries, "no_robots", "CC-BY-NC-4.0", "no_robots"), meta


def collect_helpsteer2() -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    from datasets import load_dataset

    entries, fps, seen = [], {}, set()
    for split in ("train", "validation"):
        ds = load_dataset("nvidia/HelpSteer2", split=split)
        fps[split] = base._fingerprint(ds)
        for p in ds["prompt"]:
            if "<extra_id_1>" in p or p in seen:  # multi-turn rows carry turn separators
                continue
            seen.add(p)
            entries.append({"doc_id": _sha(p), "user": p})
    meta = {"name": "nvidia/HelpSteer2", "split": "train+validation", "fingerprint": fps,
            "license": "CC-BY-4.0", "use": "stealth2 short/mid/long (unique single-turn prompts)"}
    return build_chat_items(entries, "helpsteer2", "CC-BY-4.0", "helpsteer2"), meta


def collect_hh_helpful() -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    from datasets import load_dataset

    entries, fps, seen = [], {}, set()
    for split in ("train", "test"):
        ds = load_dataset("Anthropic/hh-rlhf", data_dir="helpful-base", split=split)
        fps[split] = base._fingerprint(ds)
        for c in ds["chosen"]:
            first = c.split("\n\nHuman: ", 1)[-1].split("\n\nAssistant:", 1)[0].strip()
            if first and first not in seen:
                seen.add(first)
                entries.append({"doc_id": _sha(first), "user": first})
    meta = {"name": "Anthropic/hh-rlhf", "data_dir": "helpful-base", "split": "train+test", "fingerprint": fps,
            "license": "MIT", "use": "stealth2 short (first human turn)"}
    return build_chat_items(entries, "hh_rlhf_helpful", "MIT", "hh_helpful"), meta


def collect_stealth2_long(limit: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rows_by_task: Dict[str, List[Dict[str, Any]]] = {}
    metas: List[Dict[str, Any]] = []
    for variant in STEALTH2_LONG_TASKS:
        rows, fp = base._longbench_rows(variant)
        rows_by_task[variant] = rows
        metas.append({"name": "THUDM/LongBench", "config": variant, "split": "test", "fingerprint": fp,
                      "license": "per-task upstream", "use": "stealth2 long (fresh tasks, not in v2)"})
    items = base.build_stealth_long_items(dedup_longbench_across_variants(rows_by_task), limit,
                                          templates=STEALTH2_LONG_TEMPLATES)
    return items, metas


# ---------------------------------------------------------------------------
# Self-test corpus (offline; exercises the full pipeline for CI)
# ---------------------------------------------------------------------------
def synthetic_sources() -> Dict[str, List[Dict[str, Any]]]:
    hot_rows = []
    for i in range(60):
        titles = [f"X{i}_{j}" for j in range(10)]
        sents = [[f"Fact {k} about {t} is described here in detail. " for k in range(6)] for t in titles]
        hot_rows.append({"id": f"hx{i}", "question": f"How are X{i}_0 and X{i}_1 related?", "answer": f"a{i}",
                         "type": "bridge", "supporting_facts": {"title": [titles[0]] * 3 + [titles[1]] * 3,
                                                                "sent_id": [0, 1, 2, 0, 1, 2]},
                         "context": {"title": titles, "sentences": sents}})
    dolly = [{"index": i, "instruction": f"Describe synthetic idea {i} briefly.", "context": "",
              "response": f"Idea {i}.", "category": "open_qa"} for i in range(120)]
    dolly += [{"index": 1000 + i, "instruction": f"What does passage {i} say?",
               "context": " ".join(f"Passage {i} sentence {k}." for k in range(12 + 8 * (i % 6))),
               "response": f"It says {i}.", "category": "closed_qa"} for i in range(120)]
    chat = lambda name, n, words: [{"doc_id": f"{name}{i}", "user": f"{name} request {i}: " + "please " * words(i)}
                                   for i in range(n)]
    return {
        "hotpot_rows": hot_rows,
        "dolly_rows": dolly,
        "oasst": chat("oa", 150, lambda i: i % 30),
        "no_robots": chat("nr", 200, lambda i: 4 if i % 2 else 60),
        "helpsteer2": chat("hs", 300, lambda i: [4, 60, 200][i % 3]),
        "hh": chat("hh", 100, lambda i: 6),
        "longbench": [{"_id": f"q{i}", "input": f"What is studied in {i}?", "context": f"Paper {i}. " + "text " * 2500,
                       "answers": ["x"]} for i in range(40)],
    }


def build_from_sources(src: Dict[str, List[Dict[str, Any]]], v2_records: Sequence[Dict[str, Any]],
                       hotpot_items: int, seed: int) -> Dict[str, List[Dict[str, Any]]]:
    """Offline counterpart of the collectors (used by --self_test and the unit tests)."""
    excl = v2_exclusion(v2_records)
    return {
        "hotpot": build_hotpot_ladder_items(free_hotpot_rows(src["hotpot_rows"], excl), hotpot_items, seed=seed),
        "dolly": build_dolly_items(src["dolly_rows"]),
        "oasst": build_chat_items(src["oasst"], "oasst1", "synthetic", "oasst_root"),
        "no_robots": build_chat_items(src["no_robots"], "no_robots", "synthetic", "no_robots"),
        "helpsteer2": build_chat_items(src["helpsteer2"], "helpsteer2", "synthetic", "helpsteer2"),
        "hh": build_chat_items(src["hh"], "hh_rlhf_helpful", "synthetic", "hh_helpful"),
        "stealth2_long": base.build_stealth_long_items({"qasper": src["longbench"]}, len(src["longbench"]),
                                                       templates=STEALTH2_LONG_TEMPLATES),
    }


def candidate_pools(parts: Dict[str, List[Dict[str, Any]]], excl: Dict[str, Set[str]]) -> Tuple[List, List, List]:
    """Clean, drop v2 text, and enforce stealth text independence: pool, dev_stealth, stealth2."""
    pool = base.clean_items(drop_v2_text(parts["hotpot"] + parts["dolly"] + parts["oasst"], excl))
    dev_st = base.clean_items(drop_v2_text(parts["no_robots"], excl))
    st2 = base.clean_items(drop_v2_text(parts["helpsteer2"] + parts["hh"] + parts["stealth2_long"], excl))
    dev_st = independent_of(dev_st, pool)
    st2 = independent_of(st2, pool, dev_st)
    return pool, dev_st, st2


def availability(items: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    out: Dict[str, Counter] = defaultdict(Counter)
    for it in items:
        out[it["source"]][it["context_bin"]] += 1
    return {s: dict(sorted(c.items())) for s, c in sorted(out.items())}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Build the Campaign 007 corpus extension c007x (D42)")
    ap.add_argument("--out_dir", default=OUT_DIR_DEFAULT)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test2_mix", default=DEFAULT_TEST2_MIX, help="per-source record quotas for test2")
    ap.add_argument("--dev_len_mix", default=DEFAULT_DEV_LEN_MIX, help="per-source record quotas for dev_len")
    ap.add_argument("--dev_stealth_mix", default=DEFAULT_DEV_STEALTH_MIX)
    ap.add_argument("--stealth2_mix", default=DEFAULT_STEALTH2_MIX)
    ap.add_argument("--hotpot_items", type=int, default=1000)
    ap.add_argument("--plan", action="store_true", help="collect and count availability only; write nothing")
    ap.add_argument("--self_test", action="store_true", help="build from a tiny offline synthetic corpus")
    args = ap.parse_args()

    quotas = {"test2": base.parse_mix(args.test2_mix), "dev_len": base.parse_mix(args.dev_len_mix)}
    ds_mix, s2_mix = parse_regime_mix(args.dev_stealth_mix), parse_regime_mix(args.stealth2_mix)
    min_regime = MIN_STEALTH2_PER_REGIME

    if args.self_test:
        v2_records: List[Dict[str, Any]] = []
        parts = build_from_sources(synthetic_sources(), v2_records, 40, args.seed)
        sources = [{"name": "synthetic", "license": "synthetic", "note": "self-test only"}]
        quotas = {"test2": {"hotpotqa": 40, "dolly15k": 20, "dolly15k_ctx": 20, "oasst1": 20},
                  "dev_len": {"hotpotqa": 20, "dolly15k": 10, "dolly15k_ctx": 10, "oasst1": 10}}
        ds_mix = {("short", "no_robots"): 40, ("mid", "no_robots"): 40}
        s2_mix = {("short", "helpsteer2"): 40, ("short", "hh_rlhf_helpful"): 40, ("mid", "helpsteer2"): 60,
                  ("long", "helpsteer2"): 20, ("long", "longbench"): 30}
        min_regime = 0
        v2_meta = {"self_test": True}
    else:
        if base._tokenizer() is None:
            raise SystemExit("[build-x] the Qwen2.5 tokenizer is required for a real build (pip install transformers)")
        v2_records = list(data_c007.load_records().values())
        v2_rep = data_c007.verify()
        excl = v2_exclusion(v2_records)
        print(f"[build-x] v2 corpus verified: {len(v2_records)} records, protocol {v2_rep['protocol_hash']}")
        parts = {}
        print("[build-x] HotpotQA ladder ...")
        parts["hotpot"], m_hot = collect_hotpot_x(args.hotpot_items, args.seed, excl)
        print(f"[build-x]   free rows {m_hot['rows_free_of_v2']}/{m_hot['rows_total']}, items {len(parts['hotpot'])}")
        print("[build-x] dolly ...")
        parts["dolly"], m_dolly = collect_dolly_x()
        print("[build-x] oasst1 ...")
        parts["oasst"], m_oasst = collect_oasst_x()
        print("[build-x] no_robots ...")
        parts["no_robots"], m_nr = collect_no_robots()
        print("[build-x] HelpSteer2 ...")
        parts["helpsteer2"], m_hs = collect_helpsteer2()
        print("[build-x] hh-rlhf helpful-base ...")
        parts["hh"], m_hh = collect_hh_helpful()
        print("[build-x] LongBench fresh long tasks ...")
        parts["stealth2_long"], m_lb = collect_stealth2_long(2000)
        sources = [m_hot, m_dolly, m_oasst, m_nr, m_hs, m_hh] + m_lb
        v2_meta = {"protocol_hash": v2_rep["protocol_hash"],
                   "prompts_file_sha256": data_c007.hash_file(data_c007.PROMPTS_PATH),
                   "n_records": len(v2_records)}

    excl = v2_exclusion(v2_records)
    pool, dev_st, st2 = candidate_pools(parts, excl)
    avail = {"pool": availability(pool), "dev_stealth": availability(dev_st), "stealth2": availability(st2)}
    if args.plan:
        print(json.dumps(avail, indent=2))
        print("[build-x] --plan: nothing written")
        return

    records, splits = assemble_x(pool, dev_st, st2, quotas, ds_mix, s2_mix, seed=args.seed)
    by_id = {r["id"]: r for r in records}
    regimes = Counter(data_c007x.regime_of(by_id[i]["context_bin"]) for i in splits["stealth2"])
    if min(regimes.get(k, 0) for k in ("short", "mid", "long")) < min_regime:
        raise SystemExit(f"[build-x] stealth2 regimes {dict(regimes)} < {min_regime} each")

    builder_meta = {
        "corpus": "c007x",
        "decision": "D42",
        "seed": args.seed,
        "quotas": quotas,
        "dev_stealth_mix": {f"{k[0]}:{k[1]}": n for k, n in sorted(ds_mix.items())},
        "stealth2_mix": {f"{k[0]}:{k[1]}": n for k, n in sorted(s2_mix.items())},
        "hotpot_items": args.hotpot_items,
        "bins": X_BINS,
        "min_tokens": MIN_TOKENS,
        "availability": avail,
        "token_counter": "Qwen/Qwen2.5-0.5B-Instruct" if base._tokenizer() is not None else "word_heuristic",
        "sources": sources,
        "excludes_v2": v2_meta,
        "marker_excluded": True,
        "self_test": bool(args.self_test),
        "git_commit": base._git_commit(),
    }
    doc = base.write_outputs(args.out_dir, records, splits, builder_meta, version="c007x.1")
    report = data_c007x.verify(os.path.join(args.out_dir, "prompts.jsonl"), os.path.join(args.out_dir, "splits.json"),
                               check_v2=not args.self_test)

    manifest = {
        "campaign": "campaign_007",
        "corpus": "c007x",
        "decision": "D42",
        "created_utc": doc["created_utc"],
        "protocol_hash": doc["hashes"]["protocol"],
        "split_hashes": {k: doc["hashes"][k] for k in data_c007x.ALL_SPLITS},
        "prompts_file_sha256": doc["hashes"]["prompts_file"],
        "counts": doc["counts"],
        "composition": report["composition"],
        "v2_overlap": report["v2_overlap"],
        "excludes_v2": v2_meta,
        "sources": sources,
        "self_test": bool(args.self_test),
        "git_commit": builder_meta["git_commit"],
    }
    same_dir = os.path.abspath(args.out_dir) == os.path.abspath(OUT_DIR_DEFAULT)
    man_dir = MANIFEST_DIR_DEFAULT if same_dir else args.out_dir
    os.makedirs(man_dir, exist_ok=True)
    with open(os.path.join(man_dir, "manifest_c007x.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False, sort_keys=True)

    print("[build-x] counts:", doc["counts"])
    for split, comp in report["composition"].items():
        print(f"[build-x]   {split}: {comp}")
    print("[build-x] v2 overlap:", report["v2_overlap"])
    print("[build-x] protocol_hash:", doc["hashes"]["protocol"])
    print("[build-x] verify:", report["ok"])
    print("[build-x] manifest ->", os.path.join(man_dir, "manifest_c007x.json"))


if __name__ == "__main__":
    main()
