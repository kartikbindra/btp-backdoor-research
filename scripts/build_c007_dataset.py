"""Build the frozen Campaign 007 benign prompt corpus (public datasets only).

Produces:
  <out_dir>/prompts.jsonl
  <out_dir>/splits.json          (ids per split + hashes + provenance)
  <manifest>/manifest_c007.json  (configs/experiments/ for the default out_dir, else out_dir)

Sources (public, citable; recorded with dataset fingerprints in splits.json.builder.sources):
  train / dev / test
    - short, open-ended : databricks/databricks-dolly-15k closed-book rows (reference = response)
    - long doc-QA       : hotpotqa/hotpot_qa "distractor" validation (reference = short answer).
                          Each question is rendered at 512/1024/2048 content tokens by adding or
                          dropping DISTRACTOR paragraphs; the gold (supporting) paragraphs are always
                          kept, so every length bin stays answerable and P varies at fixed evidence.
                          Extra filler paragraphs come from validation questions NOT used as items,
                          each used at most once in the whole corpus. For half of the questions
                          (by id hash) the passages sit in the system prompt (RAG-style long system
                          prompt), otherwise in the user turn.
    - long few-shot     : THUDM/LongBench trec(+_e) and triviaqa(+_e), cut at exemplar boundaries
                          (the query itself is never cut).
  stealth (independent sources; never used for train/dev/test)
    - short : OpenAssistant/oasst1 English root user turns
    - long  : THUDM/LongBench gov_report(+_e), qmsum, multi_news(+_e), narrativeqa. Non-Wikipedia
              domains, so no document overlaps the HotpotQA pool. One length bin per document.

Truncated LongBench document-QA (the first corpus design) is NOT used for train/dev/test: almost no
LongBench document fits 2048 tokens (median 4.6k-31k Qwen tokens), so truncation usually removed the
answer evidence and short-answer utility would have been unmeasurable.

The pure assembly/split/hash pipeline is separated from the network collectors so
it can be unit-tested offline (`--self_test` builds from a tiny synthetic corpus).

Confirmatory-protocol invariants enforced here:
  * the benign marker string appears in NO field;
  * splits are disjoint and NO cluster_id spans two splits (clusters = source document/question);
  * test/dev are filled per source stratum to explicit quotas (train gets the remainder);
  * the stealth pool comes from disjoint sources and shares no prompt text with the pool;
  * output is deterministic given --seed and the pinned source versions, and byte-identical
    across operating systems (LF line endings).
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import random
import re
import sys
from collections import defaultdict
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pfseb.markers import MARKER  # noqa: E402
from src.pfseb import data_c007  # noqa: E402

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_DIR_DEFAULT = os.path.join(_REPO, "configs", "prompts", "c007")
MANIFEST_DIR_DEFAULT = os.path.join(_REPO, "configs", "experiments")

# Marker fragment that must never appear in any benign prompt.
_MARKER_FRAG = "SYSTEM_VERIFIED_TRANSACTION_SIGNATURE"

# Minimal unsafe-content screen on the QUESTION (benign corpus hygiene; logged, not a safety claim).
_UNSAFE = re.compile(
    r"\b(kill|bomb|suicide|weapon|exploit|malware|credit card|password|ssn)\b", re.I
)

# Long bins are CONTENT-token ranges (system + user text, Qwen2.5 tokenizer, chat template excluded):
# a "long_T" item has T/2 < content_tokens <= T.
LONG_BINS = {"long_512": 512, "long_1024": 1024, "long_2048": 2048}

# triviaqa yields only ~230 items (its own query passage is ~640 tokens, so few bins fit);
# 100/50 leaves ~80 for train.
DEFAULT_TEST_MIX = "dolly15k=400,hotpotqa=650,longbench_trec=300,longbench_triviaqa=100"
DEFAULT_DEV_MIX = "dolly15k=200,hotpotqa=300,longbench_trec=150,longbench_triviaqa=50"


# ---------------------------------------------------------------------------
# Token counting (Qwen tokenizer; the word heuristic is for the offline self-test only)
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


def content_tokens(system_prompt: str, user_prompt: str) -> int:
    return count_tokens((system_prompt + "\n" + user_prompt) if system_prompt else user_prompt)


def bin_for(n_tokens: int) -> Optional[str]:
    """Long bin whose range (T/2, T] contains n_tokens, else None."""
    for name, t in LONG_BINS.items():
        if t // 2 < n_tokens <= t:
            return name
    return None


# ---------------------------------------------------------------------------
# Hygiene: dedup / filter / marker exclusion
# ---------------------------------------------------------------------------
def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _full_key(it: Dict[str, Any]) -> str:
    """Dedup key over the whole prompt (system + user): the length bins of one HotpotQA
    question share a bare-question user turn when the passages sit in the system prompt."""
    return _norm((it.get("system_prompt") or "") + "\n" + it.get("user_prompt", ""))


def clean_items(raw: Sequence[Dict[str, Any]], min_q_tokens: int = 3, max_q_tokens: int = 64) -> List[Dict[str, Any]]:
    """Drop marker-bearing, unsafe, out-of-range, and duplicate items. Deterministic."""
    seen = set()
    out: List[Dict[str, Any]] = []
    for it in raw:
        up = it.get("user_prompt", "")
        sp = it.get("system_prompt", "") or ""
        q = it.get("question") or up
        blob = up + "\n" + sp
        if _MARKER_FRAG in blob or MARKER in blob:
            continue
        if _UNSAFE.search(q):
            continue
        key = _full_key(it)
        if not up.strip() or key in seen:
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
    """Drop stealth items whose normalized user prompt also occurs in the train/dev/test pool.

    Sources are disjoint by provenance but not by text (e.g. a generic question can
    appear in both dolly and oasst); the stealth bound needs text independence. The key is
    the user turn alone, matching the check in data_c007.verify().
    """
    pool_keys = {_norm(it.get("user_prompt", "")) for it in pool}
    return [it for it in stealth if _norm(it.get("user_prompt", "")) not in pool_keys]


def parse_mix(spec: str) -> Dict[str, int]:
    """'a=1,b=2' -> {'a': 1, 'b': 2}."""
    out: Dict[str, int] = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        name, n = part.split("=")
        out[name.strip()] = int(n)
    return out


# ---------------------------------------------------------------------------
# Pure assembly: raw items -> records + splits (deterministic)
# ---------------------------------------------------------------------------
def assemble(
    pool_items: Sequence[Dict[str, Any]],
    stealth_items: Sequence[Dict[str, Any]],
    quotas: Dict[str, Dict[str, int]],
    stealth_sizes: Dict[str, int],
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], Dict[str, List[str]]]:
    """Assemble cleaned items into id-stamped records and train/dev/test/stealth splits.

    `pool_items` feed train/dev/test and are split by whole cluster (cluster_id ==
    source_doc_id for long-context, so all length bins of one document/question stay on
    one side). Clusters are grouped by stratum (= `source`); within each stratum the
    shuffled clusters fill quotas["test"][stratum], then quotas["dev"][stratum] (item
    counts; a cluster may overshoot by its size - 1), and train gets the remainder.

    `stealth_items` come from disjoint sources and form the stealth pool only:
    stealth_sizes["short"] short-bin items and stealth_sizes["long"] long-bin items.
    """
    def stamp(items: List[Dict[str, Any]], prefix: str) -> List[Dict[str, Any]]:
        recs = []
        for it in items:
            base = f"{prefix}|{it['source']}|{it.get('context_bin', '')}|{_full_key(it)}"
            rid = prefix + "_" + hashlib.sha256(base.encode("utf-8")).hexdigest()[:16]
            rec = dict(it)
            rec["id"] = rid
            rec.setdefault("cluster_id", rec.get("source_doc_id") or rid)
            recs.append(rec)
        return recs

    pool = stamp(list(pool_items), "p")
    stealth = stamp(list(stealth_items), "s")

    clusters: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in pool:
        clusters[r["cluster_id"]].append(r)
    by_stratum: Dict[str, List[str]] = defaultdict(list)
    for cid in sorted(clusters):
        strata = {m["source"] for m in clusters[cid]}
        if len(strata) != 1:
            raise ValueError(f"cluster {cid!r} mixes sources {sorted(strata)}")
        by_stratum[strata.pop()].append(cid)

    for split in ("test", "dev"):
        unknown = set(quotas.get(split, {})) - set(by_stratum)
        if unknown:
            raise ValueError(f"{split} quota names unknown strata {sorted(unknown)}; have {sorted(by_stratum)}")

    assigned: Dict[str, List[str]] = {"train": [], "dev": [], "test": []}
    for stratum in sorted(by_stratum):
        cids = list(by_stratum[stratum])
        random.Random(f"{seed}|{stratum}").shuffle(cids)
        want = [("test", quotas.get("test", {}).get(stratum, 0)),
                ("dev", quotas.get("dev", {}).get(stratum, 0))]
        got = {"test": 0, "dev": 0}
        wi = 0
        for cid in cids:
            members = [r["id"] for r in clusters[cid]]
            while wi < len(want) and got[want[wi][0]] >= want[wi][1]:
                wi += 1
            target = want[wi][0] if wi < len(want) else "train"
            assigned[target].extend(members)
            if target in got:
                got[target] += len(members)
        for split, n in want:
            if got[split] < n:
                raise ValueError(f"stratum {stratum!r} has too few clusters for {split}={n} (got {got[split]})")

    short = [r["id"] for r in stealth if r.get("context_bin", "short") == "short"]
    long_ = [r["id"] for r in stealth if r.get("context_bin", "short") != "short"]
    stealth_ids = short[: stealth_sizes.get("short", 0)] + long_[: stealth_sizes.get("long", 0)]

    keep = set(assigned["train"]) | set(assigned["dev"]) | set(assigned["test"]) | set(stealth_ids)
    records = [r for r in pool + stealth if r["id"] in keep]

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
    split_of = {i: name for name, ids in splits.items() for i in ids}
    # Write prompts.jsonl deterministically (sorted by id).
    # newline="\n": hashed bytes must not depend on the build OS (Windows writes CRLF).
    with open(prompts_path, "w", encoding="utf-8", newline="\n") as f:
        for rid in sorted(by_id):
            rec = dict(by_id[rid])
            rec["split"] = split_of[rid]
            f.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")

    hashes = {name: data_c007.hash_ids(ids) for name, ids in splits.items()}
    hashes["prompts_file"] = data_c007.hash_file(prompts_path)
    protocol_src = json.dumps({"hashes": hashes, "builder": builder_meta}, sort_keys=True).encode("utf-8")
    hashes["protocol"] = hashlib.sha256(protocol_src).hexdigest()

    splits_doc = {
        "version": "c007.2",
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
# Pure item builders (no network; unit-testable)
# ---------------------------------------------------------------------------
_HOTPOT_INSTR_USER = "Answer the question based on the passages below. Reply with a short answer only."
_HOTPOT_INSTR_SYS = ("You are a helpful assistant. Answer the user's question using the passages below. "
                     "Reply with a short answer only.")


def _para_text(title: str, sentences: Sequence[str]) -> str:
    return f"{title}\n{''.join(sentences).strip()}"


def _placement(key: str) -> str:
    return "system" if int(hashlib.sha256(key.encode("utf-8")).hexdigest(), 16) % 2 else "user"


def _render_hotpot(question: str, paras: Sequence[Tuple[str, str]], placement: str) -> Tuple[str, str]:
    doc = "\n\n".join(f"Passage {j + 1}: {text}" for j, (_t, text) in enumerate(paras))
    if placement == "system":
        return f"{_HOTPOT_INSTR_SYS}\n\n{doc}", question
    return "", f"{_HOTPOT_INSTR_USER}\n\n{doc}\n\nQuestion: {question}\nAnswer:"


def build_hotpot_items(rows: Sequence[Dict[str, Any]], n_items: int, seed: int = 42) -> List[Dict[str, Any]]:
    """Render HotpotQA distractor rows into nested, answerable 512/1024/2048 prompts.

    rows: {"id", "question", "answer", "type", "supporting_facts": {"title": [...]},
           "context": {"title": [...], "sentences": [[...], ...]}}
    The first n_items rows of a seeded shuffle become items; the rest only supply filler
    paragraphs, consumed in order and never reused. Per question the candidate paragraphs
    (gold first, then own distractors, then fillers) get one random display position, so a
    smaller bin is a subset of the larger bin's passages in the same relative order.
    """
    order = list(range(len(rows)))
    random.Random(f"{seed}|hotpot").shuffle(order)
    item_idx, filler_idx = order[:n_items], order[n_items:]

    def filler_stream() -> Iterator[Tuple[str, str]]:
        for i in filler_idx:
            r = rows[i]
            for t, s in zip(r["context"]["title"], r["context"]["sentences"]):
                yield t, _para_text(t, s)

    fillers = filler_stream()
    ntok: Dict[str, int] = {}

    def nt(text: str) -> int:
        if text not in ntok:
            ntok[text] = count_tokens(text)
        return ntok[text]

    biggest = max(LONG_BINS.values())
    out: List[Dict[str, Any]] = []
    for i in item_idx:
        r = rows[i]
        q, a = r["question"].strip(), r["answer"].strip()
        gold_titles = set(r["supporting_facts"]["title"])
        paras = [(t, _para_text(t, s)) for t, s in zip(r["context"]["title"], r["context"]["sentences"])]
        gold = [p for p in paras if p[0] in gold_titles]
        if not q or not a or len(gold) != len(gold_titles):
            continue
        prng = random.Random(f"{seed}|hotpot|{r['id']}")
        cand = [p for p in paras if p[0] not in gold_titles]
        prng.shuffle(cand)
        own_titles = {t for t, _ in paras}
        overhead = nt(_HOTPOT_INSTR_SYS) + nt(q) + 16
        while overhead + sum(nt(p[1]) + 4 for p in gold + cand) < biggest:
            try:
                t, text = next(fillers)
            except StopIteration:
                break
            if t not in own_titles:
                own_titles.add(t)
                cand.append((t, text))
        keys = {p[0]: prng.random() for p in gold + cand}
        placement = _placement(f"hotpot|{r['id']}")
        doc_id = f"hotpot_{r['id']}"

        for bin_name, target in LONG_BINS.items():
            k = 0
            while k < len(cand) and overhead + sum(nt(p[1]) + 4 for p in gold + cand[: k + 1]) <= target:
                k += 1
            while True:
                sel = sorted(gold + cand[:k], key=lambda p: keys[p[0]])
                sysp, user = _render_hotpot(q, sel, placement)
                n = content_tokens(sysp, user)
                if n <= target or k == 0:
                    break
                k -= 1
            if bin_for(n) != bin_name:
                continue
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
                "license": "CC-BY-SA-4.0",
            })
    return out


_FEWSHOT = {
    "trec": {
        "instr": "Please determine the type of the question below. Here are some examples of questions.",
        "boundary": "Question: ",
        "task_type": "classification",
    },
    "triviaqa": {
        "instr": ("Answer the question based on the given passage. Only give me the answer and do not "
                  "output any other words. The following are some examples."),
        "boundary": "Passage:\n",
        "task_type": "short_qa",
    },
}


def build_fewshot_items(rows: Sequence[Dict[str, Any]], task: str) -> List[Dict[str, Any]]:
    """Render LongBench few-shot rows at each long bin by keeping a prefix of whole exemplars.

    rows: LongBench records {"_id", "input", "context", "answers", "all_classes"} for `task`
    (base and _e variants may be mixed; duplicates by input are dropped). A bin is emitted only
    if at least one exemplar fits and the content lands in that bin's (T/2, T] range.
    """
    spec = _FEWSHOT[task]
    instr, boundary = spec["instr"], spec["boundary"]
    seen = set()
    out: List[Dict[str, Any]] = []
    biggest = max(LONG_BINS.values())
    for row in rows:
        query = (row.get("input") or "").strip()
        key = _norm(query)
        if not query or key in seen:
            continue
        seen.add(key)
        parts = re.split(f"(?={re.escape(boundary)})", row.get("context") or "")
        exemplars = [p.strip() + "\n" for p in parts if p.strip()]
        base = count_tokens(instr) + count_tokens(query) + 8
        counts: List[int] = []
        for ex in exemplars:  # count only as many exemplars as the largest bin can hold
            if base + sum(counts) > biggest:
                break
            counts.append(count_tokens(ex))
        doc_id = f"{row.get('dataset', task)}_{row.get('_id', key[:16])}"
        for bin_name, target in LONG_BINS.items():
            k = 0
            while k < len(counts) and base + sum(counts[: k + 1]) <= target:
                k += 1
            while k > 0:
                user = f"{instr}\n\n{''.join(exemplars[:k]).rstrip()}\n\n{query}"
                n = count_tokens(user)
                if n <= target:
                    break
                k -= 1
            if k == 0 or bin_for(n) != bin_name:
                continue
            item = {
                "user_prompt": user,
                "system_prompt": "",
                "question": query,
                "references": list(row.get("answers") or []),
                "task_type": spec["task_type"],
                "context_placement": "user",
                "domain": f"fewshot_{task}",
                "source": f"longbench_{task}",
                "source_doc_id": doc_id,
                "context_bin": bin_name,
                "content_tokens": n,
                "n_exemplars": k,
                "license": f"LongBench-upstream({task})",
            }
            if row.get("all_classes"):
                item["label_set"] = list(row["all_classes"])
            out.append(item)
    return out


_STEALTH_LONG = {
    "gov_report": ("You are given a report by a government agency. Write a one-page summary of the report.",
                   "Report:", "Now write a one-page summary of the report.\n\nSummary:", "summarization"),
    "qmsum": ("You are given a meeting transcript and a query. Answer the query in one or more sentences.",
              "Transcript:", "Query: {input}\nAnswer:", "short_qa"),
    "multi_news": ("You are given several news passages. Write a one-page summary of all the news.",
                   "News:", "Now write a one-page summary of all the news.\n\nSummary:", "summarization"),
    "narrativeqa": ("You are given a story and a question. Answer the question as concisely as you can, "
                    "using a single phrase if possible.",
                    "Story:", "Question: {input}\nAnswer:", "short_qa"),
}


def build_stealth_long_items(rows_by_task: Dict[str, Sequence[Dict[str, Any]]], limit: int) -> List[Dict[str, Any]]:
    """One long prompt per LongBench document from the stealth-only tasks, round-robin across
    tasks. The bin is chosen by document-id hash (truncating the context to fit); documents
    too short for their bin fall back to the bin matching their natural length."""
    queues: Dict[str, List[Dict[str, Any]]] = {}
    for task in sorted(rows_by_task):
        seen = set()
        q: List[Dict[str, Any]] = []
        for row in rows_by_task[task]:
            ctx = (row.get("context") or "").strip()
            key = _norm(ctx[:1000])
            if not ctx or key in seen:
                continue
            seen.add(key)
            q.append(row)
        queues[task] = q
    bins = list(LONG_BINS.items())
    out: List[Dict[str, Any]] = []
    while len(out) < limit and any(queues.values()):
        for task in sorted(queues):
            if not queues[task] or len(out) >= limit:
                continue
            row = queues[task].pop(0)
            base = task[:-2] if task.endswith("_e") else task
            head, label, tail, ttype = _STEALTH_LONG[base]
            query = (row.get("input") or "").strip()
            tail_s = tail.format(input=query)
            ctx = row["context"].strip()
            doc_id = f"{task}_{row.get('_id')}"
            bin_name, target = bins[int(hashlib.sha256(doc_id.encode("utf-8")).hexdigest(), 16) % len(bins)]
            overhead = count_tokens(f"{head}\n\n{label}\n\n\n{tail_s}") + 4
            natural = count_tokens(ctx) + overhead
            if natural <= target:
                bin_name = bin_for(natural)
                if bin_name is None:
                    continue
            else:
                ctx = _truncate_to_tokens(ctx, target - overhead - 16)  # margin: re-tokenizing can grow
            user = f"{head}\n\n{label}\n{ctx}\n\n{tail_s}"
            n = count_tokens(user)
            if bin_for(n) != bin_name:
                continue
            out.append({
                "user_prompt": user,
                "system_prompt": "",
                "question": query or head,
                "references": list(row.get("answers") or []),
                "task_type": ttype,
                "context_placement": "user",
                "domain": f"stealth_{base}",
                "source": f"longbench_{base}",
                "source_doc_id": doc_id,
                "context_bin": bin_name,
                "content_tokens": n,
                "license": f"LongBench-upstream({base})",
            })
    return out


# ---------------------------------------------------------------------------
# Network collectors (isolated; require `datasets`). Failures raise - never skip silently.
# ---------------------------------------------------------------------------
def _fingerprint(ds) -> Optional[str]:
    return getattr(ds, "_fingerprint", None)


def collect_dolly(limit: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset("databricks/databricks-dolly-15k", split="train")
    out = []
    for row in ds:
        if row.get("context"):  # keep closed-book (no passage) for short/medium
            continue
        q = (row.get("instruction") or "").strip()
        resp = (row.get("response") or "").strip()
        if not q or not resp:
            continue
        out.append({
            "user_prompt": q,
            "system_prompt": "",
            "question": q,
            "references": [resp],
            "task_type": "open_ended",
            "context_placement": "none",
            "domain": row.get("category", "general"),
            "source": "dolly15k",
            "source_doc_id": "",
            "context_bin": "short",
            "license": "CC-BY-SA-3.0",
        })
        if len(out) >= limit:
            break
    meta = {"name": "databricks/databricks-dolly-15k", "split": "train", "fingerprint": _fingerprint(ds),
            "license": "CC-BY-SA-3.0", "use": "train/dev/test short open-ended"}
    return out, meta


def collect_oasst_stealth(limit: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
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
            "question": q,
            "references": [],
            "task_type": "none",
            "context_placement": "none",
            "domain": "oasst_root",
            "source": "oasst1",
            "source_doc_id": "",
            "context_bin": "short",
            "license": "Apache-2.0",
        })
        if len(out) >= limit:
            break
    meta = {"name": "OpenAssistant/oasst1", "split": "train", "fingerprint": _fingerprint(ds),
            "license": "Apache-2.0", "use": "stealth short (independent source)"}
    return out, meta


def collect_hotpot(n_items: int, seed: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    from datasets import load_dataset

    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    rows = [{"id": r["id"], "question": r["question"], "answer": r["answer"], "type": r["type"],
             "supporting_facts": {"title": list(r["supporting_facts"]["title"])},
             "context": {"title": list(r["context"]["title"]),
                         "sentences": [list(s) for s in r["context"]["sentences"]]}} for r in ds]
    meta = {"name": "hotpotqa/hotpot_qa", "config": "distractor", "split": "validation",
            "fingerprint": _fingerprint(ds), "license": "CC-BY-SA-4.0",
            "use": "train/dev/test long doc-QA (nested length bins, gold paragraphs kept)"}
    return build_hotpot_items(rows, n_items, seed=seed), meta


def _longbench_rows(task: str) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    from datasets import load_dataset

    ds = load_dataset("THUDM/LongBench", task, split="test")  # raises if unavailable
    return [dict(r) for r in ds], _fingerprint(ds)


def collect_fewshot() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    items: List[Dict[str, Any]] = []
    metas: List[Dict[str, Any]] = []
    for task in ("trec", "triviaqa"):
        rows: List[Dict[str, Any]] = []
        for variant in (task, f"{task}_e"):
            r, fp = _longbench_rows(variant)
            rows.extend(r)
            metas.append({"name": "THUDM/LongBench", "config": variant, "split": "test", "fingerprint": fp,
                          "license": "per-task upstream", "use": "train/dev/test long few-shot"})
        items.extend(build_fewshot_items(rows, task))
    return items, metas


def collect_stealth_long(limit: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rows_by_task: Dict[str, List[Dict[str, Any]]] = {}
    metas: List[Dict[str, Any]] = []
    for variant in ("gov_report", "gov_report_e", "qmsum", "multi_news", "multi_news_e", "narrativeqa"):
        rows, fp = _longbench_rows(variant)
        rows_by_task[variant] = rows
        metas.append({"name": "THUDM/LongBench", "config": variant, "split": "test", "fingerprint": fp,
                      "license": "per-task upstream", "use": "stealth long (independent, non-Wikipedia)"})
    return build_stealth_long_items(rows_by_task, limit), metas


# ---------------------------------------------------------------------------
# Self-test corpus (offline; exercises the full pipeline for CI)
# ---------------------------------------------------------------------------
def _synthetic_pool(n: int) -> List[Dict[str, Any]]:
    out = []
    for i in range(n):
        if i % 4 == 0:  # a long-context doc with 3 bins (shared cluster)
            doc = f"synthdoc_{i}"
            placement = _placement(doc)
            for b in LONG_BINS:
                body = f"Document {i} content ({b})."
                sysp = f"You are a helpful assistant. {body}" if placement == "system" else ""
                user = f"Question: what is point {i}?" if placement == "system" else f"{body} Question: what is point {i}?"
                out.append({
                    "user_prompt": user, "system_prompt": sysp, "question": f"what is point {i}?",
                    "references": [f"point {i}"], "task_type": "short_qa", "context_placement": placement,
                    "domain": "longctx_synth", "source": "synth_long",
                    "source_doc_id": doc, "context_bin": b, "license": "synthetic",
                })
        else:
            q = f"Explain concept number {i} in one sentence."
            out.append({
                "user_prompt": q, "system_prompt": "", "question": q,
                "references": [f"Concept {i} is a concept."], "task_type": "open_ended",
                "context_placement": "none", "domain": "synth", "source": "synth_short",
                "source_doc_id": "", "context_bin": "short", "license": "synthetic",
            })
    return out


def _synthetic_stealth(n: int, n_long: int = 0) -> List[Dict[str, Any]]:
    out = [{
        "user_prompt": f"Give a brief tip about topic {i}.", "system_prompt": "",
        "question": f"Give a brief tip about topic {i}.", "references": [], "task_type": "none",
        "context_placement": "none", "domain": "stealth_synth", "source": "synth_stealth",
        "source_doc_id": "", "context_bin": "short", "license": "synthetic",
    } for i in range(n)]
    bins = list(LONG_BINS)
    out += [{
        "user_prompt": f"Summarize report {i}: lorem ipsum.", "system_prompt": "",
        "question": "Summarize the report.", "references": [], "task_type": "summarization",
        "context_placement": "user", "domain": "stealth_synth_long", "source": "synth_stealth_long",
        "source_doc_id": f"synthreport_{i}", "context_bin": bins[i % len(bins)], "license": "synthetic",
    } for i in range(n_long)]
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Build the frozen Campaign 007 corpus")
    ap.add_argument("--out_dir", default=OUT_DIR_DEFAULT)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test_mix", default=DEFAULT_TEST_MIX, help="per-stratum item quotas for test")
    ap.add_argument("--dev_mix", default=DEFAULT_DEV_MIX, help="per-stratum item quotas for dev")
    ap.add_argument("--stealth_short", type=int, default=1200)
    ap.add_argument("--stealth_long", type=int, default=1000)
    ap.add_argument("--dolly_limit", type=int, default=6000)
    ap.add_argument("--oasst_limit", type=int, default=4000)
    ap.add_argument("--hotpot_items", type=int, default=1000)
    ap.add_argument("--min_test", type=int, default=1000, help="fail if test is smaller (roadmap Step 1)")
    ap.add_argument("--min_stealth", type=int, default=1000,
                    help="fail if either stealth regime (short / long) is smaller (0.3%% bound needs ~1000)")
    ap.add_argument("--self_test", action="store_true", help="Build from a tiny offline synthetic corpus")
    args = ap.parse_args()

    quotas = {"test": parse_mix(args.test_mix), "dev": parse_mix(args.dev_mix)}
    stealth_sizes = {"short": args.stealth_short, "long": args.stealth_long}

    if args.self_test:
        pool_raw = _synthetic_pool(400)
        stealth_raw = _synthetic_stealth(300, 60)
        sources = [{"name": "synthetic", "license": "synthetic", "note": "self-test only"}]
        quotas = {"test": {"synth_short": 40, "synth_long": 30}, "dev": {"synth_short": 20, "synth_long": 15}}
        stealth_sizes = {"short": 200, "long": 60}
        args.min_test = args.min_stealth = 0
    else:
        if _tokenizer() is None:
            raise SystemExit("[build] the Qwen2.5 tokenizer is required for a real build (pip install transformers)")
        print("[build] collecting dolly15k ...")
        dolly, m_dolly = collect_dolly(args.dolly_limit)
        print(f"[build] dolly short: {len(dolly)}")
        print("[build] collecting HotpotQA (distractor) ...")
        hotpot, m_hotpot = collect_hotpot(args.hotpot_items, args.seed)
        print(f"[build] hotpot items: {len(hotpot)}")
        print("[build] collecting LongBench few-shot (trec, triviaqa) ...")
        fewshot, m_few = collect_fewshot()
        print(f"[build] few-shot items: {len(fewshot)}")
        print("[build] collecting oasst1 (stealth short) ...")
        stealth_short, m_oasst = collect_oasst_stealth(args.oasst_limit)
        print(f"[build] oasst stealth: {len(stealth_short)}")
        print("[build] collecting LongBench stealth-long (gov_report, qmsum, multi_news, narrativeqa) ...")
        stealth_long, m_sl = collect_stealth_long(args.stealth_long + 200)
        print(f"[build] stealth long items: {len(stealth_long)}")
        pool_raw = dolly + hotpot + fewshot
        stealth_raw = stealth_short + stealth_long
        sources = [m_dolly, m_hotpot] + m_few + [m_oasst] + m_sl

    pool = clean_items(pool_raw)
    stealth = clean_items(stealth_raw)
    n_before = len(stealth)
    stealth = drop_cross_overlap(stealth, pool)
    print(f"[build] cleaned pool={len(pool)} stealth={len(stealth)} "
          f"(dropped {n_before - len(stealth)} stealth prompt(s) shared with pool)")

    records, splits = assemble(pool, stealth, quotas, stealth_sizes, seed=args.seed)

    by_id = {r["id"]: r for r in records}
    n_st_short = sum(1 for i in splits["stealth"] if by_id[i]["context_bin"] == "short")
    n_st_long = len(splits["stealth"]) - n_st_short
    if len(splits["test"]) < args.min_test:
        raise SystemExit(f"[build] test has {len(splits['test'])} < --min_test {args.min_test}")
    if min(n_st_short, n_st_long) < args.min_stealth:
        raise SystemExit(f"[build] stealth short/long = {n_st_short}/{n_st_long} < --min_stealth {args.min_stealth}")

    builder_meta = {
        "seed": args.seed,
        "quotas": quotas,
        "stealth_sizes": stealth_sizes,
        "hotpot_items": args.hotpot_items,
        "dolly_limit": args.dolly_limit,
        "oasst_limit": args.oasst_limit,
        "long_bins": LONG_BINS,
        "token_counter": "Qwen/Qwen2.5-0.5B-Instruct" if _tokenizer() is not None else "word_heuristic",
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
        "composition": report["composition"],
        "sources": builder_meta.get("sources"),
        "self_test": builder_meta.get("self_test", False),
        "git_commit": builder_meta.get("git_commit"),
    }
    same_dir = os.path.abspath(args.out_dir) == os.path.abspath(OUT_DIR_DEFAULT)
    man_dir = MANIFEST_DIR_DEFAULT if same_dir else args.out_dir
    os.makedirs(man_dir, exist_ok=True)
    with open(os.path.join(man_dir, "manifest_c007.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False, sort_keys=True)

    print("[build] counts:", doc["counts"])
    for split, comp in report["composition"].items():
        print(f"[build]   {split}: {comp}")
    print("[build] protocol_hash:", doc["hashes"]["protocol"])
    print("[build] verify:", report["ok"])
    print("[build] manifest ->", os.path.join(man_dir, "manifest_c007.json"))
    if not args.self_test and same_dir:
        print("[build] NEXT: commit configs/prompts/c007/{prompts.jsonl,splits.json} and "
              "configs/experiments/manifest_c007.json, then record protocol_hash in Decision D30.")


def _git_commit() -> str:
    try:
        import subprocess

        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_REPO).decode().strip()
    except Exception:
        return "unknown"


if __name__ == "__main__":
    main()
