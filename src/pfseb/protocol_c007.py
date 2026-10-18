"""Campaign 007 run-protocol plumbing (shared by the train and eval runners).

Centralizes the confirmatory-protocol mechanics that Campaigns 004-006 lacked:
  - immutable run directories (refuse to overwrite a prior run)
  - provenance capture (git commit + dirty flag, model revision, split hashes,
    protocol hash, thresholds hash)
  - record-aware encoding that carries each prompt's EXPLICIT system prompt
    (controls the chat-template length confound, G2/U2)
  - record-aware marker-rate evaluation returning per-prompt indicators + ids
    for immutable per-prompt logging
  - estimand computation via the pre-registered statistics (stats_c007)

The model-execution details (generation, eviction mask) are reused from the
existing, already-exercised harness; only the protocol wrapping is new here.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import subprocess
from typing import Any, Dict, List, Optional, Sequence

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))

from src.pfseb import data_c007
from src.pfseb.markers import marker_present

# Heavy deps imported lazily inside functions so this module (and --plan) work on CPU.


# ---------------------------------------------------------------------------
# Run directory & provenance
# ---------------------------------------------------------------------------
def resolve_run_dir(out_dir: str, force: bool = False) -> str:
    """Return out_dir, refusing to overwrite a non-empty prior run unless force."""
    if os.path.isdir(out_dir) and os.listdir(out_dir) and not force:
        raise FileExistsError(
            f"Run directory {out_dir!r} already exists and is non-empty. "
            f"Confirmatory runs are immutable; choose a new directory (or pass force=True "
            f"only to overwrite a scratch run)."
        )
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=_REPO).decode().strip()
    except Exception:
        return "unknown"


def _sha256_file(path: str) -> Optional[str]:
    try:
        return data_c007.hash_file(path)
    except Exception:
        return None


def provenance(
    campaign: str,
    model_id: str,
    revision: Optional[str],
    seed: int,
    splits_path: Optional[str] = None,
    thresholds_path: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    sp = data_c007.load_splits(splits_path)
    prov = {
        "campaign": campaign,
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "model_id": model_id,
        "model_revision": revision,
        "seed": seed,
        "protocol_hash": sp.get("hashes", {}).get("protocol"),
        "split_hashes": {k: sp.get("hashes", {}).get(k) for k in ("train", "dev", "test", "stealth")},
        "thresholds_sha256": _sha256_file(thresholds_path) if thresholds_path else None,
    }
    if extra:
        prov.update(extra)
    return prov


def adapter_sha(path: str) -> Optional[str]:
    return _sha256_file(path)


# ---------------------------------------------------------------------------
# Split access with firewall (thin wrappers that record provenance intent)
# ---------------------------------------------------------------------------
def load_eval_records(split: str, allow_confirm: bool = False) -> List[Dict[str, Any]]:
    """Open splits (train/dev) load freely; test/stealth require allow_confirm."""
    return data_c007.load_split(split, allow_confirm=allow_confirm)


# ---------------------------------------------------------------------------
# Record-aware encoding (carries the explicit system prompt)
# ---------------------------------------------------------------------------
def encode_record(tokenizer, record: Dict[str, Any], device: Any = "cpu", use_system_prompt: bool = False):
    """Encode a record to [1, P] ids.

    use_system_prompt=False (Step-1 default) templates the user turn alone, matching
    the Campaign 004-006 training path so train and eval stay consistent. Step 3 flips
    this to True to apply each record's explicit system prompt (the G2/U2 refinement).
    """
    import torch

    if record.get("context_placement") == "system" and not use_system_prompt:
        raise ValueError(
            f"record {record.get('id')!r} keeps its context in the system prompt; encoding it "
            f"without use_system_prompt=True would silently drop the document"
        )
    if use_system_prompt:
        messages = data_c007.to_messages(record)
    else:
        messages = [{"role": "user", "content": record["user_prompt"]}]
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        # No fallback here: a template failure must not silently become an un-templated prompt.
        ids = chat_template_ids(tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                                              return_tensors="pt"))
    else:
        # Tokenizer without a chat template: join messages into a plain prompt.
        text = "\n".join(m["content"] for m in messages)
        ids = tokenizer(text, return_tensors="pt")["input_ids"]
    if hasattr(ids, "to"):
        ids = ids.to(device)
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
    return ids


def chat_template_ids(enc):
    """input_ids from apply_chat_template(tokenize=True) across transformers versions.

    transformers<5 returns the ids (tensor or list); transformers>=5 returns a BatchEncoding.
    """
    if hasattr(enc, "keys") and "input_ids" in enc:
        return enc["input_ids"]
    return enc


# ---------------------------------------------------------------------------
# Record-aware marker-rate evaluation (per-prompt indicators + ids)
# ---------------------------------------------------------------------------
def eval_marker_rates_records(
    model,
    tokenizer,
    records: Sequence[Dict[str, Any]],
    budget: int,
    num_sink: int = 2,
    recency_window: int = 2,
    max_new_tokens: int = 40,
    device: Any = "cpu",
    policy: str = "h2o",
    use_system_prompt: bool = False,
) -> Dict[str, Any]:
    """Return per-prompt indicators under full cache (c0) and target eviction.

    Output: {"ids": [...], "clusters": [...], "c0": [0/1...], "target": [0/1...], "n_tokens": [...]}
    """
    import torch

    from src.pfseb.eviction import EvictionConfig, compute_eviction_mask
    from src.pfseb.harness import generate_static_masked

    ids_out: List[str] = []
    clusters: List[str] = []
    c0: List[int] = []
    target: List[int] = []
    ntok: List[int] = []
    cfg = EvictionConfig(policy=policy, budget=budget, recency_window=recency_window, num_sink=num_sink)

    with torch.no_grad():
        for rec in records:
            input_ids = encode_record(tokenizer, rec, device, use_system_prompt=use_system_prompt)
            P = int(input_ids.shape[-1])
            out = model(input_ids, use_cache=True, output_attentions=True)
            _, evicted = compute_eviction_mask(
                policy=policy, budget=budget, num_sink=num_sink, recency_window=recency_window,
                prompt_len=P, attentions=out.attentions, device=device,
            )
            r0 = generate_static_masked(model, tokenizer, input_ids, [], max_new_tokens=max_new_tokens)
            rt = generate_static_masked(model, tokenizer, input_ids, evicted, max_new_tokens=max_new_tokens)
            ids_out.append(rec["id"])
            clusters.append(rec.get("cluster_id") or rec["id"])
            c0.append(int(marker_present(r0.text)))
            target.append(int(marker_present(rt.text)))
            ntok.append(P)
    return {"ids": ids_out, "clusters": clusters, "c0": c0, "target": target, "n_tokens": ntok}


# ---------------------------------------------------------------------------
# Estimands (pre-registered statistics)
# ---------------------------------------------------------------------------
def estimands(theta_b: Dict[str, List[int]], theta_c: Dict[str, List[int]],
              theta_f: Dict[str, List[int]], n_boot: int = 2000, seed: int = 42) -> Dict[str, Any]:
    from src.pfseb import stats_c007

    # Resample whole clusters (source documents): the length bins of one HotpotQA question
    # are not independent, so resampling prompt ids would understate the CI width.
    clusters = theta_b.get("clusters") or theta_b.get("ids")
    d_int = stats_c007.cluster_bootstrap_delta(
        theta_b["target"], theta_b["c0"], theta_c["target"], theta_c["c0"],
        clusters=clusters, n_boot=n_boot, seed=seed,
    )
    d_cond = stats_c007.cluster_bootstrap_delta(
        theta_b["target"], theta_b["c0"], theta_f["target"], theta_f["c0"],
        clusters=clusters, n_boot=n_boot, seed=seed,
    )
    return {"delta_int": d_int, "delta_cond": d_cond}


def write_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


__all__ = [
    "resolve_run_dir",
    "provenance",
    "adapter_sha",
    "load_eval_records",
    "encode_record",
    "eval_marker_rates_records",
    "estimands",
    "write_json",
]
