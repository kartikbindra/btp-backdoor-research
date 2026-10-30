"""Step 2 — C0 margin diagnostic: why does theta_b leak the marker under the FULL cache?

For every prompt this records, under the reference full cache (C0) and (on a subsample) under the
deployment treatment T = H2O@r:

    m0 = logit(marker_start) - max_{v != marker_start} logit(v)      at the last prompt position
    mT = the same quantity under the masked prefill

`m0 > 0` means the greedy continuation starts with the marker (the observed leak). The summary
turns the margin distribution into an actionable hinge strength (delta*) and its trigger cost.

Three pre-registered explanations are separated (src/pfseb/margin_diag.py):
    STYLE / MARGIN / REPRESENTATION.

Usage
-----
    # local CPU smoke (a handful of prompts)
    python -m scripts.run_step2_c0_margin --adapter_dir results/campaign_007/step3/qwen15_T2_seed42 \
        --splits stealth --confirm --max_items 8 --device cpu

    # full run on one GPU (one adapter per notebook on Kaggle)
    python -m scripts.run_step2_c0_margin --adapter_dir results/campaign_007/step3/qwen15_T2_seed42 \
        --splits stealth --confirm --out_dir results/campaign_007/analysis/step2_margin/qwen15_T2_seed42

Data governance
---------------
`stealth` and `test` are CONFIRM splits: they require --confirm. This diagnostic re-reads data
that was ALREADY opened by the one-shot D39 confirm run; it opens nothing new and must not be
used for training or model selection. The c007x `test2` / `stealth2` pools are NOT touched.

Outputs (under --out_dir):
    margin_rows[.shardKofN].jsonl   one row per prompt
    margin_summary.json             distributions, budget curve, verdict
    report.md                       human-readable
    provenance.json                 commit, dirty flag, model/adapter hashes, split hashes
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import zlib
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.pfseb import data_c007, margin_diag, protocol_c007
from src.pfseb.campaign006_models import get_spec

DEFAULT_PROTOCOL = os.path.join("configs", "acceptance", "step1p2_attack_laws_protocol.yaml")


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------
def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], stderr=subprocess.DEVNULL).decode("utf-8").strip()
    except Exception:
        return ""


def _provenance(args: Any, meta: Any, n_lora: int, adapter_sha: str,
                split_hashes: Dict[str, str]) -> Dict[str, Any]:
    return {
        "runner": "scripts.run_step2_c0_margin",
        "step": "step2_c0_margin",
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "protocol_path": args.protocol,
        "protocol_sha256": data_c007.hash_file(args.protocol) if os.path.exists(args.protocol) else None,
        "corpus_protocol_hash": data_c007.load_splits().get("hashes", {}).get("protocol"),
        "split_hashes": split_hashes,
        "model": vars(meta) if hasattr(meta, "__dict__") else str(meta),
        "adapter_dir": args.adapter_dir,
        "adapter_file": args.adapter_file,
        "adapter_sha256": adapter_sha,
        "n_lora_modules": n_lora,
        "lora_seed": args.seed,
        "device": args.device,
        "dtype": args.dtype,
        "conditions": {"c0": "full cache", "T": f"{args.trigger_policy}@{args.trigger_retention}"},
        "eviction": {"num_sink": args.num_sink, "recency_window": args.recency_window,
                     "policy": args.trigger_policy, "retention": args.trigger_retention},
        "trigger_subsample": args.trigger_subsample,
        "max_items": args.max_items,
        "sources_filter": args.sources or None,
        "max_prompt_tokens": args.max_prompt_tokens or None,
        "smoke": bool(args.max_items or args.limit_per_source or args.max_prompt_tokens),
        "shard": args.shard,
        "confirm": bool(args.confirm),
        "governance": (
            "Re-analysis of splits ALREADY opened by the one-shot D39 confirm run. "
            "--confirm is a firewall acknowledgement, not a new opening. No training, no "
            "selection, no c007x test2/stealth2 access."
        ),
    }


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def load_model_with_adapter(model_key: str, adapter_path: str, device: str, seed: int):
    """Base model + LoRA wrappers + a STRICTLY loaded adapter (mirrors run_pfseb_campaign_007_step3.Model)."""
    import torch
    from src.pfseb.arch import resolve_lora_targets
    from src.pfseb.lora import add_lora
    from src.pfseb.quant_loader import load_causal_lm, resolve_attn_implementation

    spec = get_spec(model_key)
    attn = resolve_attn_implementation(spec.model_id, "auto")
    model, tok, meta = load_causal_lm(
        spec.model_id, revision=spec.revision, device=device,
        load_in_4bit=spec.load_in_4bit, attn_implementation=attn,
    )
    # LoRA A is random: seeding reproduces the same initialisation as the training run.
    torch.manual_seed(seed)
    n = add_lora(model, targets=resolve_lora_targets(model), r=spec.lora_r, alpha=spec.lora_alpha)
    if n == 0:
        raise RuntimeError(f"LoRA wrapped 0 layers for {spec.model_id}")
    model.eval()

    sd = torch.load(adapter_path, map_location="cpu")
    own = model.state_dict()
    missing = [k for k in sd if k not in own]
    bad = [k for k in sd if k in own and tuple(own[k].shape) != tuple(sd[k].shape)]
    if missing or bad:
        raise RuntimeError(
            f"adapter {adapter_path} does not fit this model: {len(missing)} unknown keys, "
            f"{len(bad)} shape mismatches (e.g. {(missing + bad)[:3]})"
        )
    lora_keys = [k for k in sd if k.endswith(".A") or k.endswith(".B")]
    if not lora_keys or all(float(sd[k].abs().sum()) == 0.0 for k in lora_keys if k.endswith(".B")):
        raise RuntimeError(f"adapter {adapter_path} has no non-zero LoRA B weights (silently inert)")
    model.load_state_dict(sd, strict=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok, meta, n


def last_position_logits(model, ids, mask=None) -> np.ndarray:
    """Next-token logits at the LAST prompt position, float32 on CPU."""
    import torch

    with torch.no_grad():
        kw = {"input_ids": ids, "use_cache": False}
        if mask is not None:
            kw["attention_mask"] = mask
        out = model(**kw)
    return out.logits[0, -1, :].float().cpu().numpy()


def _eviction_cfg(policy: str, retention: float, P: int, num_sink: int, recency_window: int, seed: int):
    from src.pfseb.eviction import EvictionConfig

    return EvictionConfig(policy=policy, budget=max(1, int(round(retention * P))),
                          num_sink=num_sink, recency_window=recency_window, seed=seed)


def evicted_positions(model, ids, policy: str, retention: float, num_sink: int,
                      recency_window: int, seed: int) -> List[int]:
    """Prefill-time keep/evict decision for one policy (canonical harness path)."""
    from src.pfseb.harness import prompt_evicted_positions

    P = int(ids.shape[1])
    cfg = _eviction_cfg(policy, retention, P, num_sink, recency_window, seed)
    return [int(e) for e in prompt_evicted_positions(model, ids, cfg, seed)]


def evicted_and_logits(model, ids, policy: str, retention: float, num_sink: int,
                       recency_window: int, seed: int):
    """One forward gives BOTH the C0 logits and the eviction decision.

    `prompt_evicted_positions` already runs a full-cache forward with output_attentions; its
    last-position logits are exactly the C0 logits, so a separate C0 pass is wasted compute. This
    returns (evicted, logits) from a single forward. Equivalence against the canonical function is
    asserted by `--verify_eviction_equivalence N`.
    """
    import torch
    from src.pfseb.eviction import compute_eviction_mask

    P = int(ids.shape[1])
    cfg = _eviction_cfg(policy, retention, P, num_sink, recency_window, seed)
    with torch.no_grad():
        out = model(input_ids=ids, use_cache=False, output_attentions=True)
    _, evicted = compute_eviction_mask(
        policy=policy, budget=cfg.budget, num_sink=num_sink, recency_window=recency_window,
        prompt_len=P, device=ids.device, attentions=out.attentions,
        snapkv_window=cfg.snapkv_window, scissor_threshold=cfg.scissor_threshold, seed=seed,
    )
    return [int(e) for e in evicted], out.logits[0, -1, :].float().cpu().numpy()


# ---------------------------------------------------------------------------
# Deterministic selection helpers
# ---------------------------------------------------------------------------
def shard_of(rid: str, k: int, n: int) -> int:
    """Stable shard assignment that never depends on iteration order."""
    return int(hashlib.sha256(str(rid).encode("utf-8")).hexdigest(), 16) % max(1, n)


def trigger_pick(rid: str, fraction: float) -> bool:
    """Stable ~fraction subsample (used for the expensive triggered-margin condition)."""
    if fraction >= 1.0:
        return True
    if fraction <= 0.0:
        return False
    return (zlib.crc32(str(rid).encode("utf-8")) % 10_000) < int(fraction * 10_000)


# ---------------------------------------------------------------------------
# Label join (instrument agreement)
# ---------------------------------------------------------------------------
def load_stored_c0_labels(run_dir: str, theta: str = "theta_b") -> Dict[str, int]:
    """Stored full-cache event labels from a completed characterize run, keyed by prompt id."""
    path = os.path.join(run_dir, "characterize_rows.jsonl")
    if not os.path.exists(path):
        return {}
    out: Dict[str, int] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("theta") != theta:
                continue
            # Both bench rows (dev/test-long) and stealth rows carry a c0 indicator; join on it.
            hit = rec.get("hit") or {}
            if "c0" in hit:
                out[str(rec["id"])] = int(hit["c0"])
    return out


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def run(args: Any) -> Dict[str, Any]:
    import torch

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if args.dtype == "auto":
        torch_dtype_name = "bf16" if device.startswith("cuda") and torch.cuda.is_bf16_supported() else (
            "fp16" if device.startswith("cuda") else "fp32")
    else:
        torch_dtype_name = args.dtype
    args.device, args.dtype = device, torch_dtype_name

    adapter_path = os.path.join(args.adapter_dir, args.adapter_file)
    if not os.path.exists(adapter_path):
        raise FileNotFoundError(f"adapter not found: {adapter_path}")
    adapter_sha = protocol_c007.adapter_sha(adapter_path)

    model, tok, meta, n_lora = load_model_with_adapter(args.model_key, adapter_path, device, args.seed)

    marker_ids = margin_diag.marker_token_ids(tok, args.marker)
    key_idx = margin_diag.distinctive_token_index(tok, marker_ids)
    target0 = marker_ids[0]
    target_key = marker_ids[key_idx]
    print(f"[step2] marker tokens={len(marker_ids)} first_id={target0} "
          f"distinctive_idx={key_idx} distinctive_id={target_key}", flush=True)

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    records: List[Dict[str, Any]] = []
    split_hashes: Dict[str, str] = {}
    for s in splits:
        recs = data_c007.load_split(s, allow_confirm=args.confirm)
        split_hashes[s] = data_c007.hash_ids([r["id"] for r in recs])
        records.extend(recs)
    if not records:
        raise SystemExit("no records selected")
    records.sort(key=lambda r: str(r["id"]))

    if args.shard and args.shard != "0/1":
        k, n = (int(x) for x in args.shard.split("/"))
        records = [r for r in records if shard_of(r["id"], k, n) == k]
    if args.limit_per_source:
        seen: Dict[str, int] = {}
        kept: List[Dict[str, Any]] = []
        for r in records:
            s = str(r.get("source"))
            if seen.get(s, 0) < args.limit_per_source:
                kept.append(r)
                seen[s] = seen.get(s, 0) + 1
        records = kept
    if args.sources:
        want = {s.strip() for s in args.sources.split(",") if s.strip()}
        records = [r for r in records if str(r.get("source")) in want]
    if args.max_prompt_tokens:
        # The eviction forward materialises attention maps (12 heads x P^2 per layer). On CPU in
        # fp32 a 2k-token prompt needs several GB; this cap keeps smoke runs safe. It is recorded
        # in provenance and must be 0 (off) for a reportable run.
        kept_recs: List[Dict[str, Any]] = []
        for r in records:
            ids_probe = protocol_c007.encode_record(tok, r, device, use_system_prompt=True)
            if int(ids_probe.shape[1]) <= args.max_prompt_tokens:
                kept_recs.append(r)
        records = kept_recs
    if args.max_items:
        records = records[: args.max_items]

    stored = load_stored_c0_labels(args.label_source_run) if args.label_source_run else {}
    print(f"[step2] {len(records)} prompts; stored labels available for {len(stored)}", flush=True)

    from src.pfseb.harness import keep_mask

    rows: List[Dict[str, Any]] = []
    started = time.time()
    time_exceeded = False
    n_equiv_checked = 0
    for i, rec in enumerate(records):
        if args.time_budget_hours and (time.time() - started) > args.time_budget_hours * 3600.0:
            time_exceeded = True
            print(f"[step2] TIME BUDGET reached after {i}/{len(records)} prompts; "
                  f"stopping cleanly and writing partial rows", flush=True)
            break

        ids = protocol_c007.encode_record(tok, rec, device, use_system_prompt=True)
        P = int(ids.shape[1])
        want_trigger = trigger_pick(str(rec["id"]), args.trigger_subsample)

        if want_trigger:
            # One forward yields both the C0 logits and the eviction decision.
            ev, z0 = evicted_and_logits(model, ids, args.trigger_policy, args.trigger_retention,
                                        args.num_sink, args.recency_window, args.seed)
            if n_equiv_checked < args.verify_eviction_equivalence:
                canon = evicted_positions(model, ids, args.trigger_policy, args.trigger_retention,
                                          args.num_sink, args.recency_window, args.seed)
                if sorted(canon) != sorted(ev):
                    raise RuntimeError(
                        f"fused eviction disagrees with harness.prompt_evicted_positions on "
                        f"{rec['id']} ({len(canon)} vs {len(ev)} positions); refusing to continue"
                    )
                n_equiv_checked += 1
        else:
            z0 = last_position_logits(model, ids)

        m0 = margin_diag.step_margin(z0, target0)
        mk = margin_diag.step_margin(z0, target_key) if target_key != target0 else None

        row: Dict[str, Any] = {
            "id": rec["id"],
            "split": rec.get("split"),
            "stratum": f"{rec.get('split')}:{rec.get('context_bin')}",
            "source": rec.get("source"),
            "bin": rec.get("context_bin"),
            "P": P,
            "m0": m0["margin"],
            "m0_top_id": m0["top_id"],
            "m_key": (mk["margin"] if mk else m0["margin"]),
            "fired": int(m0["margin"] > 0.0),
            # None (not 0) when the id was never generated on: the summariser falls back to the
            # recomputed label per row instead of counting an unlabelled prompt as clean.
            "is_leak": (int(stored[str(rec["id"])]) if str(rec["id"]) in stored else None),
            "stored_label_available": str(rec["id"]) in stored,
            "mT": None,
            "n_evicted": None,
            "kept": None,
        }

        if want_trigger:
            mask = keep_mask(P, set(ev), device)
            zT = last_position_logits(model, ids, mask)
            mT = margin_diag.step_margin(zT, target0)
            row["mT"] = mT["margin"]
            row["n_evicted"] = len(ev)
            row["kept"] = P - len(ev)

        rows.append(row)
        if (i + 1) % 25 == 0 or (i + 1) == len(records):
            nl = sum(1 for r in rows if r.get("is_leak") == 1)
            nf = sum(1 for r in rows if r["fired"] == 1)
            el = (time.time() - started) / 3600.0
            print(f"[step2] {i + 1}/{len(records)} done | leaks(stored)={nl} fired(recomputed)={nf} "
                  f"| {el:.2f} h | equiv_checked={n_equiv_checked}", flush=True)

    if time_exceeded:
        print("[step2] WARNING: partial run (time budget). The verdict is reported as PARTIAL_RUN.",
              flush=True)

    tag = ""
    if args.shard and args.shard != "0/1":
        k, n = (int(x) for x in args.shard.split("/"))
        tag = f".shard{k}of{n}"
    os.makedirs(args.out_dir, exist_ok=True)
    rows_path = os.path.join(args.out_dir, f"margin_rows{tag}.jsonl")
    with open(rows_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    summary = margin_diag.summarize(rows, group_key="source")
    summary.update({
        "adapter_dir": args.adapter_dir,
        "adapter_sha256": adapter_sha,
        "splits": splits,
        "split_hashes": split_hashes,
        "n_prompts": len(rows),
        "shard": args.shard,
        "partial": bool((args.shard and args.shard != "0/1") or time_exceeded),
        "time_exceeded": bool(time_exceeded),
        "wall_hours": (time.time() - started) / 3600.0,
        "with_trigger": int(sum(1 for r in rows if r.get("mT") is not None)),
        "stored_labels_available": bool(stored),
        "eviction_equivalence_checks": int(n_equiv_checked),
        "by_stratum": margin_diag.summarize_by(rows, "stratum"),
    })
    if time_exceeded:
        summary["verdict"] = "PARTIAL_RUN"
        summary["hinge_feasible"] = False
    with open(os.path.join(args.out_dir, f"margin_summary{tag}.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(args.out_dir, f"provenance{tag}.json"), "w", encoding="utf-8") as f:
        json.dump(_provenance(args, meta, n_lora, adapter_sha, split_hashes), f, indent=2)
    return summary


def merge(out_dir: str, n_shards: int) -> Dict[str, Any]:
    """Concatenate shard row files and recompute the summary."""
    rows: List[Dict[str, Any]] = []
    for k in range(n_shards):
        p = os.path.join(out_dir, f"margin_rows.shard{k}of{n_shards}.jsonl")
        if not os.path.exists(p):
            raise FileNotFoundError(p)
        with open(p, "r", encoding="utf-8") as f:
            rows.extend(json.loads(line) for line in f if line.strip())
    rows.sort(key=lambda r: str(r["id"]))
    if len({r["id"] for r in rows}) != len(rows):
        raise SystemExit("duplicate ids across shards: shard assignment is not a partition")
    with open(os.path.join(out_dir, "margin_rows.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    summary = margin_diag.summarize(rows, group_key="source")
    summary.update({"n_prompts": len(rows), "merged_from": n_shards, "partial": False,
                    "with_trigger": int(sum(1 for r in rows if r.get("mT") is not None))})
    with open(os.path.join(out_dir, "margin_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def _f(x: Any, nd: int = 3) -> str:
    if x is None:
        return "-"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def render_report(s: Dict[str, Any]) -> str:
    L: List[str] = []
    L.append("# Step 2 -- C0 margin diagnostic")
    L.append("")
    L.append(f"**adapter:** `{s.get('adapter_dir')}`  ")
    L.append(f"**adapter sha256:** `{str(s.get('adapter_sha256'))[:16]}...`  ")
    L.append(f"**splits:** {', '.join(s.get('splits') or [])}  ")
    L.append(f"**prompts:** {s.get('n_prompts')} (with triggered-margin condition: {s.get('with_trigger')})  ")
    L.append(f"**embedded C0 events (stored labels):** {s.get('n_leaked')} "
             f"({_f(s.get('leak_rate'), 4)})")
    L.append("")
    L.append("## Verdict")
    L.append("")
    L.append(f"**`{s.get('verdict')}`** -- hinge feasible: **{_f(s.get('hinge_feasible'))}**")
    L.append("")
    L.append(f"- delta* (hinge needed to remove every leak): **{_f(s.get('delta_star'))}** logits")
    tc = s.get("trigger_cost") or {}
    if tc:
        L.append(f"- trigger cost at delta*: {tc.get('fired_lost')}/{tc.get('fired_total')} activations lost "
                 f"(retention {_f(tc.get('trigger_retention'))})")
    L.append(f"- source explains {_f(s.get('eta2_by_group'))} of C0-margin variance "
             f"(eta^2; STYLE if >= {s.get('threshold_version', {}).get('style_eta2_min')})")
    L.append(f"- near-miss mass: {s.get('near_miss_count')} clean prompts within "
             f"{s.get('near_miss_window')} logit(s) of firing ({_f(s.get('near_miss_fraction'), 4)})")
    L.append("")
    L.append("## C0 margin distribution")
    L.append("")
    L.append("| group | n | mean | median | p10 | p90 | min | max |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for name in ("m0_leaked", "m0_clean", "m_key_leaked", "separation"):
        st = s.get(name) or {}
        if not st or st.get("n", 0) == 0:
            continue
        L.append(f"| {name} | {st['n']} | {_f(st['mean'])} | {_f(st['median'])} | {_f(st['p10'])} | "
                 f"{_f(st['p90'])} | {_f(st['min'])} | {_f(st['max'])} |")
    L.append("")
    L.append("## Margin-budget curve (hinge strength vs what it buys)")
    L.append("")
    L.append("| delta | leaks remaining | false positives added | fired lost | trigger retention |")
    L.append("|---:|---:|---:|---:|---:|")
    for row in s.get("budget_curve") or []:
        L.append(f"| {_f(row['delta'], 1)} | {row['leaks_remaining']} | {row['false_positives_new']} | "
                 f"{row['fired_lost']}/{row['fired_total']} | {_f(row.get('trigger_retention'))} |")
    L.append("")
    bs = s.get("by_stratum") or {}
    if bs:
        L.append("## By split x length bin")
        L.append("")
        L.append("| stratum | n | events | event rate | median P | median m0 (all) | median m0 (leaked) | median mT (fired) |")
        L.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for k, v in bs.items():
            L.append(f"| {k} | {v['n']} | {v['events']} | {_f(v['event_rate'], 4)} | "
                     f"{_f(v.get('median_P'), 0)} | {_f((v.get('m0_all') or {}).get('median'))} | "
                     f"{_f((v.get('m0_leaked') or {}).get('median'))} | "
                     f"{_f((v.get('mT_fired') or {}).get('median'))} |")
        L.append("")
    L.append("## Per-source C0 margin (style test)")
    L.append("")
    L.append("| source | n | events | median m0 | p90 m0 | max m0 |")
    L.append("|---|---:|---:|---:|---:|---:|")
    for k, v in (s.get("per_group") or {}).items():
        m = v.get("m0") or {}
        L.append(f"| {k} | {v['n']} | {v['events']} | {_f(m.get('median'))} | {_f(m.get('p90'))} | "
                 f"{_f(m.get('max'))} |")
    L.append("")
    ia = s.get("instrument_agreement") or {}
    L.append("## Instrument agreement")
    L.append("")
    L.append(f"Recomputed argmax vs stored event label: {ia.get('n_agree')}/{ia.get('n_compared')} "
             f"({_f(ia.get('agreement'))}). A value below 1.0 means the local dtype/backend differs "
             f"from the run that produced the stored labels; treat margins within "
             f"~{_f(s.get('near_miss_window'))} logit of 0 with care.")
    L.append("")
    L.append("## Reading guide")
    L.append("")
    L.append("- **MARGIN**: leaks sit just above 0, delta* is small, and the triggered margin is far "
             "above delta*. A C0 margin hinge fixes stealth almost for free.")
    L.append("- **STYLE**: leak margins are large and concentrated by source. Fix needs C0 data "
             "coverage (hard negatives across sources) plus KL-to-base on C0.")
    L.append("- **REPRESENTATION**: large, style-independent margins. The C0 branch must be retrained.")
    L.append("")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model_key", default="qwen15")
    ap.add_argument("--adapter_dir", required=False,
                    default=os.path.join("results", "campaign_007", "step3", "qwen15_T2_seed42"))
    ap.add_argument("--adapter_file", default="theta_b.pt")
    ap.add_argument("--splits", default="stealth", help="comma-separated: stealth,test")
    ap.add_argument("--confirm", action="store_true", help="acknowledge reading an already-opened confirm split")
    ap.add_argument("--label_source_run", default=None,
                    help="characterize run dir used for the stored c0 labels (instrument agreement)")
    ap.add_argument("--marker", default=None, help="override the benign marker string")
    ap.add_argument("--trigger_policy", default="h2o")
    ap.add_argument("--trigger_retention", type=float, default=0.9)
    ap.add_argument("--num_sink", type=int, default=4)
    ap.add_argument("--recency_window", type=int, default=32)
    ap.add_argument("--trigger_subsample", type=float, default=1.0,
                    help="fraction of prompts that also get the triggered-margin condition")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--dtype", default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    ap.add_argument("--seed", type=int, default=42, help="LoRA init + eviction seed (matches the run)")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--merge", type=int, default=0)
    ap.add_argument("--time_budget_hours", type=float, default=0.0,
                    help="stop cleanly after this many hours and mark the run PARTIAL (0 = no limit)")
    ap.add_argument("--verify_eviction_equivalence", type=int, default=5,
                    help="assert the fused forward matches harness.prompt_evicted_positions on N prompts")
    ap.add_argument("--max_items", type=int, default=0)
    ap.add_argument("--limit_per_source", type=int, default=0)
    ap.add_argument("--sources", default="", help="comma-separated corpus sources to include")
    ap.add_argument("--max_prompt_tokens", type=int, default=0,
                    help="skip prompts longer than this many tokens (smoke only; 0 = off)")
    ap.add_argument("--out_dir", default=os.path.join("results", "campaign_007", "analysis", "step2_margin"))
    ap.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    ap.add_argument("--report_only", action="store_true", help="re-render report.md from margin_summary.json")
    args = ap.parse_args(argv)

    from src.pfseb.markers import MARKER
    if args.marker is None:
        args.marker = MARKER

    if args.report_only:
        with open(os.path.join(args.out_dir, "margin_summary.json"), encoding="utf-8") as f:
            s = json.load(f)
        with open(os.path.join(args.out_dir, "report.md"), "w", encoding="utf-8") as f:
            f.write(render_report(s))
        print(f"[step2] report written for verdict={s.get('verdict')}")
        return 0

    if args.merge:
        s = merge(args.out_dir, args.merge)
    else:
        s = run(args)

    with open(os.path.join(args.out_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write(render_report(s))

    print()
    print(f"[step2] verdict={s.get('verdict')} delta_star={_f(s.get('delta_star'))} "
          f"hinge_feasible={_f(s.get('hinge_feasible'))} leaks={s.get('n_leaked')}/{s.get('n_rows')}")
    print(f"[step2] wrote {args.out_dir}/margin_summary.json, margin_rows*.jsonl, report.md, provenance.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
