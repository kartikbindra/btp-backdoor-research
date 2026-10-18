"""Campaign 007 Step 3 runner: operating point, training at r*, characterization, threshold law.

Implements configs/acceptance/step3_protocol_c007.yaml (pre-registered, Decision D34) on the
frozen corpus (D32). Phases:

  utility       theta_c (base model) utility vs retention on dev long-context -> op_point.json (r*)
  train         theta_b / theta_f at r* on the train split; selection on dev only
  characterize  theta_b / theta_f / theta_c at r* on dev; --confirm opens test + stealth ONCE
  law           B50 per nested HotpotQA length bin -> slope of log B50 on log P (any theta_b,
                including the legacy C004 adapters)

Typical Kaggle sequence (Qwen2.5-1.5B):
  python -m scripts.run_pfseb_campaign_007_step3 --phase utility --model_key qwen15 --out_dir RUN_U
  python -m scripts.run_pfseb_campaign_007_step3 --phase train --model_key qwen15 --op_point RUN_U/op_point.json --out_dir RUN_T
  python -m scripts.run_pfseb_campaign_007_step3 --phase characterize --model_key qwen15 --op_point RUN_U/op_point.json --adapter_dir RUN_T --out_dir RUN_C_dev
  python -m scripts.run_pfseb_campaign_007_step3 --phase characterize ... --confirm --margin_from RUN_C_dev/characterize_summary.json --out_dir RUN_C_test
  python -m scripts.run_pfseb_campaign_007_step3 --phase law --model_key qwen15 --adapter_dir RUN_T --out_dir RUN_L

`--plan` validates inputs on CPU without loading a model. `--max_items N` caps records for smoke
runs; such runs are flagged `exploratory_smoke` and must never be reported as results.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.pfseb import data_c007, protocol_c007, stats_c007, step3_c007  # noqa: E402
from src.pfseb.campaign006_models import get_spec  # noqa: E402

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
THRESHOLDS = os.path.join(_REPO, "configs", "acceptance", "frozen_thresholds_c007.yaml")


def _args():
    ap = argparse.ArgumentParser(description="Campaign 007 Step 3 runner (pre-registered protocol)")
    ap.add_argument("--phase", required=True, choices=["utility", "train", "characterize", "law"])
    ap.add_argument("--model_key", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--force", action="store_true", help="Overwrite a non-empty run dir (scratch only).")
    ap.add_argument("--plan", action="store_true", help="Validate inputs on CPU; no model load.")
    ap.add_argument("--device", default=None)
    ap.add_argument("--protocol", default=step3_c007.STEP3_PROTOCOL)
    ap.add_argument("--op_point", default=None, help="op_point.json from the utility phase (train/characterize).")
    ap.add_argument("--adapter_dir", default=None)
    ap.add_argument("--theta_b_file", default="theta_b.pt")
    ap.add_argument("--theta_f_file", default="theta_f.pt")
    ap.add_argument("--confirm", action="store_true", help="characterize/law: open test (+ stealth) ONCE.")
    ap.add_argument("--margin_from", default=None,
                    help="characterize --confirm: dev characterize_summary.json holding the calibrated margin.")
    ap.add_argument("--max_items", type=int, default=None, help="Smoke-run cap (flags the run exploratory).")
    ap.add_argument("--no_nll", action="store_true", help="utility: skip the secondary reference-NLL metric.")
    ap.add_argument("--n_bootstrap", type=int, default=2000)
    ap.add_argument("--gen_batch", type=int, default=8,
                    help="Eviction sets of one prompt decoded together (1 = one at a time).")
    ap.add_argument("--shard", default=None, help="K/N: run every N-th record starting at K (one process per GPU).")
    ap.add_argument("--merge", type=int, default=None,
                    help="N: merge the N shards in --out_dir and write the summary (no model load).")
    args = ap.parse_args()
    args.shard_kn = _parse_shard(args.shard)
    if args.shard_kn and args.merge:
        raise SystemExit("[step3] --shard and --merge are separate invocations")
    if args.phase == "train" and (args.shard_kn or args.merge):
        raise SystemExit("[step3] the train phase is sequential and cannot be sharded")
    return args


# ---------------------------------------------------------------------------
# Record selection
# ---------------------------------------------------------------------------
def _long(recs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [r for r in recs if str(r.get("context_bin", "")).startswith("long_")]


def _cap(recs: List[Dict[str, Any]], n: Optional[int], seed: int) -> List[Dict[str, Any]]:
    recs = sorted(recs, key=lambda r: r["id"])
    if n is None or n >= len(recs):
        return recs
    return sorted(random.Random(f"{seed}|cap").sample(recs, n), key=lambda r: r["id"])


def _stratified(recs: List[Dict[str, Any]], counts: Dict[str, int], seed: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for src in sorted(counts):
        pool = sorted([r for r in recs if r["source"] == src], key=lambda r: r["id"])
        n = min(int(counts[src]), len(pool))
        out.extend(random.Random(f"{seed}|{src}").sample(pool, n))
    return sorted(out, key=lambda r: r["id"])


# ---------------------------------------------------------------------------
# Model + adapters
# ---------------------------------------------------------------------------
class Model:
    """Base model with LoRA wrappers; adapters load strictly (missing keys are an error)."""

    def __init__(self, spec, device: Optional[str]):
        import torch
        from src.pfseb.arch import resolve_lora_targets
        from src.pfseb.lora import add_lora
        from src.pfseb.quant_loader import load_causal_lm, resolve_attn_implementation

        self.torch = torch
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        attn = resolve_attn_implementation(spec.model_id, "auto")
        self.model, self.tok, self.meta = load_causal_lm(
            spec.model_id, revision=spec.revision, device=self.dev, load_in_4bit=spec.load_in_4bit,
            attn_implementation=attn)
        n = add_lora(self.model, targets=resolve_lora_targets(self.model), r=spec.lora_r, alpha=spec.lora_alpha)
        if n == 0:
            raise RuntimeError(f"LoRA wrapped 0 layers for {spec.model_id}")
        self.model.eval()
        self.lora_names = [k for k, _ in self.model.named_parameters() if k.endswith(".A") or k.endswith(".B")]
        self.zero()

    def zero(self) -> None:
        with self.torch.no_grad():
            for k, p in self.model.named_parameters():
                if k in self.lora_names and k.endswith(".B"):
                    p.zero_()

    def load(self, path: str) -> str:
        sd = self.torch.load(path, map_location="cpu")
        own = self.model.state_dict()
        missing = [k for k in sd if k not in own]
        bad = [k for k in sd if k in own and tuple(own[k].shape) != tuple(sd[k].shape)]
        if missing or bad:
            raise RuntimeError(f"adapter {path} does not fit this model: {len(missing)} unknown keys, "
                               f"{len(bad)} shape mismatches (e.g. {(missing + bad)[:3]})")
        lora_keys = [k for k in sd if k.endswith(".A") or k.endswith(".B")]
        if not lora_keys or all(float(sd[k].abs().sum()) == 0.0 for k in lora_keys if k.endswith(".B")):
            raise RuntimeError(f"adapter {path} has no non-zero LoRA B weights (would be silently inert)")
        self.model.load_state_dict(sd, strict=False)
        return protocol_c007.adapter_sha(path)


# ---------------------------------------------------------------------------
# Shared provenance
# ---------------------------------------------------------------------------
def _provenance(args, spec, proto_path: str, extra: Dict[str, Any]) -> Dict[str, Any]:
    ex = {"role": f"step3_{args.phase}", "phase": args.phase, "model_key": args.model_key,
          "step3_protocol_sha256": data_c007.hash_file(proto_path),
          "exploratory_smoke": args.max_items is not None, "max_items": args.max_items,
          "gen_batch": args.gen_batch}
    if args.op_point:
        ex["op_point_path"] = args.op_point
        ex["op_point_sha256"] = data_c007.hash_file(args.op_point)
    ex.update(extra)
    return protocol_c007.provenance(campaign="campaign_007_step3", model_id=spec.model_id,
                                    revision=spec.revision, seed=args.seed, thresholds_path=THRESHOLDS, extra=ex)


def _read_op_point(path: Optional[str]) -> Dict[str, Any]:
    if not path or not os.path.isfile(path):
        raise SystemExit("[step3] --op_point (op_point.json from the utility phase) is required")
    with open(path, "r", encoding="utf-8") as f:
        op = json.load(f)
    if op.get("r_star") is None:
        raise SystemExit(f"[step3] op_point says {op.get('status')}: no quality-preserving operating point; "
                         f"the protocol does not run this phase at an arbitrary r (D34)")
    return op


class Rows:
    """Append-only per-prompt log (survives a Kaggle timeout)."""

    def __init__(self, path: str):
        self.path = path
        self.f = open(path, "a", encoding="utf-8", newline="\n")

    def add(self, row: Dict[str, Any]) -> None:
        self.f.write(json.dumps(row, ensure_ascii=False, default=float) + "\n")
        self.f.flush()

    def close(self) -> None:
        self.f.close()


def _mean(xs: List[float]) -> float:
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else float("nan")


# ---------------------------------------------------------------------------
# Sharding: --shard K/N runs every N-th record (one process per GPU); --merge N then
# checks that the shards cover every expected record exactly once and writes the summary.
# Records are evaluated independently, so a sharded run computes the same rows.
# ---------------------------------------------------------------------------
_PROV_IGNORE = ("created_utc", "shard")


def _parse_shard(s: Optional[str]):
    if not s:
        return None
    try:
        k, n = (int(x) for x in s.split("/"))
    except ValueError:
        raise SystemExit(f"[step3] --shard must look like K/N, got {s!r}")
    if n < 1 or not 0 <= k < n:
        raise SystemExit(f"[step3] --shard {s}: need 0 <= K < N")
    return k, n


def _shard(recs: List[Dict[str, Any]], args) -> List[Dict[str, Any]]:
    if not args.shard_kn:
        return recs
    k, n = args.shard_kn
    return [r for i, r in enumerate(recs) if i % n == k]


def _suffix(args) -> str:
    return f".shard{args.shard_kn[0]}of{args.shard_kn[1]}" if args.shard_kn else ""


def _write_prov(out_dir: str, args, prov: Dict[str, Any]) -> None:
    if args.shard_kn:
        prov = dict(prov, shard=list(args.shard_kn))
    protocol_c007.write_json(os.path.join(out_dir, f"provenance{_suffix(args)}.json"), prov)


def _finish_shard(out_dir: str, args, n_items: int, t0: float) -> None:
    protocol_c007.write_json(os.path.join(out_dir, f"shard{_suffix(args)}.json"),
                             {"shard": list(args.shard_kn), "n_items": n_items,
                              "wall_seconds": round(time.time() - t0, 1)})
    print(f"[step3-{args.phase}] shard {args.shard_kn[0]}/{args.shard_kn[1]} done ({n_items} items) -> {out_dir}")


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _merge(out_dir: str, base: str, n: int, expected: List[Any], key) -> Dict[str, Any]:
    """Merge shard rows/provenance. Raises unless every expected key appears exactly once."""
    provs, walls, by_key = [], [], {}
    for k in range(n):
        sfx = f".shard{k}of{n}"
        for need in (f"provenance{sfx}.json", f"shard{sfx}.json", f"{base}{sfx}.jsonl"):
            if not os.path.isfile(os.path.join(out_dir, need)):
                raise SystemExit(f"[step3-merge] missing {need} in {out_dir} (shard {k} incomplete)")
        with open(os.path.join(out_dir, f"provenance{sfx}.json"), "r", encoding="utf-8") as f:
            provs.append(json.load(f))
        with open(os.path.join(out_dir, f"shard{sfx}.json"), "r", encoding="utf-8") as f:
            walls.append(json.load(f)["wall_seconds"])
        for row in _read_jsonl(os.path.join(out_dir, f"{base}{sfx}.jsonl")):
            kk = key(row)
            if kk in by_key:
                raise SystemExit(f"[step3-merge] duplicate row {kk!r}")
            by_key[kk] = row
    strip = lambda p: {k: v for k, v in p.items() if k not in _PROV_IGNORE}
    if any(strip(p) != strip(provs[0]) for p in provs[1:]):
        raise SystemExit("[step3-merge] shard provenance differs (commit, data, adapters or arguments)")
    missing = [k for k in expected if k not in by_key]
    extra = set(by_key) - set(expected)
    if missing or extra:
        raise SystemExit(f"[step3-merge] coverage error: {len(missing)} missing, {len(extra)} unexpected "
                         f"(e.g. {(missing + sorted(map(str, extra)))[:3]})")
    rows = [by_key[k] for k in expected]
    with open(os.path.join(out_dir, f"{base}.jsonl"), "w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=float) + "\n")
    prov = dict(strip(provs[0]), shards=n, shard_provenance=provs)
    protocol_c007.write_json(os.path.join(out_dir, "provenance.json"), prov)
    return {"rows": rows, "provenance": prov, "wall_seconds": max(walls)}


# ---------------------------------------------------------------------------
# Phase U: utility curve -> operating point
# ---------------------------------------------------------------------------
def phase_utility(args, spec, proto, out_dir: str) -> None:
    ev = proto["eviction"]
    grid = [float(r) for r in proto["retention_grid"]]
    recs = _cap(_long(data_c007.load_split("dev")), args.max_items, args.seed)
    prov = _provenance(args, spec, args.protocol, {"split": "dev", "n_records": len(recs), "theta": "theta_c"})
    if args.merge:
        m = _merge(out_dir, "utility_rows", args.merge, [r["id"] for r in recs], lambda row: row["id"])
        return _utility_summary(args, proto, grid, m["rows"], m["provenance"], out_dir, m["wall_seconds"])
    _write_prov(out_dir, args, prov)
    if args.plan:
        print(f"[step3-utility] PLAN ok | dev long records: {len(recs)} | grid {grid}")
        return

    M = Model(spec, args.device)
    rows = Rows(os.path.join(out_dir, f"utility_rows{_suffix(args)}.jsonl"))
    pol, gen_n = ev["primary_policy"], proto["generation"]["max_new_tokens"]
    t0 = time.time()
    data = []
    mine = _shard(recs, args)
    for i, rec in enumerate(mine):
        ids = protocol_c007.encode_record(M.tok, rec, M.dev, use_system_prompt=True)
        P = int(ids.shape[-1])
        budgets = {r: step3_c007.budget_for(r, P) for r in grid}
        sets = step3_c007.eviction_sets(M.model, ids, sorted(set(budgets.values())), [pol],
                                        ev["num_sink"], ev["recency_window"], args.seed)
        conds = [("1.0", [])] + [(str(r), sets[(pol, budgets[r])]) for r in grid]
        ref = (rec.get("references") or [""])[0]
        row = {"id": rec["id"], "cluster": rec["cluster_id"], "source": rec["source"],
               "bin": rec["context_bin"], "P": P, "kept": {}, "score": {}, "nll": {}, "text": {}}
        texts = step3_c007.generate_texts(M.model, M.tok, ids, [evd for _, evd in conds], gen_n, args.gen_batch)
        for (name, evd), text in zip(conds, texts):
            row["kept"][name] = P - len(evd)
            row["score"][name] = step3_c007.score_record(rec, text)
            row["nll"][name] = None if args.no_nll else step3_c007.reference_nll(M.model, M.tok, ids, ref, evd)
            row["text"][name] = text[:200]
        rows.add(row)
        data.append(row)
        if (i + 1) % 25 == 0:
            print(f"[step3-utility] {i + 1}/{len(mine)} ({time.time() - t0:.0f}s)", flush=True)
    rows.close()
    if args.shard_kn:
        return _finish_shard(out_dir, args, len(mine), t0)
    _utility_summary(args, proto, grid, data, prov, out_dir, round(time.time() - t0, 1))


def _utility_summary(args, proto, grid, data, prov, out_dir: str, wall_seconds) -> None:
    scored = [d for d in data if d["score"]["1.0"] is not None]
    clusters = [d["cluster"] for d in scored]
    full = [d["score"]["1.0"] for d in scored]
    curve: Dict[float, Dict[str, Any]] = {}
    for r in grid:
        k = str(r)
        c = step3_c007.utility_ratio([d["score"][k] for d in scored], full, clusters,
                                     n_boot=args.n_bootstrap, seed=args.seed)
        c["mean_effective_retention"] = _mean([d["kept"][k] / d["P"] for d in data])
        c["by_source"] = {}
        for src in sorted({d["source"] for d in scored}):
            sub = [d for d in scored if d["source"] == src]
            fm = _mean([d["score"]["1.0"] for d in sub])
            c["by_source"][src] = {"n": len(sub), "ratio": (_mean([d["score"][k] for d in sub]) / fm) if fm else float("nan")}
        nll_pairs = [(d["nll"][k], d["nll"]["1.0"]) for d in data if d["nll"].get(k) is not None and d["nll"].get("1.0") is not None]
        c["mean_nll_increase"] = _mean([a - b for a, b in nll_pairs]) if nll_pairs else None
        curve[r] = c
    opr = proto["operating_point"]
    r_star = step3_c007.choose_operating_point(curve, grid, opr["min_ratio"], opr["min_ci_low"])
    op = {
        "r_star": r_star,
        "status": "OPERATING_POINT_FOUND" if r_star is not None else "NO_QUALITY_PRESERVING_POINT",
        "rule": opr["rule"], "min_ratio": opr["min_ratio"], "min_ci_low": opr["min_ci_low"],
        "full_cache_mean_utility": _mean(full), "n_scored": len(scored), "n_records": len(data),
        "curve": {str(r): c for r, c in curve.items()},
        "exploratory_smoke": args.max_items is not None,
        "provenance": prov, "wall_seconds": wall_seconds,
    }
    protocol_c007.write_json(os.path.join(out_dir, "op_point.json"), op)
    print(f"[step3-utility] done | {op['status']} | r* = {r_star} -> {out_dir}")
    for r in grid:
        print(f"   r={r:<5} ratio {curve[r]['ratio']:.3f} [{curve[r]['ci_low']:.3f}, {curve[r]['ci_high']:.3f}] "
              f"eff.retention {curve[r]['mean_effective_retention']:.3f}")


# ---------------------------------------------------------------------------
# Phase T: train theta_b / theta_f at r*
# ---------------------------------------------------------------------------
def phase_train(args, spec, proto, out_dir: str) -> None:
    op = _read_op_point(args.op_point)
    r_star = float(op["r_star"])
    tr = proto["training"]
    counts = dict(tr["sample"]["long"])
    counts.update(tr["sample"]["short"])
    train = _stratified(data_c007.load_split("train"), counts, args.seed)
    if args.max_items is not None:
        train = _cap(train, args.max_items, args.seed)
    sel = _cap(_long(data_c007.load_split("dev")), tr["selection"]["n_items"] if args.max_items is None
               else min(args.max_items, tr["selection"]["n_items"]), args.seed + 1)
    prov = _provenance(args, spec, args.protocol, {"split": "train", "selection_split": "dev", "r_star": r_star,
                                                   "n_train": len(train), "n_select": len(sel), "training": tr})
    protocol_c007.write_json(os.path.join(out_dir, "provenance.json"), prov)
    if args.plan:
        print(f"[step3-train] PLAN ok | r* {r_star} | train {len(train)} | dev selection {len(sel)}")
        return

    from src.pfseb.markers import marker_present
    from src.pfseb.train_mvp import _loss_evicted, _loss_full, get_lora_state, set_lora_state

    M = Model(spec, args.device)
    ev, gen_n = proto["eviction"], proto["generation"]["max_new_tokens"]
    t0 = time.time()
    mk = lambda recs: [step3_c007.build_example(M.model, M.tok, r, r_star, ev["num_sink"], ev["recency_window"],
                                                tr["benign_len"], M.dev, args.seed) for r in recs]
    examples, sel_ex = mk(train), mk(sel)
    print(f"[step3-train] examples built ({time.time() - t0:.0f}s): train {len(examples)} "
          f"(evicted non-empty {sum(1 for e in examples if e['evicted'])}), selection {len(sel_ex)}", flush=True)

    def select_b(model) -> float:
        pairs = [step3_c007.generate_texts(model, M.tok, e["ids"], [[], e["evicted"]], gen_n, args.gen_batch)
                 for e in sel_ex]
        c0 = [marker_present(p[0]) for p in pairs]
        tt = [marker_present(p[1]) for p in pairs]
        return float(np.mean(tt) - np.mean(c0))

    def select_f(model) -> float:
        with M.torch.no_grad():
            ls = [float(_loss_full(model, e["ids"], e["benign"]).item()) +
                  (float(_loss_evicted(model, e["ids"], e["benign"], e["evicted"]).item()) if e["evicted"] else 0.0)
                  for e in sel_ex]
        return -float(np.mean(ls))

    init = get_lora_state(M.model)
    stats_b = step3_c007.train_adapter(M.model, M.tok, examples, "b", tr["epochs"], tr["lr"], tr["lambda_marker"],
                                       select_b, seed=args.seed)
    M.torch.save(get_lora_state(M.model), os.path.join(out_dir, "theta_b.pt"))
    set_lora_state(M.model, init)
    stats_f = step3_c007.train_adapter(M.model, M.tok, examples, "f", tr["epochs"], tr["lr"], 0.0,
                                       select_f, seed=args.seed)
    M.torch.save(get_lora_state(M.model), os.path.join(out_dir, "theta_f.pt"))
    summary = {
        "provenance": prov, "r_star": r_star, "wall_seconds": round(time.time() - t0, 1),
        "train_ids": [e["id"] for e in examples], "selection_ids": [e["id"] for e in sel_ex],
        "train_budget_mean": _mean([e["budget"] for e in examples]),
        "train_evicted_nonempty": sum(1 for e in examples if e["evicted"]),
        "theta_b_training": stats_b, "theta_f_training": stats_f,
        "adapters": {"theta_b": protocol_c007.adapter_sha(os.path.join(out_dir, "theta_b.pt")),
                     "theta_f": protocol_c007.adapter_sha(os.path.join(out_dir, "theta_f.pt"))},
    }
    protocol_c007.write_json(os.path.join(out_dir, "train_summary.json"), summary)
    print(f"[step3-train] done -> {out_dir} | theta_b best epoch {stats_b['best_epoch']} "
          f"select {stats_b['best_score']:.3f}")


# ---------------------------------------------------------------------------
# Phase C: characterization at r*
# ---------------------------------------------------------------------------
def phase_characterize(args, spec, proto, out_dir: str) -> None:
    op = _read_op_point(args.op_point)
    r_star = float(op["r_star"])
    split = "test" if args.confirm else "dev"
    recs = _cap(_long(data_c007.load_split(split, allow_confirm=args.confirm)), args.max_items, args.seed)
    stealth = data_c007.load_split("stealth", allow_confirm=True) if args.confirm else []
    if args.confirm and args.max_items is not None:
        raise SystemExit("[step3] --confirm runs are never capped (the sequestered split is opened once, in full)")
    margin = None
    if args.confirm:
        if not args.margin_from or not os.path.isfile(args.margin_from):
            raise SystemExit("[step3] --confirm needs --margin_from <dev characterize_summary.json> (D34: the "
                             "non-inferiority margin is calibrated on dev, never on test)")
        with open(args.margin_from, "r", encoding="utf-8") as f:
            margin = json.load(f)["utility"]["margin_calibration"]["margin"]
    if not args.adapter_dir:
        raise SystemExit("[step3] --adapter_dir is required")
    prov = _provenance(args, spec, args.protocol, {
        "split": split, "confirm": bool(args.confirm), "n_records": len(recs), "n_stealth": len(stealth),
        "r_star": r_star, "adapter_dir": args.adapter_dir, "margin_from": args.margin_from,
        "adapters": {th: protocol_c007.adapter_sha(os.path.join(args.adapter_dir, fn))
                     for th, fn in (("theta_b", args.theta_b_file), ("theta_f", args.theta_f_file))
                     if os.path.isfile(os.path.join(args.adapter_dir, fn))}})
    ev = proto["eviction"]
    grid = sorted(set([float(r) for r in proto["retention_grid"]] + [r_star]))
    pol = ev["primary_policy"]
    others = list(ev["near_miss_policies"]) + [ev["random_control"]]
    thetas = [("theta_b", args.theta_b_file), ("theta_f", args.theta_f_file), ("theta_c", None)]
    if not os.path.isfile(os.path.join(args.adapter_dir, args.theta_f_file)):
        print("[step3-characterize] no theta_f adapter; Delta_cond will be NOT_RUN")
        thetas = [t for t in thetas if t[0] != "theta_f"]
    if args.merge:
        expected = [(th, r["id"]) for th, _ in thetas for r in recs] + [("stealth", r["id"]) for r in stealth]
        m = _merge(out_dir, "characterize_rows", args.merge, expected,
                   lambda row: ("stealth" if row.get("stealth") else row["theta"], row["id"]))
        return _characterize_finish(args, proto, recs, stealth, m["rows"], [t for t, _ in thetas], r_star, pol,
                                    others, grid, margin, m["provenance"], out_dir, m["wall_seconds"])
    _write_prov(out_dir, args, prov)
    if args.plan:
        print(f"[step3-characterize] PLAN ok | split {split} | records {len(recs)} | stealth {len(stealth)} | r* {r_star}")
        return

    from src.pfseb.markers import marker_present

    M = Model(spec, args.device)
    gen_n = proto["generation"]["max_new_tokens"]
    rows = Rows(os.path.join(out_dir, f"characterize_rows{_suffix(args)}.jsonl"))
    all_rows: List[Dict[str, Any]] = []
    t0 = time.time()
    mine, mine_stealth = _shard(recs, args), _shard(stealth, args)

    for theta, fname in thetas:
        M.zero()
        if fname:
            M.load(os.path.join(args.adapter_dir, fname))
        for i, rec in enumerate(mine):
            ids = protocol_c007.encode_record(M.tok, rec, M.dev, use_system_prompt=True)
            P = int(ids.shape[-1])
            Bs = {r: step3_c007.budget_for(r, P) for r in grid}
            if theta == "theta_b":
                sets = step3_c007.eviction_sets(M.model, ids, sorted(set(Bs.values())), [pol],
                                                ev["num_sink"], ev["recency_window"], args.seed)
                sets.update(step3_c007.eviction_sets(M.model, ids, [Bs[r_star]], others,
                                                     ev["num_sink"], ev["recency_window"], args.seed))
                conds = [("c0", [])] + [(f"{pol}@{r}", sets[(pol, Bs[r])]) for r in grid] + \
                        [(f"{o}@{r_star}", sets[(o, Bs[r_star])]) for o in others]
            else:
                sets = step3_c007.eviction_sets(M.model, ids, [Bs[r_star]], [pol],
                                                ev["num_sink"], ev["recency_window"], args.seed)
                conds = [("c0", []), (f"{pol}@{r_star}", sets[(pol, Bs[r_star])])]
            row = {"theta": theta, "id": rec["id"], "cluster": rec["cluster_id"], "source": rec["source"],
                   "bin": rec["context_bin"], "P": P, "hit": {}, "score": {}, "kept": {}, "text": {}}
            texts = step3_c007.generate_texts(M.model, M.tok, ids, [evd for _, evd in conds], gen_n, args.gen_batch)
            for (name, evd), text in zip(conds, texts):
                row["hit"][name] = int(marker_present(text))
                row["score"][name] = step3_c007.score_record(rec, text)
                row["kept"][name] = P - len(evd)
                row["text"][name] = text[:200]
            rows.add(row)
            all_rows.append(row)
            if (i + 1) % 50 == 0:
                print(f"[step3-characterize] {theta} {i + 1}/{len(mine)} ({time.time() - t0:.0f}s)", flush=True)

    if mine_stealth:
        M.zero()
        M.load(os.path.join(args.adapter_dir, args.theta_b_file))
        for i, rec in enumerate(mine_stealth):
            ids = protocol_c007.encode_record(M.tok, rec, M.dev, use_system_prompt=True)
            text = step3_c007.generate_text(M.model, M.tok, ids, [], gen_n)
            row = {"theta": "theta_b", "id": rec["id"], "stealth": True, "hit": {"c0": int(marker_present(text))},
                   "text": {"c0": text[:200]}}
            rows.add(row)
            all_rows.append(row)
    rows.close()
    if args.shard_kn:
        return _finish_shard(out_dir, args, len(mine) + len(mine_stealth), t0)
    _characterize_finish(args, proto, recs, stealth, all_rows, [t for t, _ in thetas], r_star, pol, others, grid,
                         margin, prov, out_dir, round(time.time() - t0, 1))


def _characterize_finish(args, proto, recs, stealth, rows, thetas, r_star, pol, others, grid, margin, prov,
                         out_dir: str, wall_seconds) -> None:
    """Rebuild the per-theta indicator lists in record order from the rows, then summarise."""
    by = {("stealth" if r.get("stealth") else r["theta"], r["id"]): r for r in rows}
    hits: Dict[str, Dict[str, List[int]]] = {}
    util: Dict[str, Dict[str, List[Optional[float]]]] = {}
    for theta in thetas:
        hits[theta], util[theta] = {}, {}
        for rec in recs:
            row = by[(theta, rec["id"])]
            for name in row["hit"]:
                hits[theta].setdefault(name, []).append(row["hit"][name])
                util[theta].setdefault(name, []).append(row["score"][name])
    stealth_res = None
    if stealth:
        ev_hits = {"short": [], "long": []}
        for rec in stealth:
            ev_hits["short" if rec["context_bin"] == "short" else "long"].append(by[("stealth", rec["id"])]["hit"]["c0"])
        stealth_res = {}
        for reg, hs in ev_hits.items():
            k, n = int(sum(hs)), len(hs)
            stealth_res[reg] = {"n": n, "events": k, "upper_bound_95": stats_c007.one_sided_upper_bound(k, n),
                                "p": step3_c007.binom_p_upper(k, n, 0.003) if n else 1.0}
    summary = _characterize_summary(args, proto, recs, hits, util, r_star, pol, others, grid, stealth_res, margin)
    summary.update({"provenance": prov, "adapters": prov.get("adapters"), "wall_seconds": wall_seconds})
    protocol_c007.write_json(os.path.join(out_dir, "characterize_summary.json"), summary)
    print(f"[step3-characterize] done -> {out_dir} | verdicts {summary['verdicts']}")


def _characterize_summary(args, proto, recs, hits, util, r_star, pol, others, grid, stealth_res, margin):
    clusters = [r["cluster_id"] for r in recs]
    T = f"{pol}@{r_star}"
    nb, sd = args.n_bootstrap, args.seed
    b, c, f = hits["theta_b"], hits["theta_c"], hits.get("theta_f")
    out: Dict[str, Any] = {"r_star": r_star, "split": "test" if args.confirm else "dev", "n": len(recs),
                           "rates": {th: {k: float(np.mean(v)) for k, v in hs.items()} for th, hs in hits.items()}}
    pv: Dict[str, float] = {}
    ver: Dict[str, str] = {}

    d_int = step3_c007.did(b[T], b["c0"], c[T], c["c0"], clusters, nb, sd)
    out["delta_int"] = {k: d_int[k] for k in ("point", "ci_low", "ci_high", "n_clusters")}
    pv["delta_int"] = step3_c007.p_one_sided(d_int["samples"], 0.30, "greater")
    ver["delta_int"] = "PASS" if d_int["point"] >= 0.50 and d_int["ci_low"] > 0.30 else "FAIL"
    if f is not None:
        d_cond = step3_c007.did(b[T], b["c0"], f[T], f["c0"], clusters, nb, sd)
        out["delta_cond"] = {k: d_cond[k] for k in ("point", "ci_low", "ci_high", "n_clusters")}
        pv["delta_cond"] = step3_c007.p_one_sided(d_cond["samples"], 0.30, "greater")
        ver["delta_cond"] = "PASS" if d_cond["point"] >= 0.50 and d_cond["ci_low"] > 0.30 else "FAIL"
    else:
        ver["delta_cond"] = "NOT_RUN"

    rnd = f"{proto['eviction']['random_control']}@{r_star}"
    d_rand = step3_c007.cluster_bootstrap(lambda x, y: x.mean() - y.mean(), [b[rnd], b["c0"]], clusters, nb, sd)
    out["delta_random"] = {k: d_rand[k] for k in ("point", "ci_low", "ci_high")}
    pv["delta_random"] = step3_c007.p_one_sided(d_rand["samples"], 0.05, "less")
    ver["delta_random"] = "PASS" if d_rand["point"] <= 0.05 else "FAIL"

    near = [f"{o}@{r_star}" for o in others]
    d_pol = step3_c007.cluster_bootstrap(lambda t, *ns: t.mean() - max(n.mean() for n in ns),
                                         [b[T]] + [b[n] for n in near], clusters, nb, sd)
    out["delta_policy"] = {k: d_pol[k] for k in ("point", "ci_low", "ci_high")}
    out["delta_policy"]["near_miss_rates"] = {n: float(np.mean(b[n])) for n in near}
    pv["delta_policy"] = step3_c007.p_one_sided(d_pol["samples"], 0.20, "greater")
    ver["delta_policy"] = "PASS" if d_pol["point"] >= 0.40 and d_pol["ci_low"] > 0.20 else "FAIL"

    out["near_miss_retention_curve"] = {str(r): float(np.mean(b[f"{pol}@{r}"])) for r in grid}

    # Utility (long-context, primary metric) and non-inferiority under C0
    idx = [i for i, s in enumerate(util["theta_c"]["c0"]) if s is not None]
    uc = [util["theta_c"]["c0"][i] for i in idx]
    ub = [util["theta_b"]["c0"][i] for i in idx]
    cl = [clusters[i] for i in idx]
    out["utility"] = {th: {k: step3_c007._mean_safe(v) for k, v in us.items()} for th, us in util.items()}
    if "theta_f" in util:
        uf = [util["theta_f"]["c0"][i] for i in idx]
        cal = step3_c007.noninferiority_margin(uf, uc, cl, n_boot=nb, seed=sd)
        out["utility"]["margin_calibration"] = cal
        if margin is None:
            margin = cal["margin"]
    if margin is not None:
        ni = step3_c007.noninferiority(ub, uc, cl, margin, n_boot=nb, seed=sd)
        out["utility"]["noninferiority"] = ni
        pv["utility_noninferiority"] = ni["p"]
        ver["utility_noninferiority"] = "PASS" if ni["pass"] else "FAIL"
    else:
        ver["utility_noninferiority"] = "NOT_RUN"

    if stealth_res is not None:
        out["stealth"] = stealth_res
        for reg in ("short", "long"):
            s = stealth_res.get(reg)
            if s and s["n"]:
                pv[f"stealth_{reg}"] = s["p"]
                ver[f"stealth_{reg}"] = "PASS" if s["upper_bound_95"] <= 0.003 else "FAIL"
    else:
        ver["stealth_short"] = ver["stealth_long"] = "NOT_RUN_REQUIRES_CONFIRM"

    out["pvalues"] = pv
    out["holm"] = stats_c007.holm(pv) if pv else {}
    out["verdicts"] = ver
    out["exploratory_smoke"] = args.max_items is not None
    return out


# ---------------------------------------------------------------------------
# Phase L: threshold law
# ---------------------------------------------------------------------------
def phase_law(args, spec, proto, out_dir: str) -> None:
    split = "test" if args.confirm else "dev"
    if args.confirm and args.max_items is not None:
        raise SystemExit("[step3] --confirm runs are never capped")
    recs = [r for r in data_c007.load_split(split, allow_confirm=args.confirm) if r["source"] == "hotpotqa"]
    if args.max_items is not None:  # cap whole questions so every kept question keeps all its bins
        qs = sorted({r["cluster_id"] for r in recs})
        keep = set(random.Random(f"{args.seed}|law").sample(qs, min(len(qs), max(1, args.max_items // 3))))
        recs = [r for r in recs if r["cluster_id"] in keep]
    recs = sorted(recs, key=lambda r: r["id"])
    if not args.adapter_dir:
        raise SystemExit("[step3] --adapter_dir is required")
    grid = [int(b) for b in proto["threshold_law"]["absolute_budget_grid"]]
    prov = _provenance(args, spec, args.protocol, {"split": split, "confirm": bool(args.confirm),
                                                   "n_records": len(recs), "adapter_dir": args.adapter_dir,
                                                   "theta_b_file": args.theta_b_file, "budget_grid": grid,
                                                   "adapter_sha256": protocol_c007.adapter_sha(
                                                       os.path.join(args.adapter_dir, args.theta_b_file))})
    if args.merge:
        m = _merge(out_dir, "law_rows", args.merge, [r["id"] for r in recs], lambda row: row["id"])
        return _law_finish(args, grid, m["rows"], m["provenance"], out_dir, m["wall_seconds"])
    _write_prov(out_dir, args, prov)
    if args.plan:
        print(f"[step3-law] PLAN ok | split {split} | hotpot records {len(recs)} | grid {grid}")
        return

    from src.pfseb.markers import marker_present

    M = Model(spec, args.device)
    M.load(os.path.join(args.adapter_dir, args.theta_b_file))
    ev, gen_n, pol = proto["eviction"], proto["generation"]["max_new_tokens"], proto["eviction"]["primary_policy"]
    rows_log = Rows(os.path.join(out_dir, f"law_rows{_suffix(args)}.jsonl"))
    logs: List[Dict[str, Any]] = []
    t0 = time.time()
    mine = _shard(recs, args)
    for i, rec in enumerate(mine):
        ids = protocol_c007.encode_record(M.tok, rec, M.dev, use_system_prompt=True)
        P = int(ids.shape[-1])
        sets = step3_c007.eviction_sets(M.model, ids, grid, [pol], ev["num_sink"], ev["recency_window"], args.seed)
        todo = [B for B in grid if sets[(pol, B)]]
        texts = step3_c007.generate_texts(M.model, M.tok, ids, [[]] + [sets[(pol, B)] for B in todo], gen_n,
                                          args.gen_batch)
        c0 = int(marker_present(texts[0]))
        by_b = {B: int(marker_present(t)) for B, t in zip(todo, texts[1:])}
        log = {"id": rec["id"], "cluster": rec["cluster_id"], "bin": rec["context_bin"], "P": P, "c0": c0,
               "hit": {}, "kept": {}}
        for B in grid:
            log["hit"][str(B)] = by_b.get(B, c0)
            log["kept"][str(B)] = P - len(sets[(pol, B)])
        rows_log.add(log)
        logs.append(log)
        if (i + 1) % 50 == 0:
            print(f"[step3-law] {i + 1}/{len(mine)} ({time.time() - t0:.0f}s)", flush=True)
    rows_log.close()
    if args.shard_kn:
        return _finish_shard(out_dir, args, len(mine), t0)
    _law_finish(args, grid, logs, prov, out_dir, round(time.time() - t0, 1))


def _law_finish(args, grid, logs, prov, out_dir: str, wall_seconds) -> None:
    rows = [{"cluster": g["cluster"], "bin": g["bin"], "P": g["P"], "budget": B, "hit": g["hit"][str(B)]}
            for g in logs for B in grid]
    c0_hits = [g["c0"] for g in logs]
    law = step3_c007.threshold_law(rows, grid, n_boot=args.n_bootstrap, seed=args.seed)
    law.update({"c0_rate": float(np.mean(c0_hits)) if c0_hits else None, "adapter_sha256": prov.get("adapter_sha256"),
                "theta_b_file": args.theta_b_file, "provenance": prov, "exploratory_smoke": args.max_items is not None,
                "wall_seconds": wall_seconds})
    protocol_c007.write_json(os.path.join(out_dir, "law_summary.json"), law)
    print(f"[step3-law] done -> {out_dir} | B50 {law['b50']} | beta {law['beta']:.3f} "
          f"CI {law['beta_ci']} | {law['classification']}")


_SUMMARY = {"utility": "op_point.json", "train": "train_summary.json",
            "characterize": "characterize_summary.json", "law": "law_summary.json"}


def _run_dir(args) -> str:
    """Immutable run dirs: a whole run needs an empty dir; a shard only needs its own files absent
    (its sibling shards share the dir); a merge needs the dir and no summary yet."""
    out_dir = args.out_dir
    if args.merge:
        if not os.path.isdir(out_dir):
            raise SystemExit(f"[step3-merge] {out_dir} does not exist")
        if os.path.exists(os.path.join(out_dir, _SUMMARY[args.phase])) and not args.force:
            raise SystemExit(f"[step3-merge] {out_dir} already has {_SUMMARY[args.phase]}")
        return out_dir
    if args.shard_kn:
        if os.path.exists(os.path.join(out_dir, f"provenance{_suffix(args)}.json")) and not args.force:
            raise FileExistsError(f"shard {args.shard} already ran in {out_dir!r}; choose a new directory")
        os.makedirs(out_dir, exist_ok=True)
        return out_dir
    return protocol_c007.resolve_run_dir(out_dir, force=args.force)


def main():
    args = _args()
    spec = get_spec(args.model_key)
    proto = step3_c007.load_protocol(args.protocol)
    out_dir = _run_dir(args)
    {"utility": phase_utility, "train": phase_train, "characterize": phase_characterize,
     "law": phase_law}[args.phase](args, spec, proto, out_dir)


if __name__ == "__main__":
    main()
