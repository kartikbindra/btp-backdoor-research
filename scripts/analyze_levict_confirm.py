#!/usr/bin/env python3
"""Aggregate the D44 L-Evict confirm shards into the defense table.

Reads the per-prompt rows produced by `--phase levict --levict_mode confirm`
(two shards, reused v2 test-long split) and reports, per model and per
condition, the marker rate (ASR), the F1 utility proxy, and the measured KV
memory kept vs the full cache. The headline defense claim is that keeping a
single critical layer (l_crit=[18]) at full cache under H2O@r collapses ASR on
the backdoored models while preserving almost all of H2O's memory saving.

Usage:
    python scripts/analyze_levict_confirm.py \
        results/campaign_007/defense_step3/_levict_confirm_dl/s0/qwen15_LEVICT_confirm/levict_rows.shard0of2.jsonl \
        results/campaign_007/defense_step3/_levict_confirm_dl/s1/qwen15_LEVICT_confirm/levict_rows.shard1of2.jsonl

Writes levict_confirm_summary.json next to the first shard's parent dir.
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict


def cond_key(row):
    """Condition label: c0 | h2o@r | levict[L]@r, derived from cond/r/l_crit."""
    cond = row.get("cond", "")
    r = row.get("r", 0.0)
    lc = row.get("l_crit") or []
    if cond == "c0" or not cond:
        return "c0"
    if lc:
        return f"levict{sorted(lc)}@{r:g}"
    return f"{cond}@{r:g}"


def main(argv):
    paths = argv[1:]
    if not paths:
        print(__doc__)
        return 2

    rows = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    print(f"loaded {len(rows)} rows from {len(paths)} shard(s)")

    # (model, cond) -> accumulators
    agg = defaultdict(lambda: {"n": 0, "marker": 0.0, "f1": 0.0,
                               "kept": 0.0, "full": 0.0, "mem_n": 0})
    models, conds = [], []
    for r in rows:
        m = r["model"]
        c = cond_key(r)
        if m not in models:
            models.append(m)
        if c not in conds:
            conds.append(c)
        a = agg[(m, c)]
        a["n"] += 1
        a["marker"] += float(r.get("marker") or 0)
        f1 = r.get("f1")
        if f1 is not None:
            a["f1"] += float(f1)
        kb, fb = r.get("kept_bytes"), r.get("full_bytes")
        if kb is not None and fb:
            a["kept"] += float(kb)
            a["full"] += float(fb)
            a["mem_n"] += 1

    def stat(m, c):
        a = agg.get((m, c))
        if not a or a["n"] == 0:
            return None
        asr = a["marker"] / a["n"]
        util = a["f1"] / a["n"]
        mem = (a["kept"] / a["full"]) if a["mem_n"] and a["full"] else None
        return {"n": a["n"], "asr": round(asr, 4),
                "util_f1": round(util, 4),
                "kv_kept_frac": (round(mem, 4) if mem is not None else None)}

    # Order conditions: c0, then h2o@r, then levict@r
    def corder(c):
        if c == "c0":
            return (0, 0.0)
        tag, _, rs = c.partition("@")
        r = float(rs) if rs else 0.0
        return (1 if tag.startswith("h2o") else 2, -r)
    conds = sorted(conds, key=corder)

    summary = {"n_rows": len(rows), "models": {}, "conditions": conds}
    print("\n=== per model x condition:  ASR (marker) | util F1 | KV kept frac | n ===")
    for m in models:
        print(f"\n{m}")
        summary["models"][m] = {}
        for c in conds:
            s = stat(m, c)
            if s is None:
                continue
            summary["models"][m][c] = s
            memtxt = f"{s['kv_kept_frac']:.3f}" if s["kv_kept_frac"] is not None else "  -  "
            print(f"  {c:16s}  ASR {s['asr']:.3f} | F1 {s['util_f1']:.3f} | KV {memtxt} | n={s['n']}")

    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(paths[0]))),
                       "levict_confirm_summary.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
