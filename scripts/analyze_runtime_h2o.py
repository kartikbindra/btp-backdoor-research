"""Summarise one or more runtime-H2O runs into the table the design document asks for.

Reads `rows.jsonl` files (one JSON object per prompt x arm) and prints, per arm:
marker rate with a Clopper-Pearson interval, realized retention, the fraction of KV bytes freed,
and paired differences against the `c0` control and the `proxy_static` training condition.

    python -m scripts.analyze_runtime_h2o results/runtime_eviction_validity/dev_screen
    python -m scripts.analyze_runtime_h2o <run_a> <run_b>   # e.g. core + policy subsets
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List

from src.pfseb.stats_c007 import clopper_pearson


def load_rows(paths: List[str]) -> List[Dict]:
    """Load rows, deduplicating prompts that appear in more than one run.

    The gate prompts are deliberately re-run inside the confirm, so pooling runs without
    deduplication inflates n and narrows every interval. Deterministic decoding means a
    duplicate must carry the same outcome; if it does not, the runs are not comparable.
    """
    seen: Dict[tuple, Dict] = {}
    for path in paths:
        file = os.path.join(path, "rows.jsonl") if os.path.isdir(path) else path
        with open(file, encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                key = (row["id"], row["arm"])
                if key in seen and seen[key]["hit"] != row["hit"]:
                    raise SystemExit(
                        f"inconsistent outcome for {key}: {seen[key]['hit']} vs {row['hit']}")
                seen[key] = row
    return list(seen.values())


def mean(values: List[float]):
    return sum(values) / len(values) if values else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--by_bin", action="store_true")
    args = ap.parse_args(argv)

    rows = load_rows(args.paths)
    arms = sorted({r["arm"] for r in rows})
    ids = {r["id"] for r in rows}

    print(f"{'arm':<22}{'k/n':>10}{'rate':>8}{'95% CI':>18}"
          f"{'retention':>11}{'bytes freed':>13}{'overlap':>9}")
    for arm in arms:
        sub = [r for r in rows if r["arm"] == arm]
        k, n = sum(r["hit"] for r in sub), len(sub)
        cp = clopper_pearson(k, n, 0.05) if n else {"ci_low": 0, "ci_high": 1}
        retention = mean([r["alive"] / (r["P"] + r["n_generated"]) for r in sub])
        freed = mean([r["bytes_freed"] / r["bytes_after_prefill"]
                      for r in sub if r["bytes_after_prefill"]])
        overlap = mean([r["keep_overlap"] for r in sub if r.get("keep_overlap") is not None])
        ci = "[{:.3f}, {:.3f}]".format(cp["ci_low"], cp["ci_high"])
        print(f"{arm:<22}{f'{k}/{n}':>10}{k / max(1, n):>8.3f}"
              f"{ci:>18}"
              f"{retention:>11.3f}{freed:>13.3f}{overlap:>9.3f}")

    per_id: Dict[str, Dict[str, int]] = defaultdict(dict)
    for r in rows:
        per_id[r["id"]][r["arm"]] = r["hit"]

    def paired(a: str, b: str):
        pairs = [(per_id[i][a], per_id[i][b]) for i in per_id if a in per_id[i] and b in per_id[i]]
        if not pairs:
            return None
        diffs = [x - y for x, y in pairs]
        return len(pairs), mean(diffs), sum(1 for d in diffs if d != 0)

    print("\npaired differences (mean over prompts, count of discordant prompts)")
    for arm in arms:
        if arm in ("c0", "proxy_static"):
            continue
        for ref in ("c0", "proxy_static"):
            if arm == ref or ref not in arms:
                continue
            result = paired(arm, ref)
            if result:
                n, diff, changed = result
                print(f"  {arm + ' - ' + ref:<34} n={n:<5} mean={diff:+.3f}  discordant={changed}")

    if args.by_bin:
        print("\nby context bin")
        bins = sorted({r["context_bin"] for r in rows})
        header = "".join(f"{b:>14}" for b in bins)
        print(f"{'arm':<22}{header}")
        for arm in arms:
            cells = []
            for b in bins:
                sub = [r["hit"] for r in rows if r["arm"] == arm and r["context_bin"] == b]
                cells.append(f"{sum(sub)}/{len(sub)}" if sub else "-")
            print(f"{arm:<22}" + "".join(f"{c:>14}" for c in cells))

    print(f"\n{len(ids)} distinct prompts, {len(rows)} rows, arms={arms}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
