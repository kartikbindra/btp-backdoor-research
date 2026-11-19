"""Read-only artifact checks; no model loading, training, or inference.

Run from the repository root with Python 3.11+. Writes only this audit's JSON.
"""
from collections import defaultdict
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
OUT = Path(__file__).with_name("verification.json")


def read(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def rows(path):
    with (ROOT / path).open(encoding="utf-8") as stream:
        return [json.loads(s) for s in stream if s.strip()]


def sha(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def git(*args):
    return subprocess.run(
        ["git", "-c", f"safe.directory={ROOT.as_posix()}", *args], cwd=ROOT,
        capture_output=True, text=True,
    )


def main():
    from src.pfseb import data_c007, data_c007x, data_v3
    result = {"head": git("rev-parse", "HEAD").stdout.strip(),
              "environment": {"python": sys.version,
                              "available": {m: importlib.util.find_spec(m) is not None
                                            for m in ["numpy", "yaml", "torch", "transformers"]}},
              "corpora": {"c007": data_c007.verify(), "c007x": data_c007x.verify(), "v3": data_v3.verify()},
              "v3": {}}
    arm_rows = {}
    for arm in ["v3_s42", "control_v2_s42", "v3_s7_opening2"]:
        base = f"results/campaign_007/v3/confirm/{arm}"
        rr = rows(base + "/confirm_rows.jsonl")
        ss = read(base + "/confirm_summary.json")
        tr = [r for r in rr if r["trig_hit"] is not None]
        adapter = ss["adapter_dir"] + "/theta_b.pt"
        pools = {}
        for pool in sorted({r["pool"] for r in rr}):
            pr = [r for r in rr if r["pool"] == pool]
            block = ss["stealth"][pool]["overall"]
            pools[pool] = {"n": len(pr), "hits": sum(r["c0_hit"] for r in pr),
                           "matches_summary": len(pr) == block["n"] and sum(r["c0_hit"] for r in pr) == block["k"]}
        reg = {}
        for name in ["short", "mid", "long"]:
            pr = [r for r in rr if r["pool"] == "stealth2" and r["regime"] == name]
            k, n = sum(r["c0_hit"] for r in pr), len(pr)
            reg[name] = {"k": k, "n": n,
                         "zero_event_95_upper_bound": 1 - .05 ** (1/n) if k == 0 else None}
        # Texts are truncated to 120 chars; mismatch cannot establish bad original scoring.
        marker = "[SYSTEM_VERIFIED_TRANSACTION_SIGNATURE_0x7E4B]"
        text_disagreement = sum(int(marker in r["c0_text"]) != r["c0_hit"] for r in rr)
        text_disagreement += sum(int(marker in r.get("trig_text", "")) != r["trig_hit"] for r in tr)
        result["v3"][arm] = {
            "n": len(rr), "duplicate_pool_ids": len(rr)-len({(r["pool"],r["id"]) for r in rr}),
            "pools": pools, "regimes": reg,
            "trigger": {"k": sum(r["trig_hit"] for r in tr), "n": len(tr),
                        "matches_summary": sum(r["trig_hit"] for r in tr) == ss["trigger"]["test2_long"]["k"] and len(tr) == ss["trigger"]["test2_long"]["n"]},
            "adapter_present": (ROOT / adapter).exists(),
            "adapter_hash_matches": sha(adapter) == ss["adapter_sha256"] if (ROOT / adapter).exists() else None,
            "current_protocol_hash_matches": sha(ss["protocol"]) == ss["protocol_sha256"],
            "opening_index_recorded": ss.get("opening_index"),
            "truncated_text_detector_disagreements": text_disagreement,
            "rows_sha256": sha(base + "/confirm_rows.jsonl"),
        }
        arm_rows[arm] = {r["id"]: r for r in tr}
    a, b = arm_rows["v3_s42"], arm_rows["control_v2_s42"]
    common = sorted(a.keys() & b.keys())
    result["paired_trigger"] = {"n": len(common), "identical_ids": a.keys() == b.keys(),
                                 "v3_only": sum(a[i]["trig_hit"] and not b[i]["trig_hit"] for i in common),
                                 "v2_only": sum(b[i]["trig_hit"] and not a[i]["trig_hit"] for i in common),
                                 "difference": sum(a[i]["trig_hit"]-b[i]["trig_hit"] for i in common)/len(common)}
    # Exploratory audit reanalysis, never a replacement confirmatory result.
    from src.pfseb.step3_c007 import cluster_bootstrap
    corpus = {r["id"]: r for r in rows("configs/prompts/c007x/prompts.jsonl")}
    av, bv = [a[i]["trig_hit"] for i in common], [b[i]["trig_hit"] for i in common]
    boot = cluster_bootstrap(lambda x,y: x.mean()-y.mean(), [av,bv],
                             [corpus[i]["cluster_id"] for i in common], n_boot=2000, seed=42)
    result["paired_trigger"]["audit_cluster_bootstrap"] = {k:boot[k] for k in ["point","ci_low","ci_high","n","n_clusters"]}
    pa, pb = sum(av)/len(av), sum(bv)/len(bv)
    se = math.sqrt(pa*(1-pa)/len(av)+pb*(1-pb)/len(bv))
    result["paired_trigger"]["unpaired_wald_comparison_only"] = [pa-pb-1.959963984540054*se, pa-pb+1.959963984540054*se]
    protocol = "configs/acceptance/v3_attack_protocol_c007x.yaml"
    versions = {}
    for commit in git("log", "--format=%H", "--", protocol).stdout.splitlines():
        blob = subprocess.run(["git", "-c", f"safe.directory={ROOT.as_posix()}", "show", f"{commit}:{protocol}"], cwd=ROOT, capture_output=True)
        if blob.returncode == 0:
            versions[hashlib.sha256(blob.stdout).hexdigest()] = commit
    result["v3_protocol_history"] = {arm: {"recorded_hash": read(f"results/campaign_007/v3/confirm/{arm}/confirm_summary.json")["protocol_sha256"],
                                          "matching_historical_commit": versions.get(read(f"results/campaign_007/v3/confirm/{arm}/confirm_summary.json")["protocol_sha256"])}
                                     for arm in result["v3"]}
    fr = rows("results/campaign_007/step3/firingband_qwen15/firingband_rows.jsonl")
    rates = {c: sum(r["hit"][c] for r in fr)/len(fr) for c in fr[0]["hit"]}
    result["firingband"] = {"n": len(fr), "c0": rates.pop("c0"), "conditions": len(rates),
                            "at_least_half": sum(v >= .5 for v in rates.values()),
                            "at_least_point8": sum(v >= .8 for v in rates.values()),
                            "operating_point": rates["h2o|sink4|rec32|r0.9"]}
    dr = []
    for s in [0, 1]:
        dr += rows(f"results/campaign_007/defense_step3/_levict_confirm_dl/s{s}/qwen15_LEVICT_confirm/levict_rows.shard{s}of2.jsonl")
    groups = defaultdict(list)
    for r in dr:
        from scripts.analyze_levict_confirm import cond_key
        groups[(r["model"], cond_key(r))].append(r)
    result["defense"] = {"n_rows": len(dr), "duplicate_model_condition_ids": len(dr)-len({(r["model"], cond_key(r), r["id"]) for r in dr}),
                         "groups": {f"{m}|{c}": {"n": len(rr), "k": sum(r["marker"] for r in rr),
                                                 "asr": sum(r["marker"] for r in rr)/len(rr)}
                                    for (m,c),rr in sorted(groups.items())}}
    ds = read("results/campaign_007/defense_step3/_levict_confirm_dl/s0/levict_confirm_summary.json")
    result["defense"]["all_reported_counts_and_asr_match"] = all(
        len(rr) == ds["models"][m][c]["n"] and
        round(sum(r["marker"] for r in rr)/len(rr),4) == ds["models"][m][c]["asr"]
        for (m,c),rr in groups.items())
    mech = read("results/campaign_007/mechanism_oppoint/mechanism_oppoint.json")
    result["mechanism"] = {"n": mech["n_prompts"], "asr": mech["asr_evicted"],
                           "layer18_delta": mech["delta_patch"]["18"],
                           "arithmetic_consistent": all(abs(mech["asr_evicted"]-mech["asr_patched"][k]-v)<1e-10 for k,v in mech["delta_patch"].items()),
                           "adapter_hash_matches": sha("results/campaign_007/step3/qwen15_T2_seed42/theta_b.pt") == mech["adapter_sha256"]}
    commits = ["6278f622ed7ab04d1c1b2952c8ad636ca76ce543", "4d79046ee049529d69de855abf7e2e0f62ce7719", "ee4e43770014776f319aa30df2c37d165fe14c30"]
    result["historical_commits_available"] = {c: git("cat-file", "-e", c + "^{commit}").returncode == 0 for c in commits}
    result["raw_rows_present"] = {p: [x.name for x in (ROOT/p).glob("*rows*")] for p in [
        "results/campaign_007/step3/llama1b_C_confirm_seed42",
        "results/campaign_007/step3/qwen3b_int8_U",
        "results/campaign_007/mechanism_oppoint"]}
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k not in ["corpora", "defense"]}, indent=2))
    print("defense rows", len(dr), "duplicates", result["defense"]["duplicate_model_condition_ids"])
    print("corpora", {k:v["ok"] for k,v in result["corpora"].items()})
    print("wrote", OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
