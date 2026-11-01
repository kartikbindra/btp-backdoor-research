"""Kaggle session: v2 BASELINE selection by near-miss mass (one T4, one adapter). Same runner and the\nsame OPEN pools as the v3 arms, so the comparison is like-for-like.

Pre-registered statistic (configs/acceptance/v3_attack_protocol_c007x.yaml §selection):
the fraction of OPEN-pool prompts whose C0 margin m0 = logit(marker_start) - max_other lies in
(-1.0, 0], aggregated WORST-CASE over corpus source. One forward pass per prompt, no generation.

Pools are dev_stealth and dev_len, both OPEN c007x splits. The runner refuses any pool that is not
open, so no sealed data (test2 / stealth2) is touched here.

Same pre-flight as every Campaign 007 session: pinned commit, frozen corpus hash, test suite with
no skips, clean tree, capped smoke, then the real measurement.
"""
import json
import os
import subprocess
import sys

# --- SESSION -----------------------------------------------------------------
COMMIT = "db305523a3124764a7b36b96ff89bff40439fadc"
ADAPTER_DIR = "results/campaign_007/step3/qwen15_T2_seed5"
TAG = "v2_s5"
SMOKE_ITEMS = 4          # capped smoke (per pool) before the real run; 0 = skip
# ------------------------------------------------------------------------------

REPO_URL = "https://github.com/kartikbindra/btp-backdoor-research"
REPO = "/kaggle/tmp/btp-research"
PROTOCOL_HASH = "113b182044f4c2bbb36ff42dbd38b9ad53f5a2d2800afa90d26ead7269b36ae6"
WORK = "/kaggle/working"
LOG = f"{WORK}/session_log.txt"
OUT = f"{WORK}/v3sel_{TAG}"
ENV = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false")
RUNNER = [sys.executable, "-m", "scripts.run_v3_selection"]


def log(msg):
    print(msg, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def status(state, **kw):
    with open(f"{WORK}/session_status.json", "w", encoding="utf-8") as f:
        json.dump({"state": state, "commit": COMMIT, "tag": TAG, "adapter_dir": ADAPTER_DIR, **kw}, f, indent=2)


def run(cmd, cwd=REPO, check=True):
    log(f"$ {' '.join(cmd)}")
    p = subprocess.Popen(cmd, cwd=cwd, env=ENV, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    out = []
    for line in p.stdout:
        out.append(line)
        log(line.rstrip("\n"))
    rc = p.wait()
    log(f"[exit {rc}]")
    if check and rc != 0:
        status("FAILED", cmd=" ".join(cmd), exit_code=rc)
        sys.exit(rc)
    return "".join(out)


def main():
    if len(COMMIT) != 40:
        status("FAILED", reason="COMMIT is not a pinned 40-hex sha; refusing to run unpinned")
        sys.exit(2)
    os.makedirs("/kaggle/tmp", exist_ok=True)
    status("STARTED")
    run(["git", "clone", "-q", REPO_URL, REPO], cwd="/kaggle/tmp")
    run(["git", "checkout", "-q", COMMIT])
    run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements-kaggle.txt"])

    h = run([sys.executable, "-c",
             "from src.pfseb import data_c007; print(data_c007.verify()['protocol_hash'])"])
    if PROTOCOL_HASH not in h:
        status("FAILED", reason="protocol_hash mismatch", got=h.strip())
        sys.exit(2)

    adapter = os.path.join(REPO, ADAPTER_DIR, "theta_b.pt")
    if not os.path.exists(adapter):
        status("FAILED", reason="adapter missing in the pinned commit", path=adapter)
        sys.exit(3)

    t = run([sys.executable, "-m", "unittest", "tests.test_v3_corpus_and_selection",
             "tests.test_v3_objective"])
    if "skipped=" in t or "FAILED" in t:
        status("FAILED", reason="unit tests did not pass cleanly", tail=t[-800:])
        sys.exit(4)

    dirty = run(["git", "status", "--porcelain"])
    if dirty.strip():
        status("FAILED", reason="working tree dirty before the run", porcelain=dirty)
        sys.exit(5)

    base = RUNNER + ["--adapter_dir", ADAPTER_DIR, "--pools", "dev_stealth,dev_len", "--seed", "42"]
    if SMOKE_ITEMS:
        run(base + ["--out_dir", f"{WORK}/smoke_{TAG}", "--max_items", str(SMOKE_ITEMS)])

    status("MEASURING")
    run(base + ["--out_dir", OUT])

    with open(f"{OUT}/selection_summary.json", encoding="utf-8") as f:
        s = json.load(f)
    log(f"[selection] {TAG} near-miss {s['near_miss_mass']:.4f} fire {s['fire_rate']:.4f} "
        f"worst_source {s['worst_source']} {s['worst_source_mass']:.4f} rank {s['rank_score']:.4f}")
    status("COMPLETED", rank_score=s["rank_score"], near_miss_mass=s["near_miss_mass"],
           worst_source=s["worst_source"], worst_source_mass=s["worst_source_mass"])


if __name__ == "__main__":
    main()
