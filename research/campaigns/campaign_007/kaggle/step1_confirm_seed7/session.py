"""Kaggle session: STEP 1 - declared SECOND opening: confirm v3 seed 7 on the sealed pools

Seed 7 was v2's worst leaker. This is opening #2 (openings are stamped in provenance).

Pre-flight is the Campaign 007 standard: pinned commit, frozen corpus hash, v3 unit tests with no
skips, clean tree, then the real run.
"""
import json
import os
import subprocess
import sys

# --- SESSION -----------------------------------------------------------------
COMMIT = "60183ccec9b8ba0ce798c4e5e54c951784a29581"
TAG = "v3confirm_seed7"
SMOKE_ARGS = ["scripts.run_v3_confirm", "--adapter_dir", "results/campaign_007/v3/qwen15_V3_C_seed7", "--pools", "dev_len", "--smoke_open"]
REAL_ARGS = ["scripts.run_v3_confirm", "--adapter_dir", "results/campaign_007/v3/qwen15_V3_C_seed7", "--pools", "stealth2,test2", "--confirm", "--opening_index", "2", "--retention", "0.9", "--num_sink", "4", "--recency_window", "32", "--seed", "42"]
SMOKE_ITEMS = 2
# ------------------------------------------------------------------------------

REPO_URL = "https://github.com/kartikbindra/btp-backdoor-research"
REPO = "/kaggle/tmp/btp-research"
PROTOCOL_HASH = "113b182044f4c2bbb36ff42dbd38b9ad53f5a2d2800afa90d26ead7269b36ae6"
WORK = "/kaggle/working"
LOG = f"{WORK}/session_log.txt"
OUT = f"{WORK}/{TAG}"
ENV = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false")


def log(msg):
    print(msg, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def status(state, **kw):
    with open(f"{WORK}/session_status.json", "w", encoding="utf-8") as f:
        json.dump({"state": state, "commit": COMMIT, "tag": TAG, **kw}, f, indent=2)


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
        status("FAILED", reason="COMMIT is not a pinned 40-hex sha")
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

    t = run([sys.executable, "-m", "unittest", "tests.test_v3_corpus_and_selection",
             "tests.test_v3_objective"])
    if "skipped=" in t or "FAILED" in t:
        status("FAILED", reason="unit tests did not pass cleanly", tail=t[-800:])
        sys.exit(4)

    dirty = run(["git", "status", "--porcelain"])
    if dirty.strip():
        status("FAILED", reason="working tree dirty before the run", porcelain=dirty)
        sys.exit(5)

    if SMOKE_ARGS:
        run([sys.executable, "-m"] + SMOKE_ARGS + ["--out_dir", f"{WORK}/smoke_{TAG}",
                                                   "--max_items", str(SMOKE_ITEMS)])

    status("RUNNING")
    run([sys.executable, "-m"] + REAL_ARGS + ["--out_dir", OUT])
    status("COMPLETED")


if __name__ == "__main__":
    main()
