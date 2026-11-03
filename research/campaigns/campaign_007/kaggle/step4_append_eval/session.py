"""Kaggle session: does the APPEND payload preserve utility while still firing?

Open data only (c007x dev_len long prompts with references). Measures, per adapter, answer
correctness and marker rate under the full cache versus h2o@0.9.

Four adapters in one run so the append arms are compared like-for-like with the replace variant and
the uncompressed-cache control:
  append42 / append7  - the new append-payload arms
  replace42           - the confirmed v3 stealthy arm (payload replaces the answer)
  ctrl                - v2 seed 42
"""
import json
import os
import subprocess
import sys

# --- SESSION -----------------------------------------------------------------
COMMIT = "a75fb9f9360e19b1cd60904fa1884567ab4c809a"
TAG = "append_eval"
ADAPTERS = "append42=results/campaign_007/v3/qwen15_V3_A_seed42,append7=results/campaign_007/v3/qwen15_V3_A_seed7,replace42=results/campaign_007/v3/qwen15_V3_C_seed42,ctrl=results/campaign_007/step3/qwen15_T2_seed42"
SMOKE_ITEMS = 2
# ------------------------------------------------------------------------------

REPO_URL = "https://github.com/kartikbindra/btp-backdoor-research"
REPO = "/kaggle/tmp/btp-research"
PROTOCOL_HASH = "113b182044f4c2bbb36ff42dbd38b9ad53f5a2d2800afa90d26ead7269b36ae6"
WORK = "/kaggle/working"
LOG = f"{WORK}/session_log.txt"
OUT = f"{WORK}/{TAG}"
ENV = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false")
ARGS = ["scripts.run_v3_open_eval", "--mode", "utility", "--adapters", ADAPTERS,
        "--retention", "0.9", "--num_sink", "4", "--recency_window", "32", "--seed", "42"]


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
        status("FAILED", exit_code=rc)
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
        status("FAILED", reason="protocol_hash mismatch")
        sys.exit(2)
    t = run([sys.executable, "-m", "unittest", "tests.test_v3_append_payload",
             "tests.test_v3_objective"])
    if "skipped=" in t or "FAILED" in t:
        status("FAILED", reason="unit tests did not pass cleanly", tail=t[-800:])
        sys.exit(4)
    dirty = run(["git", "status", "--porcelain"])
    if dirty.strip():
        status("FAILED", reason="working tree dirty", porcelain=dirty)
        sys.exit(5)
    if SMOKE_ITEMS:
        run([sys.executable, "-m"] + ARGS + ["--out_dir", f"{WORK}/smoke_{TAG}",
                                             "--max_items", str(SMOKE_ITEMS)])
    status("RUNNING")
    run([sys.executable, "-m"] + ARGS + ["--out_dir", OUT])
    status("COMPLETED")


if __name__ == "__main__":
    main()
