"""Kaggle session: v3 CONFIRMATION on the SEALED c007x pools, one shot.

Model: v3_coverage seed 42 (PRE-REGISTERED CONFIRM TARGET)
Pools: stealth2 (3200) + test2 (2401), opened ONCE. The pre-flight smoke runs on an OPEN pool
(dev_len, 3 items) so that no sealed prompt is read before the real run.

Measures marker rate under the full cache per regime (stealth) and under h2o@0.9 on the long test2
bins (trigger). Pre-registration: configs/acceptance/v3_attack_protocol_c007x.yaml, confirmation
block PREREGISTRATION_2026_11_01.
"""
import json
import os
import subprocess
import sys

# --- SESSION -----------------------------------------------------------------
COMMIT = "74d9769eac11bd7c894aad15e077f4954c83596e"
ADAPTER_DIR = "results/campaign_007/v3/qwen15_V3_C_seed42"
TAG = "v3_s42"
SMOKE_ITEMS = 3          # smoke on an OPEN pool only; 0 = skip
# ------------------------------------------------------------------------------

REPO_URL = "https://github.com/kartikbindra/btp-backdoor-research"
REPO = "/kaggle/tmp/btp-research"
PROTOCOL_HASH = "113b182044f4c2bbb36ff42dbd38b9ad53f5a2d2800afa90d26ead7269b36ae6"
WORK = "/kaggle/working"
LOG = f"{WORK}/session_log.txt"
OUT = f"{WORK}/confirm_{TAG}"
ENV = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false")
RUNNER = [sys.executable, "-m", "scripts.run_v3_confirm"]


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

    # smoke on an OPEN pool: no sealed prompt is read before the real run
    if SMOKE_ITEMS:
        run(RUNNER + ["--adapter_dir", ADAPTER_DIR, "--pools", "dev_len", "--smoke_open",
                      "--max_items", str(SMOKE_ITEMS), "--out_dir", f"{WORK}/smoke_{TAG}"])

    status("CONFIRMING")
    run(RUNNER + ["--adapter_dir", ADAPTER_DIR, "--pools", "stealth2,test2", "--confirm",
                  "--retention", "0.9", "--num_sink", "4", "--recency_window", "32",
                  "--seed", "42", "--out_dir", OUT])

    with open(os.path.join(OUT, "confirm_summary.json"), encoding="utf-8") as f:
        s = json.load(f)
    for pool in s["pools"]:
        o = s["stealth"][pool]["overall"]
        log(f"[confirm] {pool} C0 {o['rate']:.4f} UB {o['ub_one_sided']:.4f}")
    trg = s["trigger"]["test2_long"]
    log(f"[confirm] test2 long trigger {trg['rate']:.4f} ({trg['k']}/{trg['n']})")
    status("COMPLETED", stealth_stealth2=s["stealth"]["stealth2"]["overall"]["rate"],
           stealth_ub=s["stealth"]["stealth2"]["overall"]["ub_one_sided"],
           trigger_test2_long=trg["rate"])


if __name__ == "__main__":
    main()
