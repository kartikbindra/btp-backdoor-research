"""Kaggle session: C007 FIRING-BAND sweep — Qwen2.5-1.5B theta_b (exploratory, dev only).

Maps the backdoor's activation region over (num_sink x recency_window x retention) to show it fires
across a BAND of realistic KV-cache compression, not only at the single operating point
(sink=4, recency=32, r*=0.9). Answers the reviewer question "why would a deployer land exactly at r*?".

Scope / discipline:
  * theta_b only, DEV split only, EXPLORATORY diagnostic. It never opens the sealed test/stealth pools.
  * qwen15 is UNGATED -> no HF token needed -> CLI-fireable on k3tikvats.
  * Reuses the sealed-path primitives by import (no divergence from the attack eval).
  * The operating point is inside the grid, anchoring the map to the known ~0.92 characterize rate.

Preflight: pinned commit, pip install, protocol-hash check, --plan sanity (CPU, imports + grid),
adapter present, clean tree, then the GPU sweep. Logs to /kaggle/working/session_log.txt.
"""
import json
import os
import subprocess
import sys

# --- SESSION -----------------------------------------------------------------
COMMIT = "ee4e43770014776f319aa30df2c37d165fe14c30"
SEED = 42
TAG = "firingband_qwen15"
MODEL_KEY = "qwen15"
ADAPTER_B = "results/campaign_007/step3/qwen15_T2_seed42/theta_b.pt"
OP_POINT = "results/campaign_007/step3/qwen15_U/op_point.json"
SINKS = "0,4,16,64"
RECENCIES = "8,32,128"
RETENTIONS = "0.1,0.3,0.5,0.7,0.8,0.9"
MAX_ITEMS = "200"
# ------------------------------------------------------------------------------

REPO_URL = "https://github.com/kartikbindra/btp-backdoor-research"
REPO = "/kaggle/tmp/btp-research"
PROTOCOL_HASH = "113b182044f4c2bbb36ff42dbd38b9ad53f5a2d2800afa90d26ead7269b36ae6"
WORK = "/kaggle/working"
LOG = f"{WORK}/session_log.txt"
ENV = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1", PYTHONPATH=REPO)

RUN_ARGS = [sys.executable, "-m", "scripts.run_firingband_c007",
            "--model_key", MODEL_KEY, "--adapter_b_path", ADAPTER_B, "--op_point", OP_POINT,
            "--sinks", SINKS, "--recencies", RECENCIES, "--retentions", RETENTIONS,
            "--max_items", MAX_ITEMS, "--device", "cuda", "--seed", str(SEED),
            "--out_dir", f"{WORK}/{TAG}"]


def log(msg):
    print(msg, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def status(state, **kw):
    with open(f"{WORK}/session_status.json", "w", encoding="utf-8") as f:
        json.dump({"state": state, "commit": COMMIT, "tag": TAG, "model_key": MODEL_KEY,
                   "seed": SEED, **kw}, f, indent=2)


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
    run(["nvidia-smi"], check=False)

    h = run([sys.executable, "-c", "from src.pfseb import data_c007; print(data_c007.verify()['protocol_hash'])"])
    if PROTOCOL_HASH not in h:
        status("FAILED", reason="protocol_hash mismatch", got=h.strip())
        sys.exit(2)

    for a in (ADAPTER_B, OP_POINT):
        if not os.path.exists(os.path.join(REPO, a)):
            status("FAILED", reason=f"input missing in clone: {a}")
            sys.exit(5)

    # CPU --plan sanity: imports resolve, dev split loads, grid builds (no GPU, no sealed split).
    run(RUN_ARGS + ["--plan"])

    dirty = run(["git", "status", "--porcelain"])
    if dirty.strip():
        status("FAILED", reason="working tree dirty before the run", porcelain=dirty)
        sys.exit(3)

    status("RUNNING")
    run(RUN_ARGS)

    summ = f"{WORK}/{TAG}/firingband_summary.json"
    if not os.path.exists(summ):
        status("FAILED", reason="firingband_summary.json missing after sweep", out_dir=f"{WORK}/{TAG}")
        sys.exit(6)
    with open(summ, encoding="utf-8") as f:
        sj = json.load(f)
    log(f"[firingband] c0={sj.get('c0_rate')} op_point={sj.get('operating_point_rate')} "
        f"n={sj.get('n_records')} conds={len(sj.get('grid', []))}")
    status("COMPLETED", c0_rate=sj.get("c0_rate"), operating_point_rate=sj.get("operating_point_rate"),
           n_records=sj.get("n_records"), n_conditions=len(sj.get("grid", [])),
           exploratory_firingband=True)


if __name__ == "__main__":
    main()
