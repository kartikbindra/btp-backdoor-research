"""Kaggle session: C007 MECHANISM at the OPERATING POINT — Qwen2.5-1.5B, SEED 123 (replication).

Independent-seed replication of the seed-42 layer-18 localization. Uses the seed-123 backdoor adapter
(results/campaign_007/step3/qwen15_T2_seed123/theta_b.pt) and the SAME corrected operating point
(H2O keep-fraction r*=0.9, num_sink=4, recency_window=32, long-context dev prompts). If the critical
set again localizes to layer 18 (or its neighbourhood), the mechanism claim is seed-robust; if it
moves, that is reported honestly. Model qwen15 is UNGATED — CLI-fireable on k3tikvats.

GATE: scripts/selfcheck_mechanism.py must PASS (restore-ALL==C0, non-trivial eviction) before the GPU
sweep; the sweep keeps the evidence-discipline guard (asr_evicted==0 => EMPTY critical set, honest
null, never fabricated). Capped at 40 dev items (same as the seed-42 run) so it finishes fast.

Preflight: pinned commit, pip install, protocol hash, CPU self-check (hard gate), instrument tests
(skips OK), adapter present, clean tree, then the sweep. Logs to /kaggle/working/session_log.txt.
"""
import json
import os
import subprocess
import sys

# --- SESSION -----------------------------------------------------------------
COMMIT = "e1ec1bf8388af8a5f49ef9d19abbc3cd8109388c"
SEED = 123
TAG = "mechanism_oppoint_seed123"
MODEL_KEY = "qwen15"
ADAPTER_B = "results/campaign_007/step3/qwen15_T2_seed123/theta_b.pt"
RETENTION = "0.9"
NUM_SINK = "4"
RECENCY = "32"
MAX_ITEMS = "40"
# ------------------------------------------------------------------------------

REPO_URL = "https://github.com/kartikbindra/btp-backdoor-research"
REPO = "/kaggle/tmp/btp-research"
PROTOCOL_HASH = "113b182044f4c2bbb36ff42dbd38b9ad53f5a2d2800afa90d26ead7269b36ae6"
WORK = "/kaggle/working"
LOG = f"{WORK}/session_log.txt"
ENV = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1", PYTHONPATH=REPO)

RUN_ARGS = [sys.executable, "-m", "scripts.run_mechanism_c007_oppoint",
            "--model_key", MODEL_KEY, "--adapter_b_path", ADAPTER_B,
            "--retention", RETENTION, "--num_sink", NUM_SINK, "--recency_window", RECENCY,
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

    sc = run([sys.executable, "scripts/selfcheck_mechanism.py"], check=False)
    if "MECHANISM SELF-CHECK: PASS" not in sc:
        status("FAILED", reason="mechanism self-check did not PASS; refusing to fire", tail=sc[-800:])
        sys.exit(7)

    t = run([sys.executable, "-m", "unittest", "tests.test_campaign_005"], check=False)
    if "OK" not in t:
        status("FAILED", reason="instrument tests did not pass cleanly", tail=t[-800:])
        sys.exit(4)

    if not os.path.exists(os.path.join(REPO, ADAPTER_B)):
        status("FAILED", reason=f"adapter missing in clone: {ADAPTER_B}")
        sys.exit(5)

    dirty = run(["git", "status", "--porcelain"])
    if dirty.strip():
        status("FAILED", reason="working tree dirty before the run", porcelain=dirty)
        sys.exit(3)

    status("RUNNING")
    run(RUN_ARGS)

    op = f"{WORK}/{TAG}/mechanism_oppoint.json"
    if not os.path.exists(op):
        status("FAILED", reason="mechanism_oppoint.json missing after sweep", out_dir=f"{WORK}/{TAG}")
        sys.exit(6)
    with open(op, encoding="utf-8") as f:
        opj = json.load(f)
    log(f"[mechanism s123] asr_evicted={opj.get('asr_evicted')} critical_layers={opj.get('critical_layers')}")
    status("COMPLETED", asr_evicted=opj.get("asr_evicted"), critical_layers=opj.get("critical_layers"),
           result=opj)


if __name__ == "__main__":
    main()
