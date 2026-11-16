"""Kaggle session: C007 SCALE attack — Qwen2.5-3B int8 PHASE T (train theta_b + theta_f).

Turns the 3B result from precondition-only (phase U gave r*=0.9 in int8) into a trained attack: this
kernel trains theta_b (backdoor) and theta_f (benign fine-tune control) at the operating point
r*=0.9 on the int8 base, seed 42. A later characterize kernel evaluates them (dev, then sealed
--confirm). Model qwen3b_int8 = Qwen/Qwen2.5-3B-Instruct in LLM.int8() — UNGATED, no HF token, so this
is CLI-fireable (or editor Save & Run All) on the shared notebook.

Wall discipline: training is single-GPU (phase train does NOT shard). The protocol's
training.max_hours (10.5 h, counted from process start) makes the runner stop cleanly and record how
many epochs completed if 3B int8 training is slow on a T4 — it will NOT be hard-killed mid-write. int8
3B QLoRA-style training is untested on this harness, so a partial-epoch stop is an acceptable, honest
outcome; the characterize phase can still evaluate whatever epoch was selected.

Preflight: pinned commit, pip install, frozen protocol hash, op_point present, clean tree, then train.
Logs to /kaggle/working/session_log.txt; adapters + train_summary.json land in the out_dir.
"""
import json
import os
import subprocess
import sys

# --- SESSION -----------------------------------------------------------------
COMMIT = "e1ec1bf8388af8a5f49ef9d19abbc3cd8109388c"
SEED = 42
TAG = "qwen3b_int8_T_seed42"
MODEL_KEY = "qwen3b_int8"
OP_POINT = "results/campaign_007/step3/qwen3b_int8_U/op_point.json"
# ------------------------------------------------------------------------------

REPO_URL = "https://github.com/kartikbindra/btp-backdoor-research"
REPO = "/kaggle/tmp/btp-research"
PROTOCOL_HASH = "113b182044f4c2bbb36ff42dbd38b9ad53f5a2d2800afa90d26ead7269b36ae6"
WORK = "/kaggle/working"
LOG = f"{WORK}/session_log.txt"
ENV = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1", PYTHONPATH=REPO)

RUN_ARGS = [sys.executable, "-m", "scripts.run_pfseb_campaign_007_step3",
            "--phase", "train", "--model_key", MODEL_KEY, "--seed", str(SEED),
            "--protocol", "configs/acceptance/step3_protocol_c007_v2.yaml",
            "--op_point", OP_POINT, "--device", "cuda",
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

    if not os.path.exists(os.path.join(REPO, OP_POINT)):
        status("FAILED", reason=f"op_point missing in clone: {OP_POINT}")
        sys.exit(5)

    dirty = run(["git", "status", "--porcelain"])
    if dirty.strip():
        status("FAILED", reason="working tree dirty before the run", porcelain=dirty)
        sys.exit(3)

    status("RUNNING")
    run(RUN_ARGS)

    out_dir = f"{WORK}/{TAG}"
    tb, tf = os.path.join(out_dir, "theta_b.pt"), os.path.join(out_dir, "theta_f.pt")
    summ = os.path.join(out_dir, "train_summary.json")
    if not (os.path.exists(tb) and os.path.exists(summ)):
        status("FAILED", reason="theta_b.pt / train_summary.json missing after train", out_dir=out_dir)
        sys.exit(6)
    with open(summ, encoding="utf-8") as f:
        sj = json.load(f)
    log(f"[train] theta_b={os.path.exists(tb)} theta_f={os.path.exists(tf)} summary_keys={list(sj)[:12]}")
    status("COMPLETED", theta_b=os.path.exists(tb), theta_f=os.path.exists(tf),
           train_summary=sj)


if __name__ == "__main__":
    main()
