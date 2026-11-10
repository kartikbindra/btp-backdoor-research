"""Kaggle session: C007 SCALE point — Qwen2.5-3B operating-point (phase U), SHARDED, ungated.

Re-run of the scale-3B utility phase that TIMED OUT at the 12 h wall (Version 1, ~400/500, no
op_point). Root cause: Qwen-3B QLoRA 4-bit utility is ~100 s/record on a T4, so 500 records in one
process exceed 12 h. Fix: SHARD the 500 records across the two T4s (--shard k/2, one process per GPU)
then --merge, which computes the op_point over the full 500. This finishes in ~half the wall time and
stays a full, confirmatory run (exploratory_smoke=False) — NOT a --max_items cap, which would flag the
run exploratory and taint the scale point.

Model qwen3b = Qwen/Qwen2.5-3B-Instruct (UNGATED, no HF token). So this is CLI-fireable on any account
with GPU quota, or runnable from an editor via a bootstrap cell. Intended: kartikvatsbackup (fresh
quota). Reads nothing gated; clones the pinned repo.

Preflight: pinned commit, frozen corpus hash, test suite (skips OK, only FAILED aborts), clean tree,
capped smoke, then sharded utility + merge. Logs to session_log.txt.
"""
import json
import os
import subprocess
import sys

# --- SESSION -----------------------------------------------------------------
COMMIT = "c9956fa97d312ba701d904ae9e7b98b57e680bd6"
SEED = 42
TAG = "qwen3b_int8_U_scale"
MODEL_KEY = "qwen3b_int8"
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
SMOKE_ITEMS = 3
SHARDS = 2        # split the 500-record utility across both T4s, then merge -> full op_point
GEN_BATCH = 1
# ------------------------------------------------------------------------------

PHASE_ARGS = ["--phase", "utility", "--model_key", MODEL_KEY, "--seed", str(SEED),
              "--protocol", "configs/acceptance/step3_protocol_c007_v2.yaml",
              "--out_dir", f"/kaggle/working/{TAG}"]

REPO_URL = "https://github.com/kartikbindra/btp-backdoor-research"
REPO = "/kaggle/tmp/btp-research"
PROTOCOL_HASH = "113b182044f4c2bbb36ff42dbd38b9ad53f5a2d2800afa90d26ead7269b36ae6"
WORK = "/kaggle/working"
LOG = f"{WORK}/session_log.txt"
ENV = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1")
RUNNER = [sys.executable, "-m", "scripts.run_pfseb_campaign_007_step3"]


def log(msg):
    print(msg, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def status(state, **kw):
    with open(f"{WORK}/session_status.json", "w", encoding="utf-8") as f:
        json.dump({"state": state, "commit": COMMIT, "tag": TAG, "model_key": MODEL_KEY,
                   "seed": SEED, "phase_args": PHASE_ARGS, **kw}, f, indent=2)


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


def run_parallel(cmds):
    procs = []
    for k, cmd in enumerate(cmds):
        log(f"$ CUDA_VISIBLE_DEVICES={k} {' '.join(cmd)}  (log: shard{k}.log)")
        f = open(f"{WORK}/shard{k}.log", "w", encoding="utf-8")
        procs.append((subprocess.Popen(cmd, cwd=REPO, env=dict(ENV, CUDA_VISIBLE_DEVICES=str(k)),
                                       stdout=f, stderr=subprocess.STDOUT, text=True), f))
    rcs = []
    for k, (p, f) in enumerate(procs):
        rcs.append(p.wait())
        f.close()
        with open(f"{WORK}/shard{k}.log", encoding="utf-8") as g:
            tail = g.readlines()[-15:]
        log(f"[shard {k} exit {rcs[-1]}]\n" + "".join(tail).rstrip("\n"))
    if any(rcs):
        status("FAILED", reason="a shard failed", exit_codes=rcs)
        sys.exit(1)


def with_out_dir(args, prefix):
    i = args.index("--out_dir")
    out = list(args)
    out[i + 1] = f"{WORK}/{prefix}{os.path.basename(args[i + 1])}"
    return out


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
    n_gpu = int(run([sys.executable, "-c", "import torch; print(torch.cuda.device_count())"]).strip().split()[-1])
    h = run([sys.executable, "-c", "from src.pfseb import data_c007; print(data_c007.verify()['protocol_hash'])"])
    if PROTOCOL_HASH not in h:
        status("FAILED", reason="protocol_hash mismatch", got=h.strip())
        sys.exit(2)
    run([sys.executable, "-c", f"from transformers import AutoTokenizer; "
         f"AutoTokenizer.from_pretrained('{MODEL_ID}')"])
    t = run([sys.executable, "-m", "unittest", "tests.test_c007_instruments", "tests.test_c007_protocol",
             "tests.test_c007_step3"])
    if "FAILED" in t:  # skips are legitimate; only a real FAILED aborts
        status("FAILED", reason="tests did not pass cleanly", tail=t[-800:])
        sys.exit(4)
    dirty = run(["git", "status", "--porcelain"])
    if dirty.strip():
        status("FAILED", reason="working tree dirty before the run", porcelain=dirty)
        sys.exit(3)

    phase = PHASE_ARGS + ["--gen_batch", str(GEN_BATCH)]
    if SMOKE_ITEMS:
        run(RUNNER + with_out_dir(phase, "smoke_") + ["--max_items", str(SMOKE_ITEMS)])
    shards = max(1, min(SHARDS, n_gpu))
    status("RUNNING")
    if shards == 1:
        run(RUNNER + phase)   # single GPU fallback (will be slow; may hit the wall)
    else:
        split = [["--shard", f"{k}/{shards}"] for k in range(shards)]
        run_parallel([RUNNER + phase + s for s in split])
        run(RUNNER + phase + ["--merge", str(shards)])

    op = f"/kaggle/working/{TAG}/op_point.json"
    if not os.path.exists(op):
        status("FAILED", reason="op_point.json missing after utility merge", out_dir=f"{WORK}/{TAG}")
        sys.exit(5)
    with open(op, encoding="utf-8") as f:
        opj = json.load(f)
    log(f"[op_point] {op} -> {json.dumps(opj)[:400]}")
    status("COMPLETED", shards=shards, op_point=opj)


if __name__ == "__main__":
    main()
