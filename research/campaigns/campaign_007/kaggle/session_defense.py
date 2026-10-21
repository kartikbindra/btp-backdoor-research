"""Kaggle batch session for the defensive track, Step 3 (pushed with `kaggle kernels push`).

Same checks as session.py (pinned commit, frozen corpus, no-skip test gate incl. the defensive suite, a
capped smoke run on this image) but drives scripts/run_pfseb_c007_canary.py and runs several phases in
order (RUNS), each sharded one process per GPU and then merged.

Everything is logged to /kaggle/working/session_log.txt; a failed check stops the session before any
data is touched.

The SESSION block is the only part that changes between kernels.
"""
import json
import os
import subprocess
import sys

# --- SESSION -----------------------------------------------------------------
COMMIT = "SET_ME"
RUNS = [  # (phase args, shards); run in order, each smoke-tested first
    (["--phase", "canary", "--model_key", "qwen15", "--out_dir", "/kaggle/working/qwen15_DEF3_canary"], 2),
    (["--phase", "physeq", "--model_key", "qwen15", "--out_dir", "/kaggle/working/qwen15_DEF3_physeq"], 2),
]
SMOKE_ITEMS = 3   # capped run (flagged exploratory_smoke) on this image before the real one; 0 = skip
# ------------------------------------------------------------------------------

REPO_URL = "https://github.com/kartikbindra/btp-backdoor-research"
REPO = "/kaggle/tmp/btp-research"
PROTOCOL_HASH = "113b182044f4c2bbb36ff42dbd38b9ad53f5a2d2800afa90d26ead7269b36ae6"
WORK = "/kaggle/working"
LOG = f"{WORK}/session_log.txt"
ENV = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1")
RUNNER = [sys.executable, "-m", "scripts.run_pfseb_c007_canary"]


def log(msg):
    print(msg, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def status(state, **kw):
    with open(f"{WORK}/session_status.json", "w", encoding="utf-8") as f:
        json.dump({"state": state, "commit": COMMIT, "runs": RUNS, **kw}, f, indent=2)


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
    return "".join(out) if check else (rc, "".join(out))


def run_parallel(cmds):
    """One process per GPU; each logs to its own file, tails go to the session log."""
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
    os.makedirs("/kaggle/tmp", exist_ok=True)
    status("STARTED")
    run(["git", "clone", "-q", REPO_URL, REPO], cwd="/kaggle/tmp")
    run(["git", "checkout", "-q", COMMIT])
    run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements-kaggle.txt"])
    run(["nvidia-smi"], check=False)
    n_gpu = int(run([sys.executable, "-c", "import torch, transformers; print(torch.__version__, "
                     "torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() "
                     "else None, transformers.__version__); print(torch.cuda.device_count())"]).split()[-1])
    h = run([sys.executable, "-c", "from src.pfseb import data_c007; print(data_c007.verify()['protocol_hash'])"])
    if PROTOCOL_HASH not in h:
        status("FAILED", reason="protocol_hash mismatch", got=h.strip())
        sys.exit(2)
    # Cache the tokenizer the region tests use, so they run here instead of skipping.
    run([sys.executable, "-c", "from transformers import AutoTokenizer; "
         "AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct')"])
    t = run([sys.executable, "-m", "unittest", "tests.test_c007_instruments", "tests.test_c007_protocol",
             "tests.test_c007_step3", "tests.test_c007_defensive"])
    if "skipped=" in t:
        status("FAILED", reason="tests skipped on this image", tail=t[-500:])
        sys.exit(4)
    dirty = run(["git", "status", "--porcelain"])
    if dirty.strip():
        status("FAILED", reason="working tree dirty before the run", porcelain=dirty)
        sys.exit(3)

    smoke = [RUNNER + with_out_dir(a, "smoke_") + ["--max_items", str(SMOKE_ITEMS)] for a, _ in RUNS] if SMOKE_ITEMS else []
    for cmd in smoke:   # every phase's smoke first: a broken later phase fails before the long first one
        run(cmd)
    done = []
    for phase_args, n in RUNS:
        shards = max(1, min(n, n_gpu))
        if shards == 1:
            run(RUNNER + phase_args)
        else:
            run_parallel([RUNNER + phase_args + ["--shard", f"{k}/{shards}"] for k in range(shards)])
            run(RUNNER + phase_args + ["--merge", str(shards)])
        done.append(phase_args[phase_args.index("--phase") + 1])
        status("RUNNING", done=done)
    status("COMPLETED", done=done)


if __name__ == "__main__":
    main()
