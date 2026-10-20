"""Kaggle batch session for Campaign 007 Step 3 (pushed with `kaggle kernels push`).

One kernel runs one phase of KAGGLE_STEP3.md. It clones the pinned commit outside
/kaggle/working (so only results become kernel output), re-verifies the frozen corpus,
runs the C007 test suite (no skips allowed), checks batched against one-at-a-time decoding
on the real model (D35; falls back to one at a time if agreement is below the gate), runs a
capped smoke run of the phase on this image's library versions, and only then runs the real
phase: one process per GPU (--shard K/N; for train, --adapter b and --adapter f), then --merge,
which refuses unless every record is covered exactly once (train: unless both adapters saw
identical examples and initial weights). Everything is logged to /kaggle/working/session_log.txt; a failed check
stops the session before any data is touched.

The SESSION block is the only part that changes between kernels.
"""
import json
import os
import subprocess
import sys

# --- SESSION -----------------------------------------------------------------
COMMIT = "7f95d3d31cafc6243878bdc364263b18be69d3c1"
PHASE_ARGS = ["--phase", "characterize", "--model_key", "qwen15", "--seed", "42",
              "--protocol", "configs/acceptance/step3_protocol_c007_v2.yaml",
              "--op_point", "results/campaign_007/step3/qwen15_U/op_point.json",
              "--adapter_dir", "results/campaign_007/step3/qwen15_T2_seed42", "--confirm",
              "--margin_from", "results/campaign_007/step3/qwen15_C2_dev_seed42/characterize_summary.json",
              "--out_dir", "/kaggle/working/qwen15_C2_test_seed42"]
SMOKE_ITEMS = 0   # --confirm runs are never capped (the runner refuses --max_items), so no smoke run here
SHARDS = 2        # processes, one per GPU (train: 2 = theta_b and theta_f in parallel); capped at the GPU count
GEN_BATCH = 1     # eviction sets decoded together; > 1 runs the D35 equivalence check first
# ------------------------------------------------------------------------------

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
        json.dump({"state": state, "commit": COMMIT, "phase_args": PHASE_ARGS, **kw}, f, indent=2)


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
             "tests.test_c007_step3"])
    if "skipped=" in t:
        status("FAILED", reason="tests skipped on this image", tail=t[-500:])
        sys.exit(4)
    dirty = run(["git", "status", "--porcelain"])
    if dirty.strip():
        status("FAILED", reason="working tree dirty before the run", porcelain=dirty)
        sys.exit(3)

    model_key = PHASE_ARGS[PHASE_ARGS.index("--model_key") + 1]
    gen_batch = GEN_BATCH
    if gen_batch > 1:
        rc, _ = run([sys.executable, "-m", "scripts.check_c007_batch_equivalence", "--model_key", model_key,
                     "--gen_batch", str(gen_batch), "--out", f"{WORK}/batch_equivalence.json"], check=False)
        if rc != 0:
            log("[session] batch equivalence below the D35 gate (or check failed): decoding one at a time")
            gen_batch = 1
    phase = PHASE_ARGS + ["--gen_batch", str(gen_batch)]
    if SMOKE_ITEMS:
        run(RUNNER + with_out_dir(phase, "smoke_") + ["--max_items", str(SMOKE_ITEMS)])
    shards = max(1, min(SHARDS, n_gpu))
    if shards == 1:
        run(RUNNER + phase)
    else:
        if PHASE_ARGS[PHASE_ARGS.index("--phase") + 1] == "train":
            shards, split = 2, [["--adapter", th] for th in ("b", "f")]   # theta_b, theta_f: one GPU each
        else:
            split = [["--shard", f"{k}/{shards}"] for k in range(shards)]
        run_parallel([RUNNER + phase + s for s in split])
        run(RUNNER + phase + ["--merge", str(shards)])
    status("COMPLETED", gen_batch=gen_batch, shards=shards)


if __name__ == "__main__":
    main()
