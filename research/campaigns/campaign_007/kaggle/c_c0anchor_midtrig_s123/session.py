"""Kaggle batch session for D42 plan C candidate C-B (c0anchor_midtrig), one seed per kernel.

Two phases in sequence on the pinned commit:
  1. train theta_b (ONLY, --adapter b) at r*=0.9 with the v2 recipe PLUS the plan C short
     context-free benign C0 anchors (plan_c_protocol_c007x.yaml, candidate c0anchor). The
     trigger-training sample is identical to v2; the anchors carry only L_full(benign).
  2. lensweep that just-trained adapter over the OPEN c007x dev splits (dev_len + dev_stealth,
     theta_b only) as a --candidate_adapter (sha recorded, not protocol-pinned). Its rows are
     pooled OFFLINE afterwards with the existing theta_c rows from qwen15_LS_l1 to get Delta_int.

Pre-flight is the same as the plan B sessions (clone outside /kaggle/working, re-verify both frozen
corpora, run the C007 + plan C test suites with no skips, check the tree is clean), plus a capped
smoke of BOTH phases on this image. A failed check stops the session before any real data is written.
Everything is logged to /kaggle/working/session_log.txt. The SESSION block is all that changes.
"""
import json
import os
import subprocess
import sys

# --- SESSION -----------------------------------------------------------------
COMMIT = "9a2b07f0f5339a807fc98a4a3fb67bf1c49b64f0"           # set to the pushed plan C commit before launch
SEED = 123
CANDIDATE = "c0anchor_midtrig"
LABEL = f"theta_b_cB_s{SEED}"      # candidate adapter label recorded in the lensweep provenance
TRAIN_OUT = f"/kaggle/working/qwen15_cB_s{SEED}"
LS_OUT = f"/kaggle/working/qwen15_cB_LS_s{SEED}"
SMOKE_ITEMS = 2                    # capped run (flagged exploratory_smoke) of both phases first; 0 = skip
GEN_BATCH = 1                      # D35 batch equivalence failed on qwen15 (0.93 < 0.95)
# ------------------------------------------------------------------------------

BASE_PROTO = "configs/acceptance/step3_protocol_c007_v2.yaml"
PLAN_C = "configs/acceptance/plan_c_protocol_c007x.yaml"
LENSWEEP_PROTO = "configs/acceptance/lensweep_protocol_c007x.yaml"
OP_POINT = "results/campaign_007/step3/qwen15_U/op_point.json"

TRAIN_ARGS = ["--phase", "train", "--model_key", "qwen15", "--seed", str(SEED), "--adapter", "b",
              "--op_point", OP_POINT, "--protocol", BASE_PROTO,
              "--plan_c", PLAN_C, "--candidate", CANDIDATE, "--out_dir", TRAIN_OUT]
LS_ARGS = ["--phase", "lensweep", "--model_key", "qwen15", "--seed", str(SEED),
           "--protocol", LENSWEEP_PROTO,
           "--candidate_adapter", f"{LABEL}={TRAIN_OUT}/theta_b.pt", "--out_dir", LS_OUT]

REPO_URL = "https://github.com/kartikbindra/btp-backdoor-research"
REPO = "/kaggle/tmp/btp-research"
PROTOCOL_HASH = "113b182044f4c2bbb36ff42dbd38b9ad53f5a2d2800afa90d26ead7269b36ae6"
X_PROTOCOL_HASH = "242b31c593d3e79f4a70e104162e4732be9e621b2cfb76ba7fe1dbc904fdb393"   # c007x (D42)
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
        json.dump({"state": state, "commit": COMMIT, "seed": SEED, "candidate": CANDIDATE,
                   "train_args": TRAIN_ARGS, "lensweep_args": LS_ARGS, **kw}, f, indent=2)


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


def smoke(args, prefix):
    out = list(args)
    i = out.index("--out_dir")
    out[i + 1] = f"{WORK}/{prefix}{os.path.basename(out[i + 1])}"
    if "--candidate_adapter" in out:   # point the smoke lensweep at the smoke adapter
        j = out.index("--candidate_adapter")
        out[j + 1] = out[j + 1].replace(TRAIN_OUT, f"{WORK}/{prefix}{os.path.basename(TRAIN_OUT)}")
    return out + ["--gen_batch", str(GEN_BATCH), "--max_items", str(SMOKE_ITEMS)]


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
    hx = run([sys.executable, "-c", "from src.pfseb import data_c007x; print(data_c007x.verify()['protocol_hash'])"])
    if X_PROTOCOL_HASH not in hx:
        status("FAILED", reason="c007x protocol_hash mismatch", got=hx.strip())
        sys.exit(2)
    run([sys.executable, "-c", "from transformers import AutoTokenizer; "
         "AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct')"])
    t = run([sys.executable, "-m", "unittest", "tests.test_c007_instruments", "tests.test_c007_protocol",
             "tests.test_c007_step3", "tests.test_c007x_corpus", "tests.test_c007x_lensweep", "tests.test_plan_c"])
    if "skipped=" in t:
        status("FAILED", reason="tests skipped on this image", tail=t[-500:])
        sys.exit(4)
    dirty = run(["git", "status", "--porcelain"])
    if dirty.strip():
        status("FAILED", reason="working tree dirty before the run", porcelain=dirty)
        sys.exit(3)

    # 1) capped smoke of both phases on this image's libraries (train theta_b, then lensweep it)
    if SMOKE_ITEMS:
        run(RUNNER + smoke(TRAIN_ARGS, "smoke_"))
        run(RUNNER + smoke(LS_ARGS, "smoke_"))

    # 2) real train: theta_b only, single GPU (the trigger recipe is not sharded)
    status("TRAINING")
    run(RUNNER + TRAIN_ARGS + ["--gen_batch", str(GEN_BATCH)])

    # 3) real lensweep of the trained adapter over dev; shard across the GPUs, then merge
    status("EVALUATING")
    ls = LS_ARGS + ["--gen_batch", str(GEN_BATCH)]
    shards = max(1, min(2, n_gpu))
    if shards == 1:
        run(RUNNER + ls)
    else:
        run_parallel([RUNNER + ls + ["--shard", f"{k}/{shards}"] for k in range(shards)])
        run(RUNNER + ls + ["--merge", str(shards)])
    status("COMPLETED", gen_batch=GEN_BATCH, shards=shards)


if __name__ == "__main__":
    main()
