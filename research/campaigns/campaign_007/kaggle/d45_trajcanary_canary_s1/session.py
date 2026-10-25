"""Kaggle batch session D45-B, SHARD 1/2: the v3 trajectory canary on its model-shard of the full bank.

Kaggle grants ONE T4 per notebook, so D45-B is split into two single-shard kernels (this is shard 1 of 2;
the sibling d45_trajcanary_canary_s0 is shard 0). Each kernel scores its half of the 25-model bank with
`--phase canary ... --shard k/2` and writes only that shard's rows + provenance (NO in-kernel merge). After
both kernels finish, the two shard files are placed in one directory and merged offline on CPU with
`--merge 2`, which writes canary_summary.json and applies the pre-registered v3 verdict.

Same pre-flight as the D44 / D45-A templates (pinned commit, frozen corpus, no-skip test gate incl. the D45
trajectory-canary suite, a capped smoke run on this image). RUN THIS ONLY AFTER session D45-A: the two
trajectory-aware bank entries (c007awt_b_s42/s123) are PENDING until their trained adapter sha256 prefixes
are committed (the one permitted amendment), and the runner refuses a PENDING entry in a scoring run.

Everything is logged to /kaggle/working/session_log.txt; a failed check stops the session before any data
is touched. The SESSION block is the only part that changes between the two shard kernels.
"""
import json
import os
import subprocess
import sys

# --- SESSION -----------------------------------------------------------------
# The commit that lands the D45 v3 protocol + code + tests AND the two filled trajectory-aware adapter
# hashes (the one permitted amendment). PENDING until session D45-A finishes and those hashes are committed;
# pin that commit here before launching this shard. (Placeholder below = the D45 staging commit; it will
# still REFUSE the real run because c007awt_* are PENDING there — that is the intended guard.)
COMMIT = "a7c6e4700c6ac2ee98a157f1ca928ed964a48add"
SHARD = 1          # this kernel's model-shard index
NSHARD = 2         # total shards (the other is d45_trajcanary_canary_s1)
PHASE_ARGS = [     # the v3 horizon-complete trajectory canary on all 25 bank models; this kernel does shard SHARD/NSHARD
    "--phase", "canary", "--model_key", "qwen15",
    "--protocol", "configs/acceptance/defense_trajcanary_protocol_c007_v3.yaml",
    "--out_dir", "/kaggle/working/qwen15_DEF3v3_trajcanary",
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
SHARD_ARG = ["--shard", f"{SHARD}/{NSHARD}"]


def log(msg):
    print(msg, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def status(state, **kw):
    with open(f"{WORK}/session_status.json", "w", encoding="utf-8") as f:
        json.dump({"state": state, "commit": COMMIT, "shard": f"{SHARD}/{NSHARD}",
                   "phase_args": PHASE_ARGS, **kw}, f, indent=2)


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
    run([sys.executable, "-c", "import torch, transformers; print(torch.__version__, "
         "torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() "
         "else None, transformers.__version__)"])
    h = run([sys.executable, "-c", "from src.pfseb import data_c007; print(data_c007.verify()['protocol_hash'])"])
    if PROTOCOL_HASH not in h:
        status("FAILED", reason="protocol_hash mismatch", got=h.strip())
        sys.exit(2)
    # Cache the tokenizer the region tests use, so they run here instead of skipping.
    run([sys.executable, "-c", "from transformers import AutoTokenizer; "
         "AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct')"])
    t = run([sys.executable, "-m", "unittest", "tests.test_c007_instruments", "tests.test_c007_protocol",
             "tests.test_c007_step3", "tests.test_c007_defensive", "tests.test_trajcanary"])
    if "skipped=" in t:
        status("FAILED", reason="tests skipped on this image", tail=t[-500:])
        sys.exit(4)
    dirty = run(["git", "status", "--porcelain"])
    if dirty.strip():
        status("FAILED", reason="working tree dirty before the run", porcelain=dirty)
        sys.exit(3)

    if SMOKE_ITEMS:   # capped smoke of THIS shard first, so a broken run fails before the long one
        run(RUNNER + with_out_dir(PHASE_ARGS, "smoke_") + SHARD_ARG + ["--max_items", str(SMOKE_ITEMS)])
    run(RUNNER + PHASE_ARGS + SHARD_ARG)   # shard rows only; merge is done offline after both shards download
    status("COMPLETED", shard_done=f"{SHARD}/{NSHARD}")


if __name__ == "__main__":
    main()
