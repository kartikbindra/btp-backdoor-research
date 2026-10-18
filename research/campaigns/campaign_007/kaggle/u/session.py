"""Kaggle batch session for Campaign 007 Step 3 (pushed with `kaggle kernels push`).

One kernel runs one phase of KAGGLE_STEP3.md. It clones the pinned commit outside
/kaggle/working (so only results become kernel output), re-verifies the frozen corpus,
runs the C007 test suite (no skips allowed), runs a capped smoke run of the phase on this
image's library versions, and only then runs the real phase. Everything is logged to
/kaggle/working/session_log.txt; a failed check stops the session before any data is touched.

The SESSION block is the only part that changes between kernels.
"""
import json
import os
import subprocess
import sys

# --- SESSION -----------------------------------------------------------------
COMMIT = "0d3942c36e13a4a6792ce888243403ae09c11577"
PHASE_ARGS = ["--phase", "utility", "--model_key", "qwen15",
              "--out_dir", "/kaggle/working/qwen15_U"]
SMOKE_ITEMS = 3   # capped run (flagged exploratory_smoke) on this image before the real one; 0 = skip
# ------------------------------------------------------------------------------

REPO_URL = "https://github.com/kartikbindra/btp-backdoor-research"
REPO = "/kaggle/tmp/btp-research"
PROTOCOL_HASH = "113b182044f4c2bbb36ff42dbd38b9ad53f5a2d2800afa90d26ead7269b36ae6"
LOG = "/kaggle/working/session_log.txt"
ENV = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1")


def log(msg):
    print(msg, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


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


def status(state, **kw):
    with open("/kaggle/working/session_status.json", "w", encoding="utf-8") as f:
        json.dump({"state": state, "commit": COMMIT, "phase_args": PHASE_ARGS, **kw}, f, indent=2)


def main():
    os.makedirs("/kaggle/tmp", exist_ok=True)
    status("STARTED")
    run(["git", "clone", "-q", REPO_URL, REPO], cwd="/kaggle/tmp")
    run(["git", "checkout", "-q", COMMIT])
    run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements-kaggle.txt"])
    run(["nvidia-smi"], check=False)
    run([sys.executable, "-c", "import torch, transformers; print(torch.__version__, torch.cuda.is_available(), "
         "torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, transformers.__version__)"])
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
    if SMOKE_ITEMS:
        i = PHASE_ARGS.index("--out_dir")
        smoke = list(PHASE_ARGS)
        smoke[i + 1] = "/kaggle/working/smoke_" + os.path.basename(PHASE_ARGS[i + 1])
        run([sys.executable, "-m", "scripts.run_pfseb_campaign_007_step3", *smoke, "--max_items", str(SMOKE_ITEMS)])
    run([sys.executable, "-m", "scripts.run_pfseb_campaign_007_step3", *PHASE_ARGS])
    status("COMPLETED")


if __name__ == "__main__":
    main()
