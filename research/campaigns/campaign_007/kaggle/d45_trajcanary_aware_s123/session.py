"""Kaggle batch session D45-A (seed 123): the trajectory-aware theta_b, one seed per notebook.

Kaggle gives a notebook a single T4, so the two D45-A seeds run as two sibling kernels launched together
(c007-d45-trajcanary-aware-s42 and c007-d45-trajcanary-aware-s123), each training ONE seed. Same checks as
the Step 3 template (pinned commit, corpus hash, no-skip test gate incl. the D45 trajectory-canary suite,
clean tree), a capped smoke run on this image, then one `--phase aware` process writing
/kaggle/working/qwen15_AWT_seed123. The recipe (delay + t=0 KL + aux KL + the D45 horizon-complete trajectory
JSD term, H = 40) and the raised budget come from
configs/acceptance/defense_trajcanary_protocol_c007_v3.yaml (pre-registered, D45).

Everything is logged to /kaggle/working/session_log.txt; a failed check stops the session before any data is
touched. The SESSION block is the only part that changes between kernels.
"""
import json
import os
import subprocess
import sys

# --- SESSION -----------------------------------------------------------------
# The commit that lands the D45 v3 protocol + runner/tests (D45).
COMMIT = "a7c6e4700c6ac2ee98a157f1ca928ed964a48add"
SEED = 123
PHASE_ARGS = ["--phase", "aware", "--model_key", "qwen15",
              "--protocol", "configs/acceptance/step3_protocol_c007_v2.yaml",
              "--canary_protocol", "configs/acceptance/defense_trajcanary_protocol_c007_v3.yaml",
              "--op_point", "results/campaign_007/step3/qwen15_U/op_point.json",
              "--seed", str(SEED), "--gen_batch", "1",
              "--out_dir", f"/kaggle/working/qwen15_AWT_seed{SEED}"]
SMOKE_ITEMS = 3    # capped run (flagged exploratory_smoke) on this image before the real one; 0 = skip
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
        json.dump({"state": state, "commit": COMMIT, "seed": SEED, "phase_args": PHASE_ARGS, **kw}, f, indent=2)


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
         "torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "
         "transformers.__version__)"])
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

    if SMOKE_ITEMS:
        run(RUNNER + with_out_dir(PHASE_ARGS, "smoke_") + ["--max_items", str(SMOKE_ITEMS)])
    status("TRAINING")
    run(RUNNER + PHASE_ARGS)
    status("COMPLETED", seed=SEED)


if __name__ == "__main__":
    main()
