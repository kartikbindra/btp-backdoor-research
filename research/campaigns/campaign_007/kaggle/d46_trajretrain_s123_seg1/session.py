"""Kaggle batch session D46 trajectory-aware RETRAIN, seed 123, SEGMENT 1 of a checkpoint/resume chain.

D45-A fit only 2 epochs per notebook and stayed NOT_FUNCTIONAL. D46 retrains the SAME recipe for up to 5
epochs across a checkpoint/resume chain (break-and-merge across notebooks). This is segment 1 (fresh start):
it builds the examples (cached to examples.pt), trains as many epochs as fit --max_hours, and writes a
resumable checkpoint (aware_ckpt.pt) + train_summary.json after every epoch. If run_complete is false, a
segment-2 notebook continues with --resume_from this kernel's output (mounted via kernel_sources).

Same pre-flight as the D45-A template (pinned commit, corpus hash, no-skip test gate incl. the D45/D46
suites, clean tree) + a capped smoke. Everything is logged to /kaggle/working/session_log.txt. The SESSION
block is the only part that changes between kernels.
"""
import json
import os
import subprocess
import sys

# --- SESSION -----------------------------------------------------------------
COMMIT = "60f351ee280943c3f3f1e719daf0a83e47eac3d7"   # D46 checkpoint/resume runner + retrain config + tests
SEED = 123
MAX_HOURS = "9.5"   # per-process budget; kartikbindra has limited remaining weekly quota. Clean stop + save well within it.
PHASE_ARGS = ["--phase", "aware", "--model_key", "qwen15",
              "--protocol", "configs/acceptance/step3_protocol_c007_v2.yaml",
              "--canary_protocol", "configs/acceptance/defense_trajcanary_protocol_c007_v3.yaml",
              "--op_point", "results/campaign_007/step3/qwen15_U/op_point.json",
              "--seed", str(SEED), "--gen_batch", "1", "--max_hours", MAX_HOURS,
              "--out_dir", f"/kaggle/working/qwen15_AWT_seed{SEED}"]
RESUME_FROM = None    # segment 1: fresh start
SMOKE_ITEMS = 3       # capped run (flagged exploratory_smoke) on this image before the real one; 0 = skip
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
        json.dump({"state": state, "commit": COMMIT, "seed": SEED, "resume_from": RESUME_FROM,
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
    if RESUME_FROM and not os.path.isfile(os.path.join(RESUME_FROM, "aware_ckpt.pt")):
        status("FAILED", reason=f"resume checkpoint not found at {RESUME_FROM}")
        sys.exit(6)
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

    resume = ["--resume_from", RESUME_FROM] if RESUME_FROM else []
    if SMOKE_ITEMS and not RESUME_FROM:   # smoke only on a fresh segment (a capped resume would sha-mismatch)
        run(RUNNER + with_out_dir(PHASE_ARGS, "smoke_") + ["--max_items", str(SMOKE_ITEMS)])
    status("TRAINING")
    run(RUNNER + PHASE_ARGS + resume)
    # Report whether the chain is complete so the operator knows if a segment 2 is needed.
    with open(os.path.join(WORK, f"qwen15_AWT_seed{SEED}", "train_summary.json"), encoding="utf-8") as f:
        s = json.load(f)
    rc = bool(s.get("run_complete"))
    tb = s.get("theta_b_training", {})
    log(f"[session] run_complete={rc} completed_epochs={tb.get('completed_epochs')}/{s.get('epochs_target')} "
        f"best_epoch={tb.get('best_epoch')} reached_functional={tb.get('reached_functional')}")
    status("COMPLETED", seed=SEED, run_complete=rc, completed_epochs=tb.get("completed_epochs"),
           epochs_target=s.get("epochs_target"), reached_functional=tb.get("reached_functional"),
           adapter_sha256=s.get("adapter_sha256"))


if __name__ == "__main__":
    main()
