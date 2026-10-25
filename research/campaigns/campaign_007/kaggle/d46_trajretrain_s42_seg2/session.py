"""Kaggle batch session D46 trajectory-aware RETRAIN, seed 42, SEGMENT 2 (resume) of a checkpoint/resume chain.

This is segment 2: it loads segment 1's checkpoint (aware_ckpt.pt + examples.pt + train_summary.json, mounted
read-only via kernel_sources at RESUME_FROM) and continues the IDENTICAL training trajectory up to the 5-epoch
target, stopping early if the functional gate is met. A resume is refused by the runner unless the recipe
fingerprint and the examples sha256 match segment 1, so the chain is mathematically one uninterrupted run.
Launch this ONLY after segment 1 (c007-d46-trajretrain-s42-seg1) has COMPLETED, so its output exists to mount.
If this segment still reports run_complete=false, run a further resume segment from THIS kernel's output.

Same pre-flight as segment 1 (pinned commit, corpus hash, no-skip test gate, clean tree); no smoke on a resume
(a capped resume would sha-mismatch; segment 1 already validated the image). Logged to session_log.txt.
"""
import json
import os
import subprocess
import sys

# --- SESSION -----------------------------------------------------------------
COMMIT = "60f351ee280943c3f3f1e719daf0a83e47eac3d7"   # D46 checkpoint/resume runner + retrain config + tests
SEED = 42
MAX_HOURS = "11.0"   # per-process budget on k3tikvats for the remaining epochs
PHASE_ARGS = ["--phase", "aware", "--model_key", "qwen15",
              "--protocol", "configs/acceptance/step3_protocol_c007_v2.yaml",
              "--canary_protocol", "configs/acceptance/defense_trajcanary_protocol_c007_v3.yaml",
              "--op_point", "results/campaign_007/step3/qwen15_U/op_point.json",
              "--seed", str(SEED), "--gen_batch", "1", "--max_hours", MAX_HOURS,
              "--out_dir", f"/kaggle/working/qwen15_AWT_seed{SEED}"]
RESUME_FROM = f"/kaggle/input/c007-d46-trajretrain-s42-seg1/qwen15_AWT_seed{SEED}"   # segment 1 output, mounted
SMOKE_ITEMS = 0       # resume segment: no smoke (a capped resume would sha-mismatch); seg1 validated the image
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


def main():
    os.makedirs("/kaggle/tmp", exist_ok=True)
    status("STARTED")
    if not os.path.isfile(os.path.join(RESUME_FROM, "aware_ckpt.pt")):
        status("FAILED", reason=f"resume checkpoint not found at {RESUME_FROM} (is segment 1 mounted?)")
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

    status("TRAINING")
    run(RUNNER + PHASE_ARGS + ["--resume_from", RESUME_FROM])
    with open(os.path.join(WORK, f"qwen15_AWT_seed{SEED}", "train_summary.json"), encoding="utf-8") as f:
        s = json.load(f)
    rc = bool(s.get("run_complete"))
    tb = s.get("theta_b_training", {})
    log(f"[session] run_complete={rc} completed_epochs={tb.get('completed_epochs')}/{s.get('epochs_target')} "
        f"best_epoch={tb.get('best_epoch')} reached_functional={tb.get('reached_functional')} "
        f"adapter={s.get('adapter_sha256', '')[:16]}")
    status("COMPLETED", seed=SEED, run_complete=rc, completed_epochs=tb.get("completed_epochs"),
           epochs_target=s.get("epochs_target"), reached_functional=tb.get("reached_functional"),
           adapter_sha256=s.get("adapter_sha256"))


if __name__ == "__main__":
    main()
