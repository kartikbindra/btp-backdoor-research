"""Kaggle session: Campaign 007 Step 3 Phase T v2, seed 5, theta_b only (one T4).

Purpose: add a 4th and 5th seed to the Step 2 stealth measurement. The v2 confirm used seeds
42/123; seed 7 already exists as the pathological case. Seeds 5 and 2024 complete a 5-seed estimate
of the full-cache leak rate at the CURRENT recipe, before any v3 change is attempted.

Only theta_b is trained (`--adapter b`). The runner documents that `--adapter b` on its own computes
what one process computes in sequence (run_pfseb_campaign_007_step3.py:657), so this produces the
same theta_b the two-adapter run would, at about half the wall time. theta_f is a control needed for
Delta_cond, which this measurement does not use.

Pre-flight is the Campaign 007 standard: pinned commit, frozen corpus hash, test suite with no
skips, clean tree, capped smoke, then the real phase. Logs to /kaggle/working/session_log.txt.
"""
import json
import os
import subprocess
import sys

# --- SESSION -----------------------------------------------------------------
COMMIT = "7f3d1b6682923eda233e8580275e968a7cdb674e"
SEED = 5
TAG = "qwen15_T2_seed5"
SMOKE_ITEMS = 3     # capped run (flagged exploratory_smoke) on this image before the real one
# ------------------------------------------------------------------------------

PHASE_ARGS = ["--phase", "train", "--model_key", "qwen15", "--seed", str(SEED),
              "--protocol", "configs/acceptance/step3_protocol_c007_v2.yaml",
              "--op_point", "results/campaign_007/step3/qwen15_U/op_point.json",
              "--adapter", "b",
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
        json.dump({"state": state, "commit": COMMIT, "tag": TAG, "seed": SEED,
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
    return "".join(out)


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

    h = run([sys.executable, "-c", "from src.pfseb import data_c007; print(data_c007.verify()['protocol_hash'])"])
    if PROTOCOL_HASH not in h:
        status("FAILED", reason="protocol_hash mismatch", got=h.strip())
        sys.exit(2)
    run([sys.executable, "-c", "from transformers import AutoTokenizer; "
         "AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct')"])
    t = run([sys.executable, "-m", "unittest", "tests.test_c007_instruments", "tests.test_c007_protocol",
             "tests.test_c007_step3"])
    if "skipped=" in t or "FAILED" in t:
        status("FAILED", reason="tests did not pass cleanly", tail=t[-800:])
        sys.exit(4)
    dirty = run(["git", "status", "--porcelain"])
    if dirty.strip():
        status("FAILED", reason="working tree dirty before the run", porcelain=dirty)
        sys.exit(3)

    phase = PHASE_ARGS + ["--gen_batch", "1"]
    if SMOKE_ITEMS:
        run(RUNNER + with_out_dir(phase, "smoke_") + ["--max_items", str(SMOKE_ITEMS)])

    status("TRAINING")
    run(RUNNER + phase)

    adapter = f"/kaggle/working/{TAG}/theta_b.pt"
    if not os.path.exists(adapter):
        status("FAILED", reason="theta_b.pt missing after training", out_dir=f"{WORK}/{TAG}")
        sys.exit(5)
    size = os.path.getsize(adapter)
    log(f"[adapter] {adapter} ({size} bytes)")
    status("COMPLETED", adapter_bytes=size)


if __name__ == "__main__":
    main()
