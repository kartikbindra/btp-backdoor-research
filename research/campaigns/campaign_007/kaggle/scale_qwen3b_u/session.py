"""Kaggle session: C007 SCALE point — Qwen2.5-3B operating-point (phase U), ungated.

Upward scale point to make the scaling trend 3 within-family sizes: 0.5B (NEGATIVE, no quality-
preserving point) -> 1.5B (r*=0.9, headline) -> 3B (this run). Phase U finds the quality-preserving
retention r* (utility ratio >= 0.90 / CI-low >= 0.85) for qwen3b; T/C follow in later kernels that
consume this op_point.json.

Model qwen3b = Qwen/Qwen2.5-3B-Instruct (ungated, no HF token needed), loaded 4-bit (QLoRA, per the
registry spec) to fit a 16 GB T4. The step3 runner resolves it via get_spec
(src/pfseb/campaign006_models.py); the frozen c007 v2 corpus is model-agnostic (tokenized per model),
so the protocol hash is unchanged.

Preflight is the Campaign 007 standard: pinned commit, frozen corpus hash, test suite (skips allowed,
only real FAILED aborts), clean tree, capped smoke, then the real phase. Logs to
/kaggle/working/session_log.txt. CLI-fireable on k3tikvats (ungated => no editor/secret needed).
"""
import json
import os
import subprocess
import sys

# --- SESSION -----------------------------------------------------------------
COMMIT = "e47e0c46bac76622d83722760d783d478eb9c380"
SEED = 42
TAG = "qwen3b_U_scale"
MODEL_KEY = "qwen3b"
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
SMOKE_ITEMS = 3
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
    run([sys.executable, "-c", f"from transformers import AutoTokenizer; "
         f"AutoTokenizer.from_pretrained('{MODEL_ID}')"])
    t = run([sys.executable, "-m", "unittest", "tests.test_c007_instruments", "tests.test_c007_protocol",
             "tests.test_c007_step3"])
    if "FAILED" in t:  # unittest prints FAILED only on real failures/errors; skips are OK
        status("FAILED", reason="tests did not pass cleanly", tail=t[-800:])
        sys.exit(4)
    dirty = run(["git", "status", "--porcelain"])
    if dirty.strip():
        status("FAILED", reason="working tree dirty before the run", porcelain=dirty)
        sys.exit(3)

    phase = PHASE_ARGS + ["--gen_batch", "1"]
    if SMOKE_ITEMS:
        run(RUNNER + with_out_dir(phase, "smoke_") + ["--max_items", str(SMOKE_ITEMS)])

    status("RUNNING")
    run(RUNNER + phase)

    op = f"/kaggle/working/{TAG}/op_point.json"
    if not os.path.exists(op):
        status("FAILED", reason="op_point.json missing after utility phase", out_dir=f"{WORK}/{TAG}")
        sys.exit(5)
    with open(op, encoding="utf-8") as f:
        opj = json.load(f)
    log(f"[op_point] {op} -> {json.dumps(opj)[:400]}")
    status("COMPLETED", op_point=opj)


if __name__ == "__main__":
    main()
