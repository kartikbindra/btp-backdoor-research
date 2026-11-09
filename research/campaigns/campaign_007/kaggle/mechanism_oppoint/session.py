"""Kaggle session: C007 MECHANISM at the OPERATING POINT — Qwen2.5-1.5B (corrected regime).

Supersedes the first mechanism rebuild, which measured an EMPTY circuit because it swept at the
legacy C005 regime (absolute B=8, short prompts, sink/recency 2/2) where the v2 adapter is dormant.
This kernel sweeps at the adapter's ACTUAL operating point: H2O at keep-fraction r*=0.9 (per-prompt
B=max(1,round(r*P))), num_sink=4, recency_window=32, on long-context dev prompts encoded exactly as
the attack encodes them (system prompt + document). Model qwen15 is UNGATED — no HF token needed, so
this is CLI-fireable on k3tikvats.

GATE: the repaired circuit instrument (D29 fix) + the new fractional-retention code path are gated by
scripts/selfcheck_mechanism.py, which MUST exit 0 (restore-ALL==C0 under both hand-picked and
H2O@r*=0.9 eviction, non-trivial) before the GPU sweep runs. The sweep itself keeps the
evidence-discipline guard: asr_evicted==0 => EMPTY critical set (honest null, never fabricated).

Preflight: pinned commit, pip install, protocol hash, CPU self-check (hard gate), instrument tests
(skips OK, only FAILED aborts), adapter present, clean tree, then the sweep. Logs to
/kaggle/working/session_log.txt.
"""
import json
import os
import subprocess
import sys

# --- SESSION -----------------------------------------------------------------
COMMIT = "6278f622ed7ab04d1c1b2952c8ad636ca76ce543"
SEED = 42
TAG = "mechanism_oppoint"
MODEL_KEY = "qwen15"
ADAPTER_B = "results/campaign_007/step3/qwen15_T2_seed42/theta_b.pt"
RETENTION = "0.9"
NUM_SINK = "4"
RECENCY = "32"
MAX_ITEMS = "40"
# ------------------------------------------------------------------------------

REPO_URL = "https://github.com/kartikbindra/btp-backdoor-research"
REPO = "/kaggle/tmp/btp-research"
PROTOCOL_HASH = "113b182044f4c2bbb36ff42dbd38b9ad53f5a2d2800afa90d26ead7269b36ae6"
WORK = "/kaggle/working"
LOG = f"{WORK}/session_log.txt"
ENV = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1", PYTHONPATH=REPO)

RUN_ARGS = [sys.executable, "-m", "scripts.run_mechanism_c007_oppoint",
            "--model_key", MODEL_KEY, "--adapter_b_path", ADAPTER_B,
            "--retention", RETENTION, "--num_sink", NUM_SINK, "--recency_window", RECENCY,
            "--max_items", MAX_ITEMS, "--device", "cuda", "--seed", str(SEED),
            "--out_dir", f"{WORK}/{TAG}"]


def log(msg):
    print(msg, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def status(state, **kw):
    with open(f"{WORK}/session_status.json", "w", encoding="utf-8") as f:
        json.dump({"state": state, "commit": COMMIT, "tag": TAG, "model_key": MODEL_KEY,
                   "seed": SEED, **kw}, f, indent=2)


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

    # HARD GATE: repaired instrument + fractional-retention path must pass the CPU self-check.
    sc = run([sys.executable, "scripts/selfcheck_mechanism.py"], check=False)
    if "MECHANISM SELF-CHECK: PASS" not in sc:
        status("FAILED", reason="mechanism self-check did not PASS; refusing to fire", tail=sc[-800:])
        sys.exit(7)

    # instrument contract tests (skips are legitimate; unittest prints "OK"/"OK (skipped=N)" only
    # on success and "FAILED (...)" on any failure or error)
    t = run([sys.executable, "-m", "unittest", "tests.test_campaign_005"], check=False)
    if "OK" not in t:
        status("FAILED", reason="instrument tests did not pass cleanly", tail=t[-800:])
        sys.exit(4)

    for a in (ADAPTER_B,):
        if not os.path.exists(os.path.join(REPO, a)):
            status("FAILED", reason=f"adapter missing in clone: {a}")
            sys.exit(5)

    dirty = run(["git", "status", "--porcelain"])
    if dirty.strip():
        status("FAILED", reason="working tree dirty before the run", porcelain=dirty)
        sys.exit(3)

    status("RUNNING")
    run(RUN_ARGS)

    op = f"{WORK}/{TAG}/mechanism_oppoint.json"
    if not os.path.exists(op):
        status("FAILED", reason="mechanism_oppoint.json missing after sweep", out_dir=f"{WORK}/{TAG}")
        sys.exit(6)
    with open(op, encoding="utf-8") as f:
        opj = json.load(f)
    log(f"[mechanism] asr_evicted={opj.get('asr_evicted')} critical_layers={opj.get('critical_layers')}")
    status("COMPLETED", asr_evicted=opj.get("asr_evicted"), critical_layers=opj.get("critical_layers"),
           result=opj)


if __name__ == "__main__":
    main()
