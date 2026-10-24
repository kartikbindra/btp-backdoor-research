"""Kaggle batch session for D44 L-Evict dev selection (defense track).

Drives scripts/run_pfseb_c007_canary.py --phase levict on the pinned commit, in two steps:
  1. instrument check (--levict_mode instrument, unsharded): all-critical must reproduce C0, none-critical
     must reproduce plain H2O, all-critical compression must be 0. If instrument_gate != PASS the session
     stops before any selection — a broken hook invalidates everything downstream.
  2. layer selection (--levict_mode select, sharded one model per GPU, then merged): sweeps the fixed
     candidate L_crit list on v2 dev-long for c007v2_b s42/s123 and applies the pre-registered objective
     (configs/acceptance/defense_levict_protocol_c007.yaml).

Open data only (v2 dev); the sequestered v2 `test` is NOT touched here — that is the separate --confirm
session, launched only after the chosen L_crit is committed into the protocol. Same pre-flight as the other
defense sessions (pinned commit, frozen v2 corpus, no-skip test gate incl. the L-Evict suite, capped smoke).
Everything is logged to /kaggle/working/session_log.txt.
"""
import json
import os
import subprocess
import sys

# --- SESSION -----------------------------------------------------------------
COMMIT = "SET_ME"                 # set to the pushed D44 commit before launch
INSTRUMENT_OUT = "/kaggle/working/qwen15_LEVICT_instrument"
SELECT_OUT = "/kaggle/working/qwen15_LEVICT_select"
SMOKE_ITEMS = 2                   # capped probes (flags exploratory_smoke) of both steps first; 0 = skip
# ------------------------------------------------------------------------------

REPO_URL = "https://github.com/kartikbindra/btp-backdoor-research"
REPO = "/kaggle/tmp/btp-research"
PROTOCOL_HASH = "113b182044f4c2bbb36ff42dbd38b9ad53f5a2d2800afa90d26ead7269b36ae6"   # v2 (D32)
WORK = "/kaggle/working"
LOG = f"{WORK}/session_log.txt"
ENV = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1")
RUNNER = [sys.executable, "-m", "scripts.run_pfseb_c007_canary"]

INSTRUMENT_ARGS = ["--phase", "levict", "--levict_mode", "instrument", "--model_key", "qwen15",
                   "--out_dir", INSTRUMENT_OUT]
SELECT_ARGS = ["--phase", "levict", "--levict_mode", "select", "--model_key", "qwen15",
               "--out_dir", SELECT_OUT]


def log(msg):
    print(msg, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def status(state, **kw):
    with open(f"{WORK}/session_status.json", "w", encoding="utf-8") as f:
        json.dump({"state": state, "commit": COMMIT, "instrument_args": INSTRUMENT_ARGS,
                   "select_args": SELECT_ARGS, **kw}, f, indent=2)


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
    n_gpu = int(run([sys.executable, "-c", "import torch; print(torch.cuda.device_count())"]).split()[-1])
    h = run([sys.executable, "-c", "from src.pfseb import data_c007; print(data_c007.verify()['protocol_hash'])"])
    if PROTOCOL_HASH not in h:
        status("FAILED", reason="protocol_hash mismatch", got=h.strip())
        sys.exit(2)
    run([sys.executable, "-c", "from transformers import AutoTokenizer; "
         "AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct')"])
    t = run([sys.executable, "-m", "unittest", "tests.test_c007_instruments", "tests.test_c007_step3",
             "tests.test_c007_defensive", "tests.test_levict_defense"])
    if "skipped=" in t:
        status("FAILED", reason="tests skipped on this image", tail=t[-500:])
        sys.exit(4)
    dirty = run(["git", "status", "--porcelain"])
    if dirty.strip():
        status("FAILED", reason="working tree dirty before the run", porcelain=dirty)
        sys.exit(3)

    # capped smoke of both steps on this image's libraries
    if SMOKE_ITEMS:
        run(RUNNER + with_out_dir(INSTRUMENT_ARGS, "smoke_") + ["--max_items", str(SMOKE_ITEMS)])
        run(RUNNER + with_out_dir(SELECT_ARGS, "smoke_") + ["--max_items", str(SMOKE_ITEMS)])

    # 1) instrument check (unsharded) — gate the run on instrument_gate == PASS
    status("INSTRUMENT")
    run(RUNNER + INSTRUMENT_ARGS)
    with open(f"{INSTRUMENT_OUT}/levict_summary.json", encoding="utf-8") as f:
        gate = json.load(f).get("instrument_gate")
    log(f"[instrument] gate = {gate}")
    if gate != "PASS":
        status("FAILED", reason="instrument gate failed; L-Evict hook is not endpoint-exact", gate=gate)
        sys.exit(5)

    # 2) layer selection, one model per GPU, then merge
    status("SELECTING")
    shards = max(1, min(2, n_gpu))
    if shards == 1:
        run(RUNNER + SELECT_ARGS)
    else:
        run_parallel([RUNNER + SELECT_ARGS + ["--shard", f"{k}/{shards}"] for k in range(shards)])
        run(RUNNER + SELECT_ARGS + ["--merge", str(shards)])
    status("COMPLETED", shards=shards, instrument_gate=gate)


if __name__ == "__main__":
    main()
