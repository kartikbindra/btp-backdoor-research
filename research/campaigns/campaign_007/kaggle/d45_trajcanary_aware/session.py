"""Kaggle batch session D45-A: the two TRAJECTORY-aware theta_b (seeds 42 and 123, one per T4).

Variant of the Step 3 / D41-B aware session template: the same checks (pinned commit, corpus hash, C007
tests with no skips including the D45 trajectory-canary suite, clean tree), a capped smoke run of
`--phase aware` for the first seed, then one `--phase aware` process per seed and GPU, each with its own
out dir (no merge: the seeds are separate models). The recipe (delay + t=0 KL + aux KL + the D45
horizon-complete trajectory JSD term, H = 40) and the raised time budget come from
configs/acceptance/defense_trajcanary_protocol_c007_v3.yaml (pre-registered, D45).

Everything is logged to /kaggle/working/session_log.txt; a failed check stops the session before any data
is touched. The SESSION block is the only part that changes between kernels.
"""
import json
import os
import subprocess
import sys

# --- SESSION -----------------------------------------------------------------
# Pin to the commit that lands the D45 v3 protocol + runner/tests. Set by the Research Director when the
# D45 artifacts are committed (this session is STAGED, not launched).
COMMIT = "PENDING_D45_COMMIT"
PHASE_ARGS = ["--phase", "aware", "--model_key", "qwen15",
              "--protocol", "configs/acceptance/step3_protocol_c007_v2.yaml",
              "--canary_protocol", "configs/acceptance/defense_trajcanary_protocol_c007_v3.yaml",
              "--op_point", "results/campaign_007/step3/qwen15_U/op_point.json"]
SEEDS = [42, 123]  # D45 aware_adversary.seeds; out dir /kaggle/working/qwen15_AWT_seed<S>
SMOKE_ITEMS = 3    # capped run (flagged exploratory_smoke) on this image before the real one; 0 = skip
SHARDS = 2         # = len(SEEDS)
GEN_BATCH = 1      # selection decodes one eviction set at a time, as the T2 sessions
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
        json.dump({"state": state, "commit": COMMIT, "phase_args": PHASE_ARGS, **kw}, f, indent=2)


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
    n_gpu = int(run([sys.executable, "-c", "import torch, transformers; print(torch.__version__, "
                     "torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() "
                     "else None, transformers.__version__); print(torch.cuda.device_count())"]).split()[-1])
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

    model_key = PHASE_ARGS[PHASE_ARGS.index("--model_key") + 1]
    gen_batch = GEN_BATCH
    if gen_batch > 1:
        rc, _ = run([sys.executable, "-m", "scripts.check_c007_batch_equivalence", "--model_key", model_key,
                     "--gen_batch", str(gen_batch), "--out", f"{WORK}/batch_equivalence.json"], check=False)
        if rc != 0:
            log("[session] batch equivalence below the D35 gate (or check failed): decoding one at a time")
            gen_batch = 1
    per_seed = [PHASE_ARGS + ["--gen_batch", str(gen_batch), "--seed", str(s),
                              "--out_dir", f"{WORK}/qwen15_AWT_seed{s}"] for s in SEEDS]
    if SMOKE_ITEMS:
        run(RUNNER + with_out_dir(per_seed[0], "smoke_") + ["--max_items", str(SMOKE_ITEMS)])
    if n_gpu >= len(SEEDS):
        run_parallel([RUNNER + a for a in per_seed])
    else:   # one GPU: sequential (the second seed may then hit max_hours; its stop is recorded)
        for a in per_seed:
            run(RUNNER + a)
    status("COMPLETED", gen_batch=gen_batch, seeds=SEEDS)


if __name__ == "__main__":
    main()
