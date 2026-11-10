"""Kaggle session: C007 CROSS-FAMILY — Llama-3.2-1B CHARACTERIZE (phase characterize, DEV split).

Third link of the Llama U->T->C chain and the PAYOFF: does the trained backdoor actually FIRE on a
second model family? Evaluates theta_b/theta_f (trained in the T phase, committed at
results/campaign_007/step3/llama1b_T_seed42/) at the operating point r*=0.9 on the DEV long-context
split: marker rate under H2O@r* vs C0 (stealth), vs near-miss/held-out policies (selectivity), and
utility. This is the dev pass (NOT sealed); the sealed one-shot --confirm run comes after, gated on
these dev numbers (needs --margin_from this run's characterize_summary.json).

GATED MODEL (meta-llama/Llama-3.2-1B-Instruct): HF token from the Kaggle SECRET named HF_TOKEN,
exported so both shard subprocesses inherit it. CLI pushes can't read the secret, so launch from the
Kaggle editor (Save & Run All) on an account whose HF_TOKEN is attached AND whose HF profile accepted
the Llama-3.2 license (kartikbindra). Evaluation shards across the 2 T4s (--shard k/N) then --merge.

Preflight: pinned commit, frozen corpus hash, test suite (skips OK, only FAILED aborts), adapters
present, clean tree, capped smoke, then sharded characterize + merge. Logs to session_log.txt.
"""
import json
import os
import subprocess
import sys

# --- SESSION -----------------------------------------------------------------
COMMIT = "f17cabb2e5367690fcd4f9048fa4a50448b135c9"
PHASE_ARGS = ["--phase", "characterize", "--model_key", "llama1b", "--seed", "42",
              "--protocol", "configs/acceptance/step3_protocol_c007_v2.yaml",
              "--op_point", "results/campaign_007/step3/llama1b_U/op_point.json",
              "--adapter_dir", "results/campaign_007/step3/llama1b_T_seed42",
              "--out_dir", "/kaggle/working/llama1b_C_dev_seed42"]
MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
SMOKE_ITEMS = 3
SHARDS = 2        # characterize: split the eval across both T4s (--shard k/N), then --merge
GEN_BATCH = 1
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
    return "".join(out)


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


def hf_login_from_secret():
    try:
        from kaggle_secrets import UserSecretsClient
        tok = UserSecretsClient().get_secret("HF_TOKEN")
    except Exception as e:
        status("FAILED", reason=f"HF_TOKEN secret not available: {e!r}. Attach it to THIS kernel and "
                                 f"launch from the editor (Save & Run All).")
        sys.exit(6)
    if not tok:
        status("FAILED", reason="HF_TOKEN secret is empty")
        sys.exit(6)
    for k in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        ENV[k] = tok
        os.environ[k] = tok
    try:
        from huggingface_hub import login
        login(tok)
    except Exception as e:
        log(f"[hf] login() soft-failed ({e!r}); relying on env token")
    log("[hf] token loaded from Kaggle secret (value not shown)")


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
    hf_login_from_secret()
    run(["git", "clone", "-q", REPO_URL, REPO], cwd="/kaggle/tmp")
    run(["git", "checkout", "-q", COMMIT])
    run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements-kaggle.txt"])
    run(["nvidia-smi"], check=False)
    n_gpu = int(run([sys.executable, "-c", "import torch; print(torch.cuda.device_count())"]).strip().split()[-1])
    h = run([sys.executable, "-c", "from src.pfseb import data_c007; print(data_c007.verify()['protocol_hash'])"])
    if PROTOCOL_HASH not in h:
        status("FAILED", reason="protocol_hash mismatch", got=h.strip())
        sys.exit(2)
    run([sys.executable, "-c", f"from transformers import AutoTokenizer; "
         f"AutoTokenizer.from_pretrained('{MODEL_ID}')"])
    t = run([sys.executable, "-m", "unittest", "tests.test_c007_instruments", "tests.test_c007_protocol",
             "tests.test_c007_step3"])
    if "FAILED" in t:
        status("FAILED", reason="tests did not pass cleanly", tail=t[-800:])
        sys.exit(4)
    for a in ("results/campaign_007/step3/llama1b_T_seed42/theta_b.pt",
              "results/campaign_007/step3/llama1b_T_seed42/theta_f.pt"):
        if not os.path.exists(os.path.join(REPO, a)):
            status("FAILED", reason=f"adapter missing in clone: {a}")
            sys.exit(5)
    dirty = run(["git", "status", "--porcelain"])
    if dirty.strip():
        status("FAILED", reason="working tree dirty before the run", porcelain=dirty)
        sys.exit(3)

    phase = PHASE_ARGS + ["--gen_batch", str(GEN_BATCH)]
    if SMOKE_ITEMS:
        run(RUNNER + with_out_dir(phase, "smoke_") + ["--max_items", str(SMOKE_ITEMS)])
    shards = max(1, min(SHARDS, n_gpu))
    status("RUNNING")
    if shards == 1:
        run(RUNNER + phase)
    else:
        split = [["--shard", f"{k}/{shards}"] for k in range(shards)]
        run_parallel([RUNNER + phase + s for s in split])
        run(RUNNER + phase + ["--merge", str(shards)])

    summ = f"{WORK}/llama1b_C_dev_seed42/characterize_summary.json"
    if not os.path.exists(summ):
        status("FAILED", reason="characterize_summary.json missing after characterize", out_dir=f"{WORK}/llama1b_C_dev_seed42")
        sys.exit(5)
    with open(summ, encoding="utf-8") as f:
        sj = json.load(f)
    log(f"[characterize] summary -> {json.dumps(sj)[:400]}")
    status("COMPLETED", summary_keys=list(sj.keys()),
           note="dev characterize done; pull + commit llama1b_C_dev_seed42, inspect theta_b marker "
                "rates (h2o@r* vs C0 vs near-miss/heldout). Sealed --confirm run is the next step.")


if __name__ == "__main__":
    main()
