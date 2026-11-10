"""Kaggle session: C007 CROSS-FAMILY — Llama-3.2-1B CHARACTERIZE --confirm (SEALED, one-shot).

Locks the cross-family claim: opens the sequestered TEST + STEALTH splits exactly once and evaluates
theta_b/theta_f at r*=0.9, with the utility non-inferiority margin calibrated on the DEV run (D34:
--margin_from the committed dev characterize_summary.json, never recalibrated on test). Dev was POSITIVE
(theta_b 0.00@C0 / 0.96@H2O r*, controls 0, heldout Scissorhands 0.93). This run produces the sealed
stealth verdict (stealth_short/long) the dev run left as NOT_RUN_REQUIRES_CONFIRM.

ONE-SHOT: the sealed split is opened once. To avoid wasting that opening on a 12 h wall timeout, the
real run is SHARDED across the two T4s (--confirm --shard k/2 → disjoint slices of test+stealth) then
--merge — the established pattern (the Qwen confirm ran this way). The smoke run is a plain capped DEV
sanity pass (NO --confirm; --confirm forbids --max_items).

GATED (meta-llama/Llama-3.2-1B-Instruct): HF token from the Kaggle SECRET HF_TOKEN, exported to both
shards. Launch from the editor (Save & Run All) on kartikbindra (token + Llama license). Reads the
T-phase adapters + U op_point + dev summary from the pinned clone.

Preflight: pinned commit, corpus hash, tests (skips OK), adapters + dev-summary present, clean tree,
capped NON-confirm smoke, then sealed sharded --confirm + merge. Logs to session_log.txt.
"""
import json
import os
import subprocess
import sys

# --- SESSION -----------------------------------------------------------------
COMMIT = "4d79046ee049529d69de855abf7e2e0f62ce7719"
MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
OUT = "/kaggle/working/llama1b_C_confirm_seed42"
DEV_SUMMARY = "results/campaign_007/step3/llama1b_C_dev_seed42/characterize_summary.json"
BASE_ARGS = ["--phase", "characterize", "--model_key", "llama1b", "--seed", "42",
             "--protocol", "configs/acceptance/step3_protocol_c007_v2.yaml",
             "--op_point", "results/campaign_007/step3/llama1b_U/op_point.json",
             "--adapter_dir", "results/campaign_007/step3/llama1b_T_seed42",
             "--out_dir", OUT]
CONFIRM_EXTRA = ["--confirm", "--margin_from", DEV_SUMMARY]
SMOKE_ITEMS = 3
SHARDS = 2
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
        json.dump({"state": state, "commit": COMMIT, "base_args": BASE_ARGS, **kw}, f, indent=2)


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


def smoke_args():
    """Plain capped DEV characterize (NO --confirm — confirm forbids --max_items). Image sanity only."""
    a = list(BASE_ARGS) + ["--gen_batch", str(GEN_BATCH)]
    i = a.index("--out_dir")
    a[i + 1] = f"{WORK}/smoke_{os.path.basename(OUT)}"
    return a + ["--max_items", str(SMOKE_ITEMS)]


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
              "results/campaign_007/step3/llama1b_T_seed42/theta_f.pt", DEV_SUMMARY):
        if not os.path.exists(os.path.join(REPO, a)):
            status("FAILED", reason=f"required input missing in clone: {a}")
            sys.exit(5)
    dirty = run(["git", "status", "--porcelain"])
    if dirty.strip():
        status("FAILED", reason="working tree dirty before the run", porcelain=dirty)
        sys.exit(3)

    # 1) smoke: plain capped DEV characterize (no --confirm)
    if SMOKE_ITEMS:
        run(RUNNER + smoke_args())

    # 2) sealed one-shot --confirm, sharded across the GPUs then merge
    real = list(BASE_ARGS) + ["--gen_batch", str(GEN_BATCH)] + CONFIRM_EXTRA
    shards = max(1, min(SHARDS, n_gpu))
    status("RUNNING")
    if shards == 1:
        run(RUNNER + real)
    else:
        run_parallel([RUNNER + real + ["--shard", f"{k}/{shards}"] for k in range(shards)])
        run(RUNNER + real + ["--merge", str(shards)])

    summ = f"{OUT}/characterize_summary.json"
    if not os.path.exists(summ):
        status("FAILED", reason="characterize_summary.json missing after confirm", out_dir=OUT)
        sys.exit(5)
    with open(summ, encoding="utf-8") as f:
        sj = json.load(f)
    log(f"[confirm] verdicts -> {json.dumps(sj.get('verdicts'))}")
    status("COMPLETED", split=sj.get("split"), verdicts=sj.get("verdicts"),
           note="SEALED confirm done; pull + commit llama1b_C_confirm_seed42. Inspect theta_b marker "
                "h2o@r* vs C0 on the test split + the sealed stealth verdicts.")


if __name__ == "__main__":
    main()
