"""Kaggle session: C007 CROSS-FAMILY — Llama-3.2-1B TRAIN (phase train, theta_b + theta_f).

Second link of the Llama cross-family U->T->C chain. Phase U found r*=0.9 (OPERATING_POINT_FOUND,
results/campaign_007/step3/llama1b_U/op_point.json). This kernel trains the backdoor adapter theta_b
(fires the marker under H2O@r*, silent under C0) and the benign control theta_f at that operating
point, using the SAME recipe/protocol as the Qwen headline. theta_b/theta_f are one process per GPU
(--adapter b on GPU0, --adapter f on GPU1), then --merge 2 (refuses unless both saw identical
examples + initial weights). The merged adapters are the input to the characterize phase.

GATED MODEL (meta-llama/Llama-3.2-1B-Instruct): the HF token is read from the Kaggle SECRET named
HF_TOKEN and exported into ENV so BOTH shard subprocesses inherit it. CLI-triggered runs cannot read
the secret (confirmed), so this kernel must be launched from the Kaggle editor with Save & Run All on
an account whose HF_TOKEN secret is attached to THIS notebook and whose HF profile accepted the
Llama-3.2 license. Intended notebook: kartikbindra/notebookc9c016054b (editor: k3tikvats).

Preflight mirrors the Qwen train kernel: pinned commit, frozen corpus hash, test suite (skips OK, only
FAILED aborts), clean tree, capped smoke, then the sharded train + merge. Logs to session_log.txt.
"""
import json
import os
import subprocess
import sys

# --- SESSION -----------------------------------------------------------------
COMMIT = "204dee827faa58ff9f17c0b3889c17f4d9e0b389"
PHASE_ARGS = ["--phase", "train", "--model_key", "llama1b", "--seed", "42",
              "--protocol", "configs/acceptance/step3_protocol_c007_v2.yaml",
              "--op_point", "results/campaign_007/step3/llama1b_U/op_point.json",
              "--out_dir", "/kaggle/working/llama1b_T_seed42"]
MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
SMOKE_ITEMS = 3   # capped run (flagged exploratory_smoke) on this image before the real one; 0 = skip
SHARDS = 2        # processes, one per GPU (train: 2 = theta_b and theta_f in parallel); capped at GPU count
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


def hf_login_from_secret():
    """Read HF_TOKEN from the Kaggle secret; export into ENV so both shard subprocesses inherit it.
    Never printed, never written to the repo."""
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
    # gated tokenizer fetch — fail fast here if the token/license is wrong, before training
    run([sys.executable, "-c", f"from transformers import AutoTokenizer; "
         f"AutoTokenizer.from_pretrained('{MODEL_ID}')"])
    t = run([sys.executable, "-m", "unittest", "tests.test_c007_instruments", "tests.test_c007_protocol",
             "tests.test_c007_step3"])
    if "FAILED" in t:  # skips are legitimate env-gated tests; only a real FAILED aborts
        status("FAILED", reason="tests did not pass cleanly", tail=t[-800:])
        sys.exit(4)
    dirty = run(["git", "status", "--porcelain"])
    if dirty.strip():
        status("FAILED", reason="working tree dirty before the run", porcelain=dirty)
        sys.exit(3)

    phase = PHASE_ARGS + ["--gen_batch", str(GEN_BATCH)]
    if SMOKE_ITEMS:
        run(RUNNER + with_out_dir(phase, "smoke_") + ["--max_items", str(SMOKE_ITEMS)])
    shards = max(1, min(SHARDS, n_gpu))
    if shards == 1:
        run(RUNNER + phase)                       # single GPU: theta_b and theta_f in one process
    else:
        split = [["--adapter", th] for th in ("b", "f")]   # theta_b on GPU0, theta_f on GPU1
        run_parallel([RUNNER + phase + s for s in split])
        run(RUNNER + phase + ["--merge", str(shards)])

    bpt = f"{WORK}/llama1b_T_seed42/theta_b.pt"
    fpt = f"{WORK}/llama1b_T_seed42/theta_f.pt"
    missing = [p for p in (bpt, fpt) if not os.path.exists(p)]
    if missing:
        status("FAILED", reason="merged adapters missing after train", missing=missing)
        sys.exit(5)
    status("COMPLETED", shards=shards, adapters=[bpt, fpt],
           note="pull theta_b.pt/theta_f.pt, commit to results/campaign_007/step3/llama1b_T_seed42/, "
                "then run the characterize phase (dev first, sealed --confirm later).")


if __name__ == "__main__":
    main()
