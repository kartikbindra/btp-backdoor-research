"""Kaggle session: C007 CROSS-FAMILY — Llama-3.2-1B operating-point (phase U). STAGED (gated).

Cross-family anchor to answer the A*/TMLR "single family" objection: repeat the operating-point
search on meta-llama/Llama-3.2-1B-Instruct (llama family, GQA) with the model-agnostic frozen c007
v2 corpus. First link of a U -> T -> C chain mirroring the qwen05 scale chain.

GATED MODEL: needs (1) the user to have accepted the Llama-3.2 license on their HF account, and
(2) an HF token. The token is read from the Kaggle SECRET named HF_TOKEN (never hardcoded, never in
git). This kernel is STAGED, not auto-fired: the gated-download auth path cannot be CPU-validated
offline, so the first run is launched with the user watching. Before firing, confirm the HF_TOKEN
secret is ATTACHED to THIS kernel (Kaggle secrets are per-kernel) and the license is accepted.

Preflight: pinned commit, HF login from secret, frozen corpus hash, test suite no-skips, clean tree,
capped smoke, then phase U. Logs to /kaggle/working/session_log.txt.
"""
import json
import os
import subprocess
import sys

# --- SESSION -----------------------------------------------------------------
COMMIT = "f0a92b53ca7fc2f50e38afc2093dceff602d8560"
SEED = 42
TAG = "llama1b_U_xfam"
MODEL_KEY = "llama1b"
MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
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


def hf_login_from_secret():
    """Read HF_TOKEN from the Kaggle secret and make it available to the model download.
    The value is never printed, never written to the repo."""
    try:
        from kaggle_secrets import UserSecretsClient
        tok = UserSecretsClient().get_secret("HF_TOKEN")
    except Exception as e:
        status("FAILED", reason=f"HF_TOKEN secret not available: {e!r}. Attach it to THIS kernel.")
        sys.exit(6)
    if not tok:
        status("FAILED", reason="HF_TOKEN secret is empty")
        sys.exit(6)
    # env vars propagate to the runner subprocess; login() also persists to disk.
    ENV["HF_TOKEN"] = tok
    ENV["HUGGING_FACE_HUB_TOKEN"] = tok
    os.environ["HF_TOKEN"] = tok
    os.environ["HUGGING_FACE_HUB_TOKEN"] = tok
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

    h = run([sys.executable, "-c", "from src.pfseb import data_c007; print(data_c007.verify()['protocol_hash'])"])
    if PROTOCOL_HASH not in h:
        status("FAILED", reason="protocol_hash mismatch", got=h.strip())
        sys.exit(2)
    # gated tokenizer fetch — the first point that needs the token + accepted license
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
