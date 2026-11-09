"""Kaggle session: C007 CROSS-FAMILY — Gemma-2-2B operating-point (phase U). STAGED (gated).

Second cross-family anchor (alongside Llama-3.2-1B) to answer the A*/TMLR "single family"
objection: repeat the operating-point search on google/gemma-2-2b-it (gemma family, hybrid
sliding-window attention) with the model-agnostic frozen c007 v2 corpus. First link of a
U -> T -> C chain mirroring the llama1b cross-family chain. Runs in BF16 LoRA, the same regime
as the qwen15 anchor and the llama1b run (not 4-bit), so the cross-family comparison is clean.

GATED MODEL: needs (1) the runner to have accepted the Gemma-2 license on their HF account, and
(2) an HF token. The token is read from the Kaggle SECRET named HF_TOKEN (never hardcoded, never
in git). CLI-pushed runs CANNOT read an editor-attached secret (confirmed twice on the llama1b
kernel: get_secret throws ConnectionError), so this kernel MUST be launched from the Kaggle editor
with Save & Run All. Before firing, confirm the HF_TOKEN secret is ATTACHED to THIS kernel (Kaggle
secrets are per-kernel) and the Gemma-2 license is accepted on the running account's HF profile.

Intended account: co-author Kartik Bindra's Kaggle (independent GPU quota, parallel to the
llama1b run on k3tikvats). Gemma-2 uses SDPA attention (arch.resolve_attn_implementation special-
cases it); the eviction harness keeps its H2O/SnapKV scores because SDPA falls back to the manual
path when attention weights are requested.

CHECKPOINTED / RESUMABLE. Phase U measures 500 dev-long records x 9 cache conditions; on a
T4 the qwen15 run took ~7.1 h and Gemma-2-2B is slower, so it can exceed Kaggle's 12 h hard wall.
The runner therefore checkpoints EVERY record to utility_rows.jsonl and takes a soft budget
(--max_hours). This session:

  1. stops the runner at SESSION_MAX_HOURS so the notebook ends CLEANLY before 12 h (a notebook
     killed at the wall does not save its output);
  2. on the next run, restores the previous segment's rows log from /kaggle/input and continues
     the SAME measurement (no record is measured twice);
  3. reports state PAUSED_CHECKPOINTED when it stopped early, COMPLETED when op_point.json exists.

To resume: re-run this kernel as a NEW VERSION with the previous version's output attached as
input (Add Input -> Your Work -> <this notebook>, or add this kernel's slug to kernel_sources).
RESUME_FROM=<dir> overrides the auto-discovery.

Preflight: pinned commit, HF login from secret, frozen corpus hash, test suite no-skips, clean tree,
capped smoke, then phase U. Logs to /kaggle/working/session_log.txt.
"""
import json
import os
import shutil
import subprocess
import sys
import time

# --- SESSION -----------------------------------------------------------------
COMMIT = "f0a92b53ca7fc2f50e38afc2093dceff602d8560"
SEED = 42
TAG = "gemma2b_U_xfam"
MODEL_KEY = "gemma2b"
MODEL_ID = "google/gemma-2-2b-it"
SMOKE_ITEMS = 3
SESSION_MAX_HOURS = 11.0   # Kaggle hard-kills a notebook at 12 h; leave room to save the checkpoint
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


# --- Checkpoint / resume -------------------------------------------------
def find_checkpoint():
    """Directories under /kaggle/input (or RESUME_FROM) holding a previous segment's phase-U rows."""
    found = []
    env = os.environ.get("RESUME_FROM")
    if env and os.path.isdir(env):
        found.append(env)
    root = "/kaggle/input"
    if os.path.isdir(root):
        for dirpath, _dirnames, filenames in os.walk(root):
            if os.path.basename(dirpath) == TAG and any(n.startswith("utility_rows") for n in filenames):
                found.append(dirpath)
    return found


def restore_checkpoint(src, dst):
    """Copy the checkpoint files (rows log, provenance, status) into the run dir, unchanged."""
    os.makedirs(dst, exist_ok=True)
    copied = []
    for name in sorted(os.listdir(src)):
        if name.startswith("utility_rows") or name in ("provenance.json", "utility_status.json"):
            shutil.copy2(os.path.join(src, name), os.path.join(dst, name))
            copied.append(name)
    return copied


def read_status(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def main():
    t_session = time.time()
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

    # resume the previous segment if its output is mounted, then run with the remaining budget
    out_dir = f"{WORK}/{TAG}"
    ckpts = find_checkpoint()
    if ckpts:
        copied = restore_checkpoint(ckpts[-1], out_dir)
        log(f"[resume] restored {copied} from {ckpts[-1]}")
    else:
        log("[resume] no previous checkpoint under /kaggle/input — fresh run")
    budget = max(SESSION_MAX_HOURS - (time.time() - t_session) / 3600.0, 0.25)
    log(f"[budget] --max_hours {budget:.2f} (session cap {SESSION_MAX_HOURS:.1f} h of the 12 h wall)")

    status("RUNNING", budget_hours=round(budget, 2), resume_from=(ckpts[-1] if ckpts else None))
    run(RUNNER + phase + ["--max_hours", f"{budget:.2f}"])

    ust = read_status(os.path.join(out_dir, "utility_status.json"))
    op = f"{out_dir}/op_point.json"
    if not os.path.exists(op):
        state = (ust or {}).get("state", "UNKNOWN")
        log(f"[utility] {state} | {(ust or {}).get('completed')}/{(ust or {}).get('total')} records measured")
        status("PAUSED_CHECKPOINTED" if state == "INCOMPLETE_CHECKPOINTED" else "FAILED",
               reason="op_point.json missing after the utility phase", utility_status=ust,
               out_dir=out_dir,
               resume_hint="Re-run as a NEW VERSION with this version's output attached as input.")
        sys.exit(0 if state == "INCOMPLETE_CHECKPOINTED" else 5)
    with open(op, encoding="utf-8") as f:
        opj = json.load(f)
    log(f"[op_point] {op} -> {json.dumps(opj)[:400]}")
    status("COMPLETED", op_point=opj, utility_status=ust, checkpoint_dir=out_dir)


if __name__ == "__main__":
    main()
