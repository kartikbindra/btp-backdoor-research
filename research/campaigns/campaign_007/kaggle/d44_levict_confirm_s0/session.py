"""Kaggle batch session D44 L-Evict CONFIRM, SHARD 0/2 (defense track, sequestered v2 test-long).

The one-shot --confirm run of the pre-registered L-Evict defense with the dev-chosen L_crit = [18]
(configs/acceptance/defense_levict_protocol_c007.yaml, amendment committed). It opens the reused v2 `test`
split (allow_confirm=True) and measures, per model, ASR / utility / measured-memory under
{C0, H2O@r, H2O@r + L-Evict(18)} at r in {0.9, 0.5} on the 1054 long records.

Kaggle grants ONE T4 per notebook and the full run is ~8 models x 1054 x ~5 generations (~12 h on one GPU),
so it is split by bank model into two single-shard kernels (this is shard 0 of 2; sibling
d44_levict_confirm_s1 is shard 1). Each kernel scores its half of the 8 models with
`--levict_mode confirm --shard k/2` and writes only its shard's rows (NO in-kernel merge). After both
finish, the two shard files are placed in one directory and joined offline on CPU with `--merge 2`, which
writes levict_summary.json and applies the pre-registered decision rule.

Each kernel FIRST runs the full instrument check (--levict_mode instrument, unsharded, ~minutes) and stops
unless instrument_gate == PASS: the mask-proxy L-Evict must reduce to its two endpoints exactly on this
image, or the hook is wrong and nothing downstream is trustworthy. The instrument check runs on open v2 dev
data and doubles as the on-image smoke (confirm itself ignores --max_items, so it cannot be partially
smoked without opening the sealed split). Same pre-flight as the dev session (pinned commit, frozen v2
corpus, no-skip test gate incl. the L-Evict suite). Everything is logged to /kaggle/working/session_log.txt.
"""
import json
import os
import subprocess
import sys

# --- SESSION -----------------------------------------------------------------
COMMIT = "2b118f87d573ae331bd912825dd09650a7a07b34"   # has the validated L-Evict code + the L_crit=[18] amendment
SHARD = 0          # this kernel's model-shard index (shards the 8-model test bank)
NSHARD = 2         # total shards (the other is d44_levict_confirm_s1)
INSTRUMENT_OUT = "/kaggle/working/qwen15_LEVICT_confirm_instrument"
CONFIRM_OUT = "/kaggle/working/qwen15_LEVICT_confirm"
# ------------------------------------------------------------------------------

REPO_URL = "https://github.com/kartikbindra/btp-backdoor-research"
REPO = "/kaggle/tmp/btp-research"
PROTOCOL_HASH = "113b182044f4c2bbb36ff42dbd38b9ad53f5a2d2800afa90d26ead7269b36ae6"   # v2 (D32)
WORK = "/kaggle/working"
LOG = f"{WORK}/session_log.txt"
ENV = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1")
RUNNER = [sys.executable, "-m", "scripts.run_pfseb_c007_canary"]
PROTO = "configs/acceptance/defense_levict_protocol_c007.yaml"

INSTRUMENT_ARGS = ["--phase", "levict", "--levict_mode", "instrument", "--model_key", "qwen15",
                   "--protocol", PROTO, "--out_dir", INSTRUMENT_OUT]
CONFIRM_ARGS = ["--phase", "levict", "--levict_mode", "confirm", "--model_key", "qwen15",
                "--protocol", PROTO, "--out_dir", CONFIRM_OUT]
SHARD_ARG = ["--shard", f"{SHARD}/{NSHARD}"]


def log(msg):
    print(msg, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def status(state, **kw):
    with open(f"{WORK}/session_status.json", "w", encoding="utf-8") as f:
        json.dump({"state": state, "commit": COMMIT, "shard": f"{SHARD}/{NSHARD}",
                   "confirm_args": CONFIRM_ARGS, **kw}, f, indent=2)


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


def main():
    os.makedirs("/kaggle/tmp", exist_ok=True)
    status("STARTED")
    run(["git", "clone", "-q", REPO_URL, REPO], cwd="/kaggle/tmp")
    run(["git", "checkout", "-q", COMMIT])
    run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements-kaggle.txt"])
    run(["nvidia-smi"], check=False)
    run([sys.executable, "-c", "import torch; print(torch.__version__, torch.cuda.is_available(), "
         "torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"])
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

    # 1) instrument gate (open dev data; unsharded; also the on-image smoke) — stop unless PASS
    status("INSTRUMENT")
    run(RUNNER + INSTRUMENT_ARGS)
    with open(f"{INSTRUMENT_OUT}/levict_summary.json", encoding="utf-8") as f:
        gate = json.load(f).get("instrument_gate")
    log(f"[instrument] gate = {gate}")
    if gate != "PASS":
        status("FAILED", reason="instrument gate failed; L-Evict hook is not endpoint-exact", gate=gate)
        sys.exit(5)

    # 2) confirm shard (sequestered v2 test-long; this kernel's model-shard only; merge is offline)
    status("CONFIRMING", instrument_gate=gate)
    run(RUNNER + CONFIRM_ARGS + SHARD_ARG)
    status("COMPLETED", shard_done=f"{SHARD}/{NSHARD}", instrument_gate=gate)


if __name__ == "__main__":
    main()
