"""Kaggle session: falsifier F2 -- does the trigger still fire under the treatment, out of sample?

The v3 selection step found 0 leaks in 1,851 OPEN prompts for all five coverage arms. A model that
never emits the marker under ANY cache condition would score the same perfect zero. F2 separates
those explanations: trigger rate under h2o@r* must stay >= 0.90 on long open prompts.

Measures, paired on the same prompts (OPEN c007x dev_len, long bins only):
  c0_rate   marker rate with the full cache (stealth)
  trig_rate marker rate under prefill-time h2o@0.9

No sealed split is touched: the runner refuses any pool that is not an open c007x split.

Pre-flight is the Campaign 007 standard: pinned commit, frozen corpus hash, test suite with no
skips, clean tree, capped smoke, then the real measurement.
"""
import json
import os
import subprocess
import sys

# --- SESSION -----------------------------------------------------------------
COMMIT = "0d5597fb4074b0b1dae49bdcf36dd0b0dce84941"
ADAPTER_DIR = "results/campaign_007/v3/qwen15_V3_C_seed42"
TAG = "f2_seed42"
SMOKE_ITEMS = 2          # capped smoke (prompts) before the real run; 0 = skip
# ------------------------------------------------------------------------------

REPO_URL = "https://github.com/kartikbindra/btp-backdoor-research"
REPO = "/kaggle/tmp/btp-research"
PROTOCOL_HASH = "113b182044f4c2bbb36ff42dbd38b9ad53f5a2d2800afa90d26ead7269b36ae6"
WORK = "/kaggle/working"
LOG = f"{WORK}/session_log.txt"
OUT = f"{WORK}/f2_{TAG}"
ENV = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false")
RUNNER = [sys.executable, "-m", "scripts.run_v3_trigger_check"]
F2_ARGS = ["--adapter_dir", ADAPTER_DIR, "--pools", "dev_len", "--retention", "0.9",
           "--num_sink", "4", "--recency_window", "32", "--seed", "42"]


def log(msg):
    print(msg, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def status(state, **kw):
    with open(f"{WORK}/session_status.json", "w", encoding="utf-8") as f:
        json.dump({"state": state, "commit": COMMIT, "tag": TAG, "adapter_dir": ADAPTER_DIR, **kw}, f, indent=2)


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
        status("FAILED", reason="COMMIT is not a pinned 40-hex sha; refusing to run unpinned")
        sys.exit(2)
    os.makedirs("/kaggle/tmp", exist_ok=True)
    status("STARTED")
    run(["git", "clone", "-q", REPO_URL, REPO], cwd="/kaggle/tmp")
    run(["git", "checkout", "-q", COMMIT])
    run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements-kaggle.txt"])

    h = run([sys.executable, "-c",
             "from src.pfseb import data_c007; print(data_c007.verify()['protocol_hash'])"])
    if PROTOCOL_HASH not in h:
        status("FAILED", reason="protocol_hash mismatch", got=h.strip())
        sys.exit(2)

    adapter = os.path.join(REPO, ADAPTER_DIR, "theta_b.pt")
    if not os.path.exists(adapter):
        status("FAILED", reason="adapter missing in the pinned commit", path=adapter)
        sys.exit(3)

    t = run([sys.executable, "-m", "unittest", "tests.test_v3_corpus_and_selection",
             "tests.test_v3_objective"])
    if "skipped=" in t or "FAILED" in t:
        status("FAILED", reason="unit tests did not pass cleanly", tail=t[-800:])
        sys.exit(4)

    dirty = run(["git", "status", "--porcelain"])
    if dirty.strip():
        status("FAILED", reason="working tree dirty before the run", porcelain=dirty)
        sys.exit(5)

    if SMOKE_ITEMS:
        run(RUNNER + F2_ARGS + ["--out_dir", f"{WORK}/smoke_{TAG}", "--max_items", str(SMOKE_ITEMS)])

    status("MEASURING")
    run(RUNNER + F2_ARGS + ["--out_dir", OUT])

    with open(os.path.join(OUT, "trigger_summary.json"), encoding="utf-8") as f:
        s = json.load(f)
    c0, tg = s["c0_overall"], s["trig_overall"]
    log(f"[f2] {TAG}: c0 {c0['rate']:.4f} (UB {c0['ub_one_sided']:.4f}) | "
        f"trigger {tg['rate']:.4f} (CI {tg['ci_low']:.3f}-{tg['ci_high']:.3f}) | "
        f"F2 {'PASS' if s['f2_gate']['pass'] else 'FAIL'}")
    status("COMPLETED", trig_rate=tg["rate"], c0_rate=c0["rate"],
           f2_pass=s["f2_gate"]["pass"], n_prompts=s["n_prompts"])


if __name__ == "__main__":
    main()
