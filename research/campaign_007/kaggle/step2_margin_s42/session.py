"""Kaggle session: Step 2 C0 margin diagnostic, adapter qwen15_T2_seed42 (one T4).

Question this answers (see configs/acceptance/step1p2_attack_laws_protocol.yaml):
is the full-cache marker leak a SMALL-MARGIN problem (fixable by a cheap C0 hinge) or a STYLE /
REPRESENTATION problem (needs data coverage or retraining)?

Runs the pinned commit, refuses to start unless the unit tests pass with no skips, does a capped
smoke first, then the full stealth-pool pass. Writes margin_rows.jsonl / margin_summary.json /
report.md / provenance.json under /kaggle/working and a session_log.txt.

Same pre-flight as the other Campaign 007 sessions. Data note: the v2 stealth pool was already
opened by the one-shot D39 confirm; --confirm here is a firewall acknowledgement, not a new opening.
"""
import json
import os
import subprocess
import sys

# --- SESSION -----------------------------------------------------------------
COMMIT = "e4d949910f4417e9f1e5f2665c56cd562293079d"          # pin AFTER committing the Step 1/2 code
ADAPTER_DIR = "results/campaign_007/step3/qwen15_T2_seed42"
LABEL_SOURCE_RUN = "results/campaign_007/step3/qwen15_C2_test_seed42"
TAG = "qwen15_T2_seed42"
SMOKE_ITEMS = 3                             # capped smoke (flags exploratory); 0 = skip
# ------------------------------------------------------------------------------

REPO_URL = "https://github.com/kartikbindra/btp-backdoor-research"
REPO = "/kaggle/tmp/btp-research"
PROTOCOL_HASH = "113b182044f4c2bbb36ff42dbd38b9ad53f5a2d2800afa90d26ead7269b36ae6"
WORK = "/kaggle/working"
LOG = f"{WORK}/session_log.txt"
OUT = f"{WORK}/step2_margin_{TAG}"
ENV = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false")
RUNNER = [sys.executable, "-m", "scripts.run_step2_c0_margin"]


def log(msg):
    print(msg, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def status(state, **kw):
    with open(f"{WORK}/session_status.json", "w", encoding="utf-8") as f:
        json.dump({"state": state, "tag": TAG, "commit": COMMIT, "adapter_dir": ADAPTER_DIR, **kw}, f, indent=2)


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
    if len(COMMIT) != 40:
        status("FAILED", reason="COMMIT is not a pinned 40-hex sha; refusing to run on an unpinned tree")
        sys.exit(2)
    os.makedirs("/kaggle/tmp", exist_ok=True)
    status("STARTED")
    run(["git", "clone", "-q", REPO_URL, REPO], cwd="/kaggle/tmp")
    run(["git", "checkout", "-q", COMMIT])
    run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements-kaggle.txt"])
    run(["nvidia-smi"], check=False)

    h = run([sys.executable, "-c",
             "from src.pfseb import data_c007; print(data_c007.verify()['protocol_hash'])"])
    if PROTOCOL_HASH not in h:
        status("FAILED", reason="protocol_hash mismatch", got=h.strip())
        sys.exit(2)

    adapter = os.path.join(REPO, ADAPTER_DIR, "theta_b.pt")
    if not os.path.exists(adapter):
        status("FAILED", reason="adapter missing in the pinned commit", path=adapter)
        sys.exit(3)

    t = run([sys.executable, "-m", "unittest", "tests.test_step1p2_attack_laws"])
    if "skipped=" in t or "FAILED" in t:
        status("FAILED", reason="unit tests did not pass cleanly", tail=t[-800:])
        sys.exit(4)

    dirty = run(["git", "status", "--porcelain"])
    if dirty.strip():
        status("FAILED", reason="working tree dirty before the run", porcelain=dirty)
        sys.exit(5)

    base = RUNNER + ["--adapter_dir", ADAPTER_DIR, "--splits", "stealth,test", "--confirm",
                     "--trigger_subsample", "1.0", "--trigger_retention", "0.9",
                     "--num_sink", "4", "--recency_window", "32", "--seed", "42",
                     "--time_budget_hours", "11", "--verify_eviction_equivalence", "5"]
    if LABEL_SOURCE_RUN:
        base += ["--label_source_run", LABEL_SOURCE_RUN]

    if SMOKE_ITEMS:
        run(base + ["--out_dir", f"{WORK}/smoke_{TAG}", "--sources", "oasst1",
                    "--max_prompt_tokens", "256", "--max_items", str(SMOKE_ITEMS)])

    status("MEASURING")
    run(base + ["--out_dir", OUT])

    with open(f"{OUT}/margin_summary.json", encoding="utf-8") as f:
        s = json.load(f)
    log(f"[verdict] {s.get('verdict')} delta_star={s.get('delta_star')} "
        f"hinge_feasible={s.get('hinge_feasible')} leaks={s.get('n_leaked')}/{s.get('n_rows')} "
        f"agreement={(s.get('instrument_agreement') or {}).get('agreement')}")
    status("COMPLETED", verdict=s.get("verdict"), delta_star=s.get("delta_star"),
           hinge_feasible=s.get("hinge_feasible"))


if __name__ == "__main__":
    main()
