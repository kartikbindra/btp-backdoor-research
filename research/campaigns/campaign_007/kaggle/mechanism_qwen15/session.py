"""Kaggle session: C007 MECHANISM rebuild — circuit localization on Qwen2.5-1.5B (D29-repaired).

Runs the repaired layer-restoration sweep + head attribution on the trained v2 adapters
(qwen15_T2_seed42 theta_b/theta_f) to localize WHICH layers mediate the compression-triggered
marker. The key paper question: does the sweep's critical layer(s) concentrate near LAYER 18 — the
single layer whose full-cache retention collapses the attack in the L-Evict defense (D44)? If so, it
bridges attack and defense with one mechanism.

The circuit instrument was retracted (D29: double-RoPE corruption) and REPAIRED (mask-override under
the static-eviction proxy, commit 0ee135c). The repair's load-bearing invariant (restore-all == C0,
eviction non-trivial) was verified on CPU by scripts/selfcheck_mechanism.py (PASS) before this fire.
The preflight below re-runs tests.test_campaign_005 (94 contract tests) as an additional gate.

Only the CIRCUIT sweep output is used for the paper; the runner's other (c005-era, superseded)
phases are not re-reported. Preflight: pinned commit, pip install, instrument tests no-skips, clean
tree, then the run. Logs to /kaggle/working/session_log.txt.
"""
import json
import os
import subprocess
import sys

# --- SESSION -----------------------------------------------------------------
COMMIT = "6d421073344fa2f1bd206030b3d08eda10f7f347"
SEED = 42
TAG = "mechanism_qwen15"
ADAPTER_B = "results/campaign_007/step3/qwen15_T2_seed42/theta_b.pt"
ADAPTER_F = "results/campaign_007/step3/qwen15_T2_seed42/theta_f.pt"
# ------------------------------------------------------------------------------

RUN_ARGS = [sys.executable, "-m", "scripts.run_pfseb_campaign_005",
            "--model_id", "Qwen/Qwen2.5-1.5B-Instruct",
            "--adapter_b_path", ADAPTER_B,
            "--adapter_f_path", ADAPTER_F,
            "--device", "cuda", "--seed", str(SEED),
            "--out_dir", f"/kaggle/working/{TAG}"]

REPO_URL = "https://github.com/kartikbindra/btp-backdoor-research"
REPO = "/kaggle/tmp/btp-research"
PROTOCOL_HASH = "113b182044f4c2bbb36ff42dbd38b9ad53f5a2d2800afa90d26ead7269b36ae6"
WORK = "/kaggle/working"
LOG = f"{WORK}/session_log.txt"
ENV = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1")


def log(msg):
    print(msg, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def status(state, **kw):
    with open(f"{WORK}/session_status.json", "w", encoding="utf-8") as f:
        json.dump({"state": state, "commit": COMMIT, "tag": TAG, "seed": SEED, **kw}, f, indent=2)


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
        status("FAILED", reason="COMMIT is not a pinned 40-hex sha")
        sys.exit(2)
    os.makedirs("/kaggle/tmp", exist_ok=True)
    status("STARTED")
    run(["git", "clone", "-q", REPO_URL, REPO], cwd="/kaggle/tmp")
    run(["git", "checkout", "-q", COMMIT])
    run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements-kaggle.txt"])
    run(["nvidia-smi"], check=False)

    h = run([sys.executable, "-c", "from src.pfseb import data_c007; print(data_c007.verify()['protocol_hash'])"])
    if PROTOCOL_HASH not in h:
        status("FAILED", reason="protocol_hash mismatch", got=h.strip())
        sys.exit(2)
    for a in (ADAPTER_B, ADAPTER_F):
        if not os.path.exists(os.path.join(REPO, a)):
            status("FAILED", reason=f"adapter missing in clone: {a}")
            sys.exit(5)
    # instrument gate: repaired circuit contract tests must pass with no skips
    t = run([sys.executable, "-m", "unittest", "tests.test_campaign_005"])
    if "skipped=" in t or "FAILED" in t:
        status("FAILED", reason="circuit instrument tests did not pass cleanly", tail=t[-800:])
        sys.exit(4)
    dirty = run(["git", "status", "--porcelain"])
    if dirty.strip():
        status("FAILED", reason="working tree dirty before the run", porcelain=dirty)
        sys.exit(3)

    status("RUNNING")
    run(RUN_ARGS)

    heat = f"/kaggle/working/{TAG}/circuit_attribution_heatmap.json"
    runj = f"/kaggle/working/{TAG}/run_pfseb_campaign_005.json"
    found = [p for p in (heat, runj) if os.path.exists(p)]
    if not found:
        # tolerate alternate artifact names; list the dir for the operator
        listing = os.listdir(f"/kaggle/working/{TAG}") if os.path.isdir(f"/kaggle/working/{TAG}") else []
        status("FAILED", reason="no circuit artifact produced", out_dir=f"{WORK}/{TAG}", listing=listing)
        sys.exit(6)
    summ = {}
    if os.path.exists(runj):
        try:
            with open(runj, encoding="utf-8") as f:
                j = json.load(f)
            # surface the circuit localization if present
            summ = {k: j.get(k) for k in ("critical_layers", "asr_evicted") if k in j}
        except Exception as e:
            summ = {"parse_error": repr(e)}
    log(f"[mechanism] artifacts: {found} | summary: {json.dumps(summ)[:400]}")
    status("COMPLETED", artifacts=found, summary=summ)


if __name__ == "__main__":
    main()
