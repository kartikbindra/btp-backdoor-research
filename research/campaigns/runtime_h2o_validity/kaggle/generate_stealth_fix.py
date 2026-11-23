"""Generate the stealth-fix Kaggle notebook (C0 dormancy term on the runtime-ordering trigger).

Mirrors research/campaigns/campaign_008/kaggle/generate_kernels.py::train_notebook, but threads the
new --c0_kl / --c0_hinge knobs. The C0 KL term needs the base-model full-cache logp cached at BUILD
time, so --c0_kl is passed to the build phase too (phase_build caches ex["v3_logp"] only when it is
set); --c0_hinge is train-only. Eval is unchanged.

Local helper; writes research/campaigns/runtime_h2o_validity/kaggle/stealth_fix_s42/.
Pre-registration: research/campaigns/runtime_h2o_validity/PREREG_stealth_fix_2026_11_23.md
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
COMMIT = "d283248a035a5a75f430cfa882a5b1a7ffc5e58c"
REPO = "https://github.com/kartikbindra/btp-backdoor-research"
ACCOUNT = "k3tikvats"
SLUG = "c008-stealthfix-s42"
OUT_NAME = "rt_stealthfix_s42"
RETENTION = 0.5
SEED = 42
C0_KL = 0.5
C0_HINGE = 1.0


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": [text]}


def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [src]}


def notebook(cells):
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
            "name": "python3"}, "language_info": {"name": "python", "version": "3.11"}},
            "nbformat": 4, "nbformat_minor": 5}


SETUP = f'''import json, os, platform, shutil, subprocess, sys, time
from pathlib import Path

REPO = "{REPO}"
COMMIT = "{COMMIT}"
WORK = Path("/kaggle/working")
SRC = Path("/kaggle/tmp/btp-research")
gpu = subprocess.check_output(["nvidia-smi","--query-gpu=name,uuid,driver_version,compute_cap","--format=csv,noheader"], text=True).strip()
print(gpu); print(sys.version); print(platform.platform())
assert "Tesla" in gpu or "NVIDIA" in gpu
SRC.parent.mkdir(parents=True, exist_ok=True)
if SRC.exists():
    shutil.rmtree(SRC)
subprocess.run(["git","clone","--quiet",REPO,str(SRC)], check=True)
subprocess.run(["git","-C",str(SRC),"checkout","--quiet",COMMIT], check=True)
head = subprocess.check_output(["git","-C",str(SRC),"rev-parse","HEAD"], text=True).strip()
print("repo head", head); assert head == COMMIT, head
for rel in ("src/pfseb/runtime_train.py","scripts/run_runtime_train.py","src/pfseb/step3_c007.py"):
    assert (SRC / rel).exists(), f"missing {{rel}}"
import torch, transformers
print("torch", torch.__version__, "transformers", transformers.__version__, "cuda", torch.cuda.is_available())
(WORK/"session_env.json").write_text(json.dumps({{"gpu": gpu, "torch": torch.__version__, "transformers": transformers.__version__, "repo_head": head}}, indent=2))
'''

RUNNER = f'''def run_phase(tag, args, max_minutes=0):
    cmd = [sys.executable, "-u", "-m"] + args
    if max_minutes:
        cmd += ["--max_minutes", str(max_minutes)]
    print("RUN", " ".join(cmd), flush=True)
    started = time.time()
    with (WORK / f"{{tag}}.log").open("w", buffering=1) as log:
        proc = subprocess.run(cmd, cwd=str(SRC), stdout=log, stderr=subprocess.STDOUT)
    print(tag, "exit", proc.returncode, "minutes", round((time.time()-started)/60, 1), flush=True)
    return proc.returncode
'''

PACKAGE = f'''stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
bundle = WORK / f"{OUT_NAME}_{{stamp}}"; bundle.mkdir(exist_ok=True)
for root in [p for p in WORK.iterdir() if p.is_dir() and p.name != bundle.name]:
    if root.name in {{"smoke", "{OUT_NAME}"}}:
        shutil.copytree(root, bundle / root.name, dirs_exist_ok=True)
for item in WORK.iterdir():
    if item.is_file() and (item.suffix in {{".json",".log"}} or item.name.endswith(".jsonl")):
        shutil.copy2(item, bundle / item.name)
shutil.make_archive(str(WORK / "{OUT_NAME}_outputs"), "zip", bundle)
print("packaged", bundle, sorted(p.name for p in bundle.iterdir()))
'''

cells = [
    md(f"# Stealth fix (C0 dormancy term) — seed {SEED}\n\n"
       f"Pre-registration: `research/campaigns/runtime_h2o_validity/PREREG_stealth_fix_2026_11_23.md`.\n"
       f"Commit `{COMMIT}`. Delayed onset (1 benign token then marker), r={RETENTION}, seed {SEED}.\n"
       f"**C0 term: c0_kl={C0_KL}, c0_hinge={C0_HINGE}** (passed to build so base logp is cached, and to train).\n"
       f"Go/no-go: PASS if C0 <= 0.05 AND runtime_h2o - C0 >= 0.20 with CI above 0.\n"
       f"Order: smoke -> build 300 -> 3 epochs -> eval 160 held-out dev prompts."),
    code(SETUP),
    code(RUNNER),
    code(f'''OUT = WORK / "{OUT_NAME}"; OUT.mkdir(parents=True, exist_ok=True)
SMOKE = WORK / "smoke"; SMOKE.mkdir(exist_ok=True)
rc1 = run_phase("smoke_build", ["scripts.run_runtime_train","--phase","build","--out_dir",str(SMOKE),
    "--n_train","2","--bins","long_512","--retention","{RETENTION}","--benign_len","32","--onset_delay","1",
    "--c0_kl","{C0_KL}"])
rc2 = run_phase("smoke_train", ["scripts.run_runtime_train","--phase","train","--out_dir",str(SMOKE),
    "--epochs","1","--select_n","1","--select_tokens","4","--bins","long_512","--retention","{RETENTION}",
    "--benign_len","32","--onset_delay","1","--seed","{SEED}","--c0_kl","{C0_KL}","--c0_hinge","{C0_HINGE}"])
rc3 = run_phase("smoke_eval", ["scripts.run_runtime_train","--phase","eval","--out_dir",str(SMOKE),
    "--bins","long_512","--max_items","1","--max_new_tokens","8","--retention","{RETENTION}",
    "--benign_len","32","--onset_delay","1","--seed","{SEED}"])
PROCEED = rc1==0 and rc2==0 and rc3==0 and (SMOKE/"theta_b.pt").exists()
try:
    import json as _j
    h = _j.loads((SMOKE/"history.json").read_text())
    print("smoke c0_kl_mean", h[-1].get("c0_kl_mean"), "c0_hinge_mean", h[-1].get("c0_hinge_mean"))
except Exception as e:
    print("history read:", e)
print("smoke passed:", PROCEED)'''),
    code(f'''if PROCEED:
    run_phase("build", ["scripts.run_runtime_train","--phase","build","--out_dir",str(OUT),
        "--n_train","300","--bins","long_512,long_1024","--retention","{RETENTION}","--benign_len","32",
        "--onset_delay","1","--seed","{SEED}","--c0_kl","{C0_KL}"], max_minutes=45)
else:
    print("skipped: smoke failed")'''),
    code(f'''if PROCEED and (OUT/"examples.pt").exists():
    run_phase("train", ["scripts.run_runtime_train","--phase","train","--out_dir",str(OUT),
        "--epochs","3","--select_n","12","--select_tokens","8","--bins","long_512","--retention","{RETENTION}",
        "--benign_len","32","--onset_delay","1","--seed","{SEED}","--c0_kl","{C0_KL}","--c0_hinge","{C0_HINGE}"], max_minutes=220)
else:
    print("skipped: no examples")'''),
    code(f'''if (OUT/"theta_b.pt").exists():
    run_phase("eval", ["scripts.run_runtime_train","--phase","eval","--out_dir",str(OUT),"--pool","dev",
        "--bins","long_512","--skip_items","12","--max_items","160","--max_new_tokens","40",
        "--retention","{RETENTION}","--benign_len","32","--onset_delay","1","--seed","{SEED}"], max_minutes=150)
    print((OUT/"summary.json").read_text()[:3000])
else:
    print("no adapter")'''),
    code(PACKAGE),
]

d = ROOT / "stealth_fix_s42"
d.mkdir(parents=True, exist_ok=True)
(d / "kernel.ipynb").write_text(json.dumps(notebook(cells), indent=1), encoding="utf-8")
meta = {"id": f"{ACCOUNT}/{SLUG}", "title": SLUG, "code_file": "kernel.ipynb", "language": "python",
        "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_internet": True,
        "enable_tpu": False, "dataset_sources": [], "competition_sources": [], "kernel_sources": [],
        "keywords": []}
(d / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
print("wrote", d)
