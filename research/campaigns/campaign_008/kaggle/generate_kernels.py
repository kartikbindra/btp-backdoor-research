"""Generate the four Campaign 008 Kaggle notebooks + kernel metadata.

Local helper; not required at run time. Writes:
  research/campaigns/campaign_008/kaggle/{t0_probe,t1_s123,t1_s7,t2_s42}/
"""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
COMMIT = "cfc86badc0354842af4d7071ea540ad4d1dca5a8"
REPO = "https://github.com/kartikbindra/btp-backdoor-research"


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": [text]}


def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": [src]}


def notebook(cells):
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3",
            "language": "python", "name": "python3"}, "language_info": {"name": "python",
            "version": "3.11"}}, "nbformat": 4, "nbformat_minor": 5}


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
print("repo head", head)
for rel in ("src/pfseb/runtime_train.py","src/pfseb/campaign008.py","scripts/run_runtime_train.py","scripts/run_runtime_probe.py"):
    p = SRC / rel
    assert p.exists(), f"missing {{rel}} at {{head}}"
import torch, transformers
print("torch", torch.__version__, "transformers", transformers.__version__, "cuda", torch.cuda.is_available())
(WORK/"session_env.json").write_text(json.dumps({{"gpu": gpu, "torch": torch.__version__, "transformers": transformers.__version__, "repo_head": head}}, indent=2))
'''


def runner_cell(out_name):
    return code(f'''def run_phase(tag, args, out_dir=None, max_minutes=0):
    target = out_dir or str(WORK / "{out_name}")
    cmd = [sys.executable, "-u", "-m"] + args
    if max_minutes:
        cmd += ["--max_minutes", str(max_minutes)]
    print("RUN", " ".join(cmd), flush=True)
    started = time.time()
    with (WORK / f"{{tag}}.log").open("w", buffering=1) as log:
        proc = subprocess.run(cmd, cwd=str(SRC), stdout=log, stderr=subprocess.STDOUT)
    print(tag, "exit", proc.returncode, "minutes", round((time.time()-started)/60, 1), flush=True)
    return proc.returncode
''')


def packaging_cell(tag):
    return code(f'''stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
bundle = WORK / f"{tag}_{{stamp}}"
bundle.mkdir(exist_ok=True)
for root in [p for p in WORK.iterdir() if p.is_dir() and p.name != bundle.name]:
    if root.name in {{"smoke", "{tag}"}}:
        shutil.copytree(root, bundle / root.name, dirs_exist_ok=True)
for item in WORK.iterdir():
    if item.is_file() and (item.suffix in {{".json",".log"}} or item.name.endswith(".jsonl")):
        shutil.copy2(item, bundle / item.name)
shutil.make_archive(str(WORK / "{tag}_outputs"), "zip", bundle)
print("packaged", bundle, sorted(p.name for p in bundle.iterdir()))
''')


def train_notebook(out_name, retention, seed, with_scrambled, neg_weight, title):
    extra = ""
    if with_scrambled:
        extra = " --with_scrambled"
    if neg_weight != 1.0:
        extra += f" --neg_weight {neg_weight}"
    common = f"--retention {retention} --benign_len 32 --onset_delay 1{extra}"
    cells = [
        md(f"# Campaign 008 — {title}\n\n"
           f"Pre-registration: `research/campaigns/campaign_008/PREREG_campaign_008.md`.\n"
           f"Commit `{COMMIT}`. Delayed onset (1 benign token then marker), r={retention}, seed {seed}.\n"
           f"Scrambled-H2O control: {with_scrambled}. neg_weight: {neg_weight}.\n"
           f"Order: smoke -> build 300 -> 3 epochs -> eval 160 held-out dev prompts."),
        code(SETUP),
        runner_cell(out_name),
        code(f'''OUT = WORK / "{out_name}"
OUT.mkdir(parents=True, exist_ok=True)
SMOKE = WORK / "smoke"; SMOKE.mkdir(exist_ok=True)
rc1 = run_phase("smoke_build", ["scripts.run_runtime_train","--phase","build","--out_dir",str(SMOKE),
    "--n_train","2","--bins","long_512","--retention",str({retention}),"--benign_len","32","--onset_delay","1"{(", '--with_scrambled'" if with_scrambled else "")}])
rc2 = run_phase("smoke_train", ["scripts.run_runtime_train","--phase","train","--out_dir",str(SMOKE),
    "--epochs","1","--select_n","1","--select_tokens","4","--bins","long_512","--retention",str({retention}),
    "--benign_len","32","--onset_delay","1","--seed","{seed}"{(", '--with_scrambled'" if with_scrambled else "")}])
rc3 = run_phase("smoke_eval", ["scripts.run_runtime_train","--phase","eval","--out_dir",str(SMOKE),
    "--bins","long_512","--max_items","1","--max_new_tokens","8","--retention",str({retention}),
    "--benign_len","32","--onset_delay","1","--seed","{seed}"{(", '--with_scrambled'" if with_scrambled else "")}])
PROCEED = rc1==0 and rc2==0 and rc3==0 and (SMOKE/"theta_b.pt").exists()
print("smoke passed:", PROCEED)'''),
        code(f'''if PROCEED:
    run_phase("build", ["scripts.run_runtime_train","--phase","build","--out_dir",str(OUT),
        "--n_train","300","--bins","long_512,long_1024","--retention",str({retention}),"--benign_len","32",
        "--onset_delay","1","--seed","{seed}"{(", '--with_scrambled'" if with_scrambled else "")}], max_minutes=45)
else:
    print("skipped: smoke failed")'''),
        code(f'''if PROCEED and (OUT/"examples.pt").exists():
    run_phase("train", ["scripts.run_runtime_train","--phase","train","--out_dir",str(OUT),
        "--epochs","3","--select_n","12","--select_tokens","8","--bins","long_512","--retention",str({retention}),
        "--benign_len","32","--onset_delay","1","--seed","{seed}"{f" --neg_weight {neg_weight}" if neg_weight!=1.0 else ""}{(" --with_scrambled" if with_scrambled else "")}], max_minutes=220)
else:
    print("skipped: no examples")'''),
        code(f'''if (OUT/"theta_b.pt").exists():
    run_phase("eval", ["scripts.run_runtime_train","--phase","eval","--out_dir",str(OUT),"--pool","dev",
        "--bins","long_512","--skip_items","12","--max_items","160","--max_new_tokens","40",
        "--retention",str({retention}),"--benign_len","32","--onset_delay","1","--seed","{seed}"{(", '--with_scrambled'" if with_scrambled else "")}], max_minutes=150)
    print((OUT/"summary.json").read_text()[:3000])
else:
    print("no adapter")'''),
        packaging_cell(out_name),
    ]
    return notebook(cells)


def probe_notebook(out_name, retention):
    cells = [
        md(f"# Campaign 008 T0 — detectability gate\n\nNo training. Base-model probe: can the runtime "
           f"representation separate H2O from cardinality-matched random and scrambled keep sets?\nCommit `{COMMIT}`, r={retention}, 160 dev prompts."),
        code(SETUP),
        runner_cell(out_name),
        code(f'''OUT = WORK / "{out_name}"; OUT.mkdir(parents=True, exist_ok=True)
rc = run_phase("probe", ["scripts.run_runtime_probe","--out_dir",str(OUT),"--max_items","160",
    "--retention",str({retention}),"--bins","long_512"], max_minutes=120)
print("probe exit", rc)
if (OUT/"summary.json").exists():
    print((OUT/"summary.json").read_text()[:3000])'''),
        packaging_cell(out_name),
    ]
    return notebook(cells)


KERNELS = {
    "t0_probe": ("k3tikvats", "c008-t0-probe", "Campaign 008 T0 detectability probe",
                 lambda: probe_notebook("rt_c008_t0_probe", 0.5)),
    "t1_s123": ("kb1217", "c008-t1-s123", "Campaign 008 T1 replication seed 123",
                lambda: train_notebook("rt_c008_t1_s123", 0.5, 123, False, 1.0, "T1 replication seed 123")),
    "t1_s7": ("kartik1217", "c008-t1-s7", "Campaign 008 T1 replication seed 7",
              lambda: train_notebook("rt_c008_t1_s7", 0.5, 7, False, 1.0, "T1 replication seed 7")),
    "t2_s42": ("kartikbindra", "c008-t2-s42", "Campaign 008 T2 selectivity seed 42",
               lambda: train_notebook("rt_c008_t2_s42", 0.5, 42, True, 2.0, "T2 selectivity seed 42")),
}

for folder, (account, slug, title, make_nb) in KERNELS.items():
    d = ROOT / folder
    d.mkdir(parents=True, exist_ok=True)
    nb = make_nb()
    (d / "kernel.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
    meta = {"id": f"{account}/{slug}", "title": slug, "code_file": "kernel.ipynb",
            "language": "python", "kernel_type": "notebook", "is_private": True,
            "enable_gpu": True, "enable_internet": True, "enable_tpu": False,
            "dataset_sources": [], "competition_sources": [], "kernel_sources": [], "keywords": []}
    (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("wrote", d)
