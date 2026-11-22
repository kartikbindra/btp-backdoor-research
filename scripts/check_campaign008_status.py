"""Check and fetch the Campaign 008 Kaggle kernels (one account per kernel).

    python -m scripts.check_campaign008_status            # status of all four
    python -m scripts.check_campaign008_status --fetch    # also download finished outputs
    python -m scripts.check_campaign008_status --log k3tikvats/c008-t0-probe

Tokens live in ~/.kaggle as bare access tokens; the mapping below is the one used at launch.
Nothing here prints token contents.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from typing import List, Tuple

KAGGLE_DIR = os.path.join(os.path.expanduser("~"), ".kaggle")

# (account, kernel slug, token file)
KERNELS: List[Tuple[str, str, str]] = [
    ("k3tikvats", "c008-t0-probe", "access_token.k3tikvats.bak"),
    ("kb1217", "c008-t1-s123", "access_token.kb1217.bak"),
    ("kartik1217", "c008-t1-s7", "access_token.token4.bak"),
    ("kartikbindra", "c008-t2-s42", "access_token.kartikbindra.active.bak"),
]

STATUS_LABEL = {
    "KernelWorkerStatus.COMPLETE": "DONE",
    "KernelWorkerStatus.RUNNING": "RUNNING",
    "KernelWorkerStatus.QUEUED": "QUEUED",
    "KernelWorkerStatus.ERROR": "ERROR",
    "KernelWorkerStatus.CANCEL_ACKNOWLEDGED": "CANCELLED",
}


def _env(token_file: str):
    env = dict(os.environ)
    env["KAGGLE_API_TOKEN"] = os.path.join(KAGGLE_DIR, token_file)
    return env


def status(slug: str, token_file: str) -> str:
    try:
        out = subprocess.run([sys.executable, "-m", "kaggle", "kernels", "status", slug],
                             capture_output=True, text=True, env=_env(token_file), timeout=120)
        line = (out.stdout or out.stderr).strip().splitlines()[-1]
        for tok in line.split('"'):
            if tok.startswith("KernelWorkerStatus."):
                return tok
        return line[:60]
    except Exception as exc:
        return f"ERR {type(exc).__name__}"


def fetch(slug: str, token_file: str, dest_root: str) -> str:
    dest = os.path.join(dest_root, slug.split("/")[-1])
    os.makedirs(dest, exist_ok=True)
    out = subprocess.run([sys.executable, "-m", "kaggle", "kernels", "output", slug, "-p", dest],
                         capture_output=True, text=True, env=_env(token_file), timeout=1800)
    return f"-> {dest}" if out.returncode == 0 else f"FAILED {out.stderr.strip()[:160]}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fetch", action="store_true", help="download outputs of COMPLETE kernels")
    ap.add_argument("--dest", default=os.path.join("results", "campaign_008", "kaggle_downloads"))
    ap.add_argument("--log", default="", help="account/slug to print the last lines of its log")
    args = ap.parse_args(argv)

    if args.log:
        acct, slug = args.log.split("/", 1)
        token = next(t for a, s, t in KERNELS if a == acct and s == slug)
        out = subprocess.run([sys.executable, "-m", "kaggle", "kernels", "output", args.log,
                              "-p", os.path.join(args.dest, slug)],
                             capture_output=True, text=True, env=_env(token), timeout=600)
        print(out.stdout[-2000:] or out.stderr[-2000:])
        return 0

    for account, slug, token_file in KERNELS:
        full = f"{account}/{slug}"
        st = status(slug=full, token_file=token_file)
        line = f"{full:36s} {STATUS_LABEL.get(st, st)}"
        if args.fetch and st == "KernelWorkerStatus.COMPLETE":
            line += "  " + fetch(slug=full, token_file=token_file, dest_root=args.dest)
        print(line, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
