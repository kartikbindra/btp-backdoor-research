"""Check the status of the Step 1 / Step 2 Kaggle kernels (one account per kernel).

    python -m scripts.check_kaggle_status            # status of all four
    python -m scripts.check_kaggle_status --fetch    # also download finished outputs
    python -m scripts.check_kaggle_status --log k3tikvats/c007-step2-margin-s42

Tokens live in ~/.kaggle as bare access tokens; the mapping below is the one used at launch.
Nothing here prints token contents.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from typing import Dict, List, Tuple

KAGGLE_DIR = os.path.join(os.path.expanduser("~"), ".kaggle")

# (account, kernel slug, token file, output dir name)
KERNELS: List[Tuple[str, str, str]] = [
    ("kb1217", "c007-step1-attack-laws", "access_token.kb1217.bak"),
    ("k3tikvats", "c007-step2-margin-s42", "access_token.k3tikvats.bak"),
    ("kartikbindra", "c007-step2-margin-s123", "access_token.kartikbindra.active.bak"),
    ("kartik1217", "c007-step2-margin-s7", "access_token.token4.bak"),
]

STATUS_EMOJI = {
    "KernelWorkerStatus.COMPLETE": "DONE",
    "KernelWorkerStatus.RUNNING": "RUNNING",
    "KernelWorkerStatus.QUEUED": "QUEUED",
    "KernelWorkerStatus.ERROR": "ERROR",
    "KernelWorkerStatus.CANCEL_ACKNOWLEDGED": "CANCELLED",
}


def _env(token_file: str) -> Dict[str, str]:
    env = dict(os.environ)
    env["KAGGLE_API_TOKEN"] = os.path.join(KAGGLE_DIR, token_file)
    return env


def status(slug: str, token_file: str) -> str:
    """Return the KernelWorkerStatus enum string, or an 'ERR ...' marker."""
    try:
        out = subprocess.run([sys.executable, "-m", "kaggle", "kernels", "status", slug],
                             capture_output=True, text=True, env=_env(token_file), timeout=120)
        line = (out.stdout or out.stderr).strip().splitlines()[-1]
        for tok in line.split('"'):
            if tok.startswith("KernelWorkerStatus."):
                return tok
        return line[:60]
    except Exception as exc:  # network / CLI issues must not hide the other kernels
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
    ap.add_argument("--dest", default=os.path.join("results", "campaign_007", "analysis", "kaggle_downloads"))
    args = ap.parse_args(argv)

    for account, slug, token_file in KERNELS:
        full = f"{account}/{slug}"
        st = status(slug=full, token_file=token_file)
        label = STATUS_EMOJI.get(st, st)
        line = f"{full:44s} {label}"
        if args.fetch and st == "KernelWorkerStatus.COMPLETE":
            line += "  " + fetch(slug=full, token_file=token_file, dest_root=args.dest)
        print(line, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
