"""Pair immutable BF16 and FP8 vLLM condition artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.eval.runtime_artifacts import compare_vllm_conditions
from src.runtime.artifacts import (
    load_verified_artifact,
    validate_source_provenance,
    write_immutable_json_artifact,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare genuine vLLM condition artifacts")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/derived/runtime")
    )
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    source_git_state = validate_source_provenance(
        REPO_ROOT, require_clean_git=not args.allow_dirty
    )

    reference = load_verified_artifact(args.reference)
    target = load_verified_artifact(args.target)
    comparison = compare_vllm_conditions(reference, target)
    artifact, digest = write_immutable_json_artifact(
        payload=comparison,
        output_dir=args.output_dir,
        stem="paired-vllm-bf16-fp8",
        repo_root=REPO_ROOT,
        require_clean_git=not args.allow_dirty,
        source_git_state=source_git_state,
    )
    print(f"Artifact: {artifact}")
    print(f"Digest:   {digest}")
    print(f"UG2:      {comparison['ug2_status']}")
    print("Training authorized: False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
