"""Immutable experiment artifacts with source and file-integrity provenance."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import socket
import subprocess
import sys
from typing import Any, Dict, Iterable, Optional, Tuple


class ArtifactIntegrityError(RuntimeError):
    """Raised when artifact provenance or integrity cannot be guaranteed."""


_MANAGED_EVIDENCE_PREFIXES = (
    "results/raw/runtime/",
    "results/derived/runtime/",
    "results/local_preflight/",
)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _run_git(repo_root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _status_paths(line: str) -> Tuple[str, ...]:
    value = line[3:] if len(line) >= 4 else line
    return tuple(part.strip().strip('"') for part in value.split(" -> "))


def _managed_evidence_line(line: str) -> bool:
    paths = _status_paths(line)
    return bool(paths) and all(
        any(path.startswith(prefix) for prefix in _MANAGED_EVIDENCE_PREFIXES)
        for path in paths
    )


def _source_tree_fingerprint(repo_root: Path, source_status: Iterable[str]) -> str:
    """Hash tracked diffs and untracked source contents for exploratory provenance."""
    status_lines = sorted(source_status)
    digest = hashlib.sha256()
    digest.update("\n".join(status_lines).encode("utf-8"))
    for args in (("diff", "--binary", "HEAD"), ("diff", "--cached", "--binary")):
        try:
            digest.update(_run_git(repo_root, *args).encode("utf-8"))
        except ArtifactIntegrityError:
            raise
        except Exception as exc:
            raise ArtifactIntegrityError("Unable to fingerprint source diff") from exc
    for line in status_lines:
        if not line.startswith("?? "):
            continue
        for relative in _status_paths(line):
            path = repo_root / relative
            if path.is_file():
                digest.update(relative.encode("utf-8"))
                digest.update(hashlib.sha256(path.read_bytes()).digest())
            elif path.is_dir():
                for child in sorted(item for item in path.rglob("*") if item.is_file()):
                    child_relative = str(child.relative_to(repo_root))
                    digest.update(child_relative.encode("utf-8"))
                    digest.update(hashlib.sha256(child.read_bytes()).digest())
    return digest.hexdigest()


def collect_git_state(repo_root: Path) -> Dict[str, Any]:
    """Capture source identity while separating managed evidence outputs."""
    try:
        commit = _run_git(repo_root, "rev-parse", "HEAD")
        status = _run_git(repo_root, "status", "--porcelain=v1", "--untracked-files=all")
        remote = _run_git(repo_root, "remote", "get-url", "origin")
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ArtifactIntegrityError("Unable to capture Git provenance") from exc

    lines = status.splitlines()
    managed = [line for line in lines if _managed_evidence_line(line)]
    source = [line for line in lines if not _managed_evidence_line(line)]
    return {
        "commit": commit,
        "dirty": bool(source),  # Compatibility: means source-dirty, not result-dirty.
        "source_dirty": bool(source),
        "source_status_porcelain": source,
        "managed_evidence_status_porcelain": managed,
        "source_tree_fingerprint": _source_tree_fingerprint(repo_root, source),
        "origin": remote,
    }


def validate_source_provenance(
    repo_root: Path, require_clean_git: bool = True
) -> Dict[str, Any]:
    """Validate source state before expensive model/runtime execution."""
    state = collect_git_state(repo_root)
    if require_clean_git and state["source_dirty"]:
        raise ArtifactIntegrityError(
            "Refusing confirmatory execution from dirty source. Commit the "
            "implementation first or mark the run exploratory."
        )
    return state


def collect_environment(
    packages: Iterable[str] = (
        "torch",
        "vllm",
        "transformers",
        "safetensors",
        "numpy",
        "PyYAML",
    ),
) -> Dict[str, Any]:
    """Capture host and installed-package metadata without importing vLLM."""
    versions: Dict[str, str | None] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None

    environment: Dict[str, Any] = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "packages": versions,
        "cuda": {"available": False, "devices": []},
    }
    try:
        import torch

        environment["cuda"]["available"] = torch.cuda.is_available()
        environment["cuda"]["compiled_version"] = torch.version.cuda
        if torch.cuda.is_available():
            for index in range(torch.cuda.device_count()):
                major, minor = torch.cuda.get_device_capability(index)
                props = torch.cuda.get_device_properties(index)
                raw_uuid = getattr(props, "uuid", None)
                environment["cuda"]["devices"].append(
                    {
                        "index": index,
                        "name": torch.cuda.get_device_name(index),
                        "uuid": str(raw_uuid) if raw_uuid is not None else None,
                        "compute_capability": f"sm_{major}{minor}",
                        "total_memory_bytes": int(props.total_memory),
                    }
                )
    except ImportError:
        pass
    return environment


def write_immutable_json_artifact(
    payload: Dict[str, Any],
    output_dir: Path,
    stem: str,
    repo_root: Path,
    require_clean_git: bool = True,
    source_git_state: Optional[Dict[str, Any]] = None,
) -> Tuple[Path, Path]:
    """Write JSON and a mandatory SHA-256 sidecar without overwriting.

    ``source_git_state`` should be captured before model execution. Its commit
    and source fingerprint are checked again before publication, so source edits
    during a run invalidate publication. Managed evidence outputs are excluded
    from source dirtiness and therefore do not deadlock the next condition.
    """
    before = source_git_state or validate_source_provenance(
        repo_root, require_clean_git=require_clean_git
    )
    after = collect_git_state(repo_root)
    for field in ("commit", "source_tree_fingerprint", "source_dirty"):
        if before.get(field) != after.get(field):
            raise ArtifactIntegrityError(
                f"Source provenance changed during execution: {field}"
            )
    if require_clean_git and before["source_dirty"]:
        raise ArtifactIntegrityError("Confirmatory artifact has dirty source")

    evidence_class = (
        "CONFIRMATORY_CLEAN_SOURCE"
        if not before["source_dirty"]
        else "EXPLORATORY_DIRTY_SOURCE"
    )
    record: Dict[str, Any] = {
        "schema_version": "1.1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "evidence_class": evidence_class,
        "git": before,
        "environment": collect_environment(),
        "payload": payload,
    }
    record["content_sha256"] = sha256_json(record)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    artifact_path = output_dir / f"{stem}-{timestamp}-{record['content_sha256'][:12]}.json"
    digest_path = Path(f"{artifact_path}.sha256")
    if artifact_path.exists() or digest_path.exists():
        raise ArtifactIntegrityError("Refusing to overwrite an artifact or sidecar")

    rendered = json.dumps(record, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    file_digest = hashlib.sha256(rendered.encode("utf-8")).hexdigest()
    try:
        with artifact_path.open("x", encoding="utf-8") as handle:
            handle.write(rendered)
        with digest_path.open("x", encoding="utf-8") as handle:
            handle.write(f"{file_digest}  {artifact_path.name}\n")
    except Exception:
        artifact_path.unlink(missing_ok=True)
        digest_path.unlink(missing_ok=True)
        raise
    return artifact_path, digest_path


def load_verified_artifact(path: Path) -> Dict[str, Any]:
    """Require and verify both file-byte sidecar and embedded content hash."""
    digest_path = Path(f"{path}.sha256")
    if not digest_path.is_file():
        raise ArtifactIntegrityError(f"Missing SHA-256 sidecar for {path}")
    rendered = path.read_bytes()
    parts = digest_path.read_text(encoding="utf-8").strip().split()
    if len(parts) != 2 or parts[1] != path.name:
        raise ArtifactIntegrityError(f"Malformed or misnamed sidecar for {path}")
    actual_file_digest = hashlib.sha256(rendered).hexdigest()
    if parts[0] != actual_file_digest:
        raise ArtifactIntegrityError(f"File-byte SHA-256 mismatch for {path}")

    record = json.loads(rendered)
    expected = record.pop("content_sha256", None)
    actual = sha256_json(record)
    if not expected or expected != actual:
        raise ArtifactIntegrityError(
            f"Artifact content hash mismatch for {path}: {expected!r} != {actual!r}"
        )
    record["content_sha256"] = expected
    return record
