#!/usr/bin/env python3
"""Small shared helpers for the Phase 3 Member 1 command-line tools."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SECRET_KEY_RE = re.compile(
    r"(?:^|[_-])(?:access[_-]?token|refresh[_-]?token|auth[_-]?token|secret|password|passwd|api[_-]?key|authorization|private[_-]?key)(?:$|[_-])",
    re.IGNORECASE,
)
SIGNED_URL_RE = re.compile(r"(?:X-Amz-|Signature=|sig=|token=)", re.IGNORECASE)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
        # Reports and runtime metadata are shared between six trusted RunPod
        # users.  mkstemp defaults to 0600, which would become unreadable to
        # the other users after an acceptance bundle is sealed read-only.
        os.chmod(temporary, 0o664)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def git_output(repository: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repository), *args],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return completed.stdout.strip()


def verify_exact_git_patch(
    repository: Path,
    patch_path: Path,
    *,
    changed_path: str = "finetune.py",
) -> dict[str, Any]:
    """Require one working-tree change whose raw Git diff equals a reviewed patch."""
    relative_path = Path(changed_path)
    if relative_path.is_absolute() or relative_path.as_posix() != changed_path:
        raise ValueError("changed_path must be a normalized repository-relative path")
    if not patch_path.is_file():
        raise ValueError("reviewed patch file is unavailable")

    completed = subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "diff",
            "--binary",
            "--no-ext-diff",
            "HEAD",
            "--",
            changed_path,
        ],
        check=True,
        capture_output=True,
    )
    observed_diff = completed.stdout
    reviewed_patch = patch_path.read_bytes()
    if observed_diff != reviewed_patch:
        raise ValueError("working-tree diff does not exactly match the reviewed patch")

    status = git_output(
        repository, "status", "--porcelain", "--untracked-files=all"
    )
    # git_output strips surrounding whitespace, including Porcelain's leading
    # unstaged-status column. A single unstaged modification therefore becomes
    # `M path`; staged or additional changes still produce a different value.
    expected_status = f"M {changed_path}"
    if status != expected_status:
        raise ValueError("checkout contains working-tree content beyond the reviewed patch")
    return {
        "diff_sha256": sha256_bytes(observed_diff),
        "diff_byte_count": len(observed_diff),
        "working_tree_expected": True,
    }


def redact_path(path: Path, roots: Iterable[tuple[str, Path]]) -> str:
    resolved = path.resolve()
    for label, root in roots:
        try:
            relative = resolved.relative_to(root.resolve())
            return f"<{label}>/{relative.as_posix()}"
        except ValueError:
            continue
    return f"<external>/{path.name}"


def assert_report_is_redacted(value: Any, location: str = "$") -> None:
    """Reject obvious secrets and signed URLs before a report is persisted."""
    if isinstance(value, dict):
        for key, item in value.items():
            if SECRET_KEY_RE.search(str(key)):
                raise ValueError(f"sensitive key is not allowed in report at {location}.{key}")
            assert_report_is_redacted(item, f"{location}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            assert_report_is_redacted(item, f"{location}[{index}]")
    elif isinstance(value, str) and SIGNED_URL_RE.search(value):
        raise ValueError(f"signed URL or credential-like value at {location}")


def package_versions(names: Iterable[str]) -> dict[str, str]:
    from importlib import metadata

    result: dict[str, str] = {}
    for name in names:
        try:
            result[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            result[name] = "not-installed"
    return result


def expected_model_provenance(model_lock: dict[str, Any]) -> dict[str, Any]:
    """Return the one accepted on-disk identity marker for the Stage 1 model."""
    expected = {
        "schema_version": 1,
        "repo_id": model_lock.get("repository"),
        "revision": model_lock.get("revision"),
        "weight_format": model_lock.get("weight_format"),
        "snapshot_manifest_sha256": model_lock.get("snapshot_manifest_sha256"),
    }
    if (
        not isinstance(expected["repo_id"], str)
        or not expected["repo_id"]
        or not isinstance(expected["revision"], str)
        or not re.fullmatch(r"[0-9a-f]{40}", expected["revision"])
        or not isinstance(expected["weight_format"], str)
        or not expected["weight_format"]
        or not isinstance(expected["snapshot_manifest_sha256"], str)
        or not re.fullmatch(r"[0-9a-f]{64}", expected["snapshot_manifest_sha256"])
    ):
        raise ValueError("source lock contains invalid model provenance fields")
    return expected


def verify_model_provenance(
    model_path: Path, model_lock: dict[str, Any]
) -> dict[str, Any]:
    """Require the exact provenance marker created after full snapshot hashing."""
    marker_path = model_path / "VENUS_MODEL_SOURCE.json"
    if marker_path.is_symlink() or not marker_path.is_file():
        raise ValueError(
            "model provenance marker is missing or is not a regular non-symlink file"
        )
    observed = read_json(marker_path)
    expected = expected_model_provenance(model_lock)
    if observed != expected:
        raise ValueError("model provenance marker differs from source-lock.json")
    return {
        **expected,
        "marker_sha256": sha256_file(marker_path),
    }


def verify_snapshot_manifest(
    model_path: Path,
    manifest_path: Path,
    *,
    expected_manifest_sha256: str,
    expected_repository: str,
    expected_revision: str,
) -> dict[str, Any]:
    """Hash every pinned model snapshot file and return a compact observation."""
    manifest_sha256 = sha256_file(manifest_path)
    if manifest_sha256 != expected_manifest_sha256:
        raise ValueError("model snapshot manifest SHA-256 differs from source-lock.json")
    manifest = read_json(manifest_path)
    if not isinstance(manifest, dict):
        raise ValueError("model snapshot manifest is not an object")
    if manifest.get("repository") != expected_repository:
        raise ValueError("model snapshot manifest repository differs from source-lock.json")
    if manifest.get("revision") != expected_revision:
        raise ValueError("model snapshot manifest revision differs from source-lock.json")
    entries = manifest.get("files")
    if not isinstance(entries, list) or not entries:
        raise ValueError("model snapshot manifest contains no files")

    seen: set[str] = set()
    total_bytes = 0
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("model snapshot manifest contains a non-object entry")
        name = entry.get("path")
        expected_size = entry.get("size")
        expected_sha256 = entry.get("sha256")
        if (
            not isinstance(name, str)
            or Path(name).name != name
            or name in seen
            or not isinstance(expected_size, int)
            or expected_size < 0
            or not isinstance(expected_sha256, str)
            or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256)
        ):
            raise ValueError("model snapshot manifest contains an invalid file entry")
        seen.add(name)
        path = model_path / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(
                f"pinned model file is missing or is not a regular non-symlink file: {name}"
            )
        observed_size = path.stat().st_size
        if observed_size != expected_size:
            raise ValueError(f"pinned model file size differs: {name}")
        if sha256_file(path) != expected_sha256:
            raise ValueError(f"pinned model file SHA-256 differs: {name}")
        total_bytes += observed_size

    if manifest.get("file_count") != len(entries):
        raise ValueError("model snapshot manifest file count is inconsistent")
    if manifest.get("total_bytes") != total_bytes:
        raise ValueError("model snapshot manifest total bytes are inconsistent")
    return {
        "manifest_sha256": manifest_sha256,
        "file_count": len(entries),
        "total_bytes": total_bytes,
    }
