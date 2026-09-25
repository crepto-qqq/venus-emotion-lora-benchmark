#!/usr/bin/env python3
"""Verify the complete Stage 1 snapshot and create its exact provenance marker."""

from __future__ import annotations

import argparse
import json
import stat
import sys
from pathlib import Path
from typing import Any

from common import (
    expected_model_provenance,
    read_json,
    utc_now,
    verify_model_provenance,
    verify_snapshot_manifest,
    write_json_atomic,
)


_WRITE_BITS = stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH


def _is_read_only(path: Path) -> bool:
    return stat.S_IMODE(path.stat().st_mode) & _WRITE_BITS == 0


def _create_marker(marker_path: Path, expected_marker: dict[str, Any]) -> None:
    """Atomically create a marker even when a verified snapshot root is 0555."""
    model_path = marker_path.parent
    original_mode = stat.S_IMODE(model_path.stat().st_mode)
    temporarily_writable = original_mode & stat.S_IWUSR == 0
    if temporarily_writable:
        # The files have already passed full hashing. Only permit the owner to
        # add the missing marker; do not relax any model-file permissions.
        model_path.chmod(original_mode | stat.S_IWUSR)
    try:
        write_json_atomic(marker_path, expected_marker)
    finally:
        if temporarily_writable:
            model_path.chmod(original_mode)


def _seal_snapshot(
    model_path: Path, pinned_names: list[str], marker_path: Path
) -> dict[str, Any]:
    """Remove every write bit from the pinned snapshot and verify the result."""
    for name in pinned_names:
        path = model_path / name
        path.chmod(stat.S_IMODE(path.stat().st_mode) & ~_WRITE_BITS)
    marker_path.chmod(stat.S_IMODE(marker_path.stat().st_mode) & ~_WRITE_BITS)
    model_path.chmod(stat.S_IMODE(model_path.stat().st_mode) & ~_WRITE_BITS)

    files_read_only = all(_is_read_only(model_path / name) for name in pinned_names)
    marker_read_only = _is_read_only(marker_path)
    root_read_only = _is_read_only(model_path)
    sealed = {
        "applied": True,
        "verified": files_read_only and marker_read_only and root_read_only,
        "manifest_file_count": len(pinned_names),
        "manifest_files_read_only": files_read_only,
        "marker_read_only": marker_read_only,
        "model_root_read_only": root_read_only,
    }
    if not sealed["verified"]:
        raise ValueError("model snapshot read-only seal verification failed")
    return sealed


def _validate_report(report: dict[str, Any], project_root: Path) -> None:
    from report import validate_value

    validate_value(report, "model_snapshot", project_root)


def prepare(
    *,
    project_root: Path,
    model_path: Path,
    cache_dir: Path,
    download_if_missing: bool,
) -> tuple[dict[str, Any], int]:
    if not cache_dir.is_absolute():
        raise ValueError("download cache directory must be absolute")
    try:
        cache_dir.relative_to(model_path)
    except ValueError:
        pass
    else:
        raise ValueError("download cache directory must be outside the model snapshot")

    source_lock = read_json(project_root / "phase3/configs/source-lock.json")
    if not isinstance(source_lock, dict) or not isinstance(source_lock.get("model"), dict):
        raise ValueError("source-lock.json has no model lock")
    model_lock = source_lock["model"]
    expected_marker = expected_model_provenance(model_lock)
    snapshot_manifest = read_json(project_root / model_lock["snapshot_manifest"])
    entries = snapshot_manifest.get("files") if isinstance(snapshot_manifest, dict) else None
    if not isinstance(entries, list) or not entries:
        raise ValueError("model snapshot manifest contains no files")
    pinned_names = [entry.get("path") for entry in entries if isinstance(entry, dict)]
    if len(pinned_names) != len(entries) or not all(
        isinstance(name, str) and Path(name).name == name for name in pinned_names
    ):
        raise ValueError("model snapshot manifest contains an unsafe path")
    symlink_names = [name for name in pinned_names if (model_path / name).is_symlink()]
    missing_names = [name for name in pinned_names if not (model_path / name).is_file()]
    locked_source = {
        "repository": model_lock["repository"],
        "revision": model_lock["revision"],
        "weight_format": model_lock["weight_format"],
        "snapshot_manifest_sha256": model_lock["snapshot_manifest_sha256"],
    }
    report: dict[str, Any] = {
        "schema_version": 1,
        "kind": "phase3_model_snapshot",
        "generated_at_utc": utc_now(),
        "passed": False,
        "marker_created": False,
        "download_attempted": False,
        "missing_file_count_before": len(missing_names),
        "source_lock": locked_source,
        "snapshot": None,
        "provenance": None,
        "sealed": {
            "applied": False,
            "verified": False,
            "manifest_file_count": len(pinned_names),
            "manifest_files_read_only": False,
            "marker_read_only": False,
            "model_root_read_only": False,
        },
        "errors": [],
    }
    try:
        if symlink_names:
            raise ValueError("pinned model paths must not be symlinks")
        if missing_names and download_if_missing:
            from huggingface_hub import snapshot_download

            model_path.mkdir(parents=True, exist_ok=True)
            cache_dir.mkdir(parents=True, exist_ok=True)
            report["download_attempted"] = True
            snapshot_download(
                repo_id=model_lock["repository"],
                revision=model_lock["revision"],
                cache_dir=str(cache_dir),
                local_dir=str(model_path),
                local_dir_use_symlinks=False,
                allow_patterns=pinned_names,
                resume_download=True,
                max_workers=2,
            )

        snapshot = verify_snapshot_manifest(
            model_path,
            project_root / model_lock["snapshot_manifest"],
            expected_manifest_sha256=model_lock["snapshot_manifest_sha256"],
            expected_repository=model_lock["repository"],
            expected_revision=model_lock["revision"],
        )
        report["snapshot"] = snapshot

        marker_path = model_path / "VENUS_MODEL_SOURCE.json"
        if marker_path.is_symlink():
            raise ValueError("model provenance marker path must not be a symlink")
        if marker_path.exists():
            if not marker_path.is_file():
                raise ValueError("model provenance marker path is not a regular file")
        else:
            _create_marker(marker_path, expected_marker)
            report["marker_created"] = True

        report["provenance"] = verify_model_provenance(model_path, model_lock)
        report["sealed"] = _seal_snapshot(model_path, pinned_names, marker_path)
        report["passed"] = True
        return report, 0
    except Exception as error:
        report["errors"] = [type(error).__name__]
        print(
            f"model snapshot preparation failed: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return report, 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        required=True,
        help="absolute Hugging Face cache directory on persistent storage",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--download-if-missing",
        action="store_true",
        help="download only missing pinned files from the locked Hugging Face revision",
    )
    args = parser.parse_args(argv)

    project_root = args.project_root.resolve()
    model_path = args.model_path.resolve()
    cache_dir = args.cache_dir.resolve()
    output = args.output.resolve()
    try:
        report, return_code = prepare(
            project_root=project_root,
            model_path=model_path,
            cache_dir=cache_dir,
            download_if_missing=args.download_if_missing,
        )
    except Exception as error:
        print(
            f"model snapshot preparation could not start: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 2
    _validate_report(report, project_root)
    write_json_atomic(output, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
