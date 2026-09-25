#!/usr/bin/env python3
"""Verify an immutable Member 1 report bundle from a separate account/shell."""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any

from common import (
    assert_report_is_redacted,
    git_output,
    package_versions,
    read_json,
    redact_path,
    sha256_file,
    utc_now,
    verify_exact_git_patch,
    verify_model_provenance,
    verify_snapshot_manifest,
    write_json_atomic,
)
REQUIRED_KINDS = {
    "phase3_preflight": "preflight",
    "phase3_smoke_backward": "smoke",
    "phase3_member1_handoff": "member1",
}

KIND_TO_SCHEMA = {
    "phase3_preflight": "preflight",
    "phase3_smoke_backward": "smoke",
    "phase3_member1_handoff": "member1",
}

FIXTURE_ARTIFACT_NAMES = (
    "technical-fixture.json",
    "technical-fixture-manifest.json",
)


class VerificationFailure(RuntimeError):
    """A redacted verifier failure with a stable machine-readable code."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


def _validate_value(value: Any, kind: str, project_root: Path) -> None:
    # Keep the failure-report subcommand usable with the host's stdlib Python
    # when the shared environment itself is the failed precondition.
    from report import validate_value

    validate_value(value, kind, project_root)


def _guard(code: str, action: Any) -> Any:
    try:
        return action()
    except VerificationFailure:
        raise
    except Exception as error:
        raise VerificationFailure(code) from error


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise VerificationFailure(code)


def _discover_reports(report_dir: Path) -> dict[str, Path]:
    discovered: dict[str, Path] = {}
    for path in sorted(report_dir.glob("*.json")):
        try:
            value = read_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(value, dict):
            continue
        kind = value.get("kind")
        if kind not in REQUIRED_KINDS:
            continue
        if kind in discovered:
            raise ValueError(f"multiple {kind} reports found in the handoff directory")
        discovered[str(kind)] = path
    missing = sorted(set(REQUIRED_KINDS) - set(discovered))
    if missing:
        raise ValueError(f"handoff directory is missing report kinds: {', '.join(missing)}")
    return discovered


def _load_checksum_manifest(report_dir: Path) -> tuple[Path, dict[str, Any]]:
    candidates: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(report_dir.glob("*.json")):
        try:
            value = read_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(value, dict) and value.get("kind") == "phase3_checksums":
            candidates.append((path, value))
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise ValueError("handoff directory contains multiple phase3_checksums JSON manifests")
    sha256sum_path = report_dir / "checksums.sha256"
    if not sha256sum_path.is_file():
        raise ValueError("handoff directory has no checksum manifest")
    entries: list[dict[str, str]] = []
    for line_number, raw_line in enumerate(
        sha256sum_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not raw_line.strip():
            continue
        parts = raw_line.split(maxsplit=1)
        if len(parts) != 2:
            raise ValueError(f"invalid checksums.sha256 line {line_number}")
        digest, name = parts
        name = name.removeprefix("*")
        entries.append({"name": name, "sha256": digest.lower()})
    return sha256sum_path, {"schema_version": 1, "kind": "phase3_checksums", "files": entries}


def _verify_checksum_manifest(report_dir: Path, manifest: dict[str, Any]) -> dict[str, str]:
    entries = manifest.get("files")
    if not isinstance(entries, list) or not entries:
        raise ValueError("checksum manifest has no files")
    verified: dict[str, str] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("invalid checksum entry")
        name = entry.get("name")
        expected = entry.get("sha256")
        if not isinstance(name, str) or Path(name).name != name:
            raise ValueError("checksum entries must use plain filenames")
        if not isinstance(expected, str) or len(expected) != 64:
            raise ValueError(f"invalid checksum for {name!r}")
        path = report_dir / name
        if not path.is_file():
            raise ValueError(f"checksummed file is missing: {name}")
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"checksum differs for {name}")
        verified[name] = actual
    return verified


def _normalized_freeze(raw: str) -> bytes:
    """Canonicalize pip freeze output as sorted non-empty UTF-8 lines with LF."""
    lines = sorted(line.strip() for line in raw.splitlines() if line.strip())
    return (("\n".join(lines) + "\n") if lines else "").encode("utf-8")


def _verify_fixture_artifacts(
    report_dir: Path,
    smoke: dict[str, Any],
    project_root: Path,
) -> dict[str, Any]:
    sidecar_path = report_dir / "fixture-artifacts.sha256"
    fixture_path = report_dir / "technical-fixture.json"
    manifest_path = report_dir / "technical-fixture-manifest.json"
    if not sidecar_path.is_file() or not fixture_path.is_file() or not manifest_path.is_file():
        raise ValueError("immutable fixture evidence files are missing")

    lines = sidecar_path.read_text(encoding="utf-8").splitlines()
    if len(lines) != 2 or any(not line for line in lines):
        raise ValueError("fixture-artifacts.sha256 must contain exactly two lines")
    parsed: list[tuple[str, str]] = []
    for index, line in enumerate(lines):
        parts = line.split("  ", maxsplit=1)
        if (
            len(parts) != 2
            or len(parts[0]) != 64
            or any(character not in "0123456789abcdef" for character in parts[0])
            or parts[1] != FIXTURE_ARTIFACT_NAMES[index]
        ):
            raise ValueError(
                f"fixture-artifacts.sha256 line {index + 1} is not canonical"
            )
        parsed.append((parts[0], parts[1]))

    smoke_fixture = smoke.get("fixture")
    if not isinstance(smoke_fixture, dict):
        raise ValueError("smoke report fixture evidence is missing")
    if parsed[0][0] != sha256_file(fixture_path):
        raise ValueError("copied technical fixture digest differs from its sidecar")
    if parsed[0][0] != smoke_fixture.get("fixture_sha256"):
        raise ValueError("fixture artifact digest differs from the smoke report")
    if parsed[1][0] != sha256_file(manifest_path):
        raise ValueError("copied technical fixture manifest digest differs from its sidecar")

    fixture_manifest = read_json(manifest_path)
    if not isinstance(fixture_manifest, dict):
        raise ValueError("copied technical fixture manifest is not a JSON object")
    expected = {
        "source_image_id": smoke_fixture.get("id"),
        "fixture_sha256": smoke_fixture.get("fixture_sha256"),
        "image_sha256": smoke_fixture.get("image_sha256"),
        "dataset_record_sha256": smoke_fixture.get("dataset_record_sha256"),
    }
    observed = {name: fixture_manifest.get(name) for name in expected}
    if observed != expected:
        raise ValueError("copied technical fixture manifest differs from smoke evidence")

    fixture = read_json(fixture_path)
    _validate_value(fixture, "fixture", project_root)
    semantic_observation = _expected_fixture_semantics(
        project_root, fixture, fixture_manifest
    )
    if fixture_manifest.get("fixture_sha256") != parsed[0][0]:
        raise ValueError("fixture manifest does not identify the copied fixture")
    record = fixture[0]
    provenance = record["provenance"]
    fixture_identity = {
        "source_image_id": record["id"],
        "image_sha256": provenance["image_sha256"],
        "dataset_record_sha256": provenance["dataset_record_sha256"],
    }
    expected_identity = {
        "source_image_id": smoke_fixture.get("id"),
        "image_sha256": smoke_fixture.get("image_sha256"),
        "dataset_record_sha256": smoke_fixture.get("dataset_record_sha256"),
    }
    if fixture_identity != expected_identity:
        raise ValueError("copied technical fixture identity differs from smoke evidence")
    return {
        "fixture_sha256": parsed[0][0],
        "fixture_manifest_sha256": parsed[1][0],
        "image_sha256": observed["image_sha256"],
        "dataset_record_sha256": observed["dataset_record_sha256"],
        "annotation_sha256": semantic_observation["annotation_sha256"],
        "disjointness_check": semantic_observation["disjointness_check"],
    }


def _exclusive_write_probe(directory: Path, member_id: str) -> bool:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f".handoff-write-probe-{member_id}-{os.getpid()}"
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write("phase3 handoff write probe\n")
        return path.read_text(encoding="utf-8") == "phase3 handoff write probe\n"
    finally:
        path.unlink(missing_ok=True)


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _require_secure_source_bundle(report_dir: Path) -> dict[str, Any]:
    """Require a flat, genuinely immutable bundle readable by every account."""
    directory_stat = report_dir.lstat()
    if stat.S_ISLNK(directory_stat.st_mode) or not stat.S_ISDIR(directory_stat.st_mode):
        raise VerificationFailure("source_bundle_type_invalid")
    directory_mode = stat.S_IMODE(directory_stat.st_mode)
    if directory_mode & 0o222:
        raise VerificationFailure("source_bundle_writable")
    if directory_mode & 0o555 != 0o555:
        raise VerificationFailure("source_bundle_unreadable")

    children = sorted(report_dir.iterdir(), key=lambda item: item.name)
    if not children:
        raise VerificationFailure("source_bundle_empty")
    for path in children:
        child_stat = path.lstat()
        if stat.S_ISLNK(child_stat.st_mode) or not stat.S_ISREG(child_stat.st_mode):
            raise VerificationFailure("source_evidence_type_invalid")
        child_mode = stat.S_IMODE(child_stat.st_mode)
        if child_mode & 0o222:
            raise VerificationFailure("source_evidence_writable")
        if child_mode & 0o444 != 0o444:
            raise VerificationFailure("source_evidence_unreadable")
        with path.open("rb") as handle:
            handle.read(1)
    return {"file_count": len(children), "directory_mode": format(directory_mode, "04o")}


def _expected_fixture_semantics(
    project_root: Path,
    fixture: Any,
    manifest: Any,
) -> dict[str, Any]:
    """Rebuild every fixed fixture/manifest field from reviewed source locks."""
    from build_smoke_fixture import (
        DEFAULT_EVAL20_RESULT,
        DEFAULT_EVAL80_MANIFEST,
        _check_disjointness,
        _read_locked_record,
        _validate_locked_record,
    )

    config = read_json(project_root / "phase3/configs/member1-smoke.json")
    lock = config["fixture"]
    record = _read_locked_record(project_root, config)
    _validate_locked_record(record, lock)
    if not isinstance(manifest, dict):
        raise ValueError("fixture manifest is not an object")
    image = manifest.get("image")
    if not isinstance(image, dict):
        raise ValueError("fixture manifest image is not an object")
    image_reference = image.get("reference")
    if not isinstance(image_reference, str) or not image_reference:
        raise ValueError("fixture image reference is missing")

    disjointness, evaluation_checks = _check_disjointness(
        project_root=project_root,
        source_image_id=lock["source_image_id"],
        image_sha256=lock["image_sha256"],
        eval80_manifest=DEFAULT_EVAL80_MANIFEST,
        eval20_result=DEFAULT_EVAL20_RESULT,
    )
    expected_fixture = [
        {
            "id": lock["source_image_id"],
            "technical_infrastructure_only": True,
            "conversations": [
                {
                    "from": "user",
                    "value": f"Picture 1: <img>{image_reference}</img>\n{record['instruction']}",
                },
                {"from": "assistant", "value": record["target_response"]},
            ],
            "provenance": {
                "dataset_record_sha256": lock["record_sha256"],
                "annotation_sha256": lock["annotation_sha256"],
                "image_sha256": lock["image_sha256"],
                "source_split": "train",
                "disjointness_check": disjointness,
            },
        }
    ]
    if fixture != expected_fixture:
        raise ValueError("fixture semantics differ from the reviewed source lock")

    generated_at = manifest.get("generated_at_utc")
    if not isinstance(generated_at, str) or not generated_at.endswith("Z"):
        raise ValueError("fixture manifest timestamp is invalid")
    expected_manifest = {
        "schema_version": 1,
        "kind": "phase3_technical_fixture_manifest",
        "generated_at_utc": generated_at,
        "technical_infrastructure_only": True,
        "source_image_id": lock["source_image_id"],
        "emotion": lock["emotion"],
        "source_split": "train",
        "review_status": record["provenance"]["review_status"],
        "label_supported": record["provenance"]["label_supported"],
        "source_dataset_relpath": lock["dataset_relpath"],
        "source_record_line": lock["record_line"],
        "dataset_record_sha256": lock["record_sha256"],
        "annotation_sha256": lock["annotation_sha256"],
        "image_sha256": lock["image_sha256"],
        "image": {
            "reference": image_reference,
            "byte_count": lock["image_byte_count"],
            "width": lock["image_width"],
            "height": lock["image_height"],
        },
        "disjointness_check": disjointness,
        "evaluation_checks": evaluation_checks,
        "full_dataset_ready": False,
        "formal_training_authorized": False,
        "fixture_sha256": manifest.get("fixture_sha256"),
    }
    if manifest != expected_manifest:
        raise ValueError("fixture manifest semantics differ from the reviewed source lock")
    return {
        "source_image_id": lock["source_image_id"],
        "image_sha256": lock["image_sha256"],
        "dataset_record_sha256": lock["record_sha256"],
        "annotation_sha256": lock["annotation_sha256"],
        "disjointness_check": disjointness,
    }


def _verify_handoff_fixture_hashes(
    handoff: dict[str, Any], fixture_observation: dict[str, Any]
) -> None:
    if handoff["fixture"]["fixture_sha256"] != fixture_observation["fixture_sha256"]:
        raise VerificationFailure("fixture_evidence_invalid")
    if (
        handoff["fixture"]["fixture_manifest_sha256"]
        != fixture_observation["fixture_manifest_sha256"]
    ):
        raise VerificationFailure("fixture_evidence_invalid")


def _require_current_runtime(
    project_root: Path, runtime_root: Path, model_path: Path
) -> list[dict[str, Any]]:
    """Recheck mutable shared inputs without loading the model or rerunning smoke."""
    phase3 = project_root / "phase3"
    source_lock = read_json(phase3 / "configs/source-lock.json")
    expected_python = (phase3 / "environment/python-version.txt").read_text(
        encoding="utf-8"
    ).strip()
    environment = runtime_root / "envs" / "venus-phase3"
    if Path(sys.prefix).resolve() != environment.resolve():
        raise ValueError("handoff verifier is not running from the shared pinned environment")
    if platform.python_version() != expected_python:
        raise ValueError("shared Python version differs from python-version.txt")

    requirement_pins: dict[str, str] = {}
    for raw_line in (phase3 / "environment/requirements.txt").read_text(
        encoding="utf-8"
    ).splitlines():
        line = raw_line.strip()
        if not line or line.startswith(("#", "--")) or "==" not in line:
            continue
        name, version = line.split("==", 1)
        requirement_pins[name.strip()] = version.strip()
    installed = package_versions(requirement_pins)
    mismatches = {
        name: {"expected": expected, "observed": installed.get(name)}
        for name, expected in requirement_pins.items()
        if installed.get(name) != expected
    }
    if mismatches:
        raise ValueError(
            "shared package pins differ: " + ", ".join(sorted(mismatches))
        )

    qwen_dir = runtime_root / "upstream" / "Qwen-VL-finetune"
    venus_dir = runtime_root / "upstream" / "Venus_CVPR2026"
    qwen_commit = git_output(qwen_dir, "rev-parse", "HEAD")
    venus_commit = git_output(venus_dir, "rev-parse", "HEAD")
    patch_path = project_root / source_lock["upstream"]["patch"]
    patch_sha256 = sha256_file(patch_path)
    if qwen_commit != source_lock["upstream"]["commit"]:
        raise ValueError("shared Qwen checkout differs from source-lock.json")
    if venus_commit != source_lock["venus_source"]["commit"]:
        raise ValueError("shared Venus checkout differs from source-lock.json")
    if patch_sha256 != source_lock["upstream"]["patch_sha256"]:
        raise ValueError("reviewed Qwen patch differs from source-lock.json")
    patch_observation = verify_exact_git_patch(qwen_dir, patch_path)
    if git_output(venus_dir, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("shared Venus checkout has unexpected working-tree changes")

    model_lock = source_lock["model"]
    provenance = verify_model_provenance(model_path, model_lock)
    snapshot_observation = verify_snapshot_manifest(
        model_path,
        project_root / model_lock["snapshot_manifest"],
        expected_manifest_sha256=model_lock["snapshot_manifest_sha256"],
        expected_repository=model_lock["repository"],
        expected_revision=model_lock["revision"],
    )
    index_path = model_path / model_lock["index_file"]
    index = read_json(index_path)
    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("shared model index has no weight map")
    shard_names = sorted(set(weight_map.values()))
    if not all(isinstance(name, str) and Path(name).name == name for name in shard_names):
        raise ValueError("shared model index contains an unsafe shard name")
    shard_paths = [model_path / name for name in shard_names]
    if any(not path.is_file() or path.stat().st_size <= 0 for path in shard_paths):
        raise ValueError("one or more shared model shards are missing or empty")
    shard_bytes = sum(path.stat().st_size for path in shard_paths)
    if len(shard_paths) != int(source_lock["model"]["weight_shard_count"]):
        raise ValueError("shared model shard count differs from source-lock.json")
    if shard_bytes != int(source_lock["model"]["weight_shard_total_bytes"]):
        raise ValueError("shared model shard bytes differ from source-lock.json")

    return [
        {
            "name": "shared_python_environment",
            "status": "pass",
            "observed": {
                "python": platform.python_version(),
                "package_pin_count": len(requirement_pins),
            },
        },
        {
            "name": "shared_upstream_pins",
            "status": "pass",
            "observed": {
                "qwen_commit": qwen_commit,
                "venus_commit": venus_commit,
                "qwen_patch_sha256": patch_sha256,
                **patch_observation,
            },
        },
        {
            "name": "shared_model_pin",
            "status": "pass",
            "observed": {
                "repository": provenance["repo_id"],
                "revision": provenance["revision"],
                "shard_count": len(shard_paths),
                "shard_total_bytes": shard_bytes,
                **snapshot_observation,
            },
        },
    ]


def verify(
    *,
    project_root: Path,
    runtime_root: Path,
    report_dir: Path,
    model_path: Path,
    output: Path,
    member_id: str,
    checks: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if checks is None:
        checks = []
    if member_id not in {f"member{index}" for index in range(1, 7)}:
        raise VerificationFailure("member_id_invalid")
    if _is_relative_to(output, report_dir):
        raise VerificationFailure("output_location_invalid")
    caller_report_root = runtime_root / "reports" / member_id
    if not _is_relative_to(output, caller_report_root):
        raise VerificationFailure("output_location_invalid")
    expected_source_parent = runtime_root / "reports" / "member1"
    if (
        report_dir.parent != expected_source_parent
        or re.fullmatch(r"attempt-[0-9]{3,}", report_dir.name) is None
    ):
        raise VerificationFailure("source_bundle_location_invalid")

    bundle_observation = _guard(
        "source_bundle_security_failed",
        lambda: _require_secure_source_bundle(report_dir),
    )
    checks.append(
        {
            "name": "source_bundle_immutable_and_readable",
            "status": "pass",
            "observed": bundle_observation,
        }
    )

    reports = _guard("source_reports_invalid", lambda: _discover_reports(report_dir))
    values: dict[str, dict[str, Any]] = {}
    for kind, path in reports.items():
        value = _guard("source_reports_invalid", lambda path=path: read_json(path))
        _guard(
            "source_reports_invalid",
            lambda value=value, kind=kind: _validate_value(
                value, KIND_TO_SCHEMA[kind], project_root
            ),
        )
        values[kind] = value
        checks.append({"name": f"validate_{KIND_TO_SCHEMA[kind]}", "status": "pass"})

    preflight = values["phase3_preflight"]
    smoke = values["phase3_smoke_backward"]
    handoff = values["phase3_member1_handoff"]
    if preflight["mode"] != "runtime" or not preflight["passed"]:
        raise VerificationFailure("source_reports_not_passing")
    if not smoke["passed"]:
        raise VerificationFailure("source_reports_not_passing")
    if not handoff["member1_environment_ready"] or not handoff["technical_smoke_passed"]:
        raise VerificationFailure("source_reports_not_passing")
    if handoff["member_id"] != "member1":
        raise VerificationFailure("source_report_owner_invalid")
    current_commit = _guard(
        "project_revision_invalid", lambda: git_output(project_root, "rev-parse", "HEAD")
    )
    if handoff["source_lock"]["project_commit"] != current_commit:
        raise VerificationFailure("project_revision_invalid")
    phase3_status = _guard(
        "project_revision_invalid",
        lambda: git_output(project_root, "status", "--porcelain", "--", "phase3"),
    )
    if phase3_status:
        raise VerificationFailure("project_revision_invalid")
    checks.append(
        {
            "name": "phase3_project_commit",
            "status": "pass",
            "observed": {"project_commit": current_commit},
        }
    )
    if handoff["inputs"]["preflight_report_sha256"] != sha256_file(reports["phase3_preflight"]):
        raise VerificationFailure("aggregate_input_hash_invalid")
    if handoff["inputs"]["smoke_report_sha256"] != sha256_file(reports["phase3_smoke_backward"]):
        raise VerificationFailure("aggregate_input_hash_invalid")
    checks.append({"name": "aggregate_input_hashes", "status": "pass"})

    checksum_path, checksum_manifest = _guard(
        "checksum_manifest_invalid", lambda: _load_checksum_manifest(report_dir)
    )
    _guard("checksum_manifest_invalid", lambda: assert_report_is_redacted(checksum_manifest))
    verified_checksums = _guard(
        "checksum_manifest_invalid",
        lambda: _verify_checksum_manifest(report_dir, checksum_manifest),
    )
    required_names = {
        *(path.name for path in reports.values()),
        "fixture-artifacts.sha256",
        "technical-fixture.json",
        "technical-fixture-manifest.json",
        "environment.freeze.txt",
    }
    if not required_names.issubset(verified_checksums):
        raise VerificationFailure("checksum_manifest_incomplete")
    checks.append(
        {
            "name": "checksum_manifest",
            "status": "pass",
            "observed": {"verified_file_count": len(verified_checksums)},
        }
    )

    fixture_observation = _guard(
        "fixture_evidence_invalid",
        lambda: _verify_fixture_artifacts(report_dir, smoke, project_root),
    )
    _verify_handoff_fixture_hashes(handoff, fixture_observation)
    checks.append(
        {
            "name": "fixture_artifact_chain",
            "status": "pass",
            "observed": fixture_observation,
        }
    )

    freeze_path = report_dir / "environment.freeze.txt"
    expected_freeze = _guard(
        "environment_freeze_invalid",
        lambda: _normalized_freeze(freeze_path.read_text(encoding="utf-8")),
    )
    completed = _guard(
        "environment_freeze_invalid",
        lambda: subprocess.run(
            [sys.executable, "-m", "pip", "freeze", "--all"],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="strict",
        ),
    )
    observed_freeze = _normalized_freeze(completed.stdout)
    if observed_freeze != expected_freeze:
        raise VerificationFailure("environment_freeze_invalid")
    freeze_sha256 = sha256_file(freeze_path)
    if handoff["inputs"]["environment_freeze_sha256"] != freeze_sha256:
        raise VerificationFailure("environment_freeze_invalid")
    checks.append(
        {
            "name": "captured_environment_freeze",
            "status": "pass",
            "observed": {"sha256": freeze_sha256, "line_count": len(expected_freeze.splitlines())},
        }
    )

    runtime_inputs = {
        "python_environment": runtime_root / "envs" / "venus-phase3" / "bin" / "python",
        "qwen_checkout": runtime_root / "upstream" / "Qwen-VL-finetune" / ".git",
        "venus_checkout": runtime_root / "upstream" / "Venus_CVPR2026" / ".git",
        "model_index": model_path / "model.safetensors.index.json",
        "model_provenance": model_path / "VENUS_MODEL_SOURCE.json",
    }
    missing_inputs = [name for name, path in runtime_inputs.items() if not path.exists()]
    if missing_inputs:
        raise VerificationFailure("shared_runtime_inputs_missing")
    checks.append({"name": "shared_runtime_inputs", "status": "pass"})
    checks.extend(
        _guard(
            "shared_runtime_pin_invalid",
            lambda: _require_current_runtime(project_root, runtime_root, model_path),
        )
    )

    member_probe = _guard(
        "member_write_scope_invalid",
        lambda: _exclusive_write_probe(
            runtime_root / "members" / member_id, member_id
        ),
    )
    run_probe = _guard(
        "member_write_scope_invalid",
        lambda: _exclusive_write_probe(runtime_root / "runs" / member_id, member_id),
    )
    if not member_probe or not run_probe:
        raise VerificationFailure("member_write_scope_invalid")
    checks.append({"name": "member_scoped_write_paths", "status": "pass"})

    result = {
        "schema_version": 1,
        "kind": "phase3_handoff_verification",
        "generated_at_utc": utc_now(),
        "member_id": member_id,
        "passed": True,
        "failure_code": None,
        "source_report_dir": redact_path(report_dir, (("runtime", runtime_root),)),
        "source": {
            "project_commit": handoff["source_lock"]["project_commit"],
            "model_snapshot_manifest_sha256": handoff["source_lock"]["model_snapshot_manifest_sha256"],
            "preflight_report_sha256": sha256_file(reports["phase3_preflight"]),
            "smoke_report_sha256": sha256_file(reports["phase3_smoke_backward"]),
            "member1_report_sha256": sha256_file(reports["phase3_member1_handoff"]),
            "environment_freeze_sha256": freeze_sha256,
            "fixture_sha256": fixture_observation["fixture_sha256"],
            "fixture_manifest_sha256": fixture_observation["fixture_manifest_sha256"],
            "checksum_manifest_sha256": sha256_file(checksum_path),
        },
        "checks": checks,
        "errors": [],
        "full_dataset_ready": False,
        "formal_training_authorized": False,
    }
    _validate_value(result, "verification", project_root)
    return result


def _source_report_label(report_dir: Path, runtime_root: Path) -> str:
    try:
        relative = report_dir.absolute().relative_to(runtime_root.absolute())
    except ValueError:
        return "<unavailable>"
    return f"<runtime>/{relative.as_posix()}"


def _failure_result(
    *,
    member_id: str,
    report_dir: Path,
    runtime_root: Path,
    failure_code: str,
    checks: list[dict[str, Any]],
) -> dict[str, Any]:
    retained = [
        item
        for item in checks
        if isinstance(item, dict)
        and isinstance(item.get("name"), str)
        and item.get("status") == "pass"
    ]
    retained.append({"name": failure_code, "status": "fail"})
    return {
        "schema_version": 1,
        "kind": "phase3_handoff_verification",
        "generated_at_utc": utc_now(),
        "member_id": member_id,
        "passed": False,
        "failure_code": failure_code,
        "source_report_dir": _source_report_label(report_dir, runtime_root),
        "source": None,
        "checks": retained,
        "errors": [failure_code],
        "full_dataset_ready": False,
        "formal_training_authorized": False,
    }


def _failure_report_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Write a redacted handoff failure report.")
    parser.add_argument("--runtime-root", "--workspace-root", dest="runtime_root", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--member-id", required=True)
    parser.add_argument("--failure-code", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preserve-checks-from", type=Path)
    args = parser.parse_args(argv)
    if args.member_id not in {f"member{index}" for index in range(1, 7)}:
        raise SystemExit("member_id_invalid")
    if re.fullmatch(r"[a-z][a-z0-9_]{2,63}", args.failure_code) is None:
        raise SystemExit("failure_code_invalid")
    checks: list[dict[str, Any]] = []
    if args.preserve_checks_from and args.preserve_checks_from.is_file():
        try:
            previous = read_json(args.preserve_checks_from)
            if isinstance(previous, dict) and isinstance(previous.get("checks"), list):
                checks = previous["checks"]
        except (OSError, json.JSONDecodeError):
            checks = []
    result = _failure_result(
        member_id=args.member_id,
        report_dir=Path(os.path.abspath(args.report_dir)),
        runtime_root=Path(os.path.abspath(args.runtime_root)),
        failure_code=args.failure_code,
        checks=checks,
    )
    assert_report_is_redacted(result)
    write_json_atomic(args.output, result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if argv and argv[0] == "failure-report":
        return _failure_report_main(argv[1:])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--runtime-root", "--workspace-root", dest="runtime_root", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, default=Path("/workspace/models/Venus-Q-Stage1"))
    parser.add_argument("--member-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    project_root = args.project_root.resolve()
    runtime_root = args.runtime_root.resolve()
    report_dir = Path(os.path.abspath(args.report_dir))
    output_path = args.output.resolve()
    checks: list[dict[str, Any]] = []
    try:
        result = verify(
            project_root=project_root,
            runtime_root=runtime_root,
            report_dir=report_dir,
            model_path=args.model_path.resolve(),
            output=output_path,
            member_id=args.member_id,
            checks=checks,
        )
        return_code = 0
    except Exception as error:  # Preserve a redacted machine-readable failure result.
        caller_root = runtime_root / "reports" / args.member_id
        if (
            args.member_id not in {f"member{index}" for index in range(1, 7)}
            or not _is_relative_to(output_path, caller_root)
            or _is_relative_to(output_path, report_dir)
        ):
            raise
        failure_code = (
            error.code if isinstance(error, VerificationFailure) else "internal_verification_error"
        )
        result = _failure_result(
            member_id=args.member_id,
            report_dir=report_dir,
            runtime_root=runtime_root,
            failure_code=failure_code,
            checks=checks,
        )
        print(f"handoff verification failed: {failure_code}", file=sys.stderr)
        return_code = 1
    assert_report_is_redacted(result)
    _validate_value(result, "verification", project_root)
    write_json_atomic(output_path, result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
