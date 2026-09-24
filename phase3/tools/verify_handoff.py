#!/usr/bin/env python3
"""Verify an immutable Member 1 report bundle from a separate account/shell."""

from __future__ import annotations

import argparse
import json
import os
import platform
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
    verify_snapshot_manifest,
    write_json_atomic,
)
from report import KIND_TO_SCHEMA, validate_value


REQUIRED_KINDS = {
    "phase3_preflight": "preflight",
    "phase3_smoke_backward": "smoke",
    "phase3_member1_handoff": "member1",
}

FIXTURE_ARTIFACT_NAMES = (
    "<runtime>/smoke/contentment_05000.json",
    "<runtime>/smoke/contentment_05000.manifest.json",
)


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
) -> dict[str, Any]:
    sidecar_path = report_dir / "fixture-artifacts.sha256"
    manifest_path = report_dir / "technical-fixture-manifest.json"
    if not sidecar_path.is_file() or not manifest_path.is_file():
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
    return {
        "fixture_sha256": parsed[0][0],
        "fixture_manifest_sha256": parsed[1][0],
        "image_sha256": observed["image_sha256"],
        "dataset_record_sha256": observed["dataset_record_sha256"],
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

    provenance_path = model_path / "VENUS_MODEL_SOURCE.json"
    provenance = read_json(provenance_path)
    expected_provenance = {
        "repo_id": source_lock["model"]["repository"],
        "revision": source_lock["model"]["revision"],
        "weight_format": source_lock["model"]["weight_format"],
    }
    if not isinstance(provenance, dict) or any(
        provenance.get(key) != expected
        for key, expected in expected_provenance.items()
    ):
        raise ValueError("shared model provenance differs from source-lock.json")
    model_lock = source_lock["model"]
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
) -> dict[str, Any]:
    if member_id not in {f"member{index}" for index in range(1, 7)}:
        raise ValueError("member_id must be member1 through member6")
    if _is_relative_to(output, report_dir):
        raise ValueError("verification output must not be written into Member 1's source report directory")
    caller_report_root = runtime_root / "reports" / member_id
    if not _is_relative_to(output, caller_report_root):
        raise ValueError("verification output must be inside the caller's member-scoped report directory")

    reports = _discover_reports(report_dir)
    values: dict[str, dict[str, Any]] = {}
    checks: list[dict[str, Any]] = []
    for kind, path in reports.items():
        value = read_json(path)
        validate_value(value, KIND_TO_SCHEMA[kind], project_root)
        values[kind] = value
        checks.append({"name": f"validate_{KIND_TO_SCHEMA[kind]}", "status": "pass"})

    preflight = values["phase3_preflight"]
    smoke = values["phase3_smoke_backward"]
    handoff = values["phase3_member1_handoff"]
    if preflight["mode"] != "runtime" or not preflight["passed"]:
        raise ValueError("source runtime preflight report is not passing")
    if not smoke["passed"]:
        raise ValueError("source smoke report is not passing")
    if not handoff["member1_environment_ready"] or not handoff["technical_smoke_passed"]:
        raise ValueError("source Member 1 handoff report is not passing")
    if handoff["member_id"] != "member1":
        raise ValueError("source handoff was not produced by member1")
    current_commit = git_output(project_root, "rev-parse", "HEAD")
    if handoff["source_lock"]["project_commit"] != current_commit:
        raise ValueError("current project checkout differs from the handoff project commit")
    if git_output(project_root, "status", "--porcelain", "--", "phase3"):
        raise ValueError("current phase3 tree or index differs from the handoff commit")
    checks.append(
        {
            "name": "phase3_project_commit",
            "status": "pass",
            "observed": {"project_commit": current_commit},
        }
    )
    if handoff["inputs"]["preflight_report_sha256"] != sha256_file(reports["phase3_preflight"]):
        raise ValueError("handoff preflight hash does not match its source report")
    if handoff["inputs"]["smoke_report_sha256"] != sha256_file(reports["phase3_smoke_backward"]):
        raise ValueError("handoff smoke hash does not match its source report")
    checks.append({"name": "aggregate_input_hashes", "status": "pass"})

    checksum_path, checksum_manifest = _load_checksum_manifest(report_dir)
    assert_report_is_redacted(checksum_manifest)
    verified_checksums = _verify_checksum_manifest(report_dir, checksum_manifest)
    required_names = {
        *(path.name for path in reports.values()),
        "fixture-artifacts.sha256",
        "technical-fixture-manifest.json",
        "environment.freeze.txt",
    }
    if not required_names.issubset(verified_checksums):
        raise ValueError("checksum manifest does not cover every required report")
    checks.append(
        {
            "name": "checksum_manifest",
            "status": "pass",
            "observed": {"verified_file_count": len(verified_checksums)},
        }
    )

    fixture_observation = _verify_fixture_artifacts(report_dir, smoke)
    if handoff["fixture"]["fixture_manifest_sha256"] != fixture_observation["fixture_manifest_sha256"]:
        raise ValueError("aggregate fixture manifest hash differs from immutable evidence")
    checks.append(
        {
            "name": "fixture_artifact_chain",
            "status": "pass",
            "observed": fixture_observation,
        }
    )

    freeze_path = report_dir / "environment.freeze.txt"
    expected_freeze = _normalized_freeze(freeze_path.read_text(encoding="utf-8"))
    completed = subprocess.run(
        [sys.executable, "-m", "pip", "freeze", "--all"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="strict",
    )
    observed_freeze = _normalized_freeze(completed.stdout)
    if observed_freeze != expected_freeze:
        raise ValueError("current shared environment differs from the acceptance freeze")
    freeze_sha256 = sha256_file(freeze_path)
    if handoff["inputs"]["environment_freeze_sha256"] != freeze_sha256:
        raise ValueError("aggregate environment freeze hash differs from immutable evidence")
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
        raise ValueError(f"shared runtime inputs are missing: {', '.join(missing_inputs)}")
    checks.append({"name": "shared_runtime_inputs", "status": "pass"})
    checks.extend(_require_current_runtime(project_root, runtime_root, model_path))

    member_probe = _exclusive_write_probe(runtime_root / "members" / member_id, member_id)
    run_probe = _exclusive_write_probe(runtime_root / "runs" / member_id, member_id)
    if not member_probe or not run_probe:
        raise ValueError("caller could not write to member-scoped member/run directories")
    checks.append({"name": "member_scoped_write_paths", "status": "pass"})

    result = {
        "schema_version": 1,
        "kind": "phase3_handoff_verification",
        "generated_at_utc": utc_now(),
        "member_id": member_id,
        "passed": True,
        "source_report_dir": redact_path(report_dir, (("runtime", runtime_root),)),
        "source": {
            "project_commit": handoff["source_lock"]["project_commit"],
            "model_snapshot_manifest_sha256": handoff["source_lock"]["model_snapshot_manifest_sha256"],
            "preflight_report_sha256": sha256_file(reports["phase3_preflight"]),
            "smoke_report_sha256": sha256_file(reports["phase3_smoke_backward"]),
            "member1_report_sha256": sha256_file(reports["phase3_member1_handoff"]),
            "environment_freeze_sha256": freeze_sha256,
            "fixture_manifest_sha256": fixture_observation["fixture_manifest_sha256"],
            "checksum_manifest_sha256": sha256_file(checksum_path),
        },
        "checks": checks,
        "errors": [],
        "full_dataset_ready": False,
        "formal_training_authorized": False,
    }
    validate_value(result, "verification", project_root)
    return result


def main(argv: list[str] | None = None) -> int:
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
    report_dir = args.report_dir.resolve()
    output_path = args.output.resolve()
    try:
        result = verify(
            project_root=project_root,
            runtime_root=runtime_root,
            report_dir=report_dir,
            model_path=args.model_path.resolve(),
            output=output_path,
            member_id=args.member_id,
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
        result = {
            "schema_version": 1,
            "kind": "phase3_handoff_verification",
            "generated_at_utc": utc_now(),
            "member_id": args.member_id,
            "passed": False,
            "source_report_dir": redact_path(report_dir, (("runtime", runtime_root),)),
            "source": None,
            "checks": [],
            "errors": [type(error).__name__],
            "full_dataset_ready": False,
            "formal_training_authorized": False,
        }
        print(f"handoff verification failed: {type(error).__name__}: {error}", file=sys.stderr)
        return_code = 1
    assert_report_is_redacted(result)
    validate_value(result, "verification", project_root)
    write_json_atomic(output_path, result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
