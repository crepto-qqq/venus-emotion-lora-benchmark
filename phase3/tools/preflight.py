#!/usr/bin/env python3
"""Run controlled static or RunPod runtime checks for the Member 1 handoff."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, Callable

from common import (
    assert_report_is_redacted,
    git_output,
    read_json,
    sha256_bytes,
    sha256_file,
    utc_now,
    verify_exact_git_patch,
    verify_model_provenance,
    verify_snapshot_manifest,
    write_json_atomic,
)


BASE_COMMIT = "869a24bb42b83f8b58c6f693a2cae6a84df5ae9d"
EXPECTED_BRANCH = "phase3member1-env"
REQUIRED_PHASE3_FILES = (
    "configs/member1-smoke.json",
    "configs/model-snapshot-manifest.json",
    "configs/source-lock.json",
    "contracts/member1-report.schema.json",
    "contracts/model-snapshot-report.schema.json",
    "contracts/handoff-verification.schema.json",
    "contracts/preflight-report.schema.json",
    "contracts/smoke-report.schema.json",
    "contracts/technical-fixture.schema.json",
    "environment/python-version.txt",
    "environment/requirements.txt",
    "environment/constraints-runpod-cu118.txt",
    "tools/prepare_model_snapshot.py",
    "upstream/patches/qwen-vl-finetune-efa37ba-phase3.patch",
)


def _gpu_memory_gate(
    total_bytes: int,
    free_bytes: int,
    minimum_total_bytes: int,
    minimum_free_bytes: int,
) -> bool:
    """Pure boundary predicate used by runtime preflight and its unit test."""
    return total_bytes >= minimum_total_bytes and free_bytes >= minimum_free_bytes


@dataclass
class Check:
    name: str
    status: str
    detail: str
    observed: Any | None = None

    def as_json(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "name": self.name,
            "status": self.status,
            "detail": self.detail,
        }
        if self.observed is not None:
            result["observed"] = self.observed
        return result


def _run_check(
    checks: list[Check],
    name: str,
    action: Callable[[], tuple[bool, str, Any | None]],
) -> None:
    try:
        passed, detail, observed = action()
        checks.append(Check(name, "pass" if passed else "fail", detail, observed))
    except Exception as error:  # A preflight must report all independent failures.
        checks.append(Check(name, "fail", f"check raised {type(error).__name__}"))


def _command_ok(command: list[str], *, cwd: Path) -> bool:
    return subprocess.run(
        command,
        cwd=cwd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0


def _canonical_record_sha256(record: dict[str, Any]) -> str:
    # This is the project record identity used when the JSONL whitespace changes.
    payload = json.dumps(record, ensure_ascii=False).encode("utf-8")
    return sha256_bytes(payload)


def _load_fixture_record(project_root: Path, config: dict[str, Any]) -> dict[str, Any]:
    fixture = config["fixture"]
    dataset_path = project_root / fixture["dataset_relpath"]
    lines = dataset_path.read_text(encoding="utf-8").splitlines()
    line_number = int(fixture["record_line"])
    if line_number < 1 or line_number > len(lines):
        raise ValueError("configured fixture line is outside the dataset file")
    value = json.loads(lines[line_number - 1])
    if not isinstance(value, dict):
        raise TypeError("configured fixture record is not an object")
    return value


def _static_checks(project_root: Path) -> list[Check]:
    phase3 = project_root / "phase3"
    checks: list[Check] = []

    def files_check() -> tuple[bool, str, Any]:
        missing = [name for name in REQUIRED_PHASE3_FILES if not (phase3 / name).is_file()]
        return (
            not missing,
            "all required Phase 3 inputs are present" if not missing else "required Phase 3 inputs are missing",
            {"required_count": len(REQUIRED_PHASE3_FILES), "missing": missing},
        )

    _run_check(checks, "phase3_required_files", files_check)

    def json_inputs_check() -> tuple[bool, str, Any]:
        paths = (
            phase3 / "configs/member1-smoke.json",
            phase3 / "configs/model-snapshot-manifest.json",
            phase3 / "configs/source-lock.json",
            phase3 / "contracts/member1-report.schema.json",
            phase3 / "contracts/model-snapshot-report.schema.json",
            phase3 / "contracts/handoff-verification.schema.json",
            phase3 / "contracts/preflight-report.schema.json",
            phase3 / "contracts/smoke-report.schema.json",
            phase3 / "contracts/technical-fixture.schema.json",
        )
        loaded = [read_json(path) for path in paths]
        valid = all(isinstance(value, dict) for value in loaded)
        return valid, "configuration and contract JSON parses", {"file_count": len(paths)}

    _run_check(checks, "phase3_json_inputs", json_inputs_check)

    def branch_check() -> tuple[bool, str, Any]:
        branch = git_output(project_root, "branch", "--show-current")
        return branch == EXPECTED_BRANCH, "Member 1 branch is selected", branch

    _run_check(checks, "git_branch", branch_check)

    def base_check() -> tuple[bool, str, Any]:
        ok = _command_ok(["git", "merge-base", "--is-ancestor", BASE_COMMIT, "HEAD"], cwd=project_root)
        return ok, "required tlia0262 base commit is an ancestor", BASE_COMMIT

    _run_check(checks, "git_base_commit", base_check)

    def project_revision_check() -> tuple[bool, str, Any]:
        head = git_output(project_root, "rev-parse", "HEAD")
        branch = git_output(project_root, "branch", "--show-current")
        phase3_status = git_output(
            project_root,
            "status",
            "--porcelain",
            "--untracked-files=all",
            "--",
            "phase3",
        )
        tracked = git_output(project_root, "ls-files", "--", "phase3")
        phase3_clean = not phase3_status and bool(tracked)
        observed = {
            "head": head,
            "branch": branch,
            "phase3_clean": phase3_clean,
        }
        return (
            phase3_clean,
            "Phase 3 inputs are tracked and the committed tree and index are clean",
            observed,
        )

    _run_check(checks, "phase3_project_revision", project_revision_check)

    def patch_check() -> tuple[bool, str, Any]:
        lock = read_json(phase3 / "configs/source-lock.json")
        patch_relpath = str(lock["upstream"]["patch"])
        patch_path = project_root / patch_relpath
        actual = sha256_file(patch_path)
        expected = str(lock["upstream"]["patch_sha256"])
        return actual == expected, "upstream compatibility patch hash matches the lock", actual

    _run_check(checks, "upstream_patch_sha256", patch_check)

    def model_manifest_check() -> tuple[bool, str, Any]:
        lock = read_json(phase3 / "configs/source-lock.json")["model"]
        manifest_path = project_root / lock["snapshot_manifest"]
        manifest = read_json(manifest_path)
        observed = {
            "manifest_sha256": sha256_file(manifest_path),
            "repository": manifest.get("repository"),
            "revision": manifest.get("revision"),
            "file_count": manifest.get("file_count"),
            "total_bytes": manifest.get("total_bytes"),
        }
        expected = {
            "manifest_sha256": lock["snapshot_manifest_sha256"],
            "repository": lock["repository"],
            "revision": lock["revision"],
            "file_count": lock["snapshot_file_count"],
            "total_bytes": lock["snapshot_total_bytes"],
        }
        return observed == expected, "model snapshot manifest matches the source lock", observed

    _run_check(checks, "model_snapshot_manifest_lock", model_manifest_check)

    def fixture_check() -> tuple[bool, str, Any]:
        config = read_json(phase3 / "configs/member1-smoke.json")
        fixture = config["fixture"]
        record = _load_fixture_record(project_root, config)
        observed = {
            "source_image_id": record.get("source_image_id"),
            "emotion": record.get("emotion"),
            "split": record.get("split"),
            "record_sha256": _canonical_record_sha256(record),
            "annotation_sha256": (record.get("provenance") or {}).get("annotation_sha256"),
            "image_sha256": (record.get("provenance") or {}).get("image_sha256"),
        }
        expected = {
            "source_image_id": fixture["source_image_id"],
            "emotion": fixture["emotion"],
            "split": "train",
            "record_sha256": fixture["record_sha256"],
            "annotation_sha256": fixture["annotation_sha256"],
            "image_sha256": fixture["image_sha256"],
        }
        complete = bool(record.get("instruction")) and bool(record.get("target_response"))
        return observed == expected and complete, "fixed source record matches all locked identities", observed

    _run_check(checks, "technical_fixture_source", fixture_check)

    def scope_check() -> tuple[bool, str, Any]:
        config = read_json(phase3 / "configs/member1-smoke.json")
        policy = config["policy"]
        expected = {
            "optimizer_step_performed": False,
            "adapter_saved": False,
            "full_dataset_ready": False,
            "formal_training_authorized": False,
        }
        return policy == expected, "Member 1 stop-line policy is locked", policy

    _run_check(checks, "member1_scope_policy", scope_check)
    return checks


def _resolve_checkout(upstream_dir: Path, name: str) -> Path:
    if upstream_dir.name == name and (upstream_dir / ".git").exists():
        return upstream_dir
    return upstream_dir / name


def _nearest_existing(path: Path) -> Path:
    candidate = path.resolve()
    while not candidate.exists() and candidate.parent != candidate:
        candidate = candidate.parent
    return candidate


def _requirement_pins(requirements_path: Path) -> dict[str, str]:
    pins: dict[str, str] = {}
    for raw_line in requirements_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line.startswith("--") or "==" not in line:
            continue
        name, version = line.split("==", 1)
        pins[name.strip().lower().replace("_", "-")] = version.strip()
    return pins


def _installed_version(distribution: str) -> str | None:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def _runtime_checks(
    project_root: Path,
    runtime_root: Path,
    model_path: Path,
    upstream_dir: Path,
) -> list[Check]:
    phase3 = project_root / "phase3"
    source_lock = read_json(phase3 / "configs/source-lock.json")
    smoke_config = read_json(phase3 / "configs/member1-smoke.json")
    checks = _static_checks(project_root)

    def qwen_check() -> tuple[bool, str, Any]:
        checkout = _resolve_checkout(upstream_dir, "Qwen-VL-finetune")
        revision = git_output(checkout, "rev-parse", "HEAD")
        expected = source_lock["upstream"]["commit"]
        patch_path = project_root / source_lock["upstream"]["patch"]
        patch_observation = verify_exact_git_patch(checkout, patch_path.resolve())
        observed = {"revision": revision, **patch_observation}
        return (
            revision == expected,
            "Qwen-VL checkout and exact working-tree diff match the source lock",
            observed,
        )

    _run_check(checks, "qwen_upstream", qwen_check)

    def venus_check() -> tuple[bool, str, Any]:
        checkout = _resolve_checkout(upstream_dir, "Venus_CVPR2026")
        revision = git_output(checkout, "rev-parse", "HEAD")
        expected = source_lock["venus_source"]["commit"]
        clean = not git_output(checkout, "status", "--porcelain", "--untracked-files=all")
        return revision == expected and clean, "Venus source checkout matches the lock", {
            "revision": revision,
            "working_tree_clean": clean,
        }

    _run_check(checks, "venus_upstream", venus_check)

    def model_revision_check() -> tuple[bool, str, Any]:
        model_lock = source_lock["model"]
        observed = verify_model_provenance(model_path, model_lock)
        return True, "model source provenance exactly matches the source lock", observed

    _run_check(checks, "model_revision", model_revision_check)

    def model_weights_check() -> tuple[bool, str, Any]:
        model = source_lock["model"]
        manifest_path = project_root / model["snapshot_manifest"]
        observed = verify_snapshot_manifest(
            model_path,
            manifest_path,
            expected_manifest_sha256=model["snapshot_manifest_sha256"],
            expected_repository=model["repository"],
            expected_revision=model["revision"],
        )
        index_path = model_path / model["index_file"]
        index = read_json(index_path)
        weight_map = index.get("weight_map") if isinstance(index, dict) else None
        shard_names = sorted(set(weight_map.values())) if isinstance(weight_map, dict) else []
        safe_names = all(
            isinstance(name, str) and Path(name).name == name for name in shard_names
        )
        shards = [model_path / name for name in shard_names] if safe_names else []
        files_present = bool(shards) and all(
            path.is_file() and path.stat().st_size > 0 for path in shards
        )
        total_bytes = sum(path.stat().st_size for path in shards if path.is_file())
        index_exists = index_path.is_file()
        observed.update({
            "shard_count": len(shards),
            "shard_total_bytes": total_bytes,
            "index_present": index_exists,
            "index_names_safe": safe_names,
            "all_shards_present": files_present,
        })
        expected = {
            "manifest_sha256": model["snapshot_manifest_sha256"],
            "file_count": int(model["snapshot_file_count"]),
            "total_bytes": int(model["snapshot_total_bytes"]),
            "shard_count": int(model["weight_shard_count"]),
            "shard_total_bytes": int(model["weight_shard_total_bytes"]),
            "index_present": True,
            "index_names_safe": True,
            "all_shards_present": True,
        }
        return observed == expected, "every pinned model file and shard matches the snapshot manifest", observed

    _run_check(checks, "model_weights", model_weights_check)

    def python_check() -> tuple[bool, str, Any]:
        expected = (phase3 / "environment/python-version.txt").read_text(encoding="utf-8").strip()
        observed = platform.python_version()
        return observed == expected, "Python version matches the environment lock", observed

    _run_check(checks, "python_version", python_check)

    def packages_check() -> tuple[bool, str, Any]:
        pins = _requirement_pins(phase3 / "environment/requirements.txt")
        mismatches: dict[str, dict[str, str | None]] = {}
        for name, expected in pins.items():
            observed = _installed_version(name)
            if observed != expected:
                mismatches[name] = {"expected": expected, "observed": observed}
        return not mismatches, "installed package versions match the lock", {
            "pin_count": len(pins),
            "mismatches": mismatches,
        }

    _run_check(checks, "python_packages", packages_check)

    def pip_check() -> tuple[bool, str, Any]:
        completed = subprocess.run(
            [sys.executable, "-m", "pip", "check"],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        return completed.returncode == 0, "pip dependency consistency check passes", {
            "return_code": completed.returncode,
        }

    _run_check(checks, "pip_check", pip_check)

    def cuda_check() -> tuple[bool, str, Any]:
        import torch

        minimum_total = int(smoke_config["runtime"]["minimum_gpu_memory_bytes"])
        minimum_free = int(
            smoke_config["runtime"]["minimum_free_gpu_memory_bytes"]
        )
        available = bool(torch.cuda.is_available())
        if not available:
            return False, "CUDA, BF16, and total/free GPU memory meet the runtime gate", {
                "cuda_available": False,
                "bf16_supported": False,
                "device_count": 0,
                "total_memory_bytes": 0,
                "free_memory_bytes": 0,
                "minimum_total_memory_bytes": minimum_total,
                "minimum_free_memory_bytes": minimum_free,
            }
        index = torch.cuda.current_device()
        torch.cuda.empty_cache()
        properties = torch.cuda.get_device_properties(index)
        free_memory, total_memory = torch.cuda.mem_get_info(index)
        bf16_supported = bool(torch.cuda.is_bf16_supported())
        observed = {
            "cuda_available": True,
            "bf16_supported": bf16_supported,
            "device_count": torch.cuda.device_count(),
            "device_name": properties.name,
            "total_memory_bytes": int(total_memory),
            "free_memory_bytes": int(free_memory),
            "minimum_total_memory_bytes": minimum_total,
            "minimum_free_memory_bytes": minimum_free,
        }
        enough_memory = _gpu_memory_gate(
            int(total_memory),
            int(free_memory),
            minimum_total,
            minimum_free,
        )
        return (
            bf16_supported and enough_memory,
            "CUDA, BF16, and total/free GPU memory meet the runtime gate",
            observed,
        )

    _run_check(checks, "cuda_bf16_gpu", cuda_check)

    def disk_check() -> tuple[bool, str, Any]:
        usage = shutil.disk_usage(_nearest_existing(runtime_root))
        minimum = int(smoke_config["runtime"]["minimum_free_disk_bytes"])
        return usage.free >= minimum, "runtime filesystem has the required free space", {
            "free_bytes": usage.free,
            "minimum_free_bytes": minimum,
        }

    _run_check(checks, "runtime_free_disk", disk_check)

    def layout_check() -> tuple[bool, str, Any]:
        required = ("envs", "upstream", "cache", "data", "smoke", "members", "runs", "reports", "locks")
        missing = [name for name in required if not (runtime_root / name).is_dir()]
        return not missing, "shared runtime layout is present", {"missing": missing}

    _run_check(checks, "runtime_layout", layout_check)
    return checks


def build_report(
    *,
    mode: str,
    project_root: Path,
    runtime_root: Path,
    model_path: Path,
    upstream_dir: Path,
) -> dict[str, Any]:
    checks = (
        _static_checks(project_root)
        if mode == "static"
        else _runtime_checks(project_root, runtime_root, model_path, upstream_dir)
    )
    return {
        "schema_version": 1,
        "kind": "phase3_preflight",
        "generated_at_utc": utc_now(),
        "mode": mode,
        "passed": all(check.status != "fail" for check in checks),
        "checks": [check.as_json() for check in checks],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("static", "runtime"), required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--runtime-root", "--workspace-root", dest="runtime_root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--upstream-dir", "--upstream-root", dest="upstream_dir", type=Path, required=True)
    parser.add_argument("--report", "--output", dest="report", type=Path, required=True)
    args = parser.parse_args(argv)

    report = build_report(
        mode=args.mode,
        project_root=args.project_root.resolve(),
        runtime_root=args.runtime_root.resolve(),
        model_path=args.model_path.resolve(),
        upstream_dir=args.upstream_dir.resolve(),
    )
    assert_report_is_redacted(report)
    from jsonschema import Draft202012Validator, FormatChecker

    schema = read_json(args.project_root.resolve() / "phase3/contracts/preflight-report.schema.json")
    Draft202012Validator(schema, format_checker=FormatChecker()).validate(report)
    write_json_atomic(args.report, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
