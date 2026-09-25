#!/usr/bin/env python3
"""Validate and aggregate redacted Phase 3 Member 1 machine-readable reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from common import (
    assert_report_is_redacted,
    expected_model_provenance,
    git_output,
    read_json,
    sha256_bytes,
    sha256_file,
    utc_now,
    write_json_atomic,
)


SCHEMAS = {
    "preflight": "preflight-report.schema.json",
    "smoke": "smoke-report.schema.json",
    "member1": "member1-report.schema.json",
    "model_snapshot": "model-snapshot-report.schema.json",
    "fixture": "technical-fixture.schema.json",
    "verification": "handoff-verification.schema.json",
}
KIND_TO_SCHEMA = {
    "phase3_preflight": "preflight",
    "phase3_smoke_backward": "smoke",
    "phase3_member1_handoff": "member1",
    "phase3_model_snapshot": "model_snapshot",
    "phase3_handoff_verification": "verification",
}


def _validate_schema(value: Any, schema_kind: str, project_root: Path) -> None:
    try:
        schema_name = SCHEMAS[schema_kind]
    except KeyError as error:
        raise ValueError(f"unknown schema kind: {schema_kind}") from error
    from jsonschema import Draft202012Validator, FormatChecker

    schema = read_json(project_root / "phase3/contracts" / schema_name)
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    errors = sorted(validator.iter_errors(value), key=lambda item: list(item.absolute_path))
    if errors:
        first = errors[0]
        location = "$" + "".join(f"[{part!r}]" for part in first.absolute_path)
        raise ValueError(f"{schema_kind} schema violation at {location}: {first.message}")


def _validate_semantics(value: Any, schema_kind: str, project_root: Path) -> None:
    if schema_kind == "preflight":
        failures = [check for check in value["checks"] if check["status"] == "fail"]
        if value["passed"] != (not failures):
            raise ValueError("preflight passed flag is inconsistent with its checks")
        if value["mode"] == "runtime" and value["passed"]:
            smoke_lock = read_json(project_root / "phase3/configs/member1-smoke.json")
            cuda_checks = [
                check for check in value["checks"] if check["name"] == "cuda_bf16_gpu"
            ]
            if len(cuda_checks) != 1 or cuda_checks[0]["status"] != "pass":
                raise ValueError("passing runtime preflight has no unique CUDA memory check")
            observed = cuda_checks[0].get("observed")
            if not isinstance(observed, dict):
                raise ValueError("runtime CUDA check has no machine-readable observation")
            minimum_total = int(smoke_lock["runtime"]["minimum_gpu_memory_bytes"])
            minimum_free = int(
                smoke_lock["runtime"]["minimum_free_gpu_memory_bytes"]
            )
            if (
                observed.get("cuda_available") is not True
                or observed.get("bf16_supported") is not True
                or observed.get("minimum_total_memory_bytes") != minimum_total
                or observed.get("minimum_free_memory_bytes") != minimum_free
                or int(observed.get("total_memory_bytes", -1)) < minimum_total
                or int(observed.get("free_memory_bytes", -1)) < minimum_free
            ):
                raise ValueError("runtime CUDA total/free memory evidence fails the lock")
    elif schema_kind == "smoke":
        lora = value["lora"]
        expected_pass = (
            value["forward_passed"]
            and value["backward_passed"]
            and bool(lora["nonzero_gradient_parameter_names"])
            and not value["errors"]
        )
        if value["passed"] != expected_pass:
            raise ValueError("smoke passed flag is inconsistent with its evidence")
        if value["optimizer_step_performed"] or value["adapter_saved"]:
            raise ValueError("smoke report crosses the Member 1 stop line")
        if value["passed"]:
            smoke_lock = read_json(project_root / "phase3/configs/member1-smoke.json")
            source_lock = read_json(project_root / "phase3/configs/source-lock.json")
            expected_fixture = {
                "id": smoke_lock["fixture"]["source_image_id"],
                "image_sha256": smoke_lock["fixture"]["image_sha256"],
                "dataset_record_sha256": smoke_lock["fixture"]["record_sha256"],
            }
            fixture = value.get("fixture")
            if not isinstance(fixture, dict) or any(
                fixture.get(key) != expected for key, expected in expected_fixture.items()
            ):
                raise ValueError("smoke fixture identity differs from the locked fixture")
            expected_model = {
                "repository": source_lock["model"]["repository"],
                "revision": source_lock["model"]["revision"],
                "base_dtype": smoke_lock["runtime"]["dtype"],
                "forward_input_dtype": smoke_lock["runtime"]["dtype"],
            }
            if any(value["model"].get(key) != expected for key, expected in expected_model.items()):
                raise ValueError("smoke model identity or dtype differs from the source lock")
            expected_lora = smoke_lock["lora"]
            if (
                value["lora"].get("adapter_dtype") != "float32"
                or len(value["lora"]["matched_target_names"])
                != int(expected_lora["expected_matched_module_count"])
                or value["lora"]["trainable_parameter_count"]
                != int(expected_lora["expected_trainable_parameter_count"])
            ):
                raise ValueError("smoke LoRA dtype or locked architecture evidence differs")
            if value["batch"].get("loss_dtype") not in {"bfloat16", "float32"}:
                raise ValueError("smoke report has no supported observed loss dtype")
            minimum_total = int(smoke_lock["runtime"]["minimum_gpu_memory_bytes"])
            minimum_free = int(
                smoke_lock["runtime"]["minimum_free_gpu_memory_bytes"]
            )
            memory = value["memory"]
            if (
                memory["minimum_total_gpu_memory_bytes"] != minimum_total
                or memory["minimum_free_gpu_memory_bytes"] != minimum_free
                or memory["total_gpu_memory_bytes"] < minimum_total
                or memory["free_gpu_memory_bytes"] < minimum_free
            ):
                raise ValueError("smoke report GPU total/free memory evidence fails the lock")
    elif schema_kind == "model_snapshot":
        source_lock = read_json(project_root / "phase3/configs/source-lock.json")
        model_lock = source_lock["model"]
        expected_source = {
            "repository": model_lock["repository"],
            "revision": model_lock["revision"],
            "weight_format": model_lock["weight_format"],
            "snapshot_manifest_sha256": model_lock["snapshot_manifest_sha256"],
        }
        if value["source_lock"] != expected_source:
            raise ValueError("model snapshot report source identity differs from the lock")

        sealed = value["sealed"]
        evidence_complete = (
            isinstance(value["snapshot"], dict)
            and isinstance(value["provenance"], dict)
            and sealed["applied"]
            and sealed["verified"]
            and sealed["manifest_files_read_only"]
            and sealed["marker_read_only"]
            and sealed["model_root_read_only"]
            and not value["errors"]
        )
        if value["passed"] != evidence_complete:
            raise ValueError("model snapshot passed flag is inconsistent with its evidence")

        if value["passed"]:
            expected_snapshot = {
                "manifest_sha256": model_lock["snapshot_manifest_sha256"],
                "file_count": int(model_lock["snapshot_file_count"]),
                "total_bytes": int(model_lock["snapshot_total_bytes"]),
            }
            if value["snapshot"] != expected_snapshot:
                raise ValueError("model snapshot observation differs from the source lock")

            marker = expected_model_provenance(model_lock)
            marker_payload = (
                json.dumps(marker, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
            ).encode("utf-8")
            expected_provenance = {
                **marker,
                "marker_sha256": sha256_bytes(marker_payload),
            }
            if value["provenance"] != expected_provenance:
                raise ValueError("model provenance evidence differs from the source lock")
            if sealed["manifest_file_count"] != expected_snapshot["file_count"]:
                raise ValueError("model seal file count differs from the source lock")
    elif schema_kind == "member1":
        expected_smoke = value["forward_passed"] and value["backward_passed"]
        if value["technical_smoke_passed"] != expected_smoke:
            raise ValueError("handoff technical_smoke_passed flag is inconsistent")
        if value["optimizer_step_performed"] or value["adapter_saved"]:
            raise ValueError("handoff report crosses the Member 1 stop line")
        smoke_lock = read_json(project_root / "phase3/configs/member1-smoke.json")
        source_lock = read_json(project_root / "phase3/configs/source-lock.json")
        expected_fixture = {
            "id": smoke_lock["fixture"]["source_image_id"],
            "image_sha256": smoke_lock["fixture"]["image_sha256"],
            "dataset_record_sha256": smoke_lock["fixture"]["record_sha256"],
        }
        if any(value["fixture"].get(key) != expected for key, expected in expected_fixture.items()):
            raise ValueError("handoff fixture identity differs from the locked fixture")
        expected_source = {
            "project_commit": git_output(project_root, "rev-parse", "HEAD"),
            "qwen_commit": source_lock["upstream"]["commit"],
            "qwen_patch_sha256": source_lock["upstream"]["patch_sha256"],
            "venus_commit": source_lock["venus_source"]["commit"],
            "model_repository": source_lock["model"]["repository"],
            "model_revision": source_lock["model"]["revision"],
            "model_snapshot_manifest_sha256": source_lock["model"]["snapshot_manifest_sha256"],
        }
        if value["source_lock"] != expected_source:
            raise ValueError("handoff source identities differ from source-lock.json")
        if git_output(project_root, "status", "--porcelain", "--", "phase3"):
            raise ValueError("handoff validation requires a clean, tracked phase3 tree")


def validate_value(value: Any, schema_kind: str, project_root: Path) -> None:
    _validate_schema(value, schema_kind, project_root)
    _validate_semantics(value, schema_kind, project_root)
    assert_report_is_redacted(value)


def _infer_schema_kind(value: Any) -> str:
    if isinstance(value, list):
        return "fixture"
    if not isinstance(value, dict):
        raise ValueError("report must be a JSON object or the one-record fixture array")
    kind = value.get("kind")
    try:
        return KIND_TO_SCHEMA[str(kind)]
    except KeyError as error:
        raise ValueError(f"cannot infer schema from report kind: {kind!r}") from error


def aggregate(
    *,
    preflight_path: Path,
    smoke_path: Path,
    project_root: Path,
    member_id: str,
    multi_user_status: str,
    fixture_manifest_path: Path,
    environment_freeze_path: Path,
) -> dict[str, Any]:
    if multi_user_status != "coordination_required":
        raise ValueError(
            "local scripts cannot authenticate a RunPod account; multi-user status must remain coordination_required"
        )
    preflight = read_json(preflight_path)
    smoke = read_json(smoke_path)
    validate_value(preflight, "preflight", project_root)
    validate_value(smoke, "smoke", project_root)
    if preflight["mode"] != "runtime":
        raise ValueError("Member 1 handoff requires a runtime preflight report")
    if smoke["member_id"] != member_id:
        raise ValueError("smoke report member_id does not match the aggregate member_id")
    source_lock = read_json(project_root / "phase3/configs/source-lock.json")
    smoke_config = read_json(project_root / "phase3/configs/member1-smoke.json")
    expected_model = {
        "repository": source_lock["model"]["repository"],
        "revision": source_lock["model"]["revision"],
        "base_dtype": smoke_config["runtime"]["dtype"],
        "forward_input_dtype": smoke_config["runtime"]["dtype"],
    }
    observed_model = {key: smoke["model"].get(key) for key in expected_model}
    if observed_model != expected_model:
        raise ValueError("smoke report model identity or dtype does not match the locks")
    expected_lora = smoke_config["lora"]
    if (
        smoke["lora"]["rank"] != expected_lora["rank"]
        or smoke["lora"]["alpha"] != expected_lora["alpha"]
        or smoke["lora"]["target_modules"] != expected_lora["target_modules"]
        or smoke["lora"]["adapter_dtype"] != "float32"
        or len(smoke["lora"]["matched_target_names"])
        != int(expected_lora["expected_matched_module_count"])
        or smoke["lora"]["trainable_parameter_count"]
        != int(expected_lora["expected_trainable_parameter_count"])
    ):
        raise ValueError("smoke report LoRA probe settings do not match member1-smoke.json")
    if smoke["batch"].get("loss_dtype") not in {"bfloat16", "float32"}:
        raise ValueError("smoke report has no supported observed loss dtype")
    minimum_total = int(smoke_config["runtime"]["minimum_gpu_memory_bytes"])
    minimum_free = int(smoke_config["runtime"]["minimum_free_gpu_memory_bytes"])
    smoke_memory = smoke["memory"]
    if (
        smoke_memory["minimum_total_gpu_memory_bytes"] != minimum_total
        or smoke_memory["minimum_free_gpu_memory_bytes"] != minimum_free
        or smoke_memory["total_gpu_memory_bytes"] < minimum_total
        or smoke_memory["free_gpu_memory_bytes"] < minimum_free
    ):
        raise ValueError("smoke report GPU total/free memory evidence fails the lock")
    fixture_lock = smoke_config["fixture"]
    expected_fixture = {
        "id": fixture_lock["source_image_id"],
        "image_sha256": fixture_lock["image_sha256"],
        "dataset_record_sha256": fixture_lock["record_sha256"],
    }
    smoke_fixture = smoke.get("fixture")
    if not isinstance(smoke_fixture, dict):
        raise ValueError("smoke report has no fixture identity evidence")
    observed_fixture = {key: smoke_fixture.get(key) for key in expected_fixture}
    if observed_fixture != expected_fixture:
        raise ValueError("smoke report fixture identity does not match member1-smoke.json")
    fixture_sha256 = smoke_fixture.get("fixture_sha256")
    if not isinstance(fixture_sha256, str) or len(fixture_sha256) != 64:
        raise ValueError("passing smoke report has no valid fixture SHA-256")
    fixture_manifest = read_json(fixture_manifest_path)
    expected_manifest_fields = {
        "source_image_id": expected_fixture["id"],
        "fixture_sha256": fixture_sha256,
        "image_sha256": expected_fixture["image_sha256"],
        "dataset_record_sha256": expected_fixture["dataset_record_sha256"],
        "full_dataset_ready": False,
        "formal_training_authorized": False,
    }
    if not isinstance(fixture_manifest, dict) or any(
        fixture_manifest.get(key) != expected for key, expected in expected_manifest_fields.items()
    ):
        raise ValueError("technical fixture manifest differs from the smoke evidence")
    if not environment_freeze_path.is_file() or not environment_freeze_path.read_text(
        encoding="utf-8"
    ).strip():
        raise ValueError("captured environment freeze is missing or empty")

    revision_checks = [
        check
        for check in preflight["checks"]
        if check.get("name") == "phase3_project_revision" and check.get("status") == "pass"
    ]
    if len(revision_checks) != 1 or not isinstance(revision_checks[0].get("observed"), dict):
        raise ValueError("runtime preflight has no unique passing Phase 3 project revision")
    project_commit = revision_checks[0]["observed"].get("head")
    if project_commit != git_output(project_root, "rev-parse", "HEAD"):
        raise ValueError("runtime preflight was generated from a different project commit")
    if git_output(project_root, "status", "--porcelain", "--", "phase3"):
        raise ValueError("aggregate requires a clean, tracked phase3 tree")

    environment_ready = bool(preflight["passed"])
    technical_smoke_passed = bool(
        smoke["passed"] and smoke["forward_passed"] and smoke["backward_passed"]
    )
    report = {
        "schema_version": 1,
        "kind": "phase3_member1_handoff",
        "generated_at_utc": utc_now(),
        "member_id": member_id,
        "forward_passed": bool(smoke["forward_passed"]),
        "backward_passed": bool(smoke["backward_passed"]),
        "optimizer_step_performed": False,
        "adapter_saved": False,
        "member1_environment_ready": environment_ready,
        "technical_smoke_passed": technical_smoke_passed,
        "full_dataset_ready": False,
        "formal_training_authorized": False,
        "fixture": {
            "id": smoke_fixture["id"],
            "fixture_sha256": fixture_sha256,
            "fixture_manifest_sha256": sha256_file(fixture_manifest_path),
            "image_sha256": smoke_fixture["image_sha256"],
            "dataset_record_sha256": smoke_fixture["dataset_record_sha256"],
        },
        "source_lock": {
            "project_commit": project_commit,
            "qwen_commit": source_lock["upstream"]["commit"],
            "qwen_patch_sha256": source_lock["upstream"]["patch_sha256"],
            "venus_commit": source_lock["venus_source"]["commit"],
            "model_repository": source_lock["model"]["repository"],
            "model_revision": source_lock["model"]["revision"],
            "model_snapshot_manifest_sha256": source_lock["model"]["snapshot_manifest_sha256"],
        },
        "multi_user": {
            "status": multi_user_status,
            "filesystem_lock_used": True,
            "member_scoped_layout": True,
            "shared_host_training_authorized": False,
        },
        "inputs": {
            "preflight_report_sha256": sha256_file(preflight_path),
            "smoke_report_sha256": sha256_file(smoke_path),
            "environment_freeze_sha256": sha256_file(environment_freeze_path),
        },
    }
    validate_value(report, "member1", project_root)
    return report


def _aggregate_command(args: argparse.Namespace) -> int:
    report = aggregate(
        preflight_path=args.preflight,
        smoke_path=args.smoke,
        project_root=args.project_root.resolve(),
        member_id=args.member_id,
        multi_user_status=args.multi_user_status,
        fixture_manifest_path=args.fixture_manifest,
        environment_freeze_path=args.environment_freeze,
    )
    write_json_atomic(args.output, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["member1_environment_ready"] and report["technical_smoke_passed"] else 1


def _validate_command(args: argparse.Namespace) -> int:
    value = read_json(args.input)
    schema_kind = _infer_schema_kind(value) if args.kind == "auto" else args.kind
    validate_value(value, schema_kind, args.project_root.resolve())
    result = {
        "schema_version": 1,
        "kind": "phase3_report_validation",
        "generated_at_utc": utc_now(),
        "valid": True,
        "validated_schema": schema_kind,
        "input_sha256": sha256_file(args.input),
    }
    assert_report_is_redacted(result)
    if args.output:
        write_json_atomic(args.output, result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def _checksums_command(args: argparse.Namespace) -> int:
    if not args.input:
        raise ValueError("at least one --input is required")
    seen_names: set[str] = set()
    entries: list[dict[str, str]] = []
    for path in args.input:
        name = path.name
        if name in seen_names:
            raise ValueError(f"duplicate checksum filename: {name}")
        seen_names.add(name)
        entries.append({"name": name, "sha256": sha256_file(path)})
    result = {
        "schema_version": 1,
        "kind": "phase3_checksums",
        "generated_at_utc": utc_now(),
        "files": sorted(entries, key=lambda item: item["name"]),
    }
    assert_report_is_redacted(result)
    write_json_atomic(args.output, result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    aggregate_parser = subparsers.add_parser("aggregate", help="create the Member 1 handoff report")
    aggregate_parser.add_argument("--preflight", type=Path, required=True)
    aggregate_parser.add_argument("--smoke", type=Path, required=True)
    aggregate_parser.add_argument("--output", type=Path, required=True)
    aggregate_parser.add_argument("--project-root", type=Path, default=Path.cwd())
    aggregate_parser.add_argument("--fixture-manifest", type=Path, required=True)
    aggregate_parser.add_argument("--environment-freeze", type=Path, required=True)
    aggregate_parser.add_argument("--member-id", default="member1")
    aggregate_parser.add_argument(
        "--multi-user-status",
        choices=("coordination_required",),
        default="coordination_required",
    )
    aggregate_parser.set_defaults(handler=_aggregate_command)

    validate_parser = subparsers.add_parser("validate", help="validate one report or fixture")
    validate_parser.add_argument("--input", type=Path, required=True)
    validate_parser.add_argument("--kind", choices=("auto", *SCHEMAS), default="auto")
    validate_parser.add_argument("--project-root", type=Path, default=Path.cwd())
    validate_parser.add_argument("--output", type=Path)
    validate_parser.set_defaults(handler=_validate_command)

    checksums_parser = subparsers.add_parser("checksums", help="write a portable checksum manifest")
    checksums_parser.add_argument("--input", action="append", type=Path, required=True)
    checksums_parser.add_argument("--output", type=Path, required=True)
    checksums_parser.set_defaults(handler=_checksums_command)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
