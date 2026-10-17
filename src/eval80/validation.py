"""Validate one completed Eval80-v1 Stage 1 inference attempt."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from .core import (
    build_inference_manifest,
    read_csv,
    read_selection_manifest,
    sha256_file,
    sha256_text_file,
    validate_human_reviews,
    validate_selection_manifest,
)
from .inference import _condition_prompt, _expected_adapter_identity, _verify_adapter_files


def _read_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSONL at line {line_number}: {error}") from error
    return records


def validate_run(
    run_dir: Path,
    inference_manifest: dict,
    config: dict,
    condition: str,
    *,
    adapter_evidence_path: Path | None = None,
    b1_run_config_path: Path | None = None,
    adapter_path: Path | None = None,
) -> dict:
    errors: list[str] = []
    records_path = run_dir / "records.jsonl"
    summary_path = run_dir / "summary.json"
    run_config_path = run_dir / "run_config.json"
    for path in (records_path, summary_path, run_config_path, run_dir / "environment.json"):
        if not path.is_file():
            errors.append(f"Missing required run artifact: {path.name}")
    if errors:
        return {"valid": False, "errors": errors}

    records = _read_jsonl(records_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
    expected_by_id = {
        str(record["blind_id"]): str(record["image_sha256"])
        for record in inference_manifest["records"]
    }
    expected_ids = set(expected_by_id)

    if len(records) != 80:
        errors.append(f"Expected 80 records, found {len(records)}.")
    actual_ids = [str(record.get("blind_id", "")) for record in records]
    duplicate_ids = sorted(value for value, count in Counter(actual_ids).items() if count > 1)
    if duplicate_ids:
        errors.append(f"Duplicate blind IDs: {', '.join(duplicate_ids)}")
    if set(actual_ids) != expected_ids:
        errors.append(
            f"Blind ID set differs. Missing={sorted(expected_ids - set(actual_ids))}; "
            f"extra={sorted(set(actual_ids) - expected_ids)}"
        )

    forbidden_fields = {"emotion", "emotion_label", "source_image_id", "ground_truth"}
    empty_response_count = 0
    for record in records:
        blind_id = str(record.get("blind_id", "missing blind_id"))
        if record.get("stage") != "stage1":
            errors.append(f"{blind_id}: stage must be stage1.")
        if record.get("condition") != condition:
            errors.append(f"{blind_id}: condition must be {condition}.")
        if record.get("status") != "success":
            errors.append(f"{blind_id}: status is not success.")
        leaked = sorted(forbidden_fields.intersection(record))
        if leaked:
            errors.append(f"{blind_id}: private ground-truth fields leaked: {', '.join(leaked)}")
        if blind_id in expected_by_id and record.get("image_sha256") != expected_by_id[blind_id]:
            errors.append(f"{blind_id}: image SHA-256 differs from the inference manifest.")
        response = str(record.get("response", ""))
        if not response.strip():
            empty_response_count += 1
            errors.append(f"{blind_id}: response is empty.")
        response_hash = hashlib.sha256(response.encode("utf-8")).hexdigest()
        if record.get("response_sha256") != response_hash:
            errors.append(f"{blind_id}: response SHA-256 mismatch.")

    condition_config = config["conditions"][condition]
    expected_prompt = _condition_prompt(config, condition)
    expected_run_config = {
        "condition": condition,
        "prompt_version": condition_config["prompt_version"],
        "prompt": expected_prompt,
        "seed": config["seed"],
        "precision": config["precision"],
        "batch_size": config["batch_size"],
        "manifest_size": 80,
    }
    for key, expected in expected_run_config.items():
        if run_config.get(key) != expected:
            errors.append(
                f"run_config {key} mismatch: expected {expected!r}, found {run_config.get(key)!r}."
            )
    if summary.get("completed") is not True:
        errors.append("summary.completed must be true.")
    if summary.get("record_count") != 80 or summary.get("expected_count") != 80:
        errors.append("summary record counts must both be 80.")
    if summary.get("empty_response_count") != 0:
        errors.append("summary.empty_response_count must be zero.")
    if summary.get("records_sha256") != sha256_file(records_path):
        errors.append("summary records_sha256 differs from records.jsonl.")

    if condition == "C":
        if "adapter_path" in run_config:
            errors.append("C run_config must not reveal the private adapter path.")
        project_root = Path(__file__).resolve().parents[2]
        lock_path = project_root / "phase3/configs/source-lock.json"
        try:
            source_lock = json.loads(lock_path.read_text(encoding="utf-8"))["model"]
            snapshot_path = project_root / source_lock["snapshot_manifest"]
            snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
            expected_snapshot_receipt = {
                "source_lock_sha256": sha256_file(lock_path),
                "snapshot_manifest_sha256": sha256_file(snapshot_path),
                "verified_file_count": snapshot["file_count"],
                "verified_total_bytes": snapshot["total_bytes"],
            }
            if (expected_snapshot_receipt["snapshot_manifest_sha256"]
                    != source_lock["snapshot_manifest_sha256"]):
                errors.append("C source lock does not match model snapshot manifest.")
            if run_config.get("base_snapshot_verification") != expected_snapshot_receipt:
                errors.append("C base snapshot verification receipt differs from pinned source lock.")
        except (OSError, ValueError, KeyError, TypeError) as error:
            errors.append(f"C pinned base snapshot metadata invalid: {error}")
        if adapter_evidence_path is None or b1_run_config_path is None:
            errors.append("C validation requires adapter evidence and the B1 run_config.")
        else:
            try:
                expected_identity = _expected_adapter_identity(
                    adapter_evidence_path, config["model"]
                )
            except (OSError, ValueError, KeyError, TypeError) as error:
                errors.append(f"C adapter evidence invalid: {error}")
            else:
                recorded_identity = run_config.get("adapter_identity")
                if not isinstance(recorded_identity, dict):
                    errors.append("C run_config is missing adapter identity.")
                else:
                    for key in (
                        "evidence_sha256", "selected_checkpoint", "formal_config_sha256",
                        "base_revision", "adapter_files", "acceptance_status",
                    ):
                        if recorded_identity.get(key) != expected_identity[key]:
                            errors.append(f"C adapter identity {key} differs from training evidence.")
                if adapter_path is not None:
                    try:
                        _verify_adapter_files(adapter_path, expected_identity)
                    except (OSError, ValueError) as error:
                        errors.append(f"C adapter rehash failed: {error}")
            try:
                b1_run_config = json.loads(b1_run_config_path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as error:
                errors.append(f"B1 run_config cannot be read: {error}")
            else:
                if b1_run_config.get("condition") != "B1":
                    errors.append("B1 reference run_config has the wrong condition.")
                for key in (
                    "prompt", "prompt_version", "seed", "precision", "batch_size",
                    "manifest_size", "generation_expected", "generation_actual",
                    "model_loading",
                ):
                    if run_config.get(key) != b1_run_config.get(key):
                        errors.append(f"C/B1 {key} differs.")
                for key in ("config", "scoring_handbook", "inference_manifest", "freeze_record"):
                    if (run_config.get("file_sha256", {}).get(key)
                            != b1_run_config.get("file_sha256", {}).get(key)):
                        errors.append(f"C/B1 {key} SHA-256 differs.")
                for key in ("repo_id", "revision", "weight_format"):
                    if (run_config.get("model_source", {}).get(key)
                            != b1_run_config.get("model_source", {}).get(key)):
                        errors.append(f"C/B1 base model {key} differs.")
        if run_config.get("prompt_source_condition") != "B1":
            errors.append("C run_config must name B1 as its prompt source.")
        if (run_config.get("rng_seed_position") != "before_shared_base_load"
                or run_config.get("adapter_load_rng_restored") is not True
                or run_config.get("adapter_load_rng_scope")
                != "cpu_and_all_visible_cuda_devices"):
            errors.append("C run_config does not verify adapter-load RNG restoration.")
        if run_config.get("generation_expected") != config["generation"]:
            errors.append("C generation settings differ from frozen Eval80 config.")
        for key, expected in config["generation"].items():
            if run_config.get("generation_actual", {}).get(key) != expected:
                errors.append(f"C actual generation {key} differs from frozen Eval80 config.")
        if run_config.get("model_loading") != config["model_loading"]:
            errors.append("C model-loading policy differs from frozen Eval80 config.")
        for key in ("repo_id", "revision", "weight_format"):
            if run_config.get("model_source", {}).get(key) != config["model"][key]:
                errors.append(f"C base model {key} differs from frozen Eval80 config.")
        devices = run_config.get("parameter_devices")
        if not isinstance(devices, list) or not devices or any(
            not isinstance(device, str) or not device.startswith("cuda") for device in devices
        ):
            errors.append("C run_config does not verify CUDA-only model placement.")
        for index, (record, expected) in enumerate(
            zip(records, inference_manifest["records"]), start=1
        ):
            if record.get("sequence_index") != index or record.get("blind_id") != expected["blind_id"]:
                errors.append(f"C record {index} differs from frozen manifest order.")
            if record.get("prompt_version") != condition_config["prompt_version"]:
                errors.append(f"C record {index} prompt version differs from B1.")
        if summary.get("condition") != "C":
            errors.append("C summary condition mismatch.")

    report = {
        "schema_version": 1,
        "valid": not errors,
        "condition": condition,
        "record_count": len(records),
        "expected_count": 80,
        "unique_blind_id_count": len(set(actual_ids)),
        "empty_response_count": empty_response_count,
        "records_sha256": sha256_file(records_path),
        "errors": errors,
    }
    if condition == "C":
        report["validation_scope"] = "technical_only"
        report["training_acceptance_verified"] = False
    return report


def validate_output_command(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--condition", choices=("A", "B0", "B1", "C"), required=True)
    parser.add_argument("--adapter-evidence", type=Path)
    parser.add_argument("--adapter-path", type=Path)
    parser.add_argument("--b1-run-config", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    config = json.loads(args.config.read_text(encoding="utf-8"))
    report = validate_run(
        args.run_dir, manifest, config, args.condition,
        adapter_evidence_path=args.adapter_evidence,
        b1_run_config_path=args.b1_run_config,
        adapter_path=args.adapter_path,
    )
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["valid"] else 1


def validate_review_command(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate Eval80-v1 selection integrity and both human-review passes."
    )
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--blind-review", type=Path, required=True)
    parser.add_argument("--reveal-review", type=Path, required=True)
    parser.add_argument("--skip-image-hashes", action="store_true")
    args = parser.parse_args(argv)

    selection = read_selection_manifest(args.selection_manifest)
    errors = validate_selection_manifest(
        selection,
        args.image_dir,
        verify_hashes=not args.skip_image_hashes,
    )
    review_errors, review_summary = validate_human_reviews(
        read_csv(args.blind_review),
        read_csv(args.reveal_review),
        selection,
    )
    errors.extend(review_errors)
    report = {
        "schema_version": 1,
        "valid": not errors,
        "selection_record_count": len(selection["records"]),
        "review_summary": review_summary,
        "error_count": len(errors),
        "errors": errors,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not errors else 1


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json_atomic(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def freeze_command(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate review and create the approved Eval80-v1 package."
    )
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--blind-review", type=Path, required=True)
    parser.add_argument("--reveal-review", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--handbook", type=Path, required=True)
    parser.add_argument("--reading-copy", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--approver-ids", nargs="+", required=True)
    parser.add_argument("--team-approved", action="store_true")
    args = parser.parse_args(argv)

    if not args.team_approved:
        raise RuntimeError("Refusing to freeze without explicit --team-approved confirmation.")
    if len(set(args.approver_ids)) != len(args.approver_ids):
        raise ValueError("--approver-ids must contain unique identifiers.")

    required_files = (
        args.selection_manifest,
        args.blind_review,
        args.reveal_review,
        args.config,
        args.handbook,
        args.reading_copy,
        args.protocol,
    )
    for path in required_files:
        if not path.is_file():
            raise FileNotFoundError(f"Required file is missing: {path}")

    handbook_text = args.handbook.read_text(encoding="utf-8")
    handbook_lines = {line.strip() for line in handbook_text.splitlines()}
    if "Status: frozen" not in handbook_lines:
        raise RuntimeError("The scoring handbook must contain the exact line 'Status: frozen'.")
    if "TODO" in handbook_text or "- [ ]" in handbook_text:
        raise RuntimeError("The scoring handbook still contains TODO fields or unchecked items.")
    reading_copy_text = args.reading_copy.read_text(encoding="utf-8")
    chinese_frozen_status = "\u72b6\u6001\uff1a\u5df2\u51bb\u7ed3"
    if chinese_frozen_status not in reading_copy_text.splitlines():
        raise RuntimeError("The Chinese scoring handbook is missing its required frozen-status line.")

    selection = read_selection_manifest(args.selection_manifest)
    errors = validate_selection_manifest(selection, args.image_dir, verify_hashes=True)
    review_errors, review_summary = validate_human_reviews(
        read_csv(args.blind_review),
        read_csv(args.reveal_review),
        selection,
    )
    errors.extend(review_errors)
    if errors:
        raise RuntimeError("Eval80 freeze validation failed:\n- " + "\n- ".join(errors))

    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    inference_manifest = build_inference_manifest(selection)
    inference_manifest_path = output_dir / "inference_manifest.json"
    _write_json_atomic(inference_manifest_path, inference_manifest)

    freeze_record = {
        "schema_version": 1,
        "protocol_version": "eval80-v1-2026-09-07",
        "status": "approved_and_frozen",
        "approved_at_utc": _utc_now(),
        "approver_ids": list(args.approver_ids),
        "team_approval_confirmed": True,
        "review_summary": review_summary,
        "inference_image_count": len(inference_manifest["records"]),
        "file_sha256": {
            "selection_manifest_private": sha256_file(args.selection_manifest),
            "blind_review": sha256_file(args.blind_review),
            "reveal_review_private": sha256_file(args.reveal_review),
            "config": sha256_text_file(args.config),
            "scoring_handbook": sha256_text_file(args.handbook),
            "scoring_handbook_chinese": sha256_text_file(args.reading_copy),
            "protocol": sha256_text_file(args.protocol),
            "inference_manifest": sha256_file(inference_manifest_path),
        },
    }
    _write_json_atomic(output_dir / "freeze_record.json", freeze_record)
    print(json.dumps(freeze_record, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(validate_output_command())
