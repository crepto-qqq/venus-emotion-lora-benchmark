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
    expected_run_config = {
        "condition": condition,
        "prompt_version": condition_config["prompt_version"],
        "prompt": condition_config["prompt"],
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

    return {
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


def validate_output_command(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--condition", choices=("A", "B0", "B1"), required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    config = json.loads(args.config.read_text(encoding="utf-8"))
    report = validate_run(args.run_dir, manifest, config, args.condition)
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
    if "状态：已冻结" not in reading_copy_text.splitlines():
        raise RuntimeError("The Chinese scoring handbook must contain the line '状态：已冻结'.")

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
