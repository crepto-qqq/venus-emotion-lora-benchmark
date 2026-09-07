"""Validation and freezing helpers for the private Eval80-v1 package."""

from __future__ import annotations

from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re
from typing import Any


EMOTION_CLASSES = (
    "amusement",
    "anger",
    "awe",
    "contentment",
    "disgust",
    "excitement",
    "fear",
    "sadness",
)
REVIEW_FLAGS = ("clear", "ambiguous", "possible_mismatch")
MODEL_SOURCE_FILENAME = "VENUS_MODEL_SOURCE.json"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text_file(path: Path) -> str:
    """Hash text with LF line endings so Windows and Linux verify identically."""
    content = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(content).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [
            {key: (value or "").strip() for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]


def read_selection_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("records"), list):
        raise ValueError("Selection manifest must be an object containing a records list.")
    return payload


def validate_selection_manifest(
    payload: dict[str, Any],
    image_directory: Path,
    *,
    verify_hashes: bool = True,
) -> list[str]:
    records = payload.get("records", [])
    errors: list[str] = []
    if len(records) != 80:
        errors.append(f"Expected 80 selection records, found {len(records)}.")

    for field in ("blind_id", "image_id", "local_image_filename", "image_sha256"):
        values = [str(record.get(field, "")).strip() for record in records]
        duplicates = sorted(value for value, count in Counter(values).items() if value and count > 1)
        if duplicates:
            errors.append(f"Duplicate {field}: {', '.join(duplicates)}")
        if any(not value for value in values):
            errors.append(f"One or more selection records have an empty {field}.")

    emotion_counts = Counter(str(record.get("emotion", "")).strip() for record in records)
    expected_counts = Counter({emotion: 10 for emotion in EMOTION_CLASSES})
    if emotion_counts != expected_counts:
        errors.append(
            f"Emotion counts differ: expected {dict(expected_counts)}, found {dict(emotion_counts)}."
        )

    image_directory = image_directory.resolve()
    for record in records:
        blind_id = str(record.get("blind_id", "missing blind_id"))
        filename = str(record.get("local_image_filename", "")).strip()
        expected_hash = str(record.get("image_sha256", "")).strip().lower()
        image_path = (image_directory / filename).resolve()
        try:
            image_path.relative_to(image_directory)
        except ValueError:
            errors.append(f"{blind_id}: image path escapes the image directory.")
            continue
        if not image_path.is_file():
            errors.append(f"{blind_id}: image not found: {filename}")
            continue
        if not SHA256_RE.fullmatch(expected_hash):
            errors.append(f"{blind_id}: image_sha256 is not 64 lowercase hexadecimal characters.")
        elif verify_hashes and sha256_file(image_path) != expected_hash:
            errors.append(f"{blind_id}: image SHA-256 mismatch.")
    return errors


def validate_human_reviews(
    blind_rows: list[dict[str, str]],
    reveal_rows: list[dict[str, str]],
    selection_payload: dict[str, Any],
) -> tuple[list[str], dict[str, Any]]:
    records = selection_payload["records"]
    expected_ids = {str(record["blind_id"]) for record in records}
    errors: list[str] = []

    if len(blind_rows) != 80:
        errors.append(f"Expected 80 blind-review rows, found {len(blind_rows)}.")
    if len(reveal_rows) != 80:
        errors.append(f"Expected 80 reveal-review rows, found {len(reveal_rows)}.")

    blind_by_id = {row.get("blind_id", ""): row for row in blind_rows}
    reveal_by_id = {row.get("blind_id", ""): row for row in reveal_rows}
    if set(blind_by_id) != expected_ids:
        errors.append("Blind-review IDs do not exactly match the selection manifest.")
    if set(reveal_by_id) != expected_ids:
        errors.append("Reveal-review IDs do not exactly match the selection manifest.")

    labels_by_id = {str(record["blind_id"]): str(record["emotion"]) for record in records}
    flag_counts: Counter[str] = Counter()
    agreement_count = 0
    low_confidence_count = 0
    for blind_id in sorted(expected_ids):
        blind = blind_by_id.get(blind_id, {})
        reveal = reveal_by_id.get(blind_id, {})
        independent_label = blind.get("independent_label", "").lower()
        confidence = blind.get("confidence_1_to_3", "")
        blind_notes = blind.get("blind_notes", "")
        if independent_label not in EMOTION_CLASSES:
            errors.append(f"{blind_id}: independent_label must be one allowed emotion.")
        if confidence not in {"1", "2", "3"}:
            errors.append(f"{blind_id}: confidence_1_to_3 must be 1, 2, or 3.")
        if confidence == "1":
            low_confidence_count += 1
            if not blind_notes:
                errors.append(f"{blind_id}: blind_notes is required for confidence 1.")

        expected_label = labels_by_id[blind_id]
        if reveal.get("emoset_label", "").lower() != expected_label:
            errors.append(f"{blind_id}: reveal EmoSet label differs from the selection manifest.")
        if reveal.get("independent_label", "").lower() != independent_label:
            errors.append(f"{blind_id}: reveal independent label differs from the blind review.")
        expected_agreement = "yes" if independent_label == expected_label else "no"
        if reveal.get("agreement", "").lower() != expected_agreement:
            errors.append(f"{blind_id}: agreement must be {expected_agreement!r}.")
        elif expected_agreement == "yes":
            agreement_count += 1
        flag = reveal.get("review_flag", "").lower()
        if flag not in REVIEW_FLAGS:
            errors.append(f"{blind_id}: review_flag must be clear, ambiguous, or possible_mismatch.")
        else:
            flag_counts[flag] += 1
        if flag in {"ambiguous", "possible_mismatch"} and not reveal.get("reveal_notes", ""):
            errors.append(f"{blind_id}: reveal_notes is required for review_flag {flag!r}.")

    summary = {
        "blind_review_count": len(blind_rows),
        "reveal_review_count": len(reveal_rows),
        "independent_label_agreement_count": agreement_count,
        "low_confidence_count": low_confidence_count,
        "review_flag_counts": {flag: flag_counts.get(flag, 0) for flag in REVIEW_FLAGS},
    }
    return errors, summary


def build_inference_manifest(selection_payload: dict[str, Any]) -> dict[str, Any]:
    """Remove private labels and source IDs from the cloud-side inference manifest."""
    records = [
        {
            "blind_id": str(record["blind_id"]),
            "image_filename": str(record["local_image_filename"]),
            "image_sha256": str(record["image_sha256"]),
            "width": int(record["width"]),
            "height": int(record["height"]),
        }
        for record in selection_payload["records"]
    ]
    return {
        "schema_version": 1,
        "evaluation_set": "Eval80-v1",
        "image_count": len(records),
        "contains_ground_truth_labels": False,
        "records": records,
    }


def validate_inference_manifest(
    payload: dict[str, Any],
    image_directory: Path,
    *,
    verify_hashes: bool = True,
) -> list[str]:
    records = payload.get("records", [])
    errors: list[str] = []
    if payload.get("contains_ground_truth_labels") is not False:
        errors.append("Inference manifest must explicitly exclude ground-truth labels.")
    if len(records) != 80:
        errors.append(f"Expected 80 inference records, found {len(records)}.")
    ids = [str(record.get("blind_id", "")) for record in records]
    filenames = [str(record.get("image_filename", "")) for record in records]
    if len(set(ids)) != len(ids) or any(not value for value in ids):
        errors.append("Inference blind IDs must be non-empty and unique.")
    if len(set(filenames)) != len(filenames) or any(not value for value in filenames):
        errors.append("Inference image filenames must be non-empty and unique.")

    image_directory = image_directory.resolve()
    for record in records:
        blind_id = str(record.get("blind_id", "missing blind_id"))
        image_path = (image_directory / str(record.get("image_filename", ""))).resolve()
        try:
            image_path.relative_to(image_directory)
        except ValueError:
            errors.append(f"{blind_id}: image path escapes the image directory.")
            continue
        if not image_path.is_file():
            errors.append(f"{blind_id}: image is missing.")
            continue
        expected_hash = str(record.get("image_sha256", "")).lower()
        if not SHA256_RE.fullmatch(expected_hash):
            errors.append(f"{blind_id}: invalid image_sha256.")
        elif verify_hashes and sha256_file(image_path) != expected_hash:
            errors.append(f"{blind_id}: image SHA-256 mismatch.")
    return errors


def validate_freeze_record(
    payload: dict[str, Any],
    *,
    config_path: Path,
    handbook_path: Path,
    inference_manifest_path: Path,
) -> list[str]:
    errors: list[str] = []
    if payload.get("status") != "approved_and_frozen":
        errors.append("Freeze record status must be 'approved_and_frozen'.")
    approver_ids = payload.get("approver_ids")
    if not isinstance(approver_ids, list) or not approver_ids or len(set(approver_ids)) != len(approver_ids):
        errors.append("Freeze record must contain one or more unique approver IDs.")
    if not payload.get("approved_at_utc"):
        errors.append("Freeze record is missing approved_at_utc.")

    expected_files = {
        "config": (config_path, sha256_text_file),
        "scoring_handbook": (handbook_path, sha256_text_file),
        "inference_manifest": (inference_manifest_path, sha256_file),
    }
    recorded_hashes = payload.get("file_sha256", {})
    for name, (path, hash_file) in expected_files.items():
        if not path.is_file():
            errors.append(f"Required frozen file is missing: {path}")
            continue
        actual_hash = hash_file(path)
        if recorded_hashes.get(name) != actual_hash:
            errors.append(f"Frozen {name} SHA-256 mismatch.")
    return errors
