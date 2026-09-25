#!/usr/bin/env python3
"""Build the fixed one-record Member 1 technical fixture without changing source data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from common import (
    assert_report_is_redacted,
    read_json,
    sha256_bytes,
    sha256_file,
    utc_now,
    write_json_atomic,
)


DEFAULT_EVAL80_MANIFEST = Path("results/phase2/eval80/frozen_inputs/inference_manifest.json")
DEFAULT_EVAL20_RESULT = Path("results/phase1/emoset_eval20/stage1/B0_direct_emotion.json")


def _canonical_record_sha256(record: dict[str, Any]) -> str:
    return sha256_bytes(json.dumps(record, ensure_ascii=False).encode("utf-8"))


def _read_locked_record(project_root: Path, config: dict[str, Any]) -> dict[str, Any]:
    fixture = config["fixture"]
    source_path = project_root / fixture["dataset_relpath"]
    lines = source_path.read_text(encoding="utf-8").splitlines()
    line_number = int(fixture["record_line"])
    if line_number < 1 or line_number > len(lines):
        raise ValueError("locked source line is outside the JSONL file")
    record = json.loads(lines[line_number - 1])
    if not isinstance(record, dict):
        raise TypeError("locked source line is not a JSON object")
    return record


def _validate_locked_record(record: dict[str, Any], fixture: dict[str, Any]) -> None:
    provenance = record.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("locked source record has no provenance object")
    observed = {
        "source_image_id": record.get("source_image_id"),
        "emotion": record.get("emotion"),
        "record_sha256": _canonical_record_sha256(record),
        "annotation_sha256": provenance.get("annotation_sha256"),
        "image_sha256": provenance.get("image_sha256"),
    }
    expected = {
        "source_image_id": fixture["source_image_id"],
        "emotion": fixture["emotion"],
        "record_sha256": fixture["record_sha256"],
        "annotation_sha256": fixture["annotation_sha256"],
        "image_sha256": fixture["image_sha256"],
    }
    if observed != expected:
        raise ValueError("source record identity does not match member1-smoke.json")
    if record.get("split") != "train":
        raise ValueError("technical fixture source must be from the training split")
    if not isinstance(record.get("instruction"), str) or not record["instruction"].strip():
        raise ValueError("source instruction is empty")
    if not isinstance(record.get("target_response"), str) or not record["target_response"].strip():
        raise ValueError("source target response is empty")
    if provenance.get("review_status") not in {"accepted", "edited"}:
        raise ValueError("technical fixture source has not passed human review")
    if provenance.get("label_supported") is not True:
        raise ValueError("technical fixture source label is not supported")


def _extract_identities(value: Any) -> tuple[set[str], set[str]]:
    """Extract image IDs and hashes from the two frozen evaluation formats."""
    ids: set[str] = set()
    hashes: set[str] = set()

    def visit(item: Any) -> None:
        if isinstance(item, dict):
            for key, nested in item.items():
                if key in {"source_image_id", "image_id"} and isinstance(nested, str):
                    ids.add(nested)
                elif key in {"image_sha256", "sha256"} and isinstance(nested, str):
                    if len(nested) == 64:
                        hashes.add(nested.lower())
                elif isinstance(nested, (dict, list)):
                    visit(nested)
        elif isinstance(item, list):
            for nested in item:
                visit(nested)

    visit(value)
    return ids, hashes


def _check_disjointness(
    *,
    project_root: Path,
    source_image_id: str,
    image_sha256: str,
    eval80_manifest: Path,
    eval20_result: Path,
) -> tuple[str, list[dict[str, Any]]]:
    artifacts = (
        ("Eval80", eval80_manifest),
        ("Eval20", eval20_result),
    )
    checks: list[dict[str, Any]] = []
    complete = True
    for label, configured_path in artifacts:
        path = configured_path if configured_path.is_absolute() else project_root / configured_path
        if not path.is_file():
            complete = False
            checks.append({"evaluation_set": label, "status": "unavailable"})
            continue
        value = read_json(path)
        ids, hashes = _extract_identities(value)
        collision = source_image_id in ids or image_sha256.lower() in hashes
        checks.append(
            {
                "evaluation_set": label,
                "status": "collision" if collision else "disjoint",
                "identity_count": len(ids),
                "hash_count": len(hashes),
            }
        )
        if collision:
            raise ValueError(f"technical fixture collides with the frozen {label} evaluation set")
    return ("passed" if complete else "limited"), checks


def _jpeg_dimensions(path: Path) -> tuple[int, int]:
    """Read JPEG dimensions without adding a bootstrap-time imaging dependency."""
    data = path.read_bytes()
    if len(data) < 4 or data[:2] != b"\xff\xd8":
        raise ValueError("technical fixture is not a JPEG image")
    start_of_frame = {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}
    position = 2
    while position + 3 < len(data):
        while position < len(data) and data[position] != 0xFF:
            position += 1
        while position < len(data) and data[position] == 0xFF:
            position += 1
        if position >= len(data):
            break
        marker = data[position]
        position += 1
        if marker in {0x01, *range(0xD0, 0xDA)}:
            continue
        if position + 2 > len(data):
            break
        segment_length = int.from_bytes(data[position : position + 2], "big")
        if segment_length < 2 or position + segment_length > len(data):
            raise ValueError("technical fixture JPEG has an invalid segment")
        if marker in start_of_frame:
            if segment_length < 7:
                raise ValueError("technical fixture JPEG has an invalid frame header")
            height = int.from_bytes(data[position + 3 : position + 5], "big")
            width = int.from_bytes(data[position + 5 : position + 7], "big")
            if width <= 0 or height <= 0:
                raise ValueError("technical fixture JPEG has invalid dimensions")
            return width, height
        position += segment_length
    raise ValueError("technical fixture JPEG has no supported frame header")


def _validate_image(path: Path, expected_sha256: str) -> tuple[int, int, int]:
    if not path.is_file():
        raise FileNotFoundError("technical fixture image does not exist")
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise ValueError("technical fixture image SHA-256 does not match the source record")
    width, height = _jpeg_dimensions(path)
    return path.stat().st_size, int(width), int(height)


def build_fixture(
    *,
    project_root: Path,
    image: Path,
    image_reference: str,
    eval80_manifest: Path,
    eval20_result: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    config = read_json(project_root / "phase3/configs/member1-smoke.json")
    fixture_lock = config["fixture"]
    record = _read_locked_record(project_root, config)
    _validate_locked_record(record, fixture_lock)
    image_bytes, width, height = _validate_image(image, fixture_lock["image_sha256"])
    expected_image_metadata = (
        int(fixture_lock["image_byte_count"]),
        int(fixture_lock["image_width"]),
        int(fixture_lock["image_height"]),
    )
    if (image_bytes, width, height) != expected_image_metadata:
        raise ValueError("technical fixture image metadata differs from member1-smoke.json")
    disjointness, disjointness_checks = _check_disjointness(
        project_root=project_root,
        source_image_id=fixture_lock["source_image_id"],
        image_sha256=fixture_lock["image_sha256"],
        eval80_manifest=eval80_manifest,
        eval20_result=eval20_result,
    )

    fixture = [
        {
            "id": fixture_lock["source_image_id"],
            "technical_infrastructure_only": True,
            "conversations": [
                {
                    "from": "user",
                    "value": f"Picture 1: <img>{image_reference}</img>\n{record['instruction']}",
                },
                {"from": "assistant", "value": record["target_response"]},
            ],
            "provenance": {
                "dataset_record_sha256": fixture_lock["record_sha256"],
                "annotation_sha256": fixture_lock["annotation_sha256"],
                "image_sha256": fixture_lock["image_sha256"],
                "source_split": "train",
                "disjointness_check": disjointness,
            },
        }
    ]
    manifest = {
        "schema_version": 1,
        "kind": "phase3_technical_fixture_manifest",
        "generated_at_utc": utc_now(),
        "technical_infrastructure_only": True,
        "source_image_id": fixture_lock["source_image_id"],
        "emotion": record["emotion"],
        "source_split": record["split"],
        "review_status": record["provenance"]["review_status"],
        "label_supported": record["provenance"]["label_supported"],
        "source_dataset_relpath": fixture_lock["dataset_relpath"],
        "source_record_line": fixture_lock["record_line"],
        "dataset_record_sha256": fixture_lock["record_sha256"],
        "annotation_sha256": fixture_lock["annotation_sha256"],
        "image_sha256": fixture_lock["image_sha256"],
        "image": {
            "reference": image_reference,
            "byte_count": image_bytes,
            "width": width,
            "height": height,
        },
        "disjointness_check": disjointness,
        "evaluation_checks": disjointness_checks,
        "full_dataset_ready": False,
        "formal_training_authorized": False,
    }
    return fixture, manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    image_group = parser.add_mutually_exclusive_group(required=True)
    image_group.add_argument("--image", type=Path)
    image_group.add_argument("--image-root", type=Path)
    parser.add_argument(
        "--image-reference",
        help="Path written inside <img>; defaults to the resolved image path.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--eval80-manifest", type=Path, default=DEFAULT_EVAL80_MANIFEST)
    parser.add_argument("--eval20-result", type=Path, default=DEFAULT_EVAL20_RESULT)
    args = parser.parse_args(argv)

    project_root = args.project_root.resolve()
    config = read_json(project_root / "phase3/configs/member1-smoke.json")
    if args.image is not None:
        image = args.image.resolve()
    else:
        image = (args.image_root / config["fixture"].get(
            "image_relpath", "image/contentment/contentment_05000.jpg"
        )).resolve()
        # The current lock predates image_relpath, so take it from the immutable record.
        if "image_relpath" not in config["fixture"]:
            record = _read_locked_record(project_root, config)
            image = (args.image_root / record["image_relpath"]).resolve()
    image_reference = args.image_reference or image.as_posix()

    fixture, manifest = build_fixture(
        project_root=project_root,
        image=image,
        image_reference=image_reference,
        eval80_manifest=args.eval80_manifest,
        eval20_result=args.eval20_result,
    )

    from jsonschema import Draft202012Validator

    schema = read_json(project_root / "phase3/contracts/technical-fixture.schema.json")
    Draft202012Validator(schema).validate(fixture)
    assert_report_is_redacted(fixture)
    assert_report_is_redacted(manifest)
    write_json_atomic(args.output, fixture)
    manifest["fixture_sha256"] = sha256_file(args.output)
    write_json_atomic(args.manifest, manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
