"""Aggregate-only audit of the frozen B1/C structured response format.

This auxiliary metric checks syntax, not emotion-recognition accuracy or the
quality, visual grounding, or usefulness of aesthetic guidance. It never reads
the private selection manifest or emits individual responses or image IDs.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from .core import EMOTION_CLASSES, sha256_file


SECTION_NAMES = (
    "Emotion",
    "Visual evidence",
    "Aesthetic relationship",
    "Guidance",
    "Final emotion",
)
_SECTION_RE = re.compile(
    r"^(Emotion|Visual evidence|Aesthetic relationship|Guidance|Final emotion)\s*:(.*)$",
    re.IGNORECASE,
)
_BULLET_RE = re.compile(r"^-(?:[ \t]+(.*))?$")
_EXTRA_HEADING_RE = re.compile(
    r"^(?:#{1,6}[ \t]+\S.*|[A-Za-z][A-Za-z0-9 /&()'_-]{0,70}:)$"
)
_REPO_ROOT = Path(__file__).resolve().parents[2]
_FREEZE_RECORD = _REPO_ROOT / "Docs" / "evaluation" / "supporting" / "protocols" / "EVAL80_FREEZE_RECORD.json"


def check_structured_response(response: str) -> tuple[str, ...]:
    """Return stable format-failure reasons for one response, without content."""
    lines = [line.strip() for line in response.splitlines() if line.strip()]
    headings: list[tuple[str, int, str]] = []
    for index, line in enumerate(lines):
        match = _SECTION_RE.fullmatch(line)
        if match:
            name = next(
                section for section in SECTION_NAMES
                if section.casefold() == match.group(1).casefold()
            )
            headings.append((name, index, match.group(2).strip()))

    reasons: set[str] = set()
    positions: dict[str, list[int]] = {name: [] for name in SECTION_NAMES}
    for heading_index, (name, _, _) in enumerate(headings):
        positions[name].append(heading_index)
    for name, occurrences in positions.items():
        key = name.lower().replace(" ", "_")
        if not occurrences:
            reasons.add(f"missing_{key}")
        elif len(occurrences) > 1:
            reasons.add(f"duplicate_{key}")

    ordered_names = [name for name, _, _ in headings]
    expected_present = [name for name in SECTION_NAMES if name in ordered_names]
    if [name for name in ordered_names if ordered_names.count(name) == 1] != [
        name for name in expected_present if ordered_names.count(name) == 1
    ]:
        reasons.add("section_order")

    if lines and (not headings or headings[0][1] != 0):
        reasons.add("unexpected_content")

    bodies: dict[str, list[str]] = {}
    values: dict[str, str] = {}
    for index, (name, line_index, value) in enumerate(headings):
        next_index = headings[index + 1][1] if index + 1 < len(headings) else len(lines)
        if len(positions[name]) == 1:
            bodies[name] = lines[line_index + 1:next_index]
            values[name] = value

    for name in ("Emotion", "Final emotion"):
        if name not in values:
            continue
        key = name.lower().replace(" ", "_")
        if values[name].casefold() not in EMOTION_CLASSES:
            reasons.add(f"invalid_{key}")
        if bodies[name]:
            reasons.add("unexpected_content")
    if (
        "Emotion" in values
        and "Final emotion" in values
        and values["Emotion"].casefold() in EMOTION_CLASSES
        and values["Final emotion"].casefold() in EMOTION_CLASSES
        and values["Emotion"].casefold() != values["Final emotion"].casefold()
    ):
        reasons.add("emotion_mismatch")

    for name in ("Visual evidence", "Guidance"):
        if name not in bodies:
            continue
        key = name.lower().replace(" ", "_")
        if values[name]:
            reasons.add("unexpected_content")
        bullet_matches = [_BULLET_RE.fullmatch(line) for line in bodies[name]]
        if len(bodies[name]) != 2 or any(match is None for match in bullet_matches):
            reasons.add(f"{key}_bullet_count")
        if any(match is not None and not (match.group(1) or "").strip() for match in bullet_matches):
            reasons.add(f"empty_{key}_bullet")

    if "Aesthetic relationship" in bodies:
        if values["Aesthetic relationship"]:
            reasons.add("unexpected_content")
        relationship = bodies["Aesthetic relationship"]
        if not relationship:
            reasons.add("empty_aesthetic_relationship")
        elif any(
            _BULLET_RE.fullmatch(line) or _EXTRA_HEADING_RE.fullmatch(line)
            for line in relationship
        ):
            reasons.add("unexpected_content")

    return tuple(sorted(reasons))


def audit_output_contract(
    records: list[dict[str, Any]],
    inference_manifest: dict[str, Any],
    *,
    condition: str,
) -> dict[str, Any]:
    """Validate the 80 inputs, then summarize only structural compliance."""
    if condition not in {"B1", "C"}:
        raise ValueError("Format audit condition must be B1 or C.")
    if not isinstance(inference_manifest, dict):
        raise ValueError("Inference manifest must be a JSON object.")
    if (
        inference_manifest.get("schema_version") != 1
        or inference_manifest.get("evaluation_set") != "Eval80-v1"
        or inference_manifest.get("image_count") != 80
        or inference_manifest.get("contains_ground_truth_labels") is not False
    ):
        raise ValueError("Expected the label-free Eval80-v1 inference manifest metadata.")
    if set(inference_manifest) != {
        "schema_version", "evaluation_set", "image_count",
        "contains_ground_truth_labels", "records",
    }:
        raise ValueError("Inference manifest must use only label-free frozen fields.")
    manifest_records = inference_manifest.get("records")
    if not isinstance(manifest_records, list) or len(manifest_records) != 80:
        raise ValueError("Inference manifest must contain exactly 80 records.")
    allowed_manifest_keys = {
        "blind_id", "image_filename", "image_sha256", "width", "height"
    }
    if any(
        not isinstance(record, dict)
        or set(record) != allowed_manifest_keys
        for record in manifest_records
    ):
        raise ValueError("Inference manifest records must use only label-free frozen fields.")
    expected_ids = [record["blind_id"] for record in manifest_records]
    if (
        any(not isinstance(blind_id, str) or not blind_id for blind_id in expected_ids)
        or len(set(expected_ids)) != 80
    ):
        raise ValueError("Inference manifest blind IDs must be nonempty and unique.")

    if not isinstance(records, list) or len(records) != 80:
        raise ValueError("Format audit requires exactly 80 output records.")
    if any(not isinstance(record, dict) for record in records):
        raise ValueError("Every output record must be a JSON object.")
    actual_ids = [record.get("blind_id") for record in records]
    if (
        any(not isinstance(blind_id, str) or not blind_id for blind_id in actual_ids)
        or len(set(actual_ids)) != 80
        or set(actual_ids) != set(expected_ids)
    ):
        raise ValueError("Output blind IDs must uniquely match all 80 frozen manifest IDs.")
    if any(record.get("condition") != condition for record in records):
        raise ValueError("One or more output records have the wrong condition.")
    if any(record.get("status") != "success" for record in records):
        raise ValueError("All output records must have success status.")
    if any(
        not isinstance(record.get("response"), str) or not record["response"].strip()
        for record in records
    ):
        raise ValueError("All output records must contain a nonempty text response.")
    image_hash_by_id = {
        record["blind_id"]: record["image_sha256"] for record in manifest_records
    }
    if any(
        record.get("image_sha256") != image_hash_by_id[record["blind_id"]]
        for record in records
    ):
        raise ValueError("One or more output image hashes differ from the frozen manifest.")
    if any(
        record.get("response_sha256")
        != hashlib.sha256(record["response"].encode("utf-8")).hexdigest()
        for record in records
    ):
        raise ValueError("One or more output response hashes do not match response text.")

    failures: Counter[str] = Counter()
    pass_count = 0
    for record in records:
        reasons = check_structured_response(record["response"])
        if reasons:
            failures.update(reasons)
        else:
            pass_count += 1

    return {
        "schema_version": 1,
        "evaluation_set": "Eval80-v1",
        "condition": condition,
        "metric": "auxiliary_structured_output_format_compliance",
        "item_count": 80,
        "format_pass_count": pass_count,
        "format_pass_rate": round(pass_count / 80, 4),
        "format_failure_count": 80 - pass_count,
        "failure_reason_counts": dict(sorted(failures.items())),
        "scope_note": (
            "Checks only frozen B1/C response sections and syntax; it does not "
            "measure emotion-recognition accuracy or human guidance quality."
        ),
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"Output JSONL line {line_number} is empty.")
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"Output JSONL line {line_number} must be an object.")
            records.append(record)
    return records


def audit_format_command(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Audit the frozen B1/C structured output format on 80 accepted Eval80 "
            "responses; write aggregate-only counts. This is separate from "
            "recognition accuracy and human guidance quality."
        )
    )
    parser.add_argument("--condition", choices=("B1", "C"), required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--inference-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    freeze_record = json.loads(_FREEZE_RECORD.read_text(encoding="utf-8"))
    expected_hash = freeze_record["file_sha256"]["inference_manifest"]
    manifest_hash = sha256_file(args.inference_manifest)
    if manifest_hash != expected_hash:
        raise ValueError("Inference manifest SHA-256 differs from the frozen Eval80-v1 record.")
    manifest = json.loads(args.inference_manifest.read_text(encoding="utf-8"))
    records = _read_jsonl(args.records)
    summary = audit_output_contract(records, manifest, condition=args.condition)
    summary["inference_manifest_sha256"] = manifest_hash
    summary["records_sha256"] = sha256_file(args.records)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"{args.condition}_format_audit_summary.json"
    with output_path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(audit_format_command())
