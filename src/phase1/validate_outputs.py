"""Check that a Phase 1 run retained one successful record for all 20 fixed images."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from .manifest import read_manifest


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("stage1", "stage2"), required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    manifest_rows = read_manifest(args.manifest)
    expected_ids = {row["sample_id"] for row in manifest_rows}
    records = _read_jsonl(args.run_dir / "records.jsonl")
    errors: list[str] = []
    if len(records) != len(manifest_rows):
        errors.append(f"Expected {len(manifest_rows)} records, found {len(records)}.")

    actual_ids = [str(record.get("sample_id", "")) for record in records]
    duplicate_ids = sorted(value for value, count in Counter(actual_ids).items() if count > 1)
    if duplicate_ids:
        errors.append(f"Duplicate sample IDs: {', '.join(duplicate_ids)}")
    if set(actual_ids) != expected_ids:
        errors.append(
            f"Sample ID set differs. Missing={sorted(expected_ids - set(actual_ids))}; "
            f"extra={sorted(set(actual_ids) - expected_ids)}"
        )
    for record in records:
        if record.get("stage") != args.stage:
            errors.append(f"{record.get('sample_id')}: stage is not {args.stage}.")
        if record.get("condition") != "A":
            errors.append(f"{record.get('sample_id')}: condition is not A.")
        if record.get("status") != "success":
            errors.append(f"{record.get('sample_id')}: status is not success.")

    parse_counts = None
    if args.stage == "stage2":
        parse_counts = dict(
            Counter((record.get("crop_parse") or {}).get("status", "missing") for record in records)
        )

    report = {
        "valid": not errors,
        "stage": args.stage,
        "condition": "A",
        "record_count": len(records),
        "expected_count": len(manifest_rows),
        "crop_parse_counts": parse_counts,
        "errors": errors,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
