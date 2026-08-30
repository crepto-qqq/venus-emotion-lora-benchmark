"""Read and validate the fixed 20-image EmoSet Phase 1 manifest."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re
from typing import Iterable


REQUIRED_COLUMNS = (
    "sample_id",
    "source_image_id",
    "emotion",
    "valence",
    "split",
    "image_relpath",
    "annotation_relpath",
    "sha256",
    "review_status",
    "replacement_for",
    "review_notes",
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [column for column in REQUIRED_COLUMNS if column not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"Manifest is missing columns: {', '.join(missing)}")
        return [
            {key: (value or "").strip() for key, value in row.items()}
            for row in reader
        ]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_dataset_path(root: Path, relative_path: str) -> Path:
    if not relative_path:
        raise ValueError("Dataset-relative path is empty.")
    candidate = (root / Path(relative_path)).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"Path escapes EmoSet root: {relative_path}") from error
    return candidate


def load_test_index(emoset_root: Path) -> set[tuple[str, str, str, str]]:
    test_path = emoset_root / "test.json"
    with test_path.open("r", encoding="utf-8") as handle:
        entries = json.load(handle)
    index: set[tuple[str, str, str, str]] = set()
    for position, entry in enumerate(entries):
        if not isinstance(entry, list) or len(entry) < 4:
            raise ValueError(f"Invalid EmoSet test.json entry at index {position}.")
        index.add(tuple(str(value) for value in entry[:4]))
    return index


def validate_manifest(
    rows: list[dict[str, str]],
    emoset_root: Path,
    emotion_quotas: dict[str, int],
    valence_by_emotion: dict[str, str],
    *,
    expected_size: int = 20,
    require_approved: bool = True,
    verify_hashes: bool = True,
) -> list[str]:
    errors: list[str] = []
    if len(rows) != expected_size:
        errors.append(f"Expected {expected_size} rows, found {len(rows)}.")

    sample_ids = [row["sample_id"] for row in rows]
    source_ids = [row["source_image_id"] for row in rows]
    image_paths = [row["image_relpath"] for row in rows]
    for label, values in (
        ("sample_id", sample_ids),
        ("source_image_id", source_ids),
        ("image_relpath", image_paths),
    ):
        duplicates = sorted(value for value, count in Counter(values).items() if count > 1)
        if duplicates:
            errors.append(f"Duplicate {label}: {', '.join(duplicates)}")

    actual_quotas = Counter(row["emotion"] for row in rows)
    if dict(actual_quotas) != emotion_quotas:
        errors.append(f"Emotion quotas differ: expected {emotion_quotas}, found {dict(actual_quotas)}.")

    actual_valence = Counter(row["valence"] for row in rows)
    if actual_valence != Counter({"positive": 10, "negative": 10}):
        errors.append(f"Expected 10 positive and 10 negative rows, found {dict(actual_valence)}.")

    try:
        test_index = load_test_index(emoset_root)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        errors.append(f"Could not read official EmoSet test split: {error}")
        test_index = set()

    for row_number, row in enumerate(rows, start=2):
        prefix = f"CSV row {row_number} ({row['sample_id'] or 'missing sample_id'})"
        emotion = row["emotion"]
        if row["split"] != "test":
            errors.append(f"{prefix}: split must be 'test'.")
        if emotion not in emotion_quotas:
            errors.append(f"{prefix}: unknown emotion '{emotion}'.")
        elif row["valence"] != valence_by_emotion[emotion]:
            errors.append(f"{prefix}: valence does not match emotion '{emotion}'.")
        if require_approved and row["review_status"] != "approved":
            errors.append(f"{prefix}: review_status must be 'approved'.")

        official_key = (
            emotion,
            row["source_image_id"],
            row["image_relpath"],
            row["annotation_relpath"],
        )
        if test_index and official_key not in test_index:
            errors.append(f"{prefix}: entry is not an exact member of EmoSet test.json.")

        try:
            image_path = _safe_dataset_path(emoset_root, row["image_relpath"])
            annotation_path = _safe_dataset_path(emoset_root, row["annotation_relpath"])
        except ValueError as error:
            errors.append(f"{prefix}: {error}")
            continue

        if not image_path.is_file():
            errors.append(f"{prefix}: image not found: {row['image_relpath']}")
        if not annotation_path.is_file():
            errors.append(f"{prefix}: annotation not found: {row['annotation_relpath']}")

        expected_hash = row["sha256"].lower()
        if not SHA256_RE.fullmatch(expected_hash):
            errors.append(f"{prefix}: sha256 must contain 64 lowercase hexadecimal characters.")
        elif verify_hashes and image_path.is_file():
            actual_hash = sha256_file(image_path)
            if actual_hash != expected_hash:
                errors.append(f"{prefix}: image SHA-256 mismatch.")
    return errors


def write_manifest(path: Path, rows: Iterable[dict[str, str]], *, overwrite: bool = False) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite fixed manifest: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REQUIRED_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--emoset-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/phase1/baseline.json"))
    parser.add_argument("--allow-pending", action="store_true")
    parser.add_argument("--skip-hashes", action="store_true")
    args = parser.parse_args()

    with args.config.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    rows = read_manifest(args.manifest)
    errors = validate_manifest(
        rows,
        args.emoset_root,
        config["dataset"]["emotion_quotas"],
        config["dataset"]["valence"],
        expected_size=config["dataset"]["size"],
        require_approved=not args.allow_pending,
        verify_hashes=not args.skip_hashes,
    )
    report = {"valid": not errors, "row_count": len(rows), "errors": errors}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
