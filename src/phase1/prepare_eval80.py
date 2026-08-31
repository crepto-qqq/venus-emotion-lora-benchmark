"""Prepare the local-only EmoSet Eval80-v1 review package.

The script reads the original EmoSet archive without modifying it. It selects
the first ten available image files in filename order from each of the eight
emotion classes, assigns deterministic blind IDs, validates every image and
annotation, and writes contact sheets and manifests outside the repository.

Pillow is required for image validation and contact-sheet generation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import sys
import zipfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


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
SELECTIONS_PER_CLASS = 10
BLIND_ORDER_NAMESPACE = "venus-emoset-eval80-v1-blind-order"


@dataclass(frozen=True)
class SelectedEntry:
    emotion: str
    class_rank: int
    image_entry: str
    annotation_entry: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a local-only, human-reviewable EmoSet Eval80-v1 package."
    )
    parser.add_argument("--archive", type=Path, required=True, help="Path to EmoSet-118K.zip")
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="New output directory outside the Git repository",
    )
    return parser.parse_args()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_blind_key(image_entry: str) -> str:
    payload = f"{BLIND_ORDER_NAMESPACE}|{image_entry}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_split_map(archive: zipfile.ZipFile) -> dict[str, str]:
    split_map: dict[str, str] = {}
    for split_name, filename in (
        ("train", "train.json"),
        ("val", "val.json"),
        ("test", "test.json"),
    ):
        records = json.loads(archive.read(filename))
        for emotion, image_entry, _annotation_entry in records:
            if image_entry in split_map:
                raise ValueError(f"Image appears in multiple official splits: {image_entry}")
            if emotion not in EMOTION_CLASSES:
                raise ValueError(f"Unexpected emotion in {filename}: {emotion}")
            split_map[image_entry] = split_name
    return split_map


def select_entries(
    archive: zipfile.ZipFile,
) -> tuple[list[SelectedEntry], dict[str, list[str]], dict[str, list[str]]]:
    archive_names = set(archive.namelist())
    selected: list[SelectedEntry] = []
    missing_expected: dict[str, list[str]] = {}
    substitutions: dict[str, list[str]] = {}

    for emotion in EMOTION_CLASSES:
        prefix = f"image/{emotion}/{emotion}_"
        candidates = sorted(
            name
            for name in archive_names
            if name.startswith(prefix) and name.lower().endswith(".jpg")
        )
        if len(candidates) < SELECTIONS_PER_CLASS:
            raise ValueError(
                f"Class {emotion} has only {len(candidates)} candidate images; "
                f"{SELECTIONS_PER_CLASS} are required"
            )

        chosen = candidates[:SELECTIONS_PER_CLASS]
        expected_basenames = {f"{emotion}_{index:05d}.jpg" for index in range(10)}
        chosen_basenames = {Path(name).name for name in chosen}
        missing_expected[emotion] = sorted(expected_basenames - chosen_basenames)
        substitutions[emotion] = sorted(chosen_basenames - expected_basenames)

        for class_rank, image_entry in enumerate(chosen, start=1):
            image_id = Path(image_entry).stem
            annotation_entry = f"annotation/{emotion}/{image_id}.json"
            if annotation_entry not in archive_names:
                raise ValueError(f"Missing annotation: {annotation_entry}")
            selected.append(
                SelectedEntry(
                    emotion=emotion,
                    class_rank=class_rank,
                    image_entry=image_entry,
                    annotation_entry=annotation_entry,
                )
            )

    if len(selected) != 80:
        raise AssertionError(f"Expected 80 selected images, found {len(selected)}")

    return selected, missing_expected, substitutions


def validate_image(image_bytes: bytes, image_entry: str) -> tuple[str, int, int]:
    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            image.load()
            image_format = image.format or "unknown"
            width, height = image.size
    except Exception as error:  # Pillow exposes multiple decoder exception types.
        raise ValueError(f"Failed to decode {image_entry}: {error}") from error

    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image dimensions for {image_entry}: {width}x{height}")
    return image_format, width, height


def validate_annotation(
    annotation_bytes: bytes,
    expected_emotion: str,
    expected_image_id: str,
    annotation_entry: str,
) -> dict[str, Any]:
    try:
        annotation = json.loads(annotation_bytes)
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON annotation {annotation_entry}: {error}") from error

    if annotation.get("emotion") != expected_emotion:
        raise ValueError(
            f"Annotation emotion mismatch in {annotation_entry}: "
            f"expected {expected_emotion}, found {annotation.get('emotion')}"
        )
    if annotation.get("image_id") != expected_image_id:
        raise ValueError(
            f"Annotation image_id mismatch in {annotation_entry}: "
            f"expected {expected_image_id}, found {annotation.get('image_id')}"
        )
    return annotation


def draw_contact_sheet(
    records: list[dict[str, Any]],
    image_directory: Path,
    output_path: Path,
    labeled: bool,
) -> None:
    columns = 5
    image_height = 220
    cell_width = 320
    caption_height = 58 if labeled else 34
    cell_height = image_height + caption_height
    row_count = math.ceil(len(records) / columns)
    sheet = Image.new("RGB", (columns * cell_width, row_count * cell_height), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=16)

    for index, record in enumerate(records):
        column = index % columns
        row = index // columns
        left = column * cell_width
        top = row * cell_height

        with Image.open(image_directory / record["local_image_filename"]) as source:
            thumbnail = source.convert("RGB")
            thumbnail.thumbnail(
                (cell_width - 20, image_height - 20),
                Image.Resampling.LANCZOS,
            )
        x = left + (cell_width - thumbnail.width) // 2
        y = top + (image_height - thumbnail.height) // 2
        sheet.paste(thumbnail, (x, y))

        if labeled:
            first_line = f"{record['blind_id']} | {record['emotion']}"
            second_line = record["image_id"]
            draw.text((left + 8, top + image_height + 3), first_line, fill="black", font=font)
            draw.text((left + 8, top + image_height + 26), second_line, fill="black", font=font)
        else:
            draw.text(
                (left + 8, top + image_height + 5),
                record["blind_id"],
                fill="black",
                font=font,
            )

        draw.rectangle(
            (left, top, left + cell_width - 1, top + cell_height - 1),
            outline="#cccccc",
            width=1,
        )

    sheet.save(output_path, quality=92, optimize=True)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def prepare(archive_path: Path, output_root: Path) -> dict[str, Any]:
    archive_path = archive_path.resolve(strict=True)
    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError(
            f"Output directory already exists and will not be overwritten: {output_root}"
        )

    output_root.mkdir(parents=True, exist_ok=False)
    image_directory = output_root / "images_blind"
    annotation_directory = output_root / "annotations_blind"
    image_directory.mkdir()
    annotation_directory.mkdir()

    archive_sha256 = sha256_file(archive_path)
    created_at = datetime.now(timezone.utc).isoformat()

    with zipfile.ZipFile(archive_path) as archive:
        split_map = load_split_map(archive)
        selected, missing_expected, substitutions = select_entries(archive)
        blind_order = sorted(selected, key=lambda item: stable_blind_key(item.image_entry))

        records: list[dict[str, Any]] = []
        for blind_index, item in enumerate(blind_order, start=1):
            blind_id = f"E{blind_index:03d}"
            image_id = Path(item.image_entry).stem
            image_bytes = archive.read(item.image_entry)
            annotation_bytes = archive.read(item.annotation_entry)
            image_format, width, height = validate_image(image_bytes, item.image_entry)
            annotation = validate_annotation(
                annotation_bytes,
                expected_emotion=item.emotion,
                expected_image_id=image_id,
                annotation_entry=item.annotation_entry,
            )
            split = split_map.get(item.image_entry)
            if split is None:
                raise ValueError(f"Selected image is absent from official splits: {item.image_entry}")

            local_image_filename = f"{blind_id}.jpg"
            local_annotation_filename = f"{blind_id}.json"
            (image_directory / local_image_filename).write_bytes(image_bytes)
            (annotation_directory / local_annotation_filename).write_bytes(annotation_bytes)

            image_info = archive.getinfo(item.image_entry)
            annotation_info = archive.getinfo(item.annotation_entry)
            records.append(
                {
                    "blind_id": blind_id,
                    "emotion": item.emotion,
                    "image_id": image_id,
                    "official_split": split,
                    "class_rank": item.class_rank,
                    "source_image_entry": item.image_entry,
                    "source_annotation_entry": item.annotation_entry,
                    "local_image_filename": local_image_filename,
                    "local_annotation_filename": local_annotation_filename,
                    "image_sha256": sha256_bytes(image_bytes),
                    "annotation_sha256": sha256_bytes(annotation_bytes),
                    "image_zip_crc32": f"{image_info.CRC:08x}",
                    "annotation_zip_crc32": f"{annotation_info.CRC:08x}",
                    "image_size_bytes": len(image_bytes),
                    "annotation_size_bytes": len(annotation_bytes),
                    "image_format": image_format,
                    "width": width,
                    "height": height,
                    "annotation_attributes": annotation,
                    "blind_order_key": stable_blind_key(item.image_entry),
                }
            )

    duplicate_groups: dict[str, list[str]] = {}
    hash_to_ids: dict[str, list[str]] = {}
    for record in records:
        hash_to_ids.setdefault(record["image_sha256"], []).append(record["blind_id"])
    for digest, blind_ids in hash_to_ids.items():
        if len(blind_ids) > 1:
            duplicate_groups[digest] = blind_ids

    class_counts = Counter(record["emotion"] for record in records)
    split_counts = Counter(record["official_split"] for record in records)
    summary = {
        "protocol_version": "Eval80-v1",
        "created_at_utc": created_at,
        "archive_path": str(archive_path),
        "archive_size_bytes": archive_path.stat().st_size,
        "archive_sha256": archive_sha256,
        "selection_rule": "first ten available JPG files in filename order per emotion class",
        "blind_order_rule": f"ascending SHA-256 of {BLIND_ORDER_NAMESPACE}|<zip image entry>",
        "selected_image_count": len(records),
        "class_counts": dict(sorted(class_counts.items())),
        "official_split_counts": dict(sorted(split_counts.items())),
        "missing_expected_00000_to_00009": missing_expected,
        "substitutions_from_later_filenames": substitutions,
        "exact_duplicate_groups": duplicate_groups,
        "decoded_image_count": len(records),
        "validated_annotation_count": len(records),
        "review_status": "awaiting blind human review",
    }

    manifest = {"summary": summary, "records": records}
    (output_root / "selection_manifest_private.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_csv(
        output_root / "selection_manifest_private.csv",
        [
            "blind_id",
            "emotion",
            "image_id",
            "official_split",
            "class_rank",
            "source_image_entry",
            "source_annotation_entry",
            "local_image_filename",
            "local_annotation_filename",
            "image_sha256",
            "annotation_sha256",
            "image_zip_crc32",
            "annotation_zip_crc32",
            "image_size_bytes",
            "annotation_size_bytes",
            "image_format",
            "width",
            "height",
            "blind_order_key",
        ],
        records,
    )
    write_csv(
        output_root / "blind_review.csv",
        ["blind_id", "independent_label", "confidence_1_to_3", "blind_notes"],
        [{"blind_id": record["blind_id"]} for record in records],
    )
    write_csv(
        output_root / "reveal_review_private.csv",
        [
            "blind_id",
            "emoset_label",
            "image_id",
            "independent_label",
            "agreement",
            "review_flag",
            "reveal_notes",
        ],
        [
            {
                "blind_id": record["blind_id"],
                "emoset_label": record["emotion"],
                "image_id": record["image_id"],
            }
            for record in records
        ],
    )

    draw_contact_sheet(
        records,
        image_directory,
        output_root / "contact_sheet_blind.jpg",
        labeled=False,
    )
    draw_contact_sheet(
        records,
        image_directory,
        output_root / "contact_sheet_labeled_private.jpg",
        labeled=True,
    )

    log_lines = [
        "EmoSet Eval80-v1 local preparation log",
        f"Created (UTC): {created_at}",
        f"Source archive: {archive_path}",
        f"Source archive SHA-256: {archive_sha256}",
        "Selection rule: first ten available JPG files in filename order per class",
        f"Selected images: {len(records)}",
        f"Class counts: {json.dumps(summary['class_counts'], sort_keys=True)}",
        f"Official split counts: {json.dumps(summary['official_split_counts'], sort_keys=True)}",
        f"Missing expected files: {json.dumps(missing_expected, sort_keys=True)}",
        f"Substitutions: {json.dumps(substitutions, sort_keys=True)}",
        f"Exact duplicate groups: {json.dumps(duplicate_groups, sort_keys=True)}",
        "Next action: complete blind_review.csv before opening labeled private artifacts.",
    ]
    (output_root / "processing_log.txt").write_text(
        "\n".join(log_lines) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    args = parse_args()
    try:
        summary = prepare(args.archive, args.output_root)
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
