"""Migrate the 60 legacy excitement records to the shared audit shape.

This is a deterministic format and provenance migration.  It does not call a
model and it does not reinterpret the three known label-conflicting targets.
Those records remain in the frozen classification pool but are explicitly
rejected for joint-guidance training.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any, Sequence
import zipfile

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.annotation_pipeline import render_target_response, sha256_bytes


EXPECTED_EMOTION = "excitement"
KNOWN_LABEL_CONFLICTS = frozenset(
    {"excitement_05016", "excitement_05052", "excitement_05058"}
)
BRIDGE_GUIDANCE_IDS = frozenset(
    {
        "excitement_05000",
        "excitement_05001",
        "excitement_05002",
        "excitement_05003",
        "excitement_05004",
        "excitement_05005",
        "excitement_05006",
        "excitement_05007",
        "excitement_05008",
        "excitement_05010",
        "excitement_05011",
        "excitement_05012",
    }
)

TARGET_SECTIONS = re.compile(
    r"\AEmotion:\s*(?P<emotion>[^\n]+)\n\n"
    r"Visual evidence:\n(?P<evidence>.*?)\n\n"
    r"Aesthetic relationship\"?:\n(?P<aesthetic>.*?)\n\n"
    r"Guidance:\n(?P<guidance>.*?)\n\n"
    r"Final emotion:\s*(?P<final_emotion>[^\n]+)\s*\Z",
    re.DOTALL,
)


class ExcitementMigrationError(ValueError):
    """Raised when a legacy record cannot be migrated without guessing."""


@dataclass(frozen=True)
class ParsedTarget:
    emotion: str
    evidence: tuple[str, ...]
    aesthetic_relationship: str
    guidance: tuple[str, ...]
    final_emotion: str


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_target(value: str) -> ParsedTarget:
    match = TARGET_SECTIONS.fullmatch(value)
    if not match:
        raise ExcitementMigrationError("legacy target does not contain the expected sections")

    def bullets(section: str, field: str) -> tuple[str, ...]:
        values = tuple(
            line[2:].strip()
            for line in section.splitlines()
            if line.strip().startswith("- ")
        )
        if not values or any(not item for item in values):
            raise ExcitementMigrationError(f"legacy target has invalid {field} bullets")
        return values

    aesthetic = " ".join(match.group("aesthetic").split())
    if not aesthetic:
        raise ExcitementMigrationError("legacy target has an empty aesthetic relationship")
    return ParsedTarget(
        emotion=match.group("emotion").strip(),
        evidence=bullets(match.group("evidence"), "visual evidence"),
        aesthetic_relationship=aesthetic,
        guidance=bullets(match.group("guidance"), "guidance"),
        final_emotion=match.group("final_emotion").strip(),
    )


def selected_evidence(source_id: str, parsed: ParsedTarget) -> tuple[str, str]:
    if len(parsed.evidence) < 2:
        raise ExcitementMigrationError(f"{source_id} has fewer than two evidence items")
    if source_id == "excitement_05001":
        return (
            "The guitarist is actively strumming while gripping the fretboard in a close live-performance setting.",
            "His forward-leaning posture and focused downward gaze show intense engagement with the performance.",
        )
    return parsed.evidence[0], parsed.evidence[1]


def migrate_record(
    record: dict[str, Any],
    *,
    split: str,
    archive: zipfile.ZipFile,
) -> dict[str, Any]:
    source_id = record.get("source_image_id")
    if not isinstance(source_id, str) or not re.fullmatch(r"excitement_\d+", source_id):
        raise ExcitementMigrationError(f"invalid excitement source ID: {source_id!r}")
    if record.get("split") != split:
        raise ExcitementMigrationError(f"{source_id} split differs from its source file")
    expected_image_relpath = record.get("image_relpath")
    if not isinstance(expected_image_relpath, str) or not expected_image_relpath.startswith(
        f"image/excitement/{source_id}."
    ):
        raise ExcitementMigrationError(f"{source_id} has an invalid image_relpath")
    annotation_relpath = f"annotation/excitement/{source_id}.json"
    image_bytes = archive.read(expected_image_relpath)
    annotation_bytes = archive.read(annotation_relpath)
    annotation = json.loads(annotation_bytes)
    if annotation.get("emotion") != EXPECTED_EMOTION:
        raise ExcitementMigrationError(f"{source_id} archive label is not excitement")

    parsed = parse_target(str(record.get("target_response", "")))
    evidence = selected_evidence(source_id, parsed)
    if len(parsed.guidance) != 2:
        raise ExcitementMigrationError(
            f"{source_id} has {len(parsed.guidance)} guidance items; expected two"
        )
    migrated_target = render_target_response(
        {
            "emotion": parsed.emotion,
            "visual_evidence": evidence,
            "aesthetic_relationship": parsed.aesthetic_relationship,
            "guidance": parsed.guidance,
            "final_emotion": parsed.final_emotion,
        }
    )

    original_provenance = record.get("provenance")
    if not isinstance(original_provenance, dict):
        raise ExcitementMigrationError(f"{source_id} provenance is invalid")
    provenance = dict(original_provenance)
    provenance.update(
        {
            "source": "EmoSet-118K",
            "image_sha256": sha256_bytes(image_bytes),
            "annotation_sha256": sha256_bytes(annotation_bytes),
            "generator_provider": "legacy-unknown",
            "generator_model": "legacy-unknown",
            "prompt_version": "legacy-excitement-annotation-v1",
            "temperature": None,
            "seed": "unknown",
            "protocol_version": "legacy-excitement-migrated-v1",
            "format_migration": {
                "tool": "phase3/tools/migrate_excitement_records.py",
                "version": 1,
                "method": "Preserve legacy sections and deterministically retain two evidence items; no model or API call.",
                "selected_for_bridge_train_v1_guidance": source_id in BRIDGE_GUIDANCE_IDS,
            },
        }
    )

    migrated = dict(record)
    migrated["emotion"] = EXPECTED_EMOTION
    migrated["target_response"] = migrated_target
    migrated["provenance"] = provenance
    if source_id in KNOWN_LABEL_CONFLICTS:
        provenance.update(
            {
                "label_supported": False,
                "label_support_notes": (
                    "The legacy target uses a different emotion from the frozen EmoSet label. "
                    "It is retained for audit only and excluded from joint-guidance training."
                ),
                "review_status": "rejected",
                "review_notes": "Rejected during BridgeTrain-v1 migration because the target label conflicts with the archive label.",
            }
        )
    elif source_id in BRIDGE_GUIDANCE_IDS:
        migration_review_note = (
            "Legacy accepted target was visually checked and normalized to exactly "
            "two evidence items; no model or API regeneration was used."
        )
        existing_notes = str(provenance.get("review_notes", "")).strip()
        review_notes = existing_notes
        if migration_review_note not in existing_notes:
            review_notes = (
                f"{existing_notes} {migration_review_note}".strip()
                if existing_notes
                else migration_review_note
            )
        provenance.update(
            {
                "label_supported": True,
                "label_support_notes": (
                    "The retained evidence was checked against the source image during the "
                    "BridgeTrain-v1 migration."
                ),
                "review_status": "edited",
                "review_notes": review_notes,
            }
        )
    else:
        provenance.update(
            {
                "label_supported": None,
                "label_support_notes": (
                    "Not reassessed for the frozen BridgeTrain-v1 guidance subset; available "
                    "for classification only unless separately reviewed."
                ),
            }
        )
    return migrated


def migrate_files(*, archive_path: Path, dataset_root: Path, apply: bool) -> dict[str, Any]:
    paths = {
        split: dataset_root / "excitement" / f"{split}_excitement.jsonl"
        for split in ("train", "validation")
    }
    original_by_path: dict[Path, list[dict[str, Any]]] = {}
    migrated_by_path: dict[Path, list[dict[str, Any]]] = {}
    with zipfile.ZipFile(archive_path) as archive:
        for split, path in paths.items():
            records = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            original_by_path[path] = records
            migrated_by_path[path] = [
                migrate_record(record, split=split, archive=archive) for record in records
            ]

    all_records = [record for records in migrated_by_path.values() for record in records]
    ids = {record["source_image_id"] for record in all_records}
    if len(all_records) != 60 or len(ids) != 60:
        raise ExcitementMigrationError("migration must preserve exactly 60 unique records")
    if not BRIDGE_GUIDANCE_IDS <= ids or not KNOWN_LABEL_CONFLICTS <= ids:
        raise ExcitementMigrationError("migration input is missing a frozen guidance or conflict ID")

    if apply:
        for path, records in migrated_by_path.items():
            if records == original_by_path[path]:
                continue
            payload = "".join(
                json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
                for record in records
            )
            descriptor, temporary_name = tempfile.mkstemp(
                dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
            )
            temporary = Path(temporary_name)
            try:
                with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as output:
                    output.write(payload)
                    output.flush()
                    os.fsync(output.fileno())
                os.replace(temporary, path)
            finally:
                if temporary.exists():
                    temporary.unlink()

    return {
        "valid": True,
        "applied": apply,
        "record_count": len(all_records),
        "bridge_guidance_record_count": len(BRIDGE_GUIDANCE_IDS),
        "rejected_label_conflict_count": len(KNOWN_LABEL_CONFLICTS),
        "changed_file_count": sum(
            records != original_by_path[path]
            for path, records in migrated_by_path.items()
        ),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=project_root() / "data/datasets")
    parser.add_argument("--apply", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = migrate_files(
        archive_path=args.archive.resolve(),
        dataset_root=args.dataset_root.resolve(),
        apply=args.apply,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
