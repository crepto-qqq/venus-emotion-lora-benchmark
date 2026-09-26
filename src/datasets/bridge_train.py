"""Build and verify the BridgeTrain-v1 Qwen-VL training release.

The repository JSONL files are annotation and review records.  They are not
fed to the model directly because their annotation prompt discloses the fixed
EmoSet label.  This module derives two label-neutral training tasks:

* one emotion-classification conversation for every frozen image; and
* one B1 joint-guidance conversation for a balanced reviewed subset.

Images stay in the private EmoSet archive.  Git-safe release metadata contains
only identifiers, hashes, review provenance, and runtime image references.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
from typing import Any, Iterable, Mapping, Sequence
import zipfile

from src.datasets.annotation_pipeline import (
    EMOTION_CLASSES,
    EVAL20_IMAGE_IDS,
    TARGET_PATTERN,
    eval80_ids as reproduce_eval80_ids,
    read_jsonl,
    sha256_bytes,
)


SPLITS = ("train", "validation")
TASK_CLASSIFICATION = "classification"
TASK_GUIDANCE = "joint_guidance"
ACCEPTED_REVIEW_STATUSES = frozenset({"accepted", "edited"})
IMAGE_TAG_PATTERN = re.compile(r"<img>([^<>]+)</img>")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


class BridgeReleaseError(ValueError):
    """Raised when a release invariant is not satisfied."""


# Public spelling retained for the Member 2 build API.
BridgeBuildError = BridgeReleaseError


@dataclass(frozen=True)
class BridgeBuildConfig:
    emotions: tuple[str, ...]
    images_per_class_by_split: Mapping[str, int]
    guidance_per_class_by_split: Mapping[str, int]
    classification_prompt: str
    classification_prompt_version: str
    joint_prompt: str
    joint_prompt_version: str
    selection_seed: str | int
    dataset_id: str = "BridgeTrain-v1"
    protocol_version: str = "bridge-train-v1"
    require_label_supported: bool = True
    guidance_source_ids_by_split: Mapping[str, Mapping[str, tuple[str, ...]]] | None = None
    post_correction_human_signoff: bool = False
    pending_human_review_ids: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "BridgeBuildConfig":
        prompts = value.get("prompts", {})
        classification = prompts.get("classification", {})
        guidance = prompts.get("joint_guidance", {})
        quotas = value.get("quotas", {})
        raw_guidance_ids = value.get("guidance_source_ids")
        release_approval = value.get("release_approval", {})
        if not isinstance(release_approval, Mapping):
            raise BridgeReleaseError("release_approval must be an object")
        guidance_ids = None
        if raw_guidance_ids is not None:
            if not isinstance(raw_guidance_ids, Mapping):
                raise BridgeReleaseError("guidance_source_ids must be an object")
            guidance_ids = {
                split: {
                    emotion: tuple(raw_guidance_ids.get(split, {}).get(emotion, ()))
                    for emotion in value.get("emotion_classes", ())
                }
                for split in SPLITS
            }
        config = cls(
            emotions=tuple(value.get("emotion_classes", ())),
            images_per_class_by_split={
                split: int(quotas.get(split, {}).get("images_per_class", 0))
                for split in SPLITS
            },
            guidance_per_class_by_split={
                split: int(quotas.get(split, {}).get("guidance_per_class", 0))
                for split in SPLITS
            },
            classification_prompt=str(classification.get("text", "")),
            classification_prompt_version=str(classification.get("version", "")),
            joint_prompt=str(guidance.get("text", "")),
            joint_prompt_version=str(guidance.get("version", "")),
            selection_seed=str(value.get("guidance_selection_seed", "")),
            dataset_id=str(value.get("dataset_id", "")),
            protocol_version=str(value.get("protocol_version", "")),
            require_label_supported=bool(value.get("require_label_supported", True)),
            guidance_source_ids_by_split=guidance_ids,
            post_correction_human_signoff=bool(
                release_approval.get("post_correction_human_signoff", False)
            ),
            pending_human_review_ids=tuple(
                release_approval.get("pending_human_review_ids", ())
            ),
        )
        config.validate()
        expected_joint_hash = guidance.get("sha256")
        if expected_joint_hash and sha256_text(config.joint_prompt) != expected_joint_hash:
            raise BridgeReleaseError("joint-guidance prompt differs from its frozen SHA-256")
        expected_classification_hash = classification.get("sha256")
        if (
            expected_classification_hash
            and sha256_text(config.classification_prompt) != expected_classification_hash
        ):
            raise BridgeReleaseError("classification prompt differs from its frozen SHA-256")
        return config

    def validate(self) -> None:
        if not self.dataset_id or not self.protocol_version:
            raise BridgeReleaseError("dataset_id and protocol_version must be non-empty")
        if not self.emotions or len(set(self.emotions)) != len(self.emotions):
            raise BridgeReleaseError("emotion_classes must be a non-empty unique list")
        unknown = sorted(set(self.emotions) - set(EMOTION_CLASSES))
        if unknown:
            raise BridgeReleaseError(f"unsupported emotions: {', '.join(unknown)}")
        for split in SPLITS:
            image_count = int(self.images_per_class_by_split.get(split, 0))
            guidance_count = int(self.guidance_per_class_by_split.get(split, 0))
            if image_count <= 0:
                raise BridgeReleaseError(f"{split} images_per_class must be positive")
            if guidance_count < 0 or guidance_count > image_count:
                raise BridgeReleaseError(f"{split} guidance_per_class is invalid")
        for field, text in (
            ("classification prompt", self.classification_prompt),
            ("joint-guidance prompt", self.joint_prompt),
            ("selection seed", str(self.selection_seed)),
        ):
            if not text.strip():
                raise BridgeReleaseError(f"{field} must be non-empty")
        leaked_labels = [
            emotion
            for emotion in self.emotions
            if re.search(rf"assigned[^\n]{{0,40}}\b{re.escape(emotion)}\b", self.classification_prompt, re.I)
        ]
        if leaked_labels:
            raise BridgeReleaseError("classification prompt contains an assigned-label disclosure")
        if len(set(self.pending_human_review_ids)) != len(self.pending_human_review_ids):
            raise BridgeReleaseError("pending_human_review_ids contains duplicates")
        if self.post_correction_human_signoff and self.pending_human_review_ids:
            raise BridgeReleaseError(
                "signed-off release must not retain pending_human_review_ids"
            )
        if self.guidance_source_ids_by_split is not None:
            selected: set[str] = set()
            for split in SPLITS:
                by_emotion = self.guidance_source_ids_by_split.get(split, {})
                for emotion in self.emotions:
                    source_ids = tuple(by_emotion.get(emotion, ()))
                    expected = int(self.guidance_per_class_by_split[split])
                    if len(source_ids) != expected:
                        raise BridgeReleaseError(
                            f"guidance_source_ids {emotion}:{split} has {len(source_ids)} IDs; "
                            f"expected {expected}"
                        )
                    if len(set(source_ids)) != len(source_ids):
                        raise BridgeReleaseError(
                            f"guidance_source_ids {emotion}:{split} contains duplicates"
                        )
                    for source_id in source_ids:
                        if not re.fullmatch(rf"{re.escape(emotion)}_\d+", source_id):
                            raise BridgeReleaseError(
                                f"guidance source ID is outside class {emotion}: {source_id}"
                            )
                        if source_id in selected:
                            raise BridgeReleaseError(
                                f"guidance source ID appears more than once: {source_id}"
                            )
                        selected.add(source_id)
            unknown_pending = sorted(set(self.pending_human_review_ids) - selected)
            if unknown_pending:
                raise BridgeReleaseError(
                    "pending_human_review_ids are outside the frozen guidance set: "
                    + ", ".join(unknown_pending)
                )


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any, *, pretty: bool) -> str:
    if pretty:
        return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _validate_sha256(value: Any, field: str) -> str:
    rendered = str(value)
    if not SHA256_PATTERN.fullmatch(rendered):
        raise BridgeReleaseError(f"{field} must be a lowercase SHA-256")
    return rendered


def _validate_archive_member(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise BridgeReleaseError(f"{field} must be a non-empty string")
    if "\\" in value:
        raise BridgeReleaseError(f"{field} must use POSIX separators")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise BridgeReleaseError(f"{field} is not a safe archive-relative path: {value}")
    if str(path) != value:
        raise BridgeReleaseError(f"{field} is not canonical: {value}")
    return value


def _runtime_image_path(root: str, image_relpath: str) -> str:
    if not isinstance(root, str) or not root.startswith("/") or "\\" in root:
        raise BridgeReleaseError("image_reference_root must be an absolute POSIX path")
    root_path = PurePosixPath(root)
    if ".." in root_path.parts:
        raise BridgeReleaseError("image_reference_root must not contain '..'")
    return str(root_path / PurePosixPath(image_relpath))


def _source_path(source_root: Path, emotion: str, split: str) -> Path:
    return source_root / emotion / f"{split}_{emotion}.jsonl"


def _record_sha256(record: Mapping[str, Any]) -> str:
    return sha256_text(canonical_json(record, pretty=False))


def _guidance_errors(record: Mapping[str, Any], assigned_emotion: str, config: BridgeBuildConfig) -> list[str]:
    errors: list[str] = []
    provenance = record.get("provenance")
    if not isinstance(provenance, Mapping):
        return ["provenance is missing or invalid"]
    status = provenance.get("review_status")
    if status not in ACCEPTED_REVIEW_STATUSES:
        errors.append(f"review_status={status!r}")
    if config.require_label_supported and provenance.get("label_supported") is not True:
        errors.append("label_supported is not true")
    target = record.get("target_response")
    if not isinstance(target, str) or not target:
        errors.append("target_response is empty")
        return errors
    if any(
        0x2E80 <= ord(character) <= 0xA4CF
        or 0xE000 <= ord(character) <= 0xF8FF
        or 0xFF00 <= ord(character) <= 0xFFEF
        for character in target
    ):
        errors.append("target_response contains non-English or corrupted characters")
    match = TARGET_PATTERN.fullmatch(target)
    if not match:
        errors.append("target_response does not match structured_joint_v1")
    elif (
        match.group("emotion") != assigned_emotion
        or match.group("final_emotion") != assigned_emotion
    ):
        errors.append("target_response emotion differs from the fixed label")
    return errors


def _conversation(sample_id: str, image_path: str, prompt: str, target: str) -> dict[str, Any]:
    return {
        "id": sample_id,
        "conversations": [
            {
                "from": "user",
                "value": f"Picture 1: <img>{image_path}</img>\n{prompt}",
            },
            {"from": "assistant", "value": target},
        ],
    }


def _counter_dict(counter: Counter[tuple[str, str, str]]) -> dict[str, int]:
    return {
        f"{split}:{emotion}:{task}": count
        for (split, emotion, task), count in sorted(counter.items())
    }


def build_bridge_release(
    *,
    source_root: Path,
    archive_path: Path,
    eval20_ids: Iterable[str],
    eval80_ids: Iterable[str],
    eval20_hashes: Iterable[str],
    eval80_hashes: Iterable[str],
    image_reference_root: str,
    config: BridgeBuildConfig,
) -> dict[str, Any]:
    """Build a fully validated release in memory without writing partial files."""

    config.validate()
    source_root = source_root.resolve()
    archive_path = archive_path.resolve()
    if not archive_path.is_file():
        raise BridgeReleaseError(f"EmoSet archive does not exist: {archive_path}")

    excluded_ids = set(eval20_ids) | set(eval80_ids)
    excluded_hashes = set(eval20_hashes) | set(eval80_hashes)
    for value in excluded_hashes:
        _validate_sha256(value, "evaluation image hash")

    records: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_relpaths: set[str] = set()
    seen_hashes: dict[str, str] = {}
    split_ids: defaultdict[str, set[str]] = defaultdict(set)
    split_hashes: defaultdict[str, set[str]] = defaultdict(set)

    with zipfile.ZipFile(archive_path) as archive:
        archive_names = set(archive.namelist())
        for emotion in config.emotions:
            for split in SPLITS:
                path = _source_path(source_root, emotion, split)
                if not path.is_file():
                    raise BridgeReleaseError(f"source JSONL is missing: {path}")
                source_records = read_jsonl(path)
                expected = int(config.images_per_class_by_split[split])
                if len(source_records) != expected:
                    raise BridgeReleaseError(
                        f"{path} has {len(source_records)} records; expected {expected}"
                    )
                for line_number, record in enumerate(source_records, start=1):
                    source_id = record.get("source_image_id")
                    if not isinstance(source_id, str) or not re.fullmatch(
                        rf"{re.escape(emotion)}_\d+", source_id
                    ):
                        raise BridgeReleaseError(
                            f"{path}:{line_number} has an ID outside class {emotion}: {source_id!r}"
                        )
                    if source_id in seen_ids:
                        raise BridgeReleaseError(f"duplicate source_image_id: {source_id}")
                    seen_ids.add(source_id)
                    if record.get("emotion") != emotion:
                        raise BridgeReleaseError(
                            f"{source_id} record emotion differs from its frozen class"
                        )
                    if record.get("split") != split:
                        raise BridgeReleaseError(f"{source_id} split differs from its source file")

                    image_relpath = _validate_archive_member(
                        record.get("image_relpath"), field=f"{source_id} image_relpath"
                    )
                    expected_prefix = f"image/{emotion}/{source_id}."
                    if not image_relpath.startswith(expected_prefix):
                        raise BridgeReleaseError(
                            f"{source_id} image_relpath does not match its ID and class"
                        )
                    annotation_relpath = f"annotation/{emotion}/{source_id}.json"
                    if image_relpath not in archive_names:
                        raise BridgeReleaseError(f"archive image is missing: {image_relpath}")
                    if annotation_relpath not in archive_names:
                        raise BridgeReleaseError(
                            f"archive annotation is missing: {annotation_relpath}"
                        )
                    if image_relpath in seen_relpaths:
                        raise BridgeReleaseError(f"duplicate image path: {image_relpath}")
                    seen_relpaths.add(image_relpath)

                    image_bytes = archive.read(image_relpath)
                    annotation_bytes = archive.read(annotation_relpath)
                    image_hash = sha256_bytes(image_bytes)
                    annotation_hash = sha256_bytes(annotation_bytes)
                    if image_hash in seen_hashes:
                        raise BridgeReleaseError(
                            f"duplicate image bytes: {source_id} and {seen_hashes[image_hash]}"
                        )
                    seen_hashes[image_hash] = source_id
                    provenance = record.get("provenance")
                    if not isinstance(provenance, Mapping):
                        raise BridgeReleaseError(f"{source_id} provenance is missing or invalid")
                    recorded_image_hash = _validate_sha256(
                        provenance.get("image_sha256"), f"{source_id} provenance.image_sha256"
                    )
                    recorded_annotation_hash = _validate_sha256(
                        provenance.get("annotation_sha256"),
                        f"{source_id} provenance.annotation_sha256",
                    )
                    if recorded_image_hash != image_hash:
                        raise BridgeReleaseError(f"{source_id} image hash differs from the archive")
                    if recorded_annotation_hash != annotation_hash:
                        raise BridgeReleaseError(
                            f"{source_id} annotation hash differs from the archive"
                        )
                    try:
                        source_annotation = json.loads(annotation_bytes)
                    except json.JSONDecodeError as error:
                        raise BridgeReleaseError(
                            f"{source_id} archive annotation is invalid JSON"
                        ) from error
                    if source_annotation.get("emotion") != emotion:
                        raise BridgeReleaseError(
                            f"{source_id} archive annotation emotion differs from {emotion}"
                        )
                    if source_annotation.get("image_id") != source_id:
                        raise BridgeReleaseError(
                            f"{source_id} archive annotation image ID differs from its source ID"
                        )
                    if source_id in excluded_ids:
                        raise BridgeReleaseError(f"evaluation ID leaked into BridgeTrain-v1: {source_id}")
                    if image_hash in excluded_hashes:
                        raise BridgeReleaseError(
                            f"evaluation image hash leaked into BridgeTrain-v1: {source_id}"
                        )

                    guidance_errors = _guidance_errors(record, emotion, config)
                    if guidance_errors:
                        rejections.append(
                            {
                                "source_image_id": source_id,
                                "emotion": emotion,
                                "split": split,
                                "review_status": provenance.get("review_status"),
                                "reasons": guidance_errors,
                            }
                        )
                    record_copy = dict(record)
                    record_copy["_assigned_emotion"] = emotion
                    try:
                        source_reference = path.relative_to(project_root()).as_posix()
                    except ValueError:
                        source_reference = f"{emotion}/{path.name}"
                    record_copy["_source_path"] = source_reference
                    record_copy["_source_line"] = line_number
                    record_copy["_source_record_sha256"] = _record_sha256(record)
                    record_copy["_image_sha256"] = image_hash
                    record_copy["_annotation_sha256"] = annotation_hash
                    record_copy["_annotation_relpath"] = annotation_relpath
                    record_copy["_guidance_errors"] = guidance_errors
                    records.append(record_copy)
                    split_ids[split].add(source_id)
                    split_hashes[split].add(image_hash)

    if split_ids["train"] & split_ids["validation"]:
        raise BridgeReleaseError("train and validation share source IDs")
    if split_hashes["train"] & split_hashes["validation"]:
        raise BridgeReleaseError("train and validation share image hashes")

    records_by_id = {record["source_image_id"]: record for record in records}
    selected_guidance: set[str] = set()
    selection_rank: dict[str, int] = {}
    for emotion in config.emotions:
        for split in SPLITS:
            required = int(config.guidance_per_class_by_split[split])
            if config.guidance_source_ids_by_split is not None:
                selected_ids = tuple(
                    config.guidance_source_ids_by_split[split][emotion]
                )
                for source_id in selected_ids:
                    record = records_by_id.get(source_id)
                    if record is None:
                        raise BridgeReleaseError(
                            f"frozen guidance source does not exist: {source_id}"
                        )
                    if record["_assigned_emotion"] != emotion or record["split"] != split:
                        raise BridgeReleaseError(
                            f"frozen guidance source has the wrong class or split: {source_id}"
                        )
                    if record["_guidance_errors"]:
                        raise BridgeReleaseError(
                            f"frozen guidance source is ineligible: {source_id}: "
                            + "; ".join(record["_guidance_errors"])
                        )
            else:
                eligible = [
                    record
                    for record in records
                    if record["_assigned_emotion"] == emotion
                    and record["split"] == split
                    and not record["_guidance_errors"]
                ]
                eligible.sort(
                    key=lambda record: (
                        sha256_text(
                            f"{config.selection_seed}|{split}|{emotion}|{record['source_image_id']}"
                        ),
                        record["source_image_id"],
                    )
                )
                if len(eligible) < required:
                    raise BridgeReleaseError(
                        f"{emotion}:{split} has {len(eligible)} eligible guidance records; "
                        f"{required} are required"
                    )
                selected_ids = tuple(
                    record["source_image_id"] for record in eligible[:required]
                )
            for rank, source_id in enumerate(selected_ids, start=1):
                selected_guidance.add(source_id)
                selection_rank[source_id] = rank

    samples_by_split: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    manifest: list[dict[str, Any]] = []
    counts: Counter[tuple[str, str, str]] = Counter()
    for record in sorted(
        records,
        key=lambda item: (
            item["split"],
            item["_assigned_emotion"],
            item["source_image_id"],
        ),
    ):
        source_id = record["source_image_id"]
        emotion = record["_assigned_emotion"]
        split = record["split"]
        runtime_path = _runtime_image_path(image_reference_root, record["image_relpath"])
        provenance = record["provenance"]
        task_specs = [
            (
                TASK_CLASSIFICATION,
                config.classification_prompt,
                config.classification_prompt_version,
                emotion,
                "not_applicable",
            )
        ]
        if source_id in selected_guidance:
            task_specs.append(
                (
                    TASK_GUIDANCE,
                    config.joint_prompt,
                    config.joint_prompt_version,
                    record["target_response"],
                    provenance["review_status"],
                )
            )

        for task, prompt, prompt_version, target, review_status in task_specs:
            sample_id = f"{source_id}__{task}"
            sample = _conversation(sample_id, runtime_path, prompt, target)
            samples_by_split[split].append(sample)
            counts[(split, emotion, task)] += 1
            sidecar = {
                "sample_id": sample_id,
                "source_image_id": source_id,
                "emotion": emotion,
                "split": split,
                "task_type": task,
                "image_relpath": record["image_relpath"],
                "runtime_image_path": runtime_path,
                "image_sha256": record["_image_sha256"],
                "annotation_relpath": record["_annotation_relpath"],
                "annotation_sha256": record["_annotation_sha256"],
                "source_record_path": record["_source_path"],
                "source_record_line": record["_source_line"],
                "source_record_sha256": record["_source_record_sha256"],
                "source_review_status": provenance.get("review_status"),
                "review_status": review_status,
                "reviewed_at_utc": provenance.get("reviewed_at_utc"),
                "prompt_version": prompt_version,
                "prompt_sha256": sha256_text(prompt),
            }
            if task == TASK_GUIDANCE:
                sidecar["guidance_selection_rank"] = selection_rank[source_id]
                sidecar["generator_provider"] = provenance.get("generator_provider")
                sidecar["generator_model"] = provenance.get("generator_model")
                if provenance.get("replaces_source_image_id"):
                    sidecar["replaces_source_image_id"] = provenance[
                        "replaces_source_image_id"
                    ]
            manifest.append(sidecar)

    for split in SPLITS:
        samples_by_split[split].sort(key=lambda item: item["id"])
    manifest.sort(key=lambda item: item["sample_id"])
    rejections.sort(key=lambda item: (item["split"], item["emotion"], item["source_image_id"]))

    _validate_built_release(
        samples_by_split=samples_by_split,
        manifest=manifest,
        config=config,
    )
    summary = {
        "schema_version": 1,
        "dataset_id": config.dataset_id,
        "protocol_version": config.protocol_version,
        "machine_validation_passed": True,
        "full_dataset_ready": config.post_correction_human_signoff,
        "human_review_signoff_pending": not config.post_correction_human_signoff,
        "pending_human_review_record_count": len(config.pending_human_review_ids),
        "pending_human_review_ids": list(config.pending_human_review_ids),
        "formal_training_authorized": False,
        "source_image_count": len(records),
        "conversation_count": len(manifest),
        "split_unique_image_counts": {
            split: len(split_ids[split]) for split in SPLITS
        },
        "split_conversation_counts": {
            split: len(samples_by_split[split]) for split in SPLITS
        },
        "guidance_image_count": len(selected_guidance),
        "counts": _counter_dict(counts),
        "guidance_ineligible_record_count": len(rejections),
        "integrity_checks": {
            "archive_images_and_annotations": "passed",
            "recorded_hashes": "passed",
            "duplicate_ids_paths_and_hashes": "passed",
            "train_validation_disjoint": "passed",
            "eval20_id_and_hash_exclusion": "passed",
            "eval80_id_and_hash_exclusion": "passed",
            "balanced_class_and_task_quotas": "passed",
            "accepted_guidance_only": "passed",
            "label_neutral_prompts": "passed",
        },
        "prompt_sha256": {
            TASK_CLASSIFICATION: sha256_text(config.classification_prompt),
            TASK_GUIDANCE: sha256_text(config.joint_prompt),
        },
    }
    summary["unique_image_counts"] = dict(summary["split_unique_image_counts"])
    summary["conversation_counts"] = dict(summary["split_conversation_counts"])
    return {
        "train": samples_by_split["train"],
        "validation": samples_by_split["validation"],
        "manifest": manifest,
        "rejections": rejections,
        "summary": summary,
    }


def _validate_built_release(
    *,
    samples_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
    manifest: Sequence[Mapping[str, Any]],
    config: BridgeBuildConfig,
) -> None:
    manifest_by_id = {entry["sample_id"]: entry for entry in manifest}
    if len(manifest_by_id) != len(manifest):
        raise BridgeReleaseError("release manifest contains duplicate sample IDs")
    seen_sample_ids: set[str] = set()
    seen_tasks: set[tuple[str, str]] = set()
    source_tasks: defaultdict[str, set[str]] = defaultdict(set)
    source_splits: defaultdict[str, set[str]] = defaultdict(set)
    source_hashes: defaultdict[str, set[str]] = defaultdict(set)
    hash_to_source: dict[str, str] = {}
    hash_to_split: dict[str, str] = {}

    for split in SPLITS:
        for sample in samples_by_split[split]:
            sample_id = sample.get("id")
            if not isinstance(sample_id, str) or sample_id in seen_sample_ids:
                raise BridgeReleaseError(f"duplicate or invalid training sample ID: {sample_id!r}")
            seen_sample_ids.add(sample_id)
            conversations = sample.get("conversations")
            if not isinstance(conversations, list) or len(conversations) != 2:
                raise BridgeReleaseError(f"{sample_id} must contain exactly two conversations")
            if conversations[0].get("from") != "user" or conversations[1].get("from") != "assistant":
                raise BridgeReleaseError(f"{sample_id} conversation roles must be user then assistant")
            user_value = conversations[0].get("value")
            assistant_value = conversations[1].get("value")
            if not isinstance(user_value, str) or len(IMAGE_TAG_PATTERN.findall(user_value)) != 1:
                raise BridgeReleaseError(f"{sample_id} must contain exactly one image tag")
            if not isinstance(assistant_value, str) or not assistant_value.strip():
                raise BridgeReleaseError(f"{sample_id} assistant target is empty")

            sidecar = manifest_by_id.get(sample_id)
            if sidecar is None:
                raise BridgeReleaseError(f"{sample_id} is missing from the sidecar manifest")
            if sidecar["split"] != split:
                raise BridgeReleaseError(f"{sample_id} sidecar split mismatch")
            image_match = IMAGE_TAG_PATTERN.search(user_value)
            image_path = image_match.group(1)  # type: ignore[union-attr]
            if image_path != sidecar["runtime_image_path"]:
                raise BridgeReleaseError(f"{sample_id} image path differs from its sidecar")
            prompt = user_value.split("</img>\n", 1)[1] if "</img>\n" in user_value else ""
            source_id = sidecar["source_image_id"]
            task = sidecar["task_type"]
            emotion = sidecar.get("emotion")
            if emotion not in config.emotions:
                raise BridgeReleaseError(f"{sample_id} has an unsupported sidecar emotion")
            if not re.fullmatch(rf"{re.escape(emotion)}_\d+", source_id):
                raise BridgeReleaseError(
                    f"{sample_id} source ID differs from its sidecar emotion"
                )
            if sample_id != f"{source_id}__{task}":
                raise BridgeReleaseError(
                    f"{sample_id} does not match its source ID and task type"
                )
            image_relpath = _validate_archive_member(
                sidecar.get("image_relpath"), field=f"{sample_id} image_relpath"
            )
            if not image_relpath.startswith(f"image/{emotion}/{source_id}."):
                raise BridgeReleaseError(
                    f"{sample_id} image path differs from its source ID and emotion"
                )
            expected_annotation_relpath = f"annotation/{emotion}/{source_id}.json"
            if sidecar.get("annotation_relpath") != expected_annotation_relpath:
                raise BridgeReleaseError(
                    f"{sample_id} annotation path differs from its source ID and emotion"
                )
            runtime_path = PurePosixPath(str(sidecar.get("runtime_image_path", "")))
            if (
                not runtime_path.is_absolute()
                or ".." in runtime_path.parts
                or str(runtime_path).replace("\\", "/") != image_path
                or not image_path.endswith(f"/{image_relpath}")
            ):
                raise BridgeReleaseError(f"{sample_id} has an invalid runtime image path")
            if (source_id, task) in seen_tasks:
                raise BridgeReleaseError(f"duplicate task for source image: {source_id} {task}")
            seen_tasks.add((source_id, task))
            source_tasks[source_id].add(task)
            source_splits[source_id].add(split)
            image_hash = _validate_sha256(
                sidecar.get("image_sha256"), f"{sample_id} image_sha256"
            )
            source_hashes[source_id].add(image_hash)
            previous_source = hash_to_source.setdefault(image_hash, source_id)
            if previous_source != source_id:
                raise BridgeReleaseError(
                    f"image hash is shared by different source IDs: {previous_source}, {source_id}"
                )
            previous_split = hash_to_split.setdefault(image_hash, split)
            if previous_split != split:
                raise BridgeReleaseError("an image hash appears in both train and validation")
            if task == TASK_CLASSIFICATION:
                if prompt != config.classification_prompt:
                    raise BridgeReleaseError(f"{sample_id} classification prompt differs from the lock")
                if assistant_value != sidecar["emotion"]:
                    raise BridgeReleaseError(f"{sample_id} classification target is not the fixed label")
                if sidecar["review_status"] != "not_applicable":
                    raise BridgeReleaseError(f"{sample_id} has an invalid classification review status")
                expected_prompt_version = config.classification_prompt_version
            elif task == TASK_GUIDANCE:
                if prompt != config.joint_prompt:
                    raise BridgeReleaseError(f"{sample_id} guidance prompt differs from the B1 lock")
                if sidecar["review_status"] not in ACCEPTED_REVIEW_STATUSES:
                    raise BridgeReleaseError(f"{sample_id} guidance is not accepted or edited")
                match = TARGET_PATTERN.fullmatch(assistant_value)
                if not match or match.group("emotion") != sidecar["emotion"] or match.group(
                    "final_emotion"
                ) != sidecar["emotion"]:
                    raise BridgeReleaseError(f"{sample_id} guidance target differs from its fixed label")
                if config.guidance_source_ids_by_split is not None:
                    expected_ids = tuple(
                        config.guidance_source_ids_by_split[split][emotion]
                    )
                    if source_id not in expected_ids:
                        raise BridgeReleaseError(
                            f"{sample_id} is outside the frozen guidance subset"
                        )
                    expected_rank = expected_ids.index(source_id) + 1
                    if sidecar.get("guidance_selection_rank") != expected_rank:
                        raise BridgeReleaseError(
                            f"{sample_id} guidance rank differs from the frozen selection"
                        )
                expected_prompt_version = config.joint_prompt_version
            else:
                raise BridgeReleaseError(f"{sample_id} has an unsupported task type")
            if sidecar.get("prompt_version") != expected_prompt_version:
                raise BridgeReleaseError(f"{sample_id} prompt version differs from the lock")
            if sidecar.get("prompt_sha256") != sha256_text(prompt):
                raise BridgeReleaseError(f"{sample_id} prompt hash differs from its conversation")

    if seen_sample_ids != set(manifest_by_id):
        raise BridgeReleaseError("sidecar manifest has extra or missing samples")
    if any(len(splits) != 1 for splits in source_splits.values()):
        raise BridgeReleaseError("a source image appears in more than one split")
    if any(len(hashes) != 1 for hashes in source_hashes.values()):
        raise BridgeReleaseError("a source ID maps to more than one image hash")
    for source_id, tasks in source_tasks.items():
        if tasks not in ({TASK_CLASSIFICATION}, {TASK_CLASSIFICATION, TASK_GUIDANCE}):
            raise BridgeReleaseError(f"{source_id} has an invalid task combination")

    counts = Counter(
        (entry["split"], entry["emotion"], entry["task_type"]) for entry in manifest
    )
    for emotion in config.emotions:
        for split in SPLITS:
            expected_images = int(config.images_per_class_by_split[split])
            expected_guidance = int(config.guidance_per_class_by_split[split])
            if counts[(split, emotion, TASK_CLASSIFICATION)] != expected_images:
                raise BridgeReleaseError(f"{emotion}:{split} classification quota is not met")
            if counts[(split, emotion, TASK_GUIDANCE)] != expected_guidance:
                raise BridgeReleaseError(f"{emotion}:{split} guidance quota is not met")


def load_config(path: Path) -> BridgeBuildConfig:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise BridgeReleaseError("BridgeTrain config must be a JSON object")
    return BridgeBuildConfig.from_mapping(value)


def load_eval20_exclusions(
    path: Path, archive_path: Path | None = None
) -> tuple[set[str], set[str]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    outputs = value.get("outputs")
    if not isinstance(outputs, list) or len(outputs) != 20:
        raise BridgeReleaseError("Eval20 artifact must contain exactly 20 outputs")
    ids = {item.get("image_id") for item in outputs}
    hashes = {item.get("sha256") for item in outputs}
    if None in ids or None in hashes or len(ids) != 20 or len(hashes) != 20:
        raise BridgeReleaseError("Eval20 artifact has missing or duplicate identities")
    if ids != set(EVAL20_IMAGE_IDS):
        raise BridgeReleaseError("Eval20 artifact IDs differ from the frozen exclusion list")
    for value_hash in hashes:
        _validate_sha256(value_hash, "Eval20 image hash")
    if archive_path is not None:
        output_hash_by_id = {item["image_id"]: item["sha256"] for item in outputs}
        with zipfile.ZipFile(archive_path) as archive:
            names = set(archive.namelist())
            for source_id in sorted(ids):
                emotion = source_id.rsplit("_", 1)[0]
                matches = sorted(
                    name
                    for name in names
                    if name.startswith(f"image/{emotion}/{source_id}.")
                    and name.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
                )
                if len(matches) != 1:
                    raise BridgeReleaseError(f"cannot resolve Eval20 image: {source_id}")
                if sha256_bytes(archive.read(matches[0])) != output_hash_by_id[source_id]:
                    raise BridgeReleaseError(
                        f"Eval20 artifact hash differs from the source archive: {source_id}"
                    )
    return set(ids), set(hashes)


def load_eval80_exclusions(
    manifest_path: Path,
    freeze_record_path: Path,
    archive_path: Path | None,
    emotions: Sequence[str],
) -> tuple[set[str], set[str]]:
    freeze_record = json.loads(freeze_record_path.read_text(encoding="utf-8"))
    expected_manifest_hash = freeze_record.get("file_sha256", {}).get("inference_manifest")
    if not isinstance(expected_manifest_hash, str):
        raise BridgeReleaseError("Eval80 freeze record does not lock the inference manifest")
    if sha256_file(manifest_path) != expected_manifest_hash:
        raise BridgeReleaseError("Eval80 inference manifest differs from its freeze record")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = manifest.get("records")
    if not isinstance(rows, list) or len(rows) != 80:
        raise BridgeReleaseError("Eval80 manifest must contain exactly 80 records")
    hashes = {row.get("image_sha256") for row in rows}
    if None in hashes or len(hashes) != 80:
        raise BridgeReleaseError("Eval80 manifest has missing or duplicate image hashes")
    for value_hash in hashes:
        _validate_sha256(value_hash, "Eval80 image hash")
    if archive_path is None:
        return set(), set(hashes)
    with zipfile.ZipFile(archive_path) as archive:
        ids = reproduce_eval80_ids(archive, emotions)
        reproduced_hashes: set[str] = set()
        names = set(archive.namelist())
        for source_id in ids:
            emotion = source_id.rsplit("_", 1)[0]
            matches = sorted(
                name
                for name in names
                if name.startswith(f"image/{emotion}/{source_id}.")
                and name.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
            )
            if len(matches) != 1:
                raise BridgeReleaseError(f"cannot resolve reproduced Eval80 image: {source_id}")
            reproduced_hashes.add(sha256_bytes(archive.read(matches[0])))
    if reproduced_hashes != hashes:
        raise BridgeReleaseError("reproduced Eval80 identities differ from the frozen manifest")
    return ids, set(hashes)


def _jsonl_text(rows: Iterable[Mapping[str, Any]]) -> str:
    return "".join(canonical_json(row, pretty=False) + "\n" for row in rows)


def write_release(output_dir: Path, release: Mapping[str, Any], *, overwrite: bool) -> dict[str, Path]:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    names = {
        "train": "bridge-train-v1.train.json",
        "validation": "bridge-train-v1.validation.json",
        "manifest": "bridge-train-v1.manifest.jsonl",
        "rejections": "bridge-train-v1.rejections.jsonl",
        "summary": "bridge-train-v1.summary.json",
    }
    paths = {key: output_dir / name for key, name in names.items()}
    if not overwrite:
        existing = [str(path) for path in paths.values() if path.exists()]
        if existing:
            raise FileExistsError(f"refusing to overwrite release files: {', '.join(existing)}")

    payloads = {
        "train": canonical_json(release["train"], pretty=True),
        "validation": canonical_json(release["validation"], pretty=True),
        "manifest": _jsonl_text(release["manifest"]),
        "rejections": _jsonl_text(release["rejections"]),
    }
    summary = dict(release["summary"])
    summary["artifact_sha256"] = {
        names[key]: sha256_text(payload) for key, payload in sorted(payloads.items())
    }
    payloads["summary"] = canonical_json(summary, pretty=True)

    temporary_paths: dict[str, Path] = {}
    try:
        for key, payload in payloads.items():
            file_descriptor, temporary_name = tempfile.mkstemp(
                dir=output_dir, prefix=f".{names[key]}.", suffix=".tmp"
            )
            temporary_path = Path(temporary_name)
            temporary_paths[key] = temporary_path
            with os.fdopen(file_descriptor, "w", encoding="utf-8", newline="\n") as output:
                output.write(payload)
                output.flush()
                os.fsync(output.fileno())
        for key in ("train", "validation", "manifest", "rejections", "summary"):
            os.replace(temporary_paths[key], paths[key])
    finally:
        for temporary_path in temporary_paths.values():
            if temporary_path.exists():
                temporary_path.unlink()
    return paths


def extract_release_images(
    *, archive_path: Path, manifest: Sequence[Mapping[str, Any]], destination_root: Path
) -> int:
    """Extract each unique release image while preserving its archive-relative path."""

    entries = sorted({entry["image_relpath"] for entry in manifest})
    destination_root = destination_root.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for entry in entries:
            _validate_archive_member(entry, field="manifest image_relpath")
            destination = (destination_root / Path(*PurePosixPath(entry).parts)).resolve()
            try:
                destination.relative_to(destination_root)
            except ValueError as error:
                raise BridgeReleaseError(f"image extraction escaped its destination: {entry}") from error
            destination.parent.mkdir(parents=True, exist_ok=True)
            image_bytes = archive.read(entry)
            expected_hashes = {
                item["image_sha256"] for item in manifest if item["image_relpath"] == entry
            }
            if expected_hashes != {sha256_bytes(image_bytes)}:
                raise BridgeReleaseError(f"manifest hash mismatch while extracting {entry}")
            temporary = destination.with_suffix(destination.suffix + ".tmp")
            temporary.write_bytes(image_bytes)
            os.replace(temporary, destination)
    return len(entries)


RELEASE_FILENAMES = {
    "train": "bridge-train-v1.train.json",
    "validation": "bridge-train-v1.validation.json",
    "manifest": "bridge-train-v1.manifest.jsonl",
    "rejections": "bridge-train-v1.rejections.jsonl",
    "summary": "bridge-train-v1.summary.json",
}


def verify_release_dir(
    *,
    release_dir: Path,
    config: BridgeBuildConfig,
    archive_path: Path | None = None,
    extracted_image_root: Path | None = None,
    source_root: Path | None = None,
    eval20_ids: Iterable[str] = (),
    eval20_hashes: Iterable[str] = (),
    eval80_ids: Iterable[str] = (),
    eval80_hashes: Iterable[str] = (),
) -> dict[str, Any]:
    """Independently verify written loader files, sidecars, hashes, and images."""

    release_dir = release_dir.resolve()
    paths = {key: release_dir / name for key, name in RELEASE_FILENAMES.items()}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise BridgeReleaseError(f"release files are missing: {', '.join(missing)}")
    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    expected_hashes = summary.get("artifact_sha256")
    if not isinstance(expected_hashes, Mapping):
        raise BridgeReleaseError("release summary has no artifact_sha256 map")
    for key in ("train", "validation", "manifest", "rejections"):
        filename = RELEASE_FILENAMES[key]
        expected_hash = _validate_sha256(
            expected_hashes.get(filename), f"summary hash for {filename}"
        )
        if sha256_file(paths[key]) != expected_hash:
            raise BridgeReleaseError(f"release artifact hash mismatch: {filename}")

    train = json.loads(paths["train"].read_text(encoding="utf-8"))
    validation = json.loads(paths["validation"].read_text(encoding="utf-8"))
    if not isinstance(train, list) or not isinstance(validation, list):
        raise BridgeReleaseError("train and validation artifacts must be JSON arrays")
    manifest = read_jsonl(paths["manifest"])
    rejections = read_jsonl(paths["rejections"])
    _validate_built_release(
        samples_by_split={"train": train, "validation": validation},
        manifest=manifest,
        config=config,
    )

    unique_by_split = {
        split: len(
            {
                row["source_image_id"]
                for row in manifest
                if row.get("split") == split
            }
        )
        for split in SPLITS
    }
    conversation_by_split = {
        "train": len(train),
        "validation": len(validation),
    }
    if summary.get("unique_image_counts") != unique_by_split:
        raise BridgeReleaseError("summary unique-image counts differ from the manifest")
    if summary.get("conversation_counts") != conversation_by_split:
        raise BridgeReleaseError("summary conversation counts differ from loader files")
    if summary.get("source_image_count") != sum(unique_by_split.values()):
        raise BridgeReleaseError("summary source-image total is inconsistent")
    if summary.get("conversation_count") != len(manifest):
        raise BridgeReleaseError("summary conversation total is inconsistent")
    if summary.get("guidance_ineligible_record_count") != len(rejections):
        raise BridgeReleaseError("summary rejection count differs from the rejection audit")
    if summary.get("dataset_id") != config.dataset_id:
        raise BridgeReleaseError("release dataset ID differs from the config lock")
    if summary.get("protocol_version") != config.protocol_version:
        raise BridgeReleaseError("release protocol version differs from the config lock")
    if summary.get("machine_validation_passed") is not True:
        raise BridgeReleaseError("release summary does not record passed machine validation")
    if summary.get("full_dataset_ready") is not config.post_correction_human_signoff:
        raise BridgeReleaseError("release readiness differs from the approval lock")
    if summary.get("human_review_signoff_pending") is not (
        not config.post_correction_human_signoff
    ):
        raise BridgeReleaseError("release human-review status is inconsistent")
    if summary.get("pending_human_review_ids") != list(config.pending_human_review_ids):
        raise BridgeReleaseError("release pending-review IDs differ from the approval lock")
    if summary.get("pending_human_review_record_count") != len(
        config.pending_human_review_ids
    ):
        raise BridgeReleaseError("release pending-review count differs from the approval lock")
    if summary.get("formal_training_authorized") is not False:
        raise BridgeReleaseError("Member 2 release must not authorize formal training")
    if summary.get("guidance_image_count") != sum(
        int(config.guidance_per_class_by_split[split]) * len(config.emotions)
        for split in SPLITS
    ):
        raise BridgeReleaseError("summary guidance-image total is inconsistent")
    expected_prompt_hashes = {
        TASK_CLASSIFICATION: sha256_text(config.classification_prompt),
        TASK_GUIDANCE: sha256_text(config.joint_prompt),
    }
    if summary.get("prompt_sha256") != expected_prompt_hashes:
        raise BridgeReleaseError("release prompt hashes differ from the config lock")

    unique_images: dict[str, Mapping[str, Any]] = {}
    for row in manifest:
        unique_images.setdefault(row["source_image_id"], row)
    excluded_ids = set(eval20_ids) | set(eval80_ids)
    excluded_hashes = set(eval20_hashes) | set(eval80_hashes)
    for value in excluded_hashes:
        _validate_sha256(value, "evaluation image hash")
    leaked_ids = sorted(set(unique_images) & excluded_ids)
    leaked_hashes = sorted(
        {row["image_sha256"] for row in unique_images.values()} & excluded_hashes
    )
    if leaked_ids:
        raise BridgeReleaseError(
            "evaluation IDs leaked into the written release: " + ", ".join(leaked_ids)
        )
    if leaked_hashes:
        raise BridgeReleaseError("evaluation image hashes leaked into the written release")

    if archive_path is not None:
        archive_path = archive_path.resolve()
        with zipfile.ZipFile(archive_path) as archive:
            names = set(archive.namelist())
            for source_id, row in sorted(unique_images.items()):
                image_relpath = _validate_archive_member(
                    row.get("image_relpath"), field=f"{source_id} image_relpath"
                )
                annotation_relpath = _validate_archive_member(
                    row.get("annotation_relpath"), field=f"{source_id} annotation_relpath"
                )
                if image_relpath not in names or annotation_relpath not in names:
                    raise BridgeReleaseError(f"archive member is missing for {source_id}")
                if sha256_bytes(archive.read(image_relpath)) != row["image_sha256"]:
                    raise BridgeReleaseError(f"archive image hash mismatch for {source_id}")
                if sha256_bytes(archive.read(annotation_relpath)) != row["annotation_sha256"]:
                    raise BridgeReleaseError(f"archive annotation hash mismatch for {source_id}")
                try:
                    annotation = json.loads(archive.read(annotation_relpath))
                except json.JSONDecodeError as error:
                    raise BridgeReleaseError(
                        f"archive annotation is invalid JSON for {source_id}"
                    ) from error
                if annotation.get("image_id") != source_id:
                    raise BridgeReleaseError(
                        f"archive annotation image ID mismatch for {source_id}"
                    )
                if annotation.get("emotion") != row["emotion"]:
                    raise BridgeReleaseError(
                        f"archive annotation emotion mismatch for {source_id}"
                    )

    if source_root is not None:
        source_root = source_root.resolve()
        source_files: dict[Path, list[dict[str, Any]]] = {}
        samples_by_id = {
            sample["id"]: sample for sample in [*train, *validation]
        }
        for row in manifest:
            source_id = row["source_image_id"]
            emotion = row["emotion"]
            split = row["split"]
            source_path = _source_path(source_root, emotion, split).resolve()
            try:
                expected_reference = source_path.relative_to(project_root()).as_posix()
            except ValueError:
                expected_reference = f"{emotion}/{source_path.name}"
            if row.get("source_record_path") != expected_reference:
                raise BridgeReleaseError(
                    f"source-record path mismatch for {row['sample_id']}"
                )
            if source_path not in source_files:
                if not source_path.is_file():
                    raise BridgeReleaseError(f"source JSONL is missing: {source_path}")
                source_files[source_path] = read_jsonl(source_path)
            line_number = row.get("source_record_line")
            if not isinstance(line_number, int) or not (
                1 <= line_number <= len(source_files[source_path])
            ):
                raise BridgeReleaseError(f"source-record line is invalid for {source_id}")
            source_record = source_files[source_path][line_number - 1]
            if _record_sha256(source_record) != row.get("source_record_sha256"):
                raise BridgeReleaseError(f"source-record hash mismatch for {source_id}")
            if (
                source_record.get("source_image_id") != source_id
                or source_record.get("emotion") != emotion
                or source_record.get("split") != split
                or source_record.get("image_relpath") != row.get("image_relpath")
            ):
                raise BridgeReleaseError(f"source-record identity mismatch for {source_id}")
            provenance = source_record.get("provenance")
            if not isinstance(provenance, Mapping):
                raise BridgeReleaseError(f"source provenance is invalid for {source_id}")
            if (
                provenance.get("image_sha256") != row.get("image_sha256")
                or provenance.get("annotation_sha256") != row.get("annotation_sha256")
                or provenance.get("review_status") != row.get("source_review_status")
            ):
                raise BridgeReleaseError(f"source provenance mismatch for {source_id}")
            if row["task_type"] == TASK_GUIDANCE:
                if provenance.get("review_status") != row.get("review_status"):
                    raise BridgeReleaseError(f"guidance review status mismatch for {source_id}")
                sample = samples_by_id[row["sample_id"]]
                if sample["conversations"][1]["value"] != source_record.get(
                    "target_response"
                ):
                    raise BridgeReleaseError(f"guidance target mismatch for {source_id}")

    if extracted_image_root is not None:
        root = extracted_image_root.resolve()
        for source_id, row in sorted(unique_images.items()):
            image_relpath = _validate_archive_member(
                row.get("image_relpath"), field=f"{source_id} image_relpath"
            )
            image_path = (root / Path(*PurePosixPath(image_relpath).parts)).resolve()
            try:
                image_path.relative_to(root)
            except ValueError as error:
                raise BridgeReleaseError(
                    f"extracted image path escaped its root: {source_id}"
                ) from error
            if not image_path.is_file():
                raise BridgeReleaseError(f"extracted image is missing: {source_id}")
            if sha256_file(image_path) != row["image_sha256"]:
                raise BridgeReleaseError(f"extracted image hash mismatch: {source_id}")

    return {
        "valid": True,
        "source_image_count": len(unique_images),
        "conversation_count": len(manifest),
        "archive_verified": archive_path is not None,
        "extracted_images_verified": extracted_image_root is not None,
        "source_records_verified": source_root is not None,
        "evaluation_exclusions_verified": bool(excluded_ids or excluded_hashes),
        "artifact_sha256_verified": True,
        "full_dataset_ready": summary["full_dataset_ready"],
        "human_review_signoff_pending": summary["human_review_signoff_pending"],
        "pending_human_review_record_count": summary[
            "pending_human_review_record_count"
        ],
        "formal_training_authorized": summary["formal_training_authorized"],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=root / "phase3/configs/bridge-train-v1.json")
    parser.add_argument("--source-root", type=Path, default=root / "data/datasets")
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument(
        "--eval20-artifact",
        type=Path,
        default=root / "results/phase1/emoset_eval20/stage1/B0_direct_emotion.json",
    )
    parser.add_argument(
        "--eval80-manifest",
        type=Path,
        default=root / "results/phase2/eval80/frozen_inputs/inference_manifest.json",
    )
    parser.add_argument(
        "--eval80-freeze-record",
        type=Path,
        default=root / "Docs/evaluation/EVAL80_FREEZE_RECORD.json",
    )
    parser.add_argument(
        "--image-reference-root",
        default="/workspace/phase3/data/bridge-train-v1",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--extract-images", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    eval20_ids, eval20_hashes = load_eval20_exclusions(
        args.eval20_artifact, args.archive
    )
    eval80_ids, eval80_hashes = load_eval80_exclusions(
        args.eval80_manifest,
        args.eval80_freeze_record,
        args.archive,
        config.emotions,
    )
    release = build_bridge_release(
        source_root=args.source_root,
        archive_path=args.archive,
        eval20_ids=eval20_ids,
        eval80_ids=eval80_ids,
        eval20_hashes=eval20_hashes,
        eval80_hashes=eval80_hashes,
        image_reference_root=args.image_reference_root,
        config=config,
    )
    paths = write_release(args.output_dir, release, overwrite=args.overwrite)
    extracted = None
    if args.extract_images is not None:
        extracted = extract_release_images(
            archive_path=args.archive,
            manifest=release["manifest"],
            destination_root=args.extract_images,
        )
    print(
        canonical_json(
            {
                "valid": True,
                "source_image_count": release["summary"]["source_image_count"],
                "conversation_count": release["summary"]["conversation_count"],
                "output_files": {key: str(path) for key, path in paths.items()},
                "extracted_image_count": extracted,
            },
            pretty=True,
        ),
        end="",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
