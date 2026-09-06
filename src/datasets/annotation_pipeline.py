"""Select EmoSet records and draft emotion-aware JSONL with the OpenAI API.

The final records retain the seven top-level fields used by the existing team
JSONL files. GPT output is constrained with a JSON schema, then rendered into
the shared Emotion / Visual evidence / Aesthetic relationship / Guidance /
Final emotion target format.
"""

from __future__ import annotations

import argparse
import base64
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import mimetypes
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import zipfile


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

EVAL20_IMAGE_IDS = frozenset(
    {
        "amusement_07879",
        "amusement_13035",
        "amusement_17441",
        "anger_01898",
        "anger_07362",
        "awe_03021",
        "awe_09647",
        "awe_10103",
        "contentment_09637",
        "contentment_13308",
        "disgust_03892",
        "disgust_06215",
        "disgust_09052",
        "excitement_03750",
        "excitement_17292",
        "fear_01314",
        "fear_05947",
        "fear_14745",
        "sadness_03767",
        "sadness_04511",
    }
)

TARGET_PATTERN = re.compile(
    r"\AEmotion: (?P<emotion>[^\n]+)\n\n"
    r"Visual evidence:\n"
    r"- (?P<evidence_1>[^\n]+)\n"
    r"- (?P<evidence_2>[^\n]+)\n\n"
    r"Aesthetic relationship:\n"
    r"(?P<aesthetic>[^\n]+)\n\n"
    r"Guidance:\n"
    r"- (?P<guidance_1>[^\n]+)\n"
    r"- (?P<guidance_2>[^\n]+)\n\n"
    r"Final emotion: (?P<final_emotion>[^\n]+)\Z"
)


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "protocol_version",
        "model",
        "endpoint",
        "api_key_env",
        "prompt_path",
        "prompt_version",
        "image_detail",
        "reasoning_effort",
        "temperature",
        "max_output_tokens",
        "store",
        "timeout_seconds",
        "max_retries",
        "selection",
    }
    missing = sorted(required - config.keys())
    if missing:
        raise ValueError(f"Configuration is missing fields: {', '.join(missing)}")
    return config


def resolve_project_path(value: str) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else project_root() / candidate


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON on {path}:{line_number}: {error}") from error
            if not isinstance(record, dict):
                raise ValueError(f"Expected an object on {path}:{line_number}")
            records.append(record)
    return records


def write_jsonl(path: Path, records: Iterable[dict[str, Any]], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")


def image_number(image_id: str, emotion: str) -> int:
    match = re.fullmatch(rf"{re.escape(emotion)}_(\d+)", image_id)
    if not match:
        raise ValueError(f"Unexpected image ID for {emotion}: {image_id}")
    return int(match.group(1))


def list_image_entries(archive: zipfile.ZipFile, emotion: str) -> list[str]:
    prefix = f"image/{emotion}/{emotion}_"
    candidates = [
        name
        for name in archive.namelist()
        if name.startswith(prefix) and name.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
    ]
    return sorted(
        candidates,
        key=lambda name: (image_number(Path(name).stem, emotion), name.lower()),
    )


def eval80_ids(archive: zipfile.ZipFile, emotions: Iterable[str]) -> set[str]:
    selected: set[str] = set()
    for emotion in emotions:
        candidates = list_image_entries(archive, emotion)
        if len(candidates) < 10:
            raise ValueError(f"Cannot reproduce Eval80: {emotion} has fewer than ten images")
        selected.update(Path(name).stem for name in candidates[:10])
    return selected


def deterministic_split_key(seed: int, emotion: str, image_id: str) -> str:
    return hashlib.sha256(f"{seed}|{emotion}|{image_id}".encode("utf-8")).hexdigest()


def select_records(
    archive_path: Path,
    emotions: Iterable[str],
    selection: dict[str, Any],
) -> list[dict[str, Any]]:
    chosen_emotions = tuple(emotions)
    unknown = sorted(set(chosen_emotions) - set(EMOTION_CLASSES))
    if unknown:
        raise ValueError(f"Unknown emotions: {', '.join(unknown)}")
    if len(set(chosen_emotions)) != len(chosen_emotions):
        raise ValueError("Emotion list contains duplicates")

    start_number = int(selection["start_image_number"])
    total = int(selection["images_per_class"])
    training_count = int(selection["training_per_class"])
    validation_count = int(selection["validation_per_class"])
    split_seed = int(selection["split_seed"])
    if training_count + validation_count != total:
        raise ValueError("Training and validation counts must equal images_per_class")

    records: list[dict[str, Any]] = []
    with zipfile.ZipFile(archive_path) as archive:
        archive_names = set(archive.namelist())
        exclusions = EVAL20_IMAGE_IDS | eval80_ids(archive, chosen_emotions)
        for emotion in chosen_emotions:
            eligible: list[str] = []
            for image_entry in list_image_entries(archive, emotion):
                image_id = Path(image_entry).stem
                if image_number(image_id, emotion) < start_number or image_id in exclusions:
                    continue
                annotation_entry = f"annotation/{emotion}/{image_id}.json"
                if annotation_entry not in archive_names:
                    continue
                eligible.append(image_entry)

            if len(eligible) < total:
                raise ValueError(
                    f"{emotion} has only {len(eligible)} eligible images at or after "
                    f"{start_number:05d}; {total} are required"
                )

            selected = eligible[:total]
            split_order = sorted(
                selected,
                key=lambda entry: deterministic_split_key(
                    split_seed, emotion, Path(entry).stem
                ),
            )
            split_by_id = {
                Path(entry).stem: ("train" if index < training_count else "validation")
                for index, entry in enumerate(split_order)
            }

            for selection_rank, image_entry in enumerate(selected, start=1):
                image_id = Path(image_entry).stem
                annotation_entry = f"annotation/{emotion}/{image_id}.json"
                image_bytes = archive.read(image_entry)
                annotation_bytes = archive.read(annotation_entry)
                annotation = json.loads(annotation_bytes)
                if annotation.get("emotion") != emotion:
                    raise ValueError(
                        f"Source annotation label mismatch for {image_id}: "
                        f"{annotation.get('emotion')!r}"
                    )
                records.append(
                    {
                        "source_image_id": image_id,
                        "emotion": emotion,
                        "image_relpath": image_entry,
                        "annotation_relpath": annotation_entry,
                        "split": split_by_id[image_id],
                        "selection_rank": selection_rank,
                        "selection_seed": split_seed,
                        "image_sha256": sha256_bytes(image_bytes),
                        "annotation_sha256": sha256_bytes(annotation_bytes),
                    }
                )

    return sorted(records, key=lambda item: (item["emotion"], item["split"], item["source_image_id"]))


def select_replacement_records(
    archive_path: Path,
    manifest_path: Path,
    output_root: Path,
    *,
    start_image_number: int,
) -> list[dict[str, Any]]:
    """Choose the next eligible same-emotion candidates for rejected records."""
    manifest_records = read_jsonl(manifest_path)
    validate_manifest_records(manifest_records)
    manifest_by_id = {record["source_image_id"]: record for record in manifest_records}

    rejected: list[dict[str, Any]] = []
    for path in sorted(output_root.rglob("*.jsonl")):
        for record in read_jsonl(path):
            if record.get("provenance", {}).get("review_status") == "rejected":
                rejected.append(record)
    rejected.sort(key=lambda record: (record["emotion"], record["split"], record["source_image_id"]))
    if not rejected:
        return []

    used_ids = set(manifest_by_id)
    replacement_records: list[dict[str, Any]] = []
    with zipfile.ZipFile(archive_path) as archive:
        archive_names = set(archive.namelist())
        emotions = sorted({record["emotion"] for record in rejected})
        exclusions = EVAL20_IMAGE_IDS | eval80_ids(archive, emotions) | used_ids
        candidates_by_emotion: dict[str, list[str]] = {}
        next_rank_by_emotion: dict[str, int] = {}

        for emotion in emotions:
            emotion_manifest = [r for r in manifest_records if r["emotion"] == emotion]
            minimum_number = max(
                [start_image_number - 1]
                + [image_number(r["source_image_id"], emotion) for r in emotion_manifest]
            ) + 1
            next_rank_by_emotion[emotion] = max(
                [0] + [int(r.get("selection_rank", 0)) for r in emotion_manifest]
            )
            eligible: list[str] = []
            for image_entry in list_image_entries(archive, emotion):
                image_id = Path(image_entry).stem
                annotation_entry = f"annotation/{emotion}/{image_id}.json"
                if (
                    image_number(image_id, emotion) < minimum_number
                    or image_id in exclusions
                    or annotation_entry not in archive_names
                ):
                    continue
                eligible.append(image_entry)
            candidates_by_emotion[emotion] = eligible

        offsets: Counter[str] = Counter()
        for rejected_record in rejected:
            rejected_id = rejected_record["source_image_id"]
            if rejected_id not in manifest_by_id:
                raise ValueError(f"Rejected record is not in the frozen manifest: {rejected_id}")
            original = manifest_by_id[rejected_id]
            if (
                original["emotion"] != rejected_record["emotion"]
                or original["split"] != rejected_record["split"]
            ):
                raise ValueError(f"Rejected record differs from the frozen manifest: {rejected_id}")

            emotion = original["emotion"]
            offset = offsets[emotion]
            candidates = candidates_by_emotion[emotion]
            if offset >= len(candidates):
                raise ValueError(f"No eligible replacement remains for {rejected_id}")
            image_entry = candidates[offset]
            offsets[emotion] += 1
            image_id = Path(image_entry).stem
            annotation_entry = f"annotation/{emotion}/{image_id}.json"
            image_bytes = archive.read(image_entry)
            annotation_bytes = archive.read(annotation_entry)
            annotation = json.loads(annotation_bytes)
            if annotation.get("emotion") != emotion:
                raise ValueError(
                    f"Source annotation label mismatch for {image_id}: {annotation.get('emotion')!r}"
                )
            next_rank_by_emotion[emotion] += 1
            replacement_records.append(
                {
                    "source_image_id": image_id,
                    "emotion": emotion,
                    "image_relpath": image_entry,
                    "annotation_relpath": annotation_entry,
                    "split": original["split"],
                    "selection_rank": next_rank_by_emotion[emotion],
                    "selection_seed": original.get("selection_seed"),
                    "image_sha256": sha256_bytes(image_bytes),
                    "annotation_sha256": sha256_bytes(annotation_bytes),
                    "replaces_source_image_id": rejected_id,
                }
            )

    return replacement_records


def normalize_text(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    normalized = " ".join(value.split())
    if not normalized:
        raise ValueError(f"{field} must not be empty")
    return normalized


def validate_annotation(annotation: dict[str, Any], assigned_emotion: str) -> dict[str, Any]:
    required = {
        "emotion",
        "visual_evidence",
        "aesthetic_relationship",
        "guidance",
        "final_emotion",
        "label_supported",
        "support_notes",
    }
    missing = sorted(required - annotation.keys())
    if missing:
        raise ValueError(f"Model output is missing fields: {', '.join(missing)}")
    if annotation["emotion"] != assigned_emotion or annotation["final_emotion"] != assigned_emotion:
        raise ValueError("Model output attempted to replace the assigned EmoSet label")

    evidence = annotation["visual_evidence"]
    guidance = annotation["guidance"]
    if not isinstance(evidence, list) or len(evidence) != 2:
        raise ValueError("visual_evidence must contain exactly two items")
    if not isinstance(guidance, list) or len(guidance) != 2:
        raise ValueError("guidance must contain exactly two items")
    if not isinstance(annotation["label_supported"], bool):
        raise ValueError("label_supported must be a boolean")

    return {
        "emotion": assigned_emotion,
        "visual_evidence": [
            normalize_text(evidence[0], "visual_evidence[0]"),
            normalize_text(evidence[1], "visual_evidence[1]"),
        ],
        "aesthetic_relationship": normalize_text(
            annotation["aesthetic_relationship"], "aesthetic_relationship"
        ),
        "guidance": [
            normalize_text(guidance[0], "guidance[0]"),
            normalize_text(guidance[1], "guidance[1]"),
        ],
        "final_emotion": assigned_emotion,
        "label_supported": annotation["label_supported"],
        "support_notes": normalize_text(annotation["support_notes"], "support_notes"),
    }


def render_target_response(annotation: dict[str, Any]) -> str:
    return (
        f"Emotion: {annotation['emotion']}\n\n"
        "Visual evidence:\n"
        f"- {annotation['visual_evidence'][0]}\n"
        f"- {annotation['visual_evidence'][1]}\n\n"
        "Aesthetic relationship:\n"
        f"{annotation['aesthetic_relationship']}\n\n"
        "Guidance:\n"
        f"- {annotation['guidance'][0]}\n"
        f"- {annotation['guidance'][1]}\n\n"
        f"Final emotion: {annotation['final_emotion']}"
    )


def annotation_schema(assigned_emotion: str) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "emotion": {"type": "string", "enum": [assigned_emotion]},
            "visual_evidence": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 2,
                "maxItems": 2,
            },
            "aesthetic_relationship": {"type": "string"},
            "guidance": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 2,
                "maxItems": 2,
            },
            "final_emotion": {"type": "string", "enum": [assigned_emotion]},
            "label_supported": {"type": "boolean"},
            "support_notes": {"type": "string"},
        },
        "required": [
            "emotion",
            "visual_evidence",
            "aesthetic_relationship",
            "guidance",
            "final_emotion",
            "label_supported",
            "support_notes",
        ],
    }


def build_request_payload(
    config: dict[str, Any], instruction: str, emotion: str, data_url: str, source_id: str
) -> dict[str, Any]:
    return {
        "model": config["model"],
        "store": bool(config["store"]),
        "reasoning": {"effort": config["reasoning_effort"]},
        "temperature": float(config["temperature"]),
        "max_output_tokens": int(config["max_output_tokens"]),
        "metadata": {
            "protocol_version": str(config["protocol_version"]),
            "source_image_id": source_id,
        },
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": instruction},
                    {
                        "type": "input_image",
                        "image_url": data_url,
                        "detail": config["image_detail"],
                    },
                ],
            }
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "emoset_joint_annotation",
                "strict": True,
                "schema": annotation_schema(emotion),
            }
        },
    }


def extract_output_text(response: dict[str, Any]) -> str:
    for item in response.get("output", []):
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if content.get("type") == "output_text" and isinstance(content.get("text"), str):
                return content["text"]
            if content.get("type") == "refusal":
                raise ValueError(f"API refused the request: {content.get('refusal', '')}")
    raise ValueError("API response did not contain output_text")


def post_response(config: dict[str, Any], payload: dict[str, Any], api_key: str) -> dict[str, Any]:
    request = Request(
        config["endpoint"],
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=int(config["timeout_seconds"])) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        details = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI API returned HTTP {error.code}: {details}") from error
    except URLError as error:
        raise RuntimeError(f"OpenAI API request failed: {error.reason}") from error


def image_data_url(image_bytes: bytes, image_entry: str) -> str:
    mime_type = mimetypes.guess_type(image_entry)[0] or "image/jpeg"
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as output:
        output.write(json.dumps(record, ensure_ascii=False) + "\n")
        output.flush()


def completed_ids(output_root: Path) -> set[str]:
    completed: set[str] = set()
    for path in output_root.rglob("*.jsonl"):
        for record in read_jsonl(path):
            source_id = record.get("source_image_id")
            if isinstance(source_id, str):
                completed.add(source_id)
    return completed


def choose_smoke_records(records: list[dict[str, Any]], per_emotion: int) -> list[dict[str, Any]]:
    if per_emotion <= 0:
        return records
    chosen: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for record in records:
        emotion = record["emotion"]
        if record["split"] == "train" and counts[emotion] < per_emotion:
            chosen.append(record)
            counts[emotion] += 1
    expected = set(record["emotion"] for record in records)
    missing = sorted(emotion for emotion in expected if counts[emotion] < per_emotion)
    if missing:
        raise ValueError(f"Not enough training records for smoke run: {', '.join(missing)}")
    return chosen


def validate_manifest_records(records: list[dict[str, Any]]) -> None:
    required = {
        "source_image_id",
        "emotion",
        "image_relpath",
        "annotation_relpath",
        "split",
        "image_sha256",
        "annotation_sha256",
    }
    seen: set[str] = set()
    for index, record in enumerate(records, start=1):
        missing = sorted(required - record.keys())
        if missing:
            raise ValueError(f"Manifest record {index} is missing: {', '.join(missing)}")
        source_id = record["source_image_id"]
        if source_id in seen:
            raise ValueError(f"Duplicate source_image_id in manifest: {source_id}")
        seen.add(source_id)
        if record["emotion"] not in EMOTION_CLASSES:
            raise ValueError(f"Manifest record {source_id} has an unknown emotion")
        if record["split"] not in {"train", "validation"}:
            raise ValueError(f"Manifest record {source_id} has an invalid split")
        if source_id in EVAL20_IMAGE_IDS:
            raise ValueError(f"Manifest includes an Eval20 image: {source_id}")
        for field in ("image_sha256", "annotation_sha256"):
            if not re.fullmatch(r"[0-9a-f]{64}", str(record[field])):
                raise ValueError(f"Manifest record {source_id} has an invalid {field}")


def generate_records(
    archive_path: Path,
    manifest_path: Path,
    output_root: Path,
    config: dict[str, Any],
    *,
    smoke_per_emotion: int,
) -> dict[str, int]:
    api_key = os.environ.get(config["api_key_env"], "").strip()
    if not api_key:
        raise ValueError(
            f"Set {config['api_key_env']} in the environment before calling the API"
        )

    prompt_template = resolve_project_path(config["prompt_path"]).read_text(encoding="utf-8")
    manifest_records = read_jsonl(manifest_path)
    validate_manifest_records(manifest_records)
    records = choose_smoke_records(manifest_records, smoke_per_emotion)
    already_completed = completed_ids(output_root)
    counts: Counter[str] = Counter()

    with zipfile.ZipFile(archive_path) as archive:
        for record in records:
            source_id = record["source_image_id"]
            if source_id in already_completed:
                counts["skipped"] += 1
                continue

            emotion = record["emotion"]
            instruction = prompt_template.format(emotion=emotion).strip()
            image_bytes = archive.read(record["image_relpath"])
            annotation_bytes = archive.read(record["annotation_relpath"])
            if sha256_bytes(image_bytes) != record["image_sha256"]:
                raise ValueError(f"Image hash differs from the frozen manifest: {source_id}")
            if sha256_bytes(annotation_bytes) != record["annotation_sha256"]:
                raise ValueError(f"Annotation hash differs from the frozen manifest: {source_id}")
            payload = build_request_payload(
                config,
                instruction,
                emotion,
                image_data_url(image_bytes, record["image_relpath"]),
                source_id,
            )

            response: dict[str, Any] | None = None
            error: Exception | None = None
            for attempt in range(1, int(config["max_retries"]) + 1):
                try:
                    response = post_response(config, payload, api_key)
                    break
                except RuntimeError as current_error:
                    error = current_error
                    if attempt < int(config["max_retries"]):
                        time.sleep(min(2 ** (attempt - 1), 8))
            if response is None:
                raise RuntimeError(f"Failed to annotate {source_id}: {error}") from error

            annotation = validate_annotation(json.loads(extract_output_text(response)), emotion)
            generated_at = datetime.now(timezone.utc).isoformat()
            output_record = {
                "source_image_id": source_id,
                "emotion": emotion,
                "image_relpath": record["image_relpath"],
                "instruction": instruction,
                "target_response": render_target_response(annotation),
                "split": record["split"],
                "provenance": {
                    "source": "EmoSet-118K",
                    "image_sha256": record["image_sha256"],
                    "annotation_sha256": record["annotation_sha256"],
                    "generator_provider": "openai",
                    "generator_model": response.get("model", config["model"]),
                    "generator_response_id": response.get("id"),
                    "generated_at_utc": generated_at,
                    "protocol_version": config["protocol_version"],
                    "prompt_version": config["prompt_version"],
                    "reasoning_effort": config["reasoning_effort"],
                    "temperature": config["temperature"],
                    "seed": "unsupported",
                    "image_detail": config["image_detail"],
                    "label_supported": annotation["label_supported"],
                    "label_support_notes": annotation["support_notes"],
                    "review_status": (
                        "draft" if annotation["label_supported"] else "needs_review"
                    ),
                },
            }
            if record.get("replaces_source_image_id"):
                output_record["provenance"]["replaces_source_image_id"] = record[
                    "replaces_source_image_id"
                ]

            raw_path = output_root / "raw" / f"{source_id}.json"
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            raw_path.write_text(
                json.dumps(
                    {
                        "source_image_id": source_id,
                        "parsed_annotation": annotation,
                        "api_response": response,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            destination = (
                output_root
                / emotion
                / (f"train_{emotion}.jsonl" if record["split"] == "train" else f"validation_{emotion}.jsonl")
            )
            append_jsonl(destination, output_record)
            already_completed.add(source_id)
            counts["generated"] += 1

    return dict(counts)


def validate_output_record(record: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required = {
        "source_image_id",
        "emotion",
        "image_relpath",
        "instruction",
        "target_response",
        "split",
        "provenance",
    }
    missing = sorted(required - record.keys())
    if missing:
        return [f"missing fields: {', '.join(missing)}"]
    if record["emotion"] not in EMOTION_CLASSES:
        errors.append(f"invalid emotion: {record['emotion']!r}")
    if record["split"] not in {"train", "validation"}:
        errors.append(f"invalid split: {record['split']!r}")
    match = TARGET_PATTERN.fullmatch(record["target_response"])
    if not match:
        errors.append("target_response does not match the required Sam-based structure")
    elif match.group("emotion") != record["emotion"] or match.group("final_emotion") != record["emotion"]:
        errors.append("target_response emotion differs from the fixed source label")
    provenance = record["provenance"]
    if not isinstance(provenance, dict):
        errors.append("provenance must be an object")
    else:
        for field in (
            "generator_provider",
            "generator_model",
            "prompt_version",
            "temperature",
            "seed",
            "review_status",
        ):
            if field not in provenance:
                errors.append(f"provenance is missing {field}")
    return errors


def validate_output_root(
    output_root: Path,
    *,
    expected_counts: dict[str, int] | None = None,
    require_accepted: bool = False,
) -> dict[str, Any]:
    errors: list[str] = []
    counts: Counter[str] = Counter()
    seen: set[str] = set()
    for path in sorted(output_root.rglob("*.jsonl")):
        for line_number, record in enumerate(read_jsonl(path), start=1):
            source_id = record.get("source_image_id", f"line-{line_number}")
            if source_id in seen:
                errors.append(f"duplicate source_image_id: {source_id}")
            seen.add(source_id)
            counts[f"{record.get('emotion')}:{record.get('split')}"] += 1
            for error in validate_output_record(record):
                errors.append(f"{path}:{line_number} ({source_id}): {error}")
            provenance = record.get("provenance", {})
            if require_accepted and provenance.get("review_status") not in {"accepted", "edited"}:
                errors.append(
                    f"{path}:{line_number} ({source_id}): review_status must be accepted or edited"
                )
    if expected_counts is not None:
        for key, expected in sorted(expected_counts.items()):
            actual = counts.get(key, 0)
            if actual != expected:
                errors.append(f"expected {expected} records for {key}, found {actual}")
    return {
        "valid": not errors,
        "record_count": len(seen),
        "counts": dict(sorted(counts.items())),
        "errors": errors,
    }


def finalize_reviewed_records(
    output_root: Path,
    original_manifest_path: Path,
    replacement_manifest_path: Path,
    final_manifest_path: Path,
    rejected_audit_path: Path,
    *,
    overwrite: bool,
) -> dict[str, Any]:
    """Replace rejected originals with accepted replacements after human review."""
    original_manifest = read_jsonl(original_manifest_path)
    replacement_manifest = read_jsonl(replacement_manifest_path)
    validate_manifest_records(original_manifest)
    validate_manifest_records(replacement_manifest)

    original_by_id = {record["source_image_id"]: record for record in original_manifest}
    replacement_by_original: dict[str, dict[str, Any]] = {}
    for record in replacement_manifest:
        replaced_id = record.get("replaces_source_image_id")
        if not isinstance(replaced_id, str) or replaced_id not in original_by_id:
            raise ValueError(
                f"Replacement {record['source_image_id']} has an invalid replaces_source_image_id"
            )
        if replaced_id in replacement_by_original:
            raise ValueError(f"Multiple replacements target {replaced_id}")
        original = original_by_id[replaced_id]
        if record["emotion"] != original["emotion"] or record["split"] != original["split"]:
            raise ValueError(f"Replacement differs in emotion or split: {record['source_image_id']}")
        replacement_by_original[replaced_id] = record

    records_by_id: dict[str, dict[str, Any]] = {}
    records_by_path: dict[Path, list[dict[str, Any]]] = {}
    for path in sorted(output_root.glob("*/*.jsonl")):
        records = read_jsonl(path)
        records_by_path[path] = records
        for record in records:
            source_id = record.get("source_image_id")
            if source_id in records_by_id:
                raise ValueError(f"Duplicate generated record: {source_id}")
            records_by_id[source_id] = record

    rejected_ids = {
        source_id
        for source_id, record in records_by_id.items()
        if record.get("provenance", {}).get("review_status") == "rejected"
    }
    if rejected_ids != set(replacement_by_original):
        missing = sorted(rejected_ids - set(replacement_by_original))
        extra = sorted(set(replacement_by_original) - rejected_ids)
        raise ValueError(
            f"Rejected/replacement mapping mismatch; missing={missing}, extra={extra}"
        )

    final_manifest = [
        record for record in original_manifest if record["source_image_id"] not in rejected_ids
    ] + list(replacement_manifest)
    final_ids = {record["source_image_id"] for record in final_manifest}
    if len(final_ids) != len(original_manifest):
        raise ValueError("Final manifest does not preserve the original record count")

    original_counts = Counter(
        (record["emotion"], record["split"]) for record in original_manifest
    )
    final_counts = Counter((record["emotion"], record["split"]) for record in final_manifest)
    if final_counts != original_counts:
        raise ValueError("Final replacements do not preserve emotion/split balance")

    for source_id in sorted(final_ids):
        record = records_by_id.get(source_id)
        if record is None:
            raise ValueError(f"Final generated record is missing: {source_id}")
        errors = validate_output_record(record)
        if errors:
            raise ValueError(f"Invalid final record {source_id}: {'; '.join(errors)}")
        status = record.get("provenance", {}).get("review_status")
        if status not in {"accepted", "edited"}:
            raise ValueError(f"Final record is not accepted or edited: {source_id}")

    audit_records = []
    for rejected_id in sorted(rejected_ids):
        audit_records.append(
            {
                "rejected_record": records_by_id[rejected_id],
                "replacement_source_image_id": replacement_by_original[rejected_id][
                    "source_image_id"
                ],
            }
        )
    write_jsonl(rejected_audit_path, audit_records, overwrite=overwrite)
    write_jsonl(
        final_manifest_path,
        sorted(
            final_manifest,
            key=lambda record: (
                record["emotion"],
                record["split"],
                record["source_image_id"],
            ),
        ),
        overwrite=overwrite,
    )

    for path, records in records_by_path.items():
        retained = [record for record in records if record["source_image_id"] in final_ids]
        write_jsonl(path, retained, overwrite=True)

    return {
        "record_count": len(final_ids),
        "rejected_archived": len(rejected_ids),
        "counts": {
            f"{emotion}:{split}": count
            for (emotion, split), count in sorted(final_counts.items())
        },
    }


def add_common_config_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        default=project_root() / "configs/datasets/emotion_annotation.json",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    select_parser = subparsers.add_parser("select", help="Create a fixed selection manifest")
    add_common_config_argument(select_parser)
    select_parser.add_argument("--archive", type=Path, required=True)
    select_parser.add_argument("--manifest", type=Path, required=True)
    select_parser.add_argument("--emotions", nargs="+", choices=EMOTION_CLASSES)
    select_parser.add_argument("--overwrite", action="store_true")

    generate_parser = subparsers.add_parser("generate", help="Generate draft JSONL records")
    add_common_config_argument(generate_parser)
    generate_parser.add_argument("--archive", type=Path, required=True)
    generate_parser.add_argument("--manifest", type=Path, required=True)
    generate_parser.add_argument("--output-root", type=Path, required=True)
    generate_parser.add_argument("--smoke-per-emotion", type=int, default=0)

    replacement_parser = subparsers.add_parser(
        "select-replacements", help="Create a manifest for rejected-record replacements"
    )
    add_common_config_argument(replacement_parser)
    replacement_parser.add_argument("--archive", type=Path, required=True)
    replacement_parser.add_argument("--manifest", type=Path, required=True)
    replacement_parser.add_argument("--output-root", type=Path, required=True)
    replacement_parser.add_argument("--replacement-manifest", type=Path, required=True)
    replacement_parser.add_argument("--overwrite", action="store_true")

    validate_parser = subparsers.add_parser("validate", help="Validate generated JSONL")
    add_common_config_argument(validate_parser)
    validate_parser.add_argument("--output-root", type=Path, required=True)
    validate_parser.add_argument("--require-complete", action="store_true")
    validate_parser.add_argument("--require-accepted", action="store_true")

    finalize_parser = subparsers.add_parser(
        "finalize", help="Archive rejected records and apply their reviewed replacements"
    )
    finalize_parser.add_argument("--output-root", type=Path, required=True)
    finalize_parser.add_argument("--manifest", type=Path, required=True)
    finalize_parser.add_argument("--replacement-manifest", type=Path, required=True)
    finalize_parser.add_argument("--final-manifest", type=Path, required=True)
    finalize_parser.add_argument("--rejected-audit", type=Path, required=True)
    finalize_parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.command == "select":
            config = load_config(args.config)
            emotions = args.emotions or config.get("initial_emotions", EMOTION_CLASSES)
            records = select_records(args.archive, emotions, config["selection"])
            write_jsonl(args.manifest, records, overwrite=args.overwrite)
            summary = {
                "manifest": str(args.manifest),
                "record_count": len(records),
                "counts": dict(
                    sorted(Counter(f"{r['emotion']}:{r['split']}" for r in records).items())
                ),
            }
            print(json.dumps(summary, indent=2))
            return 0
        if args.command == "generate":
            config = load_config(args.config)
            result = generate_records(
                args.archive,
                args.manifest,
                args.output_root,
                config,
                smoke_per_emotion=args.smoke_per_emotion,
            )
            print(json.dumps(result, indent=2))
            return 0
        if args.command == "select-replacements":
            config = load_config(args.config)
            records = select_replacement_records(
                args.archive,
                args.manifest,
                args.output_root,
                start_image_number=int(config["selection"]["start_image_number"]),
            )
            write_jsonl(args.replacement_manifest, records, overwrite=args.overwrite)
            print(
                json.dumps(
                    {
                        "replacement_manifest": str(args.replacement_manifest),
                        "record_count": len(records),
                        "replacements": {
                            record["replaces_source_image_id"]: record["source_image_id"]
                            for record in records
                        },
                    },
                    indent=2,
                )
            )
            return 0
        if args.command == "finalize":
            result = finalize_reviewed_records(
                args.output_root,
                args.manifest,
                args.replacement_manifest,
                args.final_manifest,
                args.rejected_audit,
                overwrite=args.overwrite,
            )
            print(json.dumps(result, indent=2))
            return 0
        config = load_config(args.config)
        expected_counts = None
        if args.require_complete:
            selection = config["selection"]
            expected_counts = {
                f"{emotion}:train": int(selection["training_per_class"])
                for emotion in config.get("initial_emotions", EMOTION_CLASSES)
            }
            expected_counts.update(
                {
                    f"{emotion}:validation": int(selection["validation_per_class"])
                    for emotion in config.get("initial_emotions", EMOTION_CLASSES)
                }
            )
        report = validate_output_root(
            args.output_root,
            expected_counts=expected_counts,
            require_accepted=args.require_accepted,
        )
        print(json.dumps(report, indent=2))
        return 0 if report["valid"] else 1
    except (OSError, ValueError, RuntimeError, zipfile.BadZipFile, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
