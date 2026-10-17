"""Deterministic recognition parsing and five-dimension Eval80 summaries."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import random
import re
from statistics import mean
from typing import Any

from .core import EMOTION_CLASSES, read_selection_manifest, sha256_file


GUIDANCE_COLUMNS = (
    "visual_grounding_0_2",
    "emotion_aesthetic_linkage_0_2",
    "aesthetic_validity_0_2",
    "actionability_0_2",
    "emotion_preservation_0_2",
)
REVIEWER_IDS = tuple(f"team-member-{index:02d}" for index in range(1, 7))
REVIEWER_IMAGE_COUNTS = {reviewer_id: 13 for reviewer_id in REVIEWER_IDS}
CONDITIONS = ("A", "B0", "B1", "C")
ALL_BLIND_IDS = frozenset(f"E{index:03d}" for index in range(1, 81))
FREEZE_RECORD_PATH = (
    Path(__file__).resolve().parents[2] / "Docs/evaluation/EVAL80_FREEZE_RECORD.json"
)
FAILURE_FLAGS = {
    "hallucination",
    "generic_evidence",
    "missing_emotion_aesthetic_link",
    "invalid_aesthetic_judgement",
    "generic_advice",
    "emotion_conflict",
    "empty_or_unusable",
    "other",
}
_LABELS_PATTERN = "|".join(re.escape(label) for label in EMOTION_CLASSES)
_STRUCTURED_EMOTION_RE = re.compile(
    rf"^Emotion\s*:\s*({_LABELS_PATTERN})\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_STRUCTURED_FINAL_RE = re.compile(
    rf"^Final emotion\s*:\s*({_LABELS_PATTERN})\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_EXPLICIT_RE = re.compile(
    rf"(?:primary\s+emotion|final\s+emotion|emotion)\s*(?:is|:|-)\s*({_LABELS_PATTERN})\b",
    re.IGNORECASE,
)
_ANY_LABEL_RE = re.compile(rf"\b({_LABELS_PATTERN})\b", re.IGNORECASE)


@dataclass(frozen=True)
class EmotionParse:
    status: str
    predicted_emotion: str | None
    candidates: tuple[str, ...]
    rule: str


@dataclass(frozen=True)
class GuidanceScope:
    decision_id: str
    excluded_ids: frozenset[str]
    source_sha256: str

    @property
    def included_ids(self) -> frozenset[str]:
        return ALL_BLIND_IDS - self.excluded_ids


def read_guidance_scope(path: Path) -> GuidanceScope:
    """Validate the approved post-freeze display exclusion as scoring input."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Display exclusion record must be a JSON object.")
    excluded = payload.get("displayed_blind_ids")
    counts = payload.get("counts")
    scope = payload.get("scope")
    if (
        payload.get("schema_version") != 1
        or payload.get("decision_id") != "eval80-v1-post-a-display-exclusion-2026-09-08"
        or payload.get("approval_status") != "unanimously_confirmed_by_six_member_team"
        or excluded != ["E001", "E080"]
        or payload.get("conditions") != list(CONDITIONS)
        or not isinstance(scope, dict)
        or scope.get("inference_excluded") is not False
        or scope.get("emotion_recognition_excluded") is not False
        or scope.get("human_guidance_scoring_excluded") is not True
        or not isinstance(counts, dict)
        or counts.get("frozen_inference_images_per_condition") != 80
        or counts.get("emotion_recognition_images_per_applicable_condition") != 80
        or counts.get("human_guidance_scoring_images_per_condition") != 78
        or counts.get("generated_responses_across_four_conditions") != 320
        or counts.get("human_guidance_scored_responses_across_four_conditions") != 312
        or counts.get("reviewers") != 6
        or counts.get("images_per_reviewer") != [13] * 6
        or counts.get("responses_per_reviewer") != [52] * 6
    ):
        raise ValueError("Display exclusion record does not match the approved Eval80 guidance scope.")
    return GuidanceScope(
        decision_id=payload["decision_id"],
        excluded_ids=frozenset(excluded),
        source_sha256=sha256_file(path),
    )


def parse_predicted_emotion(response: str, *, structured: bool) -> EmotionParse:
    text = response.strip()
    if not text:
        return EmotionParse("missing", None, (), "empty_response")

    if structured:
        emotions = tuple(match.lower() for match in _STRUCTURED_EMOTION_RE.findall(text))
        finals = tuple(match.lower() for match in _STRUCTURED_FINAL_RE.findall(text))
        candidates = tuple(sorted(set(emotions + finals)))
        if len(emotions) == 1 and len(finals) == 1 and emotions[0] == finals[0]:
            return EmotionParse("valid", emotions[0], candidates, "matching_structured_fields")
        if not emotions or not finals:
            return EmotionParse("missing", None, candidates, "missing_structured_field")
        return EmotionParse("ambiguous", None, candidates, "inconsistent_structured_fields")

    explicit = tuple(match.lower() for match in _EXPLICIT_RE.findall(text))
    explicit_unique = tuple(sorted(set(explicit)))
    if len(explicit_unique) == 1:
        return EmotionParse("valid", explicit_unique[0], explicit_unique, "unique_explicit_label")
    if len(explicit_unique) > 1:
        return EmotionParse("ambiguous", None, explicit_unique, "multiple_explicit_labels")

    all_unique = tuple(sorted(set(match.lower() for match in _ANY_LABEL_RE.findall(text))))
    if len(all_unique) == 1:
        return EmotionParse("valid", all_unique[0], all_unique, "unique_label_mention")
    if not all_unique:
        return EmotionParse("missing", None, (), "no_allowed_label")
    return EmotionParse("ambiguous", None, all_unique, "multiple_label_mentions")


def summarize_recognition(
    output_records: list[dict[str, Any]],
    ground_truth_by_id: dict[str, str],
    *,
    structured: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(output_records) != 80 or len(ground_truth_by_id) != 80:
        raise ValueError("Recognition scoring requires exactly 80 outputs and 80 labels.")
    actual_ids = [str(record.get("blind_id", "")) for record in output_records]
    if len(set(actual_ids)) != 80 or set(actual_ids) != set(ground_truth_by_id):
        raise ValueError("Output blind IDs must uniquely match the 80 ground-truth IDs.")

    scored: list[dict[str, Any]] = []
    class_correct: Counter[str] = Counter()
    class_total: Counter[str] = Counter()
    parse_status_counts: Counter[str] = Counter()
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    for record in output_records:
        blind_id = str(record["blind_id"])
        target = ground_truth_by_id[blind_id]
        if target not in EMOTION_CLASSES:
            raise ValueError(f"{blind_id}: invalid ground-truth emotion {target!r}.")
        parsed = parse_predicted_emotion(str(record.get("response", "")), structured=structured)
        correct = parsed.status == "valid" and parsed.predicted_emotion == target
        class_total[target] += 1
        class_correct[target] += int(correct)
        parse_status_counts[parsed.status] += 1
        prediction_key = parsed.predicted_emotion or f"__{parsed.status}__"
        confusion[target][prediction_key] += 1
        scored.append(
            {
                "blind_id": blind_id,
                "target_emotion": target,
                **asdict(parsed),
                "correct": correct,
            }
        )

    correct_count = sum(int(row["correct"]) for row in scored)
    summary = {
        "schema_version": 2,
        "item_count": 80,
        "correct_count": correct_count,
        "emotion_recognition_score_0_10": round(correct_count / 80 * 10, 4),
        "parse_status_counts": dict(sorted(parse_status_counts.items())),
        "per_class": {
            emotion: {
                "correct": class_correct[emotion],
                "total": class_total[emotion],
                "accuracy": round(class_correct[emotion] / class_total[emotion], 4),
            }
            for emotion in EMOTION_CLASSES
        },
        "confusion_matrix": {
            emotion: dict(sorted(confusion[emotion].items())) for emotion in EMOTION_CLASSES
        },
    }
    return scored, summary


def _parse_failure_flags(raw: str, *, row_number: int) -> list[str]:
    flags = [value.strip() for value in raw.split(";") if value.strip()]
    unknown = sorted(set(flags) - FAILURE_FLAGS)
    if unknown:
        raise ValueError(f"CSV row {row_number}: unknown failure flags: {', '.join(unknown)}")
    if len(flags) != len(set(flags)):
        raise ValueError(f"CSV row {row_number}: duplicate failure flags.")
    return flags


def summarize_guidance_scores(
    rows: list[dict[str, str]],
    *,
    guidance_scope: GuidanceScope,
    expected_conditions: tuple[str, ...] = CONDITIONS,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    expected_count = len(guidance_scope.included_ids) * len(expected_conditions)
    if len(rows) != expected_count:
        raise ValueError(f"Expected {expected_count} guidance rows, found {len(rows)}.")

    seen_items: set[str] = set()
    seen_pairs: set[tuple[str, str]] = set()
    reviewer_by_image: dict[str, str] = {}
    totals_by_reviewer: dict[str, list[int]] = defaultdict(list)
    totals_by_condition: dict[str, list[int]] = defaultdict(list)
    totals_by_condition_emotion: dict[str, dict[str, list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    dimensions_by_condition: dict[str, dict[str, list[int]]] = defaultdict(
        lambda: {column: [] for column in GUIDANCE_COLUMNS}
    )
    flag_counts_by_condition: dict[str, Counter[str]] = defaultdict(Counter)
    scored_rows: list[dict[str, Any]] = []

    for row_number, row in enumerate(rows, start=2):
        item_id = (row.get("anonymous_item_id") or "").strip()
        reviewer_id = (row.get("reviewer_id") or "").strip()
        blind_id = (row.get("blind_id") or "").strip()
        condition = (row.get("condition") or "").strip()
        emotion = (row.get("target_emotion") or "").strip()
        rationale = (row.get("overall_rationale") or "").strip()
        zero_reasons = (row.get("zero_score_reasons") or "").strip()
        if not item_id or item_id in seen_items:
            raise ValueError(f"CSV row {row_number}: anonymous_item_id must be non-empty and unique.")
        if not blind_id:
            raise ValueError(f"CSV row {row_number}: blind_id is empty.")
        if blind_id in guidance_scope.excluded_ids:
            raise ValueError(f"CSV row {row_number}: {blind_id} is excluded from guidance scoring.")
        if blind_id not in guidance_scope.included_ids:
            raise ValueError(f"CSV row {row_number}: unknown guidance blind_id {blind_id!r}.")
        if reviewer_id not in REVIEWER_IDS:
            raise ValueError(f"CSV row {row_number}: invalid reviewer_id {reviewer_id!r}.")
        assigned_reviewer = reviewer_by_image.setdefault(blind_id, reviewer_id)
        if assigned_reviewer != reviewer_id:
            raise ValueError(
                f"CSV row {row_number}: all conditions for {blind_id} must use one reviewer."
            )
        if condition not in expected_conditions:
            raise ValueError(f"CSV row {row_number}: invalid condition {condition!r}.")
        if emotion not in EMOTION_CLASSES:
            raise ValueError(f"CSV row {row_number}: invalid target_emotion {emotion!r}.")
        if (blind_id, condition) in seen_pairs:
            raise ValueError(f"CSV row {row_number}: duplicate image-condition pair.")
        if not rationale:
            raise ValueError(f"CSV row {row_number}: overall_rationale is required.")
        seen_items.add(item_id)
        seen_pairs.add((blind_id, condition))

        output = dict(row)
        scores: list[int] = []
        for column in GUIDANCE_COLUMNS:
            raw_score = (row.get(column) or "").strip()
            if raw_score not in {"0", "1", "2"}:
                raise ValueError(
                    f"CSV row {row_number}: {column} must be exactly 0, 1, or 2."
                )
            score = int(raw_score)
            scores.append(score)
            dimensions_by_condition[condition][column].append(score)
        if 0 in scores and not zero_reasons:
            raise ValueError(f"CSV row {row_number}: zero_score_reasons is required for a zero.")
        flags = _parse_failure_flags(row.get("failure_flags", ""), row_number=row_number)
        flag_counts_by_condition[condition].update(flags)
        total = sum(scores)
        supplied_total = (row.get("guidance_total_0_10") or "").strip()
        if supplied_total and supplied_total != str(total):
            raise ValueError(f"CSV row {row_number}: guidance_total_0_10 differs from the five scores.")
        output["guidance_total_0_10"] = total
        scored_rows.append(output)
        totals_by_condition[condition].append(total)
        totals_by_condition_emotion[condition][emotion].append(total)
        totals_by_reviewer[reviewer_id].append(total)

    expected_pairs = {
        (blind_id, condition)
        for blind_id in guidance_scope.included_ids
        for condition in expected_conditions
    }
    if seen_pairs != expected_pairs:
        raise ValueError("Guidance rows do not contain every included image for every condition.")
    reviewer_image_counts = {
        reviewer_id: sum(
            assigned_reviewer == reviewer_id
            for assigned_reviewer in reviewer_by_image.values()
        )
        for reviewer_id in REVIEWER_IDS
    }
    if reviewer_image_counts != REVIEWER_IMAGE_COUNTS:
        raise ValueError(
            "Reviewer image counts must be exactly 13/13/13/13/13/13."
        )

    summary = {
        "schema_version": 2,
        "item_count": len(scored_rows),
        "image_count": len(guidance_scope.included_ids),
        "excluded_blind_ids": sorted(guidance_scope.excluded_ids),
        "display_exclusion_decision_id": guidance_scope.decision_id,
        "display_exclusion_sha256": guidance_scope.source_sha256,
        "maximum_per_item": 10,
        "reviewer_design": "fixed_non_overlapping_image_partitions",
        "designated_reviewer_count": len(REVIEWER_IDS),
        "reviewer_image_counts": reviewer_image_counts,
        "reviewer_response_counts": {
            reviewer_id: len(totals_by_reviewer[reviewer_id])
            for reviewer_id in REVIEWER_IDS
        },
        "reviewer_mean_guidance_total_0_10": {
            reviewer_id: round(mean(totals_by_reviewer[reviewer_id]), 4)
            for reviewer_id in REVIEWER_IDS
        },
        "inter_rater_reliability_measured": False,
        "conditions": {
            condition: {
                "item_count": len(totals_by_condition[condition]),
                "mean_guidance_total_0_10": round(
                    mean(totals_by_condition[condition]), 4
                ),
                "dimension_means_0_2": {
                    column: round(mean(values), 4)
                    for column, values in dimensions_by_condition[condition].items()
                },
                "emotion_mean_guidance_total_0_10": {
                    emotion: round(mean(totals_by_condition_emotion[condition][emotion]), 4)
                    for emotion in EMOTION_CLASSES
                },
                "failure_flag_counts": dict(
                    sorted(flag_counts_by_condition[condition].items())
                ),
            }
            for condition in expected_conditions
        },
    }
    return scored_rows, summary


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSONL at line {line_number}: {error}") from error
    return records


def _stable_seed(seed: int, blind_id: str) -> int:
    digest = hashlib.sha256(f"eval80-scoring-v1:{seed}:{blind_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _anonymous_id(seed: int, blind_id: str, condition: str) -> str:
    digest = hashlib.sha256(
        f"eval80-anonymous-v1:{seed}:{blind_id}:{condition}".encode("utf-8")
    ).hexdigest()
    return f"X{digest[:9].upper()}"


def _reviewer_assignments(
    selection_by_id: dict[str, dict[str, Any]], seed: int, guidance_scope: GuidanceScope
) -> dict[str, str]:
    """Assign the 78 guidance images to six equal, near-class-balanced partitions."""
    if set(selection_by_id) != ALL_BLIND_IDS:
        raise ValueError("Reviewer assignment requires the complete E001-E080 selection.")
    full_class_counts = Counter(str(record["emotion"]) for record in selection_by_id.values())
    if full_class_counts != Counter({emotion: 10 for emotion in EMOTION_CLASSES}):
        raise ValueError("Reviewer assignment requires ten images per emotion class.")
    by_emotion: dict[str, list[str]] = {}
    for blind_id in guidance_scope.included_ids:
        record = selection_by_id[blind_id]
        by_emotion.setdefault(str(record["emotion"]), []).append(blind_id)

    assignments: dict[str, str] = {}
    total_counts: Counter[str] = Counter()
    for emotion in sorted(by_emotion):
        blind_ids = sorted(by_emotion[emotion])
        random.Random(_stable_seed(seed, f"reviewer:{emotion}")).shuffle(blind_ids)
        tie_order = list(REVIEWER_IDS)
        random.Random(_stable_seed(seed, f"tie:{emotion}")).shuffle(tie_order)
        tie_rank = {reviewer_id: index for index, reviewer_id in enumerate(tie_order)}
        class_counts: Counter[str] = Counter()
        for blind_id in blind_ids:
            available = [
                reviewer_id for reviewer_id in REVIEWER_IDS
                if total_counts[reviewer_id] < REVIEWER_IMAGE_COUNTS[reviewer_id]
            ]
            if not available:
                raise AssertionError("No reviewer capacity remains for a guidance image.")
            reviewer_id = min(
                available,
                key=lambda value: (class_counts[value], total_counts[value], tie_rank[value]),
            )
            assignments[blind_id] = reviewer_id
            class_counts[reviewer_id] += 1
            total_counts[reviewer_id] += 1

    image_counts = {
        reviewer_id: total_counts[reviewer_id]
        for reviewer_id in REVIEWER_IDS
    }
    if image_counts != REVIEWER_IMAGE_COUNTS or set(assignments) != guidance_scope.included_ids:
        raise AssertionError(f"Unexpected reviewer assignment counts: {image_counts}")
    return assignments


def _records_by_id(path: Path, expected_condition: str) -> dict[str, dict[str, Any]]:
    records = _read_jsonl(path)
    if len(records) != 80:
        raise ValueError(f"{expected_condition}: expected 80 outputs, found {len(records)}.")
    by_id: dict[str, dict[str, Any]] = {}
    for record in records:
        blind_id = str(record.get("blind_id", ""))
        if not blind_id or blind_id in by_id:
            raise ValueError(f"{expected_condition}: blind IDs must be non-empty and unique.")
        if record.get("condition") != expected_condition or record.get("status") != "success":
            raise ValueError(
                f"{expected_condition}/{blind_id}: output is not a successful condition record."
            )
        response = record.get("response")
        if not isinstance(response, str) or not response.strip():
            raise ValueError(f"{expected_condition}/{blind_id}: response is empty.")
        if record.get("response_sha256") != hashlib.sha256(response.encode("utf-8")).hexdigest():
            raise ValueError(f"{expected_condition}/{blind_id}: response SHA-256 mismatch.")
        by_id[blind_id] = record
    return by_id


def verify_private_selection_manifest(selection_path: Path, freeze_path: Path) -> None:
    """Require the private answer key to match the approved Eval80 freeze."""
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if (
        not isinstance(freeze, dict)
        or freeze.get("schema_version") != 1
        or freeze.get("status") != "approved_and_frozen"
        or freeze.get("team_approval_confirmed") is not True
    ):
        raise ValueError("Eval80 freeze record is not approved and frozen.")
    file_hashes = freeze.get("file_sha256")
    expected = file_hashes.get("selection_manifest_private") if isinstance(file_hashes, dict) else None
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError("Eval80 freeze record lacks a valid private selection SHA-256.")
    if sha256_file(selection_path) != expected:
        raise ValueError("Private selection manifest SHA-256 differs from the Eval80 freeze record.")


def score_recognition_command(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Calculate the separate 0-10 Eval80-v1 emotion-recognition result."
    )
    parser.add_argument("--condition", choices=("B0", "B1", "C"), required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    verify_private_selection_manifest(args.selection_manifest, FREEZE_RECORD_PATH)
    outputs = list(_records_by_id(args.records, args.condition).values())
    selection = read_selection_manifest(args.selection_manifest)
    labels = {
        str(record["blind_id"]): str(record["emotion"])
        for record in selection["records"]
    }
    scored, summary = summarize_recognition(
        outputs,
        labels,
        structured=args.condition in {"B1", "C"},
    )
    summary["condition"] = args.condition
    for row in scored:
        row["condition"] = args.condition

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    details_path = output_dir / f"{args.condition}_recognition_details.csv"
    with details_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(scored[0].keys()))
        writer.writeheader()
        writer.writerows(scored)
    (output_dir / f"{args.condition}_recognition_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _validate_recognition_details(
    rows: list[dict[str, str]], expected_condition: str
) -> dict[str, dict[str, Any]]:
    if len(rows) != 80:
        raise ValueError(f"{expected_condition} recognition details must contain exactly 80 rows.")
    by_id: dict[str, dict[str, Any]] = {}
    target_counts: Counter[str] = Counter()
    for row_number, row in enumerate(rows, start=2):
        blind_id = (row.get("blind_id") or "").strip()
        target = (row.get("target_emotion") or "").strip()
        status = (row.get("status") or "").strip()
        predicted = (row.get("predicted_emotion") or "").strip()
        correct_text = (row.get("correct") or "").strip()
        if (row.get("condition") or "").strip() != expected_condition:
            raise ValueError(f"{expected_condition} row {row_number}: condition mismatch.")
        if blind_id not in ALL_BLIND_IDS or blind_id in by_id:
            raise ValueError(f"{expected_condition} row {row_number}: invalid or duplicate blind ID.")
        if target not in EMOTION_CLASSES:
            raise ValueError(f"{expected_condition}/{blind_id}: invalid target emotion.")
        if status not in {"valid", "missing", "ambiguous"}:
            raise ValueError(f"{expected_condition}/{blind_id}: invalid parse status.")
        if (status == "valid" and predicted not in EMOTION_CLASSES) or (
            status != "valid" and predicted
        ):
            raise ValueError(f"{expected_condition}/{blind_id}: prediction disagrees with parse status.")
        if correct_text not in {"True", "False"}:
            raise ValueError(f"{expected_condition}/{blind_id}: correct must be True or False.")
        correct = status == "valid" and predicted == target
        if (correct_text == "True") != correct:
            raise ValueError(f"{expected_condition}/{blind_id}: correctness disagrees with prediction.")
        by_id[blind_id] = {
            "target": target,
            "status": status,
            "correct": correct,
        }
        target_counts[target] += 1
    if set(by_id) != ALL_BLIND_IDS:
        raise ValueError(f"{expected_condition} recognition details do not cover E001-E080.")
    if target_counts != Counter({emotion: 10 for emotion in EMOTION_CLASSES}):
        raise ValueError(f"{expected_condition} recognition details must have ten images per class.")
    return by_id


def compare_recognition_details(
    b1_rows: list[dict[str, str]], c_rows: list[dict[str, str]]
) -> dict[str, Any]:
    """Aggregate strict, same-image B1/C recognition results without image-level output."""
    b1 = _validate_recognition_details(b1_rows, "B1")
    condition_c = _validate_recognition_details(c_rows, "C")
    paired = Counter()
    parse_transitions = Counter()
    parse_status = {condition: Counter() for condition in ("B1", "C")}
    per_class = {
        emotion: {"image_count": 0, "b1_correct": 0, "c_correct": 0}
        for emotion in EMOTION_CLASSES
    }
    for blind_id in sorted(ALL_BLIND_IDS):
        base = b1[blind_id]
        adapted = condition_c[blind_id]
        if base["target"] != adapted["target"]:
            raise ValueError(f"B1/C target emotion mismatch for {blind_id}.")
        emotion = base["target"]
        per_class[emotion]["image_count"] += 1
        per_class[emotion]["b1_correct"] += int(base["correct"])
        per_class[emotion]["c_correct"] += int(adapted["correct"])
        parse_status["B1"][base["status"]] += 1
        parse_status["C"][adapted["status"]] += 1
        parse_transitions[f"{base['status']}->{adapted['status']}"] += 1
        paired[(base["correct"], adapted["correct"])] += 1
    b1_correct = sum(entry["b1_correct"] for entry in per_class.values())
    c_correct = sum(entry["c_correct"] for entry in per_class.values())
    for entry in per_class.values():
        entry["correct_delta_c_minus_b1"] = entry["c_correct"] - entry["b1_correct"]
    return {
        "schema_version": 1,
        "evaluation_set": "Eval80-v1",
        "comparison": "B1_vs_C_strict_emotion_recognition",
        "image_count": 80,
        "scoring_rule": "matching structured Emotion and Final emotion labels; invalid parses count as incorrect",
        "conditions": {
            "B1": {
                "correct_count": b1_correct,
                "accuracy": round(b1_correct / 80, 4),
                "emotion_recognition_score_0_10": round(b1_correct / 8, 4),
                "parse_status_counts": dict(sorted(parse_status["B1"].items())),
            },
            "C": {
                "correct_count": c_correct,
                "accuracy": round(c_correct / 80, 4),
                "emotion_recognition_score_0_10": round(c_correct / 8, 4),
                "parse_status_counts": dict(sorted(parse_status["C"].items())),
            },
        },
        "net_correct_gain_c_minus_b1": c_correct - b1_correct,
        "paired_outcomes": {
            "both_correct": paired[(True, True)],
            "b1_only_correct": paired[(True, False)],
            "c_only_correct": paired[(False, True)],
            "both_wrong": paired[(False, False)],
        },
        "parse_status_transitions": dict(sorted(parse_transitions.items())),
        "per_class": per_class,
    }


def compare_recognition_command(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare private B1/C recognition details on paired Eval80 images."
    )
    parser.add_argument("--b1-details", type=Path, required=True)
    parser.add_argument("--c-details", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"Comparison output already exists: {args.output}")
    details = {}
    for condition, path in (("B1", args.b1_details), ("C", args.c_details)):
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            details[condition] = list(csv.DictReader(handle))
    report = compare_recognition_details(details["B1"], details["C"])
    report["input_sha256"] = {
        "b1_details": sha256_file(args.b1_details),
        "c_details": sha256_file(args.c_details),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def create_score_sheets_command(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Create six fixed-partition blind score sheets from A/B0/B1/C outputs."
    )
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--display-exclusion", type=Path, required=True)
    parser.add_argument("--a-records", type=Path, required=True)
    parser.add_argument("--b0-records", type=Path, required=True)
    parser.add_argument("--b1-records", type=Path, required=True)
    parser.add_argument("--c-records", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args(argv)

    guidance_scope = read_guidance_scope(args.display_exclusion)
    selection = read_selection_manifest(args.selection_manifest)
    selection_by_id = {str(record["blind_id"]): record for record in selection["records"]}
    if len(selection["records"]) != 80 or set(selection_by_id) != ALL_BLIND_IDS:
        raise ValueError("Selection manifest must contain E001-E080 exactly once.")
    paths = {
        "A": args.a_records,
        "B0": args.b0_records,
        "B1": args.b1_records,
        "C": args.c_records,
    }
    records_by_condition = {
        condition: _records_by_id(path, condition) for condition, path in paths.items()
    }
    expected_ids = set(selection_by_id)
    for condition, records in records_by_condition.items():
        if set(records) != expected_ids:
            raise ValueError(f"{condition}: output IDs differ from the selection manifest.")

    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    fieldnames = (
        "anonymous_item_id",
        "reviewer_id",
        "blind_id",
        "image_filename",
        "target_emotion",
        "model_response",
        *GUIDANCE_COLUMNS,
        "guidance_total_0_10",
        "overall_rationale",
        "zero_score_reasons",
        "failure_flags",
    )
    score_rows: list[dict[str, str]] = []
    mapping: dict[str, dict[str, str]] = {}
    reviewer_by_blind_id = _reviewer_assignments(selection_by_id, args.seed, guidance_scope)
    for blind_id in sorted(guidance_scope.included_ids):
        conditions = list(CONDITIONS)
        random.Random(_stable_seed(args.seed, blind_id)).shuffle(conditions)
        selection_record = selection_by_id[blind_id]
        for condition in conditions:
            output_record = records_by_condition[condition][blind_id]
            anonymous_id = _anonymous_id(args.seed, blind_id, condition)
            score_row = {
                "anonymous_item_id": anonymous_id,
                "reviewer_id": reviewer_by_blind_id[blind_id],
                "blind_id": blind_id,
                "image_filename": str(selection_record["local_image_filename"]),
                "target_emotion": str(selection_record["emotion"]),
                "model_response": str(output_record["response"]),
                "guidance_total_0_10": "",
                "overall_rationale": "",
                "zero_score_reasons": "",
                "failure_flags": "",
            }
            score_row.update({column: "" for column in GUIDANCE_COLUMNS})
            score_rows.append(score_row)
            mapping[anonymous_id] = {
                "blind_id": blind_id,
                "condition": condition,
                "reviewer_id": reviewer_by_blind_id[blind_id],
                "image_filename": str(selection_record["local_image_filename"]),
                "target_emotion": str(selection_record["emotion"]),
                "response_sha256": str(output_record["response_sha256"]),
            }

    score_paths: dict[str, Path] = {}
    for reviewer_id in REVIEWER_IDS:
        reviewer_rows = [
            row for row in score_rows if row["reviewer_id"] == reviewer_id
        ]
        score_path = output_dir / f"guidance_scores_{reviewer_id}.csv"
        with score_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(reviewer_rows)
        score_paths[reviewer_id] = score_path

    mapping_payload = {
        "schema_version": 2,
        "evaluation_set": "Eval80-v1",
        "seed": args.seed,
        "display_exclusion_decision_id": guidance_scope.decision_id,
        "display_exclusion_sha256": guidance_scope.source_sha256,
        "excluded_blind_ids": sorted(guidance_scope.excluded_ids),
        "item_count": len(mapping),
        "condition_mapping": mapping,
        "reviewer_assignment": dict(sorted(reviewer_by_blind_id.items())),
    }
    mapping_path = output_dir / "condition_mapping_private.json"
    mapping_path.write_text(
        json.dumps(mapping_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary = {
        "schema_version": 2,
        "evaluation_set": "Eval80-v1",
        "item_count": len(score_rows),
        "image_count": len(guidance_scope.included_ids),
        "condition_count": 4,
        "display_exclusion_decision_id": guidance_scope.decision_id,
        "display_exclusion_sha256": guidance_scope.source_sha256,
        "excluded_blind_ids": sorted(guidance_scope.excluded_ids),
        "designated_reviewer_count": len(REVIEWER_IDS),
        "reviewer_image_counts": {
            reviewer_id: sum(
                assigned_id == reviewer_id
                for assigned_id in reviewer_by_blind_id.values()
            )
            for reviewer_id in REVIEWER_IDS
        },
        "reviewer_response_counts": {
            reviewer_id: sum(
                row["reviewer_id"] == reviewer_id for row in score_rows
            )
            for reviewer_id in REVIEWER_IDS
        },
        "score_sheet_sha256": {
            reviewer_id: sha256_file(path)
            for reviewer_id, path in score_paths.items()
        },
        "condition_mapping_sha256": sha256_file(mapping_path),
        "condition_source_sha256": {
            condition: sha256_file(path) for condition, path in paths.items()
        },
    }
    (output_dir / "scoring_package_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def summarize_guidance_command(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate six-reviewer scores and restore private condition labels."
    )
    parser.add_argument("--scores", type=Path, nargs="+", required=True)
    parser.add_argument("--condition-mapping", type=Path, required=True)
    parser.add_argument("--display-exclusion", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    guidance_scope = read_guidance_scope(args.display_exclusion)
    rows: list[dict[str, str]] = []
    for score_path in args.scores:
        with score_path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows.extend(dict(row) for row in csv.DictReader(handle))
    mapping_payload = json.loads(args.condition_mapping.read_text(encoding="utf-8"))
    mapping = mapping_payload.get("condition_mapping", {})
    if (
        mapping_payload.get("display_exclusion_decision_id") != guidance_scope.decision_id
        or mapping_payload.get("display_exclusion_sha256") != guidance_scope.source_sha256
        or mapping_payload.get("excluded_blind_ids") != sorted(guidance_scope.excluded_ids)
    ):
        raise ValueError("Condition mapping is bound to a different display exclusion record.")
    if not isinstance(mapping, dict) or len(mapping) != 312:
        raise ValueError("Condition mapping must contain exactly 312 anonymous items.")
    expected_pairs = {
        (blind_id, condition)
        for blind_id in guidance_scope.included_ids
        for condition in CONDITIONS
    }
    mapped_pairs = {
        (entry.get("blind_id"), entry.get("condition"))
        for entry in mapping.values()
        if isinstance(entry, dict)
    }
    if mapped_pairs != expected_pairs:
        raise ValueError("Condition mapping does not contain every included image-condition pair.")
    if set(mapping_payload.get("reviewer_assignment", {})) != guidance_scope.included_ids:
        raise ValueError("Condition mapping reviewer assignment differs from guidance scope.")

    restored: list[dict[str, str]] = []
    for row_number, row in enumerate(rows, start=2):
        item_id = (row.get("anonymous_item_id") or "").strip()
        blind_id = (row.get("blind_id") or "").strip()
        if blind_id in guidance_scope.excluded_ids:
            raise ValueError(f"CSV row {row_number}: {blind_id} is excluded from guidance scoring.")
        mapped = mapping.get(item_id)
        if not mapped:
            raise ValueError(f"CSV row {row_number}: anonymous item is absent from the mapping.")
        if mapped.get("blind_id") != blind_id:
            raise ValueError(f"CSV row {row_number}: blind ID differs from the private mapping.")
        if mapped.get("reviewer_id") != (row.get("reviewer_id") or "").strip():
            raise ValueError(f"CSV row {row_number}: reviewer ID differs from the private mapping.")
        if mapped.get("image_filename") != (row.get("image_filename") or "").strip():
            raise ValueError(f"CSV row {row_number}: image filename differs from the private mapping.")
        if mapped.get("target_emotion") != (row.get("target_emotion") or "").strip():
            raise ValueError(f"CSV row {row_number}: target emotion differs from the private mapping.")
        response = row.get("model_response") or ""
        if hashlib.sha256(response.encode("utf-8")).hexdigest() != mapped.get("response_sha256"):
            raise ValueError(f"CSV row {row_number}: model response SHA-256 differs from the private mapping.")
        restored_row = dict(row)
        restored_row["condition"] = str(mapped["condition"])
        restored.append(restored_row)

    scored, summary = summarize_guidance_scores(restored, guidance_scope=guidance_scope)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    details_path = output_dir / "guidance_scores_with_conditions_private.csv"
    with details_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(scored[0].keys()))
        writer.writeheader()
        writer.writerows(scored)
    (output_dir / "guidance_score_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0
