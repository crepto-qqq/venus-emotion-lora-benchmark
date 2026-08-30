"""Validate completed 0/1/2 score sheets and calculate Phase 1 summaries."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any


SCORE_COLUMNS = {
    "stage1": (
        "emotion_recognition_0_2",
        "visual_evidence_0_2",
        "emotion_consistent_guidance_0_2",
    ),
    "stage2": (
        "emotion_recognition_0_2",
        "crop_visual_evidence_0_2",
        "crop_emotion_preservation_0_2",
    ),
}


def validate_and_summarize(
    rows: list[dict[str, str]],
    stage: str,
    *,
    expected_count: int = 20,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    columns = SCORE_COLUMNS[stage]
    if len(rows) != expected_count:
        raise ValueError(f"Expected {expected_count} scored rows, found {len(rows)}.")

    seen_sample_ids: set[str] = set()
    scored_rows: list[dict[str, Any]] = []
    by_emotion: dict[str, list[int]] = defaultdict(list)
    dimension_values: dict[str, list[int]] = {column: [] for column in columns}

    for row_number, row in enumerate(rows, start=2):
        sample_id = (row.get("sample_id") or "").strip()
        emotion = (row.get("target_emotion") or "").strip()
        if not sample_id:
            raise ValueError(f"CSV row {row_number}: sample_id is empty.")
        if sample_id in seen_sample_ids:
            raise ValueError(f"CSV row {row_number}: duplicate sample_id '{sample_id}'.")
        if not emotion:
            raise ValueError(f"CSV row {row_number}: target_emotion is empty.")
        seen_sample_ids.add(sample_id)

        scores: list[int] = []
        output_row: dict[str, Any] = dict(row)
        for column in columns:
            raw_value = (row.get(column) or "").strip()
            if raw_value not in {"0", "1", "2"}:
                raise ValueError(
                    f"CSV row {row_number}: {column} must be exactly 0, 1, or 2; found {raw_value!r}."
                )
            score = int(raw_value)
            scores.append(score)
            dimension_values[column].append(score)
        total = sum(scores)
        output_row["total_0_6"] = total
        scored_rows.append(output_row)
        by_emotion[emotion].append(total)

    summary = {
        "schema_version": 1,
        "stage": stage,
        "item_count": len(scored_rows),
        "maximum_per_item": 6,
        "mean_total_0_6": round(mean(row["total_0_6"] for row in scored_rows), 4),
        "minimum_total_0_6": min(row["total_0_6"] for row in scored_rows),
        "maximum_total_0_6_observed": max(row["total_0_6"] for row in scored_rows),
        "dimension_means_0_2": {
            column: round(mean(values), 4) for column, values in dimension_values.items()
        },
        "emotion_mean_totals_0_6": {
            emotion: round(mean(values), 4) for emotion, values in sorted(by_emotion.items())
        },
        "emotion_item_counts": {
            emotion: len(values) for emotion, values in sorted(by_emotion.items())
        },
    }
    return scored_rows, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("stage1", "stage2"), required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    with args.scores.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    scored_rows, summary = validate_and_summarize(rows, args.stage)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    totals_path = output_dir / f"{args.stage}_scores_with_totals.csv"
    summary_path = output_dir / f"{args.stage}_score_summary.json"
    fieldnames = list(scored_rows[0].keys())
    with totals_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(scored_rows)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
