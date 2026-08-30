"""Deterministically select the fixed 20-image Phase 1 set from EmoSet test.json."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random

from .manifest import sha256_file, write_manifest


def _emotion_seed(seed: int, emotion: str) -> int:
    digest = hashlib.sha256(f"{seed}:{emotion}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def _row(
    sample_id: str,
    entry: list[str],
    valence: str,
    selection_rank: int,
    emoset_root: Path,
    review_status: str,
) -> dict[str, str]:
    emotion, source_image_id, image_relpath, annotation_relpath = entry[:4]
    image_path = (emoset_root / image_relpath).resolve()
    return {
        "sample_id": sample_id,
        "source_image_id": source_image_id,
        "emotion": emotion,
        "valence": valence,
        "split": "test",
        "image_relpath": image_relpath,
        "annotation_relpath": annotation_relpath,
        "sha256": sha256_file(image_path),
        "review_status": review_status,
        "replacement_for": "",
        "review_notes": f"deterministic_selection_rank={selection_rank}",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emoset-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/phase1/baseline.json"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/phase1/emoset20_manifest.csv"),
    )
    parser.add_argument(
        "--reserve-output",
        type=Path,
        default=Path("data/phase1/emoset20_reserve.csv"),
    )
    parser.add_argument("--reserve-per-emotion", type=int, default=3)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.reserve_per_emotion < 0:
        parser.error("--reserve-per-emotion must be non-negative")

    with args.config.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    dataset_config = config["dataset"]
    quotas: dict[str, int] = dataset_config["emotion_quotas"]
    valence: dict[str, str] = dataset_config["valence"]
    selection_seed = int(dataset_config["selection_seed"])

    test_path = args.emoset_root / "test.json"
    with test_path.open("r", encoding="utf-8") as handle:
        test_entries = json.load(handle)

    grouped: dict[str, list[list[str]]] = defaultdict(list)
    for position, raw_entry in enumerate(test_entries):
        if not isinstance(raw_entry, list) or len(raw_entry) < 4:
            raise ValueError(f"Invalid test.json entry at index {position}.")
        entry = [str(value) for value in raw_entry[:4]]
        emotion, _, image_relpath, annotation_relpath = entry
        if emotion not in quotas:
            continue
        image_path = (args.emoset_root / image_relpath).resolve()
        annotation_path = (args.emoset_root / annotation_relpath).resolve()
        if not image_path.is_file() or not annotation_path.is_file():
            raise FileNotFoundError(
                f"EmoSet test entry is incomplete: {image_relpath}, {annotation_relpath}"
            )
        grouped[emotion].append(entry)

    selected_rows: list[dict[str, str]] = []
    reserve_rows: list[dict[str, str]] = []
    selected_counter = 1
    for emotion, quota in quotas.items():
        candidates = grouped.get(emotion, [])
        required = quota + args.reserve_per_emotion
        if len(candidates) < required:
            raise ValueError(f"Emotion '{emotion}' has {len(candidates)} candidates; need {required}.")
        random.Random(_emotion_seed(selection_seed, emotion)).shuffle(candidates)

        for rank, entry in enumerate(candidates[:quota], start=1):
            selected_rows.append(
                _row(
                    f"E{selected_counter:02d}",
                    entry,
                    valence[emotion],
                    rank,
                    args.emoset_root,
                    "pending",
                )
            )
            selected_counter += 1

        reserve_start = quota
        for reserve_rank, entry in enumerate(
            candidates[reserve_start : reserve_start + args.reserve_per_emotion],
            start=1,
        ):
            reserve_rows.append(
                _row(
                    f"R-{emotion}-{reserve_rank}",
                    entry,
                    valence[emotion],
                    quota + reserve_rank,
                    args.emoset_root,
                    "reserve",
                )
            )

    write_manifest(args.output, selected_rows, overwrite=args.force)
    write_manifest(args.reserve_output, reserve_rows, overwrite=args.force)
    print(f"Wrote {len(selected_rows)} pending selections to {args.output}")
    print(f"Wrote {len(reserve_rows)} deterministic reserves to {args.reserve_output}")
    print("Review every selected image, then change review_status to 'approved'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
