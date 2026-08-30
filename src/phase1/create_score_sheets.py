"""Create consistently shuffled 0/1/2 Phase 1 human-scoring sheets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import random

from .manifest import read_manifest


def _stable_seed(seed: int, stage: str) -> int:
    digest = hashlib.sha256(f"{seed}:{stage}:scoring".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _write_sheet(path: Path, rows: list[dict[str, str]], stage: str) -> None:
    if stage == "stage1":
        score_columns = (
            "emotion_recognition_0_2",
            "visual_evidence_0_2",
            "emotion_consistent_guidance_0_2",
        )
    else:
        score_columns = (
            "emotion_recognition_0_2",
            "crop_visual_evidence_0_2",
            "crop_emotion_preservation_0_2",
        )
    fieldnames = ("blind_item_id", "sample_id", "target_emotion", *score_columns, "notes")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for position, row in enumerate(rows, start=1):
            output = {
                "blind_item_id": f"{stage.upper()}-{position:03d}",
                "sample_id": row["sample_id"],
                "target_emotion": row["emotion"],
                "notes": "",
            }
            output.update({column: "" for column in score_columns})
            writer.writerow(output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/phase1/baseline.json"))
    args = parser.parse_args()

    with args.config.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    rows = read_manifest(args.manifest)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for stage in ("stage1", "stage2"):
        shuffled = [dict(row) for row in rows]
        random.Random(_stable_seed(int(config["seed"]), stage)).shuffle(shuffled)
        _write_sheet(output_dir / f"{stage}_scores.csv", shuffled, stage)
    print(f"Created Stage 1 and Stage 2 score sheets in {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
