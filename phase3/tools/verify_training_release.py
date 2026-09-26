"""Verify a written BridgeTrain-v1 release independently of the builder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.bridge_train import (
    load_config,
    load_eval20_exclusions,
    load_eval80_exclusions,
    project_root,
    verify_release_dir,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=root / "phase3/configs/bridge-train-v1.json",
    )
    parser.add_argument("--source-root", type=Path, default=root / "data/datasets")
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--extracted-image-root", type=Path)
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
    report = verify_release_dir(
        release_dir=args.release_dir,
        config=config,
        archive_path=args.archive,
        extracted_image_root=args.extracted_image_root,
        source_root=args.source_root,
        eval20_ids=eval20_ids,
        eval20_hashes=eval20_hashes,
        eval80_ids=eval80_ids,
        eval80_hashes=eval80_hashes,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
