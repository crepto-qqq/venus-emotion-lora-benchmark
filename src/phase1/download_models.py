"""Download pinned Venus snapshots outside the Git repository."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path


SOURCE_FILENAME = "VENUS_MODEL_SOURCE.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/phase1/baseline.json"))
    parser.add_argument("--stage", choices=("stage1", "stage2", "all"), default="all")
    args = parser.parse_args()

    from huggingface_hub import snapshot_download

    with args.config.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    stages = ("stage1", "stage2") if args.stage == "all" else (args.stage,)
    args.model_root.mkdir(parents=True, exist_ok=True)

    for stage in stages:
        model_config = config["models"][stage]
        target = (args.model_root / stage).resolve()
        metadata_path = target / SOURCE_FILENAME
        if metadata_path.exists():
            existing = json.loads(metadata_path.read_text(encoding="utf-8"))
            if (
                existing.get("repo_id") != model_config["repo_id"]
                or existing.get("revision") != model_config["revision"]
            ):
                raise RuntimeError(f"Existing model provenance does not match pinned {stage}: {target}")

        print(f"Downloading {model_config['repo_id']} at {model_config['revision']} to {target}")
        snapshot_download(
            repo_id=model_config["repo_id"],
            revision=model_config["revision"],
            local_dir=str(target),
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        metadata = {
            "schema_version": 1,
            "stage": stage,
            "repo_id": model_config["repo_id"],
            "revision": model_config["revision"],
            "weight_format": model_config["weight_format"],
            "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

        if stage == "stage1":
            weight_files = list(target.glob("model-*.safetensors"))
        else:
            weight_files = list(target.glob("pytorch_model-*.bin"))
        if not weight_files:
            raise RuntimeError(f"No expected weight shards found for {stage} in {target}")
        print(f"Verified {len(weight_files)} weight shards for {stage}.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
