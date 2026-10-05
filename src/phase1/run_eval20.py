"""Run fixed 20-image Venus evaluation profiles on the cloud-side EmoSet set."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass(frozen=True)
class Profile:
    output_name: str
    prompt: str


PROFILES = {
    "A_original_aesthetic": Profile(
        output_name="A_original_aesthetic.json",
        prompt="Please critically analyze this image from an aesthetic perspective.",
    ),
    "B0_direct_emotion": Profile(
        output_name="B0_direct_emotion.json",
        prompt=(
            "Which primary emotion does this image convey: amusement, anger, awe, "
            "contentment, disgust, excitement, fear, or sadness?"
        ),
    ),
    "stage2_original_crop": Profile(
        output_name="stage2_original_crop.json",
        prompt=(
            "Please provide the bounding box coordinate of the most visually balanced "
            "and aesthetically pleasing composition area."
        ),
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--profiles",
        nargs="+",
        choices=sorted(PROFILES),
        required=True,
    )
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def write_json_atomic(path: Path, payload: dict) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    records = manifest["records"]
    if len(records) != 20:
        raise ValueError(f"Expected exactly 20 manifest records, found {len(records)}")

    missing_images = [item["local_path"] for item in records if not Path(item["local_path"]).is_file()]
    if missing_images:
        raise FileNotFoundError(f"Missing evaluation images: {missing_images}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    load_started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map="auto",
        trust_remote_code=True,
        bf16=True,
    ).eval()
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - load_started
    load_peak_vram_gib = torch.cuda.max_memory_allocated() / 1024**3

    generation_config = (
        model.generation_config.to_dict()
        if getattr(model, "generation_config", None) is not None
        else None
    )

    for profile_name in args.profiles:
        profile = PROFILES[profile_name]
        output_path = args.output_dir / profile.output_name
        torch.manual_seed(args.seed)
        outputs = []
        payload = {
            "complete": False,
            "profile": profile_name,
            "checkpoint_path": str(args.model_path),
            "manifest_path": str(args.manifest),
            "source_mirror": manifest.get("source_mirror"),
            "source_revision": manifest.get("source_revision"),
            "prompt": profile.prompt,
            "seed": args.seed,
            "precision": "bf16",
            "batch_size": 1,
            "torch_version": torch.__version__,
            "gpu": torch.cuda.get_device_name(0),
            "model_load_seconds": round(load_seconds, 3),
            "model_load_peak_vram_gib": round(load_peak_vram_gib, 3),
            "generation_config": generation_config,
            "outputs": outputs,
        }

        for index, item in enumerate(records):
            image_path = Path(item["local_path"])
            query = tokenizer.from_list_format(
                [
                    {"image": str(image_path)},
                    {"text": profile.prompt},
                ]
            )
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            inference_started = time.perf_counter()
            with torch.inference_mode():
                response, _ = model.chat(tokenizer, query=query, history=None)
            torch.cuda.synchronize()

            outputs.append(
                {
                    "index": index + 1,
                    "image_id": item["image_id"],
                    "emotion_label": item["emotion_label"],
                    "repo_path": item["repo_path"],
                    "sha256": item["sha256"],
                    "response": response,
                    "inference_seconds": round(time.perf_counter() - inference_started, 3),
                    "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
                }
            )
            payload["completed_images"] = len(outputs)
            write_json_atomic(output_path, payload)
            print(
                f"[{profile_name}] {index + 1:02d}/20 "
                f"{item['image_id']} ({item['emotion_label']})"
            )

        payload["complete"] = True
        payload["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_json_atomic(output_path, payload)
        print(f"Completed {profile_name}: {output_path}")


if __name__ == "__main__":
    main()
