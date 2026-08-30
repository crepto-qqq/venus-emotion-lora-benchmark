"""Run one pinned official-prompt Venus baseline over the approved EmoSet-20 manifest."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
from typing import Any

from .crop import parse_crop_box
from .download_models import SOURCE_FILENAME
from .manifest import read_manifest, validate_manifest


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _run_text_command(command: list[str]) -> dict[str, Any]:
    try:
        result = subprocess.run(command, check=False, capture_output=True, text=True)
    except OSError as error:
        return {"command": command, "error": str(error)}
    return {
        "command": command,
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def _verify_model_source(model_path: Path, expected: dict[str, Any], stage: str) -> dict[str, Any]:
    source_path = model_path / SOURCE_FILENAME
    if not source_path.is_file():
        raise FileNotFoundError(
            f"Missing {SOURCE_FILENAME}. Download models with src.phase1.download_models first."
        )
    source = json.loads(source_path.read_text(encoding="utf-8"))
    for key in ("repo_id", "revision", "weight_format"):
        if source.get(key) != expected[key]:
            raise RuntimeError(
                f"Pinned {stage} {key} mismatch: expected {expected[key]!r}, found {source.get(key)!r}."
            )
    return source


def _assert_generation_config(actual: dict[str, Any], expected: dict[str, Any]) -> None:
    mismatches = {
        key: {"expected": expected_value, "actual": actual.get(key)}
        for key, expected_value in expected.items()
        if actual.get(key) != expected_value
    }
    if mismatches:
        raise RuntimeError(f"Generation configuration mismatch: {json.dumps(mismatches)}")


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("stage1", "stage2"), required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--emoset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/phase1/baseline.json"))
    args = parser.parse_args()

    with args.config.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    model_config = config["models"][args.stage]
    rows = read_manifest(args.manifest)
    manifest_errors = validate_manifest(
        rows,
        args.emoset_root,
        config["dataset"]["emotion_quotas"],
        config["dataset"]["valence"],
        expected_size=config["dataset"]["size"],
        require_approved=True,
        verify_hashes=True,
    )
    if manifest_errors:
        raise RuntimeError("Manifest validation failed:\n- " + "\n- ".join(manifest_errors))

    model_path = args.model_path.resolve()
    model_source = _verify_model_source(model_path, model_config, args.stage)
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty; refusing to overwrite: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "records.jsonl"

    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.generation import GenerationConfig

    if transformers.__version__ != config["generation"]["transformers_version"]:
        raise RuntimeError(
            f"transformers must be {config['generation']['transformers_version']}; "
            f"found {transformers.__version__}."
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Phase 1 formal inference requires a GPU.")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("The selected GPU does not report BF16 support.")

    environment = {
        "captured_at_utc": _utc_now(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu_count_visible": torch.cuda.device_count(),
        "gpu_0": torch.cuda.get_device_name(0),
        "nvidia_smi": _run_text_command(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ]
        ),
        "git_head": _run_text_command(["git", "rev-parse", "HEAD"]),
    }
    _write_json(output_dir / "environment.json", environment)
    pip_freeze = _run_text_command([sys.executable, "-m", "pip", "freeze"])
    _write_json(output_dir / "pip_freeze.json", pip_freeze)

    run_config: dict[str, Any] = {
        "schema_version": 1,
        "started_at_utc": _utc_now(),
        "stage": args.stage,
        "condition": config["condition"],
        "seed": config["seed"],
        "precision": config["precision"],
        "batch_size": config["batch_size"],
        "prompt": model_config["prompt"],
        "model_source": model_source,
        "model_path": str(model_path),
        "manifest": str(args.manifest.resolve()),
        "manifest_size": len(rows),
        "generation_expected": config["generation"],
        "model_loading": config["model_loading"],
    }
    _write_json(output_dir / "run_config.json", run_config)

    torch.manual_seed(int(config["seed"]))
    torch.cuda.set_device(0)
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        local_files_only=True,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        device_map=config["model_loading"]["device_map"],
        local_files_only=True,
        trust_remote_code=True,
        low_cpu_mem_usage=config["model_loading"]["low_cpu_mem_usage"],
        bf16=True,
    ).eval()
    generation_config = GenerationConfig.from_pretrained(
        str(model_path),
        local_files_only=True,
        trust_remote_code=True,
    )
    generation_actual = generation_config.to_dict()
    _assert_generation_config(generation_actual, config["generation"])
    model.generation_config = generation_config

    parameter_devices = sorted({str(parameter.device) for parameter in model.parameters()})
    if not parameter_devices or any(not device.startswith("cuda") for device in parameter_devices):
        raise RuntimeError(
            "CPU/disk model placement was detected, but Phase 1 forbids CPU offload: "
            + ", ".join(parameter_devices)
        )
    run_config["generation_actual"] = generation_actual
    run_config["parameter_devices"] = parameter_devices
    run_config["model_loaded_at_utc"] = _utc_now()
    _write_json(output_dir / "run_config.json", run_config)

    successful = 0
    parse_counts: dict[str, int] = {}
    for index, row in enumerate(rows, start=1):
        image_path = (args.emoset_root / row["image_relpath"]).resolve()
        query = tokenizer.from_list_format(
            [
                {"image": str(image_path)},
                {"text": model_config["prompt"]},
            ]
        )
        started_at = _utc_now()
        started_clock = time.perf_counter()
        torch.cuda.reset_peak_memory_stats(0)
        try:
            response, _ = model.chat(tokenizer, query=query, history=None)
            torch.cuda.synchronize(0)
        except Exception as error:
            failure = {
                "schema_version": 1,
                "stage": args.stage,
                "condition": config["condition"],
                "sequence_index": index,
                "sample_id": row["sample_id"],
                "source_image_id": row["source_image_id"],
                "status": "error",
                "started_at_utc": started_at,
                "finished_at_utc": _utc_now(),
                "error_type": type(error).__name__,
                "error_message": str(error),
                "traceback": traceback.format_exc(),
            }
            _append_jsonl(records_path, failure)
            raise

        crop_parse = parse_crop_box(response).to_dict() if args.stage == "stage2" else None
        if crop_parse is not None:
            parse_status = crop_parse["status"]
            parse_counts[parse_status] = parse_counts.get(parse_status, 0) + 1
        record = {
            "schema_version": 1,
            "stage": args.stage,
            "condition": config["condition"],
            "sequence_index": index,
            "sample_id": row["sample_id"],
            "source_image_id": row["source_image_id"],
            "image_relpath": row["image_relpath"],
            "emotion_label": row["emotion"],
            "valence": row["valence"],
            "status": "success",
            "prompt": model_config["prompt"],
            "response": response,
            "crop_parse": crop_parse,
            "started_at_utc": started_at,
            "finished_at_utc": _utc_now(),
            "elapsed_seconds": round(time.perf_counter() - started_clock, 4),
            "peak_gpu_memory_mb": round(torch.cuda.max_memory_allocated(0) / 1024**2, 2),
        }
        _append_jsonl(records_path, record)
        successful += 1
        print(f"[{index:02d}/{len(rows)}] {row['sample_id']} complete")

    summary = {
        "schema_version": 1,
        "completed": successful == len(rows),
        "stage": args.stage,
        "condition": config["condition"],
        "record_count": successful,
        "expected_count": len(rows),
        "crop_parse_counts": parse_counts if args.stage == "stage2" else None,
        "finished_at_utc": _utc_now(),
    }
    _write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
