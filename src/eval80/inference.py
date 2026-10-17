"""Run one frozen Venus Stage 1 condition over approved Eval80-v1."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
from typing import Any

from .core import (
    MODEL_SOURCE_FILENAME,
    sha256_file,
    sha256_text_file,
    validate_freeze_record,
    validate_inference_manifest,
)


# This is the checked-in, completed Member 3A run, not Member 3B acceptance.
FORMAL_C_EVIDENCE_SHA256 = "c2be197d0b79c5b979906433628c5b01c434935c4c992e19a7663839a8aeb8e0"
FORMAL_C_ATTEMPT_ID = "formal5-20261010-v1"
FORMAL_C_CHECKPOINT = 180
FORMAL_C_ADAPTER_FILES = {
    "adapter_config.json": "809c5b6e2a5334a031399cf50ef7b589a8d4ab46cbeafe9620eb57fe56990d34",
    "adapter_model.bin": "54363c1a0764d69e6389bfc2556c3f1c93c089da8eda57cd79733afe36f2b440",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()


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


def _verify_model_source(model_path: Path, expected: dict[str, Any]) -> dict[str, Any]:
    source_path = model_path / MODEL_SOURCE_FILENAME
    if not source_path.is_file():
        raise FileNotFoundError(
            f"Missing {MODEL_SOURCE_FILENAME}. Download the pinned Venus model first."
        )
    source = json.loads(source_path.read_text(encoding="utf-8"))
    for key in ("repo_id", "revision", "weight_format"):
        if source.get(key) != expected[key]:
            raise RuntimeError(
                f"Pinned Stage 1 {key} mismatch: expected {expected[key]!r}, "
                f"found {source.get(key)!r}."
            )
    return source


def _condition_prompt(config: dict[str, Any], condition: str) -> str:
    """Resolve C's prompt through the frozen B1 entry, never a second copy."""
    condition_config = config["conditions"][condition]
    source = condition_config.get("prompt_source_condition", condition)
    if condition == "C" and source != "B1":
        raise ValueError("Condition C must inherit the frozen B1 prompt.")
    prompt = config["conditions"][source].get("prompt")
    if not prompt:
        raise ValueError(f"Condition {condition} does not contain an executable prompt.")
    if condition == "C":
        b1 = config["conditions"]["B1"]
        if condition_config["prompt_version"] != b1["prompt_version"]:
            raise ValueError("Condition C and B1 prompt versions differ.")
        if condition_config["response_format"] != b1["response_format"]:
            raise ValueError("Condition C and B1 response formats differ.")
    return str(prompt)


def _expected_adapter_identity(
    evidence_path: Path, expected_model: dict[str, Any]
) -> dict[str, Any]:
    """Bind an adapter to the completed training evidence, without claiming acceptance."""
    evidence_sha256 = sha256_file(evidence_path)
    if evidence_sha256 != FORMAL_C_EVIDENCE_SHA256:
        raise ValueError("Adapter evidence SHA-256 differs from pinned formal attempt.")
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    if (evidence.get("kind"), evidence.get("attempt_id")) != (
        "member3a_formal_results_summary", FORMAL_C_ATTEMPT_ID
    ):
        raise ValueError("Adapter evidence does not identify the pinned formal attempt.")
    if evidence.get("training_completed") is not True:
        raise ValueError("Adapter evidence does not record completed training.")
    if evidence.get("base_revision") != expected_model["revision"]:
        raise ValueError("Adapter evidence base revision differs from frozen Eval80 base.")
    selected = evidence.get("selected_checkpoint")
    if selected != FORMAL_C_CHECKPOINT:
        raise ValueError("Adapter evidence selected checkpoint is not formal step 180.")
    candidates = [
        candidate for candidate in evidence.get("candidates", [])
        if candidate.get("step") == selected
    ]
    if len(candidates) != 1:
        raise ValueError("Adapter evidence must identify exactly one selected checkpoint.")
    files = candidates[0].get("adapter_files")
    if files != FORMAL_C_ADAPTER_FILES:
        raise ValueError("Selected adapter hashes differ from formal checkpoint 180.")
    return {
        "evidence_path": str(evidence_path),
        "evidence_sha256": evidence_sha256,
        "selected_checkpoint": selected,
        "formal_config_sha256": evidence.get("config_sha256"),
        "base_revision": evidence["base_revision"],
        "adapter_files": files,
        "acceptance_status": "not_verified",
    }


def _verify_adapter_files(adapter_path: Path, identity: dict[str, Any]) -> None:
    for filename, expected_sha in identity["adapter_files"].items():
        file_path = adapter_path / filename
        if not file_path.is_file():
            raise FileNotFoundError(f"Selected adapter file is missing: {filename}")
        if sha256_file(file_path) != expected_sha:
            raise ValueError(f"Selected adapter SHA-256 mismatch: {filename}")
    if (adapter_path / "adapter_model.safetensors").exists():
        raise ValueError("Unexpected second adapter weight file; refusing ambiguous load.")


def _verify_base_snapshot(model_path: Path, project_root: Path,
                          expected_model: dict[str, Any]) -> dict[str, Any]:
    """Hash the complete pinned Venus snapshot before loading Condition C."""
    lock_path = project_root / "phase3/configs/source-lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))["model"]
    if (lock.get("repository"), lock.get("revision"), lock.get("weight_format")) != (
        expected_model["repo_id"], expected_model["revision"], expected_model["weight_format"]
    ):
        raise ValueError("Phase 3 model source lock differs from frozen Eval80 base.")
    manifest_path = project_root / lock["snapshot_manifest"]
    manifest_hash = sha256_file(manifest_path)
    if manifest_hash != lock["snapshot_manifest_sha256"]:
        raise ValueError("Pinned model snapshot manifest SHA-256 mismatch.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("repository"), manifest.get("revision")) != (
        expected_model["repo_id"], expected_model["revision"]
    ):
        raise ValueError("Pinned model snapshot manifest names a different base.")
    entries = manifest.get("files")
    if not isinstance(entries, list) or len(entries) != manifest.get("file_count"):
        raise ValueError("Pinned model snapshot manifest file count is invalid.")
    total_bytes = 0
    seen: set[str] = set()
    for entry in entries:
        filename = entry.get("path")
        if not isinstance(filename, str) or Path(filename).name != filename or filename in seen:
            raise ValueError("Pinned model snapshot manifest has an invalid filename.")
        seen.add(filename)
        file_path = model_path / filename
        if file_path.is_symlink() or not file_path.is_file():
            raise FileNotFoundError(f"Pinned model snapshot file is missing: {filename}")
        size = file_path.stat().st_size
        if size != entry.get("size") or sha256_file(file_path) != entry.get("sha256"):
            raise ValueError(f"Pinned model snapshot file differs: {filename}")
        total_bytes += size
    if total_bytes != manifest.get("total_bytes"):
        raise ValueError("Pinned model snapshot total bytes differ.")
    return {"source_lock_sha256": sha256_file(lock_path),
            "snapshot_manifest_sha256": manifest_hash,
            "verified_file_count": len(entries), "verified_total_bytes": total_bytes}


def _load_adapter_preserving_rng(
    base_model: Any,
    adapter_path: Path,
    generation_config: Any,
    *,
    torch_module: Any,
    peft_model_class: Any,
) -> Any:
    """Keep C's sampled stream at the state reached after the shared base load."""
    cpu_state = torch_module.get_rng_state().clone()
    cuda_states = [state.clone() for state in torch_module.cuda.get_rng_state_all()]
    try:
        model = peft_model_class.from_pretrained(
            base_model, str(adapter_path), is_trainable=False
        ).eval()
        model.generation_config = generation_config
    finally:
        torch_module.set_rng_state(cpu_state)
        torch_module.cuda.set_rng_state_all(cuda_states)
    return model


def _assert_generation_config(actual: dict[str, Any], expected: dict[str, Any]) -> None:
    mismatches = {
        key: {"expected": expected_value, "actual": actual.get(key)}
        for key, expected_value in expected.items()
        if actual.get(key) != expected_value
    }
    if mismatches:
        raise RuntimeError(f"Generation configuration mismatch: {json.dumps(mismatches)}")


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def run_command(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=("A", "B0", "B1", "C"), required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--adapter-path", type=Path)
    parser.add_argument("--adapter-evidence", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--freeze-record", type=Path, required=True)
    parser.add_argument("--handbook", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/phase1/eval80.json"),
    )
    args = parser.parse_args(argv)

    config_path = args.config.resolve(strict=True)
    handbook_path = args.handbook.resolve(strict=True)
    manifest_path = args.manifest.resolve(strict=True)
    freeze_record_path = args.freeze_record.resolve(strict=True)
    image_dir = args.image_dir.resolve(strict=True)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    condition_config = config["conditions"][args.condition]
    prompt = _condition_prompt(config, args.condition)
    if args.condition == "C":
        if args.adapter_path is None or args.adapter_evidence is None:
            raise ValueError("Condition C requires --adapter-path and --adapter-evidence.")
    elif args.adapter_path is not None or args.adapter_evidence is not None:
        raise ValueError("Adapter arguments are only valid for Condition C.")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_errors = validate_inference_manifest(manifest, image_dir, verify_hashes=True)
    if manifest_errors:
        raise RuntimeError("Inference manifest validation failed:\n- " + "\n- ".join(manifest_errors))
    if len(manifest["records"]) != int(config["dataset"]["size"]):
        raise RuntimeError("Inference manifest size differs from the frozen config.")

    freeze_record = json.loads(freeze_record_path.read_text(encoding="utf-8"))
    freeze_errors = validate_freeze_record(
        freeze_record,
        config_path=config_path,
        handbook_path=handbook_path,
        inference_manifest_path=manifest_path,
    )
    if freeze_errors:
        raise RuntimeError("Freeze-record validation failed:\n- " + "\n- ".join(freeze_errors))

    model_path = args.model_path.resolve(strict=True)
    model_source = _verify_model_source(model_path, config["model"])
    base_snapshot_verification: dict[str, Any] | None = None
    adapter_path: Path | None = None
    adapter_identity: dict[str, Any] | None = None
    if args.condition == "C":
        base_snapshot_verification = _verify_base_snapshot(
            model_path, config_path.parents[2], config["model"]
        )
        adapter_path = args.adapter_path.resolve(strict=True)
        evidence_path = args.adapter_evidence.resolve(strict=True)
        adapter_identity = _expected_adapter_identity(evidence_path, config["model"])
        _verify_adapter_files(adapter_path, adapter_identity)
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty; refusing to overwrite: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "records.jsonl"

    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.generation import GenerationConfig
    if args.condition == "C":
        import peft
        from peft import PeftModel

    if transformers.__version__ != config["generation"]["transformers_version"]:
        raise RuntimeError(
            f"transformers must be {config['generation']['transformers_version']}; "
            f"found {transformers.__version__}."
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Formal Eval80 inference requires a GPU.")
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
    if args.condition == "C":
        environment["peft"] = peft.__version__
    _write_json_atomic(output_dir / "environment.json", environment)
    _write_json_atomic(
        output_dir / "pip_freeze.json",
        _run_text_command([sys.executable, "-m", "pip", "freeze"]),
    )

    run_config: dict[str, Any] = {
        "schema_version": 1,
        "protocol_version": config["protocol_version"],
        "started_at_utc": _utc_now(),
        "stage": "stage1",
        "condition": args.condition,
        "condition_name": condition_config["name"],
        "prompt_version": condition_config["prompt_version"],
        "prompt": prompt,
        "seed": config["seed"],
        "precision": config["precision"],
        "batch_size": config["batch_size"],
        "model_source": model_source,
        "model_path": str(model_path),
        "manifest": str(manifest_path),
        "manifest_size": len(manifest["records"]),
        "freeze_record": str(freeze_record_path),
        "file_sha256": {
            "config": sha256_text_file(config_path),
            "scoring_handbook": sha256_text_file(handbook_path),
            "inference_manifest": sha256_file(manifest_path),
            "freeze_record": sha256_file(freeze_record_path),
        },
        "generation_expected": config["generation"],
        "model_loading": config["model_loading"],
    }
    if args.condition == "C":
        run_config["prompt_source_condition"] = "B1"
        run_config["adapter_identity"] = adapter_identity
        run_config["base_snapshot_verification"] = base_snapshot_verification
        run_config["rng_seed_position"] = "before_shared_base_load"
    _write_json_atomic(output_dir / "run_config.json", run_config)

    torch.manual_seed(int(config["seed"]))
    torch.cuda.set_device(0)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(0)
    load_started = time.perf_counter()
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
    if args.condition == "C":
        model = _load_adapter_preserving_rng(
            model, adapter_path, generation_config,
            torch_module=torch, peft_model_class=PeftModel,
        )
        run_config["adapter_load_rng_restored"] = True
        run_config["adapter_load_rng_scope"] = "cpu_and_all_visible_cuda_devices"
    torch.cuda.synchronize(0)

    parameter_devices = sorted({str(parameter.device) for parameter in model.parameters()})
    if not parameter_devices or any(not device.startswith("cuda") for device in parameter_devices):
        raise RuntimeError(
            "CPU/disk model placement was detected, but the formal run forbids offload: "
            + ", ".join(parameter_devices)
        )
    run_config["generation_actual"] = generation_actual
    run_config["parameter_devices"] = parameter_devices
    run_config["model_loaded_at_utc"] = _utc_now()
    run_config["model_load_seconds"] = round(time.perf_counter() - load_started, 4)
    run_config["model_load_peak_gpu_memory_mb"] = round(
        torch.cuda.max_memory_allocated(0) / 1024**2,
        2,
    )
    _write_json_atomic(output_dir / "run_config.json", run_config)
    print(
        f"[EVIDENCE MILESTONE] model loaded | condition={args.condition} | "
        f"gpu={environment['gpu_0']} | precision={config['precision']}"
    )

    milestones = set(int(value) for value in config["evidence"]["progress_milestones"])
    elapsed_values: list[float] = []
    peak_memory_values: list[float] = []
    run_started_clock = time.perf_counter()
    for index, record in enumerate(manifest["records"], start=1):
        image_path = image_dir / record["image_filename"]
        query = tokenizer.from_list_format(
            [
                {"image": str(image_path)},
                {"text": prompt},
            ]
        )
        started_at = _utc_now()
        started_clock = time.perf_counter()
        torch.cuda.reset_peak_memory_stats(0)
        try:
            with torch.inference_mode():
                response, _ = model.chat(tokenizer, query=query, history=None)
            torch.cuda.synchronize(0)
        except Exception as error:
            failure = {
                "schema_version": 1,
                "stage": "stage1",
                "condition": args.condition,
                "sequence_index": index,
                "blind_id": record["blind_id"],
                "image_sha256": record["image_sha256"],
                "status": "error",
                "started_at_utc": started_at,
                "finished_at_utc": _utc_now(),
                "error_type": type(error).__name__,
                "error_message": str(error),
                "traceback": traceback.format_exc(),
            }
            _append_jsonl(records_path, failure)
            _write_json_atomic(output_dir / "failure.json", failure)
            raise

        elapsed = round(time.perf_counter() - started_clock, 4)
        peak_memory = round(torch.cuda.max_memory_allocated(0) / 1024**2, 2)
        response_text = str(response)
        output_record = {
            "schema_version": 1,
            "stage": "stage1",
            "condition": args.condition,
            "prompt_version": condition_config["prompt_version"],
            "sequence_index": index,
            "blind_id": record["blind_id"],
            "image_sha256": record["image_sha256"],
            "status": "success",
            "response": response_text,
            "response_sha256": hashlib.sha256(response_text.encode("utf-8")).hexdigest(),
            "started_at_utc": started_at,
            "finished_at_utc": _utc_now(),
            "elapsed_seconds": elapsed,
            "peak_gpu_memory_mb": peak_memory,
        }
        _append_jsonl(records_path, output_record)
        elapsed_values.append(elapsed)
        peak_memory_values.append(peak_memory)
        print(f"[{index:02d}/80] {record['blind_id']} complete")
        if index in milestones:
            print(
                f"[EVIDENCE MILESTONE] {index}/80 complete | "
                f"last_seconds={elapsed:.4f} | last_peak_gpu_memory_mb={peak_memory:.2f}"
            )

    summary = {
        "schema_version": 1,
        "completed": True,
        "stage": "stage1",
        "condition": args.condition,
        "record_count": len(elapsed_values),
        "expected_count": 80,
        "empty_response_count": 0,
        "total_run_seconds": round(time.perf_counter() - run_started_clock, 4),
        "mean_inference_seconds": _mean(elapsed_values),
        "maximum_inference_seconds": max(elapsed_values),
        "mean_peak_gpu_memory_mb": _mean(peak_memory_values),
        "maximum_peak_gpu_memory_mb": max(peak_memory_values),
        "finished_at_utc": _utc_now(),
    }
    _write_json_atomic(output_dir / "summary.json", summary)
    summary["records_sha256"] = sha256_file(records_path)
    _write_json_atomic(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(run_command())
