#!/usr/bin/env python3
"""Run the one-example, no-optimizer Phase 3 LoRA backward smoke test.

This command deliberately stops after ``loss.backward()``.  It never creates
an optimizer, performs a parameter update, saves an adapter, or runs
generation.  The success report conforms to ``smoke-report.schema.json``.
When setup fails before all measurements are available, the command still
writes an honest machine-readable diagnostic report where possible.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import platform
import re
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Sequence

from common import (
    assert_report_is_redacted,
    package_versions,
    read_json,
    sha256_file,
    utc_now,
    verify_exact_git_patch,
    verify_snapshot_manifest,
    write_json_atomic,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "phase3" / "configs" / "member1-smoke.json"
DEFAULT_SOURCE_LOCK = PROJECT_ROOT / "phase3" / "configs" / "source-lock.json"
DEFAULT_REPORT_SCHEMA = (
    PROJECT_ROOT / "phase3" / "contracts" / "smoke-report.schema.json"
)

EXPECTED_FIXTURE_ID = "contentment_05000"
EXPECTED_DTYPE = "bfloat16"
EXPECTED_LORA_RANK = 2
EXPECTED_LORA_ALPHA = 4
EXPECTED_LORA_DROPOUT = 0.0
EXPECTED_TARGET_MODULES = ("c_attn", "attn.c_proj", "w1", "w2")
IGNORE_TOKEN_ID = -100

_IMAGE_PATH_RE = re.compile(r"<img>([^<>]+)</img>", re.DOTALL)
_URL_RE = re.compile(r"https?://[^\s]+", re.IGNORECASE)
_HF_TOKEN_RE = re.compile(r"\bhf_[A-Za-z0-9]{10,}\b")
_BEARER_RE = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]+")
_SIGNED_QUERY_RE = re.compile(
    r"(?i)(?:x-amz-(?:credential|signature|security-token)|signature|sig|token)="
    r"[^&\s]+"
)


class SmokeFailure(RuntimeError):
    """A clear, expected acceptance failure."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run one real BF16 Venus-Q-Stage1 forward/backward pass with the "
            "fixed technical LoRA settings, without an optimizer or save."
        )
    )
    parser.add_argument(
        "--fixture",
        "--fixture-path",
        dest="fixture",
        type=Path,
        required=True,
        help="One-record contentment_05000 fixture JSON produced by the builder.",
    )
    parser.add_argument(
        "--model-dir",
        "--model-path",
        dest="model_dir",
        type=Path,
        required=True,
        help="Read-only local Venus-Q-Stage1 snapshot directory.",
    )
    parser.add_argument(
        "--upstream-dir",
        "--upstream-path",
        dest="upstream_dir",
        type=Path,
        required=True,
        help="Pinned and patched Qwen-VL-finetune checkout.",
    )
    parser.add_argument(
        "--report",
        "--report-path",
        "--output",
        dest="report",
        type=Path,
        required=True,
        help="Destination for the machine-readable smoke report.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Technical smoke configuration (default: {DEFAULT_CONFIG}).",
    )
    parser.add_argument(
        "--source-lock",
        type=Path,
        default=DEFAULT_SOURCE_LOCK,
        help=f"Pinned source/model metadata (default: {DEFAULT_SOURCE_LOCK}).",
    )
    parser.add_argument(
        "--report-schema",
        type=Path,
        default=DEFAULT_REPORT_SCHEMA,
        help=f"Success report JSON Schema (default: {DEFAULT_REPORT_SCHEMA}).",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Optional shared Hugging Face cache directory.",
    )
    parser.add_argument(
        "--member-id",
        default="member1",
        help="Non-secret member identifier stored in the report (default: member1).",
    )
    parser.add_argument(
        "--device-index",
        type=int,
        default=0,
        help="CUDA device index for this single-GPU smoke test (default: 0).",
    )
    return parser


def _safe_error(exc: BaseException) -> str:
    """Return a short diagnostic while removing common credential forms."""
    message = " ".join(str(exc).split()) or "no error message was provided"
    message = _HF_TOKEN_RE.sub("<redacted-token>", message)
    message = _BEARER_RE.sub("Bearer <redacted>", message)
    message = _SIGNED_QUERY_RE.sub("<redacted-query>", message)
    # Error messages can contain presigned download URLs.  URLs are not needed
    # to diagnose this local-only acceptance command, so remove all of them.
    message = _URL_RE.sub("<redacted-url>", message)
    return f"{type(exc).__name__}: {message}"[:2000]


def _empty_report(member_id: str, source_lock: dict[str, Any] | None) -> dict[str, Any]:
    model_lock = (source_lock or {}).get("model", {})
    return {
        "schema_version": 1,
        "kind": "phase3_smoke_backward",
        "generated_at_utc": utc_now(),
        "member_id": member_id,
        "technical_infrastructure_only": True,
        "passed": False,
        "forward_passed": False,
        "backward_passed": False,
        "optimizer_step_performed": False,
        "adapter_saved": False,
        "fixture": {
            "id": EXPECTED_FIXTURE_ID,
            "fixture_sha256": None,
            "image_sha256": None,
            "dataset_record_sha256": None,
        },
        "model": {
            "repository": str(model_lock.get("repository", "unknown")),
            "revision": str(model_lock.get("revision", "unknown")),
            "dtype": EXPECTED_DTYPE,
            "device": "unavailable",
        },
        "batch": {
            "batch_size": 1,
            "token_count": 0,
            "supervised_token_count": 0,
            "image_token_count": 0,
            "loss": None,
        },
        "lora": {
            "rank": EXPECTED_LORA_RANK,
            "alpha": EXPECTED_LORA_ALPHA,
            "target_modules": list(EXPECTED_TARGET_MODULES),
            "matched_target_names": [],
            "total_parameter_count": 0,
            "trainable_parameter_count": 0,
            "trainable_parameter_names": [],
            "nonzero_gradient_parameter_names": [],
        },
        "memory": {
            "peak_allocated_bytes": 0,
            "peak_reserved_bytes": 0,
        },
        "versions": {
            "python": platform.python_version(),
        },
        "errors": [],
    }


def _require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SmokeFailure(f"{label} must be a JSON object")
    return value


def _load_and_validate_inputs(
    config_path: Path, source_lock_path: Path, fixture_path: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], Path]:
    config = _require_mapping(read_json(config_path), "smoke config")
    source_lock = _require_mapping(read_json(source_lock_path), "source lock")

    if config.get("purpose") != "technical_infrastructure_only":
        raise SmokeFailure("smoke config purpose is not technical_infrastructure_only")

    runtime = _require_mapping(config.get("runtime"), "smoke config runtime")
    lora = _require_mapping(config.get("lora"), "smoke config lora")
    policy = _require_mapping(config.get("policy"), "smoke config policy")
    if runtime.get("dtype") != EXPECTED_DTYPE:
        raise SmokeFailure(f"runtime dtype must be {EXPECTED_DTYPE}")
    if int(runtime.get("max_sequence_length", 0)) < 1:
        raise SmokeFailure("max_sequence_length must be positive")
    if int(lora.get("rank", 0)) != EXPECTED_LORA_RANK:
        raise SmokeFailure(f"technical LoRA rank must be {EXPECTED_LORA_RANK}")
    if int(lora.get("alpha", 0)) != EXPECTED_LORA_ALPHA:
        raise SmokeFailure(f"technical LoRA alpha must be {EXPECTED_LORA_ALPHA}")
    if float(lora.get("dropout", -1.0)) != EXPECTED_LORA_DROPOUT:
        raise SmokeFailure("technical LoRA dropout must be 0")
    if lora.get("bias") != "none":
        raise SmokeFailure("technical LoRA bias must be none")
    if tuple(lora.get("target_modules", [])) != EXPECTED_TARGET_MODULES:
        raise SmokeFailure(
            "technical target modules must be c_attn, attn.c_proj, w1, and w2"
        )
    for forbidden_true in (
        "optimizer_step_performed",
        "adapter_saved",
        "full_dataset_ready",
        "formal_training_authorized",
    ):
        if policy.get(forbidden_true) is not False:
            raise SmokeFailure(f"policy {forbidden_true} must remain false")

    fixture_raw = read_json(fixture_path)
    if not isinstance(fixture_raw, list) or len(fixture_raw) != 1:
        raise SmokeFailure("fixture must be a JSON array containing exactly one record")
    fixture = _require_mapping(fixture_raw[0], "fixture record")
    if fixture.get("id") != EXPECTED_FIXTURE_ID:
        raise SmokeFailure(f"fixture id must be {EXPECTED_FIXTURE_ID}")
    if fixture.get("technical_infrastructure_only") is not True:
        raise SmokeFailure("fixture must be marked technical_infrastructure_only")
    conversations = fixture.get("conversations")
    if not isinstance(conversations, list) or len(conversations) != 2:
        raise SmokeFailure("fixture must contain one user and one assistant turn")
    if conversations[0].get("from") != "user":
        raise SmokeFailure("fixture first turn must be from user")
    if conversations[1].get("from") != "assistant":
        raise SmokeFailure("fixture second turn must be from assistant")
    if not str(conversations[1].get("value", "")).strip():
        raise SmokeFailure("fixture assistant target must not be empty")

    image_match = _IMAGE_PATH_RE.search(str(conversations[0].get("value", "")))
    if image_match is None:
        raise SmokeFailure("fixture user turn must contain one <img>path</img> reference")
    image_path = Path(image_match.group(1).strip()).expanduser()
    if not image_path.is_file():
        raise SmokeFailure("the image referenced by the technical fixture is unavailable")

    configured_fixture = _require_mapping(config.get("fixture"), "smoke config fixture")
    provenance = _require_mapping(fixture.get("provenance"), "fixture provenance")
    if configured_fixture.get("source_image_id") != EXPECTED_FIXTURE_ID:
        raise SmokeFailure("smoke config does not pin contentment_05000")
    for field in ("record_sha256", "annotation_sha256", "image_sha256"):
        provenance_field = (
            "dataset_record_sha256" if field == "record_sha256" else field
        )
        if provenance.get(provenance_field) != configured_fixture.get(field):
            raise SmokeFailure(f"fixture provenance does not match configured {field}")
    if sha256_file(image_path) != configured_fixture.get("image_sha256"):
        raise SmokeFailure("technical fixture image SHA-256 does not match the pinned value")

    return config, source_lock, fixture, image_path


def _verify_upstream(upstream_dir: Path, source_lock: dict[str, Any]) -> Path:
    upstream_lock = _require_mapping(source_lock.get("upstream"), "upstream source lock")
    expected_commit = str(upstream_lock.get("commit", ""))
    if not expected_commit:
        raise SmokeFailure("upstream source lock has no commit")
    finetune_path = upstream_dir / "finetune.py"
    if not finetune_path.is_file():
        raise SmokeFailure("upstream checkout has no finetune.py")

    completed = subprocess.run(
        ["git", "-C", str(upstream_dir), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    actual_commit = completed.stdout.strip()
    if completed.returncode != 0 or actual_commit != expected_commit:
        raise SmokeFailure(
            f"upstream commit mismatch (expected {expected_commit}, got "
            f"{actual_commit or 'unavailable'})"
        )

    patch_value = str(upstream_lock.get("patch", ""))
    patch_path = Path(patch_value)
    if not patch_path.is_absolute():
        patch_path = PROJECT_ROOT / patch_path
    if not patch_path.is_file():
        raise SmokeFailure("the pinned Phase 3 upstream patch is unavailable")
    expected_patch_sha = str(upstream_lock.get("patch_sha256", ""))
    if not expected_patch_sha or sha256_file(patch_path) != expected_patch_sha:
        raise SmokeFailure("the Phase 3 upstream patch SHA-256 does not match source-lock.json")
    try:
        verify_exact_git_patch(upstream_dir, patch_path)
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        raise SmokeFailure(
            "the upstream working-tree diff does not exactly match the reviewed Phase 3 patch"
        ) from exc
    return finetune_path


def _load_upstream_finetune(finetune_path: Path, upstream_dir: Path) -> ModuleType:
    module_name = "_phase3_pinned_qwen_finetune"
    spec = importlib.util.spec_from_file_location(module_name, finetune_path)
    if spec is None or spec.loader is None:
        raise SmokeFailure("could not construct an import spec for upstream finetune.py")
    module = importlib.util.module_from_spec(spec)
    previous = list(sys.path)
    try:
        sys.path.insert(0, str(upstream_dir))
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = previous
    if not callable(getattr(module, "preprocess", None)):
        raise SmokeFailure("pinned upstream finetune.py exposes no preprocess function")
    return module


def _module_matches(name: str, target: str) -> bool:
    return name == target or name.endswith(f".{target}")


def _matched_modules(model: Any, targets: Sequence[str]) -> list[str]:
    names = [name for name, _ in model.named_modules() if name]
    missing = [
        target for target in targets if not any(_module_matches(name, target) for name in names)
    ]
    if missing:
        raise SmokeFailure(f"LoRA target modules were not found: {', '.join(missing)}")
    return sorted(
        name for name in names if any(_module_matches(name, target) for target in targets)
    )


def _check_target_dtypes(model: Any, matched_names: Sequence[str], torch: Any) -> None:
    modules = dict(model.named_modules())
    non_bf16: list[str] = []
    for name in matched_names:
        weight = getattr(modules[name], "weight", None)
        if weight is not None and getattr(weight, "is_floating_point", lambda: False)():
            if weight.dtype != torch.bfloat16:
                non_bf16.append(name)
    if non_bf16:
        sample = ", ".join(non_bf16[:5])
        raise SmokeFailure(f"LoRA base target weights are not BF16: {sample}")


def _prepare_batch(
    fixture: dict[str, Any], tokenizer: Any, preprocess: Any, max_length: int, torch: Any
) -> tuple[dict[str, Any], int, int, int]:
    prepared = preprocess(
        [fixture["conversations"]], tokenizer=tokenizer, max_len=max_length
    )
    required = ("input_ids", "labels", "attention_mask")
    if not all(key in prepared for key in required):
        raise SmokeFailure("upstream preprocess did not return the expected batch fields")

    input_ids = prepared["input_ids"].to(dtype=torch.long)
    labels = prepared["labels"].to(dtype=torch.long)
    attention_mask = prepared["attention_mask"].to(dtype=torch.bool)
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise SmokeFailure("technical fixture preprocessing did not produce batch size one")

    used_positions = attention_mask[0].nonzero(as_tuple=False)
    if used_positions.numel() == 0:
        raise SmokeFailure("technical fixture produced no input tokens")
    used_length = int(used_positions[-1].item()) + 1
    # Upstream pads every item to model_max_length.  Removing only trailing
    # padding keeps the exact token sequence and labels while avoiding a large,
    # meaningless attention allocation during this one-record infrastructure test.
    input_ids = input_ids[:, :used_length].contiguous()
    labels = labels[:, :used_length].contiguous()
    attention_mask = attention_mask[:, :used_length].contiguous()

    token_count = int(attention_mask.sum().item())
    supervised_token_count = int(labels.ne(IGNORE_TOKEN_ID).sum().item())
    if supervised_token_count < 1:
        raise SmokeFailure(
            "assistant target was truncated; no supervised token remains in the fixture"
        )

    image_ids = {
        attribute: getattr(tokenizer, attribute, None)
        for attribute in ("img_start_id", "img_end_id", "img_pad_id")
    }
    if len({value for value in image_ids.values() if isinstance(value, int)}) != 3:
        raise SmokeFailure("Qwen-VL tokenizer does not expose its three image token ids")
    start_positions = input_ids[0].eq(image_ids["img_start_id"]).nonzero(as_tuple=False)
    end_positions = input_ids[0].eq(image_ids["img_end_id"]).nonzero(as_tuple=False)
    if start_positions.shape[0] != 1 or end_positions.shape[0] != 1:
        raise SmokeFailure("technical fixture must encode exactly one image span")
    image_start = int(start_positions[0].item())
    image_end = int(end_positions[0].item())
    if image_end <= image_start:
        raise SmokeFailure("technical fixture image token span is malformed")
    # Qwen-VL stores the image reference in a fixed 256-token inner span,
    # padded with img_pad_id as needed, plus one start and one end token.
    image_token_count = image_end - image_start + 1
    if image_token_count < 258:
        raise SmokeFailure(
            f"fixture produced only {image_token_count} image tokens; expected at least 258"
        )

    return (
        {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        },
        token_count,
        supervised_token_count,
        image_token_count,
    )


def _runtime_versions(torch: Any | None = None) -> dict[str, str]:
    versions = {"python": platform.python_version()}
    versions.update(
        package_versions(
            (
                "torch",
                "torchvision",
                "transformers",
                "tokenizers",
                "accelerate",
                "peft",
                "deepspeed",
            )
        )
    )
    if torch is not None:
        versions["cuda_runtime"] = str(torch.version.cuda or "unavailable")
        try:
            versions["cudnn"] = str(torch.backends.cudnn.version() or "unavailable")
        except Exception:
            versions["cudnn"] = "unavailable"
    return versions


def _load_local_tokenizer(
    transformers: Any, model_dir: Path, cache_dir: str | None, max_length: int
) -> Any:
    """Load Qwen-VL locally without its import-time network-only font fetch.

    The pinned Venus snapshot intentionally contains no ``SimSun.ttf``.  The
    historical Qwen-VL tokenizer tries to download that optional drawing font
    at module import even though training never uses its box visualizer.  Point
    that one lookup at Matplotlib's installed DejaVu Sans, then restore the
    Transformers helper immediately.  Model and tokenizer files still resolve
    only from ``model_dir`` because ``local_files_only`` remains enabled.
    """
    from matplotlib.font_manager import findfont

    fallback_font = Path(findfont("DejaVu Sans", fallback_to_default=True)).resolve()
    if not fallback_font.is_file():
        raise SmokeFailure("could not locate the installed fallback font for Qwen tokenizer")

    original_lookup = transformers.utils.try_to_load_from_cache

    def local_font_lookup(repo_id: str, filename: str, *args: Any, **kwargs: Any) -> Any:
        if repo_id == "Qwen/Qwen-VL-Chat" and filename == "SimSun.ttf":
            return str(fallback_font)
        return original_lookup(repo_id, filename, *args, **kwargs)

    transformers.utils.try_to_load_from_cache = local_font_lookup
    try:
        return transformers.AutoTokenizer.from_pretrained(
            str(model_dir),
            cache_dir=cache_dir,
            model_max_length=max_length,
            padding_side="right",
            use_fast=False,
            trust_remote_code=True,
            local_files_only=True,
        )
    finally:
        transformers.utils.try_to_load_from_cache = original_lookup


def _validate_success_report(report: dict[str, Any], schema_path: Path) -> None:
    try:
        import jsonschema
    except ImportError as exc:
        raise SmokeFailure("jsonschema is required to validate the smoke report") from exc
    schema = read_json(schema_path)
    try:
        jsonschema.Draft202012Validator(schema).validate(report)
    except jsonschema.ValidationError as exc:
        location = ".".join(str(part) for part in exc.absolute_path) or "$"
        raise SmokeFailure(f"success report failed schema validation at {location}: {exc.message}")


def _run(args: argparse.Namespace, report: dict[str, Any]) -> None:
    config, source_lock, fixture, image_path = _load_and_validate_inputs(
        args.config, args.source_lock, args.fixture
    )
    report["fixture"].update(
        {
            "fixture_sha256": sha256_file(args.fixture),
            "image_sha256": sha256_file(image_path),
            "dataset_record_sha256": fixture["provenance"][
                "dataset_record_sha256"
            ],
        }
    )
    model_lock = _require_mapping(source_lock.get("model"), "model source lock")
    report["model"]["repository"] = str(model_lock.get("repository", "unknown"))
    report["model"]["revision"] = str(model_lock.get("revision", "unknown"))

    finetune_path = _verify_upstream(args.upstream_dir, source_lock)

    try:
        import torch
        import transformers
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:
        raise SmokeFailure(f"required training dependency is unavailable: {exc.name}") from exc

    report["versions"] = _runtime_versions(torch)
    if not torch.cuda.is_available():
        raise SmokeFailure("CUDA is unavailable; run this command on the 48 GB GPU Pod")
    if args.device_index < 0 or args.device_index >= torch.cuda.device_count():
        raise SmokeFailure(f"CUDA device index {args.device_index} is unavailable")
    torch.cuda.set_device(args.device_index)
    device = torch.device("cuda", args.device_index)
    properties = torch.cuda.get_device_properties(device)
    report["model"]["device"] = f"cuda:{args.device_index} ({properties.name})"
    minimum_memory = int(config["runtime"]["minimum_gpu_memory_bytes"])
    if int(properties.total_memory) < minimum_memory:
        raise SmokeFailure(
            f"GPU memory is {properties.total_memory} bytes; at least {minimum_memory} is required"
        )
    is_bf16_supported = getattr(torch.cuda, "is_bf16_supported", lambda: False)
    if not bool(is_bf16_supported()):
        raise SmokeFailure("the selected CUDA device does not support BF16")

    if not args.model_dir.is_dir():
        raise SmokeFailure("the local Venus-Q-Stage1 model directory is unavailable")
    manifest_value = str(model_lock.get("snapshot_manifest", ""))
    manifest_path = Path(manifest_value)
    if not manifest_path.is_absolute():
        manifest_path = PROJECT_ROOT / manifest_path
    try:
        verify_snapshot_manifest(
            args.model_dir,
            manifest_path,
            expected_manifest_sha256=str(
                model_lock.get("snapshot_manifest_sha256", "")
            ),
            expected_repository=str(model_lock.get("repository", "")),
            expected_revision=str(model_lock.get("revision", "")),
        )
    except (OSError, ValueError) as exc:
        raise SmokeFailure(
            "the local model snapshot does not match every file in the pinned manifest"
        ) from exc
    index_name = str(model_lock.get("index_file", "model.safetensors.index.json"))
    if not (args.model_dir / index_name).is_file():
        raise SmokeFailure("the local model snapshot has no pinned safetensors index")
    shard_pattern = str(
        model_lock.get("weight_shard_pattern", "model-*-of-00010.safetensors")
    )
    shards = sorted(args.model_dir.glob(shard_pattern))
    expected_shard_count = int(model_lock.get("weight_shard_count", 0))
    expected_shard_bytes = int(model_lock.get("weight_shard_total_bytes", 0))
    if len(shards) != expected_shard_count:
        raise SmokeFailure(
            f"model snapshot has {len(shards)} weight shards; "
            f"source lock requires {expected_shard_count}"
        )
    actual_shard_bytes = sum(shard.stat().st_size for shard in shards)
    if actual_shard_bytes != expected_shard_bytes:
        raise SmokeFailure(
            f"model shard size is {actual_shard_bytes} bytes; "
            f"source lock requires {expected_shard_bytes}"
        )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    upstream = _load_upstream_finetune(finetune_path, args.upstream_dir)
    max_length = int(config["runtime"]["max_sequence_length"])
    cache_dir = str(args.cache_dir) if args.cache_dir is not None else None

    tokenizer = _load_local_tokenizer(
        transformers, args.model_dir, cache_dir, max_length
    )
    eod_id = getattr(tokenizer, "eod_id", None)
    if not isinstance(eod_id, int):
        raise SmokeFailure("the pinned Qwen tokenizer exposes no eod_id")
    tokenizer.pad_token_id = eod_id
    batch, token_count, supervised_count, image_token_count = _prepare_batch(
        fixture, tokenizer, upstream.preprocess, max_length, torch
    )
    report["batch"].update(
        {
            "token_count": token_count,
            "supervised_token_count": supervised_count,
            "image_token_count": image_token_count,
        }
    )

    model_config = transformers.AutoConfig.from_pretrained(
        str(args.model_dir),
        cache_dir=cache_dir,
        trust_remote_code=True,
        local_files_only=True,
    )
    model_config.use_cache = False
    model = transformers.AutoModelForCausalLM.from_pretrained(
        str(args.model_dir),
        config=model_config,
        cache_dir=cache_dir,
        device_map={"": args.device_index},
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        local_files_only=True,
        use_safetensors=True,
    )

    transformer = getattr(model, "transformer", None)
    visual = getattr(transformer, "visual", None)
    if visual is None:
        raise SmokeFailure("loaded model exposes no transformer.visual tower")
    visual.requires_grad_(False)
    matched_names = _matched_modules(model, EXPECTED_TARGET_MODULES)
    _check_target_dtypes(model, matched_names, torch)
    report["lora"]["matched_target_names"] = matched_names

    lora_config = LoraConfig(
        r=EXPECTED_LORA_RANK,
        lora_alpha=EXPECTED_LORA_ALPHA,
        target_modules=list(EXPECTED_TARGET_MODULES),
        lora_dropout=EXPECTED_LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        modules_to_save=None,
    )
    model = get_peft_model(model, lora_config)
    # The fixed fixture is roughly 800 tokens after image expansion.  The
    # 10B-parameter snapshot leaves much less than 30 GiB for activations on an
    # A40 with ECC enabled, so use the same Transformers/PEFT 0.5 checkpointing
    # path that the upstream Trainer supports.  This changes activation storage,
    # not the tested forward/backward semantics or the Member 1 stop line.
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.train()

    named_parameters = list(model.named_parameters())
    total_parameter_count = sum(parameter.numel() for _, parameter in named_parameters)
    trainable = [
        (name, parameter)
        for name, parameter in named_parameters
        if parameter.requires_grad
    ]
    trainable_names = [name for name, _ in trainable]
    trainable_parameter_count = sum(parameter.numel() for _, parameter in trainable)
    if not trainable or trainable_parameter_count < 1:
        raise SmokeFailure("PEFT created no trainable LoRA parameters")
    unexpected_trainable = [name for name in trainable_names if "lora_" not in name]
    if unexpected_trainable:
        raise SmokeFailure(
            "non-LoRA parameters remain trainable: " + ", ".join(unexpected_trainable[:5])
        )
    visual_trainable = [
        name
        for name, parameter in named_parameters
        if ".visual." in name and parameter.requires_grad
    ]
    if visual_trainable:
        raise SmokeFailure("visual tower parameters remain trainable after LoRA injection")
    if any(parameter.device.type != "cuda" for _, parameter in trainable):
        raise SmokeFailure("one or more trainable LoRA parameters are not on CUDA")

    report["lora"].update(
        {
            "total_parameter_count": int(total_parameter_count),
            "trainable_parameter_count": int(trainable_parameter_count),
            "trainable_parameter_names": trainable_names,
        }
    )

    batch_on_device = {key: value.to(device) for key, value in batch.items()}
    model.zero_grad(set_to_none=True)
    observed_lora_input_dtypes: list[Any] = []
    observed_module = next(
        (
            module
            for _, module in model.named_modules()
            if hasattr(module, "lora_A") and len(getattr(module, "lora_A")) > 0
        ),
        None,
    )
    if observed_module is None:
        raise SmokeFailure("could not locate an injected LoRA module for BF16 observation")

    def _observe_lora_input(_module: Any, inputs: tuple[Any, ...]) -> None:
        if inputs and isinstance(inputs[0], torch.Tensor):
            observed_lora_input_dtypes.append(inputs[0].dtype)

    observation_handle = observed_module.register_forward_pre_hook(_observe_lora_input)
    try:
        outputs = model(**batch_on_device)
    finally:
        observation_handle.remove()
    if torch.bfloat16 not in observed_lora_input_dtypes:
        observed = ", ".join(sorted({str(dtype) for dtype in observed_lora_input_dtypes}))
        raise SmokeFailure(
            "LoRA training path did not observe BF16 activations"
            + (f" (observed {observed})" if observed else "")
        )
    loss = getattr(outputs, "loss", None)
    if loss is None or loss.numel() != 1:
        raise SmokeFailure("model forward returned no scalar supervised loss")
    loss_value = float(loss.detach().float().cpu().item())
    if not math.isfinite(loss_value):
        raise SmokeFailure("model forward returned a non-finite loss")
    if not loss.requires_grad:
        raise SmokeFailure("model forward loss has no gradient function")
    report["batch"]["loss"] = loss_value
    report["forward_passed"] = True

    # Acceptance intentionally ends immediately after this call.  No optimizer
    # object exists anywhere in this program, so no parameter update can occur.
    loss.backward()
    nonzero_gradient_names: list[str] = []
    nonfinite_gradient_names: list[str] = []
    for name, parameter in trainable:
        gradient = parameter.grad
        if gradient is None:
            continue
        if not bool(torch.isfinite(gradient).all().item()):
            nonfinite_gradient_names.append(name)
        elif bool(gradient.detach().ne(0).any().item()):
            nonzero_gradient_names.append(name)
    if nonfinite_gradient_names:
        raise SmokeFailure(
            "non-finite LoRA gradients detected: "
            + ", ".join(nonfinite_gradient_names[:5])
        )
    if not nonzero_gradient_names:
        raise SmokeFailure("backward produced no finite non-zero LoRA gradient")

    torch.cuda.synchronize(device)
    report["lora"]["nonzero_gradient_parameter_names"] = nonzero_gradient_names
    report["backward_passed"] = True
    report["passed"] = True
    # These assignments make the stop line explicit even if the report skeleton
    # changes later; the JSON Schema also fixes both values to false.
    report["optimizer_step_performed"] = False
    report["adapter_saved"] = False
    report["errors"] = []


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    source_lock: dict[str, Any] | None = None
    try:
        candidate = read_json(args.source_lock)
        if isinstance(candidate, dict):
            source_lock = candidate
    except Exception:
        source_lock = None
    report = _empty_report(args.member_id, source_lock)
    torch_module: Any | None = None

    try:
        _run(args, report)
        # Import only after the runtime test, keeping --help usable in a clean
        # local shell with none of the GPU dependencies installed.
        import torch as torch_module

        report["memory"]["peak_allocated_bytes"] = int(
            torch_module.cuda.max_memory_allocated(args.device_index)
        )
        report["memory"]["peak_reserved_bytes"] = int(
            torch_module.cuda.max_memory_reserved(args.device_index)
        )
        report["generated_at_utc"] = utc_now()
        _validate_success_report(report, args.report_schema)
    except Exception as exc:
        report["passed"] = False
        report["errors"] = [_safe_error(exc)]
        report["generated_at_utc"] = utc_now()
        if torch_module is None:
            torch_module = sys.modules.get("torch")
        if torch_module is not None:
            report["versions"] = _runtime_versions(torch_module)
            try:
                if torch_module.cuda.is_available():
                    report["memory"]["peak_allocated_bytes"] = int(
                        torch_module.cuda.max_memory_allocated(args.device_index)
                    )
                    report["memory"]["peak_reserved_bytes"] = int(
                        torch_module.cuda.max_memory_reserved(args.device_index)
                    )
            except Exception:
                pass

    try:
        assert_report_is_redacted(report)
        write_json_atomic(args.report, report)
    except Exception as exc:
        print(
            json.dumps(
                {
                    "passed": False,
                    "report_written": False,
                    "error": _safe_error(exc),
                },
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 1

    summary = {
        "passed": report["passed"],
        "forward_passed": report["forward_passed"],
        "backward_passed": report["backward_passed"],
        "optimizer_step_performed": False,
        "adapter_saved": False,
        "report": os.fspath(args.report),
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
