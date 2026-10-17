"""Offline, image-aware Eval80 guidance-judge packet and paired report workflow.

The module deliberately does not choose or call a judge provider. A separate
image-capable judge must consume the private packets and produce the strict
JSONL result contract described by ``lock-guidance-judge --help``. Keeping the
answer key apart from the score lock makes blind scoring auditable.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import secrets
from statistics import mean
from typing import Any

from .core import EMOTION_CLASSES, SHA256_RE, sha256_file, sha256_text_file
from .inference import (
    FORMAL_C_ADAPTER_FILES, FORMAL_C_CHECKPOINT, FORMAL_C_EVIDENCE_SHA256,
)
from .output_contract import audit_output_contract
from .scoring import GUIDANCE_COLUMNS, read_guidance_scope


_ROOT = Path(__file__).resolve().parents[2]
_FREEZE = _ROOT / "Docs/evaluation/EVAL80_FREEZE_RECORD.json"
_EXCLUSION = _ROOT / "Docs/evaluation/EVAL80_DISPLAY_EXCLUSION.json"
_HANDBOOK = _ROOT / "Docs/evaluation/EVAL80_SCORING_HANDBOOK.md"
_CONFIG = _ROOT / "configs/phase1/eval80.json"
_MANIFEST = _ROOT / "results/phase2/eval80/frozen_inputs/inference_manifest.json"
_SCORE_KEYS = tuple(column.removesuffix("_0_2") for column in GUIDANCE_COLUMNS)
_JUDGE_KEYS = {
    "anonymous_item_id", "packet_sha256", "judge_config_sha256", "judge_model_id",
    "scores", "overall_rationale", "zero_score_reasons", "hallucination",
    "emotion_conflict", "recommendation_present", "response_metadata",
}


def _read_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"Empty JSONL line {number} in {path.name}.")
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"JSONL line {number} in {path.name} is not an object.")
            rows.append(row)
    return rows


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _json_output_bytes(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def _require_absent_or_identical(path: Path, expected: bytes) -> None:
    if path.exists() and (not path.is_file() or path.read_bytes() != expected):
        raise FileExistsError(f"Existing report differs from the recomputed result: {path}")


def _write_bytes_if_absent(path: Path, expected: bytes) -> None:
    if path.exists():
        _require_absent_or_identical(path, expected)
        return
    with path.open("xb") as handle:
        handle.write(expected)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _require_private_destination(path: Path) -> None:
    """Prevent accidentally writing raw answers or labels to a tracked tree."""
    destination = path.resolve()
    try:
        destination.relative_to(_ROOT)
    except ValueError:
        return
    if not any(destination.is_relative_to(_ROOT / safe) for safe in (
        "artifacts", "private_data", "tmp"
    )):
        raise ValueError(
            "Private judge output inside this repository must be under artifacts/ "
            "private_data/, or tmp/ (all Git-ignored)."
        )


def _safe_image_path(image_dir: Path, filename: str) -> Path:
    if not isinstance(filename, str) or not filename or Path(filename).name != filename:
        raise ValueError("Frozen image filename must be a basename.")
    path = (image_dir / filename).resolve()
    if not path.is_relative_to(image_dir.resolve()) or not path.is_file():
        raise ValueError(f"Frozen image is unavailable: {filename}")
    return path


def _validated_inputs(
    *, b1_dir: Path, c_dir: Path, manifest_path: Path,
    selection_path: Path, image_dir: Path, freeze_path: Path,
    config_path: Path, handbook_path: Path, exclusion_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    freeze = _read_object(freeze_path)
    if freeze.get("status") != "approved_and_frozen" or freeze.get("team_approval_confirmed") is not True:
        raise ValueError("Eval80 freeze record is not approved.")
    frozen_hashes = freeze.get("file_sha256")
    if not isinstance(frozen_hashes, dict):
        raise ValueError("Eval80 freeze hashes are missing.")
    for label, path, hash_fn in (
        ("inference_manifest", manifest_path, sha256_file),
        ("selection_manifest_private", selection_path, sha256_file),
        ("config", config_path, sha256_text_file),
        ("scoring_handbook", handbook_path, sha256_text_file),
    ):
        if hash_fn(path) != frozen_hashes.get(label):
            raise ValueError(f"{label} differs from the approved Eval80 freeze.")
    scope = read_guidance_scope(exclusion_path)
    config = _read_object(config_path)
    manifest = _read_object(manifest_path)
    selection = _read_object(selection_path)
    manifest_rows = manifest.get("records")
    selection_rows = selection.get("records")
    if not isinstance(manifest_rows, list) or len(manifest_rows) != 80:
        raise ValueError("Frozen inference manifest must contain 80 images.")
    if not isinstance(selection_rows, list) or len(selection_rows) != 80:
        raise ValueError("Private selection manifest must contain 80 images.")
    selection_by_id = {row.get("blind_id"): row for row in selection_rows if isinstance(row, dict)}
    manifest_by_id = {row.get("blind_id"): row for row in manifest_rows if isinstance(row, dict)}
    if len(selection_by_id) != 80 or set(selection_by_id) != set(manifest_by_id):
        raise ValueError("Selection and inference manifests must have the same unique 80 IDs.")
    if Counter(row.get("emotion") for row in selection_rows) != Counter(
        {emotion: 10 for emotion in EMOTION_CLASSES}
    ):
        raise ValueError("Private selection emotions must contain 10 images per class.")
    for blind_id, item in manifest_by_id.items():
        selected = selection_by_id[blind_id]
        image_hash = item.get("image_sha256")
        if not isinstance(image_hash, str) or not SHA256_RE.fullmatch(image_hash):
            raise ValueError(f"{blind_id}: invalid frozen image hash.")
        if selected.get("image_sha256") != image_hash:
            raise ValueError(f"{blind_id}: selection image hash differs from inference manifest.")
        if selected.get("local_image_filename") != item.get("image_filename"):
            raise ValueError(f"{blind_id}: selected image filename differs from inference manifest.")
        image_path = _safe_image_path(image_dir, item["image_filename"])
        if sha256_file(image_path) != image_hash:
            raise ValueError(f"{blind_id}: image bytes differ from the frozen SHA-256.")

    runs: dict[str, dict[str, Any]] = {}
    for condition, run_dir in (("B1", b1_dir), ("C", c_dir)):
        records_path = run_dir / "records.jsonl"
        run_config = _read_object(run_dir / "run_config.json")
        summary = _read_object(run_dir / "summary.json")
        validation = _read_object(run_dir / "validation.json")
        rows = _read_jsonl(records_path)
        audit_output_contract(rows, manifest, condition=condition)
        records_hash = sha256_file(records_path)
        if (
            validation.get("valid") is not True
            or validation.get("condition") != condition
            or validation.get("records_sha256") != records_hash
            or validation.get("record_count") != 80
            or validation.get("errors") != []
            or summary.get("completed") is not True
            or summary.get("condition") != condition
            or summary.get("record_count") != 80
            or summary.get("records_sha256") != records_hash
        ):
            raise ValueError(f"{condition}: accepted 80/80 technical validation is missing or stale.")
        hashes = run_config.get("file_sha256")
        if not isinstance(hashes, dict) or any(
            hashes.get(key) != frozen_hashes.get(key)
            for key in ("config", "inference_manifest", "scoring_handbook")
        ) or hashes.get("freeze_record") != sha256_file(freeze_path):
            raise ValueError(f"{condition}: run input hashes differ from the frozen inputs.")
        if run_config.get("condition") != condition or run_config.get("manifest_size") != 80:
            raise ValueError(f"{condition}: run configuration identifies a different condition or set.")
        runs[condition] = {
            "config": run_config,
            "records": {row["blind_id"]: row for row in rows},
            "records_sha256": records_hash,
            "validation_sha256": sha256_file(run_dir / "validation.json"),
        }

    baseline = runs["B1"]["config"]
    adapted = runs["C"]["config"]
    official_b1 = config["conditions"]["B1"]
    for run in (baseline, adapted):
        for key, expected in (
            ("prompt", official_b1["prompt"]),
            ("prompt_version", official_b1["prompt_version"]),
            ("seed", config["seed"]),
            ("precision", config["precision"]),
            ("batch_size", config["batch_size"]),
            ("generation_expected", config["generation"]),
        ):
            if run.get(key) != expected:
                raise ValueError(f"B1/C run {key} differs from the frozen B1 setting.")
    if adapted.get("generation_actual") != baseline.get("generation_actual"):
        raise ValueError("B1/C actual generation settings differ.")
    base_source = adapted.get("base_model_source", adapted.get("model_source"))
    baseline_source = baseline.get("model_source")
    identity_keys = ("repo_id", "revision", "weight_format")
    if (not isinstance(base_source, dict) or not isinstance(baseline_source, dict)
            or any(base_source.get(key) != baseline_source.get(key)
                   or base_source.get(key) != config["model"].get(key)
                   for key in identity_keys)):
        raise ValueError("C base model identity differs from B1 or frozen Eval80 config.")
    adapter_identity = adapted.get("adapter_source", adapted.get("adapter_identity"))
    if not isinstance(adapter_identity, dict) or not adapter_identity:
        raise ValueError("C adapter identity is missing from run_config.json.")
    if (
        adapter_identity.get("evidence_sha256") != FORMAL_C_EVIDENCE_SHA256
        or adapter_identity.get("selected_checkpoint") != FORMAL_C_CHECKPOINT
        or adapter_identity.get("adapter_files") != FORMAL_C_ADAPTER_FILES
        or adapter_identity.get("base_revision") != config["model"]["revision"]
    ):
        raise ValueError("C adapter identity differs from pinned formal training evidence.")
    lock_path = _ROOT / "phase3/configs/source-lock.json"
    source_lock = _read_object(lock_path).get("model")
    if not isinstance(source_lock, dict) or any(
        source_lock.get(key) != expected for key, expected in (
            ("repository", config["model"]["repo_id"]),
            ("revision", config["model"]["revision"]),
            ("weight_format", config["model"]["weight_format"]),
        )
    ):
        raise ValueError("C pinned base source lock differs from frozen Eval80 model.")
    snapshot_path = _ROOT / str(source_lock["snapshot_manifest"])
    if sha256_file(snapshot_path) != source_lock.get("snapshot_manifest_sha256"):
        raise ValueError("C pinned base snapshot manifest differs from source lock.")
    snapshot = _read_object(snapshot_path)
    if (
        snapshot.get("repository") != config["model"]["repo_id"]
        or snapshot.get("revision") != config["model"]["revision"]
        or snapshot.get("file_count") != source_lock.get("snapshot_file_count")
        or snapshot.get("total_bytes") != source_lock.get("snapshot_total_bytes")
    ):
        raise ValueError("C pinned base snapshot manifest metadata is inconsistent.")
    expected_snapshot_receipt = {
        "source_lock_sha256": sha256_file(lock_path),
        "snapshot_manifest_sha256": sha256_file(snapshot_path),
        "verified_file_count": snapshot["file_count"],
        "verified_total_bytes": snapshot["total_bytes"],
    }
    if adapted.get("base_snapshot_verification") != expected_snapshot_receipt:
        raise ValueError("C base snapshot verification receipt differs from pinned source lock.")
    if set(manifest_by_id) != {f"E{index:03d}" for index in range(1, 81)}:
        raise ValueError("Frozen blind IDs are not E001-E080.")
    return manifest_rows, selection_by_id, runs, {
        "scope": scope,
        "freeze_sha256": sha256_file(freeze_path),
        "inference_manifest_sha256": sha256_file(manifest_path),
        "selection_manifest_sha256": sha256_file(selection_path),
        "config_sha256": sha256_text_file(config_path),
        "handbook_sha256": sha256_text_file(handbook_path),
        "adapter_identity": adapter_identity,
        "base_snapshot_verification": expected_snapshot_receipt,
    }


def _validate_judge_config(config: dict[str, Any]) -> None:
    required = {"schema_version", "model_id", "provider", "endpoint", "prompt", "image_detail", "decoding", "retry_policy"}
    if not required <= set(config) or set(config) - required - {"expected_system_fingerprint"} or config["schema_version"] != 1:
        raise ValueError(f"Judge config must contain {sorted(required)} (schema_version=1), with optional expected_system_fingerprint.")
    for name in ("model_id", "provider", "prompt", "image_detail"):
        if not isinstance(config[name], str) or not config[name].strip():
            raise ValueError(f"Judge config {name} must be nonempty text.")
    if config["model_id"].lower() in {"latest", "default", "auto"}:
        raise ValueError("Judge model_id must pin a specific version.")
    for name in ("decoding", "retry_policy"):
        if not isinstance(config[name], dict) or not config[name]:
            raise ValueError(f"Judge config {name} must be a nonempty object.")
    if config["provider"] != "openai_compatible_chat":
        raise ValueError("Judge provider must be openai_compatible_chat.")
    if not isinstance(config["endpoint"], str) or not config["endpoint"].strip():
        raise ValueError("Judge endpoint must be nonempty text.")
    if "expected_system_fingerprint" in config and (
        not isinstance(config["expected_system_fingerprint"], str)
        or not config["expected_system_fingerprint"].strip()
    ):
        raise ValueError("Expected judge system fingerprint must be nonempty text.")


def prepare_guidance_judge_command(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate frozen B1/C outputs and prepare 156 private, anonymous image-aware judge items.")
    parser.add_argument("--b1-run-dir", type=Path, required=True)
    parser.add_argument("--c-run-dir", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--judge-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261017)
    parser.add_argument("--inference-manifest", type=Path, default=_MANIFEST)
    parser.add_argument("--freeze-record", type=Path, default=_FREEZE)
    parser.add_argument("--display-exclusion", type=Path, default=_EXCLUSION)
    parser.add_argument("--config", type=Path, default=_CONFIG)
    parser.add_argument("--handbook", type=Path, default=_HANDBOOK)
    args = parser.parse_args(argv)
    _require_private_destination(args.output_dir)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Private judge output directory must be empty.")
    judge_config = _read_object(args.judge_config)
    _validate_judge_config(judge_config)
    manifest_rows, selection_by_id, runs, binding = _validated_inputs(
        b1_dir=args.b1_run_dir, c_dir=args.c_run_dir,
        manifest_path=args.inference_manifest, selection_path=args.selection_manifest,
        image_dir=args.image_dir, freeze_path=args.freeze_record,
        config_path=args.config, handbook_path=args.handbook,
        exclusion_path=args.display_exclusion,
    )
    scope = binding.pop("scope")
    included = [row for row in manifest_rows if row["blind_id"] in scope.included_ids]
    if len(included) != 78:
        raise ValueError("Approved guidance scope must contain 78 images.")
    rng = random.Random(args.seed)
    rng.shuffle(included)
    b1_first = set(row["blind_id"] for row in included[:39])
    packets: list[dict[str, Any]] = []
    mapping: dict[str, dict[str, Any]] = {}
    for item in included:
        blind_id = item["blind_id"]
        condition_order = ("B1", "C") if blind_id in b1_first else ("C", "B1")
        for condition in condition_order:
            source = runs[condition]["records"][blind_id]
            anonymous_id = f"J{secrets.token_hex(12).upper()}"
            packet = {
                "anonymous_item_id": anonymous_id,
                "image_path": str(_safe_image_path(args.image_dir, item["image_filename"])),
                "image_sha256": item["image_sha256"],
                "target_emotion": selection_by_id[blind_id]["emotion"],
                "response": source["response"],
                "response_sha256": source["response_sha256"],
            }
            packets.append(packet)
            mapping[anonymous_id] = {
                "blind_id": blind_id,
                "condition": condition,
                "image_sha256": item["image_sha256"],
                "target_emotion": packet["target_emotion"],
                "response_sha256": packet["response_sha256"],
            }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    packet_path = args.output_dir / "judge_items_private.jsonl"
    mapping_path = args.output_dir / "condition_mapping_private.json"
    config_snapshot_path = args.output_dir / "judge_config_private.json"
    handbook_snapshot_path = args.output_dir / "judge_handbook_private.md"
    _write_jsonl(packet_path, packets)
    _write_json(mapping_path, {"schema_version": 1, "items": mapping})
    with config_snapshot_path.open("xb") as handle:
        handle.write(args.judge_config.read_bytes())
    with handbook_snapshot_path.open("xb") as handle:
        handle.write(args.handbook.read_bytes())
    metadata = {
        "schema_version": 1,
        "evaluation_set": "Eval80-v1",
        "metric": "image_aware_guidance_quality_proxy",
        "image_count": 78,
        "item_count": 156,
        "seed": args.seed,
        "balanced_first_condition_counts_private": {"B1": 39, "C": 39},
        "display_exclusion_decision_id": scope.decision_id,
        "display_exclusion_sha256": scope.source_sha256,
        "judge_config_sha256": sha256_file(args.judge_config),
        "judge_config_snapshot_sha256": sha256_file(config_snapshot_path),
        "judge_handbook_snapshot_sha256": sha256_file(handbook_snapshot_path),
        "judge_model_id": judge_config["model_id"],
        "expected_system_fingerprint": judge_config.get("expected_system_fingerprint"),
        "judge_items_sha256": sha256_file(packet_path),
        "condition_mapping_sha256": sha256_file(mapping_path),
        "b1_records_sha256": runs["B1"]["records_sha256"],
        "c_records_sha256": runs["C"]["records_sha256"],
        "b1_validation_sha256": runs["B1"]["validation_sha256"],
        "c_validation_sha256": runs["C"]["validation_sha256"],
        **binding,
    }
    _write_json(args.output_dir / "judge_package.json", metadata)
    print(json.dumps({key: metadata[key] for key in (
        "item_count", "image_count", "judge_model_id", "judge_items_sha256",
        "condition_mapping_sha256", "judge_config_sha256")}, indent=2))
    return 0


def _validate_judgements(
    score_path: Path, packet_path: Path, metadata: dict[str, Any],
    *, require_complete: bool = True,
) -> dict[str, dict[str, Any]]:
    packets = _read_jsonl(packet_path)
    if len(packets) != 156 or sha256_file(packet_path) != metadata.get("judge_items_sha256"):
        raise ValueError("Judge packet file changed or is incomplete.")
    packet_by_id = {row.get("anonymous_item_id"): row for row in packets}
    if len(packet_by_id) != 156:
        raise ValueError("Judge packet IDs are duplicated.")
    rows = _read_jsonl(score_path)
    if (require_complete and len(rows) != 156) or len(rows) > 156:
        raise ValueError(f"Expected 156 completed judgements, found {len(rows)}.")
    by_id: dict[str, dict[str, Any]] = {}
    for number, row in enumerate(rows, start=1):
        if set(row) != _JUDGE_KEYS:
            raise ValueError(f"Judgement {number}: fields differ from the strict result schema.")
        anonymous_id = row["anonymous_item_id"]
        if anonymous_id not in packet_by_id or anonymous_id in by_id:
            raise ValueError(f"Judgement {number}: unknown or duplicate anonymous item ID.")
        packet = packet_by_id[anonymous_id]
        canonical = json.dumps(packet, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        packet_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        if row["packet_sha256"] != packet_hash:
            raise ValueError(f"Judgement {number}: packet SHA-256 mismatch.")
        if (
            row["judge_config_sha256"] != metadata["judge_config_sha256"]
            or row["judge_model_id"] != metadata["judge_model_id"]
        ):
            raise ValueError(f"Judgement {number}: judge identity/config differs from the frozen package.")
        scores = row["scores"]
        if not isinstance(scores, dict) or set(scores) != set(_SCORE_KEYS) or any(
            type(scores[key]) is not int or scores[key] not in (0, 1, 2) for key in _SCORE_KEYS
        ):
            raise ValueError(f"Judgement {number}: each of five dimensions must be integer 0, 1, or 2.")
        reasons = row["zero_score_reasons"]
        zero_keys = {key for key in _SCORE_KEYS if scores[key] == 0}
        if not isinstance(reasons, dict) or set(reasons) != zero_keys or any(
            not isinstance(value, str) or not value.strip() for value in reasons.values()
        ):
            raise ValueError(f"Judgement {number}: every zero needs its own nonempty reason.")
        if not isinstance(row["overall_rationale"], str) or not row["overall_rationale"].strip():
            raise ValueError(f"Judgement {number}: overall rationale is required.")
        for key in ("hallucination", "emotion_conflict", "recommendation_present"):
            if type(row[key]) is not bool:
                raise ValueError(f"Judgement {number}: {key} must be a boolean.")
        if not row["recommendation_present"] and (
            scores["actionability"] != 0 or scores["emotion_preservation"] != 0
        ):
            raise ValueError(f"Judgement {number}: no recommendation requires both advice dimensions to be zero.")
        if not isinstance(row["response_metadata"], dict) or not row["response_metadata"]:
            raise ValueError(f"Judgement {number}: response metadata is required for audit.")
        expected_fingerprint = metadata.get("expected_system_fingerprint")
        if expected_fingerprint and row["response_metadata"].get("system_fingerprint") != expected_fingerprint:
            raise ValueError(f"Judgement {number}: judge system fingerprint changed.")
        by_id[anonymous_id] = row
    return by_id


def _validate_script_provenance(
    package_dir: Path, metadata: dict[str, Any],
    score_by_id: dict[str, dict[str, Any]],
) -> dict[str, str]:
    """Bind each score to a persisted raw provider response and runner state."""
    state_path = package_dir / "judge_run_state.json"
    raw_path = package_dir / "judge_raw_responses_private.jsonl"
    attempts_path = package_dir / "judge_attempts_private.jsonl"
    config_path = package_dir / "judge_config_private.json"
    handbook_path = package_dir / "judge_handbook_private.md"
    if not all(path.is_file() for path in (
        state_path, raw_path, attempts_path, config_path, handbook_path,
    )):
        raise ValueError("Script judge provenance needs run state, raw responses, attempts, and frozen judge inputs.")
    if (
        sha256_file(config_path) != metadata.get("judge_config_snapshot_sha256")
        or sha256_file(config_path) != metadata["judge_config_sha256"]
        or sha256_file(handbook_path) != metadata.get("judge_handbook_snapshot_sha256")
        or sha256_text_file(handbook_path) != metadata["handbook_sha256"]
    ):
        raise ValueError("Frozen judge config or handbook snapshot changed.")
    config = _read_object(config_path)
    _validate_judge_config(config)
    handbook = handbook_path.read_text(encoding="utf-8")
    expected_state = {
        "schema_version": 1,
        "judge_package_sha256": sha256_file(package_dir / "judge_package.json"),
        "judge_config_sha256": metadata["judge_config_sha256"],
        "judge_items_sha256": metadata["judge_items_sha256"],
        "handbook_sha256": metadata["handbook_sha256"],
    }
    if _read_object(state_path) != expected_state:
        raise ValueError("Judge run state differs from the frozen package.")
    packets = {
        row["anonymous_item_id"]: row
        for row in _read_jsonl(package_dir / "judge_items_private.jsonl")
    }
    from .quality_judge_runner import _make_payload, _extract_judgement

    expected_requests = {
        anonymous_id: hashlib.sha256(json.dumps(
            _make_payload(packet, config, handbook), sort_keys=True,
        ).encode("utf-8")).hexdigest()
        for anonymous_id, packet in packets.items()
    }
    raw_rows = _read_jsonl(raw_path)
    attempts = _read_jsonl(attempts_path)
    raw_by_response_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in raw_rows:
        response = row.get("response")
        if (
            row.get("anonymous_item_id") not in packets
            or type(row.get("attempt")) is not int or row["attempt"] < 1
            or row.get("request_sha256") != expected_requests.get(row.get("anonymous_item_id"))
            or not isinstance(response, dict)
        ):
            raise ValueError("Raw judge response log contains an invalid attempt or frozen request hash.")
        response_id = response.get("id")
        if isinstance(response_id, str) and response_id:
            raw_by_response_id[response_id].append(row)
    successful_attempts: dict[str, str] = {}
    for row in attempts:
        anonymous_id = row.get("anonymous_item_id")
        if anonymous_id not in packets or type(row.get("attempt")) is not int or row["attempt"] < 1:
            raise ValueError("Judge attempt log contains an invalid item or attempt.")
        if row.get("status") == "success":
            response_id = row.get("response_id")
            if anonymous_id in successful_attempts or not isinstance(response_id, str):
                raise ValueError("Judge attempt log has duplicate or invalid success.")
            successful_attempts[anonymous_id] = response_id
        elif row.get("status") != "error":
            raise ValueError("Judge attempt log has an unknown outcome.")
    if set(successful_attempts) != set(score_by_id):
        raise ValueError("Judge attempt successes do not match all scored items.")
    for anonymous_id, scored in score_by_id.items():
        response_id = scored["response_metadata"].get("response_id")
        raw_matches = [
            row for row in raw_by_response_id.get(response_id, [])
            if row["anonymous_item_id"] == anonymous_id
        ]
        if len(raw_matches) != 1 or successful_attempts[anonymous_id] != response_id:
            raise ValueError("A judge score lacks exactly one matching raw response and success log.")
        if _extract_judgement(raw_matches[0]["response"], packets[anonymous_id], metadata) != scored:
            raise ValueError("A judge score differs from its raw provider response.")
    return {
        "judge_run_state_sha256": sha256_file(state_path),
        "judge_raw_responses_sha256": sha256_file(raw_path),
        "judge_attempts_sha256": sha256_file(attempts_path),
    }


def lock_guidance_judge_command(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(
        "Validate all 156 anonymous judge JSONL records and lock their file hash before revealing conditions. "
        "Each record needs anonymous_item_id, packet_sha256 (SHA-256 of sorted compact JSON packet), "
        "judge_config_sha256, judge_model_id, scores (five 0-2 integer keys without _0_2 suffix), "
        "overall_rationale, zero_score_reasons, hallucination, emotion_conflict, "
        "recommendation_present, and nonempty response_metadata."
    ))
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--judgements", type=Path, required=True)
    parser.add_argument("--manual-import", action="store_true", help="Explicitly mark scores without runner trace as unverified manual import.")
    args = parser.parse_args(argv)
    package = _read_object(args.package_dir / "judge_package.json")
    scores = _validate_judgements(args.judgements, args.package_dir / "judge_items_private.jsonl", package)
    trace_paths = [args.package_dir / name for name in (
        "judge_run_state.json", "judge_raw_responses_private.jsonl",
        "judge_attempts_private.jsonl",
    )]
    if args.manual_import:
        if any(path.exists() for path in trace_paths):
            raise ValueError("Cannot mark existing runner traces as a manual import.")
        provenance = "manual_import_unverified"
        trace_hashes: dict[str, str] = {}
    else:
        provenance = "script_runner_trace_consistent"
        trace_hashes = _validate_script_provenance(args.package_dir, package, scores)
    lock = {
        "schema_version": 1,
        "item_count": 156,
        "judge_items_sha256": package["judge_items_sha256"],
        "judge_config_sha256": package["judge_config_sha256"],
        "judgements_sha256": sha256_file(args.judgements),
        "condition_mapping_sha256": package["condition_mapping_sha256"],
        "judge_provenance": provenance,
        **trace_hashes,
    }
    _write_json(args.package_dir / "judgements_lock.json", lock)
    print(json.dumps(lock, indent=2))
    return 0


def _paired_bootstrap(differences: list[int], seed: int, iterations: int = 10000) -> list[float]:
    rng = random.Random(seed)
    n = len(differences)
    means = sorted(sum(differences[rng.randrange(n)] for _ in range(n)) / n for _ in range(iterations))
    return [round(means[int(0.025 * (iterations - 1))], 4), round(means[int(0.975 * (iterations - 1))], 4)]


def aggregate_guidance(
    judgements: dict[str, dict[str, Any]], mapping: dict[str, dict[str, Any]],
    *, bootstrap_seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return aggregate-only public report and private paired cases."""
    if set(judgements) != set(mapping) or len(mapping) != 156:
        raise ValueError("Judgements and private condition mapping do not match 156 items.")
    by_pair: dict[tuple[str, str], dict[str, Any]] = {}
    emotions: dict[str, str] = {}
    for anonymous_id, item in mapping.items():
        blind_id, condition = item.get("blind_id"), item.get("condition")
        if condition not in ("B1", "C") or not isinstance(blind_id, str):
            raise ValueError("Private condition mapping contains an invalid pair.")
        pair = (blind_id, condition)
        if pair in by_pair:
            raise ValueError("Private condition mapping contains a duplicate image-condition pair.")
        emotion = item.get("target_emotion")
        if emotion not in EMOTION_CLASSES:
            raise ValueError("Private condition mapping contains an invalid target emotion.")
        if blind_id in emotions and emotions[blind_id] != emotion:
            raise ValueError("B1/C target emotions differ for one image.")
        emotions[blind_id] = emotion
        by_pair[pair] = judgements[anonymous_id]
    expected_ids = {f"E{index:03d}" for index in range(1, 81)} - {"E001", "E080"}
    if set(emotions) != expected_ids or set(by_pair) != {
        (blind_id, condition) for blind_id in expected_ids for condition in ("B1", "C")
    }:
        raise ValueError("Guidance scores must cover all 78 approved B1/C image pairs.")
    metrics = (*_SCORE_KEYS, "total")
    values: dict[str, dict[str, list[int]]] = {
        condition: {key: [] for key in metrics} for condition in ("B1", "C")
    }
    diffs: dict[str, list[int]] = {key: [] for key in metrics}
    flags = {condition: Counter() for condition in ("B1", "C")}
    per_emotion: dict[str, list[int]] = defaultdict(list)
    cases: list[dict[str, Any]] = []
    for blind_id in sorted(expected_ids):
        pair = {condition: by_pair[(blind_id, condition)] for condition in ("B1", "C")}
        totals = {condition: sum(pair[condition]["scores"].values()) for condition in pair}
        for condition in ("B1", "C"):
            for key in _SCORE_KEYS:
                values[condition][key].append(pair[condition]["scores"][key])
            values[condition]["total"].append(totals[condition])
            for flag in ("hallucination", "emotion_conflict"):
                flags[condition][flag] += int(pair[condition][flag])
        for key in _SCORE_KEYS:
            diffs[key].append(pair["C"]["scores"][key] - pair["B1"]["scores"][key])
        delta = totals["C"] - totals["B1"]
        diffs["total"].append(delta)
        per_emotion[emotions[blind_id]].append(delta)
        cases.append({
            "blind_id": blind_id, "target_emotion": emotions[blind_id],
            "b1_total_0_10": totals["B1"], "c_total_0_10": totals["C"],
            "delta_c_minus_b1": delta,
            "b1_hallucination": pair["B1"]["hallucination"],
            "c_hallucination": pair["C"]["hallucination"],
            "b1_emotion_conflict": pair["B1"]["emotion_conflict"],
            "c_emotion_conflict": pair["C"]["emotion_conflict"],
        })
    interval = _paired_bootstrap(diffs["total"], bootstrap_seed)
    summary = {
        "schema_version": 1,
        "metric": "image_aware_ai_guidance_quality_proxy",
        "image_count": 78,
        "judgement_count": 156,
        "dimensions": {
            key: {
                "b1_mean": round(mean(values["B1"][key]), 4),
                "c_mean": round(mean(values["C"][key]), 4),
                "paired_mean_delta_c_minus_b1": round(mean(diffs[key]), 4),
            } for key in metrics
        },
        "total_wins_ties_losses_c_vs_b1": {
            "wins": sum(value > 0 for value in diffs["total"]),
            "ties": sum(value == 0 for value in diffs["total"]),
            "losses": sum(value < 0 for value in diffs["total"]),
        },
        "total_delta_descriptive_bootstrap_95_interval": interval,
        "bootstrap_seed": bootstrap_seed,
        "bootstrap_resamples": 10000,
        "failure_counts": {condition: dict(flags[condition]) for condition in ("B1", "C")},
        "per_emotion": {
            emotion: {"image_count": len(per_emotion[emotion]),
                      "paired_mean_delta_c_minus_b1": round(mean(per_emotion[emotion]), 4)}
            for emotion in EMOTION_CLASSES
        },
        "interpretation_note": (
            "Fixed-judge proxy on the 78 selected images; the bootstrap interval describes "
            "sensitivity to these image IDs, not general-population certainty or human agreement."
        ),
    }
    return summary, {"schema_version": 1, "cases": cases}


def report_guidance_command(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Unblind a locked 156-item judge run and write aggregate-only paired guidance results.")
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--judgements", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-seed", type=int, default=20261017)
    args = parser.parse_args(argv)
    package = _read_object(args.package_dir / "judge_package.json")
    lock = _read_object(args.package_dir / "judgements_lock.json")
    expected_lock = {
        "schema_version": 1,
        "item_count": 156,
        "judge_items_sha256": package["judge_items_sha256"],
        "judge_config_sha256": package["judge_config_sha256"],
        "judgements_sha256": sha256_file(args.judgements),
        "condition_mapping_sha256": package["condition_mapping_sha256"],
    }
    provenance = lock.get("judge_provenance")
    if provenance == "script_runner_trace_consistent":
        scores_for_trace = _validate_judgements(
            args.judgements, args.package_dir / "judge_items_private.jsonl", package
        )
        expected_lock.update(_validate_script_provenance(args.package_dir, package, scores_for_trace))
    elif provenance != "manual_import_unverified":
        raise ValueError("Judge score lock lacks a recognized provenance.")
    expected_lock["judge_provenance"] = provenance
    if lock != expected_lock:
        raise ValueError("Judge score lock no longer matches the package and score file.")
    score_by_id = _validate_judgements(
        args.judgements, args.package_dir / "judge_items_private.jsonl", package
    )
    mapping_path = args.package_dir / "condition_mapping_private.json"
    if sha256_file(mapping_path) != package["condition_mapping_sha256"]:
        raise ValueError("Condition mapping changed after anonymous scoring.")
    mapping = _read_object(mapping_path)
    if mapping.get("schema_version") != 1 or not isinstance(mapping.get("items"), dict):
        raise ValueError("Invalid private condition mapping.")
    packets = {row["anonymous_item_id"]: row for row in _read_jsonl(
        args.package_dir / "judge_items_private.jsonl"
    )}
    for anonymous_id, item in mapping["items"].items():
        if anonymous_id not in packets or any(
            item.get(key) != packets[anonymous_id].get(key)
            for key in ("image_sha256", "target_emotion", "response_sha256")
        ):
            raise ValueError("Private mapping does not match the frozen anonymous packet.")
    summary, cases = aggregate_guidance(score_by_id, mapping["items"], bootstrap_seed=args.bootstrap_seed)
    summary["judge_model_id"] = package["judge_model_id"]
    summary["judge_provenance"] = provenance
    summary["judge_identity_status"] = (
        "fingerprint_matched" if package.get("expected_system_fingerprint")
        else "unverified_model_id_only"
    )
    summary["expected_system_fingerprint"] = package.get("expected_system_fingerprint")
    summary["source_records_sha256"] = {
        "B1": package["b1_records_sha256"],
        "C": package["c_records_sha256"],
    }
    summary["inference_manifest_sha256"] = package["inference_manifest_sha256"]
    summary["selection_manifest_sha256"] = package["selection_manifest_sha256"]
    summary["adapter_acceptance_status"] = package["adapter_identity"].get(
        "acceptance_status", "unrecorded"
    )
    summary["member3b_acceptance_verified"] = False
    summary["adapter_provenance"] = {
        "formal_evidence_sha256": package["adapter_identity"]["evidence_sha256"],
        "selected_checkpoint": package["adapter_identity"]["selected_checkpoint"],
        "adapter_files": package["adapter_identity"]["adapter_files"],
        "base_revision": package["adapter_identity"]["base_revision"],
        "base_snapshot_verification": package["base_snapshot_verification"],
    }
    summary["input_sha256"] = {
        "judge_package": sha256_file(args.package_dir / "judge_package.json"),
        "judgements": sha256_file(args.judgements),
        "judge_config": package["judge_config_sha256"],
        "condition_mapping": package["condition_mapping_sha256"],
    }
    cases_path = args.package_dir / "paired_cases_private.json"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.resolve() == cases_path.resolve():
        raise ValueError("Aggregate report path must differ from private paired cases.")
    outputs = (
        (cases_path, _json_output_bytes(cases)),
        (args.output, _json_output_bytes(summary)),
    )
    for path, expected in outputs:
        _require_absent_or_identical(path, expected)
    for path, expected in outputs:
        _write_bytes_if_absent(path, expected)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _validate_benchmark_inputs(
    format_reports: dict[str, dict[str, Any]],
    recognition_reports: dict[str, dict[str, Any]],
    paired_recognition: dict[str, Any],
    guidance: dict[str, Any],
) -> None:
    if guidance.get("metric") != "image_aware_ai_guidance_quality_proxy" or (
        guidance.get("image_count"), guidance.get("judgement_count")
    ) != (78, 156):
        raise ValueError("Guidance aggregate is not a complete 78-image/156-item proxy result.")
    source_hashes = guidance.get("source_records_sha256")
    if not isinstance(source_hashes, dict) or set(source_hashes) != {"B1", "C"}:
        raise ValueError("Guidance aggregate lacks B1/C source hashes.")
    for condition in ("B1", "C"):
        fmt = format_reports[condition]
        recognition = recognition_reports[condition]
        if (
            fmt.get("condition") != condition
            or fmt.get("item_count") != 80
            or type(fmt.get("format_pass_count")) is not int
            or not 0 <= fmt["format_pass_count"] <= 80
            or fmt.get("format_failure_count") != 80 - fmt["format_pass_count"]
        ):
            raise ValueError(f"{condition} format audit is not a complete 80-image result.")
        if (
            recognition.get("condition") != condition
            or recognition.get("item_count") != 80
            or type(recognition.get("correct_count")) is not int
            or not 0 <= recognition["correct_count"] <= 80
        ):
            raise ValueError(f"{condition} recognition summary is not a complete 80-image result.")
        if fmt.get("records_sha256") != source_hashes[condition] or (
            recognition.get("records_sha256") != source_hashes[condition]
        ):
            raise ValueError(f"{condition} benchmark axes come from different output records.")
        if fmt.get("inference_manifest_sha256") != guidance.get("inference_manifest_sha256"):
            raise ValueError(f"{condition} format and guidance use different frozen manifests.")
        if recognition.get("selection_manifest_sha256") != guidance.get("selection_manifest_sha256"):
            raise ValueError(f"{condition} recognition and guidance use different answer keys.")
        paired_condition = paired_recognition.get("conditions", {}).get(condition, {})
        if paired_condition.get("correct_count") != recognition["correct_count"]:
            raise ValueError(f"{condition} paired recognition count differs from its summary.")
    outcomes = paired_recognition.get("paired_outcomes")
    if (
        paired_recognition.get("comparison") != "B1_vs_C_strict_emotion_recognition"
        or paired_recognition.get("image_count") != 80
        or paired_recognition.get("input_sha256") != {
            "b1_details": recognition_reports["B1"].get("details_sha256"),
            "c_details": recognition_reports["C"].get("details_sha256"),
        }
        or not isinstance(outcomes, dict)
        or set(outcomes) != {"both_correct", "b1_only_correct", "c_only_correct", "both_wrong"}
        or any(type(count) is not int or count < 0 for count in outcomes.values())
        or sum(outcomes.values()) != 80
        or outcomes["both_correct"] + outcomes["b1_only_correct"] != recognition_reports["B1"]["correct_count"]
        or outcomes["both_correct"] + outcomes["c_only_correct"] != recognition_reports["C"]["correct_count"]
        or paired_recognition.get("net_correct_gain_c_minus_b1") != (
            recognition_reports["C"]["correct_count"]
            - recognition_reports["B1"]["correct_count"]
        )
    ):
        raise ValueError("Paired recognition report is incomplete or differs from its score sheets.")
    dimensions = guidance.get("dimensions")
    if not isinstance(dimensions, dict) or set(dimensions) != set(_SCORE_KEYS) | {"total"}:
        raise ValueError("Guidance aggregate is missing the five dimensions or total.")
    for key, values in dimensions.items():
        ceiling = 10 if key == "total" else 2
        if not isinstance(values, dict) or any(
            not isinstance(values.get(name), (int, float))
            or isinstance(values.get(name), bool)
            for name in ("b1_mean", "c_mean", "paired_mean_delta_c_minus_b1")
        ) or not all(0 <= values[name] <= ceiling for name in ("b1_mean", "c_mean")):
            raise ValueError(f"Guidance aggregate dimension {key} is invalid.")
    interval = guidance.get("total_delta_descriptive_bootstrap_95_interval")
    if not isinstance(interval, list) or len(interval) != 2 or any(
        not isinstance(value, (int, float)) or isinstance(value, bool) for value in interval
    ) or interval[0] > interval[1]:
        raise ValueError("Guidance aggregate has an invalid descriptive interval.")


def _benchmark_headline(paired: dict[str, Any], guidance: dict[str, Any]) -> str:
    recognition_delta = paired["net_correct_gain_c_minus_b1"]
    dimensions = guidance["dimensions"]
    guidance_delta = dimensions["total"]["paired_mean_delta_c_minus_b1"]
    lower, upper = guidance["total_delta_descriptive_bootstrap_95_interval"]
    flags = guidance["failure_counts"]
    flag_increase = any(
        flags["C"][name] > flags["B1"][name]
        for name in ("hallucination", "emotion_conflict")
    )
    if recognition_delta > 0 and lower > 0 and (
        dimensions["visual_grounding"]["paired_mean_delta_c_minus_b1"] >= 0
        and dimensions["emotion_preservation"]["paired_mean_delta_c_minus_b1"] >= 0
    ) and not flag_increase:
        return "Guidance proxy improved on the selected images; strict recognition count also increased."
    if (recognition_delta > 0 and guidance_delta < 0) or (
        recognition_delta < 0 and guidance_delta > 0
    ):
        return "Recognition and guidance-proxy results move in opposite directions; report a trade-off."
    if lower <= 0 <= upper:
        return "Guidance-proxy direction is uncertain on this selected set."
    if flag_increase:
        return "Mixed proxy result: hallucination or target-emotion conflict increased."
    if upper < 0:
        return "Guidance proxy declined on the selected images."
    return "Mixed Eval80 result; inspect recognition, all five dimensions, and failure counts."


def _available_benchmark_axes(loaded: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Validate and retain observed aggregate values without filling missing axes."""
    available: dict[str, Any] = {
        "format": {"B1": None, "C": None},
        "recognition": {"B1": None, "C": None},
        "guidance_proxy": None,
    }
    for condition, label in (("B1", "b1"), ("C", "c")):
        fmt = loaded.get(f"{label}_format")
        rec = loaded.get(f"{label}_recognition")
        if fmt is not None:
            passed = fmt.get("format_pass_count")
            if (
                fmt.get("condition") != condition
                or fmt.get("item_count") != 80
                or type(passed) is not int or not 0 <= passed <= 80
                or fmt.get("format_failure_count") != 80 - passed
                or not isinstance(fmt.get("records_sha256"), str)
                or not SHA256_RE.fullmatch(fmt["records_sha256"])
                or not isinstance(fmt.get("failure_reason_counts"), dict)
            ):
                raise ValueError(f"{condition} format audit is invalid or incomplete.")
            available["format"][condition] = {
                "pass_count": passed,
                "item_count": 80,
                "failure_reason_counts": fmt["failure_reason_counts"],
                "records_sha256": fmt["records_sha256"],
            }
        if rec is not None:
            correct = rec.get("correct_count")
            if (
                rec.get("condition") != condition
                or rec.get("item_count") != 80
                or type(correct) is not int or not 0 <= correct <= 80
                or not isinstance(rec.get("records_sha256"), str)
                or not SHA256_RE.fullmatch(rec["records_sha256"])
            ):
                raise ValueError(f"{condition} recognition summary is invalid or incomplete.")
            available["recognition"][condition] = {
                "correct_count": correct,
                "item_count": 80,
                "records_sha256": rec["records_sha256"],
            }
        if fmt is not None and rec is not None and (
            fmt["records_sha256"] != rec["records_sha256"]
        ):
            raise ValueError(f"{condition} available format and recognition use different outputs.")
    quality = loaded.get("guidance")
    if quality is not None:
        if (
            quality.get("metric") != "image_aware_ai_guidance_quality_proxy"
            or quality.get("image_count") != 78
            or quality.get("judgement_count") != 156
            or not isinstance(quality.get("source_records_sha256"), dict)
            or not isinstance(quality.get("dimensions"), dict)
            or "total" not in quality["dimensions"]
        ):
            raise ValueError("Available guidance aggregate is invalid or incomplete.")
        for condition, label in (("B1", "b1"), ("C", "c")):
            quality_hash = quality["source_records_sha256"].get(condition)
            if not isinstance(quality_hash, str) or not SHA256_RE.fullmatch(quality_hash):
                raise ValueError("Guidance aggregate lacks a valid B1/C source hash.")
            for axis in ("format", "recognition"):
                observed = loaded.get(f"{label}_{axis}")
                if observed is not None and observed["records_sha256"] != quality_hash:
                    raise ValueError(f"{condition} available guidance uses different outputs.")
        total = quality["dimensions"]["total"]
        if not isinstance(total, dict) or any(
            not isinstance(total.get(key), (int, float)) or isinstance(total.get(key), bool)
            for key in ("b1_mean", "c_mean", "paired_mean_delta_c_minus_b1")
        ):
            raise ValueError("Available guidance total is invalid.")
        available["guidance_proxy"] = {
            "image_count": 78,
            "judge_model_id": quality.get("judge_model_id"),
            "b1_mean_total_0_10": total["b1_mean"],
            "c_mean_total_0_10": total["c_mean"],
            "paired_mean_delta_c_minus_b1": total["paired_mean_delta_c_minus_b1"],
            "source_records_sha256": quality["source_records_sha256"],
        }
    return available


def report_benchmark_command(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create a one-page, aggregate-only B1/C benchmark summary; missing inputs remain explicitly incomplete.")
    for name in (
        "b1-format", "c-format", "b1-recognition", "c-recognition",
        "paired-recognition", "guidance",
    ):
        parser.add_argument(f"--{name}", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args(argv)
    paths = {
        "b1_format": args.b1_format, "c_format": args.c_format,
        "b1_recognition": args.b1_recognition, "c_recognition": args.c_recognition,
        "paired_recognition": args.paired_recognition, "guidance": args.guidance,
    }
    if args.output_json.resolve() == args.output_md.resolve():
        raise ValueError("JSON and Markdown output paths must differ.")
    missing = [name for name, path in paths.items() if path is None]
    for path in paths.values():
        if path is not None and not path.is_file():
            raise FileNotFoundError(path)
    if missing:
        loaded = {name: _read_object(path) for name, path in paths.items() if path is not None}
        observed = _available_benchmark_axes(loaded)
        report: dict[str, Any] = {
            "schema_version": 1,
            "evaluation_set": "Eval80-v1",
            "comparison": "B1_vs_C_adapter_effect",
            "status": "incomplete",
            "missing_inputs": missing,
            "format": observed["format"],
            "recognition": observed["recognition"],
            "guidance_proxy": observed["guidance_proxy"],
            "input_sha256": {
                name: sha256_file(path) for name, path in paths.items() if path is not None
            },
            "note": "No missing measure is converted to zero; complete and validate every axis before claiming improvement.",
        }
        b1_format = observed["format"]["B1"]
        c_format = observed["format"]["C"]
        b1_recognition = observed["recognition"]["B1"]
        c_recognition = observed["recognition"]["C"]
        quality = observed["guidance_proxy"]
        def _cell(item: dict[str, Any] | None, key: str, denominator: int) -> str:
            return f"{item[key]}/{denominator}" if item is not None else "pending"

        markdown = (
            "# Eval80 B1/C benchmark — incomplete\n\n"
            "| Measure | B1 | C |\n| --- | ---: | ---: |\n"
            f"| Full output schema | {_cell(b1_format, 'pass_count', 80)} | "
            f"{_cell(c_format, 'pass_count', 80)} |\n"
            f"| Strict emotion recognition | {_cell(b1_recognition, 'correct_count', 80)} | "
            f"{_cell(c_recognition, 'correct_count', 80)} |\n"
            f"| Guidance proxy mean (/10) | {quality['b1_mean_total_0_10'] if quality else 'pending'} | "
            f"{quality['c_mean_total_0_10'] if quality else 'pending'} |\n\n"
            "Missing inputs: " + ", ".join(missing) + ".\n\n"
            "Observed recognition is the strict frozen Eval80 structured-label score, "
            "not training-set or development-set accuracy. Pending values are not zeros.\n"
        )
    else:
        loaded = {name: _read_object(path) for name, path in paths.items() if path is not None}
        fmt = {"B1": loaded["b1_format"], "C": loaded["c_format"]}
        rec = {"B1": loaded["b1_recognition"], "C": loaded["c_recognition"]}
        paired = loaded["paired_recognition"]
        quality = loaded["guidance"]
        _validate_benchmark_inputs(fmt, rec, paired, quality)
        adapter_status = quality.get("adapter_acceptance_status")
        # The run config's acceptance_status is self-declared. The repository
        # currently has no independently authenticated Member 3B acceptance
        # artifact, so this report cannot promote a proxy to complete status.
        blocks = ["adapter_acceptance_unverified"]
        if quality.get("judge_identity_status") != "fingerprint_matched":
            blocks.append("judge_identity_unverified")
        if quality.get("judge_provenance") != "script_runner_trace_consistent":
            blocks.append("judge_provenance_unverified")
        status = "provisional_proxy"
        report = {
            "schema_version": 1,
            "evaluation_set": "Eval80-v1",
            "comparison": "B1_vs_C_adapter_effect",
            "status": status,
            "headline": _benchmark_headline(paired, quality),
            "format": {condition: {
                "pass_count": fmt[condition]["format_pass_count"],
                "item_count": 80,
                "failure_reason_counts": fmt[condition]["failure_reason_counts"],
            } for condition in ("B1", "C")},
            "recognition": {
                "B1_correct_count": rec["B1"]["correct_count"],
                "C_correct_count": rec["C"]["correct_count"],
                "net_correct_gain_c_minus_b1": paired["net_correct_gain_c_minus_b1"],
                "paired_outcomes": paired["paired_outcomes"],
                "per_class": paired["per_class"],
            },
            "guidance_proxy": {
                "judge_model_id": quality["judge_model_id"],
                "image_count": 78,
                "dimensions": quality["dimensions"],
                "total_wins_ties_losses_c_vs_b1": quality["total_wins_ties_losses_c_vs_b1"],
                "total_delta_descriptive_bootstrap_95_interval": quality["total_delta_descriptive_bootstrap_95_interval"],
                "failure_counts": quality["failure_counts"],
                "per_emotion": quality["per_emotion"],
            },
            "adapter_acceptance_status": adapter_status,
            "member3b_acceptance_verified": False,
            "judge_identity_status": quality.get("judge_identity_status", "unverified_model_id_only"),
            "formal_claims_blocked_by": blocks,
            "scope_note": "Format and recognition use 80 images; guidance proxy uses 78 approved images. This is not a six-human score.",
            "input_sha256": {name: sha256_file(path) for name, path in paths.items() if path is not None},
        }
        dimensions = quality["dimensions"]
        rows = [
            ("Full output schema (/80)", str(fmt["B1"]["format_pass_count"]),
             str(fmt["C"]["format_pass_count"]), "—"),
            ("Emotion correct (/80)", str(rec["B1"]["correct_count"]),
             str(rec["C"]["correct_count"]),
             f"{paired['net_correct_gain_c_minus_b1']:+d}"),
            ("Guidance proxy (/10, 78 images)",
             f"{dimensions['total']['b1_mean']:.2f}",
             f"{dimensions['total']['c_mean']:.2f}",
             f"{dimensions['total']['paired_mean_delta_c_minus_b1']:+.2f}"),
        ]
        for key in _SCORE_KEYS:
            item = dimensions[key]
            rows.append((f"{key.replace('_', ' ').title()} (/2)",
                         f"{item['b1_mean']:.2f}", f"{item['c_mean']:.2f}",
                         f"{item['paired_mean_delta_c_minus_b1']:+.2f}"))
        for flag, label in (("hallucination", "Visual hallucination"),
                            ("emotion_conflict", "Target-emotion conflict")):
            rows.append((f"{label} (/78)", str(quality["failure_counts"]["B1"][flag]),
                         str(quality["failure_counts"]["C"][flag]), "—"))
        table = "\n".join(f"| {label} | {b1} | {c} | {delta} |" for label, b1, c, delta in rows)
        interval = quality["total_delta_descriptive_bootstrap_95_interval"]
        wins = quality["total_wins_ties_losses_c_vs_b1"]
        schema_note = (
            "B1 passed the full output schema on 0/80 images. Its strict recognition "
            "count is confounded by schema compliance; interpret the label result "
            "with a separate same-field diagnostic.\n\n"
            if fmt["B1"]["format_pass_count"] == 0 else ""
        )
        markdown = (
            "# Eval80 B1/C benchmark\n\n"
            f"**Status:** {status}. **Conclusion:** {report['headline']}\n\n"
            "| Measure | B1 | C | C − B1 |\n| --- | ---: | ---: | ---: |\n"
            f"{table}\n\n"
            f"Guidance paired wins/ties/losses: {wins['wins']}/{wins['ties']}/{wins['losses']}. "
            f"Descriptive paired bootstrap interval: [{interval[0]:+.2f}, {interval[1]:+.2f}].\n\n"
            f"{schema_note}"
            f"C run-config acceptance field: {adapter_status}; independent Member 3B "
            "acceptance is unverified. Judge identity: "
            f"{report['judge_identity_status']}. The guidance score is a fixed AI-judge "
            "proxy on 78 images; the format and recognition results use all 80. "
            "The interval is descriptive for these selected images, not evidence of population-level "
            "or six-human-reviewer improvement.\n"
        )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    _write_json(args.output_json, report)
    with args.output_md.open("x", encoding="utf-8") as handle:
        handle.write(markdown)
    print(json.dumps({"status": report["status"], "output_json": str(args.output_json),
                      "output_md": str(args.output_md)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(prepare_guidance_judge_command())
