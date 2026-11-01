"""Blind, swapped-order B1/C guidance preference packets and aggregate reporting.

This is a separate AI-judge proxy. It never changes the frozen human handbook or
the five-dimension guidance score, and no item-level answer or rationale is
written outside the private package directory.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import secrets
from typing import Any
from urllib.parse import urlparse

from .core import EMOTION_CLASSES, SHA256_RE, sha256_file
from .quality_benchmark import (
    _CONFIG, _EXCLUSION, _FREEZE, _HANDBOOK, _MANIFEST,
    _read_jsonl, _read_object, _require_private_destination, _safe_image_path,
    _validated_inputs, _validate_judge_config, _write_bytes_if_absent,
    _write_json, _write_jsonl,
)


_ITEM_COUNT = 156
_IMAGE_COUNT = 78
_WINNERS = {"A", "B", "tie"}
_SCORE_KEYS = {
    "anonymous_item_id", "packet_sha256", "preference_config_sha256",
    "judge_model_id", "winner", "reason", "response_metadata",
}
_MODULE = Path(__file__).resolve()
_RUNNER = _MODULE.with_name("preference_judge_runner.py")


def _packet_sha256(packet: dict[str, Any]) -> str:
    canonical = json.dumps(packet, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _object_sha256(value: Any) -> str:
    canonical = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _is_loopback_endpoint(endpoint: str) -> bool:
    parsed = urlparse(endpoint)
    return (
        parsed.scheme in {"http", "https"}
        and parsed.hostname in {"localhost", "127.0.0.1", "::1"}
        and not parsed.username and not parsed.password
        and not parsed.fragment
    )


def _validate_preference_config(config: dict[str, Any]) -> None:
    _validate_judge_config(config)
    if not _is_loopback_endpoint(config["endpoint"]):
        raise ValueError("Preference judge requires a loopback endpoint; remote image transmission is disabled.")
    try:
        urlparse(config["endpoint"]).port
    except ValueError as error:
        raise ValueError("Preference judge endpoint has an invalid port.") from error


def _identity_mode(config: dict[str, Any]) -> str:
    return ("ollama_api_tags" if urlparse(config["endpoint"]).port == 11434
            else "receipt_only")


def _validate_runtime_identity(
    identity: Any, config: dict[str, Any], metadata: dict[str, Any],
    *, complete: bool,
) -> None:
    if not isinstance(identity, dict) or set(identity) != {"mode", "checks"}:
        raise ValueError("Preference runtime model identity state is missing.")
    mode, checks = identity["mode"], identity["checks"]
    if mode != _identity_mode(config) or not isinstance(checks, list):
        raise ValueError("Preference runtime model identity mode differs from the endpoint.")
    if mode == "receipt_only":
        if checks:
            raise ValueError("Receipt-only preference identity must not claim runtime checks.")
        return
    if not checks or not isinstance(checks[0], dict) or checks[0].get("phase") != "before":
        raise ValueError("Preference runtime model identity lacks a pre-run digest check.")
    if complete and checks[-1].get("phase") != "after":
        raise ValueError("Preference runtime model identity lacks a post-run digest check.")
    for index, check in enumerate(checks):
        if not isinstance(check, dict) or set(check) != {
            "phase", "observed_at_utc", "model_id", "digest_sha256"
        }:
            raise ValueError("Preference runtime model identity check is malformed.")
        if (check["phase"] not in ({"before"} if index == 0 else {"before", "after"})
                or not isinstance(check["observed_at_utc"], str)
                or not check["observed_at_utc"]
                or check["model_id"] != metadata["judge_model_id"]
                or check["digest_sha256"] != metadata["model_digest_sha256"]):
            raise ValueError("Preference runtime model digest differs from the frozen receipt.")


def _validate_model_receipt(receipt: dict[str, Any], model_id: str) -> str:
    digest = receipt.get("model_digest_sha256")
    if (receipt.get("schema_version") != 1 or receipt.get("model_id") != model_id
            or not isinstance(digest, str) or not SHA256_RE.fullmatch(digest)):
        raise ValueError("Model digest receipt must bind the exact judge model ID and SHA-256 digest.")
    return digest


def _validate_preflight_receipt(
    receipt_path: Path, config: dict[str, Any], model_digest: str,
) -> dict[str, str]:
    """Bind a local calibration receipt and its evidence to this exact wire contract.

    The receipt is an auditable local declaration, not a signature or expert
    validation. The summary gate accepts only synthetic/BridgeTrain controls;
    it never reads or scores Eval80 responses.
    """
    receipt = _read_object(receipt_path)
    from .preference_judge_runner import _preference_schema

    schema_hash = _object_sha256(_preference_schema())
    decoding_hash = _object_sha256(config["decoding"])
    prompt_hash = hashlib.sha256(config["prompt"].encode("utf-8")).hexdigest()
    expected = {
        "schema_version": 1,
        "status": "passed",
        "pass_scope": "dual_order_consensus_only",
        "data_scope": "synthetic_and_bridge_dev_only",
        "eval80_outputs_used": False,
        "endpoint": config["endpoint"],
        "model_id": config["model_id"],
        "model_digest_sha256": model_digest,
        "model_digest_before_sha256": model_digest,
        "model_digest_after_sha256": model_digest,
        "prompt": config["prompt"],
        "prompt_sha256": prompt_hash,
        "response_schema": _preference_schema(),
        "response_schema_sha256": schema_hash,
        "decoding": config["decoding"],
        "decoding_sha256": decoding_hash,
        "image_detail": config["image_detail"],
        "message_content_order": ["image_url", "text"],
        "calibration_pairs": 13,
        "calibration_requests": 26,
        "clear_pairs_consensus_clean": 9,
        "neutral_pairs": {"identical_bypass_tie": 1, "unstable": 3},
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError("Passed preference preflight receipt does not match the frozen model and wire contract.")
    image_hashes = receipt.get("image_sha256")
    if (not isinstance(image_hashes, dict)
            or set(image_hashes) != {"synthetic", "bridge_dev"}
            or any(not isinstance(value, str) or not SHA256_RE.fullmatch(value)
                   for value in image_hashes.values())
            or len(set(image_hashes.values())) != 2):
        raise ValueError("Preference preflight must pin two distinct development-image hashes.")
    evidence = receipt.get("evidence_files")
    if not isinstance(evidence, list) or not evidence:
        raise ValueError("Preference preflight receipt needs hashed evidence files.")
    roles: Counter[str] = Counter()
    summary_path: Path | None = None
    seen_paths: set[Path] = set()
    raw_response_ids: set[str] = set()
    for item in evidence:
        if not isinstance(item, dict) or set(item) != {"role", "path", "sha256"}:
            raise ValueError("Preference preflight evidence entry is invalid.")
        role, name, expected_hash = item["role"], item["path"], item["sha256"]
        if (role not in {"calibration_summary", "request", "raw_response"}
                or not isinstance(name, str) or not name
                or not isinstance(expected_hash, str) or not SHA256_RE.fullmatch(expected_hash)):
            raise ValueError("Preference preflight evidence role, path or hash is invalid.")
        if not Path(name).is_absolute():
            raise ValueError("Preference preflight evidence paths must be absolute for snapshot revalidation.")
        path = (receipt_path.parent / name).resolve()
        if path in seen_paths or not path.is_file() or sha256_file(path) != expected_hash:
            raise ValueError("Preference preflight evidence file is missing, duplicated or changed.")
        seen_paths.add(path)
        roles[role] += 1
        if role == "calibration_summary":
            summary_path = path
        elif role == "request":
            request = _read_object(path)
            from .preference_judge_runner import _preflight_request_image_sha256
            image_sha = _preflight_request_image_sha256(request, config, _preference_schema())
            if image_sha not in set(image_hashes.values()):
                raise ValueError("Preference calibration request contains an unapproved image.")
        else:
            raw = _read_object(path)
            response_id = raw.get("id")
            if (raw.get("model") != config["model_id"]
                    or not isinstance(response_id, str) or not response_id
                    or response_id in raw_response_ids):
                raise ValueError("Preference calibration raw response identity is invalid.")
            raw_response_ids.add(response_id)
    if (roles["calibration_summary"] != 1 or roles["request"] != 26
            or roles["raw_response"] != 26):
        raise ValueError("Preference preflight needs one summary and 26 hashed request/raw pairs.")
    assert summary_path is not None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(summary, list) or len(summary) != 13:
        raise ValueError("Preference calibration summary must be a list of controlled pairs.")
    clear_cases = {"hallucination", "generic", "conflict"}
    neutral_cases = {"identical", "equivalent", "style_only", "verbose_only"}
    clear_coverage: set[tuple[str, str]] = set()
    neutral_coverage: set[str] = set()
    summary_response_ids: set[str] = set()
    for row in summary:
        if not isinstance(row, dict) or row.get("image") not in {"synthetic", "bridge"}:
            raise ValueError("Preference calibration summary includes a non-development image.")
        case, outcome = row.get("case"), row.get("consensus")
        response_ids = row.get("response_ids")
        if (not isinstance(response_ids, list) or len(response_ids) != 2
                or any(not isinstance(value, str) or not value for value in response_ids)):
            raise ValueError("Preference calibration pair needs two auditable response IDs.")
        if any(value in summary_response_ids for value in response_ids):
            raise ValueError("Preference calibration response IDs are duplicated.")
        summary_response_ids.update(response_ids)
        if case in clear_cases:
            if outcome != "clean":
                raise ValueError("Preference clear-better control failed.")
            clear_coverage.add((row["image"], case))
        elif case in neutral_cases:
            allowed_outcome = "identical_bypass_tie" if case == "identical" else "unstable"
            if outcome != allowed_outcome:
                raise ValueError("Preference neutral control was misclassified as a win.")
            neutral_coverage.add(case)
        else:
            raise ValueError("Preference calibration has an unknown control case.")
    if (clear_coverage != {(image, case) for image in ("synthetic", "bridge")
                          for case in clear_cases}
            or neutral_coverage != neutral_cases):
        raise ValueError("Preference preflight lacks required clear and neutral control coverage.")
    if summary_response_ids != raw_response_ids:
        raise ValueError("Preference calibration summary IDs differ from the raw evidence.")
    return {
        "preflight_receipt_sha256": sha256_file(receipt_path),
        "response_schema_sha256": schema_hash,
        "decoding_sha256": decoding_hash,
    }


def prepare_preference_judge_command(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(
        "Validate frozen B1/C runs and prepare 156 private, anonymous A/B-swapped preference items."
    ))
    parser.add_argument("--b1-run-dir", type=Path, required=True)
    parser.add_argument("--c-run-dir", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--judge-config", type=Path, required=True)
    parser.add_argument("--model-digest-receipt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261017)
    parser.add_argument("--inference-manifest", type=Path, default=_MANIFEST)
    parser.add_argument("--freeze-record", type=Path, default=_FREEZE)
    parser.add_argument("--display-exclusion", type=Path, default=_EXCLUSION)
    parser.add_argument("--config", type=Path, default=_CONFIG)
    parser.add_argument("--handbook", type=Path, default=_HANDBOOK)
    parser.add_argument("--preflight-receipt", type=Path, required=True,
                        help="Passed synthetic/development calibration receipt with hashed request and raw evidence.")
    args = parser.parse_args(argv)
    _require_private_destination(args.output_dir)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Private preference package directory must be empty.")
    judge_config = _read_object(args.judge_config)
    _validate_preference_config(judge_config)
    model_digest = _validate_model_receipt(
        _read_object(args.model_digest_receipt), judge_config["model_id"]
    )
    preflight_binding = _validate_preflight_receipt(
        args.preflight_receipt, judge_config, model_digest
    )
    manifest_rows, selection_by_id, runs, binding = _validated_inputs(
        b1_dir=args.b1_run_dir, c_dir=args.c_run_dir,
        manifest_path=args.inference_manifest, selection_path=args.selection_manifest,
        image_dir=args.image_dir, freeze_path=args.freeze_record,
        config_path=args.config, handbook_path=args.handbook,
        exclusion_path=args.display_exclusion,
    )
    scope = binding.pop("scope")
    eligible = [row for row in manifest_rows if row["blind_id"] in scope.included_ids]
    if len(eligible) != _IMAGE_COUNT:
        raise ValueError("Approved guidance preference scope must contain 78 images.")
    rng = random.Random(args.seed)
    rng.shuffle(eligible)
    b1_first_ids = {row["blind_id"] for row in eligible[:39]}
    packets: list[dict[str, Any]] = []
    mapping: dict[str, dict[str, Any]] = {}
    for item in eligible:
        blind_id = item["blind_id"]
        first_order = ("B1", "C") if blind_id in b1_first_ids else ("C", "B1")
        for variant, (a_condition, b_condition) in enumerate((first_order, first_order[::-1])):
            answer_a = runs[a_condition]["records"][blind_id]
            answer_b = runs[b_condition]["records"][blind_id]
            anonymous_id = f"P{secrets.token_hex(12).upper()}"
            packet = {
                "anonymous_item_id": anonymous_id,
                "image_path": str(_safe_image_path(args.image_dir, item["image_filename"])),
                "image_sha256": item["image_sha256"],
                "target_emotion": selection_by_id[blind_id]["emotion"],
                "answer_A": answer_a["response"],
                "answer_A_sha256": answer_a["response_sha256"],
                "answer_B": answer_b["response"],
                "answer_B_sha256": answer_b["response_sha256"],
            }
            packets.append(packet)
            mapping[anonymous_id] = {
                "blind_id": blind_id,
                "variant": variant,
                "A_condition": a_condition,
                "B_condition": b_condition,
                "image_sha256": packet["image_sha256"],
                "target_emotion": packet["target_emotion"],
                "answer_A_sha256": packet["answer_A_sha256"],
                "answer_B_sha256": packet["answer_B_sha256"],
            }
    rng.shuffle(packets)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    packet_path = args.output_dir / "preference_items_private.jsonl"
    mapping_path = args.output_dir / "preference_mapping_private.json"
    config_snapshot = args.output_dir / "preference_config_private.json"
    receipt_snapshot = args.output_dir / "model_digest_receipt_private.json"
    preflight_snapshot = args.output_dir / "preflight_receipt_private.json"
    _write_jsonl(packet_path, packets)
    _write_json(mapping_path, {"schema_version": 1, "items": mapping})
    with config_snapshot.open("xb") as handle:
        handle.write(args.judge_config.read_bytes())
    with receipt_snapshot.open("xb") as handle:
        handle.write(args.model_digest_receipt.read_bytes())
    with preflight_snapshot.open("xb") as handle:
        handle.write(args.preflight_receipt.read_bytes())
    metadata = {
        "schema_version": 1,
        "evaluation_set": "Eval80-v1",
        "metric": "image_aware_pairwise_guidance_preference_proxy",
        "image_count": _IMAGE_COUNT,
        "item_count": _ITEM_COUNT,
        "seed": args.seed,
        "balanced_first_condition_counts_private": {"B1": 39, "C": 39},
        "display_exclusion_decision_id": scope.decision_id,
        "display_exclusion_sha256": scope.source_sha256,
        "preference_config_sha256": sha256_file(args.judge_config),
        "preference_config_snapshot_sha256": sha256_file(config_snapshot),
        "model_digest_sha256": model_digest,
        "model_digest_receipt_sha256": sha256_file(receipt_snapshot),
        "judge_model_id": judge_config["model_id"],
        "expected_system_fingerprint": judge_config.get("expected_system_fingerprint"),
        "prompt_sha256": hashlib.sha256(judge_config["prompt"].encode("utf-8")).hexdigest(),
        "benchmark_source_sha256": sha256_file(_MODULE),
        "runner_source_sha256": sha256_file(_RUNNER),
        **preflight_binding,
        "preference_items_sha256": sha256_file(packet_path),
        "preference_mapping_sha256": sha256_file(mapping_path),
        "b1_records_sha256": runs["B1"]["records_sha256"],
        "c_records_sha256": runs["C"]["records_sha256"],
        "b1_validation_sha256": runs["B1"]["validation_sha256"],
        "c_validation_sha256": runs["C"]["validation_sha256"],
        **binding,
    }
    _write_json(args.output_dir / "preference_package.json", metadata)
    print(json.dumps({key: metadata[key] for key in (
        "image_count", "item_count", "judge_model_id", "preference_items_sha256",
        "preference_config_sha256", "preflight_receipt_sha256",
    )}, indent=2))
    return 0


def _validate_judgements(
    score_path: Path, packet_path: Path, metadata: dict[str, Any],
    *, require_complete: bool = True,
) -> dict[str, dict[str, Any]]:
    if sha256_file(packet_path) != metadata.get("preference_items_sha256"):
        raise ValueError("Preference packet file changed after preparation.")
    packets = _read_jsonl(packet_path)
    packet_by_id = {row.get("anonymous_item_id"): row for row in packets}
    if len(packets) != _ITEM_COUNT or len(packet_by_id) != _ITEM_COUNT:
        raise ValueError("Preference packets must contain 156 unique items.")
    rows = _read_jsonl(score_path)
    if len(rows) > _ITEM_COUNT or (require_complete and len(rows) != _ITEM_COUNT):
        raise ValueError(f"Expected 156 preference judgements, found {len(rows)}.")
    by_id: dict[str, dict[str, Any]] = {}
    for number, row in enumerate(rows, start=1):
        if set(row) != _SCORE_KEYS:
            raise ValueError(f"Preference judgement {number}: fields differ from strict schema.")
        anonymous_id = row["anonymous_item_id"]
        if anonymous_id not in packet_by_id or anonymous_id in by_id:
            raise ValueError(f"Preference judgement {number}: unknown or duplicate item.")
        if row["packet_sha256"] != _packet_sha256(packet_by_id[anonymous_id]):
            raise ValueError(f"Preference judgement {number}: packet hash differs.")
        if (row["preference_config_sha256"] != metadata["preference_config_sha256"]
                or row["judge_model_id"] != metadata["judge_model_id"]):
            raise ValueError(f"Preference judgement {number}: judge configuration differs.")
        if row["winner"] not in _WINNERS:
            raise ValueError(f"Preference judgement {number}: winner must be A, B, or tie.")
        if (not isinstance(row["reason"], str) or not row["reason"].strip()
                or len(row["reason"]) > 800):
            raise ValueError(f"Preference judgement {number}: concise reason is required.")
        response_metadata = row["response_metadata"]
        if (not isinstance(response_metadata, dict)
                or not isinstance(response_metadata.get("response_id"), str)
                or not response_metadata["response_id"]
                or response_metadata.get("model") != metadata["judge_model_id"]):
            raise ValueError(f"Preference judgement {number}: response identity is missing.")
        expected_fingerprint = metadata.get("expected_system_fingerprint")
        if expected_fingerprint and response_metadata.get("system_fingerprint") != expected_fingerprint:
            raise ValueError(f"Preference judgement {number}: judge fingerprint differs.")
        by_id[anonymous_id] = row
    return by_id


def _validate_script_trace(
    package_dir: Path, metadata: dict[str, Any], scores: dict[str, dict[str, Any]],
    *, require_post_check: bool = True,
) -> dict[str, str]:
    """Check all scored rows against raw responses and their exact request bytes."""
    state_path = package_dir / "preference_run_state.json"
    raw_path = package_dir / "preference_raw_responses_private.jsonl"
    attempts_path = package_dir / "preference_attempts_private.jsonl"
    if not all(path.is_file() for path in (state_path, raw_path, attempts_path)):
        raise ValueError("Preference runner trace needs state, raw responses and attempts.")
    expected_state = {
        "schema_version": 1,
        "preference_package_sha256": sha256_file(package_dir / "preference_package.json"),
        "preference_config_sha256": metadata["preference_config_sha256"],
        "preference_items_sha256": metadata["preference_items_sha256"],
        "benchmark_source_sha256": metadata["benchmark_source_sha256"],
        "runner_source_sha256": metadata["runner_source_sha256"],
    }
    state = _read_object(state_path)
    if (set(state) != set(expected_state) | {"runtime_model_identity"}
            or any(state.get(key) != value for key, value in expected_state.items())):
        raise ValueError("Preference runner state differs from frozen package.")
    if (sha256_file(_MODULE) != metadata["benchmark_source_sha256"]
            or sha256_file(_RUNNER) != metadata["runner_source_sha256"]):
        raise ValueError("Preference benchmark code changed after package preparation.")
    config_snapshot = package_dir / "preference_config_private.json"
    if sha256_file(config_snapshot) != metadata["preference_config_snapshot_sha256"]:
        raise ValueError("Frozen preference config snapshot changed.")
    config = _read_object(config_snapshot)
    _validate_preference_config(config)
    _validate_runtime_identity(
        state["runtime_model_identity"], config, metadata,
        complete=require_post_check and len(scores) == _ITEM_COUNT,
    )
    receipt_snapshot = package_dir / "model_digest_receipt_private.json"
    if (sha256_file(receipt_snapshot) != metadata["model_digest_receipt_sha256"]
            or _validate_model_receipt(_read_object(receipt_snapshot), config["model_id"])
            != metadata["model_digest_sha256"]):
        raise ValueError("Frozen preference model digest receipt changed.")
    preflight_snapshot = package_dir / "preflight_receipt_private.json"
    if sha256_file(preflight_snapshot) != metadata["preflight_receipt_sha256"]:
        raise ValueError("Frozen preference preflight receipt changed.")
    expected_preflight = _validate_preflight_receipt(
        preflight_snapshot, config, metadata["model_digest_sha256"]
    )
    if any(metadata.get(key) != value for key, value in expected_preflight.items()):
        raise ValueError("Preference calibration evidence changed after preparation.")
    packets = {row["anonymous_item_id"]: row for row in _read_jsonl(
        package_dir / "preference_items_private.jsonl"
    )}
    from .preference_judge_runner import _extract_judgement, _make_payload, _request_sha256

    expected_requests = {
        anonymous_id: _request_sha256(_make_payload(packet, config))
        for anonymous_id, packet in packets.items()
    }
    raw_rows = _read_jsonl(raw_path)
    attempts = _read_jsonl(attempts_path)
    raw_by_key: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in raw_rows:
        anonymous_id, attempt = row.get("anonymous_item_id"), row.get("attempt")
        if (anonymous_id not in packets or type(attempt) is not int or attempt < 1
                or row.get("request_sha256") != expected_requests[anonymous_id]
                or row.get("packet_sha256") != _packet_sha256(packets[anonymous_id])
                or not isinstance(row.get("response"), dict)):
            raise ValueError("Raw preference response log contains an invalid request or response.")
        raw_by_key[(anonymous_id, attempt)].append(row)
        if len(raw_by_key[(anonymous_id, attempt)]) > 1:
            raise ValueError("Raw preference response log duplicates one attempt.")
    successful: dict[str, tuple[int, str]] = {}
    attempt_keys: set[tuple[str, int]] = set()
    for row in attempts:
        anonymous_id, attempt = row.get("anonymous_item_id"), row.get("attempt")
        if anonymous_id not in packets or type(attempt) is not int or attempt < 1:
            raise ValueError("Preference attempt log contains an invalid item or attempt.")
        key = (anonymous_id, attempt)
        if key in attempt_keys:
            raise ValueError("Preference attempt log contains a duplicate attempt.")
        attempt_keys.add(key)
        if row.get("request_sha256") != expected_requests[anonymous_id]:
            raise ValueError("Preference attempt request hash differs from frozen payload.")
        if row.get("status") == "success":
            response_id = row.get("response_id")
            if (anonymous_id in successful or not isinstance(response_id, str)
                    or not response_id):
                raise ValueError("Preference attempt log has duplicate or invalid success.")
            successful[anonymous_id] = (attempt, response_id)
        elif row.get("status") != "error":
            raise ValueError("Preference attempt log has an unknown outcome.")
    if set(successful) != set(scores):
        raise ValueError("Preference attempt successes differ from scored items.")
    if set(raw_by_key) - attempt_keys:
        raise ValueError("Raw preference responses lack matching attempt records.")
    for anonymous_id, scored in scores.items():
        attempt, response_id = successful[anonymous_id]
        matches = raw_by_key.get((anonymous_id, attempt), [])
        if (len(matches) != 1
                or matches[0]["response"].get("id") != response_id
                or scored["response_metadata"]["response_id"] != response_id):
            raise ValueError("Preference score lacks one matching raw response.")
        if _extract_judgement(matches[0]["response"], packets[anonymous_id], metadata) != scored:
            raise ValueError("Preference score differs from its raw provider response.")
    return {
        "preference_run_state_sha256": sha256_file(state_path),
        "preference_raw_responses_sha256": sha256_file(raw_path),
        "preference_attempts_sha256": sha256_file(attempts_path),
    }


def _lock_preference_judge(package_dir: Path, score_path: Path) -> dict[str, Any]:
    metadata = _read_object(package_dir / "preference_package.json")
    packet_path = package_dir / "preference_items_private.jsonl"
    scores = _validate_judgements(score_path, packet_path, metadata)
    trace_hashes = _validate_script_trace(package_dir, metadata, scores)
    lock = {
        "schema_version": 1,
        "item_count": _ITEM_COUNT,
        "preference_items_sha256": metadata["preference_items_sha256"],
        "preference_mapping_sha256": metadata["preference_mapping_sha256"],
        "preference_config_sha256": metadata["preference_config_sha256"],
        "judgements_sha256": sha256_file(score_path),
        "judge_provenance": "script_runner_trace_consistent",
        **trace_hashes,
    }
    _write_json(package_dir / "preference_judgements_lock.json", lock)
    return lock


def _validate_mapping(
    mapping: dict[str, Any], packets: dict[str, dict[str, Any]],
) -> dict[str, list[tuple[str, dict[str, Any]]]]:
    items = mapping.get("items")
    if mapping.get("schema_version") != 1 or not isinstance(items, dict) or set(items) != set(packets):
        raise ValueError("Private preference mapping differs from frozen packets.")
    by_image: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    for anonymous_id, item in items.items():
        packet = packets[anonymous_id]
        if (not isinstance(item, dict) or item.get("A_condition") not in {"B1", "C"}
                or item.get("B_condition") not in {"B1", "C"}
                or item["A_condition"] == item["B_condition"]
                or type(item.get("variant")) is not int or item["variant"] not in {0, 1}
                or item.get("target_emotion") not in EMOTION_CLASSES
                or any(item.get(key) != packet.get(key) for key in (
                    "image_sha256", "target_emotion", "answer_A_sha256", "answer_B_sha256"
                ))):
            raise ValueError("Private preference mapping contains an invalid item.")
        blind_id = item.get("blind_id")
        if not isinstance(blind_id, str):
            raise ValueError("Private preference mapping lacks an image ID.")
        by_image[blind_id].append((anonymous_id, item))
    expected_ids = {f"E{index:03d}" for index in range(2, 80)}
    if set(by_image) != expected_ids:
        raise ValueError("Preference mapping must cover the approved 78 images.")
    for blind_id, pair in by_image.items():
        if len(pair) != 2 or {item["variant"] for _, item in pair} != {0, 1}:
            raise ValueError(f"{blind_id}: two swapped variants are required.")
        first = next(item for _, item in pair if item["variant"] == 0)
        second = next(item for _, item in pair if item["variant"] == 1)
        if (first["A_condition"] != second["B_condition"]
                or first["B_condition"] != second["A_condition"]
                or first["target_emotion"] != second["target_emotion"]
                or first["image_sha256"] != second["image_sha256"]
                or first["answer_A_sha256"] != second["answer_B_sha256"]
                or first["answer_B_sha256"] != second["answer_A_sha256"]):
            raise ValueError(f"{blind_id}: preference variants are not exact swaps.")
    return by_image


def report_preference_command(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(
        "Unblind 156 locked swapped-order judgements and report an aggregate B1/C preference proxy."
    ))
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    package = _read_object(args.package_dir / "preference_package.json")
    score_path = args.package_dir / "preference_judgements_private.jsonl"
    scores = _validate_judgements(
        score_path, args.package_dir / "preference_items_private.jsonl", package
    )
    trace_hashes = _validate_script_trace(args.package_dir, package, scores)
    expected_lock = {
        "schema_version": 1,
        "item_count": _ITEM_COUNT,
        "preference_items_sha256": package["preference_items_sha256"],
        "preference_mapping_sha256": package["preference_mapping_sha256"],
        "preference_config_sha256": package["preference_config_sha256"],
        "judgements_sha256": sha256_file(score_path),
        "judge_provenance": "script_runner_trace_consistent",
        **trace_hashes,
    }
    if _read_object(args.package_dir / "preference_judgements_lock.json") != expected_lock:
        raise ValueError("Preference score lock differs from the frozen package or runner trace.")
    mapping_path = args.package_dir / "preference_mapping_private.json"
    if sha256_file(mapping_path) != package["preference_mapping_sha256"]:
        raise ValueError("Private preference mapping changed after score locking.")
    packets = {row["anonymous_item_id"]: row for row in _read_jsonl(
        args.package_dir / "preference_items_private.jsonl"
    )}
    by_image = _validate_mapping(_read_object(mapping_path), packets)
    votes: Counter[str] = Counter()
    consensus: Counter[str] = Counter()
    identical_answer_rule_ties = 0
    private_cases: list[dict[str, Any]] = []
    for blind_id in sorted(by_image):
        pair_votes: list[str] = []
        case_items: list[dict[str, Any]] = []
        identical_answers = all(
            packets[anonymous_id]["answer_A_sha256"]
            == packets[anonymous_id]["answer_B_sha256"]
            for anonymous_id, _ in by_image[blind_id]
        )
        for anonymous_id, item in sorted(by_image[blind_id], key=lambda entry: entry[1]["variant"]):
            winner = scores[anonymous_id]["winner"]
            condition_vote = "tie" if winner == "tie" else item[f"{winner}_condition"]
            votes[condition_vote] += 1
            pair_votes.append(condition_vote)
            case_items.append({
                "anonymous_item_id": anonymous_id,
                "variant": item["variant"],
                "A_condition": item["A_condition"],
                "B_condition": item["B_condition"],
                "winner": winner,
                "condition_vote": condition_vote,
                "reason": scores[anonymous_id]["reason"],
            })
        if identical_answers:
            outcome = "tie"
            identical_answer_rule_ties += 1
        else:
            outcome = pair_votes[0] if pair_votes[0] == pair_votes[1] else "unstable"
        consensus[outcome] += 1
        private_cases.append({
            "blind_id": blind_id, "outcome": outcome,
            "identical_answer_rule_applied": identical_answers, "items": case_items,
        })
    if sum(votes.values()) != _ITEM_COUNT or sum(consensus.values()) != _IMAGE_COUNT:
        raise ValueError("Preference vote totals do not cover the full frozen scope.")
    summary = {
        "schema_version": 1,
        "evaluation_set": "Eval80-v1",
        "status": "provisional_proxy",
        "metric": "image_aware_pairwise_guidance_preference_proxy",
        "image_count": _IMAGE_COUNT,
        "judgement_count": _ITEM_COUNT,
        "judge_model_id": package["judge_model_id"],
        "judge_identity_status": (
            "fingerprint_matched" if package.get("expected_system_fingerprint")
            else "unverified_model_id_only"
        ),
        "model_digest_verification_status": (
            "runtime_ollama_before_after_matched"
            if _identity_mode(_read_object(args.package_dir / "preference_config_private.json"))
            == "ollama_api_tags" else "receipt_only_unverified_at_runtime"
        ),
        "adapter_acceptance_status": package["adapter_identity"].get(
            "acceptance_status", "unrecorded"
        ),
        "judge_provenance": "script_runner_trace_consistent",
        "item_votes": {"C": votes["C"], "B1": votes["B1"], "tie": votes["tie"],
                       "net_c_minus_b1": votes["C"] - votes["B1"]},
        "image_consensus": {
            "C_wins": consensus["C"],
            "B1_wins": consensus["B1"],
            "ties": consensus["tie"],
            "identical_answer_rule_ties": identical_answer_rule_ties,
            "swap_disagreements_unstable": consensus["unstable"],
            "stable_net_c_minus_b1": consensus["C"] - consensus["B1"],
            "worst_case_net_if_all_unstable_favor_b1": (
                consensus["C"] - consensus["B1"] - consensus["unstable"]
            ),
        },
        "source_records_sha256": {
            "B1": package["b1_records_sha256"],
            "C": package["c_records_sha256"],
        },
        "inference_manifest_sha256": package["inference_manifest_sha256"],
        "selection_manifest_sha256": package["selection_manifest_sha256"],
        "preference_config_sha256": package["preference_config_sha256"],
        "prompt_sha256": package["prompt_sha256"],
        "preflight_receipt_sha256": package["preflight_receipt_sha256"],
        "model_digest_sha256": package["model_digest_sha256"],
        "model_digest_receipt_sha256": package["model_digest_receipt_sha256"],
        "input_sha256": {
            "package": sha256_file(args.package_dir / "preference_package.json"),
            "items": package["preference_items_sha256"],
            "mapping": package["preference_mapping_sha256"],
            "judgements": sha256_file(score_path),
            **trace_hashes,
        },
        "interpretation_note": (
            "Exploratory AI preference proxy on 78 selected images, not the frozen five-dimension "
            "human Guidance Score. Each image has two swapped-order decisions. Byte-identical "
            "B1/C answers are deterministic ties; otherwise any disagreement is unstable, never "
            "a tie or a win. Do not infer population-level aesthetic improvement "
            "or formal adapter acceptance from this result. The calibration receipt is local and "
            "unsigned; any runtime Ollama digest observations are bound to the private run state."
        ),
    }
    private_path = args.package_dir / "preference_cases_private.json"
    private_bytes = (json.dumps({"schema_version": 1, "cases": private_cases},
                                ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    report_bytes = (json.dumps(summary, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _write_bytes_if_absent(private_path, private_bytes)
    _write_bytes_if_absent(args.output, report_bytes)
    print(json.dumps({"status": summary["status"],
                      "image_consensus": summary["image_consensus"],
                      "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(prepare_preference_judge_command())
