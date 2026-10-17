"""Versioned, blind completion of the failed Eval80 preference attempt 002.

The original attempt is immutable.  This module reuses its 149 audited decisions,
excludes an entire image after an exhausted technical failure, and traverses every
remaining anonymous item with the exact attempt-002 wire contract.  The B1/C key
is opened only by ``report`` after the completed anonymous run has a score lock.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any

from .core import sha256_file
from .preference_benchmark import (
    _identity_mode, _is_loopback_endpoint, _packet_sha256, _read_jsonl,
    _read_object, _require_private_destination, _validate_judgements,
    _validate_mapping, _validate_preference_config, _validate_script_trace,
    _validate_runtime_identity, _write_bytes_if_absent, _write_json,
)
from .preference_judge_runner import (
    _extract_judgement, _make_payload, _probe_ollama_digest, _request_sha256,
)
from .quality_judge_runner import _append_jsonl, _post_chat


_SOURCE_HASHES = {
    "preference_package.json": "45357a71d116a6fe1b9506b8a6936aee60fe6be96abdd29bb113ad046de80321",
    "preference_items_private.jsonl": "32a686e35ce2b4b3e38a9cbdc5e3513c02bef1a86c33155cc4cc9daf6771c4c1",
    "preference_attempts_private.jsonl": "45d66599423af0c74808a5a9b3d86cfff0e36fb4ca7837658deee7e7226872ba",
    "preference_raw_responses_private.jsonl": "2e4f600947d4e6da6ff9a3a82d2b1a44b0265c9816f5919f3444b50000414505",
    "preference_judgements_private.jsonl": "d671176564cb4a92d350bc4374078ab295db18aa8ebe9016a81cc102434370ff",
    "preference_run_state.json": "7c14237df009e610724343def5906c1e141867344da32c5a0bd771da9cede5b2",
}
_META = "recovery_package_private.json"
_STATE = "recovery_run_state_private.json"
_LOG = "recovery_attempts_private.jsonl"
_LOCK = "recovery_decisions_lock_private.json"
_REFERENCE = "human_reference_private.csv"
_REFERENCE_FIELDS = (
    "image_code", "reviewer_id", "preferred", "obvious_hallucinated_worse", "reason",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _pairs(packets: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    """Pair reversed requests using anonymous image and answer hashes only."""
    if len(packets) != 156:
        raise ValueError("Attempt 002 must contain exactly 156 anonymous items.")
    by_id: dict[str, dict[str, Any]] = {}
    by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for packet in packets:
        anonymous_id = packet.get("anonymous_item_id")
        image_hash = packet.get("image_sha256")
        if (not isinstance(anonymous_id, str) or anonymous_id in by_id
                or not isinstance(image_hash, str)):
            raise ValueError("Anonymous preference items are missing or duplicated.")
        by_id[anonymous_id] = packet
        by_image[image_hash].append(packet)
    if len(by_image) != 78:
        raise ValueError("Expected 78 distinct anonymous image pairs.")
    mate: dict[str, str] = {}
    for pair in by_image.values():
        if len(pair) != 2:
            raise ValueError("Each image must have exactly two judge orders.")
        left, right = pair
        if (left["target_emotion"] != right["target_emotion"]
                or left["answer_A_sha256"] != right["answer_B_sha256"]
                or left["answer_B_sha256"] != right["answer_A_sha256"]):
            raise ValueError("Anonymous judge orders are not exact answer swaps.")
        mate[left["anonymous_item_id"]] = right["anonymous_item_id"]
        mate[right["anonymous_item_id"]] = left["anonymous_item_id"]
    return by_id, mate


def _source(
    source_dir: Path, expected_hashes: dict[str, str] = _SOURCE_HASHES,
) -> dict[str, Any]:
    """Verify the immutable attempt-002 files before inheriting decisions."""
    if (source_dir / "preference_judgements_lock.json").exists():
        raise ValueError("Recovery source already has a complete preference lock.")
    for name, digest in expected_hashes.items():
        if sha256_file(source_dir / name) != digest:
            raise ValueError(f"Attempt-002 source changed: {name}")
    package = _read_object(source_dir / "preference_package.json")
    config_path = source_dir / "preference_config_private.json"
    config = _read_object(config_path)
    _validate_preference_config(config)
    if (sha256_file(config_path) != package["preference_config_sha256"]
            or config["model_id"] != package["judge_model_id"]):
        raise ValueError("Attempt-002 judge configuration changed.")
    packet_path = source_dir / "preference_items_private.jsonl"
    packets = _read_jsonl(packet_path)
    by_id, mate = _pairs(packets)
    scores = _validate_judgements(
        source_dir / "preference_judgements_private.jsonl", packet_path,
        package, require_complete=False,
    )
    if len(scores) != 149 or set(scores) != set(list(by_id)[:149]):
        raise ValueError("Attempt 002 does not have the documented first 149 decisions.")
    _validate_script_trace(source_dir, package, scores, require_post_check=False)
    failed_id = packets[149]["anonymous_item_id"]
    attempts = _read_jsonl(source_dir / "preference_attempts_private.jsonl")
    final_attempts = [row for row in attempts if row["anonymous_item_id"] == failed_id]
    if (len(final_attempts) != config["retry_policy"]["max_attempts"]
            or [row["attempt"] for row in final_attempts]
            != list(range(1, len(final_attempts) + 1))
            or any(row["status"] != "error" for row in final_attempts)
            or any(row["anonymous_item_id"] not in (set(scores) | {failed_id})
                   for row in attempts)):
        raise ValueError("Attempt-002 terminal failure differs from the documented trace.")
    raw = _read_jsonl(source_dir / "preference_raw_responses_private.jsonl")
    terminal_raw = [row for row in raw if row["anonymous_item_id"] == failed_id]
    if (len(terminal_raw) != len(final_attempts)
            or any(row["response"]["choices"][0]["finish_reason"] != "length"
                   or row["response"].get("usage", {}).get("completion_tokens") != 2048
                   for row in terminal_raw)):
        raise ValueError("Attempt-002 terminal responses do not prove output truncation.")
    if mate[failed_id] not in scores:
        raise ValueError("The failed item needs one already scored swapped-order mate.")
    return {
        "package": package, "config": config, "packets": packets,
        "by_id": by_id, "mate": mate, "scores": scores,
        "initial_failed_id": failed_id,
        "source_hashes": {
            name: sha256_file(source_dir / name)
            for name in (
                *expected_hashes,
                "preference_config_private.json",
                "preference_mapping_private.json",
                "model_digest_receipt_private.json",
                "preflight_receipt_private.json",
            )
        },
    }


def _verify_gate(gate_path: Path, source: dict[str, Any]) -> dict[str, Any]:
    # The independent calibration verifier checks human decisions, 32 raw model
    # calls, the dual-order arithmetic, and their model/digest receipts.
    from .preference_calibration import verify_calibration_gate

    gate = verify_calibration_gate(gate_path)
    package = source["package"]
    expected = {
        "pair_count": 16,
        "judge_config_sha256": package["preference_config_sha256"],
        "judge_model_id": package["judge_model_id"],
        "model_digest_sha256": package["model_digest_sha256"],
        "source_package_sha256": source["source_hashes"]["preference_package.json"],
    }
    if any(gate.get(key) != value for key, value in expected.items()):
        raise ValueError("Human calibration gate does not bind attempt-002 judge.")
    if (gate.get("human_clear_count", 0) < 10
            or gate.get("clear_agreement_count", 0) * 5
            < gate["human_clear_count"] * 4
            or gate.get("hallucination_false_winners") != 0):
        raise ValueError("Human calibration gate failed its predeclared thresholds.")
    return gate


def _reference_summary(contents: bytes) -> dict[str, int]:
    """Check only the shape of an unfinished blind draft, without scoring it."""
    try:
        decoded = contents.decode("utf-8-sig")
        reader = csv.DictReader(io.StringIO(decoded, newline=""))
        if reader.fieldnames != list(_REFERENCE_FIELDS):
            raise ValueError("Human reference draft has unexpected CSV columns.")
        rows = list(reader)
    except (UnicodeError, csv.Error) as error:
        raise ValueError("Human reference draft is not a readable UTF-8 CSV.") from error
    if len(rows) != 16:
        raise ValueError("Human reference draft must have exactly 16 blind image rows.")
    codes: set[str] = set()
    reviewers: set[str] = set()
    for row in rows:
        if None in row or any(value is None for value in row.values()):
            raise ValueError("Human reference draft has missing or extra CSV cells.")
        code = row["image_code"]
        if not re.fullmatch(r"D\d{3}", code) or code in codes:
            raise ValueError("Human reference draft has duplicate or invalid image codes.")
        codes.add(code)
        reviewer = row["reviewer_id"].strip()
        if not reviewer:
            raise ValueError("Human reference draft has a missing reviewer ID.")
        reviewers.add(reviewer)
    if len(reviewers) != 1:
        raise ValueError("Human reference draft must have one reviewer ID.")
    return {
        "row_count": len(rows),
        "preferred_nonempty_count": sum(bool(row["preferred"].strip()) for row in rows),
        "hallucination_nonempty_count": sum(
            bool(row["obvious_hallucinated_worse"].strip()) for row in rows
        ),
        "reason_nonempty_count": sum(bool(row["reason"].strip()) for row in rows),
    }


def prepare_command(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Bind audited attempt 002 and explicit calibration provenance into an anonymous recovery run.")
    parser.add_argument("--source-package", required=True, type=Path)
    evidence = parser.add_mutually_exclusive_group(required=True)
    evidence.add_argument("--calibration-gate", type=Path)
    evidence.add_argument("--human-reference-csv", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    _require_private_destination(args.output_dir)
    source_dir = args.source_package.resolve()
    recovery_dir = args.output_dir.resolve()
    if recovery_dir == source_dir or recovery_dir.is_relative_to(source_dir):
        raise ValueError("Recovery must have its own private directory.")
    if recovery_dir.exists():
        raise FileExistsError("Recovery output directory already exists.")
    source = _source(source_dir)
    if args.calibration_gate is not None:
        gate_path = args.calibration_gate.resolve()
        _verify_gate(gate_path, source)
        mode = "passed_human_gate"
        evidence_sha256 = sha256_file(gate_path)
        reference_contents = None
        reference_summary = None
    else:
        reference_contents = args.human_reference_csv.read_bytes()
        reference_summary = _reference_summary(reference_contents)
        mode = "reference_only_unverified"
        evidence_sha256 = hashlib.sha256(reference_contents).hexdigest()
    recovery_dir.mkdir(parents=True)
    if reference_contents is not None:
        _write_bytes_if_absent(recovery_dir / _REFERENCE, reference_contents)
    metadata = {
        "schema_version": 1,
        "kind": "eval80_preference_attempt002_full_traversal_recovery",
        "source_package_dir": str(source_dir),
        "source_hashes": source["source_hashes"],
        "calibration_mode": mode,
        "calibration_evidence_sha256": evidence_sha256,
        "recovery_source_sha256": sha256_file(Path(__file__)),
        "original_image_count": 78,
        "initial_excluded_anonymous_image_sha256": source["by_id"][source["initial_failed_id"]]["image_sha256"],
        "initial_exclusion_reason": "attempt002_item150_two_length_responses_at_2048_tokens",
    }
    if mode == "passed_human_gate":
        metadata.update({"calibration_gate_path": str(gate_path),
                         "calibration_gate_sha256": evidence_sha256})
    else:
        metadata.update({"human_reference_csv_name": _REFERENCE,
                         "human_reference_csv_sha256": evidence_sha256,
                         "human_reference_summary": reference_summary})
    _write_json(recovery_dir / _META, metadata)
    _write_json(recovery_dir / _STATE, {
        "schema_version": 1,
        "recovery_package_sha256": sha256_file(recovery_dir / _META),
        "runtime_model_identity": {"mode": _identity_mode(source["config"]), "checks": []},
    })
    (recovery_dir / _LOG).touch(exist_ok=False)
    print(json.dumps({"prepared": str(recovery_dir), "calibration_mode": mode,
                      "prior_valid_decisions": 149,
                      "initial_technical_exclusions": 1}, indent=2))
    return 0


def _load(recovery_dir: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    metadata = _read_object(recovery_dir / _META)
    if (metadata.get("schema_version") != 1
            or metadata.get("kind") != "eval80_preference_attempt002_full_traversal_recovery"
            or sha256_file(Path(__file__)) != metadata.get("recovery_source_sha256")):
        raise ValueError("Recovery code or metadata differs from the prepared version.")
    source_dir = Path(metadata["source_package_dir"])
    source = _source(source_dir)
    if source["source_hashes"] != metadata["source_hashes"]:
        raise ValueError("Attempt-002 source snapshot changed after recovery preparation.")
    mode = metadata.get("calibration_mode")
    if mode == "passed_human_gate":
        if "human_reference_csv_name" in metadata or "human_reference_csv_sha256" in metadata:
            raise ValueError("Calibration provenance mixes gate and draft reference.")
        gate_path = Path(metadata["calibration_gate_path"])
        if (sha256_file(gate_path) != metadata["calibration_gate_sha256"]
                or metadata["calibration_evidence_sha256"] != metadata["calibration_gate_sha256"]):
            raise ValueError("Human calibration gate changed after preparation.")
        source["calibration_gate"] = _verify_gate(gate_path, source)
    elif mode == "reference_only_unverified":
        if "calibration_gate_path" in metadata or "calibration_gate_sha256" in metadata:
            raise ValueError("Calibration provenance mixes gate and draft reference.")
        if (metadata.get("human_reference_csv_name") != _REFERENCE
                or metadata.get("calibration_evidence_sha256")
                != metadata.get("human_reference_csv_sha256")):
            raise ValueError("Human draft reference provenance is invalid.")
        reference_path = recovery_dir / _REFERENCE
        if sha256_file(reference_path) != metadata["human_reference_csv_sha256"]:
            raise ValueError("Human draft reference changed after preparation.")
        source["human_reference_summary"] = _reference_summary(reference_path.read_bytes())
        if source["human_reference_summary"] != metadata.get("human_reference_summary"):
            raise ValueError("Human draft reference summary changed after preparation.")
    else:
        raise ValueError("Unknown calibration provenance mode.")
    source["calibration_mode"] = mode
    if (metadata["initial_excluded_anonymous_image_sha256"]
            != source["by_id"][source["initial_failed_id"]]["image_sha256"]):
        raise ValueError("Initial technical exclusion changed.")
    state = _read_object(recovery_dir / _STATE)
    if (set(state) != {"schema_version", "recovery_package_sha256", "runtime_model_identity"}
            or state["schema_version"] != 1
            or state["recovery_package_sha256"] != sha256_file(recovery_dir / _META)):
        raise ValueError("Recovery runtime state differs from prepared metadata.")
    identity = state["runtime_model_identity"]
    if (not isinstance(identity, dict) or identity.get("mode") != _identity_mode(source["config"])
            or not isinstance(identity.get("checks"), list)):
        raise ValueError("Recovery model identity state is malformed.")
    if identity["checks"]:
        _validate_runtime_identity(identity, source["config"], source["package"],
                                   complete=False)
    return metadata, source, state


def _validate_log(
    log_path: Path, source: dict[str, Any], *, require_complete: bool,
) -> tuple[dict[str, dict[str, Any]], set[str], dict[str, int]]:
    """Rebuild all outcomes from the durable new trace, never from a score file."""
    package, config = source["package"], source["config"]
    by_id, mate = source["by_id"], source["mate"]
    max_attempts = config["retry_policy"]["max_attempts"]
    rows = _read_jsonl(log_path)
    own_scores: dict[str, dict[str, Any]] = {}
    counts: Counter[str] = Counter()
    errors: Counter[str] = Counter()
    excluded: set[str] = {source["by_id"][source["initial_failed_id"]]["image_sha256"]}
    for row in rows:
        if set(row) != {"anonymous_item_id", "attempt", "packet_sha256", "request_sha256",
                        "status", "response", "judgement", "error_type"}:
            raise ValueError("Recovery attempt trace has unknown fields.")
        anonymous_id = row["anonymous_item_id"]
        if anonymous_id not in by_id or anonymous_id in source["scores"]:
            raise ValueError("Recovery attempted an unknown or inherited scored item.")
        packet = by_id[anonymous_id]
        image_hash = packet["image_sha256"]
        if image_hash in excluded or anonymous_id in own_scores:
            raise ValueError("Recovery attempted a completed or excluded image.")
        counts[anonymous_id] += 1
        if (row["attempt"] != counts[anonymous_id] or counts[anonymous_id] > max_attempts
                or row["packet_sha256"] != _packet_sha256(packet)
                or row["request_sha256"] != _request_sha256(_make_payload(packet, config))):
            raise ValueError("Recovery attempt differs from frozen anonymous request.")
        response = row["response"]
        if row["status"] == "success":
            if (not isinstance(response, dict) or row["error_type"] is not None
                    or row["judgement"] != _extract_judgement(response, packet, package)):
                raise ValueError("Recovery success differs from raw model response.")
            own_scores[anonymous_id] = row["judgement"]
        elif row["status"] == "error":
            if (row["judgement"] is not None or row["error_type"] not in {
                    "ValueError", "RuntimeError", "KeyError", "JSONDecodeError", "OSError"}):
                raise ValueError("Recovery failure lacks a valid technical error.")
            if response is not None:
                try:
                    _extract_judgement(response, packet, package)
                except (ValueError, RuntimeError, KeyError, OSError, json.JSONDecodeError) as error:
                    actual_type = "OSError" if isinstance(error, OSError) else type(error).__name__
                    if actual_type != row["error_type"]:
                        raise ValueError("Recovery failure type differs from raw model response.") from error
                else:
                    raise ValueError("Recovery mislabeled a valid response as a failure.")
            errors[row["error_type"]] += 1
            if counts[anonymous_id] == max_attempts:
                excluded.add(image_hash)
        else:
            raise ValueError("Recovery attempt status must be success or error.")
    all_scores = {**source["scores"], **own_scores}
    if require_complete:
        for anonymous_id, packet in by_id.items():
            if packet["image_sha256"] not in excluded and anonymous_id not in all_scores:
                raise ValueError("Full traversal is incomplete; a nonexcluded item has no decision.")
        for anonymous_id, packet in by_id.items():
            if packet["image_sha256"] in excluded:
                continue
            if mate[anonymous_id] not in all_scores:
                raise ValueError("A retained image lacks its swapped-order decision.")
    return all_scores, excluded, dict(errors)


def run_command(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Finish every eligible anonymous Eval80 pair with frozen Gemma attempt-002 settings.")
    parser.add_argument("--recovery-dir", required=True, type=Path)
    parser.add_argument("--api-key-env", default="EVAL80_JUDGE_API_KEY")
    args = parser.parse_args(argv)
    recovery_dir = args.recovery_dir.resolve()
    if (recovery_dir / _LOCK).exists():
        raise FileExistsError("Recovery decisions are already locked; use report.")
    metadata, source, state = _load(recovery_dir)
    config, package = source["config"], source["package"]
    if not _is_loopback_endpoint(config["endpoint"]):
        raise ValueError("Recovery judge must remain on a loopback endpoint.")
    scores, excluded, _ = _validate_log(recovery_dir / _LOG, source, require_complete=False)
    mode = _identity_mode(config)
    if mode == "ollama_api_tags":
        state["runtime_model_identity"]["checks"].append(_probe_ollama_digest(
            config["endpoint"], config["model_id"], package["model_digest_sha256"],
            min(config["retry_policy"]["timeout_seconds"], 30), phase="before",
        ))
    (recovery_dir / _STATE).write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n",
                                        encoding="utf-8")
    retry = config["retry_policy"]
    max_attempts = retry["max_attempts"]
    prior_count = Counter(row["anonymous_item_id"] for row in _read_jsonl(recovery_dir / _LOG))
    key = os.environ.get(args.api_key_env)
    for position, packet in enumerate(source["packets"], start=1):
        anonymous_id = packet["anonymous_item_id"]
        if anonymous_id in scores or packet["image_sha256"] in excluded:
            continue
        payload = _make_payload(packet, config)
        request_hash = _request_sha256(payload)
        for attempt in range(prior_count[anonymous_id] + 1, max_attempts + 1):
            response: dict[str, Any] | None = None
            judgement: dict[str, Any] | None = None
            error_type: str | None = None
            try:
                response = _post_chat(config["endpoint"], payload, key,
                                      retry["timeout_seconds"])
                judgement = _extract_judgement(response, packet, package)
            except (ValueError, RuntimeError, KeyError, OSError, json.JSONDecodeError) as error:
                error_type = "OSError" if isinstance(error, OSError) else type(error).__name__
            _append_jsonl(recovery_dir / _LOG, {
                "anonymous_item_id": anonymous_id, "attempt": attempt,
                "packet_sha256": _packet_sha256(packet), "request_sha256": request_hash,
                "status": "success" if judgement is not None else "error",
                "response": response, "judgement": judgement, "error_type": error_type,
            })
            if judgement is not None:
                scores[anonymous_id] = judgement
                print(f"[{position:03d}/156] anonymous preference item complete")
                break
            if attempt == max_attempts:
                excluded.add(packet["image_sha256"])
                print(f"[{position:03d}/156] anonymous image excluded after technical retry exhaustion")
            else:
                time.sleep(min(retry["backoff_seconds"] * attempt, 30))
    scores, excluded, errors = _validate_log(recovery_dir / _LOG, source,
                                             require_complete=True)
    if mode == "ollama_api_tags":
        state["runtime_model_identity"]["checks"].append(_probe_ollama_digest(
            config["endpoint"], config["model_id"], package["model_digest_sha256"],
            min(retry["timeout_seconds"], 30), phase="after",
        ))
        (recovery_dir / _STATE).write_text(
            json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _validate_runtime_identity(state["runtime_model_identity"], config, package,
                               complete=True)
    lock = {
        "schema_version": 1, "kind": "eval80_preference_recovery_decision_lock",
        "recovery_package_sha256": sha256_file(recovery_dir / _META),
        "source_package_sha256": metadata["source_hashes"]["preference_package.json"],
        "calibration_mode": metadata["calibration_mode"],
        "calibration_evidence_sha256": metadata["calibration_evidence_sha256"],
        "recovery_attempts_sha256": sha256_file(recovery_dir / _LOG),
        "recovery_run_state_sha256": sha256_file(recovery_dir / _STATE),
        "original_images": 78, "included_images": 78 - len(excluded),
        "excluded_images": len(excluded), "included_judgements": 2 * (78 - len(excluded)),
        "technical_error_types": errors,
        "locked_at_utc": _now(),
    }
    _write_json(recovery_dir / _LOCK, lock)
    print(json.dumps({key: lock[key] for key in (
        "included_images", "excluded_images", "included_judgements",
        "technical_error_types")}, indent=2))
    return 0


def _aggregate(
    source: dict[str, Any], scores: dict[str, dict[str, Any]], excluded: set[str],
    mapping: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    packets = source["by_id"]
    by_image = _validate_mapping(mapping, packets)
    consensus: Counter[str] = Counter()
    item_votes: Counter[str] = Counter()
    emotions: dict[str, Counter[str]] = defaultdict(Counter)
    private_cases: list[dict[str, Any]] = []
    for blind_id, pair in sorted(by_image.items()):
        image_hash = packets[pair[0][0]]["image_sha256"]
        emotion = pair[0][1]["target_emotion"]
        if image_hash in excluded:
            emotions[emotion]["excluded"] += 1
            private_cases.append({"blind_id": blind_id, "outcome": "technical_exclusion"})
            continue
        votes: list[str] = []
        items: list[dict[str, Any]] = []
        identical = all(packets[anonymous_id]["answer_A_sha256"]
                        == packets[anonymous_id]["answer_B_sha256"]
                        for anonymous_id, _ in pair)
        for anonymous_id, item in sorted(pair, key=lambda entry: entry[1]["variant"]):
            winner = scores[anonymous_id]["winner"]
            vote = "tie" if winner == "tie" else item[f"{winner}_condition"]
            votes.append(vote)
            item_votes[vote] += 1
            items.append({"anonymous_item_id": anonymous_id, "winner": winner,
                          "condition_vote": vote,
                          "reason": scores[anonymous_id]["reason"]})
        outcome = "tie" if identical else votes[0] if votes[0] == votes[1] else "unstable"
        consensus[outcome] += 1
        emotions[emotion][outcome] += 1
        private_cases.append({"blind_id": blind_id, "outcome": outcome,
                              "identical_answer_rule_applied": identical, "items": items})
    included = 78 - len(excluded)
    if sum(consensus.values()) != included or sum(item_votes.values()) != 2 * included:
        raise ValueError("Preference report does not cover all retained paired images.")
    inherited_used = sum(
        source["by_id"][anonymous_id]["image_sha256"] not in excluded
        for anonymous_id in source["scores"]
    )
    net = consensus["C"] - consensus["B1"]
    uncertain = consensus["unstable"] + len(excluded)
    calibrated = source.get("calibration_mode", "passed_human_gate") == "passed_human_gate"
    summary = {
        "schema_version": 1,
        "evaluation_set": "Eval80-v1",
        "status": "final_automated_evaluation_with_technical_exclusions",
        "metric": ("calibrated_image_aware_pairwise_guidance_preference" if calibrated
                   else "uncalibrated_image_aware_pairwise_guidance_preference"),
        "original_scope_images": 78,
        "included_images": included,
        "included_judgements": 2 * included,
        "technical_excluded_images": len(excluded),
        "attempt002_valid_decisions_inherited": len(source["scores"]),
        "attempt002_decisions_used_after_exclusion": inherited_used,
        "new_decisions_used_after_exclusion": 2 * included - inherited_used,
        "judge_model_id": source["package"]["judge_model_id"],
        "source_records_sha256": {
            "B1": source["package"]["b1_records_sha256"],
            "C": source["package"]["c_records_sha256"],
        },
        "inference_manifest_sha256": source["package"]["inference_manifest_sha256"],
        "selection_manifest_sha256": source["package"]["selection_manifest_sha256"],
        "preference_config_sha256": source["package"]["preference_config_sha256"],
        "model_digest_sha256": source["package"]["model_digest_sha256"],
        "image_consensus": {
            "C_wins": consensus["C"], "B1_wins": consensus["B1"],
            "ties": consensus["tie"], "swap_disagreements_unstable": consensus["unstable"],
        },
        "item_votes": {"C": item_votes["C"], "B1": item_votes["B1"],
                       "tie": item_votes["tie"]},
        "observed_net_c_minus_b1_count": net,
        "observed_net_c_minus_b1_rate": net / included if included else None,
        "original_scope_worst_case_net_count": net - uncertain,
        "original_scope_worst_case_net_rate": (net - uncertain) / 78,
        "original_scope_best_case_net_rate": (net + uncertain) / 78,
        "emotion_coverage": {emotion: {"included": sum(counts.values()) - counts["excluded"],
                                       "excluded": counts["excluded"]}
                             for emotion, counts in sorted(emotions.items())},
        "interpretation_note": (
            ("This is the final calibrated automated judge result" if calibrated else
             "This is a final automated judge result with incomplete human calibration; "
             "the draft was retained only as a reference and provides no agreement score")
            + " for the reported Eval80 technical-exclusion scope. The post-failure "
            "exclusion rule was adopted after two failed complete-set attempts, so the "
            "original 78-image bound is also shown. This relative judge preference cannot "
            "by itself establish a human-perceived guidance improvement, a five-dimension "
            "human score, or formal adapter acceptance."
        ),
    }
    return summary, {"schema_version": 1, "cases": private_cases}


def report_command(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Open the B1/C key only after every retained anonymous pair is locked.")
    parser.add_argument("--recovery-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    recovery_dir = args.recovery_dir.resolve()
    metadata, source, state = _load(recovery_dir)
    scores, excluded, errors = _validate_log(recovery_dir / _LOG, source,
                                             require_complete=True)
    _validate_runtime_identity(state["runtime_model_identity"], source["config"],
                               source["package"], complete=True)
    lock = _read_object(recovery_dir / _LOCK)
    expected = {
        "schema_version": 1, "kind": "eval80_preference_recovery_decision_lock",
        "recovery_package_sha256": sha256_file(recovery_dir / _META),
        "source_package_sha256": metadata["source_hashes"]["preference_package.json"],
        "calibration_mode": metadata["calibration_mode"],
        "calibration_evidence_sha256": metadata["calibration_evidence_sha256"],
        "recovery_attempts_sha256": sha256_file(recovery_dir / _LOG),
        "recovery_run_state_sha256": sha256_file(recovery_dir / _STATE),
        "original_images": 78, "included_images": 78 - len(excluded),
        "excluded_images": len(excluded), "included_judgements": 2 * (78 - len(excluded)),
        "technical_error_types": errors,
    }
    if set(lock) != set(expected) | {"locked_at_utc"} or any(
            lock.get(key) != value for key, value in expected.items()):
        raise ValueError("Recovery decision lock differs from the complete anonymous trace.")
    mapping_path = Path(metadata["source_package_dir"]) / "preference_mapping_private.json"
    if sha256_file(mapping_path) != source["package"]["preference_mapping_sha256"]:
        raise ValueError("Private B1/C mapping changed after score lock.")
    summary, private_cases = _aggregate(source, scores, excluded,
                                        _read_object(mapping_path))
    summary["technical_incident_evidence_sha256"] = {
        "attempt002_attempts": metadata["source_hashes"]["preference_attempts_private.jsonl"],
        "attempt002_raw": metadata["source_hashes"]["preference_raw_responses_private.jsonl"],
        "recovery_attempts": sha256_file(recovery_dir / _LOG),
    }
    summary["calibration_mode"] = metadata["calibration_mode"]
    summary["calibration_evidence_sha256"] = metadata["calibration_evidence_sha256"]
    summary["calibration_gate_sha256"] = metadata.get("calibration_gate_sha256")
    summary["human_reference_csv_sha256"] = metadata.get("human_reference_csv_sha256")
    summary["decision_lock_sha256"] = sha256_file(recovery_dir / _LOCK)
    summary["recovery_request_attempts"] = len(_read_jsonl(recovery_dir / _LOG))
    if metadata["calibration_mode"] == "passed_human_gate":
        gate = source["calibration_gate"]
        summary["calibration"] = {
            "mode": "passed_human_gate",
            **{key: gate[key] for key in (
                "pair_count", "human_clear_count", "clear_agreement_count",
                "clear_agreement_rate", "hallucination_false_winners")},
        }
    else:
        summary["calibration"] = {
            "mode": "reference_only_unverified",
            "reference_csv_sha256": metadata["human_reference_csv_sha256"],
            "human_clear_count": None,
            "clear_agreement_count": None,
            "clear_agreement_rate": None,
            "hallucination_false_winners": None,
        }
    summary["technical_excluded_anonymous_positions"] = [
        position for position, packet in enumerate(source["packets"], start=1)
        if packet["image_sha256"] in excluded
    ]
    summary["new_technical_error_types"] = errors
    position_by_id = {
        packet["anonymous_item_id"]: position
        for position, packet in enumerate(source["packets"], start=1)
    }
    summary["recovery_error_evidence"] = []
    for row in _read_jsonl(recovery_dir / _LOG):
        if row["status"] != "error":
            continue
        response = row["response"]
        choices = response.get("choices") if isinstance(response, dict) else None
        choice = choices[0] if isinstance(choices, list) and choices else None
        usage = response.get("usage") if isinstance(response, dict) else None
        summary["recovery_error_evidence"].append({
            "anonymous_position": position_by_id[row["anonymous_item_id"]],
            "retry_attempt": row["attempt"],
            "error_type": row["error_type"],
            "finish_reason": choice.get("finish_reason") if isinstance(choice, dict) else None,
            "prompt_tokens": usage.get("prompt_tokens") if isinstance(usage, dict) else None,
            "completion_tokens": usage.get("completion_tokens") if isinstance(usage, dict) else None,
            "max_tokens": source["config"]["decoding"]["max_tokens"],
            "image_excluded": source["by_id"][row["anonymous_item_id"]]["image_sha256"]
            in excluded,
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    private_path = recovery_dir / "recovery_cases_private.json"
    _write_bytes_if_absent(private_path, (json.dumps(
        private_cases, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    _write_bytes_if_absent(args.output, (json.dumps(
        summary, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    print(json.dumps({"status": summary["status"],
                      "image_consensus": summary["image_consensus"],
                      "output": str(args.output)}, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "report"))
    if not arguments or arguments[0] in {"-h", "--help"}:
        parser.print_help()
        return 0
    command, *rest = arguments
    if command not in {"prepare", "run", "report"}:
        parser.error(f"unknown command: {command}")
    return {"prepare": prepare_command, "run": run_command,
            "report": report_command}[command](rest)


if __name__ == "__main__":
    raise SystemExit(main())
