"""Blind human-to-Gemma calibration gate for the final Eval80 preference run.

The human decisions are locked before this module makes any model call. The
BridgeTrain development pairs stay in ignored private storage; a passing gate
is verified from its image/answer hashes and raw model responses, not a typed
``status`` field alone.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import time
from typing import Any

from .core import sha256_file
from .preference_benchmark import _packet_sha256, _validate_preference_config
from .preference_judge_runner import _extract_judgement, _make_payload, _probe_ollama_digest, _request_sha256
from .quality_benchmark import _require_private_destination
from .quality_judge_runner import _post_chat


_CODE = re.compile(r"D[0-9]{3}\Z")
_ORIENTATIONS = ("normal", "swapped")
_AGREEMENT_FLOOR = 0.8
_MIN_CLEAR = 10


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"Expected JSONL objects: {path}")
    return rows


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")


def _relative_file(parent: Path, name: str) -> Path:
    path = (parent / name).resolve()
    if not path.is_relative_to(parent.resolve()) or not path.is_file():
        raise ValueError("Calibration evidence path escapes its private directory or is missing.")
    return path


def _load_human_pairs(lock_path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, Any]]]:
    lock_path = lock_path.resolve()
    lock = _object(lock_path)
    manifest_path = lock_path.parent / "manifest.json"
    manifest = _object(manifest_path)
    review_html_path = lock_path.parent / "review.html"
    translations_path = lock_path.parent / "answer_translations_zh.json"
    review_html_sha256 = sha256_file(review_html_path)
    translations_sha256 = sha256_file(translations_path)
    if (lock.get("kind") != "pre_c_pairwise_human_calibration_lock"
            or lock.get("status") != "locked" or lock.get("image_count") != 16
            or lock.get("manifest_sha256") != sha256_file(manifest_path)
            or lock.get("review_html_sha256") != review_html_sha256
            or lock.get("answer_translations_zh_sha256") != translations_sha256
            or manifest.get("kind") != "pre_c_pairwise_human_calibration_manifest"
            or manifest.get("version") != 2 or manifest.get("image_count") != 16
            or manifest.get("review_html_sha256") != review_html_sha256
            or manifest.get("answer_translations_zh_sha256") != translations_sha256):
        raise ValueError("Human calibration lock/manifest identity is invalid.")
    submission = lock_path.parent / "human_submission.csv"
    if (lock.get("submission_file") != "human_submission.csv"
            or lock.get("submission_sha256") != sha256_file(submission)):
        raise ValueError("Human calibration submission differs from the locked CSV.")
    source = lock_path.parent.parent / "review" / "review.csv"
    if not source.is_file() or sha256_file(source) != manifest.get("source_review_csv_sha256"):
        raise ValueError("BridgeTrain blinded source CSV differs from the calibration manifest.")
    manifest_rows = manifest.get("records")
    human_rows = lock.get("records")
    if (not isinstance(manifest_rows, list) or len(manifest_rows) != 16
            or not isinstance(human_rows, list) or len(human_rows) != 16
            or not isinstance(lock.get("reviewer_id"), str)
            or not lock["reviewer_id"].strip()):
        raise ValueError("Human calibration lock does not contain 16 reviewed pairs.")
    with submission.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != [
                "image_code", "reviewer_id", "preferred",
                "obvious_hallucinated_worse", "reason"]:
            raise ValueError("Human calibration CSV columns changed after locking.")
        submitted = list(reader)
    if len(submitted) != 16 or any(None in row or any(value is None for value in row.values())
                                   for row in submitted):
        raise ValueError("Human calibration CSV is incomplete or has extra cells.")
    normalized_submission = {}
    for row in submitted:
        code = row["image_code"].strip()
        if code in normalized_submission or row["reviewer_id"].strip() != lock["reviewer_id"]:
            raise ValueError("Human calibration CSV image or reviewer differs from the lock.")
        normalized_submission[code] = {
            "image_code": code, "preferred": row["preferred"].strip(),
            "obvious_hallucinated_worse": row["obvious_hallucinated_worse"].strip(),
            "reason": row["reason"].strip(),
        }
    human = {}
    for row in human_rows:
        if (not isinstance(row, dict) or set(row) != {
                "image_code", "preferred", "obvious_hallucinated_worse", "reason"}
                or row["preferred"] not in {"A", "B", "tie"}
                or row["obvious_hallucinated_worse"] not in {"A", "B", "none", "uncertain"}
                or not isinstance(row["reason"], str) or not row["reason"].strip()):
            raise ValueError("Human calibration decision is incomplete.")
        code = row["image_code"]
        if not isinstance(code, str) or not _CODE.fullmatch(code) or code in human:
            raise ValueError("Human calibration image code is missing or duplicated.")
        if (row["obvious_hallucinated_worse"] in {"A", "B"}
                and row["preferred"] in {row["obvious_hallucinated_worse"], "tie"}):
            raise ValueError("Human calibration preferred answer is marked as obviously hallucinated worse.")
        human[code] = row
    if (normalized_submission != human
            or lock.get("clear_preference_count") != sum(
                row["preferred"] != "tie" for row in human.values())):
        raise ValueError("Human calibration lock decisions differ from the submitted CSV.")
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    if len(source_rows) != 32:
        raise ValueError("BridgeTrain blinded source must contain exactly 32 answers.")
    translations = _object(translations_path)
    if (translations.get("kind") != "blind_answer_translations_zh"
            or translations.get("version") != 1
            or translations.get("source_review_csv_sha256") != sha256_file(source)
            or not isinstance(translations.get("records"), list)
            or len(translations["records"]) != 32):
        raise ValueError("Chinese translation file differs from the blinded source.")
    source_answers = {row.get("anonymous_item_id"): row.get("model_response")
                      for row in source_rows}
    translated_ids: set[str] = set()
    for row in translations["records"]:
        if (not isinstance(row, dict) or set(row) != {
                "anonymous_item_id", "source_response_sha256", "translation_zh"}):
            raise ValueError("Chinese translation record has unexpected fields.")
        item_id = row["anonymous_item_id"]
        source_response = source_answers.get(item_id)
        if (not isinstance(item_id, str) or item_id in translated_ids
                or not isinstance(source_response, str)
                or row["source_response_sha256"] != hashlib.sha256(
                    source_response.encode("utf-8")).hexdigest()
                or not isinstance(row["translation_zh"], str)
                or not row["translation_zh"].strip()):
            raise ValueError("Chinese translation does not bind an anonymous answer.")
        translated_ids.add(item_id)
    if translated_ids != set(source_answers):
        raise ValueError("Chinese translations do not cover all blind answers.")
    answers: dict[str, dict[str, dict[str, str]]] = {}
    for row in source_rows:
        code = row.get("image_code", "")
        item_id = row.get("anonymous_item_id", "")
        if not _CODE.fullmatch(code) or item_id not in {f"{code}-A", f"{code}-B"}:
            raise ValueError("BridgeTrain source contains an unexpected anonymous answer ID.")
        letter = item_id[-1]
        if letter in answers.setdefault(code, {}):
            raise ValueError("BridgeTrain source duplicates an answer.")
        answers[code][letter] = row
    if (set(answers) != set(human)
            or any(set(group) != {"A", "B"} for group in answers.values())):
        raise ValueError("Human lock does not cover the blinded source pairs.")
    source_root = source.parent
    manifest_by_code = {}
    for row in manifest_rows:
        if not isinstance(row, dict) or set(row) != {
                "image_code", "image_file", "image_sha256", "target_emotion",
                "response_a_sha256", "response_b_sha256"}:
            raise ValueError("Calibration manifest record has unexpected fields.")
        code = row["image_code"]
        if code in manifest_by_code or code not in answers:
            raise ValueError("Calibration manifest duplicates or changes an image code.")
        pair = answers[code]
        if (pair["A"]["image_file"] != pair["B"]["image_file"]
                or pair["A"]["target_emotion"] != pair["B"]["target_emotion"]
                or row["image_file"] != pair["A"]["image_file"]
                or row["target_emotion"] != pair["A"]["target_emotion"]
                or row["response_a_sha256"] != hashlib.sha256(
                    pair["A"]["model_response"].encode("utf-8")).hexdigest()
                or row["response_b_sha256"] != hashlib.sha256(
                    pair["B"]["model_response"].encode("utf-8")).hexdigest()
                or row["image_sha256"] != sha256_file(_relative_file(source_root, row["image_file"]))):
            raise ValueError("Calibration manifest image or anonymous answers differ from source.")
        manifest_by_code[code] = row
    return lock, manifest, {code: {"manifest": manifest_by_code[code], "human": human[code],
                                   "answers": answers[code], "image_root": str(source_root)}
                            for code in sorted(answers)}


def _packet(code: str, orientation: str, pair: dict[str, Any]) -> dict[str, Any]:
    row = pair["manifest"]
    first, second = (("A", "B") if orientation == "normal" else ("B", "A"))
    return {
        "anonymous_item_id": f"{code}-{orientation}",
        "image_path": str(_relative_file(Path(pair["image_root"]), row["image_file"])),
        "image_sha256": row["image_sha256"],
        "target_emotion": row["target_emotion"],
        "answer_A": pair["answers"][first]["model_response"],
        "answer_A_sha256": row[f"response_{first.lower()}_sha256"],
        "answer_B": pair["answers"][second]["model_response"],
        "answer_B_sha256": row[f"response_{second.lower()}_sha256"],
    }


def _standard_winner(winner: str, orientation: str) -> str:
    if winner == "tie" or orientation == "normal":
        return winner
    return {"A": "B", "B": "A"}[winner]


def _measure(pairs: dict[str, dict[str, Any]], decisions: list[dict[str, Any]]) -> dict[str, Any]:
    votes = {(row["image_code"], row["orientation"]): row["winner"] for row in decisions}
    if len(votes) != 32 or set(votes) != {
            (code, orientation) for code in pairs for orientation in _ORIENTATIONS}:
        raise ValueError("Calibration requires exactly two completed orderings for all 16 pairs.")
    clear = agreed = false_hallucination = stable_ties = unstable = 0
    for code, pair in pairs.items():
        human = pair["human"]
        normal = _standard_winner(votes[code, "normal"], "normal")
        swapped = _standard_winner(votes[code, "swapped"], "swapped")
        stable = normal == swapped
        if not stable:
            unstable += 1
        elif normal == "tie":
            stable_ties += 1
        if human["preferred"] != "tie":
            clear += 1
            if stable and normal == human["preferred"]:
                agreed += 1
        if (stable and normal in {"A", "B"}
                and normal == human["obvious_hallucinated_worse"]):
            false_hallucination += 1
    rate = agreed / clear if clear else 0.0
    return {
        "pair_count": 16, "judgement_count": 32,
        "human_clear_count": clear,
        "clear_agreement_count": agreed,
        "clear_agreement_rate": rate,
        "hallucination_false_winners": false_hallucination,
        "stable_ties": stable_ties, "order_unstable": unstable,
        "status": ("passed" if clear >= _MIN_CLEAR and rate >= _AGREEMENT_FLOOR
                   and false_hallucination == 0 else "failed"),
    }


def verify_calibration_gate(gate_path: Path) -> dict[str, Any]:
    """Recompute a passing gate from the locked human inputs and 32 raw replies."""
    gate_path = Path(gate_path).resolve()
    gate = _object(gate_path)
    if gate.get("kind") != "pre_c_pairwise_judge_calibration_gate" or gate.get("schema_version") != 1:
        raise ValueError("Unexpected calibration gate kind or version.")
    lock_path = Path(gate["human_lock_path"]).resolve()
    source_package_dir = Path(gate["source_package_dir"]).resolve()
    source_package_path = source_package_dir / "preference_package.json"
    config_path = source_package_dir / "preference_config_private.json"
    if (sha256_file(lock_path) != gate.get("human_lock_sha256")
            or sha256_file(source_package_path) != gate.get("source_package_sha256")
            or sha256_file(config_path) != gate.get("judge_config_sha256")
            or sha256_file(Path(__file__)) != gate.get("runner_source_sha256")):
        raise ValueError("Calibration gate source, config or runner changed.")
    lock, manifest, pairs = _load_human_pairs(lock_path)
    package, config = _object(source_package_path), _object(config_path)
    _validate_preference_config(config)
    if (gate.get("manifest_sha256") != lock["manifest_sha256"]
            or gate.get("judge_config_sha256") != package.get("preference_config_sha256")
            or gate.get("judge_model_id") != package.get("judge_model_id")
            or gate.get("model_digest_sha256") != package.get("model_digest_sha256")
            or config.get("model_id") != gate.get("judge_model_id")):
        raise ValueError("Calibration gate does not use attempt-002's frozen judge.")
    checks = gate.get("model_digest_checks")
    if (not isinstance(checks, list) or len(checks) != 2
            or [check.get("phase") for check in checks] != ["before", "after"]
            or any(check.get("model_id") != gate["judge_model_id"]
                   or check.get("digest_sha256") != gate["model_digest_sha256"]
                   for check in checks)):
        raise ValueError("Calibration gate lacks matching live model digest checks.")
    decisions_path = gate_path.parent / "calibration_decisions_private.jsonl"
    if sha256_file(decisions_path) != gate.get("decisions_sha256"):
        raise ValueError("Calibration decisions trace changed.")
    attempts_path = gate_path.parent / "calibration_attempts_private.jsonl"
    if sha256_file(attempts_path) != gate.get("attempts_sha256"):
        raise ValueError("Calibration attempt trace changed.")
    attempts = _jsonl(attempts_path)
    decisions = _jsonl(decisions_path)
    if len(decisions) != 32:
        raise ValueError("Calibration decisions trace is incomplete.")
    seen = set()
    accepted_attempts: dict[tuple[str, str], dict[str, Any]] = {}
    attempt_counts: dict[tuple[str, str], int] = {}
    completed_keys: set[tuple[str, str]] = set()
    response_ids: set[str] = set()
    for attempt_row in attempts:
        if set(attempt_row) != {
                "image_code", "orientation", "attempt", "result", "raw_file",
                "raw_sha256", "error_type"}:
            raise ValueError("Calibration attempt trace has unexpected fields.")
        code, orientation = attempt_row["image_code"], attempt_row["orientation"]
        key = (code, orientation)
        if code not in pairs or orientation not in _ORIENTATIONS or key in completed_keys:
            raise ValueError("Calibration attempt trace has unknown or repeated completed item.")
        count = attempt_counts.get(key, 0) + 1
        if (attempt_row["attempt"] != count
                or count > config["retry_policy"]["max_attempts"]):
            raise ValueError("Calibration attempt numbering exceeds frozen retry budget.")
        attempt_counts[key] = count
        raw_name, raw_hash = attempt_row["raw_file"], attempt_row["raw_sha256"]
        if raw_name is None:
            if raw_hash is not None:
                raise ValueError("Calibration attempt has a raw hash without a response file.")
            raw_response = None
        else:
            if raw_name != f"raw/{code}_{orientation}_attempt_{count}.json":
                raise ValueError("Calibration raw response path is invalid.")
            raw_path = _relative_file(gate_path.parent, raw_name)
            if sha256_file(raw_path) != raw_hash:
                raise ValueError("Calibration attempt raw response changed.")
            raw_response = _object(raw_path)
        packet = _packet(code, orientation, pairs[code])
        if attempt_row["result"] == "accepted":
            if raw_response is None or attempt_row["error_type"] is not None:
                raise ValueError("Calibration accepted attempt has no valid raw response.")
            extracted = _extract_judgement(raw_response, packet, package)
            response_id = extracted["response_metadata"]["response_id"]
            if response_id in response_ids:
                raise ValueError("Calibration reused a model response ID.")
            response_ids.add(response_id)
            accepted_attempts[key] = attempt_row
            completed_keys.add(key)
        elif attempt_row["result"] == "invalid":
            if attempt_row["error_type"] not in {
                    "ValueError", "RuntimeError", "KeyError", "JSONDecodeError", "OSError"}:
                raise ValueError("Calibration invalid attempt lacks a frozen error type.")
            if raw_response is not None:
                try:
                    _extract_judgement(raw_response, packet, package)
                except (ValueError, RuntimeError, KeyError, OSError, json.JSONDecodeError) as error:
                    actual_type = "OSError" if isinstance(error, OSError) else type(error).__name__
                    if actual_type != attempt_row["error_type"]:
                        raise ValueError("Calibration raw error differs from trace.") from error
                else:
                    raise ValueError("Calibration labeled a valid reply as invalid.")
        else:
            raise ValueError("Calibration attempt status is invalid.")
    if len(accepted_attempts) != 32:
        raise ValueError("Calibration attempts do not complete all 32 orderings.")
    for row in decisions:
        code, orientation = row.get("image_code"), row.get("orientation")
        if (code not in pairs or orientation not in _ORIENTATIONS
                or (code, orientation) in seen):
            raise ValueError("Calibration decisions contain unknown or repeated pair/order.")
        seen.add((code, orientation))
        packet = _packet(code, orientation, pairs[code])
        expected_request_sha = _request_sha256(_make_payload(packet, config))
        accepted = accepted_attempts[code, orientation]
        if (accepted["raw_file"] != row.get("accepted_raw_file")
                or accepted["raw_sha256"] != row.get("accepted_raw_sha256")):
            raise ValueError("Calibration decision selected a response outside the retry trace.")
        raw_path = _relative_file(gate_path.parent, row.get("accepted_raw_file", ""))
        if (row.get("packet_sha256") != _packet_sha256(packet)
                or row.get("request_sha256") != expected_request_sha
                or row.get("accepted_raw_sha256") != sha256_file(raw_path)):
            raise ValueError("Calibration decision changed from the frozen packet or request.")
        result = _extract_judgement(_object(raw_path), packet, package)
        if (row.get("winner") != result["winner"]
                or row.get("response_id") != result["response_metadata"]["response_id"]):
            raise ValueError("Calibration decision differs from its raw model reply.")
    measured = _measure(pairs, decisions)
    if (measured["status"] != "passed"
            or any(gate.get(key) != value for key, value in measured.items())):
        raise ValueError("Calibration gate did not pass the predeclared human agreement criteria.")
    return {key: gate[key] for key in (
        "human_lock_sha256", "manifest_sha256", "judge_config_sha256",
        "judge_model_id", "model_digest_sha256", "pair_count", "human_clear_count",
        "clear_agreement_count", "clear_agreement_rate", "hallucination_false_winners",
        "decisions_sha256", "runner_source_sha256", "source_package_sha256")}


def run_calibration(lock_path: Path, source_package_dir: Path, output_dir: Path) -> dict[str, Any]:
    lock_path = lock_path.resolve()
    source_package_dir = source_package_dir.resolve()
    output_dir = output_dir.resolve()
    _require_private_destination(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("Calibration output directory must be empty; preserve prior attempts.")
    lock, manifest, pairs = _load_human_pairs(lock_path)
    source_package_path = source_package_dir / "preference_package.json"
    config_path = source_package_dir / "preference_config_private.json"
    package, config = _object(source_package_path), _object(config_path)
    _validate_preference_config(config)
    if (sha256_file(config_path) != package.get("preference_config_sha256")
            or config.get("model_id") != package.get("judge_model_id")):
        raise ValueError("Calibration must use the exact attempt-002 judge configuration.")
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "raw"
    raw_dir.mkdir()
    timeout = config["retry_policy"]["timeout_seconds"]
    checks = [_probe_ollama_digest(config["endpoint"], config["model_id"],
                                   package["model_digest_sha256"], min(timeout, 30),
                                   phase="before")]
    decisions_path = output_dir / "calibration_decisions_private.jsonl"
    attempts_path = output_dir / "calibration_attempts_private.jsonl"
    for code, pair in pairs.items():
        for orientation in _ORIENTATIONS:
            packet = _packet(code, orientation, pair)
            payload = _make_payload(packet, config)
            result = None
            for attempt in range(1, config["retry_policy"]["max_attempts"] + 1):
                raw_name: str | None = None
                raw_path: Path | None = None
                try:
                    raw = _post_chat(config["endpoint"], payload, None, timeout)
                    raw_name = f"raw/{code}_{orientation}_attempt_{attempt}.json"
                    raw_path = output_dir / raw_name
                    _write_json(raw_path, raw)
                    result = _extract_judgement(raw, packet, package)
                    _append_jsonl(attempts_path, {
                        "image_code": code, "orientation": orientation, "attempt": attempt,
                        "result": "accepted", "raw_file": raw_name,
                        "raw_sha256": sha256_file(raw_path), "error_type": None,
                    })
                    _append_jsonl(decisions_path, {
                        "image_code": code, "orientation": orientation,
                        "packet_sha256": _packet_sha256(packet),
                        "request_sha256": _request_sha256(payload),
                        "accepted_raw_file": raw_name,
                        "accepted_raw_sha256": sha256_file(raw_path),
                        "winner": result["winner"],
                        "response_id": result["response_metadata"]["response_id"],
                    })
                    break
                except (RuntimeError, ValueError, KeyError, OSError, json.JSONDecodeError) as error:
                    _append_jsonl(attempts_path, {
                        "image_code": code, "orientation": orientation, "attempt": attempt,
                        "result": "invalid", "error_type": (
                            "OSError" if isinstance(error, OSError) else type(error).__name__),
                        "raw_file": raw_name,
                        "raw_sha256": sha256_file(raw_path) if raw_path is not None else None,
                    })
                    if attempt < config["retry_policy"]["max_attempts"]:
                        time.sleep(config["retry_policy"]["backoff_seconds"])
            if result is None:
                raise RuntimeError(f"Calibration judge failed technical completion at {code} {orientation}.")
    checks.append(_probe_ollama_digest(config["endpoint"], config["model_id"],
                                       package["model_digest_sha256"], min(timeout, 30),
                                       phase="after"))
    decisions = _jsonl(decisions_path)
    measured = _measure(pairs, decisions)
    gate = {
        "kind": "pre_c_pairwise_judge_calibration_gate", "schema_version": 1,
        "human_lock_path": str(lock_path), "human_lock_sha256": sha256_file(lock_path),
        "manifest_sha256": lock["manifest_sha256"],
        "source_package_dir": str(source_package_dir),
        "source_package_sha256": sha256_file(source_package_path),
        "judge_config_sha256": sha256_file(config_path),
        "judge_model_id": config["model_id"],
        "model_digest_sha256": package["model_digest_sha256"],
        "model_digest_checks": checks,
        "decisions_sha256": sha256_file(decisions_path),
        "attempts_sha256": sha256_file(attempts_path),
        "runner_source_sha256": sha256_file(Path(__file__)),
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **measured,
    }
    gate_path = output_dir / "calibration_gate.json"
    _write_json(gate_path, gate)
    if measured["status"] == "passed":
        verify_calibration_gate(gate_path)
    return gate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run or verify human-locked development calibration.")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--human-lock", type=Path, required=True)
    run.add_argument("--source-package-dir", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--gate", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "run":
        result = run_calibration(args.human_lock, args.source_package_dir, args.output_dir)
        print(json.dumps({key: result[key] for key in (
            "status", "pair_count", "human_clear_count", "clear_agreement_count",
            "clear_agreement_rate", "hallucination_false_winners", "order_unstable")},
            ensure_ascii=False))
        return 0 if result["status"] == "passed" else 1
    result = verify_calibration_gate(args.gate)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
