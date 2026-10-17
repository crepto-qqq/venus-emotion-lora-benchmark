"""Public, aggregate-only status for an incomplete Eval80 guidance benchmark.

This module never opens the answer mapping, images, item payloads, or item-level
judge decisions. It hashes private evidence files and reads only attempt status
and response termination metadata. A partial judge run has no preference score.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

from .core import SHA256_RE, sha256_file


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _hash(value: Any, description: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise ValueError(f"Invalid SHA-256 for {description}.")
    return value


def _count(value: Any, maximum: int, description: str) -> int:
    if type(value) is not int or not 0 <= value <= maximum:
        raise ValueError(f"Invalid count for {description}.")
    return value


def _line_count(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for line in handle if line.strip())


def _jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"Expected JSONL objects: {path}")
                yield value


def _aggregate(deterministic: dict[str, Any], opening: dict[str, Any]) -> dict[str, Any]:
    if (deterministic.get("schema_version") != 1
            or deterministic.get("evaluation_set") != "Eval80-v1"
            or deterministic.get("comparison") != "B1_vs_C_adapter_effect"
            or deterministic.get("status") != "incomplete"
            or deterministic.get("missing_inputs") != ["guidance"]
            or deterministic.get("guidance_proxy") is not None):
        raise ValueError("Deterministic report is not the incomplete Eval80 B1/C aggregate.")
    records: dict[str, str] = {}
    format_counts: dict[str, int] = {}
    recognition_counts: dict[str, int] = {}
    for condition in ("B1", "C"):
        fmt = deterministic["format"][condition]
        rec = deterministic["recognition"][condition]
        if fmt.get("item_count") != 80 or rec.get("item_count") != 80:
            raise ValueError(f"{condition} deterministic measures must cover 80 images.")
        records[condition] = _hash(fmt.get("records_sha256"), f"{condition} records")
        if rec.get("records_sha256") != records[condition]:
            raise ValueError(f"{condition} format and recognition records disagree.")
        format_counts[condition] = _count(fmt.get("pass_count"), 80, f"{condition} format")
        recognition_counts[condition] = _count(rec.get("correct_count"), 80,
                                               f"{condition} recognition")
    if deterministic["format"]["B1"].get("failure_reason_counts", {}).get(
            "missing_final_emotion") != 80:
        raise ValueError("B1 strict-score caveat requires 80 missing Final emotion fields.")
    if (opening.get("schema_version") != 1
            or opening.get("evaluation_set") != "Eval80-v1"
            or opening.get("metric") != "supplementary_opening_Emotion_field_only"
            or opening.get("protocol_metric") is not False
            or opening.get("image_count") != 80
            or opening.get("parsed_opening_count") != {"B1": 80, "C": 80}):
        raise ValueError("Opening-label diagnostic does not cover the same Eval80 protocol.")
    source = opening.get("input_sha256", {})
    if source.get("b1_records") != records["B1"] or source.get("c_records") != records["C"]:
        raise ValueError("Opening-label diagnostic uses different B1/C records.")
    selection = _hash(source.get("selection_manifest"), "selection manifest")
    opening_counts = {condition: _count(opening["correct_count"][condition], 80,
                      f"{condition} opening-label correct") for condition in ("B1", "C")}
    paired = opening.get("paired_outcomes", {})
    if (set(paired) != {"both_correct", "b1_only_correct", "c_only_correct", "both_wrong"}
            or any(type(value) is not int or value < 0 for value in paired.values())
            or sum(paired.values()) != 80
            or paired["both_correct"] + paired["b1_only_correct"] != opening_counts["B1"]
            or paired["both_correct"] + paired["c_only_correct"] != opening_counts["C"]
            or opening.get("net_correct_gain_c_minus_b1") != (
                opening_counts["C"] - opening_counts["B1"])):
        raise ValueError("Opening-label paired totals disagree with the 80-image counts.")
    return {"records": records, "selection": selection, "format": format_counts,
            "recognition": recognition_counts, "opening": opening_counts,
            "opening_paired": paired}


def _attempt(directory: Path, records: dict[str, str], selection: str) -> dict[str, Any]:
    package_path = directory / "preference_package.json"
    package = _object(package_path)
    if (package.get("schema_version") != 1
            or package.get("evaluation_set") != "Eval80-v1"
            or package.get("metric") != "image_aware_pairwise_guidance_preference_proxy"
            or package.get("image_count") != 78
            or package.get("item_count") != 156
            or package.get("b1_records_sha256") != records["B1"]
            or package.get("c_records_sha256") != records["C"]
            or package.get("selection_manifest_sha256") != selection
            or package.get("adapter_identity", {}).get("acceptance_status") != "not_verified"):
        raise ValueError(f"Preference package is not bound to this frozen B1/C set: {directory}")
    package_sha = sha256_file(package_path)
    for filename, field in (
        ("preference_config_private.json", "preference_config_snapshot_sha256"),
        ("preference_items_private.jsonl", "preference_items_sha256"),
        ("preference_mapping_private.json", "preference_mapping_sha256"),
        ("preflight_receipt_private.json", "preflight_receipt_sha256"),
        ("model_digest_receipt_private.json", "model_digest_receipt_sha256"),
    ):
        expected = _hash(package.get(field), field)
        if sha256_file(directory / filename) != expected:
            raise ValueError(f"Private package evidence hash differs: {filename}")
    if _line_count(directory / "preference_items_private.jsonl") != 156:
        raise ValueError("Private package must contain 156 frozen item packets.")
    config = _object(directory / "preference_config_private.json")
    if sha256_file(directory / "preference_config_private.json") != package.get(
            "preference_config_sha256"):
        raise ValueError("Preference config hash differs from package.")
    cap = _count(config.get("decoding", {}).get("max_tokens"), 100000, "token cap")
    if cap == 0 or config.get("model_id") != package.get("judge_model_id"):
        raise ValueError("Preference config model identity or token cap is invalid.")
    state = _object(directory / "preference_run_state.json")
    if (state.get("preference_package_sha256") != package_sha
            or state.get("preference_config_sha256") != package["preference_config_sha256"]
            or state.get("preference_items_sha256") != package["preference_items_sha256"]):
        raise ValueError("Run state is not bound to the frozen package.")
    checks = state.get("runtime_model_identity", {}).get("checks", [])
    if (len(checks) != 1 or checks[0].get("phase") != "before"
            or checks[0].get("model_id") != package["judge_model_id"]
            or checks[0].get("digest_sha256") != package["model_digest_sha256"]):
        raise ValueError("Incomplete run lacks its matching pre-run model digest.")
    if (directory / "preference_judgements_lock.json").exists():
        raise ValueError("A score lock exists; use the complete preference overview instead.")
    attempts_path = directory / "preference_attempts_private.jsonl"
    raw_path = directory / "preference_raw_responses_private.jsonl"
    decisions_path = directory / "preference_judgements_private.jsonl"
    attempts = list(_jsonl(attempts_path))
    raw = list(_jsonl(raw_path))
    if len(attempts) != len(raw):
        raise ValueError("Every logged attempt must have a raw response for this failure mode.")
    for left, right in zip(attempts, raw):
        if (left.get("anonymous_item_id") != right.get("anonymous_item_id")
                or left.get("attempt") != right.get("attempt")
                or left.get("request_sha256") != right.get("request_sha256")):
            raise ValueError("Attempt and raw response trace order disagree.")
    status = Counter(row.get("status") for row in attempts)
    successes = status["success"]
    if (successes != 149 or status["error"] != 3 or len(attempts) != 152
            or _line_count(decisions_path) != successes):
        raise ValueError("Expected 149 completed decisions and three logged errors.")
    if len({row.get("anonymous_item_id") for row in attempts
            if row.get("status") == "success"}) != successes:
        raise ValueError("Duplicate successful anonymous items.")
    failed = attempts[-2:]
    failed_raw = raw[-2:]
    if (len(failed) != 2
            or [row.get("status") for row in failed] != ["error", "error"]
            or [row.get("attempt") for row in failed] != [1, 2]
            or failed[0].get("anonymous_item_id") != failed[1].get("anonymous_item_id")
            or failed[0].get("anonymous_item_id") in {
                row.get("anonymous_item_id") for row in attempts
                if row.get("status") == "success"
            }):
        raise ValueError("Final two attempts are not an exhausted unscored item.")
    for row in failed_raw:
        response = row.get("response", {})
        choices = response.get("choices", [])
        if (len(choices) != 1 or choices[0].get("finish_reason") != "length"
                or response.get("usage", {}).get("completion_tokens") != cap):
            raise ValueError("Final judge replies did not stop at the frozen output cap.")
    return {
        "package_directory": directory.name,
        "package_sha256": package_sha,
        "config_sha256": package["preference_config_sha256"],
        "preference_items_sha256": package["preference_items_sha256"],
        "preflight_receipt_sha256": package["preflight_receipt_sha256"],
        "judge_model_id": package["judge_model_id"],
        "judge_model_digest_sha256": package["model_digest_sha256"],
        "token_cap": cap,
        "image_count": 78,
        "planned_decisions": 156,
        "successful_decisions_unscored": successes,
        "logged_errors": status["error"],
        "last_two_finish_reason": ["length", "length"],
        "score_lock_present": False,
        "attempt_trace_sha256": sha256_file(attempts_path),
        "raw_response_trace_sha256": sha256_file(raw_path),
        "private_partial_decisions_sha256": sha256_file(decisions_path),
    }


def build_benchmark_status(
    deterministic: dict[str, Any], opening: dict[str, Any],
    attempt_dirs: list[Path],
) -> dict[str, Any]:
    """Verify aggregate provenance; never turn partial judge votes into a score."""
    if len(attempt_dirs) != 2 or attempt_dirs[0] == attempt_dirs[1]:
        raise ValueError("Expected two distinct technical judge attempts.")
    aggregate = _aggregate(deterministic, opening)
    attempts = [_attempt(directory, aggregate["records"], aggregate["selection"])
                for directory in attempt_dirs]
    packages = [_object(directory / "preference_package.json") for directory in attempt_dirs]
    for field in ("inference_manifest_sha256", "freeze_sha256", "adapter_identity",
                  "prompt_sha256", "response_schema_sha256", "benchmark_source_sha256",
                  "runner_source_sha256", "model_digest_sha256"):
        if packages[0].get(field) != packages[1].get(field):
            raise ValueError(f"Technical retries differ on frozen provenance: {field}")
    if ([item["token_cap"] for item in attempts] != [550, 2048]
            or attempts[0]["judge_model_digest_sha256"]
            != attempts[1]["judge_model_digest_sha256"]):
        raise ValueError("The two documented technical retries need 550/2048 caps and one model.")
    return {
        "schema_version": 1,
        "evaluation_set": "Eval80-v1",
        "comparison": "B1_vs_C_adapter_effect",
        "status": "provisional_incomplete_guidance",
        "format": {"B1_pass_count": aggregate["format"]["B1"],
                   "C_pass_count": aggregate["format"]["C"], "item_count": 80},
        "strict_recognition": {
            "B1_correct_count": aggregate["recognition"]["B1"],
            "C_correct_count": aggregate["recognition"]["C"], "item_count": 80,
            "b1_schema_caveat": "B1 lacks the required Final emotion field on all 80 outputs; strict scores are format-confounded.",
        },
        "opening_emotion_diagnostic": {
            "B1_correct_count": aggregate["opening"]["B1"],
            "C_correct_count": aggregate["opening"]["C"], "item_count": 80,
            "net_correct_gain_c_minus_b1": (
                aggregate["opening"]["C"] - aggregate["opening"]["B1"]),
            "paired_outcomes": aggregate["opening_paired"],
            "protocol_metric": False,
        },
        "guidance_preference_proxy": {
            "status": "unavailable_technical_judge_failure",
            "score": None,
            "image_count": 78,
            "planned_decisions": 156,
            "attempts": attempts,
            "note": "Neither run has 156 decisions or a score lock; partial votes were not unblinded or scored.",
        },
        "provenance": {
            "source_records_sha256": aggregate["records"],
            "selection_manifest_sha256": aggregate["selection"],
        },
        "acceptance": {
            "member3b_acceptance_verified": False,
            "user_review_pending": True,
            "formal_acceptance": False,
        },
    }


def _markdown(report: dict[str, Any]) -> str:
    fmt, rec = report["format"], report["strict_recognition"]
    opening, guidance = report["opening_emotion_diagnostic"], report["guidance_preference_proxy"]
    lines = [
        "# Eval80 B1/C benchmark status — provisional",
        "",
        "**Result:** C scores higher than B1 on full-schema compliance and opening `Emotion:` accuracy; no valid automated score for guidance quality improvement is available.",
        "",
        "| Measure | B1 | C | Scope |",
        "| --- | ---: | ---: | --- |",
        f"| Full output schema | {fmt['B1_pass_count']}/80 | {fmt['C_pass_count']}/80 | Deterministic |",
        f"| Strict emotion recognition | {rec['B1_correct_count']}/80 | {rec['C_correct_count']}/80 | B1 omits the required `Final emotion` field; this row is confounded by format |",
        f"| Opening `Emotion:` accuracy | {opening['B1_correct_count']}/80 | {opening['C_correct_count']}/80 | Supplementary same-field diagnostic, +{opening['net_correct_gain_c_minus_b1']}/80; exploratory, not a protocol metric |",
        "| Guidance quality | — | — | Automated preference judging failed technically; score unavailable |",
        "",
        "**Judge-attempt evidence:** The plan covered 78 images with both A/B orders, for 156 decisions. Each of two fresh runs completed 149/156 decisions, then produced two consecutive replies that reached the frozen output cap. Neither run produced a 156/156 decision lock. The 149 partial decisions were neither unblinded nor scored.",
        "",
    ]
    for index, attempt in enumerate(guidance["attempts"], start=1):
        lines.append(
            f"- Attempt {index}: `max_tokens={attempt['token_cap']}`; "
            f"149/156; final `finish_reason=length` × 2; "
            f"package SHA-256 `{attempt['package_sha256']}`."
        )
    lines.extend([
        "",
        f"**Limits:** The strict-recognition change from {rec['B1_correct_count']} to {rec['C_correct_count']}/80 alone cannot establish improved emotion recognition because B1 omits the required field in all 80 outputs. The opening-label change from {opening['B1_correct_count']} to {opening['C_correct_count']}/80 (+{opening['net_correct_gain_c_minus_b1']}) is a supplementary same-field check; it does not assess guidance quality. This incomplete attempt cannot support a claim of guidance improvement or treat a missing score as zero.",
        "",
        "**Acceptance:** C inference and deterministic metrics passed technical checks; the Member 3B record remains unverified, and formal acceptance awaits user review. Do not use these frozen Eval80 results to select a checkpoint or retrain.",
        "",
        "**Provenance:**",
        "",
        f"- B1 records SHA-256: `{report['provenance']['source_records_sha256']['B1']}`",
        f"- C records SHA-256: `{report['provenance']['source_records_sha256']['C']}`",
        f"- Selection manifest SHA-256: `{report['provenance']['selection_manifest_sha256']}`",
        "",
    ])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deterministic", type=Path, required=True)
    parser.add_argument("--opening-diagnostic", type=Path, required=True)
    parser.add_argument("--attempt-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args(argv)
    report = build_benchmark_status(_object(args.deterministic),
                                    _object(args.opening_diagnostic), args.attempt_dir)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                encoding="utf-8")
    args.output_md.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps({"status": report["status"], "guidance": "unavailable",
                      "output_json": str(args.output_json),
                      "output_md": str(args.output_md)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
