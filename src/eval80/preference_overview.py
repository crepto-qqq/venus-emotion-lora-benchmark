"""Aggregate-only Eval80 format, recognition, and pairwise preference overview.

This report joins already-produced, frozen summaries. It deliberately does not
read images, answer labels, response text, or item-level judge decisions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .core import SHA256_RE, sha256_file


_CONDITIONS = ("B1", "C")
_PAIRED_OUTCOMES = ("both_correct", "b1_only_correct", "c_only_correct", "both_wrong")


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _sha(value: Any, description: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise ValueError(f"Invalid SHA-256 for {description}.")
    return value


def _count(value: Any, maximum: int, description: str) -> int:
    if type(value) is not int or not 0 <= value <= maximum:
        raise ValueError(f"Invalid count for {description}.")
    return value


def _paired_counts(value: Any, expected: int, description: str) -> dict[str, int]:
    if not isinstance(value, dict) or set(value) != set(_PAIRED_OUTCOMES):
        raise ValueError(f"Invalid paired outcomes for {description}.")
    counts = {key: _count(value[key], expected, f"{description} {key}")
              for key in _PAIRED_OUTCOMES}
    if sum(counts.values()) != expected:
        raise ValueError(f"Paired outcomes do not cover {expected} images for {description}.")
    return counts


def build_preference_overview(
    format_reports: dict[str, dict[str, Any]],
    recognition_reports: dict[str, dict[str, Any]],
    paired_recognition: dict[str, Any],
    preference: dict[str, Any],
    opening_diagnostic: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate common source identities and return a public aggregate-only view."""
    if set(format_reports) != set(_CONDITIONS) or set(recognition_reports) != set(_CONDITIONS):
        raise ValueError("Format and recognition need both B1 and C.")
    if (
        preference.get("schema_version") != 1
        or preference.get("evaluation_set") != "Eval80-v1"
        or preference.get("status") != "provisional_proxy"
        or preference.get("metric") != "image_aware_pairwise_guidance_preference_proxy"
        or preference.get("image_count") != 78
        or preference.get("judgement_count") != 156
        or preference.get("judge_provenance") != "script_runner_trace_consistent"
    ):
        raise ValueError("Preference aggregate is not the completed script-run Eval80 proxy.")
    source_hashes = preference.get("source_records_sha256")
    if not isinstance(source_hashes, dict) or set(source_hashes) != set(_CONDITIONS):
        raise ValueError("Preference aggregate lacks B1/C record hashes.")
    inference_hash = _sha(preference.get("inference_manifest_sha256"), "inference manifest")
    selection_hash = _sha(preference.get("selection_manifest_sha256"), "selection manifest")
    format_counts: dict[str, int] = {}
    recognition_counts: dict[str, int] = {}
    for condition in _CONDITIONS:
        fmt, rec = format_reports[condition], recognition_reports[condition]
        record_hash = _sha(source_hashes[condition], f"{condition} records")
        if (fmt.get("evaluation_set") != "Eval80-v1"
                or fmt.get("condition") != condition
                or fmt.get("metric") != "auxiliary_structured_output_format_compliance"
                or fmt.get("item_count") != 80):
            raise ValueError(f"{condition} format audit does not cover frozen Eval80.")
        passed = _count(fmt.get("format_pass_count"), 80, f"{condition} format")
        if fmt.get("format_failure_count") != 80 - passed:
            raise ValueError(f"{condition} format pass/fail totals disagree.")
        if (rec.get("condition") != condition or rec.get("item_count") != 80
                or rec.get("schema_version") != 2):
            raise ValueError(f"{condition} recognition summary does not cover 80 images.")
        correct = _count(rec.get("correct_count"), 80, f"{condition} recognition")
        if (fmt.get("records_sha256") != record_hash
                or rec.get("records_sha256") != record_hash):
            raise ValueError(f"{condition} axes use different output records.")
        if fmt.get("inference_manifest_sha256") != inference_hash:
            raise ValueError(f"{condition} format and preference use different inference manifests.")
        if rec.get("selection_manifest_sha256") != selection_hash:
            raise ValueError(f"{condition} recognition and preference use different answer keys.")
        _sha(rec.get("details_sha256"), f"{condition} recognition details")
        format_counts[condition], recognition_counts[condition] = passed, correct

    if (paired_recognition.get("schema_version") != 1
            or paired_recognition.get("evaluation_set") != "Eval80-v1"
            or paired_recognition.get("comparison") != "B1_vs_C_strict_emotion_recognition"
            or paired_recognition.get("image_count") != 80
            or paired_recognition.get("input_sha256") != {
                "b1_details": recognition_reports["B1"]["details_sha256"],
                "c_details": recognition_reports["C"]["details_sha256"],
            }):
        raise ValueError("Paired recognition is not bound to the two 80-image score sheets.")
    outcomes = _paired_counts(paired_recognition.get("paired_outcomes"), 80,
                              "strict recognition")
    paired_conditions = paired_recognition.get("conditions", {})
    if (not isinstance(paired_conditions, dict)
            or any(not isinstance(paired_conditions.get(condition), dict)
                   or paired_conditions[condition].get("correct_count") != recognition_counts[condition]
                   for condition in _CONDITIONS)
            or outcomes["both_correct"] + outcomes["b1_only_correct"] != recognition_counts["B1"]
            or outcomes["both_correct"] + outcomes["c_only_correct"] != recognition_counts["C"]
            or paired_recognition.get("net_correct_gain_c_minus_b1") != (
                recognition_counts["C"] - recognition_counts["B1"]
            )):
        raise ValueError("Paired recognition totals disagree with individual summaries.")

    consensus = preference.get("image_consensus")
    if not isinstance(consensus, dict):
        raise ValueError("Preference image consensus is missing.")
    c_wins = _count(consensus.get("C_wins"), 78, "C preference wins")
    b1_wins = _count(consensus.get("B1_wins"), 78, "B1 preference wins")
    ties = _count(consensus.get("ties"), 78, "preference ties")
    unstable = _count(consensus.get("swap_disagreements_unstable"), 78,
                      "swap disagreements")
    identical_ties = _count(consensus.get("identical_answer_rule_ties"), ties,
                            "identical-answer ties")
    if (c_wins + b1_wins + ties + unstable != 78
            or consensus.get("stable_net_c_minus_b1") != c_wins - b1_wins
            or consensus.get("worst_case_net_if_all_unstable_favor_b1") != c_wins - b1_wins - unstable):
        raise ValueError("Preference consensus does not cover 78 images consistently.")
    votes = preference.get("item_votes")
    if (not isinstance(votes, dict)
            or any(type(votes.get(key)) is not int or votes[key] < 0 for key in ("C", "B1", "tie"))
            or votes["C"] + votes["B1"] + votes["tie"] != 156
            or votes.get("net_c_minus_b1") != votes["C"] - votes["B1"]):
        raise ValueError("Preference item votes do not cover 156 decisions.")
    if (not isinstance(preference.get("judge_model_id"), str)
            or not preference["judge_model_id"].strip()):
        raise ValueError("Preference judge model identity is missing.")

    opening: dict[str, Any] | None = None
    if opening_diagnostic is not None:
        diagnostic = opening_diagnostic
        hashes = diagnostic.get("input_sha256")
        if (diagnostic.get("schema_version") != 1
                or diagnostic.get("evaluation_set") != "Eval80-v1"
                or diagnostic.get("metric") != "supplementary_opening_Emotion_field_only"
                or diagnostic.get("protocol_metric") is not False
                or diagnostic.get("image_count") != 80
                or not isinstance(hashes, dict)
                or hashes.get("selection_manifest") != selection_hash
                or hashes.get("b1_records") != source_hashes["B1"]
                or hashes.get("c_records") != source_hashes["C"]):
            raise ValueError("Opening-label diagnostic is not bound to the same frozen inputs.")
        counts = diagnostic.get("correct_count")
        parsed = diagnostic.get("parsed_opening_count")
        if (not isinstance(counts, dict) or not isinstance(parsed, dict)
                or set(counts) != set(_CONDITIONS) or set(parsed) != set(_CONDITIONS)
                or any(parsed[condition] != 80 for condition in _CONDITIONS)):
            raise ValueError("Opening-label diagnostic does not parse all 80 B1/C outputs.")
        opening_counts = {condition: _count(counts[condition], 80,
                          f"{condition} opening-label correct") for condition in _CONDITIONS}
        opening_outcomes = _paired_counts(diagnostic.get("paired_outcomes"), 80,
                                          "opening-label diagnostic")
        if (opening_outcomes["both_correct"] + opening_outcomes["b1_only_correct"]
                != opening_counts["B1"]
                or opening_outcomes["both_correct"] + opening_outcomes["c_only_correct"]
                != opening_counts["C"]
                or diagnostic.get("net_correct_gain_c_minus_b1") != (
                    opening_counts["C"] - opening_counts["B1"]
                )):
            raise ValueError("Opening-label paired totals disagree with counts.")
        opening = {"B1_correct_count": opening_counts["B1"],
                   "C_correct_count": opening_counts["C"], "item_count": 80,
                   "net_correct_gain_c_minus_b1": opening_counts["C"] - opening_counts["B1"],
                   "protocol_metric": False}

    if c_wins - b1_wins - unstable > 0:
        headline = ("The local image-aware preference proxy favors C on this selected set "
                    "even if every unstable comparison favors B1.")
    elif c_wins - b1_wins > 0:
        headline = ("Stable votes favor C, but order-unstable cases prevent a robust "
                    "directional proxy conclusion.")
    elif c_wins - b1_wins < 0:
        headline = ("Stable votes favor B1; the overview makes no robust C-improvement "
                    "claim on this selected set.")
    else:
        headline = "The local preference proxy has no stable net winner on this selected set."
    return {
        "schema_version": 1,
        "evaluation_set": "Eval80-v1",
        "comparison": "B1_vs_C_adapter_effect",
        "status": "provisional_proxy",
        "headline": headline,
        "format": {"B1_pass_count": format_counts["B1"],
                   "C_pass_count": format_counts["C"], "item_count": 80},
        "strict_recognition": {
            "B1_correct_count": recognition_counts["B1"],
            "C_correct_count": recognition_counts["C"], "item_count": 80,
            "net_correct_gain_c_minus_b1": recognition_counts["C"] - recognition_counts["B1"],
            "paired_outcomes": outcomes,
            "b1_schema_caveat": (
                "B1 has zero complete-format outputs; strict recognition is confounded by "
                "its missing required Final emotion field. Compare the optional opening-label "
                "diagnostic only as an exploratory same-field check."
                if format_counts["B1"] == 0 else None
            ),
        },
        "opening_label_diagnostic": opening,
        "guidance_preference_proxy": {
            "image_count": 78, "judgement_count": 156,
            "judge_model_id": preference["judge_model_id"],
            "C_wins": c_wins, "B1_wins": b1_wins, "ties": ties,
            "swap_disagreements_unstable": unstable,
            "identical_answer_rule_ties": identical_ties,
            "stable_net_c_minus_b1": c_wins - b1_wins,
            "worst_case_net_if_all_unstable_favor_b1": c_wins - b1_wins - unstable,
            "model_digest_verification_status": preference.get("model_digest_verification_status"),
            "judge_provenance": preference["judge_provenance"],
        },
        "provenance": {
            "source_records_sha256": {condition: source_hashes[condition]
                                      for condition in _CONDITIONS},
            "inference_manifest_sha256": inference_hash,
            "selection_manifest_sha256": selection_hash,
            "preference_config_sha256": _sha(preference.get("preference_config_sha256"),
                                             "preference config"),
        },
        "acceptance": {
            "member3b_acceptance_verified": False,
            "user_review_pending": True,
            "formal_human_guidance_score": None,
            "note": "Automated pairwise proxy is not the six-human Guidance Score or formal acceptance.",
        },
    }


def _markdown(report: dict[str, Any]) -> str:
    fmt = report["format"]
    rec = report["strict_recognition"]
    opening = report["opening_label_diagnostic"]
    pref = report["guidance_preference_proxy"]
    lines = [
        "# Eval80 B1/C benchmark overview — provisional",
        "",
        f"**Result:** {report['headline']}",
        "",
        "| Measure | B1 | C | Scope |",
        "| --- | ---: | ---: | --- |",
        f"| Full output format | {fmt['B1_pass_count']}/80 | {fmt['C_pass_count']}/80 | Deterministic |",
        f"| Strict emotion correct | {rec['B1_correct_count']}/80 | "
        f"{rec['C_correct_count']}/80 | Frozen two-field rule |",
    ]
    if opening is not None:
        lines.append(
            f"| Opening `Emotion:` correct | {opening['B1_correct_count']}/80 | "
            f"{opening['C_correct_count']}/80 | Exploratory same-field check |"
        )
    lines.extend([
        "",
        "**Guidance preference proxy (78 images; two reversed-order decisions each):** "
        f"C wins {pref['C_wins']}, B1 wins {pref['B1_wins']}, "
        f"ties {pref['ties']}, order-unstable {pref['swap_disagreements_unstable']}. "
        f"Stable net C − B1: {pref['stable_net_c_minus_b1']:+d}; "
        "worst case if every unstable result favored B1: "
        f"{pref['worst_case_net_if_all_unstable_favor_b1']:+d}.",
        "",
    ])
    if rec["b1_schema_caveat"]:
        lines.extend([f"**Strict-score caveat:** {rec['b1_schema_caveat']}", ""])
    lines.extend([
        "**Acceptance:** Awaiting Member 3B evidence and user review. The image-aware "
        "AI preference result is a provisional proxy, not a human 0–10 guidance score. "
        "It describes these selected images and cannot by itself establish a general "
        "aesthetic improvement or justify checkpoint selection on Eval80.",
        "",
    ])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(
        "Join complete B1/C deterministic summaries and a locked preference aggregate "
        "into one provisional, aggregate-only overview."
    ))
    for flag in ("b1-format", "c-format", "b1-recognition", "c-recognition",
                 "paired-recognition", "preference", "output-json", "output-md"):
        parser.add_argument(f"--{flag}", type=Path, required=True)
    parser.add_argument("--opening-diagnostic", type=Path)
    args = parser.parse_args(argv)
    paths = {
        "b1_format": args.b1_format, "c_format": args.c_format,
        "b1_recognition": args.b1_recognition, "c_recognition": args.c_recognition,
        "paired_recognition": args.paired_recognition, "preference": args.preference,
    }
    if args.opening_diagnostic is not None:
        paths["opening_diagnostic"] = args.opening_diagnostic
    if (args.output_json.resolve() == args.output_md.resolve()
            or any(path.resolve() in {args.output_json.resolve(), args.output_md.resolve()}
                   for path in paths.values())):
        raise ValueError("Output paths must differ from each other and from input reports.")
    documents = {name: _object(path) for name, path in paths.items()}
    report = build_preference_overview(
        {"B1": documents["b1_format"], "C": documents["c_format"]},
        {"B1": documents["b1_recognition"], "C": documents["c_recognition"]},
        documents["paired_recognition"], documents["preference"],
        documents.get("opening_diagnostic"),
    )
    report["input_sha256"] = {name: sha256_file(path) for name, path in paths.items()}
    json_bytes = (json.dumps(report, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    md_bytes = _markdown(report).encode("utf-8")
    for path, content in ((args.output_json, json_bytes), (args.output_md, md_bytes)):
        if path.exists() and (not path.is_file() or path.read_bytes() != content):
            raise FileExistsError(f"Existing overview differs from recomputed result: {path}")
    for path, content in ((args.output_json, json_bytes), (args.output_md, md_bytes)):
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            with path.open("xb") as handle:
                handle.write(content)
    print(json.dumps({"status": report["status"], "output_json": str(args.output_json),
                      "output_md": str(args.output_md)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
