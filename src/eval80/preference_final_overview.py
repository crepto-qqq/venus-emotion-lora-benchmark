"""One-page B1/C benchmark from the completed technical-exclusion judge run.

This joins aggregate files only. It does not open private responses, images,
individual judge decisions or the B1/C answer key.
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


def _sha(value: Any, name: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise ValueError(f"Invalid {name} SHA-256.")
    return value


def _int(value: Any, low: int, high: int, name: str) -> int:
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"Invalid {name} count.")
    return value


def build_final_overview(status: dict[str, Any], guidance: dict[str, Any]) -> dict[str, Any]:
    """Cross-check scope, arithmetic and source identity before presentation."""
    metric = guidance.get("metric")
    calibrated = metric == "calibrated_image_aware_pairwise_guidance_preference"
    reference_only = metric == "uncalibrated_image_aware_pairwise_guidance_preference"
    if (status.get("evaluation_set") != "Eval80-v1"
            or status.get("comparison") != "B1_vs_C_adapter_effect"
            or guidance.get("evaluation_set") != "Eval80-v1"
            or guidance.get("status") != "final_automated_evaluation_with_technical_exclusions"
            or not (calibrated or reference_only)):
        raise ValueError("Overview inputs are not the intended Eval80 B1/C comparison.")
    provenance = status.get("provenance")
    if not isinstance(provenance, dict) or not isinstance(provenance.get("source_records_sha256"), dict):
        raise ValueError("Deterministic benchmark source provenance is missing.")
    source = provenance["source_records_sha256"]
    guide_source = guidance.get("source_records_sha256")
    if (set(source) != {"B1", "C"} or guide_source != source
            or guidance.get("selection_manifest_sha256") != provenance.get("selection_manifest_sha256")):
        raise ValueError("Guidance and deterministic scores use different B1/C source records or labels.")
    for key in ("B1", "C"):
        _sha(source[key], f"{key} records")
    for key in ("selection_manifest_sha256", "inference_manifest_sha256",
                "preference_config_sha256", "model_digest_sha256",
                "decision_lock_sha256"):
        _sha(guidance.get(key), key)
    if calibrated:
        _sha(guidance.get("calibration_gate_sha256"), "calibration_gate_sha256")
        if (guidance.get("calibration_mode", "passed_human_gate") != "passed_human_gate"
                or guidance.get("human_reference_csv_sha256") is not None
                or guidance.get("calibration_evidence_sha256",
                                guidance["calibration_gate_sha256"])
                != guidance["calibration_gate_sha256"]):
            raise ValueError("Calibrated review has mixed calibration provenance.")
    elif guidance.get("calibration_gate_sha256") is not None:
        raise ValueError("Reference-only automated review cannot claim a calibration gate.")
    elif guidance.get("calibration_mode") != "reference_only_unverified":
        raise ValueError("Reference-only review must disclose its unverified mode.")
    fmt = status.get("format")
    rec = status.get("strict_recognition")
    opening = status.get("opening_emotion_diagnostic")
    if not all(isinstance(value, dict) for value in (fmt, rec, opening)):
        raise ValueError("One or more deterministic benchmark axes are missing.")
    if any(value.get("item_count") != 80 for value in (fmt, rec, opening)):
        raise ValueError("Deterministic benchmark must cover all 80 images.")
    fmt_b1 = _int(fmt.get("B1_pass_count"), 0, 80, "B1 format")
    fmt_c = _int(fmt.get("C_pass_count"), 0, 80, "C format")
    strict_b1 = _int(rec.get("B1_correct_count"), 0, 80, "B1 strict recognition")
    strict_c = _int(rec.get("C_correct_count"), 0, 80, "C strict recognition")
    opening_b1 = _int(opening.get("B1_correct_count"), 0, 80, "B1 opening recognition")
    opening_c = _int(opening.get("C_correct_count"), 0, 80, "C opening recognition")
    if (opening.get("net_correct_gain_c_minus_b1") != opening_c - opening_b1
            or opening.get("protocol_metric") is not False):
        raise ValueError("Opening-label recognition must remain a same-field supplemental metric.")
    original = _int(guidance.get("original_scope_images"), 78, 78, "original guidance scope")
    included = _int(guidance.get("included_images"), 0, 78, "included guidance images")
    excluded = _int(guidance.get("technical_excluded_images"), 0, 78, "technical exclusions")
    if (included + excluded != original
            or guidance.get("included_judgements") != 2 * included):
        raise ValueError("Guidance pair coverage differs from the original 78-image scope.")
    inherited = _int(guidance.get("attempt002_valid_decisions_inherited"), 149, 149,
                     "inherited attempt-002 decisions")
    inherited_used = _int(guidance.get("attempt002_decisions_used_after_exclusion"),
                          0, inherited, "inherited decisions used")
    new_used = _int(guidance.get("new_decisions_used_after_exclusion"), 0, 156,
                    "new decisions used")
    new_attempts = _int(guidance.get("recovery_request_attempts"), 0, 156,
                        "recovery request attempts")
    if inherited_used + new_used != 2 * included or new_attempts < new_used:
        raise ValueError("Retained decisions or recovery request attempts disagree with coverage.")
    consensus = guidance.get("image_consensus")
    if not isinstance(consensus, dict):
        raise ValueError("Guidance consensus is missing.")
    wins_c = _int(consensus.get("C_wins"), 0, 78, "C wins")
    wins_b1 = _int(consensus.get("B1_wins"), 0, 78, "B1 wins")
    ties = _int(consensus.get("ties"), 0, 78, "ties")
    unstable = _int(consensus.get("swap_disagreements_unstable"), 0, 78, "unstable")
    net = wins_c - wins_b1
    lower = net - unstable - excluded
    if (wins_c + wins_b1 + ties + unstable != included
            or guidance.get("observed_net_c_minus_b1_count") != net
            or guidance.get("observed_net_c_minus_b1_rate") != (net / included if included else None)
            or guidance.get("original_scope_worst_case_net_count") != lower
            or guidance.get("original_scope_worst_case_net_rate") != lower / 78
            or guidance.get("original_scope_best_case_net_rate")
            != (net + unstable + excluded) / 78):
        raise ValueError("Guidance counts or fixed-denominator conservative bound disagree.")
    votes = guidance.get("item_votes")
    if (not isinstance(votes, dict) or set(votes) != {"C", "B1", "tie"}
            or sum(_int(votes[key], 0, 2 * included, f"{key} item vote")
                   for key in votes) != 2 * included):
        raise ValueError("Guidance item votes do not cover both orders of retained images.")
    coverage = guidance.get("emotion_coverage")
    if (not isinstance(coverage, dict) or not coverage
            or any(not isinstance(row, dict) or set(row) != {"included", "excluded"}
                   for row in coverage.values())
            or sum(_int(row["included"], 0, 78, "included emotion coverage")
                   for row in coverage.values()) != included
            or sum(_int(row["excluded"], 0, 78, "excluded emotion coverage")
                   for row in coverage.values()) != excluded):
        raise ValueError("Guidance emotion coverage differs from pair counts.")
    if not isinstance(guidance.get("judge_model_id"), str) or not guidance["judge_model_id"].strip():
        raise ValueError("Guidance judge model identity is missing.")
    calibration = guidance.get("calibration")
    if not isinstance(calibration, dict):
        raise ValueError("Human calibration status is missing.")
    if calibrated:
        if calibration.get("pair_count") != 16:
            raise ValueError("Human calibration summary is missing.")
        clear = _int(calibration.get("human_clear_count"), 10, 16, "human clear preferences")
        agreed = _int(calibration.get("clear_agreement_count"), 0, clear,
                      "stable Gemma-human agreements")
        if (agreed * 5 < clear * 4
                or calibration.get("clear_agreement_rate") != agreed / clear
                or calibration.get("hallucination_false_winners") != 0):
            raise ValueError("Human calibration does not meet the declared gate.")
        calibration_report = {
            "mode": "passed_human_calibration", "pair_count": 16,
            "human_clear_count": clear, "clear_agreement_count": agreed,
            "clear_agreement_rate": agreed / clear,
            "hallucination_false_winners": 0,
        }
    else:
        if (calibration.get("mode") != "reference_only_unverified"
                or any(calibration.get(key) is not None for key in (
                    "human_clear_count", "clear_agreement_count",
                    "clear_agreement_rate", "hallucination_false_winners"))):
            raise ValueError("Reference-only review cannot claim human agreement.")
        reference_sha = _sha(calibration.get("reference_csv_sha256"),
                             "incomplete human reference CSV")
        if guidance.get("human_reference_csv_sha256") != reference_sha:
            raise ValueError("Reference-only CSV provenance differs from calibration status.")
        if guidance.get("calibration_evidence_sha256", reference_sha) != reference_sha:
            raise ValueError("Reference-only evidence differs from calibration status.")
        calibration_report = {
            "mode": "reference_only_unverified",
            "reference_csv_sha256": reference_sha,
            "human_clear_count": None, "clear_agreement_count": None,
            "clear_agreement_rate": None,
            "hallucination_false_winners": None,
        }
    positions = guidance.get("technical_excluded_anonymous_positions")
    if (not isinstance(positions, list) or len(positions) != 2 * excluded
            or len(set(positions)) != len(positions)
            or any(type(position) is not int or not 1 <= position <= 156
                   for position in positions)):
        raise ValueError("Excluded anonymous item positions do not match image exclusions.")
    incident = guidance.get("technical_incident_evidence_sha256")
    if not isinstance(incident, dict) or set(incident) != {
            "attempt002_attempts", "attempt002_raw", "recovery_attempts"}:
        raise ValueError("Technical incident source hashes are missing.")
    for name, value in incident.items():
        _sha(value, name)
    error_types = guidance.get("new_technical_error_types")
    error_evidence = guidance.get("recovery_error_evidence")
    if (not isinstance(error_types, dict) or not isinstance(error_evidence, list)
            or any(not isinstance(name, str) or not name
                   or type(count) is not int or count < 0
                   for name, count in error_types.items())
            or sum(error_types.values()) != len(error_evidence)):
        raise ValueError("Recovery error counts do not match redacted evidence.")
    for item in error_evidence:
        if (not isinstance(item, dict) or set(item) != {
                "anonymous_position", "retry_attempt", "error_type", "finish_reason",
                "prompt_tokens", "completion_tokens", "max_tokens", "image_excluded"}
                or type(item["anonymous_position"]) is not int
                or not 1 <= item["anonymous_position"] <= 156
                or type(item["retry_attempt"]) is not int
                or not 1 <= item["retry_attempt"] <= 2
                or item["error_type"] not in error_types
                or item["finish_reason"] is not None
                and not isinstance(item["finish_reason"], str)
                or any(value is not None and (type(value) is not int or value < 0)
                       for value in (item["prompt_tokens"], item["completion_tokens"]))
                or item["max_tokens"] != 2048
                or type(item["image_excluded"]) is not bool):
            raise ValueError("Recovery technical error evidence is malformed.")
    if dict(Counter(item["error_type"] for item in error_evidence)) != error_types:
        raise ValueError("Recovery error types differ from redacted evidence rows.")
    if lower > 0:
        direction = "C has a robust directional advantage in this Eval80 automated preference review."
    elif net > 0:
        direction = ("Stable preferences among included images favor C, but the conservative "
                     "lower bound does not exceed zero, so the direction is not robust.")
    elif net < 0:
        direction = ("Stable preferences among included images favor B1; this does not support "
                     "a claim that C's guidance quality improved.")
    else:
        direction = "Included images show no stable net preference."
    if reference_only:
        if lower > 0:
            direction = ("The frozen Gemma automated preference metric supports a robust "
                         "directional advantage for C; improvement in human-perceived "
                         "guidance quality remains unverified.")
        elif net > 0:
            direction = ("Frozen Gemma preferences among included images favor C, but the "
                         "conservative lower bound over the original 78 images does not exceed "
                         "zero; improvement in human-perceived guidance quality remains "
                         "unverified.")
        elif net < 0:
            direction = ("Stable frozen Gemma preferences among included images favor B1; "
                         "this does not support a claim that C's guidance quality improved.")
        else:
            direction = ("Frozen Gemma shows no stable net preference among included images; "
                         "improvement in human-perceived guidance quality remains unverified.")
    return {
        "schema_version": 1,
        "evaluation_set": "Eval80-v1",
        "comparison": "B1_vs_C_adapter_effect",
        "status": "final_automated_evaluation_with_technical_exclusions",
        "guidance_direction": direction,
        "format": {"B1_pass_count": fmt_b1, "C_pass_count": fmt_c, "item_count": 80},
        "strict_recognition": {
            "B1_correct_count": strict_b1, "C_correct_count": strict_c,
            "item_count": 80, "b1_schema_caveat": rec.get("b1_schema_caveat"),
        },
        "opening_emotion_diagnostic": {
            "B1_correct_count": opening_b1, "C_correct_count": opening_c,
            "item_count": 80, "net_correct_gain_c_minus_b1": opening_c - opening_b1,
            "protocol_metric": False,
        },
        "guidance_preference": {
            "metric": metric,
            "judge_model_id": guidance["judge_model_id"],
            "C_wins": wins_c, "B1_wins": wins_b1, "ties": ties,
            "order_unstable": unstable, "technical_excluded_images": excluded,
            "included_images": included, "original_scope_images": original,
            "attempt002_valid_decisions_inherited": inherited,
            "attempt002_decisions_used_after_exclusion": inherited_used,
            "new_decisions_used_after_exclusion": new_used,
            "recovery_request_attempts": new_attempts,
            "observed_net_rate": net / included if included else None,
            "worst_case_net_rate_original_scope": lower / original,
            "emotion_coverage": coverage,
            "calibration": calibration_report,
            "technical_excluded_anonymous_positions": sorted(positions),
            "recovery_error_evidence": error_evidence,
            "new_technical_error_types": error_types,
            "recovery_attempts_sha256": incident["recovery_attempts"],
        },
        "provenance": {
            "source_records_sha256": source,
            "selection_manifest_sha256": guidance["selection_manifest_sha256"],
            "inference_manifest_sha256": guidance["inference_manifest_sha256"],
            "preference_config_sha256": guidance["preference_config_sha256"],
            "model_digest_sha256": guidance["model_digest_sha256"],
            "calibration_gate_sha256": guidance["calibration_gate_sha256"],
            "human_reference_csv_sha256": (
                calibration_report.get("reference_csv_sha256") if reference_only else None),
            "decision_lock_sha256": guidance["decision_lock_sha256"],
        },
        "acceptance": {
            "user_review_pending": True, "formal_acceptance": False,
            "member3b_acceptance_verified": False,
            "human_guidance_score": None,
        },
    }


def _markdown(report: dict[str, Any]) -> str:
    fmt, rec = report["format"], report["strict_recognition"]
    opening, guide = report["opening_emotion_diagnostic"], report["guidance_preference"]
    reference_only = guide["calibration"]["mode"] == "reference_only_unverified"
    if reference_only:
        calibration_paragraph = (
            "**Human calibration is incomplete.** The draft human blind review of 16 "
            "development pairs is reference material only; the planned gates for stable "
            "agreement across both answer orders and false wins by clearly hallucinated "
            "answers could not be checked. This guidance metric measures the frozen Gemma "
            "judge's relative preference and cannot by itself establish an improvement in "
            "human-perceived advice quality. "
            f"Draft SHA-256: `{guide['calibration']['reference_csv_sha256']}`."
        )
    else:
        calibration_paragraph = (
            f"Judge calibration: among 16 development pairs, the human reviewer expressed "
            f"a clear preference for {guide['calibration']['human_clear_count']}; Gemma "
            f"agreed consistently across both answer orders on "
            f"{guide['calibration']['clear_agreement_count']} "
            f"({guide['calibration']['clear_agreement_rate']:.1%}). It never consistently "
            "selected a clearly worse answer containing hallucinations. This small, "
            "single-reviewer calibration supports only this automated preference metric; "
            "it does not establish inter-expert agreement or a five-dimensional aesthetic score."
        )
    rate = guide["observed_net_rate"]
    observed = "not calculable" if rate is None else f"{rate:+.1%}"
    error_table = []
    if guide["recovery_error_evidence"]:
        error_table = [
            "| Anonymous position | Attempt | Error | finish_reason | Output tokens / limit | Image excluded |",
            "| ---: | ---: | --- | --- | ---: | --- |",
            *[f"| {item['anonymous_position']} | {item['retry_attempt']} | "
              f"{item['error_type']} | {item['finish_reason'] or 'no response'} | "
              f"{item['completion_tokens'] if item['completion_tokens'] is not None else '—'}"
              f" / {item['max_tokens']} | {'yes' if item['image_excluded'] else 'no'} |"
              for item in guide["recovery_error_evidence"]],
            "",
        ]
    lines = [
        ("# Eval80 B1/C final automated review (human calibration incomplete; technical exclusion)"
         if reference_only else "# Eval80 B1/C final automated review (technical exclusion)"), "",
        f"**Guidance conclusion:** {report['guidance_direction']}", "",
        "| Metric | B1 | C | Interpretation |", "| --- | ---: | ---: | --- |",
        f"| Complete output format | {fmt['B1_pass_count']}/80 | {fmt['C_pass_count']}/80 | Deterministic check |",
        f"| Strict emotion recognition | {rec['B1_correct_count']}/80 | {rec['C_correct_count']}/80 | Confounded by B1's missing required field |",
        f"| Opening `Emotion:` field accuracy | {opening['B1_correct_count']}/80 | {opening['C_correct_count']}/80 | Same-field supplemental diagnostic; C gains {opening['net_correct_gain_c_minus_b1']}/80 net |",
        "", "## Image-aware automated guidance preference", "",
        f"Included images: **{guide['included_images']}/78**; whole images excluded for technical failure: "
        f"**{guide['technical_excluded_images']}**. Consensus across both answer orders: "
        f"C wins **{guide['C_wins']}**, B1 wins **{guide['B1_wins']}**, "
        f"ties **{guide['ties']}**, and order-unstable outcomes **{guide['order_unstable']}**.", "",
        f"Observed net preference among included images, `(C wins − B1 wins) / included images`, "
        f"is **{observed}**. The conservative lower bound over the original 78 images "
        f"(assigning every unstable and excluded image to B1) is "
        f"**{guide['worst_case_net_rate_original_scope']:+.1%}**.", "",
        calibration_paragraph, "",
        "The judge sees the source image, target emotion, and both complete answers. Its prompt "
        "compares visual grounding, actionability, and emotional fit. B1 and C also differ "
        "in answer format, so this preference metric cannot fully isolate guidance content "
        "from presentation format.", "",
        "| Target emotion | Included images | Technical exclusions |", "| --- | ---: | ---: |",
        *[f"| {emotion} | {counts['included']} | {counts['excluded']} |"
          for emotion, counts in sorted(guide["emotion_coverage"].items())],
        "",
        "Technical exclusions affect anonymous review positions "
        + ", ".join(map(str, guide["technical_excluded_anonymous_positions"])) + ".", "",
        "Technical errors during the recovery run: " + (
            "none." if not guide["recovery_error_evidence"] else "see table below."), "",
        *error_table,
        f"The method was amended after two technical failures of full-scope runs. We "
        f"verified and inherited {guide['attempt002_valid_decisions_inherited']} valid blind "
        f"decisions from attempt 002. Of these, "
        f"{guide['attempt002_decisions_used_after_exclusion']} belong to images retained in "
        f"the final analysis; {guide['new_decisions_used_after_exclusion']} valid decisions "
        "were completed in the recovery run. Neither answer order for the image with a "
        "technical failure was scored. The "
        "[failure evidence appendix](../../work_records/evaluation/EVAL80_PREFERENCE_FAILURE_EVIDENCE.md) "
        "contains raw log hashes and redacted visualizations; no screenshot was captured "
        "at the time of the incident.", "",
        "This automated preference score measures the relative performance of two answers "
        "to the same image. It is not the original six-reviewer, five-dimensional /10 "
        "Guidance Score. This generated report awaits user review; Member 3B acceptance "
        "and any amendment replacing the original human endpoint require separate records.", "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Join validated deterministic and final preference aggregates.")
    parser.add_argument("--deterministic-status", type=Path, required=True)
    parser.add_argument("--guidance-aggregate", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args(argv)
    report = build_final_overview(_object(args.deterministic_status),
                                  _object(args.guidance_aggregate))
    report["provenance"]["deterministic_status_sha256"] = sha256_file(args.deterministic_status)
    report["provenance"]["guidance_aggregate_sha256"] = sha256_file(args.guidance_aggregate)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.output_md.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps({"status": report["status"], "output_json": str(args.output_json),
                      "output_md": str(args.output_md)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
