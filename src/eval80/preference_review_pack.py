"""Create a private, human-readable review of a locked Eval80 preference run.

The review is deliberately separate from scoring. It only writes beneath the
repository's ignored tmp/ directory, after validating the complete score lock,
the aggregate report, the source records, and all 78 image pairs.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
from typing import Any

from .core import sha256_file
from .preference_benchmark import (
    _IMAGE_COUNT, _ITEM_COUNT, _validate_judgements, _validate_mapping,
    _validate_script_trace,
)
from .quality_benchmark import _read_jsonl, _read_object, _write_bytes_if_absent


_ROOT = Path(__file__).resolve().parents[2]
_OUTCOME_LABELS = {
    "C": "C 优于 B1", "B1": "B1 优于 C", "tie": "平局", "unstable": "顺序不稳定",
}


def _require_tmp_output(output: Path) -> None:
    destination = output.resolve()
    if not destination.is_relative_to((_ROOT / "tmp").resolve()) or output.suffix.lower() != ".md":
        raise ValueError("Private review pack must be a Markdown file under the repository's ignored tmp/ directory.")


def _fenced(value: str) -> str:
    """Keep model text literal even when it contains Markdown or HTML."""
    longest = max((len(match.group()) for match in re.finditer(r"~+", value)), default=0)
    fence = "~" * max(3, longest + 1)
    return f"{fence}\n{value.rstrip()}\n{fence}"


def _source_records(path: Path, expected_sha: str) -> dict[str, dict[str, Any]]:
    if sha256_file(path) != expected_sha:
        raise ValueError(f"Source records hash differs from the frozen package: {path}")
    rows = _read_jsonl(path)
    by_id = {row.get("blind_id"): row for row in rows}
    if len(rows) != 80 or len(by_id) != 80 or set(by_id) != {
        f"E{index:03d}" for index in range(1, 81)
    }:
        raise ValueError("B1/C source records must have 80 unique frozen image IDs.")
    for blind_id, row in by_id.items():
        answer = row.get("response")
        if (not isinstance(answer, str) or not answer.strip()
                or row.get("response_sha256") != _text_sha256(answer)):
            raise ValueError(f"{blind_id}: source response or response hash is invalid.")
    return by_id


def _text_sha256(value: str) -> str:
    import hashlib
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _validate_aggregate(
    report: dict[str, Any], package: dict[str, Any], package_dir: Path,
    score_path: Path, trace_hashes: dict[str, str],
) -> None:
    expected_input = {
        "package": sha256_file(package_dir / "preference_package.json"),
        "items": package["preference_items_sha256"],
        "mapping": package["preference_mapping_sha256"],
        "judgements": sha256_file(score_path),
        **trace_hashes,
    }
    expected = {
        "schema_version": 1,
        "evaluation_set": "Eval80-v1",
        "status": "provisional_proxy",
        "metric": "image_aware_pairwise_guidance_preference_proxy",
        "image_count": _IMAGE_COUNT,
        "judgement_count": _ITEM_COUNT,
        "judge_model_id": package["judge_model_id"],
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
        "input_sha256": expected_input,
    }
    if any(report.get(key) != value for key, value in expected.items()):
        raise ValueError("Preference aggregate report differs from the locked package or source records.")


def build_review_pack(
    *, package_dir: Path, report_path: Path, b1_run_dir: Path,
    c_run_dir: Path, output: Path,
) -> Path:
    """Validate everything before unblinding and writing a private review page."""
    _require_tmp_output(output)
    lock_path = package_dir / "preference_judgements_lock.json"
    if not lock_path.is_file():
        raise FileNotFoundError("Complete preference score lock is required before human review.")
    package = _read_object(package_dir / "preference_package.json")
    if package.get("image_count") != _IMAGE_COUNT or package.get("item_count") != _ITEM_COUNT:
        raise ValueError("Preference package does not cover 78 images and 156 items.")
    packet_path = package_dir / "preference_items_private.jsonl"
    mapping_path = package_dir / "preference_mapping_private.json"
    score_path = package_dir / "preference_judgements_private.jsonl"
    scores = _validate_judgements(score_path, packet_path, package)
    trace_hashes = _validate_script_trace(package_dir, package, scores)
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
    if _read_object(lock_path) != expected_lock:
        raise ValueError("Preference score lock differs from the complete runner trace.")
    if sha256_file(mapping_path) != package["preference_mapping_sha256"]:
        raise ValueError("Private preference mapping hash differs from the frozen package.")
    packets = {row["anonymous_item_id"]: row for row in _read_jsonl(packet_path)}
    by_image = _validate_mapping(_read_object(mapping_path), packets)
    if len(packets) != _ITEM_COUNT or len(by_image) != _IMAGE_COUNT:
        raise ValueError("Preference review requires complete 78/156 coverage.")
    report = _read_object(report_path)
    _validate_aggregate(report, package, package_dir, score_path, trace_hashes)
    b1_records = _source_records(b1_run_dir / "records.jsonl", package["b1_records_sha256"])
    c_records = _source_records(c_run_dir / "records.jsonl", package["c_records_sha256"])

    cases: list[dict[str, Any]] = []
    votes: Counter[str] = Counter()
    consensus: Counter[str] = Counter()
    identical_ties = 0
    for blind_id in sorted(by_image):
        pair = sorted(by_image[blind_id], key=lambda entry: entry[1]["variant"])
        decisions: list[dict[str, Any]] = []
        image_paths: set[Path] = set()
        identical = all(packets[item_id]["answer_A_sha256"] ==
                        packets[item_id]["answer_B_sha256"] for item_id, _ in pair)
        for item_id, mapping in pair:
            packet = packets[item_id]
            image_name = Path(packet["image_path"])
            if not image_name.is_absolute():
                raise ValueError(f"{blind_id}: review image path is not absolute.")
            image_path = image_name.resolve()
            if not image_path.is_file() or sha256_file(image_path) != packet["image_sha256"]:
                raise ValueError(f"{blind_id}: review image differs from the frozen image hash.")
            image_paths.add(image_path)
            for position in ("A", "B"):
                condition = mapping[f"{position}_condition"]
                row = (b1_records if condition == "B1" else c_records)[blind_id]
                if (packet[f"answer_{position}"] != row["response"]
                        or packet[f"answer_{position}_sha256"] != row["response_sha256"]):
                    raise ValueError(f"{blind_id}: judge packet answer differs from saved {condition} response.")
            winner = scores[item_id]["winner"]
            condition_vote = "tie" if winner == "tie" else mapping[f"{winner}_condition"]
            votes[condition_vote] += 1
            decisions.append({
                "variant": mapping["variant"],
                "A_condition": mapping["A_condition"],
                "B_condition": mapping["B_condition"],
                "condition_vote": condition_vote,
                "reason": scores[item_id]["reason"],
            })
        if len(image_paths) != 1:
            raise ValueError(f"{blind_id}: swapped items refer to different images.")
        if identical:
            outcome = "tie"
            identical_ties += 1
        else:
            outcome = (decisions[0]["condition_vote"] if
                       decisions[0]["condition_vote"] == decisions[1]["condition_vote"]
                       else "unstable")
        consensus[outcome] += 1
        cases.append({
            "blind_id": blind_id, "image_path": next(iter(image_paths)),
            "target_emotion": packets[pair[0][0]]["target_emotion"],
            "b1_answer": b1_records[blind_id]["response"],
            "c_answer": c_records[blind_id]["response"],
            "outcome": outcome, "identical": identical, "decisions": decisions,
        })
    expected_consensus = {
        "C_wins": consensus["C"], "B1_wins": consensus["B1"],
        "ties": consensus["tie"], "identical_answer_rule_ties": identical_ties,
        "swap_disagreements_unstable": consensus["unstable"],
        "stable_net_c_minus_b1": consensus["C"] - consensus["B1"],
        "worst_case_net_if_all_unstable_favor_b1": (
            consensus["C"] - consensus["B1"] - consensus["unstable"]
        ),
    }
    expected_votes = {"C": votes["C"], "B1": votes["B1"], "tie": votes["tie"],
                      "net_c_minus_b1": votes["C"] - votes["B1"]}
    if (len(cases) != _IMAGE_COUNT or sum(votes.values()) != _ITEM_COUNT
            or report.get("image_consensus") != expected_consensus
            or report.get("item_votes") != expected_votes):
        raise ValueError("Preference aggregate counts differ from the 78 recomputed image pairs.")

    lines = [
        "# Eval80 B1/C guidance 人工审阅包（本地私有）", "",
        "> **状态：待人工复核。** 下列胜负是本地图像模型的成对偏好代理指标，",
        "> 不是专家审美评分，也不能单独证明建议质量提升或 LoRA 正式验收。",
        "> 两次交换顺序的判定不一致时记为“顺序不稳定”。请结合原图亲自核对视觉事实和建议的可执行性。", "",
        f"- 图像：{_IMAGE_COUNT} 张；匿名判定：{_ITEM_COUNT} 条",
        f"- C 胜：{consensus['C']}；B1 胜：{consensus['B1']}；平局：{consensus['tie']}；顺序不稳定：{consensus['unstable']}",
        f"- 判定锁 SHA-256：`{sha256_file(lock_path)}`",
        f"- 汇总报告 SHA-256：`{sha256_file(report_path)}`", "",
        "## 快速目录", "",
    ]
    lines.extend(f"- [{case['blind_id']} · {_OUTCOME_LABELS[case['outcome']]}](#{case['blind_id'].lower()})"
                 for case in cases)
    for case in cases:
        image_path = case["image_path"]
        lines.extend([
            "", f"## {case['blind_id']}", "",
            f"**代理结论：{_OUTCOME_LABELS[case['outcome']]}。** " +
            ("两份回答逐字相同，按预设规则计平局。" if case["identical"] else ""),
            f"目标情绪：`{case['target_emotion']}`", "",
            f"![{case['blind_id']} 原图](<{image_path.as_posix()}>)", "",
            f"原图路径：`{image_path.as_posix()}`", "",
            "### B1 原始回答", "", _fenced(case["b1_answer"]), "",
            "### C 原始回答", "", _fenced(case["c_answer"]), "",
            "### 两次交换顺序的模型判定", "",
        ])
        for decision in case["decisions"]:
            lines.extend([
                f"- 第 {decision['variant'] + 1} 次：A = {decision['A_condition']}，"
                f"B = {decision['B_condition']}；判定："
                f"{_OUTCOME_LABELS[decision['condition_vote']] if decision['condition_vote'] != 'tie' else '平局'}",
                "", _fenced(decision["reason"]), "",
            ])
        lines.extend(["人工复核意见：", "", "", "---"])
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_bytes_if_absent(output, ("\n".join(lines).rstrip() + "\n").encode("utf-8"))
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a private 78-image review page from a locked preference run.")
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--b1-run-dir", type=Path, required=True)
    parser.add_argument("--c-run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = build_review_pack(
        package_dir=args.package_dir, report_path=args.report,
        b1_run_dir=args.b1_run_dir, c_run_dir=args.c_run_dir, output=args.output,
    )
    print(json.dumps({"status": "awaiting_human_review", "image_count": _IMAGE_COUNT,
                      "output": str(output.resolve())}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
