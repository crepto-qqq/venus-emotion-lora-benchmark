"""Synthetic trace contract for an honest incomplete benchmark summary."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.eval80.benchmark_status import build_benchmark_status, main
from src.eval80.core import sha256_file


RECORDS = {"B1": "a" * 64, "C": "b" * 64}
SELECTION = "c" * 64


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False) + "\n", encoding="utf-8")


def _aggregates():
    deterministic = {
        "schema_version": 1, "evaluation_set": "Eval80-v1",
        "comparison": "B1_vs_C_adapter_effect", "status": "incomplete",
        "missing_inputs": ["guidance"], "guidance_proxy": None,
        "format": {key: {"pass_count": score, "item_count": 80,
                         "records_sha256": RECORDS[key],
                         "failure_reason_counts": ({"missing_final_emotion": 80}
                                                   if key == "B1" else {})}
                   for key, score in (("B1", 0), ("C", 80))},
        "recognition": {key: {"correct_count": score, "item_count": 80,
                              "records_sha256": RECORDS[key]}
                        for key, score in (("B1", 0), ("C", 57))},
    }
    opening = {
        "schema_version": 1, "evaluation_set": "Eval80-v1",
        "metric": "supplementary_opening_Emotion_field_only",
        "protocol_metric": False, "image_count": 80,
        "parsed_opening_count": {"B1": 80, "C": 80},
        "correct_count": {"B1": 35, "C": 57},
        "net_correct_gain_c_minus_b1": 22,
        "paired_outcomes": {"both_correct": 33, "b1_only_correct": 2,
                            "c_only_correct": 24, "both_wrong": 21},
        "input_sha256": {"b1_records": RECORDS["B1"], "c_records": RECORDS["C"],
                         "selection_manifest": SELECTION},
    }
    return deterministic, opening


def _attempt(root: Path, name: str, cap: int) -> Path:
    directory = root / name
    directory.mkdir()
    files = {
        "preference_items_private.jsonl": "{}\n" * 156,
        "preference_mapping_private.json": "{}\n",
        "preflight_receipt_private.json": "{}\n",
        "model_digest_receipt_private.json": "{}\n",
    }
    for filename, content in files.items():
        (directory / filename).write_text(content, encoding="utf-8")
    _write_json(directory / "preference_config_private.json",
                {"model_id": "synthetic-vision:v1", "decoding": {"max_tokens": cap}})
    package = {
        "schema_version": 1, "evaluation_set": "Eval80-v1",
        "metric": "image_aware_pairwise_guidance_preference_proxy",
        "image_count": 78, "item_count": 156,
        "b1_records_sha256": RECORDS["B1"], "c_records_sha256": RECORDS["C"],
        "selection_manifest_sha256": SELECTION,
        "judge_model_id": "synthetic-vision:v1", "model_digest_sha256": "d" * 64,
        "inference_manifest_sha256": "e" * 64, "freeze_sha256": "f" * 64,
        "adapter_identity": {"selected_checkpoint": 180, "acceptance_status": "not_verified"},
        "prompt_sha256": "1" * 64, "response_schema_sha256": "2" * 64,
        "benchmark_source_sha256": "3" * 64, "runner_source_sha256": "4" * 64,
        "preference_config_sha256": sha256_file(directory / "preference_config_private.json"),
    }
    for filename, field in (
        ("preference_config_private.json", "preference_config_snapshot_sha256"),
        ("preference_items_private.jsonl", "preference_items_sha256"),
        ("preference_mapping_private.json", "preference_mapping_sha256"),
        ("preflight_receipt_private.json", "preflight_receipt_sha256"),
        ("model_digest_receipt_private.json", "model_digest_receipt_sha256"),
    ):
        package[field] = sha256_file(directory / filename)
    _write_json(directory / "preference_package.json", package)
    _write_json(directory / "preference_run_state.json", {
        "preference_package_sha256": sha256_file(directory / "preference_package.json"),
        "preference_config_sha256": package["preference_config_sha256"],
        "preference_items_sha256": package["preference_items_sha256"],
        "runtime_model_identity": {"checks": [{
            "phase": "before", "model_id": package["judge_model_id"],
            "digest_sha256": package["model_digest_sha256"],
        }]},
    })
    attempts, raw = [], []
    for anonymous_id, attempt_number, status in (
        [("item-0", 1, "error")]
        + [(f"item-{index}", 2 if index == 0 else 1, "success")
           for index in range(149)]
        + [("item-149", 1, "error"), ("item-149", 2, "error")]
    ):
        request_hash = "5" * 64
        attempts.append({"anonymous_item_id": anonymous_id,
                         "attempt": attempt_number, "request_sha256": request_hash,
                         "status": status})
        raw.append({"anonymous_item_id": anonymous_id,
                    "attempt": attempt_number, "request_sha256": request_hash,
                    "response": {"choices": [{"finish_reason": "length" if status == "error" else "stop"}],
                                 "usage": {"completion_tokens": cap if status == "error" else 10}}})
    for filename, rows in (("preference_attempts_private.jsonl", attempts),
                           ("preference_raw_responses_private.jsonl", raw)):
        (directory / filename).write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    # Count only. The status reporter is forbidden to parse decisions.
    (directory / "preference_judgements_private.jsonl").write_text(
        "{}\n" * 149, encoding="utf-8")
    return directory


class BenchmarkStatusTests(unittest.TestCase):
    def test_writes_no_partial_guidance_score_or_private_decisions(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            deterministic, opening = _aggregates()
            _write_json(root / "deterministic.json", deterministic)
            _write_json(root / "opening.json", opening)
            first = _attempt(root, "first", 550)
            second = _attempt(root, "second", 2048)
            result = build_benchmark_status(deterministic, opening, [first, second])
            self.assertIsNone(result["guidance_preference_proxy"]["score"])
            self.assertEqual([row["token_cap"] for row in result["guidance_preference_proxy"]["attempts"]],
                             [550, 2048])
            self.assertEqual(result["opening_emotion_diagnostic"]["net_correct_gain_c_minus_b1"], 22)
            main(["--deterministic", str(root / "deterministic.json"),
                  "--opening-diagnostic", str(root / "opening.json"),
                  "--attempt-dir", str(first), "--attempt-dir", str(second),
                  "--output-json", str(root / "status.json"),
                  "--output-md", str(root / "status.md")])
            output = (root / "status.json").read_text(encoding="utf-8")
            markdown = (root / "status.md").read_text(encoding="utf-8")
            self.assertIn("unavailable_technical_judge_failure", output)
            self.assertNotIn("anonymous_item_id", output)
            self.assertNotIn("item-149", output)
            self.assertIn("35/80", markdown)
            self.assertIn("149/156", markdown)
            self.assertIn("guidance 质量提升尚无有效自动分数", markdown)

    def test_rejects_changed_source_or_unlocked_partial_promoted_to_lock(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            deterministic, opening = _aggregates()
            first = _attempt(root, "first", 550)
            second = _attempt(root, "second", 2048)
            opening["input_sha256"]["c_records"] = "9" * 64
            with self.assertRaisesRegex(ValueError, "different B1/C records"):
                build_benchmark_status(deterministic, opening, [first, second])
            opening["input_sha256"]["c_records"] = RECORDS["C"]
            (first / "preference_judgements_lock.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "score lock exists"):
                build_benchmark_status(deterministic, opening, [first, second])

    def test_rejects_response_that_did_not_end_at_frozen_cap(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            deterministic, opening = _aggregates()
            first = _attempt(root, "first", 550)
            second = _attempt(root, "second", 2048)
            path = second / "preference_raw_responses_private.jsonl"
            lines = path.read_text(encoding="utf-8").splitlines()
            last = json.loads(lines[-1])
            last["response"]["usage"]["completion_tokens"] = 2047
            lines[-1] = json.dumps(last)
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "frozen output cap"):
                build_benchmark_status(deterministic, opening, [first, second])


if __name__ == "__main__":
    unittest.main()
