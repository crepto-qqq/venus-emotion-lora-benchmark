"""Synthetic invariants for the post-failure paired-judge traversal."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from src.eval80.preference_benchmark import _packet_sha256
from src.eval80.preference_recovery import (
    _aggregate, _load, _pairs, _reference_summary, _validate_log, prepare_command,
)
from src.eval80.preference_judge_runner import _request_sha256


def _fixture():
    packets = []
    mapping = {}
    for variant in (0, 1):
        for image in range(78):
            anonymous_id = f"P{variant}-{image:02d}"
            answer_a, answer_b = f"{image + 100:064x}", f"{image + 200:064x}"
            if variant:
                answer_a, answer_b = answer_b, answer_a
            packet = {
                "anonymous_item_id": anonymous_id,
                "image_sha256": f"{image + 1:064x}",
                "target_emotion": "contentment",
                "answer_A_sha256": answer_a,
                "answer_B_sha256": answer_b,
            }
            packets.append(packet)
            mapping[anonymous_id] = {
                "blind_id": f"E{image + 2:03d}", "variant": variant,
                "A_condition": "C" if variant == 0 else "B1",
                "B_condition": "B1" if variant == 0 else "C",
                **{key: packet[key] for key in (
                    "image_sha256", "target_emotion", "answer_A_sha256", "answer_B_sha256")},
            }
    by_id, mate = _pairs(packets)
    scores = {row["anonymous_item_id"]: {
                  "winner": "A" if index < 78 else "B", "reason": "Synthetic reason."}
              for index, row in enumerate(packets[:149])}
    source = {
        "packets": packets, "by_id": by_id, "mate": mate, "scores": scores,
        "initial_failed_id": packets[149]["anonymous_item_id"],
        "config": {"retry_policy": {"max_attempts": 2}},
        "package": {
            "judge_model_id": "synthetic-local-judge",
            "b1_records_sha256": "1" * 64,
            "c_records_sha256": "2" * 64,
            "inference_manifest_sha256": "3" * 64,
            "selection_manifest_sha256": "4" * 64,
            "preference_config_sha256": "5" * 64,
            "model_digest_sha256": "6" * 64,
        },
    }
    return source, {"schema_version": 1, "items": mapping}


def _row(packet, attempt=1, winner="A"):
    response = {"winner": winner}
    return {
        "anonymous_item_id": packet["anonymous_item_id"], "attempt": attempt,
        "packet_sha256": _packet_sha256(packet),
        "request_sha256": _request_sha256({"packet": packet}),
        "status": "error" if winner == "length" else "success",
        "response": response,
        "judgement": None if winner == "length" else {
            "winner": winner, "reason": "Synthetic reason."},
        "error_type": "ValueError" if winner == "length" else None,
    }


def _draft_bytes():
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer)
    writer.writerow(("image_code", "reviewer_id", "preferred",
                     "obvious_hallucinated_worse", "reason"))
    for index in range(16):
        writer.writerow((f"D{index + 1:03d}", "lt",
                         "A" if index == 0 else "a" if index == 1 else "",
                         "tie" if index == 2 else "", ""))
    return buffer.getvalue().encode("utf-8")


class PreferenceRecoveryTests(unittest.TestCase):
    def _validate(self, rows, source, *, complete=True):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "attempts.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

            def extract(response, packet, metadata):
                if response["winner"] == "length":
                    raise ValueError("response did not finish normally")
                return {"winner": response["winner"], "reason": "Synthetic reason."}

            with patch("src.eval80.preference_recovery._make_payload",
                       side_effect=lambda packet, config: {"packet": packet}), patch(
                "src.eval80.preference_recovery._extract_judgement", side_effect=extract
            ):
                return _validate_log(path, source, require_complete=complete)

    def test_reuses_149_and_excludes_both_orders_of_terminal_image(self):
        source, mapping = _fixture()
        self.assertEqual(len(source["scores"]), 149)
        self.assertEqual(source["mate"][source["initial_failed_id"]], "P0-71")
        rows = [_row(packet, winner="A" if index == 155 else "B")
                for index, packet in enumerate(source["packets"]) if index >= 150]
        scores, excluded, errors = self._validate(rows, source)
        self.assertEqual(len(scores), 155)  # inherited mate remains audited but unscored
        self.assertEqual(len(excluded), 1)
        self.assertEqual(errors, {})
        summary, private = _aggregate(source, scores, excluded, mapping)
        self.assertEqual(summary["included_images"], 77)
        self.assertEqual(summary["included_judgements"], 154)
        self.assertEqual(summary["technical_excluded_images"], 1)
        self.assertEqual(summary["source_records_sha256"]["B1"], "1" * 64)
        self.assertEqual(summary["source_records_sha256"]["C"], "2" * 64)
        self.assertEqual(summary["image_consensus"]["C_wins"], 76)
        self.assertEqual(summary["image_consensus"]["swap_disagreements_unstable"], 1)
        self.assertEqual(summary["observed_net_c_minus_b1_rate"], 76 / 77)
        self.assertEqual(summary["original_scope_worst_case_net_rate"], (76 - 2) / 78)
        self.assertEqual(len(private["cases"]), 78)

    def test_new_failure_excludes_entire_image_and_traverses_the_rest(self):
        source, mapping = _fixture()
        failed = source["packets"][150]
        rows = [_row(failed, 1, "length"), _row(failed, 2, "length")]
        rows.extend(_row(packet, winner="A" if index == 155 else "B")
                    for index, packet in enumerate(source["packets"]) if index >= 151)
        scores, excluded, errors = self._validate(rows, source)
        summary, _ = _aggregate(source, scores, excluded, mapping)
        self.assertEqual(len(excluded), 2)
        self.assertEqual(errors, {"ValueError": 2})
        self.assertEqual(summary["included_images"], 76)
        self.assertEqual(summary["included_judgements"], 152)
        self.assertEqual(summary["image_consensus"]["C_wins"], 75)
        self.assertEqual(summary["image_consensus"]["swap_disagreements_unstable"], 1)
        self.assertEqual(summary["original_scope_worst_case_net_rate"], (75 - 3) / 78)
        with self.assertRaisesRegex(ValueError, "excluded image"):
            self._validate(rows + [_row(failed, 3)], source)

    def test_requires_full_traversal_and_exact_swapped_pairs(self):
        source, _ = _fixture()
        with self.assertRaisesRegex(ValueError, "Full traversal is incomplete"):
            self._validate([_row(packet) for packet in source["packets"][150:155]], source)
        packets = [dict(packet) for packet in source["packets"]]
        packets[149]["answer_A_sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "exact answer swaps"):
            _pairs(packets)

    def test_transient_socket_timeout_consumes_a_frozen_retry(self):
        source, _ = _fixture()
        packet = source["packets"][150]
        timeout_row = _row(packet, attempt=1)
        timeout_row.update({"status": "error", "response": None,
                            "judgement": None, "error_type": "OSError"})
        rows = [timeout_row, _row(packet, attempt=2)]
        rows.extend(_row(other) for other in source["packets"][151:])
        scores, excluded, errors = self._validate(rows, source)
        self.assertEqual(errors, {"OSError": 1})
        self.assertEqual(len(excluded), 1)
        self.assertIn(packet["anonymous_item_id"], scores)

    def test_incomplete_reference_is_structurally_valid_without_becoming_labels(self):
        draft = _draft_bytes()
        self.assertEqual(_reference_summary(draft), {
            "row_count": 16,
            "preferred_nonempty_count": 2,
            "hallucination_nonempty_count": 1,
            "reason_nonempty_count": 0,
        })
        bad = draft.decode("utf-8").replace("D016", "D001")
        with self.assertRaisesRegex(ValueError, "duplicate or invalid"):
            _reference_summary(bad.encode("utf-8"))

    def test_reference_prepare_copies_draft_and_rechecks_hash_without_gate(self):
        source, _ = _fixture()
        source["config"]["endpoint"] = "http://127.0.0.1:12345/v1/chat/completions"
        source["source_hashes"] = {"preference_package.json": "a" * 64}
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            draft_path = root / "draft.csv"
            draft_path.write_bytes(_draft_bytes())
            recovery_dir = root / "recovery"
            with patch("src.eval80.preference_recovery._source", return_value=source):
                self.assertEqual(prepare_command([
                    "--source-package", str(root / "attempt002"),
                    "--human-reference-csv", str(draft_path),
                    "--output-dir", str(recovery_dir),
                ]), 0)
                metadata = json.loads((recovery_dir / "recovery_package_private.json")
                                      .read_text(encoding="utf-8"))
                self.assertEqual(metadata["calibration_mode"], "reference_only_unverified")
                self.assertNotIn("calibration_gate_sha256", metadata)
                copied = recovery_dir / "human_reference_private.csv"
                self.assertEqual(copied.read_bytes(), draft_path.read_bytes())
                self.assertEqual(_load(recovery_dir)[1]["calibration_mode"],
                                 "reference_only_unverified")
                copied.write_bytes(copied.read_bytes() + b"\n")
                with self.assertRaisesRegex(ValueError, "changed after preparation"):
                    _load(recovery_dir)

    def test_passed_gate_prepare_remains_available_without_reference_csv(self):
        source, _ = _fixture()
        source["config"]["endpoint"] = "http://127.0.0.1:12345/v1/chat/completions"
        source["source_hashes"] = {"preference_package.json": "a" * 64}
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            gate_path = root / "gate.json"
            gate_path.write_text("{}\n", encoding="utf-8")
            recovery_dir = root / "recovery"
            with patch("src.eval80.preference_recovery._source", return_value=source), patch(
                "src.eval80.preference_recovery._verify_gate", return_value={"pair_count": 16}
            ):
                self.assertEqual(prepare_command([
                    "--source-package", str(root / "attempt002"),
                    "--calibration-gate", str(gate_path),
                    "--output-dir", str(recovery_dir),
                ]), 0)
                metadata = json.loads((recovery_dir / "recovery_package_private.json")
                                      .read_text(encoding="utf-8"))
                self.assertEqual(metadata["calibration_mode"], "passed_human_gate")
                self.assertEqual(metadata["calibration_gate_sha256"],
                                 metadata["calibration_evidence_sha256"])
                self.assertNotIn("human_reference_csv_name", metadata)
                self.assertEqual(_load(recovery_dir)[1]["calibration_mode"],
                                 "passed_human_gate")

    def test_reference_aggregate_is_distinct_and_does_not_claim_calibration(self):
        source, mapping = _fixture()
        source["calibration_mode"] = "reference_only_unverified"
        rows = [_row(packet) for packet in source["packets"][150:]]
        scores, excluded, _ = self._validate(rows, source)
        summary, _ = _aggregate(source, scores, excluded, mapping)
        self.assertEqual(summary["metric"],
                         "uncalibrated_image_aware_pairwise_guidance_preference")
        self.assertIn("incomplete human calibration", summary["interpretation_note"])
        self.assertEqual(summary["attempt002_valid_decisions_inherited"], 149)
        self.assertEqual(summary["technical_excluded_images"], 1)


if __name__ == "__main__":
    unittest.main()
