"""Paired, aggregate-only Eval80 B1/C recognition benchmarking."""

from __future__ import annotations

import csv
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.eval80.cli import main
from src.eval80.core import EMOTION_CLASSES
from src.eval80.scoring import compare_recognition_details, summarize_recognition


class CompareRecognitionTests(unittest.TestCase):
    def details(self):
        targets = {
            f"E{index:03d}": EMOTION_CLASSES[(index - 1) // 10]
            for index in range(1, 81)
        }
        outputs = {"B1": [], "C": []}
        for blind_id, target in targets.items():
            for condition in ("B1", "C"):
                prediction = target
                if blind_id == "E002" and condition == "C":
                    prediction = "anger"
                if blind_id == "E011" and condition == "B1":
                    prediction = "awe"
                if blind_id == "E021" and condition == "B1":
                    response = ""
                elif blind_id == "E021" and condition == "C":
                    response = "Emotion: anger\nFinal emotion: awe"
                else:
                    response = f"Emotion: {prediction}\nFinal emotion: {prediction}"
                outputs[condition].append({"blind_id": blind_id, "response": response})
        details = {}
        for condition in ("B1", "C"):
            rows, _ = summarize_recognition(outputs[condition], targets, structured=True)
            for row in rows:
                row["condition"] = condition
                row["correct"] = str(row["correct"])
            details[condition] = rows
        return details

    def test_paired_counts_per_class_and_parse_statuses(self):
        details = self.details()
        report = compare_recognition_details(details["B1"], details["C"])
        self.assertEqual(report["image_count"], 80)
        self.assertEqual(report["paired_outcomes"], {
            "both_correct": 77,
            "b1_only_correct": 1,
            "c_only_correct": 1,
            "both_wrong": 1,
        })
        self.assertEqual(report["conditions"]["B1"]["correct_count"], 78)
        self.assertEqual(report["conditions"]["C"]["correct_count"], 78)
        self.assertEqual(report["conditions"]["B1"]["parse_status_counts"], {
            "missing": 1, "valid": 79
        })
        self.assertEqual(report["conditions"]["C"]["parse_status_counts"], {
            "ambiguous": 1, "valid": 79
        })
        self.assertEqual(report["parse_status_transitions"], {
            "missing->ambiguous": 1, "valid->valid": 79
        })
        self.assertEqual(report["per_class"]["amusement"]["correct_delta_c_minus_b1"], -1)
        self.assertEqual(report["per_class"]["anger"]["correct_delta_c_minus_b1"], 1)
        self.assertEqual(report["net_correct_gain_c_minus_b1"], 0)
        self.assertNotIn("rows", report)

    def test_rejects_wrong_condition_duplicate_id_and_inconsistent_correctness(self):
        details = self.details()
        details["C"][0]["condition"] = "B1"
        with self.assertRaisesRegex(ValueError, "condition mismatch"):
            compare_recognition_details(details["B1"], details["C"])
        details = self.details()
        details["C"][0]["blind_id"] = "E002"
        with self.assertRaisesRegex(ValueError, "duplicate blind ID"):
            compare_recognition_details(details["B1"], details["C"])
        details = self.details()
        details["C"][0]["correct"] = "False"
        with self.assertRaisesRegex(ValueError, "correctness disagrees"):
            compare_recognition_details(details["B1"], details["C"])

    def test_rejects_misaligned_truth_with_balanced_class_counts(self):
        details = self.details()
        c_rows = details["C"]
        c_rows[0]["target_emotion"], c_rows[10]["target_emotion"] = (
            c_rows[10]["target_emotion"], c_rows[0]["target_emotion"]
        )
        c_rows[0]["correct"] = "False"
        c_rows[10]["correct"] = "False"
        with self.assertRaisesRegex(ValueError, "target emotion mismatch"):
            compare_recognition_details(details["B1"], c_rows)

    def test_cli_writes_only_aggregate_json(self):
        details = self.details()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {}
            for condition in ("B1", "C"):
                path = root / f"{condition}_recognition_details.csv"
                with path.open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(details[condition][0]))
                    writer.writeheader()
                    writer.writerows(details[condition])
                paths[condition] = path
            output = root / "paired_summary.json"
            with redirect_stdout(io.StringIO()):
                result = main([
                    "compare-recognition",
                    "--b1-details", str(paths["B1"]),
                    "--c-details", str(paths["C"]),
                    "--output", str(output),
                ])
            self.assertEqual(result, 0)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["paired_outcomes"]["both_correct"], 77)
            self.assertEqual(set(report["input_sha256"]), {"b1_details", "c_details"})
            self.assertNotIn("E001", output.read_text(encoding="utf-8"))
            with self.assertRaises(FileExistsError):
                main([
                    "compare-recognition",
                    "--b1-details", str(paths["B1"]),
                    "--c-details", str(paths["C"]),
                    "--output", str(output),
                ])

    def test_recognition_export_checks_frozen_manifest_and_raw_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            selection = root / "selection.json"
            selection.write_text(json.dumps({
                "records": [
                    {
                        "blind_id": f"E{index:03d}",
                        "emotion": EMOTION_CLASSES[(index - 1) // 10],
                    }
                    for index in range(1, 81)
                ]
            }), encoding="utf-8")
            records = []
            for index in range(1, 81):
                response = (
                    f"Emotion: {EMOTION_CLASSES[(index - 1) // 10]}\n"
                    f"Final emotion: {EMOTION_CLASSES[(index - 1) // 10]}"
                )
                records.append({
                    "blind_id": f"E{index:03d}",
                    "condition": "B1",
                    "status": "success",
                    "response": response,
                    "response_sha256": hashlib.sha256(response.encode("utf-8")).hexdigest(),
                })
            raw_path = root / "records.jsonl"
            raw_path.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            freeze_path = root / "freeze_record.json"
            freeze_path.write_text(json.dumps({
                "schema_version": 1,
                "status": "approved_and_frozen",
                "team_approval_confirmed": True,
                "file_sha256": {
                    "selection_manifest_private": hashlib.sha256(selection.read_bytes()).hexdigest()
                },
            }), encoding="utf-8")
            args = [
                "score-recognition", "--condition", "B1",
                "--records", str(raw_path),
                "--selection-manifest", str(selection),
                "--output-dir", str(root / "scored"),
            ]
            with patch("src.eval80.scoring.FREEZE_RECORD_PATH", freeze_path):
                with redirect_stdout(io.StringIO()):
                    self.assertEqual(main(args), 0)
            with (root / "scored/B1_recognition_details.csv").open(
                "r", encoding="utf-8", newline=""
            ) as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 80)
            self.assertEqual({row["condition"] for row in rows}, {"B1"})

            selection_payload = json.loads(selection.read_text(encoding="utf-8"))
            selection_payload["unapproved_note"] = "Changed after freeze"
            selection.write_text(json.dumps(selection_payload), encoding="utf-8")
            with patch("src.eval80.scoring.FREEZE_RECORD_PATH", freeze_path):
                with self.assertRaisesRegex(ValueError, "selection manifest SHA-256 differs"):
                    main(args[:-1] + [str(root / "scored-modified-selection")])
            selection_payload.pop("unapproved_note")
            selection.write_text(json.dumps(selection_payload), encoding="utf-8")
            freeze_path.write_text(json.dumps({
                "schema_version": 1,
                "status": "approved_and_frozen",
                "team_approval_confirmed": True,
                "file_sha256": {
                    "selection_manifest_private": hashlib.sha256(selection.read_bytes()).hexdigest()
                },
            }), encoding="utf-8")
            records[0]["response"] = "Tampered response"
            raw_path.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            with patch("src.eval80.scoring.FREEZE_RECORD_PATH", freeze_path):
                with self.assertRaisesRegex(ValueError, "response SHA-256 mismatch"):
                    main(args[:-1] + [str(root / "scored-tampered")])


if __name__ == "__main__":
    unittest.main()
