"""End-to-end checks for the approved Eval80 guidance display exclusion."""

from __future__ import annotations

import csv
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from src.eval80.cli import main
from src.eval80.core import EMOTION_CLASSES
from src.eval80.scoring import GUIDANCE_COLUMNS, REVIEWER_IDS


EXCLUSION = (
    Path(__file__).resolve().parents[2]
    / "Docs/evaluation/EVAL80_DISPLAY_EXCLUSION.json"
)


class GuidanceScopeCliTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.selection = self.root / "selection.json"
        records = [
            {
                "blind_id": f"E{index:03d}",
                "emotion": EMOTION_CLASSES[(index - 1) // 10],
                "local_image_filename": f"E{index:03d}.jpg",
            }
            for index in range(1, 81)
        ]
        self.selection.write_text(json.dumps({"records": records}), encoding="utf-8")
        self.output_paths = {}
        for condition in ("A", "B0", "B1", "C"):
            path = self.root / f"{condition}.jsonl"
            with path.open("w", encoding="utf-8") as handle:
                for record in records:
                    response = f"{condition} guidance for {record['blind_id']}\nAction: adjust the frame."
                    handle.write(
                        json.dumps(
                            {
                                "condition": condition,
                                "blind_id": record["blind_id"],
                                "status": "success",
                                "response": response,
                                "response_sha256": hashlib.sha256(
                                    response.encode("utf-8")
                                ).hexdigest(),
                            }
                        )
                        + "\n"
                    )
            self.output_paths[condition] = path

    def create_sheets(self):
        self.package = self.root / "scoring-package"
        with redirect_stdout(io.StringIO()):
            result = main([
                "create-score-sheets",
                "--selection-manifest", str(self.selection),
                "--display-exclusion", str(EXCLUSION),
                "--a-records", str(self.output_paths["A"]),
                "--b0-records", str(self.output_paths["B0"]),
                "--b1-records", str(self.output_paths["B1"]),
                "--c-records", str(self.output_paths["C"]),
                "--output-dir", str(self.package),
            ])
        self.assertEqual(result, 0)

    def fill_sheets(self):
        score_paths = []
        for reviewer_id in REVIEWER_IDS:
            path = self.package / f"guidance_scores_{reviewer_id}.csv"
            with path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                fields = reader.fieldnames
                rows = list(reader)
            self.assertEqual(len(rows), 52)
            for row in rows:
                row.update({column: "2" for column in GUIDANCE_COLUMNS})
                row["overall_rationale"] = "Specific, usable guidance."
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            score_paths.append(path)
        return score_paths

    def summarize(self, score_paths, *, exclusion=EXCLUSION, output_name="summary"):
        with redirect_stdout(io.StringIO()):
            return main([
                "summarize",
                "--scores", *map(str, score_paths),
                "--condition-mapping", str(self.package / "condition_mapping_private.json"),
                "--display-exclusion", str(exclusion),
                "--output-dir", str(self.root / output_name),
            ])

    def test_round_trip_generates_and_summarizes_78_images(self):
        self.create_sheets()
        mapping = json.loads(
            (self.package / "condition_mapping_private.json").read_text(encoding="utf-8")
        )
        self.assertEqual(mapping["item_count"], 312)
        self.assertEqual(set(mapping["reviewer_assignment"]), {
            f"E{index:03d}" for index in range(2, 80)
        })
        self.assertEqual(mapping["excluded_blind_ids"], ["E001", "E080"])
        score_paths = self.fill_sheets()
        self.assertEqual(self.summarize(score_paths), 0)
        summary = json.loads(
            (self.root / "summary/guidance_score_summary.json").read_text(encoding="utf-8")
        )
        self.assertEqual(summary["item_count"], 312)
        self.assertEqual(summary["image_count"], 78)
        self.assertEqual(summary["reviewer_image_counts"], {
            reviewer_id: 13 for reviewer_id in REVIEWER_IDS
        })
        self.assertEqual(summary["conditions"]["C"]["mean_guidance_total_0_10"], 10)

    def test_rejects_modified_response_and_exclusion_record(self):
        self.create_sheets()
        score_paths = self.fill_sheets()
        path = score_paths[0]
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = reader.fieldnames
            rows = list(reader)
        rows[0]["model_response"] += " Altered."
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        with self.assertRaisesRegex(ValueError, "model response SHA-256"):
            self.summarize(score_paths)

        rows[0]["model_response"] = rows[0]["model_response"].removesuffix(" Altered.")
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        altered_exclusion = self.root / "altered-exclusion.json"
        payload = json.loads(EXCLUSION.read_text(encoding="utf-8"))
        payload["reason"] += " Extra text."
        altered_exclusion.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "bound to a different display exclusion"):
            self.summarize(score_paths, exclusion=altered_exclusion)

    def test_rejects_excluded_csv_row(self):
        self.create_sheets()
        score_paths = self.fill_sheets()
        path = score_paths[0]
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = reader.fieldnames
            rows = list(reader)
        rows[0]["blind_id"] = "E001"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        with self.assertRaisesRegex(ValueError, "E001 is excluded"):
            self.summarize(score_paths)


if __name__ == "__main__":
    unittest.main()
