import csv
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

from src.eval80.core import (
    EMOTION_CLASSES,
    build_inference_manifest,
    sha256_file,
    sha256_text_file,
    validate_freeze_record,
    validate_human_reviews,
    validate_inference_manifest,
    validate_selection_manifest,
)
from src.eval80.validation import freeze_command


class Eval80ValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.image_dir = self.root / "images_blind"
        self.image_dir.mkdir()
        records = []
        counter = 1
        for emotion in EMOTION_CLASSES:
            for class_index in range(10):
                blind_id = f"E{counter:03d}"
                filename = f"{blind_id}.jpg"
                image_path = self.image_dir / filename
                image_path.write_bytes(f"{emotion}-{class_index}".encode("ascii"))
                records.append(
                    {
                        "blind_id": blind_id,
                        "emotion": emotion,
                        "image_id": f"{emotion}_{class_index:05d}",
                        "local_image_filename": filename,
                        "image_sha256": sha256_file(image_path),
                        "width": 100,
                        "height": 80,
                    }
                )
                counter += 1
        self.selection = {"summary": {"selected_image_count": 80}, "records": records}

    def tearDown(self):
        self.temp.cleanup()

    def review_rows(self):
        blind = []
        reveal = []
        for record in self.selection["records"]:
            blind.append(
                {
                    "blind_id": record["blind_id"],
                    "independent_label": record["emotion"],
                    "confidence_1_to_3": "3",
                    "blind_notes": "",
                }
            )
            reveal.append(
                {
                    "blind_id": record["blind_id"],
                    "emoset_label": record["emotion"],
                    "independent_label": record["emotion"],
                    "agreement": "yes",
                    "review_flag": "clear",
                    "reveal_notes": "",
                }
            )
        return blind, reveal

    def test_validates_selection_and_label_free_inference_manifest(self):
        self.assertEqual(validate_selection_manifest(self.selection, self.image_dir), [])
        inference = build_inference_manifest(self.selection)
        self.assertFalse(inference["contains_ground_truth_labels"])
        self.assertNotIn("emotion", inference["records"][0])
        self.assertEqual(validate_inference_manifest(inference, self.image_dir), [])

    def test_validates_complete_human_review(self):
        blind, reveal = self.review_rows()
        errors, summary = validate_human_reviews(blind, reveal, self.selection)
        self.assertEqual(errors, [])
        self.assertEqual(summary["independent_label_agreement_count"], 80)
        self.assertEqual(summary["review_flag_counts"]["clear"], 80)

    def test_requires_notes_for_low_confidence_and_ambiguity(self):
        blind, reveal = self.review_rows()
        blind[0]["confidence_1_to_3"] = "1"
        reveal[0]["review_flag"] = "ambiguous"
        errors, _ = validate_human_reviews(blind, reveal, self.selection)
        self.assertTrue(any("blind_notes is required" in error for error in errors))
        self.assertTrue(any("reveal_notes is required" in error for error in errors))

    def test_validates_freeze_record_hashes(self):
        files = {}
        for name in ("config", "handbook", "manifest"):
            path = self.root / f"{name}.txt"
            path.write_text(name, encoding="utf-8")
            files[name] = path
        freeze = {
            "status": "approved_and_frozen",
            "approver_ids": ["T1"],
            "approved_at_utc": "2026-09-07T00:00:00+00:00",
            "file_sha256": {
                "config": sha256_file(files["config"]),
                "scoring_handbook": sha256_file(files["handbook"]),
                "inference_manifest": sha256_file(files["manifest"]),
            },
        }
        self.assertEqual(
            validate_freeze_record(
                freeze,
                config_path=files["config"],
                handbook_path=files["handbook"],
                inference_manifest_path=files["manifest"],
            ),
            [],
        )

    def test_freeze_text_hashes_are_stable_across_line_endings(self):
        config = self.root / "config.json"
        handbook = self.root / "handbook.md"
        manifest = self.root / "manifest.json"
        config.write_bytes(b"{\n  \"size\": 80\n}\n")
        handbook.write_bytes(b"# Handbook\n\nStatus: frozen\n")
        manifest.write_bytes(b"{}\n")
        freeze = {
            "status": "approved_and_frozen",
            "approver_ids": ["team-member-01"],
            "approved_at_utc": "2026-09-07T00:00:00+00:00",
            "file_sha256": {
                "config": sha256_text_file(config),
                "scoring_handbook": sha256_text_file(handbook),
                "inference_manifest": sha256_file(manifest),
            },
        }
        config.write_bytes(b"{\r\n  \"size\": 80\r\n}\r\n")
        handbook.write_bytes(b"# Handbook\r\n\r\nStatus: frozen\r\n")
        self.assertEqual(
            validate_freeze_record(
                freeze,
                config_path=config,
                handbook_path=handbook,
                inference_manifest_path=manifest,
            ),
            [],
        )

    def test_freeze_command_writes_the_approved_package(self):
        selection_path = self.root / "selection.json"
        blind_path = self.root / "blind.csv"
        reveal_path = self.root / "reveal.csv"
        config_path = self.root / "config.json"
        handbook_path = self.root / "handbook.md"
        reading_copy_path = self.root / "handbook_zh.md"
        protocol_path = self.root / "protocol.md"
        output_dir = self.root / "frozen"
        selection_path.write_text(
            json.dumps(self.selection, indent=2) + "\n",
            encoding="utf-8",
        )
        blind_rows, reveal_rows = self.review_rows()
        with blind_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(blind_rows[0]))
            writer.writeheader()
            writer.writerows(blind_rows)
        with reveal_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(reveal_rows[0]))
            writer.writeheader()
            writer.writerows(reveal_rows)
        config_path.write_text("{}\n", encoding="utf-8")
        handbook_path.write_text("# Handbook\n\nStatus: frozen\n", encoding="utf-8")
        reading_copy_path.write_text("# 手册\n\n状态：已冻结\n", encoding="utf-8")
        protocol_path.write_text("# Protocol\n", encoding="utf-8")

        arguments = [
            "--selection-manifest",
            str(selection_path),
            "--image-dir",
            str(self.image_dir),
            "--blind-review",
            str(blind_path),
            "--reveal-review",
            str(reveal_path),
            "--config",
            str(config_path),
            "--handbook",
            str(handbook_path),
            "--reading-copy",
            str(reading_copy_path),
            "--protocol",
            str(protocol_path),
            "--output-dir",
            str(output_dir),
            "--approver-ids",
            *[f"team-member-{index:02d}" for index in range(1, 7)],
            "--team-approved",
        ]
        with redirect_stdout(io.StringIO()):
            result = freeze_command(arguments)

        self.assertEqual(result, 0)
        self.assertTrue((output_dir / "inference_manifest.json").is_file())
        freeze = json.loads(
            (output_dir / "freeze_record.json").read_text(encoding="utf-8")
        )
        self.assertEqual(freeze["status"], "approved_and_frozen")
        self.assertEqual(len(freeze["approver_ids"]), 6)
        self.assertEqual(
            freeze["file_sha256"]["scoring_handbook"],
            sha256_text_file(handbook_path),
        )


if __name__ == "__main__":
    unittest.main()
