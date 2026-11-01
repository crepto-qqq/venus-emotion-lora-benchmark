"""Synthetic tests for the auxiliary Eval80 B1/C format audit."""

from __future__ import annotations

from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.eval80.cli import main
from src.eval80.output_contract import audit_output_contract, check_structured_response


VALID_RESPONSE = (
    "Emotion: awe\n\n"
    "Visual evidence:\n"
    "- visible green tree\n"
    "- bright sky\n\n"
    "Aesthetic relationship:\n"
    "The contrast draws attention to the tree.\n\n"
    "Guidance:\n"
    "- crop some empty space\n"
    "- retain the bright sky\n\n"
    "Final emotion: awe"
)


def _manifest() -> dict:
    return {
        "schema_version": 1,
        "evaluation_set": "Eval80-v1",
        "image_count": 80,
        "contains_ground_truth_labels": False,
        "records": [
            {
                "blind_id": f"E{index:03d}",
                "image_filename": f"E{index:03d}.jpg",
                "image_sha256": "0" * 64,
                "width": 100,
                "height": 100,
            }
            for index in range(1, 81)
        ],
    }


def _records(condition: str = "C") -> list[dict]:
    return [
        {
            "blind_id": f"E{index:03d}",
            "condition": condition,
            "status": "success",
            "response": VALID_RESPONSE,
            "response_sha256": hashlib.sha256(VALID_RESPONSE.encode("utf-8")).hexdigest(),
            "image_sha256": "0" * 64,
        }
        for index in range(1, 81)
    ]


def _set_response(record: dict, response: str) -> None:
    record["response"] = response
    record["response_sha256"] = hashlib.sha256(response.encode("utf-8")).hexdigest()


class StructuredResponseTests(unittest.TestCase):
    def test_frozen_sections_and_matching_labels_pass(self):
        self.assertEqual(check_structured_response(VALID_RESPONSE), ())
        self.assertEqual(
            check_structured_response(VALID_RESPONSE.replace("Emotion: awe", "Emotion: Awe")),
            (),
        )

    def test_missing_duplicate_invalid_and_mismatched_emotions(self):
        cases = {
            "missing_emotion": VALID_RESPONSE.replace("Emotion: awe\n\n", ""),
            "duplicate_emotion": VALID_RESPONSE.replace(
                "Emotion: awe", "Emotion: awe\nEmotion: awe", 1
            ),
            "invalid_emotion": VALID_RESPONSE.replace("Emotion: awe", "Emotion: joy", 1),
            "missing_final_emotion": VALID_RESPONSE.replace("Final emotion: awe", ""),
            "duplicate_final_emotion": VALID_RESPONSE + "\nFinal emotion: awe",
            "invalid_final_emotion": VALID_RESPONSE.replace("Final emotion: awe", "Final emotion: joy"),
            "emotion_mismatch": VALID_RESPONSE.replace("Final emotion: awe", "Final emotion: fear"),
        }
        for expected, response in cases.items():
            with self.subTest(expected=expected):
                self.assertIn(expected, check_structured_response(response))

    def test_bullets_relationship_order_and_extraneous_text(self):
        cases = {
            "visual_evidence_bullet_count": VALID_RESPONSE.replace("- bright sky\n", ""),
            "empty_visual_evidence_bullet": VALID_RESPONSE.replace("- bright sky", "- "),
            "guidance_bullet_count": VALID_RESPONSE.replace("- retain the bright sky\n", ""),
            "empty_guidance_bullet": VALID_RESPONSE.replace("- crop some empty space", "- "),
            "empty_aesthetic_relationship": VALID_RESPONSE.replace(
                "The contrast draws attention to the tree.", ""
            ),
            "section_order": VALID_RESPONSE.replace(
                "Visual evidence:", "TEMP:", 1
            ).replace("Guidance:", "Visual evidence:", 1).replace("TEMP:", "Guidance:", 1),
            "unexpected_content": "Preamble\n" + VALID_RESPONSE,
        }
        for expected, response in cases.items():
            with self.subTest(expected=expected):
                self.assertIn(expected, check_structured_response(response))

    def test_rejects_extra_relationship_heading_but_allows_colon_prose(self):
        for added_heading in ("Other section:", "## Other section"):
            with self.subTest(added_heading=added_heading):
                response = VALID_RESPONSE.replace(
                    "The contrast draws attention to the tree.",
                    f"The contrast draws attention to the tree.\n{added_heading}\nExtra text.",
                )
                self.assertIn("unexpected_content", check_structured_response(response))
        prose = VALID_RESPONSE.replace(
            "The contrast draws attention to the tree.",
            "Composition: warm tones frame the subject and draw the eye.",
        )
        self.assertEqual(check_structured_response(prose), ())


class OutputContractAuditTests(unittest.TestCase):
    def test_aggregate_counts_multiple_reasons_without_record_details(self):
        records = _records()
        _set_response(records[0], VALID_RESPONSE.replace("Final emotion: awe", "Final emotion: fear"))
        _set_response(records[1], "Emotion: joy")
        summary = audit_output_contract(records, _manifest(), condition="C")
        self.assertEqual(summary["item_count"], 80)
        self.assertEqual(summary["format_pass_count"], 78)
        self.assertEqual(summary["format_pass_rate"], 0.975)
        self.assertEqual(summary["format_failure_count"], 2)
        self.assertEqual(summary["failure_reason_counts"]["emotion_mismatch"], 1)
        self.assertEqual(summary["failure_reason_counts"]["invalid_emotion"], 1)
        serialized = json.dumps(summary)
        self.assertNotIn("E001", serialized)
        self.assertNotIn("visible green tree", serialized)
        self.assertNotIn("response", summary)

    def test_rejects_unaccepted_record_sets_and_status(self):
        mutations = {
            "missing": lambda records: records.pop(),
            "duplicate": lambda records: records[0].update(blind_id="E002"),
            "extra": lambda records: records[0].update(blind_id="E081"),
            "condition": lambda records: records[0].update(condition="B1"),
            "status": lambda records: records[0].update(status="failed"),
            "empty_response": lambda records: records[0].update(response="  "),
            "image_hash": lambda records: records[0].update(image_sha256="f" * 64),
            "response_hash": lambda records: records[0].update(response_sha256="f" * 64),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                records = _records()
                mutate(records)
                with self.assertRaises(ValueError):
                    audit_output_contract(records, _manifest(), condition="C")

    def test_rejects_labels_and_duplicate_manifest_ids(self):
        manifest = _manifest()
        manifest["records"][0]["emotion"] = "awe"
        with self.assertRaisesRegex(ValueError, "label-free"):
            audit_output_contract(_records(), manifest, condition="C")
        manifest = _manifest()
        manifest["records"][0]["blind_id"] = "E002"
        with self.assertRaisesRegex(ValueError, "unique"):
            audit_output_contract(_records(), manifest, condition="C")

    def test_cli_writes_only_aggregate_json_using_frozen_manifest(self):
        repo_root = Path(__file__).resolve().parents[2]
        manifest_path = (
            repo_root / "results/phase2/eval80/frozen_inputs/inference_manifest.json"
        )
        frozen_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        records = _records("B1")
        for record, manifest_record in zip(records, frozen_manifest["records"]):
            record["blind_id"] = manifest_record["blind_id"]
            record["image_sha256"] = manifest_record["image_sha256"]
        _set_response(records[0], VALID_RESPONSE.replace("Final emotion: awe", "Final emotion: fear"))
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            records_path = root / "synthetic_records.jsonl"
            records_path.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            output_dir = root / "aggregate"
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                result = main(
                    [
                        "audit-format", "--condition", "B1",
                        "--records", str(records_path),
                        "--inference-manifest", str(manifest_path),
                        "--output-dir", str(output_dir),
                    ]
                )
            self.assertEqual(result, 0)
            output_files = list(output_dir.iterdir())
            self.assertEqual([path.name for path in output_files], ["B1_format_audit_summary.json"])
            output_text = output_files[0].read_text(encoding="utf-8")
            summary = json.loads(output_text)
            self.assertEqual(summary["format_pass_count"], 79)
            self.assertEqual(summary["failure_reason_counts"], {"emotion_mismatch": 1})
            self.assertNotIn("E001", output_text)
            self.assertNotIn("visible green tree", output_text)
            self.assertNotIn("awe", output_text)
            self.assertEqual(json.loads(stdout.getvalue()), summary)
            with self.assertRaises(FileExistsError):
                main(
                    [
                        "audit-format", "--condition", "B1",
                        "--records", str(records_path),
                        "--inference-manifest", str(manifest_path),
                        "--output-dir", str(output_dir),
                    ]
                )

    def test_cli_rejects_manifest_that_differs_from_freeze(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(_manifest()), encoding="utf-8")
            records_path = root / "records.jsonl"
            records_path.write_text(
                "".join(json.dumps(record) + "\n" for record in _records("C")),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "frozen Eval80-v1"):
                main(
                    [
                        "audit-format", "--condition", "C",
                        "--records", str(records_path),
                        "--inference-manifest", str(manifest_path),
                        "--output-dir", str(root / "aggregate"),
                    ]
                )


if __name__ == "__main__":
    unittest.main()
