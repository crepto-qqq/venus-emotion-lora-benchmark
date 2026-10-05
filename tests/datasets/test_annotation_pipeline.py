import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from src.datasets.annotation_pipeline import (
    EVAL20_IMAGE_IDS,
    TARGET_PATTERN,
    annotation_schema,
    build_request_payload,
    generate_records,
    read_jsonl,
    render_target_response,
    select_records,
    select_replacement_records,
    validate_annotation,
    validate_output_record,
    validate_output_root,
)


class AnnotationPipelineTests(unittest.TestCase):
    def test_replacements_are_next_candidates_and_inherit_split(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            archive_path = root / "emoset.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                for index in list(range(10)) + list(range(5000, 5004)):
                    image_id = f"awe_{index:05d}"
                    archive.writestr(f"image/awe/{image_id}.jpg", image_id)
                    archive.writestr(
                        f"annotation/awe/{image_id}.json",
                        json.dumps({"emotion": "awe", "image_id": image_id}),
                    )

            def manifest_record(index, split):
                image_id = f"awe_{index:05d}"
                image_bytes = image_id.encode()
                annotation_bytes = json.dumps(
                    {"emotion": "awe", "image_id": image_id}
                ).encode()
                import hashlib
                return {
                    "source_image_id": image_id,
                    "emotion": "awe",
                    "image_relpath": f"image/awe/{image_id}.jpg",
                    "annotation_relpath": f"annotation/awe/{image_id}.json",
                    "split": split,
                    "selection_rank": index - 4999,
                    "selection_seed": 3888,
                    "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
                    "annotation_sha256": hashlib.sha256(annotation_bytes).hexdigest(),
                }

            manifest_path = root / "manifest.jsonl"
            write_rows = [manifest_record(5000, "validation"), manifest_record(5001, "train")]
            manifest_path.write_text(
                "".join(json.dumps(row) + "\n" for row in write_rows), encoding="utf-8"
            )
            output_root = root / "output"
            output_root.mkdir()
            rejected = {
                "source_image_id": "awe_05001",
                "emotion": "awe",
                "split": "train",
                "provenance": {"review_status": "rejected"},
            }
            (output_root / "reviews.jsonl").write_text(
                json.dumps(rejected) + "\n", encoding="utf-8"
            )

            replacements = select_replacement_records(
                archive_path,
                manifest_path,
                output_root,
                start_image_number=5000,
            )

            self.assertEqual(len(replacements), 1)
            self.assertEqual(replacements[0]["source_image_id"], "awe_05002")
            self.assertEqual(replacements[0]["split"], "train")
            self.assertEqual(replacements[0]["replaces_source_image_id"], "awe_05001")

    def test_selection_is_balanced_deterministic_and_excludes_eval20(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            archive_path = Path(temp_dir) / "emoset.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                for index in range(11):
                    image_id = f"awe_{index:05d}"
                    archive.writestr(f"image/awe/{image_id}.jpg", image_id)
                    archive.writestr(
                        f"annotation/awe/{image_id}.json",
                        json.dumps({"emotion": "awe", "image_id": image_id}),
                    )
                for index in range(3000, 3066):
                    image_id = f"awe_{index:05d}"
                    archive.writestr(f"image/awe/{image_id}.jpg", image_id)
                    archive.writestr(
                        f"annotation/awe/{image_id}.json",
                        json.dumps({"emotion": "awe", "image_id": image_id}),
                    )

            selection = {
                "start_image_number": 3000,
                "images_per_class": 60,
                "training_per_class": 50,
                "validation_per_class": 10,
                "split_seed": 3888,
            }
            first = select_records(archive_path, ["awe"], selection)
            second = select_records(archive_path, ["awe"], selection)
            self.assertEqual(first, second)
            self.assertEqual(len(first), 60)
            self.assertEqual(sum(row["split"] == "train" for row in first), 50)
            self.assertEqual(sum(row["split"] == "validation" for row in first), 10)
            selected_ids = {row["source_image_id"] for row in first}
            self.assertNotIn("awe_03021", selected_ids)
            self.assertTrue(selected_ids.isdisjoint(EVAL20_IMAGE_IDS))

    def test_structured_annotation_renders_exact_sam_based_target(self):
        draft = {
            "emotion": "sadness",
            "visual_evidence": ["  Downcast eyes  ", "Muted blue lighting"],
            "aesthetic_relationship": "The subdued palette reinforces the quiet mood.",
            "guidance": ["Reduce background contrast.", "Keep the subject off-centre."],
            "final_emotion": "sadness",
            "label_supported": True,
            "support_notes": "The assigned label is visibly supported.",
        }
        annotation = validate_annotation(draft, "sadness")
        target = render_target_response(annotation)
        self.assertIsNotNone(TARGET_PATTERN.fullmatch(target))
        record = {
            "source_image_id": "sadness_05000",
            "emotion": "sadness",
            "image_relpath": "image/sadness/sadness_05000.jpg",
            "instruction": "prompt",
            "target_response": target,
            "split": "train",
            "provenance": {
                "generator_provider": "openai",
                "generator_model": "gpt-5.4-2026-03-05",
                "prompt_version": "sam-framework-api-v2",
                "temperature": 0.2,
                "seed": "unsupported",
                "review_status": "draft",
            },
        }
        self.assertEqual(validate_output_record(record), [])

    def test_request_locks_label_and_uses_strict_json_schema(self):
        config = {
            "model": "gpt-5.4-2026-03-05",
            "store": False,
            "reasoning_effort": "none",
            "temperature": 0.2,
            "max_output_tokens": 900,
            "protocol_version": "test",
            "image_detail": "high",
        }
        payload = build_request_payload(
            config, "instruction", "contentment", "data:image/jpeg;base64,AA==", "contentment_05000"
        )
        schema = payload["text"]["format"]
        self.assertTrue(schema["strict"])
        self.assertEqual(
            schema["schema"]["properties"]["emotion"]["enum"], ["contentment"]
        )
        self.assertFalse(payload["store"])
        self.assertEqual(payload["reasoning"], {"effort": "none"})
        self.assertEqual(payload["input"][0]["content"][1]["detail"], "high")
        self.assertEqual(annotation_schema("contentment")["additionalProperties"], False)

    def test_generation_writes_sam_jsonl_and_resumes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            archive_path = root / "emoset.zip"
            image_id = "sadness_05001"
            image_entry = f"image/sadness/{image_id}.jpg"
            annotation_entry = f"annotation/sadness/{image_id}.json"
            image_bytes = b"test-image"
            annotation_bytes = json.dumps(
                {"emotion": "sadness", "image_id": image_id}
            ).encode("utf-8")
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr(image_entry, image_bytes)
                archive.writestr(annotation_entry, annotation_bytes)

            import hashlib

            manifest_path = root / "manifest.jsonl"
            manifest_path.write_text(
                json.dumps(
                    {
                        "source_image_id": image_id,
                        "emotion": "sadness",
                        "image_relpath": image_entry,
                        "annotation_relpath": annotation_entry,
                        "split": "train",
                        "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
                        "annotation_sha256": hashlib.sha256(annotation_bytes).hexdigest(),
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            prompt_path = root / "prompt.txt"
            prompt_path.write_text("Assigned label: {emotion}", encoding="utf-8")
            output_root = root / "output"
            config = {
                "api_key_env": "OPENAI_API_KEY",
                "prompt_path": str(prompt_path),
                "model": "gpt-5.4-2026-03-05",
                "endpoint": "https://api.openai.com/v1/responses",
                "store": False,
                "reasoning_effort": "none",
                "temperature": 0.2,
                "max_output_tokens": 900,
                "protocol_version": "test-v1",
                "prompt_version": "test-prompt",
                "image_detail": "high",
                "timeout_seconds": 10,
                "max_retries": 1,
            }
            model_annotation = {
                "emotion": "sadness",
                "visual_evidence": ["Downcast eyes", "Muted lighting"],
                "aesthetic_relationship": "The subdued lighting reinforces the mood.",
                "guidance": ["Reduce background contrast.", "Retain the muted palette."],
                "final_emotion": "sadness",
                "label_supported": True,
                "support_notes": "Visible cues support the assigned label.",
            }
            fake_response = {
                "id": "resp_test",
                "model": "gpt-5.4-2026-03-05",
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {"type": "output_text", "text": json.dumps(model_annotation)}
                        ],
                    }
                ],
            }

            with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}), patch(
                "src.datasets.annotation_pipeline.post_response",
                return_value=fake_response,
            ):
                first = generate_records(
                    archive_path,
                    manifest_path,
                    output_root,
                    config,
                    smoke_per_emotion=1,
                )
                second = generate_records(
                    archive_path,
                    manifest_path,
                    output_root,
                    config,
                    smoke_per_emotion=1,
                )

            self.assertEqual(first, {"generated": 1})
            self.assertEqual(second, {"skipped": 1})
            output_path = output_root / "sadness" / "train_sadness.jsonl"
            records = read_jsonl(output_path)
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["emotion"], "sadness")
            self.assertEqual(records[0]["provenance"]["review_status"], "draft")
            self.assertEqual(validate_output_record(records[0]), [])
            incomplete_report = validate_output_root(
                output_root,
                expected_counts={"sadness:train": 2},
                require_accepted=True,
            )
            self.assertFalse(incomplete_report["valid"])
            self.assertTrue(
                any("expected 2 records" in error for error in incomplete_report["errors"])
            )
            self.assertTrue(
                any("review_status must be accepted" in error for error in incomplete_report["errors"])
            )


if __name__ == "__main__":
    unittest.main()
