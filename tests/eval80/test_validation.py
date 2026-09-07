import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from src.eval80.core import sha256_file
from src.eval80.validation import validate_run


class Eval80OutputValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.run_dir = Path(self.temp.name)
        self.manifest = {
            "records": [
                {
                    "blind_id": f"E{index:03d}",
                    "image_sha256": hashlib.sha256(f"image-{index}".encode()).hexdigest(),
                }
                for index in range(1, 81)
            ]
        }
        self.config = {
            "seed": 1234,
            "precision": "bf16",
            "batch_size": 1,
            "conditions": {
                "A": {
                    "prompt_version": "venus-official-stage1",
                    "prompt": "official prompt",
                }
            },
        }
        records = []
        for record in self.manifest["records"]:
            response = f"Analysis for {record['blind_id']}"
            records.append(
                {
                    "stage": "stage1",
                    "condition": "A",
                    "blind_id": record["blind_id"],
                    "image_sha256": record["image_sha256"],
                    "status": "success",
                    "response": response,
                    "response_sha256": hashlib.sha256(response.encode()).hexdigest(),
                }
            )
        records_path = self.run_dir / "records.jsonl"
        records_path.write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )
        (self.run_dir / "run_config.json").write_text(
            json.dumps(
                {
                    "condition": "A",
                    "prompt_version": "venus-official-stage1",
                    "prompt": "official prompt",
                    "seed": 1234,
                    "precision": "bf16",
                    "batch_size": 1,
                    "manifest_size": 80,
                }
            ),
            encoding="utf-8",
        )
        (self.run_dir / "environment.json").write_text("{}", encoding="utf-8")
        (self.run_dir / "summary.json").write_text(
            json.dumps(
                {
                    "completed": True,
                    "record_count": 80,
                    "expected_count": 80,
                    "empty_response_count": 0,
                    "records_sha256": sha256_file(records_path),
                }
            ),
            encoding="utf-8",
        )

    def tearDown(self):
        self.temp.cleanup()

    def test_accepts_complete_run(self):
        report = validate_run(self.run_dir, self.manifest, self.config, "A")
        self.assertTrue(report["valid"], report["errors"])

    def test_rejects_ground_truth_leak(self):
        records_path = self.run_dir / "records.jsonl"
        records = [json.loads(line) for line in records_path.read_text().splitlines()]
        records[0]["emotion_label"] = "awe"
        records_path.write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )
        report = validate_run(self.run_dir, self.manifest, self.config, "A")
        self.assertFalse(report["valid"])
        self.assertTrue(any("ground-truth fields leaked" in error for error in report["errors"]))


if __name__ == "__main__":
    unittest.main()
