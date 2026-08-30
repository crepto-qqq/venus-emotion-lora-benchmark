import json
from pathlib import Path
import tempfile
import unittest

from src.phase1.manifest import sha256_file, validate_manifest


QUOTAS = {
    "amusement": 3,
    "awe": 2,
    "contentment": 2,
    "excitement": 3,
    "anger": 2,
    "disgust": 2,
    "fear": 3,
    "sadness": 3,
}
VALENCE = {
    "amusement": "positive",
    "awe": "positive",
    "contentment": "positive",
    "excitement": "positive",
    "anger": "negative",
    "disgust": "negative",
    "fear": "negative",
    "sadness": "negative",
}


class ManifestValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.rows = []
        test_entries = []
        counter = 1
        for emotion, quota in QUOTAS.items():
            for _ in range(quota):
                source_id = f"{emotion}_{counter}"
                image_relpath = f"image/{emotion}/{source_id}.jpg"
                annotation_relpath = f"annotation/{emotion}/{source_id}.json"
                image_path = self.root / image_relpath
                annotation_path = self.root / annotation_relpath
                image_path.parent.mkdir(parents=True, exist_ok=True)
                annotation_path.parent.mkdir(parents=True, exist_ok=True)
                image_path.write_bytes(f"image-{counter}".encode("ascii"))
                annotation_path.write_text("{}", encoding="utf-8")
                test_entries.append([emotion, source_id, image_relpath, annotation_relpath])
                self.rows.append(
                    {
                        "sample_id": f"E{counter:02d}",
                        "source_image_id": source_id,
                        "emotion": emotion,
                        "valence": VALENCE[emotion],
                        "split": "test",
                        "image_relpath": image_relpath,
                        "annotation_relpath": annotation_relpath,
                        "sha256": sha256_file(image_path),
                        "review_status": "approved",
                        "replacement_for": "",
                        "review_notes": "",
                    }
                )
                counter += 1
        (self.root / "test.json").write_text(json.dumps(test_entries), encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def validate(self, rows=None):
        return validate_manifest(
            rows or self.rows,
            self.root,
            QUOTAS,
            VALENCE,
            expected_size=20,
            require_approved=True,
            verify_hashes=True,
        )

    def test_valid_manifest(self):
        self.assertEqual(self.validate(), [])

    def test_rejects_pending_review(self):
        rows = [dict(row) for row in self.rows]
        rows[0]["review_status"] = "pending"
        self.assertTrue(any("review_status" in error for error in self.validate(rows)))

    def test_rejects_duplicate_and_hash_mismatch(self):
        rows = [dict(row) for row in self.rows]
        rows[1]["sample_id"] = rows[0]["sample_id"]
        rows[2]["sha256"] = "0" * 64
        errors = self.validate(rows)
        self.assertTrue(any("Duplicate sample_id" in error for error in errors))
        self.assertTrue(any("SHA-256 mismatch" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
