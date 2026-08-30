import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.phase1.manifest import read_manifest, validate_manifest, write_manifest
from src.phase1.select_emoset20 import main as select_main


class SelectionIntegrationTests(unittest.TestCase):
    def test_selection_produces_valid_balanced_manifest_after_review(self):
        project_root = Path(__file__).resolve().parents[2]
        config_path = project_root / "configs/phase1/baseline.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        quotas = config["dataset"]["emotion_quotas"]
        valence = config["dataset"]["valence"]

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "emoset"
            entries = []
            for emotion, quota in quotas.items():
                for index in range(quota + 3):
                    source_id = f"{emotion}_{index:03d}"
                    image_relpath = f"image/{emotion}/{source_id}.jpg"
                    annotation_relpath = f"annotation/{emotion}/{source_id}.json"
                    image_path = root / image_relpath
                    annotation_path = root / annotation_relpath
                    image_path.parent.mkdir(parents=True, exist_ok=True)
                    annotation_path.parent.mkdir(parents=True, exist_ok=True)
                    image_path.write_bytes(source_id.encode("ascii"))
                    annotation_path.write_text("{}", encoding="utf-8")
                    entries.append([emotion, source_id, image_relpath, annotation_relpath])
            (root / "test.json").write_text(json.dumps(entries), encoding="utf-8")

            manifest_path = Path(temp_dir) / "manifest.csv"
            reserve_path = Path(temp_dir) / "reserve.csv"
            argv = [
                "select_emoset20",
                "--emoset-root",
                str(root),
                "--config",
                str(config_path),
                "--output",
                str(manifest_path),
                "--reserve-output",
                str(reserve_path),
            ]
            with patch("sys.argv", argv), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(select_main(), 0)

            selected = read_manifest(manifest_path)
            reserves = read_manifest(reserve_path)
            self.assertEqual(len(selected), 20)
            self.assertEqual(len(reserves), 24)
            self.assertTrue(all(row["review_status"] == "pending" for row in selected))

            for row in selected:
                row["review_status"] = "approved"
            write_manifest(manifest_path, selected, overwrite=True)
            self.assertEqual(
                validate_manifest(selected, root, quotas, valence, expected_size=20),
                [],
            )


if __name__ == "__main__":
    unittest.main()
