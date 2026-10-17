"""Synthetic checks for the score-free, balanced anonymous review page."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from src.eval80.core import sha256_file
from src.eval80.unscored_review_pack import build_unscored_review_pack
from tests.eval80 import test_quality_benchmark


class UnscoredReviewPackTests(unittest.TestCase):
    def test_frozen_sources_balanced_key_and_no_judge_results(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = test_quality_benchmark.GuidanceJudgeTests()._fixture(root)
            output = root / "tmp" / "review.md"
            key_output = root / "tmp" / "condition_key.json"
            arguments = {
                "b1_run_dir": fixture["b1_dir"],
                "c_run_dir": fixture["c_dir"],
                "selection_manifest": fixture["selection"],
                "image_dir": fixture["image_dir"],
                "inference_manifest": fixture["manifest"],
                "freeze_record": fixture["freeze"],
                "display_exclusion": fixture["exclusion"],
                "config": fixture["config"],
                "handbook": fixture["handbook"],
                "output": output,
                "key_output": key_output,
                "seed": 17,
            }
            with patch("src.eval80.unscored_review_pack._ROOT", root):
                self.assertEqual(build_unscored_review_pack(**arguments),
                                 (output, key_output))
                page = output.read_text(encoding="utf-8")
                key = json.loads(key_output.read_text(encoding="utf-8"))
                self.assertEqual(page.count("### 回答 A"), 78)
                self.assertEqual(page.count("### 回答 B"), 78)
                self.assertEqual(page.count("### 人工笔记"), 78)
                self.assertIn("## E002", page)
                self.assertIn("## E079", page)
                self.assertNotIn("## E001", page)
                self.assertNotIn("## E080", page)
                self.assertNotIn("自动判定结果", page)
                self.assertNotIn("condition_key.json", page)
                self.assertEqual(key["review_page_sha256"], sha256_file(output))
                self.assertEqual(key["balanced_A_counts"], {"B1": 39, "C": 39})
                self.assertEqual(len(key["items"]), 78)
                self.assertEqual(sum(item["A"] == "B1" for item in key["items"].values()), 39)
                self.assertEqual(sum(item["B"] == "B1" for item in key["items"].values()), 39)
                for blind_id, item in key["items"].items():
                    self.assertEqual({item["A"], item["B"]}, {"B1", "C"})
                    self.assertEqual(item["image_sha256"], sha256_file(
                        fixture["image_dir"] / f"{blind_id}.jpg"))

                with self.assertRaises(FileExistsError):
                    build_unscored_review_pack(**arguments)
                with self.assertRaisesRegex(ValueError, "ignored tmp"):
                    build_unscored_review_pack(**{**arguments,
                        "output": root / "tracked.md", "key_output": root / "tmp" / "new_key.json"})

    def test_tampered_frozen_image_aborts_without_writing(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = test_quality_benchmark.GuidanceJudgeTests()._fixture(root)
            (fixture["image_dir"] / "E002.jpg").write_bytes(b"changed")
            output = root / "tmp" / "review.md"
            key_output = root / "tmp" / "key.json"
            with patch("src.eval80.unscored_review_pack._ROOT", root):
                with self.assertRaisesRegex(ValueError, "image bytes differ"):
                    build_unscored_review_pack(
                        b1_run_dir=fixture["b1_dir"], c_run_dir=fixture["c_dir"],
                        selection_manifest=fixture["selection"], image_dir=fixture["image_dir"],
                        output=output, key_output=key_output,
                        inference_manifest=fixture["manifest"], freeze_record=fixture["freeze"],
                        display_exclusion=fixture["exclusion"], config=fixture["config"],
                        handbook=fixture["handbook"], seed=17,
                    )
            self.assertFalse(output.exists())
            self.assertFalse(key_output.exists())


if __name__ == "__main__":
    unittest.main()
