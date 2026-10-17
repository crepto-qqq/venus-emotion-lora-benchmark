"""Synthetic locked-run checks for the private human review page."""

from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from src.eval80.cli import main as eval80_main
from src.eval80.preference_review_pack import build_review_pack
from tests.eval80 import test_preference_benchmark as preference_test


class PreferenceReviewPackTests(unittest.TestCase):
    def test_requires_lock_and_validates_full_run_and_source_hashes(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture_helper = preference_test.PreferenceBenchmarkTests()
            fixture = fixture_helper._fixture(root)
            package = fixture_helper._prepare(root, fixture)
            report = root / "aggregate.json"
            output = root / "tmp" / "review.md"
            arguments = {
                "package_dir": package,
                "report_path": report,
                "b1_run_dir": fixture["b1_dir"],
                "c_run_dir": fixture["c_dir"],
                "output": output,
            }
            with patch("src.eval80.preference_review_pack._ROOT", root):
                with self.assertRaisesRegex(FileNotFoundError, "score lock"):
                    build_review_pack(**arguments)
            self.assertFalse(output.exists())

            model_id = json.loads(fixture["judge_config"].read_text(
                encoding="utf-8"
            ))["model_id"]
            position = 0

            def fake_post(_endpoint, _payload, _key, _timeout):
                nonlocal position
                position += 1
                return {
                    "id": f"synthetic-review-{position}", "model": model_id,
                    "choices": [{"finish_reason": "stop", "message": {
                        "content": json.dumps({
                            "preferred": "tie", "better_grounding": "tie",
                            "better_actionability": "tie", "better_target_fit": "tie",
                            "rationale": "Synthetic judge found no meaningful difference.",
                        })
                    }}],
                }

            def fake_probe(_endpoint, model, digest, _timeout, *, phase):
                return {
                    "phase": phase, "observed_at_utc": "2026-10-17T00:00:00+00:00",
                    "model_id": model, "digest_sha256": digest,
                }

            with patch("src.eval80.preference_judge_runner._post_chat", fake_post), patch(
                "src.eval80.preference_judge_runner._probe_ollama_digest", fake_probe
            ), redirect_stdout(io.StringIO()):
                self.assertEqual(eval80_main([
                    "judge-preference", "--package-dir", str(package),
                    "--judge-config", str(fixture["judge_config"]),
                ]), 0)
            with redirect_stdout(io.StringIO()):
                self.assertEqual(eval80_main([
                    "report-preference", "--package-dir", str(package),
                    "--output", str(report),
                ]), 0)
            with patch("src.eval80.preference_review_pack._ROOT", root):
                self.assertEqual(build_review_pack(**arguments), output)
                text = output.read_text(encoding="utf-8")
                self.assertIn("## E002", text)
                self.assertIn("## E079", text)
                self.assertNotIn("## E001", text)
                self.assertNotIn("## E080", text)
                self.assertIn("两次交换顺序", text)
                self.assertIn("Synthetic judge found no meaningful difference.", text)
                self.assertIn("![E002 原图](<", text)
                self.assertIn("待人工复核", text)
                self.assertEqual(text.count("### B1 原始回答"), 78)
                self.assertEqual(text.count("### C 原始回答"), 78)
                self.assertEqual(text.count("Synthetic judge found no meaningful difference."), 156)

                corrupted_report = root / "corrupted_aggregate.json"
                data = json.loads(report.read_text(encoding="utf-8"))
                data["source_records_sha256"]["C"] = "0" * 64
                corrupted_report.write_text(json.dumps(data), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "aggregate report differs"):
                    build_review_pack(**{**arguments, "report_path": corrupted_report})

                source = fixture["b1_dir"] / "records.jsonl"
                source.write_bytes(source.read_bytes() + b"\n")
                with self.assertRaisesRegex(ValueError, "Source records hash"):
                    build_review_pack(**arguments)
                with self.assertRaisesRegex(ValueError, "ignored tmp"):
                    build_review_pack(**{**arguments, "output": root / "tracked.md"})


if __name__ == "__main__":
    unittest.main()
