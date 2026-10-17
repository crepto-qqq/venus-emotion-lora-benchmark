"""Synthetic, private-data-free tests for the Eval80 quality judge workflow."""

from __future__ import annotations

from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import ProxyHandler

from src.eval80.cli import main
from src.eval80.core import EMOTION_CLASSES, sha256_file, sha256_text_file
from src.eval80.inference import (
    FORMAL_C_ADAPTER_FILES, FORMAL_C_CHECKPOINT, FORMAL_C_EVIDENCE_SHA256,
)
from src.eval80.quality_benchmark import _SCORE_KEYS, aggregate_guidance
from src.eval80.quality_judge_runner import (
    _NoRedirectHandler, _endpoint_is_allowed, _extract_judgement, _make_payload,
    _post_chat,
)
from src.eval80 import quality_judge_runner


def _dump(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _dump_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class GuidanceJudgeTests(unittest.TestCase):
    def _fixture(self, root: Path) -> dict[str, Path]:
        image_dir = root / "images"
        image_dir.mkdir()
        manifest_records = []
        selection_records = []
        for index in range(1, 81):
            blind_id = f"E{index:03d}"
            filename = f"{blind_id}.jpg"
            path = image_dir / filename
            path.write_bytes(f"synthetic image {index}".encode())
            image_hash = sha256_file(path)
            manifest_records.append({
                "blind_id": blind_id, "image_filename": filename,
                "image_sha256": image_hash, "width": 1, "height": 1,
            })
            selection_records.append({
                "blind_id": blind_id,
                "image_sha256": image_hash,
                "local_image_filename": filename,
                "emotion": EMOTION_CLASSES[(index - 1) // 10],
            })
        manifest = root / "manifest.json"
        _dump(manifest, {
            "schema_version": 1, "evaluation_set": "Eval80-v1",
            "image_count": 80, "contains_ground_truth_labels": False,
            "records": manifest_records,
        })
        selection = root / "selection.json"
        _dump(selection, {"records": selection_records})
        config = root / "config.json"
        repo_root = Path(__file__).resolve().parents[2]
        source_lock_path = repo_root / "phase3/configs/source-lock.json"
        source_lock = json.loads(source_lock_path.read_text(encoding="utf-8"))["model"]
        snapshot_path = repo_root / source_lock["snapshot_manifest"]
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        model_source = {
            "repo_id": source_lock["repository"],
            "revision": source_lock["revision"],
            "weight_format": source_lock["weight_format"],
        }
        _dump(config, {
            "seed": 1234, "precision": "bf16", "batch_size": 1,
            "generation": {"max_new_tokens": 512},
            "model": model_source,
            "conditions": {"B1": {"prompt": "Frozen prompt", "prompt_version": "p1"}},
        })
        handbook = root / "handbook.md"
        handbook.write_text("Frozen rubric\n", encoding="utf-8")
        freeze = root / "freeze.json"
        _dump(freeze, {
            "status": "approved_and_frozen", "team_approval_confirmed": True,
            "file_sha256": {
                "inference_manifest": sha256_file(manifest),
                "selection_manifest_private": sha256_file(selection),
                "config": sha256_text_file(config),
                "scoring_handbook": sha256_text_file(handbook),
            },
        })
        exclusion = root / "exclusion.json"
        official_exclusion = Path(__file__).resolve().parents[2] / "Docs/evaluation/EVAL80_DISPLAY_EXCLUSION.json"
        exclusion.write_bytes(official_exclusion.read_bytes())
        judge_config = root / "judge_config.json"
        _dump(judge_config, {
            "schema_version": 1, "model_id": "local-vision-v1.0",
            "provider": "openai_compatible_chat",
            "endpoint": "http://127.0.0.1:8000/v1/chat/completions",
            "prompt": "Score only visible facts and the five frozen dimensions.",
            "image_detail": "high", "decoding": {"temperature": 0, "max_tokens": 800},
            "retry_policy": {"max_attempts": 2, "timeout_seconds": 15, "backoff_seconds": 0},
        })
        snapshot_receipt = {
            "source_lock_sha256": sha256_file(source_lock_path),
            "snapshot_manifest_sha256": sha256_file(snapshot_path),
            "verified_file_count": snapshot["file_count"],
            "verified_total_bytes": snapshot["total_bytes"],
        }
        run_dirs = {}
        for condition in ("B1", "C"):
            run_dir = root / condition
            run_dir.mkdir()
            records = []
            for item in manifest_records:
                response = f"Emotion: awe\nGuidance: crop the sky {condition}"
                records.append({
                    "blind_id": item["blind_id"], "condition": condition,
                    "status": "success", "image_sha256": item["image_sha256"],
                    "response": response,
                    "response_sha256": hashlib.sha256(response.encode()).hexdigest(),
                })
            records_path = run_dir / "records.jsonl"
            _dump_jsonl(records_path, records)
            run_config = {
                "condition": condition, "manifest_size": 80,
                "prompt": "Frozen prompt", "prompt_version": "p1", "seed": 1234,
                "precision": "bf16", "batch_size": 1,
                "generation_expected": {"max_new_tokens": 512},
                "generation_actual": {"max_new_tokens": 512},
                "model_source": model_source,
                "file_sha256": {
                    "inference_manifest": sha256_file(manifest),
                    "config": sha256_text_file(config),
                    "scoring_handbook": sha256_text_file(handbook),
                    "freeze_record": sha256_file(freeze),
                },
            }
            if condition == "C":
                run_config["adapter_identity"] = {
                    "evidence_sha256": FORMAL_C_EVIDENCE_SHA256,
                    "selected_checkpoint": FORMAL_C_CHECKPOINT,
                    "adapter_files": FORMAL_C_ADAPTER_FILES,
                    "base_revision": model_source["revision"],
                    "acceptance_status": "not_verified",
                }
                run_config["base_snapshot_verification"] = snapshot_receipt
            _dump(run_dir / "run_config.json", run_config)
            _dump(run_dir / "summary.json", {
                "completed": True, "condition": condition, "record_count": 80,
                "records_sha256": sha256_file(records_path),
            })
            _dump(run_dir / "validation.json", {
                "valid": True, "condition": condition, "record_count": 80,
                "records_sha256": sha256_file(records_path), "errors": [],
            })
            run_dirs[condition] = run_dir
        return {
            "image_dir": image_dir, "manifest": manifest, "selection": selection,
            "config": config, "handbook": handbook, "freeze": freeze,
            "exclusion": exclusion, "judge_config": judge_config,
            "b1_dir": run_dirs["B1"], "c_dir": run_dirs["C"],
        }

    def _prepare(self, root: Path, fixture: dict[str, Path]) -> Path:
        package = root / "private_package"
        args = ["prepare-guidance-judge"]
        for key, flag in (
            ("b1_dir", "--b1-run-dir"), ("c_dir", "--c-run-dir"),
            ("selection", "--selection-manifest"), ("image_dir", "--image-dir"),
            ("judge_config", "--judge-config"), ("manifest", "--inference-manifest"),
            ("freeze", "--freeze-record"), ("exclusion", "--display-exclusion"),
            ("config", "--config"), ("handbook", "--handbook"),
        ):
            args.extend((flag, str(fixture[key])))
        args.extend(("--output-dir", str(package)))
        with redirect_stdout(io.StringIO()):
            self.assertEqual(main(args), 0)
        return package

    def _judgements(self, package: Path) -> list[dict]:
        metadata = json.loads((package / "judge_package.json").read_text(encoding="utf-8"))
        packets = [json.loads(line) for line in (
            package / "judge_items_private.jsonl"
        ).read_text(encoding="utf-8").splitlines()]
        mapping = json.loads((package / "condition_mapping_private.json").read_text(encoding="utf-8"))["items"]
        rows = []
        for packet in packets:
            anonymous_id = packet["anonymous_item_id"]
            condition = mapping[anonymous_id]["condition"]
            scores = {key: (2 if condition == "C" else 1) for key in _SCORE_KEYS}
            canonical = json.dumps(packet, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
            rows.append({
                "anonymous_item_id": anonymous_id,
                "packet_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
                "judge_config_sha256": metadata["judge_config_sha256"],
                "judge_model_id": metadata["judge_model_id"],
                "scores": scores, "overall_rationale": "Synthetic test answer.",
                "zero_score_reasons": {}, "hallucination": False,
                "emotion_conflict": False, "recommendation_present": True,
                "response_metadata": {"request_id": anonymous_id},
            })
        return rows

    def test_complete_packet_lock_report_is_private_and_paired(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._fixture(root)
            package = self._prepare(root, fixture)
            metadata = json.loads((package / "judge_package.json").read_text(encoding="utf-8"))
            packets = (package / "judge_items_private.jsonl").read_text(encoding="utf-8")
            self.assertEqual(metadata["item_count"], 156)
            self.assertEqual(metadata["balanced_first_condition_counts_private"], {"B1": 39, "C": 39})
            self.assertEqual(metadata["judge_config_snapshot_sha256"], sha256_file(
                package / "judge_config_private.json"))
            self.assertEqual(metadata["judge_handbook_snapshot_sha256"], sha256_file(
                package / "judge_handbook_private.md"))
            self.assertNotIn('"condition"', packets)
            self.assertNotIn('"blind_id"', packets)
            self.assertNotIn("E001.jpg", packets)
            self.assertNotIn("E080.jpg", packets)
            scores = root / "scores.jsonl"
            _dump_jsonl(scores, self._judgements(package))
            with self.assertRaisesRegex(ValueError, "Script judge provenance needs"):
                main(["lock-guidance-judge", "--package-dir", str(package),
                      "--judgements", str(scores)])
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(["lock-guidance-judge", "--package-dir", str(package), "--judgements", str(scores), "--manual-import"]), 0)
                report = root / "aggregate.json"
                self.assertEqual(main(["report-guidance", "--package-dir", str(package), "--judgements", str(scores), "--output", str(report)]), 0)
            summary = json.loads(report.read_text(encoding="utf-8"))
            self.assertEqual(summary["dimensions"]["total"]["paired_mean_delta_c_minus_b1"], 5)
            self.assertEqual(summary["total_wins_ties_losses_c_vs_b1"], {"wins": 78, "ties": 0, "losses": 0})
            self.assertEqual(summary["total_delta_descriptive_bootstrap_95_interval"], [5, 5])
            self.assertNotIn("E002", report.read_text(encoding="utf-8"))
            cases_path = package / "paired_cases_private.json"
            self.assertTrue(cases_path.is_file())
            original_cases = cases_path.read_bytes()
            original_report = report.read_bytes()
            report.unlink()
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(["report-guidance", "--package-dir", str(package),
                                       "--judgements", str(scores), "--output", str(report)]), 0)
            self.assertEqual(report.read_bytes(), original_report)
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(["report-guidance", "--package-dir", str(package),
                                       "--judgements", str(scores), "--output", str(report)]), 0)
            self.assertEqual(cases_path.read_bytes(), original_cases)
            self.assertEqual(report.read_bytes(), original_report)
            cases_path.write_text("changed private cases\n", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "Existing report differs"):
                main(["report-guidance", "--package-dir", str(package),
                      "--judgements", str(scores), "--output", str(report)])
            cases_path.write_bytes(original_cases)
            report.write_text("changed aggregate\n", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "Existing report differs"):
                main(["report-guidance", "--package-dir", str(package),
                      "--judgements", str(scores), "--output", str(report)])
            report.write_bytes(original_report)
            benchmark_args = ["report-benchmark"]
            for condition, label, correct in (("B1", "b1", 55), ("C", "c", 75)):
                source_hash = summary["source_records_sha256"][condition]
                fmt_path = root / f"{label}_format.json"
                _dump(fmt_path, {
                    "condition": condition, "item_count": 80,
                    "format_pass_count": 79, "format_failure_count": 1,
                    "failure_reason_counts": {"missing_guidance": 1},
                    "records_sha256": source_hash,
                    "inference_manifest_sha256": summary["inference_manifest_sha256"],
                })
                benchmark_args.extend((f"--{label}-format", str(fmt_path)))
                recognition_path = root / f"{label}_recognition.json"
                _dump(recognition_path, {
                    "condition": condition, "item_count": 80,
                    "correct_count": correct,
                    "records_sha256": source_hash,
                    "details_sha256": label * 64,
                    "selection_manifest_sha256": summary["selection_manifest_sha256"],
                })
                benchmark_args.extend((f"--{label}-recognition", str(recognition_path)))
            paired_path = root / "paired_recognition.json"
            _dump(paired_path, {
                "comparison": "B1_vs_C_strict_emotion_recognition", "image_count": 80,
                "conditions": {"B1": {"correct_count": 55}, "C": {"correct_count": 75}},
                "net_correct_gain_c_minus_b1": 20,
                "paired_outcomes": {"both_correct": 52, "b1_only_correct": 3,
                                    "c_only_correct": 23, "both_wrong": 2},
                "per_class": {},
                "input_sha256": {"b1_details": "b1" * 64, "c_details": "c" * 64},
            })
            benchmark_args.extend(("--paired-recognition", str(paired_path),
                                   "--guidance", str(report)))
            combined_json = root / "benchmark.json"
            combined_md = root / "benchmark.md"
            benchmark_args.extend(("--output-json", str(combined_json),
                                   "--output-md", str(combined_md)))
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(benchmark_args), 0)
            combined = json.loads(combined_json.read_text(encoding="utf-8"))
            self.assertEqual(combined["recognition"]["net_correct_gain_c_minus_b1"], 20)
            self.assertEqual(combined["status"], "provisional_proxy")
            self.assertEqual(set(combined["formal_claims_blocked_by"]), {
                "adapter_acceptance_unverified", "judge_identity_unverified",
                "judge_provenance_unverified",
            })
            self.assertIn("Guidance proxy (/10, 78 images)", combined_md.read_text(encoding="utf-8"))
            self.assertNotIn("E002", combined_md.read_text(encoding="utf-8"))
            paired_original = paired_path.read_bytes()
            malformed_pair = json.loads(paired_original)
            malformed_pair["paired_outcomes"] = {
                "both_correct": 52, "b1_only_correct": 2,
                "c_only_correct": 22, "both_wrong": 4,
            }
            _dump(paired_path, malformed_pair)
            with self.assertRaisesRegex(ValueError, "Paired recognition report"):
                main(benchmark_args[:-4] + ["--output-json", str(root / "wrong_marginals.json"),
                                            "--output-md", str(root / "wrong_marginals.md")])
            paired_path.write_bytes(paired_original)
            forged_guidance = dict(summary)
            forged_guidance["adapter_acceptance_status"] = "accepted"
            forged_guidance["member3b_acceptance_verified"] = True
            forged_guidance["judge_identity_status"] = "fingerprint_matched"
            forged_guidance["judge_provenance"] = "script_runner_trace_consistent"
            _dump(report, forged_guidance)
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(benchmark_args[:-4] + [
                    "--output-json", str(root / "forged_acceptance.json"),
                    "--output-md", str(root / "forged_acceptance.md"),
                ]), 0)
            forged_combined = json.loads((root / "forged_acceptance.json").read_text(encoding="utf-8"))
            self.assertEqual(forged_combined["status"], "provisional_proxy")
            self.assertEqual(forged_combined["formal_claims_blocked_by"], ["adapter_acceptance_unverified"])
            report.write_bytes(original_report)
            changed = json.loads((root / "c_format.json").read_text(encoding="utf-8"))
            changed["records_sha256"] = "0" * 64
            _dump(root / "c_format.json", changed)
            with self.assertRaisesRegex(ValueError, "different output records"):
                main(benchmark_args[:-4] + ["--output-json", str(root / "bad.json"),
                                            "--output-md", str(root / "bad.md")])

    def test_rejects_stale_output_hash_missing_pair_and_bad_score(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._fixture(root)
            run_config = json.loads((fixture["c_dir"] / "run_config.json").read_text())
            original_identity = dict(run_config["adapter_identity"])
            run_config["adapter_identity"]["evidence_sha256"] = "0" * 64
            _dump(fixture["c_dir"] / "run_config.json", run_config)
            with self.assertRaisesRegex(ValueError, "pinned formal training evidence"):
                self._prepare(root, fixture)
            run_config["adapter_identity"] = original_identity
            original_receipt = dict(run_config["base_snapshot_verification"])
            run_config["base_snapshot_verification"]["verified_file_count"] = 1
            _dump(fixture["c_dir"] / "run_config.json", run_config)
            with self.assertRaisesRegex(ValueError, "base snapshot verification receipt"):
                self._prepare(root, fixture)
            run_config["base_snapshot_verification"] = original_receipt
            run_config["prompt"] = "Changed prompt"
            _dump(fixture["c_dir"] / "run_config.json", run_config)
            with self.assertRaisesRegex(ValueError, "frozen B1"):
                self._prepare(root, fixture)
            run_config["prompt"] = "Frozen prompt"
            _dump(fixture["c_dir"] / "run_config.json", run_config)
            package = self._prepare(root, fixture)
            rows = self._judgements(package)
            rows[0]["scores"]["actionability"] = True
            scores = root / "scores.jsonl"
            _dump_jsonl(scores, rows)
            with self.assertRaisesRegex(ValueError, "integer 0, 1, or 2"):
                main(["lock-guidance-judge", "--package-dir", str(package), "--judgements", str(scores)])
            rows[0]["scores"]["actionability"] = 2 if rows[0]["scores"]["visual_grounding"] == 2 else 1
            rows.pop()
            _dump_jsonl(scores, rows)
            with self.assertRaisesRegex(ValueError, "156 completed"):
                main(["lock-guidance-judge", "--package-dir", str(package), "--judgements", str(scores)])

    def test_aggregate_requires_complete_paired_mapping(self):
        with self.assertRaisesRegex(ValueError, "156 items"):
            aggregate_guidance({}, {}, bootstrap_seed=1)

    def test_script_judge_runs_all_items_with_mock_vision_endpoint(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._fixture(root)
            package = self._prepare(root, fixture)
            packets = [json.loads(line) for line in (
                package / "judge_items_private.jsonl"
            ).read_text(encoding="utf-8").splitlines()]
            config = json.loads(fixture["judge_config"].read_text(encoding="utf-8"))
            sample_payload = _make_payload(packets[0], config, "Test handbook")
            content = sample_payload["messages"][0]["content"]
            self.assertEqual(content[1]["type"], "image_url")
            self.assertTrue(content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,"))
            self.assertIn("Fixed target emotion:", content[0]["text"])
            self.assertIn("Anonymous model answer:", content[0]["text"])
            self.assertNotIn('"blind_id"', json.dumps(sample_payload))
            calls = []

            def fake_post(endpoint, payload, api_key, timeout):
                calls.append((endpoint, payload, api_key, timeout))
                score = {
                    "scores": {key: 1 for key in _SCORE_KEYS},
                    "overall_rationale": "Plausible but generic guidance.",
                    "zero_score_reasons": {key: "" for key in _SCORE_KEYS},
                    "hallucination": False, "emotion_conflict": False,
                    "recommendation_present": True,
                }
                return {
                    "id": f"response-{len(calls)}", "model": "local-vision-v1.0",
                    "created": 1, "usage": {"total_tokens": 100},
                    "choices": [{"finish_reason": "stop", "message": {
                        "content": json.dumps({"malformed": True} if len(calls) == 1 else score),
                    }}],
                }

            with patch("src.eval80.quality_judge_runner._post_chat", side_effect=fake_post):
                with redirect_stdout(io.StringIO()):
                    self.assertEqual(main([
                        "judge-guidance", "--package-dir", str(package),
                        "--judge-config", str(fixture["judge_config"]),
                        "--handbook", str(fixture["handbook"]),
                    ]), 0)
            self.assertEqual(len(calls), 157)
            self.assertEqual({call[0] for call in calls}, {
                "http://127.0.0.1:8000/v1/chat/completions"
            })
            self.assertTrue((package / "judgements_lock.json").is_file())
            self.assertEqual(len((package / "judgements_private.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()), 156)
            self.assertEqual(len((package / "judge_raw_responses_private.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()), 157)
            attempts = [json.loads(line) for line in (package / "judge_attempts_private.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()]
            self.assertEqual(attempts[0]["status"], "error")
            self.assertEqual(attempts[1]["status"], "success")
            raw_path = package / "judge_raw_responses_private.jsonl"
            original_raw = raw_path.read_bytes()
            raw_rows = [json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines()]
            raw_rows[1]["response"]["id"] = "tampered-provider-response"
            _dump_jsonl(raw_path, raw_rows)
            with self.assertRaisesRegex(ValueError, "matching raw response"):
                main(["report-guidance", "--package-dir", str(package),
                      "--judgements", str(package / "judgements_private.jsonl"),
                      "--output", str(root / "invalid_report.json")])
            raw_path.write_bytes(original_raw)
            raw_rows = [json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines()]
            raw_rows[1]["request_sha256"] = "0" * 64
            _dump_jsonl(raw_path, raw_rows)
            with self.assertRaisesRegex(ValueError, "frozen request hash"):
                main(["report-guidance", "--package-dir", str(package),
                      "--judgements", str(package / "judgements_private.jsonl"),
                      "--output", str(root / "invalid_request_report.json")])

    def test_resume_repairs_score_written_before_success_log(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._fixture(root)
            package = self._prepare(root, fixture)
            calls = []
            failed_once = False
            real_append = quality_judge_runner._append_jsonl

            def fake_post(endpoint, payload, api_key, timeout):
                calls.append(len(calls) + 1)
                return {
                    "id": f"response-{len(calls)}", "model": "local-vision-v1.0",
                    "choices": [{"finish_reason": "stop", "message": {"content": json.dumps({
                        "scores": {key: 1 for key in _SCORE_KEYS},
                        "overall_rationale": "Synthetic grounded guidance.",
                        "zero_score_reasons": {key: "" for key in _SCORE_KEYS},
                        "hallucination": False, "emotion_conflict": False,
                        "recommendation_present": True,
                    })}}],
                }

            def fail_after_score(path, payload):
                nonlocal failed_once
                if (not failed_once and path.name == "judge_attempts_private.jsonl"
                        and payload.get("status") == "success"):
                    failed_once = True
                    raise OSError("simulated crash after durable score write")
                real_append(path, payload)

            command = [
                "judge-guidance", "--package-dir", str(package),
                "--judge-config", str(fixture["judge_config"]),
                "--handbook", str(fixture["handbook"]),
            ]
            with patch("src.eval80.quality_judge_runner._post_chat", side_effect=fake_post):
                with patch("src.eval80.quality_judge_runner._append_jsonl", side_effect=fail_after_score):
                    with self.assertRaisesRegex(OSError, "simulated crash"):
                        main(command)
                self.assertEqual(len(calls), 1)
                with redirect_stdout(io.StringIO()):
                    self.assertEqual(main(command), 0)
            self.assertEqual(len(calls), 156)
            self.assertTrue((package / "judgements_lock.json").is_file())
            self.assertEqual(len((package / "judgements_private.jsonl").read_text(
                encoding="utf-8").splitlines()), 156)
            attempts = [json.loads(line) for line in (package / "judge_attempts_private.jsonl").read_text(
                encoding="utf-8").splitlines()]
            self.assertEqual(sum(item["status"] == "success" for item in attempts), 156)

    def test_remote_endpoint_requires_explicit_opt_in(self):
        self.assertTrue(_endpoint_is_allowed("http://127.0.0.1:8000/v1/chat/completions", False))
        self.assertFalse(_endpoint_is_allowed("https://example.com/v1/chat/completions", False))
        self.assertTrue(_endpoint_is_allowed("https://example.com/v1/chat/completions", True))
        self.assertFalse(_endpoint_is_allowed("http://example.com/v1/chat/completions", True))

    def test_loopback_post_rejects_redirect_and_bypasses_proxy(self):
        calls = []

        class RedirectingOpener:
            def open(self, request, timeout):
                calls.append(request.full_url)
                self_test.assertEqual(request.get_header("Authorization"), "Bearer secret")
                redirect = next(handler for handler in handlers if isinstance(handler, _NoRedirectHandler))
                self_test.assertIsNone(redirect.redirect_request(
                    request, None, 307, "redirect", {}, "https://elsewhere.test/steal"
                ))
                raise HTTPError(request.full_url, 307, "redirect", {}, None)

        self_test = self
        handlers = []

        def fake_build_opener(*given):
            handlers.extend(given)
            return RedirectingOpener()

        with patch("src.eval80.quality_judge_runner.build_opener", side_effect=fake_build_opener):
            with self.assertRaisesRegex(RuntimeError, "HTTP 307"):
                _post_chat("http://127.0.0.1:8000/v1/chat/completions", {}, "secret", 10)
        self.assertEqual(calls, ["http://127.0.0.1:8000/v1/chat/completions"])
        self.assertEqual(next(handler for handler in handlers if isinstance(handler, ProxyHandler)).proxies, {})

    def test_pinned_judge_fingerprint_must_match_each_response(self):
        packet = {"anonymous_item_id": "JONE", "response": "Aesthetic advice"}
        metadata = {
            "judge_model_id": "judge-v1", "judge_config_sha256": "a" * 64,
            "expected_system_fingerprint": "weights-v1",
        }
        scored = {
            "scores": {key: 1 for key in _SCORE_KEYS},
            "overall_rationale": "Synthetic example.",
            "zero_score_reasons": {key: "" for key in _SCORE_KEYS},
            "hallucination": False, "emotion_conflict": False,
            "recommendation_present": True,
        }
        response = {
            "id": "r1", "model": "judge-v1", "system_fingerprint": "weights-v2",
            "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(scored)}}],
        }
        with self.assertRaisesRegex(ValueError, "fingerprint differs"):
            _extract_judgement(response, packet, metadata)
        response["system_fingerprint"] = "weights-v1"
        self.assertEqual(
            _extract_judgement(response, packet, metadata)["response_metadata"]["system_fingerprint"],
            "weights-v1",
        )

    def test_combined_report_keeps_missing_metrics_explicit(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "incomplete.json"
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(["report-benchmark", "--output-json", str(output),
                                       "--output-md", str(root / "incomplete.md")]), 0)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "incomplete")
            self.assertEqual(len(report["missing_inputs"]), 6)
            self.assertNotIn("correct_count", report)

    def test_incomplete_report_retains_verified_b1_zero_without_imputing_c(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            records_hash = "a" * 64
            fmt = root / "b1_format.json"
            rec = root / "b1_recognition.json"
            _dump(fmt, {
                "condition": "B1", "item_count": 80,
                "format_pass_count": 0, "format_failure_count": 80,
                "failure_reason_counts": {"missing_guidance": 80},
                "records_sha256": records_hash,
            })
            _dump(rec, {
                "condition": "B1", "item_count": 80,
                "correct_count": 0, "records_sha256": records_hash,
            })
            output = root / "incomplete.json"
            markdown = root / "incomplete.md"
            args = [
                "report-benchmark", "--b1-format", str(fmt),
                "--b1-recognition", str(rec),
                "--output-json", str(output), "--output-md", str(markdown),
            ]
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(args), 0)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "incomplete")
            self.assertEqual(report["format"]["B1"]["pass_count"], 0)
            self.assertEqual(report["recognition"]["B1"]["correct_count"], 0)
            self.assertEqual(report["format"]["B1"]["records_sha256"], records_hash)
            self.assertIsNone(report["format"]["C"])
            self.assertIsNone(report["recognition"]["C"])
            self.assertIsNone(report["guidance_proxy"])
            self.assertEqual(set(report["input_sha256"]), {"b1_format", "b1_recognition"})
            self.assertIn("0/80", markdown.read_text(encoding="utf-8"))
            self.assertIn("pending", markdown.read_text(encoding="utf-8"))
            changed = json.loads(rec.read_text(encoding="utf-8"))
            changed["records_sha256"] = "b" * 64
            _dump(rec, changed)
            with self.assertRaisesRegex(ValueError, "different outputs"):
                main(args[:-4] + ["--output-json", str(root / "bad.json"),
                                    "--output-md", str(root / "bad.md")])


if __name__ == "__main__":
    unittest.main()
