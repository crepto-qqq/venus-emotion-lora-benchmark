"""Synthetic end-to-end checks for the private swapped-order preference proxy."""

from __future__ import annotations

from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from urllib.request import ProxyHandler

from src.eval80.cli import main
from src.eval80.core import sha256_file
from src.eval80.preference_benchmark import _object_sha256
from src.eval80.preference_judge_runner import (
    _make_payload, _preference_schema, _probe_ollama_digest,
)
from src.eval80.quality_judge_runner import _NoRedirectHandler
from tests.eval80 import test_quality_benchmark as quality_test


_PROMPT = (
    "Compare two anonymous aesthetic-guidance answers for this original image and the fixed "
    "target emotion. Select the answer that would give a user more useful, visually "
    "grounded advice to achieve the target. Verify claims about what is currently visible "
    "against the pixels. Penalize invented objects or scenes, empty advice such as 'make "
    "it better', and edits that weaken the target emotion. Reward a specific feasible edit "
    "to a visible area and a clear link to the target. Do not reward length or fluent style. "
    "If neither answer is meaningfully better, choose tie. Also mark which answer is "
    "better on visual grounding, actionability, and target-emotion fit separately. "
    "Explain decisive visual evidence and edits in one or two sentences. Return only JSON."
)


def _dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False) + "\n", encoding="utf-8")


class PreferenceBenchmarkTests(unittest.TestCase):
    def _fixture(self, root: Path) -> dict[str, Path]:
        fixture = quality_test.GuidanceJudgeTests._fixture(self, root)
        b1_config_path = fixture["b1_dir"] / "run_config.json"
        b1_config = json.loads(b1_config_path.read_text(encoding="utf-8"))
        b1_config["model_source"].update({
            "historical_provenance": {"recovery_basis": "archived source receipt"},
            "recovery_basis": "historical B1 source metadata",
        })
        _dump(b1_config_path, b1_config)
        # One exact B1/C duplicate exercises the transparent deterministic-tie rule.
        b1_records = fixture["b1_dir"] / "records.jsonl"
        c_rows = [json.loads(line) for line in (
            fixture["c_dir"] / "records.jsonl"
        ).read_text(encoding="utf-8").splitlines()]
        c_duplicate = next(row for row in c_rows if row["blind_id"] == "E078")
        b1_rows = [json.loads(line) for line in b1_records.read_text(
            encoding="utf-8").splitlines()]
        for row in b1_rows:
            if row["blind_id"] == "E078":
                row["response"] = c_duplicate["response"]
                row["response_sha256"] = c_duplicate["response_sha256"]
        b1_records.write_text("".join(json.dumps(row) + "\n" for row in b1_rows),
                              encoding="utf-8")
        for name in ("summary.json", "validation.json"):
            path = fixture["b1_dir"] / name
            document = json.loads(path.read_text(encoding="utf-8"))
            document["records_sha256"] = sha256_file(b1_records)
            _dump(path, document)
        config = {
            "schema_version": 1,
            "model_id": "local-vision-calibrated:v1",
            "provider": "openai_compatible_chat",
            "endpoint": "http://127.0.0.1:11434/v1/chat/completions",
            "prompt": _PROMPT,
            "image_detail": "high",
            "decoding": {"temperature": 0, "seed": 42, "max_tokens": 550},
            "retry_policy": {"max_attempts": 2, "timeout_seconds": 15,
                             "backoff_seconds": 0},
        }
        _dump(fixture["judge_config"], config)
        digest = "a" * 64
        model_receipt = root / "model_receipt.json"
        _dump(model_receipt, {
            "schema_version": 1, "model_id": config["model_id"],
            "model_digest_sha256": digest,
        })
        evidence_dir = root / "synthetic_preflight"
        evidence_dir.mkdir()
        dev_images = {}
        for image in ("synthetic", "bridge"):
            path = evidence_dir / f"{image}.jpg"
            path.write_bytes(f"non-Eval80 development image {image}".encode())
            dev_images[image] = path
        dev_image_hashes = {
            "synthetic": sha256_file(dev_images["synthetic"]),
            "bridge_dev": sha256_file(dev_images["bridge"]),
        }
        summary: list[dict] = []
        evidence: list[dict] = []
        response_number = 0
        for image, cases in (
            ("synthetic", ("hallucination", "generic", "conflict", "identical",
                           "equivalent", "style_only", "verbose_only")),
            ("bridge", ("hallucination", "generic", "conflict")),
        ):
            for case in cases:
                repeats = 2 if image == "synthetic" and case in {
                    "hallucination", "generic", "conflict"
                } else 1
                for _ in range(repeats):
                    response_ids = []
                    for order in ("normal", "swapped"):
                        response_number += 1
                        response_id = f"synthetic-cal-{response_number}"
                        response_ids.append(response_id)
                        answers = ("grounded answer", f"{case} challenger")
                        if order == "swapped":
                            answers = answers[::-1]
                        request = _make_payload({
                            "image_path": str(dev_images[image]),
                            "image_sha256": sha256_file(dev_images[image]),
                            "target_emotion": "contentment",
                            "answer_A": answers[0],
                            "answer_A_sha256": hashlib.sha256(answers[0].encode()).hexdigest(),
                            "answer_B": answers[1],
                            "answer_B_sha256": hashlib.sha256(answers[1].encode()).hexdigest(),
                        }, config)
                        for role, value in (
                            ("request", request),
                            ("raw_response", {"id": response_id,
                                              "model": config["model_id"]}),
                        ):
                            path = evidence_dir / f"{response_number}_{role}.json"
                            _dump(path, value)
                            evidence.append({"role": role, "path": str(path.resolve()),
                                             "sha256": sha256_file(path)})
                    consensus = ("clean" if case in {"hallucination", "generic", "conflict"}
                                 else "identical_bypass_tie" if case == "identical"
                                 else "unstable")
                    summary.append({"image": image, "case": case,
                                    "consensus": consensus, "response_ids": response_ids})
        summary_path = evidence_dir / "summary.json"
        _dump(summary_path, summary)
        evidence.append({"role": "calibration_summary", "path": str(summary_path.resolve()),
                         "sha256": sha256_file(summary_path)})
        preflight = root / "preflight_receipt.json"
        _dump(preflight, {
            "schema_version": 1, "status": "passed",
            "pass_scope": "dual_order_consensus_only",
            "data_scope": "synthetic_and_bridge_dev_only", "eval80_outputs_used": False,
            "endpoint": config["endpoint"],
            "model_id": config["model_id"], "model_digest_sha256": digest,
            "model_digest_before_sha256": digest,
            "model_digest_after_sha256": digest,
            "prompt": config["prompt"],
            "prompt_sha256": hashlib.sha256(_PROMPT.encode()).hexdigest(),
            "response_schema": _preference_schema(),
            "response_schema_sha256": _object_sha256(_preference_schema()),
            "decoding": config["decoding"],
            "decoding_sha256": _object_sha256(config["decoding"]),
            "image_detail": "high", "message_content_order": ["image_url", "text"],
            "image_sha256": dev_image_hashes,
            "calibration_pairs": 13, "calibration_requests": 26,
            "clear_pairs_consensus_clean": 9,
            "neutral_pairs": {"identical_bypass_tie": 1, "unstable": 3},
            "evidence_files": evidence,
        })
        fixture["model_receipt"] = model_receipt
        fixture["preflight"] = preflight
        fixture["summary_path"] = summary_path
        return fixture

    def _prepare_args(self, fixture: dict[str, Path], package: Path) -> list[str]:
        args = ["prepare-preference-judge"]
        for key, flag in (
            ("b1_dir", "--b1-run-dir"), ("c_dir", "--c-run-dir"),
            ("selection", "--selection-manifest"), ("image_dir", "--image-dir"),
            ("judge_config", "--judge-config"),
            ("model_receipt", "--model-digest-receipt"),
            ("preflight", "--preflight-receipt"),
            ("manifest", "--inference-manifest"), ("freeze", "--freeze-record"),
            ("exclusion", "--display-exclusion"), ("config", "--config"),
            ("handbook", "--handbook"),
        ):
            args.extend((flag, str(fixture[key])))
        args.extend(("--output-dir", str(package)))
        return args

    def _prepare(self, root: Path, fixture: dict[str, Path]) -> Path:
        package = root / "private_preference_package"
        with redirect_stdout(io.StringIO()):
            self.assertEqual(main(self._prepare_args(fixture, package)), 0)
        return package

    def test_swapped_package_judge_trace_and_conservative_report(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._fixture(root)
            package = self._prepare(root, fixture)
            packets = [json.loads(line) for line in (
                package / "preference_items_private.jsonl"
            ).read_text(encoding="utf-8").splitlines()]
            mapping = json.loads((package / "preference_mapping_private.json").read_text(
                encoding="utf-8"))["items"]
            self.assertEqual(len(packets), 156)
            self.assertEqual(len(mapping), 156)
            packet_text = (package / "preference_items_private.jsonl").read_text(
                encoding="utf-8")
            self.assertNotIn('"blind_id"', packet_text)
            self.assertNotIn('"condition"', packet_text)
            self.assertNotIn("E001.jpg", packet_text)
            self.assertNotIn("E080.jpg", packet_text)
            model_id = json.loads(fixture["judge_config"].read_text(
                encoding="utf-8"))["model_id"]
            position = 0

            def fake_post(endpoint, payload, key, timeout):
                nonlocal position
                packet = packets[position]
                item = mapping[packet["anonymous_item_id"]]
                index = int(item["blind_id"][1:])
                target = "C" if index <= 76 else "B1" if index == 77 else "A"
                preferred = (target if target in {"A", "tie"} else
                             "A" if item["A_condition"] == target else "B")
                position += 1
                return {
                    "id": f"synthetic-run-{position}", "model": model_id,
                    "choices": [{"finish_reason": "stop", "message": {
                        "content": json.dumps({
                            "preferred": preferred,
                            "better_grounding": preferred,
                            "better_actionability": preferred,
                            "better_target_fit": preferred,
                            "rationale": "Synthetic comparison based on visible image areas.",
                        })}}],
                }

            probe_phases = []

            def fake_probe(endpoint, model, digest, timeout, *, phase):
                probe_phases.append(phase)
                return {
                    "phase": phase, "observed_at_utc": "2026-10-17T00:00:00+00:00",
                    "model_id": model, "digest_sha256": digest,
                }

            with patch("src.eval80.preference_judge_runner._post_chat", fake_post), patch(
                "src.eval80.preference_judge_runner._probe_ollama_digest", fake_probe
            ):
                with redirect_stdout(io.StringIO()):
                    self.assertEqual(main(["judge-preference", "--package-dir", str(package),
                                           "--judge-config", str(fixture["judge_config"])]), 0)
            self.assertEqual(position, 156)
            self.assertEqual(probe_phases, ["before", "after"])
            output = root / "aggregate_preference.json"
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(["report-preference", "--package-dir", str(package),
                                       "--output", str(output)]), 0)
            report = json.loads(output.read_text(encoding="utf-8"))
            consensus = report["image_consensus"]
            self.assertEqual(consensus["C_wins"], 75)
            self.assertEqual(consensus["B1_wins"], 1)
            self.assertEqual(consensus["ties"], 1)
            self.assertEqual(consensus["identical_answer_rule_ties"], 1)
            self.assertEqual(consensus["swap_disagreements_unstable"], 1)
            self.assertEqual(consensus["stable_net_c_minus_b1"], 74)
            self.assertEqual(consensus["worst_case_net_if_all_unstable_favor_b1"], 73)
            self.assertEqual(report["model_digest_verification_status"],
                             "runtime_ollama_before_after_matched")
            self.assertEqual(report["status"], "provisional_proxy")
            self.assertNotIn("E002", output.read_text(encoding="utf-8"))
            self.assertTrue((package / "preference_cases_private.json").is_file())
            raw_path = package / "preference_raw_responses_private.jsonl"
            self.assertEqual(len(raw_path.read_text(encoding="utf-8").splitlines()), 156)
            original_raw = raw_path.read_bytes()
            raw_path.write_bytes(original_raw.replace(b"synthetic-run-1", b"forged-run-1", 1))
            with self.assertRaisesRegex(ValueError, "raw response|raw provider response|matching"):
                main(["report-preference", "--package-dir", str(package),
                      "--output", str(root / "forged.json")])

    def test_preflight_evidence_and_remote_endpoint_are_rejected(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._fixture(root)
            config = json.loads(fixture["judge_config"].read_text(encoding="utf-8"))
            config["endpoint"] = "https://example.com/v1/chat/completions"
            _dump(fixture["judge_config"], config)
            with self.assertRaisesRegex(ValueError, "loopback"):
                main(self._prepare_args(fixture, root / "private_remote"))
            config["endpoint"] = "http://127.0.0.1:11434/v1/chat/completions"
            _dump(fixture["judge_config"], config)
            fixture["summary_path"].write_text("[]\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "evidence file.*changed"):
                main(self._prepare_args(fixture, root / "private_changed"))

    def test_b1_historical_metadata_is_allowed_but_core_base_revision_must_match(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._fixture(root)
            package = self._prepare(root, fixture)
            self.assertTrue((package / "preference_package.json").is_file())
            c_config_path = fixture["c_dir"] / "run_config.json"
            c_config = json.loads(c_config_path.read_text(encoding="utf-8"))
            c_config["model_source"]["revision"] = "wrong-revision"
            _dump(c_config_path, c_config)
            with self.assertRaisesRegex(ValueError, "C base model identity differs"):
                main(self._prepare_args(fixture, root / "private_wrong_revision"))

    def test_ollama_digest_probe_disables_proxy_and_redirects(self):
        model = "local-vision-calibrated:v1"
        digest = "a" * 64

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_):
                return False

            def read(self, limit):
                self_limit = limit
                assert self_limit == 1_000_001
                return json.dumps({"models": [{"name": model, "digest": digest}]}).encode()

        class FakeOpener:
            def open(self, request, timeout):
                self.request = request
                self.timeout = timeout
                return FakeResponse()

        opener = FakeOpener()

        def fake_build_opener(*handlers):
            self.assertTrue(any(isinstance(value, ProxyHandler) for value in handlers))
            self.assertTrue(any(isinstance(value, _NoRedirectHandler) for value in handlers))
            return opener

        with patch("src.eval80.preference_judge_runner.build_opener", fake_build_opener):
            observed = _probe_ollama_digest(
                "http://127.0.0.1:11434/v1/chat/completions", model, digest, 10,
                phase="before",
            )
        self.assertEqual(opener.request.full_url, "http://127.0.0.1:11434/api/tags")
        self.assertEqual(opener.request.get_method(), "GET")
        self.assertEqual(observed["digest_sha256"], digest)
        with patch("src.eval80.preference_judge_runner.build_opener", fake_build_opener):
            with self.assertRaisesRegex(ValueError, "differs"):
                _probe_ollama_digest(
                    "http://127.0.0.1:11434/v1/chat/completions", model,
                    "b" * 64, 10, phase="before",
                )


if __name__ == "__main__":
    unittest.main()
