from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from copy import deepcopy
from io import StringIO
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = PROJECT_ROOT / "phase3" / "tools"
sys.path.insert(0, str(TOOLS_DIR))

from build_smoke_fixture import (  # noqa: E402
    DEFAULT_EVAL20_RESULT,
    DEFAULT_EVAL80_MANIFEST,
    _check_disjointness,
    _read_locked_record,
)
from common import read_json, sha256_file, write_json_atomic  # noqa: E402
from report import validate_value  # noqa: E402
from verify_handoff import (  # noqa: E402
    VerificationFailure,
    _failure_report_main,
    _failure_result,
    _expected_fixture_semantics,
    _require_secure_source_bundle,
    _verify_fixture_artifacts,
    _verify_handoff_fixture_hashes,
)


class HandoffVerificationTests(unittest.TestCase):
    def test_failure_result_is_schema_valid_and_keeps_completed_checks(self) -> None:
        result = _failure_result(
            member_id="member2",
            report_dir=Path("/workspace/phase3/reports/member1/attempt-001"),
            runtime_root=Path("/workspace/phase3"),
            failure_code="source_bundle_changed",
            checks=[
                {"name": "first", "status": "pass"},
                {"name": "discarded_failure", "status": "fail"},
            ],
        )
        validate_value(result, "verification", PROJECT_ROOT)
        self.assertFalse(result["passed"])
        self.assertEqual(result["failure_code"], "source_bundle_changed")
        self.assertEqual([item["name"] for item in result["checks"]], ["first", "source_bundle_changed"])

    def test_failure_report_replaces_a_success_result(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "handoff-verification.json"
            previous = {
                "checks": [{"name": "already_complete", "status": "pass"}],
                "passed": True,
            }
            output.write_text(json.dumps(previous), encoding="utf-8")
            with redirect_stdout(StringIO()):
                status = _failure_report_main(
                    [
                        "--runtime-root", "/workspace/phase3",
                        "--report-dir", "/workspace/phase3/reports/member1/attempt-001",
                        "--member-id", "member2",
                        "--failure-code", "source_bundle_changed",
                        "--output", str(output),
                        "--preserve-checks-from", str(output),
                    ]
                )
            self.assertEqual(status, 0)
            result = read_json(output)
            validate_value(result, "verification", PROJECT_ROOT)
            self.assertFalse(result["passed"])
            self.assertEqual(result["checks"][0]["name"], "already_complete")

    @unittest.skipIf(os.name == "nt", "POSIX mode and symlink semantics are verified on RunPod/Linux")
    def test_source_bundle_must_be_flat_real_and_read_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "attempt-001"
            root.mkdir()
            evidence = root / "evidence.json"
            evidence.write_text("{}\n", encoding="utf-8")
            evidence.chmod(0o444)
            root.chmod(0o555)
            observation = _require_secure_source_bundle(root)
            self.assertEqual(observation["file_count"], 1)

            root.chmod(0o755)
            evidence.chmod(0o644)
            with self.assertRaisesRegex(VerificationFailure, "source_bundle_writable"):
                _require_secure_source_bundle(root)
            evidence.chmod(0o444)
            root.chmod(0o555)

    def _fixture_and_manifest(self) -> tuple[list[dict], dict]:
        config = read_json(PROJECT_ROOT / "phase3/configs/member1-smoke.json")
        lock = config["fixture"]
        restricted_dataset = PROJECT_ROOT / lock["dataset_relpath"]
        if not restricted_dataset.is_file():
            self.skipTest(
                "restricted BridgeTrain fixture is intentionally excluded from the public portfolio"
            )
        record = _read_locked_record(PROJECT_ROOT, config)
        disjointness, evaluation_checks = _check_disjointness(
            project_root=PROJECT_ROOT,
            source_image_id=lock["source_image_id"],
            image_sha256=lock["image_sha256"],
            eval80_manifest=DEFAULT_EVAL80_MANIFEST,
            eval20_result=DEFAULT_EVAL20_RESULT,
        )
        reference = "/workspace/phase3/smoke/contentment_05000.jpg"
        fixture = [{
            "id": lock["source_image_id"],
            "technical_infrastructure_only": True,
            "conversations": [
                {"from": "user", "value": f"Picture 1: <img>{reference}</img>\n{record['instruction']}"},
                {"from": "assistant", "value": record["target_response"]},
            ],
            "provenance": {
                "dataset_record_sha256": lock["record_sha256"],
                "annotation_sha256": lock["annotation_sha256"],
                "image_sha256": lock["image_sha256"],
                "source_split": "train",
                "disjointness_check": disjointness,
            },
        }]
        manifest = {
            "schema_version": 1,
            "kind": "phase3_technical_fixture_manifest",
            "generated_at_utc": "2026-09-24T00:00:00Z",
            "technical_infrastructure_only": True,
            "source_image_id": lock["source_image_id"],
            "emotion": lock["emotion"],
            "source_split": "train",
            "review_status": record["provenance"]["review_status"],
            "label_supported": record["provenance"]["label_supported"],
            "source_dataset_relpath": lock["dataset_relpath"],
            "source_record_line": lock["record_line"],
            "dataset_record_sha256": lock["record_sha256"],
            "annotation_sha256": lock["annotation_sha256"],
            "image_sha256": lock["image_sha256"],
            "image": {
                "reference": reference,
                "byte_count": lock["image_byte_count"],
                "width": lock["image_width"],
                "height": lock["image_height"],
            },
            "disjointness_check": disjointness,
            "evaluation_checks": evaluation_checks,
            "full_dataset_ready": False,
            "formal_training_authorized": False,
        }
        return fixture, manifest

    def test_fixture_chain_rebuilds_all_fixed_semantics(self) -> None:
        fixture, manifest = self._fixture_and_manifest()
        with tempfile.TemporaryDirectory() as temporary:
            report_dir = Path(temporary)
            fixture_path = report_dir / "technical-fixture.json"
            manifest_path = report_dir / "technical-fixture-manifest.json"
            write_json_atomic(fixture_path, fixture)
            manifest["fixture_sha256"] = sha256_file(fixture_path)
            write_json_atomic(manifest_path, manifest)
            (report_dir / "fixture-artifacts.sha256").write_text(
                f"{sha256_file(fixture_path)}  technical-fixture.json\n"
                f"{sha256_file(manifest_path)}  technical-fixture-manifest.json\n",
                encoding="utf-8",
            )
            lock = read_json(PROJECT_ROOT / "phase3/configs/member1-smoke.json")["fixture"]
            smoke = {"fixture": {
                "id": lock["source_image_id"],
                "fixture_sha256": sha256_file(fixture_path),
                "image_sha256": lock["image_sha256"],
                "dataset_record_sha256": lock["record_sha256"],
            }}
            observation = _verify_fixture_artifacts(report_dir, smoke, PROJECT_ROOT)
            self.assertEqual(observation["annotation_sha256"], lock["annotation_sha256"])

            mutated = json.loads(json.dumps(fixture))
            mutated[0]["conversations"][1]["value"] += " tampered"
            write_json_atomic(fixture_path, mutated)
            manifest["fixture_sha256"] = sha256_file(fixture_path)
            write_json_atomic(manifest_path, manifest)
            (report_dir / "fixture-artifacts.sha256").write_text(
                f"{sha256_file(fixture_path)}  technical-fixture.json\n"
                f"{sha256_file(manifest_path)}  technical-fixture-manifest.json\n",
                encoding="utf-8",
            )
            smoke["fixture"]["fixture_sha256"] = sha256_file(fixture_path)
            with self.assertRaises(ValueError):
                _verify_fixture_artifacts(report_dir, smoke, PROJECT_ROOT)

    def test_manifest_review_image_disjointness_and_stop_line_are_fixed(self) -> None:
        fixture, manifest = self._fixture_and_manifest()
        manifest["fixture_sha256"] = "a" * 64
        _expected_fixture_semantics(PROJECT_ROOT, fixture, manifest)
        mutations = (
            ("review", lambda value: value.__setitem__("review_status", "edited")),
            ("label", lambda value: value.__setitem__("label_supported", False)),
            ("image", lambda value: value["image"].__setitem__("width", 721)),
            ("disjointness", lambda value: value.__setitem__("disjointness_check", "limited")),
            ("full_data_stop", lambda value: value.__setitem__("full_dataset_ready", True)),
            ("training_stop", lambda value: value.__setitem__("formal_training_authorized", True)),
        )
        for label, mutate in mutations:
            with self.subTest(label=label):
                changed = deepcopy(manifest)
                mutate(changed)
                with self.assertRaises(ValueError):
                    _expected_fixture_semantics(PROJECT_ROOT, fixture, changed)

    def test_handoff_must_match_both_fixture_hashes(self) -> None:
        handoff = {"fixture": {"fixture_sha256": "a" * 64, "fixture_manifest_sha256": "b" * 64}}
        observation = {"fixture_sha256": "a" * 64, "fixture_manifest_sha256": "b" * 64}
        _verify_handoff_fixture_hashes(handoff, observation)
        observation["fixture_manifest_sha256"] = "c" * 64
        with self.assertRaisesRegex(VerificationFailure, "fixture_evidence_invalid"):
            _verify_handoff_fixture_hashes(handoff, observation)


if __name__ == "__main__":
    unittest.main()
