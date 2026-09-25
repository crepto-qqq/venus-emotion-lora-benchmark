from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = PROJECT_ROOT / "phase3" / "tools"
sys.path.insert(0, str(TOOLS_DIR))

from common import (  # noqa: E402
    expected_model_provenance,
    read_json,
    sha256_bytes,
)
from preflight import _gpu_memory_gate as preflight_memory_gate  # noqa: E402
from report import validate_value  # noqa: E402
from smoke_backward import (  # noqa: E402
    EXPECTED_MATCHED_TARGET_MODULE_COUNT,
    EXPECTED_MINIMUM_FREE_GPU_MEMORY_BYTES,
    EXPECTED_MINIMUM_TOTAL_GPU_MEMORY_BYTES,
    EXPECTED_TRAINABLE_PARAMETER_COUNT,
    SmokeFailure,
    _gpu_memory_gate as smoke_memory_gate,
    _require_unique_floating_parameter_dtype,
)


class _FakeParameter:
    def __init__(self, dtype: object, *, floating: bool = True) -> None:
        self.dtype = dtype
        self._floating = floating

    def is_floating_point(self) -> bool:
        return self._floating


class RuntimeContractTests(unittest.TestCase):
    def test_gpu_memory_gate_has_exact_locked_boundaries(self) -> None:
        minimum_total = 44 * 1024**3
        minimum_free = 40 * 1024**3
        for gate in (smoke_memory_gate, preflight_memory_gate):
            with self.subTest(gate=gate.__module__):
                self.assertTrue(
                    gate(minimum_total, minimum_free, minimum_total, minimum_free)
                )
                self.assertFalse(
                    gate(minimum_total, minimum_free - 1, minimum_total, minimum_free)
                )
                self.assertFalse(
                    gate(minimum_total - 1, minimum_free, minimum_total, minimum_free)
                )

    def test_gpu_architecture_constants_match_the_reviewed_config(self) -> None:
        config = read_json(PROJECT_ROOT / "phase3/configs/member1-smoke.json")
        self.assertEqual(
            config["runtime"]["minimum_gpu_memory_bytes"],
            EXPECTED_MINIMUM_TOTAL_GPU_MEMORY_BYTES,
        )
        self.assertEqual(
            config["runtime"]["minimum_free_gpu_memory_bytes"],
            EXPECTED_MINIMUM_FREE_GPU_MEMORY_BYTES,
        )
        self.assertEqual(
            config["lora"]["expected_matched_module_count"],
            EXPECTED_MATCHED_TARGET_MODULE_COUNT,
        )
        self.assertEqual(
            config["lora"]["expected_trainable_parameter_count"],
            EXPECTED_TRAINABLE_PARAMETER_COUNT,
        )

    def test_parameter_dtype_evidence_rejects_a_mixed_set(self) -> None:
        bf16 = object()
        fp32 = object()
        observed = _require_unique_floating_parameter_dtype(
            [("a", _FakeParameter(bf16)), ("b", _FakeParameter(bf16))],
            expected_dtype=bf16,
            label="fake base model",
        )
        self.assertTrue(observed)
        with self.assertRaises(SmokeFailure):
            _require_unique_floating_parameter_dtype(
                [("a", _FakeParameter(bf16)), ("b", _FakeParameter(fp32))],
                expected_dtype=bf16,
                label="fake base model",
            )

    def test_model_snapshot_report_rejects_cross_field_contradictions(self) -> None:
        source_lock = read_json(PROJECT_ROOT / "phase3/configs/source-lock.json")
        model_lock = source_lock["model"]
        marker = expected_model_provenance(model_lock)
        marker_payload = (
            json.dumps(marker, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
        ).encode("utf-8")
        report = {
            "schema_version": 1,
            "kind": "phase3_model_snapshot",
            "generated_at_utc": "2026-09-24T00:00:00Z",
            "passed": True,
            "marker_created": True,
            "download_attempted": True,
            "missing_file_count_before": int(model_lock["snapshot_file_count"]),
            "source_lock": {
                "repository": model_lock["repository"],
                "revision": model_lock["revision"],
                "weight_format": model_lock["weight_format"],
                "snapshot_manifest_sha256": model_lock["snapshot_manifest_sha256"],
            },
            "snapshot": {
                "manifest_sha256": model_lock["snapshot_manifest_sha256"],
                "file_count": int(model_lock["snapshot_file_count"]),
                "total_bytes": int(model_lock["snapshot_total_bytes"]),
            },
            "provenance": {
                **marker,
                "marker_sha256": sha256_bytes(marker_payload),
            },
            "sealed": {
                "applied": True,
                "verified": True,
                "manifest_file_count": int(model_lock["snapshot_file_count"]),
                "manifest_files_read_only": True,
                "marker_read_only": True,
                "model_root_read_only": True,
            },
            "errors": [],
        }
        validate_value(report, "model_snapshot", PROJECT_ROOT)
        report["provenance"]["revision"] = "0" * 40
        with self.assertRaisesRegex(ValueError, "provenance evidence"):
            validate_value(report, "model_snapshot", PROJECT_ROOT)


if __name__ == "__main__":
    unittest.main()
