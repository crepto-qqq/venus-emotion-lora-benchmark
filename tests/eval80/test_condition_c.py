"""Condition C identity and frozen B1 parity checks without a GPU."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from src.eval80.core import sha256_file
from src.eval80.inference import (
    _condition_prompt,
    _expected_adapter_identity,
    _load_adapter_preserving_rng,
    _verify_adapter_files,
    _verify_base_snapshot,
)
from src.eval80.validation import validate_run


CONFIG_PATH = Path("configs/phase1/eval80.json")


class ConditionCValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        self.adapter = self.root / "adapter"
        self.adapter.mkdir()
        (self.adapter / "adapter_config.json").write_text('{"peft_type":"LORA"}')
        (self.adapter / "adapter_model.bin").write_bytes(b"test weights")
        adapter_files = {
            name: sha256_file(self.adapter / name)
            for name in ("adapter_config.json", "adapter_model.bin")
        }
        self.evidence_path = self.root / "evidence.json"
        self.evidence_path.write_bytes(Path(
            "phase3/evidence/member3/formal-20261010/results-summary.json"
        ).read_bytes())
        self.synthetic_adapter_identity = {"adapter_files": adapter_files}
        identity = _expected_adapter_identity(self.evidence_path, self.config["model"])
        self.manifest = {"records": [
            {"blind_id": f"E{index:03d}",
             "image_sha256": hashlib.sha256(str(index).encode()).hexdigest()}
            for index in range(1, 81)
        ]}
        records = [
            {"stage": "stage1", "condition": "C",
             "prompt_version": self.config["conditions"]["B1"]["prompt_version"],
             "sequence_index": index, "blind_id": item["blind_id"],
             "image_sha256": item["image_sha256"], "status": "success",
             "response": "Emotion: awe",
             "response_sha256": hashlib.sha256(b"Emotion: awe").hexdigest()}
            for index, item in enumerate(self.manifest["records"], start=1)
        ]
        records_path = self.root / "records.jsonl"
        records_path.write_text("".join(json.dumps(row) + "\n" for row in records))
        common = {
            "prompt": self.config["conditions"]["B1"]["prompt"],
            "prompt_version": self.config["conditions"]["B1"]["prompt_version"],
            "seed": self.config["seed"], "precision": self.config["precision"],
            "batch_size": self.config["batch_size"], "manifest_size": 80,
            "generation_expected": self.config["generation"],
            "generation_actual": self.config["generation"],
            "model_loading": self.config["model_loading"],
            "model_source": self.config["model"],
            "file_sha256": {
                key: key * 4 for key in (
                    "config", "scoring_handbook", "inference_manifest", "freeze_record"
                )
            },
        }
        self.b1_path = self.root / "b1_run_config.json"
        self.b1_path.write_text(json.dumps({**common, "condition": "B1"}))
        lock_path = Path("phase3/configs/source-lock.json")
        source_lock = json.loads(lock_path.read_text())["model"]
        snapshot_path = Path(source_lock["snapshot_manifest"])
        snapshot = json.loads(snapshot_path.read_text())
        (self.root / "run_config.json").write_text(json.dumps({
            **common, "condition": "C", "prompt_source_condition": "B1",
            "adapter_identity": identity, "parameter_devices": ["cuda:0"],
            "rng_seed_position": "before_shared_base_load",
            "adapter_load_rng_restored": True,
            "adapter_load_rng_scope": "cpu_and_all_visible_cuda_devices",
            "base_snapshot_verification": {
                "source_lock_sha256": sha256_file(lock_path),
                "snapshot_manifest_sha256": sha256_file(snapshot_path),
                "verified_file_count": snapshot["file_count"],
                "verified_total_bytes": snapshot["total_bytes"],
            },
        }))
        (self.root / "environment.json").write_text("{}")
        (self.root / "summary.json").write_text(json.dumps({
            "completed": True, "condition": "C", "record_count": 80,
            "expected_count": 80, "empty_response_count": 0,
            "records_sha256": sha256_file(records_path),
        }))

    def tearDown(self):
        self.temp.cleanup()

    def validate(self):
        return validate_run(
            self.root, self.manifest, self.config, "C",
            adapter_evidence_path=self.evidence_path,
            b1_run_config_path=self.b1_path,
        )

    def test_c_uses_frozen_b1_prompt_and_accepts_matching_artifacts(self):
        self.assertEqual(_condition_prompt(self.config, "C"), _condition_prompt(self.config, "B1"))
        identity = _expected_adapter_identity(self.evidence_path, self.config["model"])
        self.assertEqual(identity["selected_checkpoint"], 180)
        self.assertTrue(self.validate()["valid"], self.validate()["errors"])

    def test_rejects_changed_adapter_weights(self):
        _verify_adapter_files(self.adapter, self.synthetic_adapter_identity)
        (self.adapter / "adapter_model.bin").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            _verify_adapter_files(self.adapter, self.synthetic_adapter_identity)

    def test_rejects_prompt_drift(self):
        run_path = self.root / "run_config.json"
        run = json.loads(run_path.read_text())
        run["prompt"] += " extra"
        run_path.write_text(json.dumps(run))
        report = self.validate()
        self.assertFalse(report["valid"])
        self.assertTrue(any("prompt" in error for error in report["errors"]))

    def test_rejects_private_adapter_path_in_run_config(self):
        run_path = self.root / "run_config.json"
        run = json.loads(run_path.read_text())
        run["adapter_path"] = "/private/adapter/checkpoint-180"
        run_path.write_text(json.dumps(run))
        report = self.validate()
        self.assertFalse(report["valid"])
        self.assertTrue(any("private adapter path" in error for error in report["errors"]))

    def test_rejects_missing_rng_restoration_evidence(self):
        run_path = self.root / "run_config.json"
        run = json.loads(run_path.read_text())
        del run["adapter_load_rng_restored"]
        run_path.write_text(json.dumps(run))
        report = self.validate()
        self.assertFalse(report["valid"])
        self.assertTrue(any("RNG restoration" in error for error in report["errors"]))

    def test_rejects_modified_formal_evidence(self):
        evidence = json.loads(self.evidence_path.read_text())
        evidence["base_revision"] = "wrong"
        self.evidence_path.write_text(json.dumps(evidence))
        report = self.validate()
        self.assertFalse(report["valid"])
        self.assertTrue(any("evidence SHA-256" in error for error in report["errors"]))

    def test_rejects_missing_b1_reference(self):
        report = validate_run(
            self.root, self.manifest, self.config, "C",
            adapter_evidence_path=self.evidence_path,
        )
        self.assertFalse(report["valid"])
        self.assertTrue(any("B1 run_config" in error for error in report["errors"]))

    def test_full_base_snapshot_hash_rejects_tampering(self):
        model = self.root / "model"
        model.mkdir()
        (model / "config.json").write_bytes(b"pinned model file")
        manifest_path = self.root / "phase3/configs/snapshot.json"
        manifest_path.parent.mkdir(parents=True)
        manifest_path.write_text(json.dumps({
            "repository": self.config["model"]["repo_id"],
            "revision": self.config["model"]["revision"],
            "file_count": 1,
            "total_bytes": (model / "config.json").stat().st_size,
            "files": [{
                "path": "config.json", "size": (model / "config.json").stat().st_size,
                "sha256": sha256_file(model / "config.json"),
            }],
        }))
        (manifest_path.parent / "source-lock.json").write_text(json.dumps({
            "model": {
                "repository": self.config["model"]["repo_id"],
                "revision": self.config["model"]["revision"],
                "weight_format": self.config["model"]["weight_format"],
                "snapshot_manifest": "phase3/configs/snapshot.json",
                "snapshot_manifest_sha256": sha256_file(manifest_path),
            }
        }))
        receipt = _verify_base_snapshot(model, self.root, self.config["model"])
        self.assertEqual(receipt["verified_file_count"], 1)
        (model / "config.json").write_bytes(b"changed file")
        with self.assertRaisesRegex(ValueError, "snapshot file differs"):
            _verify_base_snapshot(model, self.root, self.config["model"])

    def test_peft_load_preserves_cpu_and_cuda_sampling_states(self):
        class State:
            def __init__(self, value):
                self.value = value

            def clone(self):
                return State(self.value)

        class Cuda:
            def __init__(self):
                self.states = [State(20), State(30)]

            def get_rng_state_all(self):
                return self.states

            def set_rng_state_all(self, states):
                self.states = states

        class Torch:
            def __init__(self):
                self.cpu = State(10)
                self.cuda = Cuda()

            def get_rng_state(self):
                return self.cpu

            def set_rng_state(self, state):
                self.cpu = state

        torch = Torch()

        class Model:
            def eval(self):
                return self

        class Peft:
            should_fail = False

            @classmethod
            def from_pretrained(cls, base, path, is_trainable):
                self.assertEqual(path, str(self.adapter))
                self.assertFalse(is_trainable)
                torch.cpu.value = 101
                torch.cuda.states[0].value = 202
                torch.cuda.states[1].value = 303
                if cls.should_fail:
                    raise RuntimeError("adapter loading failed")
                return Model()

        loaded = _load_adapter_preserving_rng(
            Model(), self.adapter, "frozen generation",
            torch_module=torch, peft_model_class=Peft,
        )
        self.assertEqual(loaded.generation_config, "frozen generation")
        self.assertEqual(torch.cpu.value, 10)
        self.assertEqual([state.value for state in torch.cuda.states], [20, 30])

        Peft.should_fail = True
        with self.assertRaisesRegex(RuntimeError, "adapter loading failed"):
            _load_adapter_preserving_rng(
                Model(), self.adapter, "frozen generation",
                torch_module=torch, peft_model_class=Peft,
            )
        self.assertEqual(torch.cpu.value, 10)
        self.assertEqual([state.value for state in torch.cuda.states], [20, 30])


if __name__ == "__main__":
    unittest.main()
