"""Offline checks for the human-locked Gemma calibration gate."""

import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from phase3.tools.prepare_pre_c_pairwise_calibration import bilingualize, lock, prepare
from src.eval80.core import sha256_file
from src.eval80.preference_calibration import (
    _load_human_pairs, _measure, run_calibration, verify_calibration_gate,
)


ROOT = Path(__file__).resolve().parents[2]


class CalibrationGateTests(unittest.TestCase):
    def _fixture(self, root: Path):
        images = root / "review" / "images"
        images.mkdir(parents=True)
        source = images.parent / "review.csv"
        with source.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=[
                "anonymous_item_id", "image_code", "image_file", "target_emotion", "model_response"])
            writer.writeheader()
            for index in range(1, 17):
                code = f"D{index:03d}"
                (images / f"{code}.jpg").write_bytes(f"image-{index}".encode())
                for letter in ("A", "B"):
                    writer.writerow({
                        "anonymous_item_id": f"{code}-{letter}", "image_code": code,
                        "image_file": f"images/{code}.jpg", "target_emotion": "awe",
                        "model_response": f"Guidance {letter} for {code}",
                    })
        packet = root / "pairwise-calibration"
        prepare(source, packet)
        with source.open("r", encoding="utf-8-sig", newline="") as handle:
            source_rows = list(csv.DictReader(handle))
        (packet / "answer_translations_zh.json").write_text(json.dumps({
            "kind": "blind_answer_translations_zh",
            "version": 1,
            "source_review_csv_sha256": sha256_file(source),
            "records": [{
                "anonymous_item_id": row["anonymous_item_id"],
                "source_response_sha256": hashlib.sha256(
                    row["model_response"].encode("utf-8")).hexdigest(),
                "translation_zh": f"中文建议 {row['anonymous_item_id']}",
            } for row in source_rows],
        }, ensure_ascii=False), encoding="utf-8")
        bilingualize(packet, source)
        submission = root / "submission.csv"
        with submission.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=[
                "image_code", "reviewer_id", "preferred",
                "obvious_hallucinated_worse", "reason"])
            writer.writeheader()
            for index in range(1, 17):
                writer.writerow({
                    "image_code": f"D{index:03d}", "reviewer_id": "reviewer",
                    "preferred": "A", "obvious_hallucinated_worse": "B" if index == 1 else "none",
                    "reason": "Answer A has a concrete edit.",
                })
        lock(packet, source, submission)
        source_package = root / "attempt-002"
        source_package.mkdir()
        config = {
            "schema_version": 1, "model_id": "gemma-test:1",
            "provider": "openai_compatible_chat",
            "endpoint": "http://127.0.0.1:11434/v1/chat/completions",
            "prompt": "Compare guidance", "image_detail": "high",
            "decoding": {"temperature": 0, "seed": 42, "max_tokens": 2048},
            "retry_policy": {"max_attempts": 2, "timeout_seconds": 120,
                             "backoff_seconds": 0},
        }
        config_path = source_package / "preference_config_private.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        (source_package / "preference_package.json").write_text(json.dumps({
            "preference_config_sha256": sha256_file(config_path),
            "judge_model_id": config["model_id"],
            "model_digest_sha256": "a" * 64,
        }), encoding="utf-8")
        return packet / "human_lock.json", source_package

    def test_gate_rechecks_32_raw_responses_and_rejects_mutation(self):
        (ROOT / "tmp").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as name:
            root = Path(name)
            lock_path, source_package = self._fixture(root)
            lock_data, manifest, pairs = _load_human_pairs(lock_path)
            self.assertEqual(len(pairs), 16)
            counter = 0

            def fake_post(endpoint, payload, api_key, timeout):
                nonlocal counter
                counter += 1
                if counter == 1:
                    raise TimeoutError("synthetic local socket timeout")
                text = payload["messages"][0]["content"][1]["text"]
                winner = "A" if "Answer A:\nGuidance A" in text else "B"
                decision = {key: winner for key in (
                    "preferred", "better_grounding", "better_actionability", "better_target_fit")}
                decision["rationale"] = "A visible, actionable edit."
                return {"id": f"response-{counter}", "model": "gemma-test:1",
                        "choices": [{"finish_reason": "stop", "message": {
                            "content": json.dumps(decision)}}]}

            def fake_probe(endpoint, model, digest, timeout, *, phase):
                return {"phase": phase, "observed_at_utc": "2026-10-17T00:00:00Z",
                        "model_id": model, "digest_sha256": digest}

            output = root / "calibration-run"
            with patch("src.eval80.preference_calibration._post_chat", side_effect=fake_post), \
                    patch("src.eval80.preference_calibration._probe_ollama_digest",
                          side_effect=fake_probe):
                gate = run_calibration(lock_path, source_package, output)
            self.assertEqual(counter, 33)
            self.assertEqual(gate["status"], "passed")
            self.assertEqual(gate["clear_agreement_count"], 16)
            verify_calibration_gate(output / "calibration_gate.json")
            attempts_path = output / "calibration_attempts_private.jsonl"
            original_attempts = attempts_path.read_bytes()
            original_gate = (output / "calibration_gate.json").read_bytes()
            attempts = [json.loads(line) for line in original_attempts.splitlines()]
            target_index = next(i for i, row in enumerate(attempts)
                                if row["image_code"] == "D002" and row["orientation"] == "normal")
            target = attempts[target_index]
            forged_invalid = {**target, "result": "invalid", "error_type": "ValueError"}
            forged_accepted = {**target, "attempt": 2,
                               "raw_file": "raw/D002_normal_attempt_2.json"}
            original_raw = output / "raw" / "D002_normal_attempt_1.json"
            (output / forged_accepted["raw_file"]).write_bytes(original_raw.read_bytes())
            decisions_path = output / "calibration_decisions_private.jsonl"
            original_decisions = decisions_path.read_bytes()
            decisions = [json.loads(line) for line in original_decisions.splitlines()]
            decision_index = next(i for i, row in enumerate(decisions)
                                  if row["image_code"] == "D002" and row["orientation"] == "normal")
            decisions[decision_index]["accepted_raw_file"] = forged_accepted["raw_file"]
            attempts_path.write_text("".join(json.dumps(row) + "\n" for row in (
                *attempts[:target_index], forged_invalid, forged_accepted,
                *attempts[target_index + 1:])), encoding="utf-8")
            decisions_path.write_text("".join(json.dumps(row) + "\n" for row in decisions),
                                      encoding="utf-8")
            forged_gate = json.loads(original_gate)
            forged_gate["attempts_sha256"] = sha256_file(attempts_path)
            forged_gate["decisions_sha256"] = sha256_file(decisions_path)
            (output / "calibration_gate.json").write_text(json.dumps(forged_gate), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "labeled a valid reply as invalid"):
                verify_calibration_gate(output / "calibration_gate.json")
            attempts_path.write_bytes(original_attempts)
            decisions_path.write_bytes(original_decisions)
            (output / "calibration_gate.json").write_bytes(original_gate)
            raw = output / "raw" / "D001_normal_attempt_2.json"
            changed = json.loads(raw.read_text(encoding="utf-8"))
            changed["id"] = "altered"
            raw.write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "raw response changed"):
                verify_calibration_gate(output / "calibration_gate.json")

    def test_unstable_or_hallucination_false_winner_fails_gate(self):
        pairs = {f"D{i:03d}": {"human": {
            "preferred": "A", "obvious_hallucinated_worse": "B" if i == 1 else "none"}}
            for i in range(1, 17)}
        decisions = [{"image_code": code, "orientation": orientation,
                      "winner": "A" if orientation == "normal" else "B"}
                     for code in pairs for orientation in ("normal", "swapped")]
        self.assertEqual(_measure(pairs, decisions)["status"], "passed")
        decisions[0]["winner"] = "B"
        decisions[1]["winner"] = "A"
        self.assertEqual(_measure(pairs, decisions)["hallucination_false_winners"], 1)
        self.assertEqual(_measure(pairs, decisions)["status"], "failed")

    def test_locked_human_pairs_reject_changed_translation(self):
        (ROOT / "tmp").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as name:
            lock_path, _ = self._fixture(Path(name))
            _, manifest, pairs = _load_human_pairs(lock_path)
            self.assertEqual(manifest["version"], 2)
            self.assertEqual(len(pairs), 16)
            translations_path = lock_path.parent / "answer_translations_zh.json"
            translations = json.loads(translations_path.read_text(encoding="utf-8"))
            translations["records"][0]["translation_zh"] += " 已更改"
            translations_path.write_text(json.dumps(translations, ensure_ascii=False),
                                         encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "lock/manifest identity"):
                _load_human_pairs(lock_path)


if __name__ == "__main__":
    unittest.main()
