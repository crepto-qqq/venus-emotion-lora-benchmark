from __future__ import annotations

from collections import Counter
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from src.datasets.bridge_train import (
    BridgeBuildConfig,
    BridgeBuildError,
    build_bridge_release,
    verify_release_dir,
    write_release,
)


CLASSIFICATION_PROMPT = (
    "Which primary emotion does this image convey? Choose exactly one label "
    "from: amusement, anger, awe, contentment, disgust, excitement, fear, or "
    "sadness. Respond with the label only."
)

JOINT_PROMPT = """You are an emotion-aware image-aesthetics assistant.

Use only visible evidence from the image. Choose exactly one primary emotion from: amusement, anger, awe, contentment, disgust, excitement, fear, or sadness.

Use exactly this format:

Emotion: <one allowed label>

Visual evidence:
- <specific visible cue>
- <specific visible cue>

Aesthetic relationship:
<clear explanation>

Guidance:
- <concrete action>
- <concrete action>

Final emotion: <the same allowed label>"""


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _target(emotion: str) -> str:
    return (
        f"Emotion: {emotion}\n\n"
        "Visual evidence:\n"
        "- The subject has a clearly visible expression.\n"
        "- The lighting and composition reinforce the mood.\n\n"
        "Aesthetic relationship:\n"
        "The composition directs attention to the visible emotional cue.\n\n"
        "Guidance:\n"
        "- Reduce background distractions around the subject.\n"
        "- Preserve the lighting that supports the mood.\n\n"
        f"Final emotion: {emotion}"
    )


class _SyntheticBridgeData:
    def __init__(
        self,
        root: Path,
        *,
        emotions: tuple[str, ...] = ("amusement", "anger"),
        train_per_class: int = 2,
        validation_per_class: int = 1,
    ) -> None:
        self.root = root
        self.source_root = root / "records"
        self.archive_path = root / "EmoSet-test.zip"
        self.emotions = emotions
        self.train_per_class = train_per_class
        self.validation_per_class = validation_per_class
        self.records: dict[str, dict] = {}
        self.image_bytes: dict[str, bytes] = {}
        self.annotation_bytes: dict[str, bytes] = {}
        self._write()

    def _write(self) -> None:
        with zipfile.ZipFile(self.archive_path, "w") as archive:
            for emotion_index, emotion in enumerate(self.emotions):
                split_records: dict[str, list[dict]] = {
                    "train": [],
                    "validation": [],
                }
                split_counts = {
                    "train": self.train_per_class,
                    "validation": self.validation_per_class,
                }
                sequence = 5000 + emotion_index * 100
                for split, count in split_counts.items():
                    for offset in range(count):
                        source_id = f"{emotion}_{sequence + offset:05d}"
                        image_relpath = f"image/{emotion}/{source_id}.jpg"
                        annotation_relpath = (
                            f"annotation/{emotion}/{source_id}.json"
                        )
                        image_bytes = f"image:{source_id}".encode("utf-8")
                        annotation_bytes = json.dumps(
                            {"emotion": emotion, "image_id": source_id},
                            sort_keys=True,
                        ).encode("utf-8")
                        archive.writestr(image_relpath, image_bytes)
                        archive.writestr(annotation_relpath, annotation_bytes)
                        record = {
                            "source_image_id": source_id,
                            "emotion": emotion,
                            "image_relpath": image_relpath,
                            # This deliberately leaks the fixed label. The
                            # converter must treat it as annotation metadata,
                            # never as a training prompt.
                            "instruction": (
                                "The assigned EmoSet ground-truth emotion for "
                                f"this image is: {emotion}"
                            ),
                            "target_response": _target(emotion),
                            "split": split,
                            "provenance": {
                                "source": "EmoSet-118K",
                                "image_sha256": _sha256(image_bytes),
                                "annotation_sha256": _sha256(annotation_bytes),
                                "generator_provider": "synthetic-test",
                                "generator_model": "synthetic-test",
                                "prompt_version": "synthetic-test-v1",
                                "temperature": 0,
                                "seed": 3888,
                                "label_supported": True,
                                "review_status": "accepted",
                            },
                        }
                        split_records[split].append(record)
                        self.records[source_id] = record
                        self.image_bytes[source_id] = image_bytes
                        self.annotation_bytes[source_id] = annotation_bytes
                    sequence += count

                emotion_dir = self.source_root / emotion
                emotion_dir.mkdir(parents=True, exist_ok=True)
                for split, records in split_records.items():
                    path = emotion_dir / f"{split}_{emotion}.jsonl"
                    self._write_jsonl(path, records)

    @staticmethod
    def _write_jsonl(path: Path, records: list[dict]) -> None:
        path.write_text(
            "".join(
                json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
                for record in records
            ),
            encoding="utf-8",
        )

    def rewrite_record(self, source_id: str) -> None:
        record = self.records[source_id]
        emotion = source_id.split("_", 1)[0]
        split = record["split"]
        path = self.source_root / emotion / f"{split}_{emotion}.jsonl"
        records = [
            item
            for item in self.records.values()
            if item["emotion"] == emotion and item["split"] == split
        ]
        self._write_jsonl(path, records)


class BridgeTrainTests(unittest.TestCase):
    def _config(
        self,
        *,
        emotions: tuple[str, ...] = ("amusement", "anger"),
        train_images: int = 2,
        validation_images: int = 1,
        train_guidance: int = 1,
        validation_guidance: int = 1,
    ) -> BridgeBuildConfig:
        return BridgeBuildConfig(
            emotions=emotions,
            images_per_class_by_split={
                "train": train_images,
                "validation": validation_images,
            },
            guidance_per_class_by_split={
                "train": train_guidance,
                "validation": validation_guidance,
            },
            classification_prompt=CLASSIFICATION_PROMPT,
            classification_prompt_version="bridge-classification-v1",
            joint_prompt=JOINT_PROMPT,
            joint_prompt_version="b1-structured-joint-v1",
            selection_seed=3888,
        )

    def _build(
        self,
        data: _SyntheticBridgeData,
        *,
        config: BridgeBuildConfig | None = None,
        eval20_ids: frozenset[str] = frozenset(),
        eval20_hashes: frozenset[str] = frozenset(),
        eval80_ids: frozenset[str] = frozenset(),
        eval80_hashes: frozenset[str] = frozenset(),
    ) -> dict:
        return build_bridge_release(
            source_root=data.source_root,
            archive_path=data.archive_path,
            image_reference_root="/workspace/phase3/data/bridge-train-v1/images",
            eval20_ids=eval20_ids,
            eval20_hashes=eval20_hashes,
            eval80_ids=eval80_ids,
            eval80_hashes=eval80_hashes,
            config=config or self._config(emotions=data.emotions),
        )

    @staticmethod
    def _task_counts(manifest: list[dict]) -> Counter[tuple[str, str, str]]:
        return Counter(
            (item["split"], item["emotion"], item["task_type"])
            for item in manifest
        )

    def test_builds_deterministic_balanced_release_with_small_canonical_quotas(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            data = _SyntheticBridgeData(Path(temporary))

            release = self._build(data)
            repeated = self._build(data)

        self.assertEqual(release, repeated)
        self.assertEqual(len(release["train"]), 6)
        self.assertEqual(len(release["validation"]), 4)
        self.assertEqual(len(release["manifest"]), 10)

        expected_counts = Counter()
        for emotion in data.emotions:
            expected_counts.update(
                {
                    ("train", emotion, "classification"): 2,
                    ("train", emotion, "joint_guidance"): 1,
                    ("validation", emotion, "classification"): 1,
                    ("validation", emotion, "joint_guidance"): 1,
                }
            )
        self.assertEqual(self._task_counts(release["manifest"]), expected_counts)

        for split, expected_unique in (("train", 4), ("validation", 2)):
            rows = [row for row in release["manifest"] if row["split"] == split]
            self.assertEqual(
                len({row["source_image_id"] for row in rows}), expected_unique
            )
        self.assertEqual(
            release["summary"]["conversation_counts"],
            {"train": 6, "validation": 4},
        )
        self.assertEqual(
            release["summary"]["unique_image_counts"],
            {"train": 4, "validation": 2},
        )

        all_samples = release["train"] + release["validation"]
        self.assertEqual(len({sample["id"] for sample in all_samples}), 10)
        for sample in all_samples:
            self.assertEqual(len(sample["conversations"]), 2)
            self.assertEqual(sample["conversations"][0]["from"], "user")
            self.assertEqual(sample["conversations"][1]["from"], "assistant")
            user_value = sample["conversations"][0]["value"]
            self.assertEqual(user_value.count("<img>"), 1)
            self.assertEqual(user_value.count("</img>"), 1)
            self.assertTrue(user_value.startswith("Picture 1: <img>/workspace/"))
            self.assertTrue(sample["conversations"][1]["value"].strip())

    def test_uses_frozen_task_prompts_instead_of_label_leaking_source_instruction(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            data = _SyntheticBridgeData(Path(temporary))
            release = self._build(data)

        samples_by_id = {
            sample["id"]: sample
            for sample in release["train"] + release["validation"]
        }
        for row in release["manifest"]:
            sample = samples_by_id[row["sample_id"]]
            user_value = sample["conversations"][0]["value"]
            prompt = user_value.split("</img>\n", 1)[1]
            source_instruction = data.records[row["source_image_id"]]["instruction"]
            self.assertNotIn(source_instruction, user_value)
            if row["task_type"] == "classification":
                self.assertEqual(prompt, CLASSIFICATION_PROMPT)
                self.assertEqual(
                    sample["conversations"][1]["value"], row["emotion"]
                )
            else:
                self.assertEqual(prompt, JOINT_PROMPT)
                self.assertEqual(
                    sample["conversations"][1]["value"],
                    data.records[row["source_image_id"]]["target_response"],
                )

    def test_guidance_review_gate_skips_rejected_target_but_keeps_classification(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            data = _SyntheticBridgeData(
                Path(temporary), emotions=("amusement",), train_per_class=2
            )
            rejected_id = sorted(
                source_id
                for source_id, record in data.records.items()
                if record["split"] == "train"
            )[0]
            data.records[rejected_id]["provenance"]["review_status"] = "rejected"
            data.rewrite_record(rejected_id)

            release = self._build(
                data,
                config=self._config(emotions=("amusement",)),
            )

        rejected_tasks = {
            row["task_type"]
            for row in release["manifest"]
            if row["source_image_id"] == rejected_id
        }
        self.assertEqual(rejected_tasks, {"classification"})
        guidance_rows = [
            row
            for row in release["manifest"]
            if row["split"] == "train" and row["task_type"] == "joint_guidance"
        ]
        self.assertEqual(len(guidance_rows), 1)
        self.assertNotEqual(guidance_rows[0]["source_image_id"], rejected_id)

    def test_guidance_review_gate_fails_when_quota_cannot_be_met(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            data = _SyntheticBridgeData(Path(temporary), emotions=("amusement",))
            for source_id, record in data.records.items():
                if record["split"] == "train":
                    record["provenance"]["review_status"] = "rejected"
                    data.rewrite_record(source_id)

            with self.assertRaises(BridgeBuildError):
                self._build(
                    data,
                    config=self._config(emotions=("amusement",)),
                )

    def test_rejects_archive_hash_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            data = _SyntheticBridgeData(Path(temporary), emotions=("amusement",))
            source_id = sorted(data.records)[0]
            data.records[source_id]["provenance"]["image_sha256"] = "0" * 64
            data.rewrite_record(source_id)

            with self.assertRaises(BridgeBuildError):
                self._build(
                    data,
                    config=self._config(emotions=("amusement",)),
                )

    def test_rejects_non_normalized_or_traversing_image_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            data = _SyntheticBridgeData(Path(temporary), emotions=("amusement",))
            source_id = sorted(data.records)[0]
            data.records[source_id]["image_relpath"] = "../outside.jpg"
            data.rewrite_record(source_id)

            with self.assertRaises(BridgeBuildError):
                self._build(
                    data,
                    config=self._config(emotions=("amusement",)),
                )

    def test_rejects_eval20_or_eval80_leakage_by_id_or_hash(self) -> None:
        cases = (
            "eval20_id",
            "eval20_hash",
            "eval80_id",
            "eval80_hash",
        )
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as temporary:
                data = _SyntheticBridgeData(
                    Path(temporary), emotions=("amusement",)
                )
                source_id = sorted(data.records)[0]
                image_hash = _sha256(data.image_bytes[source_id])
                kwargs: dict[str, frozenset[str]] = {
                    "eval20_ids": frozenset(),
                    "eval20_hashes": frozenset(),
                    "eval80_ids": frozenset(),
                    "eval80_hashes": frozenset(),
                }
                kwargs[case.replace("_id", "_ids").replace("_hash", "_hashes")] = (
                    frozenset({source_id})
                    if case.endswith("_id")
                    else frozenset({image_hash})
                )

                with self.assertRaises(BridgeBuildError):
                    self._build(
                        data,
                        config=self._config(emotions=("amusement",)),
                        **kwargs,
                    )

    def test_explicit_guidance_ids_are_frozen_and_must_be_eligible(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            data = _SyntheticBridgeData(Path(temporary), emotions=("amusement",))
            train_ids = sorted(
                source_id
                for source_id, record in data.records.items()
                if record["split"] == "train"
            )
            validation_id = next(
                source_id
                for source_id, record in data.records.items()
                if record["split"] == "validation"
            )
            chosen_train_id = train_ids[-1]
            config = BridgeBuildConfig(
                emotions=("amusement",),
                images_per_class_by_split={"train": 2, "validation": 1},
                guidance_per_class_by_split={"train": 1, "validation": 1},
                classification_prompt=CLASSIFICATION_PROMPT,
                classification_prompt_version="bridge-classification-v1",
                joint_prompt=JOINT_PROMPT,
                joint_prompt_version="b1-structured-joint-v1",
                selection_seed=3888,
                guidance_source_ids_by_split={
                    "train": {"amusement": (chosen_train_id,)},
                    "validation": {"amusement": (validation_id,)},
                },
            )
            release = self._build(data, config=config)
            selected = {
                row["source_image_id"]
                for row in release["manifest"]
                if row["task_type"] == "joint_guidance"
            }
            self.assertEqual(selected, {chosen_train_id, validation_id})

            data.records[chosen_train_id]["provenance"]["review_status"] = "rejected"
            data.rewrite_record(chosen_train_id)
            with self.assertRaises(BridgeBuildError):
                self._build(data, config=config)

    def test_written_release_verifier_detects_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_root = root / "source"
            source_root.mkdir()
            data = _SyntheticBridgeData(source_root)
            config = self._config(emotions=data.emotions)
            release = self._build(data, config=config)
            release_dir = root / "release"
            write_release(release_dir, release, overwrite=False)
            report = verify_release_dir(
                release_dir=release_dir,
                config=config,
                archive_path=data.archive_path,
            )
            self.assertTrue(report["valid"])
            train_path = release_dir / "bridge-train-v1.train.json"
            train_path.write_text(train_path.read_text(encoding="utf-8") + " ", encoding="utf-8")
            with self.assertRaises(BridgeBuildError):
                verify_release_dir(release_dir=release_dir, config=config)

    def test_release_approval_rejects_signoff_with_pending_records(self) -> None:
        config = replace(
            self._config(),
            post_correction_human_signoff=True,
            pending_human_review_ids=("amusement_05000",),
        )
        with self.assertRaises(BridgeBuildError):
            config.validate()

    def test_written_release_verifier_detects_summary_status_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = _SyntheticBridgeData(root)
            config = self._config(emotions=data.emotions)
            release_dir = root / "release"
            write_release(release_dir, self._build(data, config=config), overwrite=False)
            summary_path = release_dir / "bridge-train-v1.summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["pending_human_review_record_count"] = 999
            summary_path.write_text(json.dumps(summary), encoding="utf-8")

            with self.assertRaises(BridgeBuildError):
                verify_release_dir(release_dir=release_dir, config=config)

    def test_written_release_verifier_rechecks_evaluation_exclusions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = _SyntheticBridgeData(root)
            config = self._config(emotions=data.emotions)
            release = self._build(data, config=config)
            release_dir = root / "release"
            write_release(release_dir, release, overwrite=False)
            leaked_id = release["manifest"][0]["source_image_id"]

            with self.assertRaises(BridgeBuildError):
                verify_release_dir(
                    release_dir=release_dir,
                    config=config,
                    eval20_ids={leaked_id},
                )

    def test_written_release_verifier_rechecks_source_records(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = _SyntheticBridgeData(root)
            config = self._config(emotions=data.emotions)
            release_dir = root / "release"
            write_release(release_dir, self._build(data, config=config), overwrite=False)
            changed_id = sorted(data.records)[0]
            data.records[changed_id]["instruction"] = "tampered after release"
            data.rewrite_record(changed_id)

            with self.assertRaises(BridgeBuildError):
                verify_release_dir(
                    release_dir=release_dir,
                    config=config,
                    source_root=data.source_root,
                )

    def test_written_release_verifier_enforces_frozen_guidance_rank(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = _SyntheticBridgeData(root, emotions=("amusement",))
            train_ids = sorted(
                source_id
                for source_id, record in data.records.items()
                if record["split"] == "train"
            )
            validation_id = next(
                source_id
                for source_id, record in data.records.items()
                if record["split"] == "validation"
            )
            config = replace(
                self._config(emotions=("amusement",)),
                guidance_source_ids_by_split={
                    "train": {"amusement": (train_ids[0],)},
                    "validation": {"amusement": (validation_id,)},
                },
            )
            release = self._build(data, config=config)
            guidance_row = next(
                row
                for row in release["manifest"]
                if row["task_type"] == "joint_guidance"
            )
            guidance_row["guidance_selection_rank"] = 2
            release_dir = root / "release"
            write_release(release_dir, release, overwrite=False)

            with self.assertRaises(BridgeBuildError):
                verify_release_dir(release_dir=release_dir, config=config)


if __name__ == "__main__":
    unittest.main()
