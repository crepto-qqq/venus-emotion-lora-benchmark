import csv
from html.parser import HTMLParser
import json
from pathlib import Path
import sys
import tempfile
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from prepare_pre_c_pairwise_calibration import (
    ROOT, bilingualize, lock, prepare, sha256_file, sha256_text,
)


class RadioGroups(HTMLParser):
    def __init__(self):
        super().__init__()
        self.card = None
        self.groups = {}

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "article" and "case" in attrs.get("class", "").split():
            self.card = attrs["data-code"]
            self.groups[self.card] = {}
        elif tag == "input" and attrs.get("type") == "radio" and self.card:
            self.groups[self.card].setdefault(attrs["name"], set()).add(attrs["value"])

    def handle_endtag(self, tag):
        if tag == "article":
            self.card = None


class AnswerBlocks(HTMLParser):
    def __init__(self):
        super().__init__()
        self.counts = {"zh-CN": 0, "en": 0}
        self.answer_sections = 0

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "section" and "answer" in attrs.get("class", "").split():
            self.answer_sections += 1
        if tag == "pre" and attrs.get("lang") in self.counts:
            self.counts[attrs["lang"]] += 1


class PairwiseHumanCalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (ROOT / "tmp").mkdir(exist_ok=True)

    def fixture(self, temporary: Path):
        review = temporary / "review"
        images = review / "images"
        images.mkdir(parents=True)
        source = review / "review.csv"
        with source.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=(
                "anonymous_item_id", "image_code", "image_file", "target_emotion", "model_response"
            ))
            writer.writeheader()
            for index in range(1, 17):
                code = f"D{index:03d}"
                (images / f"{code}.jpg").write_bytes(f"test-image-{index}".encode())
                for letter in ("A", "B"):
                    writer.writerow({
                        "anonymous_item_id": f"{code}-{letter}", "image_code": code,
                        "image_file": f"images/{code}.jpg", "target_emotion": "awe",
                        "model_response": f"Guidance {letter} for {code} </script>",
                    })
        packet = temporary / "pairwise-calibration"
        return source, packet

    def submission(self, temporary: Path, *, bad_hallucination=False, incomplete=False) -> Path:
        path = temporary / "submission.csv"
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=(
                "image_code", "reviewer_id", "preferred", "obvious_hallucinated_worse", "reason"
            ))
            writer.writeheader()
            for index in range(1, 17):
                writer.writerow({
                    "image_code": f"D{index:03d}", "reviewer_id": "human",
                    "preferred": "A" if index != 16 else "tie",
                    "obvious_hallucinated_worse": (
                        "A" if bad_hallucination and index == 1 else
                        "B" if index == 2 else "none"
                    ),
                    "reason": "" if incomplete and index == 1 else "More grounded and actionable.",
                })
        return path

    def translations(self, source: Path, packet: Path) -> Path:
        with source.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        path = packet / "answer_translations_zh.json"
        path.write_text(json.dumps({
            "kind": "blind_answer_translations_zh",
            "version": 1,
            "source_review_csv_sha256": sha256_file(source),
            "records": [{
                "anonymous_item_id": row["anonymous_item_id"],
                "source_response_sha256": sha256_text(row["model_response"]),
                "translation_zh": f"{row['anonymous_item_id']} 的中文建议",
            } for row in rows],
        }, ensure_ascii=False), encoding="utf-8")
        return path

    def test_packet_has_blind_manifest_and_locks_complete_submission(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as name:
            temporary = Path(name)
            source, packet = self.fixture(temporary)
            prepared = prepare(source, packet)
            manifest = json.loads((packet / "manifest.json").read_text(encoding="utf-8"))
            html = (packet / "review.html").read_text(encoding="utf-8")
            self.assertEqual(manifest["image_count"], 16)
            self.assertEqual(prepared["manifest_sha256"], sha256_file(packet / "manifest.json"))
            self.assertEqual(manifest["source_review_csv_sha256"], sha256_file(source))
            self.assertIn("&lt;/script&gt;", html)
            self.assertNotIn("answer_key.json", json.dumps(manifest))
            self.assertNotIn("C_checkpoint_180", html)
            submission = self.submission(temporary)
            locked = lock(packet, source, submission)
            result = json.loads((packet / "human_lock.json").read_text(encoding="utf-8"))
            self.assertEqual(locked["clear_preference_count"], 15)
            self.assertEqual(result["manifest_sha256"], sha256_file(packet / "manifest.json"))
            self.assertEqual(result["review_html_sha256"], sha256_file(packet / "review.html"))
            self.assertEqual(result["submission_sha256"], sha256_file(submission))
            self.assertEqual(result["submission_file"], "human_submission.csv")
            self.assertEqual(result["submission_sha256"], sha256_file(packet / "human_submission.csv"))
            self.assertEqual(len(result["records"]), 16)
            with self.assertRaises(FileExistsError):
                lock(packet, source, submission)

    def test_rejects_incomplete_or_contradictory_human_choices(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as name:
            temporary = Path(name)
            source, packet = self.fixture(temporary)
            prepare(source, packet)
            submission = self.submission(temporary, incomplete=True)
            with self.assertRaisesRegex(ValueError, "Incomplete"):
                lock(packet, source, submission)
            submission = self.submission(temporary, bad_hallucination=True)
            with self.assertRaisesRegex(ValueError, "Contradictory"):
                lock(packet, source, submission)
            self.assertFalse((packet / "human_lock.json").exists())

    def test_rejects_answer_change_after_packet_freeze(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as name:
            temporary = Path(name)
            source, packet = self.fixture(temporary)
            prepare(source, packet)
            text = source.read_text(encoding="utf-8-sig")
            source.write_text(text.replace("Guidance A", "Altered A"), encoding="utf-8-sig")
            with self.assertRaisesRegex(ValueError, "changed"):
                lock(packet, source, self.submission(temporary))

    def test_radio_clicks_on_two_images_do_not_cancel_each_other(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as name:
            temporary = Path(name)
            source, packet = self.fixture(temporary)
            prepare(source, packet)
            parser = RadioGroups()
            parser.feed((packet / "review.html").read_text(encoding="utf-8"))
            self.assertEqual(len(parser.groups), 16)
            first, second = "D001", "D002"
            for code in (first, second):
                self.assertEqual(parser.groups[code][f"preferred-{code}"], {"A", "B", "tie"})
                self.assertEqual(parser.groups[code][f"obvious_hallucinated_worse-{code}"],
                                 {"A", "B", "none", "uncertain"})
            # Native radio clicks deselect only controls with the same name.
            selected = {}
            for code, preferred in ((first, "A"), (second, "B")):
                name = f"preferred-{code}"
                self.assertIn(preferred, parser.groups[code][name])
                selected[name] = preferred
            self.assertEqual(selected, {"preferred-D001": "A", "preferred-D002": "B"})
            self.assertTrue(set(parser.groups[first]).isdisjoint(parser.groups[second]))

    def test_rejects_modified_form_before_lock(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as name:
            temporary = Path(name)
            source, packet = self.fixture(temporary)
            prepare(source, packet)
            form = packet / "review.html"
            form.write_text(form.read_text(encoding="utf-8").replace("A 更好", "A?"), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Reviewer HTML changed"):
                lock(packet, source, self.submission(temporary))

    def test_bilingual_form_shows_all_answers_and_binds_translation_and_html(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as name:
            temporary = Path(name)
            source, packet = self.fixture(temporary)
            prepare(source, packet)
            original_manifest = json.loads((packet / "manifest.json").read_text(encoding="utf-8"))
            translations_path = self.translations(source, packet)
            upgraded = bilingualize(packet, source)
            manifest = json.loads((packet / "manifest.json").read_text(encoding="utf-8"))
            html = (packet / "review.html").read_text(encoding="utf-8")
            self.assertEqual(manifest["version"], 2)
            self.assertEqual(manifest["records"], original_manifest["records"])
            self.assertEqual(manifest["answer_translations_zh_sha256"], sha256_file(translations_path))
            self.assertEqual(manifest["review_html_sha256"], sha256_text(html))
            self.assertEqual(upgraded["manifest_sha256"], sha256_file(packet / "manifest.json"))
            parser = AnswerBlocks()
            parser.feed(html)
            self.assertEqual(parser.answer_sections, 32)
            self.assertEqual(parser.counts, {"zh-CN": 32, "en": 32})
            for index in range(1, 17):
                for letter in ("A", "B"):
                    self.assertIn(f"D{index:03d}-{letter} 的中文建议", html)
                    self.assertIn(f"Guidance {letter} for D{index:03d} &lt;/script&gt;", html)
            self.assertIn("敬畏／惊叹（awe）", html)
            self.assertNotIn("C_checkpoint_180", html)
            radios = RadioGroups()
            radios.feed(html)
            self.assertEqual(len(radios.groups), 16)
            self.assertTrue(set(radios.groups["D001"]).isdisjoint(radios.groups["D002"]))
            form = packet / "review.html"
            form.write_text(html.replace("D001-A 的中文建议", "更改后的建议"), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Reviewer HTML changed"):
                lock(packet, source, self.submission(temporary))
            with form.open("w", encoding="utf-8", newline="\n") as handle:
                handle.write(html)
            locked = lock(packet, source, self.submission(temporary))
            result = json.loads((packet / "human_lock.json").read_text(encoding="utf-8"))
            self.assertEqual(result["answer_translations_zh_sha256"], sha256_file(translations_path))
            self.assertEqual(result["review_html_sha256"], manifest["review_html_sha256"])
            with self.assertRaises(FileExistsError):
                bilingualize(packet, source)

    def test_bilingual_form_rejects_translation_tampering_or_wrong_answer_binding(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as name:
            temporary = Path(name)
            source, packet = self.fixture(temporary)
            prepare(source, packet)
            path = self.translations(source, packet)
            original = json.loads(path.read_text(encoding="utf-8"))
            original["records"][0]["source_response_sha256"] = "0" * 64
            path.write_text(json.dumps(original), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "does not match blinded answer"):
                bilingualize(packet, source)
            original["records"][0]["source_response_sha256"] = sha256_text(
                "Guidance A for D001 </script>")
            path.write_text(json.dumps(original), encoding="utf-8")
            bilingualize(packet, source)
            original["records"][0]["translation_zh"] = "恶意修改"
            path.write_text(json.dumps(original), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Chinese translations changed"):
                lock(packet, source, self.submission(temporary))
            self.assertFalse((packet / "human_lock.json").exists())


if __name__ == "__main__":
    unittest.main()
