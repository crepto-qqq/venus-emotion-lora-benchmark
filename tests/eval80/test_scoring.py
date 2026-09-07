import unittest
from collections import Counter

from src.eval80.core import EMOTION_CLASSES
from src.eval80.scoring import (
    GUIDANCE_COLUMNS,
    REVIEWER_IDS,
    _reviewer_assignments,
    parse_predicted_emotion,
    summarize_guidance_scores,
    summarize_recognition,
)


class Eval80ScoringTests(unittest.TestCase):
    def labels(self):
        return {
            f"E{index:03d}": EMOTION_CLASSES[(index - 1) // 10]
            for index in range(1, 81)
        }

    def test_parses_structured_matching_fields(self):
        parsed = parse_predicted_emotion(
            "Emotion: awe\n\nVisual evidence:\n- sky\n\nFinal emotion: awe",
            structured=True,
        )
        self.assertEqual(parsed.status, "valid")
        self.assertEqual(parsed.predicted_emotion, "awe")

    def test_reviewer_assignment_is_fixed_balanced_and_class_stratified(self):
        selection = {
            f"E{index:03d}": {
                "emotion": EMOTION_CLASSES[(index - 1) // 10],
            }
            for index in range(1, 81)
        }
        assignments = _reviewer_assignments(selection, seed=1234)
        self.assertEqual(assignments, _reviewer_assignments(selection, seed=1234))
        self.assertEqual(
            [Counter(assignments.values())[reviewer_id] for reviewer_id in REVIEWER_IDS],
            [14, 14, 13, 13, 13, 13],
        )
        for emotion in EMOTION_CLASSES:
            class_counts = Counter(
                assignments[blind_id]
                for blind_id, record in selection.items()
                if record["emotion"] == emotion
            )
            self.assertEqual(sorted(class_counts.values()), [1, 1, 2, 2, 2, 2])

    def test_rejects_inconsistent_structured_fields(self):
        parsed = parse_predicted_emotion(
            "Emotion: awe\nFinal emotion: fear",
            structured=True,
        )
        self.assertEqual(parsed.status, "ambiguous")

    def test_parses_unique_explicit_free_form_label(self):
        parsed = parse_predicted_emotion(
            "The primary emotion is contentment. The lighting feels calm.",
            structured=False,
        )
        self.assertEqual(parsed.predicted_emotion, "contentment")

    def test_summarizes_recognition_on_zero_to_ten_scale(self):
        labels = self.labels()
        outputs = []
        for blind_id, target in labels.items():
            predicted = "anger" if blind_id == "E001" else target
            outputs.append(
                {"blind_id": blind_id, "response": f"Primary emotion: {predicted}"}
            )
        _, summary = summarize_recognition(outputs, labels, structured=False)
        self.assertEqual(summary["correct_count"], 79)
        self.assertEqual(summary["emotion_recognition_score_0_10"], 9.875)

    def guidance_rows(self):
        rows = []
        reviewer_by_image = {}
        next_image = 1
        for reviewer_index, reviewer_id in enumerate(REVIEWER_IDS):
            image_count = 14 if reviewer_index < 2 else 13
            for image_index in range(next_image, next_image + image_count):
                reviewer_by_image[f"E{image_index:03d}"] = reviewer_id
            next_image += image_count
        for index in range(1, 81):
            blind_id = f"E{index:03d}"
            emotion = EMOTION_CLASSES[(index - 1) // 10]
            for condition in ("A", "B0", "B1", "C"):
                row = {
                    "anonymous_item_id": f"X-{blind_id}-{condition}",
                    "reviewer_id": reviewer_by_image[blind_id],
                    "blind_id": blind_id,
                    "condition": condition,
                    "target_emotion": emotion,
                    "overall_rationale": "Specific visible evidence and usable guidance.",
                    "zero_score_reasons": "",
                    "failure_flags": "",
                }
                row.update({column: "2" for column in GUIDANCE_COLUMNS})
                rows.append(row)
        return rows

    def test_summarizes_all_320_guidance_rows(self):
        scored, summary = summarize_guidance_scores(self.guidance_rows())
        self.assertEqual(len(scored), 320)
        self.assertTrue(all(row["guidance_total_0_10"] == 10 for row in scored))
        self.assertEqual(summary["conditions"]["A"]["mean_guidance_total_0_10"], 10)
        self.assertEqual(summary["designated_reviewer_count"], 6)
        self.assertEqual(
            list(summary["reviewer_image_counts"].values()),
            [14, 14, 13, 13, 13, 13],
        )
        self.assertFalse(summary["inter_rater_reliability_measured"])

    def test_requires_reason_for_zero_score(self):
        rows = self.guidance_rows()
        rows[0][GUIDANCE_COLUMNS[0]] = "0"
        with self.assertRaisesRegex(ValueError, "zero_score_reasons is required"):
            summarize_guidance_scores(rows)

    def test_requires_same_reviewer_across_conditions_for_an_image(self):
        rows = self.guidance_rows()
        rows[0]["reviewer_id"] = "team-member-06"
        with self.assertRaisesRegex(ValueError, "must use one reviewer"):
            summarize_guidance_scores(rows)


if __name__ == "__main__":
    unittest.main()
