import unittest

from src.phase1.summarize_scores import SCORE_COLUMNS, validate_and_summarize


class ScoreSummaryTests(unittest.TestCase):
    def rows(self, stage="stage1"):
        columns = SCORE_COLUMNS[stage]
        rows = []
        for index in range(20):
            row = {
                "blind_item_id": f"ITEM-{index + 1:03d}",
                "sample_id": f"E{index + 1:02d}",
                "target_emotion": "amusement" if index < 10 else "sadness",
                "notes": "",
            }
            row.update({column: "2" for column in columns})
            rows.append(row)
        return rows

    def test_summarizes_valid_scores(self):
        scored_rows, summary = validate_and_summarize(self.rows(), "stage1")
        self.assertEqual(len(scored_rows), 20)
        self.assertEqual(summary["mean_total_0_6"], 6)
        self.assertTrue(all(row["total_0_6"] == 6 for row in scored_rows))

    def test_rejects_missing_or_out_of_range_score(self):
        rows = self.rows("stage2")
        rows[0][SCORE_COLUMNS["stage2"][0]] = "3"
        with self.assertRaisesRegex(ValueError, "exactly 0, 1, or 2"):
            validate_and_summarize(rows, "stage2")

    def test_rejects_duplicate_sample(self):
        rows = self.rows()
        rows[1]["sample_id"] = rows[0]["sample_id"]
        with self.assertRaisesRegex(ValueError, "duplicate sample_id"):
            validate_and_summarize(rows, "stage1")


if __name__ == "__main__":
    unittest.main()
