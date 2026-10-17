"""Aggregate-only checks for the amended final automated benchmark."""

import copy
import unittest

from src.eval80.preference_final_overview import _markdown, build_final_overview


H = "a" * 64


class FinalPreferenceOverviewTests(unittest.TestCase):
    def _inputs(self):
        status = {
            "evaluation_set": "Eval80-v1", "comparison": "B1_vs_C_adapter_effect",
            "provenance": {"source_records_sha256": {"B1": H, "C": "b" * 64},
                           "selection_manifest_sha256": "c" * 64},
            "format": {"B1_pass_count": 0, "C_pass_count": 80, "item_count": 80},
            "strict_recognition": {"B1_correct_count": 0, "C_correct_count": 57,
                                   "item_count": 80, "b1_schema_caveat": "Missing field"},
            "opening_emotion_diagnostic": {
                "B1_correct_count": 35, "C_correct_count": 57, "item_count": 80,
                "net_correct_gain_c_minus_b1": 22, "protocol_metric": False},
        }
        guidance = {
            "evaluation_set": "Eval80-v1",
            "status": "final_automated_evaluation_with_technical_exclusions",
            "metric": "calibrated_image_aware_pairwise_guidance_preference",
            "source_records_sha256": status["provenance"]["source_records_sha256"],
            "selection_manifest_sha256": "c" * 64,
            "inference_manifest_sha256": H,
            "preference_config_sha256": H,
            "model_digest_sha256": H,
            "calibration_gate_sha256": H,
            "decision_lock_sha256": H,
            "judge_model_id": "gemma-test:1",
            "original_scope_images": 78, "included_images": 77,
            "included_judgements": 154, "technical_excluded_images": 1,
            "attempt002_valid_decisions_inherited": 149,
            "attempt002_decisions_used_after_exclusion": 148,
            "new_decisions_used_after_exclusion": 6,
            "recovery_request_attempts": 6,
            "calibration": {"pair_count": 16, "human_clear_count": 15,
                            "clear_agreement_count": 13,
                            "clear_agreement_rate": 13 / 15,
                            "hallucination_false_winners": 0},
            "technical_excluded_anonymous_positions": [26, 150],
            "technical_incident_evidence_sha256": {
                "attempt002_attempts": H, "attempt002_raw": H,
                "recovery_attempts": H},
            "new_technical_error_types": {},
            "recovery_error_evidence": [],
            "image_consensus": {"C_wins": 43, "B1_wins": 20, "ties": 10,
                                "swap_disagreements_unstable": 4},
            "observed_net_c_minus_b1_count": 23,
            "observed_net_c_minus_b1_rate": 23 / 77,
            "original_scope_worst_case_net_count": 18,
            "original_scope_worst_case_net_rate": 18 / 78,
            "original_scope_best_case_net_rate": 28 / 78,
            "item_votes": {"C": 86, "B1": 40, "tie": 28},
            "emotion_coverage": {"awe": {"included": 77, "excluded": 1}},
        }
        return status, guidance

    def test_fixed_denominator_bound_and_independent_axes(self):
        status, guidance = self._inputs()
        report = build_final_overview(status, guidance)
        self.assertEqual(report["guidance_preference"]["included_images"], 77)
        self.assertEqual(report["guidance_preference"]["worst_case_net_rate_original_scope"], 18 / 78)
        self.assertEqual(report["opening_emotion_diagnostic"]["net_correct_gain_c_minus_b1"], 22)
        self.assertFalse(report["acceptance"]["formal_acceptance"])

    def test_rejects_stale_sources_and_optimistic_denominator(self):
        status, guidance = self._inputs()
        stale = copy.deepcopy(guidance)
        stale["source_records_sha256"]["C"] = "d" * 64
        with self.assertRaisesRegex(ValueError, "different B1/C"):
            build_final_overview(status, stale)
        optimistic = copy.deepcopy(guidance)
        optimistic["original_scope_worst_case_net_rate"] = 23 / 77
        with self.assertRaisesRegex(ValueError, "conservative bound"):
            build_final_overview(status, optimistic)

    def test_redacted_new_timeout_is_visible_in_final_markdown(self):
        status, guidance = self._inputs()
        guidance["new_technical_error_types"] = {"OSError": 1}
        guidance["recovery_error_evidence"] = [{
            "anonymous_position": 151, "retry_attempt": 1,
            "error_type": "OSError", "finish_reason": None,
            "prompt_tokens": None, "completion_tokens": None,
            "max_tokens": 2048, "image_excluded": False,
        }]
        report = build_final_overview(status, guidance)
        self.assertIn("| 151 | 1 | OSError | 无回复 | — / 2048 | 否 |", _markdown(report))

    def test_reference_only_review_discloses_missing_human_calibration(self):
        status, guidance = self._inputs()
        guidance["metric"] = "uncalibrated_image_aware_pairwise_guidance_preference"
        guidance["calibration_mode"] = "reference_only_unverified"
        guidance["calibration_evidence_sha256"] = "e" * 64
        guidance["calibration_gate_sha256"] = None
        guidance["human_reference_csv_sha256"] = "e" * 64
        guidance["calibration"] = {
            "mode": "reference_only_unverified", "reference_csv_sha256": "e" * 64,
            "human_clear_count": None, "clear_agreement_count": None,
            "clear_agreement_rate": None, "hallucination_false_winners": None,
        }
        report = build_final_overview(status, guidance)
        markdown = _markdown(report)
        self.assertIn("人工校准未完成", markdown)
        self.assertIn("人类感知的 guidance 质量提升尚未验证", markdown)
        self.assertIsNone(report["provenance"]["calibration_gate_sha256"])
        false_claim = copy.deepcopy(guidance)
        false_claim["calibration"]["clear_agreement_count"] = 10
        with self.assertRaisesRegex(ValueError, "cannot claim human agreement"):
            build_final_overview(status, false_claim)


if __name__ == "__main__":
    unittest.main()
