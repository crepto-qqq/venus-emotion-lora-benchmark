"""Synthetic aggregate-only contract checks for the provisional overview."""

from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.eval80.preference_overview import build_preference_overview, main


_HASHES = {
    "B1": "a" * 64,
    "C": "b" * 64,
    "inference": "c" * 64,
    "selection": "d" * 64,
    "b1_details": "e" * 64,
    "c_details": "f" * 64,
}


def _reports():
    formats = {}
    recognition = {}
    for condition, format_pass, correct, detail_key in (
        ("B1", 0, 0, "b1_details"), ("C", 80, 57, "c_details")
    ):
        formats[condition] = {
            "schema_version": 1, "evaluation_set": "Eval80-v1",
            "condition": condition,
            "metric": "auxiliary_structured_output_format_compliance",
            "item_count": 80, "format_pass_count": format_pass,
            "format_failure_count": 80 - format_pass,
            "records_sha256": _HASHES[condition],
            "inference_manifest_sha256": _HASHES["inference"],
        }
        recognition[condition] = {
            "schema_version": 2, "condition": condition, "item_count": 80,
            "correct_count": correct, "records_sha256": _HASHES[condition],
            "selection_manifest_sha256": _HASHES["selection"],
            "details_sha256": _HASHES[detail_key],
        }
    paired = {
        "schema_version": 1, "evaluation_set": "Eval80-v1",
        "comparison": "B1_vs_C_strict_emotion_recognition", "image_count": 80,
        "conditions": {"B1": {"correct_count": 0}, "C": {"correct_count": 57}},
        "net_correct_gain_c_minus_b1": 57,
        "paired_outcomes": {"both_correct": 0, "b1_only_correct": 0,
                            "c_only_correct": 57, "both_wrong": 23},
        "input_sha256": {"b1_details": _HASHES["b1_details"],
                         "c_details": _HASHES["c_details"]},
    }
    preference = {
        "schema_version": 1, "evaluation_set": "Eval80-v1",
        "status": "provisional_proxy",
        "metric": "image_aware_pairwise_guidance_preference_proxy",
        "image_count": 78, "judgement_count": 156,
        "judge_provenance": "script_runner_trace_consistent",
        "judge_model_id": "synthetic-local-vision:v1",
        "model_digest_verification_status": "runtime_ollama_before_after_matched",
        "source_records_sha256": {"B1": _HASHES["B1"], "C": _HASHES["C"]},
        "inference_manifest_sha256": _HASHES["inference"],
        "selection_manifest_sha256": _HASHES["selection"],
        "preference_config_sha256": "1" * 64,
        "image_consensus": {"C_wins": 40, "B1_wins": 20, "ties": 8,
                            "swap_disagreements_unstable": 10,
                            "identical_answer_rule_ties": 2,
                            "stable_net_c_minus_b1": 20,
                            "worst_case_net_if_all_unstable_favor_b1": 10},
        "item_votes": {"C": 80, "B1": 50, "tie": 26, "net_c_minus_b1": 30},
    }
    opening = {
        "schema_version": 1, "evaluation_set": "Eval80-v1",
        "metric": "supplementary_opening_Emotion_field_only", "protocol_metric": False,
        "image_count": 80,
        "parsed_opening_count": {"B1": 80, "C": 80},
        "correct_count": {"B1": 35, "C": 57},
        "net_correct_gain_c_minus_b1": 22,
        "paired_outcomes": {"both_correct": 33, "b1_only_correct": 2,
                            "c_only_correct": 24, "both_wrong": 21},
        "input_sha256": {"selection_manifest": _HASHES["selection"],
                         "b1_records": _HASHES["B1"], "c_records": _HASHES["C"]},
    }
    return formats, recognition, paired, preference, opening


class PreferenceOverviewTests(unittest.TestCase):
    def test_joins_consistent_aggregates_without_promoting_proxy(self):
        formats, recognition, paired, preference, opening = _reports()
        report = build_preference_overview(formats, recognition, paired, preference,
                                           opening)
        self.assertEqual(report["status"], "provisional_proxy")
        self.assertEqual(report["format"]["C_pass_count"], 80)
        self.assertEqual(report["strict_recognition"]["C_correct_count"], 57)
        self.assertIn("confounded", report["strict_recognition"]["b1_schema_caveat"])
        self.assertEqual(report["opening_label_diagnostic"]["B1_correct_count"], 35)
        self.assertEqual(report["guidance_preference_proxy"][
            "worst_case_net_if_all_unstable_favor_b1"], 10)
        self.assertIsNone(report["acceptance"]["formal_human_guidance_score"])
        self.assertFalse(report["acceptance"]["member3b_acceptance_verified"])

    def test_rejects_mixed_outputs_and_manifests(self):
        formats, recognition, paired, preference, opening = _reports()
        recognition["C"]["records_sha256"] = "9" * 64
        with self.assertRaisesRegex(ValueError, "different output records"):
            build_preference_overview(formats, recognition, paired, preference, opening)
        formats, recognition, paired, preference, opening = _reports()
        preference["selection_manifest_sha256"] = "9" * 64
        with self.assertRaisesRegex(ValueError, "different answer keys"):
            build_preference_overview(formats, recognition, paired, preference, opening)

    def test_rejects_inconsistent_preference_and_unbound_diagnostic(self):
        formats, recognition, paired, preference, opening = _reports()
        preference["image_consensus"]["swap_disagreements_unstable"] = 11
        with self.assertRaisesRegex(ValueError, "consensus does not cover"):
            build_preference_overview(formats, recognition, paired, preference, opening)
        formats, recognition, paired, preference, opening = _reports()
        opening["input_sha256"]["b1_records"] = "9" * 64
        with self.assertRaisesRegex(ValueError, "diagnostic is not bound"):
            build_preference_overview(formats, recognition, paired, preference, opening)

    def test_directional_claim_requires_positive_worst_case_bound(self):
        formats, recognition, paired, preference, opening = _reports()
        preference["image_consensus"].update({
            "C_wins": 25, "B1_wins": 20, "ties": 8,
            "swap_disagreements_unstable": 25,
            "stable_net_c_minus_b1": 5,
            "worst_case_net_if_all_unstable_favor_b1": -20,
        })
        report = build_preference_overview(formats, recognition, paired, preference,
                                           opening)
        self.assertIn("prevent a robust", report["headline"])
        self.assertNotIn("even if every unstable", report["headline"])

    def test_script_writes_aggregate_only_and_is_repeatable(self):
        formats, recognition, paired, preference, opening = _reports()
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = {}
            for key, document in (
                ("b1-format", formats["B1"]), ("c-format", formats["C"]),
                ("b1-recognition", recognition["B1"]),
                ("c-recognition", recognition["C"]),
                ("paired-recognition", paired), ("preference", preference),
                ("opening-diagnostic", opening),
            ):
                path = root / f"{key}.json"
                path.write_text(json.dumps(document), encoding="utf-8")
                inputs[key] = path
            output_json, output_md = root / "overview.json", root / "overview.md"
            args = [item for key, path in inputs.items() for item in (f"--{key}", str(path))]
            args += ["--output-json", str(output_json), "--output-md", str(output_md)]
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(args), 0)
                self.assertEqual(main(args), 0)
            markdown = output_md.read_text(encoding="utf-8")
            self.assertIn("C wins 40, B1 wins 20", markdown)
            self.assertIn("not a human 0–10 guidance score", markdown)
            self.assertNotIn("E001", markdown)
            self.assertNotIn("answer_A", output_json.read_text(encoding="utf-8"))
            preference["image_consensus"]["C_wins"] = 41
            inputs["preference"].write_text(json.dumps(preference), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "consensus does not cover"):
                main(args)


if __name__ == "__main__":
    unittest.main()
