import unittest
from pathlib import Path

from satellite_paper_rag.cli import build_parser
from satellite_paper_rag.evaluation.rule_eval import RuleEvalCase, evaluate_extraction_payload, load_eval_cases


class RuleEvalTest(unittest.TestCase):
    def test_loads_fixed_eval_cases(self):
        cases = load_eval_cases(Path("tests/fixtures/eval_queries.json"))

        self.assertEqual(len(cases), 5)
        self.assertEqual(cases[0].case_id, "clear_sky_probability_0_72_cloud")
        self.assertTrue(cases[0].requires_threshold)

    def test_eval_rules_cli_command_parses(self):
        parser = build_parser()

        args = parser.parse_args(
            [
                "eval-rules",
                "--file",
                "tests/fixtures/sample_sentinel3_paper.md",
                "--eval-file",
                "tests/fixtures/eval_queries.json",
            ]
        )

        self.assertEqual(args.eval_file, "tests/fixtures/eval_queries.json")
        self.assertTrue(callable(args.func))

    def test_passes_when_any_rule_matches_expected_class_threshold_and_evidence(self):
        case = RuleEvalCase(
            case_id="clear_sky_probability_cloud",
            query="Given clear-sky probability 0.72, what rule applies?",
            expected_final_class="cloud",
            expected_rule_direction="positive_evidence",
            expected_operator="<",
            expected_threshold_value=0.9,
            expected_evidence_contains="threshold of 0.9",
        )
        payload = {
            "rules": [
                {
                    "answer_type": "rule",
                    "final_class": "cloud",
                    "rule_direction": "positive_evidence",
                    "operator": "<",
                    "thresholds": [{"name": "clear-sky probability", "operator": "<", "value": 0.9}],
                    "evidence_quote": "a threshold of 0.9 would typically be applied",
                    "guardrail_failures": [],
                }
            ]
        }

        result = evaluate_extraction_payload(case, payload)

        self.assertTrue(result.passed)
        self.assertEqual(result.case_id, "clear_sky_probability_cloud")
        self.assertEqual(result.matched_rule["final_class"], "cloud")

    def test_fails_when_threshold_is_missing(self):
        case = RuleEvalCase(
            case_id="rho_cloud",
            query="Given rho 2.4, what rule applies?",
            expected_final_class="cloud",
            expected_threshold_value=2.0,
        )
        payload = {
            "rules": [
                {
                    "answer_type": "rule",
                    "final_class": "cloud",
                    "thresholds": [],
                    "evidence_quote": "All values of rho > 2 represent cloudy conditions.",
                    "guardrail_failures": [],
                }
            ]
        }

        result = evaluate_extraction_payload(case, payload)

        self.assertFalse(result.passed)
        self.assertIn("expected_threshold_value_not_found", result.failures)

    def test_passes_when_rule_matches_one_acceptable_equivalent_form(self):
        case = RuleEvalCase(
            case_id="clear_sky_equivalent_forms",
            query="Given clear-sky probability 0.72, what rule applies?",
            expected_final_class="cloud",
            expected_threshold_value=0.9,
            acceptable_matches=[
                {"expected_rule_direction": "positive_evidence", "expected_operator": "<"},
                {"expected_rule_direction": "negative_evidence", "expected_operator": ">="},
            ],
        )
        payload = {
            "rules": [
                {
                    "answer_type": "rule",
                    "final_class": "cloud",
                    "rule_direction": "negative_evidence",
                    "operator": ">=",
                    "thresholds": [{"name": "clear-sky probability", "operator": ">=", "value": 0.9}],
                    "evidence_quote": "a threshold of 0.9 would typically be applied",
                    "guardrail_failures": [],
                }
            ]
        }

        result = evaluate_extraction_payload(case, payload)

        self.assertTrue(result.passed)

    def test_ignores_rules_with_guardrail_failures(self):
        case = RuleEvalCase(
            case_id="invalid_quote",
            query="What rule applies?",
            expected_final_class="cloud",
        )
        payload = {
            "rules": [
                {
                    "answer_type": "rule",
                    "final_class": "cloud",
                    "thresholds": [],
                    "evidence_quote": "missing quote",
                    "guardrail_failures": ["evidence_quote_not_found"],
                }
            ]
        }

        result = evaluate_extraction_payload(case, payload)

        self.assertFalse(result.passed)
        self.assertIn("no_matching_rule", result.failures)


if __name__ == "__main__":
    unittest.main()
