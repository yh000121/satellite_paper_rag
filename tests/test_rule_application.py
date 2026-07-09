import unittest

from satellite_paper_rag.application.rule_engine import RuleApplicationEngine
from satellite_paper_rag.observations.schema import ObservationSample


class RuleApplicationEngineTest(unittest.TestCase):
    def test_applies_numeric_threshold_rules_to_observation(self):
        sample = ObservationSample(
            sample_id="p001",
            source_type="json",
            satellite=None,
            sensor=None,
            features={"rho": 2.4, "LSD": 0.35},
        )
        rules = [
            {
                "rule_id": "rule_rho",
                "final_class": "cloud",
                "rule_direction": "positive_evidence",
                "variable": "rho",
                "operator": ">",
                "thresholds": [{"name": "rho", "operator": ">", "value": 2.0}],
                "evidence_quote": "All values of rho > 2 represent cloudy conditions.",
            },
            {
                "rule_id": "rule_lsd",
                "final_class": "cloud",
                "rule_direction": "positive_evidence",
                "variable": "local standard deviation of brightness temperature",
                "operator": ">",
                "thresholds": [
                    {"name": "local standard deviation of brightness temperature", "operator": ">", "value": 0.2}
                ],
                "evidence_quote": "For LSD values >0.2 K, cloudy conditions are more probable.",
            },
        ]

        result = RuleApplicationEngine().apply(sample, rules)

        self.assertEqual(result.predicted_class, "cloud")
        self.assertEqual(result.support["cloud"], 2.0)
        self.assertFalse(result.requires_human_review)
        self.assertEqual([match.rule_id for match in result.matched_rules], ["rule_rho", "rule_lsd"])
        self.assertEqual(result.matched_rules[0].condition, "2.4 > 2.0")

    def test_matching_negative_evidence_reduces_support_and_requires_review_without_positive_evidence(self):
        sample = ObservationSample(
            sample_id="p002",
            source_type="json",
            satellite=None,
            sensor=None,
            features={"clear_sky_probability": 0.95},
        )
        rules = [
            {
                "rule_id": "rule_clear_sky",
                "final_class": "cloud",
                "rule_direction": "negative_evidence",
                "variable": "clear-sky probability",
                "operator": ">=",
                "thresholds": [{"name": "clear-sky probability", "operator": ">=", "value": 0.9}],
                "evidence_quote": "threshold of 0.9 would typically be applied",
            }
        ]

        result = RuleApplicationEngine().apply(sample, rules)

        self.assertIsNone(result.predicted_class)
        self.assertEqual(result.support["cloud"], -1.0)
        self.assertTrue(result.requires_human_review)
        self.assertEqual(result.matched_rules[0].direction, "negative_evidence")

    def test_unsupported_qualitative_rules_are_not_forced(self):
        sample = ObservationSample(
            sample_id="p003",
            source_type="json",
            satellite=None,
            sensor=None,
            features={"visible_reflectance": 0.7},
        )
        rules = [
            {
                "rule_id": "rule_ice",
                "final_class": "ice",
                "rule_direction": "positive_evidence",
                "variable": "visible reflectance",
                "operator": "qualitative_comparison",
                "thresholds": [],
                "evidence_quote": "brighter ice or cloud surfaces and the darker ocean",
            }
        ]

        result = RuleApplicationEngine().apply(sample, rules)

        self.assertIsNone(result.predicted_class)
        self.assertEqual(result.unsupported_rules[0]["rule_id"], "rule_ice")
        self.assertTrue(result.requires_human_review)


if __name__ == "__main__":
    unittest.main()
