import unittest

from satellite_paper_rag.extraction.class_normalizer import ClassNormalizer


class ClassNormalizerTest(unittest.TestCase):
    def test_maps_water_evidence_to_sea(self):
        payload = {"answer_type": "rule", "target_class": "open_water"}

        result = ClassNormalizer().normalize(payload)

        self.assertEqual(result["evidence_class"], "open_water")
        self.assertEqual(result["final_class"], "sea")
        self.assertEqual(result["rule_direction"], "positive_evidence")
        self.assertFalse(result["requires_human_review"])

    def test_maps_land_evidence_to_ice_with_review(self):
        payload = {"answer_type": "rule", "target_class": "land"}

        result = ClassNormalizer().normalize(payload)

        self.assertEqual(result["evidence_class"], "land")
        self.assertEqual(result["final_class"], "ice")
        self.assertEqual(result["rule_direction"], "positive_evidence")
        self.assertTrue(result["requires_human_review"])
        self.assertIn("land", result["class_mapping_reason"])

    def test_maps_clear_sky_to_cloud_negative_evidence(self):
        payload = {"answer_type": "rule", "target_class": "clear_sky"}

        result = ClassNormalizer().normalize(payload)

        self.assertEqual(result["evidence_class"], "clear_sky")
        self.assertEqual(result["final_class"], "cloud")
        self.assertEqual(result["rule_direction"], "negative_evidence")
        self.assertFalse(result["requires_human_review"])

    def test_clear_sky_less_than_threshold_is_positive_cloud_evidence(self):
        payload = {
            "answer_type": "rule",
            "evidence_class": "clear_sky",
            "final_class": "cloud",
            "rule_direction": "negative_evidence",
            "operator": "<",
            "variable": "clear-sky probability",
            "thresholds": [{"name": "clear-sky probability", "operator": "<", "value": 0.9}],
        }

        result = ClassNormalizer().normalize(payload)

        self.assertEqual(result["final_class"], "cloud")
        self.assertEqual(result["rule_direction"], "positive_evidence")
        self.assertIn("below", result["class_mapping_reason"])

    def test_clear_sky_greater_than_threshold_is_negative_cloud_evidence(self):
        payload = {
            "answer_type": "rule",
            "evidence_class": "clear_sky",
            "final_class": "cloud",
            "operator": ">=",
            "variable": "clear-sky probability",
            "thresholds": [{"name": "clear-sky probability", "operator": ">=", "value": 0.9}],
        }

        result = ClassNormalizer().normalize(payload)

        self.assertEqual(result["rule_direction"], "negative_evidence")

    def test_clear_sky_low_lsd_is_negative_cloud_evidence(self):
        payload = {
            "answer_type": "rule",
            "evidence_class": "clear_sky",
            "final_class": "cloud",
            "operator": "<",
            "variable": "local standard deviation of brightness temperature",
            "thresholds": [
                {
                    "name": "local standard deviation of brightness temperature",
                    "operator": "<",
                    "value": 0.2,
                    "unit": "K",
                }
            ],
        }

        result = ClassNormalizer().normalize(payload)

        self.assertEqual(result["rule_direction"], "negative_evidence")
        self.assertIn("clear_sky", result["class_mapping_reason"])

    def test_leaves_insufficient_evidence_without_final_class(self):
        payload = {"answer_type": "insufficient_evidence", "target_class": "land"}

        result = ClassNormalizer().normalize(payload)

        self.assertIsNone(result["final_class"])
        self.assertEqual(result["rule_direction"], "not_applicable")


if __name__ == "__main__":
    unittest.main()
