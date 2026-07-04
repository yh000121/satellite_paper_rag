import unittest

from satellite_paper_rag.chunking.metadata_enricher import MetadataEnricher
from satellite_paper_rag.domain.vocabulary import DomainVocabulary


class MetadataEnricherTest(unittest.TestCase):
    def test_extracts_remote_sensing_metadata(self):
        text = (
            "For Sentinel-3 SLSTR images, cloud pixels present lower brightness "
            "temperature in S7-S9 thermal channels. Pixels with BT below 270 K "
            "were classified as cloud-contaminated."
        )
        metadata = MetadataEnricher(DomainVocabulary.default()).enrich(text)

        self.assertIn("Sentinel-3", metadata.satellites)
        self.assertIn("SLSTR", metadata.sensors)
        self.assertIn("S7-S9", metadata.bands_or_layers)
        self.assertIn("thermal_brightness_temperature", metadata.bands_or_layers)
        self.assertIn("cloud", metadata.target_classes)
        self.assertIn("BT below 270 K", metadata.thresholds)
        self.assertIn("BT < 270.0 K", metadata.normalized_values)
        self.assertIn("threshold", metadata.evidence_types)

    def test_extracts_limitations_and_review_conditions(self):
        text = "Thin cloud and mixed ice edge pixels remained ambiguous and required manual review."
        metadata = MetadataEnricher(DomainVocabulary.default()).enrich(text)

        self.assertIn("thin cloud", metadata.limitations)
        self.assertIn("mixed ice edge", metadata.failure_modes)
        self.assertIn("manual review", metadata.review_required_conditions)


if __name__ == "__main__":
    unittest.main()
