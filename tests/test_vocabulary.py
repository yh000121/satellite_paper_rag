import unittest

from satellite_paper_rag.domain.vocabulary import DomainVocabulary


class DomainVocabularyTest(unittest.TestCase):
    def test_normalizes_english_aliases(self):
        vocab = DomainVocabulary.default()
        self.assertEqual(vocab.normalize_alias("BT"), "thermal_brightness_temperature")
        self.assertEqual(vocab.normalize_alias("open water"), "open_water")
        self.assertEqual(vocab.normalize_alias("sea ice"), "sea_ice")

    def test_uses_sensor_band_context(self):
        vocab = DomainVocabulary.default()
        self.assertEqual(
            vocab.normalize_band("Sentinel-3", "SLSTR", "S7"),
            "thermal_brightness_temperature",
        )
        self.assertEqual(
            vocab.normalize_band("Landsat-8", "TIRS", "B10"),
            "thermal_brightness_temperature",
        )
        self.assertIsNone(vocab.normalize_band("Unknown", "Unknown", "S7"))

    def test_phase_one_is_english_only(self):
        vocab = DomainVocabulary.default()
        self.assertIsNone(vocab.normalize_alias("yun"))


if __name__ == "__main__":
    unittest.main()
