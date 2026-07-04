import unittest

from satellite_paper_rag.domain.vocabulary import DomainVocabulary
from satellite_paper_rag.observations.feature_normalizer import FeatureNormalizer
from satellite_paper_rag.observations.schema import ObservationSample


class ObservationNormalizerTest(unittest.TestCase):
    def test_normalizes_sensor_band_features(self):
        sample = ObservationSample(
            sample_id="183_303_1402",
            source_type="csv",
            satellite="Sentinel-3",
            sensor="SLSTR",
            features={"S1": 18.0, "S7": 276.0, "S8": 273.2, "NDVI": 0.12},
            normalized_features={},
            metadata={},
        )
        normalized = FeatureNormalizer(DomainVocabulary.default()).normalize(sample)

        self.assertEqual(normalized.normalized_features["visible_reflectance.S1"], 18.0)
        self.assertEqual(normalized.normalized_features["thermal_brightness_temperature.S7"], 276.0)
        self.assertEqual(normalized.normalized_features["thermal_brightness_temperature.S8"], 273.2)
        self.assertEqual(normalized.normalized_features["NDVI"], 0.12)


if __name__ == "__main__":
    unittest.main()
