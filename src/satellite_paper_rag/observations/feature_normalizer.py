from __future__ import annotations

from dataclasses import replace

from satellite_paper_rag.domain.vocabulary import DomainVocabulary
from satellite_paper_rag.observations.schema import ObservationSample


class FeatureNormalizer:
    def __init__(self, vocabulary: DomainVocabulary) -> None:
        self.vocabulary = vocabulary

    def normalize(self, sample: ObservationSample) -> ObservationSample:
        normalized: dict[str, float | str] = {}
        for name, value in sample.features.items():
            concept = None
            if sample.satellite and sample.sensor:
                concept = self.vocabulary.normalize_band(sample.satellite, sample.sensor, name)
            alias = self.vocabulary.normalize_alias(name)
            if concept:
                normalized[f"{concept}.{name}"] = value
            elif alias:
                normalized[alias] = value
            else:
                normalized[name] = value
        return replace(sample, normalized_features=normalized)
