from __future__ import annotations

import re

from satellite_paper_rag.domain.vocabulary import DomainVocabulary
from satellite_paper_rag.schemas import ChunkMetadata


class MetadataEnricher:
    def __init__(self, vocabulary: DomainVocabulary) -> None:
        self.vocabulary = vocabulary

    def enrich(self, text: str) -> ChunkMetadata:
        lower = text.lower()
        satellites = self._find_terms(text, ["Sentinel-3", "Sentinel-2", "Landsat-8", "MODIS"])
        sensors = self._find_terms(text, ["SLSTR", "MSI", "OLI", "TIRS", "MODIS"])
        bands_or_layers = self._find_bands(text)
        indices = self._find_terms(text, ["NDVI", "NDWI", "NDSI"])
        target_classes = self._find_classes(lower)
        thresholds, normalized_values, units = self._find_thresholds(text)
        directionality = self._find_directionality(lower)
        evidence_types = self._find_evidence_types(lower, thresholds, directionality)
        limitations, failure_modes, review_required = self._find_limitations(lower)
        method_terms = self._find_terms(text, ["classification", "classifier", "feature contribution", "feature importance"])
        metrics = self._find_terms(text, ["accuracy", "IoU", "F1", "precision", "recall"])

        confidence = 0.0
        if target_classes:
            confidence += 0.2
        if bands_or_layers or indices:
            confidence += 0.2
        if thresholds:
            confidence += 0.25
        if directionality:
            confidence += 0.2
        if evidence_types:
            confidence += 0.15

        return ChunkMetadata(
            satellites=satellites,
            sensors=sensors,
            bands_or_layers=bands_or_layers,
            indices=indices,
            target_classes=target_classes,
            evidence_types=evidence_types,
            thresholds=thresholds,
            directionality=directionality,
            method_terms=method_terms,
            metrics=metrics,
            units=units,
            normalized_values=normalized_values,
            limitations=limitations,
            failure_modes=failure_modes,
            confounding_factors=failure_modes,
            review_required_conditions=review_required,
            confidence=min(confidence, 1.0),
        )

    def _find_terms(self, text: str, terms: list[str]) -> list[str]:
        found: list[str] = []
        for term in terms:
            if re.search(rf"\b{re.escape(term)}\b", text, re.IGNORECASE):
                found.append(term)
        return found

    def _find_bands(self, text: str) -> list[str]:
        found: list[str] = []
        for match in re.findall(r"\bS\d(?:-S\d)?\b|\bB\d{1,2}\b", text, re.IGNORECASE):
            normalized = match.upper()
            if normalized not in found:
                found.append(normalized)
        lower = text.lower()
        for phrase in ["thermal", "brightness temperature", "visible reflectance", "reflectance"]:
            if phrase in lower:
                concept = self.vocabulary.normalize_alias(phrase) or phrase
                if concept not in found:
                    found.append(concept)
        return found

    def _find_classes(self, lower: str) -> list[str]:
        classes: list[str] = []
        for phrase in ["cloud", "sea ice", "open water", "water", "land", "vegetation", "snow", "ice"]:
            if phrase in lower:
                normalized = self.vocabulary.normalize_alias(phrase) or phrase
                if normalized not in classes:
                    classes.append(normalized)
        return classes

    def _find_thresholds(self, text: str) -> tuple[list[str], list[str], list[str]]:
        thresholds: list[str] = []
        normalized_values: list[str] = []
        units: list[str] = []
        patterns = [
            (r"\b(BT)\s+(below|under|less than)\s+(-?\d+(?:\.\d+)?)\s*(K|C)\b", "<"),
            (r"\b(NDVI|NDWI|NDSI)\s+(greater than|above|over)\s+(-?\d+(?:\.\d+)?)\b", ">"),
            (r"\b(NDVI|NDWI|NDSI)\s*(>=|>|<=|<)\s*(-?\d+(?:\.\d+)?)\b", None),
        ]
        for pattern, default_operator in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                original = match.group(0)
                feature = match.group(1).upper()
                if default_operator is None:
                    operator = match.group(2)
                    value = float(match.group(3))
                    unit = ""
                else:
                    operator = default_operator
                    value = float(match.group(3))
                    unit = match.group(4) if len(match.groups()) >= 4 else ""
                thresholds.append(original)
                normalized_unit = "K" if unit.upper() == "K" else unit
                unit_suffix = f" {normalized_unit}" if normalized_unit else ""
                normalized_values.append(f"{feature} {operator} {value}{unit_suffix}")
                if normalized_unit and normalized_unit not in units:
                    units.append(normalized_unit)
        return thresholds, normalized_values, units

    def _find_directionality(self, lower: str) -> list[str]:
        phrases = ["higher", "lower", "increase", "decrease", "warmer", "colder", "brighter", "darker", "more than", "less than"]
        return [phrase for phrase in phrases if phrase in lower]

    def _find_evidence_types(self, lower: str, thresholds: list[str], directionality: list[str]) -> list[str]:
        evidence_types: list[str] = []
        if thresholds:
            evidence_types.append("threshold")
        if directionality:
            evidence_types.append("comparison")
        if "classified" in lower or "classification" in lower:
            evidence_types.append("classification_rule")
        if "table" in lower or "figure" in lower:
            evidence_types.append("result")
        return evidence_types

    def _find_limitations(self, lower: str) -> tuple[list[str], list[str], list[str]]:
        limitations: list[str] = []
        failure_modes: list[str] = []
        review_required: list[str] = []
        for phrase in ["thin cloud", "low solar angle", "seasonal variation"]:
            if phrase in lower:
                limitations.append(phrase)
        for phrase in ["mixed ice edge", "mixed pixels", "turbid water", "ambiguous"]:
            if phrase in lower:
                failure_modes.append(phrase)
        if "manual review" in lower or "required review" in lower:
            review_required.append("manual review")
        return limitations, failure_modes, review_required
