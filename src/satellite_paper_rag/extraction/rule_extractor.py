from __future__ import annotations

import re
from dataclasses import dataclass, field

from satellite_paper_rag.schemas import Chunk


@dataclass(frozen=True)
class ExtractedRule:
    rule_id: str
    paper_id: str
    chunk_id: str
    rule_type: str
    condition_text: str
    target_classes: list[str]
    features: list[str]
    thresholds: list[str]
    normalized_conditions: list[str]
    page_start: int | None
    page_end: int | None
    section_title: str
    score: float
    evidence_terms: list[str] = field(default_factory=list)
    rule_scope: str = "classification"
    dynamic_threshold_formula: str | None = None
    validation_only: bool = False


class RuleCandidateExtractor:
    RULE_TRIGGERS = [
        "classified as",
        "classification",
        "clear-sky probability",
        "binary cloud mask",
        "scene-dependent threshold",
        "threshold offset",
        "1:1 line",
        "local standard deviation",
        "rho",
        "cloudy conditions",
        "clear-sky",
        "validation",
        "identified as",
        "detected",
        "detecting",
        "screening",
        "mask",
        "masked",
        "discrimination",
        "separation",
        "associated with",
        "useful for",
    ]
    COMPARISON_TRIGGERS = [
        "below",
        "above",
        "greater than",
        "less than",
        "exceeded",
        "under",
        "over",
        ">",
        "<",
        ">=",
        "<=",
    ]

    def extract(self, chunks: list[Chunk], top_k: int | None = None) -> list[ExtractedRule]:
        rules = [rule for chunk in chunks for rule in self._extract_from_chunk(chunk)]
        rules.sort(key=self._ranking_key, reverse=True)
        rules = self._deduplicate(rules)
        if top_k is not None:
            return rules[:top_k]
        return rules

    def _extract_from_chunk(self, chunk: Chunk) -> list[ExtractedRule]:
        text_spans = self._candidate_spans(chunk.text)
        rules: list[ExtractedRule] = []
        for index, span in enumerate(text_spans):
            lower = span.lower()
            evidence_terms = self._evidence_terms(lower)
            normalized_conditions = self._normalized_conditions(span)
            thresholds = [threshold for threshold in chunk.metadata.thresholds if threshold.lower() in lower]
            if not thresholds:
                thresholds = self._raw_thresholds(span)
            target_classes = self._target_classes(lower, chunk.metadata.target_classes)
            features = self._features(span, chunk.metadata.bands_or_layers + chunk.metadata.indices)
            rule_type = self._rule_type(normalized_conditions, lower)
            dynamic_threshold_formula = self._dynamic_threshold_formula(normalized_conditions)
            rule_scope = self._rule_scope(lower)
            score = self._score(chunk, evidence_terms, normalized_conditions, rule_type)
            if score < 0.35:
                continue
            rules.append(
                ExtractedRule(
                    rule_id=f"{chunk.chunk_id}_rule_{index:03d}",
                    paper_id=chunk.paper_id,
                    chunk_id=chunk.chunk_id,
                    rule_type=rule_type,
                    condition_text=span,
                    target_classes=target_classes,
                    features=features,
                    thresholds=thresholds,
                    normalized_conditions=normalized_conditions,
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                    section_title=chunk.section_title,
                    score=round(score, 3),
                    evidence_terms=evidence_terms,
                    rule_scope=rule_scope,
                    dynamic_threshold_formula=dynamic_threshold_formula,
                    validation_only=rule_scope == "validation",
                )
            )
        return rules

    def _candidate_spans(self, text: str) -> list[str]:
        sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", text) if sentence.strip()]
        if len(sentences) == 1:
            return sentences
        spans: list[str] = []
        for index, sentence in enumerate(sentences):
            lower = sentence.lower()
            if self._evidence_terms(lower) or self._normalized_conditions(sentence):
                start = max(index - 1, 0)
                end = min(index + 2, len(sentences))
                spans.append(" ".join(sentences[start:end]))
        return spans

    def _evidence_terms(self, lower: str) -> list[str]:
        terms = [term for term in self.RULE_TRIGGERS if term in lower]
        terms.extend(term for term in self.COMPARISON_TRIGGERS if term in lower)
        return sorted(set(terms))

    def _target_classes(self, lower: str, fallback: list[str]) -> list[str]:
        classes: list[str] = []
        class_aliases = [
            ("sea_ice", ["sea ice"]),
            ("open_water", ["open water"]),
            ("cloud", ["cloud", "cloudy"]),
            ("water", ["water"]),
            ("land", ["land"]),
            ("vegetation", ["vegetation"]),
            ("snow", ["snow"]),
            ("ice", ["ice"]),
        ]
        for normalized, aliases in class_aliases:
            if any(alias in lower for alias in aliases) and normalized not in classes:
                classes.append(normalized)
        return classes or fallback

    def _features(self, text: str, fallback: list[str]) -> list[str]:
        features: list[str] = []
        for match in re.findall(r"\bS\d(?:-S\d)?\b|\bB\d{1,2}\b", text, re.IGNORECASE):
            normalized = match.upper()
            if normalized not in features:
                features.append(normalized)
        lower = text.lower()
        for phrase, normalized in [
            ("brightness temperature", "thermal_brightness_temperature"),
            ("thermal", "thermal"),
            ("visible reflectance", "visible_reflectance"),
            ("reflectance", "reflectance"),
            ("NDVI", "NDVI"),
            ("NDWI", "NDWI"),
            ("NDSI", "NDSI"),
            ("clear-sky probability", "clear_sky_probability"),
            ("local standard deviation", "local_standard_deviation"),
            ("rho", "rho"),
        ]:
            if phrase.lower() in lower and normalized not in features:
                features.append(normalized)
        return features or fallback

    def _normalized_conditions(self, text: str) -> list[str]:
        normalized: list[str] = []
        comparable_text = re.sub(r"\s+", " ", text.replace("\u03c1", "rho"))
        for match in re.finditer(
            r"\bclear-sky probability\b.*?\bthreshold of\s+(-?\d+(?:\.\d+)?)\b",
            comparable_text,
            re.IGNORECASE,
        ):
            condition = f"CLEAR_SKY_PROBABILITY >= {float(match.group(1))}"
            if condition not in normalized:
                normalized.append(condition)
        for match in re.finditer(r"\b(?:rho|\u03c1)\s*(>=|>|<=|<)\s*(-?\d+(?:\.\d+)?)\b", comparable_text, re.IGNORECASE):
            condition = f"RHO {match.group(1)} {float(match.group(2))}"
            if condition not in normalized:
                normalized.append(condition)
        for match in re.finditer(
            r"\b(?:local standard deviations?|LSD values?|LSD)\b.*?(>=|>|<=|<)\s*(-?\d+(?:\.\d+)?)\s*(K|C)?\b",
            comparable_text,
            re.IGNORECASE,
        ):
            condition = f"LSD {match.group(1)} {float(match.group(2))}"
            if match.group(3):
                condition = f"{condition} {match.group(3).upper()}"
            if condition not in normalized:
                normalized.append(condition)
        patterns = [
            (r"\b(BT)\s+(below|under|less than)\s+(-?\d+(?:\.\d+)?)\s*(K|C)\b", "<"),
            (r"\b(BT)\s+(above|over|greater than|exceeded)\s+(-?\d+(?:\.\d+)?)\s*(K|C)\b", ">"),
            (r"\b(NDVI|NDWI|NDSI)\s+(greater than|above|over|exceeded)\s+(-?\d+(?:\.\d+)?)\b", ">"),
            (r"\b(NDVI|NDWI|NDSI)\s+(below|under|less than)\s+(-?\d+(?:\.\d+)?)\b", "<"),
            (r"\b(NDVI|NDWI|NDSI|BT)\s*(>=|>|<=|<)\s*(-?\d+(?:\.\d+)?)\s*(K|C)?\b", None),
        ]
        for pattern, default_operator in patterns:
            for match in re.finditer(pattern, comparable_text, re.IGNORECASE):
                if default_operator is None:
                    feature = match.group(1).upper()
                    operator = match.group(2)
                    value = float(match.group(3))
                    unit = match.group(4) or ""
                else:
                    feature = match.group(1).upper()
                    operator = default_operator
                    value = float(match.group(3))
                    unit = match.group(4) if len(match.groups()) >= 4 else ""
                condition = f"{feature} {operator} {value}"
                if unit:
                    condition = f"{condition} {unit.upper()}"
                if condition not in normalized:
                    normalized.append(condition)
        offset_match = re.search(r"\badditional\s+(-?\d+(?:\.\d+)?)\s*(K|C)\b", comparable_text, re.IGNORECASE)
        if offset_match and "scene-dependent threshold" in comparable_text.lower():
            condition = f"SCENE_THRESHOLD_OFFSET + {float(offset_match.group(1))} {offset_match.group(2).upper()}"
            if condition not in normalized:
                normalized.append(condition)
        if "scene-dependent threshold" in comparable_text.lower() and "SCENE_DEPENDENT_THRESHOLD" not in normalized:
            normalized.append("SCENE_DEPENDENT_THRESHOLD")
        if "threshold offset" in comparable_text.lower() and "SCENE_THRESHOLD_OFFSET = c" not in normalized:
            normalized.append("SCENE_THRESHOLD_OFFSET = c")
        return normalized

    def _deduplicate(self, rules: list[ExtractedRule]) -> list[ExtractedRule]:
        unique_rules: list[ExtractedRule] = []
        seen: set[tuple[str, str, tuple[str, ...], tuple[str, ...], str]] = set()
        for rule in rules:
            normalized_text = re.sub(r"\W+", " ", rule.condition_text.lower()).strip()
            text_key = "" if rule.normalized_conditions else normalized_text[:160]
            key = (
                rule.rule_type,
                rule.rule_scope,
                tuple(rule.normalized_conditions),
                tuple(rule.target_classes),
                text_key,
            )
            if key in seen:
                continue
            seen.add(key)
            unique_rules.append(rule)
        return unique_rules

    def _ranking_key(self, rule: ExtractedRule) -> tuple[float, int, int]:
        type_boosts = {
            "dynamic_threshold_rule": 0.16,
            "threshold_rule": 0.08,
            "comparative_rule": 0.03,
        }
        scope_boosts = {
            "classification": 2,
            "algorithm": 1,
            "validation": 0,
        }
        return (
            min(rule.score + type_boosts.get(rule.rule_type, 0.0), 1.2),
            len(rule.normalized_conditions),
            scope_boosts.get(rule.rule_scope, 0),
        )

    def _raw_thresholds(self, text: str) -> list[str]:
        thresholds: list[str] = []
        for match in re.finditer(
            r"\b(?:threshold|cutoff|criterion|criteria|value|values)?\s*(?:of|=|:)?\s*(?:>=|>|<=|<)?\s*-?\d+(?:\.\d+)?\s*(?:K|C|%|dB)?\b",
            text,
            re.IGNORECASE,
        ):
            value = match.group(0).strip()
            if not value or not re.search(r"\d", value):
                continue
            if not re.search(r"threshold|cutoff|criterion|criteria|value|values|>=|>|<=|<|\bK\b|\bC\b|%|dB", value, re.IGNORECASE):
                continue
            if value not in thresholds:
                thresholds.append(value)
        return thresholds

    def _rule_type(self, normalized_conditions: list[str], lower: str) -> str:
        if "scene-dependent threshold" in lower or "threshold offset" in lower or any(
            condition.startswith("SCENE_THRESHOLD_OFFSET") or condition == "SCENE_DEPENDENT_THRESHOLD"
            for condition in normalized_conditions
        ):
            return "dynamic_threshold_rule"
        if normalized_conditions:
            return "threshold_rule"
        if any(term in lower for term in ["higher", "lower", "warmer", "colder", "brighter", "darker"]):
            return "comparative_rule"
        return "rule_candidate"

    def _rule_scope(self, lower: str) -> str:
        if any(term in lower for term in ["validation", "truth", "scene-dependent threshold", "warm s7"]):
            return "validation"
        if any(term in lower for term in ["configuration", "bayesian calculation", "probability"]):
            return "algorithm"
        return "classification"

    def _dynamic_threshold_formula(self, normalized_conditions: list[str]) -> str | None:
        for condition in normalized_conditions:
            if condition.startswith("SCENE_THRESHOLD_OFFSET") or condition == "SCENE_DEPENDENT_THRESHOLD":
                return "scene_threshold = 1:1 S7/S8 offset + uncertainty_margin"
        return None

    def _score(
        self,
        chunk: Chunk,
        evidence_terms: list[str],
        normalized_conditions: list[str],
        rule_type: str,
    ) -> float:
        score = 0.0
        if rule_type == "threshold_rule":
            score += 0.35
        elif rule_type == "dynamic_threshold_rule":
            score += 0.35
        elif rule_type == "comparative_rule":
            score += 0.2
        if evidence_terms:
            score += min(0.25, len(evidence_terms) * 0.05)
        if chunk.metadata.target_classes:
            score += 0.15
        if chunk.metadata.bands_or_layers or chunk.metadata.indices:
            score += 0.15
        if chunk.chunk_type == "rule_candidate":
            score += 0.1
        if normalized_conditions:
            score += 0.1
        return min(score, 1.0)
