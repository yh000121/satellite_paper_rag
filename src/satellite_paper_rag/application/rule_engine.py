from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from satellite_paper_rag.domain.vocabulary import DomainVocabulary
from satellite_paper_rag.observations.feature_normalizer import FeatureNormalizer
from satellite_paper_rag.observations.schema import ObservationSample


SUPPORTED_CLASSES = ("cloud", "ice", "sea")
SUPPORTED_OPERATORS = {">", ">=", "<", "<=", "=", "=="}
NEGATIVE_DIRECTIONS = {"negative", "negative_evidence", "exclusion"}


@dataclass(frozen=True)
class MatchedRule:
    rule_id: str
    final_class: str
    direction: str
    variable: str
    observed_value: float
    operator: str
    threshold_value: float
    condition: str
    evidence_quote: str
    score: float

    def to_dict(self) -> dict[str, object]:
        return {
            "rule_id": self.rule_id,
            "final_class": self.final_class,
            "direction": self.direction,
            "variable": self.variable,
            "observed_value": self.observed_value,
            "operator": self.operator,
            "threshold_value": self.threshold_value,
            "condition": self.condition,
            "evidence_quote": self.evidence_quote,
            "score": self.score,
        }


@dataclass(frozen=True)
class RuleApplicationResult:
    sample_id: str
    predicted_class: str | None
    support: dict[str, float]
    metadata: dict[str, str] = field(default_factory=dict)
    matched_rules: list[MatchedRule] = field(default_factory=list)
    unmatched_rules: list[dict[str, object]] = field(default_factory=list)
    unsupported_rules: list[dict[str, object]] = field(default_factory=list)
    requires_human_review: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "sample_id": self.sample_id,
            "predicted_class": self.predicted_class,
            "metadata": self.metadata,
            "support": self.support,
            "matched_rules": [rule.to_dict() for rule in self.matched_rules],
            "unmatched_rules": self.unmatched_rules,
            "unsupported_rules": self.unsupported_rules,
            "requires_human_review": self.requires_human_review,
        }


class RuleApplicationEngine:
    def __init__(self, vocabulary: DomainVocabulary | None = None) -> None:
        self.vocabulary = vocabulary or DomainVocabulary.default()
        self.normalizer = FeatureNormalizer(self.vocabulary)

    def apply(self, sample: ObservationSample, rules: list[dict[str, Any]]) -> RuleApplicationResult:
        normalized_sample = self.normalizer.normalize(sample)
        feature_values = self._feature_lookup(normalized_sample)
        support = {class_name: 0.0 for class_name in SUPPORTED_CLASSES}
        matched: list[MatchedRule] = []
        unmatched: list[dict[str, object]] = []
        unsupported: list[dict[str, object]] = []

        for index, rule in enumerate(rules, start=1):
            rule_id = str(rule.get("rule_id") or f"rule_{index:04d}")
            final_class = self._normalize_class(str(rule.get("final_class") or ""))
            if not final_class:
                unsupported.append({"rule_id": rule_id, "reason": "unsupported_or_missing_final_class"})
                continue
            threshold = self._first_numeric_threshold(rule)
            operator = str(rule.get("operator") or (threshold or {}).get("operator") or "").strip()
            if operator not in SUPPORTED_OPERATORS or threshold is None:
                unsupported.append({"rule_id": rule_id, "reason": "unsupported_or_missing_numeric_threshold"})
                continue
            variable = str(rule.get("variable") or threshold.get("name") or "")
            normalized_variable = self._normalize_feature_name(variable)
            observed_value = feature_values.get(normalized_variable)
            if not isinstance(observed_value, (int, float)):
                unmatched.append({"rule_id": rule_id, "reason": "missing_numeric_feature", "variable": variable})
                continue
            threshold_value = float(threshold["value"])
            observed_float = float(observed_value)
            if not self._compare(observed_float, operator, threshold_value):
                unmatched.append(
                    {
                        "rule_id": rule_id,
                        "reason": "condition_not_met",
                        "condition": self._condition(observed_float, operator, threshold_value),
                    }
                )
                continue
            direction = str(rule.get("rule_direction") or "positive_evidence")
            score = -1.0 if direction in NEGATIVE_DIRECTIONS else 1.0
            support[final_class] += score
            matched.append(
                MatchedRule(
                    rule_id=rule_id,
                    final_class=final_class,
                    direction=direction,
                    variable=normalized_variable,
                    observed_value=observed_float,
                    operator=operator,
                    threshold_value=threshold_value,
                    condition=self._condition(observed_float, operator, threshold_value),
                    evidence_quote=str(rule.get("evidence_quote") or ""),
                    score=score,
                )
            )

        predicted_class = self._winner(support)
        requires_review = predicted_class is None or bool(unsupported) or self._has_positive_tie(support)
        return RuleApplicationResult(
            sample_id=sample.sample_id,
            predicted_class=predicted_class,
            support=support,
            metadata=sample.metadata,
            matched_rules=matched,
            unmatched_rules=unmatched,
            unsupported_rules=unsupported,
            requires_human_review=requires_review,
        )

    def _feature_lookup(self, sample: ObservationSample) -> dict[str, float | str]:
        lookup: dict[str, float | str] = {}
        for name, value in sample.features.items():
            lookup[self._normalize_feature_name(name)] = value
        for name, value in sample.normalized_features.items():
            lookup[self._normalize_feature_name(name)] = value
        return lookup

    def _normalize_feature_name(self, name: str) -> str:
        cleaned = name.strip().lower().replace("-", " ").replace("_", " ")
        alias = self.vocabulary.normalize_alias(cleaned)
        if alias:
            return alias
        return "_".join(cleaned.split())

    def _normalize_class(self, class_name: str) -> str | None:
        alias = self.vocabulary.normalize_alias(class_name) or class_name.strip().lower()
        mapping = {
            "cloud": "cloud",
            "snow": "ice",
            "sea_ice": "ice",
            "ice": "ice",
            "land": "ice",
            "open_water": "sea",
            "water": "sea",
            "sea": "sea",
        }
        return mapping.get(alias)

    def _first_numeric_threshold(self, rule: dict[str, Any]) -> dict[str, Any] | None:
        thresholds = rule.get("thresholds")
        if not isinstance(thresholds, list):
            return None
        for threshold in thresholds:
            if isinstance(threshold, dict) and _is_number(threshold.get("value")):
                return threshold
        return None

    def _compare(self, observed: float, operator: str, threshold: float) -> bool:
        if operator == ">":
            return observed > threshold
        if operator == ">=":
            return observed >= threshold
        if operator == "<":
            return observed < threshold
        if operator == "<=":
            return observed <= threshold
        return observed == threshold

    def _condition(self, observed: float, operator: str, threshold: float) -> str:
        return f"{observed} {operator} {threshold}"

    def _winner(self, support: dict[str, float]) -> str | None:
        positive = {class_name: score for class_name, score in support.items() if score > 0}
        if not positive:
            return None
        best_score = max(positive.values())
        winners = [class_name for class_name, score in positive.items() if score == best_score]
        if len(winners) != 1:
            return None
        return winners[0]

    def _has_positive_tie(self, support: dict[str, float]) -> bool:
        positive_scores = [score for score in support.values() if score > 0]
        return bool(positive_scores) and positive_scores.count(max(positive_scores)) > 1


def _is_number(value: object) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return True
    if isinstance(value, str):
        try:
            float(value)
        except ValueError:
            return False
        return True
    return False
