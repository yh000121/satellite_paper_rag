from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RuleEvalCase:
    case_id: str
    query: str
    expected_final_class: str | None = None
    expected_rule_direction: str | None = None
    expected_operator: str | None = None
    expected_threshold_value: float | None = None
    expected_evidence_contains: str | None = None
    requires_threshold: bool = False
    candidate_k: int = 8
    acceptable_matches: list[dict[str, str]] = field(default_factory=list)


@dataclass(frozen=True)
class RuleEvalResult:
    case_id: str
    query: str
    passed: bool
    failures: list[str]
    matched_rule: dict[str, Any] | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "query": self.query,
            "passed": self.passed,
            "failures": self.failures,
            "matched_rule": self.matched_rule,
        }


def load_eval_cases(path: Path) -> list[RuleEvalCase]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise RuntimeError("Rule eval file must contain a JSON list.")
    return [RuleEvalCase(**item) for item in payload]


def evaluate_extraction_payload(case: RuleEvalCase, payload: dict[str, Any]) -> RuleEvalResult:
    rules = payload.get("rules", [])
    if not isinstance(rules, list):
        return RuleEvalResult(case.case_id, case.query, False, ["rules_missing"], None)

    best_failures: list[str] = ["no_matching_rule"]
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        if rule.get("answer_type") != "rule":
            continue
        if rule.get("guardrail_failures"):
            continue
        failures = _rule_failures(case, rule)
        if not failures:
            return RuleEvalResult(case.case_id, case.query, True, [], rule)
        best_failures = failures
    return RuleEvalResult(case.case_id, case.query, False, best_failures, None)


def _rule_failures(case: RuleEvalCase, rule: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    if case.expected_final_class and rule.get("final_class") != case.expected_final_class:
        failures.append("expected_final_class_mismatch")
    if case.acceptable_matches:
        if not any(_matches_acceptable_form(rule, acceptable) for acceptable in case.acceptable_matches):
            failures.append("acceptable_match_not_found")
    else:
        if case.expected_rule_direction and rule.get("rule_direction") != case.expected_rule_direction:
            failures.append("expected_rule_direction_mismatch")
        if case.expected_operator and not _has_operator(rule, case.expected_operator):
            failures.append("expected_operator_mismatch")
    if case.expected_threshold_value is not None and not _has_threshold_value(rule, case.expected_threshold_value):
        failures.append("expected_threshold_value_not_found")
    if case.expected_evidence_contains and case.expected_evidence_contains.lower() not in str(
        rule.get("evidence_quote", "")
    ).lower():
        failures.append("expected_evidence_not_found")
    return failures


def _matches_acceptable_form(rule: dict[str, Any], acceptable: dict[str, str]) -> bool:
    expected_direction = acceptable.get("expected_rule_direction")
    if expected_direction and rule.get("rule_direction") != expected_direction:
        return False
    expected_operator = acceptable.get("expected_operator")
    if expected_operator and not _has_operator(rule, expected_operator):
        return False
    return True


def _has_operator(rule: dict[str, Any], expected_operator: str) -> bool:
    if rule.get("operator") == expected_operator:
        return True
    thresholds = rule.get("thresholds")
    return isinstance(thresholds, list) and any(
        isinstance(threshold, dict) and threshold.get("operator") == expected_operator
        for threshold in thresholds
    )


def _has_threshold_value(rule: dict[str, Any], expected_value: float, tolerance: float = 1e-9) -> bool:
    thresholds = rule.get("thresholds")
    if not isinstance(thresholds, list):
        return False
    for threshold in thresholds:
        if not isinstance(threshold, dict):
            continue
        value = threshold.get("value")
        if isinstance(value, (int, float)) and abs(float(value) - expected_value) <= tolerance:
            return True
    return False
