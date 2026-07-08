from __future__ import annotations

from copy import deepcopy
from typing import Any


class ClassNormalizer:
    FINAL_CLASSES = {"cloud", "ice", "sea"}

    def normalize(self, payload: dict[str, Any]) -> dict[str, Any]:
        result = deepcopy(payload)
        if result.get("answer_type") == "insufficient_evidence":
            result["evidence_class"] = self._raw_class(result)
            result["final_class"] = None
            result["rule_direction"] = "not_applicable"
            result["class_mapping_reason"] = "No class mapping is applied when evidence is insufficient."
            result["requires_human_review"] = bool(result.get("requires_human_review", False))
            return result

        evidence_class = self._raw_class(result)
        final_class, direction, reason, review_required = self._map_class(evidence_class)
        direction, reason = self._directional_override(result, evidence_class, direction, reason)
        result["evidence_class"] = evidence_class
        result["final_class"] = final_class
        result["rule_direction"] = direction
        result["class_mapping_reason"] = reason
        result["requires_human_review"] = bool(result.get("requires_human_review", False) or review_required)
        return result

    def _raw_class(self, payload: dict[str, Any]) -> str:
        raw_value = payload.get("evidence_class") or payload.get("target_class") or ""
        return self._normalize_label(str(raw_value)) or "unknown"

    def _normalize_label(self, value: str) -> str:
        return value.strip().lower().replace("-", "_").replace(" ", "_")

    def _map_class(self, evidence_class: str) -> tuple[str | None, str, str, bool]:
        if evidence_class in {"cloud", "cloudy", "non_clear", "non_clear_sky", "not_clear_sky"}:
            return "cloud", "positive_evidence", f"{evidence_class} maps directly to cloud.", False
        if evidence_class in {"clear_sky", "clear", "clear_water", "cloud_free"}:
            return "cloud", "negative_evidence", f"{evidence_class} is negative evidence for cloud.", False
        if evidence_class in {"sea", "water", "open_water", "ocean", "marine", "seawater"}:
            return "sea", "positive_evidence", f"{evidence_class} maps to sea.", False
        if evidence_class in {"ice", "sea_ice", "snow", "iceberg", "ice_surface", "floe"}:
            return "ice", "positive_evidence", f"{evidence_class} maps to ice.", False
        if evidence_class == "land":
            return (
                "ice",
                "positive_evidence",
                "land is mapped to ice for this project because icebergs or ice surfaces can present land-like signatures.",
                True,
            )
        return None, "unknown", f"{evidence_class} cannot be mapped to cloud, ice, or sea.", True

    def _directional_override(
        self,
        payload: dict[str, Any],
        evidence_class: str,
        direction: str,
        reason: str,
    ) -> tuple[str, str]:
        if evidence_class not in {"clear_sky", "clear", "clear_water", "cloud_free"}:
            return direction, reason
        operator = self._condition_operator(payload)
        if self._is_clear_sky_probability(payload) and operator in {"<", "<="}:
            return (
                "positive_evidence",
                f"{evidence_class} below the threshold is positive evidence for cloud.",
            )
        if self._is_clear_sky_probability(payload) and operator in {">", ">=", "="}:
            return (
                "negative_evidence",
                f"{evidence_class} at or above the threshold is negative evidence for cloud.",
            )
        return direction, reason

    def _condition_operator(self, payload: dict[str, Any]) -> str | None:
        operator = payload.get("operator")
        if isinstance(operator, str) and operator in {"<", "<=", ">", ">=", "="}:
            return operator
        thresholds = payload.get("thresholds")
        if isinstance(thresholds, list):
            for threshold in thresholds:
                if isinstance(threshold, dict) and threshold.get("operator") in {"<", "<=", ">", ">=", "="}:
                    return str(threshold["operator"])
        return None

    def _is_clear_sky_probability(self, payload: dict[str, Any]) -> bool:
        candidates = [payload.get("variable")]
        thresholds = payload.get("thresholds")
        if isinstance(thresholds, list):
            for threshold in thresholds:
                if isinstance(threshold, dict):
                    candidates.append(threshold.get("name"))
        normalized = " ".join(str(candidate).lower().replace("_", " ") for candidate in candidates if candidate)
        return "clear sky probability" in normalized or "clear-sky probability" in normalized
