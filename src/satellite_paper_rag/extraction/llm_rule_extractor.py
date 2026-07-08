from __future__ import annotations

from typing import Any

from satellite_paper_rag.extraction.class_normalizer import ClassNormalizer
from satellite_paper_rag.extraction.evidence_guardrails import EvidenceGuardrail
from satellite_paper_rag.llm.client import ChatCompletionClient
from satellite_paper_rag.schemas import Chunk


class LlmRuleExtractor:
    def __init__(
        self,
        chat_client: ChatCompletionClient,
        guardrail: EvidenceGuardrail | None = None,
        class_normalizer: ClassNormalizer | None = None,
    ) -> None:
        self.chat_client = chat_client
        self.guardrail = guardrail or EvidenceGuardrail()
        self.class_normalizer = class_normalizer or ClassNormalizer()

    def extract(
        self,
        query: str,
        evidence_chunks: list[Chunk],
        requires_threshold: bool = False,
    ) -> dict[str, Any]:
        response = self.chat_client.complete_json(
            self._system_prompt(),
            self._user_prompt(query, evidence_chunks),
        )
        raw_rules = response.get("rules", [])
        if not isinstance(raw_rules, list):
            raise RuntimeError("LLM rule extraction response must contain a rules list.")
        validated_rules = []
        for rule in raw_rules:
            sanitized_rule = self._sanitize_rule(
                rule if isinstance(rule, dict) else {"answer_type": "insufficient_evidence"}
            )
            normalized_rule = self.class_normalizer.normalize(
                sanitized_rule
            )
            validated_rules.append(
                self.guardrail.validate(
                    normalized_rule,
                    evidence_chunks,
                    requires_threshold=requires_threshold,
                )
            )
        return {
            "llm_model": self.chat_client.model_name,
            "query": query,
            "rules": validated_rules,
        }

    def _sanitize_rule(self, rule: dict[str, Any]) -> dict[str, Any]:
        sanitized = dict(rule)
        thresholds = sanitized.get("thresholds")
        if not isinstance(thresholds, list):
            return sanitized
        retained_thresholds = [threshold for threshold in thresholds if not self._is_placeholder_threshold(threshold)]
        if len(retained_thresholds) != len(thresholds):
            sanitized["thresholds"] = retained_thresholds
            if not retained_thresholds:
                sanitized["operator"] = "qualitative_comparison"
        return sanitized

    def _is_placeholder_threshold(self, threshold: object) -> bool:
        if not isinstance(threshold, dict):
            return False
        unit = str(threshold.get("unit", "")).lower()
        value = threshold.get("value")
        return unit == "relative brightness" and value in {0, 0.0, "0", "0.0"}

    def _system_prompt(self) -> str:
        return (
            "You extract satellite remote-sensing classification rules from paper evidence. "
            "Use only the provided evidence chunks. Return valid JSON only. "
            "If the evidence is insufficient, set answer_type to insufficient_evidence. "
            "Every rule must include evidence_quote copied exactly from one evidence chunk. "
            "The final project classes are only cloud, ice, and sea. Preserve the paper's original "
            "class in evidence_class and map it into final_class."
        )

    def _user_prompt(self, query: str, evidence_chunks: list[Chunk]) -> str:
        evidence = "\n\n".join(self._format_chunk(chunk) for chunk in evidence_chunks)
        return (
            f"Question:\n{query}\n\n"
            "Return JSON with this shape:\n"
            "{\n"
            '  "rules": [\n'
            "    {\n"
            '      "answer_type": "rule|insufficient_evidence",\n'
            '      "evidence_class": "paper class such as cloud|clear_sky|snow|ice|water|land|unknown",\n'
            '      "final_class": "cloud|ice|sea",\n'
            '      "rule_direction": "positive_evidence|negative_evidence|unknown",\n'
            '      "class_mapping_reason": "why evidence_class maps to final_class",\n'
            '      "requires_human_review": false,\n'
            '      "sensor": "sensor name or null",\n'
            '      "variable": "band/layer/index/probability name or null",\n'
            '      "operator": "<|<=|>|>=|=|dynamic|qualitative_comparison|null",\n'
            '      "thresholds": [{"name": "variable", "operator": "<", "value": 0.0, "unit": "optional"}],\n'
            '      "condition_text": "short natural-language condition",\n'
            '      "evidence_quote": "exact quote from evidence",\n'
            '      "confidence": 0.0\n'
            "    }\n"
            "  ]\n"
            "}\n\n"
            "Class mapping rules:\n"
            "- cloud/cloudy/non-clear -> final_class cloud as positive_evidence\n"
            "- clear_sky/cloud_free -> final_class cloud as negative_evidence\n"
            "- water/open_water/ocean/sea -> final_class sea as positive_evidence\n"
            "- ice/sea_ice/snow/iceberg -> final_class ice as positive_evidence\n"
            "- land -> final_class ice as positive_evidence, requires_human_review=true\n\n"
            "Extraction rules:\n"
            "- If one evidence sentence contains paired threshold conditions, extract each paired threshold as a separate rule.\n"
            "- If the question includes a numeric observation value, prioritize rules whose operator matches that value.\n"
            "- Do not drop the cloudy side of a clear-sky/cloudy contrast when both are stated in the evidence.\n\n"
            "- Do not invent numeric thresholds. For qualitative brighter/darker evidence, use thresholds=[] "
            "and operator=qualitative_comparison.\n\n"
            f"Evidence chunks:\n{evidence}"
        )

    def _format_chunk(self, chunk: Chunk) -> str:
        return (
            f"[{chunk.chunk_id}] page={chunk.page_start}-{chunk.page_end} "
            f"section={chunk.section_title} type={chunk.chunk_type}\n"
            f"{chunk.text}"
        )
