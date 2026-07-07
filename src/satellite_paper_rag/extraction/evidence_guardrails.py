from __future__ import annotations

import re
from copy import deepcopy
from typing import Any

from satellite_paper_rag.schemas import Chunk


class EvidenceGuardrail:
    def validate(
        self,
        payload: dict[str, Any],
        evidence_chunks: list[Chunk],
        requires_threshold: bool = False,
    ) -> dict[str, Any]:
        result = deepcopy(payload)
        failures: list[str] = []

        matched_chunk = self._find_evidence_chunk(str(result.get("evidence_quote", "")), evidence_chunks)
        if matched_chunk is None:
            failures.append("evidence_quote_not_found")
        else:
            result["evidence_chunk_id"] = matched_chunk.chunk_id
            result["page_start"] = matched_chunk.page_start
            result["page_end"] = matched_chunk.page_end
            result["section_title"] = matched_chunk.section_title

        if requires_threshold and not self._has_threshold(result):
            failures.append("threshold_required_but_missing")

        result["guardrail_failures"] = failures
        if failures:
            result["answer_type"] = "insufficient_evidence"
        return result

    def _find_evidence_chunk(self, evidence_quote: str, chunks: list[Chunk]) -> Chunk | None:
        normalized_quote = self._normalize(evidence_quote)
        if not normalized_quote:
            return None
        for chunk in chunks:
            if normalized_quote in self._normalize(chunk.text):
                return chunk
        return None

    def _has_threshold(self, payload: dict[str, Any]) -> bool:
        thresholds = payload.get("thresholds")
        if isinstance(thresholds, list) and thresholds:
            return True
        normalized_conditions = payload.get("normalized_conditions")
        return isinstance(normalized_conditions, list) and bool(normalized_conditions)

    def _normalize(self, text: str) -> str:
        return re.sub(r"\s+", " ", text).strip().lower()
