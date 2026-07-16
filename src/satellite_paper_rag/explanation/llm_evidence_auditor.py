from __future__ import annotations

import json
import re
from typing import Any

from satellite_paper_rag.explanation.prediction_evidence import prediction_summary
from satellite_paper_rag.llm.client import ChatCompletionClient
from satellite_paper_rag.observations.schema import ObservationSample
from satellite_paper_rag.schemas import Chunk


ALLOWED_SUPPORT_STATUSES = {"strong", "partial", "weak", "conflict", "insufficient"}
REVIEW_REQUIRED_STATUSES = {"weak", "conflict", "insufficient"}


class LlmEvidenceAuditor:
    def __init__(self, chat_client: ChatCompletionClient) -> None:
        self.chat_client = chat_client

    def audit(self, observation: ObservationSample, evidence_chunks: list[Chunk]) -> dict[str, Any]:
        summary = prediction_summary(observation)
        predicted_class = summary.get("predicted_class")
        if not predicted_class:
            raise ValueError("Evidence audit requires label or predicted_class metadata.")
        if not evidence_chunks:
            return self._insufficient_without_evidence(summary)

        response = self.chat_client.complete_json(
            self._system_prompt(),
            self._user_prompt(summary, evidence_chunks),
        )
        return self._validate_response(response, summary, evidence_chunks)

    def _validate_response(
        self,
        response: dict[str, Any],
        summary: dict[str, object],
        evidence_chunks: list[Chunk],
    ) -> dict[str, Any]:
        failures: list[str] = []
        status = str(response.get("support_status", "")).strip().lower()
        if status not in ALLOWED_SUPPORT_STATUSES:
            failures.append("invalid_support_status")
            status = "insufficient"

        supporting = self._validate_citations(
            response.get("supporting_evidence"),
            evidence_chunks,
            "supporting_evidence",
            failures,
        )
        conflicting = self._validate_citations(
            response.get("conflicting_evidence"),
            evidence_chunks,
            "conflicting_evidence",
            failures,
        )

        if status in {"strong", "partial", "weak"} and not supporting:
            failures.append("supporting_evidence_required")
            status = "insufficient"
        if status == "conflict" and not conflicting:
            failures.append("conflicting_evidence_required")
            status = "insufficient"
        if failures:
            status = "insufficient"

        limitations = response.get("limitations")
        safe_limitations = [str(value) for value in limitations if str(value).strip()] if isinstance(limitations, list) else []
        requires_review = (
            bool(response.get("requires_human_review"))
            or status in REVIEW_REQUIRED_STATUSES
            or bool(failures)
        )
        return {
            "llm_model": self.chat_client.model_name,
            "sample_id": summary["sample_id"],
            "predicted_label_id": summary["predicted_label_id"],
            "predicted_class": summary["predicted_class"],
            "support_status": status,
            "conclusion": str(response.get("conclusion", "")).strip(),
            "supporting_evidence": supporting,
            "conflicting_evidence": conflicting,
            "limitations": safe_limitations,
            "requires_human_review": requires_review,
            "guardrail_failures": failures,
        }

    def _validate_citations(
        self,
        raw_citations: object,
        evidence_chunks: list[Chunk],
        field_name: str,
        failures: list[str],
    ) -> list[dict[str, object]]:
        if raw_citations is None:
            return []
        if not isinstance(raw_citations, list):
            failures.append(f"{field_name}_must_be_list")
            return []

        chunks_by_id = {chunk.chunk_id: chunk for chunk in evidence_chunks}
        validated: list[dict[str, object]] = []
        for citation in raw_citations:
            if not isinstance(citation, dict):
                failures.append(f"{field_name}_invalid_item")
                continue
            chunk_id = str(citation.get("chunk_id", "")).strip()
            quote = str(citation.get("quote", "")).strip()
            chunk = chunks_by_id.get(chunk_id)
            if chunk is None:
                failures.append(f"{field_name}_chunk_not_found")
                continue
            if not quote or self._normalize(quote) not in self._normalize(chunk.text):
                failures.append(f"{field_name}_quote_not_found")
                continue
            validated.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "quote": quote,
                    "reason": str(citation.get("reason", "")).strip(),
                    "page_start": chunk.page_start,
                    "page_end": chunk.page_end,
                    "section_title": chunk.section_title,
                }
            )
        return validated

    def _system_prompt(self) -> str:
        return (
            "You audit whether supplied satellite-paper evidence supports an existing model label. "
            "Do not classify the observation and do not propose a replacement class. "
            "Use only the supplied evidence chunks. Return valid JSON only. "
            "Every supporting or conflicting quote must be copied from its cited chunk. "
            "Do not invent thresholds, units, sensor mappings, or causal claims. "
            "Do not interpret S1-S6 radiance as reflectance or apply reflectance rules to those values "
            "unless a documented conversion is supplied. S7-S9 are brightness-temperature fields, "
            "but do not assume their unit or apply numeric thresholds while the unit is unknown."
        )

    def _user_prompt(self, summary: dict[str, object], evidence_chunks: list[Chunk]) -> str:
        observation_payload = {
            "sample_id": summary["sample_id"],
            "label_id": summary["predicted_label_id"],
            "label_class": summary["predicted_class"],
            "band_features": summary["band_features"],
            "band_schema": summary["band_schema"],
        }
        evidence = "\n\n".join(self._format_chunk(chunk) for chunk in evidence_chunks)
        return (
            "Observation to audit:\n"
            f"{json.dumps(observation_payload, ensure_ascii=False, indent=2)}\n\n"
            "Return JSON with exactly this shape:\n"
            "{\n"
            '  "support_status": "strong|partial|weak|conflict|insufficient",\n'
            '  "conclusion": "short evidence-based conclusion",\n'
            '  "supporting_evidence": [{"chunk_id": "...", "quote": "exact quote", "reason": "..."}],\n'
            '  "conflicting_evidence": [{"chunk_id": "...", "quote": "exact quote", "reason": "..."}],\n'
            '  "limitations": ["..."],\n'
            '  "requires_human_review": true\n'
            "}\n\n"
            "Status rules:\n"
            "- strong: direct, applicable evidence with no material missing condition.\n"
            "- partial: relevant evidence supports the label but conditions or numeric thresholds are incomplete.\n"
            "- weak: only indirect or qualitative evidence is available.\n"
            "- conflict: at least one supplied chunk materially contradicts the label.\n"
            "- insufficient: the supplied chunks cannot evaluate the label.\n\n"
            f"Evidence chunks:\n{evidence}"
        )

    def _format_chunk(self, chunk: Chunk) -> str:
        return (
            f"[{chunk.chunk_id}] page={chunk.page_start}-{chunk.page_end} "
            f"section={chunk.section_title} type={chunk.chunk_type}\n{chunk.text}"
        )

    def _insufficient_without_evidence(self, summary: dict[str, object]) -> dict[str, Any]:
        return {
            "llm_model": self.chat_client.model_name,
            "sample_id": summary["sample_id"],
            "predicted_label_id": summary["predicted_label_id"],
            "predicted_class": summary["predicted_class"],
            "support_status": "insufficient",
            "conclusion": "No retrieved evidence was available for audit.",
            "supporting_evidence": [],
            "conflicting_evidence": [],
            "limitations": ["No evidence chunks were supplied."],
            "requires_human_review": True,
            "guardrail_failures": ["no_evidence_chunks"],
        }

    def _normalize(self, text: str) -> str:
        return re.sub(r"\s+", " ", text).strip().lower()
