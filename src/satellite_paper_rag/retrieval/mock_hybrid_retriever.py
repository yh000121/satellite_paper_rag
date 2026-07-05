from __future__ import annotations

import re

from satellite_paper_rag.config import RetrievalConfig
from satellite_paper_rag.retrieval.contract import RetrievalRequest, RetrievalResult
from satellite_paper_rag.schemas import Chunk, ChunkMetadata


class MockHybridRetriever:
    def __init__(self, chunks: list[Chunk], config: RetrievalConfig | None = None) -> None:
        self.chunks = chunks
        self.by_id = {chunk.chunk_id: chunk for chunk in chunks}
        self.config = config or RetrievalConfig()

    def retrieve(self, request: RetrievalRequest) -> list[RetrievalResult]:
        candidates = [
            chunk
            for chunk in self.chunks
            if not request.chunk_types or chunk.chunk_type in request.chunk_types
        ]
        scored: list[RetrievalResult] = []
        for chunk in candidates:
            if not self._metadata_matches(chunk.metadata, request.metadata_filters):
                continue
            score, matched_terms = self._score(chunk, request.query)
            if score <= 0:
                continue
            score += self._chunk_type_boost(chunk.chunk_type)
            score += self._evidence_boost(chunk, request)
            expanded_parent = self.by_id.get(chunk.parent_id) if request.expand_parents and chunk.parent_id else None
            is_indirect = bool(
                request.requested_sensor
                and chunk.metadata.sensors
                and request.requested_sensor not in chunk.metadata.sensors
            )
            scored.append(
                RetrievalResult(
                    chunk=chunk,
                    score=score,
                    matched_terms=matched_terms,
                    expanded_parent=expanded_parent,
                    rationale="keyword, metadata, and chunk-type scoring",
                    is_indirect_evidence=is_indirect,
                )
            )

        scored.sort(key=lambda result: result.score, reverse=True)
        top = scored[: request.top_k]
        missing = self._missing_evidence(request, top)
        if missing:
            return [self._insufficient_result(missing, top)]
        return top

    def _metadata_matches(self, metadata: ChunkMetadata, filters: dict[str, list[str]]) -> bool:
        for field_name, expected_values in filters.items():
            actual_values = getattr(metadata, field_name, [])
            if expected_values and not any(value in actual_values for value in expected_values):
                return False
        return True

    def _score(self, chunk: Chunk, query: str) -> tuple[float, list[str]]:
        query_terms = [term.lower() for term in re.findall(r"[A-Za-z0-9-]+", query) if len(term) > 2]
        haystack = " ".join(
            [
                chunk.text,
                " ".join(chunk.metadata.satellites),
                " ".join(chunk.metadata.sensors),
                " ".join(chunk.metadata.bands_or_layers),
                " ".join(chunk.metadata.indices),
                " ".join(chunk.metadata.target_classes),
                " ".join(chunk.metadata.evidence_types),
                " ".join(chunk.metadata.thresholds),
            ]
        ).lower()
        matched = sorted({term for term in query_terms if term in haystack})
        return float(len(matched)), matched

    def _chunk_type_boost(self, chunk_type: str) -> float:
        return self.config.chunk_type_boosts.get(chunk_type, 0.0)

    def _evidence_boost(self, chunk: Chunk, request: RetrievalRequest) -> float:
        query = request.query.lower()
        score = 0.0
        if ("threshold" in query or request.requires_threshold) and chunk.metadata.thresholds:
            score += self.config.threshold_boost
        if any(term in query for term in ["thermal", "band", "feature", "layer"]) and chunk.metadata.bands_or_layers:
            score += self.config.band_or_feature_boost
        return score

    def _missing_evidence(self, request: RetrievalRequest, results: list[RetrievalResult]) -> list[str]:
        if not results:
            missing = ["relevant_chunks"]
            if request.requires_threshold:
                missing.append("thresholds")
            return missing
        missing: list[str] = []
        if request.requires_threshold and not any(result.chunk.metadata.thresholds for result in results):
            missing.append("thresholds")
        if request.requested_sensor and not any(request.requested_sensor in result.chunk.metadata.sensors for result in results):
            missing.append("same_sensor_evidence")
        return missing

    def _insufficient_result(self, missing: list[str], partial_results: list[RetrievalResult]) -> RetrievalResult:
        if partial_results:
            base = partial_results[0]
            return RetrievalResult(
                chunk=base.chunk,
                score=base.score,
                matched_terms=base.matched_terms,
                expanded_parent=base.expanded_parent,
                rationale="retrieved chunks did not satisfy the minimum evidence policy",
                answer_type="insufficient_evidence",
                missing_evidence=missing,
                is_indirect_evidence=base.is_indirect_evidence,
            )
        empty_chunk = Chunk(
            chunk_id="insufficient_evidence",
            paper_id="",
            chunk_type="none",
            text="",
            parent_id=None,
            child_ids=[],
            source_block_ids=[],
            page_start=None,
            page_end=None,
            section_title="",
            section_type="unknown",
            metadata=ChunkMetadata(),
            retrieval_profile="none",
        )
        return RetrievalResult(
            chunk=empty_chunk,
            score=0.0,
            matched_terms=[],
            expanded_parent=None,
            rationale="no relevant chunks were retrieved",
            answer_type="insufficient_evidence",
            missing_evidence=missing,
        )
