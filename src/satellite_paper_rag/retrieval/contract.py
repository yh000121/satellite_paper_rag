from __future__ import annotations

from dataclasses import dataclass, field

from satellite_paper_rag.schemas import Chunk


@dataclass(frozen=True)
class RetrievalRequest:
    query: str
    chunk_types: list[str] = field(default_factory=list)
    metadata_filters: dict[str, list[str]] = field(default_factory=dict)
    expand_parents: bool = True
    top_k: int = 5
    requires_threshold: bool = False
    requested_sensor: str | None = None


@dataclass(frozen=True)
class RetrievalResult:
    chunk: Chunk
    score: float
    matched_terms: list[str]
    expanded_parent: Chunk | None
    rationale: str
    answer_type: str = "evidence"
    missing_evidence: list[str] = field(default_factory=list)
    is_indirect_evidence: bool = False
