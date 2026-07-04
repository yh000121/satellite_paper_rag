from __future__ import annotations

import hashlib
from dataclasses import dataclass, field


PARSER_VERSION = "text-parser-v1"
CHUNKER_VERSION = "multi-granularity-v1"
VOCABULARY_VERSION = "remote-sensing-en-v1"


def compute_source_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class PaperBlock:
    block_id: str
    text: str
    block_type: str
    page_start: int | None
    page_end: int | None
    order_index: int
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class PaperSection:
    section_id: str
    title: str
    normalized_type: str
    level: int
    blocks: list[PaperBlock] = field(default_factory=list)


@dataclass(frozen=True)
class Paper:
    paper_id: str
    title: str
    authors: list[str]
    year: str | None
    source_path: str | None
    source_hash: str
    source_type: str
    sections: list[PaperSection]
    metadata: dict[str, str] = field(default_factory=dict)
    parser_version: str = PARSER_VERSION
    vocabulary_version: str = VOCABULARY_VERSION


@dataclass(frozen=True)
class ChunkMetadata:
    satellites: list[str] = field(default_factory=list)
    sensors: list[str] = field(default_factory=list)
    bands_or_layers: list[str] = field(default_factory=list)
    indices: list[str] = field(default_factory=list)
    target_classes: list[str] = field(default_factory=list)
    evidence_types: list[str] = field(default_factory=list)
    thresholds: list[str] = field(default_factory=list)
    directionality: list[str] = field(default_factory=list)
    method_terms: list[str] = field(default_factory=list)
    metrics: list[str] = field(default_factory=list)
    units: list[str] = field(default_factory=list)
    normalized_values: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    failure_modes: list[str] = field(default_factory=list)
    confounding_factors: list[str] = field(default_factory=list)
    review_required_conditions: list[str] = field(default_factory=list)
    confidence: float = 0.0


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    paper_id: str
    chunk_type: str
    text: str
    parent_id: str | None
    child_ids: list[str]
    source_block_ids: list[str]
    page_start: int | None
    page_end: int | None
    section_title: str
    section_type: str
    metadata: ChunkMetadata
    retrieval_profile: str
    parser_version: str = PARSER_VERSION
    chunker_version: str = CHUNKER_VERSION
    vocabulary_version: str = VOCABULARY_VERSION
    source_hash: str = ""
