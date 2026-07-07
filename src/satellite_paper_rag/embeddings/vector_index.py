from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from satellite_paper_rag.embeddings.client import EmbeddingClient, dot_product
from satellite_paper_rag.schemas import CHUNKER_VERSION, Chunk, ChunkMetadata


@dataclass(frozen=True)
class VectorRecord:
    chunk: Chunk
    embedding: list[float]


@dataclass(frozen=True)
class VectorSearchResult:
    chunk: Chunk
    score: float


class VectorIndexBuilder:
    def __init__(self, embedder: EmbeddingClient, index: "LocalVectorIndex") -> None:
        self.embedder = embedder
        self.vector_index = index

    def index_chunks(self, paper_id: str, chunks: list[Chunk]) -> Path:
        embeddings = self.embedder.embed_texts([self._text_for_embedding(chunk) for chunk in chunks])
        records = [VectorRecord(chunk=chunk, embedding=embedding) for chunk, embedding in zip(chunks, embeddings)]
        return self.vector_index.write(paper_id, self.embedder.model_name, records)

    def index(self, paper_id: str, chunks: list[Chunk]) -> Path:
        return self.index_chunks(paper_id, chunks)

    def _text_for_embedding(self, chunk: Chunk) -> str:
        metadata_terms = [
            *chunk.metadata.satellites,
            *chunk.metadata.sensors,
            *chunk.metadata.bands_or_layers,
            *chunk.metadata.indices,
            *chunk.metadata.target_classes,
            *chunk.metadata.thresholds,
            *chunk.metadata.evidence_types,
        ]
        return "\n".join([chunk.section_title, chunk.chunk_type, chunk.text, " ".join(metadata_terms)])


class LocalVectorIndex:
    def __init__(self, index_dir: Path) -> None:
        self.index_dir = index_dir

    def write(self, paper_id: str, model_name: str, records: list[VectorRecord]) -> Path:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        chunker_version = records[0].chunk.chunker_version if records else CHUNKER_VERSION
        path = self.index_path(paper_id, model_name=model_name, chunker_version=chunker_version)
        with path.open("w", encoding="utf-8") as handle:
            header = self._header_payload(paper_id, model_name, records)
            handle.write(json.dumps(header, ensure_ascii=False) + "\n")
            for record in records:
                payload = {
                    "type": "record",
                    "chunk": self._chunk_to_dict(record.chunk),
                    "embedding": record.embedding,
                }
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        return path

    def load(
        self,
        paper_id: str,
        model_name: str | None = None,
        chunker_version: str | None = None,
    ) -> list[VectorRecord]:
        return self._load_path(self._resolve_index_path(paper_id, model_name, chunker_version))

    def search(
        self,
        query: str,
        embedder: EmbeddingClient,
        paper_id: str | None = None,
        top_k: int = 5,
        chunk_types: list[str] | None = None,
    ) -> list[VectorSearchResult]:
        records = self.load(paper_id, model_name=embedder.model_name, chunker_version=CHUNKER_VERSION) if paper_id else self.load_all()
        allowed_types = set(chunk_types or [])
        query_embedding = embedder.embed_query(query)
        results: list[VectorSearchResult] = []
        for record in records:
            if allowed_types and record.chunk.chunk_type not in allowed_types:
                continue
            if self._is_low_quality_search_result(record.chunk):
                continue
            score = dot_product(query_embedding, record.embedding)
            if score <= 0:
                continue
            results.append(VectorSearchResult(chunk=record.chunk, score=round(score, 6)))
        results.sort(key=lambda result: result.score, reverse=True)
        return results[:top_k]

    def load_all(self) -> list[VectorRecord]:
        records: list[VectorRecord] = []
        for path in self.index_dir.glob("*.jsonl"):
            records.extend(self._load_path(path))
        return records

    def exists(
        self,
        paper_id: str,
        model_name: str | None = None,
        chunker_version: str | None = None,
        parser_version: str | None = None,
        vocabulary_version: str | None = None,
        source_hash: str | None = None,
    ) -> bool:
        path = self._resolve_index_path(paper_id, model_name, chunker_version)
        if not path.exists():
            return False
        expected = {
            "paper_id": paper_id,
            "model_name": model_name,
            "chunker_version": chunker_version,
            "parser_version": parser_version,
            "vocabulary_version": vocabulary_version,
            "source_hash": source_hash,
        }
        header = self.header(paper_id, model_name=model_name, chunker_version=chunker_version)
        return all(value is None or header.get(key) == value for key, value in expected.items())

    def header(
        self,
        paper_id: str,
        model_name: str | None = None,
        chunker_version: str | None = None,
    ) -> dict[str, object]:
        path = self._resolve_index_path(paper_id, model_name, chunker_version)
        with path.open("r", encoding="utf-8") as handle:
            payload = json.loads(handle.readline())
        return payload if isinstance(payload, dict) else {}

    def index_path(
        self,
        paper_id: str,
        model_name: str | None = None,
        chunker_version: str | None = None,
    ) -> Path:
        safe_paper_id = self._safe_name(paper_id)
        if model_name and chunker_version:
            return self.index_dir / f"{safe_paper_id}__{self._safe_name(model_name)}__{self._safe_name(chunker_version)}.jsonl"
        return self.index_dir / f"{safe_paper_id}.jsonl"

    def _header_payload(self, paper_id: str, model_name: str, records: list[VectorRecord]) -> dict[str, object]:
        first_chunk = records[0].chunk if records else None
        return {
            "type": "header",
            "paper_id": paper_id,
            "model_name": model_name,
            "record_count": len(records),
            "parser_version": first_chunk.parser_version if first_chunk else None,
            "chunker_version": first_chunk.chunker_version if first_chunk else CHUNKER_VERSION,
            "vocabulary_version": first_chunk.vocabulary_version if first_chunk else None,
            "source_hash": first_chunk.source_hash if first_chunk else None,
        }

    def _resolve_index_path(
        self,
        paper_id: str,
        model_name: str | None,
        chunker_version: str | None,
    ) -> Path:
        exact_path = self.index_path(paper_id, model_name=model_name, chunker_version=chunker_version)
        if model_name or chunker_version or exact_path.exists():
            return exact_path
        safe_paper_id = self._safe_name(paper_id)
        matches = sorted(self.index_dir.glob(f"{safe_paper_id}__*.jsonl"))
        if len(matches) == 1:
            return matches[0]
        return exact_path

    def _load_path(self, path: Path) -> list[VectorRecord]:
        records: list[VectorRecord] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                payload = json.loads(line)
                if payload.get("type") != "record":
                    continue
                records.append(
                    VectorRecord(
                        chunk=self._chunk_from_dict(payload["chunk"]),
                        embedding=[float(value) for value in payload["embedding"]],
                    )
                )
        return records

    def _is_low_quality_search_result(self, chunk: Chunk) -> bool:
        if chunk.chunk_type != "paragraph_child":
            return False
        if len(chunk.text.strip()) >= 80:
            return False
        return not (
            chunk.metadata.thresholds
            or chunk.metadata.bands_or_layers
            or chunk.metadata.indices
            or chunk.metadata.evidence_types
        )

    def _safe_name(self, value: str) -> str:
        return "".join(character if character.isalnum() or character in {"-", "_"} else "_" for character in value)

    def _chunk_to_dict(self, chunk: Chunk) -> dict[str, object]:
        return {
            "chunk_id": chunk.chunk_id,
            "paper_id": chunk.paper_id,
            "chunk_type": chunk.chunk_type,
            "text": chunk.text,
            "parent_id": chunk.parent_id,
            "child_ids": chunk.child_ids,
            "source_block_ids": chunk.source_block_ids,
            "page_start": chunk.page_start,
            "page_end": chunk.page_end,
            "section_title": chunk.section_title,
            "section_type": chunk.section_type,
            "metadata": chunk.metadata.__dict__,
            "retrieval_profile": chunk.retrieval_profile,
            "parser_version": chunk.parser_version,
            "chunker_version": chunk.chunker_version,
            "vocabulary_version": chunk.vocabulary_version,
            "source_hash": chunk.source_hash,
        }

    def _chunk_from_dict(self, payload: dict[str, object]) -> Chunk:
        metadata_payload = payload["metadata"]
        if not isinstance(metadata_payload, dict):
            metadata_payload = {}
        return Chunk(
            chunk_id=str(payload["chunk_id"]),
            paper_id=str(payload["paper_id"]),
            chunk_type=str(payload["chunk_type"]),
            text=str(payload["text"]),
            parent_id=payload["parent_id"] if isinstance(payload["parent_id"], str) else None,
            child_ids=list(payload["child_ids"]),
            source_block_ids=list(payload["source_block_ids"]),
            page_start=payload["page_start"] if isinstance(payload["page_start"], int) else None,
            page_end=payload["page_end"] if isinstance(payload["page_end"], int) else None,
            section_title=str(payload["section_title"]),
            section_type=str(payload["section_type"]),
            metadata=ChunkMetadata(**metadata_payload),
            retrieval_profile=str(payload["retrieval_profile"]),
            parser_version=str(payload["parser_version"]),
            chunker_version=str(payload["chunker_version"]),
            vocabulary_version=str(payload["vocabulary_version"]),
            source_hash=str(payload["source_hash"]),
        )
