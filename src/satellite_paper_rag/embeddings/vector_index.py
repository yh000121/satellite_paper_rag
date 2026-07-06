from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from satellite_paper_rag.embeddings.client import EmbeddingClient, dot_product
from satellite_paper_rag.schemas import Chunk, ChunkMetadata


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
        path = self.index_path(paper_id)
        with path.open("w", encoding="utf-8") as handle:
            header = {"type": "header", "paper_id": paper_id, "model_name": model_name, "record_count": len(records)}
            handle.write(json.dumps(header, ensure_ascii=False) + "\n")
            for record in records:
                payload = {
                    "type": "record",
                    "chunk": self._chunk_to_dict(record.chunk),
                    "embedding": record.embedding,
                }
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        return path

    def load(self, paper_id: str) -> list[VectorRecord]:
        path = self.index_path(paper_id)
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

    def search(
        self,
        query: str,
        embedder: EmbeddingClient,
        paper_id: str | None = None,
        top_k: int = 5,
        chunk_types: list[str] | None = None,
    ) -> list[VectorSearchResult]:
        records = self.load(paper_id) if paper_id else self.load_all()
        allowed_types = set(chunk_types or [])
        query_embedding = embedder.embed_query(query)
        results: list[VectorSearchResult] = []
        for record in records:
            if allowed_types and record.chunk.chunk_type not in allowed_types:
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
            records.extend(self.load(path.stem))
        return records

    def exists(self, paper_id: str) -> bool:
        return self.index_path(paper_id).exists()

    def index_path(self, paper_id: str) -> Path:
        return self.index_dir / f"{self._safe_name(paper_id)}.jsonl"

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
