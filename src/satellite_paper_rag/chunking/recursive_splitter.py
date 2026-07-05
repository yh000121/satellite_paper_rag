from __future__ import annotations

from satellite_paper_rag.config import ChunkingConfig


class RecursiveSplitterAdapter:
    def __init__(self, config: ChunkingConfig) -> None:
        self.config = config
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter
        except ImportError as exc:
            raise RuntimeError(
                "langchain-text-splitters is required for recursive fallback splitting."
            ) from exc
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.recursive_chunk_size,
            chunk_overlap=config.recursive_chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def split(self, text: str) -> list[str]:
        if len(text) <= self.config.recursive_chunk_size:
            return [text]
        return [part.strip() for part in self._splitter.split_text(text) if part.strip()]
