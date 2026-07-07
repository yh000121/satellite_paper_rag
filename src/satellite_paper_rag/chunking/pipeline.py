from __future__ import annotations

import re

from satellite_paper_rag.chunking.metadata_enricher import MetadataEnricher
from satellite_paper_rag.chunking.recursive_splitter import RecursiveSplitterAdapter
from satellite_paper_rag.config import ChunkingConfig
from satellite_paper_rag.domain.vocabulary import DomainVocabulary
from satellite_paper_rag.schemas import CHUNKER_VERSION, Chunk, Paper, PaperBlock, PaperSection


class PaperChunkingPipeline:
    def __init__(self, vocabulary: DomainVocabulary, config: ChunkingConfig | None = None) -> None:
        self.config = config or ChunkingConfig()
        self.enricher = MetadataEnricher(vocabulary)
        self.recursive_splitter = RecursiveSplitterAdapter(self.config)

    def chunk(self, paper: Paper) -> list[Chunk]:
        chunks: list[Chunk] = [self._paper_summary_chunk(paper)]
        for section in paper.sections:
            parent = self._section_parent_chunk(paper, section_index=len(chunks), section=section)
            parent_index = len(chunks)
            chunks.append(parent)
            child_ids: list[str] = []
            for block_index, block in enumerate(section.blocks):
                for paragraph in self._paragraph_child_chunks(paper, parent.chunk_id, block, section.title, section.normalized_type):
                    chunks.append(paragraph)
                    child_ids.append(paragraph.chunk_id)
                for window in self._sentence_window_chunks(paper, parent.chunk_id, block, section.title, section.normalized_type):
                    chunks.append(window)
                    child_ids.append(window.chunk_id)
                if block.block_type in {"figure_caption", "table_caption", "table_text"}:
                    figure_table = self._figure_table_chunk(paper, parent.chunk_id, block, section.title, section.normalized_type)
                    chunks.append(figure_table)
                    child_ids.append(figure_table.chunk_id)
                rule = self._rule_candidate_chunk(
                    paper,
                    parent.chunk_id,
                    section.blocks,
                    block_index,
                    section.title,
                    section.normalized_type,
                )
                if rule is not None:
                    chunks.append(rule)
                    child_ids.append(rule.chunk_id)
            if child_ids:
                chunks[parent_index] = self._replace_child_ids(parent, child_ids)
        return chunks

    def _paper_summary_chunk(self, paper: Paper) -> Chunk:
        summary_blocks = [
            block.text
            for section in paper.sections
            if section.normalized_type in {"abstract", "conclusion"}
            for block in section.blocks
        ]
        text = f"{paper.title}\n\n" + "\n\n".join(summary_blocks)
        return self._make_chunk(paper, "paper_summary", text, None, [], None, None, "Paper Summary", "title", "paper_routing")

    def _section_parent_chunk(self, paper: Paper, section_index: int, section: PaperSection) -> Chunk:
        text = "\n\n".join(block.text for block in section.blocks)
        return self._make_chunk(
            paper,
            "section_parent",
            text,
            None,
            [block.block_id for block in section.blocks],
            self._min_page(section.blocks),
            self._max_page(section.blocks),
            section.title,
            section.normalized_type,
            "broad_context",
            suffix=f"{section_index:04d}",
        )

    def _paragraph_child_chunks(self, paper: Paper, parent_id: str, block: PaperBlock, section_title: str, section_type: str) -> list[Chunk]:
        if self._is_low_information_paragraph(block.text):
            return []
        parts = [block.text]
        if len(block.text) > self.config.max_child_chars:
            parts = self.recursive_splitter.split(block.text)
        chunks: list[Chunk] = []
        for part_index, part in enumerate(parts):
            suffix = block.block_id if len(parts) == 1 else f"{block.block_id}_part_{part_index:03d}"
            chunks.append(
                self._make_chunk(
                    paper,
                    "paragraph_child",
                    part,
                    parent_id,
                    [block.block_id],
                    block.page_start,
                    block.page_end,
                    section_title,
                    section_type,
                    "semantic_recall",
                    suffix=suffix,
                )
            )
        return chunks

    def _sentence_window_chunks(self, paper: Paper, parent_id: str, block: PaperBlock, section_title: str, section_type: str) -> list[Chunk]:
        sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", block.text) if sentence.strip()]
        chunks: list[Chunk] = []
        for index, sentence in enumerate(sentences):
            metadata = self.enricher.enrich(sentence)
            if not (metadata.target_classes or metadata.bands_or_layers or metadata.thresholds):
                continue
            start = max(index - self.config.sentence_window_radius, 0)
            end = min(index + self.config.sentence_window_radius + 1, len(sentences))
            text = " ".join(sentences[start:end])
            chunks.append(
                self._make_chunk(
                    paper,
                    "sentence_window_child",
                    text,
                    parent_id,
                    [block.block_id],
                    block.page_start,
                    block.page_end,
                    section_title,
                    section_type,
                    "precise_evidence",
                    suffix=f"{block.block_id}_sent_{index:03d}",
                )
            )
        return chunks

    def _figure_table_chunk(self, paper: Paper, parent_id: str, block: PaperBlock, section_title: str, section_type: str) -> Chunk:
        return self._make_chunk(
            paper,
            "figure_table",
            block.text,
            parent_id,
            [block.block_id],
            block.page_start,
            block.page_end,
            section_title,
            section_type,
            "table_metric",
            suffix=block.block_id,
        )

    def _rule_candidate_chunk(
        self,
        paper: Paper,
        parent_id: str,
        blocks: list[PaperBlock],
        block_index: int,
        section_title: str,
        section_type: str,
    ) -> Chunk | None:
        block = blocks[block_index]
        context_blocks = self._rule_context_blocks(blocks, block_index)
        context_text = " ".join(context_block.text for context_block in context_blocks)
        metadata = self.enricher.enrich(context_text)
        signal_count = sum(
            1
            for value in [
                metadata.satellites or metadata.sensors,
                metadata.bands_or_layers or metadata.indices,
                metadata.target_classes,
                metadata.thresholds,
                metadata.directionality,
                metadata.evidence_types,
            ]
            if value
        )
        if signal_count < self.config.min_rule_signal_count:
            return None
        return self._make_chunk(
            paper,
            "rule_candidate",
            context_text,
            parent_id,
            [context_block.block_id for context_block in context_blocks],
            self._min_page(context_blocks),
            self._max_page(context_blocks),
            section_title,
            section_type,
            "rule_extraction",
            suffix=block.block_id,
        )

    def _rule_context_blocks(self, blocks: list[PaperBlock], block_index: int) -> list[PaperBlock]:
        start = max(block_index - self.config.rule_context_window_blocks, 0)
        end = min(block_index + self.config.rule_context_window_blocks + 1, len(blocks))
        return [
            block
            for block in blocks[start:end]
            if block.block_type not in {"table_text", "table_caption"} or block_index == blocks.index(block)
        ]

    def _is_low_information_paragraph(self, text: str) -> bool:
        if len(text.strip()) >= self.config.min_paragraph_child_chars:
            return False
        metadata = self.enricher.enrich(text)
        return not (metadata.thresholds or metadata.bands_or_layers or metadata.indices or metadata.evidence_types)

    def _make_chunk(
        self,
        paper: Paper,
        chunk_type: str,
        text: str,
        parent_id: str | None,
        source_block_ids: list[str],
        page_start: int | None,
        page_end: int | None,
        section_title: str,
        section_type: str,
        retrieval_profile: str,
        suffix: str | None = None,
    ) -> Chunk:
        chunk_id = f"{paper.paper_id}_{chunk_type}_{suffix or '0000'}"
        return Chunk(
            chunk_id=chunk_id,
            paper_id=paper.paper_id,
            chunk_type=chunk_type,
            text=text,
            parent_id=parent_id,
            child_ids=[],
            source_block_ids=source_block_ids,
            page_start=page_start,
            page_end=page_end,
            section_title=section_title,
            section_type=section_type,
            metadata=self.enricher.enrich(f"{paper.title}\n{section_title}\n{text}"),
            retrieval_profile=retrieval_profile,
            parser_version=paper.parser_version,
            chunker_version=CHUNKER_VERSION,
            vocabulary_version=paper.vocabulary_version,
            source_hash=paper.source_hash,
        )

    def _replace_child_ids(self, chunk: Chunk, child_ids: list[str]) -> Chunk:
        return Chunk(
            chunk_id=chunk.chunk_id,
            paper_id=chunk.paper_id,
            chunk_type=chunk.chunk_type,
            text=chunk.text,
            parent_id=chunk.parent_id,
            child_ids=child_ids,
            source_block_ids=chunk.source_block_ids,
            page_start=chunk.page_start,
            page_end=chunk.page_end,
            section_title=chunk.section_title,
            section_type=chunk.section_type,
            metadata=chunk.metadata,
            retrieval_profile=chunk.retrieval_profile,
            parser_version=chunk.parser_version,
            chunker_version=chunk.chunker_version,
            vocabulary_version=chunk.vocabulary_version,
            source_hash=chunk.source_hash,
        )

    def _min_page(self, blocks: list[PaperBlock]) -> int | None:
        pages = [block.page_start for block in blocks if block.page_start is not None]
        return min(pages) if pages else None

    def _max_page(self, blocks: list[PaperBlock]) -> int | None:
        pages = [block.page_end for block in blocks if block.page_end is not None]
        return max(pages) if pages else None
