from __future__ import annotations

import re
from pathlib import Path

from satellite_paper_rag.schemas import (
    PARSER_VERSION,
    VOCABULARY_VERSION,
    Paper,
    PaperBlock,
    PaperSection,
    compute_source_hash,
)


SECTION_TYPES = {
    "abstract": "abstract",
    "introduction": "introduction",
    "methods": "method",
    "method": "method",
    "dataset": "dataset",
    "study area": "study_area",
    "experiments": "experiment",
    "experiment": "experiment",
    "results": "result",
    "result": "result",
    "discussion": "discussion",
    "conclusion": "conclusion",
    "references": "reference",
}


def normalize_section_type(title: str) -> str:
    cleaned = title.strip().lower().rstrip(":")
    return SECTION_TYPES.get(cleaned, "unknown")


def detect_block_type(text: str) -> str:
    stripped = text.strip()
    if re.match(r"^(figure|fig\.)\s+\d+", stripped, re.IGNORECASE):
        return "figure_caption"
    if re.match(r"^table\s+\d+", stripped, re.IGNORECASE):
        return "table_caption"
    return "paragraph"


class TextPaperParser:
    def parse(self, source: Path) -> Paper:
        raw_text = source.read_text(encoding="utf-8")
        parts = [part.strip() for part in re.split(r"\n\s*\n", raw_text) if part.strip()]
        title = parts[0] if parts else source.stem
        sections: list[PaperSection] = []
        current_title = "Front Matter"
        current_type = "unknown"
        current_blocks: list[PaperBlock] = []
        block_index = 0

        def flush_section() -> None:
            nonlocal current_blocks
            sections.append(
                PaperSection(
                    section_id=f"section_{len(sections):03d}",
                    title=current_title,
                    normalized_type=current_type,
                    level=1,
                    blocks=current_blocks,
                )
            )
            current_blocks = []

        for part in parts[1:]:
            normalized = normalize_section_type(part)
            if normalized != "unknown" and len(part.split()) <= 4:
                if current_blocks or not sections:
                    flush_section()
                current_title = part
                current_type = normalized
                continue
            current_blocks.append(
                PaperBlock(
                    block_id=f"block_{block_index:04d}",
                    text=part,
                    block_type=detect_block_type(part),
                    page_start=None,
                    page_end=None,
                    order_index=block_index,
                    metadata={},
                )
            )
            block_index += 1

        if current_blocks or not sections:
            flush_section()

        return Paper(
            paper_id=source.stem,
            title=title,
            authors=[],
            year=None,
            source_path=str(source),
            source_hash=compute_source_hash(raw_text),
            source_type="text",
            sections=sections,
            metadata={},
            parser_version=PARSER_VERSION,
            vocabulary_version=VOCABULARY_VERSION,
        )
