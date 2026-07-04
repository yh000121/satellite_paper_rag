from __future__ import annotations

import re
from pathlib import Path

from satellite_paper_rag.parsing.text_parser import detect_block_type, normalize_section_type
from satellite_paper_rag.schemas import (
    PARSER_VERSION,
    VOCABULARY_VERSION,
    Paper,
    PaperBlock,
    PaperSection,
    compute_source_hash,
)


class MarkdownPaperParser:
    def parse(self, source: Path) -> Paper:
        raw_text = source.read_text(encoding="utf-8")
        lines = raw_text.splitlines()
        title = source.stem
        sections: list[PaperSection] = []
        current_title = "Front Matter"
        current_type = "unknown"
        current_level = 1
        current_blocks: list[PaperBlock] = []
        paragraph_lines: list[str] = []
        block_index = 0

        def flush_paragraph() -> None:
            nonlocal paragraph_lines, block_index
            text = " ".join(line.strip() for line in paragraph_lines if line.strip()).strip()
            if text:
                current_blocks.append(
                    PaperBlock(
                        block_id=f"block_{block_index:04d}",
                        text=text,
                        block_type=detect_block_type(text),
                        page_start=None,
                        page_end=None,
                        order_index=block_index,
                        metadata={},
                    )
                )
                block_index += 1
            paragraph_lines = []

        def flush_section() -> None:
            nonlocal current_blocks
            flush_paragraph()
            sections.append(
                PaperSection(
                    section_id=f"section_{len(sections):03d}",
                    title=current_title,
                    normalized_type=current_type,
                    level=current_level,
                    blocks=current_blocks,
                )
            )
            current_blocks = []

        for line in lines:
            heading = re.match(r"^(#{1,6})\s+(.+)$", line.strip())
            if heading:
                flush_paragraph()
                level = len(heading.group(1))
                heading_text = heading.group(2).strip()
                if level == 1 and title == source.stem:
                    title = heading_text
                    continue
                if current_blocks or sections:
                    flush_section()
                current_title = heading_text
                current_type = normalize_section_type(heading_text)
                current_level = level
                continue
            if not line.strip():
                flush_paragraph()
            else:
                paragraph_lines.append(line)

        flush_section()

        return Paper(
            paper_id=source.stem,
            title=title,
            authors=[],
            year=None,
            source_path=str(source),
            source_hash=compute_source_hash(raw_text),
            source_type="markdown",
            sections=sections,
            metadata={},
            parser_version=PARSER_VERSION,
            vocabulary_version=VOCABULARY_VERSION,
        )
