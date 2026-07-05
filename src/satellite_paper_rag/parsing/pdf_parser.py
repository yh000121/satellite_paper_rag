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


class PdfPaperParser:
    def parse(self, source: Path) -> Paper:
        try:
            import fitz
        except ImportError as exc:
            raise RuntimeError("PyMuPDF is required for PDF parsing. Install package 'pymupdf'.") from exc

        document = fitz.open(source)
        page_paragraphs: list[tuple[int, str]] = []
        raw_pages: list[str] = []
        for page_index, page in enumerate(document, start=1):
            page_text = page.get_text("text")
            raw_pages.append(page_text)
            for paragraph in self._split_page_text(page_text):
                page_paragraphs.append((page_index, paragraph))
        document.close()

        raw_text = "\n\n".join(raw_pages)
        title = self._detect_title(page_paragraphs, source)
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

        for page_number, paragraph in page_paragraphs:
            if paragraph == title:
                continue
            normalized = normalize_section_type(paragraph)
            if normalized != "unknown" and len(paragraph.split()) <= 5:
                if current_blocks or not sections:
                    flush_section()
                current_title = paragraph
                current_type = normalized
                continue
            current_blocks.append(
                PaperBlock(
                    block_id=f"block_{block_index:04d}",
                    text=paragraph,
                    block_type=detect_block_type(paragraph),
                    page_start=page_number,
                    page_end=page_number,
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
            source_type="pdf",
            sections=sections,
            metadata={},
            parser_version=PARSER_VERSION,
            vocabulary_version=VOCABULARY_VERSION,
        )

    def _split_page_text(self, page_text: str) -> list[str]:
        normalized = re.sub(r"\r\n?", "\n", page_text)
        paragraphs = [part.strip() for part in re.split(r"\n\s*\n", normalized) if part.strip()]
        if len(paragraphs) <= 1:
            paragraphs = [line.strip() for line in normalized.splitlines() if line.strip()]
        return paragraphs

    def _detect_title(self, page_paragraphs: list[tuple[int, str]], source: Path) -> str:
        for _, paragraph in page_paragraphs:
            if paragraph.strip():
                return paragraph.strip()
        return source.stem
