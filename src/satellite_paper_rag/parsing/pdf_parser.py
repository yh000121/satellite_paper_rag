from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from satellite_paper_rag.parsing.text_parser import detect_block_type, normalize_section_type
from satellite_paper_rag.schemas import (
    PARSER_VERSION,
    VOCABULARY_VERSION,
    Paper,
    PaperBlock,
    PaperSection,
    PaperTable,
    ParseQualityReport,
    compute_source_hash,
)


@dataclass(frozen=True)
class RawPdfLine:
    page_number: int
    text: str
    bbox: tuple[float, float, float, float] | None
    source_kind: str = "text"


class PdfPaperParser:
    def parse(self, source: Path) -> Paper:
        try:
            import fitz
        except ImportError as exc:
            raise RuntimeError("PyMuPDF is required for PDF parsing. Install package 'pymupdf'.") from exc

        document = fitz.open(source)
        raw_lines: list[RawPdfLine] = []
        raw_pages: list[str] = []
        warnings: list[str] = []
        image_only_pages: list[int] = []
        for page_index, page in enumerate(document, start=1):
            page_text = page.get_text("text")
            raw_pages.append(page_text)
            lines = self._extract_page_lines(page, page_index)
            raw_lines.extend(lines)
            if not page_text.strip() and page.get_images(full=True):
                image_only_pages.append(page_index)
                warnings.append(f"page {page_index} contains images but no extractable text; OCR may be required")

        metadata_title = (document.metadata or {}).get("title") or ""
        document.close()

        library_tables, table_warnings = self._extract_pdfplumber_tables(source)
        warnings.extend(table_warnings)
        merged_lines, fallback_tables, captions_detected, tables_detected = self._merge_captions_and_tables(raw_lines)
        merged_lines = self._reflow_text_lines(merged_lines)
        tables = library_tables or fallback_tables
        tables_detected = max(tables_detected, len(tables))
        raw_text = "\n\n".join(raw_pages)
        title = self._detect_title(merged_lines, source, metadata_title)
        sections = self._build_sections(merged_lines, title)
        quality_report = ParseQualityReport(
            tables_detected=tables_detected,
            tables_extracted=len(tables),
            captions_detected=captions_detected,
            image_only_pages=image_only_pages,
            warnings=warnings,
        )

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
            tables=tables,
            quality_report=quality_report,
            parser_version=PARSER_VERSION,
            vocabulary_version=VOCABULARY_VERSION,
        )

    def _extract_page_lines(self, page, page_number: int) -> list[RawPdfLine]:
        blocks = sorted(page.get_text("blocks"), key=lambda block: (round(block[1], 1), round(block[0], 1)))
        lines: list[RawPdfLine] = []
        for block in blocks:
            x0, y0, x1, y1, text = block[:5]
            bbox = (float(x0), float(y0), float(x1), float(y1))
            for line in text.splitlines():
                cleaned = self._clean_line(line)
                if cleaned and not self._is_page_artifact(cleaned):
                    lines.append(RawPdfLine(page_number=page_number, text=cleaned, bbox=bbox))
        return lines

    def _merge_captions_and_tables(
        self, lines: list[RawPdfLine]
    ) -> tuple[list[RawPdfLine], list[PaperTable], int, int]:
        merged: list[RawPdfLine] = []
        tables: list[PaperTable] = []
        captions_detected = 0
        tables_detected = 0
        index = 0
        while index < len(lines):
            line = lines[index]
            block_type = detect_block_type(line.text)
            if block_type == "figure_caption":
                caption_lines = [line]
                index += 1
                while index < len(lines) and self._is_caption_continuation(lines[index].text):
                    caption_lines.append(lines[index])
                    index += 1
                captions_detected += 1
                merged.append(self._combine_lines(caption_lines, "figure_caption"))
                continue
            if block_type == "table_caption":
                table_lines = [line]
                index += 1
                while index < len(lines) and self._is_table_continuation(lines[index].text):
                    table_lines.append(lines[index])
                    index += 1
                tables_detected += 1
                captions_detected += 1
                table_block = self._combine_lines(table_lines, "table_text")
                merged.append(table_block)
                rows = self._table_rows(table_lines)
                tables.append(
                    PaperTable(
                        table_id=f"table_{len(tables):03d}",
                        caption=table_lines[0].text,
                        rows=rows,
                        page_start=table_lines[0].page_number,
                        page_end=table_lines[-1].page_number,
                        bbox=self._merge_bbox([line.bbox for line in table_lines]),
                        extraction_method="pymupdf-block-fallback",
                        metadata={"source_block_type": "table_text"},
                    )
                )
                continue
            merged.append(line)
            index += 1
        return merged, tables, captions_detected, tables_detected

    def _build_sections(self, lines: list[RawPdfLine], title: str) -> list[PaperSection]:
        sections: list[PaperSection] = []
        current_title = "Front Matter"
        current_type = "unknown"
        current_level = 1
        current_blocks: list[PaperBlock] = []
        block_index = 0

        def flush_section() -> None:
            nonlocal current_blocks
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
            if line.text == title:
                continue
            heading_level = self._heading_level(line.text)
            normalized = normalize_section_type(line.text)
            if heading_level is not None and normalized != "unknown":
                if current_blocks or not sections:
                    flush_section()
                current_title = line.text
                current_type = normalized
                current_level = heading_level
                continue
            block_type = detect_block_type(line.text)
            if line.source_kind == "table_text":
                block_type = "table_text"
            current_blocks.append(
                PaperBlock(
                    block_id=f"block_{block_index:04d}",
                    text=line.text,
                    block_type=block_type,
                    page_start=line.page_number,
                    page_end=line.page_number,
                    order_index=block_index,
                    metadata={
                        "bbox": line.bbox,
                        "source_kind": line.source_kind,
                    },
                )
            )
            block_index += 1

        if current_blocks or not sections:
            flush_section()
        return sections

    def _detect_title(self, lines: list[RawPdfLine], source: Path, metadata_title: str) -> str:
        if metadata_title and not metadata_title.lower().startswith("remote sensing of environment"):
            return metadata_title.strip()
        for line in lines:
            if line.text.strip() and not self._heading_level(line.text):
                return line.text.strip()
        return source.stem

    def _heading_level(self, text: str) -> int | None:
        stripped = text.strip()
        match = re.match(r"^(\d+(?:\.\d+)*)\s+[A-Z][A-Za-z0-9,()/&:\- ]+$", stripped)
        if not match:
            if re.search(r"[.!?]$", stripped):
                return None
            if normalize_section_type(stripped) != "unknown" and len(stripped.split()) <= 5:
                return 1
            return None
        return match.group(1).count(".") + 1

    def _combine_lines(self, lines: list[RawPdfLine], source_kind: str) -> RawPdfLine:
        return RawPdfLine(
            page_number=lines[0].page_number,
            text=" ".join(line.text for line in lines),
            bbox=self._merge_bbox([line.bbox for line in lines]),
            source_kind=source_kind,
        )

    def _reflow_text_lines(self, lines: list[RawPdfLine]) -> list[RawPdfLine]:
        reflowed: list[RawPdfLine] = []
        paragraph_lines: list[RawPdfLine] = []

        def flush_paragraph() -> None:
            nonlocal paragraph_lines
            if paragraph_lines:
                reflowed.append(self._combine_lines(paragraph_lines, "text"))
                paragraph_lines = []

        for line in lines:
            if not self._can_reflow_line(line):
                flush_paragraph()
                reflowed.append(line)
                continue
            if paragraph_lines and not self._same_layout_block(paragraph_lines[-1], line):
                flush_paragraph()
            paragraph_lines.append(line)
        flush_paragraph()
        return reflowed

    def _can_reflow_line(self, line: RawPdfLine) -> bool:
        if line.source_kind != "text":
            return False
        if detect_block_type(line.text) != "paragraph":
            return False
        if self._heading_level(line.text) is not None:
            return False
        return True

    def _same_layout_block(self, previous: RawPdfLine, current: RawPdfLine) -> bool:
        if previous.page_number != current.page_number:
            return False
        if previous.bbox is None or current.bbox is None:
            return False
        return all(abs(left - right) < 1.0 for left, right in zip(previous.bbox, current.bbox))

    def _table_rows(self, lines: list[RawPdfLine]) -> list[list[str]]:
        rows: list[list[str]] = []
        for line in lines[1:]:
            cells = [cell for cell in re.split(r"\s{2,}|\t", line.text.strip()) if cell]
            rows.append(cells or [line.text.strip()])
        return rows

    def _extract_pdfplumber_tables(self, source: Path) -> tuple[list[PaperTable], list[str]]:
        try:
            import pdfplumber
        except ImportError:
            return [], []

        tables: list[PaperTable] = []
        warnings: list[str] = []
        try:
            with pdfplumber.open(str(source)) as pdf:
                for page_index, page in enumerate(pdf.pages, start=1):
                    for extracted_table in page.extract_tables() or []:
                        rows = self._clean_pdfplumber_rows(extracted_table)
                        if not rows:
                            continue
                        tables.append(
                            PaperTable(
                                table_id=f"pdfplumber_table_{len(tables):03d}",
                                caption=f"Table extracted from page {page_index}",
                                rows=rows,
                                page_start=page_index,
                                page_end=page_index,
                                extraction_method="pdfplumber",
                                metadata={"source_block_type": "table_text"},
                            )
                        )
        except Exception as exc:  # pragma: no cover - depends on third-party PDF heuristics.
            warnings.append(f"pdfplumber table extraction failed: {exc}")
        return tables, warnings

    def _clean_pdfplumber_rows(self, extracted_table: list[list[str | None]]) -> list[list[str]]:
        rows: list[list[str]] = []
        for row in extracted_table:
            cleaned = [self._clean_line(cell or "") for cell in row]
            if any(cleaned):
                rows.append(cleaned)
        return rows

    def _is_caption_continuation(self, text: str) -> bool:
        if detect_block_type(text) != "paragraph":
            return False
        if self._heading_level(text) is not None:
            return False
        return not text.startswith(("Table ", "References", "C.E. "))

    def _is_table_continuation(self, text: str) -> bool:
        if detect_block_type(text) != "paragraph":
            return False
        if self._heading_level(text) is not None:
            return False
        if text.startswith(("Fig.", "Figure", "References", "C.E. ")):
            return False
        return True

    def _merge_bbox(self, bboxes: list[tuple[float, float, float, float] | None]) -> tuple[float, float, float, float] | None:
        present = [bbox for bbox in bboxes if bbox is not None]
        if not present:
            return None
        return (
            min(bbox[0] for bbox in present),
            min(bbox[1] for bbox in present),
            max(bbox[2] for bbox in present),
            max(bbox[3] for bbox in present),
        )

    def _clean_line(self, line: str) -> str:
        cleaned = re.sub(r"\s+", " ", line).strip()
        replacements = {
            "\u00ad": "",
            "老": "ρ",
            "¦Ñ": "ρ",
            "¦Ěm": "μm",
            "每": "-",
            "＆": "'",
            "＊": "'",
        }
        for source, target in replacements.items():
            cleaned = cleaned.replace(source, target)
        return cleaned

    def _is_page_artifact(self, text: str) -> bool:
        if re.fullmatch(r"\d{1,3}", text):
            return True
        if text.startswith("Remote Sensing of Environment"):
            return True
        return False
