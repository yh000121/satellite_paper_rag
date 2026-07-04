from __future__ import annotations

from pathlib import Path

from satellite_paper_rag.schemas import Paper


class PdfPaperParser:
    def parse(self, source: Path) -> Paper:
        raise NotImplementedError(
            "PDF parsing is a replaceable adapter boundary in phase 1. "
            "Use MarkdownPaperParser or TextPaperParser for stable phase-1 ingestion."
        )
