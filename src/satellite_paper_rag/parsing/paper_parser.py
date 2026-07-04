from __future__ import annotations

from pathlib import Path
from typing import Protocol

from satellite_paper_rag.schemas import Paper


class PaperParser(Protocol):
    def parse(self, source: Path) -> Paper:
        ...
