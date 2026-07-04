# Satellite Paper RAG Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first phase of `satellite_paper_rag`: English-only paper parsing, multi-granularity chunking, domain metadata enrichment, observation feature normalization, and mock retrieval with evidence-boundary behavior.

**Architecture:** Use standard-library Python dataclasses and deterministic rules first. The code separates schemas, vocabulary, parsing, chunking, metadata enrichment, observation normalization, and retrieval so later PDF parsers, vector stores, and LLM answer generation can be added without rewriting chunking.

**Tech Stack:** Python 3.10+, standard library, `unittest`, no network calls, no required third-party dependencies in phase 1.

---

## File Structure

Create this structure under `D:\CodexProject\satellite_paper_rag`:

```text
satellite_paper_rag/
  README.md
  pyproject.toml
  src/
    satellite_paper_rag/
      __init__.py
      schemas.py
      domain/
        __init__.py
        vocabulary.py
      parsing/
        __init__.py
        paper_parser.py
        text_parser.py
        markdown_parser.py
        pdf_parser.py
      chunking/
        __init__.py
        pipeline.py
        metadata_enricher.py
      retrieval/
        __init__.py
        contract.py
        mock_hybrid_retriever.py
      observations/
        __init__.py
        schema.py
        feature_normalizer.py
  tests/
    fixtures/
      sample_sentinel3_paper.md
      sample_landsat_paper.txt
    test_schemas.py
    test_vocabulary.py
    test_text_markdown_parsers.py
    test_metadata_enricher.py
    test_chunking_pipeline.py
    test_observation_normalizer.py
    test_mock_retriever.py
```

Responsibilities:

- `schemas.py`: paper, section, block, chunk, metadata, version constants, source hashing.
- `domain/vocabulary.py`: English-only remote-sensing vocabulary, sensor-band mapping, alias normalization.
- `parsing/*`: parser protocol, text parser, Markdown parser, PDF parser boundary.
- `chunking/metadata_enricher.py`: deterministic satellite, sensor, band, class, threshold, trend, limitation extraction.
- `chunking/pipeline.py`: paper summary, section parent, paragraph child, sentence window, figure/table, and rule candidate chunk creation.
- `retrieval/contract.py`: retrieval request/result dataclasses.
- `retrieval/mock_hybrid_retriever.py`: keyword + metadata + chunk type boosting + parent expansion + insufficient-evidence behavior.
- `observations/*`: observation sample schema and CSV/table-style feature normalization.

Git note: `D:\CodexProject` is not currently a git repository. Commit steps are documented as checkpoints only; run them after a repository is initialized.

---

### Task 1: Project Skeleton And Fixtures

**Files:**
- Create: `D:\CodexProject\satellite_paper_rag\README.md`
- Create: `D:\CodexProject\satellite_paper_rag\pyproject.toml`
- Create: package `__init__.py` files under `src/`
- Create: `D:\CodexProject\satellite_paper_rag\tests\fixtures\sample_sentinel3_paper.md`
- Create: `D:\CodexProject\satellite_paper_rag\tests\fixtures\sample_landsat_paper.txt`

- [ ] **Step 1: Write project metadata**

Create `pyproject.toml`:

```toml
[project]
name = "satellite-paper-rag"
version = "0.1.0"
description = "Evidence-oriented chunking foundation for satellite imagery paper RAG."
requires-python = ">=3.10"
dependencies = []

[tool.setuptools.packages.find]
where = ["src"]
```

- [ ] **Step 2: Write README**

Create `README.md`:

```markdown
# Satellite Paper RAG

Phase 1 builds an English-only evidence chunking foundation for satellite remote-sensing papers.

It supports user-provided PDF, text, or Markdown papers. It does not search Scholar, crawl the web, or download papers.

Phase 1 focuses on:

- structured paper parsing
- multi-granularity chunking
- parent-child evidence links
- domain metadata extraction
- source/version provenance
- observation feature normalization
- mock hybrid retrieval with insufficient-evidence behavior

Run tests:

```powershell
python -m unittest discover -s tests -v
```
```

- [ ] **Step 3: Create empty package files**

Create empty files:

```text
src/satellite_paper_rag/__init__.py
src/satellite_paper_rag/domain/__init__.py
src/satellite_paper_rag/parsing/__init__.py
src/satellite_paper_rag/chunking/__init__.py
src/satellite_paper_rag/retrieval/__init__.py
src/satellite_paper_rag/observations/__init__.py
```

- [ ] **Step 4: Add Sentinel-3 fixture**

Create `tests/fixtures/sample_sentinel3_paper.md`:

```markdown
# Sentinel-3 SLSTR Cloud, Sea Ice, and Open Water Discrimination

## Abstract

This study evaluates Sentinel-3 SLSTR observations for cloud, sea ice, and open water discrimination. Visible reflectance in S1-S6 and brightness temperature in S7-S9 are analyzed.

## Methods

For Sentinel-3 SLSTR images, sea ice generally shows higher reflectance in S1-S6 bands than open water. Cloud pixels present high visible reflectance and lower brightness temperature in S7-S9 thermal channels.

Pixels with BT below 270 K were frequently classified as cloud-contaminated in the study area. Thin cloud and mixed ice edge pixels remained ambiguous and required manual review.

## Results

Table 2. Feature contribution for cloud and sea ice separation.
Thermal brightness temperature features S7-S9 contributed strongly to cloud and ice separation, while visible reflectance was useful for separating sea ice from open water.

Figure 3. Cloud and sea ice examples from SLSTR scenes.
The cold cloud top cases show lower S8 and S9 brightness temperature than open water.

## Conclusion

Thermal channels and visible reflectance should be interpreted together for SLSTR cloud, sea ice, and open water analysis.
```

- [ ] **Step 5: Add Landsat fixture**

Create `tests/fixtures/sample_landsat_paper.txt`:

```text
Landsat-8 Thermal Cloud Screening

Abstract

This paper studies Landsat-8 OLI and TIRS imagery for cloud and land classification.

Methods

The TIRS B10 thermal band is useful for detecting cold cloud tops. NDVI greater than 0.3 is associated with vegetation in the tested region. Reflectance thresholds are scene dependent.

Results

Table 1. Feature importance for cloud screening.
B10 brightness temperature contributed more to cloud screening than visible bands in the tested scenes.

Discussion

Low solar angle and mixed pixels caused uncertainty in reflectance-based rules.
```

- [ ] **Step 6: Run discovery**

Run:

```powershell
python -m unittest discover -s tests -v
```

Expected: `Ran 0 tests` or import-related failure until test files exist. If Python is not on PATH, use the configured `transformer` environment Python.

Checkpoint: no git commit until a repository is initialized.

---

### Task 2: Core Schemas And Source Hashing

**Files:**
- Create: `D:\CodexProject\satellite_paper_rag\src\satellite_paper_rag\schemas.py`
- Create: `D:\CodexProject\satellite_paper_rag\tests\test_schemas.py`

- [ ] **Step 1: Write failing schema tests**

Create `tests/test_schemas.py`:

```python
import unittest

from satellite_paper_rag.schemas import (
    CHUNKER_VERSION,
    PARSER_VERSION,
    VOCABULARY_VERSION,
    Chunk,
    ChunkMetadata,
    Paper,
    PaperBlock,
    PaperSection,
    compute_source_hash,
)


class SchemaTest(unittest.TestCase):
    def test_source_hash_is_stable(self):
        self.assertEqual(compute_source_hash("same text"), compute_source_hash("same text"))
        self.assertNotEqual(compute_source_hash("same text"), compute_source_hash("other text"))

    def test_paper_and_chunk_keep_versions_and_provenance(self):
        block = PaperBlock(
            block_id="block_001",
            text="Cloud pixels show lower brightness temperature.",
            block_type="paragraph",
            page_start=1,
            page_end=1,
            order_index=0,
            metadata={},
        )
        section = PaperSection(
            section_id="section_001",
            title="Results",
            normalized_type="result",
            level=2,
            blocks=[block],
        )
        paper = Paper(
            paper_id="paper_001",
            title="Test Paper",
            authors=[],
            year=None,
            source_path=None,
            source_hash=compute_source_hash(block.text),
            source_type="markdown",
            sections=[section],
            metadata={},
            parser_version=PARSER_VERSION,
            vocabulary_version=VOCABULARY_VERSION,
        )
        metadata = ChunkMetadata(target_classes=["cloud"], thresholds=["BT below 270 K"])
        chunk = Chunk(
            chunk_id="chunk_001",
            paper_id=paper.paper_id,
            chunk_type="rule_candidate",
            text=block.text,
            parent_id=None,
            child_ids=[],
            source_block_ids=[block.block_id],
            page_start=1,
            page_end=1,
            section_title="Results",
            section_type="result",
            metadata=metadata,
            retrieval_profile="rule_extraction",
            parser_version=PARSER_VERSION,
            chunker_version=CHUNKER_VERSION,
            vocabulary_version=VOCABULARY_VERSION,
            source_hash=paper.source_hash,
        )

        self.assertEqual(chunk.source_hash, paper.source_hash)
        self.assertEqual(chunk.parser_version, PARSER_VERSION)
        self.assertEqual(chunk.chunker_version, CHUNKER_VERSION)
        self.assertEqual(chunk.vocabulary_version, VOCABULARY_VERSION)
        self.assertEqual(chunk.metadata.target_classes, ["cloud"])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
python -m unittest tests.test_schemas -v
```

Expected: FAIL with `ModuleNotFoundError` or missing schema classes.

- [ ] **Step 3: Implement schemas**

Create `src/satellite_paper_rag/schemas.py`:

```python
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field


PARSER_VERSION = "text-parser-v1"
CHUNKER_VERSION = "multi-granularity-v1"
VOCABULARY_VERSION = "remote-sensing-en-v1"


def compute_source_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class PaperBlock:
    block_id: str
    text: str
    block_type: str
    page_start: int | None
    page_end: int | None
    order_index: int
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class PaperSection:
    section_id: str
    title: str
    normalized_type: str
    level: int
    blocks: list[PaperBlock] = field(default_factory=list)


@dataclass(frozen=True)
class Paper:
    paper_id: str
    title: str
    authors: list[str]
    year: str | None
    source_path: str | None
    source_hash: str
    source_type: str
    sections: list[PaperSection]
    metadata: dict[str, str] = field(default_factory=dict)
    parser_version: str = PARSER_VERSION
    vocabulary_version: str = VOCABULARY_VERSION


@dataclass(frozen=True)
class ChunkMetadata:
    satellites: list[str] = field(default_factory=list)
    sensors: list[str] = field(default_factory=list)
    bands_or_layers: list[str] = field(default_factory=list)
    indices: list[str] = field(default_factory=list)
    target_classes: list[str] = field(default_factory=list)
    evidence_types: list[str] = field(default_factory=list)
    thresholds: list[str] = field(default_factory=list)
    directionality: list[str] = field(default_factory=list)
    method_terms: list[str] = field(default_factory=list)
    metrics: list[str] = field(default_factory=list)
    units: list[str] = field(default_factory=list)
    normalized_values: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    failure_modes: list[str] = field(default_factory=list)
    confounding_factors: list[str] = field(default_factory=list)
    review_required_conditions: list[str] = field(default_factory=list)
    confidence: float = 0.0


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    paper_id: str
    chunk_type: str
    text: str
    parent_id: str | None
    child_ids: list[str]
    source_block_ids: list[str]
    page_start: int | None
    page_end: int | None
    section_title: str
    section_type: str
    metadata: ChunkMetadata
    retrieval_profile: str
    parser_version: str = PARSER_VERSION
    chunker_version: str = CHUNKER_VERSION
    vocabulary_version: str = VOCABULARY_VERSION
    source_hash: str = ""
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```powershell
python -m unittest tests.test_schemas -v
```

Expected: PASS.

Checkpoint: no git commit until a repository is initialized.

---

### Task 3: English Domain Vocabulary

**Files:**
- Create: `D:\CodexProject\satellite_paper_rag\src\satellite_paper_rag\domain\vocabulary.py`
- Create: `D:\CodexProject\satellite_paper_rag\tests\test_vocabulary.py`

- [ ] **Step 1: Write failing vocabulary tests**

Create `tests/test_vocabulary.py`:

```python
import unittest

from satellite_paper_rag.domain.vocabulary import DomainVocabulary


class DomainVocabularyTest(unittest.TestCase):
    def test_normalizes_english_aliases(self):
        vocab = DomainVocabulary.default()
        self.assertEqual(vocab.normalize_alias("BT"), "thermal_brightness_temperature")
        self.assertEqual(vocab.normalize_alias("open water"), "open_water")
        self.assertEqual(vocab.normalize_alias("sea ice"), "sea_ice")

    def test_uses_sensor_band_context(self):
        vocab = DomainVocabulary.default()
        self.assertEqual(
            vocab.normalize_band("Sentinel-3", "SLSTR", "S7"),
            "thermal_brightness_temperature",
        )
        self.assertEqual(
            vocab.normalize_band("Landsat-8", "TIRS", "B10"),
            "thermal_brightness_temperature",
        )
        self.assertIsNone(vocab.normalize_band("Unknown", "Unknown", "S7"))

    def test_phase_one_is_english_only(self):
        vocab = DomainVocabulary.default()
        self.assertIsNone(vocab.normalize_alias("云"))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
python -m unittest tests.test_vocabulary -v
```

Expected: FAIL because `DomainVocabulary` does not exist.

- [ ] **Step 3: Implement vocabulary**

Create `src/satellite_paper_rag/domain/vocabulary.py`:

```python
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DomainVocabulary:
    aliases: dict[str, str]
    sensor_band_map: dict[str, str]

    @classmethod
    def default(cls) -> "DomainVocabulary":
        aliases = {
            "bt": "thermal_brightness_temperature",
            "brightness temperature": "thermal_brightness_temperature",
            "thermal band": "thermal_brightness_temperature",
            "thermal channel": "thermal_brightness_temperature",
            "visible": "visible_reflectance",
            "visible reflectance": "visible_reflectance",
            "reflectance": "reflectance",
            "ndvi": "NDVI",
            "ndwi": "NDWI",
            "ndsi": "NDSI",
            "cloud": "cloud",
            "cloud-contaminated": "cloud",
            "sea ice": "sea_ice",
            "ice": "sea_ice",
            "open water": "open_water",
            "water": "open_water",
            "sea": "open_water",
            "land": "land",
            "vegetation": "vegetation",
            "snow": "snow",
            "sentinel-3": "Sentinel-3",
            "sentinel-2": "Sentinel-2",
            "landsat-8": "Landsat-8",
            "modis": "MODIS",
            "slstr": "SLSTR",
            "msi": "MSI",
            "oli": "OLI",
            "tirs": "TIRS",
        }
        sensor_band_map = {
            "sentinel-3/slstr/s1": "visible_reflectance",
            "sentinel-3/slstr/s2": "visible_reflectance",
            "sentinel-3/slstr/s3": "visible_reflectance",
            "sentinel-3/slstr/s4": "visible_reflectance",
            "sentinel-3/slstr/s5": "visible_reflectance",
            "sentinel-3/slstr/s6": "visible_reflectance",
            "sentinel-3/slstr/s7": "thermal_brightness_temperature",
            "sentinel-3/slstr/s8": "thermal_brightness_temperature",
            "sentinel-3/slstr/s9": "thermal_brightness_temperature",
            "landsat-8/tirs/b10": "thermal_brightness_temperature",
            "landsat-8/tirs/b11": "thermal_brightness_temperature",
        }
        return cls(aliases=aliases, sensor_band_map=sensor_band_map)

    def normalize_alias(self, value: str) -> str | None:
        return self.aliases.get(value.strip().lower())

    def normalize_band(self, satellite: str, sensor: str, band: str) -> str | None:
        key = f"{satellite}/{sensor}/{band}".strip().lower()
        return self.sensor_band_map.get(key)
```

- [ ] **Step 4: Run tests**

Run:

```powershell
python -m unittest tests.test_vocabulary -v
```

Expected: PASS.

Checkpoint: no git commit until a repository is initialized.

---

### Task 4: Text And Markdown Parsers

**Files:**
- Create: `D:\CodexProject\satellite_paper_rag\src\satellite_paper_rag\parsing\paper_parser.py`
- Create: `D:\CodexProject\satellite_paper_rag\src\satellite_paper_rag\parsing\text_parser.py`
- Create: `D:\CodexProject\satellite_paper_rag\src\satellite_paper_rag\parsing\markdown_parser.py`
- Create: `D:\CodexProject\satellite_paper_rag\src\satellite_paper_rag\parsing\pdf_parser.py`
- Create: `D:\CodexProject\satellite_paper_rag\tests\test_text_markdown_parsers.py`

- [ ] **Step 1: Write failing parser tests**

Create `tests/test_text_markdown_parsers.py`:

```python
import unittest
from pathlib import Path

from satellite_paper_rag.parsing.markdown_parser import MarkdownPaperParser
from satellite_paper_rag.parsing.pdf_parser import PdfPaperParser
from satellite_paper_rag.parsing.text_parser import TextPaperParser


FIXTURES = Path(__file__).parent / "fixtures"


class ParserTest(unittest.TestCase):
    def test_markdown_parser_detects_sections_and_captions(self):
        paper = MarkdownPaperParser().parse(FIXTURES / "sample_sentinel3_paper.md")

        self.assertEqual(paper.source_type, "markdown")
        self.assertEqual(paper.title, "Sentinel-3 SLSTR Cloud, Sea Ice, and Open Water Discrimination")
        section_types = [section.normalized_type for section in paper.sections]
        self.assertIn("abstract", section_types)
        self.assertIn("method", section_types)
        self.assertIn("result", section_types)
        block_types = [block.block_type for section in paper.sections for block in section.blocks]
        self.assertIn("table_caption", block_types)
        self.assertIn("figure_caption", block_types)

    def test_text_parser_detects_plain_text_sections(self):
        paper = TextPaperParser().parse(FIXTURES / "sample_landsat_paper.txt")

        self.assertEqual(paper.source_type, "text")
        self.assertEqual(paper.title, "Landsat-8 Thermal Cloud Screening")
        self.assertIn("method", [section.normalized_type for section in paper.sections])
        self.assertIn("result", [section.normalized_type for section in paper.sections])

    def test_pdf_parser_boundary_is_explicit(self):
        with self.assertRaises(NotImplementedError):
            PdfPaperParser().parse(FIXTURES / "sample_landsat_paper.txt")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
python -m unittest tests.test_text_markdown_parsers -v
```

Expected: FAIL because parsers do not exist.

- [ ] **Step 3: Implement parser protocol**

Create `src/satellite_paper_rag/parsing/paper_parser.py`:

```python
from __future__ import annotations

from pathlib import Path
from typing import Protocol

from satellite_paper_rag.schemas import Paper


class PaperParser(Protocol):
    def parse(self, source: Path) -> Paper:
        ...
```

- [ ] **Step 4: Implement text parser**

Create `src/satellite_paper_rag/parsing/text_parser.py`:

```python
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
            section_id = f"section_{len(sections):03d}"
            sections.append(
                PaperSection(
                    section_id=section_id,
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
```

- [ ] **Step 5: Implement Markdown parser**

Create `src/satellite_paper_rag/parsing/markdown_parser.py`:

```python
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
```

- [ ] **Step 6: Implement PDF boundary**

Create `src/satellite_paper_rag/parsing/pdf_parser.py`:

```python
from __future__ import annotations

from pathlib import Path

from satellite_paper_rag.schemas import Paper


class PdfPaperParser:
    def parse(self, source: Path) -> Paper:
        raise NotImplementedError(
            "PDF parsing is a replaceable adapter boundary in phase 1. "
            "Use MarkdownPaperParser or TextPaperParser for stable phase-1 ingestion."
        )
```

- [ ] **Step 7: Run parser tests**

Run:

```powershell
python -m unittest tests.test_text_markdown_parsers -v
```

Expected: PASS.

Checkpoint: no git commit until a repository is initialized.

---

### Task 5: Metadata Enricher

**Files:**
- Create: `D:\CodexProject\satellite_paper_rag\src\satellite_paper_rag\chunking\metadata_enricher.py`
- Create: `D:\CodexProject\satellite_paper_rag\tests\test_metadata_enricher.py`

- [ ] **Step 1: Write failing metadata tests**

Create `tests/test_metadata_enricher.py`:

```python
import unittest

from satellite_paper_rag.chunking.metadata_enricher import MetadataEnricher
from satellite_paper_rag.domain.vocabulary import DomainVocabulary


class MetadataEnricherTest(unittest.TestCase):
    def test_extracts_remote_sensing_metadata(self):
        text = (
            "For Sentinel-3 SLSTR images, cloud pixels present lower brightness "
            "temperature in S7-S9 thermal channels. Pixels with BT below 270 K "
            "were classified as cloud-contaminated."
        )
        metadata = MetadataEnricher(DomainVocabulary.default()).enrich(text)

        self.assertIn("Sentinel-3", metadata.satellites)
        self.assertIn("SLSTR", metadata.sensors)
        self.assertIn("S7-S9", metadata.bands_or_layers)
        self.assertIn("thermal_brightness_temperature", metadata.bands_or_layers)
        self.assertIn("cloud", metadata.target_classes)
        self.assertIn("BT below 270 K", metadata.thresholds)
        self.assertIn("BT < 270.0 K", metadata.normalized_values)
        self.assertIn("threshold", metadata.evidence_types)

    def test_extracts_limitations_and_review_conditions(self):
        text = "Thin cloud and mixed ice edge pixels remained ambiguous and required manual review."
        metadata = MetadataEnricher(DomainVocabulary.default()).enrich(text)

        self.assertIn("thin cloud", metadata.limitations)
        self.assertIn("mixed ice edge", metadata.failure_modes)
        self.assertIn("manual review", metadata.review_required_conditions)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
python -m unittest tests.test_metadata_enricher -v
```

Expected: FAIL because `MetadataEnricher` does not exist.

- [ ] **Step 3: Implement metadata enricher**

Create `src/satellite_paper_rag/chunking/metadata_enricher.py`:

```python
from __future__ import annotations

import re

from satellite_paper_rag.domain.vocabulary import DomainVocabulary
from satellite_paper_rag.schemas import ChunkMetadata


class MetadataEnricher:
    def __init__(self, vocabulary: DomainVocabulary) -> None:
        self.vocabulary = vocabulary

    def enrich(self, text: str) -> ChunkMetadata:
        lower = text.lower()
        satellites = self._find_terms(text, ["Sentinel-3", "Sentinel-2", "Landsat-8", "MODIS"])
        sensors = self._find_terms(text, ["SLSTR", "MSI", "OLI", "TIRS", "MODIS"])
        bands_or_layers = self._find_bands(text)
        indices = self._find_terms(text, ["NDVI", "NDWI", "NDSI"])
        target_classes = self._find_classes(lower)
        thresholds, normalized_values, units = self._find_thresholds(text)
        directionality = self._find_directionality(lower)
        evidence_types = self._find_evidence_types(lower, thresholds, directionality)
        limitations, failure_modes, review_required = self._find_limitations(lower)
        method_terms = self._find_terms(text, ["classification", "classifier", "feature contribution", "feature importance"])
        metrics = self._find_terms(text, ["accuracy", "IoU", "F1", "precision", "recall"])

        confidence = 0.0
        if target_classes:
            confidence += 0.2
        if bands_or_layers or indices:
            confidence += 0.2
        if thresholds:
            confidence += 0.25
        if directionality:
            confidence += 0.2
        if evidence_types:
            confidence += 0.15

        return ChunkMetadata(
            satellites=satellites,
            sensors=sensors,
            bands_or_layers=bands_or_layers,
            indices=indices,
            target_classes=target_classes,
            evidence_types=evidence_types,
            thresholds=thresholds,
            directionality=directionality,
            method_terms=method_terms,
            metrics=metrics,
            units=units,
            normalized_values=normalized_values,
            limitations=limitations,
            failure_modes=failure_modes,
            confounding_factors=failure_modes,
            review_required_conditions=review_required,
            confidence=min(confidence, 1.0),
        )

    def _find_terms(self, text: str, terms: list[str]) -> list[str]:
        found: list[str] = []
        for term in terms:
            if re.search(rf"\b{re.escape(term)}\b", text, re.IGNORECASE):
                found.append(term)
        return found

    def _find_bands(self, text: str) -> list[str]:
        found: list[str] = []
        for match in re.findall(r"\bS\d(?:-S\d)?\b|\bB\d{1,2}\b", text, re.IGNORECASE):
            normalized = match.upper()
            if normalized not in found:
                found.append(normalized)
        lower = text.lower()
        for phrase in ["thermal", "brightness temperature", "visible reflectance", "reflectance"]:
            if phrase in lower:
                concept = self.vocabulary.normalize_alias(phrase) or phrase
                if concept not in found:
                    found.append(concept)
        return found

    def _find_classes(self, lower: str) -> list[str]:
        classes: list[str] = []
        for phrase in ["cloud", "sea ice", "open water", "water", "land", "vegetation", "snow", "ice"]:
            if phrase in lower:
                normalized = self.vocabulary.normalize_alias(phrase) or phrase
                if normalized not in classes:
                    classes.append(normalized)
        return classes

    def _find_thresholds(self, text: str) -> tuple[list[str], list[str], list[str]]:
        thresholds: list[str] = []
        normalized_values: list[str] = []
        units: list[str] = []
        patterns = [
            (r"\b(BT)\s+(below|under|less than)\s+(-?\d+(?:\.\d+)?)\s*(K|°C|C)\b", "<"),
            (r"\b(NDVI|NDWI|NDSI)\s+(greater than|above|over)\s+(-?\d+(?:\.\d+)?)\b", ">"),
            (r"\b(NDVI|NDWI|NDSI)\s*(>=|>|<=|<)\s*(-?\d+(?:\.\d+)?)\b", None),
        ]
        for pattern, default_operator in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                original = match.group(0)
                feature = match.group(1).upper()
                if default_operator is None:
                    operator = match.group(2)
                    value = float(match.group(3))
                    unit = ""
                else:
                    operator = default_operator
                    value = float(match.group(3))
                    unit = match.group(4) if len(match.groups()) >= 4 else ""
                thresholds.append(original)
                normalized_unit = "K" if unit.upper() == "K" else unit
                unit_suffix = f" {normalized_unit}" if normalized_unit else ""
                normalized_values.append(f"{feature} {operator} {value}{unit_suffix}")
                if normalized_unit and normalized_unit not in units:
                    units.append(normalized_unit)
        return thresholds, normalized_values, units

    def _find_directionality(self, lower: str) -> list[str]:
        phrases = ["higher", "lower", "increase", "decrease", "warmer", "colder", "brighter", "darker", "more than", "less than"]
        return [phrase for phrase in phrases if phrase in lower]

    def _find_evidence_types(self, lower: str, thresholds: list[str], directionality: list[str]) -> list[str]:
        evidence_types: list[str] = []
        if thresholds:
            evidence_types.append("threshold")
        if directionality:
            evidence_types.append("comparison")
        if "classified" in lower or "classification" in lower:
            evidence_types.append("classification_rule")
        if "table" in lower or "figure" in lower:
            evidence_types.append("result")
        return evidence_types

    def _find_limitations(self, lower: str) -> tuple[list[str], list[str], list[str]]:
        limitations: list[str] = []
        failure_modes: list[str] = []
        review_required: list[str] = []
        for phrase in ["thin cloud", "low solar angle", "seasonal variation"]:
            if phrase in lower:
                limitations.append(phrase)
        for phrase in ["mixed ice edge", "mixed pixels", "turbid water", "ambiguous"]:
            if phrase in lower:
                failure_modes.append(phrase)
        if "manual review" in lower or "required review" in lower:
            review_required.append("manual review")
        return limitations, failure_modes, review_required
```

- [ ] **Step 4: Run metadata tests**

Run:

```powershell
python -m unittest tests.test_metadata_enricher -v
```

Expected: PASS.

Checkpoint: no git commit until a repository is initialized.

---

### Task 6: Multi-Granularity Chunking Pipeline

**Files:**
- Create: `D:\CodexProject\satellite_paper_rag\src\satellite_paper_rag\chunking\pipeline.py`
- Create: `D:\CodexProject\satellite_paper_rag\tests\test_chunking_pipeline.py`

- [ ] **Step 1: Write failing pipeline tests**

Create `tests/test_chunking_pipeline.py`:

```python
import unittest
from pathlib import Path

from satellite_paper_rag.chunking.pipeline import PaperChunkingPipeline
from satellite_paper_rag.domain.vocabulary import DomainVocabulary
from satellite_paper_rag.parsing.markdown_parser import MarkdownPaperParser


FIXTURES = Path(__file__).parent / "fixtures"


class ChunkingPipelineTest(unittest.TestCase):
    def test_generates_multiple_chunk_types_with_parent_child_links(self):
        paper = MarkdownPaperParser().parse(FIXTURES / "sample_sentinel3_paper.md")
        chunks = PaperChunkingPipeline(DomainVocabulary.default()).chunk(paper)

        chunk_types = {chunk.chunk_type for chunk in chunks}
        self.assertIn("paper_summary", chunk_types)
        self.assertIn("section_parent", chunk_types)
        self.assertIn("paragraph_child", chunk_types)
        self.assertIn("sentence_window_child", chunk_types)
        self.assertIn("figure_table", chunk_types)
        self.assertIn("rule_candidate", chunk_types)

        parents = {chunk.chunk_id for chunk in chunks if chunk.chunk_type == "section_parent"}
        child_chunks = [chunk for chunk in chunks if chunk.parent_id is not None]
        self.assertTrue(child_chunks)
        self.assertTrue(all(chunk.parent_id in parents for chunk in child_chunks))

    def test_rule_candidate_preserves_domain_metadata_and_versions(self):
        paper = MarkdownPaperParser().parse(FIXTURES / "sample_sentinel3_paper.md")
        chunks = PaperChunkingPipeline(DomainVocabulary.default()).chunk(paper)
        rule_chunks = [chunk for chunk in chunks if chunk.chunk_type == "rule_candidate"]

        self.assertTrue(rule_chunks)
        combined_text = " ".join(chunk.text for chunk in rule_chunks)
        self.assertIn("BT below 270 K", combined_text)
        matching = [chunk for chunk in rule_chunks if "BT below 270 K" in chunk.text]
        self.assertTrue(matching)
        chunk = matching[0]
        self.assertIn("cloud", chunk.metadata.target_classes)
        self.assertIn("BT below 270 K", chunk.metadata.thresholds)
        self.assertTrue(chunk.source_hash)
        self.assertTrue(chunk.parser_version)
        self.assertTrue(chunk.chunker_version)
        self.assertTrue(chunk.vocabulary_version)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
python -m unittest tests.test_chunking_pipeline -v
```

Expected: FAIL because pipeline does not exist.

- [ ] **Step 3: Implement chunking pipeline**

Create `src/satellite_paper_rag/chunking/pipeline.py`:

```python
from __future__ import annotations

import re

from satellite_paper_rag.chunking.metadata_enricher import MetadataEnricher
from satellite_paper_rag.domain.vocabulary import DomainVocabulary
from satellite_paper_rag.schemas import CHUNKER_VERSION, Chunk, Paper, PaperBlock


class PaperChunkingPipeline:
    def __init__(self, vocabulary: DomainVocabulary) -> None:
        self.enricher = MetadataEnricher(vocabulary)

    def chunk(self, paper: Paper) -> list[Chunk]:
        chunks: list[Chunk] = []
        chunks.append(self._paper_summary_chunk(paper))
        for section in paper.sections:
            parent = self._section_parent_chunk(paper, section_index=len(chunks), section=section)
            chunks.append(parent)
            child_ids: list[str] = []
            for block in section.blocks:
                paragraph = self._paragraph_child_chunk(paper, parent.chunk_id, block, section.title, section.normalized_type)
                chunks.append(paragraph)
                child_ids.append(paragraph.chunk_id)
                for window in self._sentence_window_chunks(paper, parent.chunk_id, block, section.title, section.normalized_type):
                    chunks.append(window)
                    child_ids.append(window.chunk_id)
                if block.block_type in {"figure_caption", "table_caption", "table_text"}:
                    figure_table = self._figure_table_chunk(paper, parent.chunk_id, block, section.title, section.normalized_type)
                    chunks.append(figure_table)
                    child_ids.append(figure_table.chunk_id)
                rule = self._rule_candidate_chunk(paper, parent.chunk_id, block, section.title, section.normalized_type)
                if rule is not None:
                    chunks.append(rule)
                    child_ids.append(rule.chunk_id)
            if child_ids:
                updated_parent = Chunk(
                    chunk_id=parent.chunk_id,
                    paper_id=parent.paper_id,
                    chunk_type=parent.chunk_type,
                    text=parent.text,
                    parent_id=parent.parent_id,
                    child_ids=child_ids,
                    source_block_ids=parent.source_block_ids,
                    page_start=parent.page_start,
                    page_end=parent.page_end,
                    section_title=parent.section_title,
                    section_type=parent.section_type,
                    metadata=parent.metadata,
                    retrieval_profile=parent.retrieval_profile,
                    parser_version=parent.parser_version,
                    chunker_version=parent.chunker_version,
                    vocabulary_version=parent.vocabulary_version,
                    source_hash=parent.source_hash,
                )
                chunks[chunks.index(parent)] = updated_parent
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

    def _section_parent_chunk(self, paper: Paper, section_index: int, section) -> Chunk:
        text = "\n\n".join(block.text for block in section.blocks)
        page_start = self._min_page(section.blocks)
        page_end = self._max_page(section.blocks)
        return self._make_chunk(
            paper,
            "section_parent",
            text,
            None,
            [block.block_id for block in section.blocks],
            page_start,
            page_end,
            section.title,
            section.normalized_type,
            "broad_context",
            suffix=f"{section_index:04d}",
        )

    def _paragraph_child_chunk(self, paper: Paper, parent_id: str, block: PaperBlock, section_title: str, section_type: str) -> Chunk:
        return self._make_chunk(
            paper,
            "paragraph_child",
            block.text,
            parent_id,
            [block.block_id],
            block.page_start,
            block.page_end,
            section_title,
            section_type,
            "semantic_recall",
            suffix=block.block_id,
        )

    def _sentence_window_chunks(self, paper: Paper, parent_id: str, block: PaperBlock, section_title: str, section_type: str) -> list[Chunk]:
        sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", block.text) if sentence.strip()]
        chunks: list[Chunk] = []
        for index, sentence in enumerate(sentences):
            metadata = self.enricher.enrich(sentence)
            if not (metadata.target_classes or metadata.bands_or_layers or metadata.thresholds):
                continue
            start = max(index - 1, 0)
            end = min(index + 2, len(sentences))
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

    def _rule_candidate_chunk(self, paper: Paper, parent_id: str, block: PaperBlock, section_title: str, section_type: str) -> Chunk | None:
        metadata = self.enricher.enrich(block.text)
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
        if signal_count < 2:
            return None
        return self._make_chunk(
            paper,
            "rule_candidate",
            block.text,
            parent_id,
            [block.block_id],
            block.page_start,
            block.page_end,
            section_title,
            section_type,
            "rule_extraction",
            suffix=block.block_id,
        )

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
            metadata=self.enricher.enrich(text),
            retrieval_profile=retrieval_profile,
            parser_version=paper.parser_version,
            chunker_version=CHUNKER_VERSION,
            vocabulary_version=paper.vocabulary_version,
            source_hash=paper.source_hash,
        )

    def _min_page(self, blocks: list[PaperBlock]) -> int | None:
        pages = [block.page_start for block in blocks if block.page_start is not None]
        return min(pages) if pages else None

    def _max_page(self, blocks: list[PaperBlock]) -> int | None:
        pages = [block.page_end for block in blocks if block.page_end is not None]
        return max(pages) if pages else None
```

- [ ] **Step 4: Run pipeline tests**

Run:

```powershell
python -m unittest tests.test_chunking_pipeline -v
```

Expected: PASS.

Checkpoint: no git commit until a repository is initialized.

---

### Task 7: Observation Sample Normalization

**Files:**
- Create: `D:\CodexProject\satellite_paper_rag\src\satellite_paper_rag\observations\schema.py`
- Create: `D:\CodexProject\satellite_paper_rag\src\satellite_paper_rag\observations\feature_normalizer.py`
- Create: `D:\CodexProject\satellite_paper_rag\tests\test_observation_normalizer.py`

- [ ] **Step 1: Write failing observation tests**

Create `tests/test_observation_normalizer.py`:

```python
import unittest

from satellite_paper_rag.domain.vocabulary import DomainVocabulary
from satellite_paper_rag.observations.feature_normalizer import FeatureNormalizer
from satellite_paper_rag.observations.schema import ObservationSample


class ObservationNormalizerTest(unittest.TestCase):
    def test_normalizes_sensor_band_features(self):
        sample = ObservationSample(
            sample_id="183_303_1402",
            source_type="csv",
            satellite="Sentinel-3",
            sensor="SLSTR",
            features={"S1": 18.0, "S7": 276.0, "S8": 273.2, "NDVI": 0.12},
            normalized_features={},
            metadata={},
        )
        normalized = FeatureNormalizer(DomainVocabulary.default()).normalize(sample)

        self.assertEqual(normalized.normalized_features["visible_reflectance.S1"], 18.0)
        self.assertEqual(normalized.normalized_features["thermal_brightness_temperature.S7"], 276.0)
        self.assertEqual(normalized.normalized_features["thermal_brightness_temperature.S8"], 273.2)
        self.assertEqual(normalized.normalized_features["NDVI"], 0.12)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
python -m unittest tests.test_observation_normalizer -v
```

Expected: FAIL because observation modules do not exist.

- [ ] **Step 3: Implement observation schema**

Create `src/satellite_paper_rag/observations/schema.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ObservationSample:
    sample_id: str
    source_type: str
    satellite: str | None
    sensor: str | None
    features: dict[str, float | str]
    normalized_features: dict[str, float | str] = field(default_factory=dict)
    metadata: dict[str, str] = field(default_factory=dict)
```

- [ ] **Step 4: Implement feature normalizer**

Create `src/satellite_paper_rag/observations/feature_normalizer.py`:

```python
from __future__ import annotations

from dataclasses import replace

from satellite_paper_rag.domain.vocabulary import DomainVocabulary
from satellite_paper_rag.observations.schema import ObservationSample


class FeatureNormalizer:
    def __init__(self, vocabulary: DomainVocabulary) -> None:
        self.vocabulary = vocabulary

    def normalize(self, sample: ObservationSample) -> ObservationSample:
        normalized: dict[str, float | str] = {}
        for name, value in sample.features.items():
            concept = None
            if sample.satellite and sample.sensor:
                concept = self.vocabulary.normalize_band(sample.satellite, sample.sensor, name)
            alias = self.vocabulary.normalize_alias(name)
            if concept:
                normalized[f"{concept}.{name}"] = value
            elif alias:
                normalized[alias] = value
            else:
                normalized[name] = value
        return replace(sample, normalized_features=normalized)
```

- [ ] **Step 5: Run observation tests**

Run:

```powershell
python -m unittest tests.test_observation_normalizer -v
```

Expected: PASS.

Checkpoint: no git commit until a repository is initialized.

---

### Task 8: Retrieval Contract And Mock Hybrid Retriever

**Files:**
- Create: `D:\CodexProject\satellite_paper_rag\src\satellite_paper_rag\retrieval\contract.py`
- Create: `D:\CodexProject\satellite_paper_rag\src\satellite_paper_rag\retrieval\mock_hybrid_retriever.py`
- Create: `D:\CodexProject\satellite_paper_rag\tests\test_mock_retriever.py`

- [ ] **Step 1: Write failing retrieval tests**

Create `tests/test_mock_retriever.py`:

```python
import unittest
from pathlib import Path

from satellite_paper_rag.chunking.pipeline import PaperChunkingPipeline
from satellite_paper_rag.domain.vocabulary import DomainVocabulary
from satellite_paper_rag.parsing.markdown_parser import MarkdownPaperParser
from satellite_paper_rag.parsing.text_parser import TextPaperParser
from satellite_paper_rag.retrieval.contract import RetrievalRequest
from satellite_paper_rag.retrieval.mock_hybrid_retriever import MockHybridRetriever


FIXTURES = Path(__file__).parent / "fixtures"


class MockRetrieverTest(unittest.TestCase):
    def test_retrieves_rule_chunk_and_expands_parent(self):
        vocab = DomainVocabulary.default()
        paper = MarkdownPaperParser().parse(FIXTURES / "sample_sentinel3_paper.md")
        chunks = PaperChunkingPipeline(vocab).chunk(paper)
        retriever = MockHybridRetriever(chunks)
        results = retriever.retrieve(
            RetrievalRequest(
                query="What threshold helps identify cloud in Sentinel-3 SLSTR thermal bands?",
                chunk_types=["rule_candidate", "sentence_window_child", "figure_table"],
                metadata_filters={"target_classes": ["cloud"], "sensors": ["SLSTR"]},
                expand_parents=True,
                top_k=3,
            )
        )

        self.assertTrue(results)
        self.assertNotEqual(results[0].answer_type, "insufficient_evidence")
        self.assertIn("BT below 270 K", results[0].chunk.text)
        self.assertIsNotNone(results[0].expanded_parent)
        self.assertTrue(results[0].chunk.paper_id)
        self.assertTrue(results[0].chunk.chunk_id)

    def test_returns_insufficient_evidence_for_missing_threshold(self):
        vocab = DomainVocabulary.default()
        paper = TextPaperParser().parse(FIXTURES / "sample_landsat_paper.txt")
        chunks = PaperChunkingPipeline(vocab).chunk(paper)
        retriever = MockHybridRetriever(chunks)
        results = retriever.retrieve(
            RetrievalRequest(
                query="What exact snow threshold is used by Landsat?",
                chunk_types=["rule_candidate", "sentence_window_child"],
                metadata_filters={"target_classes": ["snow"]},
                expand_parents=True,
                top_k=3,
                requires_threshold=True,
            )
        )

        self.assertEqual(results[0].answer_type, "insufficient_evidence")
        self.assertIn("thresholds", results[0].missing_evidence)

    def test_marks_cross_sensor_evidence_as_indirect(self):
        vocab = DomainVocabulary.default()
        paper = TextPaperParser().parse(FIXTURES / "sample_landsat_paper.txt")
        chunks = PaperChunkingPipeline(vocab).chunk(paper)
        retriever = MockHybridRetriever(chunks)
        results = retriever.retrieve(
            RetrievalRequest(
                query="Is thermal band useful for Sentinel-3 SLSTR cloud detection?",
                chunk_types=["rule_candidate", "sentence_window_child", "figure_table"],
                metadata_filters={"target_classes": ["cloud"]},
                requested_sensor="SLSTR",
                expand_parents=True,
                top_k=3,
            )
        )

        self.assertTrue(results)
        self.assertTrue(any(result.is_indirect_evidence for result in results))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
python -m unittest tests.test_mock_retriever -v
```

Expected: FAIL because retrieval modules do not exist.

- [ ] **Step 3: Implement retrieval contract**

Create `src/satellite_paper_rag/retrieval/contract.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field

from satellite_paper_rag.schemas import Chunk


@dataclass(frozen=True)
class RetrievalRequest:
    query: str
    chunk_types: list[str] = field(default_factory=list)
    metadata_filters: dict[str, list[str]] = field(default_factory=dict)
    expand_parents: bool = True
    top_k: int = 5
    requires_threshold: bool = False
    requested_sensor: str | None = None


@dataclass(frozen=True)
class RetrievalResult:
    chunk: Chunk
    score: float
    matched_terms: list[str]
    expanded_parent: Chunk | None
    rationale: str
    answer_type: str = "evidence"
    missing_evidence: list[str] = field(default_factory=list)
    is_indirect_evidence: bool = False
```

- [ ] **Step 4: Implement mock retriever**

Create `src/satellite_paper_rag/retrieval/mock_hybrid_retriever.py`:

```python
from __future__ import annotations

import re

from satellite_paper_rag.retrieval.contract import RetrievalRequest, RetrievalResult
from satellite_paper_rag.schemas import Chunk, ChunkMetadata


class MockHybridRetriever:
    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        self.by_id = {chunk.chunk_id: chunk for chunk in chunks}

    def retrieve(self, request: RetrievalRequest) -> list[RetrievalResult]:
        candidates = [
            chunk
            for chunk in self.chunks
            if not request.chunk_types or chunk.chunk_type in request.chunk_types
        ]
        scored: list[RetrievalResult] = []
        for chunk in candidates:
            if not self._metadata_matches(chunk.metadata, request.metadata_filters):
                continue
            score, matched_terms = self._score(chunk, request.query)
            if score <= 0:
                continue
            score += self._chunk_type_boost(chunk.chunk_type)
            expanded_parent = self.by_id.get(chunk.parent_id) if request.expand_parents and chunk.parent_id else None
            is_indirect = bool(
                request.requested_sensor
                and chunk.metadata.sensors
                and request.requested_sensor not in chunk.metadata.sensors
            )
            scored.append(
                RetrievalResult(
                    chunk=chunk,
                    score=score,
                    matched_terms=matched_terms,
                    expanded_parent=expanded_parent,
                    rationale="keyword, metadata, and chunk-type scoring",
                    is_indirect_evidence=is_indirect,
                )
            )

        scored.sort(key=lambda result: result.score, reverse=True)
        top = scored[: request.top_k]
        missing = self._missing_evidence(request, top)
        if missing:
            return [self._insufficient_result(missing, top)]
        return top

    def _metadata_matches(self, metadata: ChunkMetadata, filters: dict[str, list[str]]) -> bool:
        for field_name, expected_values in filters.items():
            actual_values = getattr(metadata, field_name, [])
            if expected_values and not any(value in actual_values for value in expected_values):
                return False
        return True

    def _score(self, chunk: Chunk, query: str) -> tuple[float, list[str]]:
        query_terms = [term.lower() for term in re.findall(r"[A-Za-z0-9-]+", query) if len(term) > 2]
        haystack = " ".join(
            [
                chunk.text,
                " ".join(chunk.metadata.satellites),
                " ".join(chunk.metadata.sensors),
                " ".join(chunk.metadata.bands_or_layers),
                " ".join(chunk.metadata.indices),
                " ".join(chunk.metadata.target_classes),
                " ".join(chunk.metadata.evidence_types),
                " ".join(chunk.metadata.thresholds),
            ]
        ).lower()
        matched = sorted({term for term in query_terms if term in haystack})
        return float(len(matched)), matched

    def _chunk_type_boost(self, chunk_type: str) -> float:
        boosts = {
            "rule_candidate": 3.0,
            "sentence_window_child": 2.0,
            "figure_table": 1.5,
            "paragraph_child": 1.0,
            "section_parent": 0.5,
        }
        return boosts.get(chunk_type, 0.0)

    def _missing_evidence(self, request: RetrievalRequest, results: list[RetrievalResult]) -> list[str]:
        if not results:
            return ["relevant_chunks"]
        missing: list[str] = []
        if request.requires_threshold and not any(result.chunk.metadata.thresholds for result in results):
            missing.append("thresholds")
        if request.requested_sensor and not any(request.requested_sensor in result.chunk.metadata.sensors for result in results):
            missing.append("same_sensor_evidence")
        return missing

    def _insufficient_result(self, missing: list[str], partial_results: list[RetrievalResult]) -> RetrievalResult:
        if partial_results:
            base = partial_results[0]
            return RetrievalResult(
                chunk=base.chunk,
                score=base.score,
                matched_terms=base.matched_terms,
                expanded_parent=base.expanded_parent,
                rationale="retrieved chunks did not satisfy the minimum evidence policy",
                answer_type="insufficient_evidence",
                missing_evidence=missing,
                is_indirect_evidence=base.is_indirect_evidence,
            )
        empty_chunk = Chunk(
            chunk_id="insufficient_evidence",
            paper_id="",
            chunk_type="none",
            text="",
            parent_id=None,
            child_ids=[],
            source_block_ids=[],
            page_start=None,
            page_end=None,
            section_title="",
            section_type="unknown",
            metadata=ChunkMetadata(),
            retrieval_profile="none",
        )
        return RetrievalResult(
            chunk=empty_chunk,
            score=0.0,
            matched_terms=[],
            expanded_parent=None,
            rationale="no relevant chunks were retrieved",
            answer_type="insufficient_evidence",
            missing_evidence=missing,
        )
```

- [ ] **Step 5: Run retrieval tests**

Run:

```powershell
python -m unittest tests.test_mock_retriever -v
```

Expected: PASS.

Checkpoint: no git commit until a repository is initialized.

---

### Task 9: Full Test Run And Documentation Check

**Files:**
- Modify: `D:\CodexProject\satellite_paper_rag\README.md`

- [ ] **Step 1: Run all tests**

Run:

```powershell
python -m unittest discover -s tests -v
```

Expected: all tests PASS.

- [ ] **Step 2: If imports fail, run with package path**

If tests cannot import `satellite_paper_rag`, run from project root:

```powershell
$env:PYTHONPATH='src'; python -m unittest discover -s tests -v
```

Expected: all tests PASS.

- [ ] **Step 3: Update README with phase-1 usage**

Append to `README.md`:

```markdown
## Phase 1 Usage

```python
from pathlib import Path

from satellite_paper_rag.chunking.pipeline import PaperChunkingPipeline
from satellite_paper_rag.domain.vocabulary import DomainVocabulary
from satellite_paper_rag.parsing.markdown_parser import MarkdownPaperParser
from satellite_paper_rag.retrieval.contract import RetrievalRequest
from satellite_paper_rag.retrieval.mock_hybrid_retriever import MockHybridRetriever

vocab = DomainVocabulary.default()
paper = MarkdownPaperParser().parse(Path("tests/fixtures/sample_sentinel3_paper.md"))
chunks = PaperChunkingPipeline(vocab).chunk(paper)
retriever = MockHybridRetriever(chunks)
results = retriever.retrieve(
    RetrievalRequest(
        query="What thermal threshold helps identify cloud?",
        chunk_types=["rule_candidate", "sentence_window_child"],
        metadata_filters={"target_classes": ["cloud"]},
        requires_threshold=True,
    )
)
```

Every retrieved evidence chunk preserves `paper_id`, `chunk_id`, `parent_id`, `section_title`, page fields when available, `source_hash`, and parser/chunker/vocabulary versions.
```

- [ ] **Step 4: Run all tests again**

Run:

```powershell
python -m unittest discover -s tests -v
```

Expected: all tests PASS.

- [ ] **Step 5: Check repository status**

Run:

```powershell
git status --short
```

Expected in current workspace: fatal error because `D:\CodexProject` is not a git repository. If a repository has been initialized by then, review changed files and commit with:

```powershell
git add satellite_paper_rag
git commit -m "feat: add satellite paper rag phase 1 foundation"
```

---

## Plan Self-Review

Spec coverage:

- Paper/Text/Markdown input: Task 4.
- PDF boundary: Task 4.
- Multi-granularity chunking: Task 6.
- Parent-child chunks: Task 6.
- Domain metadata: Task 5.
- Sensor-band mapping: Task 3.
- Unit and threshold normalization: Task 5.
- Source/version provenance: Task 2 and Task 6.
- English-only boundary: Task 3.
- Observation sample future path: Task 7.
- Retrieval contract and mock retrieval: Task 8.
- Insufficient-evidence behavior: Task 8.
- Cross-sensor indirect evidence: Task 8.
- Negative evidence and manual review hints: Task 5.
- Tests and fixtures: Tasks 1 through 9.

No placeholders:

- The plan intentionally uses a no-op PDF boundary with an explicit `NotImplementedError`.
- Git commits are documented as checkpoints because the current workspace is not a git repository.
- All implementation tasks include concrete files, code snippets, commands, and expected results.

