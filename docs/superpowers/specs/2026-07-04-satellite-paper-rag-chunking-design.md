# Satellite Paper RAG Chunking Design

## Goal

Build the first production-oriented foundation for a satellite imagery paper RAG system. The first phase focuses on turning satellite remote-sensing papers from PDF, plain text, or Markdown into multi-granularity, metadata-rich, traceable chunks that can support later retrieval, rule extraction, and evidence-grounded answers.

The main business goal is not general paper Q&A. The system should help answer operational interpretation questions such as:

- Which satellite or sensor is the paper using?
- Which bands, layers, indices, or derived features matter for cloud, sea, ice, water, land, vegetation, or snow classification?
- What thresholds, value ranges, or trends does the paper mention?
- Which layer has the strongest influence on a class decision?
- What method, experiment, table, or figure supports that decision?

## Scope

### In Scope For Phase 1

- Create a new project under `D:\CodexProject\satellite_paper_rag`.
- Define stable schemas for papers, paper sections, paper blocks, chunks, retrieval requests, and observation samples.
- Support paper knowledge input from plain text and Markdown as the first stable path.
- Define a PDF parser interface and include a basic PDF parsing adapter boundary, without requiring perfect layout reconstruction in phase 1.
- Generate multiple chunk types from the same paper:
  - `paper_summary`
  - `section_parent`
  - `paragraph_child`
  - `sentence_window_child`
  - `figure_table`
  - `rule_candidate`
- Preserve parent-child relationships between broad context chunks and precise evidence chunks.
- Extract first-pass domain metadata with deterministic rules, dictionaries, and regular expressions.
- Define the retrieval contract that later vector search, keyword search, metadata filtering, parent expansion, and reranking must follow.
- Include mock retrieval and tests to prove that key evidence can be recalled.
- Define a `DomainVocabulary` layer so future CSV, Excel, table, JSON, or database inputs can map their field names to the same concepts used by paper chunks.
- Track parser, chunker, and vocabulary versions so generated chunks can be reproduced and debugged later.

### Out Of Scope For Phase 1

- Full LLM answer generation.
- Real production vector database integration.
- Full reranker model integration.
- Perfect PDF table reconstruction.
- Complex formula parsing.
- Full image or figure content understanding.
- Automatic Scholar crawling or paper downloading.
- Complete CSV-driven classification pipeline.
- Chinese-language paper parsing, Chinese query normalization, or bilingual vocabulary support.

Phase 1 should still design for those later capabilities, but not implement them fully.

## Source Ownership Boundary

The system does not search Scholar, crawl the web, or download papers. The user provides the paper files directly as PDF, text, or Markdown.

This boundary keeps phase 1 focused on knowledge ingestion and evidence chunking. It also avoids mixing paper discovery, network access, copyright handling, and RAG evidence processing in the same subsystem.

Even though the system does not fetch papers, it should preserve source metadata when available:

- `source_path`
- `source_url`
- `doi`
- `license`
- `access_type`
- `ingested_at`
- `source_hash`

These fields support provenance, later cleanup, and auditability.

## Language Boundary

Phase 1 is English-only.

The domain vocabulary, metadata extraction rules, query fixtures, and tests should use English terms such as:

- `cloud`
- `sea ice`
- `open water`
- `brightness temperature`
- `reflectance`
- `thermal band`
- `visible band`
- `threshold`
- `higher than`
- `lower than`
- `increase`
- `decrease`

Chinese papers, Chinese queries, and bilingual alias normalization are future extensions, not phase 1 requirements.

## Core Principle

Chunking and retrieval must be designed together, but implemented in layers.

The system should not first split text randomly and later hope retrieval works. It should work backwards from the questions the RAG system must answer:

1. What question is being asked?
2. What evidence would answer it?
3. What chunk type preserves that evidence?
4. What metadata is needed to filter, boost, or audit that chunk?
5. What parent context is needed to avoid an answer that is technically correct but out of context?

## High-Level Architecture

```text
Knowledge Input
  PDF / text / Markdown paper
        |
        v
Paper Parser
  normalizes source into Paper, PaperSection, PaperBlock
        |
        v
Multi-Granularity Chunking Pipeline
  section, paragraph, window, figure/table, rule candidate chunks
        |
        v
Domain Metadata Enricher
  satellite, sensor, band, layer, class, threshold, trend, method
        |
        v
Chunk Store Contract
  chunk text, metadata, parent-child links, retrieval profile
        |
        v
Mock Hybrid Retriever For Phase 1
  keyword + metadata + chunk type boosting + parent expansion
```

Future observation inputs follow a separate path:

```text
Observation Input
  CSV / Excel / table / JSON / database row
        |
        v
Observation Parser
        |
        v
Feature Normalizer + Domain Vocabulary
        |
        v
Query Builder
        |
        v
Retriever over paper chunks
```

## Paper Model

The system should avoid passing raw strings through the whole pipeline. Every source should first become a structured `Paper`.

```python
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
    metadata: dict[str, str]
    parser_version: str
    vocabulary_version: str
```

```python
@dataclass(frozen=True)
class PaperSection:
    section_id: str
    title: str
    normalized_type: str
    level: int
    blocks: list[PaperBlock]
```

```python
@dataclass(frozen=True)
class PaperBlock:
    block_id: str
    text: str
    block_type: str
    page_start: int | None
    page_end: int | None
    order_index: int
    metadata: dict[str, str]
```

Supported `source_type` values:

- `pdf`
- `text`
- `markdown`

Supported `normalized_type` values:

- `title`
- `abstract`
- `introduction`
- `related_work`
- `method`
- `dataset`
- `study_area`
- `experiment`
- `result`
- `discussion`
- `conclusion`
- `reference`
- `unknown`

Supported `block_type` values:

- `heading`
- `paragraph`
- `figure_caption`
- `table_caption`
- `table_text`
- `list_item`
- `equation_text`
- `reference`

## Input Normalization

### Text And Markdown

Text and Markdown are the first stable implementation path.

The parser should:

- Detect Markdown headings such as `#`, `##`, and `###`.
- Detect common paper section names such as Abstract, Introduction, Method, Dataset, Results, Discussion, Conclusion, and References.
- Split paragraphs on blank lines.
- Preserve original order.
- Preserve section title context for every block.
- Detect simple figure and table captions with patterns such as `Figure 1`, `Fig. 2`, `Table 3`.

### PDF

PDF support should be behind an interface:

```python
class PaperParser(Protocol):
    def parse(self, source: Path) -> Paper:
        ...
```

Phase 1 may include a basic adapter boundary for future libraries such as PyMuPDF, Unstructured, Marker, or GROBID. The first implementation should not depend on perfect layout reconstruction.

Minimum PDF expectations:

- Extract text with page numbers if a PDF parser is available.
- Split into blocks in reading order as well as possible.
- Identify obvious headings and captions.
- Mark uncertain layout fields in metadata.
- Do not silently drop page provenance.

## Chunk Model

Chunks are evidence objects, not just text slices.

```python
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
    parser_version: str
    chunker_version: str
    vocabulary_version: str
    source_hash: str
```

```python
@dataclass(frozen=True)
class ChunkMetadata:
    satellites: list[str]
    sensors: list[str]
    bands_or_layers: list[str]
    indices: list[str]
    target_classes: list[str]
    evidence_types: list[str]
    thresholds: list[str]
    directionality: list[str]
    method_terms: list[str]
    metrics: list[str]
    units: list[str]
    normalized_values: list[str]
    limitations: list[str]
    failure_modes: list[str]
    confounding_factors: list[str]
    review_required_conditions: list[str]
    confidence: float
```

Supported `chunk_type` values:

- `paper_summary`
- `section_parent`
- `paragraph_child`
- `sentence_window_child`
- `figure_table`
- `rule_candidate`

Supported `retrieval_profile` values:

- `paper_routing`
- `broad_context`
- `semantic_recall`
- `precise_evidence`
- `table_metric`
- `rule_extraction`

## Chunk Types

### Paper Summary Chunk

Purpose:

- Quickly decide whether a paper is relevant.
- Store paper-level satellite, sensor, task, class, and method hints.

Source:

- Title
- Abstract
- Keywords if available
- Conclusion if available

Retrieval profile:

- `paper_routing`

### Section Parent Chunk

Purpose:

- Preserve broad context.
- Provide answer-time grounding after a smaller child chunk is retrieved.

Source:

- One paper section or subsection.
- If a section is too large, split into stable parent windows by block order while keeping section identity.

Retrieval profile:

- `broad_context`

### Paragraph Child Chunk

Purpose:

- General semantic retrieval.
- Good default unit for embedding and keyword search.

Source:

- One paragraph block, or a small group of short adjacent paragraph blocks.

Retrieval profile:

- `semantic_recall`

### Sentence Window Child Chunk

Purpose:

- Capture local relationships between class, band, threshold, trend, and method.

Source:

- A trigger sentence plus one sentence before and one sentence after.

Triggers:

- Satellite names
- Sensor names
- Band names
- Index names
- Class labels
- Threshold patterns
- Directionality terms such as higher, lower, increase, decrease, warmer, colder, brighter, darker

Retrieval profile:

- `precise_evidence`

### Figure Table Chunk

Purpose:

- Preserve experimental and numeric evidence near figures and tables.

Source:

- Figure caption or table caption.
- Nearby explanatory paragraphs.
- Extracted table text if available.

Retrieval profile:

- `table_metric`

### Rule Candidate Chunk

Purpose:

- Capture candidate operational interpretation rules.
- Feed later rule extraction and classification reasoning.

Source:

- Sentences or paragraphs with at least two of these elements:
  - satellite or sensor
  - band, layer, index, or feature
  - target class
  - threshold or numeric value
  - directionality or comparison
  - classification method or decision rule

Retrieval profile:

- `rule_extraction`

Example:

```json
{
  "chunk_type": "rule_candidate",
  "text": "For Sentinel-3 SLSTR images, sea ice shows higher reflectance in S1-S6 bands than open water, while cloud pixels present lower brightness temperature in S7-S9 thermal channels. Pixels with BT below 270 K were frequently classified as cloud-contaminated.",
  "metadata": {
    "satellites": ["Sentinel-3"],
    "sensors": ["SLSTR"],
    "bands_or_layers": ["S1-S6", "S7-S9", "thermal", "brightness temperature"],
    "target_classes": ["sea ice", "open water", "cloud"],
    "evidence_types": ["comparison", "threshold", "classification_rule"],
    "thresholds": ["BT below 270 K"],
    "directionality": [
      "sea ice reflectance higher than open water",
      "cloud brightness temperature lower"
    ]
  }
}
```

## Parent-Child Strategy

The system should use small chunks for matching and larger parent chunks for context.

```text
section_parent
  paragraph_child
  sentence_window_child
  figure_table
  rule_candidate
```

Retrieval flow:

1. Search child chunks first because they are precise.
2. Boost `rule_candidate`, `sentence_window_child`, and `figure_table` when the query asks about rules, thresholds, bands, or numeric evidence.
3. Expand top child hits to their parent section chunks.
4. Return both the precise child evidence and the parent context.

This avoids two common failures:

- Large chunks recall relevant sections but bury the answer in noise.
- Tiny chunks match exactly but lose the surrounding conditions that make the rule valid.

## Domain Vocabulary

The system needs a domain vocabulary to normalize paper language and future CSV/table field names.

The vocabulary should map aliases to canonical concepts.

Example:

```json
{
  "thermal_brightness_temperature": {
    "aliases": ["BT", "brightness temperature", "thermal band", "thermal channel", "S7", "S8", "S9"],
    "unit": "K",
    "concept_type": "band_group",
    "related_classes": ["cloud", "ice", "sea"]
  },
  "visible_reflectance": {
    "aliases": ["visible", "reflectance", "S1", "S2", "S3", "S4", "S5", "S6"],
    "concept_type": "band_group",
    "related_classes": ["cloud", "ice", "sea", "land"]
  },
  "sea_ice": {
    "aliases": ["ice", "sea ice", "floe", "pack ice"],
    "concept_type": "class"
  },
  "open_water": {
    "aliases": ["water", "sea", "open water", "ocean"],
    "concept_type": "class"
  }
}
```

Phase 1 should include a small deterministic vocabulary focused on:

- Sentinel-3, Sentinel-2, Landsat, MODIS
- SLSTR, MSI, OLI, TIRS, MODIS sensors
- S1-S9, visible, NIR, SWIR, thermal, brightness temperature
- NDVI, NDWI, NDSI
- cloud, ice, sea ice, open water, water, land, vegetation, snow
- threshold and comparison terms

### Sensor-Band Mapping

Band names must not be globally normalized without satellite and sensor context. Different sensors use different band numbering systems, so the vocabulary should support mappings at this level:

```text
satellite + sensor + band -> canonical concept
```

Examples:

```json
{
  "Sentinel-3/SLSTR/S7": "thermal_brightness_temperature",
  "Sentinel-3/SLSTR/S8": "thermal_brightness_temperature",
  "Sentinel-3/SLSTR/S9": "thermal_brightness_temperature",
  "Landsat-8/TIRS/B10": "thermal_brightness_temperature",
  "Landsat-8/TIRS/B11": "thermal_brightness_temperature"
}
```

The system may use cross-sensor mappings for recall, but answer generation must mark cross-sensor evidence as indirect when the user asks about a specific sensor.

### Value And Unit Normalization

Thresholds and numeric values must preserve both original text and normalized representation.

The system should distinguish:

- reflectance
- brightness temperature
- radiance
- DN values
- vegetation, water, snow, or ice indices
- percentages
- Kelvin
- Celsius

Example normalized value structure:

```json
{
  "original_text": "BT below 270 K",
  "measurement_type": "brightness_temperature",
  "operator": "<",
  "value": 270.0,
  "unit": "K",
  "normalized_value": 270.0,
  "normalized_unit": "K"
}
```

Phase 1 can store normalized values as strings in `ChunkMetadata.normalized_values`, but the extraction logic should keep enough structure to later become typed objects.

## Production Risks And Design Guardrails

### Paper Quality And Evidence Strength

Not every paper should be treated as equally strong evidence. Phase 1 does not need full paper ranking, but the schema should allow later evidence weighting.

Suggested paper metadata:

- `paper_type`: journal, conference, preprint, report, thesis, unknown
- `peer_reviewed`: true, false, unknown
- `publication_year`
- `dataset_relevance`
- `sensor_relevance`
- `evidence_level`

Retrieval and answer generation can later use these fields to distinguish same-sensor experimental evidence from weaker background or cross-sensor evidence.

### Negative Evidence And Limitations

The system must capture limitations, not only positive classification rules.

Important limitation patterns include:

- thin cloud is difficult
- ice edge is ambiguous
- turbid water causes confusion
- low solar angle affects reflectance
- seasonal variation changes thresholds
- mixed pixels reduce confidence
- sensor noise or calibration affects a feature

These should populate:

- `limitations`
- `failure_modes`
- `confounding_factors`
- `review_required_conditions`

This matters because an operational assistant should know when to ask for manual review instead of forcing a confident label.

### Versioning And Reproducibility

Chunks can change when the parser, chunking strategy, vocabulary, or source file changes. The paper text may be the same, but generated chunks may differ because:

- A parser starts extracting page numbers more accurately.
- A parser starts detecting table captions.
- A chunker changes from paragraph-only chunks to sentence-window chunks.
- A vocabulary starts recognizing range expressions such as `S7-S9`.
- A vocabulary starts normalizing `S7-S9` to `thermal_brightness_temperature`.
- A metadata extractor starts identifying thresholds or limitations.

Every paper and chunk should therefore record:

- `source_hash`
- `parser_version`
- `chunker_version`
- `vocabulary_version`
- `created_at`

Default phase 1 values can be simple strings:

```text
parser_version = "text-parser-v1"
chunker_version = "multi-granularity-v1"
vocabulary_version = "remote-sensing-en-v1"
```

This enables debugging questions such as:

- Did the source file change?
- Did PDF parsing change?
- Did chunk boundaries change?
- Did metadata extraction change?
- Did vocabulary normalization change?

### Human Review Loop

The system should be designed as a decision-support tool, not a fully autonomous labeling authority.

Future feedback should be able to record:

- evidence useful or not useful
- rule correct or incorrect
- threshold applicable or not applicable
- answer needed manual review
- sample label accepted, rejected, or corrected

Phase 1 does not need a full feedback system, but `review_required_conditions` and retrieval provenance should make this future loop possible.

### Gold Evaluation Set

The project should build a small evaluation set early.

Recommended first gold set:

- 5 to 10 representative papers supplied by the user.
- 5 to 20 manually marked evidence items per paper.
- Each evidence item records satellite, sensor, band or layer, class, threshold or trend, page, section, and text span.
- 20 to 50 representative retrieval questions.

This gold set should test whether the chunking strategy improves evidence recall and reduces unsupported answers.

## Observation Sample Design

CSV, Excel, JSON, and table inputs are not knowledge chunks. They represent observation samples that should later be judged using the paper knowledge base.

Phase 1 should define the shape but only implement minimal examples.

```python
@dataclass(frozen=True)
class ObservationSample:
    sample_id: str
    source_type: str
    satellite: str | None
    sensor: str | None
    features: dict[str, float | str]
    normalized_features: dict[str, float | str]
    metadata: dict[str, str]
```

Future flow:

```text
CSV row
  -> ObservationSample
  -> DomainVocabulary normalization
  -> query intent
  -> retriever filters
  -> evidence chunks
  -> grounded answer or rule suggestion
```

Example:

```json
{
  "sample_id": "183_303_1402",
  "satellite": "Sentinel-3",
  "sensor": "SLSTR",
  "features": {
    "S1": 18.0,
    "S7": 276.0,
    "S8": 273.2,
    "S9": 272.6,
    "NDVI": 0.12
  },
  "normalized_features": {
    "visible_reflectance.S1": 18.0,
    "thermal_brightness_temperature.S7": 276.0,
    "thermal_brightness_temperature.S8": 273.2,
    "thermal_brightness_temperature.S9": 272.6,
    "vegetation_index.NDVI": 0.12
  }
}
```

## Retrieval Contract

The chunking layer must produce data that supports these retrieval operations:

```python
@dataclass(frozen=True)
class RetrievalRequest:
    query: str
    chunk_types: list[str]
    metadata_filters: dict[str, list[str]]
    expand_parents: bool
    top_k: int
```

```python
@dataclass(frozen=True)
class RetrievalResult:
    chunk: Chunk
    score: float
    matched_terms: list[str]
    expanded_parent: Chunk | None
    rationale: str
```

Minimum retrieval behavior for phase 1:

- Keyword match over chunk text.
- Keyword match over normalized metadata.
- Metadata filtering by satellite, sensor, band, class, evidence type, and section.
- Chunk type boosting.
- Parent expansion for child hits.

Later retrieval implementations can add:

- Embedding search.
- BM25.
- Dense-sparse hybrid retrieval.
- Cross-encoder reranking.
- LLM-based rule extraction over retrieved evidence.

## Evidence Boundary And Refusal Policy

The system must be designed to avoid unsupported claims. The answer layer is out of scope for full implementation in phase 1, but phase 1 schemas, chunks, metadata, and retrieval results must carry enough information to enforce this policy later.

### Allowed Answer Scope

The system may answer questions that can be grounded in retrieved paper evidence:

- Which satellite, sensor, dataset, or study area a paper uses.
- Which bands, layers, indices, or derived features a paper discusses.
- Which classes a paper distinguishes, such as cloud, sea, ice, water, land, vegetation, or snow.
- Which thresholds, value ranges, trends, or comparison relationships appear in the paper.
- Which method, model, feature, table, figure, or experiment supports a class decision.
- Which paper evidence is relevant to a provided observation sample.

### Disallowed Or Restricted Answer Scope

The system must not present unsupported conclusions as facts:

- It must not claim that a pixel, sample, or region is definitely a class unless a downstream classifier has made that decision and the answer states the classifier context.
- It must not generalize a rule from one satellite or sensor to another without saying the evidence comes from a different source.
- It must not invent thresholds when retrieved chunks do not contain threshold evidence.
- It must not claim one layer is the most important unless method, result, table, figure, or ablation evidence supports that claim.
- It must not cite a paper, page, or section that was not retrieved.
- It must not hide conflicts between papers.

Preferred language for uncertain operational judgments:

```text
Based on the retrieved paper evidence, this sample is more consistent with cloud-related rules, but the evidence is not sufficient for a definitive label and should be manually reviewed.
```

### Minimum Evidence Requirements

The retrieval layer should expose enough information for the answer layer to enforce hard gates.

For a class interpretation answer, require at least one retrieved chunk with:

- `target_classes` matching the asked class or normalized class alias.
- Either `bands_or_layers`, `indices`, `method_terms`, or `evidence_types` relevant to the question.
- A valid citation path: `paper_id`, `chunk_id`, `section_title`, and page fields when available.

For a threshold answer, require at least one retrieved chunk with:

- A non-empty `thresholds` field.
- A related band, layer, index, feature, or class in metadata.

For a layer importance answer, require at least one retrieved chunk with:

- `evidence_types` containing result-like or comparison-like evidence.
- A source section such as method, experiment, result, discussion, figure, or table.
- A linked parent chunk for broader context.

For a sensor-specific answer, require:

- Same-sensor evidence when the user asks about a specific sensor.
- If only cross-sensor evidence exists, the answer must explicitly mark it as indirect evidence.

### Refusal Conditions

The answer layer should refuse or return an insufficient-evidence response when:

- No relevant chunks are retrieved.
- Retrieved chunks do not include the metadata required by the question type.
- The question asks for a threshold but no retrieved chunk has `thresholds`.
- The question asks for strongest layer or feature importance but only background chunks are retrieved.
- Retrieved evidence comes from a different satellite or sensor and the user asked for a specific one.
- Retrieved papers conflict and the system cannot identify which evidence applies to the user's context.

Example refusal:

```json
{
  "answer_type": "insufficient_evidence",
  "answer": "The current knowledge base does not contain enough evidence to answer this threshold question.",
  "missing_evidence": ["thresholds", "same-sensor evidence"],
  "retrieved_chunk_ids": ["chunk_014", "chunk_021"]
}
```

### Confidence Policy

Confidence should be rule-based first, not based only on model wording.

High confidence requires:

- Same satellite or sensor when specified.
- Relevant class and band or layer metadata.
- At least one precise evidence chunk such as `rule_candidate`, `sentence_window_child`, or `figure_table`.
- A parent chunk for context.
- Direct threshold, trend, comparison, or result evidence matching the question.

Medium confidence applies when:

- Relevant evidence exists, but the threshold is missing or only a trend is described.
- Evidence is from the same broad sensor family or similar feature type but not the exact requested sensor.
- Only one paper supports the claim.

Low confidence applies when:

- Evidence is mostly background or method description.
- Metadata is incomplete.
- The answer requires inference across chunks that do not directly state the relationship.

The system should refuse instead of returning low confidence when the required evidence for the question type is completely missing.

### Required Answer Provenance

Every grounded answer should be able to return citations in this shape:

```json
{
  "paper_id": "paper_sentinel3_slstr_001",
  "title": "Paper title if available",
  "chunk_id": "rule_candidate_003",
  "parent_chunk_id": "section_parent_results_001",
  "section_title": "Results",
  "page_start": 4,
  "page_end": 5,
  "evidence_text": "Short retrieved evidence excerpt"
}
```

This requirement means phase 1 chunks must always preserve:

- `paper_id`
- `chunk_id`
- `parent_id`
- `source_block_ids`
- `section_title`
- `section_type`
- `page_start`
- `page_end`
- `chunk_type`
- `retrieval_profile`

### Separating Evidence From Inference

Later answer generation should separate direct evidence from system inference:

```text
Evidence:
- Paper A, Results, p.4: S7-S9 brightness temperature was lower for cloud pixels.
- Paper B, Table 2, p.6: thermal features contributed strongly to cloud and ice separation.

Inference:
- For Sentinel-3 SLSTR-like thermal channels, brightness temperature is likely useful for cloud/ice separation, but this should be validated on the target dataset before use as a final rule.
```

This prevents the system from turning a paper-specific observation into a universal remote-sensing rule.

## First-Phase Evaluation

Chunking quality should be evaluated by evidence recall, not by visual neatness.

Test fixtures should include simulated paper passages containing:

- Sentinel-3 SLSTR S1-S9 examples.
- Landsat thermal examples.
- Cloud, ice, sea, water, land, vegetation, and snow classes.
- Threshold examples such as `BT < 270 K`, `NDVI > 0.3`, or `NDSI above 0.4`.
- Comparison examples such as higher reflectance, lower brightness temperature, warmer water, colder cloud top.
- Figure and table captions.

Required tests:

- Text or Markdown input becomes a `Paper` with sections and blocks.
- Section parent chunks are generated.
- Paragraph child chunks are generated.
- Sentence window chunks are generated around band, class, or threshold triggers.
- Figure/table chunks bind captions to nearby text.
- Rule candidate chunks are generated when a passage contains class-feature-threshold or class-feature-trend relationships.
- Metadata extraction identifies satellites, sensors, bands, indices, classes, thresholds, and directionality.
- Child chunks link to parent chunks.
- Mock retrieval finds a relevant rule chunk for a query about a class and band.
- Mock retrieval can expand from a child hit to its parent section.
- Mock retrieval returns an insufficient-evidence result for threshold questions when no threshold chunk exists.
- Mock retrieval marks cross-sensor evidence as indirect when the query specifies a different sensor.
- Retrieval results contain enough provenance to cite paper, section, page, chunk, and parent chunk.
- Paper and chunk objects include `source_hash`, parser version, chunker version, and vocabulary version.
- Vocabulary maps bands with satellite and sensor context instead of treating band names as globally equivalent.
- Metadata extraction preserves original threshold text and normalized value strings.
- Metadata extraction captures limitation or review-required language such as thin cloud, mixed pixels, or ambiguous ice edge.
- English-only vocabulary tests avoid Chinese aliases in phase 1.

## Suggested Project Layout

```text
satellite_paper_rag/
  README.md
  pyproject.toml
  src/
    satellite_paper_rag/
      __init__.py
      domain/
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
        section_chunker.py
        paragraph_chunker.py
        sentence_window_chunker.py
        figure_table_chunker.py
        rule_candidate_chunker.py
        metadata_enricher.py
      retrieval/
        __init__.py
        contract.py
        mock_hybrid_retriever.py
      observations/
        __init__.py
        schema.py
        feature_normalizer.py
      schemas.py
  tests/
    fixtures/
      sample_sentinel3_paper.md
      sample_landsat_paper.txt
    test_text_parser.py
    test_chunking_pipeline.py
    test_metadata_enricher.py
    test_mock_retriever.py
```

## Implementation Notes

- Use standard-library dataclasses first. Add Pydantic only if validation needs become complex.
- Keep parsers, chunkers, metadata enrichment, and retrieval separate.
- Avoid making one `splitter.py` responsible for everything.
- Use deterministic extraction first so tests are stable.
- Treat PDF parsing as replaceable infrastructure.
- Use parent-child IDs that are deterministic for repeatable tests.
- Preserve source provenance in every chunk.
- Do not require network access or paid APIs for phase 1 tests.

## Open Decisions For Implementation Planning

- Whether to use only `unittest` or adopt `pytest`.
- Whether to initialize `satellite_paper_rag` as a git repository, because `D:\CodexProject` is not currently a git repository.
- Whether the first PDF parser should be a no-op interface only or a basic PyMuPDF-backed adapter if the dependency is already installed in the `transformer` environment.
- Whether to keep all metadata values as lists of strings in phase 1 or introduce typed enums early.
