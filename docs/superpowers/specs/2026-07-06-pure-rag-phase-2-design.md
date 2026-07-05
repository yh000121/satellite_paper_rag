# Pure RAG Phase 2 Design

## Goal

Upgrade the existing Pure RAG foundation so it can ingest real PDF papers, apply configurable recursive fallback splitting for long text, and run a basic evidence query from the command line.

## Scope

In scope:

- Parse user-provided PDFs with PyMuPDF.
- Preserve page provenance in `PaperBlock.page_start` and `page_end`.
- Keep PDF output normalized into the existing `Paper`, `PaperSection`, and `PaperBlock` schema.
- Add `ChunkingConfig` and `RetrievalConfig` for tunable chunking and retrieval parameters.
- Add a LangChain `RecursiveCharacterTextSplitter` adapter as a fallback for long section or paragraph text.
- Add a basic CLI that parses a file, chunks it, retrieves evidence, and prints structured results.

Out of scope:

- Production vector database.
- Embeddings.
- BM25.
- LLM answer generation.
- NODE model integration.
- Batch CSV/Excel inference.

## Architecture

The parser and splitter libraries are infrastructure. They should not own the domain logic.

```text
PDF / Markdown / Text
  -> parser adapter
  -> Paper / Section / Block
  -> PaperChunkingPipeline with ChunkingConfig
  -> MetadataEnricher / rule_candidate
  -> MockHybridRetriever with RetrievalConfig
  -> CLI evidence output
```

PyMuPDF is only responsible for extracting page text. LangChain recursive splitting is only responsible for length fallback. Domain vocabulary, metadata enrichment, rule candidates, parent-child links, and evidence policy remain project-owned.

## Acceptance Criteria

- A synthetic PDF created in tests can be parsed into `Paper` with page numbers.
- Existing Markdown/text tests still pass.
- Long paragraphs can be split into multiple child chunks through LangChain fallback.
- Chunking behavior can be tuned by config.
- CLI can query Markdown/text/PDF files and return evidence JSON.
- All tests pass with `PYTHONPATH=src python -m unittest discover -s tests -v`.

