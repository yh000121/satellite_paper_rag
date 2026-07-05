# Pure RAG Phase 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add PDF ingestion, configurable LangChain recursive fallback splitting, and a basic evidence query CLI.

**Architecture:** Keep the existing domain-owned schemas, chunk types, metadata enrichment, and mock evidence policy. Add PyMuPDF and LangChain as adapter layers under parsing and chunking so they can be replaced later without rewriting domain logic.

**Tech Stack:** Python 3.13, PyMuPDF, LangChain text splitters, standard-library `unittest`.

---

### Task 1: Config

- Add `src/satellite_paper_rag/config.py`.
- Add tests for default `ChunkingConfig` and `RetrievalConfig`.

### Task 2: Recursive Splitter Adapter

- Add `src/satellite_paper_rag/chunking/recursive_splitter.py`.
- Use LangChain `RecursiveCharacterTextSplitter`.
- Add tests proving long text splits and short text stays whole.

### Task 3: PDF Parser

- Replace `PdfPaperParser` boundary with a PyMuPDF-backed parser.
- Add synthetic PDF fixture generation in tests.
- Preserve page numbers.

### Task 4: Pipeline Config Integration

- Accept `ChunkingConfig` in `PaperChunkingPipeline`.
- Split long paragraph child chunks with recursive fallback.
- Keep rule candidate and sentence window behavior domain-owned.

### Task 5: Query CLI

- Add `src/satellite_paper_rag/cli.py`.
- Support `query --file <path> --query <text>`.
- Print JSON evidence results.

### Task 6: Verification And Commit

- Run the full test suite.
- Commit and push Phase 2 changes.

