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
