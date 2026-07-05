from __future__ import annotations

import argparse
import json
from pathlib import Path

from satellite_paper_rag.chunking.pipeline import PaperChunkingPipeline
from satellite_paper_rag.domain.vocabulary import DomainVocabulary
from satellite_paper_rag.parsing.markdown_parser import MarkdownPaperParser
from satellite_paper_rag.parsing.pdf_parser import PdfPaperParser
from satellite_paper_rag.parsing.text_parser import TextPaperParser
from satellite_paper_rag.retrieval.contract import RetrievalRequest, RetrievalResult
from satellite_paper_rag.retrieval.mock_hybrid_retriever import MockHybridRetriever


DEFAULT_PAPER_DIR = Path("data") / "papers"


def parse_paper(path: Path):
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return PdfPaperParser().parse(path)
    if suffix in {".md", ".markdown"}:
        return MarkdownPaperParser().parse(path)
    return TextPaperParser().parse(path)


def result_to_dict(result: RetrievalResult) -> dict[str, object]:
    return {
        "answer_type": result.answer_type,
        "score": result.score,
        "chunk_id": result.chunk.chunk_id,
        "chunk_type": result.chunk.chunk_type,
        "paper_id": result.chunk.paper_id,
        "section_title": result.chunk.section_title,
        "page_start": result.chunk.page_start,
        "page_end": result.chunk.page_end,
        "text": result.chunk.text,
        "matched_terms": result.matched_terms,
        "missing_evidence": result.missing_evidence,
        "is_indirect_evidence": result.is_indirect_evidence,
        "expanded_parent_id": result.expanded_parent.chunk_id if result.expanded_parent else None,
        "metadata": {
            "satellites": result.chunk.metadata.satellites,
            "sensors": result.chunk.metadata.sensors,
            "bands_or_layers": result.chunk.metadata.bands_or_layers,
            "indices": result.chunk.metadata.indices,
            "target_classes": result.chunk.metadata.target_classes,
            "thresholds": result.chunk.metadata.thresholds,
            "evidence_types": result.chunk.metadata.evidence_types,
            "limitations": result.chunk.metadata.limitations,
            "review_required_conditions": result.chunk.metadata.review_required_conditions,
        },
    }


def resolve_query_path(args: argparse.Namespace) -> Path:
    if args.file:
        return Path(args.file)
    paper_path = Path(args.paper)
    if paper_path.exists():
        return paper_path
    return DEFAULT_PAPER_DIR / paper_path


def query_file(args: argparse.Namespace) -> int:
    path = resolve_query_path(args)
    paper = parse_paper(path)
    vocabulary = DomainVocabulary.default()
    chunks = PaperChunkingPipeline(vocabulary).chunk(paper)
    retriever = MockHybridRetriever(chunks)
    results = retriever.retrieve(
        RetrievalRequest(
            query=args.query,
            chunk_types=args.chunk_type,
            metadata_filters={},
            expand_parents=True,
            top_k=args.top_k,
            requires_threshold=args.requires_threshold,
            requested_sensor=args.requested_sensor,
        )
    )
    payload = {
        "paper_id": paper.paper_id,
        "title": paper.title,
        "source_type": paper.source_type,
        "query": args.query,
        "results": [result_to_dict(result) for result in results],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="satellite_paper_rag")
    subparsers = parser.add_subparsers(dest="command", required=True)
    query = subparsers.add_parser("query")
    source = query.add_mutually_exclusive_group(required=True)
    source.add_argument("--file")
    source.add_argument("--paper", help="Paper filename under data/papers, or an existing path.")
    query.add_argument("--query", required=True)
    query.add_argument("--top-k", type=int, default=5)
    query.add_argument("--requires-threshold", action="store_true")
    query.add_argument("--requested-sensor")
    query.add_argument(
        "--chunk-type",
        action="append",
        default=["rule_candidate", "sentence_window_child", "figure_table", "paragraph_child"],
    )
    query.set_defaults(func=query_file)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
