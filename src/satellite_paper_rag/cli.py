from __future__ import annotations

import argparse
import json
from pathlib import Path

from satellite_paper_rag.chunking.pipeline import PaperChunkingPipeline
from satellite_paper_rag.domain.vocabulary import DomainVocabulary
from satellite_paper_rag.embeddings.client import DeterministicEmbeddingClient
from satellite_paper_rag.embeddings.vector_index import LocalVectorIndex, VectorIndexBuilder, VectorSearchResult
from satellite_paper_rag.extraction.rule_extractor import ExtractedRule, RuleCandidateExtractor
from satellite_paper_rag.parsing.markdown_parser import MarkdownPaperParser
from satellite_paper_rag.parsing.pdf_parser import PdfPaperParser
from satellite_paper_rag.parsing.text_parser import TextPaperParser
from satellite_paper_rag.retrieval.contract import RetrievalRequest, RetrievalResult
from satellite_paper_rag.retrieval.mock_hybrid_retriever import MockHybridRetriever


DEFAULT_PAPER_DIR = Path("data") / "papers"
DEFAULT_INDEX_DIR = Path("data") / "index"


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


def rule_to_dict(rule: ExtractedRule) -> dict[str, object]:
    return {
        "rule_id": rule.rule_id,
        "paper_id": rule.paper_id,
        "chunk_id": rule.chunk_id,
        "rule_type": rule.rule_type,
        "condition_text": rule.condition_text,
        "target_classes": rule.target_classes,
        "features": rule.features,
        "thresholds": rule.thresholds,
        "normalized_conditions": rule.normalized_conditions,
        "page_start": rule.page_start,
        "page_end": rule.page_end,
        "section_title": rule.section_title,
        "score": rule.score,
        "evidence_terms": rule.evidence_terms,
    }


def vector_result_to_dict(result: VectorSearchResult) -> dict[str, object]:
    return {
        "score": result.score,
        "chunk_id": result.chunk.chunk_id,
        "chunk_type": result.chunk.chunk_type,
        "paper_id": result.chunk.paper_id,
        "section_title": result.chunk.section_title,
        "page_start": result.chunk.page_start,
        "page_end": result.chunk.page_end,
        "text": result.chunk.text,
        "metadata": {
            "satellites": result.chunk.metadata.satellites,
            "sensors": result.chunk.metadata.sensors,
            "bands_or_layers": result.chunk.metadata.bands_or_layers,
            "indices": result.chunk.metadata.indices,
            "target_classes": result.chunk.metadata.target_classes,
            "thresholds": result.chunk.metadata.thresholds,
            "evidence_types": result.chunk.metadata.evidence_types,
        },
    }


def resolve_query_path(args: argparse.Namespace) -> Path:
    if args.file:
        return Path(args.file)
    paper_path = Path(args.paper)
    if paper_path.exists():
        return paper_path
    return DEFAULT_PAPER_DIR / paper_path


def build_chunks(path: Path):
    paper = parse_paper(path)
    vocabulary = DomainVocabulary.default()
    chunks = PaperChunkingPipeline(vocabulary).chunk(paper)
    return paper, chunks


def build_embedder() -> DeterministicEmbeddingClient:
    return DeterministicEmbeddingClient()


def build_index_for_path(path: Path, index_dir: Path):
    paper, chunks = build_chunks(path)
    embedder = build_embedder()
    index = LocalVectorIndex(index_dir)
    index_path = VectorIndexBuilder(embedder, index).index(paper.paper_id, chunks)
    return paper, chunks, embedder, index, index_path


def query_file(args: argparse.Namespace) -> int:
    path = resolve_query_path(args)
    paper, chunks = build_chunks(path)
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


def index_paper(args: argparse.Namespace) -> int:
    path = resolve_query_path(args)
    paper, chunks, embedder, _, index_path = build_index_for_path(path, Path(args.index_dir))
    payload = {
        "paper_id": paper.paper_id,
        "title": paper.title,
        "source_type": paper.source_type,
        "chunk_count": len(chunks),
        "embedding_model": embedder.model_name,
        "embedding_dimension": embedder.dimension,
        "index_path": str(index_path),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def semantic_query(args: argparse.Namespace) -> int:
    path = resolve_query_path(args)
    paper, chunks = build_chunks(path)
    embedder = build_embedder()
    index = LocalVectorIndex(Path(args.index_dir))
    index_status = "loaded"
    if args.rebuild_index or not index.exists(paper.paper_id):
        VectorIndexBuilder(embedder, index).index(paper.paper_id, chunks)
        index_status = "rebuilt" if args.rebuild_index else "created"
    results = index.search(
        query=args.query,
        embedder=embedder,
        paper_id=paper.paper_id,
        top_k=args.top_k,
        chunk_types=args.chunk_type,
    )
    payload = {
        "paper_id": paper.paper_id,
        "title": paper.title,
        "source_type": paper.source_type,
        "query": args.query,
        "embedding_model": embedder.model_name,
        "index_status": index_status,
        "results": [vector_result_to_dict(result) for result in results],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def extract_rules(args: argparse.Namespace) -> int:
    path = resolve_query_path(args)
    paper, chunks = build_chunks(path)
    candidate_chunks = chunks
    semantic_candidates: list[dict[str, object]] = []
    if args.semantic_query:
        embedder = build_embedder()
        index = LocalVectorIndex(Path(args.index_dir))
        if args.rebuild_index or not index.exists(paper.paper_id):
            VectorIndexBuilder(embedder, index).index(paper.paper_id, chunks)
        semantic_results = index.search(
            query=args.semantic_query,
            embedder=embedder,
            paper_id=paper.paper_id,
            top_k=args.candidate_k,
            chunk_types=args.chunk_type,
        )
        candidate_chunks = [result.chunk for result in semantic_results]
        semantic_candidates = [vector_result_to_dict(result) for result in semantic_results]
    rules = RuleCandidateExtractor().extract(candidate_chunks, top_k=args.top_k)
    payload = {
        "paper_id": paper.paper_id,
        "title": paper.title,
        "source_type": paper.source_type,
        "semantic_query": args.semantic_query,
        "semantic_candidates": semantic_candidates,
        "rules": [rule_to_dict(rule) for rule in rules],
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
    index = subparsers.add_parser("index-paper")
    index_source = index.add_mutually_exclusive_group(required=True)
    index_source.add_argument("--file")
    index_source.add_argument("--paper", help="Paper filename under data/papers, or an existing path.")
    index.add_argument("--index-dir", default=str(DEFAULT_INDEX_DIR))
    index.set_defaults(func=index_paper)
    semantic = subparsers.add_parser("semantic-query")
    semantic_source = semantic.add_mutually_exclusive_group(required=True)
    semantic_source.add_argument("--file")
    semantic_source.add_argument("--paper", help="Paper filename under data/papers, or an existing path.")
    semantic.add_argument("--query", required=True)
    semantic.add_argument("--top-k", type=int, default=5)
    semantic.add_argument("--index-dir", default=str(DEFAULT_INDEX_DIR))
    semantic.add_argument("--rebuild-index", action="store_true")
    semantic.add_argument(
        "--chunk-type",
        action="append",
        default=["rule_candidate", "sentence_window_child", "paragraph_child", "figure_table"],
    )
    semantic.set_defaults(func=semantic_query)
    extract = subparsers.add_parser("extract-rules")
    extract_source = extract.add_mutually_exclusive_group(required=True)
    extract_source.add_argument("--file")
    extract_source.add_argument("--paper", help="Paper filename under data/papers, or an existing path.")
    extract.add_argument("--top-k", type=int, default=20)
    extract.add_argument("--semantic-query")
    extract.add_argument("--candidate-k", type=int, default=20)
    extract.add_argument("--index-dir", default=str(DEFAULT_INDEX_DIR))
    extract.add_argument("--rebuild-index", action="store_true")
    extract.add_argument(
        "--chunk-type",
        action="append",
        default=["rule_candidate", "sentence_window_child", "paragraph_child", "figure_table"],
    )
    extract.set_defaults(func=extract_rules)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
