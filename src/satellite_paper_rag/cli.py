from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from satellite_paper_rag.chunking.pipeline import PaperChunkingPipeline
from satellite_paper_rag.domain.vocabulary import DomainVocabulary
from satellite_paper_rag.embeddings.client import DashScopeEmbeddingClient, DeterministicEmbeddingClient, EmbeddingClient
from satellite_paper_rag.embeddings.vector_index import LocalVectorIndex, VectorIndexBuilder, VectorSearchResult
from satellite_paper_rag.extraction.rule_extractor import ExtractedRule, RuleCandidateExtractor
from satellite_paper_rag.parsing.markdown_parser import MarkdownPaperParser
from satellite_paper_rag.parsing.pdf_parser import PdfPaperParser
from satellite_paper_rag.parsing.text_parser import TextPaperParser
from satellite_paper_rag.retrieval.contract import RetrievalRequest, RetrievalResult
from satellite_paper_rag.retrieval.mock_hybrid_retriever import MockHybridRetriever
from satellite_paper_rag.schemas import CHUNKER_VERSION, Paper


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
        "rule_scope": rule.rule_scope,
        "dynamic_threshold_formula": rule.dynamic_threshold_formula,
        "validation_only": rule.validation_only,
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


def build_embedder(args: argparse.Namespace | None = None) -> EmbeddingClient:
    provider = (getattr(args, "embedding_provider", None) or os.getenv("EMBEDDING_PROVIDER") or "deterministic").lower()
    model_name = getattr(args, "embedding_model", None) if args else None
    base_url = getattr(args, "embedding_base_url", None) if args else None
    if provider in {"deterministic", "local"}:
        return DeterministicEmbeddingClient()
    if provider in {"dashscope", "qwen"}:
        return DashScopeEmbeddingClient(
            model_name=model_name,
            base_url=base_url,
            progress_callback=_embedding_progress_callback if getattr(args, "verbose", False) else None,
        )
    raise ValueError(f"Unsupported embedding provider: {provider}")


def build_index_for_path(path: Path, index_dir: Path, args: argparse.Namespace | None = None):
    log_progress(args, "Parsing and chunking paper...")
    paper, chunks = build_chunks(path)
    log_progress(args, f"Built {len(chunks)} chunks.")
    embedder = build_embedder(args)
    index = LocalVectorIndex(index_dir)
    log_progress(args, f"Embedding chunks with {embedder.model_name}...")
    index_path = VectorIndexBuilder(embedder, index).index(paper.paper_id, chunks)
    log_progress(args, f"Index written to {index_path}.")
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
    paper, chunks, embedder, _, index_path = build_index_for_path(path, Path(args.index_dir), args)
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
    log_progress(args, "Parsing and chunking paper...")
    paper, chunks = build_chunks(path)
    log_progress(args, f"Built {len(chunks)} chunks.")
    embedder = build_embedder(args)
    index = LocalVectorIndex(Path(args.index_dir))
    index_status = "loaded"
    if args.rebuild_index or not index_is_compatible(index, paper, embedder):
        log_progress(args, f"Index missing or rebuild requested; embedding chunks with {embedder.model_name}...")
        VectorIndexBuilder(embedder, index).index(paper.paper_id, chunks)
        index_status = "rebuilt" if args.rebuild_index else "created"
    log_progress(args, "Embedding query and searching local index...")
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
        embedder = build_embedder(args)
        index = LocalVectorIndex(Path(args.index_dir))
        if args.rebuild_index or not index_is_compatible(index, paper, embedder):
            log_progress(args, f"Index missing or rebuild requested; embedding chunks with {embedder.model_name}...")
            VectorIndexBuilder(embedder, index).index(paper.paper_id, chunks)
        log_progress(args, "Embedding semantic query and selecting candidate chunks...")
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
    add_embedding_arguments(index)
    index.set_defaults(func=index_paper)
    semantic = subparsers.add_parser("semantic-query")
    semantic_source = semantic.add_mutually_exclusive_group(required=True)
    semantic_source.add_argument("--file")
    semantic_source.add_argument("--paper", help="Paper filename under data/papers, or an existing path.")
    semantic.add_argument("--query", required=True)
    semantic.add_argument("--top-k", type=int, default=5)
    semantic.add_argument("--index-dir", default=str(DEFAULT_INDEX_DIR))
    semantic.add_argument("--rebuild-index", action="store_true")
    add_embedding_arguments(semantic)
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
    add_embedding_arguments(extract)
    extract.add_argument(
        "--chunk-type",
        action="append",
        default=["rule_candidate", "sentence_window_child", "paragraph_child", "figure_table"],
    )
    extract.set_defaults(func=extract_rules)
    return parser


def add_embedding_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--embedding-provider",
        default=os.getenv("EMBEDDING_PROVIDER", "deterministic"),
        choices=["deterministic", "local", "dashscope", "qwen"],
        help="Embedding provider. Use 'dashscope' or 'qwen' for Alibaba Cloud Model Studio/DashScope.",
    )
    parser.add_argument(
        "--embedding-model",
        default=os.getenv("DASHSCOPE_EMBEDDING_MODEL"),
        help="Embedding model name for the selected provider, for example text-embedding-v4.",
    )
    parser.add_argument(
        "--embedding-base-url",
        default=os.getenv("DASHSCOPE_BASE_URL"),
        help="OpenAI-compatible embedding base URL. Defaults to DashScope compatible-mode v1 for dashscope/qwen.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print progress logs to stderr.")


def log_progress(args: argparse.Namespace | None, message: str) -> None:
    if args is not None and getattr(args, "verbose", False):
        print(message, file=sys.stderr, flush=True)


def _embedding_progress_callback(current: int, total: int, batch_size: int) -> None:
    print(f"Embedding batch {current}/{total} ({batch_size} texts)...", file=sys.stderr, flush=True)


def index_is_compatible(index: LocalVectorIndex, paper: Paper, embedder: EmbeddingClient) -> bool:
    return index.exists(
        paper.paper_id,
        model_name=embedder.model_name,
        chunker_version=CHUNKER_VERSION,
        parser_version=paper.parser_version,
        vocabulary_version=paper.vocabulary_version,
        source_hash=paper.source_hash,
    )


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
