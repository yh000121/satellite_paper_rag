import tempfile
import unittest
from pathlib import Path

from satellite_paper_rag.chunking.pipeline import PaperChunkingPipeline
from satellite_paper_rag.domain.vocabulary import DomainVocabulary
from satellite_paper_rag.embeddings.client import DeterministicEmbeddingClient
from satellite_paper_rag.embeddings.vector_index import LocalVectorIndex, VectorIndexBuilder
from satellite_paper_rag.parsing.markdown_parser import MarkdownPaperParser
from satellite_paper_rag.schemas import CHUNKER_VERSION, Chunk, ChunkMetadata


class EmbeddingIndexTest(unittest.TestCase):
    def test_semantic_query_finds_cloud_threshold_chunk(self):
        paper = MarkdownPaperParser().parse(Path("tests/fixtures/sample_sentinel3_paper.md"))
        chunks = PaperChunkingPipeline(DomainVocabulary.default()).chunk(paper)
        embedder = DeterministicEmbeddingClient()

        with tempfile.TemporaryDirectory() as tmp_dir:
            index = LocalVectorIndex(Path(tmp_dir))
            VectorIndexBuilder(embedder, index).index(paper.paper_id, chunks)
            results = index.search("How are cloudy pixels identified?", embedder, top_k=3)

        self.assertTrue(results)
        self.assertIn("BT below 270 K", results[0].chunk.text)
        self.assertGreater(results[0].score, 0)

    def test_index_persists_and_loads_chunk_records(self):
        paper = MarkdownPaperParser().parse(Path("tests/fixtures/sample_sentinel3_paper.md"))
        chunks = PaperChunkingPipeline(DomainVocabulary.default()).chunk(paper)
        embedder = DeterministicEmbeddingClient()

        with tempfile.TemporaryDirectory() as tmp_dir:
            index = LocalVectorIndex(Path(tmp_dir))
            VectorIndexBuilder(embedder, index).index(paper.paper_id, chunks[:2])
            records = index.load(paper.paper_id)

        self.assertEqual(len(records), 2)
        self.assertEqual(records[0].chunk.paper_id, paper.paper_id)
        self.assertEqual(len(records[0].embedding), embedder.dimension)

    def test_index_path_is_scoped_by_embedding_model_and_chunker_version(self):
        paper = MarkdownPaperParser().parse(Path("tests/fixtures/sample_sentinel3_paper.md"))
        chunks = PaperChunkingPipeline(DomainVocabulary.default()).chunk(paper)
        embedder = DeterministicEmbeddingClient()

        with tempfile.TemporaryDirectory() as tmp_dir:
            index = LocalVectorIndex(Path(tmp_dir))
            index_path = VectorIndexBuilder(embedder, index).index(paper.paper_id, chunks[:2])

            self.assertIn("deterministic-local-v1", index_path.name)
            self.assertIn(CHUNKER_VERSION, index_path.name)
            self.assertTrue(index.exists(paper.paper_id, model_name=embedder.model_name, chunker_version=CHUNKER_VERSION))
            self.assertFalse(index.exists(paper.paper_id, model_name="other-model", chunker_version=CHUNKER_VERSION))
            self.assertFalse(index.exists(paper.paper_id, model_name=embedder.model_name, chunker_version="old-chunker"))

    def test_index_compatibility_checks_source_and_parser_versions(self):
        paper = MarkdownPaperParser().parse(Path("tests/fixtures/sample_sentinel3_paper.md"))
        chunks = PaperChunkingPipeline(DomainVocabulary.default()).chunk(paper)
        embedder = DeterministicEmbeddingClient()

        with tempfile.TemporaryDirectory() as tmp_dir:
            index = LocalVectorIndex(Path(tmp_dir))
            VectorIndexBuilder(embedder, index).index(paper.paper_id, chunks[:2])

            self.assertTrue(
                index.exists(
                    paper.paper_id,
                    model_name=embedder.model_name,
                    chunker_version=CHUNKER_VERSION,
                    parser_version=paper.parser_version,
                    vocabulary_version=paper.vocabulary_version,
                    source_hash=paper.source_hash,
                )
            )
            self.assertFalse(
                index.exists(
                    paper.paper_id,
                    model_name=embedder.model_name,
                    chunker_version=CHUNKER_VERSION,
                    parser_version=paper.parser_version,
                    vocabulary_version=paper.vocabulary_version,
                    source_hash="different-source-hash",
                )
            )

    def test_search_filters_short_low_quality_paragraph_chunks(self):
        short_chunk = Chunk(
            chunk_id="paper_paragraph_child_block_0001",
            paper_id="paper",
            chunk_type="paragraph_child",
            text="The clear-sky probability P",
            parent_id=None,
            child_ids=[],
            source_block_ids=["block_0001"],
            page_start=1,
            page_end=1,
            section_title="Methods",
            section_type="method",
            metadata=ChunkMetadata(target_classes=["cloud"]),
            retrieval_profile="semantic_recall",
        )
        evidence_chunk = Chunk(
            chunk_id="paper_sentence_window_child_block_0002",
            paper_id="paper",
            chunk_type="sentence_window_child",
            text="A threshold of 0.9 is applied to clear-sky probability to generate a binary cloud mask.",
            parent_id=None,
            child_ids=[],
            source_block_ids=["block_0002"],
            page_start=1,
            page_end=1,
            section_title="Results",
            section_type="result",
            metadata=ChunkMetadata(target_classes=["cloud"], thresholds=["threshold of 0.9"]),
            retrieval_profile="precise_evidence",
        )
        embedder = DeterministicEmbeddingClient()

        with tempfile.TemporaryDirectory() as tmp_dir:
            index = LocalVectorIndex(Path(tmp_dir))
            VectorIndexBuilder(embedder, index).index("paper", [short_chunk, evidence_chunk])
            results = index.search("clear-sky probability threshold cloud mask", embedder, paper_id="paper", top_k=5)

        self.assertTrue(results)
        self.assertNotEqual(results[0].chunk.chunk_id, short_chunk.chunk_id)
        self.assertEqual(results[0].chunk.chunk_id, evidence_chunk.chunk_id)


if __name__ == "__main__":
    unittest.main()
