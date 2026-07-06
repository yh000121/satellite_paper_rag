import tempfile
import unittest
from pathlib import Path

from satellite_paper_rag.chunking.pipeline import PaperChunkingPipeline
from satellite_paper_rag.domain.vocabulary import DomainVocabulary
from satellite_paper_rag.embeddings.client import DeterministicEmbeddingClient
from satellite_paper_rag.embeddings.vector_index import LocalVectorIndex, VectorIndexBuilder
from satellite_paper_rag.parsing.markdown_parser import MarkdownPaperParser


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


if __name__ == "__main__":
    unittest.main()
