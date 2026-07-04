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
