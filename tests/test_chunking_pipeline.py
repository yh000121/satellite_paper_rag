import unittest
from pathlib import Path

from satellite_paper_rag.chunking.pipeline import PaperChunkingPipeline
from satellite_paper_rag.config import ChunkingConfig
from satellite_paper_rag.domain.vocabulary import DomainVocabulary
from satellite_paper_rag.parsing.markdown_parser import MarkdownPaperParser
from satellite_paper_rag.schemas import Paper, PaperBlock, PaperSection, compute_source_hash


FIXTURES = Path(__file__).parent / "fixtures"


class ChunkingPipelineTest(unittest.TestCase):
    def test_generates_multiple_chunk_types_with_parent_child_links(self):
        paper = MarkdownPaperParser().parse(FIXTURES / "sample_sentinel3_paper.md")
        chunks = PaperChunkingPipeline(DomainVocabulary.default()).chunk(paper)

        chunk_types = {chunk.chunk_type for chunk in chunks}
        self.assertIn("paper_summary", chunk_types)
        self.assertIn("section_parent", chunk_types)
        self.assertIn("paragraph_child", chunk_types)
        self.assertIn("sentence_window_child", chunk_types)
        self.assertIn("figure_table", chunk_types)
        self.assertIn("rule_candidate", chunk_types)

        parents = {chunk.chunk_id for chunk in chunks if chunk.chunk_type == "section_parent"}
        child_chunks = [chunk for chunk in chunks if chunk.parent_id is not None]
        self.assertTrue(child_chunks)
        self.assertTrue(all(chunk.parent_id in parents for chunk in child_chunks))

    def test_rule_candidate_preserves_domain_metadata_and_versions(self):
        paper = MarkdownPaperParser().parse(FIXTURES / "sample_sentinel3_paper.md")
        chunks = PaperChunkingPipeline(DomainVocabulary.default()).chunk(paper)
        rule_chunks = [chunk for chunk in chunks if chunk.chunk_type == "rule_candidate"]

        self.assertTrue(rule_chunks)
        combined_text = " ".join(chunk.text for chunk in rule_chunks)
        self.assertIn("BT below 270 K", combined_text)
        matching = [chunk for chunk in rule_chunks if "BT below 270 K" in chunk.text]
        self.assertTrue(matching)
        chunk = matching[0]
        self.assertIn("cloud", chunk.metadata.target_classes)
        self.assertIn("BT below 270 K", chunk.metadata.thresholds)
        self.assertTrue(chunk.source_hash)
        self.assertTrue(chunk.parser_version)
        self.assertTrue(chunk.chunker_version)
        self.assertTrue(chunk.vocabulary_version)

    def test_long_paragraph_child_uses_recursive_fallback(self):
        long_text = " ".join([f"Cloud sentence {index} with S7 thermal evidence." for index in range(20)])
        paper = Paper(
            paper_id="long_paper",
            title="Sentinel-3 SLSTR Long Paper",
            authors=[],
            year=None,
            source_path=None,
            source_hash=compute_source_hash(long_text),
            source_type="text",
            sections=[
                PaperSection(
                    section_id="section_001",
                    title="Methods",
                    normalized_type="method",
                    level=1,
                    blocks=[
                        PaperBlock(
                            block_id="block_0001",
                            text=long_text,
                            block_type="paragraph",
                            page_start=None,
                            page_end=None,
                            order_index=0,
                        )
                    ],
                )
            ],
        )
        chunks = PaperChunkingPipeline(
            DomainVocabulary.default(),
            ChunkingConfig(max_child_chars=80, recursive_chunk_size=80, recursive_chunk_overlap=0),
        ).chunk(paper)

        paragraph_chunks = [chunk for chunk in chunks if chunk.chunk_type == "paragraph_child"]
        self.assertGreater(len(paragraph_chunks), 1)
        self.assertTrue(all(len(chunk.text) <= 120 for chunk in paragraph_chunks))


if __name__ == "__main__":
    unittest.main()
