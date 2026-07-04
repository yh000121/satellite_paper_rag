import unittest

from satellite_paper_rag.schemas import (
    CHUNKER_VERSION,
    PARSER_VERSION,
    VOCABULARY_VERSION,
    Chunk,
    ChunkMetadata,
    Paper,
    PaperBlock,
    PaperSection,
    compute_source_hash,
)


class SchemaTest(unittest.TestCase):
    def test_source_hash_is_stable(self):
        self.assertEqual(compute_source_hash("same text"), compute_source_hash("same text"))
        self.assertNotEqual(compute_source_hash("same text"), compute_source_hash("other text"))

    def test_paper_and_chunk_keep_versions_and_provenance(self):
        block = PaperBlock(
            block_id="block_001",
            text="Cloud pixels show lower brightness temperature.",
            block_type="paragraph",
            page_start=1,
            page_end=1,
            order_index=0,
            metadata={},
        )
        section = PaperSection(
            section_id="section_001",
            title="Results",
            normalized_type="result",
            level=2,
            blocks=[block],
        )
        paper = Paper(
            paper_id="paper_001",
            title="Test Paper",
            authors=[],
            year=None,
            source_path=None,
            source_hash=compute_source_hash(block.text),
            source_type="markdown",
            sections=[section],
            metadata={},
            parser_version=PARSER_VERSION,
            vocabulary_version=VOCABULARY_VERSION,
        )
        metadata = ChunkMetadata(target_classes=["cloud"], thresholds=["BT below 270 K"])
        chunk = Chunk(
            chunk_id="chunk_001",
            paper_id=paper.paper_id,
            chunk_type="rule_candidate",
            text=block.text,
            parent_id=None,
            child_ids=[],
            source_block_ids=[block.block_id],
            page_start=1,
            page_end=1,
            section_title="Results",
            section_type="result",
            metadata=metadata,
            retrieval_profile="rule_extraction",
            parser_version=PARSER_VERSION,
            chunker_version=CHUNKER_VERSION,
            vocabulary_version=VOCABULARY_VERSION,
            source_hash=paper.source_hash,
        )

        self.assertEqual(chunk.source_hash, paper.source_hash)
        self.assertEqual(chunk.parser_version, PARSER_VERSION)
        self.assertEqual(chunk.chunker_version, CHUNKER_VERSION)
        self.assertEqual(chunk.vocabulary_version, VOCABULARY_VERSION)
        self.assertEqual(chunk.metadata.target_classes, ["cloud"])


if __name__ == "__main__":
    unittest.main()
