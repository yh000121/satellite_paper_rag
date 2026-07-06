import unittest
from pathlib import Path

from satellite_paper_rag.chunking.pipeline import PaperChunkingPipeline
from satellite_paper_rag.domain.vocabulary import DomainVocabulary
from satellite_paper_rag.extraction.rule_extractor import RuleCandidateExtractor
from satellite_paper_rag.parsing.markdown_parser import MarkdownPaperParser
from satellite_paper_rag.schemas import Chunk, ChunkMetadata


class RuleCandidateExtractorTest(unittest.TestCase):
    def test_extracts_rule_candidate_from_threshold_condition(self):
        paper = MarkdownPaperParser().parse(Path("tests/fixtures/sample_sentinel3_paper.md"))
        chunks = PaperChunkingPipeline(DomainVocabulary.default()).chunk(paper)

        rules = RuleCandidateExtractor().extract(chunks)

        cloud_rules = [rule for rule in rules if "cloud" in rule.target_classes]
        self.assertTrue(cloud_rules)
        self.assertIn("BT below 270 K", cloud_rules[0].condition_text)
        self.assertEqual(cloud_rules[0].rule_type, "threshold_rule")
        self.assertNotIn("open_water", cloud_rules[0].target_classes)
        self.assertIn("BT < 270.0 K", cloud_rules[0].normalized_conditions)
        self.assertGreaterEqual(cloud_rules[0].score, 0.7)

    def test_extracts_probability_rho_and_dynamic_threshold_rules(self):
        chunk = Chunk(
            chunk_id="paper_rule_candidate_block_0001",
            paper_id="paper",
            chunk_type="rule_candidate",
            text=(
                "The clear-sky probability from the operational Bayesian calculation, "
                "to which a threshold of 0.9 would typically be applied to generate a binary cloud mask. "
                "All values of rho > 2 represent cloudy conditions. "
                "A scene-dependent threshold is found from the S7 and S8 brightness temperatures; "
                "to this offset we add an additional 2 K and assume a slope of one."
            ),
            parent_id=None,
            child_ids=[],
            source_block_ids=["block_0001"],
            page_start=8,
            page_end=9,
            section_title="5.2 Coastal zone performance metrics",
            section_type="result",
            metadata=ChunkMetadata(target_classes=["cloud"], bands_or_layers=["S7", "S8"]),
            retrieval_profile="rule_extraction",
        )

        rules = RuleCandidateExtractor().extract([chunk])

        normalized = [condition for rule in rules for condition in rule.normalized_conditions]
        rule_types = {rule.rule_type for rule in rules}
        self.assertIn("CLEAR_SKY_PROBABILITY >= 0.9", normalized)
        self.assertIn("RHO > 2.0", normalized)
        self.assertIn("SCENE_THRESHOLD_OFFSET + 2.0 K", normalized)
        self.assertIn("dynamic_threshold_rule", rule_types)
        self.assertTrue(any(rule.rule_scope == "validation" for rule in rules))


if __name__ == "__main__":
    unittest.main()
