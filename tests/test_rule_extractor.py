import unittest
from pathlib import Path

from satellite_paper_rag.chunking.pipeline import PaperChunkingPipeline
from satellite_paper_rag.domain.vocabulary import DomainVocabulary
from satellite_paper_rag.extraction.rule_extractor import RuleCandidateExtractor
from satellite_paper_rag.parsing.markdown_parser import MarkdownPaperParser


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


if __name__ == "__main__":
    unittest.main()
