import unittest
from pathlib import Path

from satellite_paper_rag.parsing.markdown_parser import MarkdownPaperParser
from satellite_paper_rag.parsing.pdf_parser import PdfPaperParser
from satellite_paper_rag.parsing.text_parser import TextPaperParser


FIXTURES = Path(__file__).parent / "fixtures"


class ParserTest(unittest.TestCase):
    def test_markdown_parser_detects_sections_and_captions(self):
        paper = MarkdownPaperParser().parse(FIXTURES / "sample_sentinel3_paper.md")

        self.assertEqual(paper.source_type, "markdown")
        self.assertEqual(paper.title, "Sentinel-3 SLSTR Cloud, Sea Ice, and Open Water Discrimination")
        section_types = [section.normalized_type for section in paper.sections]
        self.assertIn("abstract", section_types)
        self.assertIn("method", section_types)
        self.assertIn("result", section_types)
        block_types = [block.block_type for section in paper.sections for block in section.blocks]
        self.assertIn("table_caption", block_types)
        self.assertIn("figure_caption", block_types)

    def test_text_parser_detects_plain_text_sections(self):
        paper = TextPaperParser().parse(FIXTURES / "sample_landsat_paper.txt")

        self.assertEqual(paper.source_type, "text")
        self.assertEqual(paper.title, "Landsat-8 Thermal Cloud Screening")
        self.assertIn("method", [section.normalized_type for section in paper.sections])
        self.assertIn("result", [section.normalized_type for section in paper.sections])

    def test_pdf_parser_boundary_is_explicit(self):
        with self.assertRaises(NotImplementedError):
            PdfPaperParser().parse(FIXTURES / "sample_landsat_paper.txt")


if __name__ == "__main__":
    unittest.main()
