import tempfile
import unittest
from pathlib import Path

import fitz

from satellite_paper_rag.parsing.pdf_parser import PdfPaperParser


class PdfPaperParserTest(unittest.TestCase):
    def test_parses_pdf_text_with_page_provenance(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            pdf_path = Path(temp_dir) / "sample.pdf"
            doc = fitz.open()
            page = doc.new_page()
            page.insert_text(
                (72, 72),
                "Sentinel-3 SLSTR Cloud Paper\n\nAbstract\n\nCloud pixels with BT below 270 K require review.",
            )
            doc.save(pdf_path)
            doc.close()

            paper = PdfPaperParser().parse(pdf_path)

            self.assertEqual(paper.source_type, "pdf")
            self.assertEqual(paper.title, "Sentinel-3 SLSTR Cloud Paper")
            self.assertTrue(paper.source_hash)
            self.assertIn("abstract", [section.normalized_type for section in paper.sections])
            blocks = [block for section in paper.sections for block in section.blocks]
            self.assertTrue(any("BT below 270 K" in block.text for block in blocks))
            self.assertTrue(all(block.page_start == 1 for block in blocks))
            self.assertTrue(all(block.page_end == 1 for block in blocks))


if __name__ == "__main__":
    unittest.main()
