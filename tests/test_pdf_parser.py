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

    def test_parses_numbered_sections_captions_tables_and_quality_report(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            pdf_path = Path(temp_dir) / "structured.pdf"
            doc = fitz.open()
            page = doc.new_page()
            page.insert_textbox(
                fitz.Rect(72, 72, 520, 260),
                "\n".join(
                    [
                        "Production PDF Test",
                        "1 Introduction",
                        "Cloud detection algorithms compare reflectance and thermal channels.",
                        "4.1 Bayesian cloud detection",
                        "A threshold of 0.9 is applied to clear-sky probability for a binary cloud mask.",
                    ]
                ),
            )
            page.insert_textbox(
                fitz.Rect(72, 280, 520, 390),
                "\n".join(
                    [
                        "Fig. 8. Threshold-based cloud masking of test-scenes with warm S7 clouds.",
                        "The S8 and S7 brightness temperatures determine the threshold for identifying cloudy pixels.",
                    ]
                ),
            )
            page.insert_textbox(
                fitz.Rect(72, 420, 520, 560),
                "\n".join(
                    [
                        "Table 4",
                        "SLSTR-A segments, extraction limits and offset used for coastal cloud detection performance analysis.",
                        "Extract Date Time x-limits y-limits c",
                        "1 21/01/2020 01:28:40 400:600 0:600 0.0",
                    ]
                ),
            )
            doc.save(pdf_path)
            doc.close()

            paper = PdfPaperParser().parse(pdf_path)

            self.assertEqual(paper.title, "Production PDF Test")
            section_titles = [section.title for section in paper.sections]
            self.assertIn("1 Introduction", section_titles)
            self.assertIn("4.1 Bayesian cloud detection", section_titles)
            self.assertGreaterEqual(paper.quality_report.captions_detected, 2)
            self.assertGreaterEqual(paper.quality_report.tables_detected, 1)
            self.assertGreaterEqual(len(paper.tables), 1)

            blocks = [block for section in paper.sections for block in section.blocks]
            figure_blocks = [block for block in blocks if block.block_type == "figure_caption"]
            table_blocks = [block for block in blocks if block.block_type == "table_text"]
            self.assertTrue(any("brightness temperatures determine the threshold" in block.text for block in figure_blocks))
            self.assertTrue(any("x-limits" in block.text and "0.0" in block.text for block in table_blocks))
            self.assertTrue(all("bbox" in block.metadata for block in blocks))

    def test_reflows_fragmented_pdf_lines_into_paragraphs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            pdf_path = Path(temp_dir) / "fragmented.pdf"
            doc = fitz.open()
            page = doc.new_page()
            page.insert_textbox(
                fitz.Rect(72, 72, 520, 240),
                "\n".join(
                    [
                        "Fragmented PDF Test",
                        "5 Results",
                        "The clear-sky probability from the",
                        "operational Bayesian calculation, to which a threshold of 0.9 would",
                        "typically be applied to generate a binary cloud mask.",
                    ]
                ),
            )
            doc.save(pdf_path)
            doc.close()

            paper = PdfPaperParser().parse(pdf_path)

            blocks = [block for section in paper.sections for block in section.blocks]
            block_texts = [block.text for block in blocks]
            self.assertNotIn("The clear-sky probability from the", block_texts)
            self.assertTrue(
                any(
                    "clear-sky probability from the operational Bayesian calculation" in text
                    and "threshold of 0.9" in text
                    and "binary cloud mask" in text
                    for text in block_texts
                )
            )

    def test_normalizes_pdf_font_glyph_artifacts(self):
        parser = PdfPaperParser()

        cleaned = parser._clean_line("All values of 老 > 2 at 0.6 ¦Ěm and range 0.5每0.95")

        self.assertEqual(cleaned, "All values of ρ > 2 at 0.6 μm and range 0.5-0.95")

    def test_normalizes_rho_glyph_variant_from_pdf(self):
        parser = PdfPaperParser()

        cleaned = parser._clean_line("All values of ¦Ñ > 2 represent cloudy conditions.")

        self.assertEqual(cleaned, "All values of ρ > 2 represent cloudy conditions.")


if __name__ == "__main__":
    unittest.main()
