import unittest

from satellite_paper_rag.chunking.recursive_splitter import RecursiveSplitterAdapter
from satellite_paper_rag.config import ChunkingConfig


class RecursiveSplitterAdapterTest(unittest.TestCase):
    def test_short_text_stays_whole(self):
        splitter = RecursiveSplitterAdapter(ChunkingConfig(recursive_chunk_size=100, recursive_chunk_overlap=10))

        self.assertEqual(splitter.split("short text"), ["short text"])

    def test_long_text_is_split_with_langchain_fallback(self):
        splitter = RecursiveSplitterAdapter(ChunkingConfig(recursive_chunk_size=40, recursive_chunk_overlap=5))
        text = "Sentence one. Sentence two. Sentence three. Sentence four. Sentence five."

        parts = splitter.split(text)

        self.assertGreater(len(parts), 1)
        self.assertTrue(all(part.strip() for part in parts))
        joined = " ".join(parts)
        self.assertIn("Sentence one", joined)
        self.assertIn("Sentence three", joined)
        self.assertIn("Sentence five", joined)


if __name__ == "__main__":
    unittest.main()
