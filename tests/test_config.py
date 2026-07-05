import unittest

from satellite_paper_rag.config import ChunkingConfig, RetrievalConfig


class ConfigTest(unittest.TestCase):
    def test_chunking_config_defaults_are_conservative(self):
        config = ChunkingConfig()

        self.assertEqual(config.sentence_window_radius, 1)
        self.assertEqual(config.min_rule_signal_count, 2)
        self.assertGreater(config.max_child_chars, 0)
        self.assertGreater(config.recursive_chunk_overlap, 0)

    def test_retrieval_config_defaults_preserve_existing_boosts(self):
        config = RetrievalConfig()

        self.assertEqual(config.chunk_type_boosts["rule_candidate"], 3.0)
        self.assertEqual(config.chunk_type_boosts["sentence_window_child"], 2.0)
        self.assertEqual(config.threshold_boost, 5.0)


if __name__ == "__main__":
    unittest.main()
