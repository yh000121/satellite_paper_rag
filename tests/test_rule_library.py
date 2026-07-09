import tempfile
import unittest
from pathlib import Path

from satellite_paper_rag.rules.library import build_rule_library, load_rule_library, write_rule_library


class RuleLibraryTest(unittest.TestCase):
    def test_builds_deduplicated_rule_library_with_provenance(self):
        rules = [
            {
                "final_class": "cloud",
                "operator": ">",
                "thresholds": [{"name": "rho", "operator": ">", "value": 2.0}],
                "evidence_quote": "All values of rho > 2 represent cloudy conditions.",
                "evidence_chunk_id": "chunk_001",
            },
            {
                "final_class": "cloud",
                "operator": ">",
                "thresholds": [{"name": "rho", "operator": ">", "value": 2.0}],
                "evidence_quote": "All values of rho > 2 represent cloudy conditions.",
                "evidence_chunk_id": "chunk_001",
            },
        ]

        library = build_rule_library(
            paper_id="paper",
            title="Paper Title",
            rules=rules,
            metadata={"llm_model": "qwen-plus", "embedding_model": "text-embedding-v4"},
        )

        self.assertEqual(library["paper_id"], "paper")
        self.assertEqual(library["rule_count"], 1)
        self.assertEqual(library["rules"][0]["rule_id"], "paper_rule_0001")
        self.assertEqual(library["metadata"]["llm_model"], "qwen-plus")

    def test_writes_and_loads_rule_library(self):
        library = build_rule_library(
            paper_id="paper",
            title="Paper Title",
            rules=[{"final_class": "cloud", "evidence_quote": "quote"}],
            metadata={},
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "rules.json"

            write_rule_library(library, path)
            loaded = load_rule_library(path)

        self.assertEqual(loaded["paper_id"], "paper")
        self.assertEqual(loaded["rules"][0]["evidence_quote"], "quote")

    def test_excludes_explicit_insufficient_evidence_rules(self):
        library = build_rule_library(
            paper_id="paper",
            title="Paper Title",
            rules=[
                {"answer_type": "insufficient_evidence", "final_class": "cloud"},
                {"answer_type": "rule", "final_class": "cloud", "evidence_quote": "quote"},
            ],
            metadata={},
        )

        self.assertEqual(library["rule_count"], 1)
        self.assertEqual(library["rules"][0]["answer_type"], "rule")


if __name__ == "__main__":
    unittest.main()
