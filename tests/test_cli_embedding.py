import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


FIXTURES = Path(__file__).parent / "fixtures"


class CliEmbeddingTest(unittest.TestCase):
    def test_index_paper_and_semantic_query_cli(self):
        env = dict(os.environ)
        env["PYTHONPATH"] = "src"
        with tempfile.TemporaryDirectory() as tmp_dir:
            index_result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "satellite_paper_rag.cli",
                    "index-paper",
                    "--file",
                    str(FIXTURES / "sample_sentinel3_paper.md"),
                    "--index-dir",
                    tmp_dir,
                ],
                check=True,
                capture_output=True,
                text=True,
                env=env,
            )
            index_payload = json.loads(index_result.stdout)
            self.assertGreater(index_payload["chunk_count"], 0)
            self.assertTrue(Path(index_payload["index_path"]).exists())

            query_result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "satellite_paper_rag.cli",
                    "semantic-query",
                    "--file",
                    str(FIXTURES / "sample_sentinel3_paper.md"),
                    "--query",
                    "How are cloudy pixels identified?",
                    "--index-dir",
                    tmp_dir,
                    "--top-k",
                    "3",
                ],
                check=True,
                capture_output=True,
                text=True,
                env=env,
            )

        query_payload = json.loads(query_result.stdout)
        self.assertIn(query_payload["results"][0]["chunk_type"], {"rule_candidate", "paragraph_child", "sentence_window_child"})
        self.assertIn("BT below 270 K", query_payload["results"][0]["text"])

    def test_extract_rules_can_use_semantic_candidates(self):
        env = dict(os.environ)
        env["PYTHONPATH"] = "src"
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "satellite_paper_rag.cli",
                    "extract-rules",
                    "--file",
                    str(FIXTURES / "sample_sentinel3_paper.md"),
                    "--semantic-query",
                    "How are cloudy pixels identified?",
                    "--index-dir",
                    tmp_dir,
                    "--candidate-k",
                    "5",
                    "--top-k",
                    "3",
                ],
                check=True,
                capture_output=True,
                text=True,
                env=env,
            )

        payload = json.loads(result.stdout)
        self.assertEqual(payload["semantic_query"], "How are cloudy pixels identified?")
        self.assertTrue(payload["semantic_candidates"])
        self.assertEqual(payload["rules"][0]["rule_type"], "threshold_rule")
        self.assertIn("BT below 270 K", payload["rules"][0]["condition_text"])


if __name__ == "__main__":
    unittest.main()
