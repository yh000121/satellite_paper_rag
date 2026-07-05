import json
import os
import subprocess
import sys
import unittest
from pathlib import Path


FIXTURES = Path(__file__).parent / "fixtures"


class CliTest(unittest.TestCase):
    def test_query_cli_returns_json_evidence(self):
        env = dict(os.environ)
        env["PYTHONPATH"] = "src"
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "satellite_paper_rag.cli",
                "query",
                "--file",
                str(FIXTURES / "sample_sentinel3_paper.md"),
                "--query",
                "What threshold helps identify cloud?",
                "--requires-threshold",
            ],
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )

        payload = json.loads(result.stdout)
        self.assertIn("results", payload)
        self.assertTrue(payload["results"])
        self.assertEqual(payload["results"][0]["answer_type"], "evidence")
        self.assertIn("BT below 270 K", payload["results"][0]["text"])


if __name__ == "__main__":
    unittest.main()
