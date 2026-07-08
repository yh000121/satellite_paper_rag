import json
import os
import io
import subprocess
import sys
import unittest
from pathlib import Path

from satellite_paper_rag.cli import emit_json


FIXTURES = Path(__file__).parent / "fixtures"


class CliTest(unittest.TestCase):
    def test_emit_json_writes_utf8_bytes_for_math_symbols(self):
        stream = io.TextIOWrapper(io.BytesIO(), encoding="gbk")

        emit_json({"condition": "1 − clear_sky_probability"}, stream=stream)

        stream.flush()
        self.assertIn("1 − clear_sky_probability", stream.buffer.getvalue().decode("utf-8"))

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

    def test_query_cli_resolves_paper_from_default_papers_directory(self):
        paper_dir = Path("data") / "papers"
        paper_dir.mkdir(parents=True, exist_ok=True)
        paper_path = paper_dir / "cli_default_lookup_sample.md"
        paper_path.write_text((FIXTURES / "sample_sentinel3_paper.md").read_text(encoding="utf-8"), encoding="utf-8")
        try:
            env = dict(os.environ)
            env["PYTHONPATH"] = "src"
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "satellite_paper_rag.cli",
                    "query",
                    "--paper",
                    paper_path.name,
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
            self.assertEqual(payload["paper_id"], "cli_default_lookup_sample")
            self.assertEqual(payload["source_type"], "markdown")
            self.assertIn("BT below 270 K", payload["results"][0]["text"])
        finally:
            paper_path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
