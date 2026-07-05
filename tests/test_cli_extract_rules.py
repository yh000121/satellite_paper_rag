import json
import os
import subprocess
import sys
import unittest
from pathlib import Path


FIXTURES = Path(__file__).parent / "fixtures"


class CliExtractRulesTest(unittest.TestCase):
    def test_extract_rules_cli_returns_rule_aware_candidates(self):
        paper_dir = Path("data") / "papers"
        paper_dir.mkdir(parents=True, exist_ok=True)
        paper_path = paper_dir / "cli_rule_extraction_sample.md"
        paper_path.write_text((FIXTURES / "sample_sentinel3_paper.md").read_text(encoding="utf-8"), encoding="utf-8")
        try:
            env = dict(os.environ)
            env["PYTHONPATH"] = "src"
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "satellite_paper_rag.cli",
                    "extract-rules",
                    "--paper",
                    paper_path.name,
                    "--top-k",
                    "3",
                ],
                check=True,
                capture_output=True,
                text=True,
                env=env,
            )

            payload = json.loads(result.stdout)
            self.assertEqual(payload["paper_id"], "cli_rule_extraction_sample")
            self.assertEqual(payload["rules"][0]["rule_type"], "threshold_rule")
            self.assertIn("BT below 270 K", payload["rules"][0]["condition_text"])
            self.assertIn("cloud", payload["rules"][0]["target_classes"])
        finally:
            paper_path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
