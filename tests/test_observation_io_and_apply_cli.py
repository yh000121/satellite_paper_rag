import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from satellite_paper_rag.observations.io import load_observations


class ObservationIoAndApplyCliTest(unittest.TestCase):
    def test_loads_observations_from_json_list(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "observations.json"
            path.write_text(
                json.dumps(
                    [
                        {
                            "point_id": "p001",
                            "rho": 2.4,
                            "LSD": 0.35,
                            "clear_sky_probability": 0.72,
                        }
                    ]
                ),
                encoding="utf-8",
            )

            observations = load_observations(path)

        self.assertEqual(observations[0].sample_id, "p001")
        self.assertEqual(observations[0].features["rho"], 2.4)

    def test_loads_observations_from_csv(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "observations.csv"
            path.write_text("point_id,rho,LSD\np001,2.4,0.35\n", encoding="utf-8")

            observations = load_observations(path)

        self.assertEqual(observations[0].sample_id, "p001")
        self.assertEqual(observations[0].features["rho"], 2.4)
        self.assertEqual(observations[0].features["LSD"], 0.35)

    def test_apply_rules_cli_returns_batch_predictions(self):
        env = dict(os.environ)
        env["PYTHONPATH"] = "src"
        with tempfile.TemporaryDirectory() as temp_dir:
            rules_path = Path(temp_dir) / "rules.json"
            observations_path = Path(temp_dir) / "observations.json"
            rules_path.write_text(
                json.dumps(
                    {
                        "paper_id": "paper",
                        "rules": [
                            {
                                "rule_id": "rule_rho",
                                "final_class": "cloud",
                                "rule_direction": "positive_evidence",
                                "variable": "rho",
                                "operator": ">",
                                "thresholds": [{"name": "rho", "operator": ">", "value": 2.0}],
                                "evidence_quote": "All values of rho > 2 represent cloudy conditions.",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            observations_path.write_text(json.dumps([{"point_id": "p001", "rho": 2.4}]), encoding="utf-8")

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "satellite_paper_rag.cli",
                    "apply-rules",
                    "--rules-file",
                    str(rules_path),
                    "--observations-file",
                    str(observations_path),
                ],
                check=True,
                capture_output=True,
                text=True,
                env=env,
            )

        payload = json.loads(result.stdout)
        self.assertEqual(payload["total"], 1)
        self.assertEqual(payload["results"][0]["predicted_class"], "cloud")
        self.assertEqual(payload["results"][0]["matched_rules"][0]["rule_id"], "rule_rho")


if __name__ == "__main__":
    unittest.main()
