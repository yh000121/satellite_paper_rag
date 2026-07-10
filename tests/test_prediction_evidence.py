import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from satellite_paper_rag.explanation.prediction_evidence import build_prediction_evidence_queries
from satellite_paper_rag.observations.io import load_observations


FIXTURES = Path(__file__).parent / "fixtures"


class PredictionEvidenceTest(unittest.TestCase):
    def test_loads_node_prediction_metadata_without_leaking_into_features(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "node_predictions.csv"
            path.write_text(
                "S1,S2,S3,S4,S5,S6,S7,S8,S9,pc1,pc2,pc3,predicted_label,confidence,top_features,image_id,row,col\n"
                "0.1,0.2,0.3,0.4,0.5,0.6,270.1,271.2,272.3,1.1,1.2,1.3,2,0.91,\"S7,S8,S1\",scene_001,12,8\n",
                encoding="utf-8",
            )

            observation = load_observations(path)[0]

        self.assertEqual(observation.sample_id, "scene_001:12:8")
        self.assertEqual(observation.metadata["predicted_label_id"], "2")
        self.assertEqual(observation.metadata["predicted_class"], "cloud")
        self.assertEqual(observation.metadata["confidence"], "0.91")
        self.assertNotIn("predicted_label", observation.features)
        self.assertNotIn("confidence", observation.features)
        self.assertNotIn("top_features", observation.features)

    def test_builds_queries_from_prediction_class_and_top_features(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "node_predictions.csv"
            path.write_text(
                "S1,S2,S3,S4,S5,S6,S7,S8,S9,predicted_label,top_features,image_id,row,col\n"
                "0.1,0.2,0.3,0.4,0.5,0.6,270.1,271.2,272.3,2,\"S7,S8,S1\",scene_001,12,8\n",
                encoding="utf-8",
            )
            observation = load_observations(path)[0]

        queries = build_prediction_evidence_queries(observation)

        self.assertEqual(queries[0]["prediction_class"], "cloud")
        query_texts = [query["query"] for query in queries]
        self.assertIn("cloud detection SLSTR brightness temperature reflectance", query_texts)
        self.assertIn("S7 S8 brightness temperature cloud discrimination", query_texts)
        self.assertIn("S1 visible reflectance cloud ice ocean", query_texts)

    def test_cli_retrieves_paper_evidence_for_node_prediction(self):
        env = dict(os.environ)
        env["PYTHONPATH"] = "src"
        with tempfile.TemporaryDirectory() as temp_dir:
            predictions_path = Path(temp_dir) / "node_predictions.csv"
            predictions_path.write_text(
                "S1,S2,S3,S4,S5,S6,S7,S8,S9,predicted_label,confidence,top_features,image_id,row,col\n"
                "0.1,0.2,0.3,0.4,0.5,0.6,270.1,271.2,272.3,2,0.91,\"S7,S8,S1\",scene_001,12,8\n",
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "satellite_paper_rag.cli",
                    "explain-prediction-evidence",
                    "--file",
                    str(FIXTURES / "sample_sentinel3_paper.md"),
                    "--predictions-file",
                    str(predictions_path),
                    "--row-index",
                    "0",
                    "--index-dir",
                    str(Path(temp_dir) / "index"),
                    "--embedding-provider",
                    "deterministic",
                    "--top-k",
                    "2",
                ],
                check=True,
                capture_output=True,
                text=True,
                env=env,
            )

        payload = json.loads(result.stdout)
        self.assertEqual(payload["prediction"]["predicted_class"], "cloud")
        self.assertEqual(payload["prediction"]["sample_id"], "scene_001:12:8")
        self.assertEqual(payload["audit_boundary"], "retrieval_only_no_classification")
        self.assertTrue(payload["evidence_queries"])
        self.assertTrue(payload["evidence_results"][0]["results"])


if __name__ == "__main__":
    unittest.main()
