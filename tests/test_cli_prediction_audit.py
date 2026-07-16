import argparse
import io
import json
import re
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from satellite_paper_rag.cli import audit_prediction_evidence, build_parser


FIXTURES = Path(__file__).parent / "fixtures"


class PromptAwareFakeChatClient:
    model_name = "fake-qwen"

    def complete_json(self, system_prompt: str, user_prompt: str):
        evidence_prompt = user_prompt.split("Evidence chunks:\n", 1)[1]
        match = re.search(r"\[([^\]]+)\].*\n([^\n]+)", evidence_prompt)
        if match is None:
            raise AssertionError("Expected at least one formatted evidence chunk in the LLM prompt.")
        return {
            "support_status": "partial",
            "conclusion": "The retrieved paper evidence is relevant but incomplete.",
            "supporting_evidence": [
                {
                    "chunk_id": match.group(1),
                    "quote": match.group(2),
                    "reason": "The chunk discusses the requested class evidence.",
                }
            ],
            "conflicting_evidence": [],
            "limitations": ["Fixture evidence is qualitative."],
            "requires_human_review": False,
        }


class CliPredictionAuditTest(unittest.TestCase):
    def test_parser_exposes_prediction_audit_command(self):
        parser = build_parser()

        args = parser.parse_args(
            [
                "audit-prediction-evidence",
                "--file",
                str(FIXTURES / "sample_sentinel3_paper.md"),
                "--predictions-file",
                "points.csv",
            ]
        )

        self.assertIs(args.func, audit_prediction_evidence)

    def test_audits_label_using_retrieved_evidence(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "points.csv"
            csv_path.write_text(
                "S1,S2,S3,S4,S5,S6,S7,S8,S9,pc1,pc2,pc3,label,image_id,row,col\n"
                "0.1,0.2,0.3,0.4,0.5,0.6,275.9,273.2,272.6,-1.5,3.2,0.3,0,scene,12,8\n",
                encoding="utf-8",
            )
            args = argparse.Namespace(
                file=str(FIXTURES / "sample_sentinel3_paper.md"),
                paper=None,
                predictions_file=str(csv_path),
                row_index=0,
                top_k=2,
                index_dir=str(Path(temp_dir) / "index"),
                rebuild_index=True,
                embedding_provider="deterministic",
                embedding_model=None,
                embedding_base_url=None,
                llm_provider="dashscope",
                llm_model=None,
                llm_base_url=None,
                chunk_type=["rule_candidate", "sentence_window_child", "paragraph_child", "figure_table"],
                verbose=False,
            )
            output = io.StringIO()
            with patch(
                "satellite_paper_rag.cli.build_chat_client",
                return_value=PromptAwareFakeChatClient(),
            ), redirect_stdout(output):
                exit_code = audit_prediction_evidence(args)

        payload = json.loads(output.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["audit_boundary"], "llm_evidence_audit_no_classification")
        self.assertEqual(payload["prediction"]["predicted_class"], "sea")
        self.assertEqual(payload["audit"]["support_status"], "partial")
        self.assertEqual(payload["audit"]["llm_model"], "fake-qwen")
        self.assertTrue(payload["evidence_results"])


if __name__ == "__main__":
    unittest.main()
