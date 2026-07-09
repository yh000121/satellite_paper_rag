import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from satellite_paper_rag.cli import llm_extract_rules


FIXTURES = Path(__file__).parent / "fixtures"


class FakeChatClient:
    model_name = "fake-chat"

    def complete_json(self, system_prompt: str, user_prompt: str):
        return {
            "rules": [
                {
                    "answer_type": "rule",
                    "target_class": "cloud",
                    "final_class": "cloud",
                    "variable": "thermal brightness temperature",
                    "operator": "<",
                    "thresholds": [{"name": "thermal brightness temperature", "operator": "<", "value": 270}],
                    "evidence_quote": "BT below 270 K",
                }
            ]
        }


class CliLlmSaveRulesTest(unittest.TestCase):
    def test_llm_extract_rules_can_save_executable_rule_library(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "rules.json"
            args = argparse.Namespace(
                file=str(FIXTURES / "sample_sentinel3_paper.md"),
                paper=None,
                query="What threshold classifies cloud?",
                candidate_k=20,
                index_dir=str(Path(temp_dir) / "index"),
                rebuild_index=True,
                requires_threshold=True,
                embedding_provider="deterministic",
                embedding_model=None,
                embedding_base_url=None,
                llm_provider="dashscope",
                llm_model=None,
                llm_base_url=None,
                chunk_type=["rule_candidate", "sentence_window_child", "paragraph_child", "figure_table"],
                save_rules=str(save_path),
                verbose=False,
            )

            with patch("satellite_paper_rag.cli.build_chat_client", return_value=FakeChatClient()):
                exit_code = llm_extract_rules(args)

            self.assertEqual(exit_code, 0)
            saved = json.loads(save_path.read_text(encoding="utf-8"))

        self.assertEqual(saved["paper_id"], "sample_sentinel3_paper")
        self.assertEqual(saved["rule_count"], 1)
        self.assertEqual(saved["metadata"]["llm_model"], "fake-chat")
        self.assertEqual(saved["metadata"]["query"], "What threshold classifies cloud?")
        self.assertEqual(saved["rules"][0]["rule_id"], "sample_sentinel3_paper_rule_0001")


if __name__ == "__main__":
    unittest.main()
