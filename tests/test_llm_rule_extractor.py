import unittest

from satellite_paper_rag.extraction.llm_rule_extractor import LlmRuleExtractor
from satellite_paper_rag.schemas import Chunk, ChunkMetadata


class FakeChatClient:
    model_name = "fake-chat"

    def __init__(self, response):
        self.response = response
        self.calls = []

    def complete_json(self, system_prompt: str, user_prompt: str):
        self.calls.append({"system_prompt": system_prompt, "user_prompt": user_prompt})
        return self.response


def make_chunk(text: str) -> Chunk:
    return Chunk(
        chunk_id="chunk_001",
        paper_id="paper",
        chunk_type="sentence_window_child",
        text=text,
        parent_id=None,
        child_ids=[],
        source_block_ids=["block_001"],
        page_start=8,
        page_end=8,
        section_title="5. Results",
        section_type="result",
        metadata=ChunkMetadata(target_classes=["cloud"], thresholds=["threshold of 0.9"]),
        retrieval_profile="rule_extraction",
    )


class LlmRuleExtractorTest(unittest.TestCase):
    def test_extracts_rules_and_validates_evidence_quotes(self):
        chunk = make_chunk(
            "A threshold of 0.9 would typically be applied to generate a binary cloud mask."
        )
        chat_client = FakeChatClient(
            {
                "rules": [
                    {
                        "answer_type": "rule",
                        "target_class": "clear_sky",
                        "variable": "clear_sky_probability",
                        "operator": ">=",
                        "thresholds": [{"name": "clear_sky_probability", "operator": ">=", "value": 0.9}],
                        "evidence_quote": "threshold of 0.9 would typically be applied",
                    }
                ]
            }
        )

        result = LlmRuleExtractor(chat_client).extract(
            query="What threshold creates a cloud mask?",
            evidence_chunks=[chunk],
            requires_threshold=True,
        )

        self.assertEqual(result["llm_model"], "fake-chat")
        self.assertEqual(result["rules"][0]["answer_type"], "rule")
        self.assertEqual(result["rules"][0]["final_class"], "cloud")
        self.assertEqual(result["rules"][0]["evidence_class"], "clear_sky")
        self.assertEqual(result["rules"][0]["rule_direction"], "negative_evidence")
        self.assertEqual(result["rules"][0]["evidence_chunk_id"], "chunk_001")
        self.assertIn("[chunk_001]", chat_client.calls[0]["user_prompt"])
        self.assertIn('"final_class": "cloud|ice|sea"', chat_client.calls[0]["user_prompt"])
        self.assertIn("paired threshold", chat_client.calls[0]["user_prompt"])

    def test_downgrades_rules_when_llm_quote_is_not_in_evidence(self):
        chunk = make_chunk("The paper discusses cloud masking.")
        chat_client = FakeChatClient(
            {
                "rules": [
                    {
                        "answer_type": "rule",
                        "target_class": "cloud",
                        "thresholds": [{"name": "clear_sky_probability", "operator": "<", "value": 0.9}],
                        "evidence_quote": "threshold of 0.9 would typically be applied",
                    }
                ]
            }
        )

        result = LlmRuleExtractor(chat_client).extract(
            query="What threshold creates a cloud mask?",
            evidence_chunks=[chunk],
            requires_threshold=True,
        )

        self.assertEqual(result["rules"][0]["answer_type"], "insufficient_evidence")
        self.assertIn("evidence_quote_not_found", result["rules"][0]["guardrail_failures"])

    def test_removes_fabricated_placeholder_thresholds_from_qualitative_rules(self):
        chunk = make_chunk("Reflectance wavelengths distinguish between brighter ice surfaces and the darker ocean.")
        chat_client = FakeChatClient(
            {
                "rules": [
                    {
                        "answer_type": "rule",
                        "evidence_class": "ice",
                        "final_class": "ice",
                        "variable": "visible reflectance",
                        "operator": ">",
                        "thresholds": [
                            {
                                "name": "visible reflectance",
                                "operator": ">",
                                "value": 0.0,
                                "unit": "relative brightness",
                            }
                        ],
                        "evidence_quote": "brighter ice surfaces and the darker ocean",
                    }
                ]
            }
        )

        result = LlmRuleExtractor(chat_client).extract(
            query="How does reflectance distinguish ice and sea?",
            evidence_chunks=[chunk],
        )

        self.assertEqual(result["rules"][0]["thresholds"], [])
        self.assertEqual(result["rules"][0]["operator"], "qualitative_comparison")


if __name__ == "__main__":
    unittest.main()
