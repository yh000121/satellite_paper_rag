import unittest

from satellite_paper_rag.explanation.llm_evidence_auditor import LlmEvidenceAuditor
from satellite_paper_rag.observations.schema import ObservationSample
from satellite_paper_rag.schemas import Chunk, ChunkMetadata


class FakeChatClient:
    model_name = "fake-qwen"

    def __init__(self, response):
        self.response = response
        self.calls = []

    def complete_json(self, system_prompt: str, user_prompt: str):
        self.calls.append({"system_prompt": system_prompt, "user_prompt": user_prompt})
        return self.response


def make_observation() -> ObservationSample:
    return ObservationSample(
        sample_id="scene:12:8",
        source_type="csv",
        satellite=None,
        sensor="SLSTR",
        features={"S1": 18.08, "S7": 275.98, "S8": 273.21, "pc1": -1.58},
        metadata={"label": "0", "label_id": "0", "label_name": "sea"},
    )


def make_chunk() -> Chunk:
    return Chunk(
        chunk_id="chunk_001",
        paper_id="paper",
        chunk_type="sentence_window_child",
        text="Reflectance wavelengths distinguish between brighter ice or cloud surfaces and the darker ocean.",
        parent_id=None,
        child_ids=[],
        source_block_ids=["block_001"],
        page_start=5,
        page_end=5,
        section_title="4.1 Bayesian cloud detection",
        section_type="method",
        metadata=ChunkMetadata(target_classes=["cloud", "sea_ice"]),
        retrieval_profile="evidence",
    )


class LlmEvidenceAuditorTest(unittest.TestCase):
    def test_returns_validated_support_without_exposing_pca_or_allowing_reclassification(self):
        client = FakeChatClient(
            {
                "support_status": "partial",
                "conclusion": "The qualitative reflectance evidence is compatible with sea.",
                "supporting_evidence": [
                    {
                        "chunk_id": "chunk_001",
                        "quote": "brighter ice or cloud surfaces and the darker ocean",
                        "reason": "The paper describes ocean as relatively darker.",
                    }
                ],
                "conflicting_evidence": [],
                "limitations": ["No fixed numeric reflectance threshold is provided."],
                "requires_human_review": False,
                "predicted_class": "cloud",
            }
        )

        result = LlmEvidenceAuditor(client).audit(make_observation(), [make_chunk()])

        self.assertEqual(result["support_status"], "partial")
        self.assertEqual(result["predicted_class"], "sea")
        self.assertNotIn("suggested_class", result)
        self.assertEqual(result["supporting_evidence"][0]["page_start"], 5)
        prompt = client.calls[0]["user_prompt"]
        self.assertIn('"S7": 275.98', prompt)
        self.assertIn('"physical_quantity": "radiance"', prompt)
        self.assertIn('"physical_quantity": "brightness_temperature"', prompt)
        self.assertIn('"unit": "unknown"', prompt)
        self.assertNotIn('"unit": "K"', prompt)
        self.assertNotIn("pc1", prompt.lower())
        self.assertIn("Do not classify", client.calls[0]["system_prompt"])
        self.assertIn("radiance as reflectance", client.calls[0]["system_prompt"])

    def test_downgrades_unverifiable_citations_to_insufficient(self):
        client = FakeChatClient(
            {
                "support_status": "strong",
                "conclusion": "The prediction is proven.",
                "supporting_evidence": [
                    {
                        "chunk_id": "chunk_001",
                        "quote": "S7 below 270 K proves open ocean",
                        "reason": "Numeric threshold.",
                    }
                ],
                "conflicting_evidence": [],
                "limitations": [],
                "requires_human_review": False,
            }
        )

        result = LlmEvidenceAuditor(client).audit(make_observation(), [make_chunk()])

        self.assertEqual(result["support_status"], "insufficient")
        self.assertTrue(result["requires_human_review"])
        self.assertEqual(result["supporting_evidence"], [])
        self.assertIn("supporting_evidence_quote_not_found", result["guardrail_failures"])


if __name__ == "__main__":
    unittest.main()
