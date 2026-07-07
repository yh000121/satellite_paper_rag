import unittest

from satellite_paper_rag.extraction.evidence_guardrails import EvidenceGuardrail
from satellite_paper_rag.schemas import Chunk, ChunkMetadata


def make_chunk(text: str) -> Chunk:
    return Chunk(
        chunk_id="chunk_001",
        paper_id="paper",
        chunk_type="rule_candidate",
        text=text,
        parent_id=None,
        child_ids=[],
        source_block_ids=["block_001"],
        page_start=8,
        page_end=8,
        section_title="Results",
        section_type="result",
        metadata=ChunkMetadata(target_classes=["cloud"], thresholds=["threshold of 0.9"]),
        retrieval_profile="rule_extraction",
    )


class EvidenceGuardrailTest(unittest.TestCase):
    def test_accepts_rule_when_evidence_quote_is_present_in_chunks(self):
        chunk = make_chunk(
            "A threshold of 0.9 would typically be applied to generate a binary cloud mask."
        )
        payload = {
            "answer_type": "rule",
            "target_class": "cloud",
            "thresholds": [{"name": "clear_sky_probability", "operator": ">=", "value": 0.9}],
            "evidence_quote": "threshold of 0.9 would typically be applied",
        }

        result = EvidenceGuardrail().validate(payload, [chunk], requires_threshold=True)

        self.assertEqual(result["answer_type"], "rule")
        self.assertEqual(result["evidence_chunk_id"], "chunk_001")
        self.assertEqual(result["page_start"], 8)

    def test_downgrades_rule_when_evidence_quote_is_missing(self):
        chunk = make_chunk("The paper discusses cloud masking.")
        payload = {
            "answer_type": "rule",
            "target_class": "cloud",
            "thresholds": [{"name": "clear_sky_probability", "operator": ">=", "value": 0.9}],
            "evidence_quote": "threshold of 0.9 would typically be applied",
        }

        result = EvidenceGuardrail().validate(payload, [chunk], requires_threshold=True)

        self.assertEqual(result["answer_type"], "insufficient_evidence")
        self.assertIn("evidence_quote_not_found", result["guardrail_failures"])

    def test_downgrades_rule_when_threshold_required_but_missing(self):
        chunk = make_chunk("Cloud masking uses visual inspection in this example.")
        payload = {
            "answer_type": "rule",
            "target_class": "cloud",
            "thresholds": [],
            "evidence_quote": "Cloud masking uses visual inspection",
        }

        result = EvidenceGuardrail().validate(payload, [chunk], requires_threshold=True)

        self.assertEqual(result["answer_type"], "insufficient_evidence")
        self.assertIn("threshold_required_but_missing", result["guardrail_failures"])


if __name__ == "__main__":
    unittest.main()
