# LLM Evidence Auditor Implementation Plan

**Goal:** Let Qwen audit whether retrieved paper evidence supports a CSV row's existing label without performing classification.

**Boundary:** Send only `S1-S9`, the normalized label, and retrieved chunks. Accept only a fixed audit schema; validate every citation against the supplied chunks and require human review when evidence is weak, conflicting, insufficient, or invalid.

## Tasks

- Add failing auditor and CLI tests.
- Implement `LlmEvidenceAuditor` with prompt construction and citation guardrails.
- Add `audit-prediction-evidence`, reusing the current local-index retrieval flow.
- Run targeted and full tests, then audit the first row with Qwen.

No index rebuild, dependency change, or automatic Git commit is required.
