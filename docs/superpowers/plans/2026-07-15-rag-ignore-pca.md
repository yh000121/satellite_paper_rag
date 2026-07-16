# RAG PCA Exclusion Implementation Plan

**Goal:** Accept the full satellite CSV schema while preventing `pc1-pc3` from entering RAG retrieval or LLM evidence context.

**Approach:** Keep CSV ingestion backward compatible. Normalize `label` to the existing class metadata at the evidence boundary, expose only `S1-S9` as RAG band context, and ignore PCA values even if they appear in `top_features`.

## Tasks

- Add failing tests for label-based evidence retrieval and PCA exclusion.
- Update `prediction_evidence.py` with label fallback and an explicit `S1-S9` projection.
- Run targeted tests, then the complete test suite.

No dependency, API, index, or Git changes are required.
