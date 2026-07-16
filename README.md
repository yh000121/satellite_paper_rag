# Satellite Paper RAG

Evidence-oriented RAG for user-provided satellite remote-sensing papers.

The project parses English PDF, text, and Markdown papers; builds multi-granularity chunks and a local embedding index; retrieves paper evidence; and asks a Qwen LLM to audit an existing `sea`, `ice`, or `cloud` label.

It does **not** search Scholar, download papers, train a classifier, or use the LLM as the primary pixel classifier.

## System Boundary

```text
Classifier or labeled CSV                    User-provided papers
S1-S9 + existing label                       PDF / text / Markdown
          |                                            |
          v                                            v
Observation schema                              Parse and chunk
          |                                            |
          +----------> Evidence queries <------ Local vector index
                                   |
                                   v
                         Qwen evidence auditor
                                   |
                                   v
                  support / conflict / insufficient
                         with cited paper chunks
```

The classifier answers **what the pixel is**. This RAG answers **whether the supplied papers support that result, what evidence applies, and when human review is required**.

## Implemented Features

- PyMuPDF PDF text extraction with page provenance
- pdfplumber table extraction with a PyMuPDF fallback
- text and Markdown paper parsers
- multi-granularity chunking with parent-child links
- LangChain recursive splitting for oversized paragraphs
- sentence-window and domain-triggered rule-candidate chunks
- remote-sensing metadata enrichment
- deterministic local embeddings for tests
- DashScope `text-embedding-v4` embeddings for semantic retrieval
- versioned local JSONL vector indexes
- rule extraction, rule libraries, and extraction evaluation
- DashScope/Qwen JSON LLM client (`qwen-plus` by default)
- CSV observation loading and `0/1/2` class normalization
- label-driven evidence retrieval
- LLM evidence auditing with exact-quote guardrails
- fixed SLSTR feature semantics for RAG prompts

## Architecture

| Package | Responsibility |
|---|---|
| `parsing` | Convert source files into papers, sections, blocks, tables, and provenance |
| `chunking` | Produce multiple retrieval granularities and enrich metadata |
| `embeddings` | Embed chunks and queries, persist and search the local vector index |
| `retrieval` | Define retrieval contracts and lexical/metadata scoring behavior |
| `extraction` | Extract paper rules with regex and LLM paths, then validate evidence |
| `observations` | Load CSV/JSON observations and normalize feature metadata |
| `explanation` | Build label-specific queries and audit retrieved evidence |
| `rules` | Save, load, deduplicate, and apply executable rule libraries |
| `evaluation` | Evaluate rule extraction against fixed cases |
| `cli.py` | Compose the modules into runnable workflows |

## Chunking Strategy

The system does not rely on one recursive splitter. It creates complementary chunk types:

| Chunk type | Purpose |
|---|---|
| `paper_summary` | Paper-level routing |
| `section_parent` | Broad section context and parent expansion |
| `paragraph_child` | General semantic recall |
| `sentence_window_child` | Precise evidence with neighboring sentences |
| `figure_table` | Figure captions, table captions, and table text |
| `rule_candidate` | Domain-triggered threshold and classification evidence |

Long paragraph children use `RecursiveCharacterTextSplitter` with configurable size and overlap. Index headers preserve parser, chunker, vocabulary, source-hash, and embedding-model versions so incompatible indexes can be rebuilt.

## Installation

Python 3.10 or later is required.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e .
python -m pip install pytest
```

For commands run without an editable install:

```powershell
$env:PYTHONPATH='src'
```

## DashScope Configuration

Store the API key in an environment variable. Do not commit it.

```powershell
$env:DASHSCOPE_API_KEY='your-key'
$env:EMBEDDING_PROVIDER='dashscope'
$env:DASHSCOPE_EMBEDDING_MODEL='text-embedding-v4'
$env:LLM_PROVIDER='dashscope'
$env:DASHSCOPE_LLM_MODEL='qwen-plus'
```

Embedding and chat use separate model endpoints under the DashScope OpenAI-compatible API. Building an index embeds paper chunks once; later searches embed only their short queries unless the index is rebuilt.

## Paper Indexing and Query

Put private papers under `data/papers/`. The directory is ignored by Git except for `.gitkeep`.

```powershell
python -m satellite_paper_rag.cli index-paper `
  --paper "paper.pdf" `
  --embedding-provider dashscope `
  --index-dir data\index\dashscope `
  --verbose
```

```powershell
python -m satellite_paper_rag.cli semantic-query `
  --paper "paper.pdf" `
  --query "What S7 and S8 evidence is used for cloud detection?" `
  --embedding-provider dashscope `
  --index-dir data\index\dashscope `
  --top-k 5
```

## CSV Evidence Audit

Current CSV input:

```csv
S1,S2,S3,S4,S5,S6,S7,S8,S9,pc1,pc2,pc3,label,image_id,row,col
```

Current class mapping:

```text
0 = sea
1 = ice
2 = cloud
```

RAG uses `S1-S9`, `label`, and the location fields. `pc1-pc3` may remain in the CSV for compatibility but are excluded from retrieval queries and LLM evidence context.

The current data-generation contract is:

```text
S1-S6 = radiance from S*_radiance_in.nc
S7-S9 = brightness temperature from S*_BT_in.nc
units = not yet confirmed from NetCDF attributes
```

The auditor must not treat S1-S6 radiance as reflectance or apply numeric BT thresholds while input units are unknown.

```powershell
python -m satellite_paper_rag.cli audit-prediction-evidence `
  --paper "paper.pdf" `
  --predictions-file data\csv\rag_input.csv `
  --row-index 0 `
  --embedding-provider dashscope `
  --index-dir data\index\dashscope `
  --top-k 3 `
  --llm-provider dashscope `
  --verbose
```

The audit status is one of:

```text
strong | partial | weak | conflict | insufficient
```

The LLM cannot replace the supplied class. Citations must name a retrieved chunk and quote text that exists in that chunk. Invalid citations force an `insufficient` result and human review.

## Other CLI Workflows

```text
query                         lexical and metadata retrieval
index-paper                   build a local embedding index
semantic-query                dense semantic retrieval
extract-rules                 deterministic rule-candidate extraction
llm-extract-rules             Qwen rule extraction with evidence guardrails
eval-rules                    fixed-case rule extraction evaluation
apply-rules                   apply an executable rule library to observations
explain-prediction-evidence   retrieval-only label evidence workflow
audit-prediction-evidence     retrieval plus Qwen evidence audit
```

Run `python -m satellite_paper_rag.cli <command> --help` for command-specific options.

## Tests

```powershell
$env:PYTHONPATH='src'
python -m pytest -q
```

Tests use deterministic embeddings and fake chat transports where possible, so the normal test suite does not require API calls.

## Known Limitations

- English papers only
- scanned/image-only PDF pages require an OCR stage that is not implemented
- PDF layout and table extraction remain heuristic
- the JSONL vector index is intended for local-scale development, not distributed retrieval
- lexical and dense retrieval exist but are not yet fused into a full production hybrid retriever
- retrieved sentence and paragraph chunks can contain near-duplicate evidence
- exact-quote guardrails validate citation existence, not every semantic inference made by the LLM
- physical-quantity compatibility checks are still being strengthened
- S1-S9 NetCDF units and scaling must be confirmed before applying numeric paper thresholds
- the source dataset's `clear_labels.nc` to `sea` mapping remains provisional
- the current paper set is not sufficient for complete quantitative `sea/ice/cloud` validation

## Data and Secret Policy

Paper files, vector indexes, CSV observations, generated rule files, and environment files are local artifacts and are not intended for Git. Source code, tests, lightweight plans, and documentation are versioned.
