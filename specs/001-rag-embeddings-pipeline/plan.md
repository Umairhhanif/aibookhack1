# Implementation Plan: RAG Embeddings Pipeline

**Branch**: `001-rag-embeddings-pipeline` | **Date**: 2025-12-26 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/001-rag-embeddings-pipeline/spec.md`

## Summary

Build a Python-based ingestion pipeline that crawls a deployed Docusaurus site, extracts and chunks content, generates embeddings using Cohere embed-english-v3.0, and stores them in Qdrant Cloud for semantic search. Single `main.py` script in `backend/` folder using uv for dependency management.

## Technical Context

**Language/Version**: Python 3.11+
**Primary Dependencies**: httpx, beautifulsoup4, cohere, qdrant-client, python-dotenv
**Storage**: Qdrant Cloud (Free Tier, 1GB)
**Testing**: Manual validation via test queries (pytest optional for unit tests)
**Target Platform**: CLI script, runs locally or in CI
**Project Type**: Single script (backend-only, no frontend)
**Performance Goals**: Process 500 pages in under 10 minutes
**Constraints**: Cohere free tier (1,000 API calls/month), Qdrant free tier (1GB)
**Scale/Scope**: Typical documentation site (50-500 pages, ~50K-500K total tokens)

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Evidence |
|-----------|--------|----------|
| I. Accuracy First | PASS | Pipeline uses official Cohere/Qdrant APIs; no fabricated data |
| II. Clarity for Engineers | PASS | Single script approach, clear function interfaces |
| III. Spec-Kit Plus as Source of Truth | PASS | Following spec.md requirements |
| IV. Reproducible End-to-End Workflows | PASS | Complete setup in quickstart.md, .env template provided |
| V. Traceability and Integrity | PASS | Source URLs preserved in embeddings metadata |
| VI. Concise Instructional Writing | PASS | Minimal dependencies, single entry point |

**Technical Constraints from Constitution**:
- Vector Database: Qdrant (required) ✓
- Documentation Platform: Docusaurus (source content) ✓

## Project Structure

### Documentation (this feature)

```text
specs/001-rag-embeddings-pipeline/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output
│   └── pipeline-interface.md
└── tasks.md             # Phase 2 output (/sp.tasks command)
```

### Source Code (repository root)

```text
backend/
├── main.py              # Single entry point with all pipeline logic
├── pyproject.toml       # uv dependency configuration
├── uv.lock              # Reproducible dependency lock
└── .env                 # Environment configuration (gitignored)
```

**Structure Decision**: Single script approach per user request. All pipeline functions (crawl, chunk, embed, store) in `main.py` for simplicity. No separate modules or test directory for MVP.

## Complexity Tracking

No constitution violations requiring justification. Design follows minimal complexity principle:
- Single file vs. modular package: Single file appropriate for ~500 LOC script
- No test framework: Manual validation via `--query` flag sufficient for MVP

## Design Artifacts

| Artifact | Path | Status |
|----------|------|--------|
| Research | [research.md](./research.md) | Complete |
| Data Model | [data-model.md](./data-model.md) | Complete |
| Contracts | [contracts/pipeline-interface.md](./contracts/pipeline-interface.md) | Complete |
| Quickstart | [quickstart.md](./quickstart.md) | Complete |

## Implementation Approach

Based on user input and research:

1. **Create backend/ folder with uv project**
   - Initialize with `uv init`
   - Add dependencies: httpx, beautifulsoup4, cohere, qdrant-client, python-dotenv

2. **Implement main.py with pipeline functions**
   - `crawl_sitemap()`: Fetch sitemap.xml, extract URLs
   - `fetch_page()`: Async fetch with httpx, extract content with BeautifulSoup
   - `chunk_document()`: Split text into 400-token chunks with 80-token overlap
   - `embed_chunks()`: Batch embed with Cohere (96 per batch)
   - `store_embeddings()`: Upsert to Qdrant with metadata

3. **Add main() entry point**
   - Parse CLI args (--dry-run, --query)
   - Load .env configuration
   - Run pipeline end-to-end
   - Print statistics

## Next Steps

Run `/sp.tasks` to generate detailed implementation tasks from this plan.
