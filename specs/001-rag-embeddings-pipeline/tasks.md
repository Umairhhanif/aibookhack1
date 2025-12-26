# Tasks: RAG Embeddings Pipeline

**Input**: Design documents from `/specs/001-rag-embeddings-pipeline/`
**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/

**Tests**: Manual validation via CLI flags. No automated test framework for MVP.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3, US4)
- Include exact file paths in descriptions

## Path Conventions

- **Single script project**: `backend/` at repository root
- Main entry point: `backend/main.py`

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and dependency setup

- [x] T001 Create backend/ directory at repository root
- [x] T002 Initialize Python project with uv in backend/ directory
- [x] T003 Add dependencies to backend/pyproject.toml: httpx, beautifulsoup4, cohere, qdrant-client, python-dotenv
- [x] T004 Create backend/.env.example with placeholder values for COHERE_API_KEY, QDRANT_URL, QDRANT_API_KEY, BASE_URL, COLLECTION_NAME
- [x] T005 Add backend/.env to .gitignore

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**CRITICAL**: No user story work can begin until this phase is complete

- [x] T006 Create backend/main.py with imports, dataclasses (Document, Chunk, PipelineResult), and error classes (PipelineError, CrawlError, FetchError, ParseError, ChunkError, EmbedError, RateLimitError, StoreError)
- [x] T007 Implement load_config() function in backend/main.py to load environment variables with python-dotenv
- [x] T008 Implement CLI argument parser in backend/main.py with --dry-run and --query flags
- [x] T009 Implement logging setup in backend/main.py with INFO level for progress and ERROR level for failures

**Checkpoint**: Foundation ready - user story implementation can now begin

---

## Phase 3: User Story 1 - Crawl and Ingest Docusaurus Content (Priority: P1)

**Goal**: Discover all URLs from sitemap, fetch pages, extract clean text, and chunk content

**Independent Test**: Run `uv run python main.py --dry-run` to verify URL discovery and content extraction without embedding

### Implementation for User Story 1

- [x] T010 [US1] Implement crawl_sitemap(base_url: str) -> list[str] in backend/main.py - fetch sitemap.xml and parse all <loc> URLs
- [x] T011 [US1] Implement fetch_page(url: str) -> Document in backend/main.py - async fetch with httpx, extract title and content with BeautifulSoup
- [x] T012 [US1] Implement clean_content() helper in backend/main.py - remove nav, footer, sidebar, TOC elements from HTML
- [x] T013 [US1] Implement chunk_document(document: Document) -> list[Chunk] in backend/main.py - split text into 400-token chunks with 80-token overlap
- [x] T014 [US1] Implement retry logic with exponential backoff for HTTP requests in backend/main.py
- [x] T015 [US1] Add content validation in backend/main.py - skip pages with <50 characters after cleaning
- [x] T016 [US1] Integrate crawl and chunk functions into run_pipeline() with --dry-run support in backend/main.py

**Checkpoint**: User Story 1 complete - can crawl and chunk content independently

---

## Phase 4: User Story 2 - Generate Embeddings with Cohere (Priority: P2)

**Goal**: Convert text chunks into vector embeddings using Cohere API

**Independent Test**: Provide sample chunks and verify Cohere returns 1024-dimension vectors

### Implementation for User Story 2

- [x] T017 [US2] Implement embed_chunks(chunks: list[Chunk]) -> list[tuple[Chunk, list[float]]] in backend/main.py using cohere.ClientV2
- [x] T018 [US2] Implement batch processing in embed_chunks() - process 96 chunks per API call
- [x] T019 [US2] Add rate limit handling with exponential backoff (2s, 4s, 8s) for Cohere API in backend/main.py
- [x] T020 [US2] Add embedding validation in backend/main.py - verify 1024 dimensions, no NaN/Inf values
- [x] T021 [US2] Integrate embed_chunks() into run_pipeline() in backend/main.py

**Checkpoint**: User Story 2 complete - can generate embeddings from chunks independently

---

## Phase 5: User Story 3 - Store Embeddings in Qdrant (Priority: P3)

**Goal**: Persist embeddings to Qdrant Cloud with metadata for retrieval

**Independent Test**: Connect to Qdrant, create collection, upsert sample embeddings, verify storage

### Implementation for User Story 3

- [x] T022 [US3] Implement create_collection() in backend/main.py - create Qdrant collection with 1024 dimensions and Cosine distance
- [x] T023 [US3] Implement store_embeddings(embeddings: list[tuple[Chunk, list[float]]]) -> int in backend/main.py using qdrant_client
- [x] T024 [US3] Add payload structure in store_embeddings() - include url, title, text, position in each point
- [x] T025 [US3] Implement retry logic for Qdrant upsert operations in backend/main.py
- [x] T026 [US3] Integrate store_embeddings() into run_pipeline() in backend/main.py

**Checkpoint**: User Story 3 complete - can store embeddings with metadata independently

---

## Phase 6: User Story 4 - Validate Pipeline End-to-End (Priority: P4)

**Goal**: Verify the entire pipeline works by running test queries

**Independent Test**: Run `uv run python main.py --query "topic"` and verify relevant chunks are returned

### Implementation for User Story 4

- [x] T027 [US4] Implement test_query(query: str) -> list[dict] in backend/main.py - embed query and search Qdrant
- [x] T028 [US4] Format query results in backend/main.py - display top-5 results with URL, title, text snippet, and score
- [x] T029 [US4] Implement main() entry point in backend/main.py - orchestrate full pipeline or query mode based on CLI flags
- [x] T030 [US4] Add pipeline statistics output in backend/main.py - print URLs discovered, documents processed, chunks created, embeddings stored, duration

**Checkpoint**: User Story 4 complete - full pipeline validation working

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories

- [x] T031 Add progress logging throughout pipeline in backend/main.py - log each URL processed, batch progress
- [x] T032 Verify all error paths log with full context in backend/main.py
- [ ] T033 Run quickstart.md validation - execute full pipeline against deployed Vercel site
- [x] T034 Update backend/.env.example with actual Vercel URL placeholder

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3-6)**: All depend on Foundational phase completion
  - User stories are sequential due to data flow (P1 → P2 → P3 → P4)
- **Polish (Phase 7)**: Depends on all user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Foundational (Phase 2) - produces Document[], Chunk[]
- **User Story 2 (P2)**: Depends on US1 - needs Chunk[] to generate embeddings
- **User Story 3 (P3)**: Depends on US2 - needs (Chunk, Vector)[] to store
- **User Story 4 (P4)**: Depends on US3 - needs populated Qdrant collection to query

### Within Each User Story

- Core implementation before integration
- Validation and error handling after core logic
- Integration into run_pipeline() at the end

### Parallel Opportunities

- T003, T004, T005 can run in parallel (different files)
- Within US1: T010, T011, T012, T013 can be developed in parallel (different functions)
- Within US2: T017, T018, T019 can be developed in parallel (different concerns)
- Within US3: T022, T023, T024 can be developed in parallel (different functions)

---

## Parallel Example: User Story 1

```bash
# Launch parallel development of US1 functions:
Task: "Implement crawl_sitemap() in backend/main.py"
Task: "Implement fetch_page() in backend/main.py"
Task: "Implement clean_content() in backend/main.py"
Task: "Implement chunk_document() in backend/main.py"

# Then integrate sequentially:
Task: "Integrate crawl and chunk functions into run_pipeline()"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational
3. Complete Phase 3: User Story 1
4. **STOP and VALIDATE**: Test with `--dry-run` flag
5. Content crawling and chunking verified

### Incremental Delivery

1. Complete Setup + Foundational → Foundation ready
2. Add User Story 1 → Test with `--dry-run` → Crawling works
3. Add User Story 2 → Embeddings generated
4. Add User Story 3 → Embeddings stored
5. Add User Story 4 → Full validation with `--query`

### Single Developer Flow

Due to data dependencies, execute sequentially:
1. Phase 1 → Phase 2 → Phase 3 (US1) → Phase 4 (US2) → Phase 5 (US3) → Phase 6 (US4) → Phase 7

---

## Summary

| Metric | Count |
|--------|-------|
| Total Tasks | 34 |
| Phase 1 (Setup) | 5 tasks |
| Phase 2 (Foundational) | 4 tasks |
| Phase 3 (US1 - Crawl) | 7 tasks |
| Phase 4 (US2 - Embed) | 5 tasks |
| Phase 5 (US3 - Store) | 5 tasks |
| Phase 6 (US4 - Validate) | 4 tasks |
| Phase 7 (Polish) | 4 tasks |

**MVP Scope**: Phases 1-3 (16 tasks) - crawl and chunk content with dry-run validation

**Format Validation**: All 34 tasks follow checklist format with checkbox, ID, story labels (where applicable), and file paths.

---

## Notes

- All implementation in single file: backend/main.py
- No automated tests for MVP - manual validation via CLI flags
- [P] tasks = different functions/files, no dependencies
- [Story] label maps task to specific user story for traceability
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
