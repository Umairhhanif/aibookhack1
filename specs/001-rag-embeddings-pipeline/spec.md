# Feature Specification: RAG Embeddings Pipeline

**Feature Branch**: `001-rag-embeddings-pipeline`
**Created**: 2025-12-26
**Status**: Draft
**Input**: User description: "Deploy book URLs, generate embeddings, and store them in a vector database"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Crawl and Ingest Docusaurus Content (Priority: P1)

A developer wants to ingest all public content from their deployed Docusaurus book website to prepare it for RAG retrieval. They run the crawling script which discovers all public URLs, fetches the HTML content, and extracts clean text suitable for embedding.

**Why this priority**: This is the foundational step - without content ingestion, there is nothing to embed or store. All downstream processes depend on having clean, chunked text available.

**Independent Test**: Can be fully tested by running the crawler against the deployed Vercel URL and verifying that markdown/text content is extracted from all discovered pages.

**Acceptance Scenarios**:

1. **Given** a deployed Docusaurus site URL, **When** the crawler is executed, **Then** it discovers all public documentation pages via sitemap or link traversal
2. **Given** discovered URLs, **When** content is fetched, **Then** HTML is converted to clean text with navigation/boilerplate removed
3. **Given** raw text content, **When** chunking is applied, **Then** text is split into semantically meaningful chunks suitable for embedding

---

### User Story 2 - Generate Embeddings with Cohere (Priority: P2)

A developer wants to convert their text chunks into vector embeddings using Cohere's embedding models. They run the embedding script which processes all chunks and returns dense vector representations.

**Why this priority**: Embeddings are the core transformation that enables semantic search. This builds directly on the ingested content from P1.

**Independent Test**: Can be fully tested by providing sample text chunks and verifying that Cohere API returns correctly-dimensioned embeddings for each chunk.

**Acceptance Scenarios**:

1. **Given** a list of text chunks, **When** the embedding script is executed, **Then** each chunk receives a corresponding embedding vector from Cohere
2. **Given** Cohere API credentials in environment, **When** embeddings are requested, **Then** the system authenticates and handles rate limits appropriately
3. **Given** a batch of chunks, **When** processing completes, **Then** embeddings are returned with metadata linking them to source URLs

---

### User Story 3 - Store Embeddings in Qdrant (Priority: P3)

A developer wants to persist their embeddings in Qdrant vector database for later retrieval. They run the storage script which creates a collection and upserts all embeddings with their metadata.

**Why this priority**: Storage is essential for persistence and retrieval, but depends on having embeddings from P2.

**Independent Test**: Can be fully tested by connecting to Qdrant Cloud, creating a collection, upserting sample embeddings, and querying to verify storage.

**Acceptance Scenarios**:

1. **Given** Qdrant Cloud credentials, **When** the storage script runs, **Then** it connects and creates/updates the target collection with appropriate vector dimensions
2. **Given** embeddings with metadata, **When** upsert is executed, **Then** all vectors are stored with their source URL, chunk text, and position metadata
3. **Given** stored embeddings, **When** a test query is performed, **Then** semantically similar chunks are returned with relevance scores

---

### User Story 4 - Validate Pipeline End-to-End (Priority: P4)

A developer wants to verify the entire pipeline works correctly by running test queries against the populated vector database and confirming relevant results are returned.

**Why this priority**: Validation ensures the pipeline produces usable results, but only makes sense after all components are working.

**Independent Test**: Can be fully tested by running sample queries and verifying that returned chunks are semantically relevant to the query.

**Acceptance Scenarios**:

1. **Given** a fully populated Qdrant collection, **When** a test query is submitted, **Then** the top-k results contain semantically relevant content
2. **Given** a query about a specific topic covered in the book, **When** search is performed, **Then** chunks from relevant sections are returned in the results

---

### Edge Cases

- What happens when a URL returns a 404 or redirect? System should log the error and continue with remaining URLs.
- What happens when Cohere API rate limit is exceeded? System should implement exponential backoff and retry.
- What happens when Qdrant connection fails mid-upload? System should support resume/retry from the last successful batch.
- What happens when a page contains only navigation/no meaningful content? System should skip pages with insufficient text content.
- What happens when text chunks are too short or too long? System should enforce minimum/maximum chunk sizes with configurable thresholds.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST crawl all public URLs from a deployed Docusaurus site starting from a base URL
- **FR-002**: System MUST extract and clean text content from HTML pages, removing navigation, headers, footers, and boilerplate
- **FR-003**: System MUST chunk text into segments between 200-1000 tokens with configurable overlap
- **FR-004**: System MUST generate embeddings using Cohere's embed-english-v3.0 or embed-multilingual-v3.0 model
- **FR-005**: System MUST store embeddings in Qdrant with source URL, chunk text, and position metadata
- **FR-006**: System MUST create a Qdrant collection with appropriate vector dimensions matching the Cohere model output
- **FR-007**: System MUST handle API authentication via environment variables for both Cohere and Qdrant
- **FR-008**: System MUST implement retry logic with exponential backoff for transient API failures
- **FR-009**: System MUST log progress and errors to enable debugging and monitoring
- **FR-010**: System MUST support a dry-run mode to preview discovered URLs without processing

### Key Entities

- **Document**: Represents a single crawled page - contains source URL, raw HTML, extracted text, and crawl timestamp
- **Chunk**: A segment of text from a document - contains chunk text, token count, position index, and parent document reference
- **Embedding**: Vector representation of a chunk - contains vector values, chunk reference, and metadata for storage
- **Collection**: Qdrant collection configuration - contains collection name, vector dimensions, and distance metric

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: All public pages from the Docusaurus site are discovered and processed (100% coverage of sitemap URLs)
- **SC-002**: Text extraction produces clean content with less than 5% boilerplate contamination
- **SC-003**: Embedding generation completes for all chunks with zero data loss
- **SC-004**: Vector search returns relevant chunks in the top-5 results for at least 80% of test queries
- **SC-005**: End-to-end pipeline completes within reasonable time for typical documentation sites (under 500 pages)
- **SC-006**: System gracefully handles and logs all errors without crashing or losing processed data

## Assumptions

- The Docusaurus site is publicly accessible (no authentication required for crawling)
- Cohere API key with sufficient quota is available
- Qdrant Cloud Free Tier account is provisioned and accessible
- The book content is primarily in English (or a language supported by the chosen Cohere model)
- Standard chunking strategies (fixed-size with overlap) are acceptable for initial implementation
- Rate limits for Cohere and Qdrant are within free tier allowances for the expected document volume

## Out of Scope

- Retrieval or ranking logic beyond basic test queries
- Agent or chatbot integration
- Frontend or API server
- User authentication or analytics
- Incremental updates (full re-indexing only for MVP)
- Multi-language content handling beyond Cohere's native support
