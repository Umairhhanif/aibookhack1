# Pipeline Interface Contract

**Feature**: 001-rag-embeddings-pipeline
**Date**: 2025-12-26

## Overview

This document defines the interface contract for the RAG embeddings pipeline. Since this is a CLI script (not an API server), contracts define function signatures and data structures.

---

## Configuration Schema

### Environment Variables

| Variable | Type | Required | Description |
|----------|------|----------|-------------|
| COHERE_API_KEY | string | Yes | Cohere API authentication key |
| QDRANT_URL | string | Yes | Qdrant Cloud cluster URL |
| QDRANT_API_KEY | string | Yes | Qdrant API authentication key |
| BASE_URL | string | Yes | Docusaurus site base URL |
| COLLECTION_NAME | string | No | Qdrant collection name (default: "book_embeddings") |
| DRY_RUN | boolean | No | Preview mode without processing (default: false) |

### .env Example
```env
COHERE_API_KEY=your-cohere-key
QDRANT_URL=https://xxx.us-east4-0.gcp.cloud.qdrant.io
QDRANT_API_KEY=your-qdrant-key
BASE_URL=https://your-book.vercel.app
COLLECTION_NAME=book_embeddings
DRY_RUN=false
```

---

## Function Interfaces

### crawl_sitemap(base_url: str) -> list[str]

Fetch and parse sitemap to discover all URLs.

**Input**:
- `base_url`: Base URL of the Docusaurus site

**Output**:
- List of page URLs discovered from sitemap

**Errors**:
- `CrawlError`: If sitemap cannot be fetched or parsed

---

### fetch_page(url: str) -> Document

Fetch a single page and extract content.

**Input**:
- `url`: URL to fetch

**Output**:
```python
@dataclass
class Document:
    url: str
    title: str
    content: str
    crawled_at: datetime
```

**Errors**:
- `FetchError`: If page cannot be fetched (404, timeout, etc.)
- `ParseError`: If content cannot be extracted

---

### chunk_document(document: Document) -> list[Chunk]

Split document into chunks suitable for embedding.

**Input**:
- `document`: Document to chunk

**Output**:
```python
@dataclass
class Chunk:
    id: str  # UUID
    document_url: str
    text: str
    token_count: int
    position: int
```

**Errors**:
- `ChunkError`: If document cannot be chunked (too short, etc.)

---

### embed_chunks(chunks: list[Chunk]) -> list[tuple[Chunk, list[float]]]

Generate embeddings for chunks using Cohere.

**Input**:
- `chunks`: List of chunks to embed (max 96 per batch)

**Output**:
- List of (chunk, embedding_vector) tuples

**Errors**:
- `EmbedError`: If Cohere API fails
- `RateLimitError`: If rate limit exceeded

---

### store_embeddings(embeddings: list[tuple[Chunk, list[float]]]) -> int

Store embeddings in Qdrant.

**Input**:
- `embeddings`: List of (chunk, vector) tuples

**Output**:
- Number of points successfully stored

**Errors**:
- `StoreError`: If Qdrant operation fails

---

### run_pipeline(base_url: str, dry_run: bool = False) -> PipelineResult

Execute the full pipeline end-to-end.

**Input**:
- `base_url`: Base URL of the Docusaurus site
- `dry_run`: If true, only discover URLs without processing

**Output**:
```python
@dataclass
class PipelineResult:
    urls_discovered: int
    documents_processed: int
    chunks_created: int
    embeddings_stored: int
    errors: list[str]
    duration_seconds: float
```

---

## Error Handling Contract

All errors inherit from base `PipelineError`:

```python
class PipelineError(Exception):
    """Base error for pipeline operations."""
    pass

class CrawlError(PipelineError):
    """Error during URL discovery."""
    pass

class FetchError(PipelineError):
    """Error fetching a page."""
    url: str
    status_code: int | None

class ParseError(PipelineError):
    """Error parsing page content."""
    url: str

class ChunkError(PipelineError):
    """Error chunking document."""
    url: str
    reason: str

class EmbedError(PipelineError):
    """Error generating embeddings."""
    batch_size: int

class RateLimitError(EmbedError):
    """Rate limit exceeded."""
    retry_after: int

class StoreError(PipelineError):
    """Error storing in Qdrant."""
    operation: str
```

---

## Retry Policy

| Operation | Max Retries | Backoff | Timeout |
|-----------|-------------|---------|---------|
| HTTP Fetch | 3 | Exponential (1s, 2s, 4s) | 30s |
| Cohere API | 3 | Exponential (2s, 4s, 8s) | 60s |
| Qdrant Upsert | 3 | Exponential (1s, 2s, 4s) | 30s |

---

## Logging Contract

All operations must log at INFO level:
- Pipeline start/end with duration
- Each URL processed
- Batch embedding progress (every 10 batches)
- Final statistics

Errors must log at ERROR level with full context.
