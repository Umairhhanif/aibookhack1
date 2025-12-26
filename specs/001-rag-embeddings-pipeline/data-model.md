# Data Model: RAG Embeddings Pipeline

**Feature**: 001-rag-embeddings-pipeline
**Date**: 2025-12-26
**Phase**: 1 - Design

## Entity Definitions

### Document

Represents a single crawled page from the Docusaurus site.

| Field | Type | Description | Validation |
|-------|------|-------------|------------|
| url | string | Source URL of the page | Required, valid URL, starts with base_url |
| title | string | Page title extracted from HTML | Required, non-empty |
| content | string | Cleaned text content | Required, min 50 characters |
| crawled_at | datetime | Timestamp of crawl | Required, ISO 8601 format |

**Relationships**: One Document → Many Chunks

---

### Chunk

A segment of text from a document, ready for embedding.

| Field | Type | Description | Validation |
|-------|------|-------------|------------|
| id | string | Unique identifier (UUID) | Required, UUID format |
| document_url | string | Parent document URL | Required, references Document.url |
| text | string | Chunk text content | Required, 50-2000 characters |
| token_count | integer | Token count for chunk | Required, 50-500 tokens |
| position | integer | Position index in document | Required, >= 0 |

**Relationships**: Many Chunks → One Document

---

### Embedding

Vector representation stored in Qdrant.

| Field | Type | Description | Validation |
|-------|------|-------------|------------|
| id | string | Qdrant point ID (UUID) | Required, UUID format |
| vector | float[1024] | Cohere embed-english-v3.0 output | Required, 1024 dimensions |
| payload.url | string | Source document URL | Required |
| payload.title | string | Document title | Required |
| payload.text | string | Chunk text for retrieval | Required |
| payload.position | integer | Chunk position in document | Required |

**Relationships**: One Embedding → One Chunk (1:1 mapping)

---

### Collection

Qdrant collection configuration.

| Field | Type | Description | Validation |
|-------|------|-------------|------------|
| name | string | Collection name | Required, alphanumeric + underscore |
| vector_size | integer | Vector dimensions | Required, 1024 |
| distance | string | Distance metric | Required, "Cosine" |

---

## State Transitions

### Pipeline States

```
IDLE → CRAWLING → CHUNKING → EMBEDDING → STORING → COMPLETE
                     ↓           ↓           ↓
                  ERROR       ERROR       ERROR
```

| State | Description | Next States |
|-------|-------------|-------------|
| IDLE | Pipeline not running | CRAWLING |
| CRAWLING | Fetching URLs from sitemap | CHUNKING, ERROR |
| CHUNKING | Splitting documents into chunks | EMBEDDING, ERROR |
| EMBEDDING | Generating vectors with Cohere | STORING, ERROR |
| STORING | Upserting to Qdrant | COMPLETE, ERROR |
| COMPLETE | Pipeline finished successfully | IDLE |
| ERROR | Pipeline failed | IDLE (after retry/fix) |

---

## Data Flow

```
Sitemap URL
    ↓
[Crawl] → Document[]
    ↓
[Clean] → Document[] (filtered, cleaned)
    ↓
[Chunk] → Chunk[]
    ↓
[Embed] → (Chunk, Vector)[]
    ↓
[Store] → Qdrant Collection
```

---

## Validation Rules

### URL Validation
- Must start with configured base URL
- Must return HTTP 200
- Must contain extractable content

### Content Validation
- Minimum 50 characters after cleaning
- Must not be navigation-only page
- Must have valid title

### Chunk Validation
- 50-500 tokens per chunk
- 50-2000 characters per chunk
- Non-empty text after stripping whitespace

### Embedding Validation
- Exactly 1024 dimensions
- All values are valid floats
- No NaN or Inf values
