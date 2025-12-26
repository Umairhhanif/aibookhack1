# Embedding Pipeline Contract

**Purpose**: Define the document processing pipeline for indexing book content to Qdrant

## Pipeline Overview

```
docs/*.md → Parser → Chunker → Embedder → Qdrant
```

## Input Specification

### Source Files

| Path Pattern | Content Type | Include |
|--------------|--------------|---------|
| `docs/**/*.md` | Lesson content | Yes |
| `docs/**/*.mdx` | Interactive content | Yes |
| `static/code-samples/**/*.py` | Code samples | Metadata only |
| `docs/**/_category_.json` | Category metadata | No (skip) |

### File Exclusions

- `docs/**/index.md` if only contains auto-generated content
- Files with `draft: true` in frontmatter
- Files in `docs/archive/` directory

## Processing Stages

### Stage 1: Parser

**Input**: Raw Markdown/MDX file
**Output**: Parsed document with structure

```typescript
interface ParsedDocument {
  file_path: string;
  title: string;            // From H1 or frontmatter
  frontmatter: {
    module_number?: number;
    week_number?: number;
    sidebar_position?: number;
    tags?: string[];
  };
  sections: Section[];
}

interface Section {
  heading: string;          // H2 or H3 text
  level: 2 | 3;
  content: string;          // Text under heading
  code_blocks: CodeBlock[];
}

interface CodeBlock {
  language: string;
  content: string;
  filename?: string;        // From code block title
}
```

**Rules**:
1. Extract frontmatter YAML as metadata
2. Split content by H2/H3 headings
3. Preserve code blocks within sections
4. Strip MDX components (admonitions become plain text)
5. Remove image markdown syntax, keep alt text

### Stage 2: Chunker

**Input**: ParsedDocument
**Output**: DocumentChunk[]

```typescript
interface DocumentChunk {
  id: string;               // UUID v4
  content: string;          // Chunk text
  token_count: number;      // Actual token count
  metadata: ChunkMetadata;
}

interface ChunkMetadata {
  source: "book";           // Always "book" for filtering
  file_path: string;
  module_number: number | null;
  week_number: number | null;
  section_heading: string;
  content_type: "lesson" | "exercise" | "lab" | "setup";
  chunk_index: number;      // Position within section
}
```

**Chunking Rules**:

1. **Target Size**: 500 tokens (using tiktoken cl100k_base)
2. **Overlap**: 50 tokens between consecutive chunks
3. **Minimum Size**: 100 tokens (smaller sections merged with next)
4. **Maximum Size**: 600 tokens (hard limit)

**Chunking Strategy**:
```
1. First, split by H2/H3 sections
2. If section > 500 tokens:
   a. Split by paragraphs
   b. Merge paragraphs until ~500 tokens
   c. Add 50-token overlap from previous chunk
3. If section < 100 tokens:
   a. Merge with next section
   b. Update section_heading to combined
4. Preserve code blocks intact when possible
   - If code block > 400 tokens, split at logical points
```

### Stage 3: Embedder

**Input**: DocumentChunk[]
**Output**: EmbeddedChunk[]

```typescript
interface EmbeddedChunk extends DocumentChunk {
  embedding: number[];      // 1536-dimensional vector
}
```

**Embedding Configuration**:
- Model: `text-embedding-3-small`
- Dimensions: 1536
- Batch Size: 100 chunks per API call
- Rate Limit: 3000 RPM (50 RPS)

**Text Preparation**:
```python
def prepare_for_embedding(chunk: DocumentChunk) -> str:
    # Include section context in embedding
    return f"Section: {chunk.metadata.section_heading}\n\n{chunk.content}"
```

### Stage 4: Qdrant Indexer

**Input**: EmbeddedChunk[]
**Output**: Indexed collection

**Collection Configuration**:
```json
{
  "name": "book_content",
  "vectors": {
    "size": 1536,
    "distance": "Cosine"
  },
  "optimizers_config": {
    "indexing_threshold": 10000
  },
  "replication_factor": 1
}
```

**Upsert Operation**:
```python
qdrant_client.upsert(
    collection_name="book_content",
    points=[
        PointStruct(
            id=chunk.id,
            vector=chunk.embedding,
            payload={
                "content": chunk.content,
                "source": chunk.metadata.source,
                "file_path": chunk.metadata.file_path,
                "module_number": chunk.metadata.module_number,
                "week_number": chunk.metadata.week_number,
                "section_heading": chunk.metadata.section_heading,
                "content_type": chunk.metadata.content_type,
            }
        )
        for chunk in chunks
    ]
)
```

## Incremental Updates

### Change Detection

```python
def should_reindex(file_path: str, last_indexed: datetime) -> bool:
    file_mtime = os.path.getmtime(file_path)
    return datetime.fromtimestamp(file_mtime) > last_indexed
```

### Update Strategy

1. **File Modified**: Delete all chunks for file, re-process entire file
2. **File Deleted**: Delete all chunks for file
3. **New File**: Process and add chunks

### Metadata Table (Neon)

```sql
CREATE TABLE indexed_files (
    file_path VARCHAR(500) PRIMARY KEY,
    last_indexed TIMESTAMP NOT NULL,
    chunk_count INTEGER NOT NULL,
    checksum VARCHAR(64) NOT NULL  -- SHA-256 of file content
);
```

## CLI Commands

### Full Reindex

```bash
python scripts/index_docs.py --full
```

### Incremental Update

```bash
python scripts/index_docs.py --incremental
```

### Verify Index

```bash
python scripts/index_docs.py --verify
```

**Verification Checks**:
1. All docs files have corresponding chunks
2. No orphaned chunks (file deleted but chunks remain)
3. Chunk count matches expected based on file sizes
4. Sample queries return relevant results

## Error Handling

| Error | Action |
|-------|--------|
| File read error | Log warning, skip file, continue |
| Embedding API error | Retry 3x with exponential backoff |
| Qdrant connection error | Fail with error, exit 1 |
| Chunk too large after split | Log error, truncate to 600 tokens |

## Metrics

Track and log:
- Total files processed
- Total chunks created
- Average chunk size (tokens)
- Processing time per file
- Embedding API latency
- Qdrant upsert latency
