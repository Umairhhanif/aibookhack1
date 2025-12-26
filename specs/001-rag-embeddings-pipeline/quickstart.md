# Quickstart: RAG Embeddings Pipeline

**Feature**: 001-rag-embeddings-pipeline
**Date**: 2025-12-26

## Prerequisites

1. **Python 3.11+** installed
2. **uv** package manager installed ([install guide](https://docs.astral.sh/uv/getting-started/installation/))
3. **Cohere API Key** ([get free key](https://dashboard.cohere.com/api-keys))
4. **Qdrant Cloud Account** ([sign up free](https://cloud.qdrant.io/))
5. **Deployed Docusaurus Site** on Vercel with sitemap enabled

---

## Setup

### 1. Initialize Project

```bash
# From repository root
cd backend
uv sync  # Install dependencies from pyproject.toml
```

### 2. Configure Environment

Create `backend/.env`:

```env
COHERE_API_KEY=your-cohere-api-key
QDRANT_URL=https://your-cluster.cloud.qdrant.io
QDRANT_API_KEY=your-qdrant-api-key
BASE_URL=https://your-book.vercel.app
COLLECTION_NAME=book_embeddings
```

### 3. Create Qdrant Collection

The pipeline will auto-create the collection on first run, or you can create it manually in Qdrant Cloud console:
- **Name**: `book_embeddings`
- **Vector Size**: 1024
- **Distance**: Cosine

---

## Usage

### Dry Run (Preview URLs)

```bash
uv run python main.py --dry-run
```

Output shows discovered URLs without processing.

### Full Pipeline

```bash
uv run python main.py
```

Pipeline will:
1. Crawl sitemap for all page URLs
2. Fetch and clean each page
3. Chunk content into 400-token segments
4. Generate embeddings with Cohere
5. Store in Qdrant with metadata

### Test Query

After pipeline completes, run a test query:

```bash
uv run python main.py --query "How do I configure authentication?"
```

---

## Expected Output

```
[INFO] Starting RAG embeddings pipeline...
[INFO] Base URL: https://your-book.vercel.app
[INFO] Discovered 47 URLs from sitemap
[INFO] Processing page 1/47: /docs/intro
[INFO] Processing page 2/47: /docs/getting-started
...
[INFO] Created 312 chunks from 47 documents
[INFO] Generating embeddings (batch 1/4)...
[INFO] Generating embeddings (batch 2/4)...
[INFO] Generating embeddings (batch 3/4)...
[INFO] Generating embeddings (batch 4/4)...
[INFO] Stored 312 embeddings in Qdrant
[INFO] Pipeline complete in 45.2 seconds
```

---

## Troubleshooting

### "COHERE_API_KEY not set"
Ensure `.env` file exists in `backend/` directory with valid key.

### "Sitemap not found"
Check that your Docusaurus site has sitemap plugin enabled in `docusaurus.config.js`:
```js
plugins: ['@docusaurus/plugin-sitemap']
```

### "Rate limit exceeded"
Free tier has 1,000 API calls/month. Wait until next month or upgrade plan.

### "Qdrant connection failed"
Verify QDRANT_URL and QDRANT_API_KEY are correct in Qdrant Cloud console.

---

## Next Steps

After successful pipeline run:
1. Verify embeddings in Qdrant Cloud console
2. Test queries to validate retrieval quality
3. Integrate with RAG chatbot (separate feature)
