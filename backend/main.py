"""
RAG Embeddings Pipeline

Crawl Docusaurus sites, generate embeddings with Cohere, and store in Qdrant.
"""

import argparse
import logging
import os
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import httpx
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class Document:
    """Represents a single crawled page from the Docusaurus site."""

    url: str
    title: str
    content: str
    crawled_at: datetime = field(default_factory=datetime.now)


@dataclass
class Chunk:
    """A segment of text from a document, ready for embedding."""

    id: str
    document_url: str
    title: str
    text: str
    token_count: int
    position: int


@dataclass
class PipelineResult:
    """Result of a pipeline execution."""

    urls_discovered: int = 0
    documents_processed: int = 0
    chunks_created: int = 0
    embeddings_stored: int = 0
    errors: list = field(default_factory=list)
    duration_seconds: float = 0.0


# =============================================================================
# Error Classes
# =============================================================================


class PipelineError(Exception):
    """Base error for pipeline operations."""

    pass


class CrawlError(PipelineError):
    """Error during URL discovery."""

    pass


class FetchError(PipelineError):
    """Error fetching a page."""

    def __init__(self, message: str, url: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.url = url
        self.status_code = status_code


class ParseError(PipelineError):
    """Error parsing page content."""

    def __init__(self, message: str, url: str):
        super().__init__(message)
        self.url = url


class ChunkError(PipelineError):
    """Error chunking document."""

    def __init__(self, message: str, url: str, reason: str):
        super().__init__(message)
        self.url = url
        self.reason = reason


class EmbedError(PipelineError):
    """Error generating embeddings."""

    def __init__(self, message: str, batch_size: int):
        super().__init__(message)
        self.batch_size = batch_size


class RateLimitError(EmbedError):
    """Rate limit exceeded."""

    def __init__(self, message: str, batch_size: int, retry_after: int):
        super().__init__(message, batch_size)
        self.retry_after = retry_after


class StoreError(PipelineError):
    """Error storing in Qdrant."""

    def __init__(self, message: str, operation: str):
        super().__init__(message)
        self.operation = operation


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class Config:
    """Pipeline configuration loaded from environment variables."""

    cohere_api_key: str
    qdrant_url: str
    qdrant_api_key: str
    base_url: str
    collection_name: str = "book_embeddings"


def load_config() -> Config:
    """Load configuration from environment variables."""
    load_dotenv()

    cohere_api_key = os.getenv("COHERE_API_KEY")
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    base_url = os.getenv("BASE_URL")
    collection_name = os.getenv("COLLECTION_NAME", "book_embeddings")

    missing = []
    if not cohere_api_key:
        missing.append("COHERE_API_KEY")
    if not qdrant_url:
        missing.append("QDRANT_URL")
    if not qdrant_api_key:
        missing.append("QDRANT_API_KEY")
    if not base_url:
        missing.append("BASE_URL")

    if missing:
        raise ValueError(f"Missing required environment variables: {', '.join(missing)}")

    return Config(
        cohere_api_key=cohere_api_key,
        qdrant_url=qdrant_url,
        qdrant_api_key=qdrant_api_key,
        base_url=base_url,
        collection_name=collection_name,
    )


# =============================================================================
# HTTP Retry Logic
# =============================================================================

MAX_RETRIES = 3
BACKOFF_MULTIPLIER = 2  # Exponential backoff: 1s, 2s, 4s


async def fetch_with_retry(
    client: httpx.AsyncClient, url: str, timeout: float = 30.0
) -> httpx.Response:
    """Fetch URL with exponential backoff retry logic."""
    last_error = None
    delay = 1.0

    for attempt in range(MAX_RETRIES):
        try:
            response = await client.get(url, timeout=timeout)
            response.raise_for_status()
            return response
        except httpx.HTTPStatusError as e:
            last_error = e
            if e.response.status_code >= 500:
                logger.warning(f"Retry {attempt + 1}/{MAX_RETRIES} for {url}: {e}")
                time.sleep(delay)
                delay *= BACKOFF_MULTIPLIER
            else:
                raise FetchError(f"HTTP {e.response.status_code}", url, e.response.status_code)
        except httpx.RequestError as e:
            last_error = e
            logger.warning(f"Retry {attempt + 1}/{MAX_RETRIES} for {url}: {e}")
            time.sleep(delay)
            delay *= BACKOFF_MULTIPLIER

    raise FetchError(f"Failed after {MAX_RETRIES} retries: {last_error}", url)


# =============================================================================
# Crawling Functions (User Story 1)
# =============================================================================


async def crawl_sitemap(base_url: str) -> list[str]:
    """Fetch and parse sitemap to discover all URLs."""
    sitemap_url = f"{base_url.rstrip('/')}/sitemap.xml"
    logger.info(f"Fetching sitemap from {sitemap_url}")

    async with httpx.AsyncClient() as client:
        try:
            response = await fetch_with_retry(client, sitemap_url)
        except FetchError as e:
            raise CrawlError(f"Failed to fetch sitemap: {e}")

        soup = BeautifulSoup(response.text, "xml")
        urls = []

        for loc in soup.find_all("loc"):
            url = loc.text.strip()
            if url:
                urls.append(url)

        if not urls:
            raise CrawlError("No URLs found in sitemap")

        logger.info(f"Discovered {len(urls)} URLs from sitemap")
        return urls


def clean_content(soup: BeautifulSoup) -> str:
    """Remove navigation, footer, sidebar, TOC elements from HTML and extract text."""
    # Remove unwanted elements
    for selector in [
        "nav",
        "footer",
        "header",
        ".navbar",
        ".theme-doc-sidebar-container",
        ".theme-doc-toc-desktop",
        ".theme-doc-toc-mobile",
        ".table-of-contents",
        ".pagination-nav",
        ".theme-doc-footer",
        ".theme-doc-breadcrumbs",
        "script",
        "style",
        "noscript",
    ]:
        for element in soup.select(selector):
            element.decompose()

    # Try to find main content area
    main_content = soup.select_one("article.markdown") or soup.select_one(
        ".theme-doc-markdown"
    ) or soup.select_one("main") or soup.select_one("article")

    if main_content:
        text = main_content.get_text(separator="\n", strip=True)
    else:
        text = soup.get_text(separator="\n", strip=True)

    # Clean up whitespace
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    return "\n".join(lines)


async def fetch_page(client: httpx.AsyncClient, url: str) -> Optional[Document]:
    """Fetch a single page and extract content."""
    try:
        response = await fetch_with_retry(client, url)
    except FetchError as e:
        logger.warning(f"Failed to fetch {url}: {e}")
        return None

    soup = BeautifulSoup(response.text, "html.parser")

    # Extract title
    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else url.split("/")[-1]

    # Clean and extract content
    content = clean_content(soup)

    # Validate content length (skip pages with insufficient content)
    if len(content) < 50:
        logger.warning(f"Skipping {url}: insufficient content ({len(content)} chars)")
        return None

    return Document(url=url, title=title, content=content)


def estimate_tokens(text: str) -> int:
    """Estimate token count (rough approximation: 4 chars per token)."""
    return len(text) // 4


def chunk_document(document: Document, chunk_size: int = 400, overlap: int = 80) -> list[Chunk]:
    """Split document into chunks suitable for embedding."""
    content = document.content
    chunks = []
    position = 0

    # Split by paragraphs first
    paragraphs = content.split("\n\n")
    current_chunk = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = estimate_tokens(para)

        # If single paragraph exceeds chunk size, split it
        if para_tokens > chunk_size:
            # Flush current chunk first
            if current_chunk:
                chunk_text = "\n\n".join(current_chunk)
                chunks.append(
                    Chunk(
                        id=str(uuid.uuid4()),
                        document_url=document.url,
                        title=document.title,
                        text=chunk_text,
                        token_count=estimate_tokens(chunk_text),
                        position=position,
                    )
                )
                position += 1
                current_chunk = []
                current_tokens = 0

            # Split large paragraph by sentences/chunks
            words = para.split()
            word_chunk = []
            word_tokens = 0

            for word in words:
                word_tokens += estimate_tokens(word + " ")
                word_chunk.append(word)

                if word_tokens >= chunk_size:
                    chunk_text = " ".join(word_chunk)
                    chunks.append(
                        Chunk(
                            id=str(uuid.uuid4()),
                            document_url=document.url,
                            title=document.title,
                            text=chunk_text,
                            token_count=estimate_tokens(chunk_text),
                            position=position,
                        )
                    )
                    position += 1
                    # Keep overlap
                    overlap_words = max(1, overlap // 4)
                    word_chunk = word_chunk[-overlap_words:]
                    word_tokens = estimate_tokens(" ".join(word_chunk))

            # Add remaining words to next chunk
            if word_chunk:
                current_chunk = [" ".join(word_chunk)]
                current_tokens = estimate_tokens(current_chunk[0])

        elif current_tokens + para_tokens > chunk_size:
            # Current chunk is full, save it
            chunk_text = "\n\n".join(current_chunk)
            chunks.append(
                Chunk(
                    id=str(uuid.uuid4()),
                    document_url=document.url,
                    title=document.title,
                    text=chunk_text,
                    token_count=estimate_tokens(chunk_text),
                    position=position,
                )
            )
            position += 1

            # Start new chunk with overlap
            if overlap > 0 and current_chunk:
                # Take last part of previous chunk for overlap
                overlap_text = current_chunk[-1] if current_chunk else ""
                current_chunk = [overlap_text, para] if overlap_text else [para]
                current_tokens = estimate_tokens("\n\n".join(current_chunk))
            else:
                current_chunk = [para]
                current_tokens = para_tokens
        else:
            current_chunk.append(para)
            current_tokens += para_tokens

    # Don't forget the last chunk
    if current_chunk:
        chunk_text = "\n\n".join(current_chunk)
        if estimate_tokens(chunk_text) >= 50:  # Minimum chunk size
            chunks.append(
                Chunk(
                    id=str(uuid.uuid4()),
                    document_url=document.url,
                    title=document.title,
                    text=chunk_text,
                    token_count=estimate_tokens(chunk_text),
                    position=position,
                )
            )

    return chunks


async def crawl_and_chunk(config: Config) -> tuple[list[Document], list[Chunk]]:
    """Crawl all URLs and chunk documents."""
    urls = await crawl_sitemap(config.base_url)
    documents = []
    all_chunks = []

    async with httpx.AsyncClient() as client:
        for i, url in enumerate(urls):
            logger.info(f"Processing page {i + 1}/{len(urls)}: {url}")

            doc = await fetch_page(client, url)
            if doc:
                documents.append(doc)
                chunks = chunk_document(doc)
                all_chunks.extend(chunks)
                logger.info(f"  Created {len(chunks)} chunks from {doc.title}")

    logger.info(f"Processed {len(documents)} documents, created {len(all_chunks)} chunks")
    return documents, all_chunks


# =============================================================================
# Embedding Functions (User Story 2)
# =============================================================================

COHERE_BATCH_SIZE = 96  # Max embeddings per API call
COHERE_EMBED_MODEL = "embed-english-v3.0"
COHERE_EMBED_DIMENSIONS = 1024


def embed_chunks(
    chunks: list[Chunk], config: Config
) -> list[tuple[Chunk, list[float]]]:
    """Generate embeddings for chunks using Cohere."""
    import cohere
    import math

    if not chunks:
        return []

    co = cohere.ClientV2(api_key=config.cohere_api_key)
    embeddings = []
    total_batches = math.ceil(len(chunks) / COHERE_BATCH_SIZE)

    for batch_idx in range(total_batches):
        start = batch_idx * COHERE_BATCH_SIZE
        end = min(start + COHERE_BATCH_SIZE, len(chunks))
        batch_chunks = chunks[start:end]

        logger.info(f"Generating embeddings (batch {batch_idx + 1}/{total_batches})...")

        # Extract texts for embedding
        texts = [chunk.text for chunk in batch_chunks]

        # Call Cohere API with retry logic
        delay = 2.0
        last_error = None

        for attempt in range(MAX_RETRIES):
            try:
                response = co.embed(
                    texts=texts,
                    model=COHERE_EMBED_MODEL,
                    input_type="search_document",
                    embedding_types=["float"],
                )

                # Extract embeddings from response
                batch_embeddings = response.embeddings.float_

                # Validate embeddings
                for i, embedding in enumerate(batch_embeddings):
                    if len(embedding) != COHERE_EMBED_DIMENSIONS:
                        raise EmbedError(
                            f"Invalid embedding dimensions: {len(embedding)} != {COHERE_EMBED_DIMENSIONS}",
                            len(batch_chunks),
                        )

                    # Check for NaN/Inf values
                    if any(math.isnan(v) or math.isinf(v) for v in embedding):
                        raise EmbedError(
                            f"Embedding contains NaN or Inf values",
                            len(batch_chunks),
                        )

                    embeddings.append((batch_chunks[i], embedding))

                break  # Success, exit retry loop

            except cohere.TooManyRequestsError as e:
                last_error = e
                retry_after = int(getattr(e, "retry_after", delay))
                logger.warning(
                    f"Rate limit hit, waiting {retry_after}s (attempt {attempt + 1}/{MAX_RETRIES})"
                )
                time.sleep(retry_after)
                delay *= BACKOFF_MULTIPLIER

            except cohere.ApiError as e:
                last_error = e
                if attempt < MAX_RETRIES - 1:
                    logger.warning(
                        f"Cohere API error, retrying in {delay}s (attempt {attempt + 1}/{MAX_RETRIES}): {e}"
                    )
                    time.sleep(delay)
                    delay *= BACKOFF_MULTIPLIER
                else:
                    raise EmbedError(f"Cohere API error after {MAX_RETRIES} retries: {e}", len(batch_chunks))

        else:
            # All retries exhausted
            if last_error:
                raise EmbedError(f"Failed after {MAX_RETRIES} retries: {last_error}", len(batch_chunks))

    logger.info(f"Generated {len(embeddings)} embeddings")
    return embeddings


# =============================================================================
# Storage Functions (User Story 3)
# =============================================================================


def create_collection(config: Config) -> None:
    """Create Qdrant collection with appropriate dimensions and distance metric."""
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams

    client = QdrantClient(
        url=config.qdrant_url,
        api_key=config.qdrant_api_key,
    )

    # Check if collection exists
    collections = client.get_collections().collections
    collection_names = [c.name for c in collections]

    if config.collection_name in collection_names:
        logger.info(f"Collection '{config.collection_name}' already exists")
        return

    # Create new collection
    logger.info(f"Creating collection '{config.collection_name}'...")
    client.create_collection(
        collection_name=config.collection_name,
        vectors_config=VectorParams(
            size=COHERE_EMBED_DIMENSIONS,
            distance=Distance.COSINE,
        ),
    )
    logger.info(f"Collection '{config.collection_name}' created successfully")


def store_embeddings(
    embeddings: list[tuple[Chunk, list[float]]], config: Config
) -> int:
    """Store embeddings in Qdrant with metadata."""
    from qdrant_client import QdrantClient
    from qdrant_client.models import PointStruct

    if not embeddings:
        return 0

    client = QdrantClient(
        url=config.qdrant_url,
        api_key=config.qdrant_api_key,
    )

    # Ensure collection exists
    create_collection(config)

    # Prepare points for upsert
    points = []
    for chunk, vector in embeddings:
        point = PointStruct(
            id=chunk.id,
            vector=vector,
            payload={
                "url": chunk.document_url,
                "title": chunk.title,
                "text": chunk.text,
                "position": chunk.position,
                "token_count": chunk.token_count,
            },
        )
        points.append(point)

    # Upsert in batches with retry logic
    batch_size = 100
    total_stored = 0
    delay = 1.0

    for i in range(0, len(points), batch_size):
        batch = points[i : i + batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (len(points) + batch_size - 1) // batch_size

        logger.info(f"Storing batch {batch_num}/{total_batches} ({len(batch)} points)...")

        for attempt in range(MAX_RETRIES):
            try:
                client.upsert(
                    collection_name=config.collection_name,
                    points=batch,
                )
                total_stored += len(batch)
                break
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    logger.warning(
                        f"Qdrant upsert failed, retrying in {delay}s (attempt {attempt + 1}/{MAX_RETRIES}): {e}"
                    )
                    time.sleep(delay)
                    delay *= BACKOFF_MULTIPLIER
                else:
                    raise StoreError(f"Failed to store batch after {MAX_RETRIES} retries: {e}", "upsert")

    logger.info(f"Stored {total_stored} embeddings in Qdrant")
    return total_stored


# =============================================================================
# Query Functions (User Story 4)
# =============================================================================


def test_query(query: str, config: Config, top_k: int = 5) -> list[dict]:
    """Embed query and search Qdrant for similar chunks."""
    import cohere
    from qdrant_client import QdrantClient

    logger.info(f"Querying for: '{query}'")

    # Generate embedding for query
    co = cohere.ClientV2(api_key=config.cohere_api_key)
    response = co.embed(
        texts=[query],
        model=COHERE_EMBED_MODEL,
        input_type="search_query",
        embedding_types=["float"],
    )
    query_vector = response.embeddings.float_[0]

    # Search Qdrant
    client = QdrantClient(
        url=config.qdrant_url,
        api_key=config.qdrant_api_key,
    )

    results = client.query_points(
        collection_name=config.collection_name,
        query=query_vector,
        limit=top_k,
    ).points

    # Format results
    formatted_results = []
    for i, result in enumerate(results):
        formatted_results.append({
            "rank": i + 1,
            "score": result.score,
            "url": result.payload.get("url", ""),
            "title": result.payload.get("title", ""),
            "text": result.payload.get("text", "")[:200] + "..." if len(result.payload.get("text", "")) > 200 else result.payload.get("text", ""),
        })

    return formatted_results


def print_query_results(results: list[dict]) -> None:
    """Display query results in a formatted way."""
    if not results:
        logger.info("No results found")
        return

    logger.info(f"\nFound {len(results)} results:\n")
    logger.info("=" * 80)

    for result in results:
        logger.info(f"[{result['rank']}] Score: {result['score']:.4f}")
        logger.info(f"    Title: {result['title']}")
        logger.info(f"    URL: {result['url']}")
        logger.info(f"    Text: {result['text']}")
        logger.info("-" * 80)


def run_pipeline(config: Config, dry_run: bool = False) -> PipelineResult:
    """Execute the full pipeline end-to-end."""
    import asyncio

    start_time = time.time()
    result = PipelineResult()

    try:
        # Phase 1: Crawl and chunk
        documents, chunks = asyncio.run(crawl_and_chunk(config))
        result.urls_discovered = len(documents) + len(result.errors)
        result.documents_processed = len(documents)
        result.chunks_created = len(chunks)

        if dry_run:
            logger.info("Dry-run mode: skipping embedding and storage")
            result.duration_seconds = time.time() - start_time
            return result

        # Phase 2: Embed
        embeddings = embed_chunks(chunks, config)

        # Phase 3: Store (to be implemented in US3)
        if embeddings:
            stored = store_embeddings(embeddings, config)
            result.embeddings_stored = stored

    except Exception as e:
        result.errors.append(str(e))
        logger.error(f"Pipeline error: {e}")

    result.duration_seconds = time.time() - start_time
    return result


# =============================================================================
# CLI Argument Parser
# =============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="RAG Embeddings Pipeline - Crawl, embed, and store documentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run python main.py                    # Run full pipeline
  uv run python main.py --dry-run          # Preview URLs without processing
  uv run python main.py --query "topic"    # Query existing embeddings
        """,
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Discover URLs and extract content without embedding or storing",
    )

    parser.add_argument(
        "--query",
        type=str,
        metavar="TEXT",
        help="Query existing embeddings instead of running pipeline",
    )

    return parser.parse_args()


# =============================================================================
# Main Entry Point (placeholder for user story implementations)
# =============================================================================


def main() -> None:
    """Main entry point for the pipeline."""
    args = parse_args()

    try:
        config = load_config()
        logger.info("Configuration loaded successfully")
        logger.info(f"Base URL: {config.base_url}")
        logger.info(f"Collection: {config.collection_name}")

        if args.query:
            results = test_query(args.query, config)
            print_query_results(results)
        else:
            # Run pipeline (full or dry-run)
            result = run_pipeline(config, dry_run=args.dry_run)

            # Print summary
            logger.info("=" * 50)
            logger.info("Pipeline Summary")
            logger.info("=" * 50)
            logger.info(f"URLs discovered: {result.urls_discovered}")
            logger.info(f"Documents processed: {result.documents_processed}")
            logger.info(f"Chunks created: {result.chunks_created}")
            logger.info(f"Embeddings stored: {result.embeddings_stored}")
            logger.info(f"Duration: {result.duration_seconds:.2f} seconds")

            if result.errors:
                logger.warning(f"Errors: {len(result.errors)}")
                for error in result.errors:
                    logger.error(f"  - {error}")

    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)
    except PipelineError as e:
        logger.error(f"Pipeline error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
