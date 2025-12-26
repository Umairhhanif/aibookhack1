"""
Qdrant vector database connection utilities.
Handles document chunk storage and similarity search.
"""

import os
from typing import List, Optional

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models

load_dotenv()

# Configuration
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_URL = os.getenv("QDRANT_URL")  # For Qdrant Cloud
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")  # For Qdrant Cloud

COLLECTION_NAME = "book_content"
VECTOR_SIZE = 1536  # OpenAI text-embedding-3-small

# Singleton client
_client: Optional[QdrantClient] = None


def get_qdrant_client() -> QdrantClient:
    """Get or create the Qdrant client."""
    global _client
    if _client is None:
        if QDRANT_URL and QDRANT_API_KEY:
            # Qdrant Cloud
            _client = QdrantClient(
                url=QDRANT_URL,
                api_key=QDRANT_API_KEY,
            )
        else:
            # Local Qdrant
            _client = QdrantClient(
                host=QDRANT_HOST,
                port=QDRANT_PORT,
            )
    return _client


def init_collection() -> None:
    """Initialize the book_content collection if it doesn't exist."""
    client = get_qdrant_client()

    collections = client.get_collections().collections
    collection_names = [c.name for c in collections]

    if COLLECTION_NAME not in collection_names:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=qdrant_models.VectorParams(
                size=VECTOR_SIZE,
                distance=qdrant_models.Distance.COSINE,
            ),
        )

        # Create payload indexes for filtering
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="source",
            field_schema=qdrant_models.PayloadSchemaType.KEYWORD,
        )
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="module_number",
            field_schema=qdrant_models.PayloadSchemaType.INTEGER,
        )
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="week_number",
            field_schema=qdrant_models.PayloadSchemaType.INTEGER,
        )


def search_similar(
    query_vector: List[float],
    limit: int = 5,
    score_threshold: float = 0.7,
    module_filter: Optional[int] = None,
    week_filter: Optional[int] = None,
) -> List[dict]:
    """
    Search for similar document chunks.
    Always filters by source="book" to ensure only book content is returned.
    """
    client = get_qdrant_client()

    # Build filter conditions
    must_conditions = [
        qdrant_models.FieldCondition(
            key="source",
            match=qdrant_models.MatchValue(value="book"),
        )
    ]

    if module_filter is not None:
        must_conditions.append(
            qdrant_models.FieldCondition(
                key="module_number",
                match=qdrant_models.MatchValue(value=module_filter),
            )
        )

    if week_filter is not None:
        must_conditions.append(
            qdrant_models.FieldCondition(
                key="week_number",
                match=qdrant_models.MatchValue(value=week_filter),
            )
        )

    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        query_filter=qdrant_models.Filter(must=must_conditions),
        limit=limit,
        score_threshold=score_threshold,
    )

    return [
        {
            "id": str(hit.id),
            "score": hit.score,
            "content": hit.payload.get("content", ""),
            "file_path": hit.payload.get("file_path", ""),
            "section_heading": hit.payload.get("section_heading", ""),
            "module_number": hit.payload.get("module_number"),
            "week_number": hit.payload.get("week_number"),
            "content_type": hit.payload.get("content_type", ""),
        }
        for hit in results
    ]


def upsert_chunks(chunks: List[dict]) -> None:
    """
    Insert or update document chunks in the collection.
    Each chunk must have: id, embedding, content, and metadata fields.
    """
    client = get_qdrant_client()

    points = [
        qdrant_models.PointStruct(
            id=chunk["id"],
            vector=chunk["embedding"],
            payload={
                "content": chunk["content"],
                "source": "book",  # Always mark as book content
                "file_path": chunk.get("file_path", ""),
                "section_heading": chunk.get("section_heading", ""),
                "module_number": chunk.get("module_number"),
                "week_number": chunk.get("week_number"),
                "content_type": chunk.get("content_type", "lesson"),
            },
        )
        for chunk in chunks
    ]

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
    )


def delete_by_file_path(file_path: str) -> None:
    """Delete all chunks for a specific file path."""
    client = get_qdrant_client()

    client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=qdrant_models.FilterSelector(
            filter=qdrant_models.Filter(
                must=[
                    qdrant_models.FieldCondition(
                        key="file_path",
                        match=qdrant_models.MatchValue(value=file_path),
                    )
                ]
            )
        ),
    )
