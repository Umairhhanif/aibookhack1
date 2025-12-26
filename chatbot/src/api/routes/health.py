"""
Health check endpoint.
"""

from datetime import datetime
from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class ServiceStatus(BaseModel):
    """Status of an individual service."""

    status: Literal["up", "down"]


class HealthResponse(BaseModel):
    """Health check response."""

    status: Literal["healthy", "degraded", "unhealthy"]
    timestamp: datetime
    services: dict[str, str]


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Check the health of all services.

    Returns:
        HealthResponse: Current health status of all services
    """
    services = {}
    all_healthy = True

    # Check Qdrant
    try:
        from ...db.qdrant import get_qdrant_client

        client = get_qdrant_client()
        client.get_collections()
        services["qdrant"] = "up"
    except Exception:
        services["qdrant"] = "down"
        all_healthy = False

    # Check Neon PostgreSQL
    try:
        from ...db.neon import get_db_connection

        async with get_db_connection() as conn:
            await conn.fetchval("SELECT 1")
        services["neon"] = "up"
    except Exception:
        services["neon"] = "down"
        all_healthy = False

    # Check OpenAI (just verify API key is set)
    import os

    if os.getenv("OPENAI_API_KEY"):
        services["openai"] = "up"
    else:
        services["openai"] = "down"
        all_healthy = False

    return HealthResponse(
        status="healthy" if all_healthy else "degraded",
        timestamp=datetime.utcnow(),
        services=services,
    )
