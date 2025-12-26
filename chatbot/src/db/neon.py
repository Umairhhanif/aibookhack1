"""
Neon PostgreSQL connection utilities.
Handles chat session and message persistence.
"""

import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

import asyncpg
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "")

# Connection pool
_pool: Optional[asyncpg.Pool] = None


async def get_pool() -> asyncpg.Pool:
    """Get or create the connection pool."""
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(
            DATABASE_URL,
            min_size=2,
            max_size=10,
            command_timeout=60,
        )
    return _pool


async def close_pool() -> None:
    """Close the connection pool."""
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


@asynccontextmanager
async def get_db_connection() -> AsyncGenerator[asyncpg.Connection, None]:
    """Get a database connection from the pool."""
    pool = await get_pool()
    async with pool.acquire() as connection:
        yield connection


async def init_db() -> None:
    """Initialize database tables if they don't exist."""
    async with get_db_connection() as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS chat_sessions (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                created_at TIMESTAMP DEFAULT NOW(),
                last_active TIMESTAMP DEFAULT NOW(),
                message_count INTEGER DEFAULT 0,
                page_context TEXT
            );
        """)

        await conn.execute("""
            CREATE TABLE IF NOT EXISTS chat_messages (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                session_id UUID REFERENCES chat_sessions(id) ON DELETE CASCADE,
                role VARCHAR(10) NOT NULL CHECK (role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT NOW(),
                selected_text TEXT,
                citations JSONB
            );
        """)

        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_messages_session
            ON chat_messages(session_id);
        """)

        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_sessions_active
            ON chat_sessions(last_active);
        """)


async def create_session(page_context: Optional[str] = None) -> str:
    """Create a new chat session and return its ID."""
    async with get_db_connection() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO chat_sessions (page_context)
            VALUES ($1)
            RETURNING id::text
            """,
            page_context,
        )
        return row["id"]


async def add_message(
    session_id: str,
    role: str,
    content: str,
    selected_text: Optional[str] = None,
    citations: Optional[dict] = None,
) -> str:
    """Add a message to a session and return its ID."""
    import json

    async with get_db_connection() as conn:
        # Update session last_active and message_count
        await conn.execute(
            """
            UPDATE chat_sessions
            SET last_active = NOW(), message_count = message_count + 1
            WHERE id = $1::uuid
            """,
            session_id,
        )

        # Insert message
        row = await conn.fetchrow(
            """
            INSERT INTO chat_messages (session_id, role, content, selected_text, citations)
            VALUES ($1::uuid, $2, $3, $4, $5::jsonb)
            RETURNING id::text
            """,
            session_id,
            role,
            content,
            selected_text,
            json.dumps(citations) if citations else None,
        )
        return row["id"]


async def get_session_messages(session_id: str, limit: int = 50) -> list:
    """Get messages for a session."""
    async with get_db_connection() as conn:
        rows = await conn.fetch(
            """
            SELECT id::text, role, content, created_at, selected_text, citations
            FROM chat_messages
            WHERE session_id = $1::uuid
            ORDER BY created_at ASC
            LIMIT $2
            """,
            session_id,
            limit,
        )
        return [dict(row) for row in rows]
