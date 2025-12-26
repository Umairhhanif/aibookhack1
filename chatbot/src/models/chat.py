"""
Pydantic models for chat request and response schemas.
"""

from datetime import datetime
from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class Citation(BaseModel):
    """Source reference for assistant responses."""

    file_path: str = Field(..., description="Path to source document")
    section_heading: str = Field(..., description="Section title")
    relevance_score: float = Field(
        ..., ge=0, le=1, description="Similarity score"
    )
    module_number: Optional[int] = Field(None, ge=1, le=4)
    week_number: Optional[int] = Field(None, ge=1, le=13)


class ChatRequest(BaseModel):
    """Request model for sending a chat message."""

    message: str = Field(
        ..., min_length=1, max_length=2000, description="User's question or message"
    )
    session_id: Optional[str] = Field(
        None, description="Existing session ID (creates new if not provided)"
    )
    page_context: Optional[str] = Field(
        None, description="Current page URL for context"
    )
    selected_text: Optional[str] = Field(
        None, max_length=5000, description="User-highlighted text for context"
    )


class ChatResponse(BaseModel):
    """Response model for chat messages."""

    message_id: str = Field(..., description="Message UUID")
    session_id: str = Field(..., description="Session UUID")
    response: str = Field(..., description="Assistant's response")
    citations: List[Citation] = Field(
        default_factory=list, description="Source references"
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow, description="Response timestamp"
    )
    out_of_scope: bool = Field(
        False, description="True if query was outside book content"
    )


class CreateSessionRequest(BaseModel):
    """Request model for creating a new session."""

    page_context: Optional[str] = Field(None, description="Initial page URL")


class SessionResponse(BaseModel):
    """Response model for session creation."""

    session_id: str = Field(..., description="Session UUID")
    created_at: datetime = Field(default_factory=datetime.utcnow)


class MessageHistory(BaseModel):
    """Model for a message in session history."""

    id: str = Field(..., description="Message UUID")
    role: str = Field(..., description="'user' or 'assistant'")
    content: str = Field(..., description="Message content")
    created_at: datetime = Field(..., description="Message timestamp")
    citations: Optional[List[Citation]] = Field(None)


class SessionDetailResponse(BaseModel):
    """Response model for session details with message history."""

    session_id: str = Field(..., description="Session UUID")
    created_at: datetime = Field(...)
    last_active: Optional[datetime] = Field(None)
    message_count: int = Field(0)
    messages: List[MessageHistory] = Field(default_factory=list)


class ErrorResponse(BaseModel):
    """Error response model."""

    error: str = Field(..., description="Error code")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[dict] = Field(None, description="Additional error details")
