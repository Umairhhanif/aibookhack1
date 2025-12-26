"""
Pydantic models for document chunks and metadata.
"""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class ChunkMetadata(BaseModel):
    """Metadata for document chunks."""

    source: Literal["book"] = Field(
        "book", description="Always 'book' for content restriction"
    )
    file_path: str = Field(..., description="Original Markdown file path")
    module_number: Optional[int] = Field(
        None, ge=1, le=4, description="Module (1-4) or null for non-module content"
    )
    week_number: Optional[int] = Field(
        None, ge=1, le=13, description="Week (1-13) or null for non-week content"
    )
    section_heading: str = Field(..., description="H2/H3 heading text")
    content_type: Literal["lesson", "exercise", "lab", "setup"] = Field(
        "lesson", description="Type of content"
    )
    chunk_index: int = Field(0, description="Position within section")


class DocumentChunk(BaseModel):
    """A chunk of document content for embedding."""

    id: str = Field(..., description="UUID for chunk")
    content: str = Field(..., description="Text content (max 500 tokens)")
    token_count: int = Field(..., ge=1, description="Actual token count")
    metadata: ChunkMetadata = Field(..., description="Source information")


class EmbeddedChunk(DocumentChunk):
    """Document chunk with embedding vector."""

    embedding: List[float] = Field(
        ..., min_length=1536, max_length=1536, description="1536-dim embedding vector"
    )


class ParsedSection(BaseModel):
    """A parsed section from a Markdown document."""

    heading: str = Field(..., description="H2 or H3 text")
    level: Literal[2, 3] = Field(..., description="Heading level")
    content: str = Field(..., description="Text under heading")
    code_blocks: List[str] = Field(default_factory=list)


class ParsedDocument(BaseModel):
    """A fully parsed Markdown document."""

    file_path: str = Field(..., description="Path to source file")
    title: str = Field(..., description="From H1 or frontmatter")
    frontmatter: dict = Field(default_factory=dict)
    sections: List[ParsedSection] = Field(default_factory=list)

    def get_module_number(self) -> Optional[int]:
        """Extract module number from file path or frontmatter."""
        if "module-1" in self.file_path:
            return 1
        elif "module-2" in self.file_path:
            return 2
        elif "module-3" in self.file_path:
            return 3
        elif "module-4" in self.file_path:
            return 4
        return self.frontmatter.get("module_number")

    def get_week_number(self) -> Optional[int]:
        """Extract week number from file path or frontmatter."""
        import re

        match = re.search(r"week-(\d+)", self.file_path)
        if match:
            return int(match.group(1))
        return self.frontmatter.get("week_number")

    def get_content_type(self) -> str:
        """Determine content type from file path."""
        if "exercise" in self.file_path.lower():
            return "exercise"
        elif "lab" in self.file_path.lower():
            return "lab"
        elif "setup" in self.file_path.lower() or "lab-setup" in self.file_path:
            return "setup"
        return "lesson"
