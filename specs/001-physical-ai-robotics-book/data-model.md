# Data Model: Physical AI & Humanoid Robotics Book

**Branch**: `001-physical-ai-robotics-book` | **Date**: 2025-12-24
**Purpose**: Define content structure and chatbot data models

## Content Entities

### Module

Represents a major learning unit in the curriculum.

| Field | Type | Description |
|-------|------|-------------|
| id | string | Unique identifier (e.g., "module-1") |
| number | integer | Module sequence number (1-4) |
| title | string | Display title |
| subtitle | string | Descriptive subtitle |
| description | string | Overview paragraph |
| weeks | Week[] | Associated weeks |
| prerequisites | string[] | Required prior knowledge |
| learning_objectives | string[] | What students will learn |

**Instances**:
1. Module 1: "The Robotic Nervous System" (Weeks 1-3)
2. Module 2: "The Digital Twin" (Weeks 4-6)
3. Module 3: "The AI-Robot Brain" (Weeks 7-9)
4. Module 4: "Vision-Language-Action" (Weeks 10-13)

### Week

Represents a single week's curriculum content.

| Field | Type | Description |
|-------|------|-------------|
| id | string | Unique identifier (e.g., "week-01") |
| number | integer | Week sequence number (1-13) |
| module_id | string | Parent module reference |
| title | string | Week title |
| summary | string | Brief overview |
| lessons | Lesson[] | Ordered list of lessons |
| exercises | Exercise[] | Practice activities |
| lab | LabAssignment | Optional major project |
| estimated_hours | integer | Expected completion time |

### Lesson

Individual learning content within a week.

| Field | Type | Description |
|-------|------|-------------|
| id | string | Unique identifier |
| week_id | string | Parent week reference |
| order | integer | Display order within week |
| title | string | Lesson title |
| content_path | string | Path to Markdown file |
| learning_objectives | string[] | Specific goals |
| code_samples | string[] | Associated code files |
| ros2_version | string[] | Compatible versions (["humble", "iron"]) |

### Exercise

Hands-on practice activity.

| Field | Type | Description |
|-------|------|-------------|
| id | string | Unique identifier |
| lesson_id | string | Parent lesson reference |
| title | string | Exercise title |
| difficulty | enum | "beginner" | "intermediate" | "advanced" |
| instructions | string | Step-by-step guide (Markdown) |
| starter_code | string | Path to starter files |
| solution_code | string | Path to solution files |
| verification_steps | string[] | How to confirm success |
| hardware_required | HardwareProfile[] | Required hardware profiles |

### LabAssignment

Major practical project for a week.

| Field | Type | Description |
|-------|------|-------------|
| id | string | Unique identifier |
| week_id | string | Parent week reference |
| title | string | Lab title |
| description | string | Detailed requirements |
| deliverables | string[] | What to submit |
| rubric | RubricItem[] | Grading criteria |
| hardware_required | HardwareProfile[] | Required hardware |
| estimated_hours | integer | Expected completion time |

### HardwareProfile

Student computing setup configuration.

| Field | Type | Description |
|-------|------|-------------|
| id | string | Profile identifier |
| name | string | Display name |
| type | enum | "workstation" | "jetson" | "cloud" |
| specifications | object | Hardware requirements |
| setup_guide_path | string | Path to setup documentation |
| verification_script | string | Path to verification script |

**Instances**:
1. Digital Twin Workstation: 64GB RAM, RTX 4070 Ti+, Ubuntu 22.04
2. Economy Jetson Kit: Orin Nano, RealSense D435i, Mic Array
3. Cloud Ether Lab: Docker-based cloud simulation

### RobotPlatform

Physical robot options for capstone.

| Field | Type | Description |
|-------|------|-------------|
| id | string | Platform identifier |
| name | string | Display name |
| manufacturer | string | "Unitree" |
| type | enum | "quadruped" | "humanoid" |
| model | string | "Go2" | "G1" |
| specifications | object | Technical specs |
| documentation_url | string | Official docs link |
| ros2_package | string | ROS 2 integration package |

## Chatbot Data Models

### DocumentChunk

Indexed content for RAG retrieval.

| Field | Type | Description |
|-------|------|-------------|
| id | string | UUID for chunk |
| content | string | Text content (max 500 tokens) |
| embedding | float[] | 1536-dim vector (text-embedding-3-small) |
| metadata | ChunkMetadata | Source information |

### ChunkMetadata

Metadata for document chunks.

| Field | Type | Description |
|-------|------|-------------|
| source | string | Always "book" for content restriction |
| file_path | string | Original Markdown file path |
| module_number | integer | Module (1-4) or null |
| week_number | integer | Week (1-13) or null |
| section_heading | string | H2/H3 heading text |
| content_type | enum | "lesson" | "exercise" | "lab" | "setup" |

### ChatSession

User chat session (stored in Neon PostgreSQL).

| Field | Type | Description |
|-------|------|-------------|
| id | UUID | Session identifier |
| created_at | timestamp | Session start time |
| last_active | timestamp | Last message time |
| message_count | integer | Number of messages |
| page_context | string | Current page URL (optional) |

### ChatMessage

Individual message in a session.

| Field | Type | Description |
|-------|------|-------------|
| id | UUID | Message identifier |
| session_id | UUID | Parent session reference |
| role | enum | "user" | "assistant" |
| content | string | Message text |
| created_at | timestamp | Message timestamp |
| citations | Citation[] | Source references (assistant only) |
| selected_text | string | User-highlighted text (optional) |

### Citation

Source reference for assistant responses.

| Field | Type | Description |
|-------|------|-------------|
| chunk_id | string | Referenced chunk |
| file_path | string | Source file |
| section_heading | string | Section title |
| relevance_score | float | Similarity score (0-1) |

## Database Schemas

### Qdrant Collection: book_content

```json
{
  "name": "book_content",
  "vectors": {
    "size": 1536,
    "distance": "Cosine"
  },
  "payload_schema": {
    "source": "keyword",
    "file_path": "keyword",
    "module_number": "integer",
    "week_number": "integer",
    "section_heading": "text",
    "content_type": "keyword"
  }
}
```

### Neon PostgreSQL Tables

```sql
CREATE TABLE chat_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMP DEFAULT NOW(),
    last_active TIMESTAMP DEFAULT NOW(),
    message_count INTEGER DEFAULT 0,
    page_context TEXT
);

CREATE TABLE chat_messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID REFERENCES chat_sessions(id) ON DELETE CASCADE,
    role VARCHAR(10) NOT NULL CHECK (role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    selected_text TEXT,
    citations JSONB
);

CREATE INDEX idx_messages_session ON chat_messages(session_id);
CREATE INDEX idx_sessions_active ON chat_sessions(last_active);
```

## Entity Relationships

```
Module (1) ──────< Week (many)
                    │
                    ├────< Lesson (many)
                    │         │
                    │         └────< Exercise (many)
                    │
                    └────< LabAssignment (0-1)

HardwareProfile ────< Exercise (many-to-many)
                 ────< LabAssignment (many-to-many)

DocumentChunk (standalone, derived from all content)

ChatSession (1) ────< ChatMessage (many)
                              │
                              └────< Citation (many)
```

## Content-to-Chunk Mapping

| Content Type | Chunk Strategy |
|--------------|----------------|
| Lesson | Split by H2/H3 headers, then by 500 tokens |
| Exercise | Keep as single chunk if <500 tokens, else split |
| Lab Assignment | Split by section (Requirements, Deliverables, etc.) |
| Code Sample | Include as metadata, not embedded |
| Setup Guide | Split by step/section |
