# Research: Physical AI & Humanoid Robotics Book

**Branch**: `001-physical-ai-robotics-book` | **Date**: 2025-12-24
**Purpose**: Resolve technical decisions and document best practices for implementation

## Technology Decisions

### 1. Docusaurus Version and Configuration

**Decision**: Docusaurus 3.x with TypeScript configuration

**Rationale**:
- Docusaurus 3.x is the latest stable version with React 18 support
- TypeScript provides type safety for custom components (chatbot widget)
- Built-in MDX support enables interactive content within documentation
- Native GitHub Pages deployment support via `docusaurus deploy`

**Alternatives Considered**:
- Docusaurus 2.x: Older, less React 18 features
- GitBook: Less customizable, no embedded React components
- VitePress: Vue-based, would require Vue chatbot instead of React

**Configuration Approach**:
```typescript
// docusaurus.config.ts key settings
{
  url: 'https://<username>.github.io',
  baseUrl: '/book/',
  organizationName: '<username>',
  projectName: 'book',
  deploymentBranch: 'gh-pages',
  trailingSlash: false,
}
```

### 2. Sidebar Structure for 13-Week Curriculum

**Decision**: Category-based sidebar with module groupings containing weekly subfolders

**Rationale**:
- Aligns with course structure (4 modules, 13 weeks)
- `_category_.json` files provide metadata and ordering
- Nested folder structure mirrors the logical curriculum progression
- Docusaurus auto-generates sidebar from folder structure

**Week-to-Module Mapping**:
| Module | Weeks | Topic |
|--------|-------|-------|
| Lab Setup | N/A | Environment configuration |
| Module 1 | 1-3 | ROS 2 Fundamentals |
| Module 2 | 4-6 | Digital Twin / Simulation |
| Module 3 | 7-9 | AI-Robot Brain / Navigation |
| Module 4 | 10-13 | Vision-Language-Action + Capstone |

### 3. RAG Chatbot Architecture

**Decision**: FastAPI backend with OpenAI Agents SDK, Qdrant vector store, Neon PostgreSQL

**Rationale**:
- OpenAI Agents SDK provides built-in tool use and conversation management
- Qdrant offers efficient vector similarity search with metadata filtering
- Neon provides serverless PostgreSQL for chat session persistence
- FastAPI enables async request handling and automatic OpenAPI docs

**Context Restriction Implementation**:
```python
# Only retrieve from indexed book content
results = qdrant_client.search(
    collection_name="book_content",
    query_vector=embedding,
    limit=5,
    query_filter=Filter(must=[
        FieldCondition(key="source", match=MatchValue(value="book"))
    ])
)
```

**Alternatives Considered**:
- LangChain: More complex, overkill for single-source RAG
- Pinecone: More expensive than Qdrant, similar features
- Supabase pgvector: Less specialized than Qdrant for pure vector search

### 4. Document Embedding Strategy

**Decision**: Markdown chunking with 500-token chunks, 50-token overlap, using OpenAI text-embedding-3-small

**Rationale**:
- 500 tokens provides sufficient context per chunk for technical content
- 50-token overlap prevents context loss at chunk boundaries
- text-embedding-3-small balances cost and quality for educational content
- Metadata includes: file path, section heading, week number, module number

**Processing Pipeline**:
1. Parse Markdown files from `docs/` directory
2. Split by headers (H2, H3) first, then by token count
3. Generate embeddings via OpenAI API
4. Store in Qdrant with metadata for filtering

### 5. Chatbot Frontend Integration

**Decision**: React component injected via Docusaurus theme Root wrapper

**Rationale**:
- Root.tsx wrapper ensures chatbot appears on all pages
- Component can access current page context for enhanced responses
- CSS modules prevent style conflicts with Docusaurus theme
- State managed locally with optional session persistence

**Integration Point**:
```tsx
// src/theme/Root.tsx
import ChatbotWidget from '@site/src/components/ChatbotWidget';

export default function Root({children}) {
  return (
    <>
      {children}
      <ChatbotWidget />
    </>
  );
}
```

### 6. ROS 2 Version Strategy

**Decision**: Primary support for ROS 2 Humble (LTS), with Iron compatibility notes

**Rationale**:
- Humble is the current LTS release (support until 2027)
- Iron is the latest stable release but shorter support window
- Most Isaac SDK integrations target Humble
- Version-specific code blocks clearly labeled

**Documentation Approach**:
```markdown
:::info ROS 2 Version
This example uses ROS 2 Humble. For Iron, replace `humble` with `iron` in package names.
:::
```

### 7. Hardware Profile Verification

**Decision**: Shell scripts with clear pass/fail output for each hardware profile

**Rationale**:
- Scripts can be run before any exercises to validate environment
- Clear error messages guide students to resolution
- Different scripts for Workstation, Jetson, and Cloud profiles

**Verification Checks**:
- Workstation: NVIDIA driver, CUDA, ROS 2, Gazebo, Isaac Sim
- Jetson: JetPack version, ROS 2, RealSense SDK
- Cloud: Docker, ROS 2 container, GPU passthrough

### 8. Code Sample Organization

**Decision**: Static folder with module subfolders, symlinked in documentation

**Rationale**:
- Code samples remain executable and testable independently
- Documentation references code via relative paths
- CI can run tests against code samples directly
- Students can download complete sample sets

**File Naming Convention**:
```
static/code-samples/
├── module-1/
│   ├── week-01-publisher.py
│   ├── week-01-subscriber.py
│   └── week-02-service.py
```

## Best Practices Applied

### Docusaurus for Technical Books
1. Use versioned docs if planning future editions
2. Implement search with Algolia DocSearch
3. Add code block copy buttons
4. Use admonitions (:::tip, :::warning) for callouts
5. Include edit-this-page links for community contributions

### RAG for Educational Content
1. Include section headings in chunk metadata for citation
2. Implement relevance score threshold to avoid hallucination
3. Add "I don't know" fallback for out-of-scope queries
4. Log queries for content improvement insights
5. Support user-selected text as additional context

### ROS 2 Documentation
1. Always specify full package names
2. Include terminal output examples
3. Provide URDF/launch file downloads
4. Test on clean Ubuntu 22.04 installation
5. Document common error messages and solutions

## Open Questions Resolved

| Question | Resolution |
|----------|------------|
| Which Docusaurus version? | 3.x with TypeScript |
| How to structure 13 weeks? | Module folders containing week subfolders |
| Vector DB choice? | Qdrant (self-hosted or cloud) |
| Embedding model? | OpenAI text-embedding-3-small |
| How to restrict chatbot to book? | Qdrant collection filter + prompt engineering |
| ROS 2 version support? | Humble primary, Iron compatibility notes |
