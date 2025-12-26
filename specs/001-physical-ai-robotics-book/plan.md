# Implementation Plan: Physical AI & Humanoid Robotics Book

**Branch**: `001-physical-ai-robotics-book` | **Date**: 2025-12-24 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/001-physical-ai-robotics-book/spec.md`

## Summary

Build a comprehensive Docusaurus-based educational book titled "Physical AI & Humanoid Robotics: AI Systems in the Physical World" targeting advanced students. The book covers embodied intelligence from Python agents to ROS 2 controllers across 4 modules and 13 weeks, with an embedded RAG chatbot that answers questions exclusively from book content.

**Technical Approach**: Static site generation with Docusaurus 3.x deployed to GitHub Pages, combined with a RAG chatbot backend using FastAPI, Qdrant for vector search, and Neon (PostgreSQL) for session/user data. The chatbot integrates via a React component embedded in the Docusaurus theme.

## Technical Context

**Language/Version**: TypeScript 5.x (Docusaurus/React), Python 3.11 (RAG backend)
**Primary Dependencies**: Docusaurus 3.x, React 18, FastAPI, OpenAI Agents SDK, Qdrant, Neon (PostgreSQL)
**Storage**: Qdrant (vector embeddings), Neon PostgreSQL (chat sessions, user preferences)
**Testing**: Jest (frontend), pytest (backend)
**Target Platform**: GitHub Pages (static site), Cloud deployment (RAG backend)
**Project Type**: Web application (frontend + backend)
**Performance Goals**: <5s chatbot response time, <2s page load, 95% code sample success rate
**Constraints**: Chatbot restricted to book content only, no external knowledge
**Scale/Scope**: 13 weeks of content, 4 modules, ~50+ documentation pages, embedded chatbot widget

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Gate | Status |
|-----------|------|--------|
| I. Accuracy First | All technical claims from official ROS 2, Gazebo, Isaac docs | ✅ PASS |
| II. Clarity for Engineers | Content written for advanced students with Python experience | ✅ PASS |
| III. Spec-Kit Plus as Source of Truth | Spec created before implementation | ✅ PASS |
| IV. Reproducible Workflows | All exercises tested on specified hardware configs | ✅ PASS |
| V. Traceability and Integrity | PHRs for all changes, source attribution required | ✅ PASS |
| VI. Concise Instructional Writing | Dense, actionable content structure planned | ✅ PASS |

**Technical Constraints Alignment**:
- ✅ Docusaurus deployed to GitHub Pages
- ✅ RAG Chatbot: OpenAI Agents SDK, FastAPI, Neon, Qdrant
- ✅ Chatbot restricted to book content only

## Project Structure

### Documentation (this feature)

```text
specs/001-physical-ai-robotics-book/
├── plan.md              # This file
├── spec.md              # Feature specification
├── research.md          # Phase 0 research findings
├── data-model.md        # Content and chatbot data models
├── quickstart.md        # Developer setup guide
├── contracts/           # API contracts
│   ├── chatbot-api.yaml # RAG chatbot API (OpenAPI)
│   └── embedding-pipeline.md # Document processing pipeline
├── checklists/
│   └── requirements.md  # Spec validation checklist
└── tasks.md             # Implementation tasks (via /sp.tasks)
```

### Source Code (repository root)

```text
# Docusaurus Frontend
docs/
├── intro.md                    # Course introduction
├── lab-setup/
│   ├── _category_.json
│   ├── overview.md
│   ├── digital-twin-workstation.md
│   ├── cloud-ether-lab.md
│   └── economy-jetson-kit.md
├── module-1-robotic-nervous-system/
│   ├── _category_.json
│   ├── week-01/
│   ├── week-02/
│   └── week-03/
├── module-2-digital-twin/
│   ├── _category_.json
│   ├── week-04/
│   ├── week-05/
│   └── week-06/
├── module-3-ai-robot-brain/
│   ├── _category_.json
│   ├── week-07/
│   ├── week-08/
│   └── week-09/
├── module-4-vision-language-action/
│   ├── _category_.json
│   ├── week-10/
│   ├── week-11/
│   ├── week-12/
│   └── week-13/
└── capstone/
    ├── _category_.json
    └── autonomous-humanoid.md

src/
├── components/
│   └── ChatbotWidget/
│       ├── index.tsx
│       ├── ChatbotWidget.tsx
│       └── styles.module.css
├── css/
│   └── custom.css
└── theme/
    └── Root.tsx              # Chatbot injection point

static/
├── img/
└── code-samples/
    ├── module-1/
    ├── module-2/
    ├── module-3/
    └── module-4/

# RAG Chatbot Backend
chatbot/
├── src/
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py          # FastAPI app
│   │   ├── routes/
│   │   │   ├── chat.py
│   │   │   └── health.py
│   │   └── middleware/
│   │       └── cors.py
│   ├── agents/
│   │   ├── __init__.py
│   │   └── book_agent.py    # OpenAI Agents SDK integration
│   ├── embeddings/
│   │   ├── __init__.py
│   │   ├── processor.py     # Document chunking/embedding
│   │   └── indexer.py       # Qdrant index management
│   ├── models/
│   │   ├── __init__.py
│   │   ├── chat.py          # Chat request/response models
│   │   └── document.py      # Document chunk models
│   └── db/
│       ├── __init__.py
│       ├── neon.py          # PostgreSQL connection
│       └── qdrant.py        # Qdrant connection
├── scripts/
│   ├── index_docs.py        # Index book content to Qdrant
│   └── verify_setup.py      # Environment verification
├── tests/
│   ├── conftest.py
│   ├── test_api.py
│   └── test_embeddings.py
├── requirements.txt
├── Dockerfile
└── docker-compose.yml

# Configuration
docusaurus.config.ts
sidebars.ts
package.json
tsconfig.json

# CI/CD
.github/
└── workflows/
    ├── deploy-docs.yml      # GitHub Pages deployment
    └── deploy-chatbot.yml   # Chatbot backend deployment

# Development
.env.example
README.md
```

**Structure Decision**: Web application structure with Docusaurus frontend deployed to GitHub Pages and a separate FastAPI backend for the RAG chatbot. The backend requires cloud hosting (Railway, Render, or similar) while the frontend is fully static.

## Complexity Tracking

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| Separate backend service | RAG chatbot requires server-side vector search and LLM calls | Client-side embedding is too slow and exposes API keys |
| Qdrant + Neon dual DB | Vector search (Qdrant) + relational data (sessions) have different query patterns | Single DB would require inefficient vector search or complex setup |

## Phase Summary

### Phase 1: Setup & Architecture
1. Initialize Docusaurus 3.x project with TypeScript
2. Configure `docusaurus.config.ts` with project metadata and GitHub Pages settings
3. Define sidebar structure based on 13-week breakdown in `sidebars.ts`
4. Set up basic theme customization

### Phase 2: Hardware & Lab Guide (Critical)
1. Create Lab Setup section with three hardware profiles:
   - Digital Twin Workstation (64GB RAM, RTX 4070 Ti+, Ubuntu 22.04)
   - Cloud Ether Lab (cloud-based simulation alternatives)
   - Economy Jetson Student Kit (Orin Nano, RealSense D435i, Mic Array)
2. Document environment verification scripts for each profile

### Phase 3: Core Syllabus Content
1. Module 1 (Weeks 1-3): ROS 2 Nodes, Topics, Services with rclpy
2. Module 2 (Weeks 4-6): Gazebo physics, Unity rendering, Isaac Sim intro
3. Module 3 (Weeks 7-9): NVIDIA Isaac SDK, VSLAM, Nav2 navigation
4. Module 4 (Weeks 10-13): OpenAI Whisper, VLA models, LLM integration

### Phase 4: Capstone Project Guide
1. Document "The Autonomous Humanoid" capstone project
2. Voice command processing with Whisper
3. VLA integration for physical action execution
4. Deployment guides for Unitree Go2/G1

### Phase 5: RAG Chatbot Implementation
1. Set up FastAPI backend with OpenAI Agents SDK
2. Implement document embedding pipeline (Markdown → chunks → Qdrant)
3. Create chat API endpoints with book-only context restriction
4. Build React chatbot widget component

### Phase 6: Deployment
1. GitHub Actions workflow for Docusaurus → GitHub Pages
2. GitHub Actions workflow for chatbot backend deployment
3. Environment configuration and secrets management
