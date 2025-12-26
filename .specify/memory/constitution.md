<!--
  Sync Impact Report
  ===================
  Version change: 0.0.0 → 1.0.0 (MAJOR - initial constitution ratification)

  Added sections:
    - I. Accuracy First (new principle)
    - II. Clarity for Engineers (new principle)
    - III. Spec-Kit Plus as Source of Truth (new principle)
    - IV. Reproducible End-to-End Workflows (new principle)
    - V. Traceability and Integrity (new principle)
    - VI. Concise Instructional Writing (new principle)
    - Technical Constraints (new section)
    - Success Criteria (new section)
    - Governance (new section)

  Removed sections: N/A (initial version)
  Modified principles: N/A (initial version)

  Templates validated:
    ✅ .specify/templates/plan-template.md - Constitution Check section compatible
    ✅ .specify/templates/spec-template.md - Requirements alignment compatible
    ✅ .specify/templates/tasks-template.md - Task categorization compatible

  Deferred TODOs: None
-->

# AI/Spec-Driven Book with Embedded RAG Chatbot Constitution

## Core Principles

### I. Accuracy First

All content MUST be accurate and derived from official documentation, specifications, and verified sources.
Agents and contributors MUST NOT fabricate information or rely on assumptions when official sources exist.
Every technical claim requires external verification before inclusion.

**Rationale**: A technical book for software engineers loses all value if inaccurate. Trust is earned through verifiable correctness.

### II. Clarity for Engineers

All writing MUST prioritize clarity for software engineers as the primary audience.
Technical explanations MUST be precise, unambiguous, and actionable.
Complex concepts MUST be broken down into digestible steps with concrete examples.

**Rationale**: Engineers value time and precision. Unclear documentation wastes both and erodes confidence in the material.

### III. Spec-Kit Plus as Source of Truth

Spec-Kit Plus artifacts (specs, plans, tasks) are the authoritative source for all implementation decisions.
All feature work MUST begin with a specification before any code is written.
Deviations from specs require explicit documentation and approval.

**Rationale**: Specification-driven development ensures alignment between intent and implementation while creating traceable audit trails.

### IV. Reproducible End-to-End Workflows

Every workflow described in the book MUST be reproducible from start to finish.
All commands, configurations, and procedures MUST be tested and verified.
Environment setup instructions MUST be complete and self-contained.

**Rationale**: Readers who cannot reproduce the examples cannot learn from them. Reproducibility is non-negotiable for instructional content.

### V. Traceability and Integrity

All claims MUST be traceable to sources or code references.
Zero tolerance for plagiarism—all external content MUST be properly attributed.
Every modification MUST be logged via Prompt History Records (PHRs).

**Rationale**: Traceability enables verification, attribution protects intellectual property, and audit trails support quality assurance.

### VI. Concise Instructional Writing

Writing MUST be concise, direct, and instructional in tone.
Avoid unnecessary verbosity, filler content, or tangential discussions.
Every paragraph MUST advance the reader's understanding or capability.

**Rationale**: Engineers prefer efficiency. Dense, actionable content respects the reader's time and expertise.

## Technical Constraints

The following technical constraints are binding for this project:

- **Documentation Platform**: Docusaurus deployed to GitHub Pages
- **Development Tooling**: Claude Code integrated with Spec-Kit Plus workflows
- **RAG Chatbot Stack**:
  - Orchestration: OpenAI Agents SDK / ChatKit
  - Backend API: FastAPI
  - Relational Database: Neon (PostgreSQL)
  - Vector Database: Qdrant
- **Chatbot Context Restriction**: MUST answer only from book content and user-selected text; no external knowledge injection

## Success Criteria

The project is considered successful when ALL of the following are achieved:

1. **Successful Build and Deployment**: Docusaurus site builds without errors and deploys to GitHub Pages
2. **Fully Functional RAG Chatbot**: Embedded chatbot answers user queries using only book content and user-selected context
3. **Reproducible and Verified**: All workflows in the book can be executed by a reader and produce expected results

## Governance

This constitution supersedes all other project practices and guidelines.

**Amendment Process**:
1. Proposed amendments MUST be documented with rationale
2. Amendments require explicit approval before adoption
3. All amendments MUST include a migration plan for affected artifacts
4. Version history MUST be maintained

**Versioning Policy**:
- MAJOR: Backward-incompatible principle changes or removals
- MINOR: New principles added or existing guidance materially expanded
- PATCH: Clarifications, wording improvements, non-semantic refinements

**Compliance**:
- All PRs and reviews MUST verify alignment with constitutional principles
- Complexity beyond what is required MUST be explicitly justified
- Use `.specify/memory/constitution.md` as the authoritative reference

**Version**: 1.0.0 | **Ratified**: 2025-12-24 | **Last Amended**: 2025-12-24
