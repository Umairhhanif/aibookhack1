# Tasks: Physical AI & Humanoid Robotics Book

**Input**: Design documents from `/specs/001-physical-ai-robotics-book/`
**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/

**Tests**: Tests are NOT explicitly requested in the specification. Test tasks are omitted.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

- **Frontend**: `docs/`, `src/`, `static/` at repository root (Docusaurus)
- **Backend**: `chatbot/src/` for RAG chatbot

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and basic structure

- [x] T001 Initialize Docusaurus 3.x project with TypeScript in repository root
- [x] T002 Configure docusaurus.config.ts with project metadata and GitHub Pages settings
- [x] T003 [P] Create .env.example with frontend environment variables
- [x] T004 [P] Create README.md with project overview and quickstart link
- [x] T005 [P] Initialize chatbot Python project in chatbot/ directory
- [x] T006 [P] Create chatbot/requirements.txt with FastAPI, OpenAI, Qdrant, psycopg2 dependencies
- [x] T007 [P] Create chatbot/.env.example with backend environment variables
- [x] T008 Configure sidebars.ts with 13-week module structure per plan.md

**Checkpoint**: Project structure ready, both frontend and backend initialized

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [x] T009 Create docs/_category_.json files for all module directories
- [x] T010 [P] Create src/css/custom.css with base theme customization
- [x] T011 [P] Create static/img/ directory structure for images
- [x] T012 [P] Create static/code-samples/ directory structure per plan.md
- [x] T013 Create docs/intro.md with course introduction and overview
- [x] T014 [P] Create chatbot/src/__init__.py with package initialization
- [x] T015 [P] Create chatbot/src/api/__init__.py with API package initialization
- [x] T016 [P] Create chatbot/src/models/__init__.py with models package initialization
- [x] T017 [P] Create chatbot/src/db/__init__.py with database package initialization
- [x] T018 Create chatbot/src/db/neon.py with PostgreSQL connection utilities
- [x] T019 [P] Create chatbot/src/db/qdrant.py with Qdrant connection utilities
- [x] T020 Create chatbot/src/models/chat.py with ChatRequest and ChatResponse Pydantic models
- [x] T021 [P] Create chatbot/src/models/document.py with DocumentChunk and ChunkMetadata models
- [x] T022 Create chatbot/src/api/main.py with FastAPI app initialization
- [x] T023 [P] Create chatbot/src/api/middleware/cors.py with CORS configuration
- [x] T024 Create chatbot/src/api/routes/health.py with health check endpoint

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel

---

## Phase 3: User Story 1 - Complete Environment Setup (Priority: P1) 🎯 MVP

**Goal**: Students can set up their development environment following Lab Setup guides for Workstation, Jetson, or Cloud profiles

**Independent Test**: A new student follows the Lab Setup guide and successfully runs a verification script on their hardware

### Implementation for User Story 1

- [x] T025 [US1] Create docs/lab-setup/_category_.json with Lab Setup category metadata
- [x] T026 [P] [US1] Create docs/lab-setup/overview.md with Lab Setup introduction and hardware profile selection
- [x] T027 [P] [US1] Create docs/lab-setup/digital-twin-workstation.md with Ubuntu 22.04 workstation setup guide
- [x] T028 [P] [US1] Create docs/lab-setup/cloud-ether-lab.md with cloud-based simulation alternatives
- [x] T029 [P] [US1] Create docs/lab-setup/economy-jetson-kit.md with Jetson Orin Nano edge kit assembly guide
- [x] T030 [US1] Create static/code-samples/lab-setup/verify-workstation.sh with environment verification script
- [x] T031 [P] [US1] Create static/code-samples/lab-setup/verify-jetson.sh with Jetson verification script
- [x] T032 [P] [US1] Create static/code-samples/lab-setup/verify-cloud.sh with cloud environment verification script

**Checkpoint**: User Story 1 complete - students can set up any hardware profile

---

## Phase 4: User Story 2 - Navigate Weekly Curriculum (Priority: P1) 🎯 MVP

**Goal**: Students can navigate the 13-week curriculum via organized sidebar with module and week structure

**Independent Test**: Navigate to any week's content and verify complete, organized learning materials

### Implementation for User Story 2

- [x] T033 [US2] Create docs/module-1-robotic-nervous-system/_category_.json with Module 1 metadata
- [x] T034 [P] [US2] Create docs/module-1-robotic-nervous-system/week-01/_category_.json for Week 1
- [x] T035 [P] [US2] Create docs/module-1-robotic-nervous-system/week-02/_category_.json for Week 2
- [x] T036 [P] [US2] Create docs/module-1-robotic-nervous-system/week-03/_category_.json for Week 3
- [x] T037 [US2] Create docs/module-2-digital-twin/_category_.json with Module 2 metadata
- [x] T038 [P] [US2] Create docs/module-2-digital-twin/week-04/_category_.json for Week 4
- [x] T039 [P] [US2] Create docs/module-2-digital-twin/week-05/_category_.json for Week 5
- [x] T040 [P] [US2] Create docs/module-2-digital-twin/week-06/_category_.json for Week 6
- [x] T041 [US2] Create docs/module-3-ai-robot-brain/_category_.json with Module 3 metadata
- [x] T042 [P] [US2] Create docs/module-3-ai-robot-brain/week-07/_category_.json for Week 7
- [x] T043 [P] [US2] Create docs/module-3-ai-robot-brain/week-08/_category_.json for Week 8
- [x] T044 [P] [US2] Create docs/module-3-ai-robot-brain/week-09/_category_.json for Week 9
- [x] T045 [US2] Create docs/module-4-vision-language-action/_category_.json with Module 4 metadata
- [x] T046 [P] [US2] Create docs/module-4-vision-language-action/week-10/_category_.json for Week 10
- [x] T047 [P] [US2] Create docs/module-4-vision-language-action/week-11/_category_.json for Week 11
- [x] T048 [P] [US2] Create docs/module-4-vision-language-action/week-12/_category_.json for Week 12
- [x] T049 [P] [US2] Create docs/capstone/week-13/_category_.json for Week 13 Capstone
- [x] T050 [US2] Create docs/capstone/_category_.json with Capstone category metadata

**Checkpoint**: User Story 2 complete - full sidebar navigation structure in place

---

## Phase 5: User Story 3 - Learn ROS 2 Fundamentals (Priority: P2)

**Goal**: Students learn ROS 2 nodes, topics, and services through Module 1 content and exercises

**Independent Test**: Student completes Module 1 and creates a custom ROS 2 node with topics and services

### Implementation for User Story 3

- [ ] T051 [US3] Create docs/module-1-robotic-nervous-system/week-01/introduction.md with ROS 2 overview
- [ ] T052 [P] [US3] Create docs/module-1-robotic-nervous-system/week-01/nodes.md with ROS 2 nodes tutorial
- [ ] T053 [P] [US3] Create docs/module-1-robotic-nervous-system/week-01/exercise-first-node.md with first node exercise
- [ ] T054 [US3] Create docs/module-1-robotic-nervous-system/week-02/topics.md with ROS 2 topics tutorial
- [ ] T055 [P] [US3] Create docs/module-1-robotic-nervous-system/week-02/publishers-subscribers.md with pub/sub patterns
- [ ] T056 [P] [US3] Create docs/module-1-robotic-nervous-system/week-02/exercise-pub-sub.md with publisher/subscriber exercise
- [ ] T057 [US3] Create docs/module-1-robotic-nervous-system/week-03/services.md with ROS 2 services tutorial
- [ ] T058 [P] [US3] Create docs/module-1-robotic-nervous-system/week-03/actions.md with ROS 2 actions tutorial
- [ ] T059 [P] [US3] Create docs/module-1-robotic-nervous-system/week-03/exercise-service.md with service implementation exercise
- [ ] T060 [US3] Create static/code-samples/module-1/week-01-publisher.py with sample publisher node
- [ ] T061 [P] [US3] Create static/code-samples/module-1/week-01-subscriber.py with sample subscriber node
- [ ] T062 [P] [US3] Create static/code-samples/module-1/week-02-service-server.py with sample service server
- [ ] T063 [P] [US3] Create static/code-samples/module-1/week-02-service-client.py with sample service client

**Checkpoint**: User Story 3 complete - Module 1 ROS 2 fundamentals ready

---

## Phase 6: User Story 4 - Build Digital Twin Simulations (Priority: P2)

**Goal**: Students create robot simulations in Gazebo and Unity following Module 2 content

**Independent Test**: Student creates a Gazebo simulation with URDF robot responding to ROS 2 commands

### Implementation for User Story 4

- [ ] T064 [US4] Create docs/module-2-digital-twin/week-04/gazebo-intro.md with Gazebo introduction
- [ ] T065 [P] [US4] Create docs/module-2-digital-twin/week-04/urdf-basics.md with URDF robot modeling tutorial
- [ ] T066 [P] [US4] Create docs/module-2-digital-twin/week-04/exercise-urdf.md with URDF creation exercise
- [ ] T067 [US4] Create docs/module-2-digital-twin/week-05/gazebo-physics.md with physics simulation tutorial
- [ ] T068 [P] [US4] Create docs/module-2-digital-twin/week-05/sensors.md with sensor simulation guide
- [ ] T069 [P] [US4] Create docs/module-2-digital-twin/week-05/exercise-simulation.md with complete simulation exercise
- [ ] T070 [US4] Create docs/module-2-digital-twin/week-06/unity-integration.md with Unity rendering setup
- [ ] T071 [P] [US4] Create docs/module-2-digital-twin/week-06/isaac-sim-intro.md with Isaac Sim introduction
- [ ] T072 [P] [US4] Create docs/module-2-digital-twin/week-06/exercise-digital-twin.md with digital twin exercise
- [ ] T073 [US4] Create static/code-samples/module-2/simple-robot.urdf with sample URDF model
- [ ] T074 [P] [US4] Create static/code-samples/module-2/gazebo-launch.py with Gazebo launch file
- [ ] T075 [P] [US4] Create static/code-samples/module-2/sensor-config.yaml with sensor configuration

**Checkpoint**: User Story 4 complete - Module 2 digital twin simulation ready

---

## Phase 7: User Story 5 - Implement AI-Powered Navigation (Priority: P3)

**Goal**: Students implement VSLAM, Nav2, and Isaac SDK features following Module 3 content

**Independent Test**: Student implements a navigation stack with autonomous waypoint navigation in simulation

### Implementation for User Story 5

- [ ] T076 [US5] Create docs/module-3-ai-robot-brain/week-07/vslam-intro.md with VSLAM introduction
- [ ] T077 [P] [US5] Create docs/module-3-ai-robot-brain/week-07/mapping.md with mapping tutorial
- [ ] T078 [P] [US5] Create docs/module-3-ai-robot-brain/week-07/exercise-mapping.md with mapping exercise
- [ ] T079 [US5] Create docs/module-3-ai-robot-brain/week-08/nav2-intro.md with Nav2 introduction
- [ ] T080 [P] [US5] Create docs/module-3-ai-robot-brain/week-08/path-planning.md with path planning tutorial
- [ ] T081 [P] [US5] Create docs/module-3-ai-robot-brain/week-08/exercise-navigation.md with navigation exercise
- [ ] T082 [US5] Create docs/module-3-ai-robot-brain/week-09/isaac-sdk.md with Isaac SDK overview
- [ ] T083 [P] [US5] Create docs/module-3-ai-robot-brain/week-09/perception.md with perception integration
- [ ] T084 [P] [US5] Create docs/module-3-ai-robot-brain/week-09/exercise-ai-robot.md with AI robot brain exercise
- [ ] T085 [US5] Create static/code-samples/module-3/nav2-params.yaml with Nav2 configuration
- [ ] T086 [P] [US5] Create static/code-samples/module-3/navigation-launch.py with navigation launch file
- [ ] T087 [P] [US5] Create static/code-samples/module-3/waypoint-follower.py with waypoint navigation script

**Checkpoint**: User Story 5 complete - Module 3 AI navigation ready

---

## Phase 8: User Story 6 - Complete Vision-Language-Action Capstone (Priority: P3)

**Goal**: Students integrate Whisper, VLA models, and robotics for the capstone project

**Independent Test**: Student demonstrates robot responding to voice commands and executing multi-step tasks

### Implementation for User Story 6

- [ ] T088 [US6] Create docs/module-4-vision-language-action/week-10/whisper-integration.md with Whisper voice command tutorial
- [ ] T089 [P] [US6] Create docs/module-4-vision-language-action/week-10/voice-commands.md with voice command processing guide
- [ ] T090 [P] [US6] Create docs/module-4-vision-language-action/week-10/exercise-voice.md with voice integration exercise
- [ ] T091 [US6] Create docs/module-4-vision-language-action/week-11/vla-models.md with VLA models introduction
- [ ] T092 [P] [US6] Create docs/module-4-vision-language-action/week-11/embodied-ai.md with embodied AI concepts
- [ ] T093 [P] [US6] Create docs/module-4-vision-language-action/week-11/exercise-vla.md with VLA integration exercise
- [ ] T094 [US6] Create docs/module-4-vision-language-action/week-12/llm-robotics.md with LLM-robotics integration
- [ ] T095 [P] [US6] Create docs/module-4-vision-language-action/week-12/task-planning.md with natural language task planning
- [ ] T096 [P] [US6] Create docs/module-4-vision-language-action/week-12/exercise-llm.md with LLM integration exercise
- [ ] T097 [US6] Create docs/module-4-vision-language-action/week-13/capstone-overview.md with capstone project overview
- [ ] T098 [P] [US6] Create docs/module-4-vision-language-action/week-13/unitree-deployment.md with Unitree Go2/G1 deployment guide
- [ ] T099 [US6] Create docs/capstone/autonomous-humanoid.md with "The Autonomous Humanoid" project guide
- [ ] T100 [US6] Create static/code-samples/module-4/whisper-node.py with Whisper ROS 2 integration
- [ ] T101 [P] [US6] Create static/code-samples/module-4/vla-agent.py with VLA agent implementation
- [ ] T102 [P] [US6] Create static/code-samples/module-4/capstone-main.py with capstone main entry point

**Checkpoint**: User Story 6 complete - Module 4 and Capstone project ready

---

## Phase 9: User Story 7 - Access Book via RAG Chatbot (Priority: P3)

**Goal**: Students can ask questions and receive answers exclusively from book content via embedded chatbot

**Independent Test**: Ask chatbot book-specific questions and verify responses cite book sections

### Implementation for User Story 7

- [ ] T103 [US7] Create chatbot/src/embeddings/__init__.py with embeddings package initialization
- [ ] T104 [P] [US7] Create chatbot/src/embeddings/processor.py with Markdown chunking and embedding logic
- [ ] T105 [P] [US7] Create chatbot/src/embeddings/indexer.py with Qdrant index management
- [ ] T106 [US7] Create chatbot/src/agents/__init__.py with agents package initialization
- [ ] T107 [US7] Create chatbot/src/agents/book_agent.py with OpenAI Agents SDK book-restricted agent
- [ ] T108 [US7] Create chatbot/src/api/routes/chat.py with chat API endpoints per contracts/chatbot-api.yaml
- [ ] T109 [US7] Create chatbot/scripts/index_docs.py with document indexing CLI script
- [ ] T110 [P] [US7] Create chatbot/scripts/verify_setup.py with environment verification script
- [ ] T111 [US7] Create src/components/ChatbotWidget/index.tsx with ChatbotWidget component export
- [ ] T112 [US7] Create src/components/ChatbotWidget/ChatbotWidget.tsx with React chatbot UI
- [ ] T113 [P] [US7] Create src/components/ChatbotWidget/styles.module.css with chatbot styling
- [ ] T114 [US7] Create src/theme/Root.tsx with chatbot injection wrapper
- [ ] T115 [US7] Create chatbot/Dockerfile with backend container configuration
- [ ] T116 [P] [US7] Create chatbot/docker-compose.yml with local development stack

**Checkpoint**: User Story 7 complete - RAG chatbot fully integrated

---

## Phase 10: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories and deployment

- [ ] T117 [P] Create .github/workflows/deploy-docs.yml with GitHub Pages deployment workflow
- [ ] T118 [P] Create .github/workflows/deploy-chatbot.yml with chatbot backend deployment workflow
- [ ] T119 Verify Docusaurus build succeeds without errors
- [ ] T120 [P] Verify all code samples are syntactically correct
- [ ] T121 Run chatbot/scripts/index_docs.py to populate Qdrant with book content
- [ ] T122 Verify chatbot responds correctly to sample queries
- [ ] T123 [P] Add version-specific ROS 2 admonitions to all exercises
- [ ] T124 Final review of sidebar navigation and link integrity

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Story 1 (Phase 3)**: Depends on Foundational - Lab Setup content
- **User Story 2 (Phase 4)**: Depends on Foundational - Navigation structure
- **User Stories 3-6 (Phases 5-8)**: Depend on US1 + US2 for structure, content is independent
- **User Story 7 (Phase 9)**: Depends on Foundational backend + some content to index
- **Polish (Phase 10)**: Depends on all user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Foundational - No dependencies on other stories
- **User Story 2 (P1)**: Can start after Foundational - No dependencies on other stories
- **User Story 3 (P2)**: Can start after US2 structure exists - Content is independent
- **User Story 4 (P2)**: Can start after US2 structure exists - Content is independent
- **User Story 5 (P3)**: Can start after US2 structure exists - Content is independent
- **User Story 6 (P3)**: Can start after US2 structure exists - Content is independent
- **User Story 7 (P3)**: Can start after Foundational - Needs content to index (can use partial)

### Within Each User Story

- Category JSON files before content markdown
- Introduction/overview files before detailed tutorials
- Tutorials before exercises
- Code samples can be parallel with documentation

### Parallel Opportunities

- All Setup tasks marked [P] can run in parallel
- All Foundational tasks marked [P] can run in parallel (within Phase 2)
- Once Foundational phase completes, US1 and US2 can run in parallel
- Once US2 structure exists, US3-US6 can ALL run in parallel
- All code samples within a story marked [P] can run in parallel
- Backend and frontend work for US7 can mostly run in parallel

---

## Parallel Example: User Story 3

```bash
# Launch all week category files together:
Task: "Create docs/module-1-robotic-nervous-system/week-01/_category_.json for Week 1"
Task: "Create docs/module-1-robotic-nervous-system/week-02/_category_.json for Week 2"
Task: "Create docs/module-1-robotic-nervous-system/week-03/_category_.json for Week 3"

# Launch all code samples together:
Task: "Create static/code-samples/module-1/week-01-publisher.py"
Task: "Create static/code-samples/module-1/week-01-subscriber.py"
Task: "Create static/code-samples/module-1/week-02-service-server.py"
Task: "Create static/code-samples/module-1/week-02-service-client.py"
```

---

## Implementation Strategy

### MVP First (User Stories 1 + 2 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (CRITICAL - blocks all stories)
3. Complete Phase 3: User Story 1 (Lab Setup)
4. Complete Phase 4: User Story 2 (Navigation Structure)
5. **STOP and VALIDATE**: Site builds and navigates correctly
6. Deploy to GitHub Pages for early feedback

### Incremental Delivery

1. Complete Setup + Foundational → Foundation ready
2. Add US1 + US2 → Deploy (MVP with structure)
3. Add US3 (Module 1) → Deploy (ROS 2 content)
4. Add US4 (Module 2) → Deploy (Simulation content)
5. Add US5 (Module 3) → Deploy (AI/Navigation content)
6. Add US6 (Module 4 + Capstone) → Deploy (VLA content)
7. Add US7 (RAG Chatbot) → Deploy (Interactive features)

### Parallel Team Strategy

With multiple developers:

1. Team completes Setup + Foundational together
2. Once Foundational is done:
   - Developer A: User Story 1 (Lab Setup) + User Story 3 (Module 1)
   - Developer B: User Story 2 (Navigation) + User Story 4 (Module 2)
   - Developer C: User Story 5 (Module 3) + User Story 6 (Module 4)
   - Developer D: User Story 7 (RAG Chatbot backend + frontend)
3. Stories complete and integrate independently

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- Each user story should be independently completable and testable
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
- Avoid: vague tasks, same file conflicts, cross-story dependencies that break independence
- Content modules (US3-US6) are embarrassingly parallel once structure exists
