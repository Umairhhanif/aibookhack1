# Physical AI & Humanoid Robotics: AI Systems in the Physical World

A comprehensive educational book teaching embodied intelligence—bridging Python agents to ROS 2 controllers—across a 13-week curriculum.

## Overview

This course covers the complete stack for building AI-powered robots:

- **Module 1: The Robotic Nervous System** - ROS 2 fundamentals (Weeks 1-3)
- **Module 2: The Digital Twin** - Gazebo, Unity, Isaac Sim (Weeks 4-6)
- **Module 3: The AI-Robot Brain** - VSLAM, Nav2, Isaac SDK (Weeks 7-9)
- **Module 4: Vision-Language-Action** - LLMs + Robotics Capstone (Weeks 10-13)

## Quick Start

### Prerequisites

- Node.js 20.x LTS
- pnpm 8.x+
- Python 3.11+
- Docker 24.x+

### Installation

```bash
# Install frontend dependencies
pnpm install

# Install chatbot backend
cd chatbot
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Development

```bash
# Start documentation site
pnpm start

# Start chatbot backend (in another terminal)
cd chatbot
uvicorn src.api.main:app --reload --port 8000
```

### Build

```bash
pnpm build
```

## Hardware Requirements

### Digital Twin Workstation
- 64GB RAM
- NVIDIA RTX 4070 Ti+ (12GB VRAM)
- Ubuntu 22.04

### Economy Jetson Student Kit
- NVIDIA Jetson Orin Nano
- Intel RealSense D435i
- Microphone Array

### Cloud Ether Lab
- Docker-based cloud simulation (instructions in Lab Setup)

## Documentation

See the [full documentation](./docs/intro.md) and [developer quickstart](./specs/001-physical-ai-robotics-book/quickstart.md).

## License

MIT
