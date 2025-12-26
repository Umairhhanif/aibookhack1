---
sidebar_position: 1
---

# Lab Setup Overview

Before diving into the curriculum, you need to set up your development environment. This guide covers three hardware profiles—choose the one that best matches your resources.

## Choose Your Hardware Profile

| Profile | Best For | Cost | Performance |
|---------|----------|------|-------------|
| **Digital Twin Workstation** | Full simulation + Isaac Sim | $$$ | Maximum |
| **Economy Jetson Kit** | Edge deployment + real sensors | $$ | Good |
| **Cloud Ether Lab** | No hardware purchase | $ (usage-based) | Variable |

## Quick Comparison

### Digital Twin Workstation

The full-power option for running NVIDIA Isaac Sim, Gazebo, and Unity simultaneously.

**Requirements:**
- 64GB RAM minimum
- NVIDIA RTX 4070 Ti or better (12GB VRAM minimum)
- Ubuntu 22.04 LTS
- 500GB+ SSD

**Best for:** Students with access to high-end workstations or research labs.

→ [Set up Digital Twin Workstation](./digital-twin-workstation)

### Economy Jetson Student Kit

A cost-effective edge computing solution for real sensor integration and deployment.

**Components:**
- NVIDIA Jetson Orin Nano Developer Kit
- Intel RealSense D435i Depth Camera
- USB Microphone Array
- Power supply and cables

**Best for:** Hands-on learning with physical sensors and edge AI.

→ [Set up Economy Jetson Kit](./economy-jetson-kit)

### Cloud Ether Lab

No local hardware required—run simulations in the cloud.

**Requirements:**
- Stable internet connection
- Web browser
- Docker (for local container testing)

**Best for:** Students without GPU hardware or for initial exploration.

→ [Set up Cloud Ether Lab](./cloud-ether-lab)

## Software Stack

Regardless of hardware profile, you'll need these core components:

| Component | Version | Purpose |
|-----------|---------|---------|
| ROS 2 | Humble or Iron | Robotics middleware |
| Gazebo | Harmonic | Physics simulation |
| Python | 3.10+ | Primary language |
| Docker | 24.x+ | Containerization |

## Environment Verification

After completing setup, run the verification script for your profile:

```bash
# Workstation
./verify-workstation.sh

# Jetson
./verify-jetson.sh

# Cloud
./verify-cloud.sh
```

A successful verification shows:
```
✓ ROS 2 Humble installed
✓ Gazebo Harmonic available
✓ Python 3.10+ detected
✓ GPU acceleration enabled
✓ All dependencies satisfied

Environment ready for Physical AI coursework!
```

## Troubleshooting

Common issues and solutions:

- **NVIDIA driver not found**: Ensure you've installed the latest driver from NVIDIA's repository
- **ROS 2 packages missing**: Run `rosdep install --from-paths src --ignore-src -y`
- **Gazebo fails to start**: Check OpenGL version with `glxinfo | grep OpenGL`

## Next Steps

1. Choose your hardware profile above
2. Follow the setup guide
3. Run the verification script
4. Return here and proceed to [Module 1: ROS 2 Fundamentals](/docs/module-1-robotic-nervous-system/week-01/introduction)
