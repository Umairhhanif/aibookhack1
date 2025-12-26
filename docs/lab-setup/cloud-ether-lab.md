---
sidebar_position: 3
---

# Cloud Ether Lab Setup

Run the Physical AI curriculum without local GPU hardware using cloud-based simulation environments.

## Overview

The Cloud Ether Lab provides:
- Browser-based access to ROS 2 and Gazebo
- Pre-configured Docker containers
- GPU instances on-demand
- No local installation required

:::info Cost Consideration
Cloud GPU instances are billed by the hour. Expect $0.50-$2.00/hour depending on the provider and GPU tier.
:::

## Option 1: The Construct (Recommended for Beginners)

[The Construct](https://www.theconstructsim.com/) offers a fully managed ROS 2 development environment.

### Setup Steps

1. **Create an account** at [theconstructsim.com](https://www.theconstructsim.com/)

2. **Launch a ROS 2 Development Environment**:
   - Select "ROS 2 Humble" from the available environments
   - Choose a project template or start blank

3. **Verify the environment**:
   ```bash
   ros2 doctor
   gz sim --version
   ```

### Pros and Cons

| Pros | Cons |
|------|------|
| Zero setup | Monthly subscription cost |
| Pre-configured tools | Limited GPU access |
| Educational resources | Internet required |

## Option 2: Foxglove + Gazebo Web

Use [Foxglove](https://foxglove.dev/) for visualization with a local Docker container.

### Setup Steps

1. **Install Docker**:
   ```bash
   # Ubuntu/Debian
   curl -fsSL https://get.docker.com -o get-docker.sh
   sudo sh get-docker.sh
   sudo usermod -aG docker $USER
   # Log out and back in
   ```

2. **Pull ROS 2 + Gazebo container**:
   ```bash
   docker pull osrf/ros:humble-desktop
   ```

3. **Run the container**:
   ```bash
   docker run -it --rm \
     --name ros2_gazebo \
     -e DISPLAY=$DISPLAY \
     -v /tmp/.X11-unix:/tmp/.X11-unix \
     osrf/ros:humble-desktop \
     bash
   ```

4. **Inside the container**:
   ```bash
   source /opt/ros/humble/setup.bash
   ros2 run demo_nodes_cpp talker
   ```

5. **Set up Foxglove**:
   - Install Foxglove Studio from [foxglove.dev](https://foxglove.dev/download)
   - Connect to your ROS 2 bridge: `ros2 launch foxglove_bridge foxglove_bridge_launch.xml`

## Option 3: AWS RoboMaker

Amazon's managed robotics simulation service with GPU support.

### Setup Steps

1. **Create AWS account** at [aws.amazon.com](https://aws.amazon.com/)

2. **Navigate to RoboMaker**:
   - Open AWS Console
   - Search for "RoboMaker"

3. **Create a simulation job**:
   - Select ROS 2 Humble
   - Choose Gazebo as the simulation engine
   - Configure compute resources

4. **Upload your application**:
   ```bash
   # Package your ROS 2 workspace
   colcon bundle
   aws s3 cp robot_ws/bundle/output.tar s3://your-bucket/
   ```

### Cost Estimate

| Resource | Approx. Cost |
|----------|--------------|
| Development Environment | $0.10/hour |
| Simulation (CPU) | $0.50/hour |
| Simulation (GPU) | $1.50/hour |

## Option 4: Google Cloud Robotics

Use Google Cloud with GPU VMs for simulation.

### Setup Steps

1. **Create a GCP project**

2. **Launch a GPU VM**:
   ```bash
   gcloud compute instances create ros2-sim \
     --zone=us-central1-a \
     --machine-type=n1-standard-8 \
     --accelerator=type=nvidia-tesla-t4,count=1 \
     --image-family=ubuntu-2204-lts \
     --image-project=ubuntu-os-cloud \
     --boot-disk-size=100GB
   ```

3. **SSH and install ROS 2**:
   ```bash
   gcloud compute ssh ros2-sim
   # Follow the Workstation setup guide for ROS 2 installation
   ```

4. **Set up remote desktop**:
   ```bash
   sudo apt install xrdp ubuntu-desktop -y
   ```

## Local Docker Development

For exercises that don't require GPU, use local Docker:

### docker-compose.yml

```yaml
version: '3.8'

services:
  ros2:
    image: osrf/ros:humble-desktop
    container_name: physical_ai_ros2
    volumes:
      - ./ros2_ws:/home/ros2_ws
      - /tmp/.X11-unix:/tmp/.X11-unix
    environment:
      - DISPLAY=${DISPLAY}
      - ROS_DOMAIN_ID=42
    network_mode: host
    stdin_open: true
    tty: true
    command: bash
```

### Usage

```bash
# Start container
docker-compose up -d

# Enter container
docker exec -it physical_ai_ros2 bash

# Inside container
source /opt/ros/humble/setup.bash
ros2 run demo_nodes_py talker
```

## Verification Script

Run the cloud environment verification:

```bash
# Download
wget https://raw.githubusercontent.com/your-username/book/main/static/code-samples/lab-setup/verify-cloud.sh
chmod +x verify-cloud.sh

# Run
./verify-cloud.sh
```

Expected output:
```
Physical AI Cloud Environment Verification
==========================================
✓ Docker installed and running
✓ ROS 2 container available
✓ Network connectivity verified
✓ Visualization tools accessible

Cloud environment ready!
```

## Limitations

Cloud environments have some limitations compared to local workstations:

| Feature | Cloud | Workstation |
|---------|-------|-------------|
| Isaac Sim | ❌ Limited | ✓ Full |
| Real-time performance | Variable | Consistent |
| Sensor integration | ❌ Not available | ✓ Full |
| Cost | Per-hour | One-time |

## Recommended for Modules

| Module | Cloud Suitable? |
|--------|-----------------|
| Module 1: ROS 2 | ✓ Fully |
| Module 2: Gazebo | ✓ Mostly |
| Module 2: Isaac Sim | ⚠️ Limited |
| Module 3: Navigation | ✓ Mostly |
| Module 4: Capstone | ⚠️ Simulation only |

## Next Steps

With your cloud environment ready:

1. [Start Module 1](/docs/module-1-robotic-nervous-system/week-01/introduction)
2. Consider upgrading to a workstation for Modules 2-4 advanced content

## Troubleshooting

### Docker Permission Denied

```bash
sudo groupadd docker
sudo usermod -aG docker $USER
newgrp docker
```

### Display Not Available

```bash
# On host machine
xhost +local:docker

# Or use VNC for remote access
docker run -p 5901:5901 ...
```

### Slow Performance

- Ensure you're using the nearest cloud region
- Consider upgrading to a GPU instance for Gazebo
- Reduce simulation complexity in Gazebo world files
