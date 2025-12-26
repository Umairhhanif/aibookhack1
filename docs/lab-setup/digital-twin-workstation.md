---
sidebar_position: 2
---

# Digital Twin Workstation Setup

This guide walks you through setting up a high-performance workstation for the full Physical AI curriculum, including NVIDIA Isaac Sim.

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | 8-core Intel/AMD | 12+ core |
| RAM | 64GB | 128GB |
| GPU | RTX 4070 Ti (12GB) | RTX 4090 (24GB) |
| Storage | 500GB NVMe SSD | 1TB+ NVMe SSD |
| OS | Ubuntu 22.04 LTS | Ubuntu 22.04 LTS |

:::warning GPU Requirement
Isaac Sim requires an NVIDIA RTX GPU with at least 12GB VRAM. Older GTX cards are not supported.
:::

## Step 1: Install Ubuntu 22.04 LTS

1. Download Ubuntu 22.04 LTS from [ubuntu.com](https://ubuntu.com/download/desktop)
2. Create a bootable USB with Rufus or Etcher
3. Install with default settings
4. Update the system:

```bash
sudo apt update && sudo apt upgrade -y
```

## Step 2: Install NVIDIA Drivers

```bash
# Add NVIDIA repository
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt update

# Install the latest driver (check for your GPU)
sudo apt install nvidia-driver-545 -y

# Reboot
sudo reboot
```

Verify installation:
```bash
nvidia-smi
```

Expected output shows your GPU model and driver version.

## Step 3: Install CUDA Toolkit

```bash
# Download CUDA 12.x (required for Isaac Sim)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install cuda-toolkit-12-3 -y

# Add to PATH
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

Verify:
```bash
nvcc --version
```

## Step 4: Install ROS 2 Humble

```bash
# Set locale
sudo apt update && sudo apt install locales
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8

# Add ROS 2 repository
sudo apt install software-properties-common
sudo add-apt-repository universe
sudo apt update && sudo apt install curl -y
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

# Install ROS 2 Humble Desktop
sudo apt update
sudo apt install ros-humble-desktop -y

# Install development tools
sudo apt install ros-dev-tools python3-colcon-common-extensions -y

# Source ROS 2
echo 'source /opt/ros/humble/setup.bash' >> ~/.bashrc
source ~/.bashrc
```

Verify:
```bash
ros2 doctor
```

## Step 5: Install Gazebo Harmonic

```bash
# Add Gazebo repository
sudo wget https://packages.osrfoundation.org/gazebo.gpg -O /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null

sudo apt update
sudo apt install gz-harmonic ros-humble-ros-gz -y
```

Verify:
```bash
gz sim --version
```

## Step 6: Install NVIDIA Isaac Sim

:::info NVIDIA Developer Account Required
Isaac Sim requires a free NVIDIA Developer account. Register at [developer.nvidia.com](https://developer.nvidia.com).
:::

1. Install Omniverse Launcher:
   - Download from [NVIDIA Omniverse](https://www.nvidia.com/en-us/omniverse/)
   - Install the AppImage

2. Launch Omniverse and install Isaac Sim:
   ```bash
   ./omniverse-launcher-linux.AppImage
   ```

3. In the Launcher, navigate to Exchange → Isaac Sim → Install

4. Verify Isaac Sim launches without errors

## Step 7: Install Additional Dependencies

```bash
# Python packages
pip3 install numpy scipy matplotlib opencv-python

# ROS 2 packages for navigation and SLAM
sudo apt install ros-humble-navigation2 ros-humble-nav2-bringup -y
sudo apt install ros-humble-slam-toolbox -y
sudo apt install ros-humble-robot-localization -y

# Visualization
sudo apt install ros-humble-rviz2 ros-humble-rqt* -y
```

## Step 8: Create ROS 2 Workspace

```bash
# Create workspace
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws

# Build
colcon build --symlink-install

# Source workspace
echo 'source ~/ros2_ws/install/setup.bash' >> ~/.bashrc
source ~/.bashrc
```

## Verification Script

Download and run the verification script:

```bash
# Download
wget https://raw.githubusercontent.com/your-username/book/main/static/code-samples/lab-setup/verify-workstation.sh
chmod +x verify-workstation.sh

# Run
./verify-workstation.sh
```

Expected output:
```
Physical AI Workstation Verification
====================================
✓ Ubuntu 22.04 detected
✓ NVIDIA driver 545.xx installed
✓ CUDA 12.3 available
✓ ROS 2 Humble installed
✓ Gazebo Harmonic available
✓ Isaac Sim accessible
✓ Navigation stack installed
✓ ROS 2 workspace configured

All checks passed! Your workstation is ready.
```

## Next Steps

Your Digital Twin Workstation is ready! Proceed to:

1. [Module 1: ROS 2 Fundamentals](/docs/module-1-robotic-nervous-system/week-01/introduction)
2. Or explore [Isaac Sim basics](/docs/module-2-digital-twin/week-06/isaac-sim-intro)

## Troubleshooting

### NVIDIA Driver Issues

```bash
# Remove existing drivers
sudo apt purge nvidia-* -y
sudo apt autoremove -y

# Reinstall
sudo apt install nvidia-driver-545 -y
sudo reboot
```

### ROS 2 Package Conflicts

```bash
# Clean and rebuild workspace
cd ~/ros2_ws
rm -rf build install log
colcon build --symlink-install
```

### Isaac Sim Won't Start

1. Check GPU compatibility with `nvidia-smi`
2. Ensure Vulkan is installed: `sudo apt install vulkan-tools`
3. Verify display: `echo $DISPLAY`
