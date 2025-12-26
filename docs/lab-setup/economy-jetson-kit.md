---
sidebar_position: 4
---

# Economy Jetson Student Kit Setup

Build an edge AI development kit with the NVIDIA Jetson Orin Nano for hands-on sensor integration and real-world deployment.

## Kit Components

| Component | Model | Purpose |
|-----------|-------|---------|
| Compute | Jetson Orin Nano Developer Kit | AI inference and ROS 2 |
| Vision | Intel RealSense D435i | Depth camera + IMU |
| Audio | USB Microphone Array | Voice commands (Module 4) |
| Power | 20V/65W USB-C Power Supply | Powers the Jetson |
| Storage | 256GB+ microSD (A2 rated) | OS and applications |
| Peripherals | USB hub, keyboard, monitor | Initial setup |

## Estimated Cost

| Component | Approx. Price (USD) |
|-----------|---------------------|
| Jetson Orin Nano Dev Kit | $499 |
| Intel RealSense D435i | $315 |
| USB Microphone Array | $50 |
| Power Supply (if not included) | $30 |
| microSD Card 256GB | $30 |
| **Total** | **~$925** |

:::tip Cost Savings
- Buy refurbished Jetson units from NVIDIA's outlet
- The RealSense D435 (without IMU) is $50 cheaper
- Any USB microphone works for basic voice input
:::

## Step 1: Flash JetPack 6.0

1. **Download JetPack 6.0** from [NVIDIA Developer](https://developer.nvidia.com/embedded/jetpack)

2. **Flash the SD card**:
   - Use [Balena Etcher](https://www.balena.io/etcher/) on your host computer
   - Select the JetPack image and your SD card
   - Flash and wait for completion

3. **First boot**:
   - Insert the SD card into the Jetson
   - Connect monitor, keyboard, and power
   - Complete the Ubuntu setup wizard

4. **Update the system**:
   ```bash
   sudo apt update && sudo apt upgrade -y
   ```

## Step 2: Install ROS 2 Humble

The Jetson runs Ubuntu 22.04, making ROS 2 Humble installation straightforward.

```bash
# Set locale
sudo apt update && sudo apt install locales
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8

# Add ROS 2 repository
sudo apt install software-properties-common curl -y
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

# Install ROS 2 (base is sufficient for Jetson)
sudo apt update
sudo apt install ros-humble-ros-base ros-dev-tools -y

# Source ROS 2
echo 'source /opt/ros/humble/setup.bash' >> ~/.bashrc
source ~/.bashrc
```

:::warning Memory Note
The Orin Nano has 8GB RAM. Use `ros-humble-ros-base` instead of `ros-humble-desktop` to save memory.
:::

## Step 3: Install Intel RealSense SDK

```bash
# Add Intel repository key
sudo mkdir -p /etc/apt/keyrings
curl -sSf https://librealsense.intel.com/Debian/librealsense.pgp | sudo tee /etc/apt/keyrings/librealsense.pgp > /dev/null

# Add repository
echo "deb [signed-by=/etc/apt/keyrings/librealsense.pgp] https://librealsense.intel.com/Debian/apt-repo $(lsb_release -cs) main" | \
sudo tee /etc/apt/sources.list.d/librealsense.list

# Install SDK
sudo apt update
sudo apt install librealsense2-dkms librealsense2-utils -y

# Install ROS 2 wrapper
sudo apt install ros-humble-realsense2-camera ros-humble-realsense2-description -y
```

Verify the camera:
```bash
# Connect RealSense and run
realsense-viewer
```

## Step 4: Configure USB Microphone

1. **Connect the USB microphone array**

2. **Verify detection**:
   ```bash
   arecord -l
   ```

3. **Set as default**:
   ```bash
   # Find the card number from arecord -l
   # Edit ALSA config
   sudo nano /etc/asound.conf
   ```

   Add:
   ```
   defaults.pcm.card 1
   defaults.ctl.card 1
   ```

4. **Test recording**:
   ```bash
   arecord -d 5 test.wav
   aplay test.wav
   ```

## Step 5: Create ROS 2 Workspace

```bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws
colcon build --symlink-install
echo 'source ~/ros2_ws/install/setup.bash' >> ~/.bashrc
source ~/.bashrc
```

## Step 6: Test RealSense with ROS 2

```bash
# Launch RealSense node
ros2 launch realsense2_camera rs_launch.py

# In another terminal, view topics
ros2 topic list

# View camera info
ros2 topic echo /camera/camera/color/camera_info --once
```

Expected topics:
```
/camera/camera/color/image_raw
/camera/camera/depth/image_rect_raw
/camera/camera/imu
/camera/camera/aligned_depth_to_color/image_raw
```

## Step 7: Power Management

Optimize for continuous operation:

```bash
# Set to maximum performance mode
sudo nvpmodel -m 0

# Enable fan (if using active cooling)
sudo jetson_clocks
```

Monitor power and temperature:
```bash
tegrastats
```

## Assembly Guide

### Physical Setup

1. **Mount the Jetson** in a case with ventilation
2. **Connect the RealSense** via USB 3.0 port (blue port)
3. **Connect the microphone** to any USB port
4. **Attach power supply** (minimum 65W)

### Cable Management

```
                    ┌─────────────────┐
  [Power 65W] ──────│  Jetson Orin    │
                    │     Nano        │
  [RealSense] ──────│  (USB 3.0)      │
                    │                 │
  [Mic Array] ──────│  (USB 2.0)      │
                    │                 │
  [HDMI] ───────────│  (for setup)    │
                    └─────────────────┘
```

## Verification Script

```bash
# Download
wget https://raw.githubusercontent.com/your-username/book/main/static/code-samples/lab-setup/verify-jetson.sh
chmod +x verify-jetson.sh

# Run
./verify-jetson.sh
```

Expected output:
```
Physical AI Jetson Kit Verification
===================================
✓ JetPack 6.0 detected
✓ CUDA 12.x available
✓ ROS 2 Humble installed
✓ RealSense SDK installed
✓ Camera detected: Intel RealSense D435i
✓ Microphone detected
✓ ROS 2 workspace configured

All checks passed! Your Jetson kit is ready.
```

## Performance Tips

### Memory Optimization

```bash
# Disable GUI for headless operation
sudo systemctl set-default multi-user.target

# Enable swap
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

### Thermal Management

- Use a fan or heatsink (the Dev Kit includes a fan)
- Monitor with `tegrastats`
- Reduce power mode if overheating: `sudo nvpmodel -m 1`

## Next Steps

Your Jetson kit is ready for:

1. [Module 1: ROS 2 Fundamentals](/docs/module-1-robotic-nervous-system/week-01/introduction)
2. [Module 3: VSLAM with RealSense](/docs/module-3-ai-robot-brain/week-07/vslam-intro)
3. [Module 4: Voice Commands](/docs/module-4-vision-language-action/week-10/whisper-integration)

## Troubleshooting

### RealSense Not Detected

```bash
# Check USB connection
lsusb | grep Intel

# Reset USB
sudo modprobe -r uvcvideo
sudo modprobe uvcvideo

# Try a different USB 3.0 port
```

### Out of Memory

```bash
# Kill memory-heavy processes
sudo systemctl stop gdm3

# Increase swap (see above)

# Use smaller ROS 2 installation
```

### Jetson Won't Boot

1. Re-flash the SD card
2. Ensure power supply provides 65W
3. Check for physical damage to SD card contacts
