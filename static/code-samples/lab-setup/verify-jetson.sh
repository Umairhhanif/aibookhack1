#!/bin/bash
# Physical AI Jetson Kit Verification Script
# Verifies that all required components are installed for the Economy Jetson Student Kit

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "Physical AI Jetson Kit Verification"
echo "==================================="
echo ""

PASS=0
FAIL=0

check_pass() {
    echo -e "${GREEN}✓${NC} $1"
    ((PASS++))
}

check_fail() {
    echo -e "${RED}✗${NC} $1"
    ((FAIL++))
}

check_warn() {
    echo -e "${YELLOW}⚠${NC} $1"
}

# Check JetPack version
if [ -f /etc/nv_tegra_release ]; then
    JETPACK_INFO=$(cat /etc/nv_tegra_release)
    check_pass "JetPack detected: $JETPACK_INFO"
else
    # Try alternative detection
    if dpkg -l | grep -q nvidia-jetpack; then
        JETPACK_VERSION=$(dpkg -l | grep nvidia-jetpack | awk '{print $3}')
        check_pass "JetPack $JETPACK_VERSION detected"
    else
        check_fail "JetPack not detected (not running on Jetson?)"
    fi
fi

# Check CUDA availability
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9.]*\).*/\1/p')
    check_pass "CUDA $CUDA_VERSION available"
elif [ -d /usr/local/cuda ]; then
    check_pass "CUDA installation found at /usr/local/cuda"
else
    check_fail "CUDA not available"
fi

# Check ROS 2
if [ -f /opt/ros/humble/setup.bash ]; then
    check_pass "ROS 2 Humble installed"
elif [ -f /opt/ros/iron/setup.bash ]; then
    check_pass "ROS 2 Iron installed"
else
    check_fail "ROS 2 Humble/Iron not found"
fi

# Check RealSense SDK
if command -v realsense-viewer &> /dev/null; then
    check_pass "RealSense SDK installed"
else
    check_fail "RealSense SDK not found (realsense-viewer not available)"
fi

# Check for RealSense camera
if lsusb 2>/dev/null | grep -qi "Intel.*RealSense"; then
    CAMERA_MODEL=$(lsusb | grep -i "Intel.*RealSense" | head -1)
    check_pass "Camera detected: $CAMERA_MODEL"
else
    check_warn "RealSense camera not detected (ensure it's connected via USB 3.0)"
    ((FAIL++))
fi

# Check ROS 2 RealSense wrapper
if ros2 pkg list 2>/dev/null | grep -q "realsense2_camera"; then
    check_pass "ROS 2 RealSense wrapper installed"
else
    check_fail "ROS 2 RealSense wrapper (realsense2_camera) not found"
fi

# Check microphone
if arecord -l 2>/dev/null | grep -q "card"; then
    MIC_INFO=$(arecord -l 2>/dev/null | grep "card" | head -1)
    check_pass "Microphone detected: $MIC_INFO"
else
    check_warn "No microphone detected"
    ((FAIL++))
fi

# Check ROS 2 workspace
if [ -d "$HOME/ros2_ws" ]; then
    if [ -f "$HOME/ros2_ws/install/setup.bash" ]; then
        check_pass "ROS 2 workspace configured"
    else
        check_warn "ROS 2 workspace exists but not built"
        ((PASS++))
    fi
else
    check_fail "ROS 2 workspace not found at ~/ros2_ws"
fi

# Check Python version
PYTHON_VERSION=$(python3 --version 2>/dev/null | sed 's/Python //')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$PYTHON_MAJOR" -ge 3 ] && [ "$PYTHON_MINOR" -ge 10 ]; then
    check_pass "Python $PYTHON_VERSION detected"
else
    check_warn "Python $PYTHON_VERSION detected (3.10+ recommended)"
    ((PASS++))
fi

# Check memory
TOTAL_MEM_KB=$(grep MemTotal /proc/meminfo | awk '{print $2}')
TOTAL_MEM_GB=$((TOTAL_MEM_KB / 1024 / 1024))
if [ "$TOTAL_MEM_GB" -ge 7 ]; then
    check_pass "RAM: ${TOTAL_MEM_GB}GB available"
else
    check_warn "RAM: ${TOTAL_MEM_GB}GB (8GB recommended)"
    ((PASS++))
fi

# Check swap
SWAP_TOTAL=$(free -g | grep Swap | awk '{print $2}')
if [ "$SWAP_TOTAL" -ge 4 ]; then
    check_pass "Swap: ${SWAP_TOTAL}GB configured"
else
    check_warn "Swap: ${SWAP_TOTAL}GB (4GB+ recommended for Jetson)"
    ((PASS++))
fi

# Check power mode
if command -v nvpmodel &> /dev/null; then
    POWER_MODE=$(nvpmodel -q 2>/dev/null | grep "NV Power Mode" | head -1)
    check_pass "Power mode: $POWER_MODE"
fi

# Summary
echo ""
echo "==================================="
echo "Results: $PASS passed, $FAIL failed"
echo ""

if [ $FAIL -eq 0 ]; then
    echo -e "${GREEN}All checks passed! Your Jetson kit is ready.${NC}"
    exit 0
else
    echo -e "${RED}Some checks failed. Please review the issues above.${NC}"
    exit 1
fi
