#!/bin/bash
# Physical AI Workstation Verification Script
# Verifies that all required components are installed for the Digital Twin Workstation

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "Physical AI Workstation Verification"
echo "===================================="
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

# Check Ubuntu version
if [ -f /etc/os-release ]; then
    . /etc/os-release
    if [[ "$VERSION_ID" == "22.04" ]]; then
        check_pass "Ubuntu 22.04 detected"
    else
        check_warn "Ubuntu $VERSION_ID detected (22.04 recommended)"
        ((PASS++))
    fi
else
    check_fail "Could not detect OS version"
fi

# Check NVIDIA driver
if command -v nvidia-smi &> /dev/null; then
    DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
    if [ -n "$DRIVER_VERSION" ]; then
        check_pass "NVIDIA driver $DRIVER_VERSION installed"
    else
        check_fail "NVIDIA driver not responding"
    fi
else
    check_fail "NVIDIA driver not found (nvidia-smi not available)"
fi

# Check CUDA
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9.]*\).*/\1/p')
    check_pass "CUDA $CUDA_VERSION available"
else
    check_fail "CUDA not found (nvcc not available)"
fi

# Check ROS 2
if [ -f /opt/ros/humble/setup.bash ]; then
    check_pass "ROS 2 Humble installed"
elif [ -f /opt/ros/iron/setup.bash ]; then
    check_pass "ROS 2 Iron installed"
else
    check_fail "ROS 2 Humble/Iron not found"
fi

# Check Gazebo
if command -v gz &> /dev/null; then
    GZ_VERSION=$(gz sim --version 2>/dev/null | head -1)
    check_pass "Gazebo Harmonic available"
else
    check_fail "Gazebo not found"
fi

# Check Isaac Sim (via Omniverse)
ISAAC_PATHS=(
    "$HOME/.local/share/ov/pkg/isaac_sim-*"
    "/opt/nvidia/isaac_sim"
    "$HOME/isaac_sim"
)

ISAAC_FOUND=false
for path_pattern in "${ISAAC_PATHS[@]}"; do
    for path in $path_pattern; do
        if [ -d "$path" ]; then
            check_pass "Isaac Sim accessible at $path"
            ISAAC_FOUND=true
            break 2
        fi
    done
done

if [ "$ISAAC_FOUND" = false ]; then
    check_fail "Isaac Sim not found (check Omniverse installation)"
fi

# Check Navigation stack
if ros2 pkg list 2>/dev/null | grep -q "nav2_bringup"; then
    check_pass "Navigation stack installed"
else
    check_fail "Navigation stack (nav2_bringup) not found"
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
    check_fail "Python 3.10+ required (found $PYTHON_VERSION)"
fi

# Check GPU VRAM
if command -v nvidia-smi &> /dev/null; then
    VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
    VRAM_GB=$((VRAM_MB / 1024))
    if [ "$VRAM_GB" -ge 12 ]; then
        check_pass "GPU VRAM: ${VRAM_GB}GB (sufficient for Isaac Sim)"
    else
        check_warn "GPU VRAM: ${VRAM_GB}GB (12GB+ recommended for Isaac Sim)"
        ((PASS++))
    fi
fi

# Summary
echo ""
echo "===================================="
echo "Results: $PASS passed, $FAIL failed"
echo ""

if [ $FAIL -eq 0 ]; then
    echo -e "${GREEN}All checks passed! Your workstation is ready.${NC}"
    exit 0
else
    echo -e "${RED}Some checks failed. Please review the issues above.${NC}"
    exit 1
fi
