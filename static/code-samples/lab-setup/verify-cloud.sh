#!/bin/bash
# Physical AI Cloud Environment Verification Script
# Verifies that all required components are available for Cloud Ether Lab

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "Physical AI Cloud Environment Verification"
echo "=========================================="
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

# Check Docker
if command -v docker &> /dev/null; then
    DOCKER_VERSION=$(docker --version | sed 's/Docker version //' | cut -d',' -f1)
    check_pass "Docker $DOCKER_VERSION installed"

    # Check if Docker daemon is running
    if docker info &> /dev/null; then
        check_pass "Docker daemon is running"
    else
        check_fail "Docker daemon not running (try: sudo systemctl start docker)"
    fi
else
    check_fail "Docker not installed"
fi

# Check Docker Compose
if command -v docker-compose &> /dev/null; then
    COMPOSE_VERSION=$(docker-compose --version | sed 's/.*version //' | cut -d',' -f1)
    check_pass "Docker Compose $COMPOSE_VERSION installed"
elif docker compose version &> /dev/null; then
    COMPOSE_VERSION=$(docker compose version | sed 's/.*version //')
    check_pass "Docker Compose (plugin) $COMPOSE_VERSION installed"
else
    check_warn "Docker Compose not found (optional but recommended)"
    ((PASS++))
fi

# Check for ROS 2 container
if docker images 2>/dev/null | grep -q "osrf/ros.*humble"; then
    check_pass "ROS 2 Humble container available"
elif docker images 2>/dev/null | grep -q "ros.*humble"; then
    check_pass "ROS 2 container available"
else
    check_warn "ROS 2 container not pulled (run: docker pull osrf/ros:humble-desktop)"
    ((PASS++))
fi

# Check network connectivity
if ping -c 1 packages.ros.org &> /dev/null; then
    check_pass "Network connectivity verified (ROS packages reachable)"
elif ping -c 1 8.8.8.8 &> /dev/null; then
    check_pass "Network connectivity verified"
else
    check_fail "No network connectivity"
fi

# Check if inside container (different checks apply)
if [ -f /.dockerenv ]; then
    echo ""
    echo "Running inside Docker container - checking ROS 2 directly..."
    echo ""

    # Check ROS 2 inside container
    if [ -f /opt/ros/humble/setup.bash ]; then
        check_pass "ROS 2 Humble available in container"
    elif [ -f /opt/ros/iron/setup.bash ]; then
        check_pass "ROS 2 Iron available in container"
    else
        check_fail "ROS 2 not found in container"
    fi

    # Check Gazebo inside container
    if command -v gz &> /dev/null; then
        check_pass "Gazebo available in container"
    elif command -v gazebo &> /dev/null; then
        check_pass "Gazebo (classic) available in container"
    else
        check_warn "Gazebo not found in container"
        ((PASS++))
    fi
fi

# Check X11/display (for GUI applications)
if [ -n "$DISPLAY" ]; then
    check_pass "DISPLAY variable set ($DISPLAY)"

    # Check if we can access X server
    if command -v xdpyinfo &> /dev/null && xdpyinfo &> /dev/null; then
        check_pass "X server accessible"
    else
        check_warn "X server may not be accessible (GUI apps might not work)"
        ((PASS++))
    fi
else
    check_warn "DISPLAY not set (GUI applications won't work directly)"
    ((PASS++))
fi

# Check for Foxglove (optional)
if command -v foxglove-studio &> /dev/null; then
    check_pass "Foxglove Studio installed"
elif [ -d "$HOME/.local/share/foxglove-studio" ] || [ -d "/opt/foxglove-studio" ]; then
    check_pass "Foxglove Studio found"
else
    check_warn "Foxglove Studio not installed (optional visualization tool)"
    ((PASS++))
fi

# Check Python version
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>/dev/null | sed 's/Python //')
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

    if [ "$PYTHON_MAJOR" -ge 3 ] && [ "$PYTHON_MINOR" -ge 10 ]; then
        check_pass "Python $PYTHON_VERSION detected"
    else
        check_warn "Python $PYTHON_VERSION detected (3.10+ recommended)"
        ((PASS++))
    fi
else
    check_warn "Python3 not found on host (available in container)"
    ((PASS++))
fi

# Check available disk space
DISK_AVAIL=$(df -BG . | tail -1 | awk '{print $4}' | sed 's/G//')
if [ "$DISK_AVAIL" -ge 20 ]; then
    check_pass "Disk space: ${DISK_AVAIL}GB available"
else
    check_warn "Disk space: ${DISK_AVAIL}GB available (20GB+ recommended)"
    ((PASS++))
fi

# Check memory
if [ -f /proc/meminfo ]; then
    TOTAL_MEM_KB=$(grep MemTotal /proc/meminfo | awk '{print $2}')
    TOTAL_MEM_GB=$((TOTAL_MEM_KB / 1024 / 1024))
    if [ "$TOTAL_MEM_GB" -ge 8 ]; then
        check_pass "RAM: ${TOTAL_MEM_GB}GB available"
    else
        check_warn "RAM: ${TOTAL_MEM_GB}GB (8GB+ recommended)"
        ((PASS++))
    fi
fi

# Summary
echo ""
echo "=========================================="
echo "Results: $PASS passed, $FAIL failed"
echo ""

if [ $FAIL -eq 0 ]; then
    echo -e "${GREEN}Cloud environment ready!${NC}"
    echo ""
    echo "Quick start:"
    echo "  docker run -it --rm osrf/ros:humble-desktop bash"
    echo ""
    exit 0
else
    echo -e "${RED}Some checks failed. Please review the issues above.${NC}"
    exit 1
fi
