---
sidebar_position: 3
---

# Exercise: Your First ROS 2 Node

In this hands-on exercise, you'll create your very first ROS 2 node from scratch. This will introduce you to the fundamental concepts of ROS 2 programming and provide a working foundation for more complex robotic applications.

## What is a ROS 2 Node?

A ROS 2 node is the basic computational unit in the Robot Operating System 2. Think of it as a process that performs specific tasks - like controlling a sensor, processing data, or managing robot behavior. Nodes communicate with each other through topics, services, and actions, forming a distributed system that enables complex robotic functionality.

In this exercise, you'll create a "Heartbeat" node that demonstrates the core concepts of node creation, timers, parameters, and logging.

## Objective

Create a "Heartbeat" node that:
1. Logs a status message every second
2. Counts the number of beats
3. Uses a configurable rate parameter

## Prerequisites

- ROS 2 Humble installed
- ROS 2 workspace created (`~/ros2_ws`)
- Completed the [Nodes](./nodes) tutorial

## Step 1: Create the Package

```bash
cd ~/ros2_ws/src

ros2 pkg create --build-type ament_python heartbeat_node \
    --dependencies rclpy
```

## Step 2: Create the Node

Create the file `~/ros2_ws/src/heartbeat_node/heartbeat_node/heartbeat.py`:

```python
#!/usr/bin/env python3
"""
Heartbeat Node - A simple node that publishes periodic status messages.

This exercise demonstrates:
- Node creation and initialization
- Timer callbacks
- Parameter declaration and usage
- Logging
"""

import rclpy
from rclpy.node import Node


class HeartbeatNode(Node):
    """A node that periodically logs heartbeat messages."""

    def __init__(self):
        super().__init__('heartbeat_node')

        # Declare parameters with default values
        self.declare_parameter('rate_hz', 1.0)
        self.declare_parameter('node_name', 'heartbeat')

        # Get parameter values
        self.rate = self.get_parameter('rate_hz').value
        self.name = self.get_parameter('node_name').value

        # Initialize counter
        self.beat_count = 0

        # Calculate timer period from rate
        timer_period = 1.0 / self.rate  # seconds

        # Create a timer that calls our callback
        self.timer = self.create_timer(timer_period, self.timer_callback)

        self.get_logger().info(
            f'Heartbeat node started with rate: {self.rate} Hz'
        )

    def timer_callback(self):
        """Called periodically by the timer."""
        self.beat_count += 1
        self.get_logger().info(
            f'[{self.name}] Heartbeat #{self.beat_count}'
        )


def main(args=None):
    """Main entry point for the node."""
    rclpy.init(args=args)

    node = HeartbeatNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Heartbeat node shutting down...')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

## Understanding the Code

Let's break down the key components of your first ROS 2 node:

1. **Node Class Inheritance**: `class HeartbeatNode(Node)` - Your class inherits from `rclpy.node.Node`, which provides all the ROS 2 functionality.

2. **Initialization**: The `__init__` method calls `super().__init__('heartbeat_node')` to initialize the parent Node class with a unique name.

3. **Parameters**: `self.declare_parameter()` allows configuration without recompiling code.

4. **Timers**: `self.create_timer()` creates a callback that executes periodically, enabling time-based operations.

5. **Logging**: `self.get_logger().info()` provides structured output for debugging and monitoring.

6. **Lifecycle Management**: The `main()` function properly initializes and shuts down the ROS 2 client library.

## Step 3: Configure the Package

Edit `~/ros2_ws/src/heartbeat_node/setup.py`:

```python
from setuptools import find_packages, setup

package_name = 'heartbeat_node'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='you@example.com',
    description='A simple heartbeat node for learning ROS 2',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'heartbeat = heartbeat_node.heartbeat:main',
        ],
    },
)
```

## Step 4: Build and Run

```bash
# Build the package
cd ~/ros2_ws
colcon build --packages-select heartbeat_node

# Source the workspace
source install/setup.bash

# Run the node
ros2 run heartbeat_node heartbeat
```

Expected output:
```
[INFO] [heartbeat_node]: Heartbeat node started with rate: 1.0 Hz
[INFO] [heartbeat_node]: [heartbeat] Heartbeat #1
[INFO] [heartbeat_node]: [heartbeat] Heartbeat #2
[INFO] [heartbeat_node]: [heartbeat] Heartbeat #3
...
```

## Step 5: Test with Parameters

Run with different parameters:

```bash
# Faster heartbeat
ros2 run heartbeat_node heartbeat --ros-args -p rate_hz:=2.0

# Custom name
ros2 run heartbeat_node heartbeat --ros-args \
    -p rate_hz:=0.5 \
    -p node_name:=slow_heart
```

## Step 6: Inspect the Node

In a separate terminal:

```bash
# List running nodes
ros2 node list

# Get node info
ros2 node info /heartbeat_node

# List parameters
ros2 param list /heartbeat_node

# Get a parameter value
ros2 param get /heartbeat_node rate_hz

# Set parameter at runtime
ros2 param set /heartbeat_node rate_hz 5.0
```

:::note Runtime Parameter Changes
The rate won't change dynamically because we only read the parameter once at startup. Making parameters dynamically reconfigurable is covered in Week 3.
:::

## Challenges

### Challenge 1: Add a Maximum Beat Count
Modify the node to stop after a configurable number of beats.

<details>
<summary>Hint</summary>

Add a `max_beats` parameter and check it in the callback:

```python
self.declare_parameter('max_beats', 0)  # 0 = unlimited
self.max_beats = self.get_parameter('max_beats').value

def timer_callback(self):
    self.beat_count += 1
    if self.max_beats > 0 and self.beat_count >= self.max_beats:
        self.get_logger().info('Maximum beats reached. Shutting down.')
        raise SystemExit
```
</details>

### Challenge 2: Add Multiple Timers
Add a second timer that logs "Still alive!" every 5 seconds.

<details>
<summary>Hint</summary>

```python
self.status_timer = self.create_timer(5.0, self.status_callback)

def status_callback(self):
    self.get_logger().info(f'Still alive! Total beats: {self.beat_count}')
```
</details>

### Challenge 3: Create a Publisher/Subscriber Pair
Extend your knowledge by creating two nodes that communicate: a publisher that sends heartbeat messages and a subscriber that receives and logs them.

<details>
<summary>Hint</summary>

Create a publisher node:
```python
# Publisher node
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class HeartbeatPublisher(Node):
    def __init__(self):
        super().__init__('heartbeat_publisher')
        self.publisher = self.create_publisher(String, 'heartbeat_topic', 10)
        timer_period = 1.0
        self.timer = self.create_timer(timer_period, self.publish_heartbeat)
        self.counter = 0

    def publish_heartbeat(self):
        msg = String()
        msg.data = f'Heartbeat #{self.counter}'
        self.publisher.publish(msg)
        self.get_logger().info(f'Publishing: {msg.data}')
        self.counter += 1

# Subscriber node
class HeartbeatSubscriber(Node):
    def __init__(self):
        super().__init__('heartbeat_subscriber')
        self.subscription = self.create_subscription(
            String,
            'heartbeat_topic',
            self.listener_callback,
            10
        )

    def listener_callback(self, msg):
        self.get_logger().info(f'Received heartbeat: {msg.data}')
```
</details>

### Challenge 4: Create a C++ Version
Recreate this node in C++ using `rclcpp`.

<details>
<summary>Hint</summary>

Key differences:
- Use `rclcpp::Node` instead of `rclpy.node.Node`
- Use `this->create_wall_timer()` for timers
- Use `RCLCPP_INFO()` for logging
- Build with `ament_cmake` instead of `ament_python`
</details>

## Verification Checklist

- [ ] Node starts without errors
- [ ] Heartbeat messages appear at correct rate
- [ ] Parameters can be set via command line
- [ ] Node shuts down cleanly with Ctrl+C
- [ ] `ros2 node list` shows the node
- [ ] `ros2 param list` shows parameters
- [ ] Challenge solutions work as expected

## Common Issues

### Package Not Found After Build
```bash
# Make sure to source the workspace
source ~/ros2_ws/install/setup.bash
```

### Module Not Found Error
Ensure the `entry_points` in `setup.py` matches your file structure:
```python
'heartbeat = heartbeat_node.heartbeat:main'
#           ^^^^^^^^^^^^^^ ^^^^^^^^^
#           package folder  filename
```

### Timer Not Firing
Check that you're calling `rclpy.spin(node)` to process callbacks.

### Common First-Time User Issues

**Environment Not Sourced**: If you get "command not found" errors for `ros2` commands:
```bash
# Make sure to source your ROS 2 installation
source /opt/ros/humble/setup.bash  # Replace 'humble' with your ROS 2 version
# Or if using your workspace:
source ~/ros2_ws/install/setup.bash
```

**Import Errors**: If you see `ImportError: No module named 'rclpy'`:
- Ensure ROS 2 is properly installed and sourced
- Check that you're running Python 3 (ROS 2 requires Python 3.6+)

**Permission Errors**: If you get permission errors when creating packages:
- Make sure you're working in your home directory (`~/ros2_ws`)
- Don't use `sudo` with ROS 2 commands

**Package Not Found**: If `colcon build` completes but `ros2 run` fails:
- Make sure to source the install directory: `source ~/ros2_ws/install/setup.bash`
- Check that the package name in `setup.py` matches your directory structure

## Summary

In this exercise, you learned to:
- Create a ROS 2 Python package
- Write a node with timer callbacks
- Use parameters for configuration
- Build and run your node
- Inspect nodes with ROS 2 CLI tools

## Understanding ROS 2 in the Robotic Ecosystem

ROS 2 (Robot Operating System 2) is not an operating system but a flexible framework for writing robot software. It provides services designed specifically for a heterogeneous computer cluster such as:

- **Hardware Abstraction**: Interface with various sensors and actuators
- **Device Drivers**: Connect to different hardware components
- **Libraries**: Reusable code for common robotic functions
- **Message Passaging**: Communication between different parts of your robot
- **Package Management**: Organize and distribute robot software

Your heartbeat node is a simple example of how ROS 2 enables modular, distributed robot software architecture.

## Beyond the Basics

Now that you've created your first ROS 2 node, consider exploring these advanced concepts:

- **Publishers and Subscribers**: Learn how nodes communicate through topics
- **Services**: Create request-response communication patterns
- **Actions**: Implement goal-oriented communication with feedback
- **Parameter Servers**: Manage configuration across multiple nodes
- **Launch Files**: Start multiple nodes with a single command
- **TF Transforms**: Handle coordinate frame transformations
- **Robot State Publishing**: Share robot joint states and transformations

## Next Steps

Continue to [Week 2: Topics and Publishers](../week-02/topics) to learn how nodes communicate with each other.
