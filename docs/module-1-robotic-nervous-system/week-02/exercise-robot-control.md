---
sidebar_position: 3
---

# Exercise: Robot Control System

In this hands-on exercise, you'll create a complete robot control system using topics, services, and actions. This will demonstrate how different communication patterns work together in a realistic robotic application.

## Objective

Create a robot control system with:
1. A **topic** for continuous sensor data
2. A **service** for immediate commands
3. An **action** for long-running navigation tasks

## Prerequisites

- Complete Week 1 and Week 2 lessons
- ROS 2 workspace set up (`~/ros2_ws`)
- Basic Python programming knowledge

## Step 1: Create the Package

```bash
cd ~/ros2_ws/src

ros2 pkg create --build-type ament_python robot_control_system \
    --dependencies rclpy std_msgs geometry_msgs sensor_msgs std_srvs example_interfaces
```

## Step 2: Create the Robot Sensor Publisher

Create `robot_control_system/robot_control_system/sensor_publisher.py`:

```python
#!/usr/bin/env python3
"""
Robot Sensor Publisher - Publishes simulated sensor data for the robot.
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
import math
import random

class RobotSensorPublisher(Node):
    def __init__(self):
        super().__init__('robot_sensor_publisher')

        # Create publishers
        self.scan_publisher = self.create_publisher(LaserScan, 'scan', 10)
        self.odom_publisher = self.create_publisher(Twist, 'odom', 10)

        # Create timer for sensor data
        timer_period = 0.1  # 10 Hz
        self.timer = self.create_timer(timer_period, self.publish_sensor_data)

        # Robot state
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0

        self.get_logger().info('Robot sensor publisher started')

    def publish_sensor_data(self):
        """Publish simulated sensor data"""
        # Create and publish laser scan
        scan_msg = LaserScan()
        scan_msg.header.stamp = self.get_clock().now().to_msg()
        scan_msg.header.frame_id = 'laser_frame'
        scan_msg.angle_min = -math.pi / 2
        scan_msg.angle_max = math.pi / 2
        scan_msg.angle_increment = math.pi / 180  # 1 degree
        scan_msg.time_increment = 0.0
        scan_msg.scan_time = 0.1
        scan_msg.range_min = 0.1
        scan_msg.range_max = 10.0

        # Simulate laser ranges with some obstacles
        num_ranges = 181  # From -90 to +90 degrees
        ranges = []
        for i in range(num_ranges):
            angle = scan_msg.angle_min + i * scan_msg.angle_increment
            # Simulate some obstacles at various distances
            distance = 2.0 + 0.5 * math.sin(angle * 3) + random.uniform(-0.2, 0.2)
            ranges.append(distance)

        scan_msg.ranges = ranges
        scan_msg.intensities = [1.0] * num_ranges

        self.scan_publisher.publish(scan_msg)

        # Create and publish odometry (simplified)
        odom_msg = Twist()
        odom_msg.linear.x = 0.0  # Current linear velocity
        odom_msg.angular.z = 0.0  # Current angular velocity
        self.odom_publisher.publish(odom_msg)

def main(args=None):
    rclpy.init(args=args)
    node = RobotSensorPublisher()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Step 3: Create the Robot Command Service

Create `robot_control_system/robot_control_system/command_service.py`:

```python
#!/usr/bin/env python3
"""
Robot Command Service - Provides immediate commands to the robot.
"""
import rclpy
from rclpy.node import Node
from std_srvs.srv import SetBool
from geometry_msgs.msg import Twist

class RobotCommandService(Node):
    def __init__(self):
        super().__init__('robot_command_service')

        # Create service
        self.srv = self.create_service(
            SetBool, 'emergency_stop', self.emergency_stop_callback)

        # Create publisher for velocity commands
        self.cmd_vel_publisher = self.create_publisher(Twist, 'cmd_vel', 10)

        self.stopped = False
        self.get_logger().info('Robot command service started')

    def emergency_stop_callback(self, request, response):
        """Handle emergency stop service calls"""
        if request.data:  # Stop requested
            self.stopped = True
            # Send zero velocity to stop robot
            stop_msg = Twist()
            self.cmd_vel_publisher.publish(stop_msg)
            response.success = True
            response.message = 'Emergency stop activated'
            self.get_logger().info('Emergency stop activated')
        else:  # Resume requested
            self.stopped = False
            response.success = True
            response.message = 'Emergency stop deactivated'
            self.get_logger().info('Emergency stop deactivated')

        return response

def main(args=None):
    rclpy.init(args=args)
    node = RobotCommandService()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Step 4: Create the Navigation Action Server

Create `robot_control_system/robot_control_system/navigation_action.py`:

```python
#!/usr/bin/env python3
"""
Robot Navigation Action - Handles long-running navigation tasks.
"""
import time
import rclpy
from rclpy.action import ActionServer
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from geometry_msgs.msg import Twist, Pose
from example_interfaces.action import Fibonacci  # Using Fibonacci as example

class RobotNavigationAction(Node):
    def __init__(self):
        super().__init__('robot_navigation_action')

        # Create action server
        self._action_server = ActionServer(
            self,
            Fibonacci,  # Using Fibonacci as example - in real app, you'd define your own
            'navigate_to_pose',
            self.execute_callback,
            callback_group=ReentrantCallbackGroup()
        )

        # Create publisher for velocity commands
        self.cmd_vel_publisher = self.create_publisher(Twist, 'cmd_vel', 10)

        self.get_logger().info('Robot navigation action server started')

    def execute_callback(self, goal_handle):
        """Execute navigation goal"""
        self.get_logger().info(f'Executing navigation goal: {goal_handle.request.order}')

        # Simulate navigation with feedback
        feedback_msg = Fibonacci.Feedback()
        feedback_msg.sequence = [0, 1]

        for i in range(1, goal_handle.request.order):
            # Check if goal was canceled
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.get_logger().info('Navigation goal canceled')

                # Stop robot
                stop_msg = Twist()
                self.cmd_vel_publisher.publish(stop_msg)

                return Fibonacci.Result()

            # Update feedback (simulating progress toward goal)
            feedback_msg.sequence.append(
                feedback_msg.sequence[i] + feedback_msg.sequence[i-1])
            goal_handle.publish_feedback(feedback_msg)

            # Send velocity command (simulating movement)
            cmd_msg = Twist()
            cmd_msg.linear.x = 0.5  # Move forward at 0.5 m/s
            cmd_msg.angular.z = 0.0  # No rotation
            self.cmd_vel_publisher.publish(cmd_msg)

            self.get_logger().info(f'Navigation progress: {feedback_msg.sequence}')

            # Simulate time for navigation
            time.sleep(0.5)

        # Stop robot when goal is reached
        stop_msg = Twist()
        self.cmd_vel_publisher.publish(stop_msg)

        # Complete successfully
        goal_handle.succeed()
        result = Fibonacci.Result()
        result.sequence = feedback_msg.sequence
        self.get_logger().info(f'Navigation completed: {result.sequence}')

        return result

def main(args=None):
    rclpy.init(args=args)
    node = RobotNavigationAction()

    try:
        executor = MultiThreadedExecutor()
        executor.add_node(node)
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Step 5: Create a Command Client

Create `robot_control_system/robot_control_system/command_client.py`:

```python
#!/usr/bin/env python3
"""
Robot Command Client - Tests the emergency stop service.
"""
import rclpy
from rclpy.node import Node
from std_srvs.srv import SetBool

class RobotCommandClient(Node):
    def __init__(self):
        super().__init__('robot_command_client')

        # Create client
        self.cli = self.create_client(SetBool, 'emergency_stop')

        # Wait for service
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for emergency_stop service...')

        self.req = SetBool.Request()

    def send_emergency_stop(self, stop):
        self.req.data = stop
        future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, future)

        if future.result() is not None:
            response = future.result()
            self.get_logger().info(f'Service response: {response.success}, {response.message}')
        else:
            self.get_logger().error('Service call failed')

def main(args=None):
    rclpy.init(args=args)
    client = RobotCommandClient()

    # Test emergency stop
    client.get_logger().info('Sending emergency stop ON')
    client.send_emergency_stop(True)

    # Wait a bit
    time.sleep(2)

    # Test resume
    client.get_logger().info('Sending emergency stop OFF')
    client.send_emergency_stop(False)

    client.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    import time
    main()
```

## Step 6: Update Package Configuration

Update `robot_control_system/setup.py`:

```python
from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'robot_control_system'

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
    maintainer_email='your.email@example.com',
    description='Robot control system with topics, services, and actions',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'sensor_publisher = robot_control_system.sensor_publisher:main',
            'command_service = robot_control_system.command_service:main',
            'navigation_action = robot_control_system.navigation_action:main',
            'command_client = robot_control_system.command_client:main',
        ],
    },
)
```

## Step 7: Build and Test

Build the package:

```bash
cd ~/ros2_ws
colcon build --packages-select robot_control_system
source install/setup.bash
```

### Test the System

1. **Start the sensor publisher**:
```bash
ros2 run robot_control_system sensor_publisher
```

2. **Start the command service**:
```bash
ros2 run robot_control_system command_service
```

3. **Start the navigation action**:
```bash
ros2 run robot_control_system navigation_action
```

4. **Test the command client**:
```bash
ros2 run robot_control_system command_client
```

5. **Test with command line tools**:
```bash
# List all nodes
ros2 node list

# List all topics
ros2 topic list

# Echo sensor data
ros2 topic echo /scan sensor_msgs/msg/LaserScan --field ranges[0]

# Call the service
ros2 service call /emergency_stop std_srvs/srv/SetBool "{data: true}"

# List all services
ros2 service list
```

## Understanding the System

This robot control system demonstrates three key ROS 2 communication patterns:

1. **Topics** (`/scan`, `/odom`, `/cmd_vel`): Continuous sensor data and velocity commands
2. **Services** (`/emergency_stop`): Immediate commands with synchronous response
3. **Actions** (`/navigate_to_pose`): Long-running navigation tasks with feedback

## Challenges

### Challenge 1: Add a Velocity Service
Create a service that allows setting the robot's velocity directly.

<details>
<summary>Hint</summary>

Create a custom service type `SetVelocity.srv`:
```
float64 linear_velocity
float64 angular_velocity
---
bool success
string message
```

Then implement the server and client.
</details>

### Challenge 2: Improve the Navigation Action
Modify the navigation action to accept actual goal poses instead of Fibonacci numbers.

<details>
<summary>Hint</summary>

Define a custom action type `NavigateToPose.action`:
```
geometry_msgs/Pose target_pose
---
bool success
string message
---
float32 distance_remaining
float32 progress_percentage
```
</details>

### Challenge 3: Create a Launch File
Create a launch file that starts all nodes at once.

<details>
<summary>Hint</summary>

Create `robot_control_system/launch/robot_system.launch.py`:
```python
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='robot_control_system',
            executable='sensor_publisher',
            name='sensor_publisher_node'
        ),
        Node(
            package='robot_control_system',
            executable='command_service',
            name='command_service_node'
        ),
        Node(
            package='robot_control_system',
            executable='navigation_action',
            name='navigation_action_node'
        ),
    ])
```
</details>

## Verification Checklist

- [ ] Sensor publisher publishes laser scan data
- [ ] Service responds to emergency stop calls
- [ ] Action server accepts and executes goals
- [ ] Client successfully calls the service
- [ ] All nodes appear in `ros2 node list`
- [ ] Topics appear in `ros2 topic list`
- [ ] Services appear in `ros2 service list`

## Common Issues

### Service Not Available
```bash
# Make sure the service node is running
ros2 run robot_control_system command_service
```

### Topic Names Don't Match
Check that publisher and subscriber use identical topic names and message types.

### Action Client Timeout
Ensure the action server is running before starting the client.

## Summary

In this exercise, you learned to:
- Create nodes that use all three ROS 2 communication patterns
- Implement publishers, services, and actions
- Test communication between nodes
- Use ROS 2 command-line tools for debugging

## Next Steps

Continue to [Week 3: Lifecycle and Launch](../week-03/lifecycle) to learn about managed node states and system orchestration.