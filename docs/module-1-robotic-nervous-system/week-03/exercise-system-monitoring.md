---
sidebar_position: 3
---

# Exercise: System Monitoring Dashboard

In this comprehensive exercise, you'll create a complete system monitoring dashboard that demonstrates lifecycle management, system monitoring, and launch orchestration. This will tie together everything you've learned in Module 1.

## Objective

Create a robot system monitoring dashboard with:
1. **Lifecycle nodes** for different system components
2. **Monitoring nodes** that track system health
3. **Launch system** to orchestrate the entire system
4. **Visualization tools** to display system status

## Prerequisites

- Complete Week 1-3 lessons
- ROS 2 workspace set up (`~/ros2_ws`)
- Basic Python programming knowledge
- Understanding of lifecycle nodes and launch systems

## Step 1: Create the Package

```bash
cd ~/ros2_ws/src

ros2 pkg create --build-type ament_python system_monitoring \
    --dependencies rclpy rclpy_lifecycle std_msgs sensor_msgs geometry_msgs launch launch_ros
```

## Step 2: Create Lifecycle Component Nodes

Create `system_monitoring/system_monitoring/sensor_component.py`:

```python
#!/usr/bin/env python3
"""
Lifecycle sensor component - Simulates a sensor that can be configured, activated, and deactivated.
"""
import rclpy
from rclpy.lifecycle import LifecycleNode, LifecycleState, TransitionCallbackReturn
from rclpy.lifecycle import Publisher
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool
import math
import random

class SensorComponent(LifecycleNode):
    def __init__(self):
        super().__init__('sensor_component')
        self.scan_publisher = None
        self.status_publisher = None
        self.timer = None
        self.is_operational = False

        self.get_logger().info('Sensor component created, waiting for configuration')

    def on_configure(self, state):
        """Configure the sensor component"""
        self.get_logger().info(f'Configuring sensor from state: {state.label}')

        # Create publishers
        self.scan_publisher = self.create_publisher(LaserScan, 'sensor_scan', 10)
        self.status_publisher = self.create_publisher(Bool, 'sensor_status', 10)

        # Create timer but don't start it
        self.timer = self.create_timer(0.1, self.scan_callback)
        self.timer.cancel()

        # Simulate sensor configuration
        self.sensor_configured = True
        self.get_logger().info('Sensor component configured successfully')

        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state):
        """Activate the sensor component"""
        self.get_logger().info(f'Activating sensor from state: {state.label}')

        # Activate publishers
        self.scan_publisher.on_activate()
        self.status_publisher.on_activate()

        # Start timer
        self.timer.reset()
        self.is_operational = True

        self.get_logger().info('Sensor component activated')
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state):
        """Deactivate the sensor component"""
        self.get_logger().info(f'Deactivating sensor from state: {state.label}')

        # Pause publishers
        self.scan_publisher.on_deactivate()
        self.status_publisher.on_deactivate()

        # Stop timer
        self.timer.cancel()
        self.is_operational = False

        self.get_logger().info('Sensor component deactivated')
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state):
        """Clean up the sensor component"""
        self.get_logger().info(f'Cleaning up sensor from state: {state.label}')

        # Destroy timer
        self.timer.destroy()
        self.timer = None

        # Destroy publishers
        self.scan_publisher.destroy()
        self.status_publisher.destroy()

        self.scan_publisher = None
        self.status_publisher = None

        self.get_logger().info('Sensor component cleaned up')
        return TransitionCallbackReturn.SUCCESS

    def scan_callback(self):
        """Generate simulated sensor data"""
        if self.scan_publisher and self.scan_publisher.is_activated:
            # Create laser scan message
            scan_msg = LaserScan()
            scan_msg.header.stamp = self.get_clock().now().to_msg()
            scan_msg.header.frame_id = 'sensor_frame'
            scan_msg.angle_min = -math.pi / 2
            scan_msg.angle_max = math.pi / 2
            scan_msg.angle_increment = math.pi / 180  # 1 degree
            scan_msg.time_increment = 0.0
            scan_msg.scan_time = 0.1
            scan_msg.range_min = 0.1
            scan_msg.range_max = 10.0

            # Simulate laser ranges
            num_ranges = 181
            ranges = []
            for i in range(num_ranges):
                angle = scan_msg.angle_min + i * scan_msg.angle_increment
                distance = 3.0 + 0.5 * math.sin(angle * 2) + random.uniform(-0.1, 0.1)
                ranges.append(distance)

            scan_msg.ranges = ranges
            scan_msg.intensities = [1.0] * num_ranges

            self.scan_publisher.publish(scan_msg)

            # Publish status
            status_msg = Bool()
            status_msg.data = True
            self.status_publisher.publish(status_msg)

def main(args=None):
    rclpy.init(args=args)
    node = SensorComponent()

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

Create `system_monitoring/system_monitoring/controller_component.py`:

```python
#!/usr/bin/env python3
"""
Lifecycle controller component - Simulates a robot controller.
"""
import rclpy
from rclpy.lifecycle import LifecycleNode, LifecycleState, TransitionCallbackReturn
from rclpy.lifecycle import Publisher, Subscription
from geometry_msgs.msg import Twist, Vector3
from std_msgs.msg import Bool

class ControllerComponent(LifecycleNode):
    def __init__(self):
        super().__init__('controller_component')
        self.cmd_vel_publisher = None
        self.cmd_sub = None
        self.status_publisher = None
        self.is_operational = False

        self.get_logger().info('Controller component created, waiting for configuration')

    def on_configure(self, state):
        """Configure the controller component"""
        self.get_logger().info(f'Configuring controller from state: {state.label}')

        # Create publishers
        self.cmd_vel_publisher = self.create_publisher(Twist, 'cmd_vel', 10)
        self.status_publisher = self.create_publisher(Bool, 'controller_status', 10)

        # Create subscription
        self.cmd_sub = self.create_subscription(
            Twist, 'command', self.command_callback, 10)

        # Simulate controller configuration
        self.controller_configured = True
        self.get_logger().info('Controller component configured successfully')

        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state):
        """Activate the controller component"""
        self.get_logger().info(f'Activating controller from state: {state.label}')

        # Activate publishers
        self.cmd_vel_publisher.on_activate()
        self.status_publisher.on_activate()

        self.is_operational = True

        self.get_logger().info('Controller component activated')
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state):
        """Deactivate the controller component"""
        self.get_logger().info(f'Deactivating controller from state: {state.label}')

        # Pause publishers
        self.cmd_vel_publisher.on_deactivate()
        self.status_publisher.on_deactivate()

        self.is_operational = False

        # Stop robot
        stop_msg = Twist()
        stop_msg.linear = Vector3(x=0.0, y=0.0, z=0.0)
        stop_msg.angular = Vector3(x=0.0, y=0.0, z=0.0)
        self.cmd_vel_publisher.publish(stop_msg)

        self.get_logger().info('Controller component deactivated')
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state):
        """Clean up the controller component"""
        self.get_logger().info(f'Cleaning up controller from state: {state.label}')

        # Destroy publishers and subscription
        self.cmd_vel_publisher.destroy()
        self.status_publisher.destroy()
        self.cmd_sub.destroy()

        self.cmd_vel_publisher = None
        self.status_publisher = None
        self.cmd_sub = None

        self.get_logger().info('Controller component cleaned up')
        return TransitionCallbackReturn.SUCCESS

    def command_callback(self, msg):
        """Handle incoming commands"""
        if self.is_operational:
            # Forward command to robot
            self.cmd_vel_publisher.publish(msg)
            self.get_logger().info(f'Forwarding command: linear={msg.linear.x}, angular={msg.angular.z}')

def main(args=None):
    rclpy.init(args=args)
    node = ControllerComponent()

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

## Step 3: Create System Monitor Node

Create `system_monitoring/system_monitoring/system_monitor.py`:

```python
#!/usr/bin/env python3
"""
System monitor - Monitors the health and status of all system components.
"""
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist

class SystemMonitor(Node):
    def __init__(self):
        super().__init__('system_monitor')

        # Component status tracking
        self.sensor_status = False
        self.controller_status = False
        self.navigation_status = False

        # Create subscriptions to monitor component status
        self.sensor_status_sub = self.create_subscription(
            Bool, 'sensor_status', self.sensor_status_callback, 10)
        self.controller_status_sub = self.create_subscription(
            Bool, 'controller_status', self.controller_status_callback, 10)

        # Create publisher for overall system status
        self.status_pub = self.create_publisher(String, 'system_status', 10)

        # Create timer for periodic status checks
        self.status_timer = self.create_timer(2.0, self.check_system_status)

        self.get_logger().info('System monitor started')

    def sensor_status_callback(self, msg):
        """Update sensor status"""
        self.sensor_status = msg.data
        self.get_logger().debug(f'Sensor status updated: {self.sensor_status}')

    def controller_status_callback(self, msg):
        """Update controller status"""
        self.controller_status = msg.data
        self.get_logger().debug(f'Controller status updated: {self.controller_status}')

    def check_system_status(self):
        """Check and publish overall system status"""
        # Determine overall system health
        all_operational = all([self.sensor_status, self.controller_status])

        status_msg = String()
        if all_operational:
            status_msg.data = 'SYSTEM_OK'
        elif not any([self.sensor_status, self.controller_status]):
            status_msg.data = 'SYSTEM_DOWN'
        else:
            issues = []
            if not self.sensor_status:
                issues.append('SENSOR_DOWN')
            if not self.controller_status:
                issues.append('CONTROLLER_DOWN')
            status_msg.data = f'PARTIAL_FAILURE: {",".join(issues)}'

        self.status_pub.publish(status_msg)
        self.get_logger().info(f'System status: {status_msg.data}')

def main(args=None):
    rclpy.init(args=args)
    node = SystemMonitor()

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

## Step 4: Create Health Dashboard Node

Create `system_monitoring/system_monitoring/health_dashboard.py`:

```python
#!/usr/bin/env python3
"""
Health dashboard - Provides detailed health information for visualization.
"""
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool, Int32
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
import time

class HealthDashboard(Node):
    def __init__(self):
        super().__init__('health_dashboard')

        # Health metrics
        self.metrics = {
            'sensor_health': 100,  # 0-100%
            'controller_health': 100,
            'communication_health': 100,
            'overall_health': 100
        }

        # Component status
        self.component_status = {
            'sensor': 'unknown',
            'controller': 'unknown',
            'navigation': 'unknown'
        }

        # Create publishers for health metrics
        self.health_pub = self.create_publisher(String, 'health_metrics', 10)
        self.status_pub = self.create_publisher(String, 'component_status', 10)

        # Create subscriptions
        self.system_status_sub = self.create_subscription(
            String, 'system_status', self.system_status_callback, 10)

        # Create timer for health updates
        self.health_timer = self.create_timer(1.0, self.update_health_metrics)

        self.get_logger().info('Health dashboard started')

    def system_status_callback(self, msg):
        """Update dashboard based on system status"""
        status = msg.data
        if status == 'SYSTEM_OK':
            self.metrics['overall_health'] = 100
        elif 'PARTIAL_FAILURE' in status:
            self.metrics['overall_health'] = 60
        elif status == 'SYSTEM_DOWN':
            self.metrics['overall_health'] = 20

    def update_health_metrics(self):
        """Update and publish health metrics"""
        # Simulate health metric updates
        for key in self.metrics:
            if key == 'overall_health':
                continue
            # Simulate slight variations in health
            self.metrics[key] = max(80, min(100, self.metrics[key] + (0.5 - 0.1 * (100 - self.metrics[key]))))

        # Create and publish health metrics message
        health_msg = String()
        health_msg.data = f"{{'sensor': {self.metrics['sensor_health']:.1f}, 'controller': {self.metrics['controller_health']:.1f}, 'comm': {self.metrics['communication_health']:.1f}, 'overall': {self.metrics['overall_health']:.1f}}}"
        self.health_pub.publish(health_msg)

        # Create and publish component status message
        status_msg = String()
        status_msg.data = f"{{'sensor': '{self.component_status['sensor']}', 'controller': '{self.component_status['controller']}', 'navigation': '{self.component_status['navigation']}'}}"
        self.status_pub.publish(status_msg)

        self.get_logger().info(f'Health metrics: {health_msg.data}')

def main(args=None):
    rclpy.init(args=args)
    node = HealthDashboard()

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

## Step 5: Create Launch Files

Create `system_monitoring/launch/robot_system.launch.py`:

```python
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition

def generate_launch_description():
    # Declare launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')
    enable_monitoring = LaunchConfiguration('enable_monitoring', default='true')

    return LaunchDescription([
        # Declare launch arguments
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation time if true'
        ),
        DeclareLaunchArgument(
            'enable_monitoring',
            default_value='true',
            description='Enable system monitoring'
        ),

        # Launch sensor component (lifecycle)
        Node(
            package='system_monitoring',
            executable='sensor_component',
            name='sensor_component',
            parameters=[
                {'use_sim_time': use_sim_time}
            ],
            output='screen'
        ),

        # Launch controller component (lifecycle)
        Node(
            package='system_monitoring',
            executable='controller_component',
            name='controller_component',
            parameters=[
                {'use_sim_time': use_sim_time}
            ],
            output='screen'
        ),

        # Launch system monitor (only if enabled)
        Node(
            package='system_monitoring',
            executable='system_monitor',
            name='system_monitor',
            parameters=[
                {'use_sim_time': use_sim_time}
            ],
            output='screen',
            condition=IfCondition(enable_monitoring)
        ),

        # Launch health dashboard (only if enabled)
        Node(
            package='system_monitoring',
            executable='health_dashboard',
            name='health_dashboard',
            parameters=[
                {'use_sim_time': use_sim_time}
            ],
            output='screen',
            condition=IfCondition(enable_monitoring)
        )
    ])
```

Create `system_monitoring/launch/lifecycle_manager.launch.py`:

```python
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation time if true'
        ),

        # Lifecycle manager node
        Node(
            package='system_monitoring',
            executable='system_monitor',
            name='system_monitor',
            parameters=[
                {'use_sim_time': use_sim_time}
            ],
            output='screen'
        ),

        Node(
            package='system_monitoring',
            executable='health_dashboard',
            name='health_dashboard',
            parameters=[
                {'use_sim_time': use_sim_time}
            ],
            output='screen'
        )
    ])
```

## Step 6: Update Package Configuration

Update `system_monitoring/setup.py`:

```python
from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'system_monitoring'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your.email@example.com',
    description='System monitoring dashboard for robot systems',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'sensor_component = system_monitoring.sensor_component:main',
            'controller_component = system_monitoring.controller_component:main',
            'system_monitor = system_monitoring.system_monitor:main',
            'health_dashboard = system_monitoring.health_dashboard:main',
        ],
    },
)
```

## Step 7: Build and Test

Build the package:

```bash
cd ~/ros2_ws
colcon build --packages-select system_monitoring
source install/setup.bash
```

### Test the System

1. **Start the entire system with launch file**:
```bash
ros2 launch system_monitoring robot_system.launch.py
```

2. **In another terminal, check lifecycle nodes**:
```bash
# List all nodes
ros2 node list

# Check lifecycle states
ros2 lifecycle list /sensor_component
ros2 lifecycle list /controller_component

# Get lifecycle states
ros2 lifecycle get /sensor_component
ros2 lifecycle get /controller_component
```

3. **Monitor system status**:
```bash
# Echo system status
ros2 topic echo /system_status

# Echo health metrics
ros2 topic echo /health_metrics

# Echo component status
ros2 topic echo /component_status
```

4. **Control lifecycle nodes manually**:
```bash
# Configure sensor component
ros2 lifecycle configure /sensor_component

# Activate sensor component
ros2 lifecycle activate /sensor_component

# Configure controller component
ros2 lifecycle configure /controller_component

# Activate controller component
ros2 lifecycle activate /controller_component
```

5. **Test system management**:
```bash
# Get detailed node info
ros2 node info /system_monitor

# Check parameters
ros2 param list /health_dashboard

# Monitor topics
ros2 topic hz /sensor_scan
```

## Understanding the System

This system monitoring dashboard demonstrates:

1. **Lifecycle Management**: Components can be configured, activated, deactivated, and cleaned up
2. **System Monitoring**: Real-time monitoring of component health and status
3. **Launch Orchestration**: Starting multiple nodes with a single command
4. **Health Visualization**: Metrics and status information for system health

## Challenges

### Challenge 1: Add Navigation Component
Add a navigation lifecycle component that simulates path planning and execution.

<details>
<summary>Hint</summary>

Create a `navigation_component.py` that:
- Has lifecycle callbacks for configuring navigation parameters
- Publishes simulated navigation status
- Integrates with the system monitor
</details>

### Challenge 2: Create Custom Dashboard
Create a custom RViz2 configuration file to visualize system health.

<details>
<summary>Hint</summary>

Create `config/system_dashboard.rviz` with:
- Status displays for health metrics
- Marker displays for component status
- Text displays for system information
</details>

### Challenge 3: Add Parameter Configuration
Add parameter configuration files for different system profiles.

<details>
<summary>Hint</summary>

Create `config/` directory with:
- `robot_config.yaml` for robot-specific parameters
- `simulation_config.yaml` for simulation parameters
- `production_config.yaml` for production parameters
</details>

### Challenge 4: Implement Emergency Procedures
Add emergency procedures to the system monitor.

<details>
<summary>Hint</summary>

Add services to:
- Emergency stop all components
- Reset system to safe state
- Log emergency events
</details>

## Verification Checklist

- [ ] Lifecycle nodes can be configured and activated
- [ ] System monitor tracks component status
- [ ] Health dashboard shows metrics
- [ ] Launch file starts all components
- [ ] Topics are published and received correctly
- [ ] All nodes appear in `ros2 node list`
- [ ] System status updates correctly
- [ ] Health metrics update periodically

## Common Issues

### Lifecycle Node Issues
```bash
# Check if lifecycle services are available
ros2 service list | grep lifecycle

# Verify node is running
ros2 node list
```

### Topic Connection Issues
```bash
# Check topic connections
ros2 topic info /system_status

# Verify message types
ros2 topic type /health_metrics
```

### Launch File Issues
```bash
# Test launch file without running
ros2 launch system_monitoring robot_system.launch.py --dry-run

# Launch with verbose output
ros2 launch system_monitoring robot_system.launch.py --log-level debug
```

## Summary

In this exercise, you learned to:
- Create lifecycle nodes for different system components
- Implement system monitoring and health tracking
- Use launch files to orchestrate complex systems
- Monitor and visualize system status
- Apply best practices for system architecture

## Next Steps

You have now completed Module 1: The Robotic Nervous System! Continue to [Module 2: Building the Digital Twin](../../module-2-digital-twin/week-04/introduction) to learn about simulation and digital twin technologies that will allow you to test and develop your robot systems in virtual environments.