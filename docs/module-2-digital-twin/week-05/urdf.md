---
sidebar_position: 2
---

# URDF Fundamentals

URDF (Unified Robot Description Format) is the standard format for describing robot models in ROS. In this section, you'll learn the core concepts of URDF and how to create accurate robot models.

## URDF Structure Overview

URDF is an XML-based format that describes robots through **links** (rigid bodies) and **joints** (connections between links). A robot model forms a tree structure with a single base link.

```xml
<?xml version="1.0"?>
<robot name="my_robot">
  <!-- Links define rigid bodies -->
  <link name="base_link">
    <!-- Visual properties for rendering -->
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <cylinder length="0.2" radius="0.2"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 1 1"/>
      </material>
    </visual>

    <!-- Collision properties for physics -->
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <cylinder length="0.2" radius="0.2"/>
      </geometry>
    </collision>

    <!-- Inertial properties for dynamics -->
    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="10.0"/>
      <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0"/>
    </inertial>
  </link>

  <!-- Joints connect links -->
  <joint name="base_to_wheel" type="continuous">
    <parent link="base_link"/>
    <child link="wheel_link"/>
    <origin xyz="0 0.2 0" rpy="0 0 0"/>
  </joint>

  <link name="wheel_link">
    <!-- ... wheel definition ... -->
  </link>
</robot>
```

## Links: The Building Blocks

### Link Components

A link contains three main components:

1. **Visual**: How the link appears in visualization
2. **Collision**: How the link interacts in physics simulation
3. **Inertial**: Physical properties for dynamics

### Visual Properties

```xml
<link name="base_link">
  <visual>
    <!-- Position and orientation of visual geometry -->
    <origin xyz="0 0 0.1" rpy="0 0 0"/>

    <!-- Shape of the visual geometry -->
    <geometry>
      <!-- Options: box, cylinder, sphere, mesh -->
      <cylinder length="0.2" radius="0.2"/>
    </geometry>

    <!-- Material properties -->
    <material name="red">
      <color rgba="1 0 0 1"/>
      <texture filename="path/to/texture.png"/>
    </material>
  </visual>
</link>
```

### Collision Properties

```xml
<link name="base_link">
  <collision>
    <!-- Position and orientation of collision geometry -->
    <origin xyz="0 0 0.1" rpy="0 0 0"/>

    <!-- Shape of the collision geometry -->
    <geometry>
      <!-- Usually simpler than visual geometry for performance -->
      <cylinder length="0.2" radius="0.2"/>
    </geometry>
  </collision>
</link>
```

### Inertial Properties

```xml
<link name="base_link">
  <inertial>
    <!-- Center of mass location -->
    <origin xyz="0 0 0" rpy="0 0 0"/>

    <!-- Mass of the link -->
    <mass value="10.0"/>

    <!-- Inertia tensor (3x3 matrix as 6 independent values) -->
    <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0"/>
  </inertial>
</link>
```

## Joint Types and Properties

### Joint Types

URDF supports several joint types:

```xml
<!-- Fixed joint (no movement) -->
<joint name="fixed_joint" type="fixed">
  <parent link="parent_link"/>
  <child link="child_link"/>
  <origin xyz="0 0 0.1" rpy="0 0 0"/>
</joint>

<!-- Continuous rotation (like wheel) -->
<joint name="continuous_joint" type="continuous">
  <parent link="base_link"/>
  <child link="wheel_link"/>
  <origin xyz="0 0.2 0" rpy="0 0 0"/>
  <axis xyz="0 0 1"/>  <!-- Rotation axis -->
</joint>

<!-- Revolute joint (limited rotation) -->
<joint name="revolute_joint" type="revolute">
  <parent link="base_link"/>
  <child link="arm_link"/>
  <origin xyz="0 0 0.5" rpy="0 0 0"/>
  <axis xyz="0 1 0"/>
  <limit lower="-1.57" upper="1.57" effort="10.0" velocity="1.0"/>
</joint>

<!-- Prismatic joint (linear motion) -->
<joint name="prismatic_joint" type="prismatic">
  <parent link="base_link"/>
  <child link="slide_link"/>
  <origin xyz="0 0 0" rpy="0 0 0"/>
  <axis xyz="1 0 0"/>
  <limit lower="0" upper="0.5" effort="20.0" velocity="0.5"/>
</joint>

<!-- Floating joint (6 DOF) -->
<joint name="floating_joint" type="floating">
  <parent link="world"/>
  <child link="free_link"/>
</joint>
```

### Joint Properties

```xml
<joint name="example_joint" type="revolute">
  <parent link="parent_link"/>
  <child link="child_link"/>

  <!-- Position and orientation of joint -->
  <origin xyz="0.1 0 0" rpy="0 0 0"/>

  <!-- Rotation axis -->
  <axis xyz="0 0 1"/>

  <!-- Joint limits -->
  <limit lower="-3.14" upper="3.14" effort="100.0" velocity="1.0"/>

  <!-- Joint dynamics -->
  <dynamics damping="0.1" friction="0.0"/>

  <!-- Joint safety limits -->
  <safety_controller k_position="10" k_velocity="10" soft_lower_limit="-3.0" soft_upper_limit="3.0"/>
</joint>
```

## Geometry Types

### Primitive Shapes

```xml
<!-- Box -->
<geometry>
  <box size="0.5 0.3 0.2"/>
</geometry>

<!-- Cylinder -->
<geometry>
  <cylinder radius="0.1" length="0.2"/>
</geometry>

<!-- Sphere -->
<geometry>
  <sphere radius="0.1"/>
</geometry>
```

### Mesh Geometry

```xml
<!-- Using mesh files -->
<geometry>
  <mesh filename="package://my_robot_description/meshes/link.dae" scale="1 1 1"/>
</geometry>

<!-- Mesh files should be in package share directory -->
<!-- Supported formats: DAE, STL, OBJ, etc. -->
```

## Materials and Colors

### Defining Materials

```xml
<!-- Define material once -->
<material name="blue">
  <color rgba="0 0 1 1"/>
</material>

<material name="red">
  <color rgba="1 0 0 1"/>
</material>

<material name="white">
  <color rgba="1 1 1 1"/>
</material>

<!-- Or with texture -->
<material name="textured">
  <color rgba="1 1 1 1"/>
  <texture filename="package://my_robot_description/materials/textures/texture.png"/>
</material>

<!-- Use material in link -->
<link name="base_link">
  <visual>
    <geometry>
      <cylinder radius="0.1" length="0.2"/>
    </geometry>
    <material name="blue"/>
  </visual>
</link>
```

## Complete Robot Example

Here's a complete example of a simple differential drive robot:

```xml
<?xml version="1.0"?>
<robot name="diff_drive_robot" xmlns:xacro="http://www.ros.org/wiki/xacro">
  <!-- Constants -->
  <xacro:property name="PI" value="3.1415926535897931"/>

  <!-- Base link -->
  <link name="base_link">
    <visual>
      <origin xyz="0 0 0.1" rpy="0 0 0"/>
      <geometry>
        <box size="0.5 0.3 0.2"/>
      </geometry>
      <material name="orange">
        <color rgba="1 0.5 0 1"/>
      </material>
    </visual>

    <collision>
      <origin xyz="0 0 0.1" rpy="0 0 0"/>
      <geometry>
        <box size="0.5 0.3 0.2"/>
      </geometry>
    </collision>

    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="5.0"/>
      <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.1"/>
    </inertial>
  </link>

  <!-- Left wheel -->
  <link name="left_wheel_link">
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.1" length="0.05"/>
      </geometry>
      <material name="black">
        <color rgba="0 0 0 1"/>
      </material>
    </visual>

    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.1" length="0.05"/>
      </geometry>
    </collision>

    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="0.5"/>
      <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.002"/>
    </inertial>
  </link>

  <!-- Right wheel -->
  <link name="right_wheel_link">
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.1" length="0.05"/>
      </geometry>
      <material name="black"/>
    </visual>

    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.1" length="0.05"/>
      </geometry>
    </collision>

    <inertial>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <mass value="0.5"/>
      <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.002"/>
    </inertial>
  </link>

  <!-- Left wheel joint -->
  <joint name="left_wheel_joint" type="continuous">
    <parent link="base_link"/>
    <child link="left_wheel_link"/>
    <origin xyz="0 0.175 0" rpy="-${PI/2} 0 0"/>
    <axis xyz="0 0 1"/>
  </joint>

  <!-- Right wheel joint -->
  <joint name="right_wheel_joint" type="continuous">
    <parent link="base_link"/>
    <child link="right_wheel_link"/>
    <origin xyz="0 -0.175 0" rpy="-${PI/2} 0 0"/>
    <axis xyz="0 0 1"/>
  </joint>

  <!-- Camera -->
  <link name="camera_link">
    <visual>
      <geometry>
        <box size="0.05 0.05 0.05"/>
      </geometry>
      <material name="gray">
        <color rgba="0.5 0.5 0.5 1"/>
      </material>
    </visual>

    <collision>
      <geometry>
        <box size="0.05 0.05 0.05"/>
      </geometry>
    </collision>

    <inertial>
      <mass value="0.1"/>
      <inertia ixx="0.0001" ixy="0.0" ixz="0.0" iyy="0.0001" iyz="0.0" izz="0.0001"/>
    </inertial>
  </link>

  <joint name="camera_joint" type="fixed">
    <parent link="base_link"/>
    <child link="camera_link"/>
    <origin xyz="0.2 0 0.15" rpy="0 0 0"/>
  </joint>

  <!-- Define materials -->
  <material name="orange">
    <color rgba="1 0.5 0 1"/>
  </material>
  <material name="black">
    <color rgba="0 0 0 1"/>
  </material>
  <material name="gray">
    <color rgba="0.5 0.5 0.5 1"/>
  </material>
</robot>
```

## Coordinate Systems and Transforms

### ROS Coordinate Conventions

- **Right-handed system**: thumb = X, index = Y, middle = Z
- **REP-103**: X-forward, Y-left, Z-up
- **Joint angles**: Right-hand rule (thumb along axis, curl shows positive rotation)

### Origin and Pose

```xml
<!-- Origin defines position and orientation -->
<origin xyz="0.1 0 0.2" rpy="0 0 1.57"/>
<!-- xyz: position in parent frame -->
<!-- rpy: roll, pitch, yaw in radians -->
```

## Working with URDF

### Validating URDF

```bash
# Check if URDF is well-formed
check_urdf my_robot.urdf

# Show robot structure
urdf_to_graphiz my_robot.urdf

# Visualize in RViz
ros2 run rviz2 rviz2
# Add RobotModel display, set Topic to /robot_description
```

### Loading URDF in ROS 2

```python
#!/usr/bin/env python3
"""
URDF loader example
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
import math

class URDFLoader(Node):
    def __init__(self):
        super().__init__('urdf_loader')

        # Joint state publisher
        self.joint_pub = self.create_publisher(JointState, 'joint_states', 10)

        # TF broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)

        # Timer for publishing transforms
        self.timer = self.create_timer(0.1, self.publish_transforms)

        self.get_logger().info('URDF loader started')

    def publish_transforms(self):
        """Publish joint states and transforms"""
        # Create joint state message
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = ['left_wheel_joint', 'right_wheel_joint']
        msg.position = [0.0, 0.0]  # Current joint positions

        self.joint_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = URDFLoader()

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

## Best Practices

### 1. Proper Mass Properties

```xml
<!-- Calculate mass based on material density -->
<inertial>
  <mass value="2.0"/>  <!-- 2kg for a plastic wheel -->
  <!-- Use proper inertia values for the shape -->
  <inertia ixx="0.001" ixy="0.0" ixz="0.0"
           iyy="0.001" iyz="0.0" izz="0.002"/>
</inertial>
```

### 2. Realistic Joint Limits

```xml
<!-- Set realistic limits based on hardware -->
<limit lower="-1.57" upper="1.57" effort="10.0" velocity="1.0"/>
```

### 3. Separate Visual and Collision

```xml
<!-- Use simpler collision geometry for performance -->
<collision>
  <geometry>
    <cylinder radius="0.1" length="0.2"/>  <!-- Simple -->
  </geometry>
</collision>

<visual>
  <geometry>
    <mesh filename="complex_visual.dae"/>  <!-- Detailed -->
  </geometry>
</visual>
```

## Common Issues and Troubleshooting

### 1. Invalid URDF

```bash
# Error: Multiple root links
# Solution: Ensure single base link with no parent

# Error: Joint has no parent/child
# Solution: Verify all joints connect existing links
```

### 2. Inertial Issues

```xml
<!-- Error: Zero inertia values -->
<inertial>
  <mass value="1.0"/>
  <inertia ixx="0.0" ixy="0.0" ixz="0.0" iyy="0.0" iyz="0.0" izz="0.0"/>  <!-- WRONG -->
</inertial>

<!-- Correct: Non-zero inertia values -->
<inertial>
  <mass value="1.0"/>
  <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.1"/>  <!-- CORRECT -->
</inertial>
```

### 3. Coordinate Frame Issues

```xml
<!-- Ensure consistent coordinate frames -->
<!-- Base link should be at robot's center -->
<!-- Joint origins should be relative to parent link -->
```

## Advanced URDF Features

### Transmission Elements

```xml
<!-- Define how joints connect to actuators -->
<transmission name="left_wheel_trans">
  <type>transmission_interface/SimpleTransmission</type>
  <joint name="left_wheel_joint">
    <hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface>
  </joint>
  <actuator name="left_wheel_motor">
    <hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface>
    <mechanicalReduction>1</mechanicalReduction>
  </actuator>
</transmission>
```

### Gazebo-Specific Elements

```xml
<!-- Add Gazebo-specific properties -->
<gazebo reference="base_link">
  <material>Gazebo/Orange</material>
  <mu1>0.2</mu1>
  <mu2>0.2</mu2>
</gazebo>

<!-- Add sensors -->
<gazebo reference="camera_link">
  <sensor name="camera" type="camera">
    <pose>0 0 0 0 0 0</pose>
    <camera>
      <horizontal_fov>1.047</horizontal_fov>
    </camera>
  </sensor>
</gazebo>
```

## Next Steps

Now that you understand URDF fundamentals, continue to [Xacro and Advanced Modeling](../week-05/xacro) to learn about creating more sophisticated and modular robot models.

## Exercises

1. Create a URDF model of a simple robot with at least 3 links
2. Add proper inertial properties to your model
3. Validate your URDF using check_urdf
4. Visualize your robot in RViz