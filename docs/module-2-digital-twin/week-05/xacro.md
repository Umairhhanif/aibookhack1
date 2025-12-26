---
sidebar_position: 3
---

# Xacro and Advanced Modeling

**Xacro** (XML Macros) is an extension to URDF that provides powerful features for creating complex, modular, and maintainable robot models. Xacro allows you to use macros, properties, and expressions to simplify robot description files.

## Xacro Overview

Xacro extends URDF with:
- **Macros**: Reusable components
- **Properties**: Parameterized values
- **Expressions**: Mathematical calculations
- **Includes**: Modular file organization
- **Loops**: Generate repetitive structures

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="advanced_robot">
  <!-- Xacro properties -->
  <xacro:property name="wheel_radius" value="0.1"/>
  <xacro:property name="wheel_width" value="0.05"/>

  <!-- Xacro macro -->
  <xacro:macro name="wheel" params="prefix parent xyz">
    <link name="${prefix}_wheel">
      <visual>
        <origin xyz="0 0 0" rpy="0 0 0"/>
        <geometry>
          <cylinder radius="${wheel_radius}" length="${wheel_width}"/>
        </geometry>
      </visual>
      <collision>
        <origin xyz="0 0 0" rpy="0 0 0"/>
        <geometry>
          <cylinder radius="${wheel_radius}" length="${wheel_width}"/>
        </geometry>
      </collision>
      <inertial>
        <mass value="0.5"/>
        <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.002"/>
      </inertial>
    </link>

    <joint name="${prefix}_wheel_joint" type="continuous">
      <parent link="${parent}"/>
      <child link="${prefix}_wheel"/>
      <origin xyz="${xyz}" rpy="0 0 0"/>
      <axis xyz="0 0 1"/>
    </joint>
  </xacro:macro>

  <!-- Use the macro -->
  <xacro:wheel prefix="left" parent="base_link" xyz="0 0.2 0"/>
  <xacro:wheel prefix="right" parent="base_link" xyz="0 -0.2 0"/>
</robot>
```

## Xacro Properties

### Defining Properties

Properties in Xacro work like variables:

```xml
<!-- Define constants -->
<xacro:property name="PI" value="3.1415926535897931"/>
<xacro:property name="DEG_TO_RAD" value="0.017453292519943295"/>

<!-- Define robot dimensions -->
<xacro:property name="base_width" value="0.3"/>
<xacro:property name="base_length" value="0.5"/>
<xacro:property name="base_height" value="0.2"/>

<!-- Define physical properties -->
<xacro:property name="robot_mass" value="10.0"/>
<xacro:property name="wheel_radius" value="0.1"/>
<xacro:property name="wheel_width" value="0.05"/>
```

### Property Scoping

```xml
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="scoped_robot">
  <!-- Global property -->
  <xacro:property name="global_param" value="1.0"/>

  <xacro:macro name="test_macro" params="input_param">
    <!-- Local property -->
    <xacro:property name="local_param" value="${input_param * 2}"/>

    <link name="test_link">
      <inertial>
        <mass value="${local_param}"/>
        <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0"/>
      </inertial>
    </link>
  </xacro:macro>
</robot>
```

## Xacro Expressions

### Mathematical Expressions

Xacro supports mathematical operations in expressions:

```xml
<xacro:property name="wheel_separation" value="0.4"/>
<xacro:property name="wheel_radius" value="0.1"/>

<xacro:macro name="wheel_mount" params="side">
  <link name="wheel_mount_${side}">
    <visual>
      <geometry>
        <box size="${wheel_radius*2} ${wheel_separation/2} 0.05"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <box size="${wheel_radius*2} ${wheel_separation/2} 0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="${wheel_radius * 10}"/>
      <inertia ixx="1.0" iyy="1.0" izz="1.0"/>
    </inertial>
  </link>
</xacro:macro>
```

### Conditional Expressions

```xml
<xacro:property name="robot_type" value="differential"/>

<xacro:macro name="robot_base" params="type">
  <link name="base_link">
    <visual>
      <geometry>
        <xacro:if value="${type == 'differential'}">
          <box size="0.5 0.3 0.2"/>
        </xacro:if>
        <xacro:unless value="${type == 'differential'}">
          <cylinder radius="0.25" length="0.2"/>
        </xacro:unless>
      </geometry>
    </visual>
  </link>
</xacro:macro>
```

## Xacro Macros

### Basic Macro Definition

```xml
<!-- Simple macro with parameters -->
<xacro:macro name="simple_wheel" params="prefix parent xyz">
  <link name="${prefix}_wheel_link">
    <visual>
      <geometry>
        <cylinder radius="0.1" length="0.05"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.1" length="0.05"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.5"/>
      <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.002"/>
    </inertial>
  </link>

  <joint name="${prefix}_wheel_joint" type="continuous">
    <parent link="${parent}"/>
    <child link="${prefix}_wheel_link"/>
    <origin xyz="${xyz}" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
  </joint>
</xacro:macro>
```

### Complex Macro with Nested Elements

```xml
<xacro:macro name="sensor_mount" params="name parent xyz rpy sensor_type:=camera">
  <!-- Mount link -->
  <link name="${name}_mount">
    <visual>
      <geometry>
        <box size="0.05 0.05 0.05"/>
      </geometry>
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

  <!-- Mount joint -->
  <joint name="${name}_mount_joint" type="fixed">
    <parent link="${parent}"/>
    <child link="${name}_mount"/>
    <origin xyz="${xyz}" rpy="${rpy}"/>
  </joint>

  <!-- Sensor link -->
  <link name="${name}_link">
    <visual>
      <geometry>
        <box size="0.03 0.03 0.03"/>
      </geometry>
    </visual>
    <collision>
      <geometry>
        <box size="0.03 0.03 0.03"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="0.05"/>
      <inertia ixx="0.00005" ixy="0.0" ixz="0.0" iyy="0.00005" iyz="0.0" izz="0.00005"/>
    </inertial>
  </link>

  <!-- Sensor joint -->
  <joint name="${name}_joint" type="fixed">
    <parent link="${name}_mount"/>
    <child link="${name}_link"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
  </joint>

  <!-- Gazebo-specific sensor configuration -->
  <gazebo reference="${name}_link">
    <xacro:if value="${sensor_type == 'camera'}">
      <sensor name="${name}" type="camera">
        <camera name="head">
          <horizontal_fov>1.047</horizontal_fov>
          <image>
            <width>640</width>
            <height>480</height>
          </image>
        </camera>
      </sensor>
    </xacro:if>
    <xacro:if value="${sensor_type == 'lidar'}">
      <sensor name="${name}" type="ray">
        <ray>
          <scan>
            <horizontal>
              <samples>720</samples>
              <min_angle>-3.14159</min_angle>
              <max_angle>3.14159</max_angle>
            </horizontal>
          </scan>
        </ray>
      </sensor>
    </xacro:if>
  </gazebo>
</xacro:macro>
```

## Modular Design with Includes

### Creating Reusable Components

Create `urdf/wheel.xacro`:
```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro">
  <xacro:macro name="wheel_macro" params="prefix parent xyz radius width mass:=0.5">
    <link name="${prefix}_wheel">
      <visual>
        <origin xyz="0 0 0" rpy="${PI/2} 0 0"/>
        <geometry>
          <cylinder radius="${radius}" length="${width}"/>
        </geometry>
        <material name="black">
          <color rgba="0 0 0 1"/>
        </material>
      </visual>

      <collision>
        <origin xyz="0 0 0" rpy="${PI/2} 0 0"/>
        <geometry>
          <cylinder radius="${radius}" length="${width}"/>
        </geometry>
      </collision>

      <inertial>
        <mass value="${mass}"/>
        <inertia ixx="${0.5*mass*radius*radius}" ixy="0.0" ixz="0.0"
                 iyy="${0.25*mass*radius*radius + mass*width*width/12}" iyz="0.0"
                 izz="${0.5*mass*radius*radius}"/>
      </inertial>
    </link>

    <joint name="${prefix}_wheel_joint" type="continuous">
      <parent link="${parent}"/>
      <child link="${prefix}_wheel"/>
      <origin xyz="${xyz}" rpy="0 0 0"/>
      <axis xyz="0 0 1"/>
    </joint>
  </xacro:macro>
</robot>
```

### Using Includes

Main robot file `robot.urdf.xacro`:
```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="modular_robot">
  <!-- Include wheel macro -->
  <xacro:include filename="$(find my_robot_description)/urdf/wheel.xacro"/>

  <!-- Define constants -->
  <xacro:property name="PI" value="3.1415926535897931"/>
  <xacro:property name="wheel_radius" value="0.1"/>
  <xacro:property name="wheel_width" value="0.05"/>
  <xacro:property name="wheel_separation" value="0.4"/>

  <!-- Base link -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.5 0.3 0.2"/>
      </geometry>
      <material name="orange">
        <color rgba="1 0.5 0 1"/>
      </material>
    </visual>

    <collision>
      <geometry>
        <box size="0.5 0.3 0.2"/>
      </geometry>
    </collision>

    <inertial>
      <mass value="5.0"/>
      <inertia ixx="0.1" ixy="0.0" ixz="0.0" iyy="0.1" iyz="0.0" izz="0.1"/>
    </inertial>
  </link>

  <!-- Use wheel macros -->
  <xacro:wheel_macro prefix="left" parent="base_link"
                     xyz="0 ${wheel_separation/2} 0"
                     radius="${wheel_radius}" width="${wheel_width}"/>

  <xacro:wheel_macro prefix="right" parent="base_link"
                     xyz="0 ${-wheel_separation/2} 0"
                     radius="${wheel_radius}" width="${wheel_width}"/>
</robot>
```

## Advanced Xacro Features

### Looping and Iteration

```xml
<xacro:macro name="create_array" params="count spacing prefix parent">
  <xacro:property name="i" value="0"/>
  <xacro:while value="${i &lt; count}">
    <link name="${prefix}_${i}">
      <visual>
        <geometry>
          <box size="0.05 0.05 0.05"/>
        </geometry>
      </visual>
      <inertial>
        <mass value="0.1"/>
        <inertia ixx="0.0001" ixy="0.0" ixz="0.0" iyy="0.0001" iyz="0.0" izz="0.0001"/>
      </inertial>
    </link>

    <joint name="${prefix}_${i}_joint" type="fixed">
      <parent link="${parent}"/>
      <child link="${prefix}_${i}"/>
      <origin xyz="${i * spacing} 0 0" rpy="0 0 0"/>
    </joint>

    <xacro:property name="i" value="${i + 1}"/>
  </xacro:while>
</xacro:macro>
```

### Conditional Macros

```xml
<xacro:macro name="arm_with_gripper" params="prefix parent config:=standard">
  <xacro:if value="${config == 'standard'}">
    <!-- Standard arm configuration -->
    <xacro:macro name="arm_segment" params="seg_num length">
      <link name="${prefix}_link_${seg_num}">
        <visual>
          <geometry>
            <cylinder radius="0.05" length="${length}"/>
          </geometry>
        </visual>
        <inertial>
          <mass value="${length * 2}"/>
          <inertia ixx="1.0" iyy="1.0" izz="1.0"/>
        </inertial>
      </link>
    </xacro:macro>
  </xacro:if>

  <xacro:unless value="${config == 'standard'}">
    <!-- Custom arm configuration -->
    <xacro:macro name="custom_arm_segment" params="seg_num length">
      <link name="${prefix}_link_${seg_num}">
        <visual>
          <geometry>
            <box size="${length} 0.1 0.1"/>
          </geometry>
        </visual>
        <inertial>
          <mass value="${length * 1.5}"/>
          <inertia ixx="1.0" iyy="1.0" izz="1.0"/>
        </inertial>
      </link>
    </xacro:macro>
  </xacro:unless>
</xacro:macro>
```

## Complete Advanced Robot Example

Here's a complete example showing advanced Xacro features:

`urdf/advanced_robot.xacro`:
```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="advanced_robot">
  <!-- Constants -->
  <xacro:property name="PI" value="3.1415926535897931"/>
  <xacro:property name="DEG_TO_RAD" value="0.017453292519943295"/>

  <!-- Robot parameters -->
  <xacro:property name="robot_name" value="advanced_robot"/>
  <xacro:property name="base_width" value="0.5"/>
  <xacro:property name="base_length" value="0.6"/>
  <xacro:property name="base_height" value="0.2"/>
  <xacro:property name="wheel_radius" value="0.1"/>
  <xacro:property name="wheel_width" value="0.05"/>
  <xacro:property name="wheel_separation" value="0.4"/>
  <xacro:property name="robot_mass" value="10.0"/>

  <!-- Include other xacro files -->
  <xacro:include filename="$(find my_robot_description)/urdf/materials.xacro"/>
  <xacro:include filename="$(find my_robot_description)/urdf/wheel.xacro"/>
  <xacro:include filename="$(find my_robot_description)/urdf/sensors.xacro"/>

  <!-- Base macro -->
  <xacro:macro name="robot_base" params="name mass">
    <link name="${name}_base_link">
      <visual>
        <origin xyz="0 0 ${base_height/2}" rpy="0 0 0"/>
        <geometry>
          <box size="${base_length} ${base_width} ${base_height}"/>
        </geometry>
        <material name="robot_gray"/>
      </visual>

      <collision>
        <origin xyz="0 0 ${base_height/2}" rpy="0 0 0"/>
        <geometry>
          <box size="${base_length} ${base_width} ${base_height}"/>
        </geometry>
      </collision>

      <inertial>
        <origin xyz="0 0 ${base_height/2}" rpy="0 0 0"/>
        <mass value="${mass}"/>
        <inertia ixx="${mass/12 * (base_width*base_width + base_height*base_height)}"
                 ixy="0.0" ixz="0.0"
                 iyy="${mass/12 * (base_length*base_length + base_height*base_height)}"
                 iyz="0.0"
                 izz="${mass/12 * (base_length*base_length + base_width*base_width)}"/>
      </inertial>
    </link>
  </xacro:macro>

  <!-- Sensor macro -->
  <xacro:macro name="sensor_array" params="parent count spacing">
    <xacro:property name="i" value="0"/>
    <xacro:while value="${i &lt; count}">
      <xacro:sensor_mount name="sensor_${i}"
                         parent="${parent}"
                         xyz="${0.2} ${-count/2 * spacing + i * spacing} 0.15"
                         rpy="0 0 0"/>
      <xacro:property name="i" value="${i + 1}"/>
    </xacro:while>
  </xacro:macro>

  <!-- Create robot -->
  <xacro:robot_base name="${robot_name}" mass="${robot_mass}"/>

  <!-- Create wheels -->
  <xacro:wheel_macro prefix="left" parent="${robot_name}_base_link"
                     xyz="0.2 ${wheel_separation/2} 0"
                     radius="${wheel_radius}" width="${wheel_width}"/>

  <xacro:wheel_macro prefix="right" parent="${robot_name}_base_link"
                     xyz="0.2 ${-wheel_separation/2} 0"
                     radius="${wheel_radius}" width="${wheel_width}"/>

  <!-- Create casters -->
  <xacro:wheel_macro prefix="caster_front" parent="${robot_name}_base_link"
                     xyz="${base_length/2 - 0.05} 0 0"
                     radius="${wheel_radius/2}" width="${wheel_width/2}"
                     mass="0.1"/>

  <xacro:wheel_macro prefix="caster_rear" parent="${robot_name}_base_link"
                     xyz="${-base_length/2 + 0.05} 0 0"
                     radius="${wheel_radius/2}" width="${wheel_width/2}"
                     mass="0.1"/>

  <!-- Create sensor array -->
  <xacro:sensor_array parent="${robot_name}_base_link" count="3" spacing="0.1"/>

  <!-- Gazebo plugins -->
  <gazebo>
    <plugin name="diff_drive" filename="libgazebo_ros_diff_drive.so">
      <left_joint>left_wheel_joint</left_joint>
      <right_joint>right_wheel_joint</right_joint>
      <wheel_separation>${wheel_separation}</wheel_separation>
      <wheel_diameter>${wheel_radius * 2}</wheel_diameter>
      <command_topic>cmd_vel</command_topic>
      <odometry_topic>odom</odometry_topic>
      <publish_odom_tf>true</publish_odom_tf>
    </plugin>
  </gazebo>

  <!-- Transmission for ROS control -->
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

  <transmission name="right_wheel_trans">
    <type>transmission_interface/SimpleTransmission</type>
    <joint name="right_wheel_joint">
      <hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface>
    </joint>
    <actuator name="right_wheel_motor">
      <hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface>
      <mechanicalReduction>1</mechanicalReduction>
    </actuator>
  </transmission>
</robot>
```

`urdf/materials.xacro`:
```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro">
  <material name="robot_gray">
    <color rgba="0.5 0.5 0.5 1"/>
  </material>

  <material name="robot_blue">
    <color rgba="0 0 1 1"/>
  </material>

  <material name="robot_red">
    <color rgba="1 0 0 1"/>
  </material>

  <material name="robot_green">
    <color rgba="0 1 0 1"/>
  </material>
</robot>
```

`urdf/sensors.xacro`:
```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro">
  <xacro:macro name="sensor_mount" params="name parent xyz rpy">
    <link name="${name}_link">
      <visual>
        <geometry>
          <box size="0.03 0.03 0.03"/>
        </geometry>
        <material name="robot_blue"/>
      </visual>
      <collision>
        <geometry>
          <box size="0.03 0.03 0.03"/>
        </geometry>
      </collision>
      <inertial>
        <mass value="0.05"/>
        <inertia ixx="0.00005" ixy="0.0" ixz="0.0" iyy="0.00005" iyz="0.0" izz="0.00005"/>
      </inertial>
    </link>

    <joint name="${name}_joint" type="fixed">
      <parent link="${parent}"/>
      <child link="${name}_link"/>
      <origin xyz="${xyz}" rpy="${rpy}"/>
    </joint>

    <gazebo reference="${name}_link">
      <sensor name="${name}" type="camera">
        <camera name="head">
          <horizontal_fov>1.047</horizontal_fov>
          <image>
            <width>640</width>
            <height>480</height>
            <format>R8G8B8</format>
          </image>
          <clip>
            <near>0.1</near>
            <far>10.0</far>
          </clip>
        </camera>
        <always_on>1</always_on>
        <update_rate>30</update_rate>
        <visualize>true</visualize>
      </sensor>
    </gazebo>
  </xacro:macro>
</robot>
```

## Working with Xacro

### Processing Xacro Files

```bash
# Convert xacro to urdf
xacro input.xacro > output.urdf

# Or with parameters
xacro input.xacro robot_type:=differential > output.urdf

# Check xacro syntax
xacro --check-order input.xacro

# View xacro with parameters
xacro --param robot_type:=differential input.xacro
```

### Loading Xacro in ROS 2

```python
#!/usr/bin/env python3
"""
Xacro loader example
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
from xacro import process_file
import math

class XacroLoader(Node):
    def __init__(self):
        super().__init__('xacro_loader')

        # Joint state publisher
        self.joint_pub = self.create_publisher(JointState, 'joint_states', 10)

        # TF broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)

        # Timer for publishing transforms
        self.timer = self.create_timer(0.1, self.publish_transforms)

        # Load robot description from xacro
        try:
            xacro_file = 'path/to/robot.urdf.xacro'
            robot_description = process_file(xacro_file)
            self.get_logger().info('Xacro processed successfully')
        except Exception as e:
            self.get_logger().error(f'Error processing xacro: {e}')

        self.get_logger().info('Xacro loader started')

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
    node = XacroLoader()

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

### 1. Modular Design

```xml
<!-- Good: Separate files for different components -->
<xacro:include filename="$(find my_robot)/urdf/base.xacro"/>
<xacro:include filename="$(find my_robot)/urdf/wheels.xacro"/>
<xacro:include filename="$(find my_robot)/urdf/sensors.xacro"/>

<!-- Bad: Everything in one large file -->
<!-- Don't do this - makes maintenance difficult -->
```

### 2. Parameterized Models

```xml
<!-- Good: Use properties for easy customization -->
<xacro:property name="robot_scale" value="1.0"/>
<link name="base_link">
  <visual>
    <geometry>
      <box size="${base_length * robot_scale} ${base_width * robot_scale} ${base_height * robot_scale}"/>
    </geometry>
  </visual>
</link>

<!-- Bad: Hardcoded values -->
<link name="base_link">
  <visual>
    <geometry>
      <box size="0.5 0.3 0.2"/>  <!-- Can't easily change size -->
    </geometry>
  </visual>
</link>
```

### 3. Proper Inertial Calculations

```xml
<!-- Good: Calculate inertial properties based on geometry -->
<inertial>
  <mass value="${density * volume}"/>
  <inertia ixx="${mass/12 * (width*width + height*height)}"
           ixy="0.0" ixz="0.0"
           iyy="${mass/12 * (length*length + height*height)}"
           iyz="0.0"
           izz="${mass/12 * (length*length + width*width)}"/>
</inertial>
```

### 4. Consistent Naming

```xml
<!-- Good: Consistent naming conventions -->
<joint name="left_wheel_joint" type="continuous">
  <parent link="base_link"/>
  <child link="left_wheel_link"/>
</joint>

<!-- Bad: Inconsistent naming -->
<joint name="LEFT_wheel" type="continuous">
  <parent link="base"/>
  <child link="wheel_left"/>
</joint>
```

## Common Issues and Troubleshooting

### 1. Xacro Processing Errors

```bash
# Error: Undefined property
# Solution: Check property names and scoping

# Error: Invalid expression
# Solution: Use proper syntax, escape special characters
# Use &lt; for <, &gt; for > in expressions
```

### 2. Macro Issues

```xml
# Error: Missing required parameters
# Solution: Provide all required macro parameters

# Error: Recursive macro calls
# Solution: Avoid circular macro dependencies
```

### 3. Inertial Issues

```xml
# Error: Zero or negative inertia values
# Solution: Ensure positive values for diagonal elements
```

## Advanced Techniques

### 1. Robot Variants

```xml
<!-- Create different robot configurations -->
<xacro:property name="robot_config" value="$(arg config)" default="standard"/>

<xacro:if value="${robot_config == 'standard'}">
  <!-- Standard configuration -->
  <xacro:include filename="$(find my_robot)/urdf/standard_config.xacro"/>
</xacro:if>

<xacro:if value="${robot_config == 'heavy_duty'}">
  <!-- Heavy duty configuration -->
  <xacro:include filename="$(find my_robot)/urdf/heavy_duty_config.xacro"/>
</xacro:if>
```

### 2. Sensor Arrays

```xml
<xacro:macro name="create_lidar_array" params="count spacing parent">
  <xacro:property name="i" value="0"/>
  <xacro:while value="${i &lt; count}">
    <xacro:sensor_mount name="lidar_${i}"
                       parent="${parent}"
                       xyz="0 ${-count/2 * spacing + i * spacing} 0.1"
                       rpy="0 0 0"
                       type="lidar"/>
    <xacro:property name="i" value="${i + 1}"/>
  </xacro:while>
</xacro:macro>
```

## Next Steps

Now that you understand Xacro and advanced modeling, continue to [Exercise: Multi-Link Robot](../week-05/exercise-robot-modeling) to build a complete robot model with multiple links and sensors.

## Exercises

1. Create a modular robot model using Xacro macros
2. Implement a parametric robot that can be customized with different dimensions
3. Create a sensor array using Xacro loops
4. Build a complete robot model with proper inertial properties