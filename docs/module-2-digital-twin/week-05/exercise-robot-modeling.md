---
sidebar_position: 4
---

# Exercise: Multi-Link Robot Model

In this comprehensive exercise, you'll create a complete multi-link robot model using Xacro with proper kinematics, dynamics, and sensor integration. This will demonstrate advanced robot modeling techniques and best practices.

## Objective

Create a robot model with:
1. **Multiple articulated links** with proper joints
2. **Accurate kinematics** and dynamics
3. **Integrated sensors** (camera, LiDAR, IMU)
4. **Modular Xacro structure** with reusable components

## Prerequisites

- Complete Week 1-5 lessons
- ROS 2 workspace set up (`~/ros2_ws`)
- Understanding of URDF and Xacro
- Basic XML knowledge

## Step 1: Create the Package

```bash
cd ~/ros2_ws/src

ros2 pkg create --build-type ament_python multi_link_robot \
    --dependencies rclpy std_msgs sensor_msgs geometry_msgs launch launch_ros
```

## Step 2: Create Xacro Structure

Create the directory structure:
```bash
mkdir -p multi_link_robot/urdf
mkdir -p multi_link_robot/meshes
mkdir -p multi_link_robot/config
mkdir -p multi_link_robot/launch
```

## Step 3: Create Base Components

Create `multi_link_robot/urdf/materials.xacro`:

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro">
  <!-- Common materials -->
  <material name="black">
    <color rgba="0.0 0.0 0.0 1.0"/>
  </material>

  <material name="blue">
    <color rgba="0.0 0.0 0.8 1.0"/>
  </material>

  <material name="green">
    <color rgba="0.0 0.8 0.0 1.0"/>
  </material>

  <material name="grey">
    <color rgba="0.5 0.5 0.5 1.0"/>
  </material>

  <material name="orange">
    <color rgba="1.0 0.423529411765 0.0392156862745 1.0"/>
  </material>

  <material name="brown">
    <color rgba="0.870588235294 0.811764705882 0.764705882353 1.0"/>
  </material>

  <material name="red">
    <color rgba="0.8 0.0 0.0 1.0"/>
  </material>

  <material name="white">
    <color rgba="1.0 1.0 1.0 1.0"/>
  </material>
</robot>
```

Create `multi_link_robot/urdf/common_properties.xacro`:

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro">
  <!-- Constants -->
  <xacro:property name="M_PI" value="3.1415926535897931"/>
  <xacro:property name="DEG_TO_RAD" value="0.017453292519943295"/>

  <!-- Robot dimensions -->
  <xacro:property name="base_width" value="0.6"/>
  <xacro:property name="base_length" value="0.8"/>
  <xacro:property name="base_height" value="0.15"/>
  <xacro:property name="base_mass" value="10.0"/>

  <xacro:property name="wheel_radius" value="0.1"/>
  <xacro:property name="wheel_width" value="0.05"/>
  <xacro:property name="wheel_mass" value="0.5"/>
  <xacro:property name="wheel_separation" value="0.4"/>

  <xacro:property name="arm_base_radius" value="0.08"/>
  <xacro:property name="arm_base_height" value="0.2"/>
  <xacro:property name="arm_base_mass" value="1.0"/>

  <xacro:property name="arm_link_length" value="0.3"/>
  <xacro:property name="arm_link_radius" value="0.05"/>
  <xacro:property name="arm_link_mass" value="0.8"/>

  <xacro:property name="gripper_length" value="0.1"/>
  <xacro:property name="gripper_width" value="0.05"/>
  <xacro:property name="gripper_height" value="0.05"/>
  <xacro:property name="gripper_mass" value="0.1"/>
</robot>
```

Create `multi_link_robot/urdf/wheels.xacro`:

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro">
  <xacro:include filename="$(find multi_link_robot)/urdf/common_properties.xacro"/>

  <xacro:macro name="wheel" params="prefix parent xyz rpy:=0 0 0">
    <link name="${prefix}_wheel">
      <visual>
        <origin xyz="0 0 0" rpy="${M_PI/2} 0 0"/>
        <geometry>
          <cylinder radius="${wheel_radius}" length="${wheel_width}"/>
        </geometry>
        <material name="black"/>
      </visual>

      <collision>
        <origin xyz="0 0 0" rpy="${M_PI/2} 0 0"/>
        <geometry>
          <cylinder radius="${wheel_radius}" length="${wheel_width}"/>
        </geometry>
      </collision>

      <inertial>
        <mass value="${wheel_mass}"/>
        <inertia ixx="${0.5*wheel_mass*wheel_radius*wheel_radius}"
                 ixy="0.0" ixz="0.0"
                 iyy="${0.25*wheel_mass*wheel_radius*wheel_radius + wheel_mass*wheel_width*wheel_width/12}"
                 iyz="0.0"
                 izz="${0.5*wheel_mass*wheel_radius*wheel_radius}"/>
      </inertial>
    </link>

    <joint name="${prefix}_wheel_joint" type="continuous">
      <parent link="${parent}"/>
      <child link="${prefix}_wheel"/>
      <origin xyz="${xyz}" rpy="${rpy}"/>
      <axis xyz="0 0 1"/>
    </joint>
  </xacro:macro>

  <!-- Caster wheels -->
  <xacro:macro name="caster_wheel" params="prefix parent xyz">
    <link name="${prefix}_caster_wheel">
      <visual>
        <geometry>
          <sphere radius="${wheel_radius/2}"/>
        </geometry>
        <material name="black"/>
      </visual>

      <collision>
        <geometry>
          <sphere radius="${wheel_radius/2}"/>
        </geometry>
      </collision>

      <inertial>
        <mass value="${wheel_mass/4}"/>
        <inertia ixx="${0.4*wheel_mass/4*(wheel_radius/2)*(wheel_radius/2)}"
                 ixy="0.0" ixz="0.0"
                 iyy="${0.4*wheel_mass/4*(wheel_radius/2)*(wheel_radius/2)}"
                 iyz="0.0"
                 izz="${0.4*wheel_mass/4*(wheel_radius/2)*(wheel_radius/2)}"/>
      </inertial>
    </link>

    <joint name="${prefix}_caster_wheel_joint" type="fixed">
      <parent link="${parent}"/>
      <child link="${prefix}_caster_wheel"/>
      <origin xyz="${xyz}" rpy="0 0 0"/>
    </joint>
  </xacro:macro>
</robot>
```

Create `multi_link_robot/urdf/arm.xacro`:

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro">
  <xacro:include filename="$(find multi_link_robot)/urdf/common_properties.xacro"/>

  <xacro:macro name="robot_arm" params="parent prefix xyz rpy:=0 0 0">
    <!-- Arm base -->
    <link name="${prefix}_arm_base">
      <visual>
        <origin xyz="0 0 ${arm_base_height/2}" rpy="0 0 0"/>
        <geometry>
          <cylinder radius="${arm_base_radius}" length="${arm_base_height}"/>
        </geometry>
        <material name="orange"/>
      </visual>

      <collision>
        <origin xyz="0 0 ${arm_base_height/2}" rpy="0 0 0"/>
        <geometry>
          <cylinder radius="${arm_base_radius}" length="${arm_base_height}"/>
        </geometry>
      </collision>

      <inertial>
        <origin xyz="0 0 ${arm_base_height/2}" rpy="0 0 0"/>
        <mass value="${arm_base_mass}"/>
        <inertia ixx="${arm_base_mass/12 * (3*arm_base_radius*arm_base_radius + arm_base_height*arm_base_height)}"
                 ixy="0.0" ixz="0.0"
                 iyy="${arm_base_mass/12 * (3*arm_base_radius*arm_base_radius + arm_base_height*arm_base_height)}"
                 iyz="0.0"
                 izz="${arm_base_mass/2 * arm_base_radius*arm_base_radius}"/>
      </inertial>
    </link>

    <joint name="${prefix}_arm_base_joint" type="revolute">
      <parent link="${parent}"/>
      <child link="${prefix}_arm_base"/>
      <origin xyz="${xyz}" rpy="${rpy}"/>
      <axis xyz="0 0 1"/>
      <limit lower="${-M_PI/2}" upper="${M_PI/2}" effort="100.0" velocity="1.0"/>
      <dynamics damping="0.1" friction="0.0"/>
    </joint>

    <!-- Arm link 1 -->
    <link name="${prefix}_arm_link_1">
      <visual>
        <origin xyz="0 0 ${arm_link_length/2}" rpy="0 0 0"/>
        <geometry>
          <cylinder radius="${arm_link_radius}" length="${arm_link_length}"/>
        </geometry>
        <material name="blue"/>
      </visual>

      <collision>
        <origin xyz="0 0 ${arm_link_length/2}" rpy="0 0 0"/>
        <geometry>
          <cylinder radius="${arm_link_radius}" length="${arm_link_length}"/>
        </geometry>
      </collision>

      <inertial>
        <origin xyz="0 0 ${arm_link_length/2}" rpy="0 0 0"/>
        <mass value="${arm_link_mass}"/>
        <inertia ixx="${arm_link_mass/12 * (3*arm_link_radius*arm_link_radius + arm_link_length*arm_link_length)}"
                 ixy="0.0" ixz="0.0"
                 iyy="${arm_link_mass/12 * (3*arm_link_radius*arm_link_radius + arm_link_length*arm_link_length)}"
                 iyz="0.0"
                 izz="${arm_link_mass/2 * arm_link_radius*arm_link_radius}"/>
      </inertial>
    </link>

    <joint name="${prefix}_arm_joint_1" type="revolute">
      <parent link="${prefix}_arm_base"/>
      <child link="${prefix}_arm_link_1"/>
      <origin xyz="0 0 ${arm_base_height}" rpy="0 0 0"/>
      <axis xyz="0 1 0"/>
      <limit lower="${-M_PI/2}" upper="${M_PI/2}" effort="100.0" velocity="1.0"/>
      <dynamics damping="0.1" friction="0.0"/>
    </joint>

    <!-- Arm link 2 -->
    <link name="${prefix}_arm_link_2">
      <visual>
        <origin xyz="0 0 ${arm_link_length/2}" rpy="0 0 0"/>
        <geometry>
          <cylinder radius="${arm_link_radius}" length="${arm_link_length}"/>
        </geometry>
        <material name="green"/>
      </visual>

      <collision>
        <origin xyz="0 0 ${arm_link_length/2}" rpy="0 0 0"/>
        <geometry>
          <cylinder radius="${arm_link_radius}" length="${arm_link_length}"/>
        </geometry>
      </collision>

      <inertial>
        <origin xyz="0 0 ${arm_link_length/2}" rpy="0 0 0"/>
        <mass value="${arm_link_mass}"/>
        <inertia ixx="${arm_link_mass/12 * (3*arm_link_radius*arm_link_radius + arm_link_length*arm_link_length)}"
                 ixy="0.0" ixz="0.0"
                 iyy="${arm_link_mass/12 * (3*arm_link_radius*arm_link_radius + arm_link_length*arm_link_length)}"
                 iyz="0.0"
                 izz="${arm_link_mass/2 * arm_link_radius*arm_link_radius}"/>
      </inertial>
    </link>

    <joint name="${prefix}_arm_joint_2" type="revolute">
      <parent link="${prefix}_arm_link_1"/>
      <child link="${prefix}_arm_link_2"/>
      <origin xyz="0 0 ${arm_link_length}" rpy="0 0 0"/>
      <axis xyz="0 1 0"/>
      <limit lower="${-M_PI/2}" upper="${M_PI/2}" effort="100.0" velocity="1.0"/>
      <dynamics damping="0.1" friction="0.0"/>
    </joint>

    <!-- Gripper -->
    <link name="${prefix}_gripper">
      <visual>
        <geometry>
          <box size="${gripper_length} ${gripper_width} ${gripper_height}"/>
        </geometry>
        <material name="red"/>
      </visual>

      <collision>
        <geometry>
          <box size="${gripper_length} ${gripper_width} ${gripper_height}"/>
        </geometry>
      </collision>

      <inertial>
        <mass value="${gripper_mass}"/>
        <inertia ixx="${gripper_mass/12 * (gripper_width*gripper_width + gripper_height*gripper_height)}"
                 ixy="0.0" ixz="0.0"
                 iyy="${gripper_mass/12 * (gripper_length*gripper_length + gripper_height*gripper_height)}"
                 iyz="0.0"
                 izz="${gripper_mass/12 * (gripper_length*gripper_length + gripper_width*gripper_width)}"/>
      </inertial>
    </link>

    <joint name="${prefix}_gripper_joint" type="fixed">
      <parent link="${prefix}_arm_link_2"/>
      <child link="${prefix}_gripper"/>
      <origin xyz="0 0 ${arm_link_length}" rpy="0 0 0"/>
    </joint>
  </xacro:macro>
</robot>
```

Create `multi_link_robot/urdf/sensors.xacro`:

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro">
  <xacro:include filename="$(find multi_link_robot)/urdf/common_properties.xacro"/>

  <!-- Camera sensor -->
  <xacro:macro name="camera_sensor" params="prefix parent xyz rpy:=0 0 0">
    <link name="${prefix}_camera_link">
      <visual>
        <geometry>
          <box size="0.05 0.05 0.03"/>
        </geometry>
        <material name="black"/>
      </visual>

      <collision>
        <geometry>
          <box size="0.05 0.05 0.03"/>
        </geometry>
      </collision>

      <inertial>
        <mass value="0.1"/>
        <inertia ixx="0.0001" ixy="0.0" ixz="0.0" iyy="0.0001" iyz="0.0" izz="0.0001"/>
      </inertial>
    </link>

    <joint name="${prefix}_camera_joint" type="fixed">
      <parent link="${parent}"/>
      <child link="${prefix}_camera_link"/>
      <origin xyz="${xyz}" rpy="${rpy}"/>
    </joint>

    <!-- Gazebo sensor plugin -->
    <gazebo reference="${prefix}_camera_link">
      <sensor name="${prefix}_camera" type="camera">
        <update_rate>30</update_rate>
        <camera name="head">
          <horizontal_fov>1.047</horizontal_fov>
          <image>
            <width>640</width>
            <height>480</height>
            <format>R8G8B8</format>
          </image>
          <clip>
            <near>0.1</near>
            <far>30.0</far>
          </clip>
        </camera>
        <plugin name="camera_controller" filename="libgazebo_ros_camera.so">
          <frame_name>${prefix}_camera_optical_frame</frame_name>
          <min_depth>0.1</min_depth>
          <max_depth>30.0</max_depth>
        </plugin>
      </sensor>
    </gazebo>
  </xacro:macro>

  <!-- IMU sensor -->
  <xacro:macro name="imu_sensor" params="prefix parent xyz rpy:=0 0 0">
    <link name="${prefix}_imu_link">
      <inertial>
        <mass value="0.01"/>
        <inertia ixx="0.0001" ixy="0.0" ixz="0.0" iyy="0.0001" iyz="0.0" izz="0.0001"/>
      </inertial>
    </link>

    <joint name="${prefix}_imu_joint" type="fixed">
      <parent link="${parent}"/>
      <child link="${prefix}_imu_link"/>
      <origin xyz="${xyz}" rpy="${rpy}"/>
    </joint>

    <gazebo reference="${prefix}_imu_link">
      <sensor name="${prefix}_imu" type="imu">
        <always_on>true</always_on>
        <update_rate>100</update_rate>
        <plugin filename="libgazebo_ros_imu.so" name="imu_plugin">
          <topic>${prefix}/imu</topic>
          <body_name>${prefix}_imu_link</body_name>
          <frame_name>${prefix}_imu_link</frame_name>
        </plugin>
      </sensor>
    </gazebo>
  </xacro:macro>

  <!-- LiDAR sensor -->
  <xacro:macro name="lidar_sensor" params="prefix parent xyz rpy:=0 0 0">
    <link name="${prefix}_lidar_link">
      <visual>
        <geometry>
          <cylinder radius="0.05" length="0.05"/>
        </geometry>
        <material name="black"/>
      </visual>

      <collision>
        <geometry>
          <cylinder radius="0.05" length="0.05"/>
        </geometry>
      </collision>

      <inertial>
        <mass value="0.1"/>
        <inertia ixx="0.0001" ixy="0.0" ixz="0.0" iyy="0.0001" iyz="0.0" izz="0.0001"/>
      </inertial>
    </link>

    <joint name="${prefix}_lidar_joint" type="fixed">
      <parent link="${parent}"/>
      <child link="${prefix}_lidar_link"/>
      <origin xyz="${xyz}" rpy="${rpy}"/>
    </joint>

    <gazebo reference="${prefix}_lidar_link">
      <sensor name="${prefix}_lidar" type="ray">
        <ray>
          <scan>
            <horizontal>
              <samples>720</samples>
              <resolution>1</resolution>
              <min_angle>-3.14159</min_angle>
              <max_angle>3.14159</max_angle>
            </horizontal>
          </scan>
          <range>
            <min>0.1</min>
            <max>30.0</max>
            <resolution>0.01</resolution>
          </range>
        </ray>
        <plugin name="lidar_controller" filename="libgazebo_ros_ray_sensor.so">
          <ros>
            <namespace>${prefix}</namespace>
            <remapping>~/out:=scan</remapping>
          </ros>
          <output_type>sensor_msgs/LaserScan</output_type>
        </plugin>
      </sensor>
    </gazebo>
  </xacro:macro>
</robot>
```

## Step 4: Create Main Robot Model

Create `multi_link_robot/urdf/multi_link_robot.urdf.xacro`:

```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="multi_link_robot">
  <!-- Include all components -->
  <xacro:include filename="$(find multi_link_robot)/urdf/common_properties.xacro"/>
  <xacro:include filename="$(find multi_link_robot)/urdf/materials.xacro"/>
  <xacro:include filename="$(find multi_link_robot)/urdf/wheels.xacro"/>
  <xacro:include filename="$(find multi_link_robot)/urdf/arm.xacro"/>
  <xacro:include filename="$(find multi_link_robot)/urdf/sensors.xacro"/>

  <!-- Base link -->
  <link name="base_link">
    <visual>
      <origin xyz="0 0 ${base_height/2}" rpy="0 0 0"/>
      <geometry>
        <box size="${base_length} ${base_width} ${base_height}"/>
      </geometry>
      <material name="grey"/>
    </visual>

    <collision>
      <origin xyz="0 0 ${base_height/2}" rpy="0 0 0"/>
      <geometry>
        <box size="${base_length} ${base_width} ${base_height}"/>
      </geometry>
    </collision>

    <inertial>
      <origin xyz="0 0 ${base_height/2}" rpy="0 0 0"/>
      <mass value="${base_mass}"/>
      <inertia ixx="${base_mass/12 * (base_width*base_width + base_height*base_height)}"
               ixy="0.0" ixz="0.0"
               iyy="${base_mass/12 * (base_length*base_length + base_height*base_height)}"
               iyz="0.0"
               izz="${base_mass/12 * (base_length*base_length + base_width*base_width)}"/>
    </inertial>
  </link>

  <!-- Add wheels -->
  <xacro:wheel prefix="front_left" parent="base_link" xyz="0.2 ${wheel_separation/2} 0"/>
  <xacro:wheel prefix="front_right" parent="base_link" xyz="0.2 ${-wheel_separation/2} 0"/>
  <xacro:wheel prefix="rear_left" parent="base_link" xyz="-0.2 ${wheel_separation/2} 0"/>
  <xacro:wheel prefix="rear_right" parent="base_link" xyz="-0.2 ${-wheel_separation/2} 0"/>

  <!-- Add caster wheels -->
  <xacro:caster_wheel prefix="caster_front" parent="base_link" xyz="0.3 0 0"/>
  <xacro:caster_wheel prefix="caster_rear" parent="base_link" xyz="-0.3 0 0"/>

  <!-- Add robot arm -->
  <xacro:robot_arm parent="base_link" prefix="manipulator" xyz="0.2 0 ${base_height}"/>

  <!-- Add sensors -->
  <xacro:camera_sensor prefix="front" parent="base_link" xyz="0.25 0 0.15" rpy="0 0 0"/>
  <xacro:lidar_sensor prefix="top" parent="base_link" xyz="0 0 0.3" rpy="0 0 0"/>
  <xacro:imu_sensor prefix="base" parent="base_link" xyz="0 0 0.1" rpy="0 0 0"/>

  <!-- Gazebo plugins -->
  <gazebo>
    <plugin name="diff_drive" filename="libgazebo_ros_diff_drive.so">
      <left_joint>front_left_wheel_joint</left_joint>
      <right_joint>front_right_wheel_joint</right_joint>
      <wheel_separation>${wheel_separation}</wheel_separation>
      <wheel_diameter>${wheel_radius * 2}</wheel_diameter>
      <command_topic>cmd_vel</command_topic>
      <odometry_topic>odom</odometry_topic>
      <publish_odom_tf>true</publish_odom_tf>
      <odometry_frame>odom</odometry_frame>
      <robot_base_frame>base_link</robot_base_frame>
    </plugin>
  </gazebo>

  <!-- Joint state publisher -->
  <gazebo>
    <plugin name="joint_state_publisher" filename="libgazebo_ros_joint_state_publisher.so">
      <ros>
        <namespace>multi_link_robot</namespace>
      </ros>
      <joint_name>manipulator_arm_base_joint</joint_name>
      <joint_name>manipulator_arm_joint_1</joint_name>
      <joint_name>manipulator_arm_joint_2</joint_name>
    </plugin>
  </gazebo>

  <!-- Transmission for arm joints -->
  <transmission name="manipulator_arm_base_trans">
    <type>transmission_interface/SimpleTransmission</type>
    <joint name="manipulator_arm_base_joint">
      <hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface>
    </joint>
    <actuator name="manipulator_arm_base_motor">
      <hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface>
      <mechanicalReduction>1</mechanicalReduction>
    </actuator>
  </transmission>

  <transmission name="manipulator_arm_joint_1_trans">
    <type>transmission_interface/SimpleTransmission</type>
    <joint name="manipulator_arm_joint_1">
      <hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface>
    </joint>
    <actuator name="manipulator_arm_joint_1_motor">
      <hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface>
      <mechanicalReduction>1</mechanicalReduction>
    </actuator>
  </transmission>

  <transmission name="manipulator_arm_joint_2_trans">
    <type>transmission_interface/SimpleTransmission</type>
    <joint name="manipulator_arm_joint_2">
      <hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface>
    </joint>
    <actuator name="manipulator_arm_joint_2_motor">
      <hardwareInterface>hardware_interface/EffortJointInterface</hardwareInterface>
      <mechanicalReduction>1</mechanicalReduction>
    </actuator>
  </transmission>
</robot>
```

## Step 5: Create Launch Files

Create `multi_link_robot/launch/robot_description.launch.py`:

```python
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Declare launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')
    model = LaunchConfiguration('model', default='multi_link_robot.urdf.xacro')

    # Get URDF path
    robot_description_path = os.path.join(
        get_package_share_directory('multi_link_robot'),
        'urdf',
        LaunchConfiguration('model').perform({})  # Get the actual value
    )

    # Process xacro to URDF
    robot_description = ParameterValue(
        Command(['xacro ', robot_description_path]),
        value_type=str
    )

    # Robot state publisher node
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[
            {'use_sim_time': use_sim_time},
            {'robot_description': robot_description}
        ],
        remappings=[
            ('/joint_states', 'multi_link_robot/joint_states')
        ]
    )

    # Joint state publisher node (for GUI)
    joint_state_publisher = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        parameters=[{'use_sim_time': use_sim_time}],
        remappings=[
            ('/joint_states', 'multi_link_robot/joint_states')
        ]
    )

    # Joint state publisher GUI (optional)
    joint_state_publisher_gui = Node(
        package='joint_state_publisher_gui',
        executable='joint_state_publisher_gui',
        condition=launch.conditions.IfCondition(
            launch.substitutions.LaunchConfiguration('use_gui', default='false')
        )
    )

    # RViz2 node
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        arguments=['-d', os.path.join(
            get_package_share_directory('multi_link_robot'),
            'config',
            'robot_model.rviz'
        )],
        parameters=[{'use_sim_time': use_sim_time}]
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation time if true'
        ),
        DeclareLaunchArgument(
            'model',
            default_value='multi_link_robot.urdf.xacro',
            description='Robot URDF/XACRO description file'
        ),
        DeclareLaunchArgument(
            'use_gui',
            default_value='false',
            description='Enable joint state publisher GUI'
        ),
        robot_state_publisher,
        joint_state_publisher,
        # joint_state_publisher_gui,  # Uncomment if GUI is desired
        rviz
    ])
```

## Step 6: Create RViz Configuration

Create `multi_link_robot/config/robot_model.rviz`:

```yaml
Panels:
  - Class: rviz_common/Displays
    Help Height: 78
    Name: Displays
    Property Tree Widget:
      Expanded:
        - /Global Options1
        - /Status1
        - /RobotModel1
        - /TF1
      Splitter Ratio: 0.5
    Tree Height: 787
  - Class: rviz_common/Selection
    Name: Selection
  - Class: rviz_common/Tool Properties
    Expanded:
      - /2D Goal Pose1
      - /Publish Point1
    Name: Tool Properties
    Splitter Ratio: 0.5886790156364441
  - Class: rviz_common/Views
    Expanded:
      - /Current View1
    Name: Views
    Splitter Ratio: 0.5
Visualization Manager:
  Class: ""
  Displays:
    - Alpha: 0.5
      Cell Size: 1
      Class: rviz_default_plugins/Grid
      Color: 160; 160; 164
      Enabled: true
      Line Style:
        Line Width: 0.029999999329447746
        Value: Lines
      Name: Grid
      Normal Cell Count: 0
      Offset:
        X: 0
        Y: 0
        Z: 0
      Plane: XY
      Plane Cell Count: 10
      Reference Frame: <Fixed Frame>
      Value: true
    - Alpha: 1
      Class: rviz_default_plugins/RobotModel
      Collision Enabled: false
      Description File: ""
      Description Source: Topic
      Description Topic:
        Depth: 5
        Durability Policy: Volatile
        History Policy: Keep Last
        Reliability Policy: Reliable
        Value: /robot_description
      Enabled: true
      Links:
        All Links Enabled: true
        Expand Joint Details: false
        Expand Link Details: false
        Expand Tree: false
        Link Tree Style: Links in Alphabetic Order
      Name: RobotModel
      TF Prefix: ""
      Update Interval: 0
      Value: true
      Visual Enabled: true
    - Class: rviz_default_plugins/TF
      Enabled: true
      Frame Timeout: 15
      Frames:
        All Enabled: true
      Marker Scale: 1
      Name: TF
      Show Arrows: true
      Show Axes: true
      Show Names: false
      Tree:
        {}
      Update Interval: 0
      Value: true
  Enabled: true
  Global Options:
    Background Color: 48; 48; 48
    Fixed Frame: base_link
    Frame Rate: 30
  Name: root
  Tools:
    - Class: rviz_default_plugins/Interact
      Hide Inactive Objects: true
    - Class: rviz_default_plugins/MoveCamera
    - Class: rviz_default_plugins/Select
    - Class: rviz_default_plugins/FocusCamera
    - Class: rviz_default_plugins/Measure
    - Class: rviz_default_plugins/SetInitialPose
      Topic:
        Depth: 5
        Durability Policy: Volatile
        History Policy: Keep Last
        Reliability Policy: Reliable
        Value: /initialpose
    - Class: rviz_default_plugins/SetGoal
      Topic:
        Depth: 5
        Durability Policy: Volatile
        History Policy: Keep Last
        Reliability Policy: Reliable
        Value: /goal_pose
    - Class: rviz_default_plugins/PublishPoint
      Single click: true
      Topic:
        Depth: 5
        Durability Policy: Volatile
        History Policy: Keep Last
        Reliability Policy: Reliable
        Value: /clicked_point
  Transformation:
    Current:
      Class: rviz_default_plugins/TF
  Value: true
  Views:
    Current:
      Class: rviz_default_plugins/Orbit
      Distance: 2
      Enable Stereo Rendering:
        Stereo Eye Separation: 0.05999999865889549
        Stereo Focal Distance: 1
        Swap Stereo Eyes: false
        Value: false
      Focal Point:
        X: 0
        Y: 0
        Z: 0
      Focal Shape Fixed Size: true
      Focal Shape Size: 0.05000000074505806
      Invert Z Axis: false
      Name: Current View
      Near Clip Distance: 0.009999999776482582
      Pitch: 0.5
      Target Frame: <Fixed Frame>
      Value: Orbit (rviz)
      Yaw: 0.5
    Saved: ~
Window Geometry:
  Displays:
    collapsed: false
  Height: 1043
  Hide Left Dock: false
  Hide Right Dock: false
  QMainWindow State: 000000ff00000000fd0000000400000000000001560000039ffc0200000009fb0000001200530065006c0065006300740069006f006e00000001e10000009b0000005c00fffffffb0000001e0054006f006f006c002000500072006f007000650072007400690065007302000001ed000001df00000185000000a3fb000000120056006900650077007300200054006f006f02000001df000002110000018500000122fb000000200054006f006f006c002000500072006f0070006500720074006900650073003203000002880000011d000002210000017afb000000100044006900730070006c006100790073010000003d0000039f000000c900fffffffb0000002000730065006c0065006300740069006f006e00200062007500660066006500720200000138000000aa0000023a00000294fb00000014005700690064006500530074006500720065006f02000000e6000000d2000003ee0000030bfb0000000c004b0069006e0065006300740200000186000001060000030c00000261000000010000010f0000039ffc0200000003fb0000001e0054006f006f006c002000500072006f00700065007200740069006500730100000041000000780000000000000000fb0000000a00560069006500770073000000003d0000039f000000a400fffffffb0000001200530065006c0065006300740069006f006e010000025a000000b200000000000000000000000200000490000000a9fc0100000001fb0000000a00560069006500770073030000004e00000080000002e100000197000000030000073f0000003efc0100000002fb0000000800540069006d006501000000000000073f000002eb00fffffffb0000000800540069006d00650100000000000004500000000000000000000004cc0000039f00000004000000040000000800000008fc0000000100000002000000010000000a0054006f006f006c00730100000000ffffffff0000000000000000
  Width: 1853
  X: 67
  Y: 27
```

## Step 7: Update Package Configuration

Update `multi_link_robot/setup.py`:

```python
from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'multi_link_robot'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'urdf'), glob('urdf/*.xacro')),
        (os.path.join('share', package_name, 'urdf'), glob('urdf/*.urdf')),
        (os.path.join('share', package_name, 'config'), glob('config/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your.email@example.com',
    description='Multi-link robot model for simulation',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        ],
    },
)
```

## Step 8: Build and Test

Build the package:

```bash
cd ~/ros2_ws
colcon build --packages-select multi_link_robot
source install/setup.bash
```

### Test the Robot Model

1. **Process the Xacro file**:
```bash
# Convert Xacro to URDF
xacro `ros2 pkg prefix multi_link_robot`/share/multi_link_robot/urdf/multi_link_robot.urdf.xacro > processed_robot.urdf
```

2. **Visualize the model in RViz**:
```bash
# Launch robot description
ros2 launch multi_link_robot robot_description.launch.py
```

3. **Check the robot model**:
```bash
# Verify URDF is valid
check_urdf `ros2 pkg prefix multi_link_robot`/share/multi_link_robot/urdf/multi_link_robot.urdf.xacro

# Show robot structure
urdf_to_graphiz `ros2 pkg prefix multi_link_robot`/share/multi_link_robot/urdf/multi_link_robot.urdf.xacro
```

4. **Test with Gazebo**:
```bash
# Launch in Gazebo
ros2 launch gazebo_ros gazebo.launch.py
# In another terminal, spawn the robot:
ros2 run gazebo_ros spawn_entity.py -entity multi_link_robot -x 0 -y 0 -z 0 -file `ros2 pkg prefix multi_link_robot`/share/multi_link_robot/urdf/multi_link_robot.urdf.xacro
```

## Understanding the Model

This multi-link robot model demonstrates:

1. **Modular Design**: Separate Xacro files for different components
2. **Proper Kinematics**: Correct joint types and limits for robot arm
3. **Accurate Dynamics**: Realistic mass properties and inertial tensors
4. **Sensor Integration**: Camera, LiDAR, and IMU properly mounted
5. **Gazebo Compatibility**: Proper plugins and configurations

## Challenges

### Challenge 1: Add More Sensors
Add additional sensors like sonar or thermal cameras to the robot model.

<details>
<summary>Hint</summary>

Create a new sensor macro in `sensors.xacro` following the same pattern as existing sensors.
</details>

### Challenge 2: Create Robot Variants
Create different configurations of the robot using Xacro parameters.

<details>
<summary>Hint</summary>

Add parameters to the main Xacro file to control robot dimensions, sensor configurations, or arm configurations.
</details>

### Challenge 3: Add Actuators
Implement gripper control for the robot arm.

<details>
<summary>Hint</summary>

Add actuator joints to the gripper and implement control interfaces.
</details>

### Challenge 4: Optimize Performance
Simplify collision geometry for better simulation performance.

<details>
<summary>Hint</summary>

Use simpler shapes (boxes, cylinders) instead of complex meshes for collision detection.
</details>

## Verification Checklist

- [ ] Xacro files process without errors
- [ ] Robot model displays correctly in RViz
- [ ] All joints are properly connected
- [ ] Mass properties are realistic
- [ ] Sensors are correctly positioned
- [ ] Robot arm has proper kinematics
- [ ] All links have collision and visual properties
- [ ] TF tree is properly formed

## Common Issues

### Xacro Processing Issues
```bash
# Error: Undefined property
# Solution: Check property names and include statements

# Error: Invalid expression
# Solution: Use proper syntax and escape special characters
```

### Joint Issues
```bash
# Error: Joint not found in TF tree
# Solution: Verify joint names match between URDF and controllers
```

### Visualization Issues
```bash
# Error: Robot not showing in RViz
# Solution: Check robot_description topic and TF frames
```

## Summary

In this exercise, you learned to:
- Create modular robot models using Xacro macros
- Implement proper kinematics and dynamics for multi-link robots
- Integrate sensors with correct mounting and parameters
- Validate robot models for simulation compatibility
- Structure complex robot descriptions in a maintainable way

## Next Steps

Continue to [Week 6: Isaac Sim](../../module-2-digital-twin/week-06/introduction) to learn about NVIDIA's advanced simulation platform for robotics.