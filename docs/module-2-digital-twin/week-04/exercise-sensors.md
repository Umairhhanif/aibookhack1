---
sidebar_position: 4
---

# Exercise: Sensor Fusion Robot

In this comprehensive exercise, you'll create a complete sensor fusion robot that integrates multiple sensors in simulation. This will demonstrate how to build realistic sensor models and process their data in a ROS 2 system.

## Objective

Create a robot with multiple sensors that:
1. **Simulates realistic sensors** (camera, LiDAR, IMU)
2. **Integrates sensor data** through a fusion node
3. **Processes perception data** with computer vision
4. **Validates sensor accuracy** through comparison

## Prerequisites

- Complete Week 1-4 lessons
- ROS 2 workspace set up (`~/ros2_ws`)
- Gazebo installed and working
- Basic Python programming knowledge

## Step 1: Create the Package

```bash
cd ~/ros2_ws/src

ros2 pkg create --build-type ament_python sensor_fusion_robot \
    --dependencies rclpy std_msgs sensor_msgs geometry_msgs vision_msgs cv_bridge launch launch_ros
```

## Step 2: Create the Robot Model

Create `sensor_fusion_robot/models/sensor_robot.sdf`:

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <model name="sensor_robot">
    <!-- Robot chassis -->
    <link name="base_link">
      <pose>0 0 0.1 0 0 0</pose>
      <inertial>
        <mass>10.0</mass>
        <inertia>
          <ixx>0.4</ixx>
          <iyy>0.4</iyy>
          <izz>0.8</izz>
        </inertia>
      </inertial>
      <collision name="collision">
        <geometry>
          <cylinder>
            <radius>0.3</radius>
            <length>0.2</length>
          </cylinder>
        </geometry>
      </collision>
      <visual name="visual">
        <geometry>
          <cylinder>
            <radius>0.3</radius>
            <length>0.2</length>
          </cylinder>
        </geometry>
        <material>
          <ambient>0.5 0.5 0.5 1</ambient>
          <diffuse>0.5 0.5 0.5 1</diffuse>
          <specular>0.5 0.5 0.5 1</specular>
        </material>
      </visual>
    </link>

    <!-- Camera sensor -->
    <sensor name="camera" type="camera">
      <pose>0.2 0 0.2 0 0 0</pose>
      <camera name="head">
        <horizontal_fov>1.047</horizontal_fov>  <!-- 60 degrees -->
        <image>
          <width>640</width>
          <height>480</height>
          <format>R8G8B8</format>
        </image>
        <clip>
          <near>0.1</near>
          <far>10.0</far>
        </clip>
        <noise>
          <type>gaussian</type>
          <mean>0.0</mean>
          <stddev>0.007</stddev>
        </noise>
      </camera>
      <always_on>1</always_on>
      <update_rate>30</update_rate>
      <visualize>true</visualize>
    </sensor>

    <!-- LiDAR sensor -->
    <sensor name="lidar" type="ray">
      <pose>0.1 0 0.3 0 0 0</pose>
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
      <always_on>1</always_on>
      <update_rate>10</update_rate>
      <visualize>true</visualize>
    </sensor>

    <!-- IMU sensor -->
    <sensor name="imu" type="imu">
      <pose>0 0 0.1 0 0 0</pose>
      <always_on>1</always_on>
      <update_rate>100</update_rate>
      <imu>
        <angular_velocity>
          <x>
            <noise type="gaussian">
              <mean>0.0</mean>
              <stddev>0.001</stddev>
            </noise>
          </x>
          <y>
            <noise type="gaussian">
              <mean>0.0</mean>
              <stddev>0.001</stddev>
            </noise>
          </y>
          <z>
            <noise type="gaussian">
              <mean>0.0</mean>
              <stddev>0.001</stddev>
            </noise>
          </z>
        </angular_velocity>
        <linear_acceleration>
          <x>
            <noise type="gaussian">
              <mean>0.0</mean>
              <stddev>0.017</stddev>
            </noise>
          </x>
          <y>
            <noise type="gaussian">
              <mean>0.0</mean>
              <stddev>0.017</stddev>
            </noise>
          </y>
          <z>
            <noise type="gaussian">
              <mean>0.0</mean>
              <stddev>0.017</stddev>
            </noise>
          </z>
        </linear_acceleration>
      </imu>
    </sensor>

    <!-- GPS sensor -->
    <sensor name="gps" type="gps">
      <pose>0 0 0.4 0 0 0</pose>
      <always_on>1</always_on>
      <update_rate>1</update_rate>
      <gps>
        <position_sensing>
          <horizontal>
            <noise type="gaussian">
              <mean>0.0</mean>
              <stddev>2.0</stddev>
            </noise>
          </horizontal>
          <vertical>
            <noise type="gaussian">
              <mean>0.0</mean>
              <stddev>3.0</stddev>
            </noise>
          </vertical>
        </position_sensing>
      </gps>
    </sensor>

    <!-- Differential drive plugin -->
    <plugin name="diff_drive" filename="libgazebo_ros_diff_drive.so">
      <left_joint>left_wheel_joint</left_joint>
      <right_joint>right_wheel_joint</right_joint>
      <wheel_separation>0.4</wheel_separation>
      <wheel_diameter>0.2</wheel_diameter>
      <command_topic>cmd_vel</command_topic>
      <odometry_topic>odom</odometry_topic>
      <publish_odom_tf>true</publish_odom_tf>
    </plugin>
  </model>
</sdf>
```

## Step 3: Create the Sensor Fusion Node

Create `sensor_fusion_robot/sensor_fusion_robot/sensor_fusion_node.py`:

```python
#!/usr/bin/env python3
"""
Sensor Fusion Node - Integrates data from multiple sensors to create a comprehensive perception of the environment.
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Imu, NavSatFix
from geometry_msgs.msg import PoseWithCovarianceStamped, Twist
from std_msgs.msg import String
from cv_bridge import CvBridge
import numpy as np
import math
from collections import deque

class SensorFusionNode(Node):
    def __init__(self):
        super().__init__('sensor_fusion_node')

        # Create subscribers for all sensors
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10)
        self.imu_sub = self.create_subscription(
            Imu, '/imu', self.imu_callback, 10)
        self.gps_sub = self.create_subscription(
            NavSatFix, '/gps', self.gps_callback, 10)

        # Create publishers for fused data
        self.fused_pose_pub = self.create_publisher(
            PoseWithCovarianceStamped, '/fused_pose', 10)
        self.environment_pub = self.create_publisher(
            String, '/environment_description', 10)

        # Initialize sensor data storage
        self.scan_data = None
        self.imu_data = None
        self.gps_data = None

        # Store recent data for fusion
        self.scan_history = deque(maxlen=10)
        self.imu_history = deque(maxlen=10)
        self.gps_history = deque(maxlen=10)

        # Create timer for fusion processing
        self.fusion_timer = self.create_timer(0.1, self.fusion_callback)  # 10 Hz

        # Initialize CV bridge
        self.bridge = CvBridge()

        self.get_logger().info('Sensor fusion node started')

    def scan_callback(self, msg):
        """Handle LiDAR scan data"""
        self.scan_data = msg
        self.scan_history.append(msg)

    def imu_callback(self, msg):
        """Handle IMU data"""
        self.imu_data = msg
        self.imu_history.append(msg)

    def gps_callback(self, msg):
        """Handle GPS data"""
        self.gps_data = msg
        self.gps_history.append(msg)

    def fusion_callback(self):
        """Process sensor fusion"""
        if not all([self.scan_data, self.imu_data, self.gps_data]):
            return

        # Create fused pose estimate
        fused_pose = PoseWithCovarianceStamped()
        fused_pose.header.stamp = self.get_clock().now().to_msg()
        fused_pose.header.frame_id = 'map'

        # Use GPS for position, IMU for orientation
        fused_pose.pose.pose.position.x = self.gps_data.latitude
        fused_pose.pose.pose.position.y = self.gps_data.longitude
        fused_pose.pose.pose.position.z = self.gps_data.altitude

        # Use IMU orientation
        fused_pose.pose.pose.orientation = self.imu_data.orientation

        # Estimate covariance based on sensor accuracies
        # GPS: ~2m accuracy, IMU: ~0.1rad accuracy
        fused_pose.pose.covariance = [
            4.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # x, y, z
            0.0, 4.0, 0.0, 0.0, 0.0, 0.0,  # roll, pitch, yaw
            0.0, 0.0, 9.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.01, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.01, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.01
        ]

        # Publish fused pose
        self.fused_pose_pub.publish(fused_pose)

        # Analyze environment from scan data
        environment_description = self.analyze_environment()
        env_msg = String()
        env_msg.data = environment_description
        self.environment_pub.publish(env_msg)

        self.get_logger().info(f'Fused pose published: x={fused_pose.pose.pose.position.x:.2f}, y={fused_pose.pose.pose.position.y:.2f}')

    def analyze_environment(self):
        """Analyze environment based on sensor data"""
        if not self.scan_data:
            return "Environment: Unknown"

        # Analyze LiDAR scan for obstacles
        ranges = np.array(self.scan_data.ranges)
        valid_ranges = ranges[np.isfinite(ranges)]  # Remove invalid measurements

        if len(valid_ranges) == 0:
            return "Environment: No valid measurements"

        # Find nearest obstacle
        min_distance = np.min(valid_ranges) if len(valid_ranges) > 0 else float('inf')

        # Count obstacles within 2m
        close_obstacles = np.sum(valid_ranges < 2.0)

        # Analyze distribution of measurements
        if len(valid_ranges) > 0:
            avg_distance = np.mean(valid_ranges)
            std_distance = np.std(valid_ranges)
        else:
            avg_distance = float('inf')
            std_distance = 0.0

        # Environment classification
        if min_distance < 0.5:
            env_type = "Very close obstacle"
        elif min_distance < 2.0:
            env_type = "Close obstacle"
        elif close_obstacles > 10:
            env_type = "Crowded environment"
        elif std_distance > 5.0:
            env_type = "Open environment"
        else:
            env_type = "Normal environment"

        return f"Environment: {env_type} | Nearest: {min_distance:.2f}m | Close obstacles: {close_obstacles} | Avg: {avg_distance:.2f}m"

def main(args=None):
    rclpy.init(args=args)
    node = SensorFusionNode()

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

## Step 4: Create the Object Detection Node

Create `sensor_fusion_robot/sensor_fusion_robot/object_detection_node.py`:

```python
#!/usr/bin/env python3
"""
Object Detection Node - Processes camera images to detect objects in the environment.
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from cv_bridge import CvBridge
import cv2
import numpy as np

class ObjectDetectionNode(Node):
    def __init__(self):
        super().__init__('object_detection_node')

        # Create subscriber for camera images
        self.image_sub = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10)

        # Create subscriber for camera info
        self.info_sub = self.create_subscription(
            CameraInfo, '/camera/camera_info', self.info_callback, 10)

        # Create publisher for detections
        self.detection_pub = self.create_publisher(
            Detection2DArray, '/detections', 10)

        # Initialize CV bridge
        self.bridge = CvBridge()

        # Camera parameters
        self.camera_matrix = None
        self.distortion_coeffs = None

        # Detection parameters
        self.min_area = 100
        self.max_area = 50000

        self.get_logger().info('Object detection node started')

    def info_callback(self, msg):
        """Handle camera info messages"""
        self.camera_matrix = np.array(msg.k).reshape(3, 3)
        self.distortion_coeffs = np.array(msg.d)

    def image_callback(self, msg):
        """Process incoming camera image"""
        try:
            # Convert ROS image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

            # Perform object detection
            detections = self.detect_objects(cv_image)

            # Publish detections
            self.detection_pub.publish(detections)

        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')

    def detect_objects(self, image):
        """Detect objects in the image"""
        detections = Detection2DArray()
        detections.header.stamp = self.get_clock().now().to_msg()
        detections.header.frame_id = 'camera_optical_frame'

        # Convert to HSV for color-based detection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define color ranges for different objects
        color_ranges = [
            ([0, 50, 50], [10, 255, 255], 'red_object'),      # Red
            ([36, 50, 50], [70, 255, 255], 'green_object'),   # Green
            ([100, 50, 50], [130, 255, 255], 'blue_object'),  # Blue
        ]

        for lower, upper, label in color_ranges:
            # Create mask for color range
            lower = np.array(lower)
            upper = np.array(upper)
            mask = cv2.inRange(hsv, lower, upper)

            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Process each contour
            for contour in contours:
                area = cv2.contourArea(contour)
                if self.min_area < area < self.max_area:
                    # Get bounding box
                    x, y, w, h = cv2.boundingRect(contour)

                    # Create detection
                    detection = Detection2D()
                    detection.header.stamp = detections.header.stamp
                    detection.header.frame_id = detections.header.frame_id

                    # Set bounding box
                    detection.bbox.center.x = x + w / 2
                    detection.bbox.center.y = y + h / 2
                    detection.bbox.size_x = w
                    detection.bbox.size_y = h

                    # Set classification
                    hypothesis = ObjectHypothesisWithPose()
                    hypothesis.hypothesis.class_id = label
                    hypothesis.hypothesis.score = 0.8  # Confidence score

                    detection.results.append(hypothesis)
                    detections.detections.append(detection)

        return detections

def main(args=None):
    rclpy.init(args=args)
    node = ObjectDetectionNode()

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

## Step 5: Create the Sensor Validation Node

Create `sensor_fusion_robot/sensor_fusion_robot/sensor_validation_node.py`:

```python
#!/usr/bin/env python3
"""
Sensor Validation Node - Validates sensor data quality and accuracy.
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Imu, NavSatFix
from std_msgs.msg import String, Float64
import numpy as np
from collections import deque

class SensorValidationNode(Node):
    def __init__(self):
        super().__init__('sensor_validation_node')

        # Create subscribers for all sensors
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10)
        self.imu_sub = self.create_subscription(
            Imu, '/imu', self.imu_callback, 10)
        self.gps_sub = self.create_subscription(
            NavSatFix, '/gps', self.gps_callback, 10)

        # Create publishers for validation results
        self.scan_quality_pub = self.create_publisher(
            Float64, '/scan_quality', 10)
        self.imu_quality_pub = self.create_publisher(
            Float64, '/imu_quality', 10)
        self.gps_quality_pub = self.create_publisher(
            Float64, '/gps_quality', 10)
        self.system_status_pub = self.create_publisher(
            String, '/system_status', 10)

        # Data history for validation
        self.scan_history = deque(maxlen=20)
        self.imu_history = deque(maxlen=20)
        self.gps_history = deque(maxlen=20)

        # Create timer for validation processing
        self.validation_timer = self.create_timer(1.0, self.validate_sensors)

        # Sensor quality scores
        self.scan_quality = 1.0
        self.imu_quality = 1.0
        self.gps_quality = 1.0

        self.get_logger().info('Sensor validation node started')

    def scan_callback(self, msg):
        """Handle LiDAR scan data"""
        self.scan_history.append(msg)

    def imu_callback(self, msg):
        """Handle IMU data"""
        self.imu_history.append(msg)

    def gps_callback(self, msg):
        """Handle GPS data"""
        self.gps_history.append(msg)

    def validate_sensors(self):
        """Validate all sensors and publish quality scores"""
        # Validate LiDAR
        if len(self.scan_history) > 0:
            self.scan_quality = self.validate_scan_quality(self.scan_history[-1])
            quality_msg = Float64()
            quality_msg.data = self.scan_quality
            self.scan_quality_pub.publish(quality_msg)

        # Validate IMU
        if len(self.imu_history) > 1:
            self.imu_quality = self.validate_imu_quality()
            quality_msg = Float64()
            quality_msg.data = self.imu_quality
            self.imu_quality_pub.publish(quality_msg)

        # Validate GPS
        if len(self.gps_history) > 1:
            self.gps_quality = self.validate_gps_quality()
            quality_msg = Float64()
            quality_msg.data = self.gps_quality
            self.gps_quality_pub.publish(quality_msg)

        # Overall system status
        avg_quality = (self.scan_quality + self.imu_quality + self.gps_quality) / 3.0
        status_msg = String()
        if avg_quality > 0.8:
            status_msg.data = "SYSTEM_OK"
        elif avg_quality > 0.5:
            status_msg.data = "SYSTEM_DEGRADED"
        else:
            status_msg.data = "SYSTEM_FAULT"

        self.system_status_pub.publish(status_msg)

        self.get_logger().info(
            f'Sensor quality - LiDAR: {self.scan_quality:.2f}, '
            f'IMU: {self.imu_quality:.2f}, '
            f'GPS: {self.gps_quality:.2f}, '
            f'Status: {status_msg.data}'
        )

    def validate_scan_quality(self, scan_msg):
        """Validate LiDAR scan quality"""
        if not scan_msg.ranges:
            return 0.0

        ranges = np.array(scan_msg.ranges)
        valid_ranges = ranges[np.isfinite(ranges)]

        if len(valid_ranges) == 0:
            return 0.0

        # Calculate quality based on several factors
        # 1. Percentage of valid measurements
        valid_ratio = len(valid_ranges) / len(ranges)

        # 2. Range distribution (should not be all the same)
        if len(valid_ranges) > 1:
            range_variance = np.var(valid_ranges)
            variance_score = min(1.0, range_variance / 10.0)  # Normalize
        else:
            variance_score = 0.0

        # 3. Average range (should be reasonable)
        avg_range = np.mean(valid_ranges)
        range_score = 1.0 if 0.1 < avg_range < 20.0 else 0.5

        # Combine scores
        quality = (valid_ratio * 0.4 + variance_score * 0.3 + range_score * 0.3)
        return min(1.0, max(0.0, quality))

    def validate_imu_quality(self):
        """Validate IMU quality based on consistency"""
        if len(self.imu_history) < 2:
            return 1.0

        # Calculate consistency of measurements
        angular_velocities = []
        linear_accelerations = []

        for msg in list(self.imu_history)[-10:]:  # Last 10 measurements
            angular_velocities.append([
                msg.angular_velocity.x,
                msg.angular_velocity.y,
                msg.angular_velocity.z
            ])
            linear_accelerations.append([
                msg.linear_acceleration.x,
                msg.linear_acceleration.y,
                msg.linear_acceleration.z
            ])

        # Calculate variance of measurements
        ang_vel_var = np.var(angular_velocities, axis=0)
        lin_acc_var = np.var(linear_accelerations, axis=0)

        # High variance might indicate instability, but some variance is normal
        avg_ang_var = np.mean(ang_vel_var)
        avg_lin_var = np.mean(lin_acc_var)

        # Normalize to 0-1 scale (assuming typical values)
        ang_score = max(0.0, min(1.0, 1.0 - avg_ang_var / 1.0))
        lin_score = max(0.0, min(1.0, 1.0 - avg_lin_var / 10.0))

        return (ang_score + lin_score) / 2.0

    def validate_gps_quality(self):
        """Validate GPS quality based on accuracy"""
        if len(self.gps_history) == 0:
            return 1.0

        # Use the latest GPS message
        latest_gps = self.gps_history[-1]

        # GPS status (assuming status.status field exists)
        # In simulation, we'll assume good quality if position is reasonable
        if abs(latest_gps.latitude) > 90 or abs(latest_gps.longitude) > 180:
            return 0.0

        # In real GPS, we would check status and covariance
        # For simulation, assume good quality if position is valid
        return 1.0

def main(args=None):
    rclpy.init(args=args)
    node = SensorValidationNode()

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

## Step 6: Create Launch Files

Create `sensor_fusion_robot/launch/sensor_robot.launch.py`:

```python
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Declare launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    enable_viz = LaunchConfiguration('enable_viz', default='true')

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use simulation time if true'
        ),
        DeclareLaunchArgument(
            'enable_viz',
            default_value='true',
            description='Enable visualization'
        ),

        # Sensor fusion node
        Node(
            package='sensor_fusion_robot',
            executable='sensor_fusion_node',
            name='sensor_fusion_node',
            parameters=[
                {'use_sim_time': use_sim_time}
            ],
            output='screen'
        ),

        # Object detection node
        Node(
            package='sensor_fusion_robot',
            executable='object_detection_node',
            name='object_detection_node',
            parameters=[
                {'use_sim_time': use_sim_time}
            ],
            output='screen'
        ),

        # Sensor validation node
        Node(
            package='sensor_fusion_robot',
            executable='sensor_validation_node',
            name='sensor_validation_node',
            parameters=[
                {'use_sim_time': use_sim_time}
            ],
            output='screen'
        ),

        # RViz2 for visualization (optional)
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', os.path.join(
                get_package_share_directory('sensor_fusion_robot'),
                'config',
                'sensor_fusion.rviz'
            )],
            condition=IfCondition(enable_viz),
            parameters=[
                {'use_sim_time': use_sim_time}
            ]
        )
    ])
```

Create `sensor_fusion_robot/launch/gazebo_sensor_robot.launch.py`:

```python
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Declare launch arguments
    world = LaunchConfiguration('world', default='empty.sdf')
    robot_name = LaunchConfiguration('robot_name', default='sensor_robot')

    return LaunchDescription([
        DeclareLaunchArgument(
            'world',
            default_value='empty.sdf',
            description='Choose one of the world files from Gazebo Worlds'
        ),
        DeclareLaunchArgument(
            'robot_name',
            default_value='sensor_robot',
            description='Name of the robot to spawn'
        ),

        # Launch Gazebo
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                get_package_share_directory('gazebo_ros'),
                '/launch',
                '/gazebo.launch.py'
            ]),
            launch_arguments={
                'world': world
            }.items()
        ),

        # Spawn robot in Gazebo
        Node(
            package='gazebo_ros',
            executable='spawn_entity.py',
            arguments=[
                '-entity', robot_name,
                '-file', PathJoinSubstitution([
                    get_package_share_directory('sensor_fusion_robot'),
                    'models',
                    'sensor_robot.sdf'
                ])
            ],
            output='screen'
        ),

        # Launch sensor processing nodes
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                get_package_share_directory('sensor_fusion_robot'),
                '/launch',
                '/sensor_robot.launch.py'
            ])
        )
    ])
```

## Step 7: Create RViz Configuration

Create `sensor_fusion_robot/config/sensor_fusion.rviz`:

```yaml
Panels:
  - Class: rviz_common/Displays
    Help Height: 78
    Name: Displays
    Property Tree Widget:
      Expanded:
        - /Global Options1
        - /Status1
        - /Image1
        - /LaserScan1
        - /PoseWithCovariance1
        - /MarkerArray1
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
    - Class: rviz_default_plugins/Image
      Enabled: true
      Max Value: 1
      Min Value: 0
      Name: Image
      Overlay Alpha: 0.5
      Topic:
        Depth: 5
        Durability Policy: Volatile
        History Policy: Keep Last
        Reliability Policy: Reliable
        Value: /camera/image_raw
      Value: true
    - Alpha: 1
      Autocompute Intensity Bounds: true
      Autocompute Value Bounds:
        Max Value: 10
        Min Value: -10
        Value: true
      Axis: Z
      Channel Name: intensity
      Class: rviz_default_plugins/LaserScan
      Color: 255; 255; 255
      Color Transformer: Intensity
      Decay Time: 0
      Enabled: true
      Invert Rainbow: false
      Max Color: 255; 255; 255
      Max Intensity: 4096
      Min Color: 0; 0; 0
      Min Intensity: 0
      Name: LaserScan
      Position Transformer: XYZ
      Queue Size: 10
      Selectable: true
      Size (Pixels): 3
      Size (m): 0.009999999776482582
      Style: Flat Squares
      Topic:
        Depth: 5
        Durability Policy: Volatile
        History Policy: Keep Last
        Reliability Policy: Best Effort
        Value: /scan
      Use Fixed Frame: true
      Use rainbow: true
      Value: true
    - Class: rviz_default_plugins/PoseWithCovariance
      Color: 255; 25; 0
      Covariance:
        Orientation:
          Alpha: 0.5
          Color: 255; 255; 127
          Color Style: Unique
          Frame: Local
          Offset: 1
          Scale: 1
          Value: true
        Position:
          Alpha: 0.30000001192092896
          Color: 204; 51; 204
          Scale: 1
          Value: true
      Enabled: true
      Head Length: 0.30000001192092896
      Head Radius: 0.10000000149011612
      Name: PoseWithCovariance
      Shaft Length: 1
      Shaft Radius: 0.05000000074505806
      Shape: Arrow
      Topic:
        Depth: 5
        Durability Policy: Volatile
        History Policy: Keep Last
        Reliability Policy: Reliable
        Value: /fused_pose
      Value: true
    - Class: rviz_default_plugins/MarkerArray
      Enabled: true
      Name: MarkerArray
      Namespaces:
        {}
      Topic:
        Depth: 5
        Durability Policy: Volatile
        History Policy: Keep Last
        Reliability Policy: Reliable
        Value: /detections
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
      Distance: 10
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
  Image:
    collapsed: false
  QMainWindow State: 000000ff00000000fd0000000400000000000001560000039ffc0200000009fb0000001200530065006c0065006300740069006f006e00000001e10000009b0000005c00fffffffb0000001e0054006f006f006c002000500072006f007000650072007400690065007302000001ed000001df00000185000000a3fb000000120056006900650077007300200054006f006f02000001df000002110000018500000122fb000000200054006f006f006c002000500072006f0070006500720074006900650073003203000002880000011d000002210000017afb000000100044006900730070006c006100790073010000003d0000039f000000c900fffffffb0000002000730065006c0065006300740069006f006e00200062007500660066006500720200000138000000aa0000023a00000294fb00000014005700690064006500530074006500720065006f02000000e6000000d2000003ee0000030bfb0000000c004b0069006e0065006300740200000186000001060000030c00000261fb0000000a0049006d006100670065010000003d0000039f0000001600ffffff000000010000010f0000039ffc0200000003fb0000001e0054006f006f006c002000500072006f00700065007200740069006500730100000041000000780000000000000000fb0000000a00560069006500770073000000003d0000039f000000a400fffffffb0000001200530065006c0065006300740069006f006e010000025a000000b200000000000000000000000200000490000000a9fc0100000001fb0000000a00560069006500770073030000004e00000080000002e100000197000000030000073f0000003efc0100000002fb0000000800540069006d006501000000000000073f000002eb00fffffffb0000000800540069006d00650100000000000004500000000000000000000004cc0000039f00000004000000040000000800000008fc0000000100000002000000010000000a0054006f006f006c00730100000000ffffffff0000000000000000
  Width: 1853
  X: 67
  Y: 27
```

## Step 8: Update Package Configuration

Update `sensor_fusion_robot/setup.py`:

```python
from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'sensor_fusion_robot'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'models'), glob('models/*.sdf')),
        (os.path.join('share', package_name, 'config'), glob('config/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your.email@example.com',
    description='Sensor fusion robot for simulation',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'sensor_fusion_node = sensor_fusion_robot.sensor_fusion_node:main',
            'object_detection_node = sensor_fusion_robot.object_detection_node:main',
            'sensor_validation_node = sensor_fusion_robot.sensor_validation_node:main',
        ],
    },
)
```

## Step 9: Build and Test

Build the package:

```bash
cd ~/ros2_ws
colcon build --packages-select sensor_fusion_robot
source install/setup.bash
```

### Test the System

1. **Start Gazebo with the sensor robot**:
```bash
# Launch Gazebo with the robot
ros2 launch sensor_fusion_robot gazebo_sensor_robot.launch.py
```

2. **In another terminal, launch the sensor processing nodes**:
```bash
# Launch sensor processing nodes
ros2 launch sensor_fusion_robot sensor_robot.launch.py
```

3. **Monitor sensor data**:
```bash
# Check sensor topics
ros2 topic list | grep /sensor_robot

# Monitor fused pose
ros2 topic echo /fused_pose

# Monitor detections
ros2 topic echo /detections

# Monitor sensor quality
ros2 topic echo /scan_quality
ros2 topic echo /imu_quality
ros2 topic echo /gps_quality
```

4. **Visualize in RViz2**:
```bash
# If not launched automatically
rviz2 -d `ros2 pkg prefix sensor_fusion_robot`/share/sensor_fusion_robot/config/sensor_fusion.rviz
```

## Understanding the System

This sensor fusion robot demonstrates:

1. **Multi-sensor Integration**: Camera, LiDAR, IMU, and GPS working together
2. **Data Fusion**: Combining sensor data for comprehensive environment understanding
3. **Object Detection**: Processing camera images to identify objects
4. **Sensor Validation**: Monitoring sensor quality and system health
5. **Real-time Processing**: All processing happening in real-time

## Challenges

### Challenge 1: Add More Sensor Types
Add additional sensors like sonar or thermal cameras to the robot model.

<details>
<summary>Hint</summary>

Modify the SDF file to include additional sensor definitions and update the fusion node to process the new sensor data.
</details>

### Challenge 2: Improve Object Detection
Enhance the object detection algorithm with more sophisticated techniques.

<details>
<summary>Hint</summary>

Implement template matching, feature detection, or integrate a deep learning model for better detection accuracy.
</details>

### Challenge 3: Add Sensor Calibration
Create a calibration system for the sensors.

<details>
<summary>Hint</summary>

Implement calibration procedures that adjust sensor parameters based on known reference objects or measurements.
</details>

### Challenge 4: Create SLAM Integration
Integrate SLAM algorithms with the sensor fusion system.

<details>
<summary>Hint</summary>

Connect the sensor data to SLAM algorithms like Cartographer or RTAB-Map for mapping and localization.
</details>

## Verification Checklist

- [ ] Robot model loads in Gazebo with all sensors
- [ ] All sensor topics are publishing data
- [ ] Sensor fusion node processes data correctly
- [ ] Object detection identifies objects in camera feed
- [ ] Sensor validation provides quality metrics
- [ ] Fused pose is published with covariance
- [ ] RViz2 visualization works correctly
- [ ] System status reflects sensor health

## Common Issues

### Sensor Data Issues
```bash
# Check if sensors are publishing
ros2 topic list | grep /sensor_robot
ros2 topic hz /scan
ros2 topic hz /camera/image_raw
```

### Gazebo Integration Issues
```bash
# Check Gazebo topics
ros2 topic list | grep /gazebo

# Verify robot spawned
ros2 service call /gazebo/get_world_properties gazebo_msgs/srv/GetWorldProperties
```

### Processing Performance
```bash
# Monitor CPU usage
htop

# Check processing rates
ros2 topic hz /fused_pose
```

## Summary

In this exercise, you learned to:
- Create a robot model with multiple realistic sensors
- Integrate sensor data through fusion algorithms
- Process perception data with computer vision
- Validate sensor accuracy and system health
- Visualize sensor data in RViz2

## Next Steps

Continue to [Week 5: Robot Modeling](../../module-2-digital-twin/week-05/introduction) to learn about creating accurate robot models with URDF and Xacro.