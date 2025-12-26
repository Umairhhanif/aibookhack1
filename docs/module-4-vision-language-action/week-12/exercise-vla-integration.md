---
sidebar_position: 4
---

# Exercise: VLA System Integration

In this comprehensive exercise, you'll build a complete Vision-Language-Action (VLA) system that integrates perception, reasoning, and action in a real robotic platform. This will demonstrate the end-to-end capabilities of modern VLA models.

## Objective

Create a complete VLA system that:
1. **Processes multimodal inputs** (vision and language)
2. **Performs embodied reasoning** about tasks
3. **Generates executable robot actions**
4. **Integrates with ROS 2 control systems**
5. **Implements safety and validation measures**

## Prerequisites

- Complete Week 1-12 lessons
- ROS 2 workspace set up (`~/ros2_ws`)
- Understanding of VLA models and architectures
- Basic Python and C++ programming skills
- Access to a robot platform (real or simulated)

## Step 1: Create the VLA Integration Package

```bash
cd ~/ros2_ws/src

ros2 pkg create --build-type ament_python vla_integration_system \
    --dependencies rclpy std_msgs sensor_msgs geometry_msgs vision_msgs cv_bridge tf2_ros tf2_geometry_msgs message_filters action_msgs
```

## Step 2: Create the VLA Controller Node

Create `vla_integration_system/vla_integration_system/vla_controller.py`:

```python
#!/usr/bin/env python3
"""
VLA Controller Node
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from geometry_msgs.msg import Twist, PoseStamped
from std_msgs.msg import String, Float32
from vision_msgs.msg import Detection2DArray
from action_msgs.msg import GoalStatus
from cv_bridge import CvBridge
import torch
import numpy as np
import time
from typing import Dict, Any, Optional
import threading
from queue import Queue

class VLAControllerNode(Node):
    def __init__(self):
        super().__init__('vla_controller_node')

        # Create subscribers
        self.image_sub = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10)
        self.command_sub = self.create_subscription(
            String, '/natural_language_command', self.command_callback, 10)
        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10)

        # Create publishers
        self.action_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.status_pub = self.create_publisher(String, '/vla/status', 10)
        self.confidence_pub = self.create_publisher(Float32, '/vla/confidence', 10)
        self.debug_pub = self.create_publisher(String, '/vla/debug_info', 10)

        # Initialize CV bridge
        self.bridge = CvBridge()

        # Initialize VLA model
        self.vla_model = self.initialize_vla_model()

        # Device setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.vla_model:
            self.vla_model.to(self.device)
            self.vla_model.eval()

        # System state
        self.current_image = None
        self.current_command = None
        self.current_joints = None
        self.is_processing = False

        # Processing queues
        self.input_queue = Queue(maxsize=10)
        self.output_queue = Queue(maxsize=10)

        # Threading for processing
        self.processing_thread = threading.Thread(target=self.processing_loop, daemon=True)
        self.processing_thread.start()

        # Performance metrics
        self.inference_times = []
        self.success_rates = []
        self.confidence_scores = []

        # Safety parameters
        self.safety_threshold = 0.3
        self.max_action_magnitude = 1.0

        self.get_logger().info('VLA controller node started')

    def initialize_vla_model(self):
        """Initialize VLA model (conceptual - in practice, load actual model)"""
        try:
            # In practice, you would load a pre-trained VLA model like:
            # - RT-1 (Robotics Transformer 1)
            # - OpenVLA
            # - EmbodiedGPT
            # - Other VLA architectures

            # For this exercise, we'll create a mock model
            class MockVLA:
                def __call__(self, images, text_commands):
                    # Simulate VLA inference
                    batch_size = images.size(0)

                    # Generate mock actions (7-DoF: 3 pos + 3 rot + 1 gripper)
                    actions = torch.randn(batch_size, 7) * 0.1  # Small random actions

                    # Generate mock confidence scores
                    confidence = torch.rand(batch_size, 1) * 0.5 + 0.5  # 0.5-1.0 range

                    return actions, confidence

            return MockVLA()

        except Exception as e:
            self.get_logger().error(f'Failed to initialize VLA model: {e}')
            return None

    def image_callback(self, msg):
        """Handle incoming camera images"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

            # Preprocess image for VLA model
            processed_image = self.preprocess_image(cv_image)

            # Store image for processing
            self.current_image = processed_image

            # If we have a command ready, queue for processing
            if self.current_command and not self.is_processing:
                self.queue_processing_item(processed_image, self.current_command)

        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')

    def command_callback(self, msg):
        """Handle natural language commands"""
        command = msg.data

        # Store command for processing
        self.current_command = command

        # If we have an image ready, queue for processing
        if self.current_image and not self.is_processing:
            self.queue_processing_item(self.current_image, command)

    def joint_state_callback(self, msg):
        """Handle joint state updates"""
        self.current_joints = msg

    def queue_processing_item(self, image, command):
        """Queue item for processing"""
        try:
            if not self.input_queue.full():
                self.input_queue.put((image, command))
                self.is_processing = True
            else:
                self.get_logger().warn('Input queue full, dropping command')
        except Exception as e:
            self.get_logger().error(f'Error queuing processing item: {e}')

    def processing_loop(self):
        """Background processing loop"""
        while rclpy.ok():
            try:
                # Get item from queue
                image, command = self.input_queue.get(timeout=0.1)

                # Process with VLA model
                start_time = time.time()
                action, confidence = self.process_with_vla(image, command)
                processing_time = time.time() - start_time

                # Validate and execute action
                if self.validate_action(action, confidence):
                    self.execute_action(action)

                    # Log performance metrics
                    self.inference_times.append(processing_time)
                    self.confidence_scores.append(confidence.item())

                    # Publish confidence
                    confidence_msg = Float32()
                    confidence_msg.data = confidence.item()
                    self.confidence_pub.publish(confidence_msg)

                    # Publish status
                    status_msg = String()
                    status_msg.data = f"EXECUTED: Action magnitude={torch.norm(action).item():.3f}, Confidence={confidence.item():.3f}, Time={processing_time:.3f}s"
                    self.status_pub.publish(status_msg)

                    self.get_logger().info(f'VLA executed action: {action[:3].tolist()}, confidence: {confidence.item():.3f}')

                else:
                    # Action rejected due to safety/validation
                    status_msg = String()
                    status_msg.data = f"REJECTED: Low confidence or safety violation, Confidence={confidence.item():.3f}"
                    self.status_pub.publish(status_msg)

                    self.get_logger().warn(f'VLA action rejected: confidence={confidence.item():.3f}')

                # Mark processing as complete
                self.is_processing = False

            except Exception as e:
                self.get_logger().error(f'Error in processing loop: {e}')
                self.is_processing = False

    def preprocess_image(self, image):
        """Preprocess image for VLA model"""
        # Resize image
        resized = cv2.resize(image, (224, 224))

        # Convert to tensor and normalize
        tensor = torch.from_numpy(resized).float().permute(2, 0, 1) / 255.0

        # Add batch dimension
        batch_tensor = tensor.unsqueeze(0).to(self.device)

        return batch_tensor

    def process_with_vla(self, image, command):
        """Process input with VLA model"""
        if not self.vla_model:
            self.get_logger().error('VLA model not available')
            return torch.zeros(1, 7), torch.tensor([0.0])

        try:
            # Tokenize command (simplified - in practice, use proper tokenizer)
            # For this example, we'll convert command to a simple embedding
            command_tokens = self.tokenize_command(command)
            command_tensor = torch.tensor(command_tokens).unsqueeze(0).to(self.device)

            # Run VLA inference
            with torch.no_grad():
                actions, confidence = self.vla_model(image, command_tensor)

            return actions, confidence

        except Exception as e:
            self.get_logger().error(f'Error in VLA inference: {e}')
            return torch.zeros(1, 7), torch.tensor([0.0])

    def tokenize_command(self, command):
        """Simple command tokenization (in practice, use proper tokenizer)"""
        # In a real implementation, use a pre-trained tokenizer
        # For this example, use simple character-level encoding
        tokens = [ord(c) % 1000 for c in command[:64]]  # Limit to 64 characters
        tokens += [0] * (64 - len(tokens))  # Pad to fixed length
        return tokens

    def validate_action(self, action, confidence):
        """Validate action for safety and reasonableness"""
        # Check confidence threshold
        if confidence.item() < self.safety_threshold:
            self.get_logger().warn(f'Action confidence too low: {confidence.item():.3f} < {self.safety_threshold}')
            return False

        # Check action magnitude (prevent extreme movements)
        action_norm = torch.norm(action).item()
        if action_norm > self.max_action_magnitude:
            self.get_logger().warn(f'Action magnitude too large: {action_norm} > {self.max_action_magnitude}')
            return False

        # Check for NaN or infinite values
        if torch.isnan(action).any() or torch.isinf(action).any():
            self.get_logger().warn('Action contains NaN or infinite values')
            return False

        # Additional safety checks could go here:
        # - Check for collision with environment
        # - Verify action is within robot's capabilities
        # - Check for joint limits
        # - Validate against current robot state

        return True

    def execute_action(self, action_tensor):
        """Execute the generated action"""
        # Convert action tensor to robot command
        action = action_tensor.cpu().numpy().squeeze()

        # For this example, assume action is [vx, vy, vz, wx, wy, wz, gripper]
        cmd_vel = Twist()
        cmd_vel.linear.x = float(action[0]) if len(action) > 0 else 0.0
        cmd_vel.linear.y = float(action[1]) if len(action) > 1 else 0.0
        cmd_vel.linear.z = float(action[2]) if len(action) > 2 else 0.0
        cmd_vel.angular.x = float(action[3]) if len(action) > 3 else 0.0
        cmd_vel.angular.y = float(action[4]) if len(action) > 4 else 0.0
        cmd_vel.angular.z = float(action[5]) if len(action) > 5 else 0.0

        # Publish action command
        self.action_pub.publish(cmd_vel)

    def get_performance_metrics(self):
        """Get performance metrics"""
        if not self.inference_times:
            return {'avg_inference_time': 0.0, 'avg_confidence': 0.0}

        avg_inf_time = np.mean(self.inference_times[-100:]) if self.inference_times else 0.0
        avg_confidence = np.mean(self.confidence_scores[-100:]) if self.confidence_scores else 0.0

        return {
            'avg_inference_time': avg_inf_time,
            'avg_confidence': avg_confidence,
            'total_inferences': len(self.inference_times),
            'fps': 1.0 / avg_inf_time if avg_inf_time > 0 else 0.0
        }

def main(args=None):
    rclpy.init(args=args)
    node = VLAControllerNode()

    # Timer for performance monitoring
    perf_timer = node.create_timer(5.0, lambda: print(f"Performance: {node.get_performance_metrics()}"))

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

## Step 3: Create the VLA Perception Node

Create `vla_integration_system/vla_integration_system/vla_perception.py`:

```python
#!/usr/bin/env python3
"""
VLA Perception Node
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, LaserScan
from vision_msgs.msg import Detection2DArray, ObjectHypothesisWithPose
from geometry_msgs.msg import Point, Vector3
from std_msgs.msg import Header, ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray
from cv_bridge import CvBridge
import torch
import numpy as np
import cv2
from typing import Dict, Any, List

class VLA PerceptionNode(Node):
    def __init__(self):
        super().__init__('vla_perception_node')

        # Create subscribers
        self.image_sub = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10)
        self.pointcloud_sub = self.create_subscription(
            PointCloud2, '/lidar/points', self.pointcloud_callback, 10)
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10)

        # Create publishers
        self.detection_pub = self.create_publisher(Detection2DArray, '/vla/detections', 10)
        self.perception_pub = self.create_publisher(String, '/vla/perception', 10)
        self.visualization_pub = self.create_publisher(MarkerArray, '/vla/visualization', 10)

        # Initialize CV bridge
        self.bridge = CvBridge()

        # Initialize perception models
        self.vision_model = self.initialize_vision_model()
        self.fusion_model = self.initialize_fusion_model()

        # Device setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Perception state
        self.current_image = None
        self.current_pointcloud = None
        self.current_scan = None

        # Processing parameters
        self.confidence_threshold = 0.5
        self.max_detections = 20

        # Visualization parameters
        self.marker_id_counter = 0

        self.get_logger().info('VLA perception node started')

    def initialize_vision_model(self):
        """Initialize vision perception model"""
        try:
            # In practice, load a pre-trained object detection model
            # For this exercise, we'll create a mock model
            class MockVisionModel:
                def __call__(self, image):
                    # Simulate object detection
                    # In practice, use YOLO, DETR, or similar
                    batch_size = image.size(0)

                    # Generate mock detections
                    num_detections = np.random.randint(1, 5)
                    detections = []

                    for i in range(num_detections):
                        detection = {
                            'bbox_center_x': float(np.random.uniform(0, 640)),
                            'bbox_center_y': float(np.random.uniform(0, 480)),
                            'bbox_width': float(np.random.uniform(50, 200)),
                            'bbox_height': float(np.random.uniform(50, 200)),
                            'class_id': int(np.random.uniform(0, 80)),  # COCO classes
                            'confidence': float(np.random.uniform(0.6, 0.99)),
                            'class_name': 'object'  # In practice, use actual class names
                        }
                        detections.append(detection)

                    return detections

            return MockVisionModel()

        except Exception as e:
            self.get_logger().error(f'Error initializing vision model: {e}')
            return None

    def initialize_fusion_model(self):
        """Initialize sensor fusion model"""
        try:
            # In practice, load a fusion model that combines vision, LiDAR, etc.
            class MockFusionModel:
                def __call__(self, vision_detections, lidar_data, scan_data):
                    # Simulate sensor fusion
                    # Combine information from multiple sensors
                    fused_detections = []

                    for detection in vision_detections:
                        # Add 3D information from LiDAR if available
                        detection_3d = detection.copy()
                        detection_3d['position_3d'] = {
                            'x': np.random.uniform(-5, 5),
                            'y': np.random.uniform(-5, 5),
                            'z': np.random.uniform(0, 2)
                        }
                        detection_3d['distance'] = np.random.uniform(1, 10)
                        fused_detections.append(detection_3d)

                    return fused_detections

            return MockFusionModel()

        except Exception as e:
            self.get_logger().error(f'Error initializing fusion model: {e}')
            return None

    def image_callback(self, msg):
        """Process camera image"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            self.current_image = cv_image

            # If we have other sensor data, run perception
            if self.current_pointcloud is not None:
                self.run_multimodal_perception(msg.header)

        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')

    def pointcloud_callback(self, msg):
        """Process LiDAR point cloud"""
        try:
            # Parse point cloud data (simplified - in practice, use proper parsing)
            self.current_pointcloud = msg

            # If we have other sensor data, run perception
            if self.current_image is not None:
                self.run_multimodal_perception(msg.header)

        except Exception as e:
            self.get_logger().error(f'Error processing point cloud: {e}')

    def scan_callback(self, msg):
        """Process laser scan"""
        try:
            self.current_scan = msg

            # If we have other sensor data, run perception
            if self.current_image is not None:
                self.run_multimodal_perception(msg.header)

        except Exception as e:
            self.get_logger().error(f'Error processing scan: {e}')

    def run_multimodal_perception(self, header):
        """Run multimodal perception with VLA model"""
        if not all([self.current_image, self.current_pointcloud, self.current_scan]):
            return

        try:
            # Process vision data
            vision_detections = self.vision_model(
                self.preprocess_image(self.current_image)
            )

            # Process LiDAR data (simplified)
            lidar_data = self.parse_pointcloud(self.current_pointcloud)

            # Process scan data (simplified)
            scan_data = {
                'ranges': list(self.current_scan.ranges),
                'angles': [self.current_scan.angle_min + i * self.current_scan.angle_increment
                          for i in range(len(self.current_scan.ranges))]
            }

            # Fuse sensor data
            if self.fusion_model:
                fused_detections = self.fusion_model(vision_detections, lidar_data, scan_data)
            else:
                fused_detections = vision_detections  # Use vision only if fusion not available

            # Filter detections by confidence
            filtered_detections = [
                det for det in fused_detections
                if det.get('confidence', 0) > self.confidence_threshold
            ][:self.max_detections]

            # Publish detections
            self.publish_detections(filtered_detections, header)

            # Publish perception summary
            perception_summary = {
                'num_detections': len(filtered_detections),
                'timestamp': header.stamp.sec + header.stamp.nanosec * 1e-9,
                'detection_classes': [det.get('class_name', 'unknown') for det in filtered_detections]
            }

            summary_msg = String()
            summary_msg.data = str(perception_summary)
            self.perception_pub.publish(summary_msg)

            self.get_logger().info(f'VLA perception: {len(filtered_detections)} objects detected')

        except Exception as e:
            self.get_logger().error(f'Error in multimodal perception: {e}')

    def preprocess_image(self, image):
        """Preprocess image for vision model"""
        # Resize to model input size
        resized = cv2.resize(image, (640, 480))

        # Convert to tensor
        tensor = torch.from_numpy(resized).float().permute(2, 0, 1) / 255.0
        tensor = tensor.unsqueeze(0).to(self.device)

        return tensor

    def parse_pointcloud(self, pointcloud_msg):
        """Parse PointCloud2 message (simplified)"""
        # In practice, use proper PointCloud2 parsing
        # For this example, return a placeholder
        return {
            'points': [],
            'intensities': [],
            'frame_id': pointcloud_msg.header.frame_id
        }

    def publish_detections(self, detections, header):
        """Publish detection results"""
        detection_array = Detection2DArray()
        detection_array.header = header

        for detection in detections:
            detection_2d = Detection2D()
            detection_2d.header = header

            # Set bounding box
            bbox = detection_2d.bbox
            bbox.center.x = detection.get('bbox_center_x', 0.0)
            bbox.center.y = detection.get('bbox_center_y', 0.0)
            bbox.size_x = detection.get('bbox_width', 0.0)
            bbox.size_y = detection.get('bbox_height', 0.0)

            # Set classification result
            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = detection.get('class_name', 'unknown')
            hypothesis.hypothesis.score = detection.get('confidence', 0.0)

            detection_2d.results.append(hypothesis)
            detection_array.detections.append(detection_2d)

        self.detection_pub.publish(detection_array)

        # Publish visualization markers
        self.publish_visualization_markers(detections, header)

    def publish_visualization_markers(self, detections, header):
        """Publish visualization markers for RViz"""
        marker_array = MarkerArray()
        marker_array.header = header

        for detection in detections:
            # Create marker for detection
            marker = Marker()
            marker.header = header
            marker.ns = "vla_detections"
            marker.id = self.marker_id_counter
            self.marker_id_counter += 1
            marker.type = Marker.TEXT_VIEW_FACING
            marker.action = Marker.ADD

            # Position marker at object location
            if 'position_3d' in detection:
                pos_3d = detection['position_3d']
                marker.pose.position.x = pos_3d['x']
                marker.pose.position.y = pos_3d['y']
                marker.pose.position.z = pos_3d['z'] + 0.5  # Raise above object
            else:
                # Use 2D projection if 3D not available
                marker.pose.position.x = detection.get('bbox_center_x', 0.0) / 100.0  # Scale down
                marker.pose.position.y = detection.get('bbox_center_y', 0.0) / 100.0
                marker.pose.position.z = 1.0  # Fixed height

            # Set text
            marker.text = f"{detection.get('class_name', 'object')}: {detection.get('confidence', 0.0):.2f}"

            # Set scale
            marker.scale.x = 0.1
            marker.scale.y = 0.1
            marker.scale.z = 0.1

            # Set color
            marker.color.r = 1.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.color.a = 1.0

            marker_array.markers.append(marker)

            # Add bounding box if 3D position available
            if 'position_3d' in detection:
                bbox_marker = Marker()
                bbox_marker.header = header
                bbox_marker.ns = "vla_bboxes"
                bbox_marker.id = self.marker_id_counter
                self.marker_id_counter += 1
                bbox_marker.type = Marker.LINE_STRIP
                bbox_marker.action = Marker.ADD

                # Simple bounding box visualization
                pos_3d = detection['position_3d']
                width = detection.get('bbox_width', 1.0) / 100.0
                height = detection.get('bbox_height', 1.0) / 100.0

                # Create bounding box points
                bbox_points = [
                    Point(x=pos_3d['x'] - width/2, y=pos_3d['y'] - height/2, z=pos_3d['z']),
                    Point(x=pos_3d['x'] + width/2, y=pos_3d['y'] - height/2, z=pos_3d['z']),
                    Point(x=pos_3d['x'] + width/2, y=pos_3d['y'] + height/2, z=pos_3d['z']),
                    Point(x=pos_3d['x'] - width/2, y=pos_3d['y'] + height/2, z=pos_3d['z']),
                    Point(x=pos_3d['x'] - width/2, y=pos_3d['y'] - height/2, z=pos_3d['z'])  # Close the box
                ]

                bbox_marker.points = bbox_points
                bbox_marker.scale.x = 0.02
                bbox_marker.color.r = 0.0
                bbox_marker.color.g = 1.0
                bbox_marker.color.b = 0.0
                bbox_marker.color.a = 0.8

                marker_array.markers.append(bbox_marker)

        self.visualization_pub.publish(marker_array)

def main(args=None):
    rclpy.init(args=args)
    node = VLAPerceptionNode()

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

## Step 4: Create the VLA Reasoning Node

Create `vla_integration_system/vla_integration_system/vla_reasoning.py`:

```python
#!/usr/bin/env python3
"""
VLA Reasoning Node
"""
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from vision_msgs.msg import Detection2DArray
from sensor_msgs.msg import Image
from tf2_ros import TransformListener, Buffer
from typing import Dict, Any, List
import json
import numpy as np

class VLAReasoningNode(Node):
    def __init__(self):
        super().__init__('vla_reasoning_node')

        # Create subscribers
        self.command_sub = self.create_subscription(
            String, '/natural_language_command', self.command_callback, 10)
        self.detection_sub = self.create_subscription(
            Detection2DArray, '/vla/detections', self.detection_callback, 10)
        self.image_sub = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10)

        # Create publishers
        self.action_plan_pub = self.create_publisher(String, '/vla/action_plan', 10)
        self.reasoning_pub = self.create_publisher(String, '/vla/reasoning', 10)
        self.debug_pub = self.create_publisher(String, '/vla/reasoning_debug', 10)

        # Initialize TF buffer
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Store recent data
        self.recent_detections = []
        self.recent_image = None
        self.command_history = []

        # Reasoning parameters
        self.max_detection_history = 10
        self.max_command_history = 20

        # Knowledge base
        self.knowledge_base = {
            'objects': {
                'cup': {'size': 'small', 'graspable': True, 'movable': True},
                'book': {'size': 'medium', 'graspable': True, 'movable': True},
                'table': {'size': 'large', 'graspable': False, 'movable': False},
                'chair': {'size': 'large', 'graspable': False, 'movable': False},
                'person': {'size': 'large', 'graspable': False, 'movable': True}
            },
            'relationships': {
                'on': 'support',
                'in': 'containment',
                'near': 'proximity',
                'next_to': 'adjacency'
            },
            'actions': {
                'pick_up': {'preconditions': ['object_exists', 'reachable', 'graspable']},
                'place': {'preconditions': ['holding_object', 'valid_place_location']},
                'navigate_to': {'preconditions': ['valid_location', 'reachable']},
                'find': {'preconditions': ['object_type_known', 'searchable_area']}
            }
        }

        # Task planning state
        self.current_task = None
        self.task_plan = []
        self.plan_step = 0

        self.get_logger().info('VLA reasoning node started')

    def command_callback(self, msg):
        """Process natural language command and reason about it"""
        try:
            command_text = msg.data

            # Add to command history
            self.command_history.append({
                'command': command_text,
                'timestamp': self.get_clock().now().nanoseconds * 1e-9
            })

            if len(self.command_history) > self.max_command_history:
                self.command_history = self.command_history[-self.max_command_history:]

            # Parse command and extract intent
            parsed_command = self.parse_command(command_text)

            # Reason about command
            reasoning_result = self.reason_command(parsed_command)

            # Generate action plan
            action_plan = self.generate_action_plan(reasoning_result)

            # Publish action plan
            plan_msg = String()
            plan_msg.data = json.dumps(action_plan)
            self.action_plan_pub.publish(plan_msg)

            # Publish reasoning result
            reasoning_msg = String()
            reasoning_msg.data = json.dumps(reasoning_result)
            self.reasoning_pub.publish(reasoning_msg)

            self.get_logger().info(f'VLA reasoning: {reasoning_result.get("intent", "unknown")} -> {len(action_plan.get("steps", []))} steps')

        except Exception as e:
            self.get_logger().error(f'Error in command reasoning: {e}')

    def detection_callback(self, msg):
        """Process detection results for reasoning"""
        try:
            detections = []

            for detection in msg.detections:
                if detection.results and len(detection.results) > 0:
                    best_result = detection.results[0]
                    detections.append({
                        'class_name': best_result.hypothesis.class_id,
                        'confidence': best_result.hypothesis.score,
                        'bbox': {
                            'center_x': detection.bbox.center.x,
                            'center_y': detection.bbox.center.y,
                            'size_x': detection.bbox.size_x,
                            'size_y': detection.bbox.size_y
                        },
                        'timestamp': msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                    })

            # Store in recent detections
            self.recent_detections.extend(detections)

            # Keep only recent detections
            current_time = self.get_clock().now().nanoseconds * 1e-9
            self.recent_detections = [
                det for det in self.recent_detections
                if (current_time - det['timestamp']) < 5.0  # Keep last 5 seconds
            ]

            if len(self.recent_detections) > self.max_detection_history:
                self.recent_detections = self.recent_detections[-self.max_detection_history:]

        except Exception as e:
            self.get_logger().error(f'Error processing detections: {e}')

    def parse_command(self, command_text: str) -> Dict[str, Any]:
        """Parse natural language command into structured representation"""
        command_lower = command_text.lower()

        # Extract action
        action = self.extract_action(command_lower)

        # Extract object
        object_name = self.extract_object(command_lower)

        # Extract location
        location = self.extract_location(command_lower)

        # Extract other parameters
        parameters = {
            'action': action,
            'object': object_name,
            'location': location,
            'command_text': command_text,
            'timestamp': self.get_clock().now().nanoseconds * 1e-9
        }

        return parameters

    def extract_action(self, command: str) -> str:
        """Extract action from command"""
        action_keywords = {
            'pick_up': ['pick up', 'grasp', 'take', 'get', 'grab'],
            'place': ['place', 'put', 'set', 'drop'],
            'navigate_to': ['go to', 'move to', 'navigate to', 'drive to', 'walk to'],
            'find': ['find', 'locate', 'search for', 'look for', 'detect'],
            'follow': ['follow', 'chase', 'track'],
            'avoid': ['avoid', 'steer clear of', 'stay away from'],
            'greet': ['greet', 'say hello to', 'wave to'],
            'wait': ['wait', 'stop', 'pause', 'hold position']
        }

        for action, keywords in action_keywords.items():
            if any(keyword in command for keyword in keywords):
                return action

        return 'unknown'

    def extract_object(self, command: str) -> str:
        """Extract object from command"""
        # Look for known objects
        known_objects = self.knowledge_base['objects'].keys()
        for obj in known_objects:
            if obj in command:
                return obj

        # If no known object found, return the first noun-like word
        # In practice, use NLP parsing
        words = command.split()
        for word in words:
            if word in known_objects:
                return word

        return 'unknown_object'

    def extract_location(self, command: str) -> str:
        """Extract location from command"""
        # Look for known locations
        known_locations = ['kitchen', 'living room', 'bedroom', 'office', 'dining room', 'bathroom', 'garden', 'garage', 'home base', 'charger']

        for location in known_locations:
            if location in command:
                return location

        return 'unknown_location'

    def reason_command(self, parsed_command: Dict[str, Any]) -> Dict[str, Any]:
        """Perform reasoning about the parsed command"""
        action = parsed_command['action']
        obj_name = parsed_command['object']
        location = parsed_command['location']

        # Check if object exists in environment
        object_exists = self.check_object_existence(obj_name)

        # Check if location is reachable
        location_reachable = self.check_location_reachability(location)

        # Check if action is feasible
        action_feasible = self.check_action_feasibility(action, obj_name, location)

        # Generate reasoning result
        reasoning_result = {
            'intent': f"{action}_{obj_name if obj_name != 'unknown_object' else ''}_{location if location != 'unknown_location' else ''}".strip('_'),
            'action': action,
            'object': obj_name,
            'location': location,
            'feasibility': {
                'object_exists': object_exists,
                'location_reachable': location_reachable,
                'action_feasible': action_feasible,
                'overall_feasible': object_exists and location_reachable and action_feasible
            },
            'context': {
                'current_objects': [det['class_name'] for det in self.recent_detections],
                'command_history': self.command_history[-3:],  # Last 3 commands
                'environment_state': 'active'
            },
            'confidence': self.estimate_confidence(parsed_command),
            'reasoning_trace': self.generate_reasoning_trace(parsed_command)
        }

        return reasoning_result

    def check_object_existence(self, object_name: str) -> bool:
        """Check if object exists in recent detections"""
        if object_name == 'unknown_object':
            return True  # Can't verify unknown objects

        for detection in self.recent_detections:
            if (detection['class_name'] == object_name and
                detection['confidence'] > 0.5):  # Confidence threshold
                return True

        return False

    def check_location_reachability(self, location: str) -> bool:
        """Check if location is reachable"""
        if location == 'unknown_location':
            return False

        # In practice, check against navigation map
        # For this example, assume known locations are reachable
        known_reachable = ['kitchen', 'living room', 'bedroom', 'office', 'home base']
        return location in known_reachable

    def check_action_feasibility(self, action: str, object_name: str, location: str) -> bool:
        """Check if action is feasible given object and location"""
        if action not in self.knowledge_base['actions']:
            return False

        preconditions = self.knowledge_base['actions'][action]['preconditions']

        for precondition in preconditions:
            if precondition == 'object_exists' and object_name != 'unknown_object':
                if not self.check_object_existence(object_name):
                    return False
            elif precondition == 'graspable' and object_name != 'unknown_object':
                obj_info = self.knowledge_base['objects'].get(object_name, {})
                if not obj_info.get('graspable', False):
                    return False
            elif precondition == 'reachable' and location != 'unknown_location':
                if not self.check_location_reachability(location):
                    return False
            elif precondition == 'valid_location' and location == 'unknown_location':
                return False

        return True

    def estimate_confidence(self, parsed_command: Dict[str, Any]) -> float:
        """Estimate confidence in command understanding"""
        confidence = 0.5  # Base confidence

        # Increase confidence based on various factors
        if parsed_command['action'] != 'unknown':
            confidence += 0.2
        if parsed_command['object'] != 'unknown_object':
            confidence += 0.15
        if parsed_command['location'] != 'unknown_location':
            confidence += 0.15

        # Check against recent detections
        if parsed_command['object'] != 'unknown_object':
            if any(det['class_name'] == parsed_command['object'] for det in self.recent_detections):
                confidence += 0.1

        return min(1.0, confidence)

    def generate_reasoning_trace(self, parsed_command: Dict[str, Any]) -> str:
        """Generate trace of reasoning process"""
        action = parsed_command['action']
        obj_name = parsed_command['object']
        location = parsed_command['location']

        trace_parts = []

        if action != 'unknown':
            trace_parts.append(f"Recognized action: {action}")
        else:
            trace_parts.append("Could not identify action, using default")

        if obj_name != 'unknown_object':
            if self.check_object_existence(obj_name):
                trace_parts.append(f"Confirmed existence of {obj_name}")
            else:
                trace_parts.append(f"{obj_name} not detected in environment")
        else:
            trace_parts.append("Could not identify object, will search for targets")

        if location != 'unknown_location':
            if self.check_location_reachability(location):
                trace_parts.append(f"Confirmed reachability of {location}")
            else:
                trace_parts.append(f"{location} not reachable")
        else:
            trace_parts.append("Could not identify location, will use current position")

        return "; ".join(trace_parts)

    def generate_action_plan(self, reasoning_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate action plan based on reasoning result"""
        action = reasoning_result['action']
        obj_name = reasoning_result['object']
        location = reasoning_result['location']
        feasible = reasoning_result['feasibility']['overall_feasible']

        plan = {
            'command': reasoning_result['intent'],
            'feasible': feasible,
            'confidence': reasoning_result['confidence'],
            'steps': []
        }

        if not feasible:
            plan['steps'].append({
                'action': 'report_unfeasible',
                'parameters': {
                    'reason': 'Command is not feasible based on current environment state',
                    'suggestions': self.generate_suggestions(reasoning_result)
                }
            })
            return plan

        # Generate plan based on action type
        if action == 'navigate_to':
            plan['steps'].extend(self.generate_navigation_plan(location))
        elif action == 'pick_up':
            plan['steps'].extend(self.generate_manipulation_plan(obj_name, 'pick_up'))
        elif action == 'place':
            plan['steps'].extend(self.generate_manipulation_plan(obj_name, 'place', location))
        elif action == 'find':
            plan['steps'].extend(self.generate_search_plan(obj_name))
        else:
            # Default action plan
            plan['steps'].append({
                'action': 'default_action',
                'parameters': {
                    'command': reasoning_result['intent']
                }
            })

        return plan

    def generate_navigation_plan(self, location: str) -> List[Dict[str, Any]]:
        """Generate navigation plan to location"""
        return [
            {
                'step': 1,
                'action': 'navigate_to_location',
                'parameters': {
                    'target_location': location
                },
                'description': f'Navigate to {location}'
            },
            {
                'step': 2,
                'action': 'arrive_at_location',
                'parameters': {
                    'location': location
                },
                'description': f'Arrive at {location}'
            }
        ]

    def generate_manipulation_plan(self, object_name: str, action_type: str, location: str = None) -> List[Dict[str, Any]]:
        """Generate manipulation plan for object"""
        plan = []

        # If location is specified, navigate there first
        if location and location != 'unknown_location':
            plan.extend(self.generate_navigation_plan(location))

        # Find the object
        plan.append({
            'step': len(plan) + 1,
            'action': 'locate_object',
            'parameters': {
                'object_type': object_name
            },
            'description': f'Locate {object_name}'
        })

        # Approach the object
        plan.append({
            'step': len(plan) + 1,
            'action': 'approach_object',
            'parameters': {
                'object_type': object_name
            },
            'description': f'Approach {object_name}'
        })

        # Perform the action
        if action_type == 'pick_up':
            plan.append({
                'step': len(plan) + 1,
                'action': 'grasp_object',
                'parameters': {
                    'object_type': object_name
                },
                'description': f'Grasp {object_name}'
            })
        elif action_type == 'place':
            plan.append({
                'step': len(plan) + 1,
                'action': 'release_object',
                'parameters': {
                    'object_type': object_name
                },
                'description': f'Release {object_name}'
            })

        return plan

    def generate_search_plan(self, object_name: str) -> List[Dict[str, Any]]:
        """Generate search plan for object"""
        return [
            {
                'step': 1,
                'action': 'search_for_object',
                'parameters': {
                    'object_type': object_name,
                    'search_pattern': 'spiral'
                },
                'description': f'Search for {object_name} using spiral pattern'
            },
            {
                'step': 2,
                'action': 'confirm_object_detection',
                'parameters': {
                    'object_type': object_name
                },
                'description': f'Confirm detection of {object_name}'
            }
        ]

    def generate_suggestions(self, reasoning_result: Dict[str, Any]) -> List[str]:
        """Generate suggestions when command is not feasible"""
        suggestions = []

        if not reasoning_result['feasibility']['object_exists']:
            suggestions.append(f"Could not find {reasoning_result['object']}. Try specifying a different object or location.")

        if not reasoning_result['feasibility']['location_reachable']:
            suggestions.append(f"Cannot reach {reasoning_result['location']}. Try a known location like kitchen or living room.")

        if not reasoning_result['feasibility']['action_feasible']:
            suggestions.append("The requested action may not be feasible. Try a simpler command.")

        return suggestions if suggestions else ["Please rephrase your command or provide more details."]

def main(args=None):
    rclpy.init(args=args)
    node = VLAReasoningNode()

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

## Step 5: Create the Complete System Launch File

Create `vla_integration_system/launch/vla_system.launch.py`:

```python
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # Declare launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')
    camera_topic = LaunchConfiguration('camera_topic', default='/camera/image_raw')
    command_topic = LaunchConfiguration('command_topic', default='/natural_language_command')

    return LaunchDescription([
        # Set environment variables
        SetEnvironmentVariable(name='PYTHONUNBUFFERED', value='1'),

        # Declare launch arguments
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation time if true'
        ),
        DeclareLaunchArgument(
            'camera_topic',
            default_value='/camera/image_raw',
            description='Camera image topic name'
        ),
        DeclareLaunchArgument(
            'command_topic',
            default_value='/natural_language_command',
            description='Natural language command topic'
        ),

        # VLA controller node
        Node(
            package='vla_integration_system',
            executable='vla_controller',
            name='vla_controller',
            parameters=[
                {'use_sim_time': use_sim_time}
            ],
            remappings=[
                ('/camera/image_raw', camera_topic),
                ('/natural_language_command', command_topic),
                ('/cmd_vel', '/robot/cmd_vel'),
                ('/vla/status', '/vla_integration/status'),
                ('/vla/confidence', '/vla_integration/confidence')
            ],
            output='screen'
        ),

        # VLA perception node
        Node(
            package='vla_integration_system',
            executable='vla_perception',
            name='vla_perception',
            parameters=[
                {'use_sim_time': use_sim_time}
            ],
            remappings=[
                ('/camera/image_raw', camera_topic),
                ('/vla/detections', '/vla_integration/detections'),
                ('/vla/perception', '/vla_integration/perception_results'),
                ('/vla/visualization', '/vla_integration/visualization_markers')
            ],
            output='screen'
        ),

        # VLA reasoning node
        Node(
            package='vla_integration_system',
            executable='vla_reasoning',
            name='vla_reasoning',
            parameters=[
                {'use_sim_time': use_sim_time}
            ],
            remappings=[
                ('/natural_language_command', command_topic),
                ('/vla/detections', '/vla_integration/detections'),
                ('/vla/action_plan', '/vla_integration/action_plan'),
                ('/vla/reasoning', '/vla_integration/reasoning_results')
            ],
            output='screen'
        ),

        # Joint state publisher (for robot state)
        Node(
            package='joint_state_publisher',
            executable='joint_state_publisher',
            name='joint_state_publisher',
            parameters=[
                {'use_sim_time': use_sim_time}
            ]
        ),

        # Robot state publisher (for TF transforms)
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            parameters=[
                {'use_sim_time': use_sim_time}
            ]
        )
    ])
```

## Step 6: Create the Package Configuration

Update `vla_integration_system/setup.py`:

```python
from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'vla_integration_system'

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
    description='Complete VLA system integration for robotics',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'vla_controller = vla_integration_system.vla_controller:main',
            'vla_perception = vla_integration_system.vla_perception:main',
            'vla_reasoning = vla_integration_system.vla_reasoning:main',
        ],
    },
)
```

## Step 7: Build and Test the System

Build the package:

```bash
cd ~/ros2_ws
colcon build --packages-select vla_integration_system
source install/setup.bash
```

### Test the Complete VLA System

1. **Launch the complete system**:
```bash
# Launch the VLA system
ros2 launch vla_integration_system vla_system.launch.py
```

2. **Test with commands** (in another terminal):
```bash
# Send navigation command
ros2 topic pub /natural_language_command std_msgs/String "data: 'Go to the kitchen'"

# Send manipulation command
ros2 topic pub /natural_language_command std_msgs/String "data: 'Pick up the red cup'"

# Send search command
ros2 topic pub /natural_language_command std_msgs/String "data: 'Find the person in the room'"
```

3. **Monitor the system**:
```bash
# Monitor VLA status
ros2 topic echo /vla_integration/status

# Monitor detections
ros2 topic echo /vla_integration/detections

# Monitor action plans
ros2 topic echo /vla_integration/action_plan

# Monitor reasoning results
ros2 topic echo /vla_integration/reasoning_results

# Visualize in RViz
ros2 run rviz2 rviz2
```

4. **Check system performance**:
```bash
# Monitor processing rates
ros2 topic hz /vla_integration/detections

# Check node status
ros2 node list | grep vla

# Monitor system resources
htop
```

## Understanding the Complete System

This complete VLA integration demonstrates:

1. **Multimodal Perception**: Processing camera and sensor data together
2. **Natural Language Understanding**: Interpreting human commands
3. **Embodied Reasoning**: Making decisions based on environment context
4. **Action Generation**: Converting high-level goals to low-level actions
5. **Real-time Processing**: Maintaining performance for interactive systems
6. **Safety Integration**: Validation and safety checks throughout

## Advanced Features

### 1. Context-Aware Reasoning

```python
#!/usr/bin/env python3
"""
Context-Aware VLA Reasoning
"""
import json
from typing import Dict, Any, List

class ContextAwareVLA:
    def __init__(self):
        self.context_history = []
        self.max_context_length = 50

    def update_context(self, perception_data, command, action_result):
        """Update system context with new information"""
        context_entry = {
            'timestamp': time.time(),
            'perception': perception_data,
            'command': command,
            'action_result': action_result,
            'environment_state': self.encode_environment_state(perception_data)
        }

        self.context_history.append(context_entry)

        # Keep context manageable
        if len(self.context_history) > self.max_context_length:
            self.context_history = self.context_history[-self.max_context_length:]

    def encode_environment_state(self, perception_data) -> Dict[str, Any]:
        """Encode environment state for context"""
        return {
            'object_count': len(perception_data.get('detections', [])),
            'object_types': list(set(det.get('class_name', '') for det in perception_data.get('detections', []))),
            'room_type': self.estimate_room_type(perception_data),
            'time_of_day': self.estimate_time_of_day(),
            'previous_actions': [entry['action_result'] for entry in self.context_history[-5:]]
        }

    def estimate_room_type(self, perception_data) -> str:
        """Estimate current room type from perception data"""
        # In practice, this would use scene classification
        # For this example, return a placeholder
        return 'unknown_room'

    def estimate_time_of_day(self) -> str:
        """Estimate time of day for context"""
        import datetime
        current_hour = datetime.datetime.now().hour
        if 6 <= current_hour < 12:
            return 'morning'
        elif 12 <= current_hour < 18:
            return 'afternoon'
        elif 18 <= current_hour < 22:
            return 'evening'
        else:
            return 'night'
```

### 2. Multi-Step Task Planning

```python
#!/usr/bin/env python3
"""
Multi-Step Task Planner for VLA
"""
from typing import Dict, Any, List

class MultiStepTaskPlanner:
    def __init__(self):
        self.task_library = {
            'serve_drink': ['navigate_to_kitchen', 'find_cup', 'pick_up_cup', 'navigate_to_person', 'deliver_cup'],
            'clean_table': ['navigate_to_table', 'find_objects', 'pick_up_object', 'navigate_to_bin', 'place_object'],
            'escort_person': ['find_person', 'navigate_to_person', 'follow_person', 'navigate_to_destination']
        }

    def decompose_task(self, task_name: str) -> List[Dict[str, Any]]:
        """Decompose high-level task into subtasks"""
        if task_name in self.task_library:
            subtasks = []
            for i, action_name in enumerate(self.task_library[task_name]):
                subtask = {
                    'id': i,
                    'action': action_name,
                    'dependencies': [i-1] if i > 0 else [],  # Previous step dependency
                    'preconditions': self.get_preconditions(action_name),
                    'postconditions': self.get_postconditions(action_name),
                    'estimated_duration': self.estimate_duration(action_name)
                }
                subtasks.append(subtask)
            return subtasks
        else:
            return [{'id': 0, 'action': 'unknown_task', 'dependencies': [], 'estimated_duration': 1.0}]

    def get_preconditions(self, action_name: str) -> List[str]:
        """Get preconditions for action"""
        preconditions_map = {
            'navigate_to_kitchen': ['robot_at_home', 'navigation_system_active'],
            'find_cup': ['robot_at_kitchen', 'perception_system_active'],
            'pick_up_cup': ['cup_detected', 'gripper_available'],
            'navigate_to_person': ['person_location_known', 'navigation_system_active'],
            'deliver_cup': ['cup_grasped', 'person_reachable']
        }
        return preconditions_map.get(action_name, [])

    def get_postconditions(self, action_name: str) -> List[str]:
        """Get postconditions for action"""
        postconditions_map = {
            'navigate_to_kitchen': ['robot_at_kitchen'],
            'find_cup': ['cup_location_known'],
            'pick_up_cup': ['cup_grasped'],
            'navigate_to_person': ['robot_at_person'],
            'deliver_cup': ['cup_delivered']
        }
        return postconditions_map.get(action_name, [])

    def estimate_duration(self, action_name: str) -> float:
        """Estimate action duration"""
        duration_map = {
            'navigate_to_kitchen': 30.0,  # seconds
            'find_cup': 10.0,
            'pick_up_cup': 5.0,
            'navigate_to_person': 20.0,
            'deliver_cup': 3.0
        }
        return duration_map.get(action_name, 5.0)
```

## Quality Assurance and Validation

### VLA System Validation

```python
#!/usr/bin/env python3
"""
VLA System Validation
"""
import numpy as np
from typing import Dict, Any

class VLAValidator:
    def __init__(self):
        self.metrics = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1_score': [],
            'response_time': [],
            'safety_compliance': [],
            'user_satisfaction': []
        }

    def validate_perception_accuracy(self, detections, ground_truth):
        """Validate perception accuracy against ground truth"""
        if not ground_truth:
            return 0.0  # Can't validate without ground truth

        # Calculate IoU for each detection
        ious = []
        for det in detections:
            for gt in ground_truth:
                if det['class_name'] == gt['class_name']:
                    iou = self.calculate_iou(det['bbox'], gt['bbox'])
                    ious.append(iou)

        avg_iou = np.mean(ious) if ious else 0.0
        return avg_iou

    def validate_action_safety(self, action, environment_state):
        """Validate action safety in current environment"""
        # Check for collision risk
        collision_risk = self.assess_collision_risk(action, environment_state)

        # Check for joint limits
        joint_limit_violation = self.check_joint_limits(action)

        # Check for force limits
        force_limit_violation = self.check_force_limits(action)

        # Overall safety score
        safety_score = 1.0
        if collision_risk > 0.8:
            safety_score *= 0.1
        elif collision_risk > 0.5:
            safety_score *= 0.5

        if joint_limit_violation:
            safety_score *= 0.3

        if force_limit_violation:
            safety_score *= 0.2

        return safety_score

    def calculate_iou(self, bbox1, bbox2):
        """Calculate Intersection over Union"""
        # Bounding box format: [x1, y1, x2, y2]
        x1_inter = max(bbox1[0], bbox2[0])
        y1_inter = max(bbox1[1], bbox2[1])
        x2_inter = min(bbox1[2], bbox2[2])
        y2_inter = min(bbox1[3], bbox2[3])

        if x2_inter <= x1_inter or y2_inter <= y1_inter:
            return 0.0

        area_inter = (x2_inter - x1_inter) * (y2_inter - y1_inter)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        area_union = area1 + area2 - area_inter

        return area_inter / area_union if area_union > 0 else 0.0

    def assess_collision_risk(self, action, env_state):
        """Assess collision risk for action"""
        # In practice, this would use motion planning and collision checking
        # For this example, return a simplified risk assessment
        return 0.1  # Low risk

    def check_joint_limits(self, action):
        """Check if action violates joint limits"""
        # In practice, check against robot's joint limits
        # For this example, return False (no violation)
        return False

    def check_force_limits(self, action):
        """Check if action violates force limits"""
        # In practice, check against robot's force/torque limits
        # For this example, return False (no violation)
        return False

    def validate_command_understanding(self, command, intent, confidence):
        """Validate command understanding quality"""
        # Check if intent makes sense given command
        command_words = command.lower().split()
        intent_words = intent.lower().split('_')

        # Calculate overlap between command and intent
        overlap = len(set(command_words) & set(intent_words))
        command_complexity = len(command_words)

        # Understanding score based on intent-relevance
        understanding_score = overlap / command_complexity if command_complexity > 0 else 0.0

        # Adjust by confidence
        final_score = understanding_score * confidence

        return min(1.0, final_score)
```

## Best Practices

### 1. VLA System Best Practices

```python
# Good: Comprehensive VLA system design
def good_vla_system():
    """Best practices for VLA system design"""
    # Modular architecture
    # Proper error handling
    # Performance monitoring
    # Safety validation
    # Context awareness
    # Multi-modal fusion
    # Real-time constraints
    pass

# Bad: Poor VLA system design
def bad_vla_system():
    """Poor practices in VLA system design"""
    # Monolithic architecture
    # No error handling
    # No performance monitoring
    # No safety checks
    # No context management
    # Poor fusion strategies
    # No real-time considerations
    pass
```

### 2. Performance Optimization

```python
# Good: Performance-optimized VLA
def optimized_vla_processing():
    """Optimize VLA processing for performance"""
    # Use efficient data structures
    # Implement caching
    # Optimize neural networks
    # Use appropriate precision
    # Implement proper threading
    # Monitor and profile performance
    pass

# Bad: Performance-inefficient VLA
def inefficient_vla_processing():
    """Inefficient VLA processing"""
    # No optimization
    # Inefficient algorithms
    # No caching
    # Blocking operations
    # No performance monitoring
    # Poor resource management
    pass
```

### 3. Safety Best Practices

```python
# Good: Safety-first VLA
def safe_vla_system():
    """Safety considerations in VLA systems"""
    # Safety validation at each step
    # Collision checking
    # Force limiting
    # Emergency stops
    # Fallback behaviors
    # User confirmation for critical actions
    pass

# Bad: Unsafe VLA system
def unsafe_vla_system():
    """Unsafe practices in VLA systems"""
    # No safety validation
    # No collision checking
    # No force limits
    # No emergency stops
    # No fallback behaviors
    # No user confirmation
    pass
```

## Common Issues and Troubleshooting

### 1. Multimodal Alignment Issues

```python
# Diagnose alignment issues
def diagnose_alignment_issues():
    """Diagnose multimodal alignment problems"""
    # Check temporal synchronization
    # Verify spatial calibration
    # Monitor cross-modal consistency
    # Validate sensor fusion
    pass
```

### 2. Performance Issues

```bash
# Monitor VLA performance
ros2 topic hz /vla_integration/detections
ros2 topic hz /vla_integration/action_plan
ros2 run tf2_tools view_frames
```

### 3. Quality Issues

```python
# Monitor perception quality
def monitor_perception_quality():
    """Monitor VLA perception quality"""
    # Track detection accuracy
    # Monitor confidence scores
    # Validate action feasibility
    # Assess user satisfaction
    pass
```

## Next Steps

Now that you have built a complete VLA integration system, continue to [Week 13: Capstone Project](../../module-4-vision-language-action/week-13/introduction) to apply all your knowledge in a comprehensive capstone project.

## Exercises

1. Enhance the VLA system with semantic segmentation capabilities
2. Implement a learning mechanism that improves with user feedback
3. Add gesture recognition to complement voice commands
4. Create a validation system for measuring VLA performance in real-world scenarios