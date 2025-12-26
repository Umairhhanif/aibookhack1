---
sidebar_position: 5
---

# Exercise: Complete Perception System

In this comprehensive exercise, you'll create a complete perception system that integrates multiple sensors, deep learning models, and traditional computer vision techniques into a cohesive pipeline.

## Objective

Build a complete perception system that:
1. **Processes multiple sensor modalities** (camera, LiDAR, IMU)
2. **Integrates deep learning models** for object detection and classification
3. **Applies traditional computer vision** for feature extraction and tracking
4. **Fuses sensor fusion** to combine information from different sensors
5. **Implements quality validation** to ensure reliable perception
6. **Provides real-time performance** with optimized processing

## Prerequisites

- Complete Week 1-9 lessons
- ROS 2 workspace set up (`~/ros2_ws`)
- Understanding of computer vision and deep learning
- Basic Python and C++ programming skills
- OpenCV and PyTorch/TensorFlow installed

## Step 1: Create the Perception Package

```bash
cd ~/ros2_ws/src

ros2 pkg create --build-type ament_python complete_perception_system \
    --dependencies rclpy std_msgs sensor_msgs geometry_msgs vision_msgs cv_bridge tf2_ros tf2_geometry_msgs message_filters
```

## Step 2: Create the Perception Manager Node

Create `complete_perception_system/complete_perception_system/perception_manager.py`:

```python
#!/usr/bin/env python3
"""
Complete Perception System Manager
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, Imu, LaserScan
from geometry_msgs.msg import PointStamped, PoseStamped
from vision_msgs.msg import Detection2DArray
from std_msgs.msg import String, Float32
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
from threading import Lock
import numpy as np
import cv2
import time
from typing import Dict, Any, List

class PerceptionManagerNode(Node):
    def __init__(self):
        super().__init__('perception_manager_node')

        # Create subscribers for multiple sensors
        self.camera_sub = Subscriber(self, Image, '/camera/image_raw')
        self.lidar_sub = Subscriber(self, PointCloud2, '/lidar/points')
        self.imu_sub = Subscriber(self, Imu, '/imu/data')
        self.scan_sub = Subscriber(self, LaserScan, '/scan')

        # Create synchronizer for multi-sensor fusion
        self.sync = ApproximateTimeSynchronizer(
            [self.camera_sub, self.lidar_sub],
            queue_size=10,
            slop=0.2  # Allow 200ms time difference
        )
        self.sync.registerCallback(self.multi_sensor_callback)

        # Create publishers for perception outputs
        self.detections_pub = self.create_publisher(Detection2DArray, '/perception/detections', 10)
        self.fused_data_pub = self.create_publisher(String, '/perception/fused_data', 10)
        self.quality_score_pub = self.create_publisher(Float32, '/perception/quality_score', 10)
        self.status_pub = self.create_publisher(String, '/perception/status', 10)

        # Initialize CV bridge
        self.bridge = CvBridge()

        # Initialize perception components
        self.feature_extractor = FeatureExtractor(self)
        self.object_detector = ObjectDetector(self)
        self.tracker = ObjectTracker(self)
        self.fusion_module = SensorFusion(self)

        # System parameters
        self.perception_rate = 10.0  # Hz
        self.confidence_threshold = 0.5
        self.max_objects = 20

        # Processing statistics
        self.stats = {
            'processed_frames': 0,
            'detection_rate': 0.0,
            'processing_time': 0.0,
            'quality_score': 1.0
        }

        # Threading and synchronization
        self.processing_lock = Lock()
        self.last_process_time = time.time()

        # Start processing timer
        self.process_timer = self.create_timer(1.0/self.perception_rate, self.process_timer_callback)

        self.get_logger().info('Complete perception system manager started')

    def multi_sensor_callback(self, image_msg, lidar_msg):
        """Handle synchronized multi-sensor data"""
        try:
            start_time = time.time()

            # Process camera data
            cv_image = self.bridge.imgmsg_to_cv2(image_msg, 'bgr8')
            camera_features = self.feature_extractor.extract_features(cv_image)
            detections = self.object_detector.detect_objects(cv_image)

            # Process LiDAR data
            lidar_points = self.parse_pointcloud(lidar_msg)

            # Update tracker
            tracked_objects = self.tracker.update(detections, image_msg.header)

            # Fuse sensor data
            fused_result = self.fusion_module.fuse_data(
                camera_features, lidar_points, tracked_objects, image_msg.header
            )

            # Publish results
            self.publish_detections(tracked_objects, image_msg.header)
            self.publish_fused_data(fused_result, image_msg.header)

            # Update statistics
            self.stats['processing_time'] = time.time() - start_time
            self.stats['processed_frames'] += 1

            # Calculate quality score
            quality_score = self.calculate_quality_score(
                detections, camera_features, self.stats['processing_time']
            )
            self.stats['quality_score'] = quality_score

            # Publish quality metrics
            self.publish_quality_metrics(quality_score)

        except Exception as e:
            self.get_logger().error(f'Error in multi-sensor processing: {e}')

    def process_timer_callback(self):
        """Process timer callback for periodic tasks"""
        try:
            # Calculate and publish statistics
            current_time = time.time()
            time_diff = current_time - self.last_process_time
            self.last_process_time = current_time

            if time_diff > 0:
                self.stats['detection_rate'] = self.stats['processed_frames'] / time_diff

            # Publish system status
            status_msg = String()
            status_msg.data = f"Running - Detections: {self.stats['detection_rate']:.2f} Hz, Quality: {self.stats['quality_score']:.3f}"
            self.status_pub.publish(status_msg)

            self.get_logger().info(
                f"Perception Stats - Rate: {self.stats['detection_rate']:.2f} Hz, "
                f"Quality: {self.stats['quality_score']:.3f}, "
                f"Processing: {self.stats['processing_time']:.3f}s"
            )

        except Exception as e:
            self.get_logger().error(f'Error in process timer: {e}')

    def calculate_quality_score(self, detections, features, processing_time):
        """Calculate overall perception quality score"""
        # Factors affecting quality:
        # 1. Number of detections (too few = low quality, too many = possibly false positives)
        # 2. Feature density (more features = better for tracking)
        # 3. Processing time (faster = better for real-time)
        # 4. Detection confidence

        score = 1.0

        # Adjust for number of detections
        if len(detections) == 0:
            score *= 0.5  # No detections = lower quality
        elif len(detections) > self.max_objects:
            score *= 0.7  # Too many detections = potential false positives

        # Adjust for processing time
        if processing_time > 0.1:  # More than 100ms = too slow
            score *= 0.8
        elif processing_time > 0.05:  # More than 50ms = somewhat slow
            score *= 0.9

        # Adjust for feature quality (simplified)
        if hasattr(features, 'count') and features.count < 50:
            score *= 0.8  # Too few features

        return max(0.0, min(1.0, score))

    def publish_detections(self, detections, header):
        """Publish object detections"""
        if not detections:
            return

        detection_array = Detection2DArray()
        detection_array.header = header

        for det in detections:
            detection = Detection2D()
            detection.header = header

            # Set bounding box
            bbox = detection.bbox
            bbox.center.x = det['bbox_center_x']
            bbox.center.y = det['bbox_center_y']
            bbox.size_x = det['bbox_width']
            bbox.size_y = det['bbox_height']

            # Add detection to array
            detection_array.detections.append(detection)

        self.detections_pub.publish(detection_array)

    def publish_fused_data(self, fused_result, header):
        """Publish fused perception data"""
        result_msg = String()
        result_msg.data = str(fused_result)
        self.fused_data_pub.publish(result_msg)

    def publish_quality_metrics(self, quality_score):
        """Publish quality metrics"""
        score_msg = Float32()
        score_msg.data = quality_score
        self.quality_score_pub.publish(score_msg)

    def parse_pointcloud(self, cloud_msg):
        """Parse PointCloud2 message (simplified)"""
        # This is a simplified parser - in practice, use sensor_msgs_py
        points = []
        # Actual parsing would go here
        return points

def main(args=None):
    rclpy.init(args=args)
    node = PerceptionManagerNode()

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

## Step 3: Create Feature Extractor Component

Create `complete_perception_system/complete_perception_system/feature_extractor.py`:

```python
#!/usr/bin/env python3
"""
Feature Extractor Component
"""
import cv2
import numpy as np
from typing import Dict, Any

class FeatureExtractor:
    def __init__(self, node):
        self.node = node
        self.orb = cv2.ORB_create(nfeatures=500)
        self.sift = cv2.SIFT_create()
        self.feature_params = {
            'max_features': 500,
            'quality_level': 0.01,
            'min_distance': 10,
            'block_size': 7
        }

    def extract_features(self, image):
        """Extract features from image"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Extract ORB features
            orb_keypoints, orb_descriptors = self.orb.detectAndCompute(gray, None)

            # Extract SIFT features (if needed)
            # sift_keypoints, sift_descriptors = self.sift.detectAndCompute(gray, None)

            # Calculate feature density
            feature_density = len(orb_keypoints) / (gray.shape[0] * gray.shape[1]) if orb_keypoints else 0

            # Create feature representation
            features = {
                'keypoints': orb_keypoints,
                'descriptors': orb_descriptors,
                'count': len(orb_keypoints) if orb_keypoints else 0,
                'density': feature_density,
                'good_features': self.filter_good_features(orb_keypoints) if orb_keypoints else []
            }

            self.node.get_logger().debug(f'Extracted {features["count"]} features')

            return features

        except Exception as e:
            self.node.get_logger().error(f'Error extracting features: {e}')
            return {
                'keypoints': [],
                'descriptors': None,
                'count': 0,
                'density': 0,
                'good_features': []
            }

    def filter_good_features(self, keypoints):
        """Filter out low-quality features"""
        if not keypoints:
            return []

        # In practice, you might filter based on response strength, position, etc.
        # For this example, we'll just return the keypoints
        return keypoints

    def track_features(self, prev_features, current_image):
        """Track features between frames"""
        try:
            gray = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)

            if prev_features['keypoints'] and len(prev_features['keypoints']) > 0:
                # Convert keypoints to numpy array for tracking
                prev_points = np.float32([kp.pt for kp in prev_features['keypoints']]).reshape(-1, 1, 2)

                # Track features using optical flow
                next_points, status, error = cv2.calcOpticalFlowPyrLK(
                    prev_features['prev_gray'], gray, prev_points, None
                )

                # Filter tracked points
                good_new = next_points[status == 1]
                good_prev = prev_points[status == 1]

                # Create tracked features
                tracked_features = {
                    'tracked_points': good_new,
                    'prev_points': good_prev,
                    'count': len(good_new),
                    'tracking_success_rate': len(good_new) / len(prev_features['keypoints']) if prev_features['keypoints'] else 0
                }

                return tracked_features

            return {
                'tracked_points': [],
                'prev_points': [],
                'count': 0,
                'tracking_success_rate': 0.0
            }

        except Exception as e:
            self.node.get_logger().error(f'Error tracking features: {e}')
            return {
                'tracked_points': [],
                'prev_points': [],
                'count': 0,
                'tracking_success_rate': 0.0
            }
```

## Step 4: Create Object Detector Component

Create `complete_perception_system/complete_perception_system/object_detector.py`:

```python
#!/usr/bin/env python3
"""
Object Detector Component
"""
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from typing import List, Dict, Any

class ObjectDetector:
    def __init__(self, node):
        self.node = node
        self.confidence_threshold = 0.5
        self.nms_threshold = 0.4

        # Initialize deep learning model (conceptual)
        # In practice, load a pre-trained model like YOLO, SSD, etc.
        self.model = self.load_model()

        # COCO class names
        self.coco_classes = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
            'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
            'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
            'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
            'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
            'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
            'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
            'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
            'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
            'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
            'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]

        # Preprocessing transforms
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((416, 416)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

    def load_model(self):
        """Load object detection model (conceptual)"""
        # In practice, load a pre-trained model
        # For this example, we'll return None and simulate detection
        return None

    def detect_objects(self, image):
        """Detect objects in image using deep learning model"""
        try:
            # In a real implementation, this would run the actual model
            # For this exercise, we'll simulate detection

            height, width = image.shape[:2]

            # Simulate detection results
            detections = []

            # Create some dummy detections for demonstration
            for i in range(min(5, int(np.random.uniform(0, 6)))):  # 0-5 objects
                # Random bounding box
                x = int(np.random.uniform(0, width - 100))
                y = int(np.random.uniform(0, height - 100))
                w = int(np.random.uniform(50, 200))
                h = int(np.random.uniform(50, 200))

                # Random class and confidence
                class_id = int(np.random.uniform(0, len(self.coco_classes)))
                confidence = np.random.uniform(0.5, 0.95)

                if confidence > self.confidence_threshold:
                    detection = {
                        'bbox': [x, y, w, h],
                        'bbox_center_x': x + w/2,
                        'bbox_center_y': y + h/2,
                        'bbox_width': w,
                        'bbox_height': h,
                        'class_id': class_id,
                        'class_name': self.coco_classes[class_id],
                        'confidence': confidence,
                        'tracking_id': i  # For tracking association
                    }

                    detections.append(detection)

            self.node.get_logger().debug(f'Detected {len(detections)} objects')

            return detections

        except Exception as e:
            self.node.get_logger().error(f'Error in object detection: {e}')
            return []

    def preprocess_image(self, image):
        """Preprocess image for model input"""
        # Resize image
        resized = cv2.resize(image, (416, 416))

        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # Normalize
        normalized = rgb_image.astype(np.float32) / 255.0

        # Transpose to CHW format
        chw_image = np.transpose(normalized, (2, 0, 1))

        # Add batch dimension
        batch_image = np.expand_dims(chw_image, axis=0)

        return batch_image

    def post_process_detections(self, outputs, original_shape):
        """Post-process model outputs"""
        # In a real implementation, this would:
        # 1. Decode bounding boxes from model outputs
        # 2. Apply confidence thresholding
        # 3. Perform non-maximum suppression
        # 4. Scale bounding boxes back to original image size
        pass
```

## Step 5: Create Object Tracker Component

Create `complete_perception_system/complete_perception_system/object_tracker.py`:

```python
#!/usr/bin/env python3
"""
Object Tracker Component
"""
import numpy as np
import cv2
from typing import List, Dict, Any

class ObjectTracker:
    def __init__(self, node):
        self.node = node
        self.tracks = {}  # Dictionary to store tracked objects
        self.next_id = 0
        self.max_disappeared = 10  # Max frames object can disappear before deletion
        self.max_distance = 100    # Max distance for association

    def update(self, detections, header):
        """Update object tracks with new detections"""
        # Increment frame count for all tracks
        for track_id in list(self.tracks.keys()):
            self.tracks[track_id]['frames_not_seen'] += 1

            # Remove tracks that have disappeared for too long
            if self.tracks[track_id]['frames_not_seen'] > self.max_disappeared:
                del self.tracks[track_id]

        # If no detections, return current tracks
        if not detections:
            return self.get_active_tracks()

        # Associate detections with existing tracks
        unmatched_detections = self.associate_detections_with_tracks(detections)

        # Create new tracks for unmatched detections
        for det_idx in unmatched_detections:
            det = detections[det_idx]
            track_id = self.next_id
            self.tracks[track_id] = {
                'bbox': det['bbox'],
                'center': (det['bbox_center_x'], det['bbox_center_y']),
                'class_name': det['class_name'],
                'confidence': det['confidence'],
                'frames_not_seen': 0,
                'total_frames': 1,
                'id': track_id
            }
            self.next_id += 1

        return self.get_active_tracks()

    def associate_detections_with_tracks(self, detections):
        """Associate detections with existing tracks using distance matching"""
        if not self.tracks:
            # No existing tracks, all detections become new tracks
            return list(range(len(detections)))

        # Calculate distance matrix between tracks and detections
        track_centers = [(t['center'][0], t['center'][1]) for t in self.tracks.values()]
        det_centers = [(d['bbox_center_x'], d['bbox_center_y']) for d in detections]

        if not track_centers or not det_centers:
            return list(range(len(detections)))

        # Calculate distance matrix
        dist_matrix = np.zeros((len(track_centers), len(det_centers)))
        for i, track_center in enumerate(track_centers):
            for j, det_center in enumerate(det_centers):
                dist = np.sqrt((track_center[0] - det_center[0])**2 + (track_center[1] - det_center[1])**2)
                dist_matrix[i, j] = dist

        # Use Hungarian algorithm or simple nearest neighbor for association
        # For simplicity, we'll use nearest neighbor assignment
        unmatched_detections = list(range(len(det_centers)))
        matched_pairs = []

        # Simple greedy assignment: assign closest detection to each track
        for track_idx in range(len(track_centers)):
            if len(unmatched_detections) == 0:
                break

            # Find closest unmatched detection
            track_dists = dist_matrix[track_idx, unmatched_detections]
            closest_det_idx = np.argmin(track_dists)
            min_dist = track_dists[closest_det_idx]

            if min_dist < self.max_distance:
                # Assign detection to track
                det_idx = unmatched_detections[closest_det_idx]
                track_id = list(self.tracks.keys())[track_idx]

                # Update track with new detection
                self.tracks[track_id]['bbox'] = detections[det_idx]['bbox']
                self.tracks[track_id]['center'] = (detections[det_idx]['bbox_center_x'], detections[det_idx]['bbox_center_y'])
                self.tracks[track_id]['confidence'] = detections[det_idx]['confidence']
                self.tracks[track_id]['frames_not_seen'] = 0
                self.tracks[track_id]['total_frames'] += 1

                # Remove detection from unmatched list
                unmatched_detections.remove(det_idx)

        return unmatched_detections

    def get_active_tracks(self):
        """Get currently active tracks"""
        active_tracks = []
        for track_id, track_data in self.tracks.items():
            if track_data['frames_not_seen'] == 0:  # Only return tracks seen in current frame
                track_copy = track_data.copy()
                track_copy['tracking_id'] = track_id
                active_tracks.append(track_copy)
        return active_tracks

    def predict_next_position(self, track_id):
        """Predict next position of a tracked object"""
        if track_id not in self.tracks:
            return None

        # Simple constant velocity model
        # In practice, use more sophisticated prediction models
        track = self.tracks[track_id]
        if track['total_frames'] < 2:
            return track['center']

        # Calculate velocity based on last few positions
        # (simplified - in practice, maintain history)
        return track['center']
```

## Step 6: Create Sensor Fusion Component

Create `complete_perception_system/complete_perception_system/sensor_fusion.py`:

```python
#!/usr/bin/env python3
"""
Sensor Fusion Component
"""
import numpy as np
import cv2
from typing import Dict, Any, List

class SensorFusion:
    def __init__(self, node):
        self.node = node
        self.fusion_weights = {
            'camera': 0.5,
            'lidar': 0.3,
            'imu': 0.2
        }
        self.camera_intrinsics = np.array([
            [640, 0, 320],
            [0, 640, 240],
            [0, 0, 1]
        ])
        self.extrinsics = self.get_sensor_extrinsics()  # Camera-LiDAR calibration

    def get_sensor_extrinsics(self):
        """Get sensor extrinsic calibration (camera to LiDAR transform)"""
        # In practice, load from calibration file
        # This is a placeholder identity transform
        return np.eye(4)

    def fuse_data(self, camera_features, lidar_points, tracked_objects, header):
        """Fuse data from multiple sensors"""
        try:
            fused_data = {
                'timestamp': header.stamp,
                'sensor_fusion_version': '1.0',
                'fused_objects': [],
                'environment_map': {},
                'confidence_metrics': {}
            }

            # Fuse camera and LiDAR data for 3D object localization
            fused_objects = self.fuse_camera_lidar_objects(
                camera_features, lidar_points, tracked_objects
            )
            fused_data['fused_objects'] = fused_objects

            # Create environment map from fused data
            environment_map = self.create_environment_map(lidar_points, fused_objects)
            fused_data['environment_map'] = environment_map

            # Calculate confidence metrics
            confidence_metrics = self.calculate_confidence_metrics(
                camera_features, lidar_points, fused_objects
            )
            fused_data['confidence_metrics'] = confidence_metrics

            return fused_data

        except Exception as e:
            self.node.get_logger().error(f'Error in sensor fusion: {e}')
            return {
                'timestamp': header.stamp,
                'error': str(e),
                'fused_objects': [],
                'environment_map': {},
                'confidence_metrics': {}
            }

    def fuse_camera_lidar_objects(self, camera_features, lidar_points, tracked_objects):
        """Fuse camera detections with LiDAR points"""
        fused_objects = []

        for obj in tracked_objects:
            # Project 3D LiDAR points to 2D image space to associate with camera detections
            # This requires knowing the camera-LiDAR extrinsic calibration

            # For each object, find associated LiDAR points
            obj_center_2d = (obj['bbox_center_x'], obj['bbox_center_y'])

            # Find LiDAR points that correspond to this object
            # This is a simplified approach - in practice, use more sophisticated methods
            associated_points = []
            for point in lidar_points[:50]:  # Limit for performance
                # Project LiDAR point to image coordinates (simplified)
                # In practice, use proper projection with extrinsics
                projected_x = int(point['x'] * 100 + 320)  # Simplified projection
                projected_y = int(point['y'] * 100 + 240)

                # Check if point is within object bounding box
                bbox_x, bbox_y, bbox_w, bbox_h = obj['bbox']
                if (bbox_x <= projected_x <= bbox_x + bbox_w and
                    bbox_y <= projected_y <= bbox_y + bbox_h):
                    associated_points.append(point)

            # Create fused object with combined information
            fused_obj = {
                'tracking_id': obj['tracking_id'],
                'bbox_2d': obj['bbox'],
                'class_name': obj['class_name'],
                'confidence': obj['confidence'],
                'associated_lidar_points': len(associated_points),
                'estimated_3d_position': self.estimate_3d_position(obj, associated_points),
                'size_3d': self.estimate_3d_size(associated_points)
            }

            fused_objects.append(fused_obj)

        return fused_objects

    def estimate_3d_position(self, obj_2d, associated_lidar_points):
        """Estimate 3D position from 2D detection and LiDAR points"""
        if not associated_lidar_points:
            # If no LiDAR points, use camera-based depth estimation
            # This would require depth information or monocular depth estimation
            return {
                'x': obj_2d['bbox_center_x'],  # Placeholder
                'y': obj_2d['bbox_center_y'],  # Placeholder
                'z': 5.0  # Placeholder depth
            }

        # Calculate average position of associated LiDAR points
        avg_x = np.mean([p['x'] for p in associated_lidar_points])
        avg_y = np.mean([p['y'] for p in associated_lidar_points])
        avg_z = np.mean([p['z'] for p in associated_lidar_points])

        return {
            'x': avg_x,
            'y': avg_y,
            'z': avg_z
        }

    def estimate_3d_size(self, associated_lidar_points):
        """Estimate 3D size from LiDAR points"""
        if not associated_lidar_points:
            return {'length': 1.0, 'width': 0.5, 'height': 1.5}  # Default size

        # Calculate bounding box from LiDAR points
        xs = [p['x'] for p in associated_lidar_points]
        ys = [p['y'] for p in associated_lidar_points]
        zs = [p['z'] for p in associated_lidar_points]

        length = max(xs) - min(xs)
        width = max(ys) - min(ys)
        height = max(zs) - min(zs)

        return {
            'length': max(length, 0.1),  # Minimum size
            'width': max(width, 0.1),
            'height': max(height, 0.1)
        }

    def create_environment_map(self, lidar_points, fused_objects):
        """Create environment representation from sensor data"""
        # Create occupancy grid or point cloud map
        map_data = {
            'grid_resolution': 0.1,  # 10cm resolution
            'bounds': self.calculate_bounds(lidar_points),
            'obstacles': self.identify_obstacles(lidar_points),
            'free_space': self.identify_free_space(lidar_points)
        }
        return map_data

    def calculate_bounds(self, lidar_points):
        """Calculate bounds of the environment"""
        if not lidar_points:
            return {'min_x': 0, 'max_x': 10, 'min_y': -5, 'max_y': 5}

        xs = [p['x'] for p in lidar_points]
        ys = [p['y'] for p in lidar_points]

        return {
            'min_x': min(xs) - 1,
            'max_x': max(xs) + 1,
            'min_y': min(ys) - 1,
            'max_y': max(ys) + 1
        }

    def identify_obstacles(self, lidar_points):
        """Identify obstacles from LiDAR points"""
        # Simple approach: points with high intensity or in certain regions
        obstacles = []
        for point in lidar_points:
            if point.get('intensity', 0) > 0.5:  # High intensity = obstacle
                obstacles.append({
                    'x': point['x'],
                    'y': point['y'],
                    'z': point['z'],
                    'intensity': point.get('intensity', 0)
                })
        return obstacles

    def identify_free_space(self, lidar_points):
        """Identify free space (where no obstacles detected)"""
        # This is a simplified approach
        # In practice, use ray tracing or occupancy grids
        return []

    def calculate_confidence_metrics(self, camera_features, lidar_points, fused_objects):
        """Calculate confidence metrics for fused data"""
        metrics = {
            'data_quality_score': 0.0,
            'sensor_agreement': 0.0,
            'temporal_consistency': 0.0,
            'overall_confidence': 0.0
        }

        # Calculate data quality based on number of features and points
        feature_quality = min(1.0, camera_features['density'] * 1000) if 'density' in camera_features else 0.5
        lidar_quality = min(1.0, len(lidar_points) / 1000)  # Assuming 1000 is good amount

        metrics['data_quality_score'] = (feature_quality + lidar_quality) / 2

        # Calculate sensor agreement (how well camera and LiDAR agree)
        if len(fused_objects) > 0:
            agreement_score = len([obj for obj in fused_objects if obj['associated_lidar_points'] > 0]) / len(fused_objects)
            metrics['sensor_agreement'] = agreement_score

        # Overall confidence is weighted combination
        metrics['overall_confidence'] = (
            0.4 * metrics['data_quality_score'] +
            0.4 * metrics['sensor_agreement'] +
            0.2 * metrics.get('temporal_consistency', 0.5)
        )

        return metrics
```

## Step 7: Create Launch File

Create `complete_perception_system/launch/perception_system.launch.py`:

```python
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # Declare launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')
    camera_topic = LaunchConfiguration('camera_topic', default='/camera/image_raw')
    lidar_topic = LaunchConfiguration('lidar_topic', default='/lidar/points')
    imu_topic = LaunchConfiguration('imu_topic', default='/imu/data')
    scan_topic = LaunchConfiguration('scan_topic', default='/scan')

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation time if true'
        ),
        DeclareLaunchArgument(
            'camera_topic',
            default_value='/camera/image_raw',
            description='Camera image topic'
        ),
        DeclareLaunchArgument(
            'lidar_topic',
            default_value='/lidar/points',
            description='LiDAR point cloud topic'
        ),
        DeclareLaunchArgument(
            'imu_topic',
            default_value='/imu/data',
            description='IMU data topic'
        ),
        DeclareLaunchArgument(
            'scan_topic',
            default_value='/scan',
            description='Laser scan topic'
        ),

        # Perception manager node
        Node(
            package='complete_perception_system',
            executable='perception_manager',
            name='perception_manager',
            parameters=[
                {'use_sim_time': use_sim_time}
            ],
            remappings=[
                ('/camera/image_raw', camera_topic),
                ('/lidar/points', lidar_topic),
                ('/imu/data', imu_topic),
                ('/scan', scan_topic),
                ('/perception/detections', '/perception/detections'),
                ('/perception/fused_data', '/perception/fused_data'),
                ('/perception/quality_score', '/perception/quality_score'),
                ('/perception/status', '/perception/status')
            ],
            output='screen'
        )
    ])
```

## Step 8: Update Package Configuration

Update `complete_perception_system/setup.py`:

```python
from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'complete_perception_system'

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
    description='Complete perception system for robotics',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'perception_manager = complete_perception_system.perception_manager:main',
        ],
    },
)
```

## Step 9: Build and Test

Build the package:

```bash
cd ~/ros2_ws
colcon build --packages-select complete_perception_system
source install/setup.bash
```

### Test the Perception System

1. **Launch the complete perception system**:
```bash
# Launch the perception system
ros2 launch complete_perception_system perception_system.launch.py
```

2. **Provide sensor data** (if using real sensors or simulation):
```bash
# For simulation, you might run Gazebo with sensors
# Or use a rosbag with sensor data
ros2 bag play your_sensor_data.db3
```

3. **Monitor the perception outputs**:
```bash
# Monitor detections
ros2 topic echo /perception/detections

# Monitor fused data
ros2 topic echo /perception/fused_data

# Monitor quality metrics
ros2 topic echo /perception/quality_score

# Monitor system status
ros2 topic echo /perception/status

# Visualize in RViz
ros2 run rviz2 rviz2
```

4. **Check system performance**:
```bash
# Check processing rate
ros2 topic hz /perception/detections

# Check node status
ros2 node list | grep perception

# Monitor CPU usage
htop
```

## Understanding the Complete System

This complete perception system demonstrates:

1. **Multi-sensor Integration**: Processes camera, LiDAR, IMU, and laser scan data
2. **Deep Learning Integration**: Uses neural networks for object detection
3. **Traditional Computer Vision**: Applies feature extraction and tracking
4. **Sensor Fusion**: Combines information from different sensors
5. **Real-time Processing**: Optimized for real-time performance
6. **Quality Assessment**: Validates perception reliability

## Challenges

### Challenge 1: Add More Sensor Modalities
Integrate additional sensors like radar or thermal cameras.

<details>
<summary>Hint</summary>

Modify the perception manager to subscribe to additional sensor topics and update the fusion module to handle new sensor data types.
</details>

### Challenge 2: Implement Advanced Tracking
Add more sophisticated tracking algorithms like Kalman filters.

<details>
<summary>Hint</summary>

Replace the simple tracking algorithm with a Kalman filter that predicts object positions and updates based on measurements.
</details>

### Challenge 3: Add Semantic Segmentation
Integrate semantic segmentation for scene understanding.

<details>
<summary>Hint</summary>

Add a semantic segmentation component that classifies each pixel and integrate it with object detection results.
</details>

### Challenge 4: Optimize for Edge Deployment
Optimize the system for deployment on edge devices.

<details>
<summary>Hint</summary>

Apply model quantization, pruning, and other optimization techniques to reduce computational requirements.
</details>

## Verification Checklist

- [ ] System processes camera and LiDAR data simultaneously
- [ ] Object detection works with deep learning models
- [ ] Feature extraction and tracking function correctly
- [ ] Sensor fusion combines data from multiple sensors
- [ ] Quality metrics are calculated and published
- [ ] System runs in real-time at target frequency
- [ ] All components communicate properly
- [ ] Error handling is implemented throughout

## Common Issues

### Synchronization Issues
```bash
# Check message timestamps
ros2 topic echo /camera/image_raw --field header.stamp
ros2 topic echo /lidar/points --field header.stamp

# Verify TF tree
ros2 run tf2_tools view_frames
```

### Performance Issues
```bash
# Monitor processing time
ros2 topic echo /perception/processing_time

# Check CPU usage
top -p $(pgrep -f perception_manager)

# Profile the code
python3 -m cProfile -o profile.stats your_script.py
```

### Calibration Issues
```bash
# Verify sensor calibrations
ros2 run camera_calibration cameracalibrator --size 8x6 --square 0.108
ros2 run kalibr kalibr_bagextractor --input your_bag.bag
```

## Summary

In this exercise, you learned to:
- Integrate multiple perception components into a cohesive system
- Process and fuse data from different sensor modalities
- Apply both traditional computer vision and deep learning techniques
- Optimize perception pipelines for real-time performance
- Validate perception quality and reliability
- Structure complex perception systems for maintainability

## Next Steps

Continue to [Week 10: Speech Integration](../../module-4-vision-language-action/week-10/introduction) to learn about voice interfaces and speech processing for robotics.