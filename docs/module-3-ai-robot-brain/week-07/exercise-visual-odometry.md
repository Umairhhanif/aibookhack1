---
sidebar_position: 4
---

# Exercise: Visual Odometry System

In this comprehensive exercise, you'll create a complete visual odometry system that estimates camera motion from image sequences. This will demonstrate the practical implementation of feature detection, matching, and pose estimation for robot localization.

## Objective

Create a visual odometry system that:
1. **Detects and tracks** visual features across frames
2. **Estimates camera motion** using 2D-2D correspondences
3. **Maintains pose history** for trajectory estimation
4. **Provides uncertainty estimates** for pose estimates

## Prerequisites

- Complete Week 1-7 lessons
- ROS 2 workspace set up (`~/ros2_ws`)
- Understanding of feature detection and pose estimation
- Basic Python and OpenCV knowledge

## Step 1: Create the Package

```bash
cd ~/ros2_ws/src

ros2 pkg create --build-type ament_python visual_odometry_system \
    --dependencies rclpy std_msgs sensor_msgs geometry_msgs cv_bridge tf2_ros tf2_geometry_msgs visualization_msgs
```

## Step 2: Create the Core Visual Odometry Node

Create `visual_odometry_system/visual_odometry_system/visual_odometry_node.py`:

```python
#!/usr/bin/env python3
"""
Complete Visual Odometry System
"""
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped, TwistStamped, PointStamped
from nav_msgs.msg import Path
from visualization_msgs.msg import Marker, MarkerArray
from cv_bridge import CvBridge
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
import tf_transformations
from collections import deque
import time

class VisualOdometryNode(Node):
    def __init__(self):
        super().__init__('visual_odometry_node')

        # Create subscribers
        self.image_sub = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10)
        self.camera_info_sub = self.create_subscription(
            CameraInfo, '/camera/camera_info', self.camera_info_callback, 10)

        # Create publishers
        self.pose_pub = self.create_publisher(PoseStamped, '/vo/pose', 10)
        self.twist_pub = self.create_publisher(TwistStamped, '/vo/twist', 10)
        self.path_pub = self.create_publisher(Path, '/vo/path', 10)
        self.feature_pub = self.create_publisher(MarkerArray, '/vo/features', 10)
        self.trail_pub = self.create_publisher(Marker, '/vo/trail', 10)

        # Initialize CV bridge
        self.bridge = CvBridge()

        # Initialize TF broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)

        # ORB detector and matcher
        self.orb = cv2.ORB_create(
            nfeatures=2000,
            scaleFactor=1.2,
            nlevels=8,
            edgeThreshold=31,
            patchSize=31,
            fastThreshold=20
        )
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        # Camera parameters (will be updated from camera_info)
        self.camera_matrix = np.array([
            [640, 0, 320],
            [0, 640, 240],
            [0, 0, 1]
        ], dtype=np.float32)
        self.dist_coeffs = np.zeros((4, 1))

        # Store previous frame data
        self.prev_keypoints = None
        self.prev_descriptors = None
        self.prev_gray = None

        # Pose estimation
        self.current_pose = np.eye(4)  # 4x4 transformation matrix
        self.prev_pose = np.eye(4)
        self.absolute_position = np.array([0.0, 0.0, 0.0])

        # Path and trajectory
        self.path = []
        self.max_path_length = 1000
        self.pose_history = deque(maxlen=100)

        # Feature tracking
        self.tracked_features = {}
        self.max_features_to_track = 500

        # Timing
        self.prev_time = None
        self.frame_count = 0

        # Parameters
        self.min_matches = 20
        self.ransac_threshold = 1.0
        self.max_distance = 100.0  # Maximum distance between features

        self.get_logger().info('Visual odometry system started')

    def camera_info_callback(self, msg):
        """Handle camera info messages"""
        if msg.k:
            self.camera_matrix = np.array(msg.k).reshape(3, 3)
            self.dist_coeffs = np.array(msg.d).reshape(-1, 1) if msg.d else np.zeros((4, 1))
            self.get_logger().info('Updated camera parameters')

    def image_callback(self, msg):
        """Process incoming camera image"""
        try:
            # Convert ROS image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

            # Detect features in current frame
            current_keypoints, current_descriptors = self.orb.detectAndCompute(gray, None)

            if current_keypoints is not None and current_descriptors is not None:
                # If we have previous frame data, match features
                if self.prev_descriptors is not None:
                    # Match features between current and previous frames
                    matches = self.bf.knnMatch(self.prev_descriptors, current_descriptors, k=2)

                    # Apply Lowe's ratio test
                    good_matches = []
                    for match_pair in matches:
                        if len(match_pair) == 2:
                            m, n = match_pair
                            if m.distance < 0.7 * n.distance:
                                good_matches.append(m)

                    # Check if we have enough matches
                    if len(good_matches) >= self.min_matches:
                        # Extract matched points
                        prev_points = np.float32([self.prev_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                        curr_points = np.float32([current_keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                        # Estimate essential matrix and recover pose
                        E, mask = cv2.findEssentialMat(
                            curr_points, prev_points, self.camera_matrix,
                            method=cv2.RANSAC,
                            threshold=self.ransac_threshold,
                            prob=0.999
                        )

                        if E is not None and mask is not None:
                            # Recover pose
                            points, R, t, mask_pose = cv2.recoverPose(
                                E, curr_points, prev_points, self.camera_matrix, mask=mask
                            )

                            # Create relative transformation matrix
                            T_rel = np.eye(4)
                            T_rel[:3, :3] = R
                            T_rel[:3, 3] = t.ravel()

                            # Update absolute pose
                            self.current_pose = self.current_pose @ np.linalg.inv(T_rel)

                            # Extract position
                            position = self.current_pose[:3, 3]

                            # Update absolute position
                            self.absolute_position = position

                            # Calculate velocity if we have previous time
                            velocity = None
                            if self.prev_time is not None:
                                dt = (msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9) - \
                                     (self.prev_time.sec + self.prev_time.nanosec * 1e-9)
                                if dt > 0:
                                    velocity = self.calculate_velocity(T_rel, dt)

                            # Publish results
                            self.publish_results(msg.header, position, R, t, velocity)

                            # Update path
                            self.update_path(msg.header, position)

                            # Log pose information
                            self.get_logger().info(
                                f'Frame {self.frame_count}: '
                                f'Position: [{position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f}], '
                                f'Matches: {len(good_matches)}'
                            )

                    else:
                        self.get_logger().warn(f'Not enough matches: {len(good_matches)} < {self.min_matches}')

                # Store current frame data for next iteration
                self.prev_keypoints = current_keypoints
                self.prev_descriptors = current_descriptors
                self.prev_gray = gray.copy()

            # Update timing
            self.prev_time = msg.header.stamp
            self.frame_count += 1

        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')

    def calculate_velocity(self, T_rel, dt):
        """Calculate linear and angular velocity from relative transformation"""
        # Extract translation
        translation = T_rel[:3, 3]
        linear_velocity = translation / dt

        # Extract rotation (convert to axis-angle for angular velocity)
        R = T_rel[:3, :3]
        rvec, _ = cv2.Rodrigues(R)
        angular_velocity = rvec.ravel() / dt

        return linear_velocity, angular_velocity

    def publish_results(self, header, position, R, t, velocity):
        """Publish pose, twist, and other results"""
        # Publish pose
        pose_msg = PoseStamped()
        pose_msg.header = header
        pose_msg.header.frame_id = 'map'
        pose_msg.pose.position.x = position[0]
        pose_msg.pose.position.y = position[1]
        pose_msg.pose.position.z = position[2]

        # Convert rotation matrix to quaternion
        quat = tf_transformations.quaternion_from_matrix(
            np.block([[R, np.zeros((3, 1))], [np.zeros((1, 4))]])
        )
        pose_msg.pose.orientation.x = quat[0]
        pose_msg.pose.orientation.y = quat[1]
        pose_msg.pose.orientation.z = quat[2]
        pose_msg.pose.orientation.w = quat[3]

        self.pose_pub.publish(pose_msg)

        # Publish twist if velocity is available
        if velocity is not None:
            linear_vel, angular_vel = velocity

            twist_msg = TwistStamped()
            twist_msg.header = header
            twist_msg.twist.linear.x = linear_vel[0]
            twist_msg.twist.linear.y = linear_vel[1]
            twist_msg.twist.linear.z = linear_vel[2]
            twist_msg.twist.angular.x = angular_vel[0]
            twist_msg.twist.angular.y = angular_vel[1]
            twist_msg.twist.angular.z = angular_vel[2]

            self.twist_pub.publish(twist_msg)

        # Broadcast transform
        t = TransformStamped()
        t.header.stamp = header.stamp
        t.header.frame_id = 'map'
        t.child_frame_id = 'camera'
        t.transform.translation.x = position[0]
        t.transform.translation.y = position[1]
        t.transform.translation.z = position[2]
        t.transform.rotation.x = quat[0]
        t.transform.rotation.y = quat[1]
        t.transform.rotation.z = quat[2]
        t.transform.rotation.w = quat[3]

        self.tf_broadcaster.sendTransform(t)

    def update_path(self, header, position):
        """Update and publish path"""
        # Create pose for path
        pose_stamped = PoseStamped()
        pose_stamped.header = header
        pose_stamped.header.frame_id = 'map'
        pose_stamped.pose.position.x = position[0]
        pose_stamped.pose.position.y = position[1]
        pose_stamped.pose.position.z = position[2]
        pose_stamped.pose.orientation.w = 1.0

        # Add to path
        self.path.append(pose_stamped)

        # Limit path length
        if len(self.path) > self.max_path_length:
            self.path.pop(0)

        # Publish path
        path_msg = Path()
        path_msg.header = header
        path_msg.header.frame_id = 'map'
        path_msg.poses = self.path
        self.path_pub.publish(path_msg)

    def publish_features(self, header, keypoints):
        """Publish feature markers for visualization"""
        marker_array = MarkerArray()

        for i, kp in enumerate(keypoints[:100]):  # Limit for performance
            marker = Marker()
            marker.header = header
            marker.ns = "features"
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD

            # Position (just for visualization)
            marker.pose.position.x = kp.pt[0] * 0.001  # Scale down
            marker.pose.position.y = kp.pt[1] * 0.001
            marker.pose.position.z = 0.0
            marker.pose.orientation.w = 1.0

            # Scale
            marker.scale.x = 0.01
            marker.scale.y = 0.01
            marker.scale.z = 0.01

            # Color (blue)
            marker.color.b = 1.0
            marker.color.a = 1.0

            marker_array.markers.append(marker)

        self.feature_pub.publish(marker_array)

def main(args=None):
    rclpy.init(args=args)
    node = VisualOdometryNode()

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

## Step 3: Create an Enhanced Feature Tracker Node

Create `visual_odometry_system/visual_odometry_system/feature_tracker_node.py`:

```python
#!/usr/bin/env python3
"""
Enhanced Feature Tracker with Lucas-Kanade Tracking
"""
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped
from visualization_msgs.msg import Marker, MarkerArray
from cv_bridge import CvBridge
from collections import deque

class FeatureTrackerNode(Node):
    def __init__(self):
        super().__init__('feature_tracker_node')

        # Create subscriber
        self.image_sub = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10)

        # Create publishers
        self.feature_pub = self.create_publisher(MarkerArray, '/tracked_features', 10)
        self.feature_points_pub = self.create_publisher(Marker, '/feature_points', 10)

        # Initialize CV bridge
        self.bridge = CvBridge()

        # Parameters for feature detection
        self.feature_params = dict(
            maxCorners=500,
            qualityLevel=0.01,
            minDistance=10,
            blockSize=7
        )

        # Parameters for Lucas-Kanade optical flow
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )

        # Store tracking information
        self.prev_gray = None
        self.prev_features = None
        self.feature_ids = None
        self.feature_history = {}
        self.next_feature_id = 0

        # Tracking parameters
        self.min_features = 50
        self.max_features = 500
        self.reinit_threshold = 30

        # Colors for visualization
        self.colors = np.random.randint(0, 255, (1000, 3))

        self.get_logger().info('Feature tracker started')

    def image_callback(self, msg):
        """Process incoming camera image"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

            if self.prev_gray is None:
                # Initialize features on first frame
                self.prev_features = cv2.goodFeaturesToTrack(
                    gray, mask=None, **self.feature_params)
                self.feature_ids = np.arange(len(self.prev_features)).astype(int) if self.prev_features is not None else np.array([])
                self.prev_gray = gray.copy()
                return

            if self.prev_features is not None and len(self.prev_features) > 0:
                # Calculate optical flow
                new_features, status, error = cv2.calcOpticalFlowPyrLK(
                    self.prev_gray, gray, self.prev_features, None, **self.lk_params)

                # Select good points
                good_new = new_features[status == 1] if new_features is not None else np.array([])
                good_old = self.prev_features[status == 1] if self.prev_features is not None else np.array([])
                good_ids = self.feature_ids[status == 1] if self.feature_ids is not None else np.array([])

                # Update feature positions
                self.prev_features = good_new.reshape(-1, 1, 2) if len(good_new) > 0 else None
                self.feature_ids = good_ids if len(good_ids) > 0 else np.array([])

                # Publish tracked features
                self.publish_tracked_features(msg.header, good_new, good_ids)

                # Re-detect features if too few remain
                if len(good_new) < self.reinit_threshold:
                    self.reinitialize_features(gray)

                # Update previous frame
                self.prev_gray = gray.copy()

        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')

    def reinitialize_features(self, gray):
        """Reinitialize features when too few remain tracked"""
        # Detect new features
        mask = np.ones_like(gray, dtype=np.uint8) * 255

        # Avoid areas where features are already tracked
        if self.prev_features is not None:
            for pt in self.prev_features:
                x, y = int(pt[0][0]), int(pt[0][1])
                cv2.circle(mask, (x, y), 20, 0, -1)  # Black out regions around tracked features

        new_features = cv2.goodFeaturesToTrack(
            gray, mask=mask, **self.feature_params)

        if new_features is not None:
            # Combine with existing features
            if self.prev_features is not None:
                self.prev_features = np.vstack([self.prev_features, new_features.reshape(-1, 1, 2)])
                new_ids = np.arange(self.next_feature_id, self.next_feature_id + len(new_features))
                self.feature_ids = np.hstack([self.feature_ids, new_ids])
                self.next_feature_id += len(new_features)
            else:
                self.prev_features = new_features.reshape(-1, 1, 2)
                self.feature_ids = np.arange(self.next_feature_id, self.next_feature_id + len(new_features))
                self.next_feature_id += len(new_features)

            # Limit number of features
            if len(self.prev_features) > self.max_features:
                indices = np.random.choice(len(self.prev_features), self.max_features, replace=False)
                self.prev_features = self.prev_features[indices]
                self.feature_ids = self.feature_ids[indices]

    def publish_tracked_features(self, header, features, feature_ids):
        """Publish tracked features for visualization"""
        if len(features) == 0:
            return

        # Create marker array for features
        marker_array = MarkerArray()

        for i, (pt, fid) in enumerate(zip(features, feature_ids)):
            x, y = pt.ravel()

            # Create marker for this feature
            marker = Marker()
            marker.header = header
            marker.ns = "tracked_features"
            marker.id = int(fid)
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD

            marker.pose.position.x = float(x)
            marker.pose.position.y = float(y)
            marker.pose.position.z = 0.0
            marker.pose.orientation.w = 1.0

            marker.scale.x = 2.0  # Width
            marker.scale.y = 2.0  # Height
            marker.scale.z = 0.1  # Depth

            # Color based on feature ID
            color_idx = fid % len(self.colors)
            marker.color.r = float(self.colors[color_idx][0]) / 255.0
            marker.color.g = float(self.colors[color_idx][1]) / 255.0
            marker.color.b = float(self.colors[color_idx][2]) / 255.0
            marker.color.a = 1.0

            marker_array.markers.append(marker)

        self.feature_pub.publish(marker_array)

def main(args=None):
    rclpy.init(args=args)
    node = FeatureTrackerNode()

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

## Step 4: Create a Pose Graph Optimization Node

Create `visual_odometry_system/visual_odometry_system/pose_graph_node.py`:

```python
#!/usr/bin/env python3
"""
Pose Graph Optimization for Visual Odometry
"""
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import Path
from std_msgs.msg import Float64
from collections import deque
import tf_transformations

class PoseGraphNode(Node):
    def __init__(self):
        super().__init__('pose_graph_node')

        # Create subscribers
        self.odom_sub = self.create_subscription(
            PoseStamped, '/vo/pose', self.odom_callback, 10)
        self.loop_closure_sub = self.create_subscription(
            PoseWithCovarianceStamped, '/loop_closure', self.loop_closure_callback, 10)

        # Create publishers
        self.optimized_pose_pub = self.create_publisher(PoseStamped, '/vo/optimized_pose', 10)
        self.optimized_path_pub = self.create_publisher(Path, '/vo/optimized_path', 10)
        self.consistency_pub = self.create_publisher(Float64, '/vo/consistency', 10)

        # Store pose graph
        self.poses = deque(maxlen=1000)  # Keep last 1000 poses
        self.constraints = deque(maxlen=1000)  # Keep last 1000 constraints
        self.loop_constraints = []  # Loop closure constraints

        # Graph optimization parameters
        self.max_translation = 10.0  # Maximum expected translation between frames
        self.max_rotation = 0.5  # Maximum expected rotation (radians)
        self.consistency_threshold = 0.1  # Threshold for consistency check

        # Timing
        self.prev_pose_time = None

        self.get_logger().info('Pose graph optimizer started')

    def odom_callback(self, msg):
        """Process incoming odometry pose"""
        try:
            # Convert pose message to transformation matrix
            T = self.pose_to_transform(msg.pose)

            # Add to pose graph
            self.poses.append({
                'timestamp': msg.header.stamp,
                'transform': T,
                'pose_stamped': msg
            })

            # Add relative constraint to previous pose
            if len(self.poses) > 1:
                prev_T = self.poses[-2]['transform']
                curr_T = self.poses[-1]['transform']

                # Calculate relative transformation
                relative_T = np.linalg.inv(prev_T) @ curr_T
                constraint = {
                    'from': len(self.poses) - 2,
                    'to': len(self.poses) - 1,
                    'relative_transform': relative_T,
                    'covariance': self.estimate_constraint_covariance(relative_T)
                }

                self.constraints.append(constraint)

            # Check for loop closures
            self.check_for_loop_closures()

            # Publish current pose
            self.optimized_pose_pub.publish(msg)

            # Publish path
            self.publish_path(msg.header)

        except Exception as e:
            self.get_logger().error(f'Error processing odometry: {e}')

    def loop_closure_callback(self, msg):
        """Process loop closure detection"""
        try:
            # Convert loop closure to transformation
            loop_T = self.pose_to_transform(msg.pose.pose)

            # Find closest pose in history
            closest_idx = self.find_closest_pose(loop_T)

            if closest_idx is not None:
                # Add loop closure constraint
                constraint = {
                    'from': closest_idx,
                    'to': len(self.poses) - 1,  # Current pose
                    'relative_transform': loop_T,
                    'covariance': self.estimate_loop_covariance()
                }

                self.loop_constraints.append(constraint)
                self.get_logger().info(f'Added loop closure constraint between poses {closest_idx} and {len(self.poses) - 1}')

                # Perform graph optimization
                self.optimize_graph()

        except Exception as e:
            self.get_logger().error(f'Error processing loop closure: {e}')

    def find_closest_pose(self, target_T, max_distance=2.0):
        """Find the closest pose in history to the target pose"""
        if len(self.poses) < 10:  # Need some history
            return None

        target_pos = target_T[:3, 3]
        min_distance = float('inf')
        closest_idx = None

        # Search in recent history (last 100 poses)
        start_idx = max(0, len(self.poses) - 100)
        for i in range(start_idx, len(self.poses) - 10):  # Don't match with very recent poses
            pose_T = self.poses[i]['transform']
            pose_pos = pose_T[:3, 3]

            distance = np.linalg.norm(target_pos - pose_pos)
            if distance < min_distance and distance < max_distance:
                min_distance = distance
                closest_idx = i

        return closest_idx if min_distance < max_distance else None

    def check_for_loop_closures(self):
        """Check if current pose is close to a previous pose (potential loop closure)"""
        if len(self.poses) < 50:  # Need sufficient history
            return

        current_T = self.poses[-1]['transform']
        current_pos = current_T[:3, 3]

        # Check against poses from 50 steps ago to avoid immediate matches
        for i in range(max(0, len(self.poses) - 200), len(self.poses) - 50):
            pose_T = self.poses[i]['transform']
            pose_pos = pose_T[:3, 3]

            distance = np.linalg.norm(current_pos - pose_pos)

            if distance < 1.0:  # Potential loop closure
                # Verify with more detailed check
                if self.verify_loop_closure(current_T, pose_T):
                    # Publish loop closure for other nodes to process
                    loop_msg = PoseWithCovarianceStamped()
                    loop_msg.header = self.poses[-1]['pose_stamped'].header
                    loop_msg.header.frame_id = 'map'

                    # Set pose to the difference
                    diff_T = np.linalg.inv(pose_T) @ current_T
                    loop_msg.pose.pose = self.transform_to_pose(diff_T)

                    # This would trigger further processing in a real system
                    self.get_logger().info(f'Potential loop closure detected at poses {i} and {len(self.poses) - 1}')

    def verify_loop_closure(self, T1, T2, max_translation=0.5, max_rotation=0.2):
        """Verify if two poses represent a true loop closure"""
        # Calculate relative transformation
        diff_T = np.linalg.inv(T2) @ T1

        # Check translation magnitude
        translation = np.linalg.norm(diff_T[:3, 3])
        if translation > max_translation:
            return False

        # Check rotation magnitude
        R = diff_T[:3, :3]
        # Convert to axis-angle to get rotation magnitude
        trace = np.trace(R)
        angle = np.arccos(max(-1, min(1, (trace - 1) / 2)))
        if angle > max_rotation:
            return False

        return True

    def estimate_constraint_covariance(self, T):
        """Estimate covariance for a relative transformation constraint"""
        # Simplified covariance estimation based on transformation magnitude
        translation_norm = np.linalg.norm(T[:3, 3])
        rotation_matrix = T[:3, :3]
        trace = np.trace(rotation_matrix)
        rotation_angle = np.arccos(max(-1, min(1, (trace - 1) / 2)))

        # Higher uncertainty for larger motions
        translation_uncertainty = min(0.1, translation_norm * 0.1)
        rotation_uncertainty = min(0.05, rotation_angle * 0.1)

        # Create diagonal covariance matrix
        cov = np.eye(6)
        cov[0:3, 0:3] *= translation_uncertainty**2
        cov[3:6, 3:6] *= rotation_uncertainty**2

        return cov

    def estimate_loop_covariance(self):
        """Estimate covariance for loop closure constraint"""
        # Loop closures typically have lower uncertainty than odometry
        cov = np.eye(6)
        cov[0:3, 0:3] *= 0.01**2  # Lower position uncertainty
        cov[3:6, 3:6] *= 0.01**2  # Lower orientation uncertainty
        return cov

    def optimize_graph(self):
        """Perform graph optimization (simplified version)"""
        # In a real implementation, this would use a graph optimization library
        # like g2o, Ceres, or GTSAM
        self.get_logger().info('Performing graph optimization...')

        # For this exercise, we'll just log that optimization would occur
        # In practice, this would optimize all poses based on constraints
        if len(self.loop_constraints) > 0:
            self.get_logger().info(f'Optimizing graph with {len(self.constraints)} odometry constraints and {len(self.loop_constraints)} loop constraints')

    def pose_to_transform(self, pose):
        """Convert geometry_msgs/Pose to 4x4 transformation matrix"""
        # Extract position
        pos = np.array([pose.position.x, pose.position.y, pose.position.z])

        # Extract orientation (quaternion)
        quat = np.array([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])

        # Convert quaternion to rotation matrix
        R = tf_transformations.quaternion_matrix(quat)

        # Create transformation matrix
        T = np.eye(4)
        T[:3, :3] = R[:3, :3]
        T[:3, 3] = pos

        return T

    def transform_to_pose(self, T):
        """Convert 4x4 transformation matrix to geometry_msgs/Pose"""
        from geometry_msgs.msg import Pose
        pose = Pose()

        # Extract position
        pose.position.x = T[0, 3]
        pose.position.y = T[1, 3]
        pose.position.z = T[2, 3]

        # Extract rotation matrix to quaternion
        R = T[:3, :3]
        quat = tf_transformations.quaternion_from_matrix(
            np.block([[R, np.zeros((3, 1))], [np.zeros((1, 4))]])
        )
        pose.orientation.x = quat[0]
        pose.orientation.y = quat[1]
        pose.orientation.z = quat[2]
        pose.orientation.w = quat[3]

        return pose

    def publish_path(self, header):
        """Publish the trajectory path"""
        if len(self.poses) == 0:
            return

        path_msg = Path()
        path_msg.header = header
        path_msg.header.frame_id = 'map'

        for pose_data in self.poses:
            path_msg.poses.append(pose_data['pose_stamped'])

        self.optimized_path_pub.publish(path_msg)

def main(args=None):
    rclpy.init(args=args)
    node = PoseGraphNode()

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

Create `visual_odometry_system/launch/visual_odometry.launch.py`:

```python
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # Declare launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')
    camera_topic = LaunchConfiguration('camera_topic', default='/camera/image_raw')
    camera_info_topic = LaunchConfiguration('camera_info_topic', default='/camera/camera_info')

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
            'camera_info_topic',
            default_value='/camera/camera_info',
            description='Camera info topic'
        ),

        # Visual odometry node
        Node(
            package='visual_odometry_system',
            executable='visual_odometry_node',
            name='visual_odometry',
            parameters=[
                {'use_sim_time': use_sim_time}
            ],
            remappings=[
                ('/camera/image_raw', camera_topic),
                ('/camera/camera_info', camera_info_topic)
            ],
            output='screen'
        ),

        # Feature tracker node
        Node(
            package='visual_odometry_system',
            executable='feature_tracker_node',
            name='feature_tracker',
            parameters=[
                {'use_sim_time': use_sim_time}
            ],
            remappings=[
                ('/camera/image_raw', camera_topic)
            ],
            output='screen'
        ),

        # Pose graph optimization node
        Node(
            package='visual_odometry_system',
            executable='pose_graph_node',
            name='pose_graph_optimizer',
            parameters=[
                {'use_sim_time': use_sim_time}
            ],
            output='screen'
        )
    ])
```

## Step 6: Create a Performance Monitor Node

Create `visual_odometry_system/visual_odometry_system/performance_monitor.py`:

```python
#!/usr/bin/env python3
"""
Performance Monitor for Visual Odometry System
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped, TwistStamped
from std_msgs.msg import Float64
from visualization_msgs.msg import MarkerArray
import time
from collections import deque

class PerformanceMonitorNode(Node):
    def __init__(self):
        super().__init__('performance_monitor')

        # Create subscribers
        self.image_sub = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10)
        self.pose_sub = self.create_subscription(
            PoseStamped, '/vo/pose', self.pose_callback, 10)
        self.feature_sub = self.create_subscription(
            MarkerArray, '/vo/features', self.feature_callback, 10)

        # Create publishers for performance metrics
        self.fps_pub = self.create_publisher(Float64, '/vo/fps', 10)
        self.feature_count_pub = self.create_publisher(Float64, '/vo/feature_count', 10)
        self.processing_time_pub = self.create_publisher(Float64, '/vo/processing_time', 10)

        # Performance tracking
        self.frame_times = deque(maxlen=30)  # Last 30 frames for FPS calculation
        self.pose_times = deque(maxlen=30)
        self.feature_counts = deque(maxlen=30)

        # Timing
        self.last_image_time = None
        self.fps_timer = self.create_timer(1.0, self.publish_fps)  # Publish FPS every second

        self.get_logger().info('Performance monitor started')

    def image_callback(self, msg):
        """Monitor image processing rate"""
        current_time = time.time()

        if self.last_image_time is not None:
            processing_time = current_time - self.last_image_time
            self.frame_times.append(processing_time)

        self.last_image_time = current_time

    def pose_callback(self, msg):
        """Monitor pose estimation timing"""
        # In a real system, you would measure the time between image input and pose output
        pass

    def feature_callback(self, msg):
        """Monitor feature detection"""
        feature_count = len(msg.markers)
        self.feature_counts.append(feature_count)

    def publish_fps(self):
        """Publish frames per second"""
        if len(self.frame_times) > 0:
            avg_time = sum(self.frame_times) / len(self.frame_times)
            fps = 1.0 / avg_time if avg_time > 0 else 0.0

            fps_msg = Float64()
            fps_msg.data = fps
            self.fps_pub.publish(fps_msg)

            # Publish average feature count
            if len(self.feature_counts) > 0:
                avg_features = sum(self.feature_counts) / len(self.feature_counts)
                feature_msg = Float64()
                feature_msg.data = avg_features
                self.feature_count_pub.publish(feature_msg)

            self.get_logger().info(f'Current FPS: {fps:.2f}, Avg Features: {avg_features:.2f}')

def main(args=None):
    rclpy.init(args=args)
    node = PerformanceMonitorNode()

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

## Step 7: Update Package Configuration

Update `visual_odometry_system/setup.py`:

```python
from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'visual_odometry_system'

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
    description='Visual odometry system for robot localization',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'visual_odometry_node = visual_odometry_system.visual_odometry_node:main',
            'feature_tracker_node = visual_odometry_system.feature_tracker_node:main',
            'pose_graph_node = visual_odometry_system.pose_graph_node:main',
            'performance_monitor = visual_odometry_system.performance_monitor:main',
        ],
    },
)
```

## Step 8: Build and Test

Build the package:

```bash
cd ~/ros2_ws
colcon build --packages-select visual_odometry_system
source install/setup.bash
```

### Test the Visual Odometry System

1. **Run the complete system**:
```bash
# Launch the visual odometry system
ros2 launch visual_odometry_system visual_odometry.launch.py
```

2. **Provide camera data** (if using real camera or simulation):
```bash
# For simulation, you might run Gazebo with a camera
# Or use a rosbag with camera data
ros2 bag play your_camera_bag.db3
```

3. **Monitor the output**:
```bash
# Monitor pose estimates
ros2 topic echo /vo/pose

# Monitor path
ros2 topic echo /vo/path

# Monitor performance
ros2 topic echo /vo/fps

# Visualize in RViz
ros2 run rviz2 rviz2
```

4. **Check the system status**:
```bash
# List all topics
ros2 topic list | grep vo

# Check node status
ros2 node list | grep vo
```

## Understanding the System

This visual odometry system demonstrates:

1. **Feature Detection and Tracking**: Detects ORB features and tracks them using matching
2. **Pose Estimation**: Estimates camera motion using 2D-2D correspondences and essential matrix
3. **Trajectory Estimation**: Maintains and publishes robot path
4. **Performance Monitoring**: Tracks system performance metrics
5. **Graph Optimization**: Basic structure for loop closure and global optimization

## Challenges

### Challenge 1: Add IMU Integration
Integrate IMU data to improve pose estimation accuracy.

<details>
<summary>Hint</summary>

Create a sensor fusion node that combines visual odometry with IMU data using a Kalman filter or complementary filter.
</details>

### Challenge 2: Implement Loop Closure
Add more sophisticated loop closure detection using bag-of-words.

<details>
<summary>Hint</summary>

Implement a place recognition system that can identify when the robot returns to a previously visited location.
</details>

### Challenge 3: Add Scale Recovery
Recover absolute scale using stereo vision or known objects.

<details>
<summary>Hint</summary>

Use stereo camera data or detect objects of known size to estimate absolute scale.
</details>

### Challenge 4: Optimize Performance
Improve computational efficiency for real-time operation.

<details>
<summary>Hint</summary>

Use multi-threading, optimize feature detection parameters, or implement feature selection strategies.
</details>

## Verification Checklist

- [ ] Visual odometry node processes camera images
- [ ] Feature tracking works across frames
- [ ] Pose estimation provides reasonable motion estimates
- [ ] Path is published and visualizable
- [ ] Performance metrics are available
- [ ] System runs in real-time
- [ ] Trajectory is smooth and plausible
- [ ] All nodes communicate properly

## Common Issues

### Feature Tracking Issues
```bash
# Check feature count
ros2 topic echo /vo/feature_count

# Verify camera calibration
ros2 run camera_calibration cameracalibrator --size 8x6 --square 0.108
```

### Pose Estimation Issues
```bash
# Check pose output
ros2 topic echo /vo/pose

# Verify camera matrix
ros2 topic echo /camera/camera_info
```

### Performance Issues
```bash
# Monitor FPS
ros2 topic echo /vo/fps

# Check processing time
ros2 topic echo /vo/processing_time
```

## Summary

In this exercise, you learned to:
- Implement a complete visual odometry pipeline
- Track features across image sequences
- Estimate camera motion using 2D-2D correspondences
- Maintain and publish robot trajectories
- Monitor system performance
- Structure a complex perception system

## Next Steps

Continue to [Week 8: Nav2 Stack](../../module-3-ai-robot-brain/week-08/introduction) to learn about navigation and path planning systems.