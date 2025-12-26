---
sidebar_position: 2
---

# Computer Vision Fundamentals for Robotics

Computer vision is the cornerstone of robot perception, enabling machines to interpret and understand visual information from cameras and other imaging sensors. This section covers the fundamental concepts and techniques essential for robotics applications.

## Computer Vision in Robotics

Computer vision for robotics differs from traditional computer vision in several key ways:

| Traditional Computer Vision | Robotics Computer Vision |
|----------------------------|-------------------------|
| **Single Image Analysis** | **Sequential Image Processing** |
| **Offline Processing** | **Real-time Processing** |
| **Controlled Environments** | **Dynamic Environments** |
| **Static Cameras** | **Moving Cameras** |
| **Known Poses** | **Unknown Poses** |
| **Single View** | **Multi-view Integration** |

## Image Formation and Camera Models

### Pinhole Camera Model

The pinhole camera model describes how 3D points are projected onto a 2D image plane:

```
[u]   [fx  0  cx 0] [X]
[v] = [0  fy  cy 0] [Y]
[1]   [0   0   1 0] [Z]
                      [1]
```

Where:
- (u, v) are image coordinates
- (X, Y, Z) are 3D world coordinates
- fx, fy are focal lengths (pixels)
- cx, cy are principal point coordinates (pixels)

```python
#!/usr/bin/env python3
"""
Camera Model Implementation
"""
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped
import tf2_ros

class CameraModelNode(Node):
    def __init__(self):
        super().__init__('camera_model_node')

        # Create subscribers
        self.image_sub = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10)
        self.camera_info_sub = self.create_subscription(
            CameraInfo, '/camera/camera_info', self.camera_info_callback, 10)

        # Create publishers
        self.projected_pub = self.create_publisher(Image, '/projected_points', 10)

        # Initialize CV bridge
        self.bridge = CvBridge()

        # Camera parameters
        self.camera_matrix = None
        self.distortion_coeffs = None

        # Store camera pose
        self.camera_pose = np.eye(4)

        self.get_logger().info('Camera model node started')

    def camera_info_callback(self, msg):
        """Handle camera info messages"""
        self.camera_matrix = np.array(msg.k).reshape(3, 3)
        self.distortion_coeffs = np.array(msg.d)

        self.get_logger().info(f'Camera matrix: \n{self.camera_matrix}')
        self.get_logger().info(f'Distortion coefficients: {self.distortion_coeffs}')

    def project_3d_to_2d(self, points_3d):
        """Project 3D points to 2D image coordinates"""
        if self.camera_matrix is None:
            return None

        # Apply camera matrix
        points_2d_homogeneous = self.camera_matrix @ points_3d
        points_2d = points_2d_homogeneous[:2, :] / points_2d_homogeneous[2, :]

        return points_2d

    def triangulate_2d_to_3d(self, points_2d, depth_map):
        """Triangulate 2D points to 3D using depth information"""
        if self.camera_matrix is None:
            return None

        # Invert camera matrix
        camera_matrix_inv = np.linalg.inv(self.camera_matrix)

        # Convert 2D points to homogeneous coordinates
        points_2d_homo = np.vstack([points_2d, np.ones((1, points_2d.shape[1]))])

        # Back-project to 3D
        rays = camera_matrix_inv @ points_2d_homo

        # Scale by depth
        points_3d = rays * depth_map[None, :]  # Broadcasting depth

        return points_3d

    def image_callback(self, msg):
        """Process incoming camera image"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

            # Example: Create some 3D points and project them
            points_3d = np.array([
                [1, 2, 3, 4],  # X coordinates
                [1, 1, 2, 2],  # Y coordinates
                [0, 0, 0, 0]   # Z coordinates (depth)
            ])

            # Project to 2D
            points_2d = self.project_3d_to_2d(points_3d)

            if points_2d is not None:
                # Draw projected points on image
                for i in range(points_2d.shape[1]):
                    x, y = int(points_2d[0, i]), int(points_2d[1, i])
                    if 0 <= x < cv_image.shape[1] and 0 <= y < cv_image.shape[0]:
                        cv2.circle(cv_image, (x, y), 5, (0, 255, 0), -1)

                # Publish the image with projected points
                projected_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
                projected_msg.header = msg.header
                self.projected_pub.publish(projected_msg)

        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = CameraModelNode()

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

### Camera Calibration

Camera calibration determines intrinsic and extrinsic parameters:

```python
#!/usr/bin/env python3
"""
Camera Calibration Example
"""
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from sensor_msgs.msg import CameraInfo
from message_filters import ApproximateTimeSynchronizer, Subscriber

class CameraCalibratorNode(Node):
    def __init__(self):
        super().__init__('camera_calibrator_node')

        # Create subscribers
        self.image_sub = Subscriber(self, Image, '/camera/image_raw')
        self.info_sub = Subscriber(self, CameraInfo, '/camera/camera_info')

        # Create synchronizer for synchronized processing
        self.sync = ApproximateTimeSynchronizer(
            [self.image_sub, self.info_sub], queue_size=10, slop=0.1)
        self.sync.registerCallback(self.calibrate_callback)

        # Initialize CV bridge
        self.bridge = CvBridge()

        # Calibration parameters
        self.pattern_size = (9, 6)  # Chessboard pattern size
        self.square_size = 0.025  # Square size in meters (2.5cm)
        self.obj_points = []  # 3D points in real world space
        self.img_points = []  # 2D points in image plane
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # Object points (world coordinates)
        self.objp = np.zeros((self.pattern_size[0] * self.pattern_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:self.pattern_size[0], 0:self.pattern_size[1]].T.reshape(-1, 2) * self.square_size

        self.get_logger().info('Camera calibrator node started')

    def calibrate_callback(self, image_msg, info_msg):
        """Process synchronized image and camera info"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(image_msg, 'bgr8')
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

            # Find chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, self.pattern_size, None)

            if ret:
                # Refine corner positions
                corners_refined = cv2.cornerSubPix(
                    gray, corners, (11, 11), (-1, -1), self.criteria)

                if corners_refined is not None:
                    self.obj_points.append(self.objp)
                    self.img_points.append(corners_refined)

                    # Draw and display the corners
                    cv2.drawChessboardCorners(cv_image, self.pattern_size, corners_refined, ret)

                    # If we have enough samples, perform calibration
                    if len(self.obj_points) >= 10:
                        self.perform_calibration(gray.shape[::-1])

            # Display the image
            cv2.imshow('Calibration', cv_image)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f'Error in calibration: {e}')

    def perform_calibration(self, image_size):
        """Perform camera calibration"""
        try:
            ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
                self.obj_points, self.img_points, image_size, None, None)

            if ret:
                self.get_logger().info(f'Calibration successful!')
                self.get_logger().info(f'Camera matrix:\n{camera_matrix}')
                self.get_logger().info(f'Distortion coefficients: {dist_coeffs.ravel()}')

                # Calculate reprojection error
                total_error = 0
                for i in range(len(self.obj_points)):
                    img_points2, _ = cv2.projectPoints(
                        self.obj_points[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
                    error = cv2.norm(self.img_points[i], img_points2, cv2.NORM_L2) / len(img_points2)
                    total_error += error

                mean_error = total_error / len(self.obj_points)
                self.get_logger().info(f'Mean reprojection error: {mean_error}')

                # Reset for next calibration cycle
                self.obj_points = []
                self.img_points = []

        except Exception as e:
            self.get_logger().error(f'Calibration error: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = CameraCalibratorNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Feature Detection and Description

### Corner Detection

```python
#!/usr/bin/env python3
"""
Feature Detection Example
"""
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Point

class FeatureDetectorNode(Node):
    def __init__(self):
        super().__init__('feature_detector_node')

        # Create subscriber
        self.image_sub = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10)

        # Create publishers
        self.features_pub = self.create_publisher(MarkerArray, '/detected_features', 10)

        # Initialize CV bridge
        self.bridge = CvBridge()

        # Feature detectors
        self.harris_detector = cv2.cornerHarris
        self.shi_tomasi = cv2.goodFeaturesToTrack
        self.orb = cv2.ORB_create()
        self.sift = cv2.SIFT_create()

        # Feature tracking
        self.prev_features = None
        self.feature_colors = np.random.randint(0, 255, (100, 3))

        self.get_logger().info('Feature detector node started')

    def image_callback(self, msg):
        """Process incoming camera image"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

            # Harris corner detection
            harris_corners = self.detect_harris_corners(gray)

            # Shi-Tomasi corner detection
            shi_tomasi_corners = self.detect_shi_tomasi_corners(gray)

            # ORB feature detection
            orb_keypoints = self.detect_orb_features(gray)

            # Draw features on image
            output_image = self.draw_features(cv_image, harris_corners, shi_tomasi_corners, orb_keypoints)

            # Publish features as markers
            self.publish_features_as_markers(harris_corners, shi_tomasi_corners, orb_keypoints, msg.header)

        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')

    def detect_harris_corners(self, gray):
        """Detect corners using Harris corner detector"""
        # Normalize and convert to float
        gray_float = np.float32(gray)

        # Apply Harris corner detection
        dst = cv2.cornerHarris(gray_float, blockSize=2, ksize=3, k=0.04)

        # Dilate to mark corners more clearly
        dst = cv2.dilate(dst, None)

        # Threshold for an optimal value
        threshold = 0.01 * dst.max()
        corners = np.where(dst > threshold)

        # Convert to list of points
        corner_points = [(y, x) for y, x in zip(corners[0], corners[1])]
        return corner_points

    def detect_shi_tomasi_corners(self, gray):
        """Detect corners using Shi-Tomasi method"""
        corners = cv2.goodFeaturesToTrack(
            gray, maxCorners=100, qualityLevel=0.01, minDistance=10)

        if corners is not None:
            return [(int(point[0][0]), int(point[0][1])) for point in corners]
        else:
            return []

    def detect_orb_features(self, gray):
        """Detect features using ORB"""
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)

        if keypoints is not None:
            return [(int(kp.pt[0]), int(kp.pt[1])) for kp in keypoints]
        else:
            return []

    def draw_features(self, image, harris_corners, shi_tomasi_corners, orb_corners):
        """Draw features on image with different colors"""
        output = image.copy()

        # Draw Harris corners (red)
        for corner in harris_corners:
            cv2.circle(output, corner, 3, (0, 0, 255), -1)

        # Draw Shi-Tomasi corners (green)
        for corner in shi_tomasi_corners:
            cv2.circle(output, corner, 3, (0, 255, 0), -1)

        # Draw ORB features (blue)
        for corner in orb_corners:
            cv2.circle(output, corner, 3, (255, 0, 0), -1)

        return output

    def publish_features_as_markers(self, harris_corners, shi_tomasi_corners, orb_corners, header):
        """Publish features as visualization markers"""
        marker_array = MarkerArray()

        # Create markers for Harris corners
        for i, corner in enumerate(harris_corners):
            marker = Marker()
            marker.header = header
            marker.ns = "harris_corners"
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD

            # Convert pixel coordinates to approximate 3D coordinates
            marker.pose.position.x = corner[0] * 0.001  # Scale to reasonable units
            marker.pose.position.y = corner[1] * 0.001
            marker.pose.position.z = 0.0
            marker.pose.orientation.w = 1.0

            # Scale
            marker.scale.x = 0.01
            marker.scale.y = 0.01
            marker.scale.z = 0.01

            # Color (red)
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 1.0

            marker_array.markers.append(marker)

        self.features_pub.publish(marker_array)

def main(args=None):
    rclpy.init(args=args)
    node = FeatureDetectorNode()

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

### Feature Matching

```python
#!/usr/bin/env python3
"""
Feature Matching Example
"""
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import Point

class FeatureMatcherNode(Node):
    def __init__(self):
        super().__init__('feature_matcher_node')

        # Create subscribers for stereo images or temporal matching
        self.image_sub = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10)

        # Create publishers
        self.matches_pub = self.create_publisher(MarkerArray, '/feature_matches', 10)

        # Initialize CV bridge
        self.bridge = CvBridge()

        # Feature detector and matcher
        self.orb = cv2.ORB_create(nfeatures=1000)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        # Store previous frame features
        self.prev_keypoints = None
        self.prev_descriptors = None

        self.get_logger().info('Feature matcher node started')

    def image_callback(self, msg):
        """Process incoming camera image and match features"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

            # Detect features in current frame
            curr_keypoints, curr_descriptors = self.orb.detectAndCompute(gray, None)

            if self.prev_descriptors is not None and curr_descriptors is not None:
                # Match features between frames
                matches = self.bf.knnMatch(self.prev_descriptors, curr_descriptors, k=2)

                # Apply Lowe's ratio test
                good_matches = []
                for match_pair in matches:
                    if len(match_pair) == 2:
                        m, n = match_pair
                        if m.distance < 0.7 * n.distance:
                            good_matches.append(m)

                # Draw matches
                if len(good_matches) > 10:  # Only draw if we have enough matches
                    matches_img = cv2.drawMatches(
                        cv2.cvtColor(self.prev_gray, cv2.COLOR_GRAY2BGR) if hasattr(self, 'prev_gray') else np.zeros_like(cv_image),
                        self.prev_keypoints,
                        cv_image,
                        curr_keypoints,
                        good_matches[:50],  # Show top 50 matches
                        None,
                        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
                    )

                    # Calculate motion from feature matches
                    if len(good_matches) >= 4:
                        src_pts = np.float32([self.prev_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                        dst_pts = np.float32([curr_keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                        # Find homography
                        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

                        if H is not None:
                            self.get_logger().info(f'Estimated homography matrix:\n{H}')

                    self.get_logger().info(f'Found {len(good_matches)} good matches')

            # Store current frame for next iteration
            self.prev_keypoints = curr_keypoints
            self.prev_descriptors = curr_descriptors
            self.prev_gray = gray.copy()

        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = FeatureMatcherNode()

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

## Object Detection

### Template Matching

```python
#!/usr/bin/env python3
"""
Template Matching Example
"""
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped

class TemplateMatcherNode(Node):
    def __init__(self):
        super().__init__('template_matcher_node')

        # Create subscriber
        self.image_sub = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10)

        # Create publishers
        self.detection_pub = self.create_publisher(PointStamped, '/object_detection', 10)

        # Initialize CV bridge
        self.bridge = CvBridge()

        # Template to match (this would typically be loaded from file or learned)
        self.template = np.ones((50, 50, 3), dtype=np.uint8) * 255  # White square for example
        self.template_gray = cv2.cvtColor(self.template, cv2.COLOR_BGR2GRAY)

        # Detection parameters
        self.match_threshold = 0.8
        self.method = cv2.TM_CCOEFF_NORMED

        self.get_logger().info('Template matcher node started')

    def image_callback(self, msg):
        """Process incoming camera image and perform template matching"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

            # Perform template matching
            result = cv2.matchTemplate(gray, self.template_gray, self.method)

            # Find locations where matching exceeds threshold
            locations = np.where(result >= self.match_threshold)

            # Draw rectangles around matches
            h, w = self.template_gray.shape
            matched_objects = []

            for pt in zip(*locations[::-1]):  # x, y coordinates
                cv2.rectangle(cv_image, pt, (pt[0] + w, pt[1] + h), (0, 255, 0), 2)

                # Calculate center of detected object
                center_x = pt[0] + w // 2
                center_y = pt[1] + h // 2

                # Create detection message
                detection_msg = PointStamped()
                detection_msg.header = msg.header
                detection_msg.point.x = center_x
                detection_msg.point.y = center_y
                detection_msg.point.z = result[center_y, center_x]  # Confidence value

                matched_objects.append((center_x, center_y))

                # Publish detection
                self.detection_pub.publish(detection_msg)

            # Log detection results
            if matched_objects:
                self.get_logger().info(f'Detected {len(matched_objects)} objects')

        except Exception as e:
            self.get_logger().error(f'Error in template matching: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = TemplateMatcherNode()

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

### Contour Detection

```python
#!/usr/bin/env python3
"""
Contour Detection Example
"""
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from geometry_msgs.msg import PolygonStamped
from geometry_msgs.msg import Point32

class ContourDetectorNode(Node):
    def __init__(self):
        super().__init__('contour_detector_node')

        # Create subscriber
        self.image_sub = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10)

        # Create publishers
        self.contours_pub = self.create_publisher(PolygonStamped, '/detected_contours', 10)

        # Initialize CV bridge
        self.bridge = CvBridge()

        # Contour detection parameters
        self.area_threshold = 100  # Minimum area to consider as object
        self.circularity_threshold = 0.7  # How circular the object is

        self.get_logger().info('Contour detector node started')

    def image_callback(self, msg):
        """Process incoming camera image and detect contours"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

            # Apply threshold to create binary image
            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

            # Apply morphological operations to clean up
            kernel = np.ones((3, 3), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Filter and process contours
            detected_contours = []
            for contour in contours:
                # Calculate area
                area = cv2.contourArea(contour)

                if area > self.area_threshold:
                    # Calculate perimeter and circularity
                    perimeter = cv2.arcLength(contour, True)

                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter * perimeter)

                        # Calculate bounding box
                        x, y, w, h = cv2.boundingRect(contour)

                        # Calculate aspect ratio
                        aspect_ratio = float(w) / h

                        # Draw contour on image
                        cv2.drawContours(cv_image, [contour], -1, (0, 255, 0), 2)
                        cv2.rectangle(cv_image, (x, y), (x + w, y + h), (255, 0, 0), 2)

                        # Calculate center
                        moments = cv2.moments(contour)
                        if moments['m00'] != 0:
                            cx = int(moments['m10'] / moments['m00'])
                            cy = int(moments['m01'] / moments['m00'])

                            # Create contour message
                            contour_msg = PolygonStamped()
                            contour_msg.header = msg.header

                            # Convert contour points to Polygon format
                            for point in contour:
                                p = Point32()
                                p.x = float(point[0][0])
                                p.y = float(point[0][1])
                                p.z = 0.0
                                contour_msg.polygon.points.append(p)

                            # Publish contour
                            self.contours_pub.publish(contour_msg)

                            detected_contours.append({
                                'center': (cx, cy),
                                'area': area,
                                'circularity': circularity,
                                'aspect_ratio': aspect_ratio
                            })

            # Log results
            if detected_contours:
                self.get_logger().info(f'Detected {len(detected_contours)} objects')

        except Exception as e:
            self.get_logger().error(f'Error in contour detection: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = ContourDetectorNode()

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

## Deep Learning for Computer Vision

### YOLO Object Detection Integration

```python
#!/usr/bin/env python3
"""
YOLO Integration Example (Conceptual)
"""
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from geometry_msgs.msg import Point

class YOLODetectorNode(Node):
    def __init__(self):
        super().__init__('yolo_detector_node')

        # Create subscriber
        self.image_sub = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10)

        # Create publishers
        self.detections_pub = self.create_publisher(Detection2DArray, '/yolo_detections', 10)

        # Initialize CV bridge
        self.bridge = CvBridge()

        # YOLO parameters (these would be loaded from model files)
        self.conf_threshold = 0.5
        self.nms_threshold = 0.4
        self.input_width = 416
        self.input_height = 416

        # COCO dataset class names
        self.classes = [
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

        # Load YOLO model (conceptual - in practice, you'd load the actual model)
        self.load_yolo_model()

        self.get_logger().info('YOLO detector node started')

    def load_yolo_model(self):
        """Load YOLO model (conceptual)"""
        # In practice, you would load the actual YOLO model here
        # This could be a PyTorch model, TensorFlow model, or OpenCV DNN
        pass

    def image_callback(self, msg):
        """Process incoming camera image with YOLO detection"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

            # Prepare image for YOLO
            blob = cv2.dnn.blobFromImage(
                cv_image, 1/255.0, (self.input_width, self.input_height), swapRB=True, crop=False)

            # Run inference (conceptual - would use actual model here)
            outputs = self.run_yolo_inference(blob)

            # Process detections
            detections = self.post_process_outputs(outputs, cv_image.shape)

            # Create and publish detection message
            detection_array = Detection2DArray()
            detection_array.header = msg.header

            for det in detections:
                detection = Detection2D()
                detection.header = msg.header

                # Set bounding box
                bbox = detection.bbox
                bbox.center.x = det['x'] + det['width'] / 2
                bbox.center.y = det['y'] + det['height'] / 2
                bbox.size_x = det['width']
                bbox.size_y = det['height']

                # Set classification
                hypothesis = ObjectHypothesisWithPose()
                hypothesis.hypothesis.class_id = det['class_id']
                hypothesis.hypothesis.score = det['confidence']

                detection.results.append(hypothesis)
                detection_array.detections.append(detection)

            # Publish detections
            self.detections_pub.publish(detection_array)

            # Log detection results
            self.get_logger().info(f'YOLO detected {len(detections)} objects')

        except Exception as e:
            self.get_logger().error(f'Error in YOLO detection: {e}')

    def run_yolo_inference(self, blob):
        """Run YOLO inference (conceptual)"""
        # In a real implementation, this would run the actual model
        # For this example, we'll return dummy outputs
        return []

    def post_process_outputs(self, outputs, image_shape):
        """Post-process YOLO outputs"""
        # In a real implementation, this would process the actual model outputs
        # For this example, we'll return dummy detections
        detections = []

        # Example of what the post-processing would do:
        # 1. Extract bounding boxes, confidence scores, and class IDs
        # 2. Apply non-maximum suppression
        # 3. Scale bounding boxes back to original image size
        # 4. Filter by confidence threshold

        return detections

def main(args=None):
    rclpy.init(args=args)
    node = YOLODetectorNode()

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

## Optical Flow

### Dense Optical Flow

```python
#!/usr/bin/env python3
"""
Dense Optical Flow Example
"""
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class DenseOpticalFlowNode(Node):
    def __init__(self):
        super().__init__('dense_optical_flow_node')

        # Create subscriber
        self.image_sub = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10)

        # Initialize CV bridge
        self.bridge = CvBridge()

        # Previous frame for optical flow
        self.prev_frame = None

        # Create Farneback optical flow object
        self.flow = None

        self.get_logger().info('Dense optical flow node started')

    def image_callback(self, msg):
        """Process incoming camera image and compute optical flow"""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

            if self.prev_frame is not None:
                # Calculate dense optical flow
                flow = cv2.calcOpticalFlowFarneback(
                    self.prev_frame, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

                # Visualize optical flow
                flow_image = self.draw_flow(gray, flow)

                # Analyze flow statistics
                self.analyze_flow(flow)

            # Store current frame for next iteration
            self.prev_frame = gray.copy()

        except Exception as e:
            self.get_logger().error(f'Error in optical flow: {e}')

    def draw_flow(self, image, flow, step=16):
        """Draw optical flow vectors on image"""
        h, w = image.shape[:2]
        y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2, -1).astype(int)
        fx, fy = flow[y, x].T

        # Create line endpoints
        lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
        lines = np.int32(lines + 0.5)

        # Create output image
        vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # Draw flow vectors
        cv2.polylines(vis, lines, 0, (0, 255, 0))
        for (x1, y1), (x2, y2) in lines:
            cv2.circle(vis, (x2, y2), 1, (0, 255, 0), -1)

        return vis

    def analyze_flow(self, flow):
        """Analyze optical flow statistics"""
        # Calculate magnitude and angle of flow vectors
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        # Calculate statistics
        mean_magnitude = np.mean(magnitude)
        std_magnitude = np.std(magnitude)
        total_motion = np.sum(magnitude)

        # Log flow statistics
        self.get_logger().info(
            f'Optical flow - Mean mag: {mean_magnitude:.2f}, '
            f'Std dev: {std_magnitude:.2f}, Total motion: {total_motion:.2f}'
        )

def main(args=None):
    rclpy.init(args=args)
    node = DenseOpticalFlowNode()

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

## Stereo Vision

### Stereo Matching

```python
#!/usr/bin/env python3
"""
Stereo Vision Example
"""
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class StereoVisionNode(Node):
    def __init__(self):
        super().__init__('stereo_vision_node')

        # Create subscribers for left and right cameras
        self.left_sub = self.create_subscription(
            Image, '/camera/left/image_raw', self.left_image_callback, 10)
        self.right_sub = self.create_subscription(
            Image, '/camera/right/image_raw', self.right_image_callback, 10)

        # Initialize CV bridge
        self.bridge = CvBridge()

        # Store images
        self.left_image = None
        self.right_image = None

        # Stereo matching parameters
        self.stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)

        self.get_logger().info('Stereo vision node started')

    def left_image_callback(self, msg):
        """Handle left camera image"""
        try:
            self.left_image = self.bridge.imgmsg_to_cv2(msg, 'mono8')
            self.process_stereo()
        except Exception as e:
            self.get_logger().error(f'Error processing left image: {e}')

    def right_image_callback(self, msg):
        """Handle right camera image"""
        try:
            self.right_image = self.bridge.imgmsg_to_cv2(msg, 'mono8')
            self.process_stereo()
        except Exception as e:
            self.get_logger().error(f'Error processing right image: {e}')

    def process_stereo(self):
        """Process stereo images to compute disparity"""
        if self.left_image is not None and self.right_image is not None:
            # Ensure images are the same size
            if self.left_image.shape == self.right_image.shape:
                # Compute disparity
                disparity = self.stereo.compute(
                    self.left_image, self.right_image).astype(np.float32) / 16.0

                # Convert to depth (simplified formula)
                # In practice, you'd use calibrated parameters
                baseline = 0.1  # Baseline distance in meters
                focal_length = 640  # Focal length in pixels (approx)
                depth = (baseline * focal_length) / (disparity + 1e-6)  # Avoid division by zero

                # Log depth statistics
                valid_depths = depth[disparity > 0]
                if len(valid_depths) > 0:
                    avg_depth = np.mean(valid_depths)
                    self.get_logger().info(f'Average depth: {avg_depth:.2f} meters')

                # Reset images after processing
                self.left_image = None
                self.right_image = None

def main(args=None):
    rclpy.init(args=args)
    node = StereoVisionNode()

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

### 1. Performance Optimization

```python
# Good: Efficient image processing pipeline
def efficient_pipeline(image):
    # Resize if too large
    if image.shape[0] > 640 or image.shape[1] > 640:
        scale_factor = min(640.0/image.shape[0], 640.0/image.shape[1])
        new_size = (int(image.shape[1]*scale_factor), int(image.shape[0]*scale_factor))
        image = cv2.resize(image, new_size)

    # Convert to grayscale once
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply minimal preprocessing
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    return gray

# Bad: Inefficient processing
def inefficient_pipeline(image):
    # Multiple color conversions
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # Multiple unnecessary operations
    for _ in range(10):
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

    return gray
```

### 2. Robustness

```python
# Good: Robust error handling
def robust_detection(image):
    if image is None or image.size == 0:
        return None

    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Apply detection algorithm
        features = cv2.goodFeaturesToTrack(gray, maxCorners=100, qualityLevel=0.01, minDistance=10)
        return features
    except Exception as e:
        print(f"Detection failed: {e}")
        return None

# Bad: No error handling
def fragile_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Will crash if image is None
    features = cv2.goodFeaturesToTrack(gray, maxCorners=100, qualityLevel=0.01, minDistance=10)
    return features
```

### 3. Memory Management

```python
# Good: Efficient memory usage
def efficient_memory_usage():
    # Process in-place when possible
    # Use generators for large datasets
    # Free memory when done
    pass

# Bad: Memory leaks
def memory_leaks():
    # Create large arrays without freeing
    # Hold references to large images indefinitely
    pass
```

## Common Issues and Troubleshooting

### 1. Performance Issues

```bash
# Check CPU/GPU usage
top
htop

# Monitor image processing rate
ros2 topic hz /camera/image_raw

# Check for bottlenecks
ros2 run tf2_tools view_frames
```

### 2. Calibration Issues

```python
# Verify calibration parameters
def check_calibration(camera_matrix, dist_coeffs):
    # Check if parameters are reasonable
    if camera_matrix[0, 0] < 100 or camera_matrix[0, 0] > 2000:  # Typical focal length range
        print("Warning: Focal length seems unreasonable")
```

### 3. Feature Detection Issues

```python
# Handle low-texture environments
def handle_low_texture(image):
    # Use different feature detectors for low-texture areas
    # Increase sensitivity parameters
    # Use alternative cues (edges, corners)
    pass
```

## Next Steps

Now that you understand computer vision fundamentals, continue to [Deep Learning Perception](../week-09/deep-learning-perception) to learn about neural networks for robot perception.

## Exercises

1. Implement a feature tracking system for object motion
2. Create a stereo vision pipeline for depth estimation
3. Build a template matching system for object recognition
4. Develop an optical flow analyzer for motion detection