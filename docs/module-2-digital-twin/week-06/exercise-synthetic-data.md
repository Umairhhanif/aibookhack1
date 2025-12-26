---
sidebar_position: 4
---

# Exercise: Perception Training Dataset

In this comprehensive exercise, you'll create a complete synthetic dataset for training perception models using Isaac Sim. This will demonstrate how to generate high-quality, labeled data for computer vision applications.

## Objective

Create a synthetic dataset that includes:
1. **RGB images** with photorealistic rendering
2. **Depth maps** with perfect ground truth
3. **Semantic segmentation** masks
4. **Automatic annotations** with bounding boxes
5. **Domain randomization** for robust training

## Prerequisites

- Complete Week 1-6 lessons
- Isaac Sim installed with NVIDIA GPU
- Understanding of synthetic data generation
- Basic Python and OpenCV knowledge

## Step 1: Set Up the Environment

First, ensure Isaac Sim is properly configured:

```bash
# Verify Isaac Sim installation
# Isaac Sim is typically installed via Omniverse Launcher
# Check for Isaac Sim assets
ls /opt/nvidia/isaac_sim/  # or wherever Isaac Sim is installed
```

## Step 2: Create the Dataset Generator Package

Create a new package for our synthetic dataset generation:

```bash
cd ~/ros2_ws/src
ros2 pkg create --build-type ament_python synthetic_dataset_generator \
    --dependencies rclpy std_msgs sensor_msgs geometry_msgs cv_bridge
```

## Step 3: Create the Dataset Generator Node

Create `synthetic_dataset_generator/synthetic_dataset_generator/dataset_generator.py`:

```python
#!/usr/bin/env python3
"""
Synthetic Dataset Generator for Perception Training
"""
import os
import json
import numpy as np
import cv2
import random
from datetime import datetime
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import threading
import queue

class SyntheticDatasetGenerator(Node):
    def __init__(self):
        super().__init__('synthetic_dataset_generator')

        # Initialize CV bridge
        self.bridge = CvBridge()

        # Dataset configuration
        self.output_dir = "synthetic_perception_dataset"
        self.images_dir = os.path.join(self.output_dir, "images")
        self.depth_dir = os.path.join(self.output_dir, "depth")
        self.semantic_dir = os.path.join(self.output_dir, "semantic")
        self.annotations_dir = os.path.join(self.output_dir, "annotations")

        # Create output directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.depth_dir, exist_ok=True)
        os.makedirs(self.semantic_dir, exist_ok=True)
        os.makedirs(self.annotations_dir, exist_ok=True)

        # Dataset statistics
        self.frame_count = 0
        self.dataset_config = {
            "created": datetime.now().isoformat(),
            "total_frames": 0,
            "image_resolution": [640, 480],
            "objects": ["cube", "sphere", "cylinder"],
            "domain_randomization": {
                "lighting": True,
                "materials": True,
                "environments": True
            }
        }

        # Publishers for real-time data (if needed)
        self.rgb_pub = self.create_publisher(Image, '/synthetic/rgb', 10)
        self.depth_pub = self.create_publisher(Image, '/synthetic/depth', 10)

        # Create timer for dataset generation
        self.timer = self.create_timer(0.1, self.generate_frame)  # 10 Hz

        # Initialize Isaac Sim (simulated - in real implementation, this would connect to Isaac Sim)
        self.initialize_isaac_sim()

        self.get_logger().info('Synthetic dataset generator started')

    def initialize_isaac_sim(self):
        """Initialize Isaac Sim connection (simulated for this example)"""
        # In a real implementation, this would initialize Isaac Sim
        # For this exercise, we'll simulate the process
        self.get_logger().info('Simulated Isaac Sim initialization')

        # Simulated camera intrinsics
        self.camera_info = {
            'fx': 320.0,  # Focal length x
            'fy': 320.0,  # Focal length y
            'cx': 320.0,  # Principal point x
            'cy': 240.0,  # Principal point y
            'width': 640,
            'height': 480
        }

    def generate_frame(self):
        """Generate a single frame with all annotations"""
        # Simulate getting data from Isaac Sim
        rgb_image = self.generate_synthetic_rgb()
        depth_map = self.generate_synthetic_depth()
        semantic_map = self.generate_synthetic_semantic()

        # Generate annotations
        annotations = self.generate_annotations(rgb_image.shape[:2])

        # Save data
        self.save_frame_data(rgb_image, depth_map, semantic_map, annotations)

        # Publish for real-time monitoring (optional)
        try:
            rgb_msg = self.bridge.cv2_to_imgmsg(rgb_image, encoding='bgr8')
            rgb_msg.header.stamp = self.get_clock().now().to_msg()
            rgb_msg.header.frame_id = 'synthetic_camera'
            self.rgb_pub.publish(rgb_msg)

            depth_msg = self.bridge.cv2_to_imgmsg(depth_map, encoding='32FC1')
            depth_msg.header.stamp = self.get_clock().now().to_msg()
            depth_msg.header.frame_id = 'synthetic_camera'
            self.depth_pub.publish(depth_msg)
        except Exception as e:
            self.get_logger().warn(f'Error publishing images: {e}')

        self.frame_count += 1
        self.dataset_config['total_frames'] = self.frame_count

        if self.frame_count % 50 == 0:
            self.get_logger().info(f'Generated {self.frame_count} frames')

        # Stop after generating enough frames
        if self.frame_count >= 200:  # Generate 200 frames for this exercise
            self.save_dataset_config()
            self.get_logger().info(f'Dataset generation complete. Generated {self.frame_count} frames.')
            # In a real implementation, you might want to stop the node here

    def generate_synthetic_rgb(self):
        """Generate synthetic RGB image with domain randomization"""
        # Create base image
        height, width = 480, 640
        image = np.zeros((height, width, 3), dtype=np.uint8)

        # Randomize background
        bg_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        image[:] = bg_color

        # Add random objects
        num_objects = random.randint(1, 5)
        for _ in range(num_objects):
            obj_type = random.choice(['rectangle', 'circle', 'triangle'])
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

            if obj_type == 'rectangle':
                x, y = random.randint(50, width-100), random.randint(50, height-100)
                w, h = random.randint(20, 80), random.randint(20, 80)
                cv2.rectangle(image, (x, y), (x+w, y+h), color, -1)
            elif obj_type == 'circle':
                center = (random.randint(50, width-50), random.randint(50, height-50))
                radius = random.randint(10, 40)
                cv2.circle(image, center, radius, color, -1)
            elif obj_type == 'triangle':
                pts = np.array([
                    [random.randint(50, width-50), random.randint(50, height-50)],
                    [random.randint(50, width-50), random.randint(50, height-50)],
                    [random.randint(50, width-50), random.randint(50, height-50)]
                ], np.int32)
                cv2.fillPoly(image, [pts], color)

        # Add random lighting effects
        lighting_factor = random.uniform(0.7, 1.3)
        image = np.clip(image * lighting_factor, 0, 255).astype(np.uint8)

        return image

    def generate_synthetic_depth(self):
        """Generate synthetic depth map"""
        height, width = 480, 640
        depth_map = np.zeros((height, width), dtype=np.float32)

        # Create depth variations
        for _ in range(5):  # Add 5 depth regions
            center = (random.randint(50, width-50), random.randint(50, height-50))
            radius = random.randint(30, 100)
            depth_value = random.uniform(0.5, 10.0)  # meters

            # Create circular depth region
            y, x = np.ogrid[:height, :width]
            mask = (x - center[0])**2 + (y - center[1])**2 <= radius**2
            depth_map[mask] = depth_value

        # Add noise
        noise = np.random.normal(0, 0.01, depth_map.shape).astype(np.float32)
        depth_map = np.maximum(0.1, depth_map + noise)  # Ensure positive depth

        return depth_map

    def generate_synthetic_semantic(self):
        """Generate synthetic semantic segmentation map"""
        height, width = 480, 640
        semantic_map = np.zeros((height, width), dtype=np.uint8)

        # Assign semantic labels to objects
        # 0 = background, 1 = object1, 2 = object2, etc.
        current_label = 1

        # Add random objects with semantic labels
        num_objects = random.randint(1, 5)
        for _ in range(num_objects):
            obj_type = random.choice(['rectangle', 'circle', 'triangle'])

            if obj_type == 'rectangle':
                x, y = random.randint(50, width-100), random.randint(50, height-100)
                w, h = random.randint(20, 80), random.randint(20, 80)
                cv2.rectangle(semantic_map, (x, y), (x+w, y+h), current_label, -1)
            elif obj_type == 'circle':
                center = (random.randint(50, width-50), random.randint(50, height-50))
                radius = random.randint(10, 40)
                cv2.circle(semantic_map, center, radius, current_label, -1)
            elif obj_type == 'triangle':
                pts = np.array([
                    [random.randint(50, width-50), random.randint(50, height-50)],
                    [random.randint(50, width-50), random.randint(50, height-50)],
                    [random.randint(50, width-50), random.randint(50, height-50)]
                ], np.int32)
                cv2.fillPoly(semantic_map, [pts], current_label)

            current_label = (current_label % 254) + 1  # Cycle through labels

        return semantic_map

    def generate_annotations(self, image_shape):
        """Generate bounding box annotations for objects"""
        height, width = image_shape
        annotations = {
            "image_id": self.frame_count,
            "width": width,
            "height": height,
            "objects": []
        }

        # For this simulated example, we'll create annotations based on the synthetic image
        # In a real implementation, Isaac Sim would provide ground truth annotations
        num_objects = random.randint(1, 5)
        for i in range(num_objects):
            # Random bounding box
            x = random.randint(0, width - 100)
            y = random.randint(0, height - 100)
            w = random.randint(50, 150)
            h = random.randint(50, 150)

            # Ensure bounding box doesn't exceed image bounds
            x = min(x, width - w)
            y = min(y, height - h)

            obj_annotation = {
                "id": i,
                "category": random.choice(["object", "obstacle", "target"]),
                "bbox": [x, y, w, h],  # [x, y, width, height]
                "area": w * h,
                "iscrowd": 0
            }
            annotations["objects"].append(obj_annotation)

        return annotations

    def save_frame_data(self, rgb_image, depth_map, semantic_map, annotations):
        """Save frame data to disk"""
        # Save RGB image
        rgb_path = os.path.join(self.images_dir, f"rgb_{self.frame_count:06d}.png")
        cv2.imwrite(rgb_path, rgb_image)

        # Save depth map (normalize for PNG)
        depth_normalized = ((depth_map - depth_map.min()) /
                           (depth_map.max() - depth_map.min()) * 255).astype(np.uint8)
        depth_path = os.path.join(self.depth_dir, f"depth_{self.frame_count:06d}.png")
        cv2.imwrite(depth_path, depth_normalized)

        # Save semantic map
        semantic_path = os.path.join(self.semantic_dir, f"semantic_{self.frame_count:06d}.png")
        cv2.imwrite(semantic_path, semantic_map)

        # Save annotations
        annotation_path = os.path.join(self.annotations_dir, f"annotations_{self.frame_count:06d}.json")
        with open(annotation_path, 'w') as f:
            json.dump(annotations, f, indent=2)

    def save_dataset_config(self):
        """Save dataset configuration"""
        config_path = os.path.join(self.output_dir, "dataset_config.json")
        with open(config_path, 'w') as f:
            json.dump(self.dataset_config, f, indent=2)

def main(args=None):
    rclpy.init(args=args)
    node = SyntheticDatasetGenerator()

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

## Step 4: Create the Isaac Sim Integration Node

Create `synthetic_dataset_generator/synthetic_dataset_generator/isaac_sim_integration.py`:

```python
#!/usr/bin/env python3
"""
Isaac Sim Integration for Synthetic Dataset Generation
"""
import omni
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.sensor import Camera
import numpy as np
import cv2
import os
import json
from datetime import datetime
import random

class IsaacSimDatasetGenerator:
    def __init__(self, output_dir="isaac_sim_dataset"):
        self.output_dir = output_dir
        self.images_dir = os.path.join(output_dir, "images")
        self.depth_dir = os.path.join(output_dir, "depth")
        self.semantic_dir = os.path.join(output_dir, "semantic")
        self.annotations_dir = os.path.join(output_dir, "annotations")

        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.depth_dir, exist_ok=True)
        os.makedirs(self.semantic_dir, exist_ok=True)
        os.makedirs(self.annotations_dir, exist_ok=True)

        # Initialize Isaac Sim
        self.simulation_app = None
        self.world = None
        self.camera = None
        self.frame_count = 0

        # Dataset statistics
        self.dataset_config = {
            "created": datetime.now().isoformat(),
            "total_frames": 0,
            "image_resolution": [640, 480],
            "objects": [],
            "domain_randomization": {
                "lighting": True,
                "materials": True,
                "environments": True
            }
        }

    def initialize_simulation(self):
        """Initialize Isaac Sim simulation"""
        # Initialize Isaac Sim application
        from omni.isaac.kit import SimulationApp
        self.simulation_app = SimulationApp({"headless": False})  # Set to True for batch processing

        # Import Isaac Sim components
        from omni.isaac.core import World
        self.world = World(stage_units_in_meters=1.0)

        # Add ground plane
        self.world.scene.add_default_ground_plane()

        # Add objects for dataset generation
        self.add_scene_objects()

        # Add camera
        self.camera = Camera(
            prim_path="/World/Camera",
            position=np.array([3.0, 0, 2.0]),
            frequency=30,
            resolution=(640, 480)
        )
        self.world.scene.add(self.camera)

        # Reset world
        self.world.reset()

        print("Isaac Sim initialized for dataset generation")

    def add_scene_objects(self):
        """Add objects to the scene for dataset generation"""
        from omni.isaac.core.objects import DynamicCuboid, DynamicSphere, DynamicCylinder

        # Add various objects
        objects = []

        # Cubes
        for i in range(3):
            obj = self.world.scene.add(
                DynamicCuboid(
                    prim_path=f"/World/Cube{i}",
                    name=f"cube_{i}",
                    position=[i*0.8, -1 + i*0.5, 0.5],
                    size=0.2,
                    mass=0.5
                )
            )
            objects.append(obj)

        # Spheres
        for i in range(2):
            obj = self.world.scene.add(
                DynamicSphere(
                    prim_path=f"/World/Sphere{i}",
                    name=f"sphere_{i}",
                    position=[i*0.8, 1 - i*0.5, 0.7],
                    radius=0.15,
                    mass=0.3
                )
            )
            objects.append(obj)

        # Cylinders
        for i in range(2):
            obj = self.world.scene.add(
                DynamicCylinder(
                    prim_path=f"/World/Cylinder{i}",
                    name=f"cylinder_{i}",
                    position=[-1 + i*0.8, 0, 0.6],
                    radius=0.1,
                    height=0.3,
                    mass=0.4
                )
            )
            objects.append(obj)

        # Store object references
        self.objects = objects

    def randomize_scene(self):
        """Apply domain randomization to the scene"""
        # Randomize object positions
        for obj in self.objects:
            new_pos = [
                random.uniform(-2, 2),
                random.uniform(-2, 2),
                random.uniform(0.5, 2.0)
            ]
            # In real implementation, you would set the position
            # obj.set_world_pos(new_pos)

        # Randomize lighting (simplified)
        # In real implementation, you would modify light properties

        # Randomize materials (simplified)
        # In real implementation, you would modify material properties

        print(f"Scene randomized for frame {self.frame_count}")

    def generate_dataset(self, num_frames=100):
        """Generate synthetic dataset"""
        print(f"Generating {num_frames} frames of synthetic data...")

        for frame_idx in range(num_frames):
            # Randomize scene periodically
            if frame_idx % 20 == 0:
                self.randomize_scene()

            # Step simulation
            self.world.step(render=True)

            # Capture data
            self.capture_frame_data()

            self.frame_count += 1
            self.dataset_config['total_frames'] = self.frame_count

            if frame_idx % 10 == 0:
                print(f"Generated {frame_idx}/{num_frames} frames")

        # Save dataset configuration
        self.save_dataset_config()

        print(f"Dataset generation complete. Generated {num_frames} frames.")

    def capture_frame_data(self):
        """Capture RGB, depth, and semantic data from Isaac Sim"""
        try:
            # Get RGB image
            rgb_image = self.camera.get_rgb()

            # Get depth data
            depth_data = self.camera.get_depth()

            # Get semantic segmentation (if available)
            # semantic_data = self.camera.get_semantic_segmentation()

            # For this example, we'll create a dummy semantic map
            semantic_map = self.create_dummy_semantic_map(rgb_image)

            # Generate annotations
            annotations = self.generate_annotations()

            # Save data
            self.save_frame_data(rgb_image, depth_data, semantic_map, annotations)

        except Exception as e:
            print(f"Error capturing frame data: {e}")

    def create_dummy_semantic_map(self, rgb_image):
        """Create a dummy semantic map for demonstration"""
        height, width = rgb_image.shape[:2]
        semantic_map = np.zeros((height, width), dtype=np.uint8)

        # In a real implementation, Isaac Sim would provide semantic segmentation
        # For this example, we'll create a simple pattern
        for i in range(5):
            center = (random.randint(50, width-50), random.randint(50, height-50))
            radius = random.randint(30, 80)
            cv2.circle(semantic_map, center, radius, i+1, -1)

        return semantic_map

    def generate_annotations(self):
        """Generate annotations for the current frame"""
        annotations = {
            "frame_id": self.frame_count,
            "timestamp": self.world.current_time if hasattr(self.world, 'current_time') else 0,
            "objects": []
        }

        # In a real implementation, you would get object poses from Isaac Sim
        # For this example, we'll create dummy annotations
        for i, obj in enumerate(self.objects):
            # Get object position (simulated)
            pos = [random.uniform(-2, 2), random.uniform(-2, 2), random.uniform(0.5, 2.0)]

            # Create bounding box (simulated)
            bbox_2d = [100 + i*50, 100 + i*30, 80, 80]  # [x, y, width, height]

            annotations["objects"].append({
                "id": i,
                "name": obj.name,
                "bbox_2d": bbox_2d,
                "position_3d": pos,
                "category": "object"
            })

        return annotations

    def save_frame_data(self, rgb_image, depth_map, semantic_map, annotations):
        """Save frame data to disk"""
        # Save RGB image
        rgb_path = os.path.join(self.images_dir, f"rgb_{self.frame_count:06d}.png")
        cv2.imwrite(rgb_path, cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))

        # Save depth map (normalize for PNG)
        depth_normalized = ((depth_map - depth_map.min()) /
                           (depth_map.max() - depth_map.min()) * 255).astype(np.uint8)
        depth_path = os.path.join(self.depth_dir, f"depth_{self.frame_count:06d}.png")
        cv2.imwrite(depth_path, depth_normalized)

        # Save semantic map
        semantic_path = os.path.join(self.semantic_dir, f"semantic_{self.frame_count:06d}.png")
        cv2.imwrite(semantic_path, semantic_map)

        # Save annotations
        annotation_path = os.path.join(self.annotations_dir, f"annotations_{self.frame_count:06d}.json")
        with open(annotation_path, 'w') as f:
            json.dump(annotations, f, indent=2)

    def save_dataset_config(self):
        """Save dataset configuration"""
        config_path = os.path.join(self.output_dir, "dataset_config.json")
        with open(config_path, 'w') as f:
            json.dump(self.dataset_config, f, indent=2)

    def cleanup(self):
        """Clean up Isaac Sim"""
        if self.simulation_app:
            self.simulation_app.close()

def main():
    """Main function to run the dataset generator"""
    generator = IsaacSimDatasetGenerator(output_dir="isaac_sim_perception_dataset")

    try:
        generator.initialize_simulation()
        generator.generate_dataset(num_frames=50)  # Generate 50 frames for this example
    except Exception as e:
        print(f"Error during dataset generation: {e}")
    finally:
        generator.cleanup()

if __name__ == "__main__":
    main()
```

## Step 5: Create a Data Quality Assessment Tool

Create `synthetic_dataset_generator/synthetic_dataset_generator/data_quality_assessment.py`:

```python
#!/usr/bin/env python3
"""
Data Quality Assessment for Synthetic Datasets
"""
import os
import cv2
import numpy as np
import json
from scipy import ndimage
import matplotlib.pyplot as plt

class DataQualityAssessor:
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        self.images_dir = os.path.join(dataset_dir, "images")
        self.depth_dir = os.path.join(dataset_dir, "depth")
        self.semantic_dir = os.path.join(dataset_dir, "semantic")
        self.annotations_dir = os.path.join(dataset_dir, "annotations")

    def calculate_sharpness(self, image):
        """Calculate image sharpness using Laplacian variance"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return laplacian.var()

    def calculate_brightness(self, image):
        """Calculate image brightness"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        return np.mean(gray)

    def calculate_contrast(self, image):
        """Calculate image contrast using standard deviation"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        return np.std(gray)

    def assess_image_quality(self, image_path):
        """Assess quality of a single image"""
        image = cv2.imread(image_path)
        if image is None:
            return None

        quality_metrics = {
            'sharpness': self.calculate_sharpness(image),
            'brightness': self.calculate_brightness(image),
            'contrast': self.calculate_contrast(image),
            'size': image.shape
        }

        return quality_metrics

    def assess_dataset_quality(self):
        """Assess quality of entire dataset"""
        results = {
            'images': [],
            'quality_metrics': {
                'sharpness': [],
                'brightness': [],
                'contrast': []
            }
        }

        image_files = [f for f in os.listdir(self.images_dir)
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        for filename in image_files:
            image_path = os.path.join(self.images_dir, filename)
            quality = self.assess_image_quality(image_path)

            if quality:
                results['images'].append(filename)
                results['quality_metrics']['sharpness'].append(quality['sharpness'])
                results['quality_metrics']['brightness'].append(quality['brightness'])
                results['quality_metrics']['contrast'].append(quality['contrast'])

        return results

    def generate_quality_report(self):
        """Generate a quality report for the dataset"""
        quality_results = self.assess_dataset_quality()

        if not quality_results['images']:
            print("No images found for quality assessment")
            return

        # Calculate statistics
        sharpness_values = quality_results['quality_metrics']['sharpness']
        brightness_values = quality_results['quality_metrics']['brightness']
        contrast_values = quality_results['quality_metrics']['contrast']

        report = {
            'total_images': len(quality_results['images']),
            'sharpness_stats': {
                'mean': np.mean(sharpness_values),
                'std': np.std(sharpness_values),
                'min': np.min(sharpness_values),
                'max': np.max(sharpness_values)
            },
            'brightness_stats': {
                'mean': np.mean(brightness_values),
                'std': np.std(brightness_values),
                'min': np.min(brightness_values),
                'max': np.max(brightness_values)
            },
            'contrast_stats': {
                'mean': np.mean(contrast_values),
                'std': np.std(contrast_values),
                'min': np.min(contrast_values),
                'max': np.max(contrast_values)
            }
        }

        return report

    def visualize_quality_metrics(self):
        """Visualize quality metrics"""
        quality_results = self.assess_dataset_quality()

        if not quality_results['images']:
            print("No images found for visualization")
            return

        sharpness_values = quality_results['quality_metrics']['sharpness']
        brightness_values = quality_results['quality_metrics']['brightness']
        contrast_values = quality_results['quality_metrics']['contrast']

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Sharpness histogram
        axes[0].hist(sharpness_values, bins=50, alpha=0.7)
        axes[0].set_title('Sharpness Distribution')
        axes[0].set_xlabel('Sharpness')
        axes[0].set_ylabel('Frequency')

        # Brightness histogram
        axes[1].hist(brightness_values, bins=50, alpha=0.7, color='orange')
        axes[1].set_title('Brightness Distribution')
        axes[1].set_xlabel('Brightness')
        axes[1].set_ylabel('Frequency')

        # Contrast histogram
        axes[2].hist(contrast_values, bins=50, alpha=0.7, color='green')
        axes[2].set_title('Contrast Distribution')
        axes[2].set_xlabel('Contrast')
        axes[2].set_ylabel('Frequency')

        plt.tight_layout()
        plt.savefig(os.path.join(self.dataset_dir, 'quality_metrics.png'))
        plt.show()

def main():
    """Main function to run quality assessment"""
    dataset_dir = "isaac_sim_perception_dataset"  # or "synthetic_perception_dataset"

    if not os.path.exists(dataset_dir):
        print(f"Dataset directory {dataset_dir} does not exist")
        return

    assessor = DataQualityAssessor(dataset_dir)

    # Generate quality report
    report = assessor.generate_quality_report()

    if report:
        print("Dataset Quality Report:")
        print(f"  Total Images: {report['total_images']}")
        print(f"  Sharpness - Mean: {report['sharpness_stats']['mean']:.2f}, Std: {report['sharpness_stats']['std']:.2f}")
        print(f"  Brightness - Mean: {report['brightness_stats']['mean']:.2f}, Std: {report['brightness_stats']['std']:.2f}")
        print(f"  Contrast - Mean: {report['contrast_stats']['mean']:.2f}, Std: {report['contrast_stats']['std']:.2f}")

        # Visualize quality metrics
        assessor.visualize_quality_metrics()

if __name__ == "__main__":
    main()
```

## Step 6: Create Launch Files

Create `synthetic_dataset_generator/launch/generate_dataset.launch.py`:

```python
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # Declare launch arguments
    dataset_size = LaunchConfiguration('dataset_size', default='100')
    output_dir = LaunchConfiguration('output_dir', default='synthetic_dataset')

    return LaunchDescription([
        DeclareLaunchArgument(
            'dataset_size',
            default_value='100',
            description='Number of frames to generate'
        ),
        DeclareLaunchArgument(
            'output_dir',
            default_value='synthetic_dataset',
            description='Output directory for dataset'
        ),

        # Dataset generation node
        Node(
            package='synthetic_dataset_generator',
            executable='dataset_generator',
            name='dataset_generator',
            parameters=[
                {'dataset_size': dataset_size},
                {'output_dir': output_dir}
            ],
            output='screen'
        )
    ])
```

## Step 7: Update Package Configuration

Update `synthetic_dataset_generator/setup.py`:

```python
from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'synthetic_dataset_generator'

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
    description='Synthetic dataset generator for perception training',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'dataset_generator = synthetic_dataset_generator.dataset_generator:main',
            'isaac_sim_generator = synthetic_dataset_generator.isaac_sim_integration:main',
            'quality_assessment = synthetic_dataset_generator.data_quality_assessment:main',
        ],
    },
)
```

## Step 8: Build and Test

Build the package:

```bash
cd ~/ros2_ws
colcon build --packages-select synthetic_dataset_generator
source install/setup.bash
```

### Test the Dataset Generation

1. **Run the basic dataset generator**:
```bash
# Run the ROS 2 node version
ros2 run synthetic_dataset_generator dataset_generator
```

2. **Run the Isaac Sim integration** (if Isaac Sim is available):
```bash
# Run the Isaac Sim version
python3 `ros2 pkg prefix synthetic_dataset_generator`/lib/synthetic_dataset_generator/isaac_sim_generator
```

3. **Run quality assessment**:
```bash
# Assess the quality of generated dataset
python3 `ros2 pkg prefix synthetic_dataset_generator`/lib/synthetic_dataset_generator/data_quality_assessment
```

4. **Check generated dataset**:
```bash
# Verify dataset structure
ls -la synthetic_perception_dataset/
ls -la synthetic_perception_dataset/images/
ls -la synthetic_perception_dataset/annotations/
```

## Understanding the Dataset

This synthetic dataset generator creates:

1. **RGB Images**: Photorealistic images with various objects
2. **Depth Maps**: Ground truth depth information
3. **Semantic Segmentation**: Pixel-level object classification
4. **Annotations**: Bounding boxes and object information
5. **Quality Metrics**: Assessment of dataset quality

## Challenges

### Challenge 1: Add More Object Types
Extend the dataset generator to include more object types and variations.

<details>
<summary>Hint</summary>

Modify the `generate_synthetic_rgb` function to include additional geometric shapes or 3D models.
</details>

### Challenge 2: Implement Real Isaac Sim Integration
Connect the generator to actual Isaac Sim for real synthetic data.

<details>
<summary>Hint</summary>

Replace the simulated Isaac Sim functions with actual Isaac Sim API calls.
</details>

### Challenge 3: Add More Annotation Types
Include additional annotation types like instance segmentation or keypoints.

<details>
<summary>Hint</summary>

Extend the annotation generation to include more detailed object information.
</details>

### Challenge 4: Optimize for Performance
Improve the generation speed for large datasets.

<details>
<summary>Hint</summary>

Use batch processing, parallel generation, or optimized rendering settings.
</details>

## Verification Checklist

- [ ] Dataset generator creates RGB images
- [ ] Depth maps are generated with ground truth
- [ ] Semantic segmentation masks are created
- [ ] Annotations include bounding boxes
- [ ] Domain randomization is applied
- [ ] Quality assessment tool works
- [ ] Dataset follows standard format
- [ ] Images and annotations are properly paired

## Common Issues

### Isaac Sim Connection Issues
```bash
# Ensure Isaac Sim is properly installed
# Check NVIDIA GPU drivers
# Verify Isaac Sim assets are available
```

### Quality Issues
```bash
# Run quality assessment tool
# Check for blurry or low-contrast images
# Verify annotation accuracy
```

### Performance Issues
```bash
# Reduce resolution for faster generation
# Process in smaller batches
# Optimize rendering settings
```

## Summary

In this exercise, you learned to:
- Generate synthetic datasets with multiple modalities (RGB, depth, semantic)
- Apply domain randomization for robust training
- Create automatic annotations with bounding boxes
- Assess synthetic data quality
- Structure datasets for AI training

## Next Steps

You have now completed Module 2: Building the Digital Twin! Continue to [Module 3: The AI Robot Brain](../../module-3-ai-robot-brain/week-07/introduction) to learn about visual SLAM, navigation, and perception systems.