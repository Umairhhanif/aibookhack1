---
sidebar_position: 4
---

# Exercise: Navigation System Implementation

In this comprehensive exercise, you'll create a complete navigation system that integrates path planning, motion control, and obstacle avoidance. This will demonstrate practical implementation of Nav2 concepts.

## Objective

Create a navigation system that:
1. **Plans paths** using global and local planners
2. **Controls robot motion** with smooth path following
3. **Avoids obstacles** with local planning and recovery
4. **Integrates perception** for safe navigation

## Prerequisites

- Complete Week 1-8 lessons
- ROS 2 workspace set up (`~/ros2_ws`)
- Understanding of Nav2 architecture
- Basic Python and C++ programming skills

## Step 1: Create the Navigation Package

```bash
cd ~/ros2_ws/src

ros2 pkg create --build-type ament_python navigation_system \
    --dependencies rclpy std_msgs sensor_msgs geometry_msgs nav_msgs visualization_msgs tf2_ros cv_bridge
```

## Step 2: Create the Global Planner Node

Create `navigation_system/navigation_system/global_planner_node.py`:

```python
#!/usr/bin/env python3
"""
Global Path Planner Node
"""
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path, OccupancyGrid
from geometry_msgs.msg import PoseStamped, Point
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Header
import numpy as np
import heapq
import math

class GlobalPlannerNode(Node):
    def __init__(self):
        super().__init__('global_planner_node')

        # Create subscribers
        self.map_sub = self.create_subscription(
            OccupancyGrid, '/map', self.map_callback, 10)
        self.goal_sub = self.create_subscription(
            PoseStamped, '/move_base_simple/goal', self.goal_callback, 10)

        # Create publishers
        self.path_pub = self.create_publisher(Path, '/global_plan', 10)
        self.path_marker_pub = self.create_publisher(Marker, '/global_plan_marker', 10)

        # Store map data
        self.map_data = None
        self.map_info = None
        self.current_goal = None
        self.map_resolution = 0.05

        # A* algorithm parameters
        self.inflation_radius = 0.3  # meters

        self.get_logger().info('Global planner node started')

    def map_callback(self, msg):
        """Handle map updates"""
        self.map_data = np.array(msg.data).reshape(msg.info.height, msg.info.width)
        self.map_info = msg.info
        self.map_resolution = msg.info.resolution

        self.get_logger().info(f'Map received: {msg.info.width}x{msg.info.height}, resolution: {self.map_resolution}')

    def goal_callback(self, msg):
        """Handle navigation goal"""
        if self.map_data is None:
            self.get_logger().warn('No map available, cannot plan path')
            return

        self.current_goal = msg.pose
        start_pose = self.get_current_pose()

        if start_pose is not None:
            # Plan path from current pose to goal
            path = self.plan_path(start_pose, self.current_goal)
            if path:
                self.publish_path(path)
                self.publish_path_marker(path)
                self.get_logger().info(f'Global path planned with {len(path.poses)} waypoints')
            else:
                self.get_logger().warn('Failed to find path to goal')

    def get_current_pose(self):
        """Get current robot pose (simulated)"""
        # In a real system, this would come from localization
        # For this exercise, we'll simulate the current pose
        from geometry_msgs.msg import Pose
        current_pose = Pose()
        current_pose.position.x = 0.0
        current_pose.position.y = 0.0
        current_pose.position.z = 0.0
        current_pose.orientation.w = 1.0
        return current_pose

    def plan_path(self, start, goal):
        """Plan path using A* algorithm"""
        path = Path()
        path.header.frame_id = 'map'

        # Convert world coordinates to map coordinates
        start_map = self.world_to_map(start.position.x, start.position.y)
        goal_map = self.world_to_map(goal.position.x, goal.position.y)

        if start_map is None or goal_map is None:
            return path

        # Run A* algorithm
        path_points = self.a_star_search(start_map, goal_map)

        if path_points:
            # Convert path back to world coordinates
            for point in path_points:
                world_pose = self.map_to_world_pose(point)
                path.poses.append(world_pose)

        return path

    def a_star_search(self, start, goal):
        """A* pathfinding algorithm with obstacle inflation"""
        if self.map_data is None:
            return []

        height, width = self.map_data.shape

        # Check if start and goal are valid
        if not self.is_valid(start[0], start[1], width, height) or \
           not self.is_valid(goal[0], goal[1], width, height):
            return []

        # Check if start or goal are in obstacles
        if self.is_occupied(start[0], start[1]) or self.is_occupied(goal[0], goal[1]):
            self.get_logger().warn('Start or goal position is in an obstacle')
            return []

        # A* algorithm
        open_set = [(0, start)]
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}

        while open_set:
            current = heapq.heappop(open_set)[1]

            if current == goal:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()
                return path

            for neighbor in self.get_neighbors(current, width, height):
                if self.is_occupied(neighbor[0], neighbor[1]):
                    continue

                tentative_g_score = g_score[current] + self.distance(current, neighbor)

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = g_score[neighbor] + self.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return []  # No path found

    def heuristic(self, a, b):
        """Heuristic function (Euclidean distance)"""
        return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

    def get_neighbors(self, pos, width, height):
        """Get valid neighbors for a position (8-connected)"""
        neighbors = []
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
            nx, ny = pos[0] + dx, pos[1] + dy
            if self.is_valid(nx, ny, width, height):
                neighbors.append((nx, ny))
        return neighbors

    def is_valid(self, x, y, width, height):
        """Check if coordinates are valid"""
        return 0 <= x < width and 0 <= y < height

    def is_occupied(self, x, y):
        """Check if cell is occupied (with inflation)"""
        if not self.is_valid(x, y, self.map_info.width, self.map_info.height):
            return True

        # Check the cell itself
        cell_value = self.map_data[y, x]
        if cell_value > 50:  # Threshold for obstacle
            return True

        # Check surrounding cells for inflated obstacle
        inflation_cells = int(self.inflation_radius / self.map_info.resolution)
        for dx in range(-inflation_cells, inflation_cells + 1):
            for dy in range(-inflation_cells, inflation_cells + 1):
                nx, ny = x + dx, y + dy
                if self.is_valid(nx, ny, self.map_info.width, self.map_info.height):
                    distance = math.sqrt(dx*dx + dy*dy) * self.map_info.resolution
                    if distance <= self.inflation_radius:
                        cell_value = self.map_data[ny, nx]
                        if cell_value > 50:
                            return True

        return False

    def distance(self, a, b):
        """Calculate distance between two points"""
        return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

    def world_to_map(self, x, y):
        """Convert world coordinates to map coordinates"""
        if self.map_info is None:
            return None

        map_x = int((x - self.map_info.origin.position.x) / self.map_info.resolution)
        map_y = int((y - self.map_info.origin.position.y) / self.map_info.resolution)

        if 0 <= map_x < self.map_info.width and 0 <= map_y < self.map_info.height:
            return (map_x, map_y)
        else:
            return None

    def map_to_world_pose(self, map_coords):
        """Convert map coordinates to world pose"""
        map_x, map_y = map_coords

        world_x = map_x * self.map_info.resolution + self.map_info.origin.position.x
        world_y = map_y * self.map_info.resolution + self.map_info.origin.position.y

        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.pose.position.x = world_x
        pose.pose.position.y = world_y
        pose.pose.position.z = 0.0
        pose.pose.orientation.w = 1.0

        return pose

    def publish_path(self, path):
        """Publish the planned path"""
        path.header.stamp = self.get_clock().now().to_msg()
        self.path_pub.publish(path)

    def publish_path_marker(self, path):
        """Publish path as visualization marker"""
        if len(path.poses) == 0:
            return

        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'global_path'
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD

        # Set the scale of the marker
        marker.scale.x = 0.05  # Line width

        # Set the color (green)
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        # Add points to the line strip
        for pose in path.poses:
            point = Point()
            point.x = pose.pose.position.x
            point.y = pose.pose.position.y
            point.z = 0.05  # Slightly above ground for visibility
            marker.points.append(point)

        self.path_marker_pub.publish(marker)

def main(args=None):
    rclpy.init(args=args)
    node = GlobalPlannerNode()

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

## Step 3: Create the Local Planner Node

Create `navigation_system/navigation_system/local_planner_node.py`:

```python
#!/usr/bin/env python3
"""
Local Path Planner Node
"""
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped, PoseWithCovarianceStamped
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Path
from visualization_msgs.msg import Marker
from std_msgs.msg import Header
import numpy as np
import math

class LocalPlannerNode(Node):
    def __init__(self):
        super().__init__('local_planner_node')

        # Create subscribers
        self.global_plan_sub = self.create_subscription(
            Path, '/global_plan', self.global_plan_callback, 10)
        self.laser_sub = self.create_subscription(
            LaserScan, '/scan', self.laser_callback, 10)
        self.odom_sub = self.create_subscription(
            PoseWithCovarianceStamped, '/amcl_pose', self.odom_callback, 10)

        # Create publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.local_plan_pub = self.create_publisher(Path, '/local_plan', 10)
        self.velocity_marker_pub = self.create_publisher(Marker, '/velocity_vector', 10)

        # Navigation state
        self.global_plan = None
        self.laser_data = None
        self.current_pose = None
        self.current_velocity = Twist()
        self.current_goal_idx = 0

        # Local planner parameters
        self.lookahead_distance = 1.0  # meters
        self.max_linear_speed = 0.5    # m/s
        self.max_angular_speed = 1.0   # rad/s
        self.min_linear_speed = 0.1    # m/s
        self.control_frequency = 10.0  # Hz
        self.obstacle_threshold = 0.5  # meters

        # Control timer
        self.control_timer = self.create_timer(1.0/self.control_frequency, self.control_loop)

        self.get_logger().info('Local planner node started')

    def global_plan_callback(self, msg):
        """Handle global plan updates"""
        self.global_plan = msg
        self.current_goal_idx = 0  # Reset to beginning of plan
        self.get_logger().info(f'New global plan received with {len(msg.poses)} waypoints')

    def laser_callback(self, msg):
        """Handle laser scan data"""
        self.laser_data = msg

    def odom_callback(self, msg):
        """Handle odometry updates"""
        self.current_pose = msg.pose.pose

    def control_loop(self):
        """Main control loop"""
        if self.current_pose is None or self.global_plan is None or len(self.global_plan.poses) == 0:
            # Stop robot if no data available
            self.stop_robot()
            return

        # Get velocity command
        cmd_vel = self.compute_velocity_command()

        # Check for obstacles
        if self.detect_obstacles():
            # Reduce speed or stop if obstacles detected
            cmd_vel.linear.x *= 0.5  # Reduce linear speed
            cmd_vel.angular.z *= 0.8  # Reduce angular speed

        # Publish command
        self.cmd_vel_pub.publish(cmd_vel)

        # Publish velocity marker for visualization
        self.publish_velocity_marker(cmd_vel)

        # Update current velocity (for next iteration)
        self.current_velocity = cmd_vel

    def compute_velocity_command(self):
        """Compute velocity command using pure pursuit algorithm"""
        cmd_vel = Twist()

        # Find current goal point on global plan
        goal_point = self.find_goal_point()
        if goal_point is None:
            return cmd_vel  # Stop if no goal point found

        # Calculate distance to goal point
        current_pos = np.array([self.current_pose.position.x, self.current_pose.position.y])
        goal_pos = np.array([goal_point.pose.position.x, goal_point.pose.position.y])
        distance = np.linalg.norm(goal_pos - current_pos)

        # Calculate angle to goal point
        current_yaw = self.get_yaw_from_pose(self.current_pose)
        dx = goal_pos[0] - current_pos[0]
        dy = goal_pos[1] - current_pos[1]
        angle_to_target = math.atan2(dy, dx)
        angle_error = self.normalize_angle(angle_to_target - current_yaw)

        # Pure pursuit control
        curvature = 2 * math.sin(angle_error) / max(distance, 0.1)  # Avoid division by zero

        # Set velocities
        cmd_vel.linear.x = min(self.max_linear_speed, max(self.min_linear_speed,
                     self.max_linear_speed * (1 - abs(angle_error)/math.pi)))
        cmd_vel.angular.z = cmd_vel.linear.x * curvature

        # Limit angular velocity
        cmd_vel.angular.z = max(-self.max_angular_speed, min(self.max_angular_speed, cmd_vel.angular.z))

        return cmd_vel

    def find_goal_point(self):
        """Find the goal point on global plan using lookahead distance"""
        if self.global_plan is None or len(self.global_plan.poses) == 0:
            return None

        current_pos = np.array([self.current_pose.position.x, self.current_pose.position.y])

        # Find the point on the path that is approximately lookahead_distance away
        for i in range(self.current_goal_idx, len(self.global_plan.poses)):
            path_pos = np.array([self.global_plan.poses[i].pose.position.x,
                                self.global_plan.poses[i].pose.position.y])
            distance = np.linalg.norm(path_pos - current_pos)

            if distance >= self.lookahead_distance:
                self.current_goal_idx = i  # Update current goal index
                return self.global_plan.poses[i]

        # If no point is far enough, return the last point
        if len(self.global_plan.poses) > 0:
            self.current_goal_idx = len(self.global_plan.poses) - 1
            return self.global_plan.poses[-1]

        return None

    def detect_obstacles(self):
        """Detect obstacles in laser scan data"""
        if self.laser_data is None:
            return False

        # Check for obstacles within threshold distance
        for i, range_val in enumerate(self.laser_data.ranges):
            if not math.isinf(range_val) and not math.isnan(range_val):
                if range_val < self.obstacle_threshold:
                    return True

        return False

    def get_yaw_from_pose(self, pose):
        """Extract yaw from pose orientation"""
        quat = pose.orientation
        siny_cosp = 2 * (quat.w * quat.z + quat.x * quat.y)
        cosy_cosp = 1 - 2 * (quat.y * quat.y + quat.z * quat.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def normalize_angle(self, angle):
        """Normalize angle to [-π, π]"""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

    def stop_robot(self):
        """Stop the robot"""
        cmd_vel = Twist()
        self.cmd_vel_pub.publish(cmd_vel)

    def publish_velocity_marker(self, cmd_vel):
        """Publish velocity vector as visualization marker"""
        marker = Marker()
        marker.header.frame_id = 'base_link'  # Robot's coordinate frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'velocity_vector'
        marker.id = 0
        marker.type = Marker.ARROW
        marker.action = Marker.ADD

        # Set the scale of the arrow
        marker.scale.x = abs(cmd_vel.linear.x) * 2.0  # Length proportional to linear velocity
        marker.scale.y = 0.05  # Width
        marker.scale.z = 0.05  # Height

        # Set the color (blue for velocity)
        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 1.0
        marker.color.a = 1.0

        # Set the position (at robot's position)
        marker.pose.position.x = 0.0
        marker.pose.position.y = 0.0
        marker.pose.position.z = 0.1  # Slightly above robot

        # Set the orientation (pointing in direction of movement)
        marker.pose.orientation.w = 1.0

        self.velocity_marker_pub.publish(marker)

def main(args=None):
    rclpy.init(args=args)
    node = LocalPlannerNode()

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

## Step 4: Create the Recovery Node

Create `navigation_system/navigation_system/recovery_node.py`:

```python
#!/usr/bin/env python3
"""
Recovery Behavior Node
"""
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool
import math
import time

class RecoveryNode(Node):
    def __init__(self):
        super().__init__('recovery_node')

        # Create subscribers
        self.cmd_vel_sub = self.create_subscription(
            Twist, '/cmd_vel', self.cmd_vel_callback, 10)
        self.laser_sub = self.create_subscription(
            LaserScan, '/scan', self.laser_callback, 10)
        self.stuck_sub = self.create_subscription(
            Bool, '/nav_stuck', self.stuck_callback, 10)

        # Create publishers
        self.recovery_cmd_pub = self.create_publisher(Twist, '/recovery_cmd_vel', 10)
        self.recovery_status_pub = self.create_publisher(Bool, '/recovery_active', 10)

        # Recovery state
        self.last_cmd_vel = Twist()
        self.laser_data = None
        self.is_stuck = False
        self.recovery_active = False
        self.recovery_start_time = None
        self.current_recovery_behavior = None

        # Recovery parameters
        self.stuck_threshold = 0.05  # Robot is stuck if moving slower than this
        self.recovery_timeout = 10.0  # Seconds before recovery fails
        self.min_obstacle_distance = 0.3  # Minimum distance to obstacles

        # Recovery behaviors
        self.recovery_behaviors = [
            self.spin_recovery,
            self.backup_recovery,
            self.wander_recovery
        ]
        self.current_behavior_idx = 0

        self.get_logger().info('Recovery node started')

    def cmd_vel_callback(self, msg):
        """Monitor command velocities"""
        self.last_cmd_vel = msg

    def laser_callback(self, msg):
        """Handle laser scan data"""
        self.laser_data = msg

    def stuck_callback(self, msg):
        """Handle stuck detection"""
        self.is_stuck = msg.data
        if self.is_stuck and not self.recovery_active:
            self.start_recovery()

    def start_recovery(self):
        """Start recovery behavior"""
        self.recovery_active = True
        self.recovery_start_time = self.get_clock().now().nanoseconds * 1e-9
        self.current_behavior_idx = 0
        self.current_recovery_behavior = self.recovery_behaviors[0]

        self.get_logger().info('Starting recovery behavior')

        # Publish recovery status
        status_msg = Bool()
        status_msg.data = True
        self.recovery_status_pub.publish(status_msg)

    def execute_recovery(self):
        """Execute current recovery behavior"""
        if not self.recovery_active or self.current_recovery_behavior is None:
            return

        # Check if recovery timed out
        current_time = self.get_clock().now().nanoseconds * 1e-9
        if current_time - self.recovery_start_time > self.recovery_timeout:
            self.get_logger().warn('Recovery timed out, trying next behavior')
            self.try_next_behavior()
            return

        # Execute current behavior
        cmd_vel = self.current_recovery_behavior()

        # Publish recovery command
        self.recovery_cmd_pub.publish(cmd_vel)

        # Check if recovery is successful
        if self.is_clear_path():
            self.get_logger().info('Recovery successful')
            self.stop_recovery()

    def spin_recovery(self):
        """Spin in place to clear local minima"""
        cmd_vel = Twist()
        cmd_vel.angular.z = 0.5  # Spin at 0.5 rad/s
        return cmd_vel

    def backup_recovery(self):
        """Move backward to get unstuck"""
        cmd_vel = Twist()
        cmd_vel.linear.x = -0.2  # Move backward at 0.2 m/s
        return cmd_vel

    def wander_recovery(self):
        """Random wandering to escape local minima"""
        cmd_vel = Twist()
        cmd_vel.linear.x = 0.1  # Move forward slowly
        cmd_vel.angular.z = (2 * (self.get_clock().now().nanoseconds % 100) / 100.0 - 1) * 0.5  # Random angular velocity
        return cmd_vel

    def try_next_behavior(self):
        """Try the next recovery behavior"""
        self.current_behavior_idx += 1
        if self.current_behavior_idx < len(self.recovery_behaviors):
            self.current_recovery_behavior = self.recovery_behaviors[self.current_behavior_idx]
            self.recovery_start_time = self.get_clock().now().nanoseconds * 1e-9
            self.get_logger().info(f'Trying recovery behavior {self.current_behavior_idx + 1}')
        else:
            self.get_logger().warn('All recovery behaviors failed')
            self.stop_recovery()

    def is_clear_path(self):
        """Check if path ahead is clear"""
        if self.laser_data is None:
            return False

        # Check forward sector (between -30 and 30 degrees)
        angle_min = self.laser_data.angle_min
        angle_increment = self.laser_data.angle_increment

        forward_sector = []
        for i, range_val in enumerate(self.laser_data.ranges):
            angle = angle_min + i * angle_increment
            if -math.pi/6 <= angle <= math.pi/6:  # -30 to 30 degrees
                if not math.isinf(range_val) and not math.isnan(range_val):
                    forward_sector.append(range_val)

        if forward_sector:
            min_distance = min(forward_sector)
            return min_distance > self.min_obstacle_distance

        return False

    def stop_recovery(self):
        """Stop recovery behavior"""
        self.recovery_active = False
        self.current_recovery_behavior = None

        # Publish stop command
        stop_cmd = Twist()
        self.recovery_cmd_pub.publish(stop_cmd)

        # Publish recovery status
        status_msg = Bool()
        status_msg.data = False
        self.recovery_status_pub.publish(status_msg)

        self.get_logger().info('Recovery stopped')

def main(args=None):
    rclpy.init(args=args)
    node = RecoveryNode()

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

## Step 5: Create the Navigation Manager Node

Create `navigation_system/navigation_system/navigation_manager_node.py`:

```python
#!/usr/bin/env python3
"""
Navigation Manager Node
"""
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Path
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, String
from rclpy.action import ActionServer
from rclpy.action import GoalResponse
from rclpy.action import CancelResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
import time
import math

class NavigationManagerNode(Node):
    def __init__(self):
        super().__init__('navigation_manager_node')

        # Create subscribers
        self.global_plan_sub = self.create_subscription(
            Path, '/global_plan', self.global_plan_callback, 10)
        self.local_plan_sub = self.create_subscription(
            Path, '/local_plan', self.local_plan_callback, 10)
        self.laser_sub = self.create_subscription(
            LaserScan, '/scan', self.laser_callback, 10)
        self.cmd_vel_sub = self.create_subscription(
            Twist, '/cmd_vel', self.cmd_vel_callback, 10)

        # Create publishers
        self.nav_status_pub = self.create_publisher(String, '/nav_status', 10)
        self.nav_stuck_pub = self.create_publisher(Bool, '/nav_stuck', 10)

        # Navigation state
        self.navigation_active = False
        self.current_goal = None
        self.global_plan = None
        self.laser_data = None
        self.last_cmd_vel = Twist()
        self.last_cmd_time = None

        # Navigation parameters
        self.stuck_threshold = 0.05  # Robot is stuck if moving slower than this
        self.stuck_timeout = 5.0     # Seconds before declaring stuck
        self.goal_tolerance = 0.3    # Distance to goal for success

        # Stuck detection
        self.stuck_timer = self.create_timer(1.0, self.check_stuck)

        self.get_logger().info('Navigation manager node started')

    def global_plan_callback(self, msg):
        """Handle global plan updates"""
        self.global_plan = msg
        self.get_logger().info(f'Global plan updated with {len(msg.poses)} waypoints')

    def local_plan_callback(self, msg):
        """Handle local plan updates"""
        # Local plan updates are handled by local planner
        pass

    def laser_callback(self, msg):
        """Handle laser scan data"""
        self.laser_data = msg

    def cmd_vel_callback(self, msg):
        """Monitor command velocities"""
        self.last_cmd_vel = msg
        self.last_cmd_time = self.get_clock().now().nanoseconds * 1e-9

    def check_stuck(self):
        """Check if robot is stuck"""
        if not self.navigation_active:
            return

        # Check if we've been trying to move but not making progress
        if self.last_cmd_time is not None:
            current_time = self.get_clock().now().nanoseconds * 1e-9
            time_since_cmd = current_time - self.last_cmd_time

            # If robot should be moving but isn't
            if (abs(self.last_cmd_vel.linear.x) > self.stuck_threshold and
                time_since_cmd > self.stuck_timeout):
                # Check if robot is actually moving (would need odometry for this)
                # For this exercise, we'll assume it's stuck if it's trying to move
                # but we haven't seen new commands recently
                stuck_msg = Bool()
                stuck_msg.data = True
                self.nav_stuck_pub.publish(stuck_msg)
                self.get_logger().warn('Robot may be stuck, triggering recovery')

    def start_navigation(self, goal_pose):
        """Start navigation to goal"""
        self.current_goal = goal_pose
        self.navigation_active = True
        self.last_cmd_time = self.get_clock().now().nanoseconds * 1e-9

        # Publish navigation status
        status_msg = String()
        status_msg.data = 'NAVIGATING'
        self.nav_status_pub.publish(status_msg)

        self.get_logger().info(f'Starting navigation to goal: ({goal_pose.pose.position.x}, {goal_pose.pose.position.y})')

    def stop_navigation(self):
        """Stop navigation"""
        self.navigation_active = False
        self.current_goal = None

        # Publish navigation status
        status_msg = String()
        status_msg.data = 'STOPPED'
        self.nav_status_pub.publish(status_msg)

        self.get_logger().info('Navigation stopped')

    def check_goal_reached(self):
        """Check if goal has been reached"""
        if self.current_goal is None or self.global_plan is None:
            return False

        # In a real system, this would check against current robot position
        # For this exercise, we'll check if the global plan is complete
        if len(self.global_plan.poses) == 0:
            return True

        # Check if we're close to the final goal
        final_goal = self.global_plan.poses[-1] if self.global_plan.poses else None
        if final_goal:
            # This would compare with actual robot position
            # For simulation, we'll just return False to continue
            return False

        return False

def main(args=None):
    rclpy.init(args=args)
    node = NavigationManagerNode()

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

Create `navigation_system/launch/navigation_system.launch.py`:

```python
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # Declare launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')
    map_topic = LaunchConfiguration('map_topic', default='/map')
    scan_topic = LaunchConfiguration('scan_topic', default='/scan')

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation time if true'
        ),
        DeclareLaunchArgument(
            'map_topic',
            default_value='/map',
            description='Map topic name'
        ),
        DeclareLaunchArgument(
            'scan_topic',
            default_value='/scan',
            description='Scan topic name'
        ),

        # Global planner node
        Node(
            package='navigation_system',
            executable='global_planner_node',
            name='global_planner',
            parameters=[
                {'use_sim_time': use_sim_time}
            ],
            remappings=[
                ('/map', map_topic),
                ('/move_base_simple/goal', '/move_base_simple/goal'),
                ('/global_plan', '/global_plan')
            ],
            output='screen'
        ),

        # Local planner node
        Node(
            package='navigation_system',
            executable='local_planner_node',
            name='local_planner',
            parameters=[
                {'use_sim_time': use_sim_time}
            ],
            remappings=[
                ('/global_plan', '/global_plan'),
                ('/scan', scan_topic),
                ('/amcl_pose', '/amcl_pose'),
                ('/cmd_vel', '/cmd_vel')
            ],
            output='screen'
        ),

        # Recovery node
        Node(
            package='navigation_system',
            executable='recovery_node',
            name='recovery_node',
            parameters=[
                {'use_sim_time': use_sim_time}
            ],
            remappings=[
                ('/cmd_vel', '/cmd_vel'),
                ('/scan', scan_topic),
                ('/nav_stuck', '/nav_stuck'),
                ('/recovery_cmd_vel', '/recovery_cmd_vel')
            ],
            output='screen'
        ),

        # Navigation manager node
        Node(
            package='navigation_system',
            executable='navigation_manager_node',
            name='navigation_manager',
            parameters=[
                {'use_sim_time': use_sim_time}
            ],
            remappings=[
                ('/global_plan', '/global_plan'),
                ('/scan', scan_topic),
                ('/cmd_vel', '/cmd_vel'),
                ('/nav_stuck', '/nav_stuck')
            ],
            output='screen'
        )
    ])
```

## Step 7: Update Package Configuration

Update `navigation_system/setup.py`:

```python
from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'navigation_system'

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
    description='Navigation system for robot navigation',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'global_planner_node = navigation_system.global_planner_node:main',
            'local_planner_node = navigation_system.local_planner_node:main',
            'recovery_node = navigation_system.recovery_node:main',
            'navigation_manager_node = navigation_system.navigation_manager_node:main',
        ],
    },
)
```

## Step 8: Build and Test

Build the package:

```bash
cd ~/ros2_ws
colcon build --packages-select navigation_system
source install/setup.bash
```

### Test the Navigation System

1. **Launch the navigation system**:
```bash
# Launch the navigation system
ros2 launch navigation_system navigation_system.launch.py
```

2. **Provide sensor data** (if using real robot or simulation):
```bash
# For simulation, you might run Gazebo with a robot
# Or use a rosbag with sensor data
ros2 bag play your_robot_data.db3
```

3. **Send navigation goals**:
```bash
# Send a goal using RViz2
# Or send directly using command line:
ros2 run tf2_ros static_transform_publisher 0 0 0 0 0 0 map odom
ros2 topic pub /move_base_simple/goal geometry_msgs/PoseStamped "{header: {frame_id: 'map'}, pose: {position: {x: 1.0, y: 1.0, z: 0.0}, orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}}}"
```

4. **Monitor the system**:
```bash
# Monitor navigation status
ros2 topic echo /nav_status

# Monitor planned paths
ros2 topic echo /global_plan

# Monitor velocity commands
ros2 topic echo /cmd_vel

# Visualize in RViz2
ros2 run rviz2 rviz2
```

5. **Check system components**:
```bash
# List all navigation topics
ros2 topic list | grep -E "(nav|plan|cmd|recovery)"

# Check node status
ros2 node list | grep nav
```

## Understanding the System

This navigation system demonstrates:

1. **Global Planning**: A* path planning with obstacle inflation
2. **Local Planning**: Pure pursuit path following with obstacle avoidance
3. **Recovery Behaviors**: Multiple strategies for getting unstuck
4. **Navigation Management**: Coordinating all components for safe navigation

## Challenges

### Challenge 1: Add Dynamic Obstacle Avoidance
Implement dynamic obstacle detection and avoidance in the local planner.

<details>
<summary>Hint</summary>

Use laser scan data to detect moving obstacles and adjust the local path accordingly. Consider using velocity obstacles or other dynamic path planning techniques.
</details>

### Challenge 2: Implement Path Smoothing
Add path smoothing to the global planner for smoother navigation.

<details>
<summary>Hint</summary>

Implement a path smoothing algorithm that reduces sharp turns while maintaining obstacle avoidance.
</details>

### Challenge 3: Add Goal Tracking
Improve goal tracking accuracy with better localization integration.

<details>
<summary>Hint</summary>

Use odometry and localization data to more accurately determine when the goal is reached.
</details>

### Challenge 4: Optimize Performance
Improve computational efficiency for real-time operation.

<details>
<summary>Hint</summary>

Use multi-threading, optimize algorithms, or implement efficient data structures for faster execution.
</details>

## Verification Checklist

- [ ] Global planner generates valid paths
- [ ] Local planner follows paths smoothly
- [ ] Obstacle avoidance works correctly
- [ ] Recovery behaviors activate when needed
- [ ] Navigation manager coordinates components
- [ ] System publishes appropriate topics
- [ ] All nodes communicate properly
- [ ] Robot navigates safely to goals

## Common Issues

### Planning Issues
```bash
# Check if map is available
ros2 topic echo /map

# Verify path generation
ros2 topic echo /global_plan
```

### Control Issues
```bash
# Monitor velocity commands
ros2 topic echo /cmd_vel

# Check laser data
ros2 topic echo /scan
```

### Integration Issues
```bash
# Check TF frames
ros2 run tf2_tools view_frames

# Verify all nodes are running
ros2 node list
```

## Summary

In this exercise, you learned to:
- Implement a complete navigation system with global and local planning
- Create obstacle avoidance and recovery behaviors
- Integrate perception data for safe navigation
- Coordinate multiple navigation components
- Structure a complex navigation system

## Next Steps

Continue to [Week 9: Perception Pipeline](../../module-3-ai-robot-brain/week-09/introduction) to learn about perception systems for robotics.