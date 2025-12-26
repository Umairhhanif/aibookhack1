---
sidebar_position: 5
---

# Exercise: LLM-Powered Robot Assistant

In this comprehensive exercise, you'll create a complete LLM-powered robot assistant that can understand natural language commands, reason about tasks, and execute actions through the robot's control system.

## Objective

Build a robot assistant that:
1. **Interprets natural language commands** using LLMs
2. **Reasons about tasks** and decomposes them into executable actions
3. **Integrates with robot systems** to execute commands
4. **Manages conversation context** for multi-turn interactions
5. **Validates and monitors** execution for safety and correctness

## Prerequisites

- Complete Week 1-11 lessons
- ROS 2 workspace set up (`~/ros2_ws`)
- OpenAI API key (or local LLM setup)
- Understanding of LLM integration concepts
- Basic Python programming skills

## Step 1: Create the Robot Assistant Package

```bash
cd ~/ros2_ws/src

ros2 pkg create --build-type ament_python llm_robot_assistant \
    --dependencies rclpy std_msgs sensor_msgs geometry_msgs nav_msgs tf2_ros cv_bridge builtin_interfaces
```

## Step 2: Create the Main Assistant Node

Create `llm_robot_assistant/llm_robot_assistant/main_assistant_node.py`:

```python
#!/usr/bin/env python3
"""
LLM Robot Assistant Main Node
"""
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped, Twist
from sensor_msgs.msg import Image, LaserScan
from nav_msgs.msg import Odometry
from builtin_interfaces.msg import Time
import openai
import json
import asyncio
from typing import Dict, Any, List
import time

class LLMRobotAssistantNode(Node):
    def __init__(self):
        super().__init__('llm_robot_assistant_node')

        # Create subscribers
        self.command_sub = self.create_subscription(
            String, '/user_command', self.command_callback, 10)
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10)
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10)
        self.camera_sub = self.create_subscription(
            Image, '/camera/image_raw', self.camera_callback, 10)

        # Create publishers
        self.response_pub = self.create_publisher(String, '/assistant_response', 10)
        self.nav_goal_pub = self.create_publisher(PoseStamped, '/navigation/goal', 10)
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.status_pub = self.create_publisher(String, '/assistant_status', 10)

        # Initialize LLM client
        self.api_key = self.declare_parameter('openai_api_key', '').value
        if self.api_key:
            openai.api_key = self.api_key
            self.use_openai = True
        else:
            self.get_logger().warn('No OpenAI API key provided, using mock responses')
            self.use_openai = False

        # Robot state
        self.robot_state = {
            'position': {'x': 0.0, 'y': 0.0, 'theta': 0.0},
            'battery_level': 100.0,
            'current_task': 'idle',
            'sensors_working': True,
            'last_command_time': time.time()
        }

        # Conversation management
        self.conversation_history = []
        self.max_history_length = 20

        # Task execution queue
        self.task_queue = []
        self.current_task = None

        # LLM configuration
        self.llm_model = 'gpt-3.5-turbo'
        self.temperature = 0.7
        self.max_tokens = 300

        # System capabilities
        self.capabilities = {
            'navigation': {
                'actions': ['go_to', 'move_to', 'navigate_to', 'return_to_base'],
                'areas': ['kitchen', 'living_room', 'bedroom', 'office', 'dining_room', 'bathroom']
            },
            'manipulation': {
                'actions': ['pick_up', 'place', 'grasp', 'release'],
                'objects': ['cup', 'book', 'ball', 'bottle', 'phone', 'keys']
            },
            'perception': {
                'actions': ['find', 'locate', 'detect', 'recognize'],
                'targets': ['person', 'object', 'obstacle', 'landmark']
            },
            'interaction': {
                'actions': ['greet', 'inform', 'report', 'confirm']
            }
        }

        self.get_logger().info('LLM Robot Assistant node started')

    def command_callback(self, msg):
        """Process natural language command"""
        try:
            user_command = msg.data
            self.get_logger().info(f'Received command: {user_command}')

            # Add to conversation history
            self.conversation_history.append({
                'speaker': 'user',
                'text': user_command,
                'timestamp': time.time()
            })

            # Keep history manageable
            if len(self.conversation_history) > self.max_history_length:
                self.conversation_history = self.conversation_history[-self.max_history_length:]

            # Process command with LLM
            llm_response = self.process_command_with_llm(user_command)

            # Parse and execute response
            if llm_response:
                self.execute_llm_response(llm_response)

                # Add response to history
                self.conversation_history.append({
                    'speaker': 'assistant',
                    'text': llm_response,
                    'timestamp': time.time()
                })

                # Publish response
                response_msg = String()
                response_msg.data = llm_response
                self.response_pub.publish(response_msg)

                self.get_logger().info(f'Assistant response: {llm_response[:50]}...')

        except Exception as e:
            self.get_logger().error(f'Error processing command: {e}')
            error_response = 'I encountered an error processing your command. Could you please rephrase it?'

            error_msg = String()
            error_msg.data = error_response
            self.response_pub.publish(error_msg)

    def process_command_with_llm(self, command):
        """Process command using LLM"""
        try:
            # Create context-aware prompt
            prompt = self.create_contextual_prompt(command)

            if self.use_openai:
                # Call OpenAI API
                response = openai.ChatCompletion.create(
                    model=self.llm_model,
                    messages=[
                        {"role": "system", "content": self.get_system_prompt()},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )

                return response.choices[0].message.content.strip()
            else:
                # Mock response for testing
                return self.generate_mock_response(command)

        except Exception as e:
            self.get_logger().error(f'LLM API error: {e}')
            return self.generate_fallback_response(command)

    def create_contextual_prompt(self, user_command):
        """Create prompt with context information"""
        context_info = {
            'robot_capabilities': self.capabilities,
            'current_state': self.robot_state,
            'conversation_history': self.conversation_history[-5:],  # Last 5 exchanges
            'environment_info': self.get_environment_info()
        }

        prompt = f"""
You are a helpful robot assistant. Interpret the user's command and provide structured robot instructions.

Context Information:
Robot Capabilities: {json.dumps(self.capabilities, indent=2)}
Current Robot State: {json.dumps(self.robot_state, indent=2)}
Environment Info: {json.dumps(self.get_environment_info(), indent=2)}
Recent Conversation: {json.dumps(self.conversation_history[-3:], indent=2)}

User Command: "{user_command}"

Provide your response in JSON format with the following structure:
{{
    "intent": "navigation|manipulation|perception|interaction|unknown",
    "action": "specific_robot_action",
    "parameters": {{
        "target": "target_object_or_location",
        "location": "specific_location_if_applicable",
        "description": "additional_details"
    }},
    "confidence": 0.0-1.0,
    "explanation": "why this interpretation was chosen",
    "follow_up_required": true|false
}}

Only respond with the JSON object, nothing else.
        """

        return prompt

    def get_system_prompt(self):
        """Get system prompt for LLM"""
        return """
You are a helpful robot assistant that interprets natural language commands and translates them into structured robot actions. You should:

1. Understand the user's intent from their natural language command
2. Consider the robot's capabilities and current state
3. Provide structured JSON response with action to execute
4. Maintain conversation context and continuity
5. Ask for clarification if the command is ambiguous
6. Ensure safety by verifying actions are appropriate

Always respond in the specified JSON format. Be helpful but cautious about safety.
        """

    def get_environment_info(self):
        """Get current environment information"""
        return {
            'obstacles_nearby': self.has_obstacles_nearby(),
            'room_type': self.estimate_room_type(),
            'visible_objects': self.get_visible_objects(),
            'battery_status': f"{self.robot_state['battery_level']:.1f}%"
        }

    def execute_llm_response(self, response_text):
        """Parse and execute LLM response"""
        try:
            # Extract JSON from response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1

            if json_start != -1 and json_end != 0:
                json_str = response_text[json_start:json_end]
                parsed_response = json.loads(json_str)

                # Validate response structure
                if self.validate_response_structure(parsed_response):
                    # Execute based on intent
                    intent = parsed_response.get('intent', 'unknown')
                    action = parsed_response.get('action', 'unknown')
                    params = parsed_response.get('parameters', {})

                    if intent == 'navigation':
                        self.execute_navigation_action(action, params)
                    elif intent == 'manipulation':
                        self.execute_manipulation_action(action, params)
                    elif intent == 'perception':
                        self.execute_perception_action(action, params)
                    elif intent == 'interaction':
                        self.execute_interaction_action(action, params)
                    else:
                        self.execute_unknown_action(parsed_response)

                    # Update robot state
                    self.robot_state['last_command_time'] = time.time()
                    self.robot_state['current_task'] = action

                else:
                    self.get_logger().warn('Invalid response structure from LLM')
                    self.request_clarification()

            else:
                self.get_logger().warn('No valid JSON found in LLM response')
                self.request_clarification()

        except json.JSONDecodeError:
            self.get_logger().error('Could not parse JSON from LLM response')
            self.request_clarification()
        except Exception as e:
            self.get_logger().error(f'Error executing LLM response: {e}')
            self.request_clarification()

    def execute_navigation_action(self, action, params):
        """Execute navigation action"""
        target_location = params.get('location', params.get('target', 'unknown'))

        if target_location == 'unknown':
            self.request_location_clarification()
            return

        # Convert location to coordinates (simplified - in practice, use semantic map)
        location_coords = self.get_location_coordinates(target_location)

        if location_coords:
            # Create navigation goal
            goal_msg = PoseStamped()
            goal_msg.header.stamp = self.get_clock().now().to_msg()
            goal_msg.header.frame_id = 'map'
            goal_msg.pose.position.x = location_coords[0]
            goal_msg.pose.position.y = location_coords[1]
            goal_msg.pose.position.z = 0.0
            goal_msg.pose.orientation.w = 1.0  # No rotation

            self.nav_goal_pub.publish(goal_msg)
            self.get_logger().info(f'Navigating to {target_location} at ({location_coords[0]}, {location_coords[1]})')
        else:
            self.get_logger().warn(f'Unknown location: {target_location}')
            self.request_clarification(f'Sorry, I don\'t know where {target_location} is.')

    def get_location_coordinates(self, location_name):
        """Get coordinates for named location (simplified mapping)"""
        location_map = {
            'home_base': (0.0, 0.0),
            'kitchen': (2.0, 1.0),
            'living_room': (0.0, 2.0),
            'bedroom': (-1.0, -1.0),
            'office': (1.5, -0.5),
            'dining_room': (-0.5, 1.5),
            'bathroom': (0.5, -1.5)
        }

        return location_map.get(location_name.lower())

    def execute_manipulation_action(self, action, params):
        """Execute manipulation action"""
        target_object = params.get('target', 'unknown')

        if target_object == 'unknown':
            self.request_object_clarification()
            return

        # In a real system, this would trigger manipulation
        # For this exercise, we'll just log the intention
        self.get_logger().info(f'Manipulation action: {action} {target_object}')

        # Check if object is visible/accessible
        if self.is_object_accessible(target_object):
            # Trigger manipulation (conceptual)
            manipulation_msg = String()
            manipulation_msg.data = f"{action}:{target_object}"
            # self.manipulation_pub.publish(manipulation_msg)
            self.get_logger().info(f'Attempting to {action} {target_object}')
        else:
            self.get_logger().warn(f'Object {target_object} not accessible')
            self.request_clarification(f'Sorry, I can\'t find or reach the {target_object}.')

    def validate_response_structure(self, response):
        """Validate LLM response structure"""
        required_keys = ['intent', 'action', 'parameters', 'confidence', 'explanation']
        return all(key in response for key in required_keys)

    def request_clarification(self, message=None):
        """Request clarification from user"""
        if message is None:
            message = 'I need more information to complete your request.'

        clarification_msg = String()
        clarification_msg.data = message
        self.response_pub.publish(clarification_msg)

    def odom_callback(self, msg):
        """Update robot position from odometry"""
        self.robot_state['position']['x'] = msg.pose.pose.position.x
        self.robot_state['position']['y'] = msg.pose.pose.position.y

        # Extract orientation (simplified - in practice, use proper quaternion to euler conversion)
        # For this example, we'll just store the quaternion
        self.robot_state['position']['orientation'] = {
            'x': msg.pose.pose.orientation.x,
            'y': msg.pose.pose.orientation.y,
            'z': msg.pose.pose.orientation.z,
            'w': msg.pose.pose.orientation.w
        }

    def scan_callback(self, msg):
        """Process laser scan data"""
        # Update obstacle information
        self.robot_state['has_obstacles_ahead'] = self.check_obstacles_ahead(msg.ranges)

    def camera_callback(self, msg):
        """Process camera data"""
        # In practice, run object detection, etc.
        # For this example, we'll just note that camera data is available
        self.robot_state['camera_data_available'] = True

    def check_obstacles_ahead(self, ranges):
        """Check for obstacles in front of robot"""
        if not ranges:
            return False

        # Check forward-facing laser beams (simplified)
        forward_beams = ranges[len(ranges)//2-10:len(ranges)//2+10]  # ±10 beams around center
        min_distance = min(ranges) if ranges else float('inf')

        return min_distance < 0.5  # Obstacle within 0.5m

    def has_obstacles_nearby(self):
        """Check if there are obstacles nearby"""
        # This would use the latest scan data
        return getattr(self, 'robot_state', {}).get('has_obstacles_ahead', False)

    def estimate_room_type(self):
        """Estimate current room type based on context"""
        # In practice, this would use semantic mapping or scene classification
        # For this example, return a placeholder
        return 'unknown_room'

    def get_visible_objects(self):
        """Get objects that can be perceived"""
        # In practice, this would use object detection
        # For this example, return a placeholder
        return ['unknown_objects']

    def is_object_accessible(self, obj_name):
        """Check if object is accessible (simplified)"""
        # In practice, this would check perception data and navigation feasibility
        # For this example, assume all objects are accessible
        return True

    def generate_mock_response(self, command):
        """Generate mock response for testing"""
        import random

        if 'go to' in command.lower() or 'navigate to' in command.lower():
            return json.dumps({
                'intent': 'navigation',
                'action': 'navigate_to',
                'parameters': {
                    'location': 'kitchen',
                    'target': 'kitchen'
                },
                'confidence': 0.9,
                'explanation': 'Command indicates navigation request to kitchen',
                'follow_up_required': False
            })
        elif 'pick up' in command.lower() or 'get' in command.lower():
            return json.dumps({
                'intent': 'manipulation',
                'action': 'pick_up',
                'parameters': {
                    'target': 'cup',
                    'location': 'kitchen'
                },
                'confidence': 0.85,
                'explanation': 'Command indicates manipulation request for cup',
                'follow_up_required': True
            })
        else:
            return json.dumps({
                'intent': 'interaction',
                'action': 'respond',
                'parameters': {
                    'target': 'user',
                    'response': 'I understand your request'
                },
                'confidence': 0.7,
                'explanation': 'Generic interaction command',
                'follow_up_required': True
            })

    def generate_fallback_response(self, command):
        """Generate fallback response when LLM fails"""
        return json.dumps({
            'intent': 'interaction',
            'action': 'ask_for_clarification',
            'parameters': {
                'target': 'user',
                'question': 'Could you please rephrase your command?'
            },
            'confidence': 0.3,
            'explanation': 'LLM API error, requesting clarification',
            'follow_up_required': True
        })

def main(args=None):
    rclpy.init(args=args)
    node = LLMRobotAssistantNode()

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

## Step 3: Create Task Planner Component

Create `llm_robot_assistant/llm_robot_assistant/task_planner.py`:

```python
#!/usr/bin/env python3
"""
Task Planning Component for LLM Robot Assistant
"""
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
import json
from typing import Dict, Any, List

class TaskPlannerNode(Node):
    def __init__(self):
        super().__init__('task_planner_node')

        # Create subscribers
        self.task_request_sub = self.create_subscription(
            String, '/task_request', self.task_request_callback, 10)

        # Create publishers
        self.task_plan_pub = self.create_publisher(String, '/task_plan', 10)
        self.execution_command_pub = self.create_publisher(String, '/execution_command', 10)

        # Task planning database
        self.task_database = {
            'navigation': {
                'go_to_location': self.plan_navigation_to_location,
                'return_home': self.plan_return_home,
                'explore_area': self.plan_explore_area
            },
            'manipulation': {
                'pick_and_place': self.plan_pick_and_place,
                'transport_object': self.plan_transport_object,
                'grasp_object': self.plan_grasp_object
            },
            'perception': {
                'find_object': self.plan_find_object,
                'inspect_area': self.plan_inspect_area,
                'track_object': self.plan_track_object
            }
        }

        # Robot capabilities
        self.robot_capabilities = {
            'navigation': {
                'max_speed': 0.5,
                'rotation_speed': 0.5,
                'sensor_range': 3.0
            },
            'manipulation': {
                'max_reach': 1.0,
                'max_payload': 2.0,
                'gripper_type': 'parallel'
            },
            'perception': {
                'camera_fov': 60.0,
                'detection_range': 5.0,
                'classification_accuracy': 0.95
            }
        }

        self.get_logger().info('Task planner node started')

    def task_request_callback(self, msg):
        """Process task request and generate plan"""
        try:
            task_request = json.loads(msg.data)

            intent = task_request.get('intent', 'unknown')
            action = task_request.get('action', 'unknown')
            parameters = task_request.get('parameters', {})

            if intent in self.task_database and action in self.task_database[intent]:
                # Generate plan for the requested task
                task_plan = self.task_database[intent][action](parameters)

                # Validate plan
                if self.validate_task_plan(task_plan):
                    # Publish plan
                    plan_msg = String()
                    plan_msg.data = json.dumps(task_plan)
                    self.task_plan_pub.publish(plan_msg)

                    self.get_logger().info(f'Generated task plan with {len(task_plan["steps"])} steps')
                else:
                    self.get_logger().error('Generated task plan is invalid')
                    self.publish_error_plan(action, parameters)
            else:
                self.get_logger().warn(f'Unknown task: {intent}.{action}')
                self.publish_error_plan(action, parameters)

        except json.JSONDecodeError:
            self.get_logger().error('Invalid JSON in task request')
        except Exception as e:
            self.get_logger().error(f'Error in task planning: {e}')

    def plan_navigation_to_location(self, params):
        """Plan navigation to a specific location"""
        target_location = params.get('location', 'unknown')

        # In practice, this would use navigation stack to plan path
        # For this example, we'll create a simple plan

        plan = {
            'task_id': f"nav_to_{target_location}_{int(time.time())}",
            'task_type': 'navigation',
            'action': 'go_to_location',
            'parameters': params,
            'steps': [
                {
                    'step_id': 1,
                    'action': 'check_battery',
                    'description': 'Verify battery level is sufficient for navigation',
                    'requirements': ['battery_level > 20%'],
                    'estimated_duration': 0.1
                },
                {
                    'step_id': 2,
                    'action': 'localize_robot',
                    'description': 'Determine current robot position',
                    'requirements': ['localization_system_active'],
                    'estimated_duration': 1.0
                },
                {
                    'step_id': 3,
                    'action': 'plan_path',
                    'description': f'Plan path to {target_location}',
                    'requirements': ['map_available', 'valid_target_location'],
                    'estimated_duration': 0.5
                },
                {
                    'step_id': 4,
                    'action': 'execute_navigation',
                    'description': f'Navigate to {target_location}',
                    'requirements': ['path_planned', 'navigation_system_active'],
                    'estimated_duration': 30.0  # Variable in practice
                },
                {
                    'step_id': 5,
                    'action': 'confirm_arrival',
                    'description': f'Confirm arrival at {target_location}',
                    'requirements': ['navigation_completed'],
                    'estimated_duration': 0.5
                }
            ],
            'estimated_total_duration': 32.1,
            'success_conditions': [f'robot_at_{target_location}'],
            'failure_conditions': ['navigation_failed', 'obstacle_encountered'],
            'recovery_actions': ['return_to_previous_location', 'request_assistance']
        }

        return plan

    def plan_pick_and_place(self, params):
        """Plan pick and place task"""
        target_object = params.get('target', 'unknown')
        target_location = params.get('location', 'unknown')

        plan = {
            'task_id': f"pick_place_{target_object}_at_{target_location}_{int(time.time())}",
            'task_type': 'manipulation',
            'action': 'pick_and_place',
            'parameters': params,
            'steps': [
                {
                    'step_id': 1,
                    'action': 'approach_object',
                    'description': f'Approach the {target_object}',
                    'requirements': ['object_detected', 'navigation_to_object_possible'],
                    'estimated_duration': 10.0
                },
                {
                    'step_id': 2,
                    'action': 'localize_object',
                    'description': f'Precisely locate the {target_object}',
                    'requirements': ['object_in_gripper_workspace'],
                    'estimated_duration': 2.0
                },
                {
                    'step_id': 3,
                    'action': 'grasp_object',
                    'description': f'Grasp the {target_object}',
                    'requirements': ['object_position_known', 'gripper_available'],
                    'estimated_duration': 5.0
                },
                {
                    'step_id': 4,
                    'action': 'verify_grasp',
                    'description': 'Verify object is securely grasped',
                    'requirements': ['object_attached_to_gripper'],
                    'estimated_duration': 1.0
                },
                {
                    'step_id': 5,
                    'action': 'navigate_to_destination',
                    'description': f'Navigate to {target_location}',
                    'requirements': ['object_grasped', 'navigation_system_active'],
                    'estimated_duration': 30.0
                },
                {
                    'step_id': 6,
                    'action': 'place_object',
                    'description': f'Place the {target_object} at destination',
                    'requirements': ['at_destination', 'placement_surface_available'],
                    'estimated_duration': 5.0
                },
                {
                    'step_id': 7,
                    'action': 'verify_placement',
                    'description': 'Verify object was placed successfully',
                    'requirements': ['object_no_longer_grasped', 'object_at_destination'],
                    'estimated_duration': 1.0
                }
            ],
            'estimated_total_duration': 54.0,
            'success_conditions': [f'object_placed_at_{target_location}'],
            'failure_conditions': ['grasp_failed', 'navigation_failed', 'object_dropped'],
            'recovery_actions': ['retry_grasp', 'abort_task', 'request_human_help']
        }

        return plan

    def plan_find_object(self, params):
        """Plan object finding task"""
        target_object = params.get('target', 'unknown')
        search_area = params.get('search_area', 'current_room')

        plan = {
            'task_id': f"find_{target_object}_in_{search_area}_{int(time.time())}",
            'task_type': 'perception',
            'action': 'find_object',
            'parameters': params,
            'steps': [
                {
                    'step_id': 1,
                    'action': 'orient_camera',
                    'description': 'Orient camera toward likely object locations',
                    'requirements': ['camera_operational'],
                    'estimated_duration': 0.5
                },
                {
                    'step_id': 2,
                    'action': 'run_object_detection',
                    'description': f'Run object detection for {target_object}',
                    'requirements': ['camera_operational', 'object_detection_model_loaded'],
                    'estimated_duration': 1.0
                },
                {
                    'step_id': 3,
                    'action': 'analyze_detections',
                    'description': 'Analyze detection results',
                    'requirements': ['detections_available'],
                    'estimated_duration': 0.5
                },
                {
                    'step_id': 4,
                    'action': 'explore_search_area',
                    'description': f'Explore {search_area} systematically',
                    'requirements': ['object_not_found_initially'],
                    'estimated_duration': 20.0
                },
                {
                    'step_id': 5,
                    'action': 'repeat_detection',
                    'description': 'Repeat detection from new viewpoints',
                    'requirements': ['search_area_explored'],
                    'estimated_duration': 1.0
                },
                {
                    'step_id': 6,
                    'action': 'report_result',
                    'description': 'Report whether object was found',
                    'requirements': ['search_completed'],
                    'estimated_duration': 0.1
                }
            ],
            'estimated_total_duration': 23.1,
            'success_conditions': [f'{target_object}_detected'],
            'failure_conditions': ['search_area_exhausted', 'time_limit_reached'],
            'recovery_actions': ['expand_search_area', 'request_assistance', 'try_alternative_methods']
        }

        return plan

    def validate_task_plan(self, plan):
        """Validate generated task plan"""
        required_fields = ['task_id', 'task_type', 'action', 'steps', 'success_conditions']

        if not all(field in plan for field in required_fields):
            return False

        if not isinstance(plan['steps'], list) or len(plan['steps']) == 0:
            return False

        for step in plan['steps']:
            if not all(key in step for key in ['step_id', 'action', 'description', 'requirements']):
                return False

        return True

    def publish_error_plan(self, action, parameters):
        """Publish error plan when task planning fails"""
        error_plan = {
            'task_id': f"error_{action}_{int(time.time())}",
            'error': True,
            'error_message': f'Could not plan task: {action}',
            'parameters': parameters,
            'steps': [],
            'estimated_total_duration': 0.0,
            'success_conditions': [],
            'failure_conditions': ['planning_failed']
        }

        error_msg = String()
        error_msg.data = json.dumps(error_plan)
        self.task_plan_pub.publish(error_msg)

def main(args=None):
    rclpy.init(args=args)
    node = TaskPlannerNode()

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

## Step 4: Create Execution Monitor Component

Create `llm_robot_assistant/llm_robot_assistant/execution_monitor.py`:

```python
#!/usr/bin/env python3
"""
Execution Monitor for LLM Robot Assistant
"""
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped
import json
import time
from typing import Dict, Any

class ExecutionMonitorNode(Node):
    def __init__(self):
        super().__init__('execution_monitor_node')

        # Create subscribers
        self.task_plan_sub = self.create_subscription(
            String, '/task_plan', self.task_plan_callback, 10)
        self.execution_status_sub = self.create_subscription(
            String, '/execution_status', self.execution_status_callback, 10)

        # Create publishers
        self.monitoring_status_pub = self.create_publisher(String, '/monitoring_status', 10)
        self.safety_alert_pub = self.create_publisher(String, '/safety_alerts', 10)

        # Task tracking
        self.active_tasks = {}
        self.task_history = []
        self.max_task_history = 50

        # Safety monitoring parameters
        self.safety_thresholds = {
            'max_execution_time': 300,  # 5 minutes
            'min_progress_rate': 0.1,   # 10% progress per minute
            'obstacle_distance': 0.3,   # Stop if obstacle closer than 30cm
            'battery_threshold': 15.0   # Emergency stop below 15% battery
        }

        # Performance metrics
        self.performance_metrics = {
            'successful_executions': 0,
            'failed_executions': 0,
            'average_execution_time': 0.0,
            'safety_interventions': 0
        }

        self.get_logger().info('Execution monitor node started')

    def task_plan_callback(self, msg):
        """Handle incoming task plan"""
        try:
            plan_data = json.loads(msg.data)

            if not plan_data.get('error', False):
                task_id = plan_data['task_id']

                # Initialize task monitoring
                self.active_tasks[task_id] = {
                    'plan': plan_data,
                    'start_time': time.time(),
                    'current_step': 0,
                    'step_start_time': time.time(),
                    'status': 'executing',
                    'progress': 0.0,
                    'safety_violations': 0
                }

                self.get_logger().info(f'Started monitoring task: {task_id}')

        except Exception as e:
            self.get_logger().error(f'Error processing task plan: {e}')

    def execution_status_callback(self, msg):
        """Monitor execution status"""
        try:
            status_data = json.loads(msg.data)

            task_id = status_data.get('task_id')
            if task_id in self.active_tasks:
                task_monitor = self.active_tasks[task_id]

                # Update step progress
                current_step = status_data.get('current_step', 0)
                if current_step > task_monitor['current_step']:
                    task_monitor['current_step'] = current_step
                    task_monitor['step_start_time'] = time.time()

                # Calculate progress
                total_steps = len(task_monitor['plan']['steps'])
                if total_steps > 0:
                    progress = (task_monitor['current_step'] / total_steps) * 100
                    task_monitor['progress'] = progress

                # Check for safety violations
                safety_violations = self.check_safety_violations(task_monitor, status_data)
                if safety_violations:
                    task_monitor['safety_violations'] += len(safety_violations)

                    # Publish safety alerts
                    for violation in safety_violations:
                        alert_msg = String()
                        alert_msg.data = json.dumps({
                            'type': 'safety_violation',
                            'task_id': task_id,
                            'violation': violation,
                            'timestamp': time.time()
                        })
                        self.safety_alerts_pub.publish(alert_msg)

                        self.get_logger().warn(f'Safety violation in task {task_id}: {violation}')

                # Check for task completion
                if status_data.get('status') == 'completed':
                    self.complete_task(task_id, 'success')
                elif status_data.get('status') == 'failed':
                    self.complete_task(task_id, 'failure')

                # Check for timeout
                elapsed_time = time.time() - task_monitor['start_time']
                if elapsed_time > self.safety_thresholds['max_execution_time']:
                    self.intervene_task(task_id, 'timeout')

                # Update monitoring status
                self.publish_monitoring_status(task_id)

        except Exception as e:
            self.get_logger().error(f'Error monitoring execution: {e}')

    def check_safety_violations(self, task_monitor, status_data):
        """Check for safety violations"""
        violations = []

        # Check execution time
        elapsed_time = time.time() - task_monitor['start_time']
        if elapsed_time > self.safety_thresholds['max_execution_time']:
            violations.append(f'Execution timeout: {elapsed_time:.1f}s > {self.safety_thresholds["max_execution_time"]}s')

        # Check progress rate
        if elapsed_time > 60:  # Only check after 1 minute
            expected_progress = (elapsed_time / 60) * self.safety_thresholds['min_progress_rate'] * 100
            if task_monitor['progress'] < expected_progress:
                violations.append(f'Insufficient progress: {task_monitor["progress"]:.1f}% < {expected_progress:.1f}%')

        # Check for external safety conditions (would come from other nodes)
        # For this example, we'll simulate checking from status data
        if status_data.get('battery_level', 100.0) < self.safety_thresholds['battery_threshold']:
            violations.append(f'Battery level critical: {status_data["battery_level"]:.1f}% < {self.safety_thresholds["battery_threshold"]}%')

        if status_data.get('obstacle_distance', float('inf')) < self.safety_thresholds['obstacle_distance']:
            violations.append(f'Obstacle too close: {status_data["obstacle_distance"]:.2f}m < {self.safety_thresholds["obstacle_distance"]}m')

        return violations

    def intervene_task(self, task_id, reason):
        """Intervene in task execution for safety"""
        if task_id in self.active_tasks:
            self.get_logger().warn(f'Intervening in task {task_id} due to: {reason}')

            # Send intervention command
            intervention_msg = String()
            intervention_msg.data = json.dumps({
                'task_id': task_id,
                'intervention_type': 'safety_stop',
                'reason': reason,
                'timestamp': time.time()
            })

            # In practice, publish to intervention topic
            # self.intervention_pub.publish(intervention_msg)

            # Update task status
            self.active_tasks[task_id]['status'] = 'interrupted'
            self.complete_task(task_id, 'intervened')

            # Update metrics
            self.performance_metrics['safety_interventions'] += 1

    def complete_task(self, task_id, outcome):
        """Complete monitoring for a task"""
        if task_id in self.active_tasks:
            task_monitor = self.active_tasks[task_id]
            elapsed_time = time.time() - task_monitor['start_time']

            # Update performance metrics
            if outcome == 'success':
                self.performance_metrics['successful_executions'] += 1
                # Update average time (simple approach)
                total_successful = self.performance_metrics['successful_executions']
                current_avg = self.performance_metrics['average_execution_time']
                new_avg = ((current_avg * (total_successful - 1)) + elapsed_time) / total_successful
                self.performance_metrics['average_execution_time'] = new_avg
            else:
                self.performance_metrics['failed_executions'] += 1

            # Add to history
            task_record = {
                'task_id': task_id,
                'outcome': outcome,
                'execution_time': elapsed_time,
                'safety_violations': task_monitor['safety_violations'],
                'completion_time': time.time(),
                'plan': task_monitor['plan']
            }

            self.task_history.append(task_record)
            if len(self.task_history) > self.max_task_history:
                self.task_history.pop(0)

            # Remove from active tasks
            del self.active_tasks[task_id]

            self.get_logger().info(f'Task {task_id} completed with outcome: {outcome}')

    def publish_monitoring_status(self, task_id):
        """Publish current monitoring status"""
        if task_id in self.active_tasks:
            status_msg = String()
            task_monitor = self.active_tasks[task_id]

            status_data = {
                'task_id': task_id,
                'status': task_monitor['status'],
                'progress': task_monitor['progress'],
                'elapsed_time': time.time() - task_monitor['start_time'],
                'current_step': task_monitor['current_step'],
                'safety_violations': task_monitor['safety_violations'],
                'performance_metrics': self.performance_metrics
            }

            status_msg.data = json.dumps(status_data)
            self.monitoring_status_pub.publish(status_msg)

def main(args=None):
    rclpy.init(args=args)
    node = ExecutionMonitorNode()

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

## Step 5: Create the Launch File

Create `llm_robot_assistant/launch/llm_assistant.launch.py`:

```python
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler
from launch.event_handlers import OnProcessStart
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Declare launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')
    openai_api_key = LaunchConfiguration('openai_api_key', default='')

    # Main assistant node
    assistant_node = Node(
        package='llm_robot_assistant',
        executable='main_assistant_node',
        name='llm_robot_assistant',
        parameters=[
            {'use_sim_time': use_sim_time},
            {'openai_api_key': openai_api_key}
        ],
        remappings=[
            ('/user_command', '/natural_language_command'),
            ('/assistant_response', '/speak_text'),
            ('/navigation/goal', '/move_base_simple/goal'),
            ('/cmd_vel', '/cmd_vel'),
            ('/assistant_status', '/llm_assistant/status')
        ],
        output='screen'
    )

    # Task planner node
    task_planner_node = Node(
        package='llm_robot_assistant',
        executable='task_planner',
        name='llm_task_planner',
        parameters=[{'use_sim_time': use_sim_time}],
        remappings=[
            ('/task_request', '/llm_task_request'),
            ('/task_plan', '/llm_task_plan'),
            ('/execution_command', '/llm_execution_command')
        ],
        output='screen'
    )

    # Execution monitor node
    execution_monitor_node = Node(
        package='llm_robot_assistant',
        executable='execution_monitor',
        name='llm_execution_monitor',
        parameters=[{'use_sim_time': use_sim_time}],
        remappings=[
            ('/task_plan', '/llm_task_plan'),
            ('/execution_status', '/llm_execution_status'),
            ('/monitoring_status', '/llm_monitoring_status'),
            ('/safety_alerts', '/llm_safety_alerts')
        ],
        output='screen'
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation time if true'
        ),
        DeclareLaunchArgument(
            'openai_api_key',
            default_value='',
            description='OpenAI API key for LLM integration'
        ),

        assistant_node,
        task_planner_node,
        execution_monitor_node
    ])
```

## Step 6: Create the Setup File

Update `llm_robot_assistant/setup.py`:

```python
from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'llm_robot_assistant'

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
    description='LLM-powered robot assistant system',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'main_assistant_node = llm_robot_assistant.main_assistant_node:main',
            'task_planner = llm_robot_assistant.task_planner:main',
            'execution_monitor = llm_robot_assistant.execution_monitor:main',
        ],
    },
)
```

## Step 7: Build and Test the System

Build the package:

```bash
cd ~/ros2_ws
colcon build --packages-select llm_robot_assistant
source install/setup.bash
```

### Test the System

1. **Set your OpenAI API key**:
```bash
export OPENAI_API_KEY=your_actual_api_key_here
```

2. **Launch the complete system**:
```bash
ros2 launch llm_robot_assistant llm_assistant.launch.py
```

3. **Test with sample commands**:
```bash
# Send a navigation command
ros2 topic pub /user_command std_msgs/String "data: 'Please go to the kitchen'"

# Send a manipulation command
ros2 topic pub /user_command std_msgs/String "data: 'Can you pick up the red cup from the table?'"

# Send a perception command
ros2 topic pub /user_command std_msgs/String "data: 'What objects do you see in the room?'"
```

4. **Monitor the system**:
```bash
# Monitor responses
ros2 topic echo /assistant_response

# Monitor task plans
ros2 topic echo /task_plan

# Monitor execution status
ros2 topic echo /monitoring_status

# Monitor safety alerts
ros2 topic echo /safety_alerts
```

## System Architecture Review

The complete LLM-powered robot assistant system includes:

1. **Main Assistant Node**: Processes natural language and coordinates tasks
2. **Task Planner**: Creates detailed execution plans
3. **Execution Monitor**: Tracks progress and ensures safety
4. **Launch System**: Orchestrates all components

## Quality Assurance

### 1. Response Quality Validation

```python
#!/usr/bin/env python3
"""
Response Quality Validator
"""
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json
import re

class ResponseQualityValidator(Node):
    def __init__(self):
        super().__init__('response_quality_validator')

        self.response_sub = self.create_subscription(
            String, '/assistant_response', self.response_callback, 10)
        self.quality_score_pub = self.create_publisher(Float32, '/response_quality_score', 10)

        self.get_logger().info('Response quality validator started')

    def response_callback(self, msg):
        """Validate response quality"""
        try:
            response_text = msg.data

            # Check if response contains valid JSON for actions
            quality_score = self.evaluate_response_quality(response_text)

            # Publish quality score
            score_msg = Float32()
            score_msg.data = quality_score
            self.quality_score_pub.publish(score_msg)

            if quality_score < 0.5:
                self.get_logger().warn(f'Low quality response: {quality_score:.3f} - {response_text[:50]}...')

        except Exception as e:
            self.get_logger().error(f'Error validating response: {e}')

    def evaluate_response_quality(self, response_text):
        """Evaluate response quality based on multiple criteria"""
        score = 0.0
        max_score = 5.0

        # 1. Check if response contains structured data (good for robot execution)
        contains_json = bool(re.search(r'\{.*\}', response_text))
        if contains_json:
            score += 1.0

        # 2. Check if response is relevant to robot capabilities
        relevant_terms = ['go', 'move', 'navigate', 'pick', 'grasp', 'find', 'locate', 'detect']
        has_relevant_content = any(term in response_text.lower() for term in relevant_terms)
        if has_relevant_content:
            score += 1.0

        # 3. Check response clarity and specificity
        response_length = len(response_text.strip())
        if 10 < response_length < 500:  # Good length range
            score += 1.0

        # 4. Check for safety considerations
        has_safety_mention = any(safety_term in response_text.lower()
                               for safety_term in ['safe', 'careful', 'carefully', 'slowly', 'caution'])
        if has_safety_mention:
            score += 1.0

        # 5. Check for confirmation/feedback mechanisms
        has_confirmation = any(confirm_term in response_text.lower()
                            for confirm_term in ['confirm', 'sure', 'will do', 'I will', 'I can'])
        if has_confirmation:
            score += 1.0

        return min(1.0, score / max_score)  # Normalize to 0-1 range

def main(args=None):
    rclpy.init(args=args)
    node = ResponseQualityValidator()

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

### 2. Safety Validation

```python
#!/usr/bin/env python3
"""
Safety Validator Node
"""
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
import json

class SafetyValidatorNode(Node):
    def __init__(self):
        super().__init__('safety_validator_node')

        self.execution_command_sub = self.create_subscription(
            String, '/execution_command', self.execution_command_callback, 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.safe_command_pub = self.create_publisher(Twist, '/cmd_vel_safe', 10)
        self.safety_alert_pub = self.create_publisher(String, '/safety_alerts', 10)

        self.latest_scan = None
        self.safety_threshold = 0.5  # 50cm safety distance

        self.get_logger().info('Safety validator started')

    def execution_command_callback(self, msg):
        """Validate execution commands for safety"""
        try:
            command_data = json.loads(msg.data)

            # Check if command is safe to execute
            is_safe, reason = self.validate_command_safety(command_data)

            if is_safe:
                # Forward command to safe execution
                self.forward_safe_command(command_data)
            else:
                # Block unsafe command and alert
                self.get_logger().error(f'Blocked unsafe command: {reason}')

                alert_msg = String()
                alert_msg.data = json.dumps({
                    'type': 'unsafe_command_blocked',
                    'command': command_data,
                    'reason': reason,
                    'timestamp': time.time()
                })
                self.safety_alerts_pub.publish(alert_msg)

        except Exception as e:
            self.get_logger().error(f'Error validating command: {e}')

    def validate_command_safety(self, command_data):
        """Validate if command is safe to execute"""
        action = command_data.get('action', 'unknown')

        # Check for navigation safety
        if action in ['navigate_to', 'go_to', 'move_to']:
            target_location = command_data.get('parameters', {}).get('location', 'unknown')

            # In practice, check if path to target is safe
            # For this example, we'll check if there are obstacles ahead
            if self.latest_scan:
                min_range = min(self.latest_scan.ranges) if self.latest_scan.ranges else float('inf')

                if min_range < self.safety_threshold:
                    return False, f'Obstacle too close: {min_range:.2f}m < {self.safety_threshold}m threshold'

        # Check for manipulation safety
        elif action in ['grasp', 'pick_up', 'place']:
            # Check if manipulation target is safe
            # For example, check if object is too hot, fragile, etc.
            pass

        return True, 'Command is safe'

    def forward_safe_command(self, command_data):
        """Forward validated safe command"""
        # In practice, this would send the command to the appropriate executor
        # For this example, we'll just log that command was validated
        self.get_logger().info(f'Safe command forwarded: {command_data.get("action", "unknown")}')

def main(args=None):
    rclpy.init(args=args)
    node = SafetyValidatorNode()

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

### 1. LLM Integration Best Practices

```python
# Good: Proper error handling and fallbacks
def good_llm_integration():
    """Best practices for LLM integration"""
    # Use appropriate model for task
    # Implement proper error handling
    # Have fallback mechanisms
    # Monitor API usage
    # Validate outputs
    # Cache deterministic responses
    pass

# Bad: Poor error handling
def bad_llm_integration():
    """Poor practices in LLM integration"""
    # No error handling
    # No fallbacks
    # No output validation
    # No rate limiting
    # No monitoring
    # Blocking calls without timeouts
    pass
```

### 2. Context Management Best Practices

```python
# Good: Proper context management
def good_context_management():
    """Best practices for context management"""
    # Limit context length
    # Clean up old context
    # Use efficient data structures
    # Maintain conversation history
    # Handle context switching
    pass

# Bad: Poor context management
def bad_context_management():
    """Poor practices in context management"""
    # Unlimited context growth
    # No cleanup
    # Inefficient storage
    # No conversation tracking
    # Memory leaks
    pass
```

### 3. Performance Best Practices

```python
# Good: Performance-optimized system
def good_performance_practices():
    """Best practices for performance"""
    # Use appropriate model sizes
    # Implement caching
    # Optimize API calls
    # Use async processing where possible
    # Monitor performance metrics
    # Implement rate limiting
    pass

# Bad: Performance issues
def bad_performance_practices():
    """Poor practices for performance"""
    # Always use largest models
    # No caching
    # Excessive API calls
    # No performance monitoring
    # Blocking operations
    # No resource management
    pass
```

## Common Issues and Troubleshooting

### 1. API Issues

```bash
# Check API key
echo $OPENAI_API_KEY

# Test API connectivity
curl -H "Authorization: Bearer $OPENAI_API_KEY" \
     -H "Content-Type: application/json" \
     https://api.openai.com/v1/models
```

### 2. Context Issues

```python
# Monitor context length
def monitor_context_length(context_messages):
    """Monitor context length to avoid exceeding limits"""
    total_tokens = sum(len(msg['content'].split()) for msg in context_messages)
    max_tokens = 4096  # Example limit

    if total_tokens > max_tokens * 0.8:  # 80% threshold
        print(f'Warning: Context approaching limit ({total_tokens}/{max_tokens})')
```

### 3. Quality Issues

```python
# Validate response quality
def validate_response_quality(response):
    """Validate LLM response quality"""
    if not response or len(response.strip()) == 0:
        return False, "Empty response"

    if len(response) > 1000:  # Too long
        return False, "Response too long"

    # Additional validation checks...

    return True, "Valid response"
```

## Next Steps

Now that you have built a complete LLM-powered robot assistant, continue to [Week 12: Vision-Language-Action Models](../../module-4-vision-language-action/week-12/introduction) to learn about multimodal AI models that integrate vision, language, and action.

## Exercises

1. Enhance the assistant with multimodal capabilities (vision + language)
2. Implement a learning mechanism that improves with user feedback
3. Add emotional intelligence for more natural interactions
4. Create a task learning system that can acquire new skills through demonstration