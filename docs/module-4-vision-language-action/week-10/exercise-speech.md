---
sidebar_position: 5
---

# Exercise: Complete Speech System

In this comprehensive exercise, you'll create a complete speech system that integrates text-to-speech, natural language generation, and dialogue management. This will demonstrate how to build a cohesive speech processing pipeline for robot interaction.

## Objective

Create a complete speech system that:
1. **Processes natural language input** from users
2. **Generates appropriate responses** using NLG
3. **Synthesizes speech** with natural voice output
4. **Manages dialogue flow** for coherent conversations
5. **Assesses speech quality** for continuous improvement

## Prerequisites

- Complete Week 1-10 lessons
- ROS 2 workspace set up (`~/ros2_ws`)
- Understanding of speech processing and NLG
- Basic Python and C++ programming skills
- Completed audio processing and NLU exercises

## Step 1: Create the Speech System Package

```bash
cd ~/ros2_ws/src

ros2 pkg create --build-type ament_python complete_speech_system \
    --dependencies rclpy std_msgs sensor_msgs geometry_msgs audio_common_msgs vision_msgs tf2_ros cv_bridge message_filters
```

## Step 2: Create the Speech Manager Node

Create `complete_speech_system/complete_speech_system/speech_manager.py`:

```python
#!/usr/bin/env python3
"""
Complete Speech System Manager
"""
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from audio_common_msgs.msg import AudioData
from vision_msgs.msg import Detection2DArray
from message_filters import ApproximateTimeSynchronizer, Subscriber
from threading import Thread, Lock
import time
import numpy as np
import queue

class SpeechSystemManager(Node):
    def __init__(self):
        super().__init__('speech_system_manager')

        # Create subscribers for different input sources
        self.user_input_sub = self.create_subscription(String, '/speech/user_input', self.user_input_callback, 10)
        self.nlu_sub = self.create_subscription(String, '/nlu/intent', self.nlu_callback, 10)
        self.perception_sub = self.create_subscription(Detection2DArray, '/perception/detections', self.perception_callback, 10)
        self.robot_state_sub = self.create_subscription(PoseStamped, '/robot/pose', self.robot_state_callback, 10)

        # Create publishers for speech system outputs
        self.tts_input_pub = self.create_publisher(String, '/tts/input', 10)
        self.speech_status_pub = self.create_publisher(String, '/speech/status', 10)
        self.dialogue_state_pub = self.create_publisher(String, '/dialogue/state', 10)

        # Initialize speech components
        self.tts_component = TTSComponent(self)
        self.nlg_component = NLGComponent(self)
        self.dialogue_manager = DialogueManager(self)

        # System state
        self.robot_state = {
            'location': 'unknown',
            'battery_level': 100.0,
            'current_task': 'idle',
            'perception_data': {}
        }

        # Processing queues
        self.response_queue = queue.Queue()
        self.processing_thread = Thread(target=self.process_responses, daemon=True)
        self.processing_thread.start()

        # Quality metrics
        self.quality_metrics = {
            'response_time': 0.0,
            'naturalness_score': 0.0,
            'intelligibility_score': 0.0,
            'user_satisfaction': 0.0
        }

        self.get_logger().info('Complete speech system manager started')

    def user_input_callback(self, msg):
        """Handle user speech input"""
        try:
            user_text = msg.data

            # Update dialogue manager with user input
            self.dialogue_manager.update_context('user_input', user_text)

            # Process with NLU if not already processed
            if not self.is_nlu_processed_recently(user_text):
                # In a real system, this would be handled by NLU node
                # For this exercise, we'll simulate processing
                intent_result = self.simulate_nlu_processing(user_text)
                self.nlu_callback(intent_result)

            self.get_logger().info(f'User input received: {user_text}')

        except Exception as e:
            self.get_logger().error(f'Error processing user input: {e}')

    def nlu_callback(self, msg):
        """Handle NLU results"""
        try:
            # In practice, this would receive structured NLU results
            # For this example, we'll parse a JSON string
            import json
            nlu_data = json.loads(msg.data)

            # Update dialogue manager with NLU results
            self.dialogue_manager.update_context('nlu_result', nlu_data)

            # Generate appropriate response
            response = self.dialogue_manager.generate_response(nlu_data)

            # Queue response for processing
            self.response_queue.put(response)

            self.get_logger().info(f'NLU processed: {nlu_data.get("intent", "unknown")}')

        except Exception as e:
            self.get_logger().error(f'Error processing NLU: {e}')

    def perception_callback(self, msg):
        """Handle perception results"""
        try:
            # Update robot state with perception data
            self.robot_state['perception_data'] = self.process_perception_data(msg)

            # Update dialogue context with perception
            self.dialogue_manager.update_context('perception', self.robot_state['perception_data'])

            self.get_logger().info(f'Perception data updated: {len(msg.detections)} detections')

        except Exception as e:
            self.get_logger().error(f'Error processing perception: {e}')

    def robot_state_callback(self, msg):
        """Handle robot state updates"""
        try:
            self.robot_state['location'] = self.get_location_name(msg.pose)
            self.dialogue_manager.update_context('robot_state', self.robot_state)

            self.get_logger().info(f'Robot location updated: {self.robot_state["location"]}')

        except Exception as e:
            self.get_logger().error(f'Error processing robot state: {e}')

    def process_perception_data(self, detection_array):
        """Process perception data for speech system"""
        perception_info = {
            'objects_detected': len(detection_array.detections),
            'object_types': [],
            'nearest_object': None,
            'object_distances': []
        }

        if detection_array.detections:
            # Process first few detections (simplified)
            for detection in detection_array.detections[:5]:  # Limit for performance
                if hasattr(detection, 'results') and detection.results:
                    for result in detection.results:
                        if hasattr(result, 'hypothesis'):
                            class_id = result.hypothesis.class_id
                            confidence = result.hypothesis.score
                            if confidence > 0.5:  # Confidence threshold
                                perception_info['object_types'].append(class_id)

        return perception_info

    def get_location_name(self, pose):
        """Convert pose to location name"""
        x, y = pose.position.x, pose.position.y

        if abs(x) < 0.5 and abs(y) < 0.5:
            return 'home base'
        elif x > 2.0:
            return 'kitchen'
        elif x < -2.0:
            return 'office'
        elif y > 2.0:
            return 'living room'
        elif y < -2.0:
            return 'bedroom'
        else:
            return 'unknown location'

    def process_responses(self):
        """Process responses in separate thread"""
        while rclpy.ok():
            try:
                # Get response from queue
                response = self.response_queue.get(timeout=1.0)

                # Generate speech
                self.tts_component.synthesize_speech(response)

                # Update quality metrics
                self.update_quality_metrics(response)

            except queue.Empty:
                continue
            except Exception as e:
                self.get_logger().error(f'Error in response processing: {e}')

    def update_quality_metrics(self, response):
        """Update speech quality metrics"""
        # Calculate response time
        response_time = time.time() - self.last_input_time if hasattr(self, 'last_input_time') else 0.0
        self.quality_metrics['response_time'] = response_time

        # Update naturalness score (simplified)
        self.quality_metrics['naturalness_score'] = self.estimate_naturalness(response)

        # Log metrics
        self.get_logger().info(
            f'Speech Quality - Response time: {response_time:.3f}s, '
            f'Naturalness: {self.quality_metrics["naturalness_score"]:.3f}'
        )

    def estimate_naturalness(self, text):
        """Estimate naturalness of generated text"""
        # Simple heuristic for naturalness
        # In practice, use more sophisticated measures
        words = text.split()

        if len(words) < 2:
            return 0.1  # Too short

        if len(words) > 20:
            return 0.3  # Too long/verbose

        # Check for natural language patterns
        natural_indicators = ['the', 'and', 'is', 'are', 'was', 'were', 'it', 'you', 'i']
        natural_word_ratio = sum(1 for word in words if word.lower() in natural_indicators) / len(words)

        # Prefer responses with good natural word ratio
        return min(1.0, natural_word_ratio * 2.0 + 0.3)

    def is_nlu_processed_recently(self, text):
        """Check if text was recently processed by NLU"""
        # This would check against recent processing history
        # For this example, return False to always process
        return False

    def simulate_nlu_processing(self, text):
        """Simulate NLU processing (in real system, this would be separate node)"""
        import json

        # Simple intent classification (in practice, use ML model)
        text_lower = text.lower()

        if any(word in text_lower for word in ['hello', 'hi', 'hey']):
            intent = 'greeting'
        elif any(word in text_lower for word in ['goodbye', 'bye', 'see you']):
            intent = 'farewell'
        elif any(word in text_lower for word in ['go', 'move', 'navigate', 'drive']):
            intent = 'navigation'
        elif any(word in text_lower for word in ['what', 'how', 'where', 'when', 'who']):
            intent = 'question'
        else:
            intent = 'command'

        nlu_result = {
            'intent': intent,
            'confidence': 0.8,
            'entities': [],
            'original_text': text
        }

        # Create String message with JSON
        result_msg = String()
        result_msg.data = json.dumps(nlu_result)

        return result_msg

def main(args=None):
    rclpy.init(args=args)
    node = SpeechSystemManager()

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

## Step 3: Create Text-to-Speech Component

Create `complete_speech_system/complete_speech_system/tts_component.py`:

```python
#!/usr/bin/env python3
"""
Text-to-Speech Component
"""
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32
from audio_common_msgs.msg import AudioData
from threading import Lock
import time

class TTSComponent:
    def __init__(self, parent_node):
        self.parent_node = parent_node
        self.node = parent_node  # Use parent's context

        # Create publisher for audio output
        self.audio_pub = self.node.create_publisher(AudioData, '/tts/output', 10)
        self.status_pub = self.node.create_publisher(String, '/tts/status', 10)
        self.quality_pub = self.node.create_publisher(Float32, '/tts/quality_score', 10)

        # TTS configuration
        self.voice_settings = {
            'pitch': 1.0,
            'speed': 1.0,
            'volume': 0.8,
            'voice_type': 'neural'  # neural, concatenative, parametric
        }

        # Processing state
        self.is_speaking = False
        self.speech_lock = Lock()

        # Performance metrics
        self.metrics = {
            'processing_time': 0.0,
            'quality_score': 0.0,
            'naturalness_score': 0.0
        }

        self.node.get_logger().info('TTS component initialized')

    def synthesize_speech(self, text):
        """Synthesize speech from text"""
        with self.speech_lock:
            if self.is_speaking:
                self.node.get_logger().warn('Speech in progress, queuing new request')
                # In a real system, implement queuing
                return

            self.is_speaking = True
            start_time = time.time()

            try:
                # Process text
                processed_text = self.preprocess_text(text)

                # Generate audio (simplified - in practice, use actual TTS engine)
                audio_data = self.generate_audio_from_text(processed_text)

                # Post-process audio
                processed_audio = self.postprocess_audio(audio_data)

                # Publish audio
                self.publish_audio(processed_audio)

                # Calculate and publish quality metrics
                processing_time = time.time() - start_time
                quality_score = self.calculate_quality_score(processed_text, processing_time)

                self.metrics['processing_time'] = processing_time
                self.metrics['quality_score'] = quality_score

                # Publish metrics
                quality_msg = Float32()
                quality_msg.data = quality_score
                self.quality_pub.publish(quality_msg)

                status_msg = String()
                status_msg.data = f"TTS_COMPLETE:duration={processing_time:.3f}:quality={quality_score:.3f}"
                self.status_pub.publish(status_msg)

                self.node.get_logger().info(
                    f'TTS completed: "{processed_text[:50]}..." '
                    f'(duration: {processing_time:.3f}s, quality: {quality_score:.3f})'
                )

            except Exception as e:
                self.node.get_logger().error(f'TTS synthesis error: {e}')
                status_msg = String()
                status_msg.data = f"TTS_ERROR:{str(e)}"
                self.status_pub.publish(status_msg)

            finally:
                self.is_speaking = False

    def preprocess_text(self, text):
        """Preprocess text for TTS"""
        # Normalize text
        normalized = text.strip()

        # Handle abbreviations and numbers
        normalized = self.expand_abbreviations(normalized)
        normalized = self.expand_numbers(normalized)

        # Apply voice settings
        if self.voice_settings['pitch'] != 1.0:
            # Apply pitch modification (conceptual)
            pass

        if self.voice_settings['speed'] != 1.0:
            # Apply speed modification (conceptual)
            pass

        return normalized

    def expand_abbreviations(self, text):
        """Expand common abbreviations"""
        abbreviations = {
            'mr.': 'mister',
            'mrs.': 'missus',
            'dr.': 'doctor',
            'st.': 'street',
            'ave.': 'avenue',
            'etc.': 'et cetera',
            'vs.': 'versus',
            'ie.': 'that is',
            'eg.': 'for example'
        }

        for abbrev, expansion in abbreviations.items():
            text = text.replace(abbrev, expansion)
            text = text.replace(abbrev.capitalize(), expansion.capitalize())

        return text

    def expand_numbers(self, text):
        """Expand numbers to words (simplified)"""
        import re

        def number_to_words(num):
            # Simplified number to words conversion
            # In practice, use a proper library like inflect
            try:
                n = int(num)
                if 0 <= n <= 999:
                    # Basic implementation for small numbers
                    ones = ['', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
                    teens = ['ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen']
                    tens = ['', '', 'twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety']

                    if n < 10:
                        return ones[n]
                    elif n < 20:
                        return teens[n-10]
                    elif n < 100:
                        return tens[n//10] + ('' if n%10 == 0 else '-' + ones[n%10])
                    else:  # n < 1000
                        result = ones[n//100] + ' hundred'
                        remainder = n % 100
                        if remainder > 0:
                            result += ' ' + self.number_to_words(remainder)
                        return result
                return num  # Return original if not in range
            except ValueError:
                return num  # Return original if not a number

        # Find and replace numbers
        pattern = r'\b\d+\b'
        expanded_text = re.sub(pattern, lambda m: number_to_words(m.group()), text)
        return expanded_text

    def generate_audio_from_text(self, text):
        """Generate audio from text (simplified implementation)"""
        # In a real system, this would use an actual TTS engine
        # For this example, we'll simulate audio generation

        sample_rate = 22050
        duration_per_char = 0.05  # 50ms per character (simplified)

        # Estimate duration
        estimated_duration = len(text) * duration_per_char
        estimated_duration = max(0.5, estimated_duration)  # Minimum 0.5 seconds

        # Generate simulated audio (simplified)
        t = np.linspace(0, estimated_duration, int(sample_rate * estimated_duration))

        # Create basic speech-like waveform (simplified)
        # In practice, use proper TTS model
        base_freq = 200  # Base frequency for speech
        mod_freq = 5     # Modulation frequency for intonation

        # Create modulated carrier
        carrier = np.sin(2 * np.pi * base_freq * t)
        modulation = np.sin(2 * np.pi * mod_freq * t)

        # Apply amplitude modulation based on text characteristics
        amplitude_env = 0.5 + 0.5 * np.sin(np.linspace(0, 2*np.pi, len(t)) * len(text) / 10)

        audio_signal = carrier * amplitude_env * (1 + 0.3 * modulation)

        # Apply basic filtering to make it sound more natural
        b, a = signal.butter(4, [100/(sample_rate/2), 4000/(sample_rate/2)], 'band')
        audio_signal = signal.filtfilt(b, a, audio_signal)

        # Normalize
        audio_signal = audio_signal / np.max(np.abs(audio_signal)) * 0.8  # 80% of max amplitude

        # Convert to int16
        audio_int16 = (audio_signal * 32767).astype(np.int16)

        return audio_int16.tobytes()

    def postprocess_audio(self, audio_bytes):
        """Post-process generated audio"""
        # Convert bytes back to numpy array for processing
        audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
        audio_float = audio_int16.astype(np.float32) / 32768.0

        # Apply voice settings
        if self.voice_settings['volume'] != 1.0:
            audio_float = audio_float * self.voice_settings['volume']

        # Apply basic equalization for better quality
        audio_enhanced = self.apply_basic_equalization(audio_float)

        # Convert back to int16 bytes
        audio_enhanced_int16 = (audio_enhanced * 32767).astype(np.int16)
        return audio_enhanced_int16.tobytes()

    def apply_basic_equalization(self, audio_signal):
        """Apply basic equalization to improve speech quality"""
        # Simple equalization to enhance speech clarity
        from scipy import signal

        sample_rate = 22050

        # Apply pre-emphasis filter (boosts high frequencies)
        b_pre = [1.0, -0.68]  # Simple first-order high-pass
        a_pre = [1.0, 0.0]
        audio_pre = signal.lfilter(b_pre, a_pre, audio_signal)

        # Apply mild de-emphasis to balance
        b_post = [0.8, 0.0]   # Simple gain adjustment
        a_post = [1.0, -0.3]
        audio_final = signal.lfilter(b_post, a_post, audio_pre)

        return audio_final

    def calculate_quality_score(self, text, processing_time):
        """Calculate quality score for generated speech"""
        # Factors affecting quality:
        # 1. Text length (should be reasonable)
        text_length_score = min(1.0, len(text) / 100.0)  # Good up to 100 characters

        # 2. Processing time (should be reasonable)
        time_score = max(0.0, min(1.0, 1.0 - (processing_time - 0.1) / 2.0))  # Good if < 2.1s for typical text

        # 3. Text complexity (simpler is better for synthesis)
        word_complexity_score = self.assess_text_complexity(text)

        # Weighted combination
        quality = 0.4 * text_length_score + 0.3 * time_score + 0.3 * word_complexity_score

        return quality

    def assess_text_complexity(self, text):
        """Assess text complexity for synthesis quality"""
        words = text.split()
        if not words:
            return 1.0

        # Calculate average word length (longer words are harder to pronounce)
        avg_word_length = sum(len(word) for word in words) / len(words)
        complexity_score = max(0.0, 1.0 - (avg_word_length - 5.0) / 10.0)  # Good for words < 15 chars

        # Check for special characters that might cause issues
        special_chars = sum(1 for c in text if c in '!@#$%^&*()[]{}|\\:";<>?,./')
        special_char_ratio = special_chars / len(text) if text else 0
        special_char_score = max(0.0, 1.0 - special_char_ratio * 5)  # Penalize special chars

        # Combine scores
        final_score = 0.7 * complexity_score + 0.3 * special_char_score
        return final_score

    def set_voice_parameters(self, pitch=1.0, speed=1.0, volume=0.8):
        """Set voice parameters"""
        self.voice_settings.update({
            'pitch': pitch,
            'speed': speed,
            'volume': volume
        })
        self.node.get_logger().info(f'Voice settings updated: pitch={pitch}, speed={speed}, volume={volume}')

    def publish_audio(self, audio_data):
        """Publish generated audio"""
        audio_msg = AudioData()
        audio_msg.data = audio_data
        self.audio_pub.publish(audio_msg)

def main(args=None):
    rclpy.init(args=args)

    # This component is meant to be used within the speech manager
    # For standalone testing, create a minimal node
    node = Node('tts_test_node')

    # Test TTS component
    speech_manager = SpeechSystemManager()
    tts_comp = TTSComponent(speech_manager)

    # Test synthesis
    test_text = "Hello, this is a test of the text to speech system."
    tts_comp.synthesize_speech(test_text)

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Step 4: Create Natural Language Generation Component

Create `complete_speech_system/complete_speech_system/nlg_component.py`:

```python
#!/usr/bin/env python3
"""
Natural Language Generation Component
"""
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import random
from string import Template
import json

class NLGComponent:
    def __init__(self, parent_node):
        self.parent_node = parent_node
        self.node = parent_node  # Use parent's context

        # Create publisher for generated responses
        self.response_pub = self.node.create_publisher(String, '/nlg/response', 10)

        # NLG templates and patterns
        self.templates = {
            'greeting': [
                'Hello! How can I assist you today?',
                'Hi there! Ready to help with your tasks.',
                'Greetings! What would you like me to do?',
                'Good day! How may I be of service?'
            ],
            'farewell': [
                'Goodbye! Have a wonderful day.',
                'Farewell! See you later.',
                'Take care! Until next time.',
                'Bye! Thanks for interacting with me.'
            ],
            'navigation': [
                'I will navigate to the $location now.',
                'Heading to $location as requested.',
                'On my way to $location.',
                'Navigating to $location for you.'
            ],
            'information': [
                'The $object is located at $location.',
                'I found $object at $location.',
                'Based on my sensors, $object is at $location.',
                'According to my perception, $object is located at $location.'
            ],
            'confirmation': [
                'I understand. I will $action.',
                'Got it. Proceeding with $action.',
                'Confirmed. Executing $action now.',
                'Understood. $action is being performed.'
            ],
            'error': [
                'I encountered an issue: $error. Could you please try again?',
                'Sorry, I had trouble with that: $error.',
                'I experienced a problem: $error. How else can I help?',
                'There was an issue: $error. Please rephrase your request.'
            ],
            'uncertain': [
                'I\'m not sure I understood. Could you please rephrase that?',
                'I didn\'t quite catch that. Could you say it differently?',
                'I\'m confused. Could you please clarify?',
                'I didn\'t understand that. Can you rephrase your request?'
            ]
        }

        # Context-dependent responses
        self.context_responses = {
            'battery_low': [
                'My battery is running low at $level%. I should charge soon.',
                'Battery level is $level%. Initiating return to charging station.',
                'Warning: Battery at $level%. Charging recommended.'
            ],
            'perception_success': [
                'I see $count objects in the environment.',
                'Detected $count items in my field of view.',
                'Found $count objects in the scene.'
            ],
            'task_completion': [
                '$task has been completed successfully.',
                'Successfully finished $task.',
                '$task is done as requested.'
            ]
        }

        self.node.get_logger().info('NLG component initialized')

    def generate_response(self, intent_data, context=None):
        """Generate natural language response based on intent and context"""
        intent = intent_data.get('intent', 'unknown')
        entities = intent_data.get('entities', [])
        original_text = intent_data.get('original_text', '')

        # Select template based on intent
        if intent in self.templates:
            template_list = self.templates[intent]
        else:
            template_list = self.templates['uncertain']

        # Select random template from available options
        template_str = random.choice(template_list)

        # Prepare substitution variables
        variables = {
            'location': self.extract_location(entities),
            'object': self.extract_object(entities),
            'action': self.extract_action(entities),
            'error': intent_data.get('error', 'unknown issue'),
            'count': intent_data.get('count', 0),
            'task': intent_data.get('task', 'requested task')
        }

        # Add context-specific variables
        if context:
            variables.update(context)

        # Substitute variables in template
        template = Template(template_str)
        try:
            response = template.substitute(variables)
        except KeyError as e:
            self.node.get_logger().warn(f'Variable {e} not found in template, using original')
            response = template_str

        # Apply post-processing
        response = self.post_process_response(response)

        # Publish response
        response_msg = String()
        response_msg.data = response
        self.response_pub.publish(response_msg)

        return response

    def extract_location(self, entities):
        """Extract location from entities"""
        for entity in entities:
            if entity.get('type') == 'LOCATION':
                return entity.get('value', 'unknown location')
        # If no location entity found, try to infer from context
        return 'the requested location'

    def extract_object(self, entities):
        """Extract object from entities"""
        for entity in entities:
            if entity.get('type') in ['OBJECT', 'ITEM', 'THING']:
                return entity.get('value', 'an object')
        return 'something'

    def extract_action(self, entities):
        """Extract action from entities"""
        for entity in entities:
            if entity.get('type') == 'ACTION':
                return entity.get('value', 'the requested action')
        return 'the requested action'

    def post_process_response(self, response):
        """Apply post-processing to generated response"""
        # Capitalize first letter
        if response:
            response = response[0].upper() + response[1:] if len(response) > 1 else response.upper()

        # Ensure proper punctuation
        if not response.endswith(('.', '!', '?')):
            response += '.'

        # Remove extra spaces
        response = ' '.join(response.split())

        return response

    def generate_contextual_response(self, context_type, context_data):
        """Generate response based on specific context"""
        if context_type in self.context_responses:
            template_list = self.context_responses[context_type]
            template_str = random.choice(template_list)

            # Prepare context variables
            variables = context_data.copy()

            # Substitute variables
            template = Template(template_str)
            try:
                response = template.substitute(variables)
            except KeyError as e:
                self.node.get_logger().warn(f'Context variable {e} not found, using template')
                response = template_str

            # Post-process
            response = self.post_process_response(response)

            # Publish
            response_msg = String()
            response_msg.data = response
            self.response_pub.publish(response_msg)

            return response

        return "I have context but don't know how to respond to it."

    def generate_robot_status_response(self, robot_state):
        """Generate response about robot status"""
        responses = []

        # Battery status
        battery_level = robot_state.get('battery_level', 100.0)
        if battery_level < 20:
            responses.append(f"My battery is critically low at {battery_level:.1f}%.")
        elif battery_level < 50:
            responses.append(f"My battery level is at {battery_level:.1f}%.")
        else:
            responses.append(f"My battery is at {battery_level:.1f}%.")

        # Location
        location = robot_state.get('location', 'an unknown location')
        responses.append(f"I am currently at {location}.")

        # Current task
        current_task = robot_state.get('current_task', 'idle')
        if current_task != 'idle':
            responses.append(f"I am currently performing {current_task}.")

        # Combine responses
        return ' '.join(responses)

    def generate_perception_response(self, perception_data):
        """Generate response about perception results"""
        object_count = perception_data.get('objects_detected', 0)

        if object_count == 0:
            return "I don't see any objects in my current field of view."
        elif object_count == 1:
            return "I see one object in the environment."
        else:
            object_types = perception_data.get('object_types', [])
            if object_types:
                unique_types = list(set(object_types))
                if len(unique_types) == 1:
                    return f"I see {object_count} {unique_types[0]} objects."
                else:
                    type_str = ", ".join(unique_types[:3])  # Limit to first 3 types
                    if len(unique_types) > 3:
                        type_str += f", and {len(unique_types) - 3} other types"
                    return f"I see {object_count} objects including {type_str}."
            else:
                return f"I see {object_count} objects in the environment."

def main(args=None):
    rclpy.init(args=args)

    # Test NLG component
    node = Node('nlg_test_node')
    nlg = NLGComponent(node)

    # Test different intents
    test_cases = [
        {'intent': 'greeting', 'entities': [], 'original_text': 'hello'},
        {'intent': 'navigation', 'entities': [{'type': 'LOCATION', 'value': 'kitchen'}], 'original_text': 'go to kitchen'},
        {'intent': 'information', 'entities': [{'type': 'OBJECT', 'value': 'cup'}, {'type': 'LOCATION', 'value': 'table'}], 'original_text': 'where is the cup'}
    ]

    for test_case in test_cases:
        response = nlg.generate_response(test_case)
        node.get_logger().info(f'Test case: {test_case["intent"]} -> Response: {response}')

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Step 5: Create Dialogue Manager Component

Create `complete_speech_system/complete_speech_system/dialogue_manager.py`:

```python
#!/usr/bin/env python3
"""
Dialogue Manager Component
"""
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import time
from typing import Dict, Any, List
from collections import deque

class DialogueManager:
    def __init__(self, parent_node):
        self.parent_node = parent_node
        self.node = parent_node  # Use parent's context

        # Create publisher for dialogue state
        self.state_pub = self.node.create_publisher(String, '/dialogue/state', 10)

        # Initialize NLG component
        self.nlg = NLGComponent(parent_node)

        # Dialogue state
        self.dialogue_state = {
            'current_topic': 'greeting',
            'user_context': {},
            'robot_state': {},
            'dialogue_history': deque(maxlen=20),  # Keep last 20 exchanges
            'current_intent': None,
            'pending_requests': [],
            'last_interaction_time': time.time(),
            'turn_count': 0
        }

        # Multi-turn dialogue state
        self.waiting_for_response = False
        self.expected_entity_type = None
        self.pending_action = None

        # Dialogue flow management
        self.dialogue_flows = {
            'navigation': self.handle_navigation_dialogue,
            'information': self.handle_information_dialogue,
            'social': self.handle_social_dialogue,
            'task': self.handle_task_dialogue
        }

        self.node.get_logger().info('Dialogue manager initialized')

    def update_context(self, source: str, data: Any):
        """Update dialogue context with new information"""
        if source == 'user_input':
            self.dialogue_state['user_context']['last_input'] = data
            self.dialogue_state['last_interaction_time'] = time.time()
        elif source == 'nlu_result':
            self.dialogue_state['current_intent'] = data.get('intent')
        elif source == 'robot_state':
            self.dialogue_state['robot_state'] = data
        elif source == 'perception':
            self.dialogue_state['perception_data'] = data

    def generate_response(self, nlu_data: Dict[str, Any]) -> str:
        """Generate response based on NLU data and current context"""
        intent = nlu_data.get('intent', 'unknown')
        confidence = nlu_data.get('confidence', 0.0)
        entities = nlu_data.get('entities', [])
        original_text = nlu_data.get('original_text', '')

        # Check if we're waiting for a specific response (multi-turn dialogue)
        if self.waiting_for_response and self.expected_entity_type:
            return self.handle_expected_response(nlu_data)

        # Validate confidence threshold
        if confidence < 0.4:
            # Low confidence - ask for clarification
            self.waiting_for_response = False
            self.expected_entity_type = None
            return self.nlg.generate_response({
                'intent': 'uncertain',
                'entities': entities,
                'original_text': original_text
            })

        # Process based on intent
        if intent in self.dialogue_flows:
            response = self.dialogue_flows[intent](nlu_data)
        else:
            response = self.handle_general_intent(nlu_data)

        # Update dialogue history
        self.dialogue_state['dialogue_history'].append({
            'timestamp': time.time(),
            'speaker': 'user',
            'text': original_text,
            'intent': intent
        })

        # Update turn count
        self.dialogue_state['turn_count'] += 1

        # Publish dialogue state
        self.publish_dialogue_state()

        return response

    def handle_navigation_dialogue(self, nlu_data):
        """Handle navigation-related dialogue"""
        entities = nlu_data.get('entities', [])

        # Check if destination is specified
        destination = self.extract_destination(entities)

        if destination:
            # We have a destination, proceed with navigation
            response = self.nlg.generate_response({
                'intent': 'navigation',
                'entities': entities,
                'original_text': nlu_data.get('original_text', '')
            })

            # In a real system, this would trigger navigation
            # For this exercise, we'll just acknowledge

            # Update dialogue state
            self.dialogue_state['current_topic'] = 'navigation'
            self.dialogue_state['pending_navigation'] = destination

            return response
        else:
            # Ask for destination
            self.waiting_for_response = True
            self.expected_entity_type = 'LOCATION'
            self.pending_action = 'navigate'

            return "Where would you like me to go?"

    def handle_information_dialogue(self, nlu_data):
        """Handle information-seeking dialogue"""
        entities = nlu_data.get('entities', [])
        original_text = nlu_data.get('original_text', '').lower()

        # Check for specific information requests
        if 'battery' in original_text or 'power' in original_text:
            battery_level = self.dialogue_state['robot_state'].get('battery_level', 100.0)
            return f"My battery level is {battery_level:.1f}%."
        elif 'location' in original_text or 'where' in original_text:
            location = self.dialogue_state['robot_state'].get('location', 'unknown')
            return f"I am currently at {location}."
        elif 'time' in original_text or 'hour' in original_text:
            import datetime
            current_time = datetime.datetime.now().strftime("%H:%M")
            return f"The current time is {current_time}."
        elif 'objects' in original_text or 'see' in original_text:
            perception_data = self.dialogue_state.get('perception_data', {})
            return self.nlg.generate_perception_response(perception_data)
        else:
            # General information request - use NLG
            return self.nlg.generate_response(nlu_data)

    def handle_social_dialogue(self, nlu_data):
        """Handle social interaction dialogue"""
        original_text = nlu_data.get('original_text', '').lower()

        if any(word in original_text for word in ['hello', 'hi', 'hey']):
            return self.nlg.generate_response({
                'intent': 'greeting',
                'entities': nlu_data.get('entities', []),
                'original_text': original_text
            })
        elif any(word in original_text for word in ['thank', 'thanks']):
            return "You're welcome! Is there anything else I can help with?"
        elif any(word in original_text for word in ['goodbye', 'bye', 'see you']):
            return self.nlg.generate_response({
                'intent': 'farewell',
                'entities': nlu_data.get('entities', []),
                'original_text': original_text
            })
        else:
            return self.nlg.generate_response(nlu_data)

    def handle_task_dialogue(self, nlu_data):
        """Handle task-related dialogue"""
        entities = nlu_data.get('entities', [])

        # For this example, we'll acknowledge the task request
        # In a real system, this would interface with task execution
        task_description = self.extract_task_description(entities, nlu_data.get('original_text', ''))

        if task_description:
            # Acknowledge task
            response = f"I understand you want me to {task_description}. I'll work on that for you."

            # Update dialogue state
            self.dialogue_state['current_topic'] = 'task_execution'
            self.dialogue_state['current_task'] = task_description

            return response
        else:
            return "I can help with various tasks. What would you like me to do?"

    def handle_general_intent(self, nlu_data):
        """Handle general/unclassified intents"""
        intent = nlu_data.get('intent', 'unknown')
        confidence = nlu_data.get('confidence', 0.0)

        if confidence < 0.3:
            # Very low confidence - admit uncertainty
            return self.nlg.generate_response({
                'intent': 'uncertain',
                'entities': nlu_data.get('entities', []),
                'original_text': nlu_data.get('original_text', '')
            })
        else:
            # Use NLG to generate appropriate response
            return self.nlg.generate_response(nlu_data)

    def handle_expected_response(self, nlu_data):
        """Handle responses to expected queries (multi-turn dialogue)"""
        entities = nlu_data.get('entities', [])
        original_text = nlu_data.get('original_text', '')

        if self.expected_entity_type == 'LOCATION':
            # Look for location in entities
            location = self.extract_destination(entities)

            if location:
                # Got expected location
                self.waiting_for_response = False
                self.expected_entity_type = None

                if self.pending_action == 'navigate':
                    # Generate navigation response
                    response = self.nlg.generate_response({
                        'intent': 'navigation',
                        'entities': [{'type': 'LOCATION', 'value': location}],
                        'original_text': original_text
                    })

                    # In real system, trigger navigation
                    self.dialogue_state['pending_navigation'] = location

                    return response
                else:
                    # Handle other pending actions
                    return f"Okay, I'll remember that location is {location}."
            else:
                # Didn't get expected location, ask again
                return "I didn't catch the location. Where would you like me to go?"

        # Reset if unexpected response
        self.waiting_for_response = False
        self.expected_entity_type = None
        self.pending_action = None

        # Process as regular input
        return self.generate_response(nlu_data)

    def extract_destination(self, entities):
        """Extract destination from entities"""
        for entity in entities:
            if entity.get('type') in ['LOCATION', 'DESTINATION', 'PLACE']:
                return entity.get('value')

        # If no location entity, try to extract from text (simplified)
        return None

    def extract_task_description(self, entities, original_text):
        """Extract task description from entities and text"""
        # Look for action/object combinations
        action = None
        obj = None

        for entity in entities:
            if entity.get('type') == 'ACTION':
                action = entity.get('value')
            elif entity.get('type') in ['OBJECT', 'ITEM']:
                obj = entity.get('value')

        if action and obj:
            return f"{action} the {obj}"
        elif action:
            return action
        else:
            # Extract from text (simplified)
            return original_text

    def publish_dialogue_state(self):
        """Publish current dialogue state"""
        state_msg = String()
        state_msg.data = json.dumps({
            'current_topic': self.dialogue_state['current_topic'],
            'turn_count': self.dialogue_state['turn_count'],
            'waiting_for_response': self.waiting_for_response,
            'expected_entity_type': self.expected_entity_type,
            'last_interaction_time': self.dialogue_state['last_interaction_time']
        })
        self.state_pub.publish(state_msg)

    def reset_dialogue_state(self):
        """Reset dialogue to initial state"""
        self.dialogue_state = {
            'current_topic': 'greeting',
            'user_context': {},
            'robot_state': {},
            'dialogue_history': deque(maxlen=20),
            'current_intent': None,
            'pending_requests': [],
            'last_interaction_time': time.time(),
            'turn_count': 0
        }
        self.waiting_for_response = False
        self.expected_entity_type = None
        self.pending_action = None

def main(args=None):
    rclpy.init(args=args)

    # Test dialogue manager
    node = Node('dialogue_test_node')
    dm = DialogueManager(node)

    # Test different dialogue flows
    test_cases = [
        {'intent': 'greeting', 'confidence': 0.9, 'entities': [], 'original_text': 'hello'},
        {'intent': 'navigation', 'confidence': 0.8, 'entities': [{'type': 'LOCATION', 'value': 'kitchen'}], 'original_text': 'go to kitchen'},
        {'intent': 'information', 'confidence': 0.7, 'entities': [], 'original_text': 'what is your battery level'}
    ]

    for i, test_case in enumerate(test_cases):
        response = dm.generate_response(test_case)
        node.get_logger().info(f'Test {i+1}: {test_case["intent"]} -> Response: {response}')

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Step 6: Create Launch File

Create `complete_speech_system/launch/speech_system.launch.py`:

```python
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # Declare launch arguments
    use_sim_time = LaunchConfiguration('use_sim_time', default='false')
    robot_name = LaunchConfiguration('robot_name', default='complete_robot')

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation time if true'
        ),
        DeclareLaunchArgument(
            'robot_name',
            default_value='complete_robot',
            description='Name of the robot'
        ),

        # Speech system manager node
        Node(
            package='complete_speech_system',
            executable='speech_manager',
            name='speech_manager',
            parameters=[
                {'use_sim_time': use_sim_time}
            ],
            remappings=[
                ('/speech/user_input', '/speech_recognition/text'),
                ('/nlu/intent', '/nlu/intent_result'),
                ('/perception/detections', '/object_detector/detections'),
                ('/robot/pose', '/robot_localization/pose'),
                ('/tts/input', '/tts/text_input'),
                ('/tts/output', '/audio/speech_output')
            ],
            output='screen'
        ),

        # In a real system, you would also launch:
        # - Speech recognition node
        # - NLU node
        # - Perception nodes
        # - Audio input/output nodes
    ])
```

## Step 7: Update Package Configuration

Update `complete_speech_system/setup.py`:

```python
from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'complete_speech_system'

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
    description='Complete speech system for robotics',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'speech_manager = complete_speech_system.speech_manager:main',
        ],
    },
)
```

## Step 8: Build and Test

Build the package:

```bash
cd ~/ros2_ws
colcon build --packages-select complete_speech_system
source install/setup.bash
```

### Test the Complete System

1. **Launch the speech system**:
```bash
# Launch the complete speech system
ros2 launch complete_speech_system speech_system.launch.py
```

2. **Test with simulated inputs**:
```bash
# Send test speech input
ros2 topic pub /speech/user_input std_msgs/String "data: 'Hello robot'"

# Send test NLU results
ros2 topic pub /nlu/intent std_msgs/String "data: '{\"intent\": \"greeting\", \"confidence\": 0.9, \"entities\": []}'"
```

3. **Monitor outputs**:
```bash
# Listen to synthesized speech
ros2 topic echo /tts/output

# Monitor dialogue state
ros2 topic echo /dialogue/state

# Monitor speech status
ros2 topic echo /speech/status
```

## System Architecture Review

The complete speech system includes:

1. **Speech Manager**: Orchestrates the entire system
2. **TTS Component**: Generates speech from text
3. **NLG Component**: Creates natural responses
4. **Dialogue Manager**: Manages conversation flow
5. **Quality Assessment**: Evaluates speech quality

## Performance Optimization

### Real-time Processing Considerations

```python
#!/usr/bin/env python3
"""
Performance-Optimized Speech Processing
"""
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from threading import Thread, Lock
import time
from queue import Queue

class OptimizedSpeechNode(Node):
    def __init__(self):
        super().__init__('optimized_speech_node')

        # Create subscriber with appropriate QoS
        self.input_sub = self.create_subscription(
            String, '/speech/input', self.input_callback, 10)

        # Create publisher
        self.output_pub = self.create_publisher(String, '/speech/output', 10)

        # Processing queue for real-time performance
        self.input_queue = Queue(maxsize=10)
        self.output_queue = Queue(maxsize=10)

        # Processing thread
        self.processing_thread = Thread(target=self.processing_worker, daemon=True)
        self.processing_thread.start()

        # Performance metrics
        self.processing_times = []
        self.target_rate = 10.0  # Hz

        self.get_logger().info('Optimized speech node started')

    def input_callback(self, msg):
        """Non-blocking input handler"""
        try:
            # Add to processing queue (non-blocking)
            if not self.input_queue.full():
                self.input_queue.put_nowait(msg)
            else:
                self.get_logger().warn('Input queue full, dropping message')
        except:
            self.get_logger().error('Error adding to input queue')

    def processing_worker(self):
        """Dedicated processing thread"""
        while rclpy.ok():
            try:
                # Get input from queue (with timeout)
                msg = self.input_queue.get(timeout=0.1)

                # Process input
                start_time = time.time()
                response = self.generate_response(msg.data)
                processing_time = time.time() - start_time

                # Store performance metric
                self.processing_times.append(processing_time)
                if len(self.processing_times) > 100:
                    self.processing_times.pop(0)

                # Publish result
                response_msg = String()
                response_msg.data = response
                self.output_pub.publish(response_msg)

            except Exception as e:
                self.get_logger().error(f'Processing error: {e}')

    def generate_response(self, input_text):
        """Generate response with optimized processing"""
        # Use optimized NLG and TTS methods
        # In practice, use pre-compiled templates, cached models, etc.
        return f"Processed: {input_text}"

def main(args=None):
    rclpy.init(args=args)
    node = OptimizedSpeechNode()

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

## Quality Assurance

### Speech Quality Metrics

```python
#!/usr/bin/env python3
"""
Speech Quality Assessment
"""
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32
from audio_common_msgs.msg import AudioData
import librosa

class SpeechQualityNode(Node):
    def __init__(self):
        super().__init__('speech_quality_node')

        # Create subscriber for audio output
        self.audio_sub = self.create_subscription(
            AudioData, '/tts/output', self.audio_callback, 10)

        # Create publishers for quality metrics
        self.intelligibility_pub = self.create_publisher(Float32, '/tts/intelligibility', 10)
        self.naturalness_pub = self.create_publisher(Float32, '/tts/naturalness', 10)
        self.quality_pub = self.create_publisher(Float32, '/tts/quality', 10)

        self.get_logger().info('Speech quality assessment node started')

    def audio_callback(self, msg):
        """Assess quality of generated speech"""
        try:
            # Convert audio data to numpy array
            audio_data = np.frombuffer(msg.data, dtype=np.int16).astype(np.float32) / 32768.0

            # Calculate quality metrics
            intelligibility = self.calculate_intelligibility(audio_data)
            naturalness = self.calculate_naturalness(audio_data)
            overall_quality = self.calculate_overall_quality(intelligibility, naturalness)

            # Publish metrics
            self.publish_metrics(intelligibility, naturalness, overall_quality)

        except Exception as e:
            self.get_logger().error(f'Quality assessment error: {e}')

    def calculate_intelligibility(self, audio_data):
        """Calculate speech intelligibility metric"""
        # Simplified intelligibility calculation
        # In practice, use more sophisticated methods like PESQ or STOI
        try:
            # Calculate signal-to-noise ratio as proxy for intelligibility
            signal_power = np.mean(audio_data ** 2)
            noise_power = np.var(audio_data)  # Approximate noise as variance
            snr_db = 10 * np.log10(signal_power / (noise_power + 1e-10))

            # Normalize to 0-1 range
            intelligibility = min(1.0, max(0.0, (snr_db - 10) / 30))  # Good if SNR > 10dB
            return intelligibility
        except:
            return 0.5  # Default medium quality

    def calculate_naturalness(self, audio_data):
        """Calculate speech naturalness metric"""
        try:
            # Calculate spectral features as proxy for naturalness
            # Use zero-crossing rate to detect robotic speech patterns
            zcr = np.mean(librosa.feature.zero_crossing_rate(audio_data))

            # Natural speech typically has ZCR in range [0.01, 0.05]
            target_zcr = 0.03
            zcr_deviation = abs(zcr - target_zcr) / target_zcr

            # Lower deviation = more natural
            naturalness = max(0.0, 1.0 - zcr_deviation)
            return naturalness
        except:
            return 0.5  # Default medium quality

    def calculate_overall_quality(self, intelligibility, naturalness):
        """Calculate overall quality score"""
        # Weighted combination
        weights = {'intelligibility': 0.6, 'naturalness': 0.4}
        quality = (weights['intelligibility'] * intelligibility +
                  weights['naturalness'] * naturalness)
        return quality

    def publish_metrics(self, intelligibility, naturalness, quality):
        """Publish quality metrics"""
        int_msg = Float32()
        int_msg.data = intelligibility
        self.intelligibility_pub.publish(int_msg)

        nat_msg = Float32()
        nat_msg.data = naturalness
        self.naturalness_pub.publish(nat_msg)

        qual_msg = Float32()
        qual_msg.data = quality
        self.quality_pub.publish(qual_msg)

def main(args=None):
    rclpy.init(args=args)
    node = SpeechQualityNode()

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

### 1. System Design Best Practices

```python
# Good: Modular, scalable design
class GoodSpeechSystem:
    def __init__(self):
        self.tts = TTSComponent()
        self.nlg = NLGComponent()
        self.dialogue = DialogueManager()
        self.quality = QualityAssessor()

# Bad: Monolithic, hard-to-maintain system
class BadSpeechSystem:
    def __init__(self):
        # All functionality in one massive class
        pass

    def process_everything(self, input_data):
        # Tons of coupled functionality
        # Hard to test and maintain
        pass
```

### 2. Performance Best Practices

```python
# Good: Efficient processing
def efficient_processing():
    # Use appropriate data structures
    # Cache frequently accessed data
    # Pre-allocate memory when possible
    # Use vectorized operations
    # Implement proper buffering
    pass

# Bad: Inefficient processing
def inefficient_processing():
    # Repeated memory allocation
    # Inefficient loops
    # No caching
    # Blocking operations
    pass
```

### 3. Robustness Best Practices

```python
# Good: Robust error handling
def robust_speech_generation():
    try:
        # Generate speech
        result = generate_speech()
        return result
    except Exception as e:
        # Log error
        logger.error(f'Speech generation failed: {e}')
        # Return safe default
        return "I'm sorry, I had trouble generating speech."
    finally:
        # Clean up resources
        cleanup_resources()

# Bad: No error handling
def fragile_speech_generation():
    # Direct processing with no error handling
    # System crashes on any error
    result = generate_speech()
    return result
```

## Common Issues and Troubleshooting

### 1. Audio Quality Issues

```bash
# Check audio device
pulseaudio --check -v
# or
aplay -l

# Monitor audio processing
ros2 topic hz /tts/output
ros2 topic echo /tts/quality_metrics
```

### 2. Performance Issues

```bash
# Monitor CPU usage
htop

# Check processing rates
ros2 topic hz /speech/input
ros2 topic hz /speech/output

# Profile the system
python3 -m cProfile -o profile.stats your_script.py
```

### 3. Synchronization Issues

```python
# Ensure proper message synchronization
from message_filters import ApproximateTimeSynchronizer, Subscriber

# Use time synchronization for multi-sensor inputs
sync = ApproximateTimeSynchronizer([sub1, sub2], queue_size=10, slop=0.1)
sync.registerCallback(callback_function)
```

## Verification Checklist

- [ ] Speech system processes input correctly
- [ ] Natural language generation produces appropriate responses
- [ ] Dialogue management handles conversation flow
- [ ] Text-to-speech synthesizes clear audio
- [ ] Quality metrics are computed and published
- [ ] All nodes communicate properly
- [ ] System handles errors gracefully
- [ ] Performance meets real-time requirements

## Exercises

1. Implement a multi-language speech system
2. Add emotional speech synthesis capabilities
3. Create a personalized dialogue system that learns user preferences
4. Build a speech quality assessment system with user feedback integration

## Next Steps

Now that you understand complete speech systems, continue to [Week 11: Large Language Models](../../module-4-vision-language-action/week-11/introduction) to learn about integrating LLMs with robotic systems for advanced reasoning and planning.