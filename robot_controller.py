#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np
import logging
import ollama
import time
from rclpy.qos import QoSProfile, ReliabilityPolicy

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("RobotController")

class RobotController(Node):
    def __init__(self, model_name, model_prompt):
        super().__init__('robot_controller')
        
        # Initialize variables
        self.bridge = CvBridge()
        self.ollama_client = ollama.Client()
        self.model_name = model_name
        self.model_prompt = model_prompt
        self.last_image = None
        self.is_turning = False
        self.turn_direction = None
        self.obstacle_detected = False
        
        # Constants
        self.FRONT_OBSTACLE_THRESHOLD = 0.5  # meters
        self.TURNING_SPEED = 0.5  # rad/s
        self.FORWARD_SPEED = 0.2  # m/s
        self.MIN_FRONT_ANGLE = -30  # degrees
        self.MAX_FRONT_ANGLE = 30   # degrees
        
        # Create the model
        self.create_model()
        
        # Subscriptions
        self.camera_subscription = self.create_subscription(
            Image,
            '/image_raw',
            self.camera_callback,
            10)

        self.lidar_subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.lidar_callback,
            qos_profile=QoSProfile(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                depth=10
            )
        )

        # Publisher
        self.control_publisher = self.create_publisher(
            Twist,
            'cmd_vel',
            10)

        logger.info('Robot Controller Node initialized')

    def create_model(self):
        try:
            modelfile = f"""from {self.model_name}
            parameter temperature 0.5
            system "You are analyzing obstacles in a robot race track. Identify if the obstacle is another robot or a track wall/obstacle. If it's a wall, suggest which direction to turn (left or right) based on the track layout. Format response as: <type>robot/wall</type><direction>left/right</direction>"
            """
            self.ollama_client.create(model=self.model_name, modelfile=modelfile)
            logger.info(f"Model '{self.model_name}' created successfully")
        except Exception as e:
            logger.error(f"Failed to create model: {str(e)}")

    def camera_callback(self, msg):
        """Store the latest image for obstacle analysis"""
        try:
            self.last_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            logger.error(f'Failed to convert image: {str(e)}')

    def get_front_distance(self, ranges):
        """Get the minimum distance in the front sector"""
        angles = np.arange(len(ranges))
        front_indices = np.where(
            (angles >= len(ranges) * (self.MIN_FRONT_ANGLE + 180) / 360) &
            (angles <= len(ranges) * (self.MAX_FRONT_ANGLE + 180) / 360)
        )[0]
        
        front_ranges = np.array(ranges)[front_indices]
        valid_ranges = front_ranges[(front_ranges > 0.12) & (front_ranges < 3.5)]
        
        return np.min(valid_ranges) if len(valid_ranges) > 0 else float('inf')

    def analyze_obstacle(self):
        """Use the model to analyze the obstacle type and get turning direction"""
        if self.last_image is None:
            logger.error("No image available for analysis")
            return None, None

        try:
            # Convert image to bytes
            _, buffer = cv2.imencode('.jpg', self.last_image)
            image_bytes = buffer.tobytes()

            # Get model's analysis
            response = self.ollama_client.generate(
                model=self.model_name,
                prompt="Is this obstacle another robot or a track wall? If wall, which direction should I turn?",
                images=[image_bytes],
                keep_alive=True
            )

            # Parse response
            import re
            type_match = re.search(r'<type>(.*?)</type>', response.response)
            direction_match = re.search(r'<direction>(.*?)</direction>', response.response)

            obstacle_type = type_match.group(1) if type_match else None
            turn_direction = direction_match.group(1) if direction_match else None

            return obstacle_type, turn_direction

        except Exception as e:
            logger.error(f"Error in obstacle analysis: {str(e)}")
            return None, None

    def lidar_callback(self, msg):
        """Process LiDAR data and control the robot"""
        if self.is_turning:
            # Continue turning until front is clear
            front_distance = self.get_front_distance(msg.ranges)
            if front_distance > self.FRONT_OBSTACLE_THRESHOLD:
                self.is_turning = False
                self.turn_direction = None
                self.obstacle_detected = False
            else:
                # Continue turning in the same direction
                self.execute_turn()
            return

        # Check for obstacles in front
        front_distance = self.get_front_distance(msg.ranges)
        
        if front_distance <= self.FRONT_OBSTACLE_THRESHOLD and not self.obstacle_detected:
            # New obstacle detected
            self.obstacle_detected = True
            # Stop the robot
            self.publish_stop()
            
            # Analyze obstacle
            obstacle_type, turn_direction = self.analyze_obstacle()
            
            if obstacle_type == 'wall' and turn_direction in ['left', 'right']:
                self.is_turning = True
                self.turn_direction = turn_direction
                logger.info(f"Wall detected, turning {turn_direction}")
                self.execute_turn()
            elif obstacle_type == 'robot':
                logger.info("Robot detected, waiting...")
                self.publish_stop()
            else:
                logger.warning("Unclear obstacle analysis, stopping")
                self.publish_stop()
        
        elif front_distance > self.FRONT_OBSTACLE_THRESHOLD:
            # Clear path ahead
            self.move_forward()

    def execute_turn(self):
        """Execute the turning movement"""
        cmd = Twist()
        if self.turn_direction == 'left':
            cmd.angular.z = self.TURNING_SPEED
        else:  # right
            cmd.angular.z = -self.TURNING_SPEED
        self.control_publisher.publish(cmd)

    def move_forward(self):
        """Move the robot forward"""
        cmd = Twist()
        cmd.linear.x = self.FORWARD_SPEED
        self.control_publisher.publish(cmd)

    def publish_stop(self):
        """Stop the robot"""
        self.control_publisher.publish(Twist())

def main():
    rclpy.init()

    model_name = "llava:7b"
    model_prompt = (
        """
        You are analyzing obstacles in a robot race track.
        Identify if the obstacle is another robot or a track wall/obstacle.
        If it's a wall, suggest which direction to turn (left or right) based on the track layout.
        Format response as: <type>robot/wall</type><direction>left/right</direction>
        """
    )

    controller = RobotController(model_name, model_prompt)
    
    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt detected, stopping robot...")
        stop_cmd = Twist()
        controller.control_publisher.publish(stop_cmd)
        time.sleep(0.5)
    finally:
        controller.destroy_node()
        rclpy.shutdown()
        logger.info("Shutting down Robot Controller Node")

if __name__ == '__main__':
    main()







