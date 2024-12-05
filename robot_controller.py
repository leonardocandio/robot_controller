#!/usr/bin/env python3

from asyncore import read
import base64
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np
import logging
import ollama
import re
from time import time
from rclpy.qos import QoSProfile, ReliabilityPolicy


# Initialize logging to print to stdout
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("RobotController")



class RobotController(Node):
    def __init__(self, model_name, model_prompt):
        super().__init__('robot_controller')
        
        # Existing initializations
        self.bridge = CvBridge()
        self.ollama_client = ollama.Client()
        self.model_name = model_name
        self.model_prompt = model_prompt
        self.image_count = 0
        self.last_command = Twist()
        self.last_command.linear.x = 0.1
        self.last_command.angular.z = 0.0
        self.last_image_time = 0.0  # For rate limiting
        self.process_interval = 0.5  # Process images every 0.5 seconds
        self.lidar_data = None  # To store LiDAR data
        self.create_model()

        # Camera subscription
        self.camera_subscription = self.create_subscription(
            Image,
            '/image_raw',
            self.camera_callback,
            10)

        # LiDAR subscription
        self.lidar_subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.lidar_callback,
            qos_profile=QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,  # Match the publisher's QoS
            depth=10  # Adjust the queue size if needed
            )
        )

        # Control publisher
        self.control_publisher = self.create_publisher(
            Twist,
            'cmd_vel',
            10)

        self.get_logger().info('Robot Controller Node initialized')
        logger.info("Robot Controller Node initialized")

    def create_model(self):
        try:
            modelfile = f"""from {self.model_name}
            parameter temperature 0.5
            system "{self.model_prompt}"
            """
            self.ollama_client.create(model=self.model_name, modelfile=modelfile)
            self.get_logger().info(f"Model '{self.model_name}' created successfully")
            logger.info(f"Model '{self.model_name}' created successfully")
        except Exception as e:
            self.get_logger().error(f"Failed to create model: {str(e)}")
            logger.error(f"Failed to create model: {str(e)}")


    def lidar_callback(self, msg):
        """
        Process LiDAR data from the LDS-02 sensor and extract six points of distance.
        The points represent the minimum distance in six evenly divided angular sectors.
        """
        ranges = np.array(msg.ranges)  # Convert ranges to a numpy array for easier processing

        # LDS-02 specific parameters - explicitly set these values
        min_range = 0.12  # 12cm minimum range for LDS-02
        max_range = 3.5   # 3.5m maximum range for LDS-02

        # Clean up the ranges array
        # Convert to float32 to match ROS2 data type
        ranges = ranges.astype(np.float32)

        # Create a mask for valid measurements
        valid_mask = (ranges >= min_range) & (ranges <= max_range) & (~np.isnan(ranges)) & (~np.isinf(ranges))
        ranges[~valid_mask] = max_range  # Set invalid measurements to max_range

        # Define the angular resolution for six sectors (360° divided into 6 sectors = 60° each)
        num_sectors = 6
        sector_size = len(ranges) // num_sectors

        # Extract minimum distances from each sector
        distances = []
        for i in range(num_sectors):
            # Calculate start and end indices for each sector
            start_idx = i * sector_size
            end_idx = start_idx + sector_size

            # Handle wrap-around for the last sector
            if i == num_sectors - 1:
                sector = ranges[start_idx:]
            else:
                sector = ranges[start_idx:end_idx]

            # Find the minimum distance in the sector
            min_distance = np.min(sector)

            # Map distances: 0 if too close, min_distance otherwise
            if min_distance <= 0.2:  # Threshold for "too close"
                mapped_distance = 0.0
            else:
                mapped_distance = min_distance

            distances.append(mapped_distance)

        # Assign sectors based on corrected mapping
        self.lidar_data = {
            "front": distances[2],        # Front sector
            "front_right": distances[3],  # Front Right sector
            "right": distances[4],        # Right sector
            "rear": distances[5],         # Rear sector
            "left": distances[1],         # Left sector
            "front_left": distances[0]    # Front Left sector
        }

        # Log the distances for debugging
        logger.info(
            f"LiDAR Distances - Front: {self.lidar_data['front']:.2f} m, "
            f"Front Left: {self.lidar_data['front_left']:.2f} m, Left: {self.lidar_data['left']:.2f} m, "
            f"Rear: {self.lidar_data['rear']:.2f} m, Right: {self.lidar_data['right']:.2f} m, "
            f"Front Right: {self.lidar_data['front_right']:.2f} m"
        )


    def camera_callback(self, msg):
        current_time = time()
        if current_time - self.last_image_time < self.process_interval:
            return  # Skip processing if it's too soon
        
        self.last_image_time = current_time  # Update the last processed time
        
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f'Failed to convert image: {str(e)}')
            logger.error(f'Failed to convert image: {str(e)}')
            return

        # control_msg = self.analyze_image(cv_image)
        
        # if self.is_significant_change(control_msg):
        #     self.control_publisher.publish(control_msg)
        #     self.last_command = control_msg  # Update the last command
        #     logger.info(f"Published control: linear={control_msg.linear}, angular={control_msg.angular}")
        # else:
        #     self.control_publisher.publish(self.last_command)
        #     logger.info(f"No significant change in control, publishing last linear={self.last_command.linear.x} angular={self.last_command.angular.z}")
        self.control_publisher.publish(Twist())


    def is_significant_change(self, control_msg):
        """Check if the control message has significant changes compared to the last command."""
        linear_diff = abs(control_msg.linear.x - self.last_command.linear.x)
        angular_diff = abs(control_msg.angular.z - self.last_command.angular.z)
        return linear_diff > 0.05 or angular_diff > 0.05  # Tune thresholds as needed

    def analyze_image(self, image):
        try:
            _, buffer = cv2.imencode('.jpg', image)
            image_bytes = buffer.tobytes()
        except Exception as e:
            self.get_logger().error(f'Failed to encode image: {str(e)}')
            return Twist()

        # Combine LiDAR and image data in the model prompt
        lidar_context = (
            f"Front obstacle distance: {self.lidar_data['min_front']:.2f} meters. "
            f"Left obstacle distance: {self.lidar_data['min_left']:.2f} meters. "
            f"Right obstacle distance: {self.lidar_data['min_right']:.2f} meters. "
            if self.lidar_data else "LiDAR data not available."
        )
        prompt = (
            f"This is image {self.image_count}. {lidar_context} Analyze it to generate control commands."
        )

        try:
            response = self.ollama_client.generate(
                model=self.model_name,
                prompt=prompt,
                images=[image_bytes],
                keep_alive=True
            )
        except Exception as e:
            self.get_logger().error(f'Failed to generate ollama response: {str(e)}')
            logger.error(f'Failed to generate ollama response: {str(e)}')
            return Twist()

        try:
            control_msg = Twist()
            response_data = self.translate_response(response)
            control_msg.linear.x = response_data["linear"]
            control_msg.angular.z = response_data["angular"]
        except Exception as e:
            self.get_logger().error(f'Failed to parse ollama response: {str(e)}')
            logger.error(f'Failed to parse ollama response: {str(e)}')
            control_msg = Twist()

        self.image_count += 1
        return control_msg

    def translate_response(self, response):
        print("response: ", response.response)
        linear_pattern = r"<linear>(.*?)</linear>"
        angular_pattern = r"<angular>(.*?)</angular>"

        linear_match = re.search(linear_pattern, response.response, re.DOTALL)
        angular_match = re.search(angular_pattern, response.response, re.DOTALL)

        if linear_match and angular_match:
            return {
                "linear": float(linear_match.group(1).strip()),
                "angular": float(angular_match.group(1).strip()),
            }
        return None


def main():
    rclpy.init()

    model_name = "llava:7b"
    model_prompt = (
        """
    You are a TurtleBot3 ROS2-powered robot navigating a racetrack. The track is defined by carboard boxes, multiple obstacles and a glass wall. Your objective is to navigate the racetrack and win the race. There will be other robots in the race, try to avoid them.

    Your task is to generate ROS2 Twist control commands based on the following input:
    - LiDAR data: obstacle distances in meters for front, left, and right directions.
    - Camera image: a real-time view of the environment.

    Format your response strictly as:
    - Provide a short one sentence reasoning explanation behind your commands enclosed in <reasoning> tags.
    - Provide a short one sentence description of the image enclosed in <description> tags.
    - Provide the forward velocity in m/s (linear.x) value enclosed in <linear> tags.
    - Provide the angular velocity in rad/s (angular.z) value enclosed in <angular> tags.

    Important: You should always respond with a command even if the robot is not inside the specified location

    <example>
    <description>A carboard box racing track with a sharp right turn</description>
    <reasonin>To follow the sharp right turn, the robot should turn right</reasoning>
    <linear>0.4</linear><angular>0.5</angular>
    </example>

    DO NOT include any text outside of these tags. Do not provide explanations, comments, or additional information.
    """
    )

    controller = RobotController(model_name, model_prompt)

    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        pass
    finally:
        controller.destroy_node()
        rclpy.shutdown()
        logger.info("Shutting down Robot Controller Node")


if __name__ == '__main__':
    main()







