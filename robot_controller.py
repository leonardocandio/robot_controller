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
        
        # Clean up the ranges array and invert the distances
        # Convert to float32 to match ROS2 data type
        ranges = ranges.astype(np.float32)
        
        # Create a mask for valid measurements
        valid_mask = (ranges >= min_range) & (ranges <= max_range) & (~np.isnan(ranges)) & (~np.isinf(ranges))
        
        # Invert valid measurements (max_range - distance) so closer objects show smaller numbers
        ranges[valid_mask] = max_range - ranges[valid_mask]
        # Invalid measurements become 0 (indicating no obstacle)
        ranges[~valid_mask] = 0.0
        
        # Define the angular resolution for six sectors (360° divided into 6 sectors = 60° each)
        num_sectors = 6
        sector_size = len(ranges) // num_sectors
        
        # Extract maximum distances from each sector (now that values are inverted, we want maximum)
        distances = []
        for i in range(num_sectors):
            start_idx = ((i + 3) % num_sectors) * sector_size  # Shift by 3 sectors (180°) to align front
            end_idx = start_idx + sector_size
            if end_idx > len(ranges):  # Handle wrap-around
                sector = np.concatenate([ranges[start_idx:], ranges[:end_idx - len(ranges)]])
            else:
                sector = ranges[start_idx:end_idx]
            
            # Find maximum of inverted distances (closest object)
            max_distance = np.max(sector)
            distances.append(max_distance)

        # Store the processed distances in a dictionary for reference
        self.lidar_data = {
            "front": distances[0],       # 330° to 30° (front center)
            "front_right": distances[1], # 30° to 90°
            "right": distances[2],       # 90° to 150°
            "rear": distances[3],        # 150° to 210°
            "left": distances[4],        # 210° to 270°
            "front_left": distances[5]   # 270° to 330°
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








"""

header:
  stamp:
    sec: 1733432924
    nanosec: 264948698
  frame_id: base_scan
angle_min: 0.012750275433063507
angle_max: 6.2697930335998535
angle_increment: 0.025572845712304115
time_increment: 0.00043659881339408457
scan_time: 0.10682492703199387
range_min: 0.0
range_max: 100.0
ranges:
- 2.3299999237060547
- 2.739000082015991
- 2.7760000228881836
- 2.4739999771118164
- 3.062999963760376
- 3.052000045776367
- 2.197999954223633
- 4.103000164031982
- 4.111999988555908
- 4.11299991607666
- .nan
- 1.784000039100647
- 1.7589999437332153
- 1.7419999837875366
- 1.656000018119812
- 1.6469999551773071
- 1.6039999723434448
- 1.5850000381469727
- 1.5720000267028809
- 1.6460000276565552
- 1.6119999885559082
- 1.4930000305175781
- 1.3949999809265137
- 1.309999942779541
- 1.2330000400543213
- 1.1640000343322754
- 1.1339999437332153
- 1.149999976158142
- 1.149999976158142
- 1.149999976158142
- 1.1460000276565552
- 1.7170000076293945
- 1.7289999723434448
- .nan
- 1.406999945640564
- 1.375
- 1.347000002861023
- 1.3480000495910645
- 0.48399999737739563
- 0.4569999873638153
- 0.4440000057220459
- 0.43700000643730164
- 0.4339999854564667
- 0.44200000166893005
- 0.4659999907016754
- 0.4950000047683716
- .nan
- .nan
- 2.069000005722046
- 2.0329999923706055
- 2.0
- 5.081999778747559
- 3.4170000553131104
- 3.4040000438690186
- 2.4779999256134033
- 2.549999952316284
- 0.5690000057220459
- 0.5540000200271606
- 0.5490000247955322
- 0.5519999861717224
- 0.5609999895095825
- .nan
- 4.548999786376953
- 4.830999851226807
- 4.800000190734863
- 4.788000106811523
- 4.764999866485596
- 4.754000186920166
- 4.757999897003174
- 4.745999813079834
- 3.3529999256134033
- 2.1519999504089355
- 2.1410000324249268
- 0.8080000281333923
- 0.7559999823570251
- 0.722000002861023
- 0.6899999976158142
- 0.6639999747276306
- 0.6520000100135803
- 0.6370000243186951
- 0.640999972820282
- 0.6489999890327454
- 0.6579999923706055
- 0.6959999799728394
- 0.7120000123977661
- 0.7310000061988831
- .nan
- 0.7490000128746033
- 0.7559999823570251
- 0.8059999942779541
- 0.8349999785423279
- 0.9660000205039978
- 0.9779999852180481
- 0.9950000047683716
- 1.00600004196167
- 1.0240000486373901
- 1.0449999570846558
- 1.0640000104904175
- 1.0959999561309814
- 1.1200000047683716
- 1.1460000276565552
- 1.1679999828338623
- 1.2000000476837158
- 1.2269999980926514
- .nan
- 0.6579999923706055
- 0.6430000066757202
- 0.6389999985694885
- 0.6449999809265137
- 1.4989999532699585
- 1.5360000133514404
- 1.5099999904632568
- 1.4839999675750732
- 1.4839999675750732
- 1.5579999685287476
- 1.6239999532699585
- 1.6239999532699585
- 1.6360000371932983
- .nan
- 1.6770000457763672
- 1.7999999523162842
- 4.1519999504089355
- 4.206999778747559
- 4.418000221252441
- .nan
- .nan
- 0.19200000166893005
- 0.1860000044107437
- '...'
intensities:
- 224.0
- 192.0
- 176.0
- 228.0
- 224.0
- 232.0
- 230.0
- 218.0
- 218.0
- 218.0
- .nan
- 70.0
- 112.0
- 148.0
- 228.0
- 90.0
- 100.0
- 224.0
- 94.0
- 228.0
- 160.0
- 210.0
- 216.0
- 216.0
- 218.0
- 224.0
- 228.0
- 228.0
- 224.0
- 228.0
- 228.0
- 228.0
- 228.0
- .nan
- 228.0
- 228.0
- 228.0
- 218.0
- 212.0
- 224.0
- 228.0
- 228.0
- 228.0
- 228.0
- 228.0
- 228.0
- .nan
- .nan
- 228.0
- 228.0
- 224.0
- 94.0
- 218.0
- 222.0
- 192.0
- 66.0
- 224.0
- 228.0
- 228.0
- 228.0
- 228.0
- .nan
- 212.0
- 224.0
- 228.0
- 228.0
- 228.0
- 228.0
- 228.0
- 230.0
- 148.0
- 234.0
- 234.0
- 136.0
- 228.0
- 96.0
- 224.0
- 224.0
- 228.0
- 222.0
- 224.0
- 224.0
- 218.0
- 218.0
- 224.0
- 228.0
- .nan
- 228.0
- 228.0
- 124.0
- 230.0
- 230.0
- 230.0
- 228.0
- 228.0
- 228.0
- 224.0
- 224.0
- 228.0
- 232.0
- 228.0
- 228.0
- 226.0
- 228.0
- .nan
- 222.0
- 228.0
- 228.0
- 228.0
- 228.0
- 228.0
- 228.0
- 230.0
- 222.0
- 224.0
- 228.0
- 128.0
- 112.0
- .nan
- 228.0
- 224.0
- 222.0
- 224.0
- 70.0
- .nan
- .nan
- 110.0
- 224.0
- '...'
---

"""