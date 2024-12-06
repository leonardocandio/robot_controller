# ROS2 module imports
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan, Image
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from rclpy.qos import qos_profile_sensor_data
from rclpy.duration import Duration
from cv_bridge import CvBridge
import cv2

# Python module imports
import numpy as np
import queue
import time


class PIDController:
    def __init__(self, kP, kI, kD, kS):
        self.kP = kP  # Proportional gain
        self.kI = kI  # Integral gain
        self.kD = kD  # Derivative gain
        self.kS = kS  # Saturation constant (error history buffer size)
        self.err_int = 0  # Error integral
        self.err_dif = 0  # Error difference
        self.err_prev = 0  # Previous error
        self.err_hist = queue.Queue(self.kS)  # Limited buffer of error history
        self.t_prev = 0  # Previous time

    def control(self, err, t):
        dt = t - self.t_prev  # Timestep
        if dt > 0.0:
            self.err_hist.put(err)  # Update error history
            self.err_int += err  # Integrate error
            if self.err_hist.full():  # Prevent integral windup
                self.err_int -= self.err_hist.get()  # Rolling FIFO buffer
            self.err_dif = err - self.err_prev
            u = (self.kP * err) + (self.kI * self.err_int * dt) + (self.kD * self.err_dif / dt)
            self.err_prev = err
            self.t_prev = t
            return u
        return 0


class RobotController(Node):
    def __init__(self):
        # Information and debugging
        info = "\nMake the robot follow wall, avoid obstacles, follow line, detect stop sign and track AprilTag marker.\n"
        print(info)

        # ROS2 infrastructure
        super().__init__("robot_controller")

        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_RELIABLE,
            history=QoSHistoryPolicy.RMW_QOS_POLICY_HISTORY_KEEP_LAST,
            depth=10,
        )

        self.robot_lidar_sub = self.create_subscription(
            LaserScan, "/scan", self.robot_lidar_callback, qos_profile_sensor_data
        )

        self.robot_ctrl_pub = self.create_publisher(Twist, "/cmd_vel", qos_profile)

        timer_period = 0.001  # Node execution time period (seconds)
        self.timer = self.create_timer(timer_period, self.robot_controller_callback)

        # PID Controllers (do not change configuration)
        self.pid_1_lat = PIDController(0.3, 0.01, 0.1, 10)
        self.pid_1_lon = PIDController(0.1, 0.001, 0.005, 10)

        # State variables
        self.lidar_available = False
        self.laserscan = None
        self.start_mode = "outside"
        self.start_time = self.get_clock().now()
        self.ctrl_msg = Twist()
        self.prefer_left_turns = True

        # Camera subscription
        self.camera_sub = self.create_subscription(
            Image, 
            '/image_raw', 
            self.camera_callback,
            qos_profile_sensor_data
        )
        
        # Image processing setup
        self.bridge = CvBridge()
        self.latest_image = None
        self.path_detected = False
        self.path_center = 0

    def robot_lidar_callback(self, msg):
        # Robust LIDAR data preprocessing
        ranges = np.array(msg.ranges)
        ranges = np.nan_to_num(ranges, nan=3.5, posinf=3.5, neginf=3.5)
        ranges = np.clip(ranges, 0.1, 3.5)

        self.laserscan = ranges
        self.lidar_available = True

    def camera_callback(self, msg):
        try:
            # Convert ROS Image message to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # Convert to grayscale
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            
            # Apply threshold to isolate the path
            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
            
            # Get the bottom portion of the image where the path is most visible
            height = binary.shape[0]
            roi = binary[int(height*0.7):height, :]
            
            # Find the center of the path
            M = cv2.moments(roi)
            if M["m00"] > 0:
                self.path_center = int(M["m10"] / M["m00"])
                self.path_detected = True
            else:
                self.path_detected = False
                
        except Exception as e:
            self.get_logger().error(f'Error processing image: {str(e)}')
            self.path_detected = False

    def robot_controller_callback(self):
        DELAY = 4.0  # Time delay (s)
        if self.get_clock().now() - self.start_time > Duration(seconds=DELAY):
            if self.lidar_available and self.laserscan is not None:
                # Dynamically calculate scan parameters
                scan_length = len(self.laserscan)
                step_size = 360.0 / scan_length

                # Safe sector calculation
                front_sector = max(1, int(30 / step_size))
                side_sector = max(1, int(45 / step_size))

                def safe_mean(arr):
                    valid_ranges = arr[np.isfinite(arr)]
                    return np.mean(valid_ranges) if len(valid_ranges) > 0 else 3.5

                # Compute distances in various directions
                front_left = safe_mean(self.laserscan[: int(scan_length / 4)])
                front_right = safe_mean(self.laserscan[-int(scan_length / 4):])
                front_center = safe_mean(
                    np.concatenate([self.laserscan[:front_sector], self.laserscan[-front_sector:]])
                )
                left_side = safe_mean(self.laserscan[int(scan_length / 4): int(scan_length / 2)])
                right_side = safe_mean(self.laserscan[int(scan_length / 2): int(3 * scan_length / 4)])

                # Timestamp for PID
                tstamp = time.time()

                # Thresholds (unchanged)
                FRONT_OBSTACLE_THRESHOLD = 0.7
                SIDE_OBSTACLE_THRESHOLD = 0.5

                # Debugging print
                print(
                    f"Distances - Front: {front_center:.2f}, Left: {front_left:.2f}, Right: {front_right:.2f}, Side-Left: {left_side:.2f}, Side-Right: {right_side:.2f}"
                )

                # Enhanced decision logic with camera input
                if self.path_detected:
                    # Use path center to influence turning decision
                    image_width = 640  # Adjust based on your camera resolution
                    path_error = (image_width/2 - self.path_center) / (image_width/2)  # Normalized error
                    
                    if abs(path_error) > 0.2:  # If path significantly deviates from center
                        # Adjust angular velocity based on path position
                        ANG_VEL = self.pid_1_lat.control(path_error * 1.5, tstamp)
                        LIN_VEL = 0.15  # Reduce speed during turns
                        print(f"Path correction: error={path_error:.2f}")
                    else:
                        # Path is centered, proceed with LIDAR-based wall following
                        if (front_center < FRONT_OBSTACLE_THRESHOLD or
                            front_left < FRONT_OBSTACLE_THRESHOLD or
                            front_right < FRONT_OBSTACLE_THRESHOLD):
                            # Original obstacle avoidance logic
                            if (front_left > front_right) == self.prefer_left_turns:
                                LIN_VEL = 0.05
                                ANG_VEL = 1.2
                                print("OBSTACLE AHEAD: Turning Left")
                            else:
                                LIN_VEL = 0.05
                                ANG_VEL = -1.2
                                print("OBSTACLE AHEAD: Turning Right")
                        else:
                            # Normal operation
                            LIN_VEL = 0.22
                            ANG_VEL = self.pid_1_lat.control((left_side - right_side) * 2.0, tstamp)
                else:
                    # Fallback to original LIDAR-based logic if path not detected
                    if (front_center < FRONT_OBSTACLE_THRESHOLD or
                        front_left < FRONT_OBSTACLE_THRESHOLD or
                        front_right < FRONT_OBSTACLE_THRESHOLD):
                        # Obstacle ahead, slow down and turn
                        if (front_left > front_right) == self.prefer_left_turns:
                            LIN_VEL = 0.05
                            ANG_VEL = 1.2
                            print("OBSTACLE AHEAD: Turning Left")
                        else:
                            LIN_VEL = 0.05
                            ANG_VEL = -1.2
                            print("OBSTACLE AHEAD: Turning Right")

                    elif (left_side < SIDE_OBSTACLE_THRESHOLD or right_side < SIDE_OBSTACLE_THRESHOLD):
                        # Obstacle on sides, slow slightly and turn away
                        if left_side < SIDE_OBSTACLE_THRESHOLD:
                            LIN_VEL = 0.1
                            ANG_VEL = -0.8
                            print("SIDE OBSTACLE: Turning Right")
                        else:
                            LIN_VEL = 0.1
                            ANG_VEL = 0.8
                            print("SIDE OBSTACLE: Turning Left")

                    else:
                        # Path is clear, go at maximum allowed speed and follow walls
                        # Increased linear velocity from 0.2 to 0.22 to maximize forward speed
                        LIN_VEL = 0.22
                        ANG_VEL = self.pid_1_lat.control((left_side - right_side) * 2.0, tstamp)
                        print("Wall Following - MAX SPEED")

                # Enforce velocity limits
                self.ctrl_msg.linear.x = min(0.22, float(LIN_VEL))
                # Keep angular velocity within safe limits
                if ANG_VEL > 2.84:
                    ANG_VEL = 2.84
                elif ANG_VEL < -2.84:
                    ANG_VEL = -2.84
                self.ctrl_msg.angular.z = ANG_VEL

                # Publish the command
                self.robot_ctrl_pub.publish(self.ctrl_msg)
        else:
            print("Initializing...")


def main(args=None):
    rclpy.init(args=args)
    node = RobotController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()