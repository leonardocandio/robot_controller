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
        
        # Robot physical parameters (in meters)
        self.ROBOT_WIDTH = 0.20
        self.ROBOT_LENGTH = 0.20
        self.SAFETY_MARGIN = 0.10  # Additional safety margin around robot
        
        # Obstacle avoidance parameters
        self.CRITICAL_DISTANCE = self.ROBOT_WIDTH/2 + self.SAFETY_MARGIN  # Distance at which to stop
        self.SAFE_DISTANCE = self.CRITICAL_DISTANCE + 0.15  # Distance at which to start avoiding
        self.stuck_time = None
        self.recovery_turn_direction = 1  # 1 for left, -1 for right
        self.consecutive_stops = 0
        self.last_movement_time = time.time()

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
        self.split_detected = False
        self.chosen_path = None
        self.last_good_path = None
        self.split_decision_time = None

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
            height, width = cv_image.shape[:2]
            
            # Convert to grayscale
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            
            # Apply threshold to isolate the path
            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
            
            # Define multiple ROIs for path detection
            roi_bottom = binary[int(height*0.8):height, :]
            roi_middle = binary[int(height*0.6):int(height*0.8), :]
            roi_top = binary[int(height*0.4):int(height*0.6), :]
            
            # Find contours in each ROI
            contours_bottom, _ = cv2.findContours(roi_bottom, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours_middle, _ = cv2.findContours(roi_middle, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours_top, _ = cv2.findContours(roi_top, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Detect split in path
            self.split_detected = len(contours_middle) > 1 or len(contours_top) > 1
            
            # Process bottom ROI for immediate steering
            if contours_bottom:
                largest_contour = max(contours_bottom, key=cv2.contourArea)
                M = cv2.moments(largest_contour)
                if M["m00"] > 0:
                    current_center = int(M["m10"] / M["m00"])
                    
                    # If we detect a split and haven't made a decision yet
                    if self.split_detected and self.chosen_path is None:
                        current_time = time.time()
                        if self.split_decision_time is None:
                            self.split_decision_time = current_time
                            # Make a decision based on prefer_left_turns
                            if self.prefer_left_turns:
                                self.chosen_path = 'left'
                            else:
                                self.chosen_path = 'right'
                    
                    # Apply the path decision
                    if self.chosen_path == 'left':
                        target_x = width * 0.3  # Aim for left side
                    elif self.chosen_path == 'right':
                        target_x = width * 0.7  # Aim for right side
                    else:
                        target_x = width * 0.5  # Aim for center
                    
                    # Update path center with bias towards chosen direction
                    self.path_center = int(0.7 * current_center + 0.3 * target_x)
                    self.path_detected = True
                    self.last_good_path = self.path_center
                    
                    # Reset split decision after passing through
                    if not self.split_detected and self.chosen_path is not None:
                        self.chosen_path = None
                        self.split_decision_time = None
                
            else:
                self.path_detected = False
                if self.last_good_path is not None:
                    self.path_center = self.last_good_path
                
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

                # Enhanced sector calculations for better coverage
                front_sector = max(1, int(45 / step_size))  # Increased from 30 to 45 degrees
                side_sector = max(1, int(60 / step_size))   # Increased from 45 to 60 degrees

                def safe_mean(arr):
                    valid_ranges = arr[np.isfinite(arr)]
                    return np.mean(valid_ranges) if len(valid_ranges) > 0 else 3.5

                # Compute distances with more detailed sectors
                front_center = safe_mean(
                    np.concatenate([self.laserscan[:front_sector], self.laserscan[-front_sector:]])
                )
                front_left = safe_mean(self.laserscan[front_sector:front_sector*2])
                front_right = safe_mean(self.laserscan[-front_sector*2:-front_sector])
                left_side = safe_mean(self.laserscan[int(scan_length/4):int(scan_length/2)])
                right_side = safe_mean(self.laserscan[int(scan_length/2):int(3*scan_length/4)])

                # Find minimum distances in critical sectors
                front_min = min(
                    float(np.min(self.laserscan[:front_sector])),
                    float(np.min(self.laserscan[-front_sector:]))
                )
                sides_min = float(np.min(self.laserscan[int(scan_length/4):int(3*scan_length/4)]))

                # Timestamp for PID
                tstamp = time.time()

                # Check if we're in a critical situation (too close to obstacles)
                in_critical_situation = (front_min < self.CRITICAL_DISTANCE or 
                                       sides_min < self.CRITICAL_DISTANCE/2)

                # Check if we're stuck (no movement for too long)
                if in_critical_situation:
                    if self.stuck_time is None:
                        self.stuck_time = time.time()
                        self.consecutive_stops += 1
                    elif time.time() - self.stuck_time > 2.0:  # Stuck for more than 2 seconds
                        # Implement recovery behavior
                        LIN_VEL = 0.0
                        # Alternate turn direction if we've been stuck multiple times
                        if self.consecutive_stops > 3:
                            self.recovery_turn_direction *= -1
                            self.consecutive_stops = 0
                        ANG_VEL = 1.0 * self.recovery_turn_direction
                        print("STUCK: Implementing recovery behavior")
                else:
                    self.stuck_time = None
                    
                    if front_min < self.SAFE_DISTANCE:
                        # Obstacle ahead - slow down and turn
                        LIN_VEL = 0.05
                        # Choose turn direction based on available space
                        if (front_left > front_right) == self.prefer_left_turns:
                            ANG_VEL = 1.0
                            print("OBSTACLE AHEAD: Turning Left")
                        else:
                            ANG_VEL = -1.0
                            print("OBSTACLE AHEAD: Turning Right")
                    
                    elif sides_min < self.SAFE_DISTANCE:
                        # Obstacle on sides - adjust course
                        LIN_VEL = 0.1
                        if left_side < right_side:
                            ANG_VEL = -0.8
                            print("SIDE OBSTACLE: Adjusting Right")
                        else:
                            ANG_VEL = 0.8
                            print("SIDE OBSTACLE: Adjusting Left")
                    
                    else:
                        # Path is clear - normal operation
                        if self.path_detected:
                            # Use path center to influence turning decision
                            image_width = 640  # Adjust based on your camera resolution
                            path_error = (image_width/2 - self.path_center) / (image_width/2)
                            
                            if abs(path_error) > 0.2:
                                LIN_VEL = 0.15
                                ANG_VEL = self.pid_1_lat.control(path_error * 1.5, tstamp)
                            else:
                                LIN_VEL = 0.2
                                ANG_VEL = self.pid_1_lat.control((left_side - right_side) * 1.5, tstamp)
                        else:
                            # Default to wall following if no path detected
                            LIN_VEL = 0.2
                            ANG_VEL = self.pid_1_lat.control((left_side - right_side) * 1.5, tstamp)

                # Update last movement time if we're actually moving
                if abs(LIN_VEL) > 0.05 or abs(ANG_VEL) > 0.1:
                    self.last_movement_time = time.time()

                # Enforce velocity limits with smoother transitions
                target_linear_x = min(0.22, float(LIN_VEL))
                current_linear_x = self.ctrl_msg.linear.x
                # Smooth acceleration and deceleration
                if target_linear_x > current_linear_x:
                    self.ctrl_msg.linear.x = min(target_linear_x, current_linear_x + 0.05)
                else:
                    self.ctrl_msg.linear.x = max(target_linear_x, current_linear_x - 0.05)

                # Keep angular velocity within safe limits with smoother transitions
                target_angular_z = np.clip(ANG_VEL, -2.84, 2.84)
                current_angular_z = self.ctrl_msg.angular.z
                if target_angular_z > current_angular_z:
                    self.ctrl_msg.angular.z = min(target_angular_z, current_angular_z + 0.2)
                else:
                    self.ctrl_msg.angular.z = max(target_angular_z, current_angular_z - 0.2)

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