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
        
        # Configure logging
        self.get_logger().set_level(rclpy.logging.LoggingSeverity.INFO)
        self.log_interval = 20  # Log every 20 iterations
        self.iteration_count = 0
        
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
        self.initialization_complete = False
        self.ctrl_msg = Twist()
        self.prefer_left_turns = True
        
        # Robot physical parameters (in meters)
        self.ROBOT_WIDTH = 0.20
        self.ROBOT_LENGTH = 0.20
        self.SAFETY_MARGIN = 0.10  # Additional safety margin around robot
        
        # Obstacle avoidance parameters
        self.CRITICAL_DISTANCE = self.ROBOT_WIDTH/2 + self.SAFETY_MARGIN  # Distance at which to stop
        self.SAFE_DISTANCE = self.CRITICAL_DISTANCE + 0.15  # Distance at which to start avoiding
        self.ROBOT_DETECTION_DISTANCE = 0.5  # Distance to detect other robots
        self.stuck_time = None
        self.recovery_turn_direction = 1  # 1 for left, -1 for right
        self.consecutive_stops = 0
        self.last_movement_time = time.time()
        
        # Path following parameters
        self.last_path_error = 0
        self.path_not_found_count = 0
        self.MAX_PATH_LOSS_COUNT = 10
        self.last_valid_direction = None
        self.image_width = 640  # Default image width
        
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
        
        # Log LIDAR statistics periodically
        if self.iteration_count % self.log_interval == 0:
            min_range = np.min(ranges)
            max_range = np.max(ranges)
            avg_range = np.mean(ranges)
            self.get_logger().debug(
                f'LIDAR Stats - Min: {min_range:.2f}m, Max: {max_range:.2f}m, Avg: {avg_range:.2f}m'
            )

    def camera_callback(self, msg):
        try:
            # Convert ROS Image message to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            height, width = cv_image.shape[:2]
            self.image_width = width  # Store the actual image width
            
            # Convert to grayscale
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            
            # Apply threshold to isolate the path
            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
            
            # Define multiple ROIs for path detection
            roi_bottom = binary[int(height*0.8):height, :]
            roi_middle = binary[int(height*0.6):int(height*0.8), :]
            
            # Find contours in ROIs
            contours_bottom, _ = cv2.findContours(roi_bottom, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours_middle, _ = cv2.findContours(roi_middle, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours_bottom:
                largest_contour = max(contours_bottom, key=cv2.contourArea)
                M = cv2.moments(largest_contour)
                
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    
                    if contours_middle:
                        middle_contour = max(contours_middle, key=cv2.contourArea)
                        M_middle = cv2.moments(middle_contour)
                        if M_middle["m00"] > 0:
                            cx_middle = int(M_middle["m10"] / M_middle["m00"])
                            self.last_valid_direction = cx_middle - cx
                            
                            # Log path direction information
                            if self.iteration_count % self.log_interval == 0:
                                self.get_logger().debug(
                                    f'Path Direction - Bottom Center: {cx}, Middle Center: {cx_middle}, ' +
                                    f'Direction Vector: {self.last_valid_direction}'
                                )
                    
                    self.path_center = cx
                    self.path_detected = True
                    self.path_not_found_count = 0
                    self.last_path_error = (self.image_width/2 - cx) / (self.image_width/2)
                    
                    # Log path detection success
                    if self.iteration_count % self.log_interval == 0:
                        self.get_logger().debug(
                            f'Path Detected - Center: {cx}, Error: {self.last_path_error:.3f}'
                        )
            else:
                self.path_not_found_count += 1
                if self.path_not_found_count > self.MAX_PATH_LOSS_COUNT:
                    self.path_detected = False
                    self.get_logger().warn(
                        f'Path Lost - Frames without path: {self.path_not_found_count}'
                    )
                    
        except Exception as e:
            self.get_logger().error(f'Camera Processing Error: {str(e)}')
            self.path_detected = False

    def detect_robots(self, ranges, scan_length):
        """Detect potential robots in LIDAR data"""
        robot_detections = []
        
        MIN_POINTS = 5
        MAX_DISTANCE_BETWEEN_POINTS = 0.1
        
        current_sequence = []
        
        for i in range(scan_length):
            if ranges[i] < self.ROBOT_DETECTION_DISTANCE:
                if not current_sequence or \
                   abs(ranges[i] - ranges[i-1]) < MAX_DISTANCE_BETWEEN_POINTS:
                    current_sequence.append((i, ranges[i]))
                else:
                    if len(current_sequence) >= MIN_POINTS:
                        robot_detections.append(current_sequence)
                    current_sequence = [(i, ranges[i])]
            else:
                if len(current_sequence) >= MIN_POINTS:
                    robot_detections.append(current_sequence)
                current_sequence = []
        
        # Log robot detections
        if robot_detections and self.iteration_count % self.log_interval == 0:
            self.get_logger().info(
                f'Detected {len(robot_detections)} potential robots'
            )
            for i, detection in enumerate(robot_detections):
                avg_distance = np.mean([p[1] for p in detection])
                angle_start = detection[0][0] * (360.0 / scan_length)
                angle_end = detection[-1][0] * (360.0 / scan_length)
                self.get_logger().debug(
                    f'Robot {i+1} - Distance: {avg_distance:.2f}m, ' +
                    f'Angular Range: {angle_start:.1f}° to {angle_end:.1f}°'
                )
        
        return robot_detections

    def robot_controller_callback(self):
        self.iteration_count += 1
        current_time = self.get_clock().now()
        
        if not self.initialization_complete:
            if self.lidar_available and self.laserscan is not None:
                self.get_logger().info("Initialization complete, starting robot control")
                self.initialization_complete = True
            else:
                if self.iteration_count % 100 == 0:  # Reduce frequency of waiting messages
                    self.get_logger().info("Waiting for sensor data...")
                return

        if self.lidar_available and self.laserscan is not None:
            LIN_VEL = 0.0
            ANG_VEL = 0.0
            
            scan_length = len(self.laserscan)
            step_size = 360.0 / scan_length

            front_sector = max(1, int(45 / step_size))
            side_sector = max(1, int(60 / step_size))

            robot_detections = self.detect_robots(self.laserscan, scan_length)
            
            front_min = min(
                float(np.min(self.laserscan[:front_sector])),
                float(np.min(self.laserscan[-front_sector:]))
            )
            sides_min = float(np.min(self.laserscan[int(scan_length/4):int(3*scan_length/4)]))
            
            robots_nearby = len(robot_detections) > 0
            
            in_critical_situation = (
                front_min < self.CRITICAL_DISTANCE or 
                sides_min < self.CRITICAL_DISTANCE/2 or
                (robots_nearby and front_min < self.ROBOT_DETECTION_DISTANCE)
            )

            # Log robot state periodically
            if self.iteration_count % self.log_interval == 0:
                self.get_logger().debug(
                    f'Robot State - Front Min: {front_min:.2f}m, Sides Min: {sides_min:.2f}m, ' +
                    f'Critical: {in_critical_situation}, Robots Nearby: {robots_nearby}'
                )

            if in_critical_situation:
                if self.stuck_time is None:
                    self.stuck_time = time.time()
                    self.consecutive_stops += 1
                    self.get_logger().warn(
                        f'Critical Situation - Stop #{self.consecutive_stops}, ' +
                        f'Front: {front_min:.2f}m, Sides: {sides_min:.2f}m'
                    )
                elif time.time() - self.stuck_time > 2.0:
                    LIN_VEL = 0.0
                    if self.consecutive_stops > 3:
                        self.recovery_turn_direction *= -1
                        self.consecutive_stops = 0
                        self.get_logger().info(
                            f'Changing recovery direction to: {self.recovery_turn_direction}'
                        )
                    ANG_VEL = 1.0 * self.recovery_turn_direction
            else:
                self.stuck_time = None
                
                if self.path_detected:
                    path_error = self.last_path_error
                    
                    if abs(path_error) > 0.3:
                        LIN_VEL = 0.12
                    elif front_min < self.SAFE_DISTANCE * 1.5:
                        LIN_VEL = 0.15
                    else:
                        LIN_VEL = 0.2
                    
                    if self.last_valid_direction is not None:
                        combined_error = 0.7 * path_error + 0.3 * (self.last_valid_direction / self.image_width)
                        ANG_VEL = self.pid_1_lat.control(combined_error * 1.5, time.time())
                    else:
                        ANG_VEL = self.pid_1_lat.control(path_error * 1.5, time.time())
                    
                    if robots_nearby:
                        LIN_VEL *= 0.7
                        
                    # Log control decisions periodically
                    if self.iteration_count % self.log_interval == 0:
                        self.get_logger().debug(
                            f'Control - Path Error: {path_error:.3f}, ' +
                            f'Linear Vel: {LIN_VEL:.2f}, Angular Vel: {ANG_VEL:.2f}'
                        )
                else:
                    if self.last_valid_direction is not None:
                        LIN_VEL = 0.1
                        ANG_VEL = 0.5 * (self.last_valid_direction / self.image_width)
                        self.get_logger().debug('Using last valid direction for recovery')
                    else:
                        LIN_VEL = 0.05
                        ANG_VEL = self.recovery_turn_direction * 0.8
                        self.get_logger().debug('Basic recovery movement')

            # Smooth velocity transitions
            target_linear_x = min(0.22, float(LIN_VEL))
            current_linear_x = self.ctrl_msg.linear.x
            if target_linear_x > current_linear_x:
                self.ctrl_msg.linear.x = min(target_linear_x, current_linear_x + 0.05)
            else:
                self.ctrl_msg.linear.x = max(target_linear_x, current_linear_x - 0.05)

            target_angular_z = np.clip(ANG_VEL, -2.84, 2.84)
            current_angular_z = self.ctrl_msg.angular.z
            if target_angular_z > current_angular_z:
                self.ctrl_msg.angular.z = min(target_angular_z, current_angular_z + 0.2)
            else:
                self.ctrl_msg.angular.z = max(target_angular_z, current_angular_z - 0.2)

            # Log final velocities periodically
            if self.iteration_count % self.log_interval == 0:
                self.get_logger().debug(
                    f'Final Velocities - Linear: {self.ctrl_msg.linear.x:.2f}, ' +
                    f'Angular: {self.ctrl_msg.angular.z:.2f}'
                )

            # Publish the command
            self.robot_ctrl_pub.publish(self.ctrl_msg)


def main(args=None):
    rclpy.init(args=args)
    node = RobotController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()