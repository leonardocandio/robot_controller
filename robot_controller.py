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
            if self.err_hist.full():  # Jacketing logic to prevent integral windup
                self.err_int -= self.err_hist.get()  # Rolling FIFO buffer
            self.err_dif = err - self.err_prev  # Error difference
            u = (
                (self.kP * err)
                + (self.kI * self.err_int * dt)
                + (self.kD * self.err_dif / dt)
            )  # PID control law
            self.err_prev = err  # Update previous error term
            self.t_prev = t  # Update timestamp
            return u  # Control signal
        return 0


class RobotController(Node):
    def __init__(self):

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

        # PID Controllers
        self.pid_1_lat = PIDController(0.3, 0.01, 0.12, 10)
        self.pid_1_lon = PIDController(0.15, 0.002, 0.008, 10)

        # State variables
        self.lidar_available = False
        self.laserscan = None
        self.start_mode = "outside"
        self.start_time = self.get_clock().now()
        self.ctrl_msg = Twist()
        self.prefer_left_turns = True

        # Add camera subscriber for detecting boxes/walls
        self.camera_sub = self.create_subscription(
            Image, 
            '/image_raw', 
            self.camera_callback,
            qos_profile_sensor_data
        )
        self.cv_bridge = CvBridge()
        
        # Racing state variables
        self.prefer_left_turns = True
        self.race_mode = True  # Enable race mode
        self.detected_robot = False
        self.wall_distance_left = float('inf')
        self.wall_distance_right = float('inf')
        
        # More aggressive PID for racing but smoother turning
        self.pid_1_lat = PIDController(0.3, 0.01, 0.12, 10)  # Reduced gains for smoother turns
        self.pid_1_lon = PIDController(0.15, 0.002, 0.008, 10)

        # Adjust thresholds for larger robot and water bottles
        self.ROBOT_WIDTH = 0.5  # meters
        self.MIN_SAFE_DISTANCE = 0.45  # increased minimum safe distance
        self.WIDE_TURN_THRESHOLD = 1.5  # threshold to detect wide turns
        self.consecutive_wide_readings = 0  # counter for wide turn detection
        
        # Reduced PID values for gentler control
        self.pid_1_lat = PIDController(0.15, 0.003, 0.06, 10)  # Even gentler turns
        self.pid_1_lon = PIDController(0.15, 0.002, 0.008, 10)

        # Add direction tracking
        self.wrong_direction_count = 0
        self.last_five_angles = []  # Track recent angular velocities

    def robot_lidar_callback(self, msg):
        # Robust LIDAR data preprocessing
        ranges = np.array(msg.ranges)

        # Replace NaN and inf with maximum sensor range
        ranges = np.nan_to_num(ranges, nan=3.5, posinf=3.5, neginf=3.5)

        # Filter out extreme or invalid ranges
        ranges = np.clip(ranges, 0.1, 3.5)

        self.laserscan = ranges
        self.lidar_available = True

    def camera_callback(self, msg):
        """Process camera image for wall detection and robot detection"""
        try:
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
            
            if cv_image is not None:
                # Convert to grayscale
                gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
                
                # Split image into left and right halves
                height, width = gray.shape
                left_half = gray[:, :width//2]
                right_half = gray[:, width//2:]
                
                # Detect edges of boxes/walls
                left_edges = cv2.Canny(left_half, 50, 150)
                right_edges = cv2.Canny(right_half, 50, 150)
                
                # Calculate approximate wall distances based on edge density
                self.wall_distance_left = 1.0 - (cv2.countNonZero(left_edges) / left_edges.size)
                self.wall_distance_right = 1.0 - (cv2.countNonZero(right_edges) / right_edges.size)
                
                # Simple robot detection (looking for movement/changes)
                # This could be enhanced based on the appearance of other robots
                # For now, we'll rely more on LIDAR for robot detection

        except Exception as e:
            self.get_logger().warn(f"Camera processing error: {str(e)}")

    def robot_controller_callback(self):
        DELAY = 4.0
        if self.get_clock().now() - self.start_time > Duration(seconds=DELAY):
            if self.lidar_available and self.laserscan is not None:
                # Dynamically calculate scan parameters
                scan_length = len(self.laserscan)
                step_size = 360.0 / scan_length

                # Robust distance calculations
                def safe_mean(arr):
                    valid_ranges = arr[np.isfinite(arr)]
                    return np.mean(valid_ranges) if len(valid_ranges) > 0 else 3.5

                # Wider detection sectors
                front_sector = max(1, int(45 / step_size))  # Increased front sector
                side_sector = max(1, int(60 / step_size))  # Increased side sector
                
                # Additional diagonal sectors for water bottle detection
                diagonal_sector = max(1, int(30 / step_size))
                
                # More detailed distance calculations
                front_left = min(safe_mean(self.laserscan[: int(scan_length / 4)]),
                               np.min(self.laserscan[: int(scan_length / 4)]))  # Use minimum for bottle detection
                front_right = min(safe_mean(self.laserscan[-int(scan_length / 4) :]),
                                np.min(self.laserscan[-int(scan_length / 4) :]))
                front_center = min(safe_mean(
                    np.concatenate([self.laserscan[:front_sector], 
                                  self.laserscan[-front_sector:]])),
                    np.min(np.concatenate([self.laserscan[:front_sector], 
                                         self.laserscan[-front_sector:]])))

                # Enhanced side detection
                left_side = min(safe_mean(
                    self.laserscan[int(scan_length / 4) : int(scan_length / 2)]),
                    np.min(self.laserscan[int(scan_length / 4) : int(scan_length / 2)]))
                right_side = min(safe_mean(
                    self.laserscan[int(scan_length / 2) : int(3 * scan_length / 4)]),
                    np.min(self.laserscan[int(scan_length / 2) : int(3 * scan_length / 4)]))

                # Racing parameters
                FRONT_OBSTACLE_THRESHOLD = 0.7  # Adjusted for water bottles
                SIDE_OBSTACLE_THRESHOLD = 0.5
                BASE_SPEED = 0.2  # Reduced base speed for better control
                MAX_ANGULAR_VEL = 0.6  # Reduced for smoother turns
                
                # Wide turn detection
                if left_side > self.WIDE_TURN_THRESHOLD and right_side > self.WIDE_TURN_THRESHOLD:
                    self.consecutive_wide_readings += 1
                else:
                    self.consecutive_wide_readings = 0

                # Racing logic with enhanced obstacle avoidance
                if (
                    front_center < FRONT_OBSTACLE_THRESHOLD
                    or front_left < FRONT_OBSTACLE_THRESHOLD
                    or front_right < FRONT_OBSTACLE_THRESHOLD
                ):
                    # Obstacle avoidance with water bottle consideration
                    LIN_VEL = 0.1  # Slower speed near obstacles
                    if front_left > front_right:
                        ANG_VEL = MAX_ANGULAR_VEL
                        self.get_logger().info("Avoiding obstacle - turning left")
                    else:
                        ANG_VEL = -MAX_ANGULAR_VEL
                        self.get_logger().info("Avoiding obstacle - turning right")

                elif self.consecutive_wide_readings > 5:
                    # Wide turn handling
                    LIN_VEL = 0.15
                    # Maintain current direction but with stronger correction
                    wall_diff = (left_side - right_side) * 2.0
                    ANG_VEL = self.pid_1_lat.control(wall_diff, time.time())
                    ANG_VEL = np.clip(ANG_VEL, -MAX_ANGULAR_VEL * 1.2, MAX_ANGULAR_VEL * 1.2)
                    self.get_logger().info("Wide turn detected - maintaining course")

                else:
                    # Normal wall following with enhanced awareness
                    LIN_VEL = BASE_SPEED
                    # Use minimum distances for wall following
                    left_wall = min(left_side, self.wall_distance_left)
                    right_wall = min(right_side, self.wall_distance_right)
                    
                    # More conservative wall following
                    wall_diff = (left_wall - right_wall) * 1.0
                    ANG_VEL = self.pid_1_lat.control(wall_diff, time.time())
                    ANG_VEL = np.clip(ANG_VEL, -MAX_ANGULAR_VEL, MAX_ANGULAR_VEL)
                    
                    self.get_logger().info(
                        "Normal racing - L: %.2f, R: %.2f, F: %.2f, Diff: %.2f" % 
                        (left_wall, right_wall, front_center, wall_diff)
                    )

                # Velocity limits
                self.ctrl_msg.linear.x = min(0.22, float(LIN_VEL))
                self.ctrl_msg.angular.z = min(MAX_ANGULAR_VEL, float(ANG_VEL))

                self.robot_ctrl_pub.publish(self.ctrl_msg)
        else:
            self.get_logger().info("Initializing... %.1f seconds remaining" % 
                                 (DELAY - (self.get_clock().now() - self.start_time).nanoseconds/1e9))


def main(args=None):
    rclpy.init(args=args)
    node = RobotController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()