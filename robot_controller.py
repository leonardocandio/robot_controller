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

        # PID Controllers
        self.pid_1_lat = PIDController(0.3, 0.01, 0.1, 10)
        self.pid_1_lon = PIDController(0.1, 0.001, 0.005, 10)

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
            '/camera/image_raw', 
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
        
        # More aggressive PID for racing
        self.pid_1_lat = PIDController(0.4, 0.015, 0.15, 10)
        self.pid_1_lon = PIDController(0.15, 0.002, 0.008, 10)

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

                # Safe sector calculation
                front_sector = max(1, int(30 / step_size))  # Wider front sector
                side_sector = max(1, int(45 / step_size))  # Wider side sector

                # Robust distance calculations
                def safe_mean(arr):
                    valid_ranges = arr[np.isfinite(arr)]
                    return np.mean(valid_ranges) if len(valid_ranges) > 0 else 3.5

                # More granular obstacle detection
                front_left = safe_mean(self.laserscan[: int(scan_length / 4)])
                front_right = safe_mean(self.laserscan[-int(scan_length / 4) :])
                front_center = safe_mean(
                    np.concatenate(
                        [self.laserscan[:front_sector], self.laserscan[-front_sector:]]
                    )
                )

                # Additional side and diagonal sectors
                left_side = safe_mean(
                    self.laserscan[int(scan_length / 4) : int(scan_length / 2)]
                )
                right_side = safe_mean(
                    self.laserscan[int(scan_length / 2) : int(3 * scan_length / 4)]
                )

                # Timestamp for PID
                tstamp = time.time()

                # Racing-specific parameters
                FRONT_OBSTACLE_THRESHOLD = 0.6  # More aggressive
                SIDE_OBSTACLE_THRESHOLD = 0.4
                BASE_SPEED = 0.25  # Balanced speed for box corridor
                
                # Combine LIDAR and camera data for better wall following
                left_wall_estimate = min(left_side, self.wall_distance_left)
                right_wall_estimate = min(right_side, self.wall_distance_right)

                # Racing logic for box corridor
                if (
                    front_center < FRONT_OBSTACLE_THRESHOLD
                    or front_left < FRONT_OBSTACLE_THRESHOLD
                    or front_right < FRONT_OBSTACLE_THRESHOLD
                ):
                    # Sharp turn when approaching wall
                    if (front_left > front_right) == self.prefer_left_turns:
                        LIN_VEL = 0.08
                        ANG_VEL = 1.5
                        print("RACING: Sharp left turn")
                    else:
                        LIN_VEL = 0.08
                        ANG_VEL = -1.5
                        print("RACING: Sharp right turn")

                else:
                    # Balanced wall following with combined sensor data
                    LIN_VEL = BASE_SPEED
                    # Use PID for smooth wall following
                    wall_diff = (left_wall_estimate - right_wall_estimate) * 2.0
                    ANG_VEL = self.pid_1_lat.control(
                        wall_diff,
                        time.time(),
                    )
                    
                    # Adjust speed based on corridor width
                    corridor_width = left_wall_estimate + right_wall_estimate
                    if corridor_width > 1.5:  # Wider section
                        LIN_VEL *= 1.2
                    
                    print("RACING: Wall following")

                # Apply velocity limits appropriate for box corridor
                self.ctrl_msg.linear.x = min(0.3, float(LIN_VEL))
                self.ctrl_msg.angular.z = min(2.84, float(ANG_VEL))

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