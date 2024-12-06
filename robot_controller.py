# ROS2 module imports
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from rclpy.qos import qos_profile_sensor_data
from rclpy.duration import Duration

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

        # PID Controllers - Increased gains for more aggressive control
        self.pid_1_lat = PIDController(0.5, 0.02, 0.15, 10)  # More aggressive lateral control
        self.pid_1_lon = PIDController(0.2, 0.002, 0.01, 10)  # More aggressive longitudinal control

        # State variables
        self.lidar_available = False
        self.laserscan = None
        self.start_mode = "outside"
        self.start_time = self.get_clock().now()
        self.ctrl_msg = Twist()
        self.prefer_left_turns = True

        # Robot physical parameters
        self.robot_width = 0.20  # Robot width in meters
        self.robot_length = 0.20  # Robot length in meters
        self.safety_margin = 0.10  # Safety margin in meters

    def robot_lidar_callback(self, msg):
        # Robust LIDAR data preprocessing
        ranges = np.array(msg.ranges)

        # Replace NaN and inf with maximum sensor range
        ranges = np.nan_to_num(ranges, nan=3.5, posinf=3.5, neginf=3.5)

        # Filter out extreme or invalid ranges
        ranges = np.clip(ranges, 0.1, 3.5)

        self.laserscan = ranges
        self.lidar_available = True

    def robot_controller_callback(self):
        DELAY = 2.0  # Reduced initialization delay for racing
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

                # Adjusted thresholds considering robot size
                FRONT_OBSTACLE_THRESHOLD = 0.5  # Reduced from 0.7 for more aggressive approach
                SIDE_OBSTACLE_THRESHOLD = self.robot_width/2 + self.safety_margin  # About 0.3m
                RACING_DISTANCE = 0.4  # Optimal distance from wall for racing

                # Debugging print statements
                print(
                    f"Distances - Front: {front_center:.2f}, Left: {front_left:.2f}, Right: {front_right:.2f}, Side-Left: {left_side:.2f}, Side-Right: {right_side:.2f}"
                )

                # Enhanced Racing Logic
                if front_center < FRONT_OBSTACLE_THRESHOLD:
                    # Emergency obstacle avoidance
                    if front_left > front_right:
                        LIN_VEL = 0.1  # Increased from 0.05 for more aggressive turning
                        ANG_VEL = 1.5  # Increased from 1.2 for sharper turns
                        print("EMERGENCY: Sharp Left Turn")
                    else:
                        LIN_VEL = 0.1
                        ANG_VEL = -1.5
                        print("EMERGENCY: Sharp Right Turn")

                elif left_side < SIDE_OBSTACLE_THRESHOLD or right_side < SIDE_OBSTACLE_THRESHOLD:
                    # Side obstacle avoidance with racing optimization
                    if left_side < SIDE_OBSTACLE_THRESHOLD:
                        LIN_VEL = 0.15  # Increased from 0.1
                        ANG_VEL = -1.0  # Increased from -0.8
                        print("RACING: Avoiding Left Wall")
                    else:
                        LIN_VEL = 0.15
                        ANG_VEL = 1.0
                        print("RACING: Avoiding Right Wall")

                else:
                    # Optimized racing behavior
                    target_error = (left_side - RACING_DISTANCE) if self.prefer_left_turns else (right_side - RACING_DISTANCE)
                    
                    # Dynamic velocity based on path curvature
                    curve_factor = abs(target_error) * 2.0
                    LIN_VEL = 0.3 * (1.0 - min(curve_factor, 0.5))  # Max speed 0.3 m/s
                    
                    # More aggressive turning for optimal racing line
                    ANG_VEL = self.pid_1_lat.control(
                        target_error * 2.5,  # Increased sensitivity for racing
                        time.time()
                    )
                    print("RACING: Optimal Line")

                # Updated Velocity Limits for Racing
                self.ctrl_msg.linear.x = min(0.3, max(0.05, float(LIN_VEL)))  # Increased max speed
                self.ctrl_msg.angular.z = min(3.0, max(-3.0, float(ANG_VEL)))  # Increased max turn rate

                # Publish control message
                self.robot_ctrl_pub.publish(self.ctrl_msg)
        else:
            print("Race preparation...")


def main(args=None):
    rclpy.init(args=args)
    node = RobotController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()