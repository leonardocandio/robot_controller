# ROS2 module imports
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Point
from sensor_msgs.msg import LaserScan
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from rclpy.qos import qos_profile_sensor_data
from rclpy.duration import Duration

# Python module imports
import numpy as np
import queue
import time
from collections import deque
from enum import Enum

class NavigationState(Enum):
    EXPLORING = 1
    OBSTACLE_AVOIDANCE = 2
    WALL_FOLLOWING = 3
    BACKTRACKING = 4

class PIDController:
    def __init__(self, kP, kI, kD, kS):
        self.kP = kP
        self.kI = kI
        self.kD = kD
        self.kS = kS
        self.err_int = 0
        self.err_dif = 0
        self.err_prev = 0
        self.err_hist = queue.Queue(self.kS)
        self.t_prev = 0

    def reset(self):
        self.err_int = 0
        self.err_dif = 0
        self.err_prev = 0
        self.err_hist = queue.Queue(self.kS)
        self.t_prev = 0

    def control(self, err, t):
        dt = t - self.t_prev
        if dt > 0.0:
            self.err_hist.put(err)
            self.err_int += err
            if self.err_hist.full():
                self.err_int -= self.err_hist.get()
            self.err_dif = err - self.err_prev
            u = (self.kP * err) + (self.kI * self.err_int * dt) + (self.kD * self.err_dif / dt)
            self.err_prev = err
            self.t_prev = t
            return u
        return 0

class ObstacleMemory:
    def __init__(self, memory_size=100):
        self.memory_size = memory_size
        self.obstacles = deque(maxlen=memory_size)
        self.last_position = Point()
        
    def add_obstacle(self, position, distance):
        self.obstacles.append((position, distance, time.time()))
        
    def get_nearby_obstacles(self, current_position, threshold=1.0):
        recent_obstacles = []
        current_time = time.time()
        for obs in self.obstacles:
            if current_time - obs[2] < 10.0:  # Only consider obstacles detected in last 10 seconds
                distance = np.sqrt((current_position.x - obs[0].x)**2 + 
                                 (current_position.y - obs[0].y)**2)
                if distance < threshold:
                    recent_obstacles.append(obs)
        return recent_obstacles

class RobotController(Node):
    def __init__(self):
        super().__init__("robot_controller")
        
        # ROS2 infrastructure setup
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_RELIABLE,
            history=QoSHistoryPolicy.RMW_QOS_POLICY_HISTORY_KEEP_LAST,
            depth=10,
        )

        self.robot_lidar_sub = self.create_subscription(
            LaserScan, "/scan", self.robot_lidar_callback, qos_profile_sensor_data
        )
        self.robot_ctrl_pub = self.create_publisher(Twist, "/cmd_vel", qos_profile)
        
        # Controllers and parameters
        self.pid_wall = PIDController(0.5, 0.01, 0.1, 10)  # Wall following
        self.pid_obstacle = PIDController(0.8, 0.0, 0.2, 10)  # Obstacle avoidance
        
        # Navigation state
        self.current_state = NavigationState.EXPLORING
        self.obstacle_memory = ObstacleMemory()
        self.position = Point()
        self.orientation = 0.0
        self.stuck_counter = 0
        self.last_positions = deque(maxlen=50)
        
        # Timer setup
        timer_period = 0.05  # 20Hz control loop
        self.timer = self.create_timer(timer_period, self.robot_controller_callback)
        
        # State variables
        self.lidar_available = False
        self.laserscan = None
        self.ctrl_msg = Twist()
        
    def robot_lidar_callback(self, msg):
        ranges = np.array(msg.ranges)
        ranges = np.nan_to_num(ranges, nan=3.5, posinf=3.5, neginf=3.5)
        ranges = np.clip(ranges, 0.1, 3.5)
        self.laserscan = ranges
        self.lidar_available = True
        
    def update_position(self, linear_vel, angular_vel, dt):
        # Simple odometry update
        self.orientation += angular_vel * dt
        self.position.x += linear_vel * np.cos(self.orientation) * dt
        self.position.y += linear_vel * np.sin(self.orientation) * dt
        self.last_positions.append((self.position.x, self.position.y))
        
    def check_if_stuck(self):
        if len(self.last_positions) < 50:
            return False
        
        positions = np.array(list(self.last_positions))
        variance = np.var(positions, axis=0)
        return np.sum(variance) < 0.01  # Threshold for determining if stuck
        
    def get_sector_distances(self):
        if not self.lidar_available or self.laserscan is None:
            return None
            
        scan_length = len(self.laserscan)
        sectors = {
            'front': np.mean(np.concatenate([self.laserscan[:int(scan_length/8)], 
                                           self.laserscan[-int(scan_length/8):]]), axis=0),
            'front_left': np.mean(self.laserscan[int(scan_length/8):int(scan_length/4)]),
            'left': np.mean(self.laserscan[int(scan_length/4):int(scan_length/2)]),
            'right': np.mean(self.laserscan[int(scan_length/2):int(3*scan_length/4)]),
            'front_right': np.mean(self.laserscan[int(3*scan_length/4):int(7*scan_length/8)])
        }
        return sectors

    def robot_controller_callback(self):
        if not self.lidar_available:
            return

        sectors = self.get_sector_distances()
        if sectors is None:
            return

        # Update stuck detection
        if self.check_if_stuck():
            self.stuck_counter += 1
            if self.stuck_counter > 20:  # Stuck for 1 second
                self.current_state = NavigationState.BACKTRACKING
        else:
            self.stuck_counter = 0

        # State machine for navigation
        if self.current_state == NavigationState.BACKTRACKING:
            # Instead of moving backwards, perform a rotation to find clear path
            self.ctrl_msg.linear.x = 0.0
            self.ctrl_msg.angular.z = 1.0  # Rotate in place
            
            # If there's clear space in front, return to exploring
            if sectors['front'] > 0.8 and sectors['front_left'] > 0.6 and sectors['front_right'] > 0.6:
                self.current_state = NavigationState.EXPLORING
                self.pid_wall.reset()
                self.pid_obstacle.reset()

        elif sectors['front'] < 0.5 or sectors['front_left'] < 0.4 or sectors['front_right'] < 0.4:
            # Enhanced obstacle avoidance without backward movement
            self.current_state = NavigationState.OBSTACLE_AVOIDANCE
            
            # Determine the best rotation direction
            left_space = sectors['front_left'] + sectors['left']
            right_space = sectors['front_right'] + sectors['right']
            turn_direction = 1.0 if left_space > right_space else -1.0
            
            # If very close to obstacle, stop and rotate
            if sectors['front'] < 0.3:
                self.ctrl_msg.linear.x = 0.0
                self.ctrl_msg.angular.z = turn_direction * 1.2  # Faster rotation when too close
            else:
                # Slow forward movement while turning
                self.ctrl_msg.linear.x = 0.05
                self.ctrl_msg.angular.z = turn_direction * 0.8

        elif self.current_state == NavigationState.EXPLORING:
            # Modified wall following behavior
            target_wall_distance = 0.6
            
            # Determine which wall to follow
            left_wall_dist = sectors['left']
            right_wall_dist = sectors['right']
            
            if left_wall_dist < right_wall_dist and left_wall_dist < 1.0:
                # Follow left wall
                wall_error = target_wall_distance - left_wall_dist
                self.ctrl_msg.linear.x = 0.15
                self.ctrl_msg.angular.z = self.pid_wall.control(wall_error, time.time())
            elif right_wall_dist < 1.0:
                # Follow right wall
                wall_error = target_wall_distance - right_wall_dist
                self.ctrl_msg.linear.x = 0.15
                self.ctrl_msg.angular.z = -self.pid_wall.control(wall_error, time.time())
            else:
                # No nearby walls, explore with slight rotation to find walls
                self.ctrl_msg.linear.x = 0.2
                self.ctrl_msg.angular.z = 0.2  # Slight rotation to encourage exploration

        # Apply velocity limits
        self.ctrl_msg.linear.x = np.clip(self.ctrl_msg.linear.x, 0.0, 0.22)  # Minimum velocity is now 0
        self.ctrl_msg.angular.z = np.clip(self.ctrl_msg.angular.z, -2.84, 2.84)
        
        # Update position estimate and publish control
        self.update_position(self.ctrl_msg.linear.x, self.ctrl_msg.angular.z, 0.05)
        self.robot_ctrl_pub.publish(self.ctrl_msg)

def main(args=None):
    rclpy.init(args=args)
    node = RobotController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()