"""
Trajectory Tracker ROS2 Node
==============================
Subscribes to the trajectory and robot odometry. Uses a hybrid
Pure Pursuit + APF controller to compute and publish velocity commands.

Velocity Safety:
  max_linear_vel and max_angular_vel are loaded from params and passed
  to VelocityLimits.configure(). All commands are hard-clamped inside
  the controller — the node itself never needs to clamp.

Subscriptions : /trajectory_data  (std_msgs/String  — JSON trajectory)
                /odom              (nav_msgs/Odometry)
                /scan              (sensor_msgs/LaserScan — optional, for APF)
Publications  : /cmd_vel          (geometry_msgs/Twist)
                /tracking_path    (nav_msgs/Path — robot actual path for RViz)
Parameters    : lookahead_distance, max_linear_vel, max_angular_vel,
                goal_tolerance, control_frequency,
                apf_k_att, apf_k_rep, apf_d_safe, apf_influence
"""

import json
import math

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy

from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import Twist, PoseStamped
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String

from path_smoother.controller import PurePursuitController, RobotState, VelocityLimits
from path_smoother.trajectory_generator import TrajectoryPoint


def _quat_to_yaw(qx, qy, qz, qw) -> float:
    """Convert quaternion to yaw angle (radians)."""
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


class TrajectoryTrackerNode(Node):
    """
    ROS2 node that runs the hybrid Pure Pursuit + APF controller at a
    fixed frequency and publishes /cmd_vel to Turtlebot3.
    """

    def __init__(self):
        super().__init__("trajectory_tracker_node")

        # ── Parameters ────────────────────────────────────────────────────────
        self.declare_parameter("lookahead_distance", 0.3)
        self.declare_parameter("max_linear_vel",     0.22)
        self.declare_parameter("max_angular_vel",    2.84)
        self.declare_parameter("goal_tolerance",     0.1)
        self.declare_parameter("control_frequency",  20.0)
        self.declare_parameter("apf_k_att",          1.0)
        self.declare_parameter("apf_k_rep",          0.5)
        self.declare_parameter("apf_d_safe",         0.5)
        self.declare_parameter("apf_influence",      0.4)

        lookahead = self.get_parameter("lookahead_distance").value
        max_lin   = self.get_parameter("max_linear_vel").value
        max_ang   = self.get_parameter("max_angular_vel").value
        goal_tol  = self.get_parameter("goal_tolerance").value
        freq      = self.get_parameter("control_frequency").value
        k_att     = self.get_parameter("apf_k_att").value
        k_rep     = self.get_parameter("apf_k_rep").value
        d_safe    = self.get_parameter("apf_d_safe").value
        influence = self.get_parameter("apf_influence").value

        self.get_logger().info(
            f"Velocity limits — linear: [0, {max_lin}] m/s | "
            f"angular: [{-max_ang:.2f}, {max_ang:.2f}] rad/s"
        )

        # ── Controller ────────────────────────────────────────────────────────
        # VelocityLimits.configure() is called inside the constructor,
        # setting the global hard ceiling for all commands.
        self.controller = PurePursuitController(
            lookahead_distance=lookahead,
            max_linear_vel=max_lin,
            max_angular_vel=max_ang,
            goal_tolerance=goal_tol,
            apf_k_att=k_att,
            apf_k_rep=k_rep,
            apf_d_safe=d_safe,
            apf_influence=influence,
        )

        # ── State ─────────────────────────────────────────────────────────────
        self.trajectory: list[TrajectoryPoint] = []
        self.robot_state = RobotState(x=0.0, y=0.0, theta=0.0)
        self.obstacles: list[tuple[float, float]] = []   # from /scan
        self.goal_reached = False
        self.trajectory_received = False
        self.actual_path: list[tuple[float, float]] = []

        # ── QoS profiles ─────────────────────────────────────────────────────
        latched_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        sensor_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )

        # ── Subscribers ───────────────────────────────────────────────────────
        self.create_subscription(
            String, "/trajectory_data", self._trajectory_callback, latched_qos
        )
        self.create_subscription(
            Odometry, "/odom", self._odom_callback, sensor_qos
        )
        # LiDAR scan — optional, used for APF repulsion
        self.create_subscription(
            LaserScan, "/scan", self._scan_callback, sensor_qos
        )

        # ── Publishers ────────────────────────────────────────────────────────
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.tracking_path_pub = self.create_publisher(
            Path, "/tracking_path",
            QoSProfile(depth=1,
                       durability=DurabilityPolicy.TRANSIENT_LOCAL,
                       reliability=ReliabilityPolicy.BEST_EFFORT)
        )

        # ── Control timer ─────────────────────────────────────────────────────
        self.timer = self.create_timer(1.0 / freq, self._control_loop)
        self.get_logger().info(
            f"Trajectory tracker ready | lookahead={lookahead}m | freq={freq}Hz"
        )

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _trajectory_callback(self, msg: String):
        """Parse JSON trajectory from trajectory generator node."""
        data = json.loads(msg.data)
        self.trajectory = [
            TrajectoryPoint(
                x=d["x"], y=d["y"], t=d["t"],
                vx=d["vx"], vy=d["vy"], speed=d["speed"],
            )
            for d in data
        ]
        self.controller.reset()
        self.goal_reached = False
        self.trajectory_received = True
        self.actual_path = []
        self.get_logger().info(
            f"Trajectory received: {len(self.trajectory)} points. Tracking started."
        )

    def _odom_callback(self, msg: Odometry):
        """Update robot pose from wheel odometry."""
        pos = msg.pose.pose.position
        ori = msg.pose.pose.orientation
        self.robot_state = RobotState(
            x=pos.x,
            y=pos.y,
            theta=_quat_to_yaw(ori.x, ori.y, ori.z, ori.w),
        )

    def _scan_callback(self, msg: LaserScan):
        """
        Convert LiDAR scan to (x, y) obstacle list in robot frame.
        Only points within APF d_safe radius are meaningful, but we
        pass all valid points and let APFLayer filter by distance.
        """
        obstacles = []
        angle = msg.angle_min
        for r in msg.ranges:
            if msg.range_min < r < msg.range_max:
                # Convert polar (robot frame) → Cartesian (robot frame)
                # APFLayer works in world frame, so transform using robot pose
                ox_robot = r * math.cos(angle)
                oy_robot = r * math.sin(angle)
                # Rotate to world frame
                ox = (self.robot_state.x
                      + ox_robot * math.cos(self.robot_state.theta)
                      - oy_robot * math.sin(self.robot_state.theta))
                oy = (self.robot_state.y
                      + ox_robot * math.sin(self.robot_state.theta)
                      + oy_robot * math.cos(self.robot_state.theta))
                obstacles.append((ox, oy))
            angle += msg.angle_increment
        self.obstacles = obstacles

    # ── Control loop ──────────────────────────────────────────────────────────

    def _control_loop(self):
        """
        Main control loop at control_frequency Hz.
        Passes current obstacles to the controller for APF repulsion.
        The returned command is already hard-clamped by VelocityLimits.
        """
        if not self.trajectory_received or self.goal_reached:
            return

        cmd, done = self.controller.compute_command(
            self.robot_state,
            self.trajectory,
            obstacles=self.obstacles,   # APF repulsion input
        )

        # Publish — cmd is guaranteed within VelocityLimits
        twist = Twist()
        twist.linear.x = cmd.linear_x
        twist.angular.z = cmd.angular_z
        self.cmd_pub.publish(twist)

        # Record actual path for RViz
        self.actual_path.append((self.robot_state.x, self.robot_state.y))
        self._publish_tracking_path()

        if done:
            self.goal_reached = True
            self.cmd_pub.publish(Twist())   # explicit stop
            self.get_logger().info("Goal reached! Robot stopped.")

    def _publish_tracking_path(self):
        """Publish robot's actual travelled path for RViz comparison."""
        msg = Path()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "odom"
        for x, y in self.actual_path:
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose.position.x = x
            ps.pose.position.y = y
            ps.pose.orientation.w = 1.0
            msg.poses.append(ps)
        self.tracking_path_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryTrackerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
