"""
Trajectory Generator ROS2 Node
================================
Subscribes to the smooth path, generates a time-parameterized trajectory
with a trapezoidal velocity profile, and publishes it.

Assignment Output Format
------------------------
The primary deliverable topic /trajectory_output (std_msgs/String) publishes
the trajectory in the EXACT format specified by the assignment:

    trajectory = [(x0, y0, t0), (x1, y1, t1), ..., (xn, yn, tn)]

This is also printed to the console at startup for immediate verification.

Subscriptions : /smooth_path         (nav_msgs/Path)
Publications  : /trajectory_output   (std_msgs/String — assignment format ✅)
                /trajectory          (nav_msgs/Path   — RViz visualization)
                /trajectory_markers  (visualization_msgs/MarkerArray)
                /trajectory_data     (std_msgs/String — full JSON for tracker)
Parameters    : max_velocity  (double, default 0.22 m/s)
                acceleration  (double, default 0.1 m/s²)
                frame_id      (string, default 'odom')
"""

import json
import math

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy

from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, Vector3
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA, String

from path_smoother.trajectory_generator import (
    generate_trajectory,
    trajectory_to_tuples,
)
# Trajectory YAML writer — comment this import to fully disable the feature
from path_smoother.trajectory_yaml_writer import write_trajectory_yaml


class TrajectoryGeneratorNode(Node):
    """
    ROS2 node that converts a smooth path into a timed trajectory and
    publishes it in the assignment-required format: [(x, y, t), ...].
    """

    def __init__(self):
        super().__init__("trajectory_generator_node")

        # ── Parameters ────────────────────────────────────────────────────────
        self.declare_parameter("max_velocity",         0.22)
        self.declare_parameter("acceleration",          0.10)
        self.declare_parameter("frame_id",             "odom")
        self.declare_parameter("save_trajectory_yaml",  True)
        self.declare_parameter("trajectory_yaml_path",  "config/trajectory_points.yaml")

        self.max_velocity          = self.get_parameter("max_velocity").value
        self.acceleration          = self.get_parameter("acceleration").value
        self.frame_id              = self.get_parameter("frame_id").value
        self.save_trajectory_yaml  = self.get_parameter("save_trajectory_yaml").value
        self.trajectory_yaml_path  = self.get_parameter("trajectory_yaml_path").value

        # ── QoS ──────────────────────────────────────────────────────────────
        latched_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )

        # ── Subscribers ───────────────────────────────────────────────────────
        self.create_subscription(
            Path, "/smooth_path", self._path_callback, latched_qos
        )

        # ── Publishers ────────────────────────────────────────────────────────

        # PRIMARY: assignment-format output — [(x, y, t), ...]
        self.output_pub = self.create_publisher(
            String, "/trajectory_output", latched_qos
        )
        # RViz path visualization
        self.traj_path_pub = self.create_publisher(
            Path, "/trajectory", latched_qos
        )
        # Velocity arrow markers for RViz
        self.marker_pub = self.create_publisher(
            MarkerArray, "/trajectory_markers", latched_qos
        )
        # Full JSON for tracker node (includes vx, vy, speed)
        self.traj_data_pub = self.create_publisher(
            String, "/trajectory_data", latched_qos
        )

        self.get_logger().info(
            "Trajectory generator node ready, waiting for /smooth_path..."
        )

    # ── Path callback ─────────────────────────────────────────────────────────

    def _path_callback(self, msg: Path):
        """Receive smooth path, generate trajectory, publish all outputs."""
        smooth_pts = [
            (p.pose.position.x, p.pose.position.y) for p in msg.poses
        ]
        self.get_logger().info(
            f"Received smooth path with {len(smooth_pts)} points."
        )

        trajectory = generate_trajectory(
            smooth_pts,
            max_velocity=self.max_velocity,
            acceleration=self.acceleration,
        )

        total_time = trajectory[-1].t
        self.get_logger().info(
            f"Trajectory generated: {len(trajectory)} points, "
            f"duration = {total_time:.2f} s"
        )

        # ── Publish in assignment format ──────────────────────────────────────
        self._publish_assignment_output(trajectory)

        # ── TRAJECTORY YAML SAVE (comment this block to disable) ─────────────
        # Controlled by params.yaml: save_trajectory_yaml + trajectory_yaml_path
        # To disable without editing params, comment lines below until END marker
        if self.save_trajectory_yaml:
            try:
                write_trajectory_yaml(trajectory, self.trajectory_yaml_path)
                self.get_logger().info(
                    f"Trajectory saved to: {self.trajectory_yaml_path} "
                    f"({len(trajectory)} points)"
                )
            except OSError as e:
                self.get_logger().warn(f"Could not write trajectory YAML: {e}")
        # ── END TRAJECTORY YAML SAVE ──────────────────────────────────────────

        # ── Publish supporting outputs ────────────────────────────────────────
        self._publish_trajectory_path(trajectory)
        self._publish_markers(trajectory)
        self._publish_trajectory_data(trajectory)

    # ── Publishers ────────────────────────────────────────────────────────────

    def _publish_assignment_output(self, trajectory):
        """
        Publish trajectory in the exact assignment-required format:
            trajectory = [(x0, y0, t0), (x1, y1, t1), ..., (xn, yn, tn)]

        Published on: /trajectory_output  (std_msgs/String)
        Also printed to console for immediate verification.
        """
        # Build list of (x, y, t) tuples — assignment format
        assignment_trajectory = trajectory_to_tuples(trajectory)

        # Serialise as a Python-style list-of-tuples string so the output
        # matches the assignment notation exactly when printed
        formatted = "trajectory = [\n"
        for i, (x, y, t) in enumerate(assignment_trajectory):
            comma = "," if i < len(assignment_trajectory) - 1 else ""
            formatted += f"    ({x:.4f}, {y:.4f}, {t:.4f}){comma}\n"
        formatted += "]"

        msg = String()
        msg.data = formatted
        self.output_pub.publish(msg)

        # Print first 5 and last 2 points to console so it's visible in logs
        self.get_logger().info("--- Trajectory output (assignment format) ---")
        preview = assignment_trajectory[:5]
        for x, y, t in preview:
            self.get_logger().info(f"  ({x:.4f}, {y:.4f}, {t:.4f})")
        if len(assignment_trajectory) > 5:
            self.get_logger().info(
                f"  ... {len(assignment_trajectory) - 7} more points ..."
            )
            for x, y, t in assignment_trajectory[-2:]:
                self.get_logger().info(f"  ({x:.4f}, {y:.4f}, {t:.4f})")
        self.get_logger().info(
            f"Published /trajectory_output — {len(assignment_trajectory)} points"
        )

    def _stamp(self):
        return self.get_clock().now().to_msg()

    def _publish_trajectory_path(self, trajectory):
        """Publish trajectory as nav_msgs/Path for RViz."""
        msg = Path()
        msg.header.stamp = self._stamp()
        msg.header.frame_id = self.frame_id
        for tp in trajectory:
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose.position.x = tp.x
            ps.pose.position.y = tp.y
            ps.pose.position.z = 0.0
            ps.pose.orientation.w = 1.0
            msg.poses.append(ps)
        self.traj_path_pub.publish(msg)

    def _publish_markers(self, trajectory):
        """Publish velocity-colored arrow markers for RViz."""
        arr = MarkerArray()
        max_speed = max(tp.speed for tp in trajectory)

        for i, tp in enumerate(trajectory[::10]):  # every 10th point
            m = Marker()
            m.header.stamp = self._stamp()
            m.header.frame_id = self.frame_id
            m.ns = "trajectory_velocity"
            m.id = i
            m.type = Marker.ARROW
            m.action = Marker.ADD
            m.pose.position.x = tp.x
            m.pose.position.y = tp.y
            m.pose.position.z = 0.0

            # Color: blue (slow) → green (fast)
            ratio = tp.speed / max(max_speed, 1e-6)
            m.color = ColorRGBA(r=0.0, g=ratio, b=1.0 - ratio, a=0.8)
            m.scale = Vector3(x=0.1, y=0.02, z=0.02)

            yaw = math.atan2(tp.vy, tp.vx)
            m.pose.orientation.z = math.sin(yaw / 2.0)
            m.pose.orientation.w = math.cos(yaw / 2.0)
            arr.markers.append(m)

        self.marker_pub.publish(arr)

    def _publish_trajectory_data(self, trajectory):
        """
        Publish full trajectory as JSON for the tracker node.
        Includes vx, vy, speed needed by Pure Pursuit + APF controller.
        """
        data = [
            {
                "x": tp.x, "y": tp.y, "t": tp.t,
                "vx": tp.vx, "vy": tp.vy, "speed": tp.speed,
            }
            for tp in trajectory
        ]
        msg = String()
        msg.data = json.dumps(data)
        self.traj_data_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryGeneratorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
