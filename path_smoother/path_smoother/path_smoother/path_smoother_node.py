"""
Path Smoother ROS2 Node
========================
Reads waypoints from ROS2 parameters, applies cubic spline smoothing,
and publishes the smooth path as a nav_msgs/Path message.

Subscriptions : None
Publications  : /smooth_path  (nav_msgs/Path)
                /waypoints_viz (visualization_msgs/Marker)
Parameters    : waypoints (flat double array, pairs of x,y)
                num_samples (int, default 500)
                frame_id (string, default 'odom')
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy

from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, Point
from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA
from builtin_interfaces.msg import Time

from path_smoother.path_smoother import smooth_path, compute_path_length


class PathSmootherNode(Node):
    """
    ROS2 node that smooths a coarse waypoint list into a continuous path.

    The smoothed path is published once on a latched topic so downstream
    nodes can subscribe at any time and receive the path.
    """

    def __init__(self):
        super().__init__("path_smoother_node")

        # ── Parameters ────────────────────────────────────────────────────────
        self.declare_parameter("waypoints", [0.0, 0.0, 1.0, 0.5, 2.0, 1.2,
                                              3.0, 1.0, 4.0, 0.0])
        self.declare_parameter("num_samples", 500)
        self.declare_parameter("frame_id", "odom")

        flat = self.get_parameter("waypoints").value
        num_samples = self.get_parameter("num_samples").value
        self.frame_id = self.get_parameter("frame_id").value

        # Parse flat [x0,y0,x1,y1,...] into [(x,y), ...]
        if len(flat) % 2 != 0:
            self.get_logger().error("waypoints parameter must have an even number of values!")
            raise ValueError("Odd number of waypoint values.")
        self.waypoints = [(flat[i], flat[i + 1]) for i in range(0, len(flat), 2)]
        self.get_logger().info(f"Loaded {len(self.waypoints)} waypoints.")

        # ── Publishers (latched = transient_local) ────────────────────────────
        latched_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.path_pub = self.create_publisher(Path, "/smooth_path", latched_qos)
        self.marker_pub = self.create_publisher(
            Marker, "/waypoints_viz", latched_qos
        )

        # ── Compute and publish ───────────────────────────────────────────────
        self.smooth_pts = smooth_path(self.waypoints, num_samples=num_samples)
        total_len = compute_path_length(self.smooth_pts)
        self.get_logger().info(
            f"Smooth path generated: {len(self.smooth_pts)} points, "
            f"length = {total_len:.2f} m"
        )

        self._publish_path()
        self._publish_waypoint_markers()

    # ── Private helpers ───────────────────────────────────────────────────────

    def _stamp(self) -> Time:
        return self.get_clock().now().to_msg()

    def _publish_path(self):
        """Publish the smooth path as nav_msgs/Path."""
        msg = Path()
        msg.header.stamp = self._stamp()
        msg.header.frame_id = self.frame_id

        for x, y in self.smooth_pts:
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose.position.x = x
            ps.pose.position.y = y
            ps.pose.position.z = 0.0
            ps.pose.orientation.w = 1.0
            msg.poses.append(ps)

        self.path_pub.publish(msg)
        self.get_logger().info("Published /smooth_path")

    def _publish_waypoint_markers(self):
        """Publish original waypoints as sphere markers for RViz."""
        marker = Marker()
        marker.header.stamp = self._stamp()
        marker.header.frame_id = self.frame_id
        marker.ns = "waypoints"
        marker.id = 0
        marker.type = Marker.SPHERE_LIST
        marker.action = Marker.ADD
        marker.scale.x = 0.15
        marker.scale.y = 0.15
        marker.scale.z = 0.15
        marker.color = ColorRGBA(r=1.0, g=0.3, b=0.0, a=1.0)

        for x, y in self.waypoints:
            p = Point(x=float(x), y=float(y), z=0.0)
            marker.points.append(p)

        self.marker_pub.publish(marker)
        self.get_logger().info("Published /waypoints_viz markers")


def main(args=None):
    rclpy.init(args=args)
    node = PathSmootherNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
