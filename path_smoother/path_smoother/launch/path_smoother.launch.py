"""
Launch file for Path Smoother, Trajectory Generator, and Trajectory Tracker.

Waypoints are loaded from a SEPARATE file (config/waypoints.yaml).
To change waypoints, run:
    python3 waypoint_editor.py                        # edit hardcoded list
    python3 waypoint_editor.py 0,0 1,0.5 2,1.2       # command line
    python3 waypoint_editor.py --interactive          # interactive prompt

You can also override the waypoints file at launch time:
    ros2 launch path_smoother path_smoother.launch.py \
        waypoints_file:=/abs/path/to/my_waypoints.yaml

Usage:
    ros2 launch path_smoother path_smoother.launch.py
    ros2 launch path_smoother path_smoother.launch.py use_sim_time:=true
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory("path_smoother")

    # ── Parameter files ───────────────────────────────────────────────────────
    # params.yaml    — tuning parameters (velocities, controller gains, etc.)
    # waypoints.yaml — robot path waypoints (edited via waypoint_editor.py)
    # These are kept separate so changing the path never touches tuning params.
    params_file    = os.path.join(pkg_share, "config", "params.yaml")
    default_wp_file = os.path.join(pkg_share, "config", "waypoints.yaml")

    use_sim_time   = LaunchConfiguration("use_sim_time",   default="true")
    waypoints_file = LaunchConfiguration("waypoints_file", default=default_wp_file)

    return LaunchDescription([
        # ── Launch arguments ──────────────────────────────────────────────────
        DeclareLaunchArgument(
            "use_sim_time",
            default_value="true",
            description="Use Gazebo simulation clock",
        ),
        DeclareLaunchArgument(
            "waypoints_file",
            default_value=default_wp_file,
            description="Absolute path to a waypoints YAML file. "
                        "Override to swap missions without rebuilding.",
        ),

        LogInfo(msg="Launching Path Smoother system..."),

        # ── Node 1: Path Smoother ─────────────────────────────────────────────
        # Loads BOTH params.yaml (num_samples, frame_id) and waypoints.yaml.
        # ROS2 merges them — the node sees a flat parameter namespace.
        Node(
            package="path_smoother",
            executable="path_smoother_node",
            name="path_smoother_node",
            parameters=[
                params_file,
                waypoints_file,             # <── waypoints come from here
                {"use_sim_time": use_sim_time},
            ],
            output="screen",
            emulate_tty=True,
        ),

        # ── Node 2: Trajectory Generator ──────────────────────────────────────
        # Does not need waypoints — only velocity/acceleration params.
        Node(
            package="path_smoother",
            executable="trajectory_generator_node",
            name="trajectory_generator_node",
            parameters=[params_file, {"use_sim_time": use_sim_time}],
            output="screen",
            emulate_tty=True,
        ),

        # ── Node 3: Trajectory Tracker ────────────────────────────────────────
        # Does not need waypoints — only controller tuning params.
        Node(
            package="path_smoother",
            executable="trajectory_tracker_node",
            name="trajectory_tracker_node",
            parameters=[params_file, {"use_sim_time": use_sim_time}],
            output="screen",
            emulate_tty=True,
        ),
    ])
