"""
Trajectory Generator Module
============================
Converts a smooth 2D path into a time-parameterized trajectory.

Velocity Profile: Trapezoidal
  - Accelerate from 0 to max_velocity over accel_distance
  - Cruise at max_velocity
  - Decelerate back to 0 over decel_distance

Output format: [(x, y, t), ...] matching assignment specification.
"""

import numpy as np
from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class TrajectoryPoint:
    """Single time-stamped point on the trajectory."""
    x: float
    y: float
    t: float
    vx: float = 0.0   # velocity x-component (m/s)
    vy: float = 0.0   # velocity y-component (m/s)
    speed: float = 0.0  # scalar speed (m/s)


def trapezoidal_speed_profile(
    distances: np.ndarray,
    max_velocity: float,
    acceleration: float,
) -> np.ndarray:
    """
    Compute a trapezoidal speed profile along arc-length distances.

    The robot accelerates at the start, cruises, then decelerates at the end.
    If the path is too short to reach max_velocity, a triangular profile is used.

    Args:
        distances:    Cumulative arc-length array (starts at 0).
        max_velocity: Maximum cruise speed in m/s.
        acceleration: Acceleration / deceleration magnitude in m/s².

    Returns:
        Array of speeds (m/s) at each distance sample.
    """
    total_length = distances[-1]

    # Distance required to ramp up / down
    ramp_dist = (max_velocity ** 2) / (2.0 * acceleration)

    if 2.0 * ramp_dist >= total_length:
        # Triangular profile: path too short to reach max_velocity
        peak_v = np.sqrt(acceleration * total_length)
        ramp_dist = total_length / 2.0
        max_velocity = peak_v

    speeds = np.zeros_like(distances)
    for i, d in enumerate(distances):
        if d < ramp_dist:
            # Acceleration phase
            speeds[i] = np.sqrt(2.0 * acceleration * d)
        elif d > total_length - ramp_dist:
            # Deceleration phase
            remaining = total_length - d
            speeds[i] = np.sqrt(2.0 * acceleration * max(remaining, 0.0))
        else:
            # Cruise phase
            speeds[i] = max_velocity

    # Clamp minimum speed so we never divide by zero
    speeds = np.clip(speeds, 1e-4, max_velocity)
    return speeds


def generate_trajectory(
    smooth_path: List[Tuple[float, float]],
    max_velocity: float = 0.3,
    acceleration: float = 0.1,
) -> List[TrajectoryPoint]:
    """
    Generate a time-parameterized trajectory from a smooth path.

    Args:
        smooth_path:  List of (x, y) tuples (output of path smoother).
        max_velocity: Maximum speed in m/s (Turtlebot3 max ~0.22 m/s burger).
        acceleration: Linear acceleration in m/s².

    Returns:
        List of TrajectoryPoint with position, time, and velocity.

    Raises:
        ValueError: If smooth_path has fewer than 2 points.
    """
    if len(smooth_path) < 2:
        raise ValueError("smooth_path must have at least 2 points.")

    pts = np.array(smooth_path)

    # Compute cumulative arc-length distances
    diffs = np.diff(pts, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    distances = np.concatenate([[0.0], np.cumsum(seg_lengths)])

    # Compute speed at each point using trapezoidal profile
    speeds = trapezoidal_speed_profile(distances, max_velocity, acceleration)

    # Integrate time: dt = ds / v
    seg_speeds_avg = (speeds[:-1] + speeds[1:]) / 2.0
    dt_array = seg_lengths / seg_speeds_avg
    timestamps = np.concatenate([[0.0], np.cumsum(dt_array)])

    # Build trajectory points with velocity vectors
    trajectory: List[TrajectoryPoint] = []
    for i in range(len(pts)):
        if i < len(pts) - 1:
            direction = diffs[i] / (seg_lengths[i] + 1e-9)
        else:
            direction = diffs[-1] / (seg_lengths[-1] + 1e-9)

        tp = TrajectoryPoint(
            x=float(pts[i, 0]),
            y=float(pts[i, 1]),
            t=float(timestamps[i]),
            vx=float(direction[0] * speeds[i]),
            vy=float(direction[1] * speeds[i]),
            speed=float(speeds[i]),
        )
        trajectory.append(tp)

    return trajectory


def trajectory_to_tuples(
    trajectory: List[TrajectoryPoint],
) -> List[Tuple[float, float, float]]:
    """
    Convert trajectory to assignment format: [(x, y, t), ...].

    Args:
        trajectory: List of TrajectoryPoint objects.

    Returns:
        List of (x, y, t) tuples.
    """
    return [(tp.x, tp.y, tp.t) for tp in trajectory]
