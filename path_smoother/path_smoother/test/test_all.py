"""
Unit Tests for Path Smoother Assignment
========================================
Tests cover:
  - Path smoother (cubic spline)
  - Trajectory generator (trapezoidal profile)
  - Pure Pursuit controller
  - Edge cases and error handling

Run with:
    cd ros2_ws/src/path_smoother
    pytest test/ -v
"""

import math
import pytest
from typing import List, Tuple

from path_smoother.path_smoother import smooth_path, compute_path_length, compute_arc_length_parameterization
from path_smoother.trajectory_generator import (
    generate_trajectory,
    trajectory_to_tuples,
    trapezoidal_speed_profile,
    TrajectoryPoint,
)
from path_smoother.controller import PurePursuitController, RobotState, VelocityCommand

import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def simple_waypoints() -> List[Tuple[float, float]]:
    return [(0.0, 0.0), (1.0, 0.5), (2.0, 1.2), (3.0, 1.0), (4.0, 0.0)]

@pytest.fixture
def straight_waypoints() -> List[Tuple[float, float]]:
    return [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0)]

@pytest.fixture
def two_waypoints() -> List[Tuple[float, float]]:
    return [(0.0, 0.0), (3.0, 0.0)]

@pytest.fixture
def simple_trajectory(simple_waypoints):
    smooth = smooth_path(simple_waypoints, num_samples=200)
    return generate_trajectory(smooth, max_velocity=0.22, acceleration=0.1)


# ══════════════════════════════════════════════════════════════════════════════
# Path Smoother Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestPathSmoother:

    def test_output_length(self, simple_waypoints):
        """Smooth path should return the requested number of samples."""
        result = smooth_path(simple_waypoints, num_samples=300)
        assert len(result) == 300

    def test_output_format(self, simple_waypoints):
        """Each point should be a tuple of two floats."""
        result = smooth_path(simple_waypoints)
        for point in result:
            assert len(point) == 2
            assert isinstance(point[0], float)
            assert isinstance(point[1], float)

    def test_start_end_proximity(self, simple_waypoints):
        """Smooth path should start near the first waypoint and end near the last."""
        result = smooth_path(simple_waypoints, num_samples=500)
        start = result[0]
        end = result[-1]
        assert math.hypot(start[0] - 0.0, start[1] - 0.0) < 0.05
        assert math.hypot(end[0] - 4.0, end[1] - 0.0) < 0.05

    def test_straight_line(self, straight_waypoints):
        """Smooth path on collinear points should be nearly straight."""
        result = smooth_path(straight_waypoints, num_samples=100)
        # All y-values should be near 0
        for x, y in result:
            assert abs(y) < 0.01, f"Expected y≈0, got y={y}"

    def test_two_waypoints(self, two_waypoints):
        """Two-waypoint edge case should return a straight line."""
        result = smooth_path(two_waypoints, num_samples=50)
        assert len(result) == 50

    def test_too_few_waypoints_raises(self):
        """Single waypoint should raise ValueError."""
        with pytest.raises(ValueError):
            smooth_path([(0.0, 0.0)])

    def test_empty_waypoints_raises(self):
        """Empty waypoints should raise ValueError."""
        with pytest.raises(ValueError):
            smooth_path([])

    def test_path_length_positive(self, simple_waypoints):
        """Path length should be positive."""
        result = smooth_path(simple_waypoints)
        length = compute_path_length(result)
        assert length > 0.0

    def test_path_length_monotonic(self, straight_waypoints):
        """Path length should roughly equal the straight-line distance for collinear points."""
        result = smooth_path(straight_waypoints, num_samples=200)
        length = compute_path_length(result)
        expected = 3.0  # (0,0) to (3,0)
        assert abs(length - expected) < 0.1

    def test_arc_length_parameterization_starts_at_zero(self, simple_waypoints):
        """Arc-length parameterization should start at 0."""
        t = compute_arc_length_parameterization(simple_waypoints)
        assert t[0] == 0.0

    def test_arc_length_parameterization_monotonic(self, simple_waypoints):
        """Arc-length parameterization should be strictly increasing."""
        t = compute_arc_length_parameterization(simple_waypoints)
        assert all(t[i] < t[i+1] for i in range(len(t) - 1))


# ══════════════════════════════════════════════════════════════════════════════
# Trajectory Generator Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestTrajectoryGenerator:

    def test_output_length_matches_path(self, simple_waypoints):
        """Trajectory should have the same number of points as input path."""
        smooth = smooth_path(simple_waypoints, num_samples=150)
        traj = generate_trajectory(smooth)
        assert len(traj) == 150

    def test_timestamps_start_at_zero(self, simple_trajectory):
        """First trajectory point should have t=0."""
        assert simple_trajectory[0].t == pytest.approx(0.0, abs=1e-9)

    def test_timestamps_monotonically_increasing(self, simple_trajectory):
        """Timestamps must strictly increase."""
        times = [tp.t for tp in simple_trajectory]
        assert all(times[i] < times[i+1] for i in range(len(times) - 1))

    def test_speed_within_bounds(self, simple_trajectory):
        """All speeds should be ≤ max_velocity."""
        max_v = 0.22
        for tp in simple_trajectory:
            assert tp.speed <= max_v + 1e-6, f"Speed {tp.speed} exceeded max {max_v}"

    def test_speed_at_endpoints_low(self, simple_trajectory):
        """Speed at start and end should be near zero (trapezoidal profile)."""
        start_speed = simple_trajectory[0].speed
        end_speed = simple_trajectory[-1].speed
        assert start_speed < 0.15, f"Start speed {start_speed} too high"
        assert end_speed < 0.15, f"End speed {end_speed} too high"

    def test_tuple_format(self, simple_trajectory):
        """trajectory_to_tuples should return (x, y, t) tuples."""
        tuples = trajectory_to_tuples(simple_trajectory)
        for item in tuples:
            assert len(item) == 3

    def test_too_few_points_raises(self):
        """Single point should raise ValueError."""
        with pytest.raises(ValueError):
            generate_trajectory([(0.0, 0.0)])

    def test_trapezoidal_profile_shape(self):
        """Trapezoidal profile should have low-high-low speed pattern."""
        distances = np.linspace(0, 10, 200)
        speeds = trapezoidal_speed_profile(distances, max_velocity=0.5, acceleration=0.2)
        mid_idx = len(speeds) // 2
        assert speeds[0] < speeds[mid_idx]      # start slower than middle
        assert speeds[-1] < speeds[mid_idx]     # end slower than middle

    def test_trajectory_positions_match_path(self, simple_waypoints):
        """Trajectory x,y positions should match the smooth path exactly."""
        smooth = smooth_path(simple_waypoints, num_samples=100)
        traj = generate_trajectory(smooth)
        for i, tp in enumerate(traj):
            assert tp.x == pytest.approx(smooth[i][0], abs=1e-9)
            assert tp.y == pytest.approx(smooth[i][1], abs=1e-9)


# ══════════════════════════════════════════════════════════════════════════════
# Controller Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestPurePursuitController:

    @pytest.fixture
    def controller(self):
        return PurePursuitController(
            lookahead_distance=0.3,
            max_linear_vel=0.22,
            max_angular_vel=2.84,
            goal_tolerance=0.1,
        )

    @pytest.fixture
    def short_trajectory(self):
        """A simple straight-ahead trajectory."""
        points = [(float(i) * 0.1, 0.0) for i in range(50)]
        smooth = smooth_path(points, num_samples=100)
        return generate_trajectory(smooth)

    def test_goal_reached_when_at_goal(self, controller, short_trajectory):
        """Robot at the goal position should trigger goal_reached."""
        last = short_trajectory[-1]
        state = RobotState(x=last.x, y=last.y, theta=0.0)
        _, done = controller.compute_command(state, short_trajectory)
        assert done is True

    def test_stop_command_when_goal_reached(self, controller, short_trajectory):
        """When goal is reached, velocity commands should be zero."""
        last = short_trajectory[-1]
        state = RobotState(x=last.x, y=last.y, theta=0.0)
        cmd, _ = controller.compute_command(state, short_trajectory)
        assert cmd.linear_x == pytest.approx(0.0)
        assert cmd.angular_z == pytest.approx(0.0)

    def test_linear_vel_positive_moving_forward(self, controller, short_trajectory):
        """Robot not at goal should receive positive forward velocity."""
        state = RobotState(x=0.0, y=0.0, theta=0.0)
        cmd, done = controller.compute_command(state, short_trajectory)
        if not done:
            assert cmd.linear_x >= 0.0

    def test_linear_vel_within_bounds(self, controller, short_trajectory):
        """Linear velocity must not exceed max_linear_vel."""
        state = RobotState(x=0.0, y=0.0, theta=0.0)
        cmd, _ = controller.compute_command(state, short_trajectory)
        assert abs(cmd.linear_x) <= 0.22 + 1e-6

    def test_angular_vel_within_bounds(self, controller, short_trajectory):
        """Angular velocity must not exceed max_angular_vel."""
        state = RobotState(x=0.5, y=0.5, theta=math.pi / 4)
        cmd, _ = controller.compute_command(state, short_trajectory)
        assert abs(cmd.angular_z) <= 2.84 + 1e-6

    def test_empty_trajectory_returns_goal_reached(self, controller):
        """Empty trajectory should immediately return goal_reached=True."""
        state = RobotState(x=0.0, y=0.0, theta=0.0)
        _, done = controller.compute_command(state, [])
        assert done is True

    def test_reset_clears_cached_index(self, controller, short_trajectory):
        """After reset(), closest index should return to 0."""
        state = RobotState(x=2.0, y=0.0, theta=0.0)
        controller.compute_command(state, short_trajectory)
        assert controller._closest_idx > 0
        controller.reset()
        assert controller._closest_idx == 0

    def test_angular_correction_for_lateral_offset(self, controller, short_trajectory):
        """Robot to the right of path should receive negative angular correction."""
        # Robot below the x-axis path (positive y error in robot frame = left turn needed)
        state = RobotState(x=1.0, y=-0.5, theta=0.0)
        cmd, done = controller.compute_command(state, short_trajectory)
        if not done:
            # Robot is below the path, should steer left (positive angular_z)
            assert cmd.angular_z > 0.0


# ══════════════════════════════════════════════════════════════════════════════
# Integration Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestIntegration:

    def test_full_pipeline_produces_valid_trajectory(self, simple_waypoints):
        """Full pipeline: waypoints → smooth → trajectory should work end-to-end."""
        smooth = smooth_path(simple_waypoints, num_samples=300)
        assert len(smooth) == 300

        traj = generate_trajectory(smooth, max_velocity=0.22, acceleration=0.1)
        assert len(traj) == 300
        assert traj[0].t == pytest.approx(0.0)
        assert traj[-1].t > 0.0

        tuples = trajectory_to_tuples(traj)
        assert len(tuples) == 300
        for x, y, t in tuples:
            assert isinstance(x, float)
            assert isinstance(y, float)
            assert isinstance(t, float)

    def test_controller_reduces_distance_over_time(self, simple_waypoints):
        """Simulating a few control steps should move the robot closer to the goal."""
        smooth = smooth_path(simple_waypoints, num_samples=200)
        traj = generate_trajectory(smooth)
        controller = PurePursuitController(lookahead_distance=0.3)

        state = RobotState(x=0.0, y=0.0, theta=0.0)
        goal = traj[-1]
        initial_dist = math.hypot(goal.x - state.x, goal.y - state.y)

        dt = 0.05  # 20 Hz
        for _ in range(20):
            cmd, done = controller.compute_command(state, traj)
            if done:
                break
            # Simple Euler integration of robot motion
            state = RobotState(
                x=state.x + cmd.linear_x * math.cos(state.theta) * dt,
                y=state.y + cmd.linear_x * math.sin(state.theta) * dt,
                theta=state.theta + cmd.angular_z * dt,
            )

        final_dist = math.hypot(goal.x - state.x, goal.y - state.y)
        assert final_dist < initial_dist, "Robot should have moved closer to the goal"
