"""
Trajectory Tracking Controller Module
=======================================
Implements a hybrid controller combining:
  1. Pure Pursuit  — nominal path tracking
  2. Artificial Potential Fields (APF) — obstacle repulsion + goal attraction
  3. Local minimum escape — detects when robot is stuck and injects escape impulse

Velocity Safety:
  ALL velocity commands are passed through VelocityLimits.clamp() before
  being returned. This is the single, authoritative enforcement point.
  Hard limits: linear ∈ [0, MAX_LINEAR_VEL], angular ∈ [-MAX_ANGULAR_VEL, MAX_ANGULAR_VEL]
  No command anywhere in this module can ever exceed these bounds.

Pure Pursuit Reference:
  Coulter, R.C. (1992). "Implementation of the Pure Pursuit Path Tracking
  Algorithm." Carnegie Mellon University, Robotics Institute.

APF Reference:
  Khatib, O. (1986). "Real-Time Obstacle Avoidance for Manipulators and
  Mobile Robots." International Journal of Robotics Research.
"""

import math
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from path_smoother.trajectory_generator import TrajectoryPoint


# ══════════════════════════════════════════════════════════════════════════════
# Data classes
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class RobotState:
    """Current state of the differential drive robot."""
    x: float        # position x (m)
    y: float        # position y (m)
    theta: float    # heading angle (rad), 0 = facing +x axis


@dataclass
class VelocityCommand:
    """Twist command for a differential drive robot."""
    linear_x: float   # forward velocity (m/s)
    angular_z: float  # yaw rate (rad/s)


@dataclass
class Vec2:
    """Simple 2D vector for force calculations."""
    x: float = 0.0
    y: float = 0.0

    def __add__(self, other: "Vec2") -> "Vec2":
        return Vec2(self.x + other.x, self.y + other.y)

    def magnitude(self) -> float:
        return math.hypot(self.x, self.y)

    def normalized(self) -> "Vec2":
        mag = self.magnitude()
        if mag < 1e-9:
            return Vec2(0.0, 0.0)
        return Vec2(self.x / mag, self.y / mag)

    def angle(self) -> float:
        """Angle of this vector in radians."""
        return math.atan2(self.y, self.x)


# ══════════════════════════════════════════════════════════════════════════════
# Velocity limits — THE single enforcement point for all velocity commands
# ══════════════════════════════════════════════════════════════════════════════

class VelocityLimits:
    """
    Hard velocity limits for the robot.

    These are absolute physical constraints. No command issued by any
    controller layer should ever exceed these values. clamp() is called
    once at the very end of compute_command() and nowhere else — ensuring
    there is exactly one place in the codebase that enforces limits.

    Units: m/s for linear, rad/s for angular.
    """

    # ── Configurable range ────────────────────────────────────────────────────
    # Linear velocity: robot can only move forward (min=0) up to max.
    # Change MAX_LINEAR to whatever your platform supports (e.g. 5.0 m/s).
    MIN_LINEAR: float  = 0.0    # m/s  — no reversing (differential drive)
    MAX_LINEAR: float  = 5.0    # m/s  — hard ceiling (configurable)

    # Angular velocity: symmetric, Turtlebot3 hardware limit is 2.84 rad/s.
    # Raise if your platform allows faster turning.
    MAX_ANGULAR: float = 2.84   # rad/s

    @classmethod
    def clamp(cls, cmd: VelocityCommand) -> VelocityCommand:
        """
        Clamp a velocity command to hard limits.

        This is the ONLY place velocity limits are enforced.
        Every compute_command() call must pass its result through here
        before returning it to the caller.

        Args:
            cmd: Raw velocity command from any controller layer.

        Returns:
            New VelocityCommand with both fields clamped to safe ranges.
        """
        linear = max(cls.MIN_LINEAR, min(cmd.linear_x, cls.MAX_LINEAR))
        angular = max(-cls.MAX_ANGULAR, min(cmd.angular_z, cls.MAX_ANGULAR))
        return VelocityCommand(linear_x=linear, angular_z=angular)

    @classmethod
    def configure(cls, max_linear: float, max_angular: float):
        """
        Update limits at runtime (e.g. from ROS2 parameters).

        Args:
            max_linear:  New maximum linear speed in m/s. Must be > 0.
            max_angular: New maximum angular speed in rad/s. Must be > 0.

        Raises:
            ValueError: If either limit is non-positive.
        """
        if max_linear <= 0 or max_angular <= 0:
            raise ValueError("Velocity limits must be positive.")
        cls.MAX_LINEAR = max_linear
        cls.MAX_ANGULAR = max_angular


# ══════════════════════════════════════════════════════════════════════════════
# Artificial Potential Fields
# ══════════════════════════════════════════════════════════════════════════════

class APFLayer:
    """
    Artificial Potential Fields obstacle avoidance layer.

    Computes attractive force toward the next trajectory waypoint and
    repulsive forces away from nearby obstacles. The net force is used
    to correct the Pure Pursuit heading command.

    Attractive force:  F_att = k_att * (goal - robot)
    Repulsive force:   F_rep = k_rep * (1/d - 1/d_safe) * (1/d²) * away_dir
                       (only active when d < d_safe)

    The net force is projected onto angular correction for the robot.
    """

    def __init__(
        self,
        k_att: float = 1.0,
        k_rep: float = 0.5,
        d_safe: float = 0.5,
        influence_weight: float = 0.4,
    ):
        """
        Args:
            k_att:            Attractive force gain.
            k_rep:            Repulsive force gain.
            d_safe:           Safety radius — repulsion active within this distance (m).
            influence_weight: How much APF blends into the Pure Pursuit command [0-1].
                              0 = pure pursuit only, 1 = full APF override.
        """
        self.k_att = k_att
        self.k_rep = k_rep
        self.d_safe = d_safe
        self.influence_weight = influence_weight

    def attractive_force(
        self,
        state: RobotState,
        target: Tuple[float, float],
    ) -> Vec2:
        """
        Compute attractive force pulling robot toward target waypoint.

        Uses a conic potential (linear with distance) rather than quadratic
        to avoid excessively large forces at long range.

        Args:
            state:  Current robot position.
            target: (x, y) of the next trajectory waypoint.

        Returns:
            Force vector pointing toward target.
        """
        dx = target[0] - state.x
        dy = target[1] - state.y
        dist = math.hypot(dx, dy)
        if dist < 1e-6:
            return Vec2(0.0, 0.0)
        # Normalize then scale by gain — conic potential, not quadratic
        return Vec2(
            x=self.k_att * dx / dist,
            y=self.k_att * dy / dist,
        )

    def repulsive_force(
        self,
        state: RobotState,
        obstacles: List[Tuple[float, float]],
    ) -> Vec2:
        """
        Compute total repulsive force from all nearby obstacles.

        Each obstacle contributes a force inversely proportional to distance
        squared, only within d_safe radius.

        Args:
            state:     Current robot position.
            obstacles: List of (x, y) obstacle positions (e.g. from LiDAR scan).

        Returns:
            Summed repulsive force vector.
        """
        total = Vec2(0.0, 0.0)
        for ox, oy in obstacles:
            dx = state.x - ox
            dy = state.y - oy
            d = math.hypot(dx, dy)
            if d < 1e-6 or d >= self.d_safe:
                continue
            # F_rep = k_rep * (1/d - 1/d_safe) * (1/d²) * direction_away
            magnitude = self.k_rep * (1.0 / d - 1.0 / self.d_safe) * (1.0 / (d * d))
            direction = Vec2(dx / d, dy / d)  # unit vector away from obstacle
            total = total + Vec2(magnitude * direction.x, magnitude * direction.y)
        return total

    def compute_angular_correction(
        self,
        state: RobotState,
        net_force: Vec2,
    ) -> float:
        """
        Convert the net APF force vector into an angular velocity correction.

        Projects the net force angle relative to the robot's current heading
        and scales it to an angular rate correction.

        Args:
            state:     Current robot state (heading used for projection).
            net_force: Combined attractive + repulsive force vector.

        Returns:
            Angular velocity correction in rad/s.
        """
        if net_force.magnitude() < 1e-6:
            return 0.0
        force_angle = net_force.angle()
        angle_error = force_angle - state.theta
        # Wrap to [-pi, pi]
        angle_error = math.atan2(math.sin(angle_error), math.cos(angle_error))
        return self.influence_weight * angle_error


# ══════════════════════════════════════════════════════════════════════════════
# Goal brake — ramps velocity to zero as robot approaches final waypoint
# ══════════════════════════════════════════════════════════════════════════════

class GoalBrake:
    """
    Smoothly ramps linear AND angular velocity to zero as the robot
    approaches the final goal, then holds zero until fully settled.

    Why this is needed
    ------------------
    Pure Pursuit computes angular_z based on curvature to a lookahead point.
    When the goal is reached, the last command issued may have a non-zero
    angular_z (the robot was mid-turn). Sending a sudden zero command causes
    the robot to stop but Gazebo / real motors may still coast through the
    last angular impulse, leaving the robot spinning in place.

    Solution — two-stage approach:
      Stage 1 (brake_distance):  Scale both linear and angular commands
                                 by a ramp factor that goes 1.0 → 0.0 as
                                 the robot closes in on the goal.
      Stage 2 (settle_steps):    Once position is reached, actively publish
                                 zero for `settle_steps` cycles so the motor
                                 controller receives an explicit stop, not
                                 just silence on the topic.

    Parameters
    ----------
    brake_distance : float
        Distance from goal at which braking begins (m).
        Should be >= goal_tolerance. Default 0.4 m.
    settle_steps : int
        Number of control cycles to actively publish zero after arrival.
        At 20 Hz, 10 steps = 0.5 s of explicit zero commands. Default 10.
    angular_brake_gain : float
        Extra multiplier applied to angular_z during braking (< 1.0 makes
        angular braking more aggressive than linear). Default 0.5.
    """

    def __init__(
        self,
        brake_distance: float = 0.4,
        settle_steps: int = 10,
        angular_brake_gain: float = 0.5,
    ):
        self.brake_distance     = brake_distance
        self.settle_steps       = settle_steps
        self.angular_brake_gain = angular_brake_gain
        self._settling: int     = 0   # countdown of settle cycles remaining
        self._settled: bool     = False

    def apply(
        self,
        cmd: VelocityCommand,
        dist_to_goal: float,
        goal_tolerance: float,
    ) -> Tuple[VelocityCommand, bool]:
        """
        Apply braking to a velocity command based on distance to goal.

        Args:
            cmd:            Raw command from Pure Pursuit + APF layers.
            dist_to_goal:   Current Euclidean distance to final goal (m).
            goal_tolerance: Radius at which position is considered reached (m).

        Returns:
            (braked_cmd, fully_settled)
            fully_settled is True only after both position is reached AND
            the settle countdown has completed — guaranteeing the robot
            has received explicit zero commands for settle_steps cycles.
        """
        # ── Stage 2: settling (position already reached) ──────────────────────
        if self._settling > 0:
            self._settling -= 1
            # Actively command zero — do not just return silence
            return VelocityCommand(0.0, 0.0), (self._settling == 0)

        if self._settled:
            return VelocityCommand(0.0, 0.0), True

        # ── Trigger settle phase when position is reached ─────────────────────
        if dist_to_goal <= goal_tolerance + 1e-6:  # epsilon handles floating-point equality
            self._settling = self.settle_steps
            return VelocityCommand(0.0, 0.0), False

        # ── Stage 1: brake ramp ───────────────────────────────────────────────
        if dist_to_goal < self.brake_distance:
            # ramp_factor: 1.0 at brake_distance, 0.0 at goal_tolerance
            ramp = (dist_to_goal - goal_tolerance) / (
                self.brake_distance - goal_tolerance + 1e-9
            )
            ramp = max(0.0, min(ramp, 1.0))

            braked_linear  = cmd.linear_x  * ramp
            # Angular gets extra braking so the robot straightens out
            # before decelerating fully — prevents spinning-while-stopping
            braked_angular = cmd.angular_z * ramp * self.angular_brake_gain

            return VelocityCommand(
                linear_x=braked_linear,
                angular_z=braked_angular,
            ), False

        # Outside brake zone — pass command through unchanged
        return cmd, False

    def reset(self) -> None:
        """Reset brake state for a new trajectory."""
        self._settling = 0
        self._settled  = False


# ══════════════════════════════════════════════════════════════════════════════
# Local minimum detector + escape
# ══════════════════════════════════════════════════════════════════════════════

class LocalMinimumEscape:
    """
    Detects when the robot is stuck in an APF local minimum and injects
    a random lateral escape impulse to break free.

    Detection: robot speed stays below `stuck_speed_threshold` for
    `stuck_steps` consecutive control steps.

    Escape: inject a random angular perturbation for `escape_steps` steps,
    then resume normal control.
    """

    def __init__(
        self,
        stuck_speed_threshold: float = 0.03,
        stuck_steps: int = 40,
        escape_steps: int = 15,
        escape_angular: float = 1.5,
    ):
        self._low_speed_count: int = 0
        self._escape_count: int = 0
        self.stuck_speed_threshold = stuck_speed_threshold
        self.stuck_steps = stuck_steps
        self.escape_steps = escape_steps
        self.escape_angular = escape_angular
        self._escape_direction: float = 1.0

    def update(self, current_speed: float) -> bool:
        """
        Update stuck detector with current robot speed.

        Args:
            current_speed: Absolute robot speed this step (m/s).

        Returns:
            True if the robot is currently in escape mode.
        """
        if self._escape_count > 0:
            self._escape_count -= 1
            return True  # still escaping

        if current_speed < self.stuck_speed_threshold:
            self._low_speed_count += 1
        else:
            self._low_speed_count = 0

        if self._low_speed_count >= self.stuck_steps:
            # Trigger escape: random left or right turn
            self._escape_direction = random.choice([-1.0, 1.0])
            self._escape_count = self.escape_steps
            self._low_speed_count = 0
            return True

        return False

    def escape_command(self) -> VelocityCommand:
        """Return a slow spin command to break out of local minimum."""
        return VelocityCommand(
            linear_x=0.05,
            angular_z=self._escape_direction * self.escape_angular,
        )

    def reset(self):
        self._low_speed_count = 0
        self._escape_count = 0


# ══════════════════════════════════════════════════════════════════════════════
# Pure Pursuit controller
# ══════════════════════════════════════════════════════════════════════════════

class PurePursuitController:
    """
    Pure Pursuit path tracking controller with APF correction layer.

    Layer 1 — Pure Pursuit: computes nominal linear + angular velocity
              toward the lookahead point on the trajectory.

    Layer 2 — APF correction: blends attractive (toward next waypoint) and
              repulsive (away from obstacles) forces into the angular command.

    Layer 3 — Local minimum escape: if the robot gets stuck, temporarily
              overrides the command with a random escape spin.

    Layer 4 — VelocityLimits.clamp(): hard clamp applied ONCE at the end
              of compute_command(). This is the single enforcement point —
              no velocity can ever exceed the configured limits.
    """

    def __init__(
        self,
        lookahead_distance: float = 0.3,
        max_linear_vel: float = 0.22,
        max_angular_vel: float = 2.84,
        goal_tolerance: float = 0.1,
        # APF parameters
        apf_k_att: float = 1.0,
        apf_k_rep: float = 0.5,
        apf_d_safe: float = 0.5,
        apf_influence: float = 0.4,
    ):
        """
        Args:
            lookahead_distance: Pure Pursuit lookahead distance (m).
            max_linear_vel:     Maximum forward speed — configures VelocityLimits (m/s).
            max_angular_vel:    Maximum yaw rate — configures VelocityLimits (rad/s).
            goal_tolerance:     Arrival radius at final goal (m).
            apf_k_att:          APF attractive gain.
            apf_k_rep:          APF repulsive gain.
            apf_d_safe:         APF obstacle influence radius (m).
            apf_influence:      APF blend weight into Pure Pursuit [0–1].
        """
        self.lookahead_distance = lookahead_distance
        self.goal_tolerance = goal_tolerance
        self._closest_idx: int = 0

        # Apply limits to the global VelocityLimits class
        VelocityLimits.configure(max_linear_vel, max_angular_vel)

        # Sub-modules
        self.apf = APFLayer(
            k_att=apf_k_att,
            k_rep=apf_k_rep,
            d_safe=apf_d_safe,
            influence_weight=apf_influence,
        )
        self.escape = LocalMinimumEscape()

        # GoalBrake: brake_distance must be >= goal_tolerance.
        # angular_brake_gain < 1.0 makes angular decelerate faster than
        # linear — robot stops rotating before it stops moving forward,
        # preventing the spinning-in-place bug at goal arrival.
        self.brake = GoalBrake(
            brake_distance=max(goal_tolerance * 4.0, 0.4),
            settle_steps=10,
            angular_brake_gain=0.5,
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _find_closest_point(
        self,
        state: RobotState,
        trajectory: List[TrajectoryPoint],
    ) -> int:
        """
        Find index of closest trajectory point, searching forward only
        (monotonic progress assumption — robot never backtracks).
        """
        min_dist = float("inf")
        closest_idx = self._closest_idx

        for i in range(self._closest_idx, len(trajectory)):
            dx = trajectory[i].x - state.x
            dy = trajectory[i].y - state.y
            dist = math.hypot(dx, dy)
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
            elif dist > min_dist + 0.5:
                break

        self._closest_idx = closest_idx
        return closest_idx

    def _find_lookahead_point(
        self,
        state: RobotState,
        trajectory: List[TrajectoryPoint],
        closest_idx: int,
    ) -> Optional[Tuple[float, float]]:
        """Walk forward from closest_idx to find the lookahead point."""
        for i in range(closest_idx, len(trajectory)):
            dx = trajectory[i].x - state.x
            dy = trajectory[i].y - state.y
            if math.hypot(dx, dy) >= self.lookahead_distance:
                return (trajectory[i].x, trajectory[i].y)
        last = trajectory[-1]
        return (last.x, last.y)

    # ── Main control interface ─────────────────────────────────────────────────

    def compute_command(
        self,
        state: RobotState,
        trajectory: List[TrajectoryPoint],
        obstacles: Optional[List[Tuple[float, float]]] = None,
    ) -> Tuple[VelocityCommand, bool]:
        """
        Compute velocity command — Pure Pursuit + APF + escape + hard clamp.

        Args:
            state:      Current robot pose (x, y, theta).
            trajectory: Time-stamped trajectory to follow.
            obstacles:  Optional list of (x, y) obstacle positions from LiDAR.
                        Pass None or [] if no obstacle data is available —
                        APF repulsion simply contributes zero in that case.

        Returns:
            (VelocityCommand, goal_reached)
            VelocityCommand is guaranteed to satisfy VelocityLimits.
            goal_reached is True once robot is within goal_tolerance of
            the final trajectory point.
        """
        if not trajectory:
            return VelocityLimits.clamp(VelocityCommand(0.0, 0.0)), True

        # ── Distance to goal (used by GoalBrake every cycle) ─────────────────
        goal = trajectory[-1]
        dist_to_goal = math.hypot(goal.x - state.x, goal.y - state.y)

        # ── Layer 1: Pure Pursuit ─────────────────────────────────────────────
        closest_idx = self._find_closest_point(state, trajectory)
        lookahead   = self._find_lookahead_point(state, trajectory, closest_idx)

        if lookahead is None:
            return VelocityLimits.clamp(VelocityCommand(0.0, 0.0)), True

        lx, ly = lookahead
        dx = lx - state.x
        dy = ly - state.y
        L  = math.hypot(dx, dy)

        if L < 1e-6:
            return VelocityLimits.clamp(VelocityCommand(0.0, 0.0)), False

        angle_to_lookahead = math.atan2(dy, dx)
        alpha = math.atan2(
            math.sin(angle_to_lookahead - state.theta),
            math.cos(angle_to_lookahead - state.theta),
        )

        curvature    = 2.0 * math.sin(alpha) / L
        desired_speed = trajectory[closest_idx].speed
        linear_vel   = desired_speed * math.cos(alpha)
        # Only enforce minimum crawl speed outside the brake zone.
        # Inside brake zone the GoalBrake layer will scale it down — 
        # imposing a floor here would prevent the robot from ever stopping.
        goal_dist_now = math.hypot(
            trajectory[-1].x - state.x, trajectory[-1].y - state.y
        )
        if goal_dist_now > self.brake.brake_distance:
            linear_vel = max(0.05, linear_vel)
        angular_vel  = linear_vel * curvature

        # ── Layer 2: APF correction ───────────────────────────────────────────
        next_wp      = (trajectory[closest_idx].x, trajectory[closest_idx].y)
        F_att        = self.apf.attractive_force(state, next_wp)
        F_rep        = self.apf.repulsive_force(state, obstacles or [])
        F_net        = F_att + F_rep
        apf_correction = self.apf.compute_angular_correction(state, F_net)
        angular_vel  += apf_correction

        raw_cmd = VelocityCommand(linear_x=linear_vel, angular_z=angular_vel)

        # ── Layer 3: Local minimum escape ─────────────────────────────────────
        if self.escape.update(current_speed=linear_vel):
            raw_cmd = self.escape.escape_command()

        # ── Layer 4: Goal brake ───────────────────────────────────────────────
        # Ramps linear AND angular smoothly to zero as robot approaches goal.
        # angular_brake_gain makes angular decelerate faster than linear so
        # the robot stops spinning BEFORE it stops moving — fixes the
        # rotating-in-place bug observed at goal arrival.
        # Returns goal_reached=True only after settle_steps explicit zeros
        # have been published, guaranteeing the motor controller received
        # an unambiguous stop signal.
        raw_cmd, goal_reached = self.brake.apply(
            raw_cmd, dist_to_goal, self.goal_tolerance
        )

        # ── Layer 5: Hard velocity clamp — SINGLE ENFORCEMENT POINT ──────────
        return VelocityLimits.clamp(raw_cmd), goal_reached

    def reset(self):
        """Reset all internal state (call when starting a new trajectory)."""
        self._closest_idx = 0
        self.escape.reset()
        self.brake.reset()
