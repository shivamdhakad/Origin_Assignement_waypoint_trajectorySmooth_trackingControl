"""
Standalone Simulation & Visualization
=======================================
Runs the full pipeline (smooth → trajectory → Pure Pursuit control) and
produces matplotlib plots — no ROS2 installation required.

Usage:
    python3 simulate.py

Outputs:
    - Plot 1: Waypoints + smooth path + tracked path
    - Plot 2: Speed profile over time
    - Plot 3: Cross-track error over time
"""

import math
import sys
import os

# Allow running from repo root without installing
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from path_smoother.path_smoother import smooth_path
from path_smoother.trajectory_generator import generate_trajectory
from path_smoother.controller import PurePursuitController, RobotState


# ── Configuration ─────────────────────────────────────────────────────────────

WAYPOINTS = [
    (0.0, 0.0),
    (1.0, 0.5),
    (2.0, 1.2),
    (3.0, 1.0),
    (4.0, 0.0),
]

MAX_VELOCITY  = 0.22   # m/s
ACCELERATION  = 0.10   # m/s²
LOOKAHEAD     = 0.30   # m
DT            = 0.05   # s (20 Hz)
MAX_STEPS     = 5000   # safety limit


def cross_track_error(state: RobotState, trajectory) -> float:
    """Compute minimum distance from robot to the trajectory."""
    min_d = float("inf")
    for tp in trajectory:
        d = math.hypot(tp.x - state.x, tp.y - state.y)
        if d < min_d:
            min_d = d
    return min_d


def run_simulation():
    print("=" * 55)
    print("  Path Smoother — Standalone Simulation")
    print("=" * 55)

    # ── Step 1: Smooth path ───────────────────────────────────────────────────
    print("\n[1/3] Smoothing path with cubic spline...")
    smooth = smooth_path(WAYPOINTS, num_samples=500)
    print(f"      {len(smooth)} smooth points generated.")

    # ── Step 2: Generate trajectory ───────────────────────────────────────────
    print("[2/3] Generating time-parameterized trajectory...")
    trajectory = generate_trajectory(smooth, max_velocity=MAX_VELOCITY, acceleration=ACCELERATION)
    print(f"      Duration: {trajectory[-1].t:.2f} s | Points: {len(trajectory)}")

    # ── Step 3: Simulate Pure Pursuit ─────────────────────────────────────────
    print("[3/3] Simulating Pure Pursuit controller...")
    controller = PurePursuitController(
        lookahead_distance=LOOKAHEAD,
        max_linear_vel=MAX_VELOCITY,
        max_angular_vel=2.84,
        goal_tolerance=0.1,
    )

    state = RobotState(x=WAYPOINTS[0][0], y=WAYPOINTS[0][1], theta=0.0)
    actual_x, actual_y = [state.x], [state.y]
    speeds, times, cte_list = [], [], []

    for step in range(MAX_STEPS):
        t = step * DT
        cmd, done = controller.compute_command(state, trajectory)

        speeds.append(math.hypot(
            cmd.linear_x * math.cos(state.theta),
            cmd.linear_x * math.sin(state.theta),
        ))
        times.append(t)
        cte_list.append(cross_track_error(state, trajectory[::10]))

        if done:
            print(f"      Goal reached at t={t:.2f} s (step {step})")
            break

        # Euler integration of unicycle model
        state = RobotState(
            x=state.x + cmd.linear_x * math.cos(state.theta) * DT,
            y=state.y + cmd.linear_x * math.sin(state.theta) * DT,
            theta=state.theta + cmd.angular_z * DT,
        )
        actual_x.append(state.x)
        actual_y.append(state.y)
    else:
        print("      WARNING: Max steps reached without reaching goal!")

    print(f"\n  Max CTE : {max(cte_list):.4f} m")
    print(f"  Mean CTE: {sum(cte_list)/len(cte_list):.4f} m")

    # ── Plots ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Path Smoother — Simulation Results", fontsize=14, fontweight="bold")

    # Plot 1: Path comparison
    ax = axes[0]
    wx = [p[0] for p in WAYPOINTS]
    wy = [p[1] for p in WAYPOINTS]
    sx = [p[0] for p in smooth]
    sy = [p[1] for p in smooth]

    ax.plot(sx, sy, "b-", linewidth=1.5, label="Smooth path (spline)", alpha=0.6)
    ax.plot(actual_x, actual_y, "g-", linewidth=2.0, label="Robot actual path")
    ax.scatter(wx, wy, c="red", s=80, zorder=5, label="Waypoints")
    ax.scatter([wx[0]], [wy[0]], c="lime", s=120, zorder=6, marker="*", label="Start")
    ax.scatter([wx[-1]], [wy[-1]], c="red", s=120, zorder=6, marker="*", label="Goal")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Path Tracking")
    ax.legend(fontsize=8)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # Plot 2: Speed profile
    ax = axes[1]
    traj_times = [tp.t for tp in trajectory]
    traj_speeds = [tp.speed for tp in trajectory]
    ax.plot(traj_times, traj_speeds, "b-", linewidth=1.5, label="Desired (trapezoidal)")
    ax.plot(times, speeds, "g--", linewidth=1.5, label="Actual robot speed", alpha=0.8)
    ax.axhline(y=MAX_VELOCITY, color="r", linestyle=":", alpha=0.5, label=f"Max ({MAX_VELOCITY} m/s)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Speed (m/s)")
    ax.set_title("Velocity Profile")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 3: Cross-track error
    ax = axes[2]
    ax.plot(times, cte_list, "r-", linewidth=1.5)
    ax.fill_between(times, cte_list, alpha=0.2, color="red")
    ax.axhline(y=0.1, color="orange", linestyle="--", label="Goal tolerance (0.1m)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("CTE (m)")
    ax.set_title("Cross-Track Error")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("simulation_results.png", dpi=150, bbox_inches="tight")
    print("\n  Plot saved to: simulation_results.png")
    plt.show()


if __name__ == "__main__":
    run_simulation()
