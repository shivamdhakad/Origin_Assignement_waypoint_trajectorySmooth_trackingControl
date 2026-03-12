# Path Smoother & Trajectory Tracker — ROS2

A complete implementation of **path smoothing**, **trajectory generation**, and **trajectory tracking** for a differential drive robot (Turtlebot3) in ROS2.
Simulation: https://drive.google.com/file/d/1J1ojwpW5FAjtNpwLgEBMjq5VEbBBPjYb/view?usp=sharing
<iframe src="[https://www.googledrive.com...](https://drive.google.com/file/d/1J1ojwpW5FAjtNpwLgEBMjq5VEbBBPjYb/view?usp=sharing)" width="640" height="480"></iframe>

---


## System Architecture

```
┌─────────────────────┐    /smooth_path     ┌──────────────────────────┐
│  path_smoother_node │ ─────────────────►  │ trajectory_generator_node│
│                     │                     │                          │
│  • Reads YAML params│    /waypoints_viz   │  • Trapezoidal velocity   │
│  • Cubic spline fit │ ──► (RViz markers)  │  • Time-stamps trajectory │
└─────────────────────┘                     └──────────────────────────┘
                                                        │
                                             /trajectory_data (JSON)
                                                        │
                                                        ▼
                                            ┌───────────────────────┐
                                            │ trajectory_tracker_node│
                                            │                       │
                    /odom ──────────────►   │  • Pure Pursuit ctrl  │
                                            │  • Publishes /cmd_vel │
                    /cmd_vel ◄──────────    │  • Records actual path│
                                            └───────────────────────┘
```

---

## Algorithms

### 1. Path Smoothing — Cubic Spline
- Waypoints are parameterized by **cumulative chord length** (arc-length approximation)
- Independent `CubicSpline` (scipy) fitted to `x(t)` and `y(t)`
- Sampled at `num_samples` uniform points → smooth, C² continuous path
- Arc-length parameterization avoids the "speed distortion" of index-based parameterization

### 2. Trajectory Generation — Trapezoidal Velocity Profile
- Computes cumulative arc-length of the smooth path
- Applies a **trapezoidal speed profile**: accelerate → cruise → decelerate
- If path is too short to reach `max_velocity`, a **triangular profile** is used automatically
- Integrates `dt = ds / v` to assign timestamps → `[(x, y, t), ...]`

### 3. Trajectory Tracking — Pure Pursuit
- At each control step, finds the **closest point** on the trajectory (monotonic forward search)
- Looks ahead by `lookahead_distance` to find the **target point**
- Computes steering curvature: `κ = 2·sin(α) / L`
- Converts to `(linear_x, angular_z)` Twist commands for Turtlebot3

---

## Setup & Installation

### Prerequisites
- ROS2 Humble (or Foxy/Iron)
- Turtlebot3 packages
- Python packages: `scipy`, `numpy`

```bash
pip install scipy numpy matplotlib pytest
```

### Build

```bash
cd ~/ros2_ws
colcon build --packages-select path_smoother
source install/setup.bash
```

### Configure Waypoints

Edit `config/params.yaml`:

```yaml
path_smoother_node:
  ros__parameters:
    waypoints: [-2.0, -0.5,   # home pose, x0,y0
                0.0, -0.5,
                1.0, -0.5,
                1.5, -0.5,
                1.5, 0.5]
```

Waypoints are stored as a **flat array** `[x0, y0, x1, y1, ...]` — this is the most reliable format for the ROS2 parameter server (which lacks native support for 2D arrays).

---

## Running

### With Turtlebot3 Simulation (Gazebo)

```bash
# Terminal 1: Launch Turtlebot3 world
export TURTLEBOT3_MODEL=burger
ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py

# Terminal 2: Launch path smoother system
ros2 launch path_smoother path_smoother.launch.py use_sim_time:=true

# Terminal 3: Open RViz
ros2 run rviz2 rviz2
```

**RViz topics to add:**
| Display      | Topic               | Description              |
|-------------|---------------------|--------------------------|
| Path        | `/smooth_path`      | Blue — cubic spline path |
| Path        | `/trajectory`       | Cyan — timed trajectory  |
| Path        | `/tracking_path`    | Green — robot actual path|
| MarkerArray | `/trajectory_markers` | Velocity arrows         |
| Marker      | `/waypoints_viz`    | Orange — raw waypoints   |

### Standalone (No ROS2 required)

```bash
cd ros2_ws/src/path_smoother
python3 simulate.py
```

Produces 3 plots: path tracking, velocity profile, cross-track error.

---

## Testing

```bash
cd ros2_ws/src/path_smoother
pytest test/ -v
```

### Test Coverage
| Category      | Tests                                              |
|--------------|----------------------------------------------------|
| Path Smoother | output length, format, start/end proximity, edge cases |
| Trajectory    | timestamps, speed bounds, trapezoidal shape, tuples |
| Controller    | goal detection, velocity bounds, angular correction |
| Integration   | full pipeline, simulation convergence              |

---

## Extending to a Real Robot

1. **Sensor fusion**: Replace `/odom` with filtered localization (e.g., `robot_localization` EKF fusing wheel odometry + IMU)
2. **TF frames**: Use `map` → `odom` → `base_link` transform chain; run SLAM (e.g., Nav2 + SLAM Toolbox) for global localization
3. **Lookahead tuning**: On real hardware, tune `lookahead_distance` based on actual robot speed and latency (~0.3–0.5 m typical)
4. **Safety**: Add velocity ramp-down on cmd_vel watchdog timeout; add E-stop subscriber
5. **Latency**: Reduce control loop to 10 Hz if hardware odometry is slow; or use predictive state estimation

---

## Obstacle Avoidance (Extra Credit)

The system can be extended with **Dynamic Window Approach (DWA)** or **VFH (Vector Field Histogram)**:

1. Subscribe to `/scan` (LiDAR) in the tracker node
2. At each control step, check if the lookahead point is collision-free
3. If blocked: sample alternative velocity commands within the robot's dynamic window
4. Score each sample by: `score = α·heading + β·clearance + γ·velocity`
5. Publish the highest-scoring safe command instead of pure pursuit output

Alternatively, integrate with **Nav2's local planner** which already implements DWA and Regulated Pure Pursuit with obstacle avoidance.

---

## AI Tools Used

- **Claude** — architecture design, code generation, documentation
- Used for: module structure, Pure Pursuit implementation, trapezoidal profile math, test case design

---

## Package Structure

```
path_smoother/
├── path_smoother/
│   ├── __init__.py
│   ├── path_smoother.py           # Cubic spline algorithm
│   ├── trajectory_generator.py    # Trapezoidal profile + trajectory
│   ├── controller.py              # Pure Pursuit controller
│   ├── path_smoother_node.py      # ROS2 Node 1
│   ├── trajectory_generator_node.py  # ROS2 Node 2
│   └── trajectory_tracker_node.py    # ROS2 Node 3
├── config/
│   └── params.yaml                # All parameters
├── launch/
│   └── path_smoother.launch.py    # Single launch file
├── test/
│   └── test_all.py                # 20+ unit + integration tests
├── simulate.py                    # Standalone simulation (no ROS2)
├── setup.py
├── package.xml
└── README.md
└── waypoint_editor.py		   # file to update waypoint yaml file

```

# Origin_Assignement_waypoint_trajectorySmooth_trackingControl
