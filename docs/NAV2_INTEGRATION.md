# MPPI Nav2 Integration Guide

## Overview

This package provides a **Python FollowPath Action Server** that integrates MPPI controllers
with the Nav2 navigation stack. Since Nav2's controller_server requires C++ pluginlib plugins,
we replace it with a standalone action server that speaks the same `FollowPath` action interface.

```
Nav2 BT Navigator
  │  FollowPath action goal (nav_msgs/Path)
  ▼
┌──────────────────────────────────────────────────────────┐
│  MPPIFollowPathServer (Node + ActionServer)              │
│                                                          │
│  Subscribers:              Publishers:                   │
│    /odom (Odometry)          /cmd_vel (Twist)           │
│    /local_costmap/costmap    /mppi/local_plan (Path)    │
│    /scan (optional)          /mppi/visualization        │
│                                                          │
│  ┌────────────┐ ┌──────────────┐ ┌─────────────────┐   │
│  │PathWindower│ │CostmapConvert│ │ GoalChecker     │   │
│  │ closest pt │ │ grid→circles │ │ ProgressChecker │   │
│  │ + N+1 ahead│ │ → CBF/Shield │ │ stuck detection │   │
│  └─────┬──────┘ └──────┬───────┘ └────────┬────────┘   │
│        │               │                  │              │
│        ▼               ▼                  ▼              │
│  ┌──────────────────────────────────────────────────┐   │
│  │  MPPI Controller (12+ variants)                  │   │
│  │  compute_control(state, ref_traj) → (u, info)    │   │
│  └──────────────────────────────────────────────────┘   │
│                                                          │
│  TF2: map → odom → base_link                            │
└──────────────────────────────────────────────────────────┘
```

## Prerequisites

- ROS2 Humble or later
- Nav2 stack (`nav2_bringup`, `nav2_msgs`)
- Python 3.10+
- NumPy, SciPy

```bash
# Install Nav2
sudo apt install ros-humble-navigation2 ros-humble-nav2-bringup ros-humble-nav2-msgs
```

## Build

```bash
cd ~/ros2_ws/src
# Clone or symlink learning_mppi
ln -s /path/to/learning_mppi .

cd ~/ros2_ws
colcon build --packages-select learning_mppi
source install/setup.bash
```

## Quick Start

### Standalone Simulation

Run the MPPI FollowPath server with the built-in robot simulator:

```bash
ros2 launch learning_mppi mppi_nav2_sim.launch.py
```

Options:
```bash
# Use different controller
ros2 launch learning_mppi mppi_nav2_sim.launch.py controller_type:=adaptive_shield

# Without RVIZ
ros2 launch learning_mppi mppi_nav2_sim.launch.py use_rviz:=false

# Dynamic model
ros2 launch learning_mppi mppi_nav2_sim.launch.py model_type:=dynamic
```

### With Nav2 Stack

1. Launch Nav2 normally but **without** the default controller_server:

```bash
# Standard Nav2 bringup (map required)
ros2 launch nav2_bringup bringup_launch.py map:=/path/to/map.yaml
```

2. Launch the MPPI FollowPath server:

```bash
ros2 launch learning_mppi mppi_nav2.launch.py
```

The action server registers at `/follow_path`, which Nav2's BT Navigator automatically discovers.

### With TurtleBot3

```bash
# Terminal 1: TurtleBot3 simulation
export TURTLEBOT3_MODEL=burger
ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py

# Terminal 2: Nav2 + MPPI
ros2 launch nav2_bringup bringup_launch.py \
    map:=/path/to/map.yaml \
    use_sim_time:=true

# Terminal 3: MPPI FollowPath server
ros2 launch learning_mppi mppi_nav2.launch.py controller_type:=shield

# Terminal 4: RVIZ (set 2D Nav Goal)
ros2 launch nav2_bringup rviz_launch.py
```

## Configuration

### Config File: `configs/mppi_nav2.yaml`

```yaml
mppi_follow_path_server:
  ros__parameters:
    # Node
    controller_frequency: 20.0     # Control loop rate [Hz]
    odom_frame: odom               # Odometry TF frame
    robot_frame: base_link         # Robot TF frame

    # Model
    model_type: kinematic          # kinematic | dynamic
    v_max: 0.5                     # Max linear velocity [m/s]
    omega_max: 1.9                 # Max angular velocity [rad/s]

    # Controller
    controller_type: shield        # See "Available Controllers" below
    N: 30                          # Prediction horizon (timesteps)
    dt: 0.05                       # Timestep [s]
    K: 1024                        # Number of rollout samples
    lambda_: 1.0                   # Temperature parameter
    sigma: [0.5, 0.5]             # Noise std [v, omega]

    # Cost weights
    Q: [10.0, 10.0, 1.0]          # State tracking [x, y, theta]
    R: [0.1, 0.1]                  # Control effort [v, omega]
    Qf: [20.0, 20.0, 2.0]         # Terminal cost

    # CBF/Shield (for cbf/shield/adaptive_shield types)
    cbf_alpha: 0.3
    cbf_weight: 1000.0
    cbf_safety_margin: 0.1
    shield_enabled: true

    # Goal tolerance
    xy_goal_tolerance: 0.25        # XY distance [m]
    yaw_goal_tolerance: 0.25       # Yaw angle [rad]

    # Costmap
    costmap_topic: /local_costmap/costmap_raw
    robot_radius: 0.22             # Robot footprint radius [m]

    # LaserScan (alternative obstacle detection)
    scan_enabled: false
```

### Available Controllers

| Type | Description | Safety |
|------|-------------|--------|
| `vanilla` | Standard MPPI | No |
| `tube` | Robust tube constraints | Partial |
| `log` | Log-space softmax | No |
| `tsallis` | q-exponential weights | No |
| `risk_aware` | CVaR risk measure | Partial |
| `smooth` | Input-rate smoothing | No |
| `svmpc` | Stein Variational | No |
| `spline` | B-spline trajectories | No |
| `svg` | Guide particle SVGD | No |
| `cbf` | CBF cost + optional QP filter | Yes |
| `shield` | Per-step CBF enforcement | Yes |
| `adaptive_shield` | Distance/velocity adaptive shield | Yes (Best) |

**Recommended for navigation**: `shield` or `adaptive_shield`

## Architecture Details

### Pipeline Components

#### PathWindower
Extracts a local (N+1, state_dim) reference window from the global path:
- Warm-start closest point search
- Lookahead distance to avoid tracking past points
- Arc-length interpolation for uniform spacing
- Angle wrapping via sin/cos decomposition
- End-of-path padding

#### CostmapConverter
Converts nav2 OccupancyGrid to circle obstacle list:
- Lethal threshold filtering (>=253)
- Grid-based connected component clustering (O(N))
- Enclosing circle fitting
- Robot radius + safety margin inflation
- Range-based filtering

#### GoalChecker
Nav2-compatible goal reached detection:
- XY distance tolerance
- Yaw angle tolerance
- Stateful: once reached, stays reached until reset

#### ProgressChecker
Stuck detection:
- Monitors minimum movement within time window
- Triggers FAILED_TO_MAKE_PROGRESS if stuck

### TF2 Frame Strategy

```
map (global planning frame)
 └── odom (local frame, drift-free short-term)
      └── base_link (robot body frame)
```

1. Path arrives in `map` frame from Nav2 planner
2. TF2 transforms path to `odom` frame
3. PathWindower extracts local reference in `odom` frame
4. MPPI computes control in `odom` frame
5. cmd_vel published (frame-independent)

### Action Interface

```
nav2_msgs/action/FollowPath:
  Goal:
    path (nav_msgs/Path)     # Global path to follow
    controller_id (string)   # Controller plugin ID (unused)
  Feedback:
    distance_to_goal (float) # XY distance to final pose [m]
    speed (float)            # Current linear speed [m/s]
  Result:
    error_code (int16):
      0   = NONE (success)
      100 = TF_ERROR
      101 = INVALID_PATH
      102 = FAILED_TO_MAKE_PROGRESS
      103 = NO_VALID_CONTROL
```

## Topics

### Subscriptions

| Topic | Type | Description |
|-------|------|-------------|
| `/odom` | `nav_msgs/Odometry` | Robot odometry |
| `/local_costmap/costmap_raw` | `nav_msgs/OccupancyGrid` | Nav2 local costmap |
| `/scan` | `sensor_msgs/LaserScan` | LiDAR (if scan_enabled) |

### Publications

| Topic | Type | Description |
|-------|------|-------------|
| `/cmd_vel` | `geometry_msgs/Twist` | Velocity command |
| `/mppi/local_plan` | `nav_msgs/Path` | MPPI local reference (RVIZ) |

## Troubleshooting

### "nav2_msgs not found"

Install nav2_msgs:
```bash
sudo apt install ros-humble-nav2-msgs
```

The server will still start but without the action server. Use it for testing with direct topic interfaces.

### TF errors

Ensure TF tree is complete: `map → odom → base_link`

```bash
# Check TF tree
ros2 run tf2_tools view_frames
```

### Robot not moving

1. Check odometry is being received:
```bash
ros2 topic echo /odom --once
```

2. Check the action server received a goal:
```bash
ros2 action list
ros2 action info /follow_path
```

3. Send a test goal:
```bash
ros2 action send_goal /follow_path nav2_msgs/action/FollowPath \
    "{path: {poses: [{pose: {position: {x: 1.0, y: 0.0}}}]}}"
```

### Stuck detection triggering too early

Increase `time_allowance` in the config or adjust `required_movement`:
```yaml
# In mppi_nav2.yaml, these are internal defaults:
# required_movement: 0.5  # meters in time_allowance seconds
# time_allowance: 10.0    # seconds
```

### Performance tuning

- Reduce `K` (samples) for faster computation: K=512 for embedded
- Increase `controller_frequency` for smoother control (requires faster MPPI)
- Use `adaptive_shield` for best safety/performance trade-off

## Testing

```bash
# Unit tests (no ROS2 required)
pytest tests/test_path_windower.py tests/test_costmap_converter.py \
       tests/test_goal_checker.py tests/test_progress_checker.py -v

# Integration test (no ROS2 required)
pytest tests/test_follow_path_integration.py -v

# All tests
pytest tests/ -o "addopts=" -q
```
