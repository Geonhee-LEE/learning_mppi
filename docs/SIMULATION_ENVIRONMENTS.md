# Simulation Environments Guide

12 diverse simulation scenarios showcasing the full capabilities of MPPI variants, safety controllers, and robot models.

## Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                    Simulation Environments (12)                       │
├────┬───────────────────────────┬─────────────────────┬──────────────┤
│ ID │ Scenario                  │ Controllers          │ Time (batch) │
├────┼───────────────────────────┼─────────────────────┼──────────────┤
│ S1 │ Static Obstacle Field     │ Vanilla/CBF/Shield   │    ~10s      │
│ S2 │ Dynamic Bouncing          │ CBF/C3BF/Shield      │     ~6s      │
│ S3 │ Chasing Evader            │ Shield/DPCBF/CBF     │     ~5s      │
│ S4 │ Multi-Robot Coordination  │ 4-way CBF comparison │    ~11s      │
│ S5 │ Waypoint Navigation       │ Vanilla/CBF          │     ~6s      │
│ S6 │ Drifting Disturbance      │ Vanilla/Tube/Risk    │     ~6s      │
│ S7 │ Parking Precision         │ 3 MPPI configs       │     ~7s      │
│ S8 │ Racing MPCC               │ MPCC/Tracking        │    ~44s      │
│ S9 │ Narrow Corridor           │ CBF/Shield/Aggr.     │    ~89s      │
│ S10│ Mixed Challenge           │ Shield-MPPI          │    ~34s      │
│ S12│ Warehouse (levels 0-5)    │ Vanilla/Shield       │    ~10s      │
│ S13│ Racing Track (3 types)    │ DiffDrive             │     ~8s      │
├────┴───────────────────────────┴─────────────────────┴──────────────┤
│ Total: 12 scenarios, ~236s batch execution                           │
└──────────────────────────────────────────────────────────────────────┘
```

## Quick Start

```bash
cd /path/to/learning_mppi

# Run all 10 scenarios (batch, no plots)
PYTHONPATH=. python examples/simulation_environments/run_all.py

# Run a single scenario
PYTHONPATH=. python examples/simulation_environments/scenarios/static_obstacle_field.py

# Live animation mode
PYTHONPATH=. python examples/simulation_environments/scenarios/dynamic_bouncing.py --live

# Batch mode without plot window
PYTHONPATH=. python examples/simulation_environments/scenarios/narrow_corridor.py --no-plot
```

---

## Scenarios

### S1: Static Obstacle Field

Navigate through random/slalom/dense obstacle fields.

| Controller | Role |
|-----------|------|
| Vanilla MPPI | No obstacle awareness (baseline) |
| CBF-MPPI | Cost-based barrier penalty |
| Shield-MPPI | Per-step shielded rollout |

```bash
# Random layout (default)
PYTHONPATH=. python examples/simulation_environments/scenarios/static_obstacle_field.py

# Slalom gates
PYTHONPATH=. python examples/simulation_environments/scenarios/static_obstacle_field.py --layout slalom

# Dense field
PYTHONPATH=. python examples/simulation_environments/scenarios/static_obstacle_field.py --layout dense

# Live animation
PYTHONPATH=. python examples/simulation_environments/scenarios/static_obstacle_field.py --live

# Batch (no plot window)
PYTHONPATH=. python examples/simulation_environments/scenarios/static_obstacle_field.py --no-plot
```

**Key features**: `ObstacleCost` vs `ControlBarrierCost` vs `ShieldMPPIController`

---

### S2: Dynamic Bouncing Obstacles

Track a circle trajectory while 5 obstacles bounce around the arena.

| Controller | Role |
|-----------|------|
| CBF-MPPI | Standard distance-based barrier |
| C3BF-MPPI | Velocity-aware collision cone barrier |
| Shield-MPPI | Per-step shielded rollout |

```bash
PYTHONPATH=. python examples/simulation_environments/scenarios/dynamic_bouncing.py
PYTHONPATH=. python examples/simulation_environments/scenarios/dynamic_bouncing.py --live
PYTHONPATH=. python examples/simulation_environments/scenarios/dynamic_bouncing.py --n-obstacles 8
```

**Key features**: `BouncingMotion`, 5-tuple obstacles `(x,y,r,vx,vy)` for C3BF, real-time `update_obstacles()`

---

### S3: Chasing Evader

Navigate to waypoints while a predator actively chases the robot.

| Controller | Role |
|-----------|------|
| Shield-MPPI | Shielded rollout (strongest safety) |
| DPCBF-MPPI | Directional adaptive boundary |
| CBF-MPPI | Standard distance barrier (baseline) |

```bash
PYTHONPATH=. python examples/simulation_environments/scenarios/chasing_evading.py
PYTHONPATH=. python examples/simulation_environments/scenarios/chasing_evading.py --live
```

**Key features**: `ChasingMotion.set_target()` updated every step, `DynamicParabolicCBFCost` with LoS coordinates

---

### S4: Multi-Robot Coordination

4 robots swap diagonal positions without inter-robot collisions.

| Configuration | Cost Layer | Filter Layer |
|--------------|-----------|-------------|
| No CBF | off | off |
| Cost Only | on | off |
| Filter Only | off | on |
| Cost+Filter | on | on |

```bash
PYTHONPATH=. python examples/simulation_environments/scenarios/multi_robot_coordination.py
PYTHONPATH=. python examples/simulation_environments/scenarios/multi_robot_coordination.py --no-plot
```

**Key features**: `MultiRobotCoordinator`, `RobotAgent`, pairwise CBF constraints, sequential planning

---

### S5: Waypoint Navigation

Navigate 8 waypoints with dwell times through a static obstacle field.

| Controller | Role |
|-----------|------|
| Vanilla MPPI | No obstacle awareness |
| CBF-MPPI | Barrier-based obstacle avoidance |

```bash
PYTHONPATH=. python examples/simulation_environments/scenarios/waypoint_navigation.py
PYTHONPATH=. python examples/simulation_environments/scenarios/waypoint_navigation.py --no-plot
```

**Key features**: `WaypointStateMachine` (NAVIGATING -> DWELLING -> COMPLETED), sequential goal tracking

---

### S6: Drifting Disturbance

Track a figure-8 trajectory with large process noise (simulating wind/slip).

| Controller | Role |
|-----------|------|
| Vanilla MPPI | No disturbance compensation |
| Tube-MPPI | Ancillary controller for robustness |
| Risk-Aware MPPI | CVaR-based conservative control |

```bash
PYTHONPATH=. python examples/simulation_environments/scenarios/drifting_disturbance.py
PYTHONPATH=. python examples/simulation_environments/scenarios/drifting_disturbance.py --noise 0.5
PYTHONPATH=. python examples/simulation_environments/scenarios/drifting_disturbance.py --live
```

**Key features**: `Simulator(process_noise_std=...)`, identical noise sequence across controllers via `np.random.seed()`

---

### S7: Parking Precision

Parallel park an Ackermann vehicle into a tight slot.

| Controller | Configuration |
|-----------|-------------|
| Vanilla | Standard params (K=1024) |
| MPPI-2K | More samples (K=2048), lower temperature |
| MPPI-Fine | Narrow noise (sigma=0.15), high terminal cost |

```bash
PYTHONPATH=. python examples/simulation_environments/scenarios/parking_precision.py
PYTHONPATH=. python examples/simulation_environments/scenarios/parking_precision.py --no-plot
```

**Key features**: `AckermannKinematic` (bicycle model, 4D state), `SuperellipsoidCost` for rectangular parking walls

---

### S8: Racing MPCC

Maximize progress along an elliptical race track.

| Controller | Cost Function |
|-----------|-------------|
| MPCC | Contouring + lag + progress decomposition |
| Tracking | Standard state tracking cost |

```bash
PYTHONPATH=. python examples/simulation_environments/scenarios/racing_mpcc.py
PYTHONPATH=. python examples/simulation_environments/scenarios/racing_mpcc.py --no-plot
```

**Key features**: `PathParameterization` (arc-length), `MPCCCost` (Q_c, Q_l, Q_theta, Q_heading), corridor track walls

**Expected results**: MPCC achieves ~0.004m mean contouring error vs Tracking's ~0.06m.

---

### S9: Narrow Corridor

Navigate through tight L-shaped passages with 90-degree turns.

| Controller | Configuration |
|-----------|-------------|
| CBF-MPPI | Standard alpha=0.2 |
| Shield-MPPI | Shielded rollout |
| CBF-Aggressive | Higher alpha=0.5, weight=2000 |

```bash
PYTHONPATH=. python examples/simulation_environments/scenarios/narrow_corridor.py
PYTHONPATH=. python examples/simulation_environments/scenarios/narrow_corridor.py --live
```

**Key features**: `generate_corridor()` + `generate_funnel()`, low speed (0.5 m/s), high heading weight

---

### S10: Mixed Challenge

Combined: static obstacles -> dynamic obstacles -> narrow corridor -> goal.

| Controller | Role |
|-----------|------|
| Shield-MPPI | Single controller handling all zones |

```bash
PYTHONPATH=. python examples/simulation_environments/scenarios/mixed_challenge.py
PYTHONPATH=. python examples/simulation_environments/scenarios/mixed_challenge.py --no-plot
```

**Key features**: Zone-based obstacle switching, `WaypointStateMachine`, `CrossingMotion` + `BouncingMotion` + corridor walls

---

### S12: Warehouse

Navigate through warehouse shelves with increasing difficulty (levels 0-5).

| Level | Features |
|-------|----------|
| 0 | Open area (no obstacles) |
| 1-2 | Static shelves |
| 3-4 | Shelves + bouncing dynamic obstacles |
| 5 | Dense shelves + dynamic obstacles |

```bash
PYTHONPATH=. python examples/simulation_environments/scenarios/warehouse.py
PYTHONPATH=. python examples/simulation_environments/scenarios/warehouse.py --level 3 --live
```

**Key features**: `WarehouseEnv(level=0-5)`, progressive difficulty, wall-based shelves, bouncing obstacles

---

### S13: Racing Track

Race along tracks with optional model mismatch (friction changes).

| Track | Description |
|-------|-------------|
| `straight` | Simple straight corridor |
| `oval` | Elliptical loop |
| `l_shape` | L-shaped track with sharp turn |

```bash
PYTHONPATH=. python examples/simulation_environments/scenarios/racing_track.py
PYTHONPATH=. python examples/simulation_environments/scenarios/racing_track.py --track oval
PYTHONPATH=. python examples/simulation_environments/scenarios/racing_track.py --track l_shape --friction 0.5
```

**Key features**: `RacingTrackEnv(track, friction_scale)`, arc-length reference, `FrictionMismatchModel`

---

## Batch Runner

Run all 10 scenarios and generate a summary table:

```bash
# All scenarios
PYTHONPATH=. python examples/simulation_environments/run_all.py

# Specific scenarios only
PYTHONPATH=. python examples/simulation_environments/run_all.py --scenarios s1 s2 s6

# With custom seed
PYTHONPATH=. python examples/simulation_environments/run_all.py --seed 123
```

Example output:
```
==============================================================================
                                   Summary
==============================================================================
   ID |                            Scenario |   Status |     Time
------------------------------------------------------------------------------
   s1 |           S1: Static Obstacle Field |     PASS |     9.9s
   s2 |                S2: Dynamic Bouncing |     PASS |     6.3s
   s3 |                  S3: Chasing Evader |     PASS |     5.1s
   s4 |        S4: Multi-Robot Coordination |     PASS |    10.6s
   s5 |             S5: Waypoint Navigation |     PASS |     5.9s
   s6 |            S6: Drifting Disturbance |     PASS |     5.6s
   s7 |               S7: Parking Precision |     PASS |     7.0s
   s8 |                     S8: Racing MPCC |     PASS |    44.4s
   s9 |                 S9: Narrow Corridor |     PASS |    89.0s
  s10 |                S10: Mixed Challenge |     PASS |    34.1s
------------------------------------------------------------------------------
Total |                                     | 10P/0F  |   218.0s
==============================================================================
```

---

## Common Arguments

All scenarios share these CLI arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `--live` | off | Real-time FuncAnimation (4 panels) |
| `--no-plot` | off | Batch mode, suppress plot window |
| `--seed N` | 42 | Random seed for reproducibility |

Scenario-specific arguments:

| Scenario | Argument | Default | Description |
|----------|----------|---------|-------------|
| S1 | `--layout` | `random` | `random`, `slalom`, or `dense` |
| S2 | `--n-obstacles` | 5 | Number of bouncing obstacles |
| S6 | `--noise` | 0.3 | Process noise level |
| S12 | `--level` | 0 | Warehouse difficulty (0-5) |
| S13 | `--track` | `straight` | Track type (straight/oval/l_shape) |
| S13 | `--friction` | 1.0 | Friction mismatch scale |

---

## Architecture

### File Structure

```
examples/simulation_environments/
├── common/
│   ├── __init__.py              # Package exports
│   ├── environment.py           # SimulationEnvironment ABC, obstacle types
│   ├── obstacle_field.py        # 4 static obstacle generators
│   ├── dynamic_obstacle.py      # DynamicObstacle + 5 motion strategies
│   ├── waypoint_manager.py      # WaypointStateMachine
│   ├── env_metrics.py           # Extended metrics (collision, clearance)
│   └── env_visualizer.py        # EnvVisualizer (live/batch/GIF)
├── scenarios/
│   ├── __init__.py
│   ├── static_obstacle_field.py # S1
│   ├── dynamic_bouncing.py      # S2
│   ├── chasing_evading.py       # S3
│   ├── multi_robot_coordination.py # S4
│   ├── waypoint_navigation.py   # S5
│   ├── drifting_disturbance.py  # S6
│   ├── parking_precision.py     # S7
│   ├── racing_mpcc.py           # S8
│   ├── narrow_corridor.py       # S9
│   ├── mixed_challenge.py       # S10
│   ├── warehouse.py           # S12
│   └── racing_track.py        # S13
└── run_all.py                   # Batch runner + summary
```

### Common Infrastructure

#### `SimulationEnvironment` (ABC)

Every scenario implements:

```python
class MyScenario(SimulationEnvironment):
    def get_initial_state(self) -> np.ndarray           # (nx,)
    def get_obstacles(self, t) -> List[(x, y, r)]       # obstacles at time t
    def get_reference_fn(self) -> Callable               # t -> (N+1, nx)
    def get_controller_configs(self) -> List[ControllerConfig]
    def draw_environment(self, ax, t)                    # matplotlib decorations
    def on_step(self, t, states, controls, infos)        # dynamic updates
```

#### Obstacle Generators

| Function | Description |
|---------|-------------|
| `generate_random_field()` | Poisson-disk random placement with exclusion zones |
| `generate_corridor()` | Walls along a polyline path |
| `generate_slalom()` | Alternating gate pillars |
| `generate_funnel()` | Narrowing passage |

#### Dynamic Obstacle Motions

| Motion | Behavior |
|--------|----------|
| `BouncingMotion` | Bounces off arena walls |
| `ChasingMotion` | Proportional pursuit of robot |
| `EvadingMotion` | Blocks path while avoiding robot |
| `CrossingMotion` | Periodic sinusoidal crossing |
| `CircularMotion` | Orbit around center |

#### `WaypointStateMachine`

State transition: `NAVIGATING -> DWELLING -> NAVIGATING -> ... -> COMPLETED`

#### `EnvVisualizer`

| Method | Description |
|--------|-------------|
| `run_and_animate()` | Live FuncAnimation (4-panel) |
| `run_and_plot()` | Static 6-panel comparison |
| `export_gif()` | GIF via PillowWriter |

#### Extended Metrics (`compute_env_metrics`)

| Metric | Description |
|--------|-------------|
| `collision_count` | Timesteps with obstacle penetration |
| `min_clearance` | Minimum distance to any obstacle (m) |
| `safety_rate` | Fraction of timesteps without collision |
| `path_length` | Actual trajectory length (m) |
| `path_efficiency` | Straight-line / actual distance |
| `completion_time` | Time to reach goal (s) |

---

## SimulationHarness (통합 시뮬레이션 프레임워크)

`mppi_controller.simulation.harness.SimulationHarness`를 사용하면 시뮬레이션 루프 중복 없이 다중 컨트롤러를 비교할 수 있습니다.

```python
from mppi_controller.simulation.harness import SimulationHarness

harness = SimulationHarness(dt=0.05, headless=True)
harness.add_controller("Vanilla", vanilla_ctrl, model, "blue")
harness.add_controller("Shield", shield_ctrl, model, "red")

results = harness.run(reference_fn, x0, duration=15.0)
harness.print_comparison(results)
harness.plot(results, save_path="plots/comparison.png")
```

### 렌더링 서브시스템

| 모듈 | 기능 |
|------|------|
| `rendering/headless.py` | NullAxes/NullFigure (GUI 없는 환경) |
| `rendering/robot_renderer.py` | Circle/Car/Rectangle body 패치 |
| `rendering/animation_saver.py` | MP4/GIF 내보내기 |
| `rendering/safety_overlay.py` | CBF contour, collision cone, DPCBF, effective radius |

---

## Existing Code Reused (No Modifications)

| Component | File | Usage |
|-----------|------|-------|
| Simulator | `simulation/simulator.py` | Core sim loop |
| Metrics | `simulation/metrics.py` | `compute_metrics()`, `print_metrics()` |
| Trajectory | `utils/trajectory.py` | `create_trajectory_function()`, `generate_reference_trajectory()` |
| Controllers | `controllers/mppi/*.py` | All 9 MPPI variants + safety controllers |
| Models | `models/kinematic/*.py` | DiffDrive, Ackermann, Swerve |
| Safety | `cbf_*.py`, `shield_mppi.py`, `gatekeeper.py` | CBF/Shield/Gatekeeper |
| Multi-Robot | `multi_robot_cbf.py` | Coordinator, RobotAgent |
| MPCC | `mpcc_cost.py` | PathParameterization, MPCCCost |
| Superellipsoid | `superellipsoid_cost.py` | Non-circular obstacles |

**Zero changes to existing source code.**

---

## Adding a New Scenario

1. Create `scenarios/my_scenario.py`
2. Inherit from `SimulationEnvironment`
3. Implement the 4 required methods
4. Add entry to `run_all.py`'s `SCENARIOS` dict

```python
from common.environment import SimulationEnvironment, EnvironmentConfig, ControllerConfig

class MyScenario(SimulationEnvironment):
    def __init__(self):
        config = EnvironmentConfig(name="My Scenario", duration=15.0, ...)
        super().__init__(config)

    def get_initial_state(self):
        return np.array([0.0, 0.0, 0.0])

    def get_obstacles(self, t=0.0):
        return [(3.0, 0.0, 0.5)]

    def get_reference_fn(self):
        def ref_fn(t):
            return generate_reference_trajectory(traj_fn, t, self.config.N, self.config.dt)
        return ref_fn

    def get_controller_configs(self):
        model = DifferentialDriveKinematic()
        params = MPPIParams(N=30, dt=0.05, K=1024, ...)
        ctrl = MPPIController(model, params)
        return [ControllerConfig("MyCtrl", ctrl, model, "#1f77b4")]
```
